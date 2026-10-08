#!/usr/bin/env python3
"""ingest Lambda(`jev-graphrag-ingest`)の入口。`ingest_core.run_ingest(dry_run=False)` を呼ぶだけ。

event(全部省略可。知らないキーがあれば何もせずに失敗を返す):

    {"reset": false, "node_th": 0.20, "edge_th": 0.55, "include_rows": true,
     "chunk_ids": null, "expect_data": null, "slim_records": false, "node_prefilter": "dictionary"}

- `reset`: 判定が全部終わってから、書く直前に 2 テーブルを空にする
- `chunk_ids`: 処理するチャンク(null なら全部)。1 回の Invoke に収まらないので `scripts/20_invoke_ingest.py` が分けて呼ぶ
- `expect_data`: `{"master_version", "chunks_sha256"}`。Lambda が読んだデータと違えば Jev を呼ばずに失敗する
- `node_prefilter`: `"dictionary"`(辞書の候補だけを Jev に問う)/ `"none"`(全エンティティ)
- `include_rows` / `slim_records`: 判定ログ・呼び出しレコードを応答に入れるか / 費用計算に要る項目だけにするか

失敗しても課金済みの呼び出しレコード(`records`)は必ず返す(`ok: false`)。応答が 6MB を超えそうなら判定ログを省く。
API キーは SSM から読んで `JevClient` に渡すだけ(応答にも標準出力にも出さない)。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import aws_secrets
import jev_client          # 先頭で読む: typesafe_sdk / pydantic_core(C 拡張)の読み込み失敗を初期化時に出す
import ratelimit
from common import JEV_MODEL, error_code, utc_now_iso
from graph_store import GraphStore, TableGuardError
from ingest_core import (
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_NODE_PREFILTER,
    DEFAULT_NODE_THRESHOLD,
    NODE_PREFILTERS,
    EdgeCandidateLimit,
    missing_score_rows,
    run_ingest,
)
from master import CHUNKS_PATH, MASTER_PATH, MasterCache, file_sha256, load_chunks

FUNCTION_NAME = "jev-graphrag-ingest"
HERE = Path(__file__).resolve().parent

JEV_RATE_PER_MIN = 1000
PAYLOAD_LIMIT_BYTES = 6 * 1024 * 1024     # 同期 Invoke の応答の上限
PAYLOAD_SAFE_BYTES = 5_500_000            # これを超えたら削る(見積もりの誤差ぶんの余裕)

EVENT_DEFAULTS: dict[str, Any] = {
    "reset": False,
    "node_th": DEFAULT_NODE_THRESHOLD,
    "edge_th": DEFAULT_EDGE_THRESHOLD,
    "include_rows": True,
    "chunk_ids": None,
    "expect_data": None,
    "slim_records": False,
    "node_prefilter": DEFAULT_NODE_PREFILTER,
}
EXPECT_DATA_KEYS = ("master_version", "chunks_sha256")

# 呼び出しレコードを削るときに残す項目(budget.cost_of と突き合わせに要るものだけ)
SLIM_RECORD_KEYS = ("key", "kind", "model", "usage", "latency_ms", "retries", "ts")

# メッセージを応答に載せてよい例外(自前の例外だけ。SDK の例外は本文を載せない)
_SAFE_MESSAGE_ERRORS = (ValueError, EdgeCandidateLimit, TableGuardError, aws_secrets.SecretUnavailable)

# ============================================================== モジュール変数(ウォームスタートで使い回す)

_client: Any = None
_store: Any = None
_master_cache: MasterCache | None = None
_chunks: list[dict] | None = None
_chunks_sha256: str | None = None
_data_info: dict | None = None
_cold = True


def get_client() -> Any:
    """JevClient を 1 つだけ作って使い回す。"""
    global _client
    if _client is None:
        _client = jev_client.JevClient(
            api_key=aws_secrets.get_api_key(),
            bucket=ratelimit.TokenBucket(JEV_RATE_PER_MIN),
            cache=None,
        )
    return _client


def get_store() -> Any:
    global _store
    if _store is None:
        _store = GraphStore()
    return _store


def _data_path(name: str) -> Path | None:
    """zip のルートにあればそれを使う。無ければ None(= ローカルの `data/` を `master.py` が読む)。"""
    path = HERE / name
    return path if path.exists() else None


def master_cache() -> MasterCache:
    global _master_cache
    if _master_cache is None:
        _master_cache = MasterCache(_data_path("master.json") or MASTER_PATH)
    return _master_cache


def get_chunks() -> list[dict]:
    """zip のチャンク(起動時に 1 回だけ読む)。"""
    global _chunks, _chunks_sha256
    if _chunks is None:
        chunks_path = _data_path("chunks.json") or CHUNKS_PATH
        _chunks = load_chunks(chunks_path)
        _chunks_sha256 = file_sha256(chunks_path)
    return _chunks


def get_data() -> tuple[dict, list[dict]]:
    """(マスター, チャンク)。**呼び出しごとに**呼ぶ(マスターは META の version を見て読み直す)。"""
    global _data_info
    chunks = get_chunks()
    master, info = master_cache().get(get_store())
    _data_info = {"entities": len(master["entities"]), "chunks": len(chunks),
                  "master_source": info["source"], "master_version": info["version"],
                  "chunks_sha256": _chunks_sha256}
    return master, chunks


def get_data_info() -> dict | None:
    """`get_data()` 済みなら `{entities, chunks, master_source, master_version, chunks_sha256}`。まだなら None。"""
    return dict(_data_info) if _data_info is not None else None


def check_expected_data(expect: Mapping[str, Any] | None, info: Mapping[str, Any]) -> None:
    """`expect_data` と Lambda のデータが違えば ValueError(Jev を呼ぶ前に止める)。"""
    if expect is None:
        return
    diff = [k for k in EXPECT_DATA_KEYS if expect.get(k) != info.get(k)]
    if diff:
        hints = []
        if "master_version" in diff:
            hints.append("マスターは scripts/60_put_master.py で DynamoDB に入れてください"
                         f"(Lambda が読んだのは {info.get('master_source')})")
        if "chunks_sha256" in diff:
            hints.append("チャンクは infra/build.sh と deploy.py deploy-lambdas で入れ直してください")
        raise ValueError(f"Lambda のデータが手元と違います({', '.join(diff)})。" + "。".join(hints))


def select_chunks(chunks: list[dict], chunk_ids: list[str] | None) -> list[dict]:
    """`chunk_ids` のチャンクだけを、**zip の並び順で**返す。None なら全部。知らない id は ValueError。"""
    if chunk_ids is None:
        return list(chunks)
    known = {c["chunk_id"] for c in chunks}
    unknown = [cid for cid in chunk_ids if cid not in known]
    if unknown:
        shown = ", ".join(unknown[:5]) + ("…" if len(unknown) > 5 else "")
        raise ValueError(f"zip のチャンクに無い chunk_id があります({len(unknown)} 件): {shown}")
    wanted = set(chunk_ids)
    return [c for c in chunks if c["chunk_id"] in wanted]


# ============================================================== event

def _threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} は 0〜1 の数値です: {value!r}")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} は 0〜1 の数値です: {value!r}")
    return value


def _flag(name: str, value: Any) -> bool:
    # "false" のような文字列を真とみなして全削除する事故を防ぐため、bool だけ受け付ける
    if not isinstance(value, bool):
        raise ValueError(f"{name} は true / false です: {value!r}")
    return value


def _chunk_ids(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise ValueError("chunk_ids は空でない文字列の配列(または null)です")
    if len(set(value)) != len(value):
        raise ValueError("chunk_ids に重複があります")
    return list(value)


def _expect_data(value: Any) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or sorted(value) != sorted(EXPECT_DATA_KEYS) \
            or not all(isinstance(value[k], str) for k in EXPECT_DATA_KEYS):
        raise ValueError(f"expect_data は {list(EXPECT_DATA_KEYS)} の 2 つを文字列で持つオブジェクト(または null)です")
    return {k: value[k] for k in EXPECT_DATA_KEYS}


def _node_prefilter(value: Any) -> str:
    if not isinstance(value, str) or value not in NODE_PREFILTERS:
        raise ValueError(f"node_prefilter は {list(NODE_PREFILTERS)} のどれかです: {value!r}")
    return value


def parse_event(event: Any) -> dict:
    """event を `{reset, node_th, edge_th, include_rows, chunk_ids, expect_data, slim_records,
    node_prefilter}` に正規化する。おかしければ ValueError。"""
    if event is None:
        event = {}
    if not isinstance(event, Mapping):
        raise ValueError(f"event は JSON オブジェクトです: {type(event).__name__}")
    unknown = sorted(set(event) - set(EVENT_DEFAULTS))
    if unknown:
        raise ValueError(f"知らないキーがあります: {unknown}(有効: {sorted(EVENT_DEFAULTS)})")
    merged = {**EVENT_DEFAULTS, **event}
    return {
        "reset": _flag("reset", merged["reset"]),
        "node_th": _threshold("node_th", merged["node_th"]),
        "edge_th": _threshold("edge_th", merged["edge_th"]),
        "include_rows": _flag("include_rows", merged["include_rows"]),
        "chunk_ids": _chunk_ids(merged["chunk_ids"]),
        "expect_data": _expect_data(merged["expect_data"]),
        "slim_records": _flag("slim_records", merged["slim_records"]),
        "node_prefilter": _node_prefilter(merged["node_prefilter"]),
    }


# ============================================================== 応答サイズ

def payload_bytes(obj: Any) -> int:
    """Lambda ランタイムと同じく `json.dumps` 既定(ensure_ascii=True)で数える。
    日本語は `\\uXXXX` の 6 バイトになるので、UTF-8 で数えるより大きめ(安全側)に出る。"""
    return len(json.dumps(obj).encode("utf-8"))


def slim_records(records: list[dict]) -> list[dict]:
    """呼び出しレコードを費用計算・突き合わせに要る項目(SLIM_RECORD_KEYS)だけにする。"""
    return [{k: r.get(k) for k in SLIM_RECORD_KEYS} for r in records]


def fit_payload(response: dict, include_rows: bool, safe_bytes: int = PAYLOAD_SAFE_BYTES,
                slim_first: bool = False) -> dict:
    """応答を上限に収める。順に (1) 判定ログを省く (2) 呼び出しレコードを削る。
    `slim_first=True`(event の `slim_records`)なら、先に呼び出しレコードを削ってから (1) を判断する。"""
    info: dict[str, Any] = {
        "limit_bytes": PAYLOAD_LIMIT_BYTES,
        "safe_bytes": safe_bytes,
        "full_bytes": payload_bytes(response),
        "rows_omitted": None,          # None / "include_rows=false" / "size"
        "records_slimmed": False,
    }
    if slim_first:
        response["records"] = slim_records(response.get("records") or [])
        info["records_slimmed"] = True
    if not include_rows:
        info["rows_omitted"] = "include_rows=false"
    elif payload_bytes(response) > safe_bytes:
        info["rows_omitted"] = "size"
    if info["rows_omitted"]:
        response["node_rows"] = None
        response["edge_rows"] = None
    if not info["records_slimmed"] and payload_bytes(response) > safe_bytes:
        response["records"] = slim_records(response.get("records") or [])
        info["records_slimmed"] = True
    response["payload"] = info
    info["estimated_bytes"] = payload_bytes(response)
    return response


# ============================================================== 本体

def _error_info(exc: Exception) -> dict:
    info = {"type": type(exc).__name__, "status": getattr(exc, "status", None),
            "code": error_code(exc) or None}
    if isinstance(exc, _SAFE_MESSAGE_ERRORS):
        info["message"] = str(exc)
    return info


def _summary_line(response: dict) -> str:
    """CloudWatch Logs に出す 1 行。判定ログ・レコード本体・キーは入れない。"""
    payload = response.get("payload") or {}
    counts = response.get("counts") or {}
    return json.dumps({
        "kind": "ingest_summary",
        "ok": response.get("ok"),
        "cold_start": response.get("cold_start"),
        "params": response.get("params"),
        "data": response.get("data"),
        "written": response.get("written"),
        "deleted": response.get("deleted"),
        "jev_requests": counts.get("jev_requests"),
        "input_tokens": counts.get("input_tokens"),
        "retries": counts.get("retries"),
        "latency_ms": response.get("latency_ms"),
        "payload_bytes": payload.get("estimated_bytes"),
        "rows_omitted": payload.get("rows_omitted"),
        "records_slimmed": payload.get("records_slimmed"),
        "error": response.get("error"),
    }, ensure_ascii=False)


def _record_counts(records: list[dict]) -> dict:
    return {
        "jev_requests": len(records),
        "input_tokens": sum(int((r.get("usage") or {}).get("input_tokens") or 0) for r in records),
        "retries": sum(int(r.get("retries") or 0) for r in records),
    }


def lambda_handler(event: Any, context: Any = None) -> dict:
    global _cold
    cold, _cold = _cold, False
    started = time.perf_counter()
    records: list[dict] = []

    def on_record(record: dict) -> None:
        record["ts"] = utc_now_iso()
        records.append(record)

    base: dict[str, Any] = {"ok": False, "function": FUNCTION_NAME, "model": JEV_MODEL,
                            "cold_start": cold, "params": None, "data": None}
    slim_first = False
    try:
        params = parse_event(event)
        base["params"] = params
        slim_first = params["slim_records"]
        master, all_chunks = get_data()
        base["data"] = get_data_info()
        check_expected_data(params["expect_data"], base["data"])     # Jev・SSM より先に確かめる
        chunks = select_chunks(all_chunks, params["chunk_ids"])
        result = run_ingest(
            get_client(), master, chunks,
            dry_run=False, stages=("nodes", "edges"), mode="packed",
            node_prefilter=params["node_prefilter"],
            node_threshold=params["node_th"], edge_threshold=params["edge_th"],
            store=get_store(), reset=params["reset"],
            on_record=on_record,
        )
    except Exception as exc:  # noqa: BLE001
        response = {
            **base,
            "error": _error_info(exc),
            "counts": _record_counts(records),
            "latency_ms": {"total": (time.perf_counter() - started) * 1000.0,
                           "jev_sum": sum(float(r.get("latency_ms") or 0.0) for r in records)},
            "node_rows": None, "edge_rows": None,
            "records": records,
        }
        response = fit_payload(response, include_rows=False, slim_first=slim_first)
        print(_summary_line(response), flush=True)
        return response

    node_rows = result["node_rows"]
    edge_rows = result["edge_rows"]
    response = {
        **base,
        "ok": True,
        "written": result["written"],
        "deleted": result["deleted"],
        "counts": {
            "chunks": len(chunks),
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "node_judgments": sum(1 for r in node_rows if not r.get("prefiltered")),
            "node_prefiltered": sum(1 for r in node_rows if r.get("prefiltered")),
            "node_chunks_no_candidates": sum(1 for v in result["node_candidates"].values() if not v),
            "present": sum(len(v) for v in result["present"].values()),
            "edge_candidates": len(edge_rows),
            "missing_scores": len(missing_score_rows(node_rows))
            + sum(1 for r in edge_rows if r.get("prob") is None),
            **_record_counts(records),
        },
        "latency_ms": {
            "total": (time.perf_counter() - started) * 1000.0,
            **result["timings_ms"],
            "jev_sum": sum(float(r.get("latency_ms") or 0.0) for r in records),
        },
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "records": records,
    }
    response = fit_payload(response, include_rows=params["include_rows"], slim_first=params["slim_records"])
    print(_summary_line(response), flush=True)
    return response
