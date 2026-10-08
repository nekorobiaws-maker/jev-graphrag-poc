#!/usr/bin/env python3
"""query Lambda(`jev-graphrag-query`)の入口。`query_core.run_query()` を呼ぶだけ。

直接 Invoke の event(question 以外は省略可。知らないキーがあれば BadEvent):

    {"question": "...", "mode": "adaptive", "max_hops": 3, "max_neighbors": 3, "max_chunks": 10,
     "max_chunks_per_node": 5, "entry_th": 0.5, "sufficiency_th": 0.7, "min_neighbor_th": 0.5,
     "hop_chunk_source": "edge_evidence", "max_chunks_per_hop": 3, "entry_prefilter": "dictionary",
     "entry_fallback": "semantic", "semantic_entry_th": 0.7, "entry_chunk_rank": "jev",
     "answer_style": "concise", "hop_dup_policy": "count", "answer_filter": "none",
     "answer_filter_threshold": 0.5, "answer_filter_max": 5}

応答は `run_query()` のトレースに `ok` / `function` / `cold_start` / `records`(Jev と Bedrock の呼び出しレコード)を
足したもの。失敗しても課金済みのレコードは返す(`ok: false`)。
マスターは呼び出しごとに DynamoDB の META を見て、版が変わっていれば読み直す(無ければ同梱の master.json)。
Lambda の残り時間から締め切りを決めて `run_query(deadline=...)` に渡す。

API Gateway 経由(`requestContext` と `body` がある event)は、body の `question`(1〜200 文字)と
`mode`(`fixed1` | `fixed2` | `adaptive`)だけを受け付け、締め切りは 25 秒。応答のレコードからプロンプト本文を外す。

zip に `botocore_data/` があれば、boto3 を使う前に `AWS_DATA_PATH` に入れる(同梱した新しい DynamoDB の API 定義を使う)。"""

from __future__ import annotations

import base64
import binascii
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
BOTOCORE_DATA_DIR = HERE / "botocore_data"     # infra/build.sh が入れる DynamoDB の新しい API 定義
# boto3 のセッション(= botocore の Loader)を作る前に置く。deploy.py が環境変数で渡していれば何もしない
if BOTOCORE_DATA_DIR.is_dir():
    os.environ.setdefault("AWS_DATA_PATH", str(BOTOCORE_DATA_DIR))

import aws_secrets
import bedrock_llm
import jev_client          # 先頭で読む: typesafe_sdk / pydantic_core(C 拡張)の読み込み失敗を初期化時に出す
import ratelimit
from common import DEFAULT_GEN_MODEL, REGION, error_code, utc_now_iso
from graph_store import GraphStore
from master import MasterCache
from query_core import (
    BadEvent,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_ENTRY_PREFILTER,
    DEFAULT_ENTRY_FALLBACK,
    DEFAULT_ENTRY_CHUNK_RANK,
    DEFAULT_ANSWER_STYLE,
    DEFAULT_ANSWER_FILTER,
    DEFAULT_ANSWER_FILTER_MAX,
    DEFAULT_ANSWER_FILTER_THRESHOLD,
    DEFAULT_HOP_DUP_POLICY,
    DEFAULT_SEMANTIC_ENTRY_THRESHOLD,
    DEFAULT_HOP_CHUNK_SOURCE,
    DEFAULT_MAX_CHUNKS_PER_HOP,
    DEFAULT_MAX_CHUNKS_PER_NODE,
    DEFAULT_MAX_HOPS,
    DEFAULT_MAX_NEIGHBORS,
    DEFAULT_MIN_NEIGHBOR_SCORE,
    DEFAULT_ENTRY_THRESHOLD,
    DEFAULT_SUFFICIENCY_THRESHOLD,
    make_params,
    run_query,
    summary,
)

FUNCTION_NAME = "jev-graphrag-query"
METHOD_JEV_GRAPH = "jev_graph"            # 応答の method 欄に入る方式名
ALL_METHODS = (METHOD_JEV_GRAPH,)

JEV_RATE_PER_MIN = 1000
JEV_MAX_RETRIES = 1                       # Lambda のタイムアウトに収めるため
JEV_TIMEOUT_SEC = 15.0
BEDROCK_RATE_PER_MIN = 37                 # Haiku のクォータ 50/分の 75%
BEDROCK_READ_TIMEOUT = 30
BEDROCK_RETRY = {"max_attempts": 3, "max_delay": 5.0}   # call_with_retry へ(既定は 8 回・最大 30 秒)
DEADLINE_MARGIN_SEC = 5.0                 # Lambda の残り時間から差し引く余裕(応答を返す時間)

EVENT_DEFAULTS: dict[str, Any] = {
    "question": None,
    "mode": "adaptive",
    "max_hops": DEFAULT_MAX_HOPS,
    "max_neighbors": DEFAULT_MAX_NEIGHBORS,
    "max_chunks": DEFAULT_MAX_CHUNKS,
    "max_chunks_per_node": DEFAULT_MAX_CHUNKS_PER_NODE,
    "entry_th": DEFAULT_ENTRY_THRESHOLD,
    "sufficiency_th": DEFAULT_SUFFICIENCY_THRESHOLD,
    "min_neighbor_th": DEFAULT_MIN_NEIGHBOR_SCORE,
    "hop_chunk_source": DEFAULT_HOP_CHUNK_SOURCE,
    "max_chunks_per_hop": DEFAULT_MAX_CHUNKS_PER_HOP,
    "entry_prefilter": DEFAULT_ENTRY_PREFILTER,
    "entry_fallback": DEFAULT_ENTRY_FALLBACK,
    "semantic_entry_th": DEFAULT_SEMANTIC_ENTRY_THRESHOLD,
    "entry_chunk_rank": DEFAULT_ENTRY_CHUNK_RANK,
    "answer_style": DEFAULT_ANSWER_STYLE,
    "hop_dup_policy": DEFAULT_HOP_DUP_POLICY,
    "answer_filter": DEFAULT_ANSWER_FILTER,
    "answer_filter_threshold": DEFAULT_ANSWER_FILTER_THRESHOLD,
    "answer_filter_max": DEFAULT_ANSWER_FILTER_MAX,
    "method": METHOD_JEV_GRAPH,
}

# ---- API Gateway 経由(濫用防止のため、受け付ける口を絞る)
API_ALLOWED_KEYS = frozenset({"question", "mode"})
API_DEFAULT_MODE = "adaptive"
# scripts/21_invoke_query.py の MODE_EVENTS と同じ対応
API_MODES: dict[str, dict[str, Any]] = {
    "fixed1": {"mode": "fixed", "max_hops": 1},
    "fixed2": {"mode": "fixed", "max_hops": 2},
    "adaptive": {"mode": "adaptive"},
}
API_QUESTION_MAX_CHARS = 200
API_BODY_MAX_BYTES = 4096
API_DEADLINE_SEC = 25.0                   # 統合タイムアウト 29 秒より前に必ず返す
# 締め切りの予約値(秒)。query_core の既定では 25 秒に入らないので小さくする
API_RESERVES: dict[str, float] = {"jev": 4.0, "ddb": 2.0, "gen": 10.0}
API_HEADERS = {"Content-Type": "application/json"}


# ============================================================== モジュール変数(ウォームスタートで使い回す)

_client: Any = None
_store: Any = None
_bedrock: Any = None
_bedrock_raw: Any = None
_bedrock_bucket: Any = None
_ddb: Any = None
_master_cache: MasterCache | None = None
_cold = True


def get_client() -> Any:
    """JevClient を 1 つだけ作って使い回す。"""
    global _client
    if _client is None:
        _client = jev_client.JevClient(
            api_key=aws_secrets.get_api_key(),
            bucket=ratelimit.TokenBucket(JEV_RATE_PER_MIN),
            cache=None,
            max_retries=JEV_MAX_RETRIES,
            timeout=JEV_TIMEOUT_SEC,
        )
    return _client


def get_store() -> Any:
    global _store
    if _store is None:
        _store = GraphStore()
    return _store


class BedrockReadTimeout(RuntimeError):
    """Bedrock の応答待ちが read_timeout を超えた。**課金済みの可能性があるので再試行しない**
    (`call_with_retry` は ReadTimeoutError を一時障害として再試行するので、別の例外に包み替えて止める)。"""


class NoReadTimeoutRetry:
    """bedrock-runtime クライアントの包み。`converse` の ReadTimeoutError を BedrockReadTimeout に替える。"""

    def __init__(self, client: Any) -> None:
        self._client = client

    def converse(self, **kwargs: Any) -> Any:
        try:
            return self._client.converse(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "ReadTimeoutError":
                raise BedrockReadTimeout(
                    f"Bedrock の応答が {BEDROCK_READ_TIMEOUT} 秒で返りませんでした(課金済みの可能性があるので再試行しない)"
                ) from None
            raise


def bedrock_config() -> Any:
    """bedrock-runtime 用。botocore の内部再試行は**ゼロ**(`total_max_attempts=1`)。
    標準モードの内部再試行は ReadTimeoutError もやり直してしまうため。"""
    from botocore.config import Config

    return Config(read_timeout=BEDROCK_READ_TIMEOUT, connect_timeout=5,
                  retries={"total_max_attempts": 1, "mode": "standard"})


def get_llm() -> Any:
    """boto3 クライアントとレート制限は使い回し、`ConverseCaller` は呼び出しごとに作る
    (`fatal` が立ったままウォームスタートの次の呼び出しまで引きずらないように)。"""
    global _bedrock, _bedrock_bucket
    if _bedrock is None:
        _bedrock = NoReadTimeoutRetry(get_bedrock_raw())
        _bedrock_bucket = ratelimit.TokenBucket(BEDROCK_RATE_PER_MIN)
    return bedrock_llm.ConverseCaller(client=_bedrock, model=DEFAULT_GEN_MODEL, kind="gen",
                                      cache=None, bucket=_bedrock_bucket, label="answer",
                                      retry=BEDROCK_RETRY)


def get_bedrock_raw() -> Any:
    """bedrock-runtime の client(1 つを使い回す)。"""
    global _bedrock_raw
    if _bedrock_raw is None:
        import boto3

        _bedrock_raw = boto3.client("bedrock-runtime", region_name=REGION, config=bedrock_config())
    return _bedrock_raw


_SAFE_MESSAGE_ERRORS = (BadEvent,)      # メッセージを応答に載せてよい例外(それ以外は種類とコードだけ)


def _data_path(name: str) -> Path | None:
    """zip のルートにあればそれを使う。無ければ None(= ローカルの `data/` を `master.py` が読む)。"""
    path = HERE / name
    return path if path.exists() else None


def get_master_info() -> tuple[dict, dict]:
    """マスターと `{source, version, entities}`。**呼び出しごとに**呼ぶ(META だけ GetItem して、
    version が変わっていれば DynamoDB から読み直す。MASTER が無ければ同梱の master.json)。"""
    global _master_cache
    if _master_cache is None:
        _master_cache = MasterCache(_data_path("master.json"))
    return _master_cache.get(get_store())


def get_master() -> dict:
    return get_master_info()[0]


# ============================================================== event

def parse_event(event: Any) -> dict:
    """event を `{question, mode, max_hops, max_neighbors, max_chunks, max_chunks_per_node, entry_th,
    sufficiency_th, min_neighbor_th, hop_chunk_source, max_chunks_per_hop, entry_prefilter, entry_fallback,
    semantic_entry_threshold, entry_chunk_rank, answer_style, hop_dup_policy, answer_filter,
    answer_filter_threshold, answer_filter_max, answer_filter_min}` に
    正規化する。おかしければ BadEvent。"""
    if not isinstance(event, Mapping):
        raise BadEvent(f"event は JSON オブジェクトです: {type(event).__name__}")
    unknown = sorted(set(event) - set(EVENT_DEFAULTS))
    if unknown:
        raise BadEvent(f"知らないキーがあります: {unknown}(有効: {sorted(EVENT_DEFAULTS)})")
    merged = {**EVENT_DEFAULTS, **event}
    question = merged["question"]
    if not isinstance(question, str) or not question.strip():
        raise BadEvent("question は空でない文字列です")
    method = merged["method"]
    if method not in ALL_METHODS:
        raise BadEvent(f"method は {list(ALL_METHODS)} のどれかです: {method!r}")
    params = make_params(mode=merged["mode"], entry_th=merged["entry_th"],
                         sufficiency_th=merged["sufficiency_th"], max_hops=merged["max_hops"],
                         max_neighbors=merged["max_neighbors"], max_chunks=merged["max_chunks"],
                         min_neighbor_score=merged["min_neighbor_th"],
                         max_chunks_per_node=merged["max_chunks_per_node"],
                         hop_chunk_source=merged["hop_chunk_source"],
                         max_chunks_per_hop=merged["max_chunks_per_hop"],
                         entry_prefilter=merged["entry_prefilter"],
                         entry_fallback=merged["entry_fallback"],
                         semantic_entry_th=merged["semantic_entry_th"],
                         entry_chunk_rank=merged["entry_chunk_rank"],
                         answer_style=merged["answer_style"],
                         hop_dup_policy=merged["hop_dup_policy"],
                         answer_filter=merged["answer_filter"],
                         answer_filter_threshold=merged["answer_filter_threshold"],
                         answer_filter_max=merged["answer_filter_max"])
    params.pop("in_scope_th")
    return {"question": question.strip(), "method": method, **params}


# ============================================================== 本体

def _error_info(exc: Exception) -> dict:
    info = {"type": type(exc).__name__, "status": getattr(exc, "status", None),
            "code": error_code(exc) or None}
    if isinstance(exc, _SAFE_MESSAGE_ERRORS):
        info["message"] = str(exc)
    return info


def deadline_from(context: Any, now: float, max_sec: float | None = None) -> float | None:
    """Lambda の残り時間から締め切り(`time.perf_counter()` の時計)を決める。context が無ければ None。"""
    remaining = getattr(context, "get_remaining_time_in_millis", None)
    deadline = None if remaining is None else now + float(remaining()) / 1000.0 - DEADLINE_MARGIN_SEC
    if max_sec is not None:
        cap = now + float(max_sec)
        deadline = cap if deadline is None else min(deadline, cap)
    return deadline


def record_line(record: Mapping[str, Any]) -> str:
    """呼び出しレコード 1 件ぶんの課金の記録(CloudWatch 用)。本文・キー・リクエストは入れない。
    タイムアウトで応答を持ち帰れなかったときに、課金をログから突き合わせるため。"""
    return json.dumps({"rec": record.get("kind"), "model": record.get("model"),
                       "usage": record.get("usage"), "ts": record.get("ts")}, ensure_ascii=False)


def lambda_handler(event: Any, context: Any = None) -> dict:
    if is_api_event(event):
        return api_handler(event, context)
    return invoke(event, context)


def invoke(event: Any, context: Any = None, *, max_sec: float | None = None,
           reserves: Mapping[str, float] | None = None) -> dict:
    """直接 Invoke の本体(API 経由もここを通る)。`max_sec` / `reserves` は API 経由のときだけ渡す。"""
    global _cold
    cold, _cold = _cold, False
    started = time.perf_counter()
    deadline = deadline_from(context, started, max_sec)
    records: list[dict] = []

    def on_record(record: dict) -> None:
        record["ts"] = utc_now_iso()   # 返ってきた時刻を付ける(ingest と同じ)
        records.append(record)
        print(record_line(record), flush=True)

    base: dict[str, Any] = {"ok": False, "function": FUNCTION_NAME, "cold_start": cold}
    method = event.get("method", METHOD_JEV_GRAPH) if isinstance(event, Mapping) else None
    base["method"] = method if method in ALL_METHODS else None
    try:
        args = parse_event(event)
        master, master_info = get_master_info()
        base["master_source"] = master_info["source"]
        base["master_version"] = master_info["version"]
        trace = run_query(
            args["question"],
            jev=get_client(), store=get_store(), llm=get_llm(), master=master,
            mode=args["mode"], entry_th=args["entry_threshold"], sufficiency_th=args["sufficiency_th"],
            max_hops=args["max_hops"], max_neighbors=args["max_neighbors"],
            max_chunks=args["max_chunks"], min_neighbor_score=args["min_neighbor_score"],
            max_chunks_per_node=args["max_chunks_per_node"],
            hop_chunk_source=args["hop_chunk_source"],
            max_chunks_per_hop=args["max_chunks_per_hop"],
            entry_prefilter=args["entry_prefilter"],
            entry_fallback=args["entry_fallback"],
            semantic_entry_th=args["semantic_entry_threshold"],
            entry_chunk_rank=args["entry_chunk_rank"],
            answer_style=args["answer_style"],
            hop_dup_policy=args["hop_dup_policy"],
            answer_filter=args["answer_filter"],
            answer_filter_threshold=args["answer_filter_threshold"],
            answer_filter_max=args["answer_filter_max"],
            answer_filter_min=args["answer_filter_min"],
            on_record=on_record, deadline=deadline,
            **({"reserves": reserves} if reserves is not None else {}),
        )
    except Exception as exc:  # noqa: BLE001
        response = {
            **base,
            "error": _error_info(exc),
            "question": event.get("question") if isinstance(event, Mapping) else None,
            "mode": event.get("mode") if isinstance(event, Mapping) else None,
            "jev_calls": sum(1 for r in records if r.get("kind") == "jev"),
            "latency_ms": {"total": (time.perf_counter() - started) * 1000.0},
            "records": records,
        }
        line = {"kind": "query_summary", "ok": False, "cold_start": cold, "error": response["error"],
                "method": base.get("method"),
                "master_source": base.get("master_source"), "master_version": base.get("master_version"),
                "mode": response["mode"], "jev_calls": response["jev_calls"],
                "n_records": len(records), "latency_ms": response["latency_ms"]}
        print(json.dumps(line, ensure_ascii=False), flush=True)
        return response

    return _ok_response(base, trace, records, cold, summary(trace))


def _ok_response(base: Mapping[str, Any], trace: Mapping[str, Any], records: list[dict], cold: bool,
                 line_summary: Mapping[str, Any]) -> dict:
    """成功時の応答とサマリー 1 行。"""
    response = {**base, "ok": True, **trace}
    response["records"] = records       # on_record で ts を付けたもの(trace["records"] と同じ物)
    line = {**line_summary, "ok": True, "cold_start": cold, "n_records": len(records),
            "method": base.get("method"),
            "master_source": base.get("master_source"), "master_version": base.get("master_version")}
    print(json.dumps(line, ensure_ascii=False), flush=True)
    return response


# ============================================================== API Gateway 経由

def is_api_event(event: Any) -> bool:
    """API Gateway のプロキシ統合の event か(`requestContext` と `body` の両方がある)。"""
    return isinstance(event, Mapping) and "requestContext" in event and "body" in event


def parse_api_event(event: Mapping[str, Any]) -> tuple[dict, str]:
    """API の body を読んで、直接 Invoke と同じ形の event に直す。戻り値は `(event, mode 名)`。"""
    raw = event.get("body")
    if raw is None or raw == "":
        raise BadEvent("body が空です(JSON で {\"question\": ..., \"mode\": ...} を送ってください)")
    if not isinstance(raw, str):
        raise BadEvent("body は JSON 文字列です")
    if event.get("isBase64Encoded"):
        try:
            raw = base64.b64decode(raw, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            raise BadEvent("body を base64 として読めません") from None
    if len(raw.encode("utf-8")) > API_BODY_MAX_BYTES:
        raise BadEvent(f"body が大きすぎます(上限 {API_BODY_MAX_BYTES} バイト)")
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        raise BadEvent("body が JSON として読めません") from None
    if not isinstance(body, Mapping):
        raise BadEvent("body は JSON オブジェクトです")
    unknown = sorted(set(body) - API_ALLOWED_KEYS)
    if unknown:
        raise BadEvent(f"受け付けないキーがあります: {unknown}(有効: {sorted(API_ALLOWED_KEYS)})")
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise BadEvent("question は空でない文字列です")
    question = question.strip()
    if len(question) > API_QUESTION_MAX_CHARS:
        raise BadEvent(f"question は {API_QUESTION_MAX_CHARS} 文字までです({len(question)} 文字)")
    mode = body.get("mode", API_DEFAULT_MODE)
    if not isinstance(mode, str) or mode not in API_MODES:
        raise BadEvent(f"mode は {list(API_MODES)} のどれかです")
    return {"question": question, **API_MODES[mode]}, mode


def strip_records(records: Any) -> list[dict]:
    """呼び出しレコードから `request`(プロンプト本文)を外した写し。費用の計算に要る
    key/kind/model/usage/ts などは残す。"""
    return [{k: v for k, v in r.items() if k != "request"}
            for r in records or [] if isinstance(r, Mapping)]


def _http(status: int, body: Mapping[str, Any]) -> dict:
    return {"statusCode": status, "headers": dict(API_HEADERS),
            "body": json.dumps(body, ensure_ascii=False, default=str)}


def api_handler(event: Mapping[str, Any], context: Any = None) -> dict:
    """API Gateway(プロキシ統合)の入口。応答は `{statusCode, headers, body}`。"""
    try:
        inner, mode_name = parse_api_event(event)
    except BadEvent as exc:
        return _http(400, {"ok": False, "function": FUNCTION_NAME,
                           "error": {"type": "BadEvent", "message": str(exc)}, "records": []})
    try:
        result = invoke(inner, context, max_sec=API_DEADLINE_SEC, reserves=API_RESERVES)
    except Exception as exc:  # noqa: BLE001
        return _http(500, {"ok": False, "function": FUNCTION_NAME,
                           "error": {"type": type(exc).__name__}, "records": []})
    body = {**result, "api_mode": mode_name, "records": strip_records(result.get("records"))}
    if result.get("ok"):
        return _http(200, body)
    error = result.get("error") or {}
    if error.get("type") == "BadEvent":
        return _http(400, body)
    body["error"] = {"type": error.get("type") or "Error"}       # 500 は種類だけ
    return _http(500, body)
