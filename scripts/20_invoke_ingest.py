#!/usr/bin/env python3
"""ingest Lambda(`jev-graphrag-ingest`)を同期 Invoke し、結果を持ち帰って件数を突き合わせる。

    ./.venv/bin/python scripts/20_invoke_ingest.py --dry-estimate          # 見積もりだけ(AWS に触れない)
    ./.venv/bin/python scripts/20_invoke_ingest.py --reset --yes           # 全部登録し直す
    ./.venv/bin/python scripts/20_invoke_ingest.py --start-batch 3 --yes   # 途中で止まった続きから(reset なし)

- チャンクを `--batch-chunks`(既定 40)件ずつに分けて Invoke する。`reset` は最初の 1 回だけ
- Invoke の前に DynamoDB のマスターの版と手元の版を突き合わせ、違えば止まる(先に `60_put_master.py`)
- 自動で再 Invoke しない。呼び出しレコードは `results/cache/jev.jsonl` に追記し、判定ログは `results/ingest/` に残す
- 最後に、判定ログから数えた期待件数・Lambda が書いた件数・DynamoDB の実件数を並べる"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

LIB_DIR = Path(__file__).resolve().parent.parent / "lib"
_lib = str(LIB_DIR)
if _lib in sys.path:
    sys.path.remove(_lib)
sys.path.insert(0, _lib)   # 同名モジュールが他所にあっても lib/ を優先させる

from budget import Budget, cost_of  # noqa: E402
from common import (  # noqa: E402
    require_home_region,
    DATASET,
    JEV_MODEL,
    REGION,
    RESULTS_DIR,
    append_jsonl,
    boto_config,
    cache_path,
    check_account,
    ensure_dirs,
    iter_jsonl,
)
from ingest_core import (  # noqa: E402
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_NODE_PREFILTER,
    DEFAULT_NODE_THRESHOLD,
    MAX_EDGE_CANDIDATES,
    NODE_PREFILTERS,
    edge_rows as make_edge_rows,
    estimate_nodes,
    node_rows as make_node_rows,
    node_scores_from_rows,
    plan_edges,
    present_ids,
)
from master import (  # noqa: E402
    CHUNKS_PATH,
    entity_by_id,
    file_sha256,
    load_chunks,
    load_master,
    master_version,
    symmetric_edge_names,
)

INGEST_FUNCTION = "jev-graphrag-ingest"
INGEST_DIR = RESULTS_DIR / "ingest"


FUNCTION_TIMEOUT_SEC = 900
READ_TIMEOUT_SEC = FUNCTION_TIMEOUT_SEC + 60   # 関数側のタイムアウト応答を受け取れるように余裕を持たせる
LOOK_ENTITY = "luffy"

DEFAULT_INVOKE_CHUNKS = 40          # 1 回の Invoke に渡すチャンク数
SEC_PER_JEV_REQUEST_EST = 0.6
SEC_PER_CHUNK_WRITE_EST = 0.15      # DynamoDB への書き込み(チャンク・登場・関係の順逆)
TIME_WARN_RATIO = 0.6               # 見積もりが関数のタイムアウトのこの割合を超えたら警告
SLIM_RECORD_BYTES_EST = 400         # slim にした呼び出しレコード 1 件の大きさ(ensure_ascii の JSON)
PAYLOAD_SAFE_BYTES = 5_500_000      # handler.py と同じ(これを超えると判定ログが省かれる)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ingest Lambda を同期 Invoke して突き合わせる")
    parser.add_argument("--reset", action="store_true",
                        help="書く前に 2 テーブルを空にする(判定が全部終わってから消す)")
    parser.add_argument("--node-th", type=float, default=DEFAULT_NODE_THRESHOLD,
                        help=f"登場とみなすしきい値(既定 {DEFAULT_NODE_THRESHOLD})")
    parser.add_argument("--edge-th", type=float, default=DEFAULT_EDGE_THRESHOLD,
                        help=f"関係ありとみなすしきい値(既定 {DEFAULT_EDGE_THRESHOLD})")
    parser.add_argument("--node-prefilter", choices=NODE_PREFILTERS, default=DEFAULT_NODE_PREFILTER,
                        help="ノード判定で Jev に問うエンティティの絞り方。dictionary=本文から辞書で拾った候補だけ"
                             f" / none=全エンティティの総当たり(旧動作)。既定 {DEFAULT_NODE_PREFILTER}")
    parser.add_argument("--no-rows", action="store_true",
                        help="判定ログを応答に入れさせない(応答が 6MB を超えそうなとき用)")
    parser.add_argument("--batch-chunks", type=int, default=DEFAULT_INVOKE_CHUNKS,
                        help=f"1 回の Invoke に渡すチャンク数(既定 {DEFAULT_INVOKE_CHUNKS})")
    parser.add_argument("--start-batch", type=int, default=0,
                        help="この番号(0 始まり)の組から Invoke する。途中で止まったときの続き用(--reset と併用不可)")
    parser.add_argument("--full-records", action="store_true",
                        help="呼び出しレコードを request 本文つきで持ち帰る(既定は費用計算の項目だけ)")
    parser.add_argument("--local-nodes-log", type=Path, default=None,
                        help="比較・見積もりに使う手元のノード判定ログ(ingest_core.node_rows() の形の JSONL。省略可)")
    parser.add_argument("--local-edges-log", type=Path, default=None,
                        help="比較に使う手元の関係判定ログ(ingest_core.edge_rows() の形の JSONL。省略可)")
    parser.add_argument("--dry-estimate", action="store_true",
                        help="見積もりだけ表示して終わる(AWS にも Jev にも触れない)")
    parser.add_argument("--yes", action="store_true", help="実行前の Enter 確認を飛ばす")
    args = parser.parse_args(argv)
    if args.batch_chunks < 1:
        parser.error("--batch-chunks は 1 以上です")
    if args.start_batch < 0:
        parser.error("--start-batch は 0 以上です")
    if args.start_batch > 0 and args.reset:
        parser.error("--start-batch 1 以上と --reset は併用できません(続きの前に書いた分まで消えるため)")

    for name in ("node_th", "edge_th"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} は 0〜1 です: {value}")
    return args


def invoke_batches(chunk_ids: list[str], size: int) -> list[list[str]]:
    """チャンク id を `size` 件ずつの組に分ける(並び順を保つ)。"""
    if size < 1:
        raise ValueError(f"size は 1 以上です: {size}")
    return [list(chunk_ids[i:i + size]) for i in range(0, len(chunk_ids), size)]


def batch_events(chunk_ids: list[str], size: int, *, reset: bool, node_th: float, edge_th: float,
                 include_rows: bool, slim_records: bool, expect: Mapping[str, str] | None,
                 start: int = 0, node_prefilter: str = DEFAULT_NODE_PREFILTER) -> list[dict]:
    """組ごとの event。**reset は全体の最初の組(start=0 の 0 番)だけ**。start 以降の組だけ返す。"""
    events = []
    for i, ids in enumerate(invoke_batches(chunk_ids, size)):
        if i < start:
            continue
        events.append({"reset": bool(reset and i == 0), "node_th": node_th, "edge_th": edge_th,
                       "include_rows": include_rows, "chunk_ids": ids,
                       "expect_data": dict(expect) if expect is not None else None,
                       "slim_records": slim_records, "node_prefilter": node_prefilter})
    return events


def present_by_string(master: Mapping[str, Any], chunks: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """見積もり用の登場ノード: 正式名か別名が本文に**文字として含まれる**エンティティ。
    部分一致も数えるので Jev より多め・言い換えは拾えないので少なめ。"""
    out: dict[str, list[str]] = {}
    for chunk in chunks:
        text = chunk["text"]
        out[chunk["chunk_id"]] = [e["id"] for e in master["entities"]
                                  if any(n and n in text for n in [e["name"], *(e.get("aliases") or [])])]
    return out


def plan_invokes(master: Mapping[str, Any], chunks: list[dict], present: Mapping[str, list[str]],
                 size: int, *, include_rows: bool = True, slim_records: bool = True,
                 node_prefilter: str = DEFAULT_NODE_PREFILTER) -> list[dict]:
    """組ごとの見積もり: `{index, chunk_ids, n_chunks, node_requests, node_judgments, edge_candidates,
    edge_requests, node_input_tokens, node_usd, edge_input_tokens, edge_usd, est_input_tokens, est_usd,
    est_sec, est_payload_bytes, over_candidates, over_time, over_payload}`。
    ノード判定は `node_prefilter`(既定 dictionary = 辞書の候補だけを問う)で数える。"""
    by_id = {c["chunk_id"]: c for c in chunks}
    plans = []
    for i, ids in enumerate(invoke_batches([c["chunk_id"] for c in chunks], size)):
        part = [by_id[cid] for cid in ids]
        nodes = estimate_nodes(part, master["entities"], "packed", node_prefilter)
        edges = plan_edges(master, part, {cid: present.get(cid, []) for cid in ids})
        n_req = nodes["n_requests"] + edges["n_requests"]
        rows_bytes = 0
        if include_rows:
            fake = {e["id"]: 0.123456 for e in master["entities"]}
            for chunk in part:
                rows_bytes += len(json.dumps(make_node_rows(chunk, master["entities"], fake, "packed")))
                cands = edges["candidates"][chunk["chunk_id"]]
                rows_bytes += len(json.dumps(make_edge_rows(chunk, cands, {c["qid"]: 0.123456 for c in cands})))
        if slim_records:
            rec_bytes = n_req * SLIM_RECORD_BYTES_EST
        else:
            rec_bytes = int((nodes["est_input_tokens"] + edges["est_input_tokens"]) / 1.5 * 3)
        est_sec = n_req * SEC_PER_JEV_REQUEST_EST + len(part) * SEC_PER_CHUNK_WRITE_EST
        payload = rows_bytes + rec_bytes
        plans.append({
            "index": i, "chunk_ids": ids, "n_chunks": len(ids),
            "node_requests": nodes["n_requests"], "node_judgments": nodes["n_judgments"],
            "edge_candidates": edges["n_candidates"], "edge_requests": edges["n_requests"],
            "node_input_tokens": nodes["est_input_tokens"], "node_usd": nodes["est_usd"],
            "edge_input_tokens": edges["est_input_tokens"], "edge_usd": edges["est_usd"],
            "est_input_tokens": nodes["est_input_tokens"] + edges["est_input_tokens"],
            "est_usd": nodes["est_usd"] + edges["est_usd"],
            "est_sec": est_sec, "est_payload_bytes": payload,
            "over_candidates": edges["n_candidates"] > MAX_EDGE_CANDIDATES,
            "over_time": est_sec > FUNCTION_TIMEOUT_SEC * TIME_WARN_RATIO,
            "over_payload": payload > PAYLOAD_SAFE_BYTES,
        })
    return plans


def invoke_config() -> Any:
    """Lambda 用の botocore Config。**再試行なし**(`total_max_attempts=1`)。"""
    from botocore.config import Config

    return Config(read_timeout=READ_TIMEOUT_SEC, connect_timeout=10, tcp_keepalive=True,
                  retries={"total_max_attempts": 1, "mode": "standard"})


def confirm() -> bool:
    try:
        input("Enter で実行、Ctrl-C で中止: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return True


# ============================================================== 突き合わせ(純粋関数)

def expected_counts(node_rows: Iterable[Mapping[str, Any]], edge_rows: Iterable[Mapping[str, Any]],
                    node_th: float, edge_th: float, n_chunks: int,
                    symmetric: Iterable[str] = ()) -> dict:
    """判定ログをしきい値で数えた、書かれているはずの件数(向きのある関係は順＝逆)。
    `symmetric`(対称な関係のエッジ名)の関係は SYM# の行で両端に 1 行ずつ = 件数の 2 倍(`sym` は行数)。"""
    sym = frozenset(symmetric)
    n_app = sum(1 for r in node_rows if r.get("prob") is not None and r["prob"] >= node_th)
    passed = [r for r in edge_rows if r.get("prob") is not None and r["prob"] >= edge_th]
    n_sym = sum(1 for r in passed if r.get("edge") in sym)
    n_edge = len(passed) - n_sym
    return {"chunk": n_chunks, "appearance": n_app, "out": n_edge, "in": n_edge, "sym": 2 * n_sym}


def node_key(row: Mapping[str, Any]) -> tuple:
    return (row["chunk_id"], row["entity_id"])


def edge_key(row: Mapping[str, Any]) -> tuple:
    return (row["chunk_id"], row["source_id"], row["edge"], row["target_id"])


def compare_scores(local_rows: Iterable[Mapping[str, Any]], lambda_rows: Iterable[Mapping[str, Any]],
                   key_fn: Any, threshold: float) -> dict:
    """同じ組のスコアをローカルと Lambda で比べる。"""
    local = {key_fn(r): r for r in local_rows}
    remote = {key_fn(r): r for r in lambda_rows}
    common = [k for k in remote if k in local]
    diffs: list[float] = []
    flips: list[dict] = []
    for k in common:
        a, b = local[k].get("prob"), remote[k].get("prob")
        if a is None or b is None:
            continue
        diffs.append(abs(a - b))
        if (a >= threshold) != (b >= threshold):
            flips.append({"key": k, "local": a, "lambda": b, "row": remote[k]})
    flips.sort(key=lambda f: -abs(f["local"] - f["lambda"]))
    pref_hits = [{"key": k, "local": local[k]["prob"], "row": remote[k]} for k in common
                 if remote[k].get("prefiltered") and local[k].get("prob") is not None
                 and local[k]["prob"] >= threshold]
    pref_hits.sort(key=lambda f: -f["local"])
    return {
        "n_common": len(common),
        "only_local": sorted(k for k in local if k not in remote),
        "only_lambda": sorted(k for k in remote if k not in local),
        "mean_abs_diff": (sum(diffs) / len(diffs)) if diffs else None,
        "max_abs_diff": max(diffs) if diffs else None,
        "flips": flips,
        "prefiltered_hits": pref_hits,
    }


# ============================================================== 表示

def _fmt_diff(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_compare(label: str, cmp: dict, threshold: float, describe: Any) -> None:
    print(f"\n[{label}] ローカルとのスコア比較(しきい値 {threshold})")
    print(f"  共通の組 {cmp['n_common']} / ローカルだけ {len(cmp['only_local'])} / "
          f"Lambda だけ {len(cmp['only_lambda'])}")
    print(f"  |差| 平均 {_fmt_diff(cmp['mean_abs_diff'])} / 最大 {_fmt_diff(cmp['max_abs_diff'])}")
    if not cmp["flips"]:
        print("  しきい値をまたいで判定が変わった組: なし")
    else:
        print(f"  しきい値をまたいで判定が変わった組: {len(cmp['flips'])} 件")
        for f in cmp["flips"]:
            print(f"    ローカル {f['local']:.2f} → Lambda {f['lambda']:.2f}  {describe(f['row'])}")
    hits = cmp.get("prefiltered_hits") or []
    if hits:
        print(f"  Lambda で辞書の候補にならず問わなかったが、ローカルではしきい値以上だった組: {len(hits)} 件")
        for f in hits[:20]:
            print(f"    ローカル {f['local']:.2f}  {describe(f['row'])}")
        if len(hits) > 20:
            print(f"    …ほか {len(hits) - 20} 件")
    for name in ("only_local", "only_lambda"):
        if cmp[name]:
            shown = ", ".join("/".join(k) for k in cmp[name][:10])
            more = "" if len(cmp[name]) <= 10 else f" …ほか {len(cmp[name]) - 10} 件"
            print(f"  {'ローカルだけ' if name == 'only_local' else 'Lambda だけ'}: {shown}{more}")


def local_expect(master: Mapping[str, Any], chunks_path: Path = CHUNKS_PATH) -> dict:
    """event の `expect_data`: 手元のマスターの版とチャンクファイルの SHA-256。"""
    return {"master_version": master_version(master), "chunks_sha256": file_sha256(chunks_path)}


def dynamodb_master_problem(meta: Mapping[str, Any] | None, expect: Mapping[str, Any]) -> str | None:
    """DynamoDB の META と手元の版を比べる。問題が無ければ None、あれば説明の文。"""
    if meta is None:
        return ("DynamoDB にマスター(PK=MASTER)がありません。先に "
                "./.venv/bin/python scripts/60_put_master.py で入れてください")
    if meta.get("version") != expect["master_version"]:
        return (f"DynamoDB のマスターの版({str(meta.get('version'))[:8]})が手元({expect['master_version'][:8]})と"
                "違います。scripts/60_put_master.py で入れ直してください")
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    label = "invoke ingest"
    master = load_master()
    chunks = load_chunks()
    ents = entity_by_id(master)
    expect = local_expect(master)
    chunk_ids = [c["chunk_id"] for c in chunks]
    events = batch_events(chunk_ids, args.batch_chunks, reset=args.reset, node_th=args.node_th,
                          edge_th=args.edge_th, include_rows=not args.no_rows,
                          slim_records=not args.full_records, expect=expect, start=args.start_batch,
                          node_prefilter=args.node_prefilter)
    n_batches_all = len(invoke_batches(chunk_ids, args.batch_chunks))
    if not events:
        print(f"[{label}] --start-batch {args.start_batch} より後の組がありません(全 {n_batches_all} 組)")
        return 2
    target_ids = [cid for e in events for cid in e["chunk_ids"]]
    target_chunks = [c for c in chunks if c["chunk_id"] in set(target_ids)]

    # ---------------------------------------------------------------- 見積もり(AWS に触れない)
    if args.local_nodes_log is not None and args.local_nodes_log.exists():
        scores = node_scores_from_rows(iter_jsonl(args.local_nodes_log))
        present = {c["chunk_id"]: present_ids(scores.get(c["chunk_id"], {}), args.node_th) for c in chunks}
        lacking = [c["chunk_id"] for c in target_chunks if c["chunk_id"] not in scores]
        edge_note = f"関係判定は手元のノード判定ログ({args.local_nodes_log.name})の登場ノードから概算"
        if lacking:
            edge_note += f"。ログに無い {len(lacking)} チャンクは登場 0 とみなした(少なめに出る)"
    else:
        present = present_by_string(master, chunks)
        edge_note = "関係判定は正式名・別名の文字列一致を登場ノードとみなして概算"
    plans = [p for p in plan_invokes(master, chunks, present, args.batch_chunks,
                                     include_rows=not args.no_rows, slim_records=not args.full_records,
                                     node_prefilter=args.node_prefilter)
             if p["index"] >= args.start_batch]
    n_node_req = sum(p["node_requests"] for p in plans)
    n_edge_req = sum(p["edge_requests"] for p in plans)
    n_cands = sum(p["edge_candidates"] for p in plans)
    n_requests = n_node_req + n_edge_req
    est_usd = sum(p["est_usd"] for p in plans)
    est_tokens = sum(p["est_input_tokens"] for p in plans)
    print(f"[{label}] 見積もり(データ {DATASET}: エンティティ {len(master['entities'])} / チャンク {len(chunks)})")
    print(f"  関数         : {INGEST_FUNCTION}")
    print(f"  Invoke       : {len(events)} 回(全 {n_batches_all} 組のうち {args.start_batch} 番から。"
          f"1 回 {args.batch_chunks} チャンクまで。reset は {'最初の 1 回だけ' if events[0]['reset'] else 'なし'})")
    n_node_judg = sum(p["node_judgments"] for p in plans)
    node_tokens = sum(p["node_input_tokens"] for p in plans)
    edge_tokens = sum(p["edge_input_tokens"] for p in plans)
    print(f"  ノード判定   : {n_node_req} リクエスト / {n_node_judg:,} 判定(総当たりなら "
          f"{len(target_chunks) * len(master['entities']):,}。絞り込み {args.node_prefilter})/ "
          f"約 {node_tokens:,} トークン / ${sum(p['node_usd'] for p in plans):.4f}")
    print(f"  関係判定     : 約 {n_edge_req} リクエスト / 約 {n_cands:,} 候補 / 約 {edge_tokens:,} トークン / "
          f"${sum(p['edge_usd'] for p in plans):.4f}")
    print(f"                 ({edge_note})")
    print(f"  概算入力     : 約 {est_tokens:,} トークン(文字数 × 1.5)")
    print(f"  概算費用     : ${est_usd:.4f}")
    print(f"  {'組':>3}{'チャンク':>8}{'ノード':>7}{'関係':>6}{'候補':>7}{'所要(秒)':>10}{'応答(MB)':>10}  注意")
    warn = False
    for p in plans:
        notes = []
        if p["over_candidates"]:
            notes.append(f"候補が上限 {MAX_EDGE_CANDIDATES:,} 超え(Lambda 内で止まる)")
        if p["over_time"]:
            notes.append(f"所要が {FUNCTION_TIMEOUT_SEC} 秒の {TIME_WARN_RATIO:.0%} 超え")
        if p["over_payload"]:
            notes.append("応答が大きく判定ログが省かれる")
        warn |= bool(notes)
        print(f"  {p['index']:>3}{p['n_chunks']:>8}{p['node_requests']:>7}{p['edge_requests']:>6}"
              f"{p['edge_candidates']:>7,}{p['est_sec']:>10.0f}{p['est_payload_bytes'] / 1e6:>10.2f}  "
              f"{'、'.join(notes)}")
    print(f"  (所要は Jev 1 リクエスト {SEC_PER_JEV_REQUEST_EST} 秒・書き込み 1 チャンク "
          f"{SEC_PER_CHUNK_WRITE_EST} 秒とみた上振れ寄りの値)")
    if warn:
        print(f"  注意         : 上の組は --batch-chunks を小さくしてください")
    if args.reset:
        print("  注意         : --reset なので、最初の組の判定が終わったあと 2 テーブルの中身を全部消してから書きます")
    else:
        print("  注意         : --reset なし。前回の行が残ったまま上書き・追加されます")
    if args.dry_estimate:
        print(f"[{label}] --dry-estimate のため AWS には触れずに終わります", flush=True)
        return 0

    import boto3

    session = boto3.Session(region_name=REGION)
    require_home_region(__file__.rsplit('/', 1)[-1])  # 東京・大阪専用(JEV_REGION がそれ以外なら止める)
    check_account(session)                      # JEV_AWS_ACCOUNT_ID と違えばここで例外

    from graph_store import GraphStore

    problem = dynamodb_master_problem(
        GraphStore(session.resource("dynamodb", region_name=REGION,
                                    config=boto_config(max_attempts=5))).get_master_meta(), expect)
    if problem:
        print(f"[{label}] {problem}(Lambda は呼んでいません)", flush=True)
        return 2
    print(f"[{label}] DynamoDB のマスターの版 {expect['master_version'][:8]} は手元と一致")

    ensure_dirs()
    budget = Budget()
    budget.announce(n_requests, est_usd, label)
    budget.check(est_usd, label)
    if not args.yes and not confirm():
        print(f"[{label}] 中止しました(Lambda は呼んでいません)", flush=True)
        return 1

    # ---------------------------------------------------------------- Invoke(組の数だけ。各 1 回だけ)
    lam = session.client("lambda", region_name=REGION, config=invoke_config())
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jev_path = cache_path("jev")
    nodes_path = INGEST_DIR / f"lambda_nodes_{ts}.jsonl"
    edges_path = INGEST_DIR / f"lambda_edges_{ts}.jsonl"
    node_rows: list[dict] | None = []
    edge_rows: list[dict] | None = []
    all_records: list[dict] = []
    written_total = {"chunks": 0, "appearances": 0, "edges_out": 0, "edges_in": 0, "edges_sym": 0}
    deleted = None
    bodies: list[dict] = []
    wall_ms_total = 0.0
    for n, event in enumerate(events):
        idx = args.start_batch + n
        tag = f"{label} 組 {idx}/{n_batches_all - 1}"
        print(f"\n[{tag}] Invoke します({len(event['chunk_ids'])} チャンク {event['chunk_ids'][0]}〜"
              f"{event['chunk_ids'][-1]}、reset={event['reset']}。最大 {FUNCTION_TIMEOUT_SEC} 秒。"
              "自動では再 Invoke しません)", flush=True)
        resume = (f"./.venv/bin/python scripts/20_invoke_ingest.py --start-batch {idx} "
                  f"--batch-chunks {args.batch_chunks} --node-th {args.node_th} --edge-th {args.edge_th} "
                  f"--node-prefilter {args.node_prefilter}")
        started = time.perf_counter()
        try:
            resp = lam.invoke(FunctionName=INGEST_FUNCTION, InvocationType="RequestResponse",
                              Payload=json.dumps(event).encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] Invoke 失敗: {type(exc).__name__}。Lambda 側で走ったかは CloudWatch Logs で確認して"
                  "ください(この組の課金レコードは持ち帰れていません)", flush=True)
            print(f"  続きから: {resume}")
            return 2
        wall_ms = (time.perf_counter() - started) * 1000.0
        wall_ms_total += wall_ms
        raw = resp["Payload"].read()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[{tag}] 応答が JSON ではありません({len(raw)} bytes)", flush=True)
            print(f"  続きから: {resume}")
            return 2

        if resp.get("FunctionError"):
            # タイムアウト・メモリ不足・import 失敗など、ハンドラーの外で落ちたとき
            print(f"[{tag}] 関数エラー: {resp['FunctionError']} / {body.get('errorType')}: "
                  f"{str(body.get('errorMessage'))[:300]}", flush=True)
            print("  呼び出しレコードは持ち帰れていません。課金ぶんは CloudWatch Logs のサマリー行で確認してください")
            print(f"  続きから(--batch-chunks を小さくして): {resume}")
            return 2

        # 課金レコードを先に残す
        records = body.get("records") or []
        for record in records:
            append_jsonl(jev_path, record)
        all_records.extend(records)
        run_cost = sum(cost_of(r) for r in records)
        print(f"[{tag}] 呼び出しレコード {len(records)} 件を {jev_path} に追記(この組 ${run_cost:.6f})", flush=True)

        if not body.get("ok"):
            print(f"[{tag}] Lambda 内で失敗: {json.dumps(body.get('error'), ensure_ascii=False)}", flush=True)
            if body.get("data"):
                print(f"  Lambda のデータ: {json.dumps(body['data'], ensure_ascii=False)}")
            print(f"  累計費用 : ${Budget().spent():.6f}")
            print(f"  続きから: {resume}")
            return 2
        bodies.append(body)
        if body.get("deleted") is not None:
            deleted = body["deleted"]
        for key in written_total:
            written_total[key] += int((body.get("written") or {}).get(key) or 0)

        # 判定ログ(全組で 1 ファイルに追記)
        b_nodes, b_edges = body.get("node_rows"), body.get("edge_rows")
        if b_nodes is not None and b_edges is not None:
            for row in b_nodes:
                append_jsonl(nodes_path, row)
            for row in b_edges:
                append_jsonl(edges_path, row)
            if node_rows is not None and edge_rows is not None:
                node_rows.extend(b_nodes)
                edge_rows.extend(b_edges)
        else:
            node_rows = edge_rows = None            # 1 組でも欠けたら、全体の期待件数・比較は出さない

        lat = body.get("latency_ms") or {}
        counts = body.get("counts") or {}
        payload = body.get("payload") or {}
        print(f"  リクエスト {counts.get('jev_requests')}(再試行 {counts.get('retries')})/ "
              f"ノード判定 {counts.get('node_judgments')}(辞書で除外 {counts.get('node_prefiltered')})/ "
              f"関係候補 {counts.get('edge_candidates')} / スコア欠け {counts.get('missing_scores')} / "
              f"書いた {body.get('written')}")
        print(f"  所要 {lat.get('total', 0) / 1000:.1f} 秒(ノード {lat.get('nodes', 0) / 1000:.1f} / "
              f"関係 {lat.get('edges', 0) / 1000:.1f} / 書き込み {lat.get('write', 0) / 1000:.1f})、"
              f"応答 {payload.get('estimated_bytes', 0):,} bytes"
              f"{'(判定ログ省略: ' + str(payload['rows_omitted']) + ')' if payload.get('rows_omitted') else ''}",
              flush=True)

    # ---------------------------------------------------------------- まとめ
    run_cost = sum(cost_of(r) for r in all_records)
    served = sorted({str((r.get("response") or {}).get("model")) for r in all_records if r.get("response")})
    print(f"\n[{label}] まとめ({len(bodies)} 回の Invoke)")
    print(f"  データ       : {json.dumps(bodies[0].get('data'), ensure_ascii=False)}")
    print(f"  コールド     : {[b.get('cold_start') for b in bodies]}")
    print(f"  リクエスト   : {sum(int((b.get('counts') or {}).get('jev_requests') or 0) for b in bodies)}"
          f"(再試行 {sum(int((b.get('counts') or {}).get('retries') or 0) for b in bodies)} 回)")
    print(f"  入力トークン : {sum(int((b.get('counts') or {}).get('input_tokens') or 0) for b in bodies):,}")
    print(f"  消した件数   : {deleted}")
    print(f"  書いた件数   : {written_total}")
    print(f"  所要         : 最長 {max((b.get('latency_ms') or {}).get('total', 0) for b in bodies) / 1000:.1f} 秒 / "
          f"Invoke の往復の合計 {wall_ms_total / 1000:.1f} 秒")
    print(f"  応答サイズ   : 最大 {max((b.get('payload') or {}).get('estimated_bytes', 0) for b in bodies):,} bytes")
    if served and served != [JEV_MODEL]:
        print(f"  注意         : 応答のモデルが {served} です(期待値 {JEV_MODEL})")
    print(f"  今回の費用   : ${run_cost:.6f}(見積もり ${est_usd:.6f})")
    print(f"  累計費用     : ${Budget().spent():.6f}")
    if node_rows is not None:
        print(f"  判定ログ     : {nodes_path}")
        print(f"                 {edges_path}")

    # ---------------------------------------------------------------- 件数の突き合わせ
    from graph_store import GraphStore

    store = GraphStore(session.resource("dynamodb", region_name=REGION, config=boto_config(max_attempts=5)))
    actual = store.count_items()
    from_lambda = {"chunk": written_total["chunks"], "appearance": written_total["appearances"],
                   "out": written_total["edges_out"], "in": written_total["edges_in"],
                   "sym": 2 * written_total["edges_sym"]}   # SYM# は両端に 1 行ずつ(関係の件数の 2 倍)
    expected = (expected_counts(node_rows, edge_rows, args.node_th, args.edge_th, len(target_chunks),
                                symmetric_edge_names(master))
                if node_rows is not None and edge_rows is not None else {})
    whole = args.start_batch == 0 and args.reset
    print(f"\n[{label}] 件数の突き合わせ(期待 = 判定ログをしきい値で数えた値。Lambda = 全組の合計)")
    print(f"  {'種類':<12}{'期待':>8}{'Lambda':>8}{'DynamoDB':>10}  一致")
    all_ok = True
    for kind in ("chunk", "appearance", "out", "in", "sym"):
        exp, lam_n, act = expected.get(kind), from_lambda.get(kind), actual.get(kind)
        ok = (exp is None or exp == lam_n) and (act == lam_n if whole else act >= lam_n)
        all_ok &= ok
        print(f"  {kind:<12}{'-' if exp is None else exp:>8}{lam_n:>8}{act:>10}  "
              f"{'○' if ok else '×'}")
    if actual.get("out") != actual.get("in"):
        all_ok = False
        print("  × 順方向と逆方向の件数がずれています")
    if actual.get("sym", 0) % 2:
        all_ok = False
        print("  × 対称な関係(SYM)の行数が奇数です(両端に 1 行ずつのはず)")
    if actual.get("other"):
        all_ok = False
        print(f"  × 形の分からない行が {actual['other']} 件あります")
    if not whole:
        print("  (--reset なし・または続きからの実行なので、DynamoDB 側は前回・前の組の行を含み、多くなります)")
    print(f"  → {'全部一致' if all_ok else '不一致あり'}")

    # ---------------------------------------------------------------- 手元の判定ログと比較
    if node_rows is None or edge_rows is None:
        print(f"\n[{label}] 判定ログを持ち帰っていない組があるので、ローカルとの比較は省きます")
    else:
        nodes_log, edges_log = args.local_nodes_log, args.local_edges_log
        if nodes_log is not None and nodes_log.exists():
            cmp = compare_scores(iter_jsonl(nodes_log), node_rows, node_key, args.node_th)
            print_compare(f"{label} ノード", cmp, args.node_th,
                          lambda r: f"{r['chunk_id']} {r['entity_name']}({r['entity_id']})")
        else:
            print(f"\n[{label}] 手元のノード判定ログ(--local-nodes-log)が無いのでノードの比較は省きます")
        if edges_log is not None and edges_log.exists():
            cmp = compare_scores(iter_jsonl(edges_log), edge_rows, edge_key, args.edge_th)
            print_compare(f"{label} 関係", cmp, args.edge_th,
                          lambda r: f"{r['chunk_id']} {r['sentence']}")
            print("  (ローカルだけ/Lambda だけの組は、登場ノードの判定が揺れて候補文が変わったもの)")
        else:
            print(f"\n[{label}] 手元の関係判定ログ(--local-edges-log)が無いので関係の比較は省きます")

    # ---------------------------------------------------------------- ENT#luffy
    node = store.query_node(LOOK_ENTITY)
    name = lambda eid: ents.get(eid, {}).get("name", eid)  # noqa: E731
    print(f"\n[{label}] ENT#{LOOK_ENTITY}({name(LOOK_ENTITY)})の Query 結果")
    print(f"  登場チャンク ({len(node['appearances'])}):")
    for a in node["appearances"]:
        print(f"    {a['chunk_id']:<6} {a['score']:.2f}")
    print(f"  関係・順方向 OUT ({len(node['out'])}):")
    for e in node["out"]:
        print(f"    {e['score']:.2f}  {name(LOOK_ENTITY)} -[{e['edge']}]-> {name(e['target_id'])}  ({e['chunk_id']})")
    print(f"  関係・逆方向 IN ({len(node['in'])}):")
    for e in node["in"]:
        print(f"    {e['score']:.2f}  {name(e['source_id'])} -[{e['edge']}]-> {name(LOOK_ENTITY)}  ({e['chunk_id']})")
    print(f"  対称な関係 SYM ({len(node.get('sym') or [])}):")
    for e in node.get("sym") or []:
        print(f"    {e['score']:.2f}  {name(LOOK_ENTITY)} -[{e['edge']}]- {name(e['other_id'])}  ({e['chunk_id']})")
    return 0 if all_ok else 3


if __name__ == "__main__":
    sys.exit(main())
