#!/usr/bin/env python3
"""質問セットを 3 モードで query Lambda に流し、1 行 1 回のトレースを残す。

    ./.venv/bin/python scripts/30_run_eval.py --run-name dev1
    ./.venv/bin/python scripts/30_run_eval.py --run-name dev1 --qids q03 --modes adaptive --append
    ./.venv/bin/python scripts/30_run_eval.py --run-name dev1 --closed-book
    JEV_QUESTIONS=data/questions_test_v2.json ./.venv/bin/python scripts/30_run_eval.py --run-name test1

- 質問ファイルは環境変数 `JEV_QUESTIONS`(既定 `data/questions_v2.json`)
- 質問×モードを同時 2 本まで Invoke する(自動で再 Invoke しない)。Haiku のクォータに合わせ、
  Lambda 1 回 = Bedrock 最大 2 回と数えて 37 回/分以下に抑える
- 結果は `results/eval/<run>/raw.jsonl`、呼び出しレコードは `results/cache/` に追記
- `--closed-book`: Lambda を使わず、チャンクなしで Haiku に質問だけを投げる(比較用の基準)"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_common as ec  # noqa: E402
from bedrock_llm import ConverseCaller, converse_key, converse_payload  # noqa: E402
from budget import Budget, cost_of  # noqa: E402
from cache import JsonlCache  # noqa: E402
from common import (  # noqa: E402
    CACHE_DIR,
    DEFAULT_GEN_MODEL,
    ERRORS_JSONL_NAME,
    PRICES,
    REGION,
    append_jsonl,
    cache_path,
    check_account,
    ensure_dirs,
    load_json,
    save_json,
    utc_now_iso,
)
from master import load_chunks, load_master  # noqa: E402
from query_core import (  # noqa: E402
    ANSWER_STYLES,
    DEFAULT_ANSWER_STYLE,
    DEFAULT_ANSWER_FILTER,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_HOP_CHUNK_SOURCE,
    DEFAULT_MAX_CHUNKS_PER_HOP,
    DEFAULT_MAX_CHUNKS_PER_NODE,
    DEFAULT_MAX_HOPS,
    DEFAULT_ENTRY_PREFILTER,
    DEFAULT_ENTRY_FALLBACK,
    DEFAULT_SEMANTIC_ENTRY_THRESHOLD,
    ENTRY_FALLBACKS,
    ENTRY_PREFILTERS,
    HOP_CHUNK_SOURCES,
    TOKENS_PER_CHAR,
    estimate_usd,
)
from ratelimit import TokenBucket  # noqa: E402

q21 = ec.q21
QUERY_FUNCTION = q21.QUERY_FUNCTION

BEDROCK_PER_MIN = 37          # Haiku のクォータ 50/分の 75%
BEDROCK_PER_LAMBDA = 2        # query Lambda 1 回で Bedrock は最大 2 回(回答 + JSON の再依頼)
MAX_CONCURRENCY = 2

CLOSED_BOOK_SYSTEM = "知っている範囲で簡潔に答えてください。知らなければ「分かりません」と答えてください。"
CLOSED_BOOK_MAX_TOKENS = 512
CLOSED_BOOK_READ_TIMEOUT = 60


# ============================================================== 引数

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="評価: 質問×モードで query Lambda を呼ぶ")
    parser.add_argument("--modes", default=None,
                        help=f"カンマ区切り。選べるのは {', '.join(ec.MODE_ORDER)}(省略時は全部)")
    parser.add_argument("--qids", default="", help="カンマ区切り(省略時は質問ファイルの全部)")
    parser.add_argument("--run-name", default=None, help="results/eval/<run> の名前(省略時は UTC 時刻)")
    parser.add_argument("--entry-th", type=float, default=None, help="入口のしきい値(省略時は Lambda の既定 0.5)")
    parser.add_argument("--min-neighbor-th", type=float, default=None,
                        help="隣ノードへ進むスコアの下限(省略時は Lambda の既定 0.5)")
    parser.add_argument("--sufficiency-th", type=float, default=None,
                        help="十分性のしきい値(省略時は Lambda の既定 0.7)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help=f"集めるチャンクの全体の上限(省略時は Lambda の既定 {DEFAULT_MAX_CHUNKS})")
    parser.add_argument("--max-chunks-per-node", type=int, default=None,
                        help=f"1 ノードから取るチャンクの上限(省略時は Lambda の既定 {DEFAULT_MAX_CHUNKS_PER_NODE})")
    parser.add_argument("--max-chunks-per-hop", type=int, default=None,
                        help="1 回のホップで新しく足すチャンクの上限"
                             f"(省略時は Lambda の既定 {DEFAULT_MAX_CHUNKS_PER_HOP})")
    parser.add_argument("--hop-chunk-source", choices=HOP_CHUNK_SOURCES, default=None,
                        help="ホップ先のノードから取るチャンク。edge_evidence = たどった関係の根拠チャンク、"
                             f"appearance = 登場チャンク上位(旧動作)。省略時は Lambda の既定 {DEFAULT_HOP_CHUNK_SOURCE}")
    parser.add_argument("--entry-prefilter", choices=ENTRY_PREFILTERS, default=None,
                        help="前段判定で Jev に聞くエンティティの絞り方。dictionary = 名前・別名の辞書で拾った候補だけ、"
                             f"none = 全エンティティ(旧動作)。省略時は Lambda の既定 {DEFAULT_ENTRY_PREFILTER}")
    parser.add_argument("--entry-fallback", choices=ENTRY_FALLBACKS, default=None,
                        help="入口が 0 件だったときの聞き直し。semantic = 種別を選ばせてその種別のエンティティだけに"
                             "「同じものを指しているか」を聞く、none = 聞き直さない(旧動作)。"
                             f"省略時は Lambda の既定 {DEFAULT_ENTRY_FALLBACK}")
    parser.add_argument("--semantic-entry-th", type=float, default=None,
                        help=f"聞き直しの入口のしきい値(省略時は Lambda の既定 {DEFAULT_SEMANTIC_ENTRY_THRESHOLD})")
    parser.add_argument("--answer-style", choices=ANSWER_STYLES, default=None,
                        help="回答の書き方。concise = 答えだけを原則 1〜2 文・引用は最小限、full = 旧動作。"
                             f"省略時は event に入れない(Lambda の既定 {DEFAULT_ANSWER_STYLE}。"
                             "answer_style を知らない古い Lambda にもそのまま投げられる)")
    q21.add_filter_args(parser)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY,
                        help=f"同時 Invoke 数(1〜{MAX_CONCURRENCY})")
    parser.add_argument("--append", action="store_true",
                        help="既にある raw.jsonl / closed_book.jsonl に追記する(失敗分のやり直し用)")
    parser.add_argument("--closed-book", action="store_true",
                        help="Lambda を使わず、チャンクなしで Haiku に質問だけを投げる")
    parser.add_argument("--yes", action="store_true", help="実行前の Enter 確認を飛ばす")
    args = parser.parse_args(argv)

    raw_modes = args.modes if args.modes is not None else ",".join(ec.MODE_ORDER)
    modes = [m.strip() for m in raw_modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in ec.MODE_ORDER]
    if not modes or bad:
        parser.error(f"--modes が不正です: {bad or args.modes}(選べるのは {', '.join(ec.MODE_ORDER)})")
    args.modes = list(dict.fromkeys(modes))
    args.qids = [q.strip() for q in args.qids.split(",") if q.strip()]
    for name in ("sufficiency_th", "entry_th", "min_neighbor_th", "semantic_entry_th", "answer_filter_threshold"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} は 0〜1 です: {value}")
    for name in ("max_chunks", "max_chunks_per_node", "max_chunks_per_hop", "answer_filter_max"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 100:
            parser.error(f"--{name.replace('_', '-')} は 1〜100 です: {value}")
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        parser.error(f"--concurrency は 1〜{MAX_CONCURRENCY} です: {args.concurrency}")
    if args.run_name is None:
        args.run_name = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        ec.run_dir(args.run_name)
    except ValueError as exc:
        parser.error(str(exc))
    return args


# ============================================================== レート制限

def bedrock_bucket(per_call: int, per_min: int = BEDROCK_PER_MIN, *,
                   clock: Callable[[], float] = time.monotonic,
                   sleep: Callable[[float], None] = time.sleep) -> TokenBucket:
    """1 回の呼び出しで Bedrock を最大 `per_call` 回使うときのバケット。`acquire(per_call)` で使う。"""
    if not 0 < per_call < per_min:
        raise ValueError(f"per_call は 1〜{per_min - 1} です: {per_call}")
    return TokenBucket(per_min - per_call, burst=per_call, clock=clock, sleep=sleep)


def run_jobs(jobs: Sequence[Any], call: Callable[[Any], Any], *, bucket: TokenBucket, per_call: int,
             max_workers: int = MAX_CONCURRENCY) -> list[Any]:
    """jobs を同時 `max_workers` 本で回す。各 job の直前に `bucket.acquire(per_call)`。
    結果は入力順。`call` の例外はその位置に例外オブジェクトを入れる(1 件の失敗で止めない)。"""
    if not jobs:
        return []

    def guarded(job: Any) -> Any:
        bucket.acquire(per_call)
        return call(job)

    out: list[Any] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs)))) as pool:
        futures = [pool.submit(guarded, job) for job in jobs]
        for fut in futures:
            try:
                out.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                out.append(exc)
    return out


# ============================================================== 見積もり

def estimate_jobs(jobs: Sequence[Mapping[str, Any]], master: Mapping[str, Any],
                  chunk_texts: Sequence[str]) -> dict:
    n_calls = 0
    usd = 0.0
    for job in jobs:
        ev = q21.MODE_EVENTS[job["mode"]]
        est = estimate_usd(job["question"], master, chunk_texts, mode=ev["mode"],
                           max_hops=ev.get("max_hops", DEFAULT_MAX_HOPS),
                           max_chunks=(job.get("event") or {}).get("max_chunks", DEFAULT_MAX_CHUNKS),
                           entry_prefilter=(job.get("event") or {}).get("entry_prefilter",
                                                                       DEFAULT_ENTRY_PREFILTER),
                           entry_fallback=(job.get("event") or {}).get("entry_fallback",
                                                                      DEFAULT_ENTRY_FALLBACK),
                           answer_style=(job.get("event") or {}).get("answer_style", DEFAULT_ANSWER_STYLE),
                           answer_filter=(job.get("event") or {}).get("answer_filter", DEFAULT_ANSWER_FILTER))
        n_calls += est["jev_requests"] + est["gen_requests"]
        usd += est["est_usd"]
    return {"n_calls": n_calls, "est_usd": usd}


def make_jobs(questions: Sequence[Mapping[str, Any]], modes: Sequence[str],
              entry_th: float | None, sufficiency_th: float | None,
              min_neighbor_th: float | None = None, *, max_chunks: int | None = None,
              max_chunks_per_node: int | None = None,
              hop_chunk_source: str | None = None,
              max_chunks_per_hop: int | None = None,
              entry_prefilter: str | None = None, entry_fallback: str | None = None,
              semantic_entry_th: float | None = None,
              answer_style: str | None = None, hop_dup_policy: str | None = None,
              answer_filter: str | None = None, answer_filter_threshold: float | None = None,
              answer_filter_max: int | None = None) -> list[dict]:
    """質問×モード(質問の順 → モードの順)。"""
    jobs = []
    for q in questions:
        for m in modes:
            jobs.append({"qid": q["id"], "type": q.get("type"), "question": q["question"],
                         "required_chunks": list(q.get("required_chunks") or []),
                         "required_any": q.get("required_any"), "mode": m,
                         "method": "jev_graph",
                         "event": q21.build_event(q["question"], m, sufficiency_th, entry_th,
                                                  min_neighbor_th, max_chunks=max_chunks,
                                                  max_chunks_per_node=max_chunks_per_node,
                                                  hop_chunk_source=hop_chunk_source,
                                                  max_chunks_per_hop=max_chunks_per_hop,
                                                  entry_prefilter=entry_prefilter,
                                                  entry_fallback=entry_fallback,
                                                  semantic_entry_th=semantic_entry_th,
                                                  answer_style=answer_style,
                                                  hop_dup_policy=hop_dup_policy,
                                                  answer_filter=answer_filter,
                                                  answer_filter_threshold=answer_filter_threshold,
                                                  answer_filter_max=answer_filter_max)})
    return jobs


# ============================================================== Lambda 評価

class EvalWriter:
    """raw.jsonl と呼び出しレコードの追記(2 スレッドから呼ばれるので鍵をかける)。"""

    def __init__(self, raw_path: Path, cache_dir: Path | None = None) -> None:
        self.raw_path = raw_path
        self.cache_dir = cache_dir
        self.lock = threading.Lock()

    def records(self, records: Sequence[Mapping[str, Any]]) -> None:
        with self.lock:
            for record in records:
                kind = q21.ledger_kind(record)
                if kind is not None:
                    append_jsonl(cache_path(kind, self.cache_dir), dict(record))

    def row(self, row: Mapping[str, Any]) -> None:
        with self.lock:
            append_jsonl(self.raw_path, dict(row))


def invoke_one(lam: Any, job: Mapping[str, Any], writer: EvalWriter, run: str) -> dict:
    """1 回だけ Invoke(再試行なし)。課金レコードを先に追記し、raw.jsonl に 1 行書く。戻り値はその行。"""
    row: dict[str, Any] = {"run": run, "qid": job["qid"], "mode": job["mode"],
                           "method": job.get("method") or "jev_graph", "type": job["type"],
                           "question": job["question"], "required_chunks": job["required_chunks"],
                           "required_any": job.get("required_any"),
                           "event": job["event"], "ts": utc_now_iso(), "function": QUERY_FUNCTION, "region": REGION,
                           "status": None, "error": None, "wall_ms": None, "cost_usd": 0.0,
                           "required_check": None, "body": None}
    started = time.perf_counter()
    try:
        resp = lam.invoke(FunctionName=QUERY_FUNCTION, InvocationType="RequestResponse",
                          Payload=json.dumps(job["event"]).encode("utf-8"))
        raw = resp["Payload"].read()
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "invoke_error", "error": {"type": type(exc).__name__}})
        writer.row(row)
        return row
    row["wall_ms"] = (time.perf_counter() - started) * 1000.0
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        row.update({"status": "invoke_error", "error": {"type": "non-json", "bytes": len(raw)}})
        writer.row(row)
        return row
    if resp.get("FunctionError"):
        row.update({"status": "function_error",
                    "error": {"type": resp["FunctionError"], "errorType": body.get("errorType"),
                              "errorMessage": str(body.get("errorMessage"))[:300]}})
        writer.row(row)
        return row

    records = body.get("records") or []
    writer.records(records)                     # 課金レコードを先に残す
    row["cost_usd"] = sum(cost_of(r) for r in records)
    row["body"] = body
    if not body.get("ok"):
        row.update({"status": "lambda_error", "error": body.get("error")})
        writer.row(row)
        return row
    row["status"] = "ok"
    row["required_check"] = q21.required_check(body, job["required_chunks"])
    writer.row(row)
    return row


def print_row(row: Mapping[str, Any]) -> None:
    body = row.get("body") or {}
    head = f"  {row['qid']} {row['mode']:<9}"
    if row["status"] != "ok":
        print(f"{head}: !! {row['status']} {json.dumps(row.get('error'), ensure_ascii=False)[:200]}"
              f"(${row.get('cost_usd', 0):.6f} は追記済み)", flush=True)
        return
    req = row.get("required_check") or {}
    print(f"{head}: stop={body.get('stop_reason')} / ホップ {ec.last_hop(body)} / "
          f"チャンク {len(body.get('chunks') or [])}(渡した {len(ec.answer_chunk_ids(body))})/ 引用OK={body.get('citations_valid')} / "
          f"required 収集={req.get('collected_cover_required')}・引用={req.get('citations_cover_required')} / "
          f"{ec.fmt((body.get('latency_ms') or {}).get('total'), 0)}ms / ${row['cost_usd']:.6f}", flush=True)


def prepare_out(path: Path, append: bool) -> None:
    if path.exists() and path.stat().st_size > 0 and not append:
        raise FileExistsError(f"{path} が既にあります。別の --run-name にするか、追記なら --append を付けてください")


def write_meta(out_dir: Path, entry: Mapping[str, Any]) -> None:
    path = out_dir / ec.META_NAME
    meta = load_json(path, default={"runs": []}) or {"runs": []}
    meta.setdefault("runs", []).append(dict(entry))
    save_json(path, meta)


def confirm() -> bool:
    return q21.confirm()


def main_lambda(args: argparse.Namespace, questions: list[dict], master: dict, chunks: list[dict]) -> int:
    label = "run eval"
    out_dir = ec.run_dir(args.run_name)
    raw_path = out_dir / ec.RAW_NAME
    try:
        prepare_out(raw_path, args.append)
    except FileExistsError as exc:
        print(f"[{label}] {exc}")
        return 1

    jobs = make_jobs(questions, args.modes, args.entry_th, args.sufficiency_th, args.min_neighbor_th,
                     max_chunks=args.max_chunks, max_chunks_per_node=args.max_chunks_per_node,
                     hop_chunk_source=args.hop_chunk_source, max_chunks_per_hop=args.max_chunks_per_hop,
                     entry_prefilter=args.entry_prefilter, entry_fallback=args.entry_fallback,
                     semantic_entry_th=args.semantic_entry_th, answer_style=args.answer_style,
                     **q21.filter_kwargs(args))

    import boto3

    session = boto3.Session(region_name=REGION)
    check_account(session)                      # JEV_AWS_ACCOUNT_ID と違えばここで例外

    est = estimate_jobs(jobs, master, [c["text"] for c in chunks])
    n_lambda = len(jobs)
    per_lambda = BEDROCK_PER_LAMBDA
    min_minutes = max(n_lambda - 1, 0) * per_lambda / (BEDROCK_PER_MIN - per_lambda)
    print(f"[{label}] 見積もり(上振れ寄り)")
    print(f"  関数     : {QUERY_FUNCTION}(リージョン {REGION})")
    print(f"  run      : {args.run_name} → {raw_path}{'(追記)' if args.append else ''}")
    print(f"  質問     : {', '.join(q['id'] for q in questions)}({len(questions)} 問)")
    print(f"  モード   : {', '.join(args.modes)}")
    print(f"  しきい値 : entry_th={args.entry_th if args.entry_th is not None else '既定'} / "
          f"sufficiency_th={args.sufficiency_th if args.sufficiency_th is not None else '既定'} / "
          f"min_neighbor_th={args.min_neighbor_th if args.min_neighbor_th is not None else '既定'}")
    print(f"  チャンク : max_chunks={args.max_chunks if args.max_chunks is not None else '既定'} / "
          f"max_chunks_per_node={args.max_chunks_per_node if args.max_chunks_per_node is not None else '既定'} / "
          f"max_chunks_per_hop={args.max_chunks_per_hop if args.max_chunks_per_hop is not None else '既定'} / "
          f"hop_chunk_source={args.hop_chunk_source or '既定'}")
    print(f"  前段判定 : entry_prefilter={args.entry_prefilter or '既定'} / "
          f"entry_fallback={args.entry_fallback or '既定'} / "
          f"semantic_entry_th={args.semantic_entry_th if args.semantic_entry_th is not None else '既定'}")
    print(f"  回答     : answer_style={args.answer_style or '既定'} / "
          f"hop_dup_policy={args.hop_dup_policy or '既定'} / answer_filter={args.answer_filter or '既定'} / "
          f"answer_filter_threshold={args.answer_filter_threshold if args.answer_filter_threshold is not None else '既定'} / "
          f"answer_filter_max={args.answer_filter_max if args.answer_filter_max is not None else '既定'}")
    print(f"  Invoke   : {n_lambda} 回(同時 {args.concurrency} 本、自動再試行なし)")
    print(f"  Bedrock  : 最大 {n_lambda * per_lambda} 回を {BEDROCK_PER_MIN} 回/分以下に抑える"
          f"(最短でも約 {min_minutes:.1f} 分)")
    ensure_dirs()
    budget = Budget()
    budget.announce(est["n_calls"], est["est_usd"], label)
    budget.check(est["est_usd"], label)
    if not args.yes and not confirm():
        print(f"[{label}] 中止しました(Lambda は呼んでいません)", flush=True)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    lam = session.client("lambda", region_name=REGION, config=q21.invoke_config())
    writer = EvalWriter(raw_path)
    bucket = bedrock_bucket(per_lambda)

    def call(job: Mapping[str, Any]) -> dict:
        row = invoke_one(lam, job, writer, args.run_name)
        print_row(row)
        return row

    print(f"[{label}] 実行")
    results = run_jobs(jobs, call, bucket=bucket, per_call=per_lambda, max_workers=args.concurrency)

    n_ok = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "ok")
    unexpected = [r for r in results if isinstance(r, Exception)]
    total_cost = sum(r.get("cost_usd", 0.0) for r in results if isinstance(r, dict))
    write_meta(out_dir, {"kind": "lambda", "started_at": started_at, "finished_at": utc_now_iso(),
                         "region": REGION, "function": QUERY_FUNCTION,
                         "method": "jev_graph",
                         "modes": args.modes, "qids": [q["id"] for q in questions],
                         "entry_th": args.entry_th, "sufficiency_th": args.sufficiency_th,
                         "min_neighbor_th": args.min_neighbor_th,
                         "max_chunks": args.max_chunks, "max_chunks_per_node": args.max_chunks_per_node,
                         "hop_chunk_source": args.hop_chunk_source,
                         "max_chunks_per_hop": args.max_chunks_per_hop,
                         "entry_prefilter": args.entry_prefilter,
                         "entry_fallback": args.entry_fallback,
                         "semantic_entry_th": args.semantic_entry_th,
                         "answer_style": args.answer_style,
                         **q21.filter_kwargs(args),
                         "append": args.append, "n_jobs": n_lambda, "n_ok": n_ok,
                         "cost_usd": total_cost, "est_usd": est["est_usd"]})
    print(f"\n[{label}] まとめ")
    print(f"  成功       : {n_ok}/{n_lambda}")
    for exc in unexpected:
        print(f"  !! スクリプト側の例外: {type(exc).__name__}: {exc}")
    print(f"  今回の費用 : ${total_cost:.6f}(見積もり ${est['est_usd']:.4f})")
    print(f"  累計費用   : ${Budget().spent():.6f}")
    print(f"  保存先     : {raw_path}")
    return 0 if n_ok == n_lambda else 2


# ============================================================== クローズドブック

def closed_book_task(qid: str, question: str, model: str = DEFAULT_GEN_MODEL) -> dict:
    """チャンクを渡さない、質問だけの Converse タスク(`ConverseCaller.run` に渡す形)。"""
    payload = converse_payload(CLOSED_BOOK_SYSTEM, question, temperature=0.0,
                               max_tokens=CLOSED_BOOK_MAX_TOKENS)
    return {"key": converse_key(model, payload, kind="gen"), "system": CLOSED_BOOK_SYSTEM,
            "user": question, "payload": payload,
            "extra": {"query_id": qid, "method": "closed-book"}}


def estimate_closed_book(questions: Sequence[Mapping[str, Any]]) -> float:
    price = PRICES[DEFAULT_GEN_MODEL]
    usd = 0.0
    for q in questions:
        tin = int((len(CLOSED_BOOK_SYSTEM) + len(q["question"])) * TOKENS_PER_CHAR) + 20
        usd += tin * price["input"] / 1e6 + CLOSED_BOOK_MAX_TOKENS * price["output"] / 1e6
    return usd


class _NoReadTimeoutRetry:
    """`converse` の ReadTimeoutError を再試行されない例外に替える(課金済みの可能性があるため。
    query Lambda の `NoReadTimeoutRetry` と同じ考え方。Lambda のコードは import しない)。"""

    def __init__(self, client: Any) -> None:
        self._client = client

    def converse(self, **kwargs: Any) -> Any:
        try:
            return self._client.converse(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "ReadTimeoutError":
                raise RuntimeError("Bedrock の応答が返りませんでした(課金済みの可能性があるので再試行しない)") from None
            raise


def run_closed_book(questions: Sequence[Mapping[str, Any]], caller: ConverseCaller, out_path: Path,
                    run: str) -> list[dict]:
    """1 問 1 回。結果を out_path に 1 行ずつ追記して返す。"""
    rows = []
    for q in questions:
        task = closed_book_task(q["id"], q["question"], caller.model)
        cached_before = caller.n_cached
        record = caller.run(task, use_cache=True)
        cached = caller.n_cached > cached_before
        row: dict[str, Any] = {"run": run, "qid": q["id"], "type": q.get("type"), "question": q["question"],
                               "system": CLOSED_BOOK_SYSTEM, "model": caller.model, "ts": utc_now_iso(),
                               "cached": cached, "answer": None, "stop_reason": None, "usage": None,
                               "latency_ms": None, "cost_usd": 0.0, "error": None}
        if record is None:
            fail = caller.fatal or (caller.failures[-1]["error"] if caller.failures else {"type": "unknown"})
            row["error"] = fail
        else:
            response = record.get("response") or {}
            row.update({"answer": response.get("text"), "stop_reason": response.get("stop_reason"),
                        "usage": record.get("usage"), "latency_ms": record.get("latency_ms"),
                        "cost_usd": 0.0 if cached else cost_of(record)})
        append_jsonl(out_path, row)
        rows.append(row)
        tag = "(キャッシュ)" if cached else ""
        print(f"  {q['id']}: {str(row['answer'] or row['error'])[:120]}{tag}", flush=True)
    return rows


def main_closed_book(args: argparse.Namespace, questions: list[dict]) -> int:
    label = "closed book"
    out_dir = ec.run_dir(args.run_name)
    out_path = out_dir / ec.CLOSED_BOOK_NAME
    try:
        prepare_out(out_path, args.append)
    except FileExistsError as exc:
        print(f"[{label}] {exc}")
        return 1

    import boto3
    from botocore.config import Config

    session = boto3.Session(region_name=REGION)
    check_account(session)

    ensure_dirs()
    cache = JsonlCache(cache_path("gen"))
    to_call = [q for q in questions if closed_book_task(q["id"], q["question"])["key"] not in cache]
    est_usd = estimate_closed_book(to_call)
    print(f"[{label}] 見積もり(上振れ寄り)")
    print(f"  モデル   : {DEFAULT_GEN_MODEL}(temperature 0、チャンクは渡さない)")
    print(f"  system   : {CLOSED_BOOK_SYSTEM}")
    print(f"  質問     : {', '.join(q['id'] for q in questions)}"
          f"(うちキャッシュ済み {len(questions) - len(to_call)} 問は呼ばない)")
    print(f"  保存先   : {out_path}{'(追記)' if args.append else ''}")
    budget = Budget()
    budget.announce(len(to_call), est_usd, label)
    budget.check(est_usd, label)
    if not args.yes and not confirm():
        print(f"[{label}] 中止しました(Bedrock は呼んでいません)", flush=True)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    client = session.client("bedrock-runtime", region_name=REGION,
                            config=Config(read_timeout=CLOSED_BOOK_READ_TIMEOUT, connect_timeout=5,
                                          retries={"total_max_attempts": 1, "mode": "standard"}))
    caller = ConverseCaller(client=_NoReadTimeoutRetry(client), model=DEFAULT_GEN_MODEL, kind="gen",
                            cache=cache, errors_path=CACHE_DIR / ERRORS_JSONL_NAME,
                            bucket=bedrock_bucket(1), label="closed-book",
                            retry={"max_attempts": 3, "max_delay": 5.0})
    print(f"[{label}] 実行")
    rows = run_closed_book(questions, caller, out_path, args.run_name)
    cost = sum(r["cost_usd"] for r in rows)
    write_meta(out_dir, {"kind": "closed_book", "finished_at": utc_now_iso(), "region": REGION,
                         "qids": [q["id"] for q in questions], "n_called": caller.n_called,
                         "n_cached": caller.n_cached, "cost_usd": cost})
    print(f"\n[{label}] 今回の費用 ${cost:.6f}(見積もり ${est_usd:.4f})/ 累計 ${Budget().spent():.6f}")
    print(f"  保存先: {out_path}")
    return 0 if all(r["error"] is None for r in rows) else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    master = load_master()
    chunks = load_chunks()
    try:
        questions = ec.load_questions(args.qids or None, chunk_ids={c["chunk_id"] for c in chunks})
    except ValueError as exc:
        print(f"[run eval] {exc}")
        return 1
    if not questions:
        print("[run eval] 質問がありません")
        return 1
    if args.closed_book:
        return main_closed_book(args, questions)
    return main_lambda(args, questions, master, chunks)


if __name__ == "__main__":
    sys.exit(main())
