#!/usr/bin/env python3
"""query Lambda(`jev-graphrag-query`)を同期 Invoke し、トレースと回答を表示する。

    ./.venv/bin/python scripts/21_invoke_query.py --qid q03
    ./.venv/bin/python scripts/21_invoke_query.py --qid q03 --modes adaptive --yes
    ./.venv/bin/python scripts/21_invoke_query.py --question "ルフィの祖父は誰か？" --modes adaptive

モード: `fixed1` = fixed・max_hops=1、`fixed2` = fixed・max_hops=2、`adaptive` = adaptive・max_hops=3(既定)

`--qid` は `data/questions_v2.json` から質問を引く。見積もり → 予算の確認 → Invoke(自動で再 Invoke しない)の順に進み、
呼び出しレコードを `results/cache/` に追記して、結果を `results/query/<qid>_<ts>.json` に保存する。
Invoke するリージョンは環境変数 `JEV_REGION`(既定 ap-northeast-1)。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

LIB_DIR = Path(__file__).resolve().parent.parent / "lib"
_lib = str(LIB_DIR)
if _lib in sys.path:
    sys.path.remove(_lib)
sys.path.insert(0, _lib)   # 同名モジュールが他所にあっても lib/ を優先させる

from budget import Budget, cost_of  # noqa: E402
from common import (  # noqa: E402
    DATA_DIR,
    REGION,
    RESULTS_DIR,
    append_jsonl,
    cache_path,
    check_account,
    ensure_dirs,
    load_json,
    save_json,
)
from master import entity_by_id, load_chunks, load_master  # noqa: E402
from query_core import (  # noqa: E402
    DEFAULT_ANSWER_STYLE,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_HOP_CHUNK_SOURCE,
    DEFAULT_MAX_CHUNKS_PER_HOP,
    DEFAULT_MAX_CHUNKS_PER_NODE,
    DEFAULT_MAX_HOPS,
    DEFAULT_ENTRY_PREFILTER,
    DEFAULT_ENTRY_FALLBACK,
    DEFAULT_SEMANTIC_ENTRY_THRESHOLD,
    DEFAULT_ANSWER_FILTER,
    DEFAULT_ANSWER_FILTER_MAX,
    DEFAULT_ANSWER_FILTER_THRESHOLD,
    DEFAULT_HOP_DUP_POLICY,
    ANSWER_FILTERS,
    ENTRY_FALLBACKS,
    ENTRY_PREFILTERS,
    HOP_CHUNK_SOURCES,
    HOP_DUP_POLICIES,
    estimate_usd,
)  # noqa: E402
QUERY_FUNCTION = "jev-graphrag-query"
QUESTIONS_PATH = DATA_DIR / "questions_v2.json"
QUERY_DIR = RESULTS_DIR / "query"

FUNCTION_TIMEOUT_SEC = 120
READ_TIMEOUT_SEC = FUNCTION_TIMEOUT_SEC + 30

MODE_EVENTS: dict[str, dict] = {
    "fixed1": {"mode": "fixed", "max_hops": 1},
    "fixed2": {"mode": "fixed", "max_hops": 2},
    "adaptive": {"mode": "adaptive"},
}
DEFAULT_MODES = "fixed1,fixed2,adaptive"
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="query Lambda を同期 Invoke して結果を見る")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--qid", help="data/questions_v2.json の id(例 q03)")
    src.add_argument("--question", help="質問文を直接渡す(required_chunks との照合はしない)")
    parser.add_argument("--modes", default=None,
                        help=f"カンマ区切り。選べるのは {', '.join(MODE_EVENTS)}(省略時は {DEFAULT_MODES})")
    parser.add_argument("--sufficiency-th", type=float, default=None,
                        help="十分性のしきい値(省略時は Lambda の既定 0.7)")
    parser.add_argument("--entry-th", type=float, default=None,
                        help="入口のしきい値(省略時は Lambda の既定 0.5)")
    parser.add_argument("--min-neighbor-th", type=float, default=None,
                        help="隣ノードへ進むスコアの下限(省略時は Lambda の既定 0.5)")
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
    add_filter_args(parser)
    parser.add_argument("--yes", action="store_true", help="実行前の Enter 確認を飛ばす")
    args = parser.parse_args(argv)
    modes = [m.strip() for m in (args.modes if args.modes is not None else DEFAULT_MODES).split(",") if m.strip()]
    bad = [m for m in modes if m not in MODE_EVENTS]
    if not modes or bad:
        parser.error(f"--modes が不正です: {bad or args.modes}(選べるのは {', '.join(MODE_EVENTS)})")
    args.modes = list(dict.fromkeys(modes))
    for name in ("sufficiency_th", "entry_th", "min_neighbor_th", "semantic_entry_th", "answer_filter_threshold"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} は 0〜1 です: {value}")
    for name in ("max_chunks", "max_chunks_per_node", "max_chunks_per_hop", "answer_filter_max"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 100:
            parser.error(f"--{name.replace('_', '-')} は 1〜100 です: {value}")
    return args


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    """重複の数え方と回答前の絞り込みの引数(30 番と共通。どれも省略時は event に入れない)。"""
    parser.add_argument("--hop-dup-policy", choices=HOP_DUP_POLICIES, default=None,
                        help="各ホップで既に集めたチャンクの数え方。count = 上位 max_chunks_per_hop 件をそのまま取り"
                             "新顔だけ足す、skip = 既に集めたものを飛ばして新しいものを足す(旧動作)。"
                             f"省略時は Lambda の既定 {DEFAULT_HOP_DUP_POLICY}")
    parser.add_argument("--answer-filter", choices=ANSWER_FILTERS, default=None,
                        help="回答前の絞り込み。jev = 累積チャンクごとに Jev に聞いて上位だけ Haiku に渡す、"
                             f"none = 全部渡す。省略時は Lambda の既定 {DEFAULT_ANSWER_FILTER}")
    parser.add_argument("--answer-filter-threshold", type=float, default=None,
                        help=f"回答前の絞り込みで残すスコアの下限(省略時は Lambda の既定 {DEFAULT_ANSWER_FILTER_THRESHOLD})")
    parser.add_argument("--answer-filter-max", type=int, default=None,
                        help=f"回答前の絞り込みで残す件数の上限(省略時は Lambda の既定 {DEFAULT_ANSWER_FILTER_MAX})")


def ledger_kind(record: Mapping[str, Any]) -> str:
    """呼び出しレコードを追記する台帳(`results/cache/<kind>.jsonl`)。Jev → jev、それ以外(Haiku)→ gen。"""
    kind = record.get("kind")
    if kind == "jev":
        return "jev"
    return "gen"


def build_event(question: str, mode_name: str | None, sufficiency_th: float | None,
                entry_th: float | None = None, min_neighbor_th: float | None = None, *,
                max_chunks: int | None = None, max_chunks_per_node: int | None = None,
                hop_chunk_source: str | None = None, max_chunks_per_hop: int | None = None,
                entry_prefilter: str | None = None, entry_fallback: str | None = None,
                semantic_entry_th: float | None = None, answer_style: str | None = None,
                hop_dup_policy: str | None = None, answer_filter: str | None = None,
                answer_filter_threshold: float | None = None, answer_filter_max: int | None = None) -> dict:
    """Lambda に送る event。指定した(None でない)パラメータだけを入れる(省略は Lambda の既定)。"""
    event = {"question": question, **MODE_EVENTS[mode_name]}
    if sufficiency_th is not None:
        event["sufficiency_th"] = sufficiency_th
    if entry_th is not None:
        event["entry_th"] = entry_th
    if min_neighbor_th is not None:
        event["min_neighbor_th"] = min_neighbor_th
    if max_chunks is not None:
        event["max_chunks"] = max_chunks
    if max_chunks_per_node is not None:
        event["max_chunks_per_node"] = max_chunks_per_node
    if hop_chunk_source is not None:
        event["hop_chunk_source"] = hop_chunk_source
    if max_chunks_per_hop is not None:
        event["max_chunks_per_hop"] = max_chunks_per_hop
    if entry_prefilter is not None:
        event["entry_prefilter"] = entry_prefilter
    if entry_fallback is not None:
        event["entry_fallback"] = entry_fallback
    if semantic_entry_th is not None:
        event["semantic_entry_th"] = semantic_entry_th
    if answer_style is not None:                 # 指定したときだけ送る(古い Lambda は知らないキーを断るため)
        event["answer_style"] = answer_style
    # 以下も指定したときだけ送る(知らない古い Lambda にもそのまま投げられるように)
    if hop_dup_policy is not None:
        event["hop_dup_policy"] = hop_dup_policy
    if answer_filter is not None:
        event["answer_filter"] = answer_filter
    if answer_filter_threshold is not None:
        event["answer_filter_threshold"] = answer_filter_threshold
    if answer_filter_max is not None:
        event["answer_filter_max"] = answer_filter_max
    return event


def filter_kwargs(args: argparse.Namespace) -> dict:
    """`build_event()` に渡す重複の数え方・回答前の絞り込みの引数。"""
    return {"hop_dup_policy": args.hop_dup_policy, "answer_filter": args.answer_filter,
            "answer_filter_threshold": args.answer_filter_threshold,
            "answer_filter_max": args.answer_filter_max}


def format_answer_filter(body: Mapping[str, Any]) -> list[str]:
    """回答前の絞り込みの表示行(絞らなかったら空)。"""
    af = body.get("answer_filter")
    if not af:
        return []
    def row(r: Mapping[str, Any]) -> str:
        return f"{r['id']}({_f(r.get('score'))}・{r.get('hits')}回)"
    head = (f"{af.get('n_in')} 件 → {af.get('n_out')} 件(しきい値 {_f(af.get('threshold'))}・最大 {af.get('max')})"
            + (f"  !! 絞らずに全部渡した: {af['fallback']}" if af.get("fallback") else "")
            + ("  (しきい値以上が 0 件なので上位を残した)" if af.get("min_applied") else ""))
    return [head,
            "  渡した   : " + (", ".join(row(r) for r in af.get("kept") or []) or "(なし)"),
            "  渡さない : " + (", ".join(row(r) for r in af.get("dropped") or []) or "(なし)")]


VIA_LABELS = {"dictionary": "辞書", "semantic": "聞き直し"}


def format_entry(entry: Sequence[Mapping[str, Any]], names: Mapping[str, str]) -> list[str]:
    """入口候補の表示行。選ばれたものは「→」、選ばれなかったものは「(選外)」を付ける。
    意味で聞き直して見つけた入口(via=semantic)は「[聞き直し]」、辞書の候補は「[辞書]」を前に付ける
    (via の無い古いトレースは付けない)。"""
    lines = []
    for e in entry:
        mark = "→" if e.get("selected") else " "
        tail = "" if e.get("selected") else "  (選外)"
        via = VIA_LABELS.get(e.get("via"))
        tag = f"[{via}] " if via else ""
        lines.append(f"{mark} {tag}{e['score']:.2f} {names.get(e['id'], e['id'])}({e['id']}){tail}")
    return lines


def format_fallback(body: Mapping[str, Any]) -> list[str]:
    """聞き直しの表示行(聞き直さなかったら空)。種別のスコアと、問うた種別・問いの数・しきい値。"""
    if not body.get("entry_fallback_used"):
        return []
    fb = body.get("entry_fallback") or {}
    types = " / ".join(f"{t['type']} {_f(t.get('score'))}" for t in (fb.get("type_scores") or [])[:4])
    selected = "、".join(fb.get("selected_types") or []) or "(なし)"
    found = sum(1 for e in body.get("entry") or [] if e.get("via") == "semantic" and e.get("selected"))
    return [f"辞書で入口が見つからなかったので、意味で聞き直した(しきい値 {_f(fb.get('threshold'))})",
            f"  種別の判定 : {types or '(取れず)'}",
            f"  問うた種別 : {selected}({fb.get('n_questions', 0)} 件に質問)→ 入口 {found} 件"]


def invoke_config() -> Any:
    """Lambda 用の botocore Config。**再試行なし**(`total_max_attempts=1`。20 番と同じ理由)。"""
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

def load_question(qid: str, chunk_ids: set[str], path: Path = QUESTIONS_PATH) -> dict:
    """質問ファイルから 1 問。required_chunks がチャンクに無ければ ValueError。"""
    questions = load_json(path, default=[])
    found = [q for q in questions if q.get("id") == qid]
    if not found:
        raise ValueError(f"{path.name} に {qid!r} がありません")
    q = found[0]
    unknown = [c for c in q.get("required_chunks") or [] if c not in chunk_ids]
    if unknown:
        raise ValueError(f"{qid} の required_chunks が chunks.json にありません: {unknown}")
    return q


def required_check(body: Mapping[str, Any], required: Sequence[str]) -> dict:
    """required_chunks と、集めたチャンク・引用を照合する。"""
    collected = {c["id"]: c for c in body.get("chunks") or []}
    cited = {cid for c in body.get("citations") or [] for cid in c.get("chunk_ids") or []}
    req = list(required)
    return {
        "required": req,
        "collected_cover_required": all(c in collected for c in req),
        "citations_cover_required": all(c in cited for c in req),
        "missing_in_collected": [c for c in req if c not in collected],
        "missing_in_citations": [c for c in req if c not in cited],
        "first_hop": {c: (collected[c]["hop"] if c in collected else None) for c in req},
    }


# ============================================================== 表示

def _f(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_run(mode_name: str, body: Mapping[str, Any], names: Mapping[str, str],
              req: dict | None, cost: float, wall_ms: float) -> None:
    name = lambda eid: names.get(eid, eid)  # noqa: E731
    print(f"\n==================== {mode_name}(mode={body.get('mode')} / "
          f"max_hops={(body.get('params') or {}).get('max_hops')})")
    print(f"  in_scope     : {_f(body.get('in_scope'))}"
          + ("  (しきい値未満/取れずだが入口があるので続行)" if body.get("in_scope_overridden") else ""))
    entry_th = (body.get("params") or {}).get("entry_threshold")
    lines = format_entry(body.get("entry") or [], names)
    print(f"  入口候補     : しきい値 {_f(entry_th)}(スコア 0.1 以上を全部表示)")
    for line in lines or ["(なし)"]:
        print(f"    {line}")
    for line in format_fallback(body):
        print(f"  聞き直し     : {line}" if not line.startswith("  ") else f"               {line.strip()}")
    for h in body.get("hops") or []:
        print(f"  -- ホップ {h['hop']}")
        if h["hop"] > 0:
            print(f"     frontier  : {', '.join(name(e) for e in h.get('frontier') or [])}")
            scores = h.get("neighbor_scores") or []
            min_th = (body.get("params") or {}).get("min_neighbor_score")
            n_below = sum(1 for s in scores if s.get("excluded") == "below_min")
            print(f"     候補      : {len(scores)} 件(下限 {_f(min_th)} 未満 {n_below} 件。上位 8 件を表示)")
            for s in scores[:8]:
                mark = "→" if s["entity"] in (h.get("selected") or []) else " "
                tail = "  (下限未満)" if s.get("excluded") == "below_min" else ""
                print(f"      {mark} {_f(s.get('score'))} [{s['direction']}] {s['sentence']}{tail}")
        print(f"     進んだ    : {', '.join(name(e) for e in h.get('selected') or []) or '(なし)'}"
              + ("(関係の根拠チャンクを取得)" if h.get("chunk_source") == "edge_evidence" and h.get("hop") else ""))
        print(f"     新規チャンク: {h.get('new_chunk_ids')}"
              + (f" / 切り捨て {h['dropped_chunk_ids']}" if h.get("dropped_chunk_ids") else "")
              + (f" / 見つからず {h['missing_chunk_ids']}" if h.get("missing_chunk_ids") else ""))
        for eid, capped in (h.get("capped_chunk_ids") or {}).items():
            total = (h.get("node_chunk_counts") or {}).get(eid)
            what = "関係の根拠" if h.get("chunk_source") == "edge_evidence" else "登場"
            print(f"     1ノード上限: {name(eid)} は{what} {total if total is not None else '?'} 件のうち"
                  f"上位のみ。外した {capped}")
        if h.get("hop_capped_chunk_ids"):
            per_hop = (body.get("params") or {}).get("max_chunks_per_hop", "?")
            print(f"     1ホップ上限: このホップで足すのは {per_hop} 件まで(スコア順)。"
                  f"あふれた {h['hop_capped_chunk_ids']}")
        suff = "-" if h.get("sufficiency") is None else f"{h['sufficiency']:.3f}"
        print(f"     十分性    : {suff}{'(累積が同じなので前の値)' if h.get('sufficiency_reused') else ''}")
    print("  集めたチャンク(累積順):")
    for c in body.get("chunks") or []:
        also = "; ".join(f"h{a['hop']} {name(a['entity'])}/{a['edge'] or '入口'}" for a in c.get("also_via") or [])
        ev = "; ".join(f"{name(r['from'])}→{name(r['to'])}" for r in c.get("evidence_of") or [])
        print(f"    {c['id']:<6} hop {c['hop']}  via {name(c['via_entity'])} / {c['via_edge'] or '入口'}"
              + (f"  [根拠: {ev}]" if ev else "")
              + (f"  (ほかに {also})" if also else "")
              + (f"  上位 {c['hits']} 回" if (c.get("hits") or 0) > 1 else ""))
    for i, line in enumerate(format_answer_filter(body)):
        print(f"  回答前の絞り込み: {line}" if i == 0 else f"    {line.strip()}")
    halted = f"(打ち切り位置 {body['halted_at']})" if body.get("halted_at") else ""
    print(f"  stop_reason  : {body.get('stop_reason')}{'(truncated)' if body.get('truncated') else ''}{halted}")
    print(f"  answerable   : {body.get('answerable')}"
          + ("  !! answer と食い違い" if body.get("answerable_mismatch") else ""))
    print(f"  回答         : {body.get('answer')}")
    if body.get("parse_error"):
        cut = "max_tokens で切れたため再依頼なし。" if body.get("answer_truncated") else ""
        print(f"  !! JSON として読めませんでした(試行 {body.get('answer_attempts')} 回)。{cut}生の出力: "
              f"{str(body.get('answer_raw'))[:300]}")
    if body.get("bedrock_error"):
        print(f"  !! Bedrock 失敗: {body['bedrock_error']}")
    print("  引用:")
    for c in body.get("citations") or []:
        print(f"    - {c['claim']}  ← {c['chunk_ids'] or '(根拠なし)'}")
    print(f"  引用の検査   : citations_valid={body.get('citations_valid')} / "
          f"invalid={body.get('invalid_citation_ids')} / uncited_claims={body.get('uncited_claims')} / "
          f"no_citations={body.get('no_citations')} / 表記の直し {body.get('citation_format_fixes')} 件")
    if req is not None:
        print(f"  required     : {req['required']}")
        print(f"    collected に全部 : {'○' if req['collected_cover_required'] else '×'}"
              + (f"(欠け {req['missing_in_collected']})" if req["missing_in_collected"] else ""))
        print(f"    引用に全部       : {'○' if req['citations_cover_required'] else '×'}"
              + (f"(欠け {req['missing_in_citations']})" if req["missing_in_citations"] else ""))
        print(f"    最初に取ったホップ: {req['first_hop']}")
    lat = body.get("latency_ms") or {}
    tok = body.get("tokens") or {}
    print(f"  レイテンシ   : 全体 {_f(lat.get('total'), 0)}ms(Jev {_f(lat.get('jev'), 0)} / "
          f"DynamoDB {_f(lat.get('ddb'), 0)} / Bedrock {_f(lat.get('bedrock'), 0)})、"
          f"Invoke の往復 {wall_ms:.0f}ms、コールド {body.get('cold_start')}")
    print(f"  呼び出し     : Jev {body.get('jev_calls')} 回({tok.get('jev_input', 0):,} トークン)/ "
          f"Bedrock 入力 {tok.get('bedrock_input', 0):,}・出力 {tok.get('bedrock_output', 0):,} トークン"
          f"(試行 {body.get('answer_attempts')} 回)")
    print(f"  費用         : ${cost:.6f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    label = "invoke query"
    master = load_master()
    chunks = load_chunks()
    names = {eid: e["name"] for eid, e in entity_by_id(master).items()}

    if args.qid:
        q = load_question(args.qid, {c["chunk_id"] for c in chunks})
        qid, question, required = q["id"], q["question"], list(q.get("required_chunks") or [])
    else:
        qid, question, required = "adhoc", args.question.strip(), None
        if not question:
            print(f"[{label}] --question が空です")
            return 1

    import boto3

    session = boto3.Session(region_name=REGION)
    check_account(session)                      # JEV_AWS_ACCOUNT_ID と違えばここで例外

    # ---------------------------------------------------------------- 見積もり
    texts = [c["text"] for c in chunks]
    ests = {}
    for m in args.modes:
        ev = MODE_EVENTS[m]
        ests[m] = estimate_usd(question, master, texts, mode=ev["mode"],
                               max_hops=ev.get("max_hops", DEFAULT_MAX_HOPS),
                               max_chunks=args.max_chunks or DEFAULT_MAX_CHUNKS,
                               entry_prefilter=args.entry_prefilter or DEFAULT_ENTRY_PREFILTER,
                               entry_fallback=args.entry_fallback or DEFAULT_ENTRY_FALLBACK,
                               answer_filter=args.answer_filter or DEFAULT_ANSWER_FILTER)
    n_calls = sum(e["jev_requests"] + e["gen_requests"] for e in ests.values())
    est_usd = sum(e["est_usd"] for e in ests.values())
    print(f"[{label}] 見積もり(上振れ寄り)")
    print(f"  関数     : {QUERY_FUNCTION}(リージョン {REGION})")
    print(f"  質問     : {qid}: {question}")
    if required is not None:
        print(f"  required : {required}")
    for m in args.modes:
        e = ests[m]
        print(f"  {m:<9}: event {json.dumps(build_event(question, m, args.sufficiency_th, args.entry_th, args.min_neighbor_th,
                                 max_chunks=args.max_chunks, max_chunks_per_node=args.max_chunks_per_node,
                                 hop_chunk_source=args.hop_chunk_source,
                                 max_chunks_per_hop=args.max_chunks_per_hop,
                                 entry_prefilter=args.entry_prefilter,
                                 entry_fallback=args.entry_fallback,
                                 semantic_entry_th=args.semantic_entry_th,
                                 **filter_kwargs(args)), ensure_ascii=False)}")
        print(f"             Jev 最大 {e['jev_requests']} リクエスト / Bedrock 最大 {e['gen_requests']} 回"
              + f" / ${e['est_usd']:.4f}")

    ensure_dirs()
    budget = Budget()
    budget.announce(n_calls, est_usd, label)
    budget.check(est_usd, label)
    if not args.yes and not confirm():
        print(f"[{label}] 中止しました(Lambda は呼んでいません)", flush=True)
        return 1

    # ---------------------------------------------------------------- Invoke(モードごとに 1 回だけ)
    lam = session.client("lambda", region_name=REGION, config=invoke_config())
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out: dict[str, Any] = {"qid": qid, "question": question, "required_chunks": required, "ts": ts,
                           "function": QUERY_FUNCTION, "region": REGION, "method": "jev_graph", "runs": {}}
    status = 0
    total_cost = 0.0
    for m in args.modes:
        event = build_event(question, m, args.sufficiency_th, args.entry_th, args.min_neighbor_th,
                                 max_chunks=args.max_chunks, max_chunks_per_node=args.max_chunks_per_node,
                                 hop_chunk_source=args.hop_chunk_source,
                                 max_chunks_per_hop=args.max_chunks_per_hop,
                                 entry_prefilter=args.entry_prefilter,
                                 entry_fallback=args.entry_fallback,
                                 semantic_entry_th=args.semantic_entry_th,
                                 **filter_kwargs(args))
        started = time.perf_counter()
        try:
            resp = lam.invoke(FunctionName=QUERY_FUNCTION, InvocationType="RequestResponse",
                              Payload=json.dumps(event).encode("utf-8"))
            raw = resp["Payload"].read()
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{label}] {m}: Invoke 失敗: {type(exc).__name__}。課金は CloudWatch Logs のサマリー行で"
                  "確認してください", flush=True)
            out["runs"][m] = {"event": event, "invoke_error": type(exc).__name__}
            status = 2
            continue
        wall_ms = (time.perf_counter() - started) * 1000.0
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            print(f"\n[{label}] {m}: 応答が JSON ではありません({len(raw)} bytes)", flush=True)
            out["runs"][m] = {"event": event, "invoke_error": "non-json"}
            status = 2
            continue
        if resp.get("FunctionError"):
            print(f"\n[{label}] {m}: 関数エラー: {resp['FunctionError']} / {body.get('errorType')}: "
                  f"{str(body.get('errorMessage'))[:300]}", flush=True)
            print("  呼び出しレコードは持ち帰れていません。課金ぶんは CloudWatch Logs のサマリー行で確認してください")
            out["runs"][m] = {"event": event, "function_error": body}
            status = 2
            continue

        # 課金レコードを先に残す
        records = body.get("records") or []
        for record in records:
            kind = ledger_kind(record)
            if kind is not None:
                append_jsonl(cache_path(kind), record)
        cost = sum(cost_of(r) for r in records)
        total_cost += cost

        if not body.get("ok"):
            print(f"\n[{label}] {m}: Lambda 内で失敗: {json.dumps(body.get('error'), ensure_ascii=False)}"
                  f"(レコード {len(records)} 件・${cost:.6f} は追記済み)", flush=True)
            out["runs"][m] = {"event": event, "wall_ms": wall_ms, "cost_usd": cost, "body": body}
            status = 2
            continue

        req = required_check(body, required) if required is not None else None
        print_run(m, body, names, req, cost, wall_ms)
        out["runs"][m] = {"event": event, "wall_ms": wall_ms, "cost_usd": cost,
                          "required_check": req, "body": body}

    # ---------------------------------------------------------------- 保存とまとめ
    path = QUERY_DIR / f"{qid}_{ts}.json"
    save_json(path, out)
    print(f"\n[{label}] まとめ")
    for m, r in out["runs"].items():
        body = r.get("body") or {}
        req = r.get("required_check") or {}
        print(f"  {m:<9}: stop={body.get('stop_reason')} / チャンク {len(body.get('chunks') or [])} / "
              f"answerable={body.get('answerable')} / 引用OK={body.get('citations_valid')} / "
              f"required 収集={req.get('collected_cover_required')}・引用={req.get('citations_cover_required')} / "
              f"{_f((body.get('latency_ms') or {}).get('total'), 0)}ms / ${r.get('cost_usd', 0):.6f}")
    print(f"  今回の費用 : ${total_cost:.6f}(見積もり ${est_usd:.4f})")
    print(f"  累計費用   : ${Budget().spent():.6f}")
    print(f"  保存先     : {path}")
    return status


if __name__ == "__main__":
    sys.exit(main())
