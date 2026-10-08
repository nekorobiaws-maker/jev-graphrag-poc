#!/usr/bin/env python3
"""`results/eval/<run>/raw.jsonl`(と `closed_book.jsonl`)から、目視採点用の `review.md` を作る。

    ./.venv/bin/python scripts/31_make_review_sheet.py --run dev1
    ./.venv/bin/python scripts/31_make_review_sheet.py --run dev1 --force   # 採点済みでも作り直す

採点欄(`<!-- grade q01 fixed1 -->` 〜 `<!-- /grade -->`)は 32 番が読むので、マーカーは消さないこと。
AWS・Jev・Bedrock には触れない。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_common as ec  # noqa: E402
from master import entity_by_id, load_chunks, load_master  # noqa: E402

CRITERIA = """## 採点の基準

- **正誤**(○/×/△): 模範解答と意味が合っているか。答えのない質問(type=no_answer)は「回答できません」と答えたら ○。
  一部だけ合っている・余計な誤りが混ざるなら △
- **根拠**(○/×): 引用したチャンク本文**だけ**で、回答のすべての記述が言えるか。言えない記述があれば ×(知識で答えた疑い)。
  「回答できません」で主張が無いときは ○
- **回答文と引用の食い違い**(○/×): 回答文に書かれているのに引用(claim)に入っていない事実があれば ×。無ければ ○
- **メモ**: 自由記述(気づいたこと)

書き方: `- 正誤: ○` のように、コロンの後ろに ○ / × / △ を 1 文字(o / x でも可)。
`<!-- grade ... -->` と `<!-- /grade -->` の行は消さないでください(32 番が読みます)。
正解 = 正誤 ○ かつ 根拠 ○。正誤 ○ かつ 根拠 × は「知識で答えた疑い」として別に数えます。
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="目視採点シートを作る")
    parser.add_argument("--run", required=True, help="results/eval/<run> の名前")
    parser.add_argument("--force", action="store_true", help="採点欄が埋まっていても上書きする")
    return parser.parse_args(argv)


# ============================================================== 部品

def _quote(text: str) -> str:
    return "\n".join("> " + line if line else ">" for line in str(text).splitlines() or [""])


def mode_title(mode: str) -> str:
    ev = ec.q21.MODE_EVENTS[mode]
    hops = ev.get("max_hops", "既定 3")
    return f"{mode}(mode={ev['mode']}, max_hops={hops})"


def entry_lines(body: Mapping[str, Any], name) -> list[str]:
    entry = body.get("entry") or []
    chosen = [e for e in entry if e.get("selected")]
    others = [e for e in entry if not e.get("selected")]
    th = (body.get("params") or {}).get("entry_threshold")
    sel = "、".join(f"{name(e['id'])}({e['id']}) {ec.fmt(e['score'])}" for e in chosen) or "(なし)"
    lines = [f"- **入口**(しきい値 {ec.fmt(th)}、in_scope {ec.fmt(body.get('in_scope'))}): {sel}"]
    if others:
        lines.append("  - 選外: " + "、".join(f"{name(e['id'])} {ec.fmt(e['score'])}" for e in others[:6])
                     + (" ほか" if len(others) > 6 else ""))
    return lines


def hop_lines(body: Mapping[str, Any], name) -> list[str]:
    lines = ["- **ホップ**:"]
    for h in body.get("hops") or []:
        suff = h.get("sufficiency")
        suff_s = "-" if suff is None else f"{suff:.3f}" + ("(前の値)" if h.get("sufficiency_reused") else "")
        new = ", ".join(h.get("new_chunk_ids") or []) or "なし"
        extra = ""
        if h.get("dropped_chunk_ids"):
            extra += f" / 切り捨て {h['dropped_chunk_ids']}"
        if h.get("missing_chunk_ids"):
            extra += f" / 見つからず {h['missing_chunk_ids']}"
        counts = h.get("node_chunk_counts") or {}
        for eid, capped in (h.get("capped_chunk_ids") or {}).items():
            total = counts.get(eid)
            what = "関係の根拠" if h.get("chunk_source") == "edge_evidence" else "登場"
            extra += (f" / {name(eid)} は{what} {total if total is not None else '?'} 件のうち上位のみ"
                      f"(外した {', '.join(capped)})")
        if h.get("hop_capped_chunk_ids"):
            per_hop = (body.get("params") or {}).get("max_chunks_per_hop", "?")
            extra += (f" / 1ホップ {per_hop} 件まで(スコア順。あふれた "
                      f"{', '.join(h['hop_capped_chunk_ids'])})")
        if h["hop"] == 0:
            dest = "入口 " + ("、".join(name(e) for e in h.get("selected") or []) or "(なし)")
        else:
            scores = h.get("neighbor_scores") or []
            parts = []
            for eid in h.get("selected") or []:
                s = next((s for s in scores if s.get("entity") == eid), None)
                if s is None:
                    parts.append(name(eid))
                    continue
                direction = str(s.get("direction") or "").lower()
                arrow = {"out": "→(OUT)", "sym": "⇄(SYM)"}.get(direction, "←(IN)")
                parts.append(f"{name(s.get('from'))} {arrow}[{s.get('edge')} {ec.fmt(s.get('score'))}] "
                             f"{name(eid)}(「{s.get('sentence')}」)")
            dest = "、".join(parts) or f"(進めず。候補 {len(scores)} 件)"
        lines.append(f"  - ホップ{h['hop']}: {dest} / 新規チャンク {new}{extra} / 十分性 {suff_s}")
    return lines


def chunk_lines(body: Mapping[str, Any], name) -> list[str]:
    lines = ["- **集めたチャンク**(累積順):"]
    for c in body.get("chunks") or []:
        also = "; ".join(f"h{a['hop']} {name(a['entity'])}/{a.get('edge') or '入口'}"
                         for a in c.get("also_via") or [])
        lines.append(f"  - {c['id']}: hop {c['hop']} / via {name(c['via_entity'])} / {c.get('via_edge') or '入口'}"
                     + (f"(ほかに {also})" if also else ""))
    if len(lines) == 1:
        lines.append("  - (なし)")
    return lines


def citation_lines(body: Mapping[str, Any], texts: Mapping[str, str]) -> list[str]:
    collected = {c["id"] for c in body.get("chunks") or []}
    lines = ["- **引用**:"]
    cits = body.get("citations") or []
    if not cits:
        lines.append("  - (引用なし)")
    for i, c in enumerate(cits, 1):
        lines.append(f"  {i}. 主張: {c.get('claim') or '(空)'}")
        ids = c.get("chunk_ids") or []
        if not ids:
            lines.append("     - (根拠 ID なし)")
        for cid in ids:
            note = "" if cid in collected else " **!! 渡していない ID**"
            lines.append(f"     - [{cid}]{note}")
            text = texts.get(cid)
            body_text = text if text is not None else "(chunks.json に無い ID)"
            lines.append("\n".join("       " + ln for ln in _quote(body_text).splitlines()))
    return lines


def check_lines(body: Mapping[str, Any], req: Mapping[str, Any] | None,
                groups: Sequence[Sequence[str]] | None = None) -> list[str]:
    lines = [f"- **機械検査**: citations_valid={body.get('citations_valid')} / "
             f"no_citations={body.get('no_citations')} / uncited_claims={body.get('uncited_claims')} / "
             f"invalid={body.get('invalid_citation_ids') or []} / answerable={body.get('answerable')}"
             + (" / !! answer と answerable が食い違い" if body.get("answerable_mismatch") else "")
             + (" / !! JSON パース失敗" if body.get("parse_error") else "")]
    if req is not None and req.get("required"):
        lines.append(f"  - required 収集: {'○' if req['collected_cover_required'] else '×'}"
                     + (f"(欠け {req['missing_in_collected']})" if req["missing_in_collected"] else "")
                     + f" / 引用: {'○' if req['citations_cover_required'] else '×'}"
                     + (f"(欠け {req['missing_in_citations']})" if req["missing_in_citations"] else "")
                     + f" / 最初に取ったホップ {req['first_hop']}")
    elif req is not None:
        lines.append("  - required: なし(答えのない質問)")
    if groups:
        chk = ec.groups_check(groups, [c["id"] for c in body.get("chunks") or []], ec.cited_ids(body))
        lines.append(f"  - 正解の組(どれか 1 組): 収集 {'○' if chk['collected_any'] else '×'}"
                     + (f"({'+'.join(chk['collected_group'])})" if chk["collected_group"] else "")
                     + f" / 引用 {'○' if chk['cited_any'] else '×'}"
                     + (f"({'+'.join(chk['cited_group'])})" if chk["cited_group"] else ""))
    return lines


def run_section(mode: str, row: Mapping[str, Any] | None, failed: Sequence[Mapping[str, Any]],
                name, texts: Mapping[str, str], groups: Sequence[Sequence[str]] | None = None) -> list[str]:
    out = [f"### {mode_title(mode)}", ""]
    if row is None:
        fails = [f for f in failed if f.get("mode") == mode]
        if fails:
            f = fails[-1]
            out += [f"(失敗: {f.get('status')} {f.get('error')})", ""]
        else:
            out += ["(結果なし)", ""]
        return out
    body = row["body"]
    req = row.get("required_check") or ec.q21.required_check(body, row.get("required_chunks") or [])
    out += entry_lines(body, name)
    out += hop_lines(body, name)
    halted = f"(打ち切り位置 {body['halted_at']})" if body.get("halted_at") else ""
    out.append(f"- **stop_reason**: {body.get('stop_reason')}{'(truncated)' if body.get('truncated') else ''}{halted}")
    out += chunk_lines(body, name)
    out.append("- **回答文**:")
    out.append(_quote(body.get("answer") or "(空)"))
    out += citation_lines(body, texts)
    out += check_lines(body, req, groups)
    out += ["", f"#### 採点({row['qid']} / {mode})", "", ec.grade_block(row["qid"], mode), ""]
    return out


def build_review(run: str, questions: Sequence[Mapping[str, Any]], ok: Mapping[tuple[str, str], dict],
                 failed: Sequence[Mapping[str, Any]], closed: Mapping[str, dict], master: Mapping[str, Any],
                 chunks: Sequence[Mapping[str, Any]], modes: Sequence[str] = ec.MODE_ORDER) -> str:
    ents = entity_by_id(master)

    def name(eid: Any) -> str:
        return ents[eid]["name"] if eid in ents else str(eid)

    texts = {c["chunk_id"]: c["text"] for c in chunks}
    out = [f"# 採点シート({run})", "", CRITERIA]
    for q in questions:
        qid = q["id"]
        v2 = "required_any" in q or "answer_v2" in q
        out += [f"## {qid}({ec.question_type(q)})", "",
                f"- **質問**: {q['question']}",
                f"- **模範解答**: {ec.model_answer(q)}"]
        if v2:
            groups_text = " / ".join("+".join(g) for g in ec.required_groups(q)) or "(なし。答えのない質問)"
            out += [f"- **正解の組(required_any。どれか 1 組で足りる)**: {groups_text}",
                    f"- **v1 の模範解答・type**: {q.get('answer')}({q.get('type')})"]
        out += [f"- **required_chunks**: {', '.join(q.get('required_chunks') or []) or '(なし)'}",
                f"- **expected_path**: {q.get('expected_path') or '(なし)'}",
                f"- **notes**: {q.get('notes') or ''}"]
        if v2 and q.get("notes_v2"):
            out.append(f"- **notes_v2**: {q['notes_v2']}")
        out.append("")
        body_ids = list(dict.fromkeys(c for g in ec.required_groups(q) for c in g)) if v2 \
            else list(q.get("required_chunks") or [])
        if body_ids:
            out.append("<details><summary>正解チャンクの本文</summary>" if v2
                       else "<details><summary>required_chunks の本文</summary>")
            out.append("")
            for cid in body_ids:
                out.append(f"- [{cid}]")
                out.append(_quote(texts.get(cid, "(chunks.json に無い ID)")))
            out += ["", "</details>", ""]
        out += ["### クローズドブック(チャンクなしの Haiku)", ""]
        cb = closed.get(qid)
        if cb is None:
            out += ["(未実行)", ""]
        elif cb.get("error"):
            out += [f"(失敗: {cb['error']})", ""]
        else:
            out += [_quote(cb.get("answer") or "(空)"), ""]
        q_failed = [f for f in failed if f.get("qid") == qid]
        for mode in modes:
            out += run_section(mode, ok.get((qid, mode)), q_failed, name, texts,
                               ec.required_groups(q) if v2 else None)
    return "\n".join(out).rstrip() + "\n"


def write_review(path: Path, text: str, force: bool) -> bool:
    """採点欄が埋まった既存ファイルは force が無ければ書かない。書いたら True。"""
    if path.exists() and not force and ec.has_filled_grades(path.read_text(encoding="utf-8")):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    label = "review sheet"
    try:
        out_dir = ec.run_dir(args.run)
    except ValueError as exc:
        print(f"[{label}] {exc}")
        return 1
    raw_path = out_dir / ec.RAW_NAME
    if not raw_path.exists():
        print(f"[{label}] {raw_path} がありません(先に 30 番を実行)")
        return 1
    ok, failed = ec.load_raw(raw_path)
    closed = ec.load_closed_book(out_dir / ec.CLOSED_BOOK_NAME)
    qids = list(dict.fromkeys([k[0] for k in ok] + [f.get("qid") for f in failed if f.get("qid")]))
    questions = [q for q in ec.load_questions() if q["id"] in qids]
    modes = [m for m in ec.MODE_ORDER if any(k[1] == m for k in ok) or any(f.get("mode") == m for f in failed)]
    text = build_review(args.run, questions, ok, failed, closed, load_master(), load_chunks(), modes)
    path = out_dir / ec.REVIEW_NAME
    if not write_review(path, text, args.force):
        print(f"[{label}] {path} は採点済みの欄があるので上書きしません(作り直すなら --force)")
        return 1
    print(f"[{label}] {len(questions)} 問 × {len(modes)} モード(成功 {len(ok)} / 失敗 {len(failed)})"
          f"、クローズドブック {len(closed)} 問")
    print(f"  保存先: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
