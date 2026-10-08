#!/usr/bin/env python3
"""review.md の採点欄と raw.jsonl を集計して `results/eval/<run>/summary.md` を作る。

    ./.venv/bin/python scripts/32_summarize.py --run dev1

モード別の正解率・機械指標(必要なチャンクの収集率・引用の妥当性・ホップ数・レイテンシ・費用など)と、
十分性・入口のしきい値の分析を出す。採点欄が空なら機械指標だけで作る。AWS・Jev・Bedrock には触れない。"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_common as ec  # noqa: E402
from budget import cost_of  # noqa: E402
from common import md_table  # noqa: E402
from master import entity_by_id, load_master  # noqa: E402
from query_core import ENTRY_LOG_MIN_SCORE  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="採点と機械指標の集計")
    parser.add_argument("--run", required=True, help="results/eval/<run> の名前")
    return parser.parse_args(argv)


def _mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _rate(num: int, den: int) -> str:
    return f"{num}/{den}({num / den:.0%})" if den else "-"


# ============================================================== 採点

def grade_stats(rows: Mapping[tuple[str, str], dict], grades: Mapping[tuple[str, str], dict],
                mode: str) -> dict:
    """1 モードぶんの採点集計。正誤が空の行は「未採点」として分母から外す。"""
    keys = [k for k in rows if k[1] == mode]
    graded = [(k, grades[k]) for k in keys if k in grades and grades[k]["correct"] is not None]
    correct = [k for k, g in graded if g["correct"] == "○"]
    strict = [k for k, g in graded if g["correct"] == "○" and g["grounded"] == "○"]
    suspect = [k for k, g in graded if g["correct"] == "○" and g["grounded"] == "×"]
    partial = [k for k, g in graded if g["correct"] == "△"]
    inconsistent = [k for k in keys if k in grades and grades[k]["consistent"] == "×"]
    missing_ground = [k for k, g in graded if g["correct"] == "○" and g["grounded"] is None]
    unreadable = [(k, f) for k in keys if k in grades for f in ec.MARK_FIELDS if grades[k][f] == "?"]
    return {"n_runs": len(keys), "n_graded": len(graded), "correct": correct, "strict": strict,
            "suspect": suspect, "partial": partial, "inconsistent": inconsistent,
            "missing_ground": missing_ground, "unreadable": unreadable}


# ============================================================== 機械指標

def run_metrics(row: Mapping[str, Any], groups: Sequence[Sequence[str]] | None = None) -> dict:
    """1 実行ぶんの機械指標。`groups` は正解の組の候補(`ec.required_groups()`)。None なら
    row の required_chunks を 1 組とみなす。"""
    body = row["body"]
    required = list(row.get("required_chunks") or [])
    collected = {c["id"] for c in body.get("chunks") or []}
    cited = set(ec.cited_ids(body))
    first_hop = {c["id"]: c.get("hop") for c in body.get("chunks") or []}
    lat = body.get("latency_ms") or {}
    if groups is None:
        groups = [required] if required else []
    any_check = ec.groups_check(groups, collected, cited)
    return {
        "required": required,
        "groups": any_check["groups"],
        "collect_any": any_check["collected_any"],
        "cite_any": any_check["cited_any"],
        "collect_group": any_check["collected_group"],
        "collect_recall": (sum(c in collected for c in required) / len(required)) if required else None,
        "cite_recall": (sum(c in cited for c in required) / len(required)) if required else None,
        "citations_valid": body.get("citations_valid"),
        "no_citations": bool(body.get("no_citations")),
        "uncited_claims": int(body.get("uncited_claims") or 0),
        "hops": ec.last_hop(body),
        "stop_reason": body.get("stop_reason"),
        "jev_calls": int(body.get("jev_calls") or 0),
        "latency": {k: lat.get(k) for k in ("total", "jev", "ddb", "bedrock")},
        "cost": sum(cost_of(r) for r in body.get("records") or []),
        "first_hop": {c: first_hop.get(c) for c in required},
        "n_collected": len(collected),
        "n_passed": len(ec.answer_chunk_ids(body)),
    }


def machine_stats(rows: Mapping[tuple[str, str], dict], mode: str,
                  groups_by_qid: Mapping[str, Sequence[Sequence[str]]] | None = None) -> dict:
    groups_by_qid = groups_by_qid or {}
    ms = [run_metrics(r, groups_by_qid.get(k[0])) for k, r in sorted(rows.items()) if k[1] == mode]
    any_c = [m["collect_any"] for m in ms if m["collect_any"] is not None]
    any_t = [m["cite_any"] for m in ms if m["cite_any"] is not None]
    valid = [m["citations_valid"] for m in ms if m["citations_valid"] is not None]
    first = Counter()
    for m in ms:
        for _cid, hop in m["first_hop"].items():
            first["未取得" if hop is None else f"ホップ{hop}"] += 1
    return {
        "n": len(ms),
        "collect_recall": _mean([m["collect_recall"] for m in ms]),
        "cite_recall": _mean([m["cite_recall"] for m in ms]),
        "collect_any": f"{sum(any_c)}/{len(any_c)}" if any_c else "-",
        "collect_any_rate": (sum(any_c) / len(any_c)) if any_c else None,
        "cite_any": f"{sum(any_t)}/{len(any_t)}" if any_t else "-",
        "cite_any_rate": (sum(any_t) / len(any_t)) if any_t else None,
        "valid_rate": (sum(valid) / len(valid)) if valid else None,
        "n_valid": f"{sum(valid)}/{len(valid)}" if valid else "-",
        "no_citations": sum(m["no_citations"] for m in ms),
        "uncited_sum": sum(m["uncited_claims"] for m in ms),
        "uncited_mean": _mean([m["uncited_claims"] for m in ms]),
        "hops_mean": _mean([m["hops"] for m in ms]),
        "collected_mean": _mean([m["n_collected"] for m in ms]),
        "passed_mean": _mean([m["n_passed"] for m in ms]),
        "stop": Counter(m["stop_reason"] for m in ms),
        "jev_sum": sum(m["jev_calls"] for m in ms),
        "jev_mean": _mean([m["jev_calls"] for m in ms]),
        "latency": {k: _mean([m["latency"][k] for m in ms]) for k in ("total", "jev", "ddb", "bedrock")},
        "cost_sum": sum(m["cost"] for m in ms),
        "cost_mean": _mean([m["cost"] for m in ms]),
        "first_hop": first,
    }


# ============================================================== 十分性・入口の分析

def sufficiency_rows(rows: Mapping[tuple[str, str], dict], mode: str = "adaptive",
                     groups_by_qid: Mapping[str, Sequence[Sequence[str]]] | None = None,
                     types: Mapping[str, Any] | None = None) -> list[dict]:
    """adaptive の各ホップ: 十分性スコアと、その時点の累積に required が全部揃っていたか。"""
    out = []
    for (qid, m), row in sorted(rows.items()):
        if m != mode:
            continue
        body = row["body"]
        required = list(row.get("required_chunks") or [])
        groups = (groups_by_qid or {}).get(qid)
        if groups is None:
            groups = [required] if required else []
        for h in body.get("hops") or []:
            if h.get("sufficiency") is None:
                continue
            upto = ec.chunks_upto(body, h["hop"])
            complete = bool(ec.groups_check(groups, upto, []).get("collected_any"))
            missing = min(([c for c in g if c not in upto] for g in groups), key=len, default=[])
            out.append({"qid": qid, "type": (types or {}).get(qid, row.get("type")), "hop": h["hop"],
                        "score": h["sufficiency"],
                        "reused": bool(h.get("sufficiency_reused")), "complete": complete,
                        "collected": upto, "missing": missing,
                        "threshold": (body.get("params") or {}).get("sufficiency_th")})
    return out


def entry_rows(rows: Mapping[tuple[str, str], dict], master: Mapping[str, Any],
               questions: Sequence[Mapping[str, Any]]) -> list[dict]:
    """全質問の入口候補(selected を含む全件)× 質問文に出てくるか。`(qid, entity)` ごとに 1 行。"""
    ents = entity_by_id(master)
    qtext = {q["id"]: q["question"] for q in questions}
    table: dict[tuple[str, str], dict] = {}
    for (qid, mode), row in sorted(rows.items()):
        question = qtext.get(qid, row.get("question") or "")
        for e in row["body"].get("entry") or []:
            key = (qid, e["id"])
            item = table.setdefault(key, {"qid": qid, "id": e["id"], "name": ents.get(e["id"], {}).get("name", e["id"]),
                                          "scores": {}, "selected": set(),
                                          "term": ec.matched_term(question, ents.get(e["id"], {"name": e["id"]}))})
            item["scores"][mode] = e["score"]
            if e.get("selected"):
                item["selected"].add(mode)
    for qid in sorted({k[0] for k in rows}):
        question = qtext.get(qid, "")
        for eid, ent in ents.items():
            term = ec.matched_term(question, ent)
            if term and (qid, eid) not in table:
                table[(qid, eid)] = {"qid": qid, "id": eid, "name": ent["name"], "scores": {},
                                     "selected": set(), "term": term}
    return sorted(table.values(), key=lambda r: (r["qid"], -max(r["scores"].values(), default=0.0), r["id"]))


def entry_samples(items: Sequence[Mapping[str, Any]]) -> list[tuple[float, bool]]:
    """しきい値の候補探し用。1 回の実行 × 1 ノードを 1 サンプル(下限未満のノードは入れない)。"""
    return [(score, item["term"] is not None) for item in items for score in item["scores"].values()]


# ============================================================== 書き出し

def _cand_table(cands: Sequence[Mapping[str, Any]], pos: str, neg: str) -> str:
    rows = [[ec.fmt(c["threshold"], 3), c["tp"], c["fp"], c["fn"], c["tn"],
             ec.fmt(c["balanced_accuracy"], 3), ec.fmt(c["accuracy"], 3)] for c in cands]
    return md_table(["しきい値", f"TP({pos}を通す)", f"FP({neg}を通す)", f"FN({pos}を落とす)",
                     f"TN({neg}を落とす)", "balanced acc", "accuracy"], rows)


def build_summary(run: str, rows: Mapping[tuple[str, str], dict], failed: Sequence[Mapping[str, Any]],
                  grades: Mapping[tuple[str, str], dict], closed: Mapping[str, dict],
                  questions: Sequence[Mapping[str, Any]], master: Mapping[str, Any]) -> str:
    modes = [m for m in ec.MODE_ORDER if any(k[1] == m for k in rows)]
    qids = sorted({k[0] for k in rows})
    any_graded = any(g["correct"] is not None for k, g in grades.items() if k in rows)
    qmap = {q["id"]: q for q in questions}
    groups_by_qid = {qid: ec.required_groups(q) for qid, q in qmap.items()}
    types = {qid: ec.question_type(q) for qid, q in qmap.items()}
    has_v2 = any("required_any" in q for q in questions)
    out = [f"# 集計({run})", "",
           f"- 質問: {', '.join(qids)}({len(qids)} 問)/ モード: {', '.join(modes)}",
           f"- 成功した実行: {len(rows)} 件 / 失敗(後で成功していないもの): {len(failed)} 件",
           "- 正解の基準: " + ("v2(required_any・type_v2・answer_v2)" if has_v2 else "v1(required_chunks)")]
    for f in failed:
        out.append(f"  - {f.get('qid')} {f.get('mode')}: {f.get('status')} {f.get('error')}")
    out.append("")

    # ---- 採点
    out += ["## 採点結果(目視)", ""]
    if not any_graded:
        out += ["**未採点**(review.md の採点欄が空のため、以下は機械指標だけ)", ""]
    else:
        stats = {m: grade_stats(rows, grades, m) for m in modes}
        table = []
        for m in modes:
            s = stats[m]
            table.append([m, f"{s['n_graded']}/{s['n_runs']}", _rate(len(s["strict"]), s["n_graded"]),
                          _rate(len(s["correct"]), s["n_graded"]), len(s["partial"]), len(s["suspect"]),
                          len(s["inconsistent"])])
        out.append(md_table(["モード", "採点済み", "正解率(正誤○かつ根拠○)", "正誤だけの正解率", "△",
                             "知識回答の疑い(正誤○・根拠×)", "回答文と引用の食い違い(×)"], table))
        out.append("")
        for m in modes:
            s = stats[m]
            notes = []
            if s["n_graded"] < s["n_runs"]:
                notes.append(f"未採点 {s['n_runs'] - s['n_graded']} 件")
            if s["suspect"]:
                notes.append("知識回答の疑い: " + ", ".join(k[0] for k in s["suspect"]))
            if s["inconsistent"]:
                notes.append("食い違い: " + ", ".join(k[0] for k in s["inconsistent"]))
            if s["missing_ground"]:
                notes.append("正誤○だが根拠が空欄(正解に数えていない): " + ", ".join(k[0] for k in s["missing_ground"]))
            if s["unreadable"]:
                notes.append("読めない記入: " + ", ".join(f"{k[0]}/{f}" for k, f in s["unreadable"]))
            if notes:
                out.append(f"- {m}: " + " / ".join(notes))
        out.append("")
        detail = []
        for qid in qids:
            line = [qid]
            for m in modes:
                g = grades.get((qid, m))
                if (qid, m) not in rows:
                    line.append("(結果なし)")
                elif g is None:
                    line.append("(欄なし)")
                else:
                    line.append(f"{g['correct'] or '_'}/{g['grounded'] or '_'}/{g['consistent'] or '_'}"
                                + (f" {ec.md_cell(g['memo'])}" if g["memo"] else ""))
            detail.append(line)
        out += ["質問別(正誤/根拠/食い違い。_ は空欄)", "", md_table(["qid", *modes], detail), ""]

    # ---- 機械指標
    out += ["## 機械指標(モード別)", ""]
    ms = {m: machine_stats(rows, m, groups_by_qid) for m in modes}
    metric_rows = [
        ("実行数", lambda s: s["n"]),
        ("required 収集再現率(平均)", lambda s: ec.fmt(s["collect_recall"], 3)),
        ("required 引用再現率(平均)", lambda s: ec.fmt(s["cite_recall"], 3)),
        ("正解の組がそろった(集めたチャンク)", lambda s: f"{ec.fmt(s['collect_any_rate'], 3)}({s['collect_any']})"),
        ("正解の組がそろった(引用)", lambda s: f"{ec.fmt(s['cite_any_rate'], 3)}({s['cite_any']})"),
        ("引用の妥当率(citations_valid)", lambda s: f"{ec.fmt(s['valid_rate'], 3)}({s['n_valid']})"),
        ("no_citations 件数", lambda s: s["no_citations"]),
        ("uncited_claims 合計(平均)", lambda s: f"{s['uncited_sum']}({ec.fmt(s['uncited_mean'])})"),
        ("ホップ数(平均)", lambda s: ec.fmt(s["hops_mean"])),
        ("集めたチャンク数(平均)", lambda s: ec.fmt(s["collected_mean"], 1)),
        ("Haiku に渡したチャンク数(平均。絞った後)", lambda s: ec.fmt(s["passed_mean"], 1)),
        ("Jev 呼び出し 合計(平均)", lambda s: f"{s['jev_sum']}({ec.fmt(s['jev_mean'], 1)})"),
        ("レイテンシ total 平均 ms", lambda s: ec.fmt(s["latency"]["total"], 0)),
        ("　jev 平均 ms", lambda s: ec.fmt(s["latency"]["jev"], 0)),
        ("　ddb 平均 ms", lambda s: ec.fmt(s["latency"]["ddb"], 0)),
        ("　bedrock 平均 ms", lambda s: ec.fmt(s["latency"]["bedrock"], 0)),
        ("費用 合計 $(平均)", lambda s: f"{s['cost_sum']:.6f}({ec.fmt(s['cost_mean'], 6)})"),
        ("stop_reason", lambda s: "、".join(f"{k} {v}" for k, v in sorted(s["stop"].items(), key=lambda kv: str(kv[0])))),
        ("required を最初に取ったホップ", lambda s: "、".join(f"{k} {v}" for k, v in sorted(s["first_hop"].items()))),
    ]
    out.append(md_table(["指標", *modes], [[name, *[str(fn(ms[m])) for m in modes]] for name, fn in metric_rows]))
    out += ["", "- 再現率は required_chunks がある質問だけの平均(答えのない質問は除く)",
            "- 「正解の組がそろった」は、正解の組の候補(v2 は required_any、v1 は required_chunks の 1 組)の"
            "どれか 1 組が全部入っていた実行の割合。組の無い質問(答えのない質問)は除く", ""]
    per_q = []
    for (qid, m), row in sorted(rows.items(), key=lambda kv: (kv[0][0], ec.MODE_ORDER.index(kv[0][1]))):
        rm = run_metrics(row, groups_by_qid.get(qid))

        def mark(value: Any) -> str:
            return "-" if value is None else ("○" if value else "×")

        per_q.append([qid, types.get(qid, row.get("type")), m, rm["stop_reason"], rm["hops"],
                      ec.fmt(rm["collect_recall"]), ec.fmt(rm["cite_recall"]),
                      mark(rm["collect_any"]) + (f" {'+'.join(rm['collect_group'])}" if rm["collect_group"] else ""),
                      mark(rm["cite_any"]), rm["citations_valid"], rm["uncited_claims"], rm["jev_calls"],
                      ec.fmt(rm["latency"]["total"], 0), f"{rm['cost']:.6f}",
                      ", ".join(f"{c}:{'-' if h is None else h}" for c, h in rm["first_hop"].items()) or "-"])
    out += ["質問別", "", md_table(["qid", "type", "モード", "stop", "ホップ", "収集再現", "引用再現",
                                     "組そろう(収集)", "組そろう(引用)", "引用OK",
                                     "uncited", "Jev", "total ms", "費用 $", "required の初出ホップ"], per_q), ""]

    # ---- 十分性
    out += ["## 十分性しきい値の分析(adaptive の各ホップ)", ""]
    srows = sufficiency_rows(rows, groups_by_qid=groups_by_qid, types=types)
    if not srows:
        out += ["(adaptive の結果が無い、または十分性スコアが無い)", ""]
    else:
        th_now = next((r["threshold"] for r in srows if r["threshold"] is not None), None)
        out.append(md_table(["qid", "type", "ホップ", "十分性", "累積に required が揃った", "その時点の累積", "欠け"],
                            [[r["qid"], r["type"], r["hop"], ec.fmt(r["score"], 3) + ("(前の値)" if r["reused"] else ""),
                              "○" if r["complete"] else "×", ", ".join(r["collected"]),
                              ", ".join(r["missing"]) or "-"] for r in srows]))
        out += ["", f"- 今回のしきい値: {ec.fmt(th_now, 3)}(adaptive はしきい値以上で止まるので、"
                     "それより後のホップは観測されていない)",
                "- 答えのない質問は「揃う」ことが無いので × として数える(十分と判定してほしくない側)",
                "- 「(前の値)」の行(累積が前のホップと同じで判定を呼んでいない)は候補探しから外す", ""]
        samples = [(r["score"], r["complete"]) for r in srows if not r["reused"]]
        cands = ec.threshold_candidates(samples)
        if th_now is not None:
            now = ec.confusion(samples, th_now)
            out += [f"- 今のしきい値 {ec.fmt(th_now, 3)} の成績: TP {now['tp']} / FP {now['fp']} / FN {now['fn']} / "
                    f"TN {now['tn']} / balanced acc {ec.fmt(now['balanced_accuracy'], 3)}", ""]
        out += ["しきい値の候補(balanced accuracy の高い順。同点は高いしきい値を先に。値は隣り合うスコアの中点で、"
                "前後のスコアの間ならどこでも同じ成績)", "",
                _cand_table(cands, "揃った", "揃っていない"), ""]

    # ---- 入口
    out += ["## 入口しきい値の分析", ""]
    erows = entry_rows(rows, master, questions)
    if not erows:
        out += ["(結果なし)", ""]
    else:
        th_entry = next((r["body"].get("params", {}).get("entry_threshold") for r in rows.values()), None)
        table = []
        for r in erows:
            scores = " / ".join(f"{m} {ec.fmt(r['scores'][m])}" for m in modes if m in r["scores"]) \
                or f"{ENTRY_LOG_MIN_SCORE} 未満(ログに無い)"
            table.append([r["qid"], f"{r['name']}({r['id']})", scores,
                          "、".join(m for m in modes if m in r["selected"]) or "-",
                          f"○({r['term']})" if r["term"] else "×"])
        out.append(md_table(["qid", "ノード", "スコア(モード別)", "入口に選ばれた", "質問文に出る(一致した語)"], table))
        out += ["", f"- 今回の entry_threshold: {ec.fmt(th_entry, 3)}。ログに残るのはスコア {ENTRY_LOG_MIN_SCORE} 以上だけ",
                "- 「質問文に出る」は正式名・別名の部分一致(機械判定。短い別名の誤一致に注意)", ""]
        cands = ec.threshold_candidates(entry_samples(erows))
        out += ["しきい値の候補(質問文に出る=陽性。1 実行 × 1 ノードを 1 件として数える)", "",
                _cand_table(cands, "出る語", "出ない語"), ""]

    # ---- クローズドブック
    out += ["## クローズドブック(チャンクなしの Haiku)", ""]
    if not closed:
        out += ["(未実行)", ""]
    else:
        table = []
        for qid in sorted(closed):
            cb = closed[qid]
            q = qmap.get(qid, {})
            table.append([qid, ec.question_type(q, cb.get("type")), ec.md_cell(ec.model_answer(q)),
                          ec.md_cell(cb.get("answer") if not cb.get("error") else f"(失敗 {cb['error']})")])
        out.append(md_table(["qid", "type", "模範解答", "クローズドブックの回答"], table))
        out += ["", "- 素で答えられている質問は、その質問の正解をとくに引用(根拠)で厳しく見る", ""]
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    label = "summarize"
    try:
        out_dir = ec.run_dir(args.run)
    except ValueError as exc:
        print(f"[{label}] {exc}")
        return 1
    raw_path = out_dir / ec.RAW_NAME
    if not raw_path.exists():
        print(f"[{label}] {raw_path} がありません(先に 30 番を実行)")
        return 1
    rows, failed = ec.load_raw(raw_path)
    review = out_dir / ec.REVIEW_NAME
    grades = ec.parse_grades(review.read_text(encoding="utf-8")) if review.exists() else {}
    if not review.exists():
        print(f"[{label}] {review} がありません。機械指標だけで作ります")
    closed = ec.load_closed_book(out_dir / ec.CLOSED_BOOK_NAME)
    text = build_summary(args.run, rows, failed, grades, closed, ec.load_questions(), load_master())
    path = out_dir / ec.SUMMARY_NAME
    path.write_text(text, encoding="utf-8")
    n_graded = sum(1 for k, g in grades.items() if k in rows and g["correct"] is not None)
    print(f"[{label}] 成功 {len(rows)} 件 / 失敗 {len(failed)} 件 / 採点済み {n_graded} 件"
          + ("(未採点)" if n_graded == 0 else ""))
    print(f"  保存先: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
