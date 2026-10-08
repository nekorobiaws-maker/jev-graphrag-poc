#!/usr/bin/env python3
"""評価スクリプト(30〜32 番)の共通部品。AWS・Jev・Bedrock には触れない。

- 出力先: `results/eval/<run>/`(`raw.jsonl` / `closed_book.jsonl` / `review.md` / `summary.md` / `meta.json`)
- 21 番の `MODE_EVENTS`・`build_event` などを `q21` として読み込んで使う
- 採点欄の書式(31 番が書き、32 番が読む)と、しきい値の候補探し"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPTS_DIR.parent / "lib"
for _p in (str(LIB_DIR),):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)   # 同名モジュールが他所にあっても lib/ を優先させる

from common import DATA_DIR, RESULTS_DIR, iter_jsonl, load_json  # noqa: E402


def _load_script(filename: str, module_name: str) -> Any:
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


q21 = _load_script("21_invoke_query.py", "invoke_query_21")

EVAL_DIR = RESULTS_DIR / "eval"


def questions_path(value: str | None = None) -> Path:
    """質問ファイルの場所。環境変数 `JEV_QUESTIONS` で差し替える(既定 data/questions_v2.json)。
    相対パスは、今いるディレクトリに無ければプロジェクトのルートからの相対とみなす。"""
    if not value:
        return DATA_DIR / "questions_v2.json"
    path = Path(value)
    if not path.is_absolute() and not path.exists():
        path = DATA_DIR.parent / path
    return path


QUESTIONS_PATH = questions_path(os.environ.get("JEV_QUESTIONS"))
MODE_ORDER: tuple[str, ...] = tuple(q21.MODE_EVENTS)          # fixed1, fixed2, adaptive
RAW_NAME = "raw.jsonl"
CLOSED_BOOK_NAME = "closed_book.jsonl"
REVIEW_NAME = "review.md"
SUMMARY_NAME = "summary.md"
META_NAME = "meta.json"

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


# ============================================================== 実行ディレクトリ・質問

def run_dir(run: str, eval_dir: Path | None = None) -> Path:
    """`results/eval/<run>`。run 名に `/` や `..` は使わせない。"""
    if not _RUN_NAME_RE.match(run or "") or ".." in run:
        raise ValueError(f"run 名は英数字と _ . - だけにしてください: {run!r}")
    return Path(eval_dir or EVAL_DIR) / run


def load_questions(qids: Sequence[str] | None = None, *, chunk_ids: set[str] | None = None,
                   path: Path = QUESTIONS_PATH) -> list[dict]:
    """質問ファイルから質問を引く(qids 省略時は全部、ファイルの順)。
    無い qid・チャンクに無い required_chunks(と required_any)は ValueError。"""
    questions = load_json(path, default=[]) or []
    by_id = {q["id"]: q for q in questions}
    if qids:
        missing = [q for q in qids if q not in by_id]
        if missing:
            raise ValueError(f"{path.name} にありません: {missing}")
        picked = [by_id[q] for q in dict.fromkeys(qids)]
    else:
        picked = list(questions)
    if chunk_ids is not None:
        for q in picked:
            unknown = [c for c in q.get("required_chunks") or [] if c not in chunk_ids]
            if unknown:
                raise ValueError(f"{q['id']} の required_chunks が chunks.json にありません: {unknown}")
            unknown = [c for group in q.get("required_any") or [] for c in group if c not in chunk_ids]
            if unknown:
                raise ValueError(f"{q['id']} の required_any が chunks.json にありません: {unknown}")
    return picked


def load_raw(path: Path) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    """raw.jsonl を読む。`(qid, mode)` ごとに**最後の成功行**を返し、失敗行は別に返す
    (`--append` で失敗分だけやり直したときに、後の成功が勝つように)。"""
    ok: dict[tuple[str, str], dict] = {}
    failed: list[dict] = []
    for row in iter_jsonl(path):
        if row.get("status") == "ok" and isinstance(row.get("body"), Mapping):
            ok[(row["qid"], row["mode"])] = row
        else:
            failed.append(row)
    # 後で成功した組の失敗は数えない
    failed = [r for r in failed if (r.get("qid"), r.get("mode")) not in ok]
    return ok, failed


def load_closed_book(path: Path) -> dict[str, dict]:
    """closed_book.jsonl を qid ごとの最後の行にする。"""
    out: dict[str, dict] = {}
    for row in iter_jsonl(path):
        out[row["qid"]] = row
    return out


# ============================================================== 採点欄(31 番が書き、32 番が読む)

GRADE_FIELDS: tuple[tuple[str, str], ...] = (
    ("correct", "正誤"),
    ("grounded", "根拠"),
    ("consistent", "回答文と引用の食い違い"),
    ("memo", "メモ"),
)
MARK_FIELDS = ("correct", "grounded", "consistent")
_LABEL_TO_KEY = {label: key for key, label in GRADE_FIELDS}

_MARKS = {
    "○": "○", "〇": "○", "◯": "○", "o": "○", "O": "○",
    "×": "×", "x": "×", "X": "×", "✕": "×", "✗": "×", "╳": "×",
    "△": "△", "▲": "△",
}

_BLOCK_RE = re.compile(r"<!--\s*grade\s+(\S+)\s+(\S+)\s*-->(.*?)<!--\s*/grade\s*-->", re.DOTALL)
_LINE_RE = re.compile(r"^\s*[-*]\s*(" + "|".join(re.escape(lbl) for _k, lbl in GRADE_FIELDS)
                      + r")\s*[:：]\s?(.*)$")


def grade_block(qid: str, mode: str) -> str:
    """空の採点欄。マーカーの HTML コメントは 32 番が読むので消さないこと。"""
    lines = [f"<!-- grade {qid} {mode} -->"]
    lines += [f"- {label}: " for _key, label in GRADE_FIELDS]
    lines.append("<!-- /grade -->")
    return "\n".join(lines)


def normalize_mark(value: str) -> str | None:
    """○/×/△ にそろえる。空なら None。先頭の 1 文字で判定し、読めなければ "?"。"""
    text = (value or "").strip()
    if not text:
        return None
    return _MARKS.get(text[0], "?")


def parse_grades(text: str) -> dict[tuple[str, str], dict]:
    """review.md から採点欄を読む。`{(qid, mode): {correct, grounded, consistent, memo, raw}}`。
    correct/grounded/consistent は ○/×/△/None/"?"(読めない)、memo は文字列(空なら "")。"""
    grades: dict[tuple[str, str], dict] = {}
    for match in _BLOCK_RE.finditer(text):
        qid, mode, body = match.group(1), match.group(2), match.group(3)
        entry: dict[str, Any] = {"correct": None, "grounded": None, "consistent": None, "memo": "",
                                 "raw": {}}
        for line in body.splitlines():
            m = _LINE_RE.match(line)
            if not m:
                continue
            key = _LABEL_TO_KEY[m.group(1)]
            value = m.group(2).strip()
            entry["raw"][key] = value
            if key == "memo":
                entry["memo"] = value
            else:
                entry[key] = normalize_mark(value)
        grades[(qid, mode)] = entry
    return grades


def has_filled_grades(text: str) -> bool:
    """採点欄のどれか 1 つでも書き込まれているか(31 番の上書き防止)。"""
    for entry in parse_grades(text).values():
        if any(entry[k] is not None for k in MARK_FIELDS) or entry["memo"]:
            return True
    return False


# ============================================================== トレースの読み方

def chunks_upto(body: Mapping[str, Any], hop: int) -> list[str]:
    """ホップ `hop` を読み終えた時点の累積チャンク ID(最初に取ったホップが hop 以下のもの)。"""
    return [c["id"] for c in body.get("chunks") or [] if c.get("hop") is not None and c["hop"] <= hop]


def last_hop(body: Mapping[str, Any]) -> int:
    hops = body.get("hops") or []
    return int(hops[-1]["hop"]) if hops else 0


def cited_ids(body: Mapping[str, Any]) -> list[str]:
    seen: dict[str, None] = {}
    for c in body.get("citations") or []:
        for cid in c.get("chunk_ids") or []:
            seen.setdefault(cid, None)
    return list(seen)


def answer_chunk_ids(body: Mapping[str, Any]) -> list[str]:
    """Haiku に渡したチャンク ID(回答前の絞り込みの後)。`answer_chunk_ids` の無い古いトレースは
    集めたチャンク全部を渡していたので `chunks` の ID を返す。"""
    if "answer_chunk_ids" in body:
        return list(body.get("answer_chunk_ids") or [])
    return [c["id"] for c in body.get("chunks") or []]


# ============================================================== 正解の基準

def required_groups(question: Mapping[str, Any] | None) -> list[list[str]]:
    """正解に要るチャンクの組の候補。**どれか 1 組が全部そろえば足りる**。"""
    if not question:
        return []
    if "required_any" in question and question["required_any"] is not None:
        return [list(group) for group in question["required_any"] if group]
    required = list(question.get("required_chunks") or [])
    return [required] if required else []


def groups_check(groups: Sequence[Sequence[str]], collected: Iterable[str], cited: Iterable[str]) -> dict:
    """組の候補と、集めたチャンク・引用を照合する。組が無ければ判定は None(答えのない質問)。"""
    have, used = set(collected), set(cited)
    if not groups:
        return {"groups": [], "collected_any": None, "cited_any": None,
                "collected_group": None, "cited_group": None}
    got = next((list(g) for g in groups if all(c in have for c in g)), None)
    cit = next((list(g) for g in groups if all(c in used for c in g)), None)
    return {"groups": [list(g) for g in groups], "collected_any": got is not None, "cited_any": cit is not None,
            "collected_group": got, "cited_group": cit}


def question_type(question: Mapping[str, Any] | None, fallback: Any = None) -> Any:
    """`type_v2` があればそれ、無ければ `type`。"""
    if not question:
        return fallback
    return question.get("type_v2") or question.get("type") or fallback


def model_answer(question: Mapping[str, Any] | None) -> Any:
    """模範解答。`answer_v2` があればそれ、無ければ `answer`。"""
    if not question:
        return None
    return question.get("answer_v2") if question.get("answer_v2") is not None else question.get("answer")


def matched_term(question: str, entity: Mapping[str, Any]) -> str | None:
    """エンティティの正式名・別名のどれかが質問文に部分一致すれば、その語を返す(長い語を優先)。
    英字は大文字小文字を区別しない。"""
    terms = [entity.get("name") or ""] + list(entity.get("aliases") or [])
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    q = question.casefold()
    for t in terms:
        if t.casefold() in q:
            return t
    return None


# ============================================================== しきい値の候補探し

def confusion(samples: Iterable[tuple[float, bool]], threshold: float) -> dict:
    """`score >= threshold` を陽性と予測したときの混同行列と balanced accuracy。
    陽性(または陰性)が 0 件なら、その側の率は計算せず、ある方だけで平均する。"""
    tp = fp = fn = tn = 0
    for score, label in samples:
        pred = score >= threshold
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    rates = []
    if tp + fn:
        rates.append(tp / (tp + fn))
    if tn + fp:
        rates.append(tn / (tn + fp))
    bal = sum(rates) / len(rates) if rates else None
    n = tp + fp + fn + tn
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "balanced_accuracy": bal, "accuracy": (tp + tn) / n if n else None}


def threshold_candidates(samples: Sequence[tuple[float, bool]], top: int = 5) -> list[dict]:
    """両者を最もよく分けるしきい値の候補。候補は「隣り合うスコアの中点」(と最小値・最大値の外側)。"""
    scores = sorted({round(float(s), 6) for s, _l in samples})
    if not scores:
        return []
    cands = [max(scores[0] - 0.01, 0.0), min(scores[-1] + 0.01, 1.0)]
    cands += [round((a + b) / 2, 4) for a, b in zip(scores, scores[1:])]
    results = [confusion(samples, t) for t in sorted(set(cands))]
    results.sort(key=lambda r: (-(r["balanced_accuracy"] or 0.0), -(r["accuracy"] or 0.0),
                                -r["threshold"]))
    return results[:top]


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def md_cell(text: Any) -> str:
    """表のセルに入れる文字列(改行と | を潰す)。"""
    return str("" if text is None else text).replace("\n", " ").replace("|", "／")
