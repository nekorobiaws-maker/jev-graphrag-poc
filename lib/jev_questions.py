#!/usr/bin/env python3
"""Jev に渡す問いの組み立てと、結果の読み出し。問いの文面の正本はこのファイル。

問いの形式は `{"type": "noul", "instructions": ..., "criteria": {"true": ..., "false": ...}}`(criteria は必ず付ける)。"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

NODE_QID_PREFIX = "node_"
ALIAS_SEP = "、"
NO_ALIAS = "なし"

NODE_CRITERIA = {
    "true": "このエンティティ本人に言及している",
    "false": "言及していない、または同名・部分一致の別物",
}


def noul_value(record: Mapping[str, Any], qid: str) -> float | None:
    """Noul の確率(はいの確からしさ)。取れなければ None。"""
    answer = ((record.get("response") or {}).get("answers") or {}).get(qid)
    if not isinstance(answer, Mapping):
        return None
    value = answer.get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# ============================================================== 1 リクエストの大きさ(64k トークン制限への対応)

TOKENS_PER_CHAR = 1.5                 # 概算の係数(ingest_core / query_core もここを使う)
MAX_REQUEST_TOKENS_EST = 40_000       # 1 リクエストの見積もりトークンの上限(64k に対して余裕を見る)
MAX_QUESTIONS_PER_REQUEST = 100       # 1 リクエストの問いの数の上限


def request_chars(state: Any, questions: Mapping[str, Any]) -> int:
    """実際に送る JSON(`{"state": ..., "questions": ...}`)の文字数。"""
    return len(json.dumps({"state": state, "questions": questions}, ensure_ascii=False))


def estimate_tokens(chars: int) -> int:
    """文字数から見積もりトークン(文字数 × TOKENS_PER_CHAR)。"""
    return int(chars * TOKENS_PER_CHAR)


def split_questions(
    state: Any,
    questions: Mapping[str, Any],
    *,
    max_tokens: int = MAX_REQUEST_TOKENS_EST,
    max_questions: int = MAX_QUESTIONS_PER_REQUEST,
) -> list[dict[str, Any]]:
    """問いを 1 リクエストぶんずつに分ける。並び順を保ち、前から貪欲に詰める。"""
    if max_questions < 1:
        raise ValueError(f"max_questions は 1 以上です: {max_questions}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens は 1 以上です: {max_tokens}")
    items = list(questions.items())
    if not items:
        return []
    # JSON の文字数は「state と空の questions」+「各問いの `"qid": {...}`」+「区切りの `, `」で足し算できる
    base = request_chars(state, {})
    pieces = [len(json.dumps({qid: q}, ensure_ascii=False)) - 2 for qid, q in items]
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    chars = base
    for (qid, q), piece in zip(items, pieces):
        added = piece + (2 if current else 0)
        if current and (len(current) >= max_questions or estimate_tokens(chars + added) > max_tokens):
            groups.append(current)
            current, chars, added = {}, base, piece
        if estimate_tokens(base + piece) > max_tokens:
            raise ValueError(f"問い {qid!r} は 1 個だけで見積もり {estimate_tokens(base + piece):,} トークンあり、"
                             f"上限 {max_tokens:,} を超えています")
        current[qid] = q
        chars += added
    groups.append(current)
    return groups


def split_items(
    state: Any,
    items: Sequence[Any],
    to_questions: Any,
    qid_of: Any,
    *,
    max_tokens: int = MAX_REQUEST_TOKENS_EST,
    max_questions: int = MAX_QUESTIONS_PER_REQUEST,
) -> list[list[Any]]:
    """`split_questions()` を、エンティティや候補の並び(items)のまま使う版。"""
    by_qid = {qid_of(item): item for item in items}
    groups = split_questions(state, to_questions(items), max_tokens=max_tokens, max_questions=max_questions)
    return [[by_qid[qid] for qid in group] for group in groups]


# ============================================================== ノード判定(登録)

def node_qid(entity_id: str) -> str:
    """エンティティ id から問いの id(`node_<id>`)を作る。"""
    return f"{NODE_QID_PREFIX}{entity_id}"


def node_instructions(entity: Mapping[str, Any]) -> str:
    """ノード判定の instructions。別名が無ければ「なし」と書く。"""
    aliases = ALIAS_SEP.join(entity.get("aliases") or []) or NO_ALIAS
    return (
        f"本文に『{entity['name']}』（別名: {aliases}／種別: {entity['type']}／"
        f"{entity['description']}）そのものが登場するか。"
        "呼び名の一部が他の語に含まれるだけ、または同じ呼び名の別物なら登場しない"
    )


def node_question(entity: Mapping[str, Any]) -> dict:
    """1エンティティぶんの Noul の問い。"""
    return {
        "type": "noul",
        "instructions": node_instructions(entity),
        "criteria": dict(NODE_CRITERIA),
    }


def node_questions(entities: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """`{node_<id>: 問い}`。エンティティの並び順を保つ。id が重複していたら ValueError。"""
    questions: dict[str, dict] = {}
    for entity in entities:
        qid = node_qid(entity["id"])
        if qid in questions:
            raise ValueError(f"問いの id が重複しています: {qid!r}")
        questions[qid] = node_question(entity)
    return questions


def node_scores(record: Mapping[str, Any], entity_ids: Iterable[str]) -> dict[str, float | None]:
    """呼び出しレコードから `{entity_id: prob}` を読み出す。取れなかったものは None。"""
    return {eid: noul_value(record, node_qid(eid)) for eid in entity_ids}


# ============================================================== 関係判定(登録)

EDGE_QID_PREFIX = "edge_"
EDGE_QID_SEP = "__"

EDGE_CRITERIA = {
    "true": "本文に明記されている、または本文から直接読み取れる",
    "false": "本文に書かれていない、または推測が必要",
}


def edge_qid(source_id: str, edge_index: int, target_id: str) -> str:
    """関係判定の問いの id(`edge_<source_id>__e<NN>__<target_id>`)。"""
    return f"{EDGE_QID_PREFIX}{source_id}{EDGE_QID_SEP}e{edge_index:02d}{EDGE_QID_SEP}{target_id}"


def edge_instructions(sentence: str) -> str:
    """関係判定の instructions。候補文は `master.render()` で正式名を埋めたもの。"""
    return f"本文は次の文を裏付けるか：『{sentence}』"


def edge_question(sentence: str) -> dict:
    """1 候補文ぶんの Noul の問い。"""
    return {
        "type": "noul",
        "instructions": edge_instructions(sentence),
        "criteria": dict(EDGE_CRITERIA),
    }


def edge_questions(candidates: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """候補(`ingest_core.edge_candidates()` の要素)から `{qid: 問い}`。並び順を保つ。
    qid が重複していたら ValueError。"""
    questions: dict[str, dict] = {}
    for cand in candidates:
        qid = cand["qid"]
        if qid in questions:
            raise ValueError(f"問いの id が重複しています: {qid!r}")
        questions[qid] = edge_question(cand["sentence"])
    return questions


def edge_scores(record: Mapping[str, Any], qids: Iterable[str]) -> dict[str, float | None]:
    """呼び出しレコードから `{qid: prob}` を読み出す。取れなかったものは None。"""
    return {qid: noul_value(record, qid) for qid in qids}


# ============================================================== 検索
#
# query Lambda の 3 種類の判定。どれも state に `question` を入れる。
# - 前段判定: `in_scope` と、質問文に名前が書かれたノード `node_<id>`×エンティティ数。
#   `split_questions()` で分ける(in_scope は先頭の組)
# - 十分性判定: `sufficient` 1 問。state に累積チャンク全件 `chunks: [{id, text}]` を入れる
# - 隣ノード選択: 候補文ごとに `nb_<NNN>`(候補の通し番号。3 桁)。最大 50 問ずつ 1 リクエスト
#   (見積もりトークンの上限でも分ける)

IN_SCOPE_QID = "in_scope"
SUFFICIENCY_QID = "sufficient"
NEIGHBOR_QID_PREFIX = "nb_"

IN_SCOPE_INSTRUCTIONS = (
    "この質問は、漫画・アニメ作品（ONE PIECE、名探偵コナン、名探偵プリキュア!。プリキュアシリーズを含む）について、"
    "作品の内容（登場人物・組織・設定・出来事・道具など）と、"
    "作品そのものの情報（作者・出版社・掲載誌・連載や放送の時期・放送局・制作会社・シリーズ・映画化・コラボなど）の"
    "どちらかを聞く質問か"
)
IN_SCOPE_CRITERIA = {
    "true": "上記の作品の内容、または作品そのものの情報についての質問",
    "false": "上記の作品と無関係な質問（ほかの作品だけについての質問、一般的な雑談、天気、計算など）",
}

QUESTION_NODE_CRITERIA = {
    "true": "質問文に、このエンティティを指す名前・別名が明示されている",
    "false": "質問文に書かれていない（答えとして連想されるだけのものを含む）、または同名・部分一致の別物",
}

SUFFICIENCY_INSTRUCTIONS = "これらの本文だけで質問に答えられるか"
SUFFICIENCY_CRITERIA = {
    "true": "本文の記述だけで答えが確定する",
    "false": "答えに必要な情報が本文に欠けている、または一般知識が必要",
}

NEIGHBOR_CRITERIA = {
    "true": "この関係の先を調べると、質問の答えやその手がかりが見つかりそう",
    "false": "質問と関係がない、または答えにつながらない",
}


def in_scope_question() -> dict:
    """前段判定の「作品についての質問か」。"""
    return {"type": "noul", "instructions": IN_SCOPE_INSTRUCTIONS, "criteria": dict(IN_SCOPE_CRITERIA)}


def question_node_instructions(entity: Mapping[str, Any]) -> str:
    """前段判定の「**質問文に名前が書かれているか**」。"""
    aliases = ALIAS_SEP.join(entity.get("aliases") or []) or NO_ALIAS
    return (
        f"質問文の中に、『{entity['name']}』またはその別名（{aliases}）が文字として書かれているか"
        f"（種別: {entity['type']}／{entity['description']}）。"
        "質問から答えとして連想される・推測されるだけのもの、質問文に書かれていないものは該当しない。"
        "名前の一部が別の語に含まれるだけ（例：作品名『名探偵コナン』の中の「コナン」）も該当しない"
    )


def question_node_question(entity: Mapping[str, Any]) -> dict:
    return {
        "type": "noul",
        "instructions": question_node_instructions(entity),
        "criteria": dict(QUESTION_NODE_CRITERIA),
    }


def question_node_questions(entities: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """`{node_<id>: 問い}`(質問に登場するか)。id が重複していたら ValueError。"""
    questions: dict[str, dict] = {}
    for entity in entities:
        qid = node_qid(entity["id"])
        if qid in questions:
            raise ValueError(f"問いの id が重複しています: {qid!r}")
        questions[qid] = question_node_question(entity)
    return questions


def entry_questions(entities: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """前段判定の 1 リクエストぶん: `in_scope` + `node_<id>`×エンティティ数。"""
    return {IN_SCOPE_QID: in_scope_question(), **question_node_questions(entities)}


# ============================================================== 前段判定の聞き直し

ENTRY_TYPE_QID = "entry_type"
ENTRY_TYPE_LABEL_PREFIX = "type_"
SEMANTIC_QID_PREFIX = "sem_"

ENTRY_TYPE_INSTRUCTIONS = (
    "この質問が主に尋ねている対象のうち、質問文の中で起点（主語）として言及されているもの"
    "（名前のほか、言い換え・説明的な言い方・通称で書かれていてもよい）は、次のどの種別か。"
    "質問の答えとして尋ねられているものの種別ではなく、質問文に出てくる起点の種別を選ぶ"
)

SEMANTIC_ENTRY_CRITERIA = {
    "true": "質問文中のある表現が、このエンティティそのものを指している",
    "false": "指していない、または答えとして連想されるだけ",
}


def entry_type_label(index: int) -> str:
    """種別の Choice のラベル(`type_00` から。node_types の添字)。"""
    return f"{ENTRY_TYPE_LABEL_PREFIX}{index:02d}"


def entry_type_question(node_types: Sequence[Mapping[str, Any]]) -> dict:
    """種別の判定(Choice 1 問)。criteria は `{type_NN: "種別名: 説明"}`。"""
    if not node_types:
        raise ValueError("node_types が空です")
    criteria = {entry_type_label(i): f"{t['name']}：{t.get('description') or t['name']}"
                for i, t in enumerate(node_types)}
    return {"type": "choice", "instructions": ENTRY_TYPE_INSTRUCTIONS, "criteria": criteria}


def entry_type_questions(node_types: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    return {ENTRY_TYPE_QID: entry_type_question(node_types)}


def entry_type_scores(record: Mapping[str, Any], node_types: Sequence[Mapping[str, Any]]) -> list[dict]:
    """種別の判定の結果を `[{type, score}]`(スコアの高い順。同点は node_types の順)にする。"""
    answer = ((record.get("response") or {}).get("answers") or {}).get(ENTRY_TYPE_QID)
    if not isinstance(answer, Mapping):
        return []
    by_label = {entry_type_label(i): t["name"] for i, t in enumerate(node_types)}
    order = {t["name"]: i for i, t in enumerate(node_types)}
    scores: dict[str, float] = {}
    probs = answer.get("probabilities")
    if isinstance(probs, Mapping):
        for label, value in probs.items():
            if label in by_label and isinstance(value, (int, float)) and not isinstance(value, bool):
                scores[by_label[label]] = float(value)
    if not scores:
        label, conf = answer.get("choice"), answer.get("confidence")
        if label in by_label and isinstance(conf, (int, float)) and not isinstance(conf, bool):
            scores[by_label[label]] = float(conf)
    out = [{"type": name, "score": s} for name, s in scores.items()]
    out.sort(key=lambda item: (-item["score"], order[item["type"]]))
    return out


def select_entry_types(type_scores: Sequence[Mapping[str, Any]], cover: float, max_types: int,
                       min_score: float = 0.0) -> list[str]:
    """スコアの高い順に、合計が cover 以上になるまで(最大 max_types 種別)採用する。
    min_score 未満の種別は採用しない。"""
    picked: list[str] = []
    total = 0.0
    for item in type_scores:
        if len(picked) >= max_types or total >= cover:
            break
        if item["score"] is None or item["score"] < min_score or item["score"] <= 0.0:
            break
        picked.append(item["type"])
        total += item["score"]
    return picked


def semantic_qid(entity_id: str) -> str:
    """聞き直しの問いの id(`sem_<id>`)。"""
    return f"{SEMANTIC_QID_PREFIX}{entity_id}"


def semantic_entry_instructions(entity: Mapping[str, Any]) -> str:
    """聞き直しの instructions。「文字として書かれているか」ではなく「同じものを指しているか」。"""
    aliases = ALIAS_SEP.join(entity.get("aliases") or []) or NO_ALIAS
    return (
        "質問の中の言葉（言い換え・説明的な言い方・通称を含む）が、"
        f"『{entity['name']}』（別名: {aliases}／種別: {entity['type']}／{entity['description']}）"
        "と同じものを指しているか。"
        "質問から答えとして連想されるだけのもの（質問が尋ねている答えの側）は該当しない"
    )


def semantic_entry_question(entity: Mapping[str, Any]) -> dict:
    return {
        "type": "noul",
        "instructions": semantic_entry_instructions(entity),
        "criteria": dict(SEMANTIC_ENTRY_CRITERIA),
    }


def semantic_entry_questions(entities: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """`{sem_<id>: 問い}`。並び順を保つ。id が重複していたら ValueError。"""
    questions: dict[str, dict] = {}
    for entity in entities:
        qid = semantic_qid(entity["id"])
        if qid in questions:
            raise ValueError(f"問いの id が重複しています: {qid!r}")
        questions[qid] = semantic_entry_question(entity)
    return questions


def sufficiency_question() -> dict:
    return {"type": "noul", "instructions": SUFFICIENCY_INSTRUCTIONS,
            "criteria": dict(SUFFICIENCY_CRITERIA)}


def sufficiency_questions() -> dict[str, dict]:
    return {SUFFICIENCY_QID: sufficiency_question()}


def neighbor_qid(index: int) -> str:
    """隣ノード候補の問いの id(`nb_000` から)。"""
    return f"{NEIGHBOR_QID_PREFIX}{index:03d}"


def neighbor_instructions(sentence: str) -> str:
    """隣ノード選択の instructions。候補文は `master.render()` で正式名を埋めたもの。"""
    return f"この関係は、質問に答えるための手がかりになるか：『{sentence}』"


def neighbor_question(sentence: str) -> dict:
    return {"type": "noul", "instructions": neighbor_instructions(sentence),
            "criteria": dict(NEIGHBOR_CRITERIA)}


def neighbor_questions(candidates: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """候補(`query_core.neighbor_candidates()` の要素。`qid` と `sentence` を持つ)から `{qid: 問い}`。"""
    questions: dict[str, dict] = {}
    for cand in candidates:
        qid = cand["qid"]
        if qid in questions:
            raise ValueError(f"問いの id が重複しています: {qid!r}")
        questions[qid] = neighbor_question(cand["sentence"])
    return questions


# ============================================================== 入口の登場チャンクの選び方
#
# 入口ノードの登場チャンクから、ホップ 0 で足す上位を選ぶ。チャンク本文ごとに 1 問
# 「この本文は質問に答える情報を含むか」を聞き、そのスコア順に並べる(登場スコアの上位だと、
# 属性を問う質問で答えの書かれたチャンクが上位に入らないことがあるため)。
# state は `{"question"}`、本文は instructions に入れる。qid は `ec_<NNN>`(3 桁の通し番号)。

ENTRY_CHUNK_QID_PREFIX = "ec_"

ENTRY_CHUNK_CRITERIA = {
    "true": "本文に、質問の答え、または答えにたどり着く手がかり（人物の属性・関係・出来事など）が書かれている",
    "false": "質問と関係がない、または質問の対象が登場するだけで答えにつながる記述がない",
}


def entry_chunk_qid(index: int) -> str:
    """入口の登場チャンクの問いの id(`ec_000` から)。"""
    return f"{ENTRY_CHUNK_QID_PREFIX}{index:03d}"


def entry_chunk_instructions(text: str) -> str:
    """入口の登場チャンクの instructions。本文はそのまま埋める。"""
    return f"この本文は、質問に答えるための情報を含むか：『{text}』"


def entry_chunk_question(text: str) -> dict:
    return {"type": "noul", "instructions": entry_chunk_instructions(text),
            "criteria": dict(ENTRY_CHUNK_CRITERIA)}


def entry_chunk_questions(items: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """`[{qid, text}]`(`query_core.entry_chunk_items()` の要素)から `{qid: 問い}`。並び順を保つ。
    qid が重複していたら ValueError。"""
    questions: dict[str, dict] = {}
    for item in items:
        qid = item["qid"]
        if qid in questions:
            raise ValueError(f"問いの id が重複しています: {qid!r}")
        questions[qid] = entry_chunk_question(item["text"])
    return questions


# ============================================================== 回答前の絞り込み
#
# ホップのループが終わったあと、累積チャンク(最大 max_chunks 件)のそれぞれに、入口の登場チャンクと
# **同じ問い・同じ criteria**(`entry_chunk_question()`)を聞き、スコアの高いものだけを Haiku に渡す。
# qid だけ別の接頭辞 `af_<NNN>`(累積の通し番号)にして、入口チャンクの `ec_` と重ならないようにする。

ANSWER_FILTER_QID_PREFIX = "af_"


def answer_filter_qid(index: int) -> str:
    """回答前の絞り込みの問いの id(`af_000` から)。"""
    return f"{ANSWER_FILTER_QID_PREFIX}{index:03d}"


def answer_filter_questions(items: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """`[{qid, text}]`(`query_core.answer_filter_items()` の要素)から `{qid: 問い}`。問いの中身は
    入口チャンクと同じ(`entry_chunk_question()`)。並び順を保つ。qid が重複していたら ValueError。"""
    questions: dict[str, dict] = {}
    for item in items:
        qid = item["qid"]
        if qid in questions:
            raise ValueError(f"問いの id が重複しています: {qid!r}")
        questions[qid] = entry_chunk_question(item["text"])
    return questions
