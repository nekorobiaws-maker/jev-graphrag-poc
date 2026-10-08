#!/usr/bin/env python3
"""検索処理の中身。query Lambda から使う。

1. 前段判定: 辞書(`entry_dictionary`)で質問文から候補を拾い、Jev に `in_scope` と候補ごとの「書かれているか」を聞く。
   入口(entry_threshold 以上の上位 max_neighbors 件)が 0 件なら、Jev に種別を選ばせてその種別のエンティティだけに
   聞き直す(`entry_fallback="semantic"`)。それでも 0 件なら Bedrock を呼ばずに「回答できません」を返す
2. ホップ 0: 入口ノードの登場チャンクを読み、Jev の「質問に答える情報を含むか」のスコア順に上位を取る
3. 十分性判定(adaptive のみ、各ホップの後): 累積チャンク全件で Jev に聞き、しきい値以上で終了
4. ホップ n: 前のホップで進んだノードの関係(OUT/IN/SYM)から候補文を作り、Jev の「手がかりになるか」の
   スコア上位の隣ノードへ進み、たどった関係の根拠チャンクを足す
5. 終了: `sufficient` / `max_chunks` / `max_hops` / `no_neighbors` / `jev_cap` / `deadline`
6. 回答: Haiku に累積チャンクを `[id] 本文` の形で渡し、根拠チャンク ID 付きの JSON で答えさせる
   (`answer_filter="jev"` なら事前に Jev で絞る)。引用の検査は Haiku に渡したチャンクだけを正とする

fixed モードは十分性判定をしないことだけが違う。`max_hops=1` は「入口(ホップ 0)+隣 1 段」。
締め切り(`deadline`)を渡すと、各呼び出しの前に残り時間を見て足りなければ打ち切り、集めたチャンクで回答する。

依存は全部引数で受け取る。
- `jev`: `judge(state, questions, use_cache=...)` を持つ物(本物は `jev_client.JevClient`)
- `store`: `query_node(entity_id)` と `batch_get_chunks(ids)` を持つ物(本物は `graph_store.GraphStore`)
- `llm`: `run(task, use_cache=...)` を持つ物(本物は `bedrock_llm.ConverseCaller`)。失敗時は None を返す"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

from bedrock_llm import converse_key, converse_payload
from common import DEFAULT_GEN_MODEL, JEV_MODEL, PRICES, strip_code_fence
from jev_questions import (
    IN_SCOPE_QID,
    answer_filter_qid,
    answer_filter_questions,
    SUFFICIENCY_QID,
    TOKENS_PER_CHAR,
    request_chars,
    split_items,
    split_questions,
    entry_chunk_qid,
    entry_chunk_questions,
    entry_questions,
    entry_type_questions,
    entry_type_scores,
    neighbor_qid,
    neighbor_questions,
    node_qid,
    noul_value,
    select_entry_types,
    semantic_entry_questions,
    semantic_qid,
    sufficiency_questions,
)
from entry_dictionary import find_candidates
from master import entity_by_id, render

MODES = ("adaptive", "fixed")

DEFAULT_ENTRY_THRESHOLD = 0.5
ENTRY_LOG_MIN_SCORE = 0.1              # トレースの entry に残す候補の下限(選外も含めて残す)
DEFAULT_SUFFICIENCY_THRESHOLD = 0.7
DEFAULT_MAX_HOPS = 3
DEFAULT_MAX_NEIGHBORS = 3
DEFAULT_MIN_NEIGHBOR_SCORE = 0.5
EXCLUDED_BELOW_MIN = "below_min"       # neighbor_scores の excluded: 下限未満で選外
DEFAULT_MAX_CHUNKS = 10
DEFAULT_MAX_CHUNKS_PER_NODE = 5
DEFAULT_MAX_CHUNKS_PER_HOP = 3
# ホップ先から取るチャンク: edge_evidence = たどった関係の根拠チャンク / appearance = 登場チャンクの上位
CHUNK_SOURCE_APPEARANCE = "appearance"
CHUNK_SOURCE_EDGE_EVIDENCE = "edge_evidence"
HOP_CHUNK_SOURCES = (CHUNK_SOURCE_EDGE_EVIDENCE, CHUNK_SOURCE_APPEARANCE)
DEFAULT_HOP_CHUNK_SOURCE = CHUNK_SOURCE_EDGE_EVIDENCE
# 入口の登場チャンクの並べ方: jev = Jev の「質問に答える情報を含むか」のスコア順 / appearance = 登場スコア順
ENTRY_CHUNK_RANK_JEV = "jev"
ENTRY_CHUNK_RANK_APPEARANCE = "appearance"
ENTRY_CHUNK_RANKS = (ENTRY_CHUNK_RANK_JEV, ENTRY_CHUNK_RANK_APPEARANCE)
DEFAULT_ENTRY_CHUNK_RANK = ENTRY_CHUNK_RANK_JEV
ENTRY_CHUNK_MAX_QUESTIONS = 100        # 入口チャンクの判定の 1 リクエストの問いの上限(split_items で分ける)
ENTRY_CHUNKS_EST = 150                 # 見積もり用: 入口の登場チャンクの合計(上振れ寄り)
IN_SCOPE_THRESHOLD = 0.5
# 前段判定で Jev に聞くエンティティ: dictionary = 辞書で拾った候補だけ / none = 全エンティティ
ENTRY_PREFILTER_DICTIONARY = "dictionary"
ENTRY_PREFILTER_NONE = "none"
ENTRY_PREFILTERS = (ENTRY_PREFILTER_DICTIONARY, ENTRY_PREFILTER_NONE)
DEFAULT_ENTRY_PREFILTER = ENTRY_PREFILTER_DICTIONARY
# 入口が 0 件のときの聞き直し: semantic = 種別を選ばせ、その種別のエンティティに意味で聞く / none = しない
ENTRY_FALLBACK_SEMANTIC = "semantic"
ENTRY_FALLBACK_NONE = "none"
ENTRY_FALLBACKS = (ENTRY_FALLBACK_SEMANTIC, ENTRY_FALLBACK_NONE)
DEFAULT_ENTRY_FALLBACK = ENTRY_FALLBACK_SEMANTIC
DEFAULT_SEMANTIC_ENTRY_THRESHOLD = 0.7    # 聞き直しの入口のしきい値(辞書の entry_th より高め)
SEMANTIC_TYPE_COVER = 0.8                 # 種別はスコアの高い順に、合計がこれ以上になるまで採用
SEMANTIC_MAX_TYPES = 2                    # 採用する種別の上限
SEMANTIC_TYPE_MIN_SCORE = 0.1             # これ未満の種別は採用しない(2 つ目に小さな種別を足さない)
SEMANTIC_MAX_QUESTIONS = 100              # 聞き直しの 1 リクエストの問いの上限(split_items で分ける)
# entry_chunk_ranking.skipped: 入口チャンクの判定を呼ばなかった理由(None = 呼んだ)
ENTRY_RANK_SKIP_WITHIN_CAPACITY = "within_capacity"   # 全部取れる件数なので並べても結果が同じ
ENTRY_RANK_SKIP_NO_CHUNKS = "no_chunks"               # 入口に登場チャンクが無い
VIA_DICTIONARY = "dictionary"
VIA_SEMANTIC = "semantic"
# 既に集めたチャンクの数え方: count = 候補の上位 N 件をそのまま取り新顔だけ足す / skip = 飛ばして新顔を N 件足す
HOP_DUP_COUNT = "count"
HOP_DUP_SKIP = "skip"
HOP_DUP_POLICIES = (HOP_DUP_COUNT, HOP_DUP_SKIP)
DEFAULT_HOP_DUP_POLICY = HOP_DUP_COUNT
# 回答前の絞り込み: jev = Jev のスコア上位だけを Haiku に渡す / none = 累積を全部渡す
ANSWER_FILTER_JEV = "jev"
ANSWER_FILTER_NONE = "none"
ANSWER_FILTERS = (ANSWER_FILTER_JEV, ANSWER_FILTER_NONE)
DEFAULT_ANSWER_FILTER = ANSWER_FILTER_NONE
DEFAULT_ANSWER_FILTER_THRESHOLD = 0.5    # これ以上のスコアのチャンクを残す
DEFAULT_ANSWER_FILTER_MAX = 5            # 残す件数の上限
# しきい値以上が 0 件のとき、スコア上位から残す件数(全部捨てて「回答できません」に倒れないように。
# 本当に答えが無い質問は Haiku が断る)。answer_filter_max より大きければ answer_filter_max 件
DEFAULT_ANSWER_FILTER_MIN = 2
ANSWER_FILTER_MAX_QUESTIONS = 100        # 1 リクエストの問いの上限(split_items で分ける。累積 10 件なら 1 回)
# answer_filter.fallback: 絞らずに全部渡した理由(None = 絞った)
ANSWER_FILTER_NO_SCORES = "no_scores"    # Jev の応答にスコアが 1 件も無かった

MAX_HOPS_LIMIT = 10                    # 引数の書き間違い対策の上限
HOP_MAX_CHUNKS_PER_NODE_LIMIT = 10_000
NEIGHBOR_BATCH_SIZE = 50               # 1 リクエストに詰める隣ノード候補の上限
# 1 質問あたりの Jev 呼び出しの上限(超えるなら jev_cap)。分割したリクエストも 1 回ずつ数える
MAX_JEV_CALLS = 20
NEIGHBOR_CANDIDATES_EST = 100          # 見積もり用: 1 ホップあたりの隣候補数(上振れ寄り)

# 締め切りの確認に使う「1 回の呼び出しの最悪時間」(秒)。Lambda 側の設定に合わせる:
# Jev は timeout 15 秒 × (1 + 再試行 1 回)+待ち、Bedrock は read_timeout 30 秒 + スロットルの待ち直し
JEV_RESERVE_SEC = 35.0
DDB_RESERVE_SEC = 5.0
GEN_RESERVE_SEC = 45.0

GEN_TEMPERATURE = 0.0
GEN_MAX_TOKENS = 1024

CANNOT_ANSWER = "回答できません"

STOP_REASONS = ("entry_judge_missing", "out_of_scope", "no_entry", "no_chunks", "sufficient",
                "max_hops", "max_chunks", "no_neighbors", "jev_cap", "deadline")

# 概算費用用の TOKENS_PER_CHAR は jev_questions から読む(ingest_core と同じ。実費は usage から出す)

ANSWER_STYLE_CONCISE = "concise"
ANSWER_STYLE_FULL = "full"
ANSWER_STYLES = (ANSWER_STYLE_CONCISE, ANSWER_STYLE_FULL)
DEFAULT_ANSWER_STYLE = ANSWER_STYLE_CONCISE

ANSWER_SYSTEM_FULL = (
    "あなたは、与えられた本文だけを根拠に質問に答えるアシスタントです。\n"
    "- 一般知識・事前知識は使わないでください。本文に書かれていないことは書かないでください。\n"
    f"- 本文に答えが無ければ answerable を false にして、answer は「{CANNOT_ANSWER}」としてください。\n"
    "- 回答に含まれるすべての主張に、根拠となる本文の ID(例: op-3)を 1 つ以上付けてください。"
    "ID は本文の先頭の [ ] の中の文字列をそのまま使ってください。\n"
    "- 出力は次の形の JSON オブジェクトだけにしてください。説明文やコードフェンスは付けないでください。\n"
    '{"answer": "回答の文章", "citations": [{"claim": "回答中の主張", "chunk_ids": ["op-3"]}], '
    '"answerable": true}'
)
ANSWER_SYSTEM = ANSWER_SYSTEM_FULL

# "concise" の system。full と同じルール(本文だけを根拠に・全主張に ID・JSON だけ)で言い回しを短くし、
# 出力を 1 行の JSON にさせる
ANSWER_SYSTEM_CONCISE = (
    "本文だけを根拠に質問に答える。一般知識・事前知識は使わない。\n"
    "- answer は答えだけを原則1〜2文で。質問に聞かれていない補足は書かない。\n"
    "- 質問が複数の意味に取れるときは、本文で答えられる意味で解釈して答え、どう解釈したかを一言添える。\n"
    f"- 本文に答えが無ければ answerable=false、answer=「{CANNOT_ANSWER}」。\n"
    "- citations は答えを裏付ける最小限の主張だけ。各主張に根拠の本文ID(本文先頭の[ ]内。例: op-3)を1つ以上。\n"
    "- 出力は次の形のJSONだけを1行で(改行・インデント・説明文・コードフェンスなし):\n"
    '{"answer":"…","citations":[{"claim":"…","chunk_ids":["op-3"]}],"answerable":true}'
)

ANSWER_SYSTEMS = {ANSWER_STYLE_CONCISE: ANSWER_SYSTEM_CONCISE, ANSWER_STYLE_FULL: ANSWER_SYSTEM_FULL}

RETRY_NOTE = (
    "\n\n前回の出力は JSON として読めませんでした。説明文やコードフェンスを付けず、"
    "指定の形の JSON オブジェクトだけを出力してください。"
)


class BadEvent(ValueError):
    """event・パラメータの書き間違い。**メッセージをそのまま応答・ログに出してよい唯一の例外**。"""


class _Halt(Exception):
    """ループの打ち切り(deadline / jev_cap)。run_query の中だけで使う。"""

    def __init__(self, reason: str, where: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.where = where


# ============================================================== パラメータ

def _number(name: str, value: Any, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadEvent(f"{name} は {lo}〜{hi} の数値です: {value!r}")
    value = float(value)
    if not lo <= value <= hi:
        raise BadEvent(f"{name} は {lo}〜{hi} の数値です: {value!r}")
    return value


def _integer(name: str, value: Any, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadEvent(f"{name} は {lo}〜{hi} の整数です: {value!r}")
    if not lo <= value <= hi:
        raise BadEvent(f"{name} は {lo}〜{hi} の整数です: {value!r}")
    return value


def make_params(
    *,
    mode: str = "adaptive",
    entry_th: Any = DEFAULT_ENTRY_THRESHOLD,
    sufficiency_th: Any = DEFAULT_SUFFICIENCY_THRESHOLD,
    max_hops: Any = DEFAULT_MAX_HOPS,
    max_neighbors: Any = DEFAULT_MAX_NEIGHBORS,
    max_chunks: Any = DEFAULT_MAX_CHUNKS,
    min_neighbor_score: Any = DEFAULT_MIN_NEIGHBOR_SCORE,
    max_chunks_per_node: Any = DEFAULT_MAX_CHUNKS_PER_NODE,
    hop_chunk_source: Any = DEFAULT_HOP_CHUNK_SOURCE,
    max_chunks_per_hop: Any = DEFAULT_MAX_CHUNKS_PER_HOP,
    entry_prefilter: Any = DEFAULT_ENTRY_PREFILTER,
    entry_fallback: Any = DEFAULT_ENTRY_FALLBACK,
    semantic_entry_th: Any = DEFAULT_SEMANTIC_ENTRY_THRESHOLD,
    entry_chunk_rank: Any = DEFAULT_ENTRY_CHUNK_RANK,
    answer_style: Any = DEFAULT_ANSWER_STYLE,
    hop_dup_policy: Any = DEFAULT_HOP_DUP_POLICY,
    answer_filter: Any = DEFAULT_ANSWER_FILTER,
    answer_filter_threshold: Any = DEFAULT_ANSWER_FILTER_THRESHOLD,
    answer_filter_max: Any = DEFAULT_ANSWER_FILTER_MAX,
    answer_filter_min: Any = DEFAULT_ANSWER_FILTER_MIN,
) -> dict:
    """パラメータを検査して dict にする。おかしければ BadEvent。"""
    if hop_dup_policy not in HOP_DUP_POLICIES:
        raise BadEvent(f"hop_dup_policy は {HOP_DUP_POLICIES} のどれかです: {hop_dup_policy!r}")
    if answer_filter not in ANSWER_FILTERS:
        raise BadEvent(f"answer_filter は {ANSWER_FILTERS} のどれかです: {answer_filter!r}")
    if mode not in MODES:
        raise BadEvent(f"mode は {MODES} のどれかです: {mode!r}")
    if entry_prefilter not in ENTRY_PREFILTERS:
        raise BadEvent(f"entry_prefilter は {ENTRY_PREFILTERS} のどれかです: {entry_prefilter!r}")
    if entry_fallback not in ENTRY_FALLBACKS:
        raise BadEvent(f"entry_fallback は {ENTRY_FALLBACKS} のどれかです: {entry_fallback!r}")
    if hop_chunk_source not in HOP_CHUNK_SOURCES:
        raise BadEvent(f"hop_chunk_source は {HOP_CHUNK_SOURCES} のどれかです: {hop_chunk_source!r}")
    if entry_chunk_rank not in ENTRY_CHUNK_RANKS:
        raise BadEvent(f"entry_chunk_rank は {ENTRY_CHUNK_RANKS} のどれかです: {entry_chunk_rank!r}")
    if answer_style not in ANSWER_STYLES:
        raise BadEvent(f"answer_style は {ANSWER_STYLES} のどれかです: {answer_style!r}")
    return {
        "mode": mode,
        "entry_threshold": _number("entry_th", entry_th, 0.0, 1.0),
        "sufficiency_th": _number("sufficiency_th", sufficiency_th, 0.0, 1.0),
        "max_hops": _integer("max_hops", max_hops, 0, MAX_HOPS_LIMIT),
        "max_neighbors": _integer("max_neighbors", max_neighbors, 1, 50),
        "min_neighbor_score": _number("min_neighbor_th", min_neighbor_score, 0.0, 1.0),
        "max_chunks": _integer("max_chunks", max_chunks, 1, 100),
        "max_chunks_per_node": _integer("max_chunks_per_node", max_chunks_per_node, 1, 100),
        "max_chunks_per_hop": _integer("max_chunks_per_hop", max_chunks_per_hop, 1, 100),
        "hop_chunk_source": hop_chunk_source,
        "entry_prefilter": entry_prefilter,
        "entry_fallback": entry_fallback,
        "semantic_entry_threshold": _number("semantic_entry_th", semantic_entry_th, 0.0, 1.0),
        "entry_chunk_rank": entry_chunk_rank,
        "answer_style": answer_style,
        "hop_dup_policy": hop_dup_policy,
        "answer_filter": answer_filter,
        "answer_filter_threshold": _number("answer_filter_threshold", answer_filter_threshold, 0.0, 1.0),
        "answer_filter_max": _integer("answer_filter_max", answer_filter_max, 1, 100),
        "answer_filter_min": _integer("answer_filter_min", answer_filter_min, 1, 100),
        "in_scope_th": IN_SCOPE_THRESHOLD,
    }


# ============================================================== 純粋関数

def select_entries(scores: Mapping[str, float | None], entry_th: float, max_neighbors: int) -> list[dict]:
    """entry_th 以上のノードをスコアの高い順に、上位 max_neighbors 件。`[{id, score}]`。
    同点は entity_id の順(結果を安定させるため)。None は除く。"""
    hits = [(eid, p) for eid, p in scores.items() if p is not None and p >= entry_th]
    hits.sort(key=lambda item: (-item[1], item[0]))
    return [{"id": eid, "score": p} for eid, p in hits[:max_neighbors]]


def entry_log(scores: Mapping[str, float | None], selected: Sequence[Mapping[str, Any]],
              min_score: float = ENTRY_LOG_MIN_SCORE, via: str = VIA_DICTIONARY) -> list[dict]:
    """トレース用の入口候補。スコア min_score 以上のノードを、選ばれなかったものも含めて
    `{id, score, selected, via}` でスコアの高い順に全部返す。
    `via` はどの判定のスコアか(`dictionary` = 辞書の候補への「書かれているか」、`semantic` = 聞き直し)。"""
    chosen = {e["id"] for e in selected}
    hits = [(eid, p) for eid, p in scores.items() if p is not None and p >= min_score]
    hits.sort(key=lambda item: (-item[1], item[0]))
    return [{"id": eid, "score": p, "selected": eid in chosen, "via": via} for eid, p in hits]


def merge_entry_logs(semantic: Sequence[Mapping[str, Any]],
                     dictionary: Sequence[Mapping[str, Any]]) -> list[dict]:
    """聞き直しの入口候補(先)と辞書の入口候補(後)を 1 本にする。同じ id は聞き直しの方を残す
    (図の 0 列目でノードが重ならないように)。"""
    seen = {e["id"] for e in semantic}
    return [dict(e) for e in semantic] + [dict(e) for e in dictionary if e["id"] not in seen]


def semantic_entities(master: Mapping[str, Any], types: Sequence[str]) -> list[dict]:
    """聞き直しで問うエンティティ。採用した種別の順 → マスターの並び順。"""
    return [e for t in types for e in master["entities"] if e["type"] == t]


def semantic_question_groups(state: Mapping[str, Any], entities: Sequence[Mapping[str, Any]]) -> list[list[dict]]:
    """聞き直しの問いをリクエストに分けたもの(エンティティの組)。100 問ずつ・見積もりトークンの上限でも分ける。"""
    return split_items(state, list(entities), semantic_entry_questions, lambda e: semantic_qid(e["id"]),
                       max_questions=SEMANTIC_MAX_QUESTIONS)


def appearance_chunk_ids(node: Mapping[str, Any]) -> list[str]:
    """`query_node()` の登場チャンクを、登場スコアの高い順(同点は chunk_id 順)で返す。
    max_chunks で切り捨てるときに、登場の確からしいチャンクを先に残すため。"""
    rows = sorted(node.get("appearances") or [],
                  key=lambda r: (-(r.get("score") or 0.0), r["chunk_id"]))
    return list(dict.fromkeys(r["chunk_id"] for r in rows))


def cap_node_chunks(chunk_ids: Sequence[str], per_node: int) -> tuple[list[str], list[str]]:
    """1 ノードの登場チャンク(`appearance_chunk_ids()` の順)を、上位 per_node 件と
    あふれた分に分ける。並べ替えはしない(登場スコア順のまま)。"""
    ids = list(chunk_ids)
    return ids[:per_node], ids[per_node:]


def entry_chunk_items(chunk_ids: Sequence[str], fetched: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    """入口の登場チャンクの判定で Jev に聞く `[{qid, id, text}]`。本文が取れたものだけ、chunk_ids の順
    (重複は 1 回)。qid は `ec_000` からの通し番号。"""
    items: list[dict] = []
    for cid in dict.fromkeys(chunk_ids):
        text = (fetched.get(cid) or {}).get("text")
        if isinstance(text, str) and text.strip():
            items.append({"qid": entry_chunk_qid(len(items)), "id": cid, "text": text})
    return items


def entry_chunk_groups(state: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> list[list[dict]]:
    """入口チャンクの問いをリクエストに分けたもの。100 問ずつ・見積もりトークンの上限でも分ける。"""
    return split_items(state, list(items), entry_chunk_questions, lambda item: item["qid"],
                       max_questions=ENTRY_CHUNK_MAX_QUESTIONS)


def rank_by_jev(chunk_ids: Sequence[str], scores: Mapping[str, float | None]) -> list[str]:
    """1 ノードの登場チャンク(`appearance_chunk_ids()` の順)を、Jev の「質問に答える情報を含むか」の
    スコアの高い順に並べ直す。同点・スコアなし(None・本文なし)は元の登場スコア順のまま後ろへ。"""
    def key(cid: str) -> float:
        score = scores.get(cid)
        return -(score if score is not None else -1.0)
    return sorted(chunk_ids, key=key)          # 安定ソートなので同点は登場スコア順


def rank_by_score(chunk_ids: Sequence[str], scores: Mapping[str, float | None]) -> list[str]:
    """`rank_by_jev()` の、範囲の決まっていないスコア版(`scorer` を渡したとき用)。
    スコアの高い順。スコアなし(None・本文なし)は一番後ろ。同点は元の登場スコア順のまま。"""
    def key(cid: str) -> float:
        score = scores.get(cid)
        return -score if score is not None else math.inf
    return sorted(chunk_ids, key=key)


def entry_capacity(per_hop: int, room: int) -> int:
    """ホップ 0 で実際に足せる件数の上限(1 ホップの枠と全体の残りの小さい方)。"""
    return max(min(per_hop, room), 0)


def ranking_unneeded(order_of: Mapping[str, Sequence[str]], per_node: int, capacity: int) -> bool:
    """入口チャンクを Jev で並べても取れるチャンクが変わらないか。"""
    if any(len(ids) > per_node for ids in order_of.values()):
        return False
    n_unique = len({cid for ids in order_of.values() for cid in ids})
    return n_unique <= capacity


def round_robin(lists: Sequence[Sequence[Any]]) -> list[Any]:
    """リストの先頭から 1 件ずつ交互に取り出して 1 本に並べる(`[[a1, a2], [b1]]` → `[a1, b1, a2]`)。
    入口が複数ノードのとき、スコアの高いノードの順に各ノードの上位を詰めるのに使う。"""
    out: list[Any] = []
    depth = max((len(items) for items in lists), default=0)
    for i in range(depth):
        for items in lists:
            if i < len(items):
                out.append(items[i])
    return out


def relation_order(scored: Sequence[Mapping[str, Any]], entities: Sequence[str],
                   min_score: float) -> list[dict]:
    """選んだ隣ノード `entities` への**たどった関係**(下限以上の候補)を、Jev の手がかりスコアの高い順
    (同点は候補の並び順)に並べる。各関係の `evidence` は登録時スコアの高い順(None は最後、同点は
    chunk_id 順)に並べ直して返す。ホップ 1 以降で根拠チャンクを足す順番に使う。"""
    chosen = set(entities)
    rels = [c for c in scored if c.get("entity") in chosen and not below_min(c.get("score"), min_score)]
    rels.sort(key=lambda c: -c["score"])            # 安定ソートなので同点は候補の並び順
    out = []
    for cand in rels:
        rows = sorted(cand.get("evidence") or [],
                      key=lambda r: (-(r.get("score") if r.get("score") is not None else -1.0), r["chunk_id"]))
        out.append({"entity": cand["entity"], "edge": cand["edge"], "score": cand["score"], "evidence": rows})
    return out


def split_hop_overflow(found: Sequence[str], per_hop: int, room: int) -> tuple[list[str], list[str], list[str]]:
    """そのホップで本文が取れた新しいチャンク(足す順)を、足す分・全体上限で捨てる分・ホップの上限で
    あふれた分に分ける。`(kept, dropped, hop_capped)`。"""
    ids = list(found)
    take = min(per_hop, max(room, 0))
    kept = ids[:take]
    dropped = ids[take:max(per_hop, take)]
    capped = ids[max(per_hop, take):]
    return kept, dropped, capped


def pick_hop_top(order_ids: Sequence[str], collected_ids: set[str] | Sequence[str], found: Sequence[str],
                 per_hop: int, room: int) -> dict:
    """`hop_dup_policy="count"` のホップの取り方。`order_ids` はそのホップの候補(足す順。重複なし。
    既に集めたチャンクも含む)、`found` はそのうち**新しくて本文が取れた**もの。"""
    have = set(collected_ids)
    found_set = set(found)
    ranked = [cid for cid in order_ids if cid in have or cid in found_set]
    top = ranked[:max(per_hop, 0)]
    top_new = [cid for cid in top if cid in found_set]
    kept, dropped, _ = split_hop_overflow(top_new, per_hop, room)
    top_new_set = set(top_new)
    return {"top": top, "kept": kept, "dropped": dropped,
            "hop_capped": [cid for cid in found if cid not in top_new_set],
            "dups": [cid for cid in top if cid in have]}


def answer_filter_items(collected: Sequence[Mapping[str, Any]]) -> list[dict]:
    """回答前の絞り込みで Jev に聞く `[{qid, id, text}]`。累積の順、qid は `af_000` からの通し番号。"""
    return [{"qid": answer_filter_qid(i), "id": c["id"], "text": c["text"]} for i, c in enumerate(collected)]


def answer_filter_groups(state: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> list[list[dict]]:
    """回答前の絞り込みの問いをリクエストに分けたもの。100 問ずつ・見積もりトークンの上限でも分ける。"""
    return split_items(state, list(items), answer_filter_questions, lambda item: item["qid"],
                       max_questions=ANSWER_FILTER_MAX_QUESTIONS)


def select_answer_chunks(rows: Sequence[Mapping[str, Any]], threshold: float, max_n: int,
                         min_n: int) -> tuple[list[dict], list[dict], bool]:
    """回答前の絞り込み。`rows` は累積の順の `[{id, score, hits}]`(score は None もあり)。"""
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda ir: (-(ir[1]["score"] if ir[1].get("score") is not None else -1.0),
                                 -int(ir[1].get("hits") or 0), ir[0]))
    ranked = [dict(r) for _i, r in indexed]
    passing = [r for r in ranked if r.get("score") is not None and r["score"] >= threshold][:max_n]
    min_applied = not passing
    kept = ranked[:min(min_n, max_n)] if min_applied else passing
    kept_ids = {r["id"] for r in kept}
    return kept, [r for r in ranked if r["id"] not in kept_ids], min_applied


def edge_evidence_chunks(evidence: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, list[dict]]]:
    """たどった関係の根拠(`neighbor_candidates()` の `evidence` の行)から、チャンク ID を
    関係の登録時スコアの高い順(None は最後、同点は chunk_id 順)に重複を除いて並べる。"""
    rows = sorted(evidence, key=lambda r: (-(r.get("score") if r.get("score") is not None else -1.0),
                                           r["chunk_id"]))
    ids: list[str] = []
    of: dict[str, list[dict]] = {}
    for row in rows:
        cid = row["chunk_id"]
        if cid not in of:
            ids.append(cid)
            of[cid] = []
        rel = {"from": row["from"], "edge": row["edge"], "to": row["to"], "direction": row["direction"]}
        if rel not in of[cid]:
            of[cid].append(rel)
    return ids, of


def neighbor_candidates(
    frontier: Sequence[str],
    nodes: Mapping[str, Mapping[str, Any]],
    master: Mapping[str, Any],
    visited: set[str],
) -> list[dict]:
    """frontier の OUT/IN/SYM 行から隣ノードへの候補を作る。"""
    ents = entity_by_id(master)
    templates = {e["name"]: e["template"] for e in master["edge_types"]}
    seen: dict[tuple[str, str], dict] = {}
    for me in frontier:
        node = nodes.get(me) or {}
        for direction, rows in (("out", node.get("out") or []), ("in", node.get("in") or []),
                                ("sym", node.get("sym") or [])):
            for row in rows:
                other = {"out": row.get("target_id"), "in": row.get("source_id"),
                         "sym": row.get("other_id")}[direction]
                if other in visited or other == me or other not in ents or me not in ents:
                    continue
                template = templates.get(row["edge"])
                if template is None:
                    continue
                key = (row["edge"], other)
                evidence = {"from": me, "edge": row["edge"], "to": other, "direction": direction,
                            "chunk_id": row.get("chunk_id"), "score": row.get("score")}
                if key in seen:
                    cand = seen[key]
                    cand["n_rows"] += 1
                    if evidence["chunk_id"] is not None:
                        cand["evidence"].append(evidence)
                    score = row.get("score")
                    if score is not None and (cand["edge_score"] is None or score > cand["edge_score"]):
                        cand["edge_score"] = score
                    if me != cand["from"] and all(a["from"] != me for a in cand["also_from"]):
                        cand["also_from"].append({"from": me, "direction": direction})
                    continue
                # out・sym は (自分, 隣)、in は (隣, 自分) をテンプレートの (source, target) に入れる
                source, target = (ents[other], ents[me]) if direction == "in" else (ents[me], ents[other])
                seen[key] = {
                    "from": me,
                    "entity": other,
                    "edge": row["edge"],
                    "direction": direction,
                    "sentence": render(template, source, target),
                    "edge_score": row.get("score"),
                    "n_rows": 1,
                    "also_from": [],
                    "evidence": [evidence] if evidence["chunk_id"] is not None else [],
                }
    candidates = list(seen.values())
    for i, cand in enumerate(candidates):
        cand["qid"] = neighbor_qid(i)
    return candidates


def below_min(score: float | None, min_score: float) -> bool:
    """下限未満(None も含む)なら True。下限ちょうどは進める。"""
    return score is None or score < min_score


def select_neighbors(scored: Sequence[Mapping[str, Any]], max_neighbors: int,
                     min_score: float = 0.0) -> list[dict]:
    """スコアが min_score 以上の候補から、スコアの高い順に**異なる隣ノード**を max_neighbors 件選ぶ。
    スコアが None の候補は選ばない。同点は候補の並び順。戻り値は選んだ候補(隣ノードごとに最良の 1 件)。"""
    order = sorted(
        (c for c in scored if not below_min(c.get("score"), min_score)),
        key=lambda c: -c["score"],
    )
    picked: dict[str, dict] = {}
    for cand in order:
        if cand["entity"] in picked:
            continue
        picked[cand["entity"]] = dict(cand)
        if len(picked) >= max_neighbors:
            break
    return list(picked.values())


def format_chunks(collected: Sequence[Mapping[str, Any]]) -> str:
    """回答生成に渡す本文。`[op-3] 本文…` を 1 チャンク 1 段落で並べる。"""
    return "\n\n".join(f"[{c['id']}] {c['text']}" for c in collected)


def answer_user_prompt(question: str, collected: Sequence[Mapping[str, Any]]) -> str:
    return f"質問: {question}\n\n本文:\n{format_chunks(collected)}"


def first_json_object(text: str) -> dict | None:
    """コードフェンスを剥がしたうえで、`{` が出てくる位置ごとに `raw_decode` を試し、
    最初に dict として読めたものを返す(前後に地の文があっても、`}` が後ろにあっても読める)。"""
    body = strip_code_fence(text or "")
    decoder = json.JSONDecoder()
    pos = body.find("{")
    while pos >= 0:
        try:
            obj, _end = decoder.raw_decode(body, pos)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return obj
        pos = body.find("{", pos + 1)
    return None


def normalize_chunk_id(value: str) -> str:
    """引用 ID の表記揺れをそろえる: 前後の空白と `[` `]` を除き、小文字にする(`[DC-3] ` → `dc-3`)。"""
    text = value.strip()
    while text.startswith("["):
        text = text[1:].strip()
    while text.endswith("]"):
        text = text[:-1].strip()
    return text.lower()


def parse_answer(text: str) -> dict | None:
    """Haiku の出力を読む。形が違えば None。"""
    data = first_json_object(text)
    if data is None:
        return None
    answer, citations, answerable = data.get("answer"), data.get("citations"), data.get("answerable")
    if not isinstance(answer, str) or not isinstance(answerable, bool):
        return None
    fixes = 0
    raw = citations
    if citations is None:
        citations, fixes = [], fixes + 1
    if not isinstance(citations, list):
        return None
    cleaned = []
    for cit in citations:
        if not isinstance(cit, dict):
            return None
        claim = cit.get("claim")
        if claim is None:
            claim, fixes = "", fixes + 1
        if not isinstance(claim, str):
            return None
        ids = cit.get("chunk_ids")
        if ids is None:
            ids, fixes = [], fixes + 1
        elif isinstance(ids, str):
            ids, fixes = [ids], fixes + 1
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            return None
        out_ids = []
        for cid in ids:
            norm = normalize_chunk_id(cid)
            if not norm:
                fixes += 1
                continue
            if norm != cid:
                fixes += 1
            out_ids.append(norm)
        cleaned.append({"claim": claim, "chunk_ids": out_ids})
    return {"answer": answer, "citations": cleaned, "answerable": answerable,
            "citations_raw": raw, "citation_format_fixes": fixes}


def check_citations(citations: Sequence[Mapping[str, Any]], collected_ids: Sequence[str],
                    answerable: bool | None = None) -> dict:
    """引用の機械検査。`{citations_valid, invalid_citation_ids, uncited_claims, no_citations}`。"""
    allowed = {normalize_chunk_id(c) for c in collected_ids}
    per_claim = [[normalize_chunk_id(i) for i in c.get("chunk_ids") or [] if normalize_chunk_id(i)]
                 for c in citations]
    cited = [cid for ids in per_claim for cid in ids]
    invalid = sorted({cid for cid in cited if cid not in allowed})
    no_citations = answerable is True and not any(cid in allowed for cid in cited)
    return {
        "citations_valid": not invalid and not no_citations,
        "invalid_citation_ids": invalid,
        "uncited_claims": sum(1 for ids in per_claim if not ids),
        "no_citations": no_citations,
    }


def answerable_mismatch(answer: Any, answerable: Any) -> bool:
    """answerable と answer の食い違い。false なのに「回答できません」以外/true なのに「回答できません」。"""
    if not isinstance(answer, str) or not isinstance(answerable, bool):
        return False
    text = answer.strip().rstrip("。.").strip()
    if answerable is False:
        return text != CANNOT_ANSWER
    return CANNOT_ANSWER in answer


def entry_candidates(question: str, master: Mapping[str, Any], entry_prefilter: str) -> list[dict] | None:
    """前段判定で Jev に聞くエンティティ。"dictionary" なら辞書で拾った `[{id, matched_terms}]`、
    "none" なら None。"""
    if entry_prefilter == ENTRY_PREFILTER_NONE:
        return None
    return find_candidates(question, master)


def entry_question_groups(state: Mapping[str, Any], master: Mapping[str, Any],
                          candidates: Sequence[Mapping[str, Any]] | None) -> list[dict]:
    """前段判定のリクエストに分けた問い。candidates が None なら全エンティティ、空なら in_scope の 1 問だけ。"""
    if candidates is None:
        entities = list(master["entities"])
    else:
        ents = entity_by_id(master)
        entities = [ents[c["id"]] for c in candidates]
    return split_questions(state, entry_questions(entities))


# ============================================================== 見積もり

def estimate_usd(question: str, master: Mapping[str, Any], chunk_texts: Sequence[str], *,
                 mode: str, max_hops: int, max_chunks: int = DEFAULT_MAX_CHUNKS,
                 neighbor_candidates_per_hop: int = NEIGHBOR_CANDIDATES_EST,
                 entry_prefilter: str = DEFAULT_ENTRY_PREFILTER,
                 entry_fallback: str = DEFAULT_ENTRY_FALLBACK,
                 entry_chunk_rank: str = DEFAULT_ENTRY_CHUNK_RANK,
                 entry_chunks_est: int = ENTRY_CHUNKS_EST,
                 answer_style: str = DEFAULT_ANSWER_STYLE,
                 answer_filter: str = DEFAULT_ANSWER_FILTER) -> dict:
    """Invoke **前**に出す上振れ寄りの見積もり(1 質問 1 モード)。"""
    state = {"question": question}
    entry_groups = entry_question_groups(
        state, master, entry_candidates(question, master, entry_prefilter))
    entry_chars = sum(request_chars(state, g) for g in entry_groups)
    n_fallback = 0
    if entry_fallback == ENTRY_FALLBACK_SEMANTIC:
        counts: dict[str, int] = {}
        for e in master["entities"]:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        worst = sorted(counts, key=lambda t: -counts[t])[:SEMANTIC_MAX_TYPES]
        groups = semantic_question_groups(state, semantic_entities(master, worst))
        entry_chars += request_chars(state, entry_type_questions(master["node_types"]))
        entry_chars += sum(request_chars(state, semantic_entry_questions(g)) for g in groups)
        n_fallback = 1 + len(groups)
    n_entry_chunk = 0
    if entry_chunk_rank == ENTRY_CHUNK_RANK_JEV and entry_chunks_est > 0:
        texts = sorted(chunk_texts, key=len, reverse=True)[:entry_chunks_est]   # 全チャンク数より多くはならない
        items = [{"qid": entry_chunk_qid(i), "text": t} for i, t in enumerate(texts)]
        ec_groups = entry_chunk_groups(state, items)
        entry_chars += sum(request_chars(state, entry_chunk_questions(g)) for g in ec_groups)
        n_entry_chunk = len(ec_groups)
    longest = sorted(chunk_texts, key=len, reverse=True)[:max_chunks]
    chunks_chars = sum(len(t) + 20 for t in longest)
    suff_chars = len(json.dumps({"state": state, "questions": sufficiency_questions()},
                                ensure_ascii=False)) + chunks_chars
    sample = {"qid": neighbor_qid(0), "sentence": "候補文の例として二十文字ほどの文をここに置く"}
    per_question = len(json.dumps(neighbor_questions([sample]), ensure_ascii=False))
    per_batch_state = len(json.dumps({"state": state}, ensure_ascii=False))
    n_batches = math.ceil(neighbor_candidates_per_hop / NEIGHBOR_BATCH_SIZE)
    nb_chars_per_hop = per_question * neighbor_candidates_per_hop + per_batch_state * n_batches
    n_suff = (max_hops + 1) if mode == "adaptive" else 0
    af_chars, n_af = 0, 0
    if answer_filter == ANSWER_FILTER_JEV and longest:
        af_items = [{"qid": answer_filter_qid(i), "text": t} for i, t in enumerate(longest)]
        af_groups = answer_filter_groups(state, af_items)
        af_chars = sum(request_chars(state, answer_filter_questions(g)) for g in af_groups)
        n_af = len(af_groups)
    jev_chars = entry_chars + n_suff * suff_chars + max_hops * nb_chars_per_hop + af_chars
    jev_tokens = int(jev_chars * TOKENS_PER_CHAR)
    system = ANSWER_SYSTEMS.get(answer_style, ANSWER_SYSTEM_FULL)
    gen_in = int((len(system) + len(question) + chunks_chars) * TOKENS_PER_CHAR) * 2
    gen_out = GEN_MAX_TOKENS * 2
    jev_price = PRICES[JEV_MODEL]
    gen_price = PRICES[DEFAULT_GEN_MODEL]
    usd = (jev_tokens * jev_price["input"] / 1e6
           + gen_in * gen_price["input"] / 1e6 + gen_out * gen_price["output"] / 1e6)
    jev_requests = min(len(entry_groups) + n_fallback + n_entry_chunk + n_suff + max_hops * n_batches + n_af,
                       MAX_JEV_CALLS)
    return {"jev_requests": jev_requests, "jev_input_tokens": jev_tokens,
            "gen_requests": 2, "gen_input_tokens": gen_in, "gen_output_tokens": gen_out,
            "est_usd": usd}


# ============================================================== 本体

class _Timer:
    """種類ごとの壁時計の合計(ミリ秒)。"""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.ms = {"jev": 0.0, "ddb": 0.0, "bedrock": 0.0}

    def run(self, kind: str, fn: Callable[[], Any]) -> Any:
        started = self.clock()
        try:
            return fn()
        finally:
            self.ms[kind] += (self.clock() - started) * 1000.0


_RESERVE = {"jev": JEV_RESERVE_SEC, "ddb": DDB_RESERVE_SEC, "gen": GEN_RESERVE_SEC}


def run_query(
    question: str,
    *,
    jev: Any,
    store: Any,
    llm: Any,
    master: Mapping[str, Any],
    mode: str = "adaptive",
    entry_th: float = DEFAULT_ENTRY_THRESHOLD,
    sufficiency_th: float = DEFAULT_SUFFICIENCY_THRESHOLD,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    min_neighbor_score: float = DEFAULT_MIN_NEIGHBOR_SCORE,
    max_chunks_per_node: int = DEFAULT_MAX_CHUNKS_PER_NODE,
    hop_chunk_source: str = DEFAULT_HOP_CHUNK_SOURCE,
    max_chunks_per_hop: int = DEFAULT_MAX_CHUNKS_PER_HOP,
    entry_prefilter: str = DEFAULT_ENTRY_PREFILTER,
    entry_fallback: str = DEFAULT_ENTRY_FALLBACK,
    semantic_entry_th: float = DEFAULT_SEMANTIC_ENTRY_THRESHOLD,
    entry_chunk_rank: str = DEFAULT_ENTRY_CHUNK_RANK,
    answer_style: str = DEFAULT_ANSWER_STYLE,
    hop_dup_policy: str = DEFAULT_HOP_DUP_POLICY,
    answer_filter: str = DEFAULT_ANSWER_FILTER,
    answer_filter_threshold: float = DEFAULT_ANSWER_FILTER_THRESHOLD,
    answer_filter_max: int = DEFAULT_ANSWER_FILTER_MAX,
    answer_filter_min: int = DEFAULT_ANSWER_FILTER_MIN,
    gen_model: str = DEFAULT_GEN_MODEL,
    on_record: Callable[[dict], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    deadline: float | None = None,
    reserves: Mapping[str, float] | None = None,
    preset_entries: Sequence[Mapping[str, Any]] | None = None,
    preset_entry_log: Sequence[Mapping[str, Any]] | None = None,
    method_label: str | None = None,
    scorer: Any | None = None,
    hop_max_chunks_per_node: int | None = None,
    answer_chunk_selector: Callable[[list[dict]], list[dict]] | None = None,
) -> dict:
    """1 質問ぶんの検索と回答。戻り値はトレース(`records` に Jev と Bedrock の呼び出しレコード)。

    `preset_entries` 以降の引数は判定を差し替えるための口。既定(None)なら本手法の動作になる。"""
    if not isinstance(question, str) or not question.strip():
        raise BadEvent("question は空でない文字列です")
    question = question.strip()
    params = make_params(mode=mode, entry_th=entry_th, sufficiency_th=sufficiency_th,
                         max_hops=max_hops, max_neighbors=max_neighbors, max_chunks=max_chunks,
                         min_neighbor_score=min_neighbor_score, max_chunks_per_node=max_chunks_per_node,
                         hop_chunk_source=hop_chunk_source, max_chunks_per_hop=max_chunks_per_hop,
                         entry_prefilter=entry_prefilter, entry_fallback=entry_fallback,
                         semantic_entry_th=semantic_entry_th, entry_chunk_rank=entry_chunk_rank,
                         answer_style=answer_style, hop_dup_policy=hop_dup_policy,
                         answer_filter=answer_filter, answer_filter_threshold=answer_filter_threshold,
                         answer_filter_max=answer_filter_max, answer_filter_min=answer_filter_min)
    if scorer is not None and params["mode"] != "fixed":
        raise BadEvent("scorer を渡すときは mode=fixed です(十分性の判定は Jev のため)")
    if (scorer is not None or answer_chunk_selector is not None) and params["answer_filter"] != ANSWER_FILTER_NONE:
        raise BadEvent("scorer / answer_chunk_selector を渡すときは answer_filter=none です")
    if hop_max_chunks_per_node is not None:
        params["hop_max_chunks_per_node"] = _integer("hop_max_chunks_per_node", hop_max_chunks_per_node,
                                                     1, HOP_MAX_CHUNKS_PER_NODE_LIMIT)
    started = clock()
    timer = _Timer(clock)
    records: list[dict] = []
    ents = entity_by_id(master)
    counter = {"jev": 0}
    reserve = {**_RESERVE, **(reserves or {})}
    if set(reserve) != set(_RESERVE):
        raise BadEvent(f"reserves のキーは {sorted(_RESERVE)} です: {sorted(reserves or {})}")

    def keep(record: dict) -> dict:
        records.append(record)
        if on_record is not None:
            on_record(record)
        return record

    def guard(kind: str, where: str, n_jev: int = 1) -> None:
        """呼ぶ前の確認。Jev の回数上限と締め切り。足りなければ _Halt。"""
        if kind == "jev" and counter["jev"] + n_jev > MAX_JEV_CALLS:
            raise _Halt("jev_cap", where)
        if deadline is None:
            return
        need = reserve[kind] + (reserve["gen"] if kind != "gen" else 0.0)
        if deadline - clock() < need:
            raise _Halt("deadline", where)

    def judge(state: dict, questions: dict, where: str) -> dict:
        guard("jev", where)
        counter["jev"] += 1
        return keep(timer.run("jev", lambda: jev.judge(state, questions, use_cache=False)))

    trace: dict[str, Any] = {
        "question": question, "mode": params["mode"], "params": params,
        "in_scope": None, "in_scope_overridden": False,
        "entry_prefilter": params["entry_prefilter"], "entry_candidates": None,
        "entry": [], "entry_fallback_used": False, "entry_fallback": None, "hops": [], "chunks": [],
        "stop_reason": None, "halted_at": None, "truncated": False,
        "answer": None, "citations": [], "citations_raw": None, "citation_format_fixes": 0,
        "answerable": None, "answerable_mismatch": False,
        "citations_valid": None, "invalid_citation_ids": [], "uncited_claims": 0, "no_citations": False,
        "parse_error": False, "answer_attempts": 0, "answer_stop_reasons": [], "answer_truncated": False,
        "bedrock_error": None, "answer_filter": None, "answer_chunk_ids": [],
    }

    def cannot_answer(reason: str) -> dict:
        trace.update({"stop_reason": reason, "answer": CANNOT_ANSWER, "answerable": False,
                      "citations_valid": True})
        return _finish(trace, records, timer, started, clock)

    def no_answer_halt(halt: _Halt) -> dict:
        trace.update({"stop_reason": halt.reason, "halted_at": halt.where, "answer": "", "answerable": None})
        return _finish(trace, records, timer, started, clock)

    # ---- 1. 前段判定
    q_state = {"question": question}
    if preset_entries is not None:
        entries = [{"id": e["id"], "score": e.get("score")} for e in preset_entries if e.get("id") in ents]
        trace["entry_prefilter"] = None
        trace["entry"] = ([dict(e) for e in preset_entry_log] if preset_entry_log is not None else
                          [{**e, "selected": True, "via": "preset"} for e in entries])
        if not entries:
            return cannot_answer("no_entry")
    else:
        candidates = entry_candidates(question, master, params["entry_prefilter"])
        trace["entry_candidates"] = candidates
        entry_groups = entry_question_groups(q_state, master, candidates)
        entry_records: list[dict] = []
        try:
            # 途中の組だけ判定して半端な入口を選ばないよう、全組ぶんの回数を先に確かめる
            guard("jev", "entry", n_jev=len(entry_groups))
            for group in entry_groups:
                entry_records.append(judge(q_state, group, "entry"))
        except _Halt as halt:
            return no_answer_halt(halt)
        in_scope = next((noul_value(r, IN_SCOPE_QID) for r, g in zip(entry_records, entry_groups)
                         if IN_SCOPE_QID in g), None)
        trace["in_scope"] = in_scope
        node_scores: dict[str, float | None] = {eid: None for eid in ents}
        for record, group in zip(entry_records, entry_groups):
            for eid in ents:
                if node_qid(eid) in group:
                    node_scores[eid] = noul_value(record, node_qid(eid))
        entries = select_entries(node_scores, params["entry_threshold"], params["max_neighbors"])
        trace["entry"] = entry_log(node_scores, entries, via=VIA_DICTIONARY)
        if not entries and params["entry_fallback"] == ENTRY_FALLBACK_SEMANTIC:
            # 辞書で入口が見つからなかったときだけ、意味で聞き直す(種別 1 回 → その種別のエンティティだけ)
            trace["entry_fallback_used"] = True
            fb: dict[str, Any] = {"type_scores": [], "selected_types": [], "n_questions": 0,
                                  "candidates": [], "threshold": params["semantic_entry_threshold"]}
            trace["entry_fallback"] = fb
            try:
                type_record = judge(q_state, entry_type_questions(master["node_types"]), "entry_fallback:type")
                fb["type_scores"] = entry_type_scores(type_record, master["node_types"])
                fb["selected_types"] = select_entry_types(fb["type_scores"], SEMANTIC_TYPE_COVER,
                                                          SEMANTIC_MAX_TYPES, SEMANTIC_TYPE_MIN_SCORE)
                targets = semantic_entities(master, fb["selected_types"])
                fb["n_questions"] = len(targets)
                groups = semantic_question_groups(q_state, targets)
                # 途中の組だけ判定して半端な入口を選ばないよう、全組ぶんの回数を先に確かめる
                guard("jev", "entry_fallback:semantic", n_jev=len(groups))
                sem_scores: dict[str, float | None] = {}
                for group in groups:
                    record = judge(q_state, semantic_entry_questions(group), "entry_fallback:semantic")
                    for entity in group:
                        sem_scores[entity["id"]] = noul_value(record, semantic_qid(entity["id"]))
            except _Halt as halt:
                return no_answer_halt(halt)
            entries = select_entries(sem_scores, params["semantic_entry_threshold"], params["max_neighbors"])
            sem_log = entry_log(sem_scores, entries, via=VIA_SEMANTIC)
            fb["candidates"] = [{"id": e["id"], "score": e["score"]} for e in sem_log]
            trace["entry"] = merge_entry_logs(sem_log, trace["entry"])
        if not entries:
            if in_scope is None:
                return cannot_answer("entry_judge_missing")
            if in_scope < params["in_scope_th"]:
                return cannot_answer("out_of_scope")
            return cannot_answer("no_entry")
        # 入口があれば in_scope が低くても(取れなくても)続ける。続けたことをトレースに残す
        trace["in_scope_overridden"] = in_scope is None or in_scope < params["in_scope_th"]

    # ---- 2〜5. ホップのループ
    collected: list[dict] = []          # 累積。[{id, text}]。置き換えない
    chunk_log: dict[str, dict] = {}     # id -> {id, hop, via_entity, via_edge, evidence_of, also_via, hits}
    nodes: dict[str, dict] = {}         # 訪問したノードの query_node() 結果(OUT/IN の材料)
    visited: set[str] = set()
    state_box: dict[str, Any] = {"last_sufficiency": None}

    def rank_entry_chunks(order_of: Mapping[str, list[str]],
                          hop_log: dict) -> tuple[dict[str, list[str]], dict[str, dict] | None]:
        """入口ノードごとの登場チャンク(登場スコア順)を、Jev の「質問に答える情報を含むか」のスコア順に
        並べ直す。戻り値は `(並べ直した order_of, 読んだ本文 {chunk_id: {text, doc_id}} | None)`。"""
        capacity = entry_capacity(params["max_chunks_per_hop"], params["max_chunks"] - len(collected))
        ranking: dict[str, Any] = {"n_chunks": 0, "n_questions": 0, "requests": 0, "input_tokens": 0,
                                   "usd": 0.0, "scores": [], "fallback": None, "halted_at": None,
                                   "skipped": None, "capacity": capacity}
        hop_log["entry_chunk_ranking"] = ranking
        all_ids = list(dict.fromkeys(cid for ids in order_of.values() for cid in ids))
        ranking["n_chunks"] = len(all_ids)
        if not all_ids:
            ranking["skipped"] = ENTRY_RANK_SKIP_NO_CHUNKS
            return dict(order_of), None
        if ranking_unneeded(order_of, params["max_chunks_per_node"], capacity):
            ranking["skipped"] = ENTRY_RANK_SKIP_WITHIN_CAPACITY
            return dict(order_of), None
        guard("ddb", "hop0:entry_chunks:batch_get")
        fetched = timer.run("ddb", lambda: store.batch_get_chunks(all_ids))
        items = entry_chunk_items(all_ids, fetched)
        ranking["n_questions"] = len(items)
        if scorer is not None:
            got = timer.run("jev", lambda: scorer.score_chunks([item["id"] for item in items]))
            local = {item["id"]: got.get(item["id"]) for item in items}
            ranking["scores"] = sorted(({"id": cid, "score": s} for cid, s in local.items()),
                                       key=lambda r: -(r["score"] if r["score"] is not None else -math.inf))
            return {eid: rank_by_score(ids, local) for eid, ids in order_of.items()}, fetched
        groups = entry_chunk_groups(q_state, items)
        scores: dict[str, float | None] = {}
        try:
            # 途中の組だけ判定して半端に並べないよう、全組ぶんの回数を先に確かめる(問いが無ければ呼ばない)
            if groups:
                guard("jev", "hop0:entry_chunks", n_jev=len(groups))
            for group in groups:
                record = judge(q_state, entry_chunk_questions(group), "hop0:entry_chunks")
                ranking["requests"] += 1
                ranking["input_tokens"] += int((record.get("usage") or {}).get("input_tokens") or 0)
                for item in group:
                    scores[item["id"]] = noul_value(record, item["qid"])
        except _Halt as halt:
            ranking["fallback"], ranking["halted_at"] = halt.reason, halt.where
        ranking["usd"] = ranking["input_tokens"] * PRICES[JEV_MODEL]["input"] / 1e6
        ranking["scores"] = sorted(({"id": cid, "score": s} for cid, s in scores.items()),
                                   key=lambda r: -(r["score"] if r["score"] is not None else -1.0))
        if ranking["fallback"] is not None:
            return dict(order_of), fetched                 # 半端なスコアでは並べない(登場スコア順のまま)
        return {eid: rank_by_jev(ids, scores) for eid, ids in order_of.items()}, fetched

    def visit(hop: int, arrivals: Sequence[tuple[str, str | None]], hop_log: dict,
              relations: Sequence[Mapping[str, Any]] | None = None) -> None:
        """arrivals = [(entity_id, via_edge)](スコアの高い順)のノードに進み、チャンクを累積に足す。"""
        for eid, _via_edge in arrivals:
            guard("ddb", f"hop{hop}:query_node")
            visited.add(eid)
            nodes[eid] = timer.run("ddb", lambda e=eid: store.query_node(e))
        per_node = params["max_chunks_per_node"]
        if hop > 0 and "hop_max_chunks_per_node" in params:
            per_node = params["hop_max_chunks_per_node"]
        queue: list[tuple[str, str | None, str]] = []      # (entity, via_edge, chunk_id) を足す順に
        evidence_of: dict[str, dict[str, list[dict]]] = {}
        prefetched: dict[str, dict] | None = None          # 入口チャンクの判定で先に読んだ本文(全件)
        if relations is None:
            order_of = {eid: appearance_chunk_ids(nodes[eid]) for eid, _ in arrivals}
            if hop == 0 and params["entry_chunk_rank"] == ENTRY_CHUNK_RANK_JEV:
                order_of, prefetched = rank_entry_chunks(order_of, hop_log)
            lists = []
            for eid, via_edge in arrivals:
                all_ids = order_of[eid]
                top, capped = cap_node_chunks(all_ids, per_node)
                hop_log["node_chunk_counts"][eid] = len(all_ids)
                if capped:
                    hop_log["capped_chunk_ids"][eid] = capped
                lists.append([(eid, via_edge, cid) for cid in top])
            queue = round_robin(lists)
        else:
            order: dict[str, list[tuple[str, str]]] = {eid: [] for eid, _ in arrivals}
            for rel in relations:
                seen_ids = {cid for cid, _ in order[rel["entity"]]}
                for row in rel["evidence"]:
                    if row["chunk_id"] not in seen_ids:
                        seen_ids.add(row["chunk_id"])
                        order[rel["entity"]].append((row["chunk_id"], rel["edge"]))
            kept_of: dict[str, set[str]] = {}
            for eid, items in order.items():
                _ids, evidence_of[eid] = edge_evidence_chunks(
                    [row for rel in relations if rel["entity"] == eid for row in rel["evidence"]])
                top, capped = cap_node_chunks([cid for cid, _ in items], per_node)
                hop_log["node_chunk_counts"][eid] = len(items)
                if capped:
                    hop_log["capped_chunk_ids"][eid] = capped
                kept_of[eid] = set(top)
            done: set[tuple[str, str]] = set()
            for rel in relations:
                eid = rel["entity"]
                for row in rel["evidence"]:
                    cid = row["chunk_id"]
                    if cid in kept_of[eid] and (eid, cid) not in done:
                        done.add((eid, cid))
                        queue.append((eid, rel["edge"], cid))
        new_ids: list[str] = []
        order_ids: list[str] = []                           # 候補の順(重複なし。既に集めたものも含む)
        dup_vias: dict[str, list[dict]] = {}                # 既に集めたチャンク -> このホップの経路
        pending: dict[str, dict] = {}
        for eid, via_edge, cid in queue:
            via = {"hop": hop, "entity": eid, "edge": via_edge}
            if cid in chunk_log:            # 前のホップで取得済み → 別経路(数えるかは下で決める)
                if cid not in dup_vias:
                    dup_vias[cid] = []
                    order_ids.append(cid)
                dup_vias[cid].append(via)
                continue
            if cid in pending:              # 同じホップで別のノードからも出会った
                pending[cid]["also_via"].append(via)
                continue
            new_ids.append(cid)
            order_ids.append(cid)
            pending[cid] = {"id": cid, "hop": hop, "via_entity": eid, "via_edge": via_edge,
                            "evidence_of": list(evidence_of.get(eid, {}).get(cid, [])), "also_via": [],
                            "hits": 1}
        fetched: dict[str, dict] = {}
        if prefetched is not None:
            fetched = prefetched                        # 入口の登場チャンクは全件読み済み(new_ids はその一部)
        elif new_ids:
            guard("ddb", f"hop{hop}:batch_get")
            fetched = timer.run("ddb", lambda: store.batch_get_chunks(new_ids))
        found = [cid for cid in new_ids if isinstance((fetched.get(cid) or {}).get("text"), str)]
        room = max(params["max_chunks"] - len(collected), 0)
        if params["hop_dup_policy"] == HOP_DUP_COUNT:
            picked = pick_hop_top(order_ids, set(chunk_log), found, params["max_chunks_per_hop"], room)
            kept, dropped, hop_capped = picked["kept"], picked["dropped"], picked["hop_capped"]
            counted = picked["dups"]
        else:
            kept, dropped, hop_capped = split_hop_overflow(found, params["max_chunks_per_hop"], room)
            counted = list(dup_vias)
        for cid in counted:
            chunk_log[cid]["also_via"].extend(dup_vias[cid])
            chunk_log[cid]["hits"] += 1
        hop_log["dup_chunk_ids"] = counted
        for cid in kept:
            collected.append({"id": cid, "text": fetched[cid]["text"]})
            chunk_log[cid] = pending[cid]
        if dropped:
            trace["truncated"] = True
        hop_log["new_chunk_ids"] = kept
        hop_log["dropped_chunk_ids"] = dropped
        hop_log["hop_capped_chunk_ids"] = hop_capped
        for cid in hop_capped:
            hop_log["hop_capped_by_entity"].setdefault(pending[cid]["via_entity"], []).append(cid)
        hop_log["missing_chunk_ids"] = [cid for cid in new_ids if cid not in found]

    def sufficiency(hop: int, hop_log: dict) -> bool:
        """adaptive なら十分性を判定して hop_log に残す。しきい値以上なら True。"""
        if params["mode"] != "adaptive":
            return False
        if hop_log["new_chunk_ids"] or state_box["last_sufficiency"] is None:
            state = {"question": question, "chunks": [dict(c) for c in collected]}
            record = judge(state, sufficiency_questions(), f"hop{hop}:sufficiency")
            state_box["last_sufficiency"] = noul_value(record, SUFFICIENCY_QID)
        else:
            hop_log["sufficiency_reused"] = True      # 累積が前と同じなので呼ばない
        hop_log["sufficiency"] = state_box["last_sufficiency"]
        score = state_box["last_sufficiency"]
        return score is not None and score >= params["sufficiency_th"]

    def after_hop(hop: int, hop_log: dict) -> str | None:
        if sufficiency(hop, hop_log):
            return "sufficient"
        if len(collected) >= params["max_chunks"] or trace["truncated"]:
            return "max_chunks"
        if hop >= params["max_hops"]:
            return "max_hops"
        return None

    def filter_for_answer(chunks: list[dict]) -> list[dict]:
        """回答の前に累積チャンクを Jev で絞る(answer_filter="jev")。Haiku に渡すチャンクを並べた順で返す。"""
        if params["answer_filter"] != ANSWER_FILTER_JEV:
            return list(chunks)
        af: dict[str, Any] = {"n_in": len(chunks), "n_out": len(chunks), "kept": [], "dropped": [],
                              "requests": 0, "input_tokens": 0, "usd": 0.0, "fallback": None,
                              "halted_at": None, "min_applied": False,
                              "threshold": params["answer_filter_threshold"],
                              "max": params["answer_filter_max"], "min": params["answer_filter_min"]}
        trace["answer_filter"] = af
        items = answer_filter_items(chunks)
        groups = answer_filter_groups(q_state, items)
        scores: dict[str, float | None] = {}
        try:
            # 途中の組だけ判定して半端に絞らないよう、全組ぶんの回数を先に確かめる
            guard("jev", "answer_filter", n_jev=len(groups))
            for group in groups:
                record = judge(q_state, answer_filter_questions(group), "answer_filter")
                af["requests"] += 1
                af["input_tokens"] += int((record.get("usage") or {}).get("input_tokens") or 0)
                for item in group:
                    scores[item["id"]] = noul_value(record, item["qid"])
        except _Halt as halt:
            af["fallback"], af["halted_at"] = halt.reason, halt.where
        af["usd"] = af["input_tokens"] * PRICES[JEV_MODEL]["input"] / 1e6
        rows = [{"id": c["id"], "score": scores.get(c["id"]), "hits": chunk_log[c["id"]]["hits"]}
                for c in chunks]
        if af["fallback"] is None and all(r["score"] is None for r in rows):
            af["fallback"] = ANSWER_FILTER_NO_SCORES
        if af["fallback"] is not None:
            af["kept"] = rows                           # 絞らずに全部(集めた順)
            return list(chunks)
        kept, dropped, af["min_applied"] = select_answer_chunks(
            rows, params["answer_filter_threshold"], params["answer_filter_max"], params["answer_filter_min"])
        af["kept"], af["dropped"], af["n_out"] = kept, dropped, len(kept)
        by_id = {c["id"]: c for c in chunks}
        return [by_id[r["id"]] for r in kept]

    stop: str | None = None
    try:
        hop_log = {"hop": 0, "frontier": [], "neighbor_scores": [],
                   "selected": [e["id"] for e in entries], "chunk_source": CHUNK_SOURCE_APPEARANCE,
                   "new_chunk_ids": [], "dropped_chunk_ids": [],
                   "missing_chunk_ids": [], "capped_chunk_ids": {}, "node_chunk_counts": {},
                   "hop_capped_chunk_ids": [], "hop_capped_by_entity": {}, "dup_chunk_ids": [],
                   "sufficiency": None, "sufficiency_reused": False,
                   "in_scope_overridden": trace["in_scope_overridden"],
                   "entry_chunk_rank": params["entry_chunk_rank"], "entry_chunk_ranking": None}
        trace["hops"].append(hop_log)
        visit(0, [(e["id"], None) for e in entries], hop_log)
        if not collected:
            stop = "no_chunks"
        else:
            stop = after_hop(0, hop_log)
        frontier = [e["id"] for e in entries]
        hop = 0
        while stop is None:
            hop += 1
            candidates = neighbor_candidates(frontier, nodes, master, visited)
            hop_log = {"hop": hop, "frontier": list(frontier), "neighbor_scores": [], "selected": [],
                       "chunk_source": params["hop_chunk_source"], "new_chunk_ids": [], "dropped_chunk_ids": [], "missing_chunk_ids": [],
                       "capped_chunk_ids": {}, "node_chunk_counts": {},
                       "hop_capped_chunk_ids": [], "hop_capped_by_entity": {}, "dup_chunk_ids": [],
                       "sufficiency": None, "sufficiency_reused": False}
            trace["hops"].append(hop_log)
            if not candidates:
                stop = "no_neighbors"
                break
            scored = []
            if scorer is not None:
                got = timer.run("jev", lambda c=candidates: scorer.score_neighbors(c))
                scored = [{**cand, "score": got.get(cand["qid"])} for cand in candidates]
                min_score = -math.inf
            else:
                batches = split_items(q_state, candidates, neighbor_questions, lambda c: c["qid"],
                                      max_questions=NEIGHBOR_BATCH_SIZE)
                # 途中のバッチだけ判定して半端に選ばないよう、全バッチぶんの回数を先に確かめる
                guard("jev", f"hop{hop}:neighbors", n_jev=len(batches))
                for batch in batches:
                    record = judge(q_state, neighbor_questions(batch), f"hop{hop}:neighbors")
                    for cand in batch:
                        scored.append({**cand, "score": noul_value(record, cand["qid"])})
                min_score = params["min_neighbor_score"]
            hop_log["neighbor_scores"] = [
                {**{k: c[k] for k in ("entity", "edge", "direction", "score", "from", "also_from", "sentence")},
                 **({"excluded": EXCLUDED_BELOW_MIN} if below_min(c["score"], min_score) else {})}
                for c in sorted(scored, key=lambda c: -(c["score"] if c["score"] is not None else -math.inf))
            ]
            chosen = select_neighbors(scored, params["max_neighbors"], min_score)
            if not chosen:
                stop = "no_neighbors"
                break
            hop_log["selected"] = [c["entity"] for c in chosen]
            by_evidence = params["hop_chunk_source"] == CHUNK_SOURCE_EDGE_EVIDENCE
            relations = relation_order(scored, [c["entity"] for c in chosen], min_score) \
                if by_evidence else None
            visit(hop, [(c["entity"], c["edge"]) for c in chosen], hop_log, relations)
            frontier = [c["entity"] for c in chosen]
            stop = after_hop(hop, hop_log)
    except _Halt as halt:
        stop = halt.reason
        trace["halted_at"] = halt.where

    trace["stop_reason"] = stop
    trace["chunks"] = [chunk_log[c["id"]] for c in collected]

    # ---- 6. 回答
    if not collected:
        if stop == "deadline":
            trace.update({"answer": "", "answerable": None})
            return _finish(trace, records, timer, started, clock)
        return cannot_answer("no_chunks")
    passed = answer_chunk_selector(list(collected)) if answer_chunk_selector is not None \
        else filter_for_answer(collected)
    trace["answer_chunk_ids"] = [c["id"] for c in passed]
    _answer(trace, question, passed, llm, gen_model, keep, timer, guard,
            answer_style=params["answer_style"], method_label=method_label)
    return _finish(trace, records, timer, started, clock)


def _answer(trace: dict, question: str, collected: Sequence[Mapping[str, Any]], llm: Any,
            gen_model: str, keep: Callable[[dict], dict], timer: _Timer,
            guard: Callable[..., None], *, answer_style: str = DEFAULT_ANSWER_STYLE,
            method_label: str | None = None) -> None:
    """Haiku で回答し、引用を検査してトレースに書く。system は `answer_style` の文面
    (`ANSWER_SYSTEMS`。user・JSON の形・引用の検査はどちらも同じ)。"""
    system = ANSWER_SYSTEMS[answer_style]
    base_user = answer_user_prompt(question, collected)
    parsed = None
    raw_text = None
    for attempt in range(2):
        try:
            guard("gen", "answer" if attempt == 0 else "answer_retry")
        except _Halt as halt:
            trace.update({"stop_reason": halt.reason, "halted_at": halt.where,
                          "answer": "", "answerable": None})
            if raw_text is not None:
                trace["parse_error"] = True
                trace["answer_raw"] = raw_text
            return
        user = base_user if attempt == 0 else base_user + RETRY_NOTE
        payload = converse_payload(system, user, temperature=GEN_TEMPERATURE,
                                   max_tokens=GEN_MAX_TOKENS)
        task = {"key": converse_key(gen_model, payload, kind="gen"), "system": system,
                "user": user, "payload": payload,
                "extra": {"method": method_label or f"graphrag-{trace['mode']}", "attempt": attempt,
                          "answer_style": answer_style}}
        record = timer.run("bedrock", lambda t=task: llm.run(t, use_cache=False))
        trace["answer_attempts"] = attempt + 1
        if record is None:
            fatal = getattr(llm, "fatal", None)
            failures = getattr(llm, "failures", None) or []
            trace["bedrock_error"] = fatal or (failures[-1]["error"] if failures else {"type": "unknown"})
            trace.update({"answer": "", "answerable": None})
            return
        keep(record)
        response = record.get("response") or {}
        raw_text = response.get("text") or ""
        stop_reason = response.get("stop_reason")
        trace["answer_stop_reasons"].append(stop_reason)
        parsed = parse_answer(raw_text)
        if parsed is not None:
            break
        if stop_reason == "max_tokens":
            trace["answer_truncated"] = True
            break
    if parsed is None:
        trace.update({"parse_error": True, "answer_raw": raw_text, "answer": "", "answerable": None})
        return
    trace.update(parsed)
    trace.update(check_citations(parsed["citations"], [c["id"] for c in collected], parsed["answerable"]))
    trace["answerable_mismatch"] = answerable_mismatch(parsed["answer"], parsed["answerable"])


def _finish(trace: dict, records: list[dict], timer: _Timer, started: float,
            clock: Callable[[], float]) -> dict:
    jev_records = [r for r in records if r.get("kind") == "jev"]
    gen_records = [r for r in records if r.get("kind") != "jev"]
    trace["jev_calls"] = len(jev_records)
    trace["latency_ms"] = {"total": (clock() - started) * 1000.0, **timer.ms}
    trace["tokens"] = {
        "jev_input": sum(int((r.get("usage") or {}).get("input_tokens") or 0) for r in jev_records),
        "bedrock_input": sum(int((r.get("usage") or {}).get("input_tokens") or 0) for r in gen_records),
        "bedrock_output": sum(int((r.get("usage") or {}).get("output_tokens") or 0) for r in gen_records),
    }
    trace["records"] = records
    return trace


def _ranking_summary(ranking: Mapping[str, Any] | None) -> dict | None:
    """サマリー用の入口チャンクの判定。スコアは上位 10 件だけ(チャンク数が多いとログが長くなるため)。"""
    if ranking is None:
        return None
    return {**{k: v for k, v in ranking.items() if k != "scores"},
            "top_scores": list(ranking.get("scores") or [])[:10]}


def summary(trace: Mapping[str, Any]) -> dict:
    """CloudWatch Logs 用のサマリー。**本文テキスト(チャンク・回答・主張)と呼び出しレコードは入れない**。
    質問文は識別のために入れる。"""
    return {
        "kind": "query_summary",
        "question": trace.get("question"),
        "mode": trace.get("mode"),
        "params": trace.get("params"),
        "in_scope": trace.get("in_scope"),
        "entry_prefilter": trace.get("entry_prefilter"),
        "entry_candidates": trace.get("entry_candidates"),
        "in_scope_overridden": trace.get("in_scope_overridden", False),
        "entry": trace.get("entry"),
        "entry_fallback_used": trace.get("entry_fallback_used", False),
        "entry_fallback": trace.get("entry_fallback"),
        "hops": [
            {"hop": h.get("hop"), "frontier": h.get("frontier"), "selected": h.get("selected"),
             "chunk_source": h.get("chunk_source"),
             "n_candidates": len(h.get("neighbor_scores") or []),
             "n_below_min": sum(1 for c in h.get("neighbor_scores") or []
                                if c.get("excluded") == EXCLUDED_BELOW_MIN),
             "new_chunk_ids": h.get("new_chunk_ids"), "dropped_chunk_ids": h.get("dropped_chunk_ids"),
             "missing_chunk_ids": h.get("missing_chunk_ids"),
             "capped_chunk_ids": h.get("capped_chunk_ids"),
             "hop_capped_chunk_ids": h.get("hop_capped_chunk_ids"),
             "dup_chunk_ids": h.get("dup_chunk_ids"), "sufficiency": h.get("sufficiency"),
             **({"in_scope_overridden": h["in_scope_overridden"]} if "in_scope_overridden" in h else {}),
             **({"entry_chunk_rank": h["entry_chunk_rank"]} if "entry_chunk_rank" in h else {}),
             **({"entry_chunk_ranking": _ranking_summary(h["entry_chunk_ranking"])}
                if "entry_chunk_ranking" in h else {})}
            for h in trace.get("hops") or []
        ],
        "chunks": trace.get("chunks"),
        "answer_filter": trace.get("answer_filter"),
        "answer_chunk_ids": trace.get("answer_chunk_ids"),
        "stop_reason": trace.get("stop_reason"),
        "halted_at": trace.get("halted_at"),
        "truncated": trace.get("truncated"),
        "answerable": trace.get("answerable"),
        "answerable_mismatch": trace.get("answerable_mismatch"),
        "answer_chars": len(trace.get("answer") or ""),
        "citation_ids": [c.get("chunk_ids") for c in trace.get("citations") or []],
        "citation_format_fixes": trace.get("citation_format_fixes"),
        "citations_valid": trace.get("citations_valid"),
        "invalid_citation_ids": trace.get("invalid_citation_ids"),
        "uncited_claims": trace.get("uncited_claims"),
        "no_citations": trace.get("no_citations"),
        "parse_error": trace.get("parse_error"),
        "answer_attempts": trace.get("answer_attempts"),
        "answer_stop_reasons": trace.get("answer_stop_reasons"),
        "answer_truncated": trace.get("answer_truncated"),
        "bedrock_error": trace.get("bedrock_error"),
        "jev_calls": trace.get("jev_calls"),
        "latency_ms": trace.get("latency_ms"),
        "tokens": trace.get("tokens"),
    }
