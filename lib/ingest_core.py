#!/usr/bin/env python3
"""登録処理の中身。ingest Lambda から使う。

1. ノード判定: チャンク本文から辞書で候補を拾い、候補のエンティティだけを Jev に「登場するか」を問う
   (`node_prefilter="none"` なら全エンティティ)。候補 0 件のチャンクは Jev を呼ばない
2. 関係判定: 登場ノードの順列 × 許可されたエッジ種別で候補文を作り、Jev に判定させる
   (対称な関係は順番なしのペアで 1 回だけ)。候補が `MAX_EDGE_CANDIDATES` を超えたら 1 問も投げずに止まる
3. 書き込み: `dry_run=False` のとき、判定が全部終わってから チャンク → 登場 → 関係 の順に書く。
   `reset=True` なら書く直前に 2 テーブルを空にする

ファイル・ネットワークには直接触らない(Jev クライアントと store は引数で受け取る)。"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from itertools import permutations

from common import JEV_MODEL, PRICES
from entry_dictionary import INGEST_MIN_TERM_CHARS, find_candidates
from jev_questions import (
    MAX_REQUEST_TOKENS_EST,
    TOKENS_PER_CHAR,
    edge_qid,
    edge_questions,
    edge_scores,
    node_qid,
    node_questions,
    node_scores,
    split_items,
)
from master import allowed_edges, is_symmetric, render, symmetric_edge_names

STAGES = ("nodes", "edges")
NODE_MODES = ("packed", "single")
NODE_PREFILTER_DICTIONARY = "dictionary"
NODE_PREFILTER_NONE = "none"
NODE_PREFILTERS = (NODE_PREFILTER_DICTIONARY, NODE_PREFILTER_NONE)
DEFAULT_NODE_PREFILTER = NODE_PREFILTER_DICTIONARY
DEFAULT_NODE_THRESHOLD = 0.20  # 登場・関係とみなすしきい値
DEFAULT_EDGE_THRESHOLD = 0.55
EDGE_BATCH_SIZE = 50           # 1 リクエストに詰める関係判定の問いの上限
MAX_EDGE_CANDIDATES = 5000


RecordHook = Callable[[dict], None]
RowsHook = Callable[[list[dict]], None]


class EdgeCandidateLimit(RuntimeError):
    """関係判定の候補が多すぎる(`MAX_EDGE_CANDIDATES` 超)。Jev は呼んでいない。"""


def node_state(chunk: Mapping[str, Any]) -> dict:
    """ノード判定の state。packed / single で同じものを使う(比較の条件をそろえるため)。"""
    return {"chunk": chunk["text"]}


def check_node_prefilter(prefilter: str) -> str:
    if prefilter not in NODE_PREFILTERS:
        raise ValueError(f"node_prefilter は {NODE_PREFILTERS} のどれかです: {prefilter!r}")
    return prefilter


def node_candidates(
    chunk: Mapping[str, Any],
    master: Mapping[str, Any],
    prefilter: str = DEFAULT_NODE_PREFILTER,
) -> list[Mapping[str, Any]]:
    """ノード判定で Jev に問うエンティティ(マスターの並び順)。"""
    check_node_prefilter(prefilter)
    if prefilter == NODE_PREFILTER_NONE:
        return list(master["entities"])
    hit = {c["id"] for c in find_candidates(chunk["text"], master, INGEST_MIN_TERM_CHARS)}
    return [e for e in master["entities"] if e["id"] in hit]


def _node_groups(entities: Sequence[Mapping[str, Any]], mode: str,
                 state: Mapping[str, Any] | None = None) -> list[list[Mapping[str, Any]]]:
    """1 リクエストに詰めるエンティティの組。"""
    if mode == "packed":
        return split_items(state if state is not None else {"chunk": ""}, list(entities), node_questions,
                           lambda e: node_qid(e["id"]))
    if mode == "single":
        return [[entity] for entity in entities]
    raise ValueError(f"mode は {NODE_MODES} のどれかです: {mode!r}")


def judge_nodes(
    client: Any,
    chunk: Mapping[str, Any],
    entities: Sequence[Mapping[str, Any]],
    mode: str = "packed",
    *,
    on_record: RecordHook | None = None,
) -> tuple[dict[str, float | None], list[dict]]:
    """1 チャンクのノード判定。"""
    state = node_state(chunk)
    scores: dict[str, float | None] = {}
    records: list[dict] = []
    if not entities:
        _node_groups(entities, mode, state)    # mode の書き間違いは候補 0 件でも弾く
        return scores, records
    for group in _node_groups(entities, mode, state):
        record = client.judge(state, node_questions(group), use_cache=False)
        records.append(record)
        if on_record is not None:
            on_record(record)
        scores.update(node_scores(record, [entity["id"] for entity in group]))
    return scores, records


def node_rows(
    chunk: Mapping[str, Any],
    entities: Iterable[Mapping[str, Any]],
    scores: Mapping[str, float | None],
    mode: str,
    judged: Iterable[str] | None = None,
) -> list[dict]:
    """判定ログの行(チャンク × 全エンティティで 1 組 1 行)。キーは
    `chunk_id, entity_id, entity_name, entity_type, prob, mode, prefiltered` の 7 つだけ。"""
    asked = None if judged is None else set(judged)
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "entity_id": entity["id"],
            "entity_name": entity["name"],
            "entity_type": entity["type"],
            "prob": scores.get(entity["id"]),
            "mode": mode,
            "prefiltered": asked is not None and entity["id"] not in asked,
        }
        for entity in entities
    ]


def missing_score_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Jev に問うたのにスコアが取れなかった行(prob が None で、辞書で除いた行ではないもの)。"""
    return [r for r in rows if r.get("prob") is None and not r.get("prefiltered")]


def present_ids(
    scores: Mapping[str, float | None], threshold: float = DEFAULT_NODE_THRESHOLD
) -> list[str]:
    """しきい値以上の entity_id を、スコアの高い順に返す(None は除く)。"""
    hits = [(eid, p) for eid, p in scores.items() if p is not None and p >= threshold]
    return [eid for eid, _ in sorted(hits, key=lambda item: -item[1])]


def estimate_nodes(
    chunks: Sequence[Mapping[str, Any]],
    entities: Sequence[Mapping[str, Any]],
    mode: str = "packed",
    prefilter: str = DEFAULT_NODE_PREFILTER,
) -> dict:
    """ノード判定を呼ぶ**前に**出す見積もり。実際に送る JSON を組み立てて文字数から概算する。
    `prefilter="dictionary"` なら辞書の候補(`node_candidates()`)だけを問う前提で数える。"""
    check_node_prefilter(prefilter)
    master = {"entities": list(entities)}     # 辞書は 1 回だけ組み立てる(同じオブジェクトを使い回す)
    n_requests = 0
    n_judgments = 0
    n_empty = 0
    max_cands = 0
    chars = 0
    max_chars = 0
    for chunk in chunks:
        state = node_state(chunk)
        cands = node_candidates(chunk, master, prefilter)
        n_judgments += len(cands)
        max_cands = max(max_cands, len(cands))
        if not cands:
            n_empty += 1
            continue
        for group in _node_groups(cands, mode, state):
            payload = {"state": state, "questions": node_questions(group)}
            size = len(json.dumps(payload, ensure_ascii=False))
            chars += size
            max_chars = max(max_chars, size)
            n_requests += 1
    est_tokens = int(chars * TOKENS_PER_CHAR)
    est_usd = est_tokens * float(PRICES[JEV_MODEL]["input"]) / 1e6
    return {
        "n_requests": n_requests,
        "n_judgments": n_judgments,
        "n_pairs": len(chunks) * len(entities),
        "n_chunks_no_candidates": n_empty,
        "max_candidates": max_cands,
        "prefilter": prefilter,
        "est_input_tokens": est_tokens,
        "est_usd": est_usd,
        "max_request_tokens": int(max_chars * TOKENS_PER_CHAR),
    }


# ============================================================== 関係判定

def edge_state(chunk: Mapping[str, Any]) -> dict:
    """関係判定の state。ノード判定と同じ形(`{"chunk": 本文}`)。"""
    return {"chunk": chunk["text"]}


def edge_candidates(
    master: Mapping[str, Any], present_entities: Sequence[Mapping[str, Any]]
) -> list[dict]:
    """登場ノードの `permutations(present, 2)` × `allowed_edges(source.type, target.type)` の候補。"""
    edge_index = {edge["name"]: i for i, edge in enumerate(master["edge_types"])}
    sym_edges = [e for e in master["edge_types"] if is_symmetric(e)]
    candidates: list[dict] = []
    for source, target in permutations(present_entities, 2):
        allowed = [e for e in allowed_edges(master, source["type"], target["type"]) if not is_symmetric(e)]
        if source["id"] < target["id"]:
            pair_ok = {e["name"] for e in allowed_edges(master, source["type"], target["type"])}
            pair_ok |= {e["name"] for e in allowed_edges(master, target["type"], source["type"])}
            allowed += [e for e in sym_edges if e["name"] in pair_ok]
        for edge in sorted(allowed, key=lambda e: edge_index[e["name"]]):
            idx = edge_index[edge["name"]]
            candidates.append({
                "qid": edge_qid(source["id"], idx, target["id"]),
                "source_id": source["id"],
                "source_name": source["name"],
                "edge": edge["name"],
                "edge_index": idx,
                "target_id": target["id"],
                "target_name": target["name"],
                "sentence": render(edge["template"], source, target),
                "symmetric": is_symmetric(edge),
            })
    return candidates


def edge_batches(candidates: Sequence[Mapping[str, Any]], batch_size: int = EDGE_BATCH_SIZE,
                 state: Mapping[str, Any] | None = None,
                 max_tokens: int = MAX_REQUEST_TOKENS_EST) -> list[list]:
    """候補を最大 `batch_size` 問ずつに分ける(順序を保つ)。候補 0 件なら空リスト(=呼ばない)。"""
    if batch_size < 1:
        raise ValueError(f"batch_size は 1 以上です: {batch_size}")
    if state is None:
        return [list(candidates[i:i + batch_size]) for i in range(0, len(candidates), batch_size)]
    return split_items(state, list(candidates), edge_questions, lambda c: c["qid"],
                       max_tokens=max_tokens, max_questions=batch_size)


def check_edge_candidates(n_candidates: int, limit: int = MAX_EDGE_CANDIDATES) -> None:
    """候補数が上限を超えていたら EdgeCandidateLimit(ちょうど上限は通す)。"""
    if n_candidates > limit:
        raise EdgeCandidateLimit(
            f"関係判定の候補が {n_candidates:,} 件あり、上限 {limit:,} 件を超えています(Jev は呼んでいません)"
        )


def edge_rows(chunk: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]],
              scores: Mapping[str, float | None]) -> list[dict]:
    """判定ログの行(1 判定 1 行)。キーは
    `chunk_id, source_id, source_name, edge, target_id, target_name, sentence, prob` の 8 つだけ。"""
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "source_id": cand["source_id"],
            "source_name": cand["source_name"],
            "edge": cand["edge"],
            "target_id": cand["target_id"],
            "target_name": cand["target_name"],
            "sentence": cand["sentence"],
            "prob": scores.get(cand["qid"]),
        }
        for cand in candidates
    ]


def judge_edges(
    client: Any,
    chunk: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    batch_size: int = EDGE_BATCH_SIZE,
    on_record: RecordHook | None = None,
) -> tuple[list[dict], list[dict]]:
    """1 チャンクの関係判定。最大 `batch_size` 問ずつ 1 リクエスト(`state={"chunk": 本文}`)。"""
    state = edge_state(chunk)
    scores: dict[str, float | None] = {}
    records: list[dict] = []
    for batch in edge_batches(candidates, batch_size, state):
        record = client.judge(state, edge_questions(batch), use_cache=False)
        records.append(record)
        if on_record is not None:
            on_record(record)
        scores.update(edge_scores(record, [cand["qid"] for cand in batch]))
    return edge_rows(chunk, candidates, scores), records


def node_scores_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    """ノード判定ログの行(`node_rows()` の形)を `{chunk_id: {entity_id: prob}}` に戻す。"""
    out: dict[str, dict[str, float | None]] = {}
    modes: set[str] = set()
    for row in rows:
        modes.add(str(row.get("mode")))
        prob = row.get("prob")
        out.setdefault(row["chunk_id"], {})[row["entity_id"]] = None if prob is None else float(prob)
    if len(modes) > 1:
        raise ValueError(f"判定ログにモードが混ざっています: {sorted(modes)}")
    return out


def plan_edges(
    master: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    present: Mapping[str, Sequence[str]],
    batch_size: int = EDGE_BATCH_SIZE,
) -> dict:
    """関係判定を呼ぶ**前に**出す計画と見積もり。実際に送る JSON の文字数から概算する。"""
    ents = {e["id"]: e for e in master["entities"]}
    per_chunk: list[dict] = []
    all_cands: dict[str, list[dict]] = {}
    chars = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        cands = edge_candidates(master, [ents[eid] for eid in present.get(cid, [])])
        batches = edge_batches(cands, batch_size, edge_state(chunk))
        for batch in batches:
            payload = {"state": edge_state(chunk), "questions": edge_questions(batch)}
            chars += len(json.dumps(payload, ensure_ascii=False))
        all_cands[cid] = cands
        per_chunk.append({
            "chunk_id": cid,
            "n_present": len(present.get(cid, [])),
            "n_candidates": len(cands),
            "n_requests": len(batches),
        })
    est_tokens = int(chars * TOKENS_PER_CHAR)
    return {
        "per_chunk": per_chunk,
        "candidates": all_cands,
        "n_candidates": sum(p["n_candidates"] for p in per_chunk),
        "n_requests": sum(p["n_requests"] for p in per_chunk),
        "est_input_tokens": est_tokens,
        "est_usd": est_tokens * float(PRICES[JEV_MODEL]["input"]) / 1e6,
    }


# ============================================================== 本体

def run_ingest(
    client: Any,
    master: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool = True,
    stages: Sequence[str] = ("nodes",),
    mode: str = "packed",
    node_prefilter: str = DEFAULT_NODE_PREFILTER,
    node_threshold: float = DEFAULT_NODE_THRESHOLD,
    node_scores: Mapping[str, Mapping[str, float | None]] | None = None,
    edge_batch_size: int = EDGE_BATCH_SIZE,
    max_edge_candidates: int = MAX_EDGE_CANDIDATES,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    store: Any = None,
    reset: bool = False,
    on_record: RecordHook | None = None,
    on_node_rows: RowsHook | None = None,
    on_edge_rows: RowsHook | None = None,
) -> dict:
    """登録処理の本体。"""
    check_node_prefilter(node_prefilter)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"未知の stage です: {unknown}(有効: {STAGES})")
    if not stages:
        raise ValueError("stages が空です")
    if node_scores is not None and "nodes" in stages:
        raise ValueError("node_scores を渡すときは stages に nodes を入れない(ノード判定をやり直さない)")
    if node_scores is None and "nodes" not in stages:
        raise ValueError("edges だけ実行するときは node_scores(ノード判定のスコア)が必要です")
    if not dry_run:
        if store is None:
            raise ValueError("dry_run=False には store(graph_store.GraphStore)が必要です")
        if "edges" not in stages:
            raise ValueError("dry_run=False のときは stages に edges が必要です(登場だけの半端なグラフを作らない)")
    elif reset:
        raise ValueError("reset=True は dry_run=False のときだけ指定できます")

    entities = list(master["entities"])
    result: dict[str, Any] = {
        "nodes": {}, "present": {}, "node_candidates": {}, "node_rows": [], "edge_plan": None, "edge_rows": [], "records": [],
        "written": None, "deleted": None,
        "timings_ms": {"nodes": 0.0, "edges": 0.0, "reset": 0.0, "write": 0.0},
    }
    started = time.perf_counter()

    # ---- 1. 登場ノード(Jev で判定 / 渡されたスコアを使う)
    if node_scores is not None:
        missing = [c["chunk_id"] for c in chunks if c["chunk_id"] not in node_scores]
        if missing:
            raise ValueError(f"node_scores にチャンクがありません: {missing}")
        for chunk in chunks:
            scores = dict(node_scores[chunk["chunk_id"]])
            result["nodes"][chunk["chunk_id"]] = scores
            result["present"][chunk["chunk_id"]] = present_ids(scores, node_threshold)
    else:
        for chunk in chunks:
            cands = node_candidates(chunk, master, node_prefilter)
            scores, records = judge_nodes(client, chunk, cands, mode, on_record=on_record)
            judged = [e["id"] for e in cands]
            rows = node_rows(chunk, entities, scores, mode, judged=judged)
            result["node_candidates"][chunk["chunk_id"]] = judged
            result["nodes"][chunk["chunk_id"]] = scores
            result["present"][chunk["chunk_id"]] = present_ids(scores, node_threshold)
            result["node_rows"].extend(rows)
            result["records"].extend(records)
            if on_node_rows is not None:
                on_node_rows(rows)

    result["timings_ms"]["nodes"] = (time.perf_counter() - started) * 1000.0

    if "edges" not in stages:
        return result

    # ---- 2. 関係判定(全候補を作ってから件数ガード → 判定)
    started = time.perf_counter()
    plan = plan_edges(master, chunks, result["present"], edge_batch_size)
    check_edge_candidates(plan["n_candidates"], max_edge_candidates)
    result["edge_plan"] = plan
    for chunk in chunks:
        rows, records = judge_edges(
            client, chunk, plan["candidates"][chunk["chunk_id"]], edge_batch_size, on_record=on_record
        )
        result["edge_rows"].extend(rows)
        result["records"].extend(records)
        if on_edge_rows is not None:
            on_edge_rows(rows)
    result["timings_ms"]["edges"] = (time.perf_counter() - started) * 1000.0

    if dry_run:
        return result

    # ---- 3. 書き込み(判定が全部終わってから。途中で Jev が落ちたら何も消さない・書かない)
    if reset:
        started = time.perf_counter()
        result["deleted"] = store.reset()
        result["timings_ms"]["reset"] = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    appearances = appearance_records(result["nodes"], node_threshold)
    edges = passing_edges(result["edge_rows"], edge_threshold, symmetric_edge_names(master))
    n_chunks = store.put_chunks(chunks)
    n_app = store.put_appearances(appearances)
    n_out, n_in = store.put_edges([e for e in edges if not e["symmetric"]])
    n_sym = store.put_sym_edges([e for e in edges if e["symmetric"]])
    result["timings_ms"]["write"] = (time.perf_counter() - started) * 1000.0
    result["written"] = {"chunks": n_chunks, "appearances": n_app, "edges_out": n_out, "edges_in": n_in,
                         "edges_sym": n_sym}
    return result


def appearance_records(
    nodes: Mapping[str, Mapping[str, float | None]], threshold: float
) -> list[dict]:
    """`{chunk_id: {entity_id: prob}}` から、しきい値以上の登場行 `{entity_id, chunk_id, score}`。"""
    return [
        {"entity_id": eid, "chunk_id": cid, "score": nodes[cid][eid]}
        for cid in nodes
        for eid in present_ids(nodes[cid], threshold)
    ]


def passing_edges(rows: Iterable[Mapping[str, Any]], threshold: float,
                  symmetric: Iterable[str] = ()) -> list[dict]:
    """関係判定ログの行から、しきい値以上を `{source_id, edge, target_id, chunk_id, score, symmetric}` で返す。
    `symmetric` は対称な関係のエッジ名(`master.symmetric_edge_names()`)。その関係は `symmetric: True`
    (書き込みは `put_sym_edges()`)。"""
    sym = frozenset(symmetric)
    return [
        {"source_id": r["source_id"], "edge": r["edge"], "target_id": r["target_id"],
         "chunk_id": r["chunk_id"], "score": r["prob"], "symmetric": r["edge"] in sym}
        for r in rows
        if r.get("prob") is not None and r["prob"] >= threshold
    ]
