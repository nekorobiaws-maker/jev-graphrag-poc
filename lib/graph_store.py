#!/usr/bin/env python3
"""DynamoDB の `chunk` / `graph` テーブルの読み書き。

| テーブル | 種類 | PK | SK | 属性 |
|---|---|---|---|---|
| chunk | チャンク | `CHUNK#<chunk_id>` | - | `text`、`doc_id` |
| graph | 登場 | `ENT#<id>` | `CHUNK#<chunk_id>` | `score` |
| graph | 関係(順方向) | `ENT#<src>` | `OUT#<edge名>#ENT#<tgt>#<chunk_id>` | `score` |
| graph | 関係(逆方向) | `ENT#<tgt>` | `IN#<edge名>#ENT#<src>#<chunk_id>` | `score` |
| graph | 対称な関係 | `ENT#<a>` / `ENT#<b>` | `SYM#<edge名>#ENT#<相手>#<chunk_id>` | `score` |
| graph | マスターの版 | `MASTER` | `META` | `version`、`entities`、`node_types`、`edge_types`、`updated_at`、`source_sha256` |
| graph | ノード種別 | `MASTER` | `NODETYPE#<name>` | `name`、`description`、`index` |
| graph | エッジ種別 | `MASTER` | `EDGE#<添字 4 桁>` | `name`、`template`、`allowed`、`symmetric`、`index` |
| graph | エンティティ | `MASTER` | `ENT#<id>` | `name`、`aliases`、`type`、`description`、`index` |

- 1 つの根拠チャンクを 1 アイテムにする。向きのある関係は順・逆の 2 行、対称な関係は両端に `SYM#` の 2 行
- マスター(`PK=MASTER`)は `reset()` で消さない。`META` は最後に書く
- キーの部品に `#` が入っていたら ValueError。`score` は Decimal で書き、読むときは float に戻す
- 触れるテーブルは `jev-graphrag-chunk` と `jev-graphrag-graph` だけ(完全一致。読み書きの直前に毎回確かめる)"""

from __future__ import annotations

import math
import time
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence

from common import REGION, boto_config

CHUNK_TABLE = "jev-graphrag-chunk"
GRAPH_TABLE = "jev-graphrag-graph"
ALLOWED_TABLES = frozenset((CHUNK_TABLE, GRAPH_TABLE))

SEP = "#"
CHUNK_PREFIX = "CHUNK"
ENT_PREFIX = "ENT"
OUT_PREFIX = "OUT"
IN_PREFIX = "IN"
SYM_PREFIX = "SYM"

MASTER_PK = "MASTER"
_MASTER_PK_GUARD = "MASTER"   # reset 直前のガード用。除外の条件(MASTER_PK)とは別に持つ(片方が壊れても止まる)
MASTER_META_SK = "META"
NODETYPE_PREFIX = "NODETYPE"
EDGE_PREFIX = "EDGE"
EDGE_INDEX_WIDTH = 4

BATCH_GET_MAX = 100          # BatchGetItem の 1 回あたりのキー数上限
BATCH_GET_RETRIES = 5        # UnprocessedKeys を投げ直す回数


class TableGuardError(RuntimeError):
    """許可されていないテーブル名を渡された。"""


class MasterGuardError(RuntimeError):
    """全削除(reset)でマスターの行(`PK=MASTER`)を消そうとした。"""


def assert_not_master(key: Mapping[str, Any]) -> Mapping[str, Any]:
    """削除するキーがマスターの行でないことを確かめる。マスターなら MasterGuardError。"""
    if str(key.get("PK")) == _MASTER_PK_GUARD:
        raise MasterGuardError(f"マスターの行は reset で消しません: {dict(key)!r}")
    return key


def assert_table(table_name: Any, expected: str) -> str:
    """テーブル名が `expected` と**完全一致**するか確かめる。違えば TableGuardError。"""
    if expected not in ALLOWED_TABLES:
        raise TableGuardError(f"許可リストに無いテーブルです: {expected!r}")
    if not isinstance(table_name, str) or table_name != expected:
        raise TableGuardError(
            f"このモジュールが触れるテーブルは {expected!r} です: 渡されたのは {table_name!r}"
        )
    return table_name


# ============================================================== キーの組み立て(純粋関数)

def _part(value: Any, what: str) -> str:
    """キーの部品。空・文字列以外・`#` 入りは ValueError。"""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} が空か文字列ではありません: {value!r}")
    if SEP in value:
        raise ValueError(f"{what} に区切り文字 {SEP!r} が入っています: {value!r}")
    return value


def to_decimal(value: Any) -> Decimal:
    """スコアを DynamoDB に書ける Decimal にする。bool・NaN・無限大は ValueError。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"score は数値です: {value!r}")
    as_float = float(value)
    if math.isnan(as_float) or math.isinf(as_float):
        raise ValueError(f"score が有限の数ではありません: {value!r}")
    return Decimal(str(as_float))


def chunk_pk(chunk_id: str) -> str:
    return f"{CHUNK_PREFIX}{SEP}{_part(chunk_id, 'chunk_id')}"


def ent_pk(entity_id: str) -> str:
    return f"{ENT_PREFIX}{SEP}{_part(entity_id, 'entity_id')}"


def appearance_sk(chunk_id: str) -> str:
    return chunk_pk(chunk_id)


def out_sk(edge: str, target_id: str, chunk_id: str) -> str:
    return SEP.join((OUT_PREFIX, _part(edge, "edge"), ENT_PREFIX,
                     _part(target_id, "target_id"), _part(chunk_id, "chunk_id")))


def in_sk(edge: str, source_id: str, chunk_id: str) -> str:
    return SEP.join((IN_PREFIX, _part(edge, "edge"), ENT_PREFIX,
                     _part(source_id, "source_id"), _part(chunk_id, "chunk_id")))


def sym_sk(edge: str, other_id: str, chunk_id: str) -> str:
    return SEP.join((SYM_PREFIX, _part(edge, "edge"), ENT_PREFIX,
                     _part(other_id, "other_id"), _part(chunk_id, "chunk_id")))


def chunk_item(chunk: Mapping[str, Any]) -> dict:
    """chunk テーブルの 1 アイテム。"""
    text = chunk.get("text")
    if not isinstance(text, str):
        raise ValueError(f"chunk の text が文字列ではありません: {chunk.get('chunk_id')!r}")
    return {"PK": chunk_pk(chunk["chunk_id"]), "text": text, "doc_id": str(chunk.get("doc_id") or "")}


def appearance_item(entity_id: str, chunk_id: str, score: Any) -> dict:
    """登場エッジ 1 行。"""
    return {"PK": ent_pk(entity_id), "SK": appearance_sk(chunk_id), "score": to_decimal(score)}


def edge_items(source_id: str, edge: str, target_id: str, chunk_id: str, score: Any) -> tuple[dict, dict]:
    """関係 1 件ぶんの (順方向, 逆方向)。2 行は同じ score を持つ。"""
    if source_id == target_id:
        raise ValueError(f"自分自身への関係は書きません: {source_id!r}")
    dec = to_decimal(score)
    forward = {"PK": ent_pk(source_id), "SK": out_sk(edge, target_id, chunk_id), "score": dec}
    reverse = {"PK": ent_pk(target_id), "SK": in_sk(edge, source_id, chunk_id), "score": dec}
    return forward, reverse


def sym_items(a_id: str, edge: str, b_id: str, chunk_id: str, score: Any) -> tuple[dict, dict]:
    """対称な関係 1 件ぶんの (a 側の行, b 側の行)。どちらも `SYM#<edge>#ENT#<相手>#<chunk_id>`、同じ score。"""
    if a_id == b_id:
        raise ValueError(f"自分自身への関係は書きません: {a_id!r}")
    dec = to_decimal(score)
    return ({"PK": ent_pk(a_id), "SK": sym_sk(edge, b_id, chunk_id), "score": dec},
            {"PK": ent_pk(b_id), "SK": sym_sk(edge, a_id, chunk_id), "score": dec})


def parse_graph_item(item: Mapping[str, Any]) -> dict:
    """graph テーブルの 1 行を読みやすい形に戻す(SK を分解し、score を float に)。"""
    pk = str(item["PK"])
    sk = str(item["SK"])
    pk_parts = pk.split(SEP)
    if len(pk_parts) != 2 or pk_parts[0] != ENT_PREFIX:
        raise ValueError(f"graph の PK の形が違います: {pk!r}")
    entity_id = pk_parts[1]
    score = item.get("score")
    score_f = None if score is None else float(score)
    parts = sk.split(SEP)
    if len(parts) == 2 and parts[0] == CHUNK_PREFIX:
        return {"kind": "appearance", "entity_id": entity_id, "chunk_id": parts[1], "score": score_f}
    if len(parts) == 5 and parts[2] == ENT_PREFIX and parts[0] in (OUT_PREFIX, IN_PREFIX):
        if parts[0] == OUT_PREFIX:
            return {"kind": "out", "entity_id": entity_id, "edge": parts[1], "target_id": parts[3],
                    "chunk_id": parts[4], "score": score_f}
        return {"kind": "in", "entity_id": entity_id, "edge": parts[1], "source_id": parts[3],
                "chunk_id": parts[4], "score": score_f}
    if len(parts) == 5 and parts[2] == ENT_PREFIX and parts[0] == SYM_PREFIX:
        return {"kind": "sym", "entity_id": entity_id, "edge": parts[1], "other_id": parts[3],
                "chunk_id": parts[4], "score": score_f}
    raise ValueError(f"graph の SK の形が違います: {sk!r}")


# ---------------------------------------------------------- マスター(PK=MASTER)

def nodetype_sk(name: str) -> str:
    return f"{NODETYPE_PREFIX}{SEP}{_part(name, 'node_type')}"


def edge_type_sk(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 10 ** EDGE_INDEX_WIDTH:
        raise ValueError(f"エッジ種別の添字が範囲外です: {index!r}")
    return f"{EDGE_PREFIX}{SEP}{index:0{EDGE_INDEX_WIDTH}d}"


def master_entity_sk(entity_id: str) -> str:
    return ent_pk(entity_id)


def master_items(master: Mapping[str, Any]) -> list[dict]:
    """マスターを graph テーブルの行にする(META は含まない。`master_meta_item()` で別に作る)。
    並びは ノード種別 → エッジ種別 → エンティティ(それぞれマスターの順。`index` に順番を持つ)。"""
    items: list[dict] = []
    for i, nt in enumerate(master["node_types"]):
        items.append({"PK": MASTER_PK, "SK": nodetype_sk(nt["name"]), "index": i,
                      "name": nt["name"], "description": nt.get("description", "")})
    for i, et in enumerate(master["edge_types"]):
        items.append({"PK": MASTER_PK, "SK": edge_type_sk(i), "index": i,
                      "name": et["name"], "template": et["template"],
                      "symmetric": et.get("symmetric") is True,
                      "allowed": [{"source_type": r["source_type"], "target_type": r["target_type"]}
                                  for r in et.get("allowed", [])]})
    for i, ent in enumerate(master["entities"]):
        items.append({"PK": MASTER_PK, "SK": master_entity_sk(ent["id"]), "index": i,
                      "name": ent["name"], "aliases": list(ent.get("aliases") or []),
                      "type": ent["type"], "description": ent.get("description", "")})
    return items


def master_meta_item(*, version: str, entities: int, node_types: int, edge_types: int,
                     updated_at: str, source_sha256: str | None = None) -> dict:
    item = {"PK": MASTER_PK, "SK": MASTER_META_SK, "version": version, "entities": entities,
            "node_types": node_types, "edge_types": edge_types, "updated_at": updated_at}
    if source_sha256:
        item["source_sha256"] = source_sha256
    return item


def _index(item: Mapping[str, Any]) -> int:
    return int(item["index"])


def master_from_items(items: Iterable[Mapping[str, Any]]) -> dict:
    """`PK=MASTER` の行(META 以外)からマスターの dict を組み立て直す(`index` の順に並べる)。
    形の分からない SK があれば ValueError。エッジ種別の `symmetric` は true のときだけキーを付ける。"""
    nts, ets, ents = [], [], []
    for item in items:
        sk = str(item["SK"])
        if sk == MASTER_META_SK:
            continue
        head, _, rest = sk.partition(SEP)
        if head == NODETYPE_PREFIX and rest:
            nts.append(item)
        elif head == EDGE_PREFIX and rest:
            ets.append(item)
        elif head == ENT_PREFIX and rest:
            ents.append(item)
        else:
            raise ValueError(f"マスターの SK の形が違います: {sk!r}")
    return {
        "node_types": [{"name": str(i["name"]), "description": str(i.get("description", ""))}
                       for i in sorted(nts, key=_index)],
        "edge_types": [_edge_type_from_item(i) for i in sorted(ets, key=_index)],
        "entities": [{"id": str(i["SK"]).partition(SEP)[2], "name": str(i["name"]),
                      "aliases": [str(a) for a in (i.get("aliases") or [])], "type": str(i["type"]),
                      "description": str(i.get("description", ""))}
                     for i in sorted(ents, key=_index)],
    }


def _edge_type_from_item(item: Mapping[str, Any]) -> dict:
    edge: dict[str, Any] = {"name": str(item["name"]), "template": str(item["template"])}
    if item.get("symmetric") is True:
        edge["symmetric"] = True
    edge["allowed"] = [{"source_type": str(r["source_type"]), "target_type": str(r["target_type"])}
                       for r in (item.get("allowed") or [])]
    return edge


def master_meta(item: Mapping[str, Any] | None) -> dict | None:
    """META の行を読みやすい形に(数値は int に)。None ならそのまま None。"""
    if not item:
        return None
    out = {"version": str(item["version"]), "updated_at": str(item.get("updated_at", ""))}
    for key in ("entities", "node_types", "edge_types"):
        out[key] = int(item.get(key) or 0)
    if item.get("source_sha256"):
        out["source_sha256"] = str(item["source_sha256"])
    return out


# ============================================================== 読み書き

class GraphStore:
    """2 テーブルをまとめて扱う。`resource` は boto3 の DynamoDB ServiceResource。"""

    def __init__(
        self,
        resource: Any = None,
        *,
        chunk_table: str = CHUNK_TABLE,
        graph_table: str = GRAPH_TABLE,
        sleep: Any = time.sleep,
    ) -> None:
        assert_table(chunk_table, CHUNK_TABLE)
        assert_table(graph_table, GRAPH_TABLE)
        if resource is None:
            import boto3

            resource = boto3.resource("dynamodb", region_name=REGION,
                                      config=boto_config(max_attempts=5))
        self._resource = resource
        self._sleep = sleep
        self.chunk = resource.Table(chunk_table)
        self.graph = resource.Table(graph_table)
        self._check()

    def _check(self) -> None:
        """操作の直前に毎回呼ぶ。Table オブジェクトの実名で確かめる。"""
        assert_table(getattr(self.chunk, "name", None), CHUNK_TABLE)
        assert_table(getattr(self.graph, "name", None), GRAPH_TABLE)

    # ---------------------------------------------------------- 書き込み

    def put_chunks(self, chunks: Iterable[Mapping[str, Any]]) -> int:
        """チャンクを書く。書いた件数を返す。"""
        items = [chunk_item(c) for c in chunks]          # 書き始める前に全部検査する
        self._check()
        with self.chunk.batch_writer(overwrite_by_pkeys=["PK"]) as writer:
            for item in items:
                writer.put_item(Item=item)
        return len(items)

    def put_appearances(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """登場エッジを書く。`rows` の要素は `{entity_id, chunk_id, score}`。書いた件数を返す。"""
        items = [appearance_item(r["entity_id"], r["chunk_id"], r["score"]) for r in rows]
        self._check()
        with self.graph.batch_writer(overwrite_by_pkeys=["PK", "SK"]) as writer:
            for item in items:
                writer.put_item(Item=item)
        return len(items)

    def put_edges(self, edges: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
        """関係を順方向・逆方向の 2 行ずつ書く。`edges` の要素は
        `{source_id, edge, target_id, chunk_id, score}`。戻り値は (順方向の件数, 逆方向の件数)。
        `symmetric: true` の要素が混ざっていたら ValueError(対称な関係は `put_sym_edges()`)。"""
        edges = list(edges)
        sym = [e for e in edges if e.get("symmetric") is True]
        if sym:
            raise ValueError(f"対称な関係は put_sym_edges() で書きます: {sym[0].get('edge')!r}")
        pairs = [edge_items(e["source_id"], e["edge"], e["target_id"], e["chunk_id"], e["score"])
                 for e in edges]
        self._check()
        with self.graph.batch_writer(overwrite_by_pkeys=["PK", "SK"]) as writer:
            for forward, reverse in pairs:
                writer.put_item(Item=forward)
                writer.put_item(Item=reverse)
        return len(pairs), len(pairs)

    def put_sym_edges(self, edges: Iterable[Mapping[str, Any]]) -> int:
        """対称な関係を、両端のノードに `SYM#` の行を 1 行ずつ(計 2 行)書く。`edges` の要素は
        `{source_id, edge, target_id, chunk_id, score}`(source/target は並べ替えた順で、向きの意味は無い)。
        戻り値は関係の件数(書いた行はその 2 倍)。"""
        pairs = [sym_items(e["source_id"], e["edge"], e["target_id"], e["chunk_id"], e["score"])
                 for e in edges]
        self._check()
        with self.graph.batch_writer(overwrite_by_pkeys=["PK", "SK"]) as writer:
            for a_row, b_row in pairs:
                writer.put_item(Item=a_row)
                writer.put_item(Item=b_row)
        return len(pairs)

    # ---------------------------------------------------------- 全削除

    def _scan_keys(self, table: Any, key_names: Sequence[str]) -> list[dict]:
        names = {f"#k{i}": k for i, k in enumerate(key_names)}
        kwargs: dict[str, Any] = {
            "ProjectionExpression": ", ".join(names),
            "ExpressionAttributeNames": names,
        }
        keys: list[dict] = []
        while True:
            resp = table.scan(**kwargs)
            keys.extend({k: item[k] for k in key_names} for item in resp.get("Items", []))
            last = resp.get("LastEvaluatedKey")
            if not last:
                return keys
            kwargs["ExclusiveStartKey"] = last

    def reset(self) -> dict:
        """2 テーブルを Scan して全アイテムを消す。テーブル自体は消さない。"""
        self._check()
        deleted: dict[str, int] = {}
        for label, table, key_names in (("chunk", self.chunk, ("PK",)),
                                        ("graph", self.graph, ("PK", "SK"))):
            # マスター(PK=MASTER)は消さない。Scan の結果から外し、消す直前にも 1 件ずつ確かめる
            keys = [k for k in self._scan_keys(table, key_names) if str(k.get("PK")) != MASTER_PK]
            self._check()
            with table.batch_writer(overwrite_by_pkeys=list(key_names)) as writer:
                for key in keys:
                    writer.delete_item(Key=assert_not_master(key))
            deleted[label] = len(keys)
        return deleted

    # ---------------------------------------------------------- 読み出し

    def count_items(self) -> dict:
        """件数を種類別に数える(Scan。20 番スクリプトの突き合わせ用。データが小さい前提)。"""
        self._check()
        counts = {"chunk": len(self._scan_keys(self.chunk, ("PK",))),
                  "appearance": 0, "out": 0, "in": 0, "sym": 0, "master": 0, "other": 0}
        for key in self._scan_keys(self.graph, ("PK", "SK")):
            if str(key.get("PK")) == MASTER_PK:
                counts["master"] += 1
                continue
            try:
                kind = parse_graph_item(key)["kind"]
            except ValueError:
                kind = "other"
            counts[kind] += 1
        return counts

    def query_node(self, entity_id: str) -> dict:
        """`PK = ENT#<id>` を 1 回の Query(ページ送りあり)で読み、種類別に分けて返す。"""
        self._check()
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": "#pk = :pk",
            "ExpressionAttributeNames": {"#pk": "PK"},
            "ExpressionAttributeValues": {":pk": ent_pk(entity_id)},
        }
        out: dict[str, Any] = {"entity_id": entity_id, "appearances": [], "out": [], "in": [], "sym": []}
        bucket = {"appearance": "appearances", "out": "out", "in": "in", "sym": "sym"}
        while True:
            resp = self.graph.query(**kwargs)
            for item in resp.get("Items", []):
                parsed = parse_graph_item(item)
                out[bucket[parsed["kind"]]].append(parsed)
            last = resp.get("LastEvaluatedKey")
            if not last:
                return out
            kwargs["ExclusiveStartKey"] = last

    def batch_get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, dict]:
        """チャンク本文をまとめて読む。戻り値は `{chunk_id: {"text", "doc_id"}}`、
        並びは `chunk_ids` の順(重複は 1 回)。見つからない id は入らない。"""
        self._check()
        wanted = list(dict.fromkeys(chunk_ids))
        found: dict[str, dict] = {}
        for i in range(0, len(wanted), BATCH_GET_MAX):
            request = {CHUNK_TABLE: {"Keys": [{"PK": chunk_pk(cid)} for cid in wanted[i:i + BATCH_GET_MAX]]}}
            for attempt in range(BATCH_GET_RETRIES + 1):
                self._check()
                resp = self._resource.batch_get_item(RequestItems=request)
                for item in (resp.get("Responses") or {}).get(CHUNK_TABLE, []):
                    cid = str(item["PK"]).split(SEP, 1)[1]
                    found[cid] = {"text": item.get("text"), "doc_id": item.get("doc_id")}
                request = resp.get("UnprocessedKeys") or {}
                if not request:
                    break
                if attempt == BATCH_GET_RETRIES:
                    raise RuntimeError(f"BatchGetItem の未処理キーが残りました: {request}")
                self._sleep(min(2.0, 0.1 * (2 ** attempt)))
        return {cid: found[cid] for cid in wanted if cid in found}

    # ---------------------------------------------------------- マスター(PK=MASTER)

    def get_master_meta(self) -> dict | None:
        """META だけを GetItem(強い整合性)。無ければ None。呼び出しごとの版の確認用。"""
        self._check()
        resp = self.graph.get_item(Key={"PK": MASTER_PK, "SK": MASTER_META_SK}, ConsistentRead=True)
        return master_meta(resp.get("Item"))

    def query_master_items(self) -> list[dict]:
        """`PK=MASTER` を Query(ページ送りあり)。META も含む全行。"""
        self._check()
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": "#pk = :pk",
            "ExpressionAttributeNames": {"#pk": "PK"},
            "ExpressionAttributeValues": {":pk": MASTER_PK},
        }
        items: list[dict] = []
        while True:
            resp = self.graph.query(**kwargs)
            items.extend(dict(i) for i in resp.get("Items", []))
            last = resp.get("LastEvaluatedKey")
            if not last:
                return items
            kwargs["ExclusiveStartKey"] = last

    def write_master(self, master: Mapping[str, Any], meta: Mapping[str, Any]) -> dict:
        """マスターを書き換える。①新しい行を全部 put → ②新しい版に無い行(消えたエンティティなど)を delete
        → ③**最後に META** を put。途中で失敗しても META は旧 version のまま残る。"""
        if meta.get("PK") != MASTER_PK or meta.get("SK") != MASTER_META_SK:
            raise ValueError("meta は master_meta_item() の行です")
        items = master_items(master)                     # 書き始める前に全部検査する
        new_sks = {i["SK"] for i in items}
        stale = [{"PK": MASTER_PK, "SK": i["SK"]} for i in self.query_master_items()
                 if i["SK"] != MASTER_META_SK and i["SK"] not in new_sks]
        self._check()
        with self.graph.batch_writer(overwrite_by_pkeys=["PK", "SK"]) as writer:
            for item in items:
                writer.put_item(Item=item)
        self._check()
        with self.graph.batch_writer(overwrite_by_pkeys=["PK", "SK"]) as writer:
            for key in stale:
                writer.delete_item(Key=key)
        self._check()
        self.graph.put_item(Item=dict(meta))
        return {"put": len(items) + 1, "deleted": len(stale)}
