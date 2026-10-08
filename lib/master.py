#!/usr/bin/env python3
"""マスター(`data/master_<版>.json`)とチャンク(`data/chunks_<版>.json`)の読み込みと、引き方の小道具。

既定のパスは `common.dataset_paths()`(環境変数 `JEV_DATASET`)。Lambda は zip の `master.json` / `chunks.json` を渡す。

- エッジ種別に `"symmetric": true` を持たせると向きのない関係になる
- マスターの正本は DynamoDB(`jev-graphrag-graph` の `PK=MASTER`)。`scripts/60_put_master.py` で書けば、
  Lambda をデプロイし直さずに反映される
- `master_version(master)`: 内容の正規化 JSON の SHA-256 = 版
- `MasterCache`: 呼び出しごとに META だけ読み、版が変わっていたら読み直す。MASTER が無ければ同梱の master.json"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from common import DATASET, dataset_paths

MASTER_PATH, CHUNKS_PATH = dataset_paths(DATASET)

_ENTITY_KEYS = ("id", "name", "aliases", "type", "description")


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_master(path: Path | str | None = None) -> dict:
    """マスターを読む。`node_types` / `edge_types` / `entities` の3つが揃っているかと、
    エンティティの必須キー・id の重複・種別の未定義だけ確かめる。"""
    return validate_master(_read_json(Path(path) if path is not None else MASTER_PATH))


def validate_master(master: Any) -> dict:
    """`load_master()` と同じ確認を dict にかける(DynamoDB から組み立てたマスター用)。"""
    if not isinstance(master, dict):
        raise ValueError("マスターは JSON オブジェクトです")
    for key in ("node_types", "edge_types", "entities"):
        if not isinstance(master.get(key), list):
            raise ValueError(f"マスターに {key} の配列がありません")

    type_names = {t["name"] for t in master["node_types"]}
    seen: set[str] = set()
    for entity in master["entities"]:
        missing = [k for k in _ENTITY_KEYS if k not in entity]
        if missing:
            raise ValueError(f"エンティティ {entity.get('id')!r} に {missing} がありません")
        if entity["id"] in seen:
            raise ValueError(f"エンティティの id が重複しています: {entity['id']!r}")
        if entity["type"] not in type_names:
            raise ValueError(f"エンティティ {entity['id']!r} の種別 {entity['type']!r} が node_types にありません")
        seen.add(entity["id"])
    for i, edge in enumerate(master["edge_types"]):
        _validate_edge_type(i, edge)
    return master


def _validate_edge_type(index: int, edge: Any) -> None:
    """エッジ種別 1 件の確認。`symmetric` は bool だけ。symmetric なら allowed が向きの入れ替えで閉じているか。"""
    if not isinstance(edge, dict) or not edge.get("name"):
        raise ValueError(f"エッジ種別 [{index}] に name がありません")
    sym = edge.get("symmetric", False)
    if not isinstance(sym, bool):
        raise ValueError(f"エッジ種別 [{index}] {edge['name']!r} の symmetric は true/false です: {sym!r}")
    if not sym:
        return
    pairs = {(r.get("source_type"), r.get("target_type")) for r in edge.get("allowed", [])}
    missing = sorted((t, s) for s, t in pairs if (t, s) not in pairs)
    if missing:
        shown = ", ".join(f"{a}→{b}" for a, b in missing)
        raise ValueError(
            f"対称な関係 [{index}] {edge['name']!r} の allowed に逆向きの組がありません: {shown}"
            "(対称な関係は source_type と target_type を入れ替えた組も許可する必要があります)"
        )


def is_symmetric(edge: Mapping[str, Any]) -> bool:
    """対称な(向きのない)関係か。`symmetric` が true のときだけ True。"""
    return edge.get("symmetric") is True


def symmetric_edge_names(master: Mapping[str, Any]) -> frozenset[str]:
    """対称な関係のエッジ種別名の集合。"""
    return frozenset(e["name"] for e in master["edge_types"] if is_symmetric(e))


def normalize_edge_type(edge: Mapping[str, Any]) -> dict:
    """比較・版の計算用。`symmetric` は true のときだけ残す(false と省略を同じに扱う。
    DynamoDB から組み立て直したマスターは true のときだけ持つので、版がずれないように)。"""
    out = {k: v for k, v in edge.items() if k != "symmetric"}
    if is_symmetric(edge):
        out["symmetric"] = True
    return out


def load_chunks(path: Path | str | None = None) -> list[dict]:
    """チャンクを読む。`chunk_id` と `text` を持ち、chunk_id が一意であることだけ確かめる。
    **分割はしない**。"""
    chunks = _read_json(Path(path) if path is not None else CHUNKS_PATH)
    if not isinstance(chunks, list):
        raise ValueError("チャンクのファイルはチャンクの配列です")
    seen: set[str] = set()
    for chunk in chunks:
        if not chunk.get("chunk_id") or not isinstance(chunk.get("text"), str):
            raise ValueError(f"chunk_id と text が必要です: {chunk!r:.80}")
        if chunk["chunk_id"] in seen:
            raise ValueError(f"chunk_id が重複しています: {chunk['chunk_id']!r}")
        seen.add(chunk["chunk_id"])
    return chunks


def file_sha256(path: Path | str) -> str:
    """ファイルの中身の SHA-256(16 進)。Lambda の zip に入ったデータと手元のデータが同じ版かを突き合わせる。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_fingerprint(master_path: Path | str | None = None,
                     chunks_path: Path | str | None = None) -> dict:
    """`{master_sha256, chunks_sha256}`。省略時は既定の版のパス。"""
    return {
        "master_sha256": file_sha256(master_path if master_path is not None else MASTER_PATH),
        "chunks_sha256": file_sha256(chunks_path if chunks_path is not None else CHUNKS_PATH),
    }


def entity_by_id(master: Mapping[str, Any]) -> dict[str, dict]:
    """`{entity_id: entity}`。マスターの並び順を保つ。"""
    return {e["id"]: e for e in master["entities"]}


def allowed_edges(master: Mapping[str, Any], src_type: str, tgt_type: str) -> list[dict]:
    """種別ペア (src_type → tgt_type) で許可されたエッジ種別を、マスターの並び順で返す。"""
    return [
        edge
        for edge in master["edge_types"]
        if any(
            rule.get("source_type") == src_type and rule.get("target_type") == tgt_type
            for rule in edge.get("allowed", [])
        )
    ]


def render(template: str, a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    """候補文テンプレートの `{source}` を a、`{target}` を b の**正式名**で埋める。"""
    return template.replace("{source}", str(a["name"])).replace("{target}", str(b["name"]))


# ============================================================== 版と DynamoDB

MASTER_SOURCE_DYNAMODB = "dynamodb"
MASTER_SOURCE_BUNDLED = "bundled"


def master_version(master: Mapping[str, Any]) -> str:
    """マスターの版 = 中身(node_types / edge_types / entities)を正規化した JSON の SHA-256。
    キーの順・空白・ファイルの改行に左右されない(DynamoDB から組み立て直しても同じ値になる)。
    並び順は意味を持つ(エッジ種別の添字は qid に使われる)ので、配列の順はそのまま。
    エッジ種別の `symmetric: false` は省略と同じ扱い(`normalize_edge_type()`)。"""
    body = {k: master[k] for k in ("node_types", "entities")}
    body["edge_types"] = [normalize_edge_type(e) for e in master["edge_types"]]
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MasterInconsistent(RuntimeError):
    """DynamoDB の MASTER の行と META の version が合わない(書き込み途中・途中で失敗した)。"""


def load_master_from_store(store: Any) -> tuple[dict, dict] | None:
    """DynamoDB(`store.query_master_items()`)からマスターを読む。MASTER が無ければ None。"""
    from graph_store import MASTER_META_SK, master_from_items, master_meta

    items = store.query_master_items()
    if not items:
        return None
    meta_item = next((i for i in items if i.get("SK") == MASTER_META_SK), None)
    meta = master_meta(meta_item) if meta_item is not None else None
    if meta is None:
        raise MasterInconsistent("MASTER の行はあるのに META がありません(書き込み途中か、途中で失敗しました)")
    master = validate_master(master_from_items(items))
    got = master_version(master)
    if got != meta["version"]:
        raise MasterInconsistent(f"MASTER の中身({got[:8]})が META の version({meta['version'][:8]})と違います")
    return master, meta


class MasterCache:
    """Lambda 用のマスターの読み込み(モジュール変数に 1 つ置いて使い回す)。"""

    def __init__(self, bundled_path: Path | str | None = None,
                 log: Any = None) -> None:
        self.bundled_path = bundled_path
        self.log = log or (lambda msg: print(msg, flush=True))
        self._master: dict | None = None
        self._info: dict | None = None
        self._bundled: tuple[dict, dict] | None = None

    def bundled(self) -> tuple[dict, dict]:
        if self._bundled is None:
            master = load_master(self.bundled_path)
            self._bundled = (master, {"source": MASTER_SOURCE_BUNDLED, "version": master_version(master),
                                      "entities": len(master["entities"])})
        return self._bundled

    def _use(self, master: dict, info: dict) -> tuple[dict, dict]:
        self._master, self._info = master, info
        return master, dict(info)

    def get(self, store: Any) -> tuple[dict, dict]:
        meta = store.get_master_meta()
        if meta is None:
            return self._use(*self.bundled())
        if (self._info is not None and self._info["source"] == MASTER_SOURCE_DYNAMODB
                and self._info["version"] == meta["version"]):
            return self._master, dict(self._info)
        try:
            loaded = load_master_from_store(store)
        except MasterInconsistent as exc:
            self.log(json.dumps({"kind": "master_inconsistent", "message": str(exc)}, ensure_ascii=False))
            if self._master is not None:
                return self._master, dict(self._info)
            return self._use(*self.bundled())
        if loaded is None:
            return self._use(*self.bundled())
        master, meta = loaded
        return self._use(master, {"source": MASTER_SOURCE_DYNAMODB, "version": meta["version"],
                                  "entities": len(master["entities"])})
