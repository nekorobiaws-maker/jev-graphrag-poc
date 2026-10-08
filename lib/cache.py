#!/usr/bin/env python3
"""API 応答のキャッシュ(追記専用の JSONL)。

- キーは入力から決まる(同じ入力の再実行は再課金しない)
- 追記のみ。同じキーを put し直しても過去の行は消さない(`budget.py` が課金回数を数えるため)
- 読み出しは後の行が勝つ"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

from common import append_jsonl, utc_now_iso


def make_key(kind: str, model: str, payload: Any, trial: int = 0) -> str:
    """キャッシュキー(sha256 の 16 進 64 文字)を作る。"""
    canonical = json.dumps(
        {"kind": kind, "model": model, "payload": payload, "trial": trial},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class JsonlCache:
    """1 つの JSONL ファイルを、キー引きできる追記専用のキャッシュとして扱う。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._index: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """既存ファイルを読んで索引を作る。壊れた行は飛ばす(途中で止まった行があっても動く)。"""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                key = record.get("key")
                if isinstance(key, str) and key:
                    self._index[key] = record   # 同じキーが複数行あれば後の行が勝つ

    def get(self, key: str) -> dict | None:
        with self._lock:
            record = self._index.get(key)
            return dict(record) if record is not None else None

    def put(self, key: str, record: dict) -> dict:
        """レコードに `key` と `ts` を補って 1 行追記し、そのレコードを返す。"""
        stored = dict(record)
        stored["key"] = key
        # 既に ts があるレコードはその時刻を残す(再投入しても「いつ課金されたか」を保つ)
        stored.setdefault("ts", utc_now_iso())
        with self._lock:
            append_jsonl(self.path, stored)     # open→write→close で即フラッシュされる
            self._index[key] = stored
        return dict(stored)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._index

    def __len__(self) -> int:
        """一意なキーの数(ファイルの行数ではない)。"""
        with self._lock:
            return len(self._index)

    def records(self) -> Iterator[dict]:
        """一意なキーごとに最新の 1 件を返す(索引に影響しないよう、写しを返す)。"""
        with self._lock:
            snapshot = [dict(record) for record in self._index.values()]
        return iter(snapshot)


def cached_call(
    cache: JsonlCache | None,
    key: str,
    fn: Callable[[], dict],
    *,
    read: bool = True,
) -> tuple[dict, bool]:
    """キャッシュを引いてから `fn()` を呼ぶ。戻り値は (レコード, ヒットしたか)。"""
    if cache is None:
        return fn(), False
    if read:
        hit = cache.get(key)
        if hit is not None:
            return hit, True
    record = fn()
    return cache.put(key, record), False
