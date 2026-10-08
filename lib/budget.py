#!/usr/bin/env python3
"""予算の見張り。

キャッシュに残った実測 usage × 単価(`common.PRICES`)で累計費用を出し、上限を超える呼び出しの手前で止める。"""

from __future__ import annotations

import math
from pathlib import Path

from common import BUDGET_LIMIT_USD, CACHE_DIR, ERRORS_JSONL_NAME, PRICES, iter_jsonl

RERANK_CHUNKS_PER_QUERY = 100


class BudgetExceeded(RuntimeError):
    """このまま呼ぶと上限を超える、という停止シグナル。"""


def cost_of(record: dict) -> float:
    """キャッシュの 1 レコードの費用(USD)を出す。"""
    kind = record.get("kind")
    if kind == "search":
        return 0.0

    model = record.get("model")
    prices = PRICES.get(model) if isinstance(model, str) else None
    if prices is None:
        raise ValueError(f"単価が未登録のモデルです: {model!r}(common.PRICES に追記してください)")

    if kind == "rerank":
        if "per_query" not in prices:
            raise ValueError(f"リランクの単価(per_query)が未登録のモデルです: {model!r}")
        n_docs = record.get("n_docs")
        # n_docs が無ければ 1 クエリ扱い(リランクは 100 チャンクごとに 1 クエリぶん課金)
        queries = 1 if n_docs is None else math.ceil(float(n_docs) / RERANK_CHUNKS_PER_QUERY)
        return queries * float(prices["per_query"])

    if "input" not in prices:
        raise ValueError(f"トークン単価が未登録のモデルです: {model!r}")
    usage = record.get("usage") or {}
    input_tokens = float(usage.get("input_tokens") or 0)
    output_tokens = float(usage.get("output_tokens") or 0)
    return (
        input_tokens * float(prices["input"]) / 1e6
        + output_tokens * float(prices.get("output", 0.0)) / 1e6
    )


class Budget:
    """`results/cache/*.jsonl` を数えて累計費用を出し、上限の手前で止める。"""

    def __init__(self, cache_dir: Path = CACHE_DIR, limit_usd: float = BUDGET_LIMIT_USD) -> None:
        self.cache_dir = Path(cache_dir)
        self.limit_usd = float(limit_usd)

    def _files(self) -> list[Path]:
        """集計対象のキャッシュファイル。`errors.jsonl` は課金されていないので除く。"""
        if not self.cache_dir.exists():
            return []
        return sorted(
            p for p in self.cache_dir.glob("*.jsonl") if p.name != ERRORS_JSONL_NAME
        )

    def spent(self) -> float:
        """これまでに使った額(USD)。"""
        return sum(
            cost_of(record) for path in self._files() for record in iter_jsonl(path)
        )

    def by_kind(self) -> dict[str, float]:
        """種別ごとの内訳(どこに金がかかっているかを見るため)。"""
        totals: dict[str, float] = {}
        for path in self._files():
            for record in iter_jsonl(path):
                kind = str(record.get("kind", "unknown"))
                totals[kind] = totals.get(kind, 0.0) + cost_of(record)
        return totals

    def check(self, planned_usd: float, label: str) -> None:
        """これから `planned_usd` 使うと上限を超えるなら BudgetExceeded で止める。"""
        spent = self.spent()
        if spent + planned_usd > self.limit_usd:
            raise BudgetExceeded(
                f"[{label}] 予算の上限を超えます: 累計 ${spent:.4f} + 概算 ${planned_usd:.4f} "
                f"> 上限 ${self.limit_usd:.2f}"
            )

    def announce(self, n_calls: int, est_usd: float, label: str) -> str:
        """「これから何コール、概算いくら、累計いくら」を表示しつつ、同じ文字列を返す。"""
        spent = self.spent()
        message = (
            f"[{label}] これから {n_calls} コール、概算 ${est_usd:.4f}、"
            f"累計 ${spent:.4f}/上限 ${self.limit_usd:.2f}"
        )
        print(message, flush=True)
        return message
