#!/usr/bin/env python3
"""レート制限(トークンバケット)と、入力順を保つ並列実行。

Jev の上限(1,200 リクエスト/分)に対し、既定は 1,000 リクエスト/分に抑える。"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_RATE_PER_MIN = 1000.0
DEFAULT_MAX_WORKERS = 20


class TokenBucket:
    """1 分あたり `rate_per_min` 個までに均すトークンバケット。"""

    def __init__(
        self,
        rate_per_min: float = DEFAULT_RATE_PER_MIN,
        burst: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_min <= 0:
            raise ValueError(f"rate_per_min は正の数です: {rate_per_min}")
        self.rate_per_min = float(rate_per_min)
        self.rate_per_sec = self.rate_per_min / 60.0
        self.burst = float(math.ceil(self.rate_per_sec) if burst is None else burst)
        if self.burst <= 0:
            raise ValueError(f"burst は正の数です: {self.burst}")
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens = self.burst          # 開始時は満タン
        self._updated = clock()

    def acquire(self, n: float = 1) -> float:
        """トークンが貯まるまで待ち、待った秒数を返す。"""
        if n <= 0:
            raise ValueError(f"n は正の数です: {n}")
        if n > self.burst:
            raise ValueError(f"n={n} は burst={self.burst} を超えています")

        with self._lock:
            now = self._clock()
            self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.rate_per_sec)
            self._updated = now
            if self._tokens >= n:
                self._tokens -= n
                return 0.0
            # 足りないぶんが貯まるまでの秒数。眠っている間に貯まる前提で時計を進めるので、
            # `sleep` が実時間を消費しない偽物でも辻褄が合う。
            waited = (n - self._tokens) / self.rate_per_sec
            self._sleep(waited)
            self._tokens = 0.0
            self._updated = now + waited
            return waited


def run_parallel(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    bucket: TokenBucket | None = None,
) -> list[R | Exception]:
    """`items` を並列に処理し、**入力順のまま**結果を返す。"""
    if not items:
        return []

    def guarded(item: T) -> R:
        if bucket is not None:
            bucket.acquire()
        return fn(item)

    results: list[R | Exception] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(items)))) as pool:
        futures = [pool.submit(guarded, item) for item in items]
        for future in futures:                  # submit した順に回収するので入力順が保たれる
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                results.append(exc)
    return results
