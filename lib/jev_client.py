#!/usr/bin/env python3
"""Jev(TypeSafe AI System One)公式 SDK の薄い包み。

- `TypeSafeClient` は 1 つ作って使い回す(接続の確立に時間がかかるため)
- SDK の自動再試行は切り、再試行は自前で数えて記録する(レイテンシの計測を汚さないため)
- 所要時間・usage・再試行回数を 1 レコードにまとめる。API キーやヘッダはレコードに入れない
- レートは `TokenBucket` で抑える(待ち時間は `latency_ms` に含めない)"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from typing import Any, Callable

from pydantic import BaseModel
from typesafe_sdk import (
    Choice,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeClient,
)

from cache import JsonlCache, cached_call, make_key
from common import JEV_MODEL, load_api_key
from ratelimit import TokenBucket

REQUEST_TIMEOUT = 30.0

# 自前再試行の対象にする HTTP ステータス。5xx は下の `_is_retryable` でまとめて拾う。
# 529 は過負荷を表す応答で、5xx なのでここに書かなくても拾えるが、
# 「何を再試行するつもりか」を明示するために並べておく。
RETRY_STATUSES = frozenset({408, 429, 529})

RETRY_BASE_DELAY = 0.5   # 秒。以降 2 倍ずつ(ジッタ付き)
RETRY_MAX_DELAY = 8.0


def _to_question(name: str, question: Any) -> Any:
    """素の dict を SDK の質問オブジェクト(Noul / Choice / Score)に変換する。"""
    if isinstance(question, (Noul, Choice, Score)):
        return question
    if not isinstance(question, Mapping):
        raise TypeError(f"質問 {name!r} は dict か SDK の質問オブジェクトです: {type(question).__name__}")
    data = dict(question)
    qtype = data.pop("type", None)
    if qtype == "noul":
        return Noul(**data)
    if qtype == "choice":
        return Choice(**data)
    if qtype == "score":
        return Score(**data)
    raise ValueError(f"質問 {name!r} の type が不正です: {qtype!r}(noul / choice / score)")


def _jsonable(value: Any) -> Any:
    """レコードに残す用に、pydantic のオブジェクトを素の dict へ落とす。"""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _is_retryable(exc: Exception) -> bool:
    """再試行してよい失敗か。"""
    if isinstance(exc, TypeSafeAPIResponseValidationError):
        # 200 が返ったのに中身が想定外。再送しても同じなので即送出する
        return False
    if isinstance(exc, TypeSafeAPIConnectionError):
        return True
    if isinstance(exc, TypeSafeAPIError):
        status = getattr(exc, "status", 0)
        return status >= 500 or status in RETRY_STATUSES
    return False


def _retry_delay(exc: Exception, attempt: int) -> float:
    """次の再試行までの待ち秒数。サーバが `Retry-After` を返していればそれに従う。"""
    retry_after_ms = getattr(exc, "retry_after_ms", None)   # TypeSafeRateLimitError だけ持つ
    if retry_after_ms:
        return float(retry_after_ms) / 1000.0
    delay = min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * (2 ** attempt))
    return delay * (0.5 + random.random())                  # ジッタ(同時再送の山を崩す)


class JevClient:
    """Jev を 1 回叩いて 1 レコードを返す包み。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = JEV_MODEL,
        cache: JsonlCache | None = None,
        bucket: TokenBucket | None = None,
        transport: Any = None,
        max_retries: int = 3,
        timeout: float = REQUEST_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """`api_key=None` なら `.env` から読む(値はここでしか触らず、属性にも残さない)。"""
        self.model = model
        self.max_retries = max_retries
        self._cache = cache
        self._bucket = bucket
        self._sleep = sleep
        # api_key はローカル変数のまま SDK へ渡す。self には持たせない(repr やダンプで漏らさない)
        self._client = TypeSafeClient(
            api_key=api_key if api_key is not None else load_api_key(),
            model=model,
            retry=RetryPolicy(max_retries=0),   # SDK の自動再試行は切る(再試行は自前で数える)
            timeout=timeout,
            transport=transport,
        )

    def judge(
        self,
        state: str | dict | list,
        questions: dict[str, dict],
        *,
        trial: int = 0,
        use_cache: bool = True,
    ) -> dict:
        """state について questions を判定させ、キャッシュのレコードを返す。"""
        payload = {"state": _jsonable(state), "questions": _jsonable(questions)}
        key = make_key("jev", self.model, payload, trial)
        record, _hit = cached_call(
            self._cache, key, lambda: self._call(state, questions, payload, key), read=use_cache
        )
        return record

    def _call(self, state: Any, questions: Mapping[str, Any], payload: dict, key: str) -> dict:
        """実リクエスト 1 件。再試行を自前で回し、**最後に成功した 1 回**の所要時間を測る。"""
        prepared = {name: _to_question(name, question) for name, question in questions.items()}
        retries = 0
        while True:
            if self._bucket is not None:
                self._bucket.acquire()      # 待ち時間は計測の外(acquire は測り始める前に呼ぶ)
            started = time.perf_counter()
            try:
                response = self._client.system_one(state, prepared)
            except Exception as exc:  # noqa: BLE001
                if retries >= self.max_retries or not _is_retryable(exc):
                    raise
                self._sleep(_retry_delay(exc, retries))
                retries += 1
                continue
            latency_ms = (time.perf_counter() - started) * 1000.0
            return self._record(response, payload, latency_ms, retries, key)

    def _record(
        self, response: Any, payload: dict, latency_ms: float, retries: int, key: str
    ) -> dict:
        """SDK の応答をキャッシュのレコード形へ落とす。"""
        usage = getattr(response, "usage", None)
        answers = {
            name: _jsonable(answer) for name, answer in getattr(response, "answers", {}).items()
        }
        return {
            "key": key,                     # cache=None で呼ばれてもレコード単体で辿れるように
            "kind": "jev",
            "model": self.model,
            "request": payload,             # state と questions だけ。キーもヘッダも入れない
            "response": {
                "model": getattr(response, "model", None),
                "answers": answers,
            },
            "usage": {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            },
            "latency_ms": latency_ms,
            "retries": retries,
        }

    def close(self) -> None:
        """HTTP 接続を閉じる。"""
        self._client.close()

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
