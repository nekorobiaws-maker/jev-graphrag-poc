#!/usr/bin/env python3
"""Bedrock の Converse を 1 回呼んで 1 レコードにする薄い包み。

- 所要時間と usage を測って記録する(レート制限の待ち時間は `latency_ms` に含めない)
- キャッシュにあるものは呼ばない。`use_cache=False` は読みだけ飛ばす
- スロットル・一時障害は `common.call_with_retry` で待ち直し、待っても直らない失敗(権限・モデル ID・クォータ)は
  再試行せずに `self.fatal` を立てる
- botocore の内部再試行回数を `boto_retry_attempts` に残す"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from cache import JsonlCache, make_key
from common import (
    DEFAULT_GEN_MODEL,
    append_jsonl,
    boto_retry_attempts,
    call_with_retry,
    error_code,
    is_throttle,
    log,
    utc_now_iso,
)
from ratelimit import TokenBucket

# ============================================================== 失敗の分類

ERROR_BODY_MAX_CHARS = 2000

# Bedrock の設定の問題。**再試行しない**(IAM もモデルアクセスもこちらでは触らない)
FATAL_BEDROCK_CODES: frozenset[str] = frozenset({
    "AccessDeniedException",        # モデルアクセス未有効 / Cohere のサブスクリプション未購入
    "UnrecognizedClientException",
    "InvalidSignatureException",
    "ExpiredTokenException",
    "ResourceNotFoundException",    # モデル ARN が無い
    "ValidationException",          # モデル ARN の形式・パラメータ不正
    "ServiceQuotaExceededException",
})

FATAL_EXC_TYPES: frozenset[str] = frozenset({
    "NoCredentialsError", "PartialCredentialsError", "ProfileNotFound",
    "NoRegionError", "TokenRetrievalError", "SSOTokenLoadError",
    "UnauthorizedSSOTokenError", "EndpointConnectionError",
})


class FatalBedrockError(RuntimeError):
    """待っても直らない Bedrock の失敗(権限・モデルアクセス・モデル ID)。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = (message or "")[:ERROR_BODY_MAX_CHARS]
        super().__init__(f"{code}: {self.message}")


def error_message(exc: Exception) -> str:
    """例外から人に見せる文だけを取り出す(認証情報は入らない)。"""
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        message = response.get("Error", {}).get("Message")
        if message:
            return str(message)
    return str(exc)


def is_fatal(exc: Exception) -> bool:
    """待っても直らない失敗か。"""
    return error_code(exc) in FATAL_BEDROCK_CODES or type(exc).__name__ in FATAL_EXC_TYPES


def error_detail(exc: Exception) -> dict:
    """記録に残す失敗の中身。**API キーもヘッダも入らない**。"""
    return {
        "type": type(exc).__name__,
        "code": error_code(exc) or None,
        "message": error_message(exc)[:ERROR_BODY_MAX_CHARS],
    }


# ============================================================== 要求と応答


def converse_payload(system: str, user: str, *, temperature: float, max_tokens: int) -> dict:
    """キャッシュキーのもと。**送る中身そのもの**(文面が変われば別のキーになる)。"""
    return {"system": str(system), "user": str(user),
            "temperature": float(temperature), "max_tokens": int(max_tokens)}


def converse_key(model: str, payload: Mapping[str, Any], *, kind: str = "gen",
                 trial: int = 0) -> str:
    return make_key(kind, model, dict(payload), trial)


def converse_text(response: Mapping[str, Any]) -> str:
    """`converse` の応答からテキストだけを取り出す。"""
    content = (response.get("output") or {}).get("message", {}).get("content") or []
    return "".join(str(block.get("text") or "") for block in content if isinstance(block, Mapping))


# ============================================================== 呼び出し


class ConverseCaller:
    """Converse 1 コールぶんの呼び出しとキャッシュ。"""

    def __init__(self, *, client: Any, model: str = DEFAULT_GEN_MODEL,
                 kind: str = "gen", cache: JsonlCache | None = None,
                 errors_path: Path | None = None, bucket: TokenBucket | None = None,
                 label: str = "gen", retry: Mapping[str, Any] | None = None) -> None:
        self.retry = dict(retry or {})
        self.client = client
        self.model = model
        self.kind = kind
        self.cache = cache
        self.errors_path = errors_path
        self.bucket = bucket
        self.label = label

        self.n_called = 0
        self.n_cached = 0
        self.n_throttled = 0                # 表に出たスロットルを待ち直した回数
        self.failures: list[dict] = []
        self.fatal: dict | None = None      # 立ったらこのモデルは以降呼ばない

    def run(self, task: Mapping[str, Any], *, use_cache: bool = True) -> dict | None:
        """1 コール。キャッシュにあれば呼ばない。飛ばした・失敗したときは None。"""
        if self.fatal is not None:
            return None
        if self.cache is not None and use_cache:
            hit = self.cache.get(task["key"])
            if hit is not None:
                self.n_cached += 1
                return hit
        return self._call(task)

    def _call(self, task: Mapping[str, Any]) -> dict | None:
        # 待ち時間は計測の外(測り始める前に待ち切る)
        bucket_wait_ms = (self.bucket.acquire() * 1000.0) if self.bucket is not None else 0.0
        payload = dict(task["payload"])
        # 再試行の待ち時間を `latency_ms` に混ぜない。**最後に成功した 1 回**だけを測る
        # (`JevClient._call` と同じ流儀)。再試行の回数はレコードに残す
        clock = {"started": time.perf_counter(), "retries": 0}

        def invoke() -> Any:
            clock["started"] = time.perf_counter()
            return self.client.converse(
                modelId=self.model,
                system=[{"text": task["system"]}],
                messages=[{"role": "user", "content": [{"text": task["user"]}]}],
                inferenceConfig={"temperature": payload["temperature"],
                                 "maxTokens": payload["max_tokens"]},
            )

        def on_retry(attempt: int, exc: Exception) -> None:
            clock["retries"] = attempt
            if is_throttle(exc):
                self.n_throttled += 1
            log(f"  [retry] {self.label}: {type(exc).__name__} — {attempt} 回目")

        try:
            response = call_with_retry(
                invoke,
                what=f"{self.label}({task.get('extra', {}).get('query_id', '?')})",
                on_retry=on_retry,
                **self.retry,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            detail = error_detail(exc)
            self._record_error(task, detail)
            if is_fatal(exc):
                # **この手法だけ飛ばして続行**(再試行しない。IAM もモデルアクセスも触らない)
                self.fatal = detail
                log(f"  [{self.label}] {self.model} を飛ばします: "
                    f"{detail['code'] or detail['type']} — {detail['message']}")
            else:
                log(f"  [{self.label}] 失敗: {detail['code'] or detail['type']}")
            return None

        latency_ms = (time.perf_counter() - clock["started"]) * 1000.0
        self.n_called += 1
        usage = response.get("usage") or {}
        record = {
            "key": task["key"],
            "kind": self.kind,
            "model": self.model,
            "request": {**payload, **{k: v for k, v in (task.get("extra") or {}).items()
                                      if k in ("query_id", "split", "method", "n_docs")}},
            "response": {
                "text": converse_text(response),
                "stop_reason": response.get("stopReason"),
            },
            "usage": {
                "input_tokens": int(usage.get("inputTokens") or 0),
                "output_tokens": int(usage.get("outputTokens") or 0),
            },
            "latency_ms": latency_ms,
            "retries": int(clock["retries"]),
            # botocore が内部でやり直した回数(None = 不明)。0 より大きい行は
            # 再試行の待ちが latency_ms に混ざっているので、集計から外す
            "boto_retry_attempts": boto_retry_attempts(response),
            "bucket_wait_ms": bucket_wait_ms,       # 計測には含めていない待ち
        }
        if task.get("extra"):
            record["extra"] = dict(task["extra"])
        if self.cache is not None:
            return self.cache.put(task["key"], record)      # 課金されたので必ず 1 行残す
        return record

    def _record_error(self, task: Mapping[str, Any], detail: Mapping[str, Any]) -> None:
        extra = dict(task.get("extra") or {})
        self.failures.append({"extra": extra, "error": dict(detail)})
        if self.errors_path is not None:
            append_jsonl(self.errors_path, {
                "kind": self.kind, "model": self.model, "error": dict(detail),
                "extra": extra, "ts": utc_now_iso(),
            })
