#!/usr/bin/env python3
"""Lambda 用の API キー読み出し。

SSM SecureString を 1 度だけ読み、モジュール変数にキャッシュする。値は戻り値以外に出さない。
読む先のリージョンは環境変数 `SECRET_REGION`、無ければ `common.HOME_REGION`。
(ファイル名を `secrets.py` にしないのは標準ライブラリと衝突するため)"""

from __future__ import annotations

import os
import threading
from typing import Any

from common import HOME_REGION, REGIONS, boto_config, error_code

SECRET_PARAM = "/jev-graphrag-poc/typesafe-api-key"
SECRET_REGION_ENV = "SECRET_REGION"


def secret_region() -> str:
    """SSM を読むリージョン。環境変数 `SECRET_REGION` → 東京。知らないリージョンは SecretUnavailable。"""
    name = (os.environ.get(SECRET_REGION_ENV) or HOME_REGION).strip()
    if name not in REGIONS:
        raise SecretUnavailable(f"{SECRET_REGION_ENV} は {REGIONS} のどれかです: {name!r}")
    return name

_lock = threading.Lock()
_api_key: str | None = None


class SecretUnavailable(RuntimeError):
    """SSM から API キーを読めなかった(値はメッセージに含めない)。"""


def get_api_key(client: Any = None) -> str:
    """API キーを返す。初回だけ SSM を呼び、以後はキャッシュを返す。"""
    global _api_key
    with _lock:
        if _api_key is not None:
            return _api_key
        if client is None:
            import boto3

            client = boto3.client("ssm", region_name=secret_region(), config=boto_config(max_attempts=3))
        try:
            resp = client.get_parameter(Name=SECRET_PARAM, WithDecryption=True)
            value = resp["Parameter"]["Value"]
        except Exception as exc:  # noqa: BLE001
            raise SecretUnavailable(
                f"{SECRET_PARAM} を読めませんでした: {type(exc).__name__} / "
                f"{error_code(exc) or '(コード無し)'}"
            ) from None
        if not isinstance(value, str) or not value.strip():
            raise SecretUnavailable(f"{SECRET_PARAM} の値が空です")
        _api_key = value.strip()
        return _api_key


def _clear_cache() -> None:
    """キャッシュを捨てる。"""
    global _api_key
    with _lock:
        _api_key = None
