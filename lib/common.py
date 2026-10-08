#!/usr/bin/env python3
"""lib/ とスクリプトが共有する定数とユーティリティ。

- モデル名・単価・パスの正本はこのファイル(単価は `budget.py` がここだけを見る)
- API キーはローカルでは `.env` から `load_api_key()` でだけ読む(Lambda は `aws_secrets.py` が SSM から読む)。
  値は表示・ログ・例外メッセージ・`os.environ` に出さない
- import した時点ではディレクトリもファイルも作らない(Lambda は読み取り専用 FS)
- boto3 は必要な関数の中で遅延 import する"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

# ============================================================== パス
#
# パスを組み立てるだけで、ここでは作らない(Lambda 上では書けない場所を指していてよい)。

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
ENV_FILE = PROJECT_ROOT / ".env"                        # 中身は load_api_key() 以外が触らない

# ============================================================== データセット(マスターとチャンクの組)
# `data/master_<版>.json` と `data/chunks_<版>.json` を組で使う。同梱は v3(チャンクは v2 を共用)。

DATASETS = ("v1", "v2", "v3")
DEFAULT_DATASET = "v3"
DATASET_ENV = "JEV_DATASET"
# 版 → チャンクの版(マスターだけ変えた版はチャンクを前の版と共用する。書いていない版は同じ名前)
_CHUNKS_OF = {"v3": "v2"}


def dataset_name(value: str | None = None) -> str:
    """版の名前。引数 → 環境変数 `JEV_DATASET` → 既定 v3 の順に決める。知らない版は ValueError。"""
    name = (value or os.environ.get(DATASET_ENV) or DEFAULT_DATASET).strip()
    if name not in DATASETS:
        raise ValueError(f"{DATASET_ENV} は {DATASETS} のどれかです: {name!r}")
    return name


def dataset_paths(name: str | None = None) -> tuple[Path, Path]:
    """`(マスターのパス, チャンクのパス)`。ファイルの有無は確かめない(読む側が確かめる)。"""
    ds = dataset_name(name)
    return DATA_DIR / f"master_{ds}.json", DATA_DIR / f"chunks_{_CHUNKS_OF.get(ds, ds)}.json"


DATASET = dataset_name()

# ============================================================== 環境
# 既定は東京。us-west-2 は query Lambda の写しを置くときだけ使う(環境変数 JEV_REGION)。

HOME_REGION = "ap-northeast-1"
REGIONS: tuple[str, ...] = (HOME_REGION, "us-west-2")
REGION_ENV = "JEV_REGION"


def region_name(value: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    """使うリージョン。引数 → (Lambda の中なら)`AWS_REGION` → 環境変数 `JEV_REGION` → 既定 東京 の順。"""
    env = os.environ if environ is None else environ
    in_lambda = bool(env.get("AWS_LAMBDA_FUNCTION_NAME"))
    name = (value or (env.get("AWS_REGION") if in_lambda else None)
            or env.get(REGION_ENV) or HOME_REGION).strip()
    if name not in REGIONS:
        raise ValueError(f"リージョンは {REGIONS} のどれかです: {name!r}")
    return name


REGION = region_name()
ACCOUNT_ID_ENV = "JEV_AWS_ACCOUNT_ID"
ACCOUNT_ID = (os.environ.get(ACCOUNT_ID_ENV) or "").strip()

JEV_MODEL = "jev-1.13.0"                     # jev-latest は差し替わりうるので版を固定する

EMBED_MODEL = "amazon.titan-embed-text-v2:0"
GEN_MODELS = {
    "sonnet": "global.anthropic.claude-sonnet-5",
    "haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
}
RERANK_MODELS = {
    "cohere": "cohere.rerank-v3-5:0",
    "amazon": "amazon.rerank-v1:0",
}

DEFAULT_GEN_MODEL = GEN_MODELS["haiku"]      # 回答生成

# ============================================================== 単価

# USD。トークン単価は 1M トークンあたり、リランクは 1 クエリ(100 チャンクまで)あたり。
# `budget.py::cost_of` がここだけを見る。ここに無いモデルを呼んだら ValueError になる。
PRICES: dict[str, dict[str, float]] = {
    "jev-1.13.0": {"input": 0.042, "output": 0.0},   # 出力は現時点で無料
    GEN_MODELS["sonnet"]: {"input": 2.00, "output": 10.00},
    GEN_MODELS["haiku"]: {"input": 1.00, "output": 5.00},
    EMBED_MODEL: {"input": 0.02, "output": 0.0},
    RERANK_MODELS["cohere"]: {"per_query": 0.002},
    RERANK_MODELS["amazon"]: {"per_query": 0.001},
}

VECTOR_SEARCH_USD_PER_GB = 0.00228
VECTOR_SEARCH_MIN_BYTES = 1024

KINDS: tuple[str, ...] = ("jev", "gen", "embed", "rerank")

# 失敗は種別ごとのファイルに混ぜず、ここへ分けて置く。費用の集計からも除く(budget.py)。
ERRORS_JSONL_NAME = "errors.jsonl"

BUDGET_LIMIT_USD = 9.0     # 上限 $10 に対する自主規制ライン


def cache_path(kind: str, cache_dir: Path | None = None) -> Path:
    """種別からキャッシュファイルのパスを返す。未知の種別は弾く。"""
    if kind not in KINDS:
        raise ValueError(f"未知のキャッシュ種別です: {kind!r}(有効: {', '.join(KINDS)})")
    return Path(cache_dir or CACHE_DIR) / f"{kind}.jsonl"


def ensure_dirs() -> None:
    """出力先を作る。**ローカルのスクリプトから明示的に呼ぶ**(import 時には呼ばない)。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================== API キー(ローカル専用)

# `.env` の変数名は `apikey` で、SDK 既定の `TYPESAFE_API_KEY` と違う。どちらの書き方でも
# 読めるようにしておく(先に見つかった行を採る)。
_API_KEY_NAMES = ("apikey", "typesafe_api_key")


def _unquote(value: str) -> str:
    """前後の空白を落とし、引用符で囲まれていれば外す。"""
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


def load_api_key(env_path: Path | None = None) -> str:
    """`.env` から API キーを読む。**値は戻り値以外のどこにも出さない**。"""
    path = Path(env_path) if env_path is not None else ENV_FILE
    if not path.exists():
        raise RuntimeError(f"API キーのファイルがありません: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip().lower() not in _API_KEY_NAMES:
                continue
            key = _unquote(value)
            if not key:
                # 値そのものは絶対に書かない。どの行が空だったかだけ伝える。
                raise RuntimeError(f"{path} の {name.strip()} が空です")
            return key

    raise RuntimeError(
        f"{path} に apikey / TYPESAFE_API_KEY の行がありません"
    )


# ============================================================== AWS

def boto_config(*, max_attempts: int = 1, mode: str = "standard", **kwargs: Any) -> Any:
    """botocore の `Config`。**既定は内部再試行なし**(計測用)。"""
    from botocore.config import Config

    return Config(retries={"max_attempts": int(max_attempts), "mode": str(mode)}, **kwargs)


def boto_retry_attempts(response: Any) -> int | None:
    """応答の `ResponseMetadata.RetryAttempts`(= botocore が内部でやり直した回数)。"""
    if not isinstance(response, Mapping):
        return None
    meta = response.get("ResponseMetadata")
    if not isinstance(meta, Mapping):
        return None
    value = meta.get("RetryAttempts")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def check_account(session: Any = None, region: str | None = None) -> str:
    """意図しないアカウントで走らせないためのガード。AWS を触るスクリプトの最初に呼ぶ。"""
    import boto3

    if not ACCOUNT_ID:
        raise RuntimeError(f"環境変数 {ACCOUNT_ID_ENV} に AWS アカウント ID を設定してください")
    region = region or REGION
    client = (session or boto3).client("sts", region_name=region)
    account = client.get_caller_identity()["Account"]
    if account != ACCOUNT_ID:
        raise RuntimeError(
            f"AWS アカウントが違います: expected={ACCOUNT_ID} actual={account}"
        )
    print(f"account: {account} / region: {region} (ok)")
    return account


def require_home_region(label: str = "") -> None:
    """東京(HOME_REGION)専用のスクリプト(登録・マスター)の最初に呼ぶ。`JEV_REGION` が東京以外なら止める。"""
    if REGION != HOME_REGION:
        raise RuntimeError(f"{label + ': ' if label else ''}このスクリプトは {HOME_REGION} 専用です"
                           f"({REGION_ENV}={REGION})")


# ============================================================== JSON / JSONL 入出力

def utc_now_iso() -> str:
    """キャッシュの `ts` に入れる ISO8601(UTC)。"""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    """途中で落ちても壊れたファイルを残さないよう、一時ファイル経由で置き換える。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
        fh.write("\n")
    os.replace(tmp, path)


def iter_jsonl(path: Path) -> Iterator[dict]:
    """JSONL を 1 行ずつ読む。ファイルが無ければ何も返さない。空行は飛ばす。"""
    if not Path(path).exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, record: dict) -> None:
    """JSONL に 1 行だけ追記する(親ディレクトリが無ければ作る)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================== モデル出力の後始末

def strip_code_fence(text: str) -> str:
    """```json ... ``` のコードフェンスを剥がす。"""
    body = (text or "").strip()
    if not body.startswith("```"):
        return body
    lines = body.splitlines()
    # 先頭の ``` / ```json 行を落とす
    lines = lines[1:]
    # 末尾の ``` 行を落とす
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def parse_json_payload(text: str) -> Any:
    """モデル出力を JSON として読む。フェンスを剥がし、前後の余計な地の文も削る。"""
    body = strip_code_fence(text)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    # 前後に説明文が付いた場合に備えて、最初の [ または { から最後の ] または } までを拾う
    starts = [i for i in (body.find("["), body.find("{")) if i >= 0]
    ends = [i for i in (body.rfind("]"), body.rfind("}")) if i >= 0]
    if starts and ends:
        sliced = body[min(starts): max(ends) + 1]
        return json.loads(sliced)
    raise json.JSONDecodeError("JSON として読めませんでした", body, 0)


# ============================================================== AWS 呼び出しの再試行

THROTTLE_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ProvisionedThroughputExceededException",
    "RequestLimitExceeded",
    "ServiceUnavailableException",
    "ServiceUnavailable",
    "ModelTimeoutException",
    "ModelNotReadyException",
}

RETRYABLE_CODES = THROTTLE_CODES | {
    "InternalServerException",
    "InternalServerError",
    "LimitExceededException",
}


def error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return response.get("Error", {}).get("Code", "")
    return ""


def is_throttle(exc: Exception) -> bool:
    """スロットル(待てば直る上限超過)か。**内部再試行を切ったときに表に出てくる**。"""
    return error_code(exc) in THROTTLE_CODES


def _is_transient(exc: Exception) -> bool:
    """ClientError 以外(接続断・読み取りタイムアウト等)の一時障害判定。"""
    name = type(exc).__name__
    return name in {
        "ConnectionError",
        "EndpointConnectionError",
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "IncompleteReadError",
        "ResponseStreamingError",
        "ConnectionClosedError",
    }


def call_with_retry(
    fn: Callable[[], Any],
    what: str = "呼び出し",
    *,
    max_attempts: int = 8,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Any:
    """スロットル・一時障害を指数バックオフ(ジッタ付き)で待ち直す。"""
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            code = error_code(exc)
            retryable = code in RETRYABLE_CODES or (not code and _is_transient(exc))
            last = exc
            if not retryable or attempt == max_attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt)) * (0.5 + random.random())
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            else:
                print(f"  [retry] {what}: {code or type(exc).__name__} -> {delay:.1f}s 待機 "
                      f"({attempt + 1}/{max_attempts - 1})")
            time.sleep(delay)
    raise RuntimeError(f"{what}: リトライ上限に到達しました ({last})")


# ============================================================== ログ

_T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {message}", flush=True)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown の表を組み立てる(results/*.md 用)。"""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    # 設定の目視確認用。AWS も Jev も呼ばない。API キーの値は読まない・出さない
    # (.env の有無だけ表示する)。ディレクトリも作らない。
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"REGION       : {REGION}({REGION_ENV} で切り替え。有効: {', '.join(REGIONS)})/ ACCOUNT_ID: {ACCOUNT_ID}")
    print(f"JEV_MODEL    : {JEV_MODEL}")
    print(f"GEN_MODELS   : {GEN_MODELS}")
    print(f"  既定の生成 : {DEFAULT_GEN_MODEL}")
    print(f"EMBED_MODEL  : {EMBED_MODEL}")
    print(f"RERANK_MODELS: {RERANK_MODELS}")
    print("PRICES       : (USD。トークンは 1M あたり、リランクは 1 クエリあたり)")
    for name, price in PRICES.items():
        print(f"  {name}: {price}")
    print(f"BUDGET_LIMIT_USD: {BUDGET_LIMIT_USD}")
    print(f"DATA_DIR     : {DATA_DIR}")
    print(f"DATASET      : {DATASET}({DATASET_ENV} で切り替え。有効: {', '.join(DATASETS)})")
    for _p in dataset_paths():
        print(f"  {_p.name}{'' if _p.exists() else '(まだ無い)'}")
    print(f"RESULTS_DIR  : {RESULTS_DIR}")
    print(f"CACHE_DIR    : {CACHE_DIR}")
    for kind in KINDS:
        print(f"  {kind:7s} -> {cache_path(kind).name}")
    print(f"  失敗       -> {ERRORS_JSONL_NAME}")
    print(f"ENV_FILE     : {ENV_FILE}({'あり' if ENV_FILE.exists() else 'まだ無い'}。中身は表示しない)")
