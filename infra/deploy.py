#!/usr/bin/env python3
"""AWS リソースを作る・見る・消す。

    ./.venv/bin/python infra/deploy.py create-tables    # DynamoDB 2 本(既にあればスキップ)
    ./.venv/bin/python infra/deploy.py put-secret       # .env の API キーを SSM SecureString に登録
    ./.venv/bin/python infra/deploy.py deploy-lambdas --only ingest   # build/ingest.zip を配置
    ./.venv/bin/python infra/deploy.py deploy-lambdas --only query    # build/query.zip を配置
    ./.venv/bin/python infra/deploy.py create-api       # API Gateway(REST)+ API キー + 使用量プラン
    ./.venv/bin/python infra/deploy.py status           # 全リソースの有無を一覧(読むだけ)
    ./.venv/bin/python infra/deploy.py teardown         # 一覧を出して yes で削除

- 最初に `common.check_account()` で AWS アカウントを確かめる(環境変数 `JEV_AWS_ACCOUNT_ID`)
- リソース名はこのファイルの決め打ちだけ。削除は許可リスト(`DELETABLE`)にある名前しか通さない
- IAM ロールは作らない・消さない(status で読むだけ)
- 既にある関数はコードだけ更新し、設定は変えない(query の `AWS_DATA_PATH` だけは無ければ足す)
- API キーの値は表示しない。create-api は URL とキーを `.api.local.json`(権限 600)に書く
- `--region` で東京以外(us-west-2)に query Lambda とテーブルだけを置ける(SSM と API Gateway は東京だけ)
- us-west-2 は計測用。東京のデータを写すスクリプトは同梱していない(テーブルは空で作られる)
- ap-northeast-3(大阪)は CDK(`infra/cdk/`)で一式を置く。このスクリプトは大阪では status(読むだけ)しか
  受け付けない。消すのは `npx aws-cdk@2 destroy -c region=ap-northeast-3`"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

LIB_DIR = Path(__file__).resolve().parent.parent / "lib"
_lib = str(LIB_DIR)
if _lib in sys.path:
    sys.path.remove(_lib)
sys.path.insert(0, _lib)   # 同名モジュールが他所にあっても lib/ を優先させる

from common import (  # noqa: E402
    ACCOUNT_ID,
    HOME_REGION,
    REGION,
    REGIONS,
    boto_config,
    check_account,
    error_code,
    load_api_key,
)

# ============================================================== 決め打ちのリソース名

PROJECT_TAG = {"Key": "project", "Value": "jev-graphrag-poc"}

CHUNK_TABLE = "jev-graphrag-chunk"
GRAPH_TABLE = "jev-graphrag-graph"

TEARDOWN_TABLES: tuple[str, ...] = (CHUNK_TABLE, GRAPH_TABLE)

TABLE_KEYS: dict[str, tuple[str, ...]] = {
    CHUNK_TABLE: ("PK",),
    GRAPH_TABLE: ("PK", "SK"),
}

SECRET_PARAM = "/jev-graphrag-poc/typesafe-api-key"

LAMBDA_ROLE = "jev-graphrag-poc-lambda-role"      # status で読むだけ。消さない
INGEST_FUNCTION = "jev-graphrag-ingest"
QUERY_FUNCTION = "jev-graphrag-query"
FUNCTIONS: tuple[str, ...] = (INGEST_FUNCTION, QUERY_FUNCTION)
LOG_GROUPS: tuple[str, ...] = tuple(f"/aws/lambda/{name}" for name in FUNCTIONS)
LOG_GROUP_PREFIX = "/aws/lambda/jev-graphrag-"
LOG_RETENTION_DAYS = 7

LAMBDA_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/{LAMBDA_ROLE}"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"

# API Gateway(REST API。API キーと使用量プランを使うため HTTP API ではなく REST)
API_NAME = "jev-graphrag-api"
API_STAGE = "poc"
API_PATH_PART = "query"
API_METHOD = "POST"
API_KEY_NAME = "jev-graphrag-poc-key"
USAGE_PLAN_NAME = "jev-graphrag-poc-plan"
USAGE_THROTTLE = {"rateLimit": 1.0, "burstLimit": 2}
USAGE_QUOTA = {"limit": 200, "period": "DAY"}
INTEGRATION_TIMEOUT_MS = 29000            # REST API の上限(既定値)。Lambda 側は 25 秒で返す
PERMISSION_SID = "jev-graphrag-api-invoke"
PERMISSION_NAME = f"{QUERY_FUNCTION}:{PERMISSION_SID}"   # 許可リストでの呼び名
API_LOCAL_PATH = PROJECT_ROOT / ".api.local.json"          # URL とキー。権限 600・git 除外

# deploy-lambdas で配置する関数(--only の選択肢もここから作る)
LAMBDA_SPECS: dict[str, dict[str, Any]] = {
    "ingest": {
        "FunctionName": INGEST_FUNCTION,
        "Runtime": "python3.13",
        "Architectures": ["arm64"],
        "MemorySize": 1024,
        "Timeout": 900,
        "Handler": "handler.lambda_handler",
        "Role": LAMBDA_ROLE_ARN,
        "Description": "jev-graphrag-poc ingest (Jev node/edge judgement -> DynamoDB)",
        "zip": "ingest.zip",
    },
    "query": {
        "FunctionName": QUERY_FUNCTION,
        "Runtime": "python3.13",
        "Architectures": ["arm64"],
        "MemorySize": 1024,
        "Timeout": 120,
        "Handler": "handler.lambda_handler",
        "Role": LAMBDA_ROLE_ARN,
        "Description": "jev-graphrag-poc query (Jev adaptive hop -> Bedrock answer)",
        # zip に同梱した DynamoDB の API 定義(infra/build.sh)を、ランタイム同梱の botocore より先に読ませる。
        # 検証時と同じ zip にするため残している(本手法は SearchVectors を使わない)
        "Environment": {"Variables": {"AWS_DATA_PATH": "/var/task/botocore_data"}},
        "zip": "query.zip",
    },
}

# 東京以外のリージョンに置く関数(本手法の query の写しだけ)と、足す環境変数(東京の SSM を読ませる)
REMOTE_LAMBDAS: tuple[str, ...] = ("query",)
SECRET_REGION_ENV = "SECRET_REGION"       # lib/aws_secrets.py と同じ名前
REMOTE_COMMANDS: tuple[str, ...] = ("create-tables", "deploy-lambdas", "status", "teardown")
# CDK で一式を管理するリージョン。作る・消すは CDK に任せ、ここでは status(読むだけ)だけ通す
CDK_ONLY_REGIONS: tuple[str, ...] = ("ap-northeast-3",)
CDK_ONLY_COMMANDS: tuple[str, ...] = ("status",)


def lambda_spec(key: str, region: str = REGION) -> dict:
    """配置する関数の設定。東京なら `LAMBDA_SPECS` そのまま。東京以外は query だけで、
    環境変数に `SECRET_REGION=東京` を足したもの(元の辞書は変えない)。"""
    base = LAMBDA_SPECS[key]
    if region == HOME_REGION:
        return base
    if key not in REMOTE_LAMBDAS:
        raise ValueError(f"{region} に置けるのは {REMOTE_LAMBDAS} だけです: {key!r}")
    spec = dict(base)
    env = dict(((base.get("Environment") or {}).get("Variables") or {}))
    env[SECRET_REGION_ENV] = HOME_REGION
    spec["Environment"] = {"Variables": env}
    return spec


FUNCTION_TIMEOUT = 300.0          # 作成・更新が終わるまで待つ上限(秒)
ROLE_PROPAGATION_RETRIES = 6      # ロール作成直後の「引き受けられない」を待ち直す回数

# 削除してよい名前の許可リスト。**ここに無い名前は何があっても消さない**。
# IAM は種類ごと入れていない(= IAM の削除はこのスクリプトでは絶対に起きない)。
DELETABLE: dict[str, frozenset[str]] = {
    "lambda": frozenset(FUNCTIONS),
    "log_group": frozenset(LOG_GROUPS),
    "table": frozenset(TEARDOWN_TABLES),
    "ssm_parameter": frozenset((SECRET_PARAM,)),
    "usage_plan": frozenset((USAGE_PLAN_NAME,)),
    "api_key": frozenset((API_KEY_NAME,)),
    "rest_api": frozenset((API_NAME,)),
    "lambda_permission": frozenset((PERMISSION_NAME,)),
}

POLL_INTERVAL = 5.0
TABLE_TIMEOUT = 300.0

EXIT_OK = 0
EXIT_FAILED = 1


class RefusedToDelete(RuntimeError):
    """許可リスト外の名前を消そうとした。"""


def _guard_delete(kind: str, name: str) -> None:
    """削除 API を呼ぶ直前に必ず通す。完全一致で許可リストにあるときだけ通る。"""
    allowed = DELETABLE.get(kind)
    if allowed is None or name not in allowed:
        raise RefusedToDelete(f"許可リストに無いので消しません: {kind} {name!r}")


# ============================================================== AWS クライアント


class Aws:
    """必要なクライアントをまとめて持つ。session は差し替えられる。"""

    def __init__(self, session: Any, region: str = REGION) -> None:
        if region not in REGIONS:
            raise ValueError(f"リージョンは {REGIONS} のどれかです: {region!r}")
        cfg = boto_config(max_attempts=5)
        self.region = region
        self.ddb = session.client("dynamodb", region_name=region, config=cfg)
        # SSM の API キーは東京にだけ置く(東京以外のリージョンで動かしても、見るのは東京)
        self.ssm = session.client("ssm", region_name=HOME_REGION, config=cfg)
        self.iam = session.client("iam", region_name=region, config=cfg)
        self.lam = session.client("lambda", region_name=region, config=cfg)
        self.logs = session.client("logs", region_name=region, config=cfg)
        self.apigw = session.client("apigateway", region_name=region, config=cfg)

    @property
    def is_home(self) -> bool:
        """東京(SSM・API Gateway・ingest を置くリージョン)か。"""
        return self.region == HOME_REGION


def _not_found(exc: Exception) -> bool:
    return error_code(exc) in {"ResourceNotFoundException", "NoSuchEntity", "ParameterNotFound",
                               "NotFoundException"}


# ============================================================== 読むだけの関数


def describe_table(aws: Aws, name: str) -> dict | None:
    try:
        return aws.ddb.describe_table(TableName=name)["Table"]
    except Exception as exc:  # noqa: BLE001
        if _not_found(exc):
            return None
        raise


def describe_secret(aws: Aws) -> dict | None:
    """SSM パラメータのメタデータだけ(値は取らない)。無ければ None。"""
    resp = aws.ssm.describe_parameters(
        ParameterFilters=[{"Key": "Name", "Option": "Equals", "Values": [SECRET_PARAM]}]
    )
    for meta in resp.get("Parameters", []) or []:
        if meta.get("Name") == SECRET_PARAM:
            return meta
    return None


def get_function(aws: Aws, name: str) -> dict | None:
    try:
        return aws.lam.get_function_configuration(FunctionName=name)
    except Exception as exc:  # noqa: BLE001
        if _not_found(exc):
            return None
        raise


def existing_log_groups(aws: Aws, prefix: str = LOG_GROUP_PREFIX) -> list[str]:
    names: list[str] = []
    kwargs: dict[str, Any] = {"logGroupNamePrefix": prefix}
    while True:
        resp = aws.logs.describe_log_groups(**kwargs)
        names.extend(g["logGroupName"] for g in resp.get("logGroups", []) or [])
        token = resp.get("nextToken")
        if not token:
            return names
        kwargs["nextToken"] = token


def describe_role(aws: Aws) -> dict | None:
    """IAM ロールの有無とインラインポリシー名。**読むだけ**。"""
    try:
        role = aws.iam.get_role(RoleName=LAMBDA_ROLE)["Role"]
    except Exception as exc:  # noqa: BLE001
        if _not_found(exc):
            return None
        raise
    policies = aws.iam.list_role_policies(RoleName=LAMBDA_ROLE).get("PolicyNames", [])
    return {"arn": role.get("Arn"), "inline_policies": list(policies)}


# ============================================================== create-tables


def create_table_params(name: str) -> dict:
    keys = TABLE_KEYS[name]
    key_types = ("HASH", "RANGE")
    return {
        "TableName": name,
        "AttributeDefinitions": [{"AttributeName": k, "AttributeType": "S"} for k in keys],
        "KeySchema": [{"AttributeName": k, "KeyType": t} for k, t in zip(keys, key_types)],
        "BillingMode": "PAY_PER_REQUEST",
        "Tags": [dict(PROJECT_TAG)],
    }


def wait_table_active(aws: Aws, name: str, *, interval: float = POLL_INTERVAL,
                      timeout: float = TABLE_TIMEOUT,
                      sleep_fn: Callable[[float], None] = time.sleep,
                      clock: Callable[[], float] = time.time) -> None:
    deadline = clock() + timeout
    while True:
        table = describe_table(aws, name)
        status = table.get("TableStatus", "?") if table else "MISSING"
        print(f"  {name}: {status}", flush=True)
        if status == "ACTIVE":
            return
        if clock() >= deadline:
            raise TimeoutError(f"{name} が {timeout:.0f} 秒で ACTIVE になりませんでした({status})")
        sleep_fn(interval)


def run_create_tables(aws: Aws, *, sleep_fn: Callable[[float], None] = time.sleep) -> int:
    for name in TABLE_KEYS:
        table = describe_table(aws, name)
        if table is not None:
            print(f"{name}: 既にあります({table.get('TableStatus')})。作成はスキップ")
        else:
            print(f"{name}: 作成します(キー {'+'.join(TABLE_KEYS[name])}、オンデマンド)")
            aws.ddb.create_table(**create_table_params(name))
        wait_table_active(aws, name, sleep_fn=sleep_fn)
    print("2 テーブルとも ACTIVE です")
    return EXIT_OK


# ============================================================== put-secret


def run_put_secret(aws: Aws, *, key_loader: Callable[[], str] = load_api_key) -> int:
    """API キーを SecureString で登録する。**値はどこにも出さない**。"""
    try:
        resp = aws.ssm.put_parameter(
            Name=SECRET_PARAM,
            Value=key_loader(),
            Type="SecureString",
            Overwrite=True,
            Description="TypeSafe (Jev) API key for jev-graphrag-poc",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"!! 登録に失敗しました: {type(exc).__name__} / {error_code(exc) or '(コード無し)'}")
        return EXIT_FAILED
    version = resp.get("Version")
    # Overwrite=True と Tags は同時に渡せないので、タグは別 API で付ける
    aws.ssm.add_tags_to_resource(ResourceType="Parameter", ResourceId=SECRET_PARAM,
                                 Tags=[dict(PROJECT_TAG)])
    print(f"{SECRET_PARAM} に登録しました(バージョン {version}。値は表示しません)")
    return EXIT_OK


# ============================================================== deploy-lambdas


def ensure_log_group(aws: Aws, name: str) -> None:
    """ロググループを先に作り、保持期間を 7 日にする。既にあれば作成はスキップ
    (保持期間が 7 日でなければ 7 日にそろえる)。決め打ちの名前以外は作らない。"""
    if name not in LOG_GROUPS:
        raise ValueError(f"決め打ちのロググループではありません: {name!r}")
    resp = aws.logs.describe_log_groups(logGroupNamePrefix=name)
    found = [g for g in resp.get("logGroups", []) or [] if g.get("logGroupName") == name]
    if found:
        retention = found[0].get("retentionInDays")
        print(f"  {name}: 既にあります(保持 {retention or '無期限'} 日)。作成はスキップ")
        if retention == LOG_RETENTION_DAYS:
            return
    else:
        aws.logs.create_log_group(logGroupName=name, tags={PROJECT_TAG["Key"]: PROJECT_TAG["Value"]})
        print(f"  {name}: 作成しました")
    aws.logs.put_retention_policy(logGroupName=name, retentionInDays=LOG_RETENTION_DAYS)
    print(f"  {name}: 保持期間を {LOG_RETENTION_DAYS} 日にしました")


def wait_function_ready(aws: Aws, name: str, *, interval: float = POLL_INTERVAL,
                        timeout: float = FUNCTION_TIMEOUT,
                        sleep_fn: Callable[[float], None] = time.sleep,
                        clock: Callable[[], float] = time.time) -> dict:
    """State=Active かつ LastUpdateStatus=Successful になるまで待つ。失敗なら RuntimeError。"""
    deadline = clock() + timeout
    while True:
        fn = get_function(aws, name) or {}
        state = fn.get("State", "?")
        update = fn.get("LastUpdateStatus", "Successful")
        print(f"  {name}: State={state} / LastUpdateStatus={update}", flush=True)
        if state == "Failed" or update == "Failed":
            raise RuntimeError(f"{name} の作成・更新に失敗しました: "
                               f"{fn.get('StateReason') or fn.get('LastUpdateStatusReason')}")
        if state == "Active" and update == "Successful":
            return fn
        if clock() >= deadline:
            raise TimeoutError(f"{name} が {timeout:.0f} 秒で準備完了になりませんでした")
        sleep_fn(interval)


def _config_drift(current: dict, spec: dict) -> list[str]:
    """既存の関数の設定と spec の違い(表示用)。"""
    diffs = []
    for key in ("Runtime", "MemorySize", "Timeout", "Handler", "Role"):
        if current.get(key) != spec[key]:
            diffs.append(f"{key}: {current.get(key)!r} -> 期待 {spec[key]!r}")
    if current.get("Architectures") not in (None, spec["Architectures"]):
        diffs.append(f"Architectures: {current.get('Architectures')!r} -> 期待 {spec['Architectures']!r}")
    return diffs


def missing_env(current: dict, spec: dict) -> dict[str, str]:
    """spec の環境変数のうち、既存の関数に無い・値が違うもの(spec に Environment が無ければ空)。"""
    want = ((spec.get("Environment") or {}).get("Variables") or {})
    have = ((current.get("Environment") or {}).get("Variables") or {})
    return {k: v for k, v in want.items() if have.get(k) != v}


def create_function_params(spec: dict, zip_bytes: bytes) -> dict:
    params = {k: v for k, v in spec.items() if k != "zip"}
    params["Code"] = {"ZipFile": zip_bytes}
    params["Tags"] = {PROJECT_TAG["Key"]: PROJECT_TAG["Value"]}
    params["Publish"] = False
    return params


def deploy_function(aws: Aws, spec: dict, zip_bytes: bytes, *,
                    sleep_fn: Callable[[float], None] = time.sleep) -> str:
    """関数を作る(無ければ)/コードだけ更新する(あれば)。戻り値は "created" / "updated"。"""
    name = spec["FunctionName"]
    if name not in FUNCTIONS:
        raise ValueError(f"決め打ちの関数名ではありません: {name!r}")
    current = get_function(aws, name)
    if current is None:
        print(f"  {name}: 作成します({spec['Runtime']} / {spec['Architectures'][0]} / "
              f"{spec['MemorySize']}MB / {spec['Timeout']}秒)")
        for attempt in range(ROLE_PROPAGATION_RETRIES + 1):
            try:
                aws.lam.create_function(**create_function_params(spec, zip_bytes))
                break
            except Exception as exc:  # noqa: BLE001
                # ロール作成直後は「Lambda がロールを引き受けられない」で弾かれることがある
                if error_code(exc) != "InvalidParameterValueException" or attempt == ROLE_PROPAGATION_RETRIES:
                    raise
                print(f"  {name}: ロールの反映待ち({attempt + 1}/{ROLE_PROPAGATION_RETRIES})", flush=True)
                sleep_fn(10.0)
        wait_function_ready(aws, name, sleep_fn=sleep_fn)
        return "created"

    drift = _config_drift(current, spec)
    if drift:
        print(f"  {name}: 注意 設定が想定と違います(コードだけ更新し、設定は変えません)")
        for line in drift:
            print(f"    - {line}")
    wait_function_ready(aws, name, sleep_fn=sleep_fn)       # 前の更新が終わっていないと弾かれる
    print(f"  {name}: 既にあるのでコードだけ更新します")
    aws.lam.update_function_code(FunctionName=name, ZipFile=zip_bytes,
                                 Architectures=spec["Architectures"], Publish=False)
    current = wait_function_ready(aws, name, sleep_fn=sleep_fn)
    env = missing_env(current, spec)
    if env:
        # 環境変数だけ足す(ほかの設定と、既にある環境変数はそのまま)。値は秘密ではないので表示する
        have = ((current.get("Environment") or {}).get("Variables") or {})
        print(f"  {name}: 環境変数を設定します: {env}")
        aws.lam.update_function_configuration(FunctionName=name,
                                              Environment={"Variables": {**have, **env}})
        wait_function_ready(aws, name, sleep_fn=sleep_fn)
    return "updated"


def run_deploy_lambdas(aws: Aws, only: Sequence[str] | None = None, *,
                       build_dir: Path | None = None,
                       sleep_fn: Callable[[float], None] = time.sleep) -> int:
    if aws.is_home:
        targets = list(only) if only else list(LAMBDA_SPECS)
    else:
        targets = list(only) if only else list(REMOTE_LAMBDAS)
        refused = [k for k in targets if k not in REMOTE_LAMBDAS]
        if refused:                         # AWS を変える前に止める(登録は東京だけでやる)
            print(f"!! {aws.region} に置けるのは {', '.join(REMOTE_LAMBDAS)} だけです(指定: {', '.join(refused)})")
            return EXIT_FAILED
        print(f"リージョン {aws.region}: {', '.join(targets)} だけを配置します"
              f"(環境変数 {SECRET_REGION_ENV}={HOME_REGION} を足す)")
    build_dir = Path(build_dir) if build_dir is not None else BUILD_DIR
    zips: dict[str, bytes] = {}
    for key in targets:                     # AWS を変える前に zip が揃っているか全部確かめる
        path = build_dir / LAMBDA_SPECS[key]["zip"]
        if not path.exists():
            print(f"!! {path} がありません。先に bash infra/build.sh を実行してください")
            return EXIT_FAILED
        zips[key] = path.read_bytes()
        print(f"{key}: {path}({len(zips[key]):,} bytes)")
    for key in targets:
        spec = lambda_spec(key, aws.region)
        ensure_log_group(aws, f"/aws/lambda/{spec['FunctionName']}")
        result = deploy_function(aws, spec, zips[key], sleep_fn=sleep_fn)
        fn = get_function(aws, spec["FunctionName"]) or {}
        print(f"  {spec['FunctionName']}: {result} / CodeSha256={fn.get('CodeSha256')} / "
              f"CodeSize={fn.get('CodeSize')}")
    return EXIT_OK


# ============================================================== API Gateway(読むだけの関数)


def _paged(fn: Callable[..., dict], **kwargs: Any) -> list[dict]:
    """API Gateway の `items` / `position` 形式のページ送り。"""
    items: list[dict] = []
    kwargs = {"limit": 500, **kwargs}
    while True:
        resp = fn(**kwargs)
        items.extend(resp.get("items", []) or [])
        position = resp.get("position")
        if not position:
            return items
        kwargs["position"] = position


def _one_by_name(items: Sequence[dict], name: str, what: str) -> dict | None:
    """名前の完全一致で 1 つ。同名が複数あれば、どれを触るべきか決められないので止める。"""
    found = [i for i in items if i.get("name") == name]
    if len(found) > 1:
        raise RuntimeError(f"{what} {name!r} が {len(found)} 個あります。手で確認してください"
                           f"(id: {[i.get('id') for i in found]})")
    return found[0] if found else None


def find_rest_api(aws: Aws) -> dict | None:
    return _one_by_name(_paged(aws.apigw.get_rest_apis), API_NAME, "REST API")


def find_usage_plan(aws: Aws) -> dict | None:
    return _one_by_name(_paged(aws.apigw.get_usage_plans), USAGE_PLAN_NAME, "使用量プラン")


def find_api_key(aws: Aws) -> dict | None:
    """API キーのメタデータだけ(**値は取らない**。`includeValues=False`)。"""
    items = _paged(aws.apigw.get_api_keys, nameQuery=API_KEY_NAME, includeValues=False)
    return _one_by_name(items, API_KEY_NAME, "API キー")


def get_stage(aws: Aws, api_id: str) -> dict | None:
    try:
        return aws.apigw.get_stage(restApiId=api_id, stageName=API_STAGE)
    except Exception as exc:  # noqa: BLE001
        if _not_found(exc):
            return None
        raise


def get_method(aws: Aws, api_id: str, resource_id: str) -> dict | None:
    try:
        return aws.apigw.get_method(restApiId=api_id, resourceId=resource_id, httpMethod=API_METHOD)
    except Exception as exc:  # noqa: BLE001
        if _not_found(exc):
            return None
        raise


def get_integration(aws: Aws, api_id: str, resource_id: str) -> dict | None:
    try:
        return aws.apigw.get_integration(restApiId=api_id, resourceId=resource_id, httpMethod=API_METHOD)
    except Exception as exc:  # noqa: BLE001
        if _not_found(exc):
            return None
        raise


def api_resources(aws: Aws, api_id: str) -> dict[str, dict]:
    """パス → リソース。"""
    return {r.get("path"): r for r in _paged(aws.apigw.get_resources, restApiId=api_id)}


def source_arn(api_id: str) -> str:
    """Lambda を呼んでよい送信元。この API のステージ poc の POST /query だけ。"""
    return f"arn:aws:execute-api:{REGION}:{ACCOUNT_ID}:{api_id}/{API_STAGE}/{API_METHOD}/{API_PATH_PART}"


def invoke_url(api_id: str) -> str:
    return f"https://{api_id}.execute-api.{REGION}.amazonaws.com/{API_STAGE}/{API_PATH_PART}"


def integration_uri(function_arn: str) -> str:
    return f"arn:aws:apigateway:{REGION}:lambda:path/2015-03-31/functions/{function_arn}/invocations"


def get_permission_statement(aws: Aws) -> dict | None:
    """query Lambda のリソースベースポリシーから、決め打ちの StatementId の文だけを返す。"""
    try:
        policy = aws.lam.get_policy(FunctionName=QUERY_FUNCTION).get("Policy") or "{}"
    except Exception as exc:  # noqa: BLE001
        if _not_found(exc):
            return None
        raise
    for stmt in json.loads(policy).get("Statement", []) or []:
        if stmt.get("Sid") == PERMISSION_SID:
            return stmt
    return None


def _statement_source_arn(stmt: dict) -> str | None:
    cond = stmt.get("Condition") or {}
    return (cond.get("ArnLike") or {}).get("AWS:SourceArn")


def write_api_local(api_id: str, api_key_value: str, *, path: Path | None = None) -> Path:
    """URL とキーを `.api.local.json` に書く(権限 600)。キーの値は表示しない。"""
    path = Path(path) if path is not None else API_LOCAL_PATH
    data = {"url": invoke_url(api_id), "api_key": api_key_value, "api_id": api_id,
            "stage": API_STAGE, "region": REGION}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.chmod(path, 0o600)                       # 既にあったファイルでも 600 にそろえる
    return path


# ============================================================== create-api


def _tags() -> dict[str, str]:
    return {PROJECT_TAG["Key"]: PROJECT_TAG["Value"]}


def run_create_api(aws: Aws, *, out_path: Path | None = None) -> int:
    """REST API 一式を作る(既にあるものは作らずに使う)。最後に URL とキーを `.api.local.json` へ。"""
    fn = get_function(aws, QUERY_FUNCTION)
    if fn is None:
        print(f"!! {QUERY_FUNCTION} がありません。先に deploy-lambdas --only query を実行してください")
        return EXIT_FAILED
    function_arn = fn.get("FunctionArn") or \
        f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{QUERY_FUNCTION}"
    changed = False                             # メソッド・統合を変えたら再デプロイする

    # ---- REST API
    api = find_rest_api(aws)
    if api is None:
        api = aws.apigw.create_rest_api(
            name=API_NAME, description="jev-graphrag-poc query API (API key required)",
            endpointConfiguration={"types": ["REGIONAL"]}, apiKeySource="HEADER", tags=_tags())
        print(f"REST API {API_NAME}: 作成しました(id {api['id']})")
        changed = True
    else:
        print(f"REST API {API_NAME}: 既にあります(id {api['id']})。作成はスキップ")
    api_id = api["id"]

    # ---- リソース /query
    resources = api_resources(aws, api_id)
    resource = resources.get(f"/{API_PATH_PART}")
    if resource is None:
        root = resources.get("/")
        if root is None:
            print("!! ルートリソース / が見つかりません")
            return EXIT_FAILED
        resource = aws.apigw.create_resource(restApiId=api_id, parentId=root["id"], pathPart=API_PATH_PART)
        print(f"  リソース /{API_PATH_PART}: 作成しました")
        changed = True
    else:
        print(f"  リソース /{API_PATH_PART}: 既にあります")
    resource_id = resource["id"]

    # ---- メソッド POST(API キー必須)
    method = get_method(aws, api_id, resource_id)
    if method is None:
        aws.apigw.put_method(restApiId=api_id, resourceId=resource_id, httpMethod=API_METHOD,
                             authorizationType="NONE", apiKeyRequired=True)
        print(f"  メソッド {API_METHOD}: 作成しました(API キー必須)")
        changed = True
    elif not method.get("apiKeyRequired"):
        aws.apigw.update_method(restApiId=api_id, resourceId=resource_id, httpMethod=API_METHOD,
                                patchOperations=[{"op": "replace", "path": "/apiKeyRequired",
                                                  "value": "true"}])
        print(f"  メソッド {API_METHOD}: API キー必須に直しました")
        changed = True
    else:
        print(f"  メソッド {API_METHOD}: 既にあります(API キー必須)")

    # ---- 統合(Lambda プロキシ)
    uri = integration_uri(function_arn)
    integ = get_integration(aws, api_id, resource_id)
    if integ is None or integ.get("type") != "AWS_PROXY" or integ.get("uri") != uri:
        aws.apigw.put_integration(restApiId=api_id, resourceId=resource_id, httpMethod=API_METHOD,
                                  type="AWS_PROXY", integrationHttpMethod="POST", uri=uri,
                                  timeoutInMillis=INTEGRATION_TIMEOUT_MS)
        print(f"  統合: {QUERY_FUNCTION} へのプロキシ統合を{'作成' if integ is None else '更新'}しました")
        changed = True
    else:
        print("  統合: 既にあります(Lambda プロキシ)")

    # ---- Lambda のリソースベースポリシー(送信元をこの API の POST /query に限定)
    want = source_arn(api_id)
    stmt = get_permission_statement(aws)
    if stmt is not None and _statement_source_arn(stmt) != want:
        print(f"  呼び出し許可 {PERMISSION_SID}: 送信元が違うので付け直します")
        delete_lambda_permission(aws, PERMISSION_NAME)
        stmt = None
    if stmt is None:
        aws.lam.add_permission(FunctionName=QUERY_FUNCTION, StatementId=PERMISSION_SID,
                               Action="lambda:InvokeFunction", Principal="apigateway.amazonaws.com",
                               SourceArn=want, SourceAccount=ACCOUNT_ID)
        print(f"  呼び出し許可 {PERMISSION_SID}: 追加しました(送信元 {want})")
    else:
        print(f"  呼び出し許可 {PERMISSION_SID}: 既にあります")

    # ---- デプロイ(ステージ poc)
    stage = get_stage(aws, api_id)
    if stage is None or changed:
        aws.apigw.create_deployment(restApiId=api_id, stageName=API_STAGE,
                                    description="jev-graphrag-poc create-api")
        print(f"  ステージ {API_STAGE}: デプロイしました")
    else:
        print(f"  ステージ {API_STAGE}: 既にあります(変更なしなので再デプロイしない)")

    # ---- API キー
    key = find_api_key(aws)
    if key is None:
        key = aws.apigw.create_api_key(name=API_KEY_NAME, enabled=True,
                                       description="jev-graphrag-poc local UI", tags=_tags())
        print(f"API キー {API_KEY_NAME}: 作成しました(値は表示しません)")
    else:
        print(f"API キー {API_KEY_NAME}: 既にあります(値は表示しません)")
    key_id = key["id"]

    # ---- 使用量プラン
    plan = find_usage_plan(aws)
    stage_ref = {"apiId": api_id, "stage": API_STAGE}
    if plan is None:
        plan = aws.apigw.create_usage_plan(
            name=USAGE_PLAN_NAME, description="jev-graphrag-poc: 1 req/s, 200 req/day",
            apiStages=[stage_ref], throttle=dict(USAGE_THROTTLE), quota=dict(USAGE_QUOTA), tags=_tags())
        print(f"使用量プラン {USAGE_PLAN_NAME}: 作成しました(1 回/秒・バースト 2・200 回/日)")
    else:
        print(f"使用量プラン {USAGE_PLAN_NAME}: 既にあります")
        patches = []
        stages = plan.get("apiStages") or []
        if not any(s.get("apiId") == api_id and s.get("stage") == API_STAGE for s in stages):
            patches.append({"op": "add", "path": "/apiStages", "value": f"{api_id}:{API_STAGE}"})
        throttle, quota = plan.get("throttle") or {}, plan.get("quota") or {}
        if throttle.get("rateLimit") != USAGE_THROTTLE["rateLimit"]:
            patches.append({"op": "replace", "path": "/throttle/rateLimit",
                            "value": str(USAGE_THROTTLE["rateLimit"])})
        if throttle.get("burstLimit") != USAGE_THROTTLE["burstLimit"]:
            patches.append({"op": "replace", "path": "/throttle/burstLimit",
                            "value": str(USAGE_THROTTLE["burstLimit"])})
        if quota.get("limit") != USAGE_QUOTA["limit"]:
            patches.append({"op": "replace", "path": "/quota/limit", "value": str(USAGE_QUOTA["limit"])})
        if quota.get("period") != USAGE_QUOTA["period"]:
            patches.append({"op": "replace", "path": "/quota/period", "value": USAGE_QUOTA["period"]})
        if patches:
            aws.apigw.update_usage_plan(usagePlanId=plan["id"], patchOperations=patches)
            print(f"  {len(patches)} 項目を直しました: {[p['path'] for p in patches]}")
    plan_keys = _paged(aws.apigw.get_usage_plan_keys, usagePlanId=plan["id"])
    if any(k.get("id") == key_id for k in plan_keys):
        print("  API キーとの紐づけ: 既にあります")
    else:
        aws.apigw.create_usage_plan_key(usagePlanId=plan["id"], keyId=key_id, keyType="API_KEY")
        print("  API キーとの紐づけ: 作りました")

    # ---- URL とキーをローカルに(キーの値はここで初めて取る。表示しない)
    value = aws.apigw.get_api_key(apiKey=key_id, includeValue=True).get("value")
    if not value:
        print("!! API キーの値が取れませんでした")
        return EXIT_FAILED
    path = write_api_local(api_id, value, path=out_path)
    print(f"URL : {invoke_url(api_id)}")
    print(f"{path.name} に URL とキーを書きました(権限 600。キーの値は表示しません)")
    print("使用量プランの反映には数十秒かかることがあります(直後は 403 になる場合あり)")
    return EXIT_OK


# ============================================================== status


def _try(fn: Callable[[], Any]) -> tuple[Any, str | None]:
    """権限不足などで読めなくても一覧は最後まで出したいので、エラーは文字列で返す。"""
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001
        return None, f"確認できません({error_code(exc) or type(exc).__name__})"


def run_status(aws: Aws) -> int:
    if aws.region in CDK_ONLY_REGIONS:
        print(f"リージョン {aws.region}(CDK のスタック。SSM は東京を読むだけ、API Gateway は見ません。"
              f"IAM は東京の {LAMBDA_ROLE} を見ます。スタックのロールは {LAMBDA_ROLE}-{aws.region})")
    elif not aws.is_home:
        print(f"リージョン {aws.region}(本手法の query の写し。SSM は東京を読むだけ、API Gateway は見ません)")
    print("[DynamoDB]")
    for name in TEARDOWN_TABLES:
        table, err = _try(lambda n=name: describe_table(aws, n))
        if err:
            print(f"  {name}: {err}")
        elif table is None:
            print(f"  {name}: 無し")
        else:
            print(f"  {name}: {table.get('TableStatus')} / 件数(概算) {table.get('ItemCount')}")

    print("[SSM]" if aws.is_home else f"[SSM]({HOME_REGION}。読むだけ)")
    meta, err = _try(lambda: describe_secret(aws))
    if err:
        print(f"  {SECRET_PARAM}: {err}")
    elif meta is None:
        print(f"  {SECRET_PARAM}: 無し")
    else:
        print(f"  {SECRET_PARAM}: あり / {meta.get('Type')} / バージョン {meta.get('Version')}")

    print("[IAM](読むだけ)")
    role, err = _try(lambda: describe_role(aws))
    if err:
        print(f"  {LAMBDA_ROLE}: {err}")
    elif role is None:
        print(f"  {LAMBDA_ROLE}: 無し")
    else:
        names = ", ".join(role["inline_policies"]) or "(無し)"
        print(f"  {LAMBDA_ROLE}: あり / インラインポリシー: {names}")

    print("[Lambda]")
    for name in FUNCTIONS:
        fn, err = _try(lambda n=name: get_function(aws, n))
        if err:
            print(f"  {name}: {err}")
        elif fn is None:
            print(f"  {name}: 無し")
        else:
            print(f"  {name}: あり / {fn.get('Runtime')} / {fn.get('State', '?')}")

    print("[CloudWatch Logs]")
    groups, err = _try(lambda: existing_log_groups(aws))
    if err:
        print(f"  {LOG_GROUP_PREFIX}*: {err}")
    elif not groups:
        print(f"  {LOG_GROUP_PREFIX}*: 無し")
    else:
        for g in groups:
            note = "" if g in LOG_GROUPS else "(想定外の名前。teardown では消さない)"
            print(f"  {g}: あり{note}")

    if aws.is_home:
        run_status_api(aws)
    return EXIT_OK


def run_status_api(aws: Aws) -> None:
    """API Gateway・使用量プラン・API キー(有無だけ)・Lambda の呼び出し許可。読むだけ。"""
    print("[API Gateway]")
    api, err = _try(lambda: find_rest_api(aws))
    if err:
        print(f"  {API_NAME}: {err}")
    elif api is None:
        print(f"  {API_NAME}: 無し")
    else:
        api_id = api["id"]
        print(f"  {API_NAME}: あり / id {api_id}")
        stage, err = _try(lambda: get_stage(aws, api_id))
        if err:
            print(f"  ステージ {API_STAGE}: {err}")
        elif stage is None:
            print(f"  ステージ {API_STAGE}: 無し(未デプロイ)")
        else:
            print(f"  ステージ {API_STAGE}: あり / URL {invoke_url(api_id)}")
        res, err = _try(lambda: api_resources(aws, api_id).get(f"/{API_PATH_PART}"))
        if err:
            print(f"  {API_METHOD} /{API_PATH_PART}: {err}")
        elif res is None:
            print(f"  {API_METHOD} /{API_PATH_PART}: 無し")
        else:
            method, err = _try(lambda: get_method(aws, api_id, res["id"]))
            if err:
                print(f"  {API_METHOD} /{API_PATH_PART}: {err}")
            elif method is None:
                print(f"  {API_METHOD} /{API_PATH_PART}: メソッド無し")
            else:
                need = "必須" if method.get("apiKeyRequired") else "!! 不要になっています"
                print(f"  {API_METHOD} /{API_PATH_PART}: あり / API キー {need}")

    print("[使用量プラン]")
    plan, err = _try(lambda: find_usage_plan(aws))
    if err:
        print(f"  {USAGE_PLAN_NAME}: {err}")
    elif plan is None:
        print(f"  {USAGE_PLAN_NAME}: 無し")
    else:
        th, qu = plan.get("throttle") or {}, plan.get("quota") or {}
        stages = [f"{s.get('apiId')}:{s.get('stage')}" for s in plan.get("apiStages") or []]
        print(f"  {USAGE_PLAN_NAME}: あり / {th.get('rateLimit')} 回/秒・バースト {th.get('burstLimit')} / "
              f"{qu.get('limit')} 回/{qu.get('period')} / ステージ {stages or '(無し)'}")

    print("[API キー](有無だけ。値は取らない)")
    key, err = _try(lambda: find_api_key(aws))
    if err:
        print(f"  {API_KEY_NAME}: {err}")
    elif key is None:
        print(f"  {API_KEY_NAME}: 無し")
    else:
        print(f"  {API_KEY_NAME}: あり / {'有効' if key.get('enabled') else '無効'}")

    print("[Lambda の呼び出し許可]")
    stmt, err = _try(lambda: get_permission_statement(aws))
    if err:
        print(f"  {PERMISSION_NAME}: {err}")
    elif stmt is None:
        print(f"  {PERMISSION_NAME}: 無し")
    else:
        print(f"  {PERMISSION_NAME}: あり / 送信元 {_statement_source_arn(stmt)}")

    print("[ローカル]")
    if API_LOCAL_PATH.exists():
        mode = API_LOCAL_PATH.stat().st_mode & 0o777
        note = "" if mode == 0o600 else "(!! 600 ではありません)"
        print(f"  {API_LOCAL_PATH.name}: あり / 権限 {oct(mode)}{note}")
    else:
        print(f"  {API_LOCAL_PATH.name}: 無し")


# ============================================================== teardown


def delete_function(aws: Aws, name: str) -> None:
    _guard_delete("lambda", name)
    aws.lam.delete_function(FunctionName=name)


def delete_log_group(aws: Aws, name: str) -> None:
    _guard_delete("log_group", name)
    aws.logs.delete_log_group(logGroupName=name)


def delete_table(aws: Aws, name: str) -> None:
    _guard_delete("table", name)
    aws.ddb.delete_table(TableName=name)


def delete_parameter(aws: Aws, name: str) -> None:
    _guard_delete("ssm_parameter", name)
    aws.ssm.delete_parameter(Name=name)


def delete_usage_plan(aws: Aws, name: str) -> bool:
    """ステージの紐づけを外してから消す。無ければ False。"""
    _guard_delete("usage_plan", name)
    plan = find_usage_plan(aws)
    if plan is None:
        return False
    stages = plan.get("apiStages") or []
    if stages:
        aws.apigw.update_usage_plan(usagePlanId=plan["id"], patchOperations=[
            {"op": "remove", "path": "/apiStages", "value": f"{s['apiId']}:{s['stage']}"} for s in stages])
    aws.apigw.delete_usage_plan(usagePlanId=plan["id"])
    return True


def delete_api_key(aws: Aws, name: str) -> bool:
    _guard_delete("api_key", name)
    key = find_api_key(aws)
    if key is None:
        return False
    aws.apigw.delete_api_key(apiKey=key["id"])
    return True


def delete_rest_api(aws: Aws, name: str) -> bool:
    _guard_delete("rest_api", name)
    api = find_rest_api(aws)
    if api is None:
        return False
    aws.apigw.delete_rest_api(restApiId=api["id"])
    return True


def delete_lambda_permission(aws: Aws, name: str) -> bool:
    """query Lambda のリソースベースポリシーから決め打ちの 1 文だけ外す(関数・ロールは触らない)。"""
    _guard_delete("lambda_permission", name)
    function, sid = name.split(":", 1)
    aws.lam.remove_permission(FunctionName=function, StatementId=sid)
    return True


def plan_teardown(aws: Aws) -> list[tuple[str, str]]:
    """消す順(使用量プラン → API キー → REST API → Lambda の呼び出し許可 → Lambda → ロググループ
    → テーブル → SSM)に、**今あるものだけ**を並べる。"""
    targets: list[tuple[str, str]] = []
    if aws.is_home:
        if find_usage_plan(aws) is not None:
            targets.append(("usage_plan", USAGE_PLAN_NAME))
        if find_api_key(aws) is not None:
            targets.append(("api_key", API_KEY_NAME))
        if find_rest_api(aws) is not None:
            targets.append(("rest_api", API_NAME))
        if get_permission_statement(aws) is not None:
            targets.append(("lambda_permission", PERMISSION_NAME))
    for name in FUNCTIONS:
        if get_function(aws, name) is not None:
            targets.append(("lambda", name))
    present_groups = set(existing_log_groups(aws))
    for name in LOG_GROUPS:
        if name in present_groups:
            targets.append(("log_group", name))
    for name in TEARDOWN_TABLES:
        if describe_table(aws, name) is not None:
            targets.append(("table", name))
    if aws.is_home and describe_secret(aws) is not None:
        targets.append(("ssm_parameter", SECRET_PARAM))
    # 念のため、計画の時点でも許可リストと突き合わせる
    for kind, name in targets:
        if not aws.is_home and kind in ("ssm_parameter", "usage_plan", "api_key", "rest_api",
                                        "lambda_permission"):
            raise RefusedToDelete(f"{aws.region} では消しません: {kind} {name!r}")
        _guard_delete(kind, name)
    return targets


def wait_table_gone(aws: Aws, name: str, *, interval: float = POLL_INTERVAL,
                    timeout: float = TABLE_TIMEOUT,
                    sleep_fn: Callable[[float], None] = time.sleep,
                    clock: Callable[[], float] = time.time) -> None:
    deadline = clock() + timeout
    while describe_table(aws, name) is not None:
        if clock() >= deadline:
            raise TimeoutError(f"{name} が {timeout:.0f} 秒で消えませんでした")
        sleep_fn(interval)


_DELETERS: dict[str, Callable[[Aws, str], Any]] = {
    "usage_plan": delete_usage_plan,
    "api_key": delete_api_key,
    "rest_api": delete_rest_api,
    "lambda_permission": delete_lambda_permission,
    "lambda": delete_function,
    "log_group": delete_log_group,
    "table": delete_table,
    "ssm_parameter": delete_parameter,
}


def run_teardown(aws: Aws, *, assume_yes: bool = False,
                 input_fn: Callable[[str], str] = input,
                 sleep_fn: Callable[[float], None] = time.sleep) -> int:
    targets = plan_teardown(aws)
    if not targets:
        print("消すものはありません(全部無いか、もう消えています)")
    else:
        if not aws.is_home:
            print(f"リージョン {aws.region}(本手法の query の写し。SSM・API Gateway には触れません)")
        print("次のリソースを削除します:")
        for kind, name in targets:
            print(f"  - {kind}: {name}")
        if not assume_yes:
            try:
                answer = input_fn("本当に削除するなら yes と入力してください: ")
            except EOFError:
                answer = ""
            if answer.strip() != "yes":
                print("yes ではないので何もしませんでした")
                return EXIT_FAILED
        for kind, name in targets:
            try:
                result = _DELETERS[kind](aws, name)
            except RefusedToDelete:
                raise
            except Exception as exc:  # noqa: BLE001
                if _not_found(exc):
                    print(f"  {kind}: {name} はもうありません(スキップ)")
                    continue
                raise
            if result is False:
                print(f"  {kind}: {name} はもうありません(スキップ)")
                continue
            print(f"  {kind}: {name} を削除しました")
            if kind == "table":
                wait_table_gone(aws, name, sleep_fn=sleep_fn)
    if aws.is_home and API_LOCAL_PATH.exists():     # 東京の API の URL とキー。他リージョンでは消さない
        API_LOCAL_PATH.unlink()
        print(f"  ローカル: {API_LOCAL_PATH.name} を削除しました")
    print(f"IAM ロール {LAMBDA_ROLE} はこのスクリプトでは消しません。"
          "必要なら手動で削除してください")
    return EXIT_OK


# ============================================================== CLI

SUBCOMMANDS: tuple[str, ...] = ("create-tables", "put-secret", "deploy-lambdas", "create-api",
                                "status", "teardown")


def build_parser() -> argparse.ArgumentParser:
    region_help = (f"触るリージョン({', '.join(REGIONS)})。省略時は {REGION}(環境変数 JEV_REGION、既定 {HOME_REGION})。"
                   f"{HOME_REGION} 以外では {', '.join(REMOTE_COMMANDS)} だけ")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", choices=list(REGIONS), default=None, help=region_help)
    # サブコマンドの後ろにも書けるように(後ろに書いた値が前の値を上書きする。省略時は前の値を残す)
    common_opts = argparse.ArgumentParser(add_help=False)
    common_opts.add_argument("--region", choices=list(REGIONS), default=argparse.SUPPRESS, help=region_help)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create-tables", parents=[common_opts], help="DynamoDB 2 本を作る(既にあればスキップ)")
    sub.add_parser("put-secret", parents=[common_opts], help="API キーを SSM SecureString に登録(値は表示しない)")
    dl = sub.add_parser("deploy-lambdas", parents=[common_opts],
                        help="build/*.zip を Lambda に配置(無ければ作成、あればコードだけ更新)")
    dl.add_argument("--only", choices=list(LAMBDA_SPECS), action="append", default=None,
                    help="配置する関数(繰り返し指定可)。省略時は全部(東京以外は query だけ)")
    sub.add_parser("create-api", parents=[common_opts],
                   help="REST API・API キー・使用量プラン・呼び出し許可を作る(既存は使う)")
    sub.add_parser("status", parents=[common_opts], help="全リソースの有無を一覧(読むだけ)")
    td = sub.add_parser("teardown", parents=[common_opts], help="決め打ちのリソースを削除(IAM は消さない)")
    td.add_argument("--yes", action="store_true", help="yes の入力確認を省略する")
    return parser


def main(argv: Sequence[str] | None = None, *, session: Any = None,
         input_fn: Callable[[str], str] = input,
         sleep_fn: Callable[[float], None] = time.sleep) -> int:
    args = build_parser().parse_args(argv)
    region = args.region or REGION
    if region in CDK_ONLY_REGIONS and args.command not in CDK_ONLY_COMMANDS:
        # AWS を呼ぶ前に止める(大阪は CDK のスタック。作る・消すは cdk deploy / cdk destroy で)
        print(f"!! {region} は CDK で管理します。このスクリプトで使えるのは {', '.join(CDK_ONLY_COMMANDS)} だけです"
              f"(指定: {args.command})。消すときは npx aws-cdk@2 destroy -c region={region}")
        return EXIT_FAILED
    if region != HOME_REGION and args.command not in REMOTE_COMMANDS:
        # AWS を呼ぶ前に止める(SSM と API Gateway は東京にだけ置く)
        print(f"!! {args.command} は {HOME_REGION} だけです(指定: {region})。"
              f"{region} で使えるのは {', '.join(REMOTE_COMMANDS)}")
        return EXIT_FAILED

    if session is None:
        import boto3
        session = boto3.Session(region_name=region)
    check_account(session, region)              # JEV_AWS_ACCOUNT_ID と違えばここで例外
    aws = Aws(session, region)

    if args.command == "create-tables":
        return run_create_tables(aws, sleep_fn=sleep_fn)
    if args.command == "put-secret":
        return run_put_secret(aws)
    if args.command == "deploy-lambdas":
        return run_deploy_lambdas(aws, args.only, sleep_fn=sleep_fn)
    if args.command == "create-api":
        return run_create_api(aws)
    if args.command == "status":
        return run_status(aws)
    if args.command == "teardown":
        return run_teardown(aws, assume_yes=args.yes, input_fn=input_fn, sleep_fn=sleep_fn)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
