"""CDK スタックが使う値。**`infra/deploy.py` の現在の値をそのまま写したもの**。

- 名前は Lambda のコード(`lib/graph_store.py` のテーブル名・`lib/aws_secrets.py` の SSM パラメータ名)と
  スクリプト(`scripts/20_invoke_ingest.py`・`scripts/21_invoke_query.py` の関数名)が決め打ちにしているので変えない
- deploy.py・lib と食い違っていないかは `check_parity.py` で確かめる(AWS は呼ばない)
- ここには秘密もアカウント ID も書かない"""

from __future__ import annotations

from typing import Any

# ============================================================== リージョン

HOME_REGION = "ap-northeast-1"                    # lib/common.py と同じ
REGIONS: tuple[str, ...] = (HOME_REGION, "us-west-2", "ap-northeast-3")   # lib/common.py と同じ
FULL_REGIONS: tuple[str, ...] = (HOME_REGION, "ap-northeast-3")  # 一式(Lambda 2 つ・API Gateway)を置く。lib/common.INGEST_REGIONS と同じ
REMOTE_LAMBDAS: tuple[str, ...] = ("query",)      # それ以外(us-west-2)に置くのは query の写しだけ
SECRET_REGION_ENV = "SECRET_REGION"               # lib/aws_secrets.py と同じ名前

# ============================================================== タグ

PROJECT_TAG_KEY = "project"
PROJECT_TAG_VALUE = "jev-graphrag-poc"

# ============================================================== DynamoDB

CHUNK_TABLE = "jev-graphrag-chunk"
GRAPH_TABLE = "jev-graphrag-graph"
TABLE_KEYS: dict[str, tuple[str, ...]] = {        # 属性はすべて S。オンデマンド
    CHUNK_TABLE: ("PK",),
    GRAPH_TABLE: ("PK", "SK"),
}

# ============================================================== SSM(スタックには入れない。名前で権限だけ参照)

SECRET_PARAM = "/jev-graphrag-poc/typesafe-api-key"

# ============================================================== IAM / Lambda

LAMBDA_ROLE = "jev-graphrag-poc-lambda-role"      # 東京のスタックの名前(deploy.py status が読む名前)
LAMBDA_POLICY = "jev-graphrag-poc-policy"         # README の手順で付けていたインラインポリシー名
INGEST_FUNCTION = "jev-graphrag-ingest"
QUERY_FUNCTION = "jev-graphrag-query"
LOG_GROUP_PREFIX = "/aws/lambda/jev-graphrag-"
LOG_RETENTION_DAYS = 7

# deploy.py の LAMBDA_SPECS から Role と Tags/Publish(作成時の付加)を除いた部分
LAMBDA_SPECS: dict[str, dict[str, Any]] = {
    "ingest": {
        "FunctionName": INGEST_FUNCTION,
        "Runtime": "python3.13",
        "Architectures": ["arm64"],
        "MemorySize": 1024,
        "Timeout": 900,
        "Handler": "handler.lambda_handler",
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
        "Description": "jev-graphrag-poc query (Jev adaptive hop -> Bedrock answer)",
        "Environment": {"Variables": {"AWS_DATA_PATH": "/var/task/botocore_data"}},
        "zip": "query.zip",
    },
}

# Bedrock(infra/iam/policy.json と同じ。回答生成は Haiku 4.5 の global 推論プロファイル)
BEDROCK_INFERENCE_PROFILE = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_FOUNDATION_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

# ============================================================== API Gateway(FULL_REGIONS だけ)

API_NAME = "jev-graphrag-api"
API_DESCRIPTION = "jev-graphrag-poc query API (API key required)"
API_STAGE = "poc"
API_PATH_PART = "query"
API_METHOD = "POST"
API_DEPLOYMENT_DESCRIPTION = "jev-graphrag-poc create-api"
API_KEY_NAME = "jev-graphrag-poc-key"
API_KEY_DESCRIPTION = "jev-graphrag-poc local UI"
USAGE_PLAN_NAME = "jev-graphrag-poc-plan"
USAGE_PLAN_DESCRIPTION = "jev-graphrag-poc: 1 req/s, 200 req/day"
USAGE_THROTTLE = {"rateLimit": 1.0, "burstLimit": 2}
USAGE_QUOTA = {"limit": 200, "period": "DAY"}
INTEGRATION_TIMEOUT_MS = 29000

# ============================================================== スタック

STACK_NAMES: dict[str, str] = {
    HOME_REGION: "JevGraphragPoc",
    "ap-northeast-3": "JevGraphragPocApNortheast3",
    "us-west-2": "JevGraphragPocUsWest2",
}

# CfnOutput のキー(write_api_local.py が読む)
OUTPUT_API_URL = "ApiUrl"
OUTPUT_API_ID = "ApiId"
OUTPUT_API_KEY_ID = "ApiKeyId"


def role_name(region: str) -> str:
    """IAM ロール名はアカウントで 1 つなので、東京以外はリージョンを足して分ける。"""
    return LAMBDA_ROLE if region == HOME_REGION else f"{LAMBDA_ROLE}-{region}"


def lambda_targets(region: str) -> list[str]:
    """そのリージョンに置く Lambda(LAMBDA_SPECS のキー)。FULL_REGIONS は全部、それ以外は query だけ。"""
    return list(LAMBDA_SPECS) if region in FULL_REGIONS else list(REMOTE_LAMBDAS)
