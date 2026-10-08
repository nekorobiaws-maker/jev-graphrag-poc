#!/usr/bin/env python3
"""CDK アプリの入口。

    npx aws-cdk@2 synth                        # 東京(ap-northeast-1)。スタック JevGraphragPoc
    npx aws-cdk@2 synth -c region=ap-northeast-3   # 大阪(東京と同じ一式)。スタック JevGraphragPocApNortheast3
    npx aws-cdk@2 synth -c region=us-west-2    # us-west-2(query とテーブルだけ)。スタック JevGraphragPocUsWest2

- リージョン: context `region` → 環境変数 `JEV_REGION` → 既定 ap-northeast-1
- アカウント: 環境変数 `JEV_AWS_ACCOUNT_ID` → `CDK_DEFAULT_ACCOUNT`(CDK CLI が認証情報から入れる)。
  両方あって食い違っていたら止める(deploy.py の check_account と同じ考え方)
- zip の場所: context `build_dir` → 既定はリポジトリ直下の build/(infra/build.sh の出力)
- lookups は使わない(synth に AWS の認証は要らない)"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import aws_cdk as cdk

import config as c
from jev_graphrag_stack import JevGraphragStack

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent


def fail(message: str) -> None:
    print(f"!! {message}", file=sys.stderr)
    sys.exit(1)


app = cdk.App()

region = (app.node.try_get_context("region") or os.environ.get("JEV_REGION") or c.HOME_REGION).strip()
if region not in c.REGIONS:
    fail(f"リージョンは {c.REGIONS} のどれかです: {region!r}")

jev_account = (os.environ.get("JEV_AWS_ACCOUNT_ID") or "").strip()
cdk_account = (os.environ.get("CDK_DEFAULT_ACCOUNT") or "").strip()
if jev_account and cdk_account and jev_account != cdk_account:
    fail(f"AWS アカウントが違います: JEV_AWS_ACCOUNT_ID={jev_account} / 認証情報のアカウント={cdk_account}")
account = jev_account or cdk_account
if not account:
    fail("環境変数 JEV_AWS_ACCOUNT_ID に AWS アカウント ID を設定してください"
         "(synth だけならダミーの 12 桁でよい)")

build_dir = Path(app.node.try_get_context("build_dir") or PROJECT_ROOT / "build").resolve()
targets = c.lambda_targets(region)
missing = [str(build_dir / c.LAMBDA_SPECS[k]["zip"]) for k in targets
           if not (build_dir / c.LAMBDA_SPECS[k]["zip"]).is_file()]
if missing:
    fail(f"zip がありません: {', '.join(missing)}。先にリポジトリ直下で bash infra/build.sh を実行してください")

JevGraphragStack(
    app, c.STACK_NAMES[region],
    build_dir=build_dir,
    env=cdk.Environment(account=account, region=region),
    description="jev-graphrag-poc (DynamoDB / Lambda / API Gateway). Jev API key in SSM is managed outside this stack.",
)

app.synth()
