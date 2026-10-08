#!/usr/bin/env python3
"""CDK 側の値が `infra/deploy.py`・`infra/iam/policy.json`・Lambda のコードと食い違っていないかを確かめる。
AWS は呼ばない(boto3 も要らない)。

    ./.venv/bin/python check_parity.py                                         # config.py と deploy.py / lib の突き合わせ
    ./.venv/bin/python check_parity.py --template cdk.out/JevGraphragPoc.template.json   # 生成テンプレートも突き合わせる

テンプレートのリージョンはファイル名(`config.STACK_NAMES` のスタック名)から決める。

テンプレートを見るときは、synth と同じ `JEV_AWS_ACCOUNT_ID` を環境変数に入れておく(policy.json の
`<ACCOUNT_ID>` をこれで置き換えて比べる)。食い違いがあれば一覧を出して終了コード 1。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for p in (str(ROOT / "infra"), str(ROOT / "lib"), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import config as c  # noqa: E402
import deploy  # noqa: E402   infra/deploy.py(import しても AWS は呼ばない)
import aws_secrets  # noqa: E402
import common  # noqa: E402
import graph_store  # noqa: E402

problems: list[str] = []


def same(label: str, got: Any, want: Any) -> None:
    if got != want:
        problems.append(f"{label}: CDK={got!r} / 期待={want!r}")


def check_config() -> None:
    # deploy.py と
    for name in ("CHUNK_TABLE", "GRAPH_TABLE", "TABLE_KEYS", "SECRET_PARAM", "LAMBDA_ROLE",
                 "INGEST_FUNCTION", "QUERY_FUNCTION", "LOG_GROUP_PREFIX", "LOG_RETENTION_DAYS",
                 "API_NAME", "API_STAGE", "API_PATH_PART", "API_METHOD", "API_KEY_NAME",
                 "USAGE_PLAN_NAME", "USAGE_THROTTLE", "USAGE_QUOTA", "INTEGRATION_TIMEOUT_MS",
                 "REMOTE_LAMBDAS", "SECRET_REGION_ENV"):
        same(f"deploy.{name}", getattr(c, name), getattr(deploy, name))
    same("deploy.PROJECT_TAG", {"Key": c.PROJECT_TAG_KEY, "Value": c.PROJECT_TAG_VALUE}, deploy.PROJECT_TAG)
    for key, spec in deploy.LAMBDA_SPECS.items():
        want = {k: v for k, v in spec.items() if k != "Role"}
        same(f"deploy.LAMBDA_SPECS[{key}]", c.LAMBDA_SPECS.get(key), want)
    same("LAMBDA_SPECS のキー", sorted(c.LAMBDA_SPECS), sorted(deploy.LAMBDA_SPECS))
    # Lambda のコード・lib と
    same("lib/common.HOME_REGION", c.HOME_REGION, common.HOME_REGION)
    same("lib/common.REGIONS", c.REGIONS, common.REGIONS)
    same("lib/common.INGEST_REGIONS", c.FULL_REGIONS, common.INGEST_REGIONS)
    same("lib/graph_store.CHUNK_TABLE", c.CHUNK_TABLE, graph_store.CHUNK_TABLE)
    same("lib/graph_store.GRAPH_TABLE", c.GRAPH_TABLE, graph_store.GRAPH_TABLE)
    same("lib/aws_secrets.SECRET_PARAM", c.SECRET_PARAM, aws_secrets.SECRET_PARAM)
    same("lib/aws_secrets.SECRET_REGION_ENV", c.SECRET_REGION_ENV, aws_secrets.SECRET_REGION_ENV)
    same("lib/common.GEN_MODELS[haiku]", c.BEDROCK_INFERENCE_PROFILE, common.GEN_MODELS["haiku"])
    for rel, const, want in (("scripts/20_invoke_ingest.py", "INGEST_FUNCTION", c.INGEST_FUNCTION),
                             ("scripts/21_invoke_query.py", "QUERY_FUNCTION", c.QUERY_FUNCTION),
                             ("lambdas/ingest/handler.py", "FUNCTION_NAME", c.INGEST_FUNCTION),
                             ("lambdas/query/handler.py", "FUNCTION_NAME", c.QUERY_FUNCTION)):
        m = re.search(rf'^{const} = "([^"]+)"', (ROOT / rel).read_text(encoding="utf-8"), re.M)
        same(f"{rel} の {const}", want, m.group(1) if m else None)
    # deploy.py の文字列(説明文など。deploy.py では関数の中に直書きなのでソースから拾う)
    src = (ROOT / "infra" / "deploy.py").read_text(encoding="utf-8")
    for label, text in (("API の説明", c.API_DESCRIPTION), ("デプロイの説明", c.API_DEPLOYMENT_DESCRIPTION),
                        ("API キーの説明", c.API_KEY_DESCRIPTION), ("使用量プランの説明", c.USAGE_PLAN_DESCRIPTION)):
        if f'"{text}"' not in src:
            problems.append(f"{label} {text!r} が deploy.py に見つかりません")
    for frag in ('types": ["REGIONAL"]', 'apiKeySource="HEADER"', 'authorizationType="NONE", apiKeyRequired=True',
                 'type="AWS_PROXY", integrationHttpMethod="POST"', 'Principal="apigateway.amazonaws.com"',
                 "SourceAccount=ACCOUNT_ID", '"BillingMode": "PAY_PER_REQUEST"', '"AttributeType": "S"'):
        if frag not in src:
            problems.append(f"deploy.py に {frag!r} が見つかりません(写した前提が変わっている)")


def by_type(template: dict, rtype: str) -> list[dict]:
    return [r.get("Properties", {}) for r in template["Resources"].values() if r["Type"] == rtype]


def expected_policy(region: str, account: str) -> dict:
    """policy.json の <ACCOUNT_ID> を埋め、東京以外ならリージョンを置き換える(SSM は東京のまま)。"""
    doc = json.loads((ROOT / "infra" / "iam" / "policy.json").read_text(encoding="utf-8"))
    for stmt in doc["Statement"]:
        res = stmt["Resource"]
        res_list = res if isinstance(res, list) else [res]
        fixed = []
        for arn in res_list:
            arn = arn.replace("<ACCOUNT_ID>", account)
            if stmt.get("Sid") != "ApiKey":
                arn = arn.replace(f":{c.HOME_REGION}:", f":{region}:")
            fixed.append(arn)
        stmt["Resource"] = fixed if isinstance(res, list) else fixed[0]
    return doc


def region_of_template(path: Path) -> str:
    """`cdk.out/<スタック名>.template.json` のスタック名からリージョンを引く。"""
    stack = path.name.removesuffix(".template.json")
    for region, name in c.STACK_NAMES.items():
        if name == stack:
            return region
    raise SystemExit(f"!! {path.name} のスタック名が config.STACK_NAMES にありません")


def expected_lambda(key: str, region: str) -> dict:
    """期待する関数の設定。東京と us-west-2 は deploy.lambda_spec そのもの。大阪の ingest は deploy.py では
    置けないので、deploy.py の LAMBDA_SPECS に `SECRET_REGION=東京` を足したもの(query と同じ足し方)。"""
    if region == c.HOME_REGION or key in deploy.REMOTE_LAMBDAS:
        return deploy.lambda_spec(key, region)
    spec = dict(deploy.LAMBDA_SPECS[key])
    env = dict(((spec.get("Environment") or {}).get("Variables") or {}))
    env[deploy.SECRET_REGION_ENV] = c.HOME_REGION
    spec["Environment"] = {"Variables": env}
    return spec


def check_template(path: Path, account: str) -> None:
    t = json.loads(path.read_text(encoding="utf-8"))
    types = sorted({r["Type"] for r in t["Resources"].values()})
    region = region_of_template(path)
    is_full = region in c.FULL_REGIONS
    print(f"テンプレート: {path.name}(リージョン {region})")

    # IAM: ロールは 1 つ・インラインポリシー 1 つ・別立てのポリシーは無し
    roles = by_type(t, "AWS::IAM::Role")
    same("IAM ロールの数", len(roles), 1)
    for rtype in ("AWS::IAM::Policy", "AWS::IAM::ManagedPolicy", "AWS::ApiGateway::Account"):
        if rtype in types:
            problems.append(f"想定外のリソース {rtype} があります")
    if roles:
        role = roles[0]
        same("ロール名", role.get("RoleName"), c.role_name(region))
        same("信頼ポリシー", role.get("AssumeRolePolicyDocument"),
             json.loads((ROOT / "infra" / "iam" / "trust.json").read_text(encoding="utf-8")))
        same("管理ポリシーの付与", role.get("ManagedPolicyArns"), None)
        policies = role.get("Policies") or []
        same("インラインポリシーの数", len(policies), 1)
        if policies:
            same("インラインポリシー名", policies[0]["PolicyName"], c.LAMBDA_POLICY)
            same("インラインポリシーの中身(policy.json と)", policies[0]["PolicyDocument"],
                 expected_policy(region, account))

    # DynamoDB
    tables = {p["TableName"]: p for p in by_type(t, "AWS::DynamoDB::Table")}
    same("テーブル名", sorted(tables), sorted(c.TABLE_KEYS))
    for name, p in tables.items():
        want = deploy.create_table_params(name)
        for k in ("AttributeDefinitions", "KeySchema", "BillingMode"):
            same(f"{name}.{k}", p.get(k), want[k])
    for lid, r in t["Resources"].items():
        if r["Type"] in ("AWS::DynamoDB::Table", "AWS::Logs::LogGroup"):
            same(f"{lid} の DeletionPolicy", r.get("DeletionPolicy"), "Delete")

    # ロググループ
    targets = c.lambda_targets(region)
    groups = {p["LogGroupName"]: p for p in by_type(t, "AWS::Logs::LogGroup")}
    same("ロググループ", sorted(groups),
         sorted(f"/aws/lambda/{c.LAMBDA_SPECS[k]['FunctionName']}" for k in targets))
    for name, p in groups.items():
        same(f"{name} の保持日数", p.get("RetentionInDays"), deploy.LOG_RETENTION_DAYS)

    # Lambda(期待値は expected_lambda。東京以外は SECRET_REGION が足される)
    fns = {p["FunctionName"]: p for p in by_type(t, "AWS::Lambda::Function")}
    same("Lambda 関数", sorted(fns), sorted(c.LAMBDA_SPECS[k]["FunctionName"] for k in targets))
    for key in targets:
        want = expected_lambda(key, region)
        got = fns.get(want["FunctionName"], {})
        for k in ("Runtime", "Architectures", "MemorySize", "Timeout", "Handler", "Description"):
            same(f"{want['FunctionName']}.{k}", got.get(k), want[k])
        same(f"{want['FunctionName']}.Environment", got.get("Environment"), want.get("Environment"))
        for k in ("Layers", "VpcConfig", "TracingConfig", "LoggingConfig", "DeadLetterConfig",
                  "ReservedConcurrentExecutions"):
            same(f"{want['FunctionName']}.{k}(付けていないこと)", got.get(k), None)

    if not is_full:
        for rtype in types:
            if rtype.startswith("AWS::ApiGateway::") or rtype == "AWS::Lambda::Permission":
                problems.append(f"{region} に {rtype} があります(API は {', '.join(c.FULL_REGIONS)} だけ)")
        return

    # API Gateway
    apis = by_type(t, "AWS::ApiGateway::RestApi")
    same("REST API の数", len(apis), 1)
    api = apis[0] if apis else {}
    same("API 名", api.get("Name"), c.API_NAME)
    same("API の説明", api.get("Description"), c.API_DESCRIPTION)
    same("エンドポイント", api.get("EndpointConfiguration"), {"Types": ["REGIONAL"]})
    same("apiKeySource", api.get("ApiKeySourceType"), "HEADER")
    resources = by_type(t, "AWS::ApiGateway::Resource")
    same("リソース", [r.get("PathPart") for r in resources], [c.API_PATH_PART])
    methods = by_type(t, "AWS::ApiGateway::Method")
    same("メソッドの数", len(methods), 1)
    m = methods[0] if methods else {}
    same("メソッド", m.get("HttpMethod"), c.API_METHOD)
    same("API キー必須", m.get("ApiKeyRequired"), True)
    same("認可", m.get("AuthorizationType"), "NONE")
    integ = m.get("Integration") or {}
    same("統合の種類", integ.get("Type"), "AWS_PROXY")
    same("統合の HTTP メソッド", integ.get("IntegrationHttpMethod"), "POST")
    same("統合のタイムアウト", integ.get("TimeoutInMillis"), c.INTEGRATION_TIMEOUT_MS)
    uri = integ.get("Uri")
    uri_parts = (uri or {}).get("Fn::Join", ["", []])[1] if isinstance(uri, dict) else [uri]
    uri_literal = "".join(x if isinstance(x, str) else "{fn}" for x in uri_parts)
    same("統合の URI", uri_literal, f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/{{fn}}/invocations")
    stages = by_type(t, "AWS::ApiGateway::Stage")
    same("ステージ", [s.get("StageName") for s in stages], [c.API_STAGE])
    deps = by_type(t, "AWS::ApiGateway::Deployment")
    same("デプロイの説明", [d.get("Description") for d in deps], [c.API_DEPLOYMENT_DESCRIPTION])
    keys = by_type(t, "AWS::ApiGateway::ApiKey")
    same("API キー", [(k.get("Name"), k.get("Enabled"), k.get("Description")) for k in keys],
         [(c.API_KEY_NAME, True, c.API_KEY_DESCRIPTION)])
    if keys and "Value" in keys[0]:
        problems.append("API キーの値がテンプレートに書かれています")
    plans = by_type(t, "AWS::ApiGateway::UsagePlan")
    same("使用量プランの数", len(plans), 1)
    plan = plans[0] if plans else {}
    same("使用量プラン名", plan.get("UsagePlanName"), c.USAGE_PLAN_NAME)
    same("使用量プランの説明", plan.get("Description"), c.USAGE_PLAN_DESCRIPTION)
    same("スロットル", plan.get("Throttle"),
         {"RateLimit": c.USAGE_THROTTLE["rateLimit"], "BurstLimit": c.USAGE_THROTTLE["burstLimit"]})
    same("クォータ", plan.get("Quota"), {"Limit": c.USAGE_QUOTA["limit"], "Period": c.USAGE_QUOTA["period"]})
    same("使用量プランのステージ数", len(plan.get("ApiStages") or []), 1)
    same("使用量プランとキーの紐づけ", len(by_type(t, "AWS::ApiGateway::UsagePlanKey")), 1)

    perms = by_type(t, "AWS::Lambda::Permission")
    same("呼び出し許可の数(テスト呼び出し用は無し)", len(perms), 1)
    if perms:
        p = perms[0]
        same("許可の Action", p.get("Action"), "lambda:InvokeFunction")
        same("許可の Principal", p.get("Principal"), "apigateway.amazonaws.com")
        same("許可の SourceAccount", p.get("SourceAccount"), account)
        parts = (p.get("SourceArn") or {}).get("Fn::Join", ["", []])[1]
        literal = "".join(x if isinstance(x, str) else "{api_id}" for x in parts)
        same("許可の SourceArn", literal,
             f"arn:aws:execute-api:{region}:{account}:{{api_id}}/{c.API_STAGE}/{c.API_METHOD}/{c.API_PATH_PART}")

    outputs = t.get("Outputs") or {}
    for k in (c.OUTPUT_API_URL, c.OUTPUT_API_ID, c.OUTPUT_API_KEY_ID):
        if k not in outputs:
            problems.append(f"出力 {k} がありません")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--template", type=Path, action="append", default=[],
                    help="cdk.out の *.template.json(繰り返し指定可)")
    args = ap.parse_args()
    check_config()
    if args.template:
        account = (os.environ.get("JEV_AWS_ACCOUNT_ID") or "").strip()
        if not account:
            print("!! テンプレートを見るときは JEV_AWS_ACCOUNT_ID を synth と同じ値で入れてください")
            return 1
        for path in args.template:
            check_template(path, account)
    if problems:
        print(f"!! 食い違い {len(problems)} 件")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("OK: deploy.py・policy.json・Lambda のコードの決め打ちと一致しています")
    return 0


if __name__ == "__main__":
    sys.exit(main())
