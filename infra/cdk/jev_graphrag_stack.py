"""jev-graphrag-poc の AWS リソース一式(CDK)。

`infra/deploy.py` の create-tables / deploy-lambdas / create-api と、README の手順で手作りしていた
IAM ロールを 1 つのスタックにまとめたもの。値はすべて `config.py`(= deploy.py の写し)から取る。

- 東京(ap-northeast-1): テーブル 2 本・ロググループ 2 つ・IAM ロール・Lambda 2 つ・REST API・API キー・使用量プラン
- ap-northeast-3(大阪): 東京と同じ一式。IAM ロール名にリージョンを足し、Lambda 2 つに `SECRET_REGION=東京` を入れる
- us-west-2: テーブル 2 本・query のロググループ・IAM ロール・query Lambda だけ(deploy.py の `--region` と同じ制限)
- Jev の API キー(SSM SecureString)はスタックに入れない。`deploy.py put-secret` で東京に登録し、ここでは
  パラメータ名で IAM の読み取り権限だけを書く
- Lambda の zip は `infra/build.sh` が作った `build/*.zip` をそのまま使う(CDK 側でバンドルし直さない)
- 削除方針は PoC なので全部 DESTROY(テーブルも消える)"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

import config as c

_RUNTIMES = {"python3.13": lambda_.Runtime.PYTHON_3_13}
_ARCHITECTURES = {"arm64": lambda_.Architecture.ARM_64}


class JevGraphragStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, build_dir: Path, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        region = self.region
        account = self.account
        if region not in c.REGIONS:
            raise ValueError(f"リージョンは {c.REGIONS} のどれかです: {region!r}")
        is_home = region == c.HOME_REGION
        is_full = region in c.FULL_REGIONS          # 東京・大阪は一式(Lambda 2 つ・API Gateway)
        targets = c.lambda_targets(region)

        Tags.of(self).add(c.PROJECT_TAG_KEY, c.PROJECT_TAG_VALUE)

        # ------------------------------------------------------------ DynamoDB
        for name, keys in c.TABLE_KEYS.items():
            logical = "ChunkTable" if name == c.CHUNK_TABLE else "GraphTable"
            dynamodb.Table(
                self, logical,
                table_name=name,
                partition_key=dynamodb.Attribute(name=keys[0], type=dynamodb.AttributeType.STRING),
                sort_key=(dynamodb.Attribute(name=keys[1], type=dynamodb.AttributeType.STRING)
                          if len(keys) > 1 else None),
                billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
                removal_policy=RemovalPolicy.DESTROY,
            )

        # ------------------------------------------------------------ IAM(infra/iam/policy.json と同じ文)
        role = iam.Role(
            self, "LambdaRole",
            role_name=c.role_name(region),
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="jev-graphrag-poc Lambda execution role",
            inline_policies={c.LAMBDA_POLICY: self._policy_document(region, account)},
        )

        # ------------------------------------------------------------ Lambda + ロググループ
        functions: dict[str, lambda_.Function] = {}
        for key in targets:
            spec = c.LAMBDA_SPECS[key]
            zip_path = Path(build_dir) / spec["zip"]
            env = dict(((spec.get("Environment") or {}).get("Variables") or {}))
            if not is_home:
                env[c.SECRET_REGION_ENV] = c.HOME_REGION     # SSM は東京のものを読ませる
            title = key.capitalize()

            log_group = logs.LogGroup(
                self, f"{title}LogGroup",
                log_group_name=f"/aws/lambda/{spec['FunctionName']}",
                retention=logs.RetentionDays.ONE_WEEK,       # LOG_RETENTION_DAYS = 7
                removal_policy=RemovalPolicy.DESTROY,
            )
            fn = lambda_.Function(
                self, f"{title}Function",
                function_name=spec["FunctionName"],
                runtime=_RUNTIMES[spec["Runtime"]],
                architecture=_ARCHITECTURES[spec["Architectures"][0]],
                memory_size=spec["MemorySize"],
                timeout=Duration.seconds(spec["Timeout"]),
                handler=spec["Handler"],
                description=spec["Description"],
                role=role,
                environment=env or None,
                code=lambda_.Code.from_asset(str(zip_path)),   # .zip はそのまま上げる(展開・再圧縮しない)
            )
            fn.node.add_dependency(log_group)    # deploy.py と同じく、ロググループを先に作る
            functions[key] = fn

        if not is_full:
            return       # API Gateway は FULL_REGIONS(東京・大阪)だけ

        # ------------------------------------------------------------ API Gateway(REST)
        query_fn = functions["query"]
        api = apigw.RestApi(
            self, "Api",
            rest_api_name=c.API_NAME,
            description=c.API_DESCRIPTION,
            endpoint_configuration=apigw.EndpointConfiguration(types=[apigw.EndpointType.REGIONAL]),
            api_key_source_type=apigw.ApiKeySourceType.HEADER,
            cloud_watch_role=False,          # アカウント設定用の IAM ロールは作らない
            deploy=True,
            deploy_options=apigw.StageOptions(stage_name=c.API_STAGE),
        )
        # 自動で作られるデプロイの説明を deploy.py と同じ文にそろえる
        api.latest_deployment.node.default_child.add_property_override(
            "Description", c.API_DEPLOYMENT_DESCRIPTION)

        integration = apigw.Integration(
            type=apigw.IntegrationType.AWS_PROXY,
            integration_http_method="POST",
            uri=(f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/"
                 f"{query_fn.function_arn}/invocations"),
            options=apigw.IntegrationOptions(timeout=Duration.millis(c.INTEGRATION_TIMEOUT_MS)),
        )
        resource = api.root.add_resource(c.API_PATH_PART)
        resource.add_method(
            c.API_METHOD, integration,
            authorization_type=apigw.AuthorizationType.NONE,
            api_key_required=True,
        )

        # 呼び出し許可: この API のステージ poc の POST /query だけ(テスト呼び出し用の許可は付けない)
        query_fn.add_permission(
            "ApiInvokePermission",
            principal=iam.ServicePrincipal("apigateway.amazonaws.com"),
            action="lambda:InvokeFunction",
            source_arn=(f"arn:aws:execute-api:{region}:{account}:{api.rest_api_id}/"
                        f"{c.API_STAGE}/{c.API_METHOD}/{c.API_PATH_PART}"),
            source_account=account,
        )

        api_key = apigw.ApiKey(
            self, "ApiKey",
            api_key_name=c.API_KEY_NAME,
            description=c.API_KEY_DESCRIPTION,
            enabled=True,
        )
        plan = api.add_usage_plan(
            "UsagePlan",
            name=c.USAGE_PLAN_NAME,
            description=c.USAGE_PLAN_DESCRIPTION,
            throttle=apigw.ThrottleSettings(rate_limit=c.USAGE_THROTTLE["rateLimit"],
                                            burst_limit=c.USAGE_THROTTLE["burstLimit"]),
            quota=apigw.QuotaSettings(limit=c.USAGE_QUOTA["limit"],
                                      period=apigw.Period[c.USAGE_QUOTA["period"]]),
            api_stages=[apigw.UsagePlanPerApiStage(api=api, stage=api.deployment_stage)],
        )
        plan.add_api_key(api_key)

        # ------------------------------------------------------------ 出力(キーの値は出さない)
        CfnOutput(self, c.OUTPUT_API_URL,
                  value=(f"https://{api.rest_api_id}.execute-api.{region}.amazonaws.com/"
                         f"{c.API_STAGE}/{c.API_PATH_PART}"),
                  description="POST /query の URL(x-api-key ヘッダー必須)")
        CfnOutput(self, c.OUTPUT_API_ID, value=api.rest_api_id, description="REST API の ID")
        CfnOutput(self, c.OUTPUT_API_KEY_ID, value=api_key.key_id,
                  description="API キーの ID(値ではない。値は write_api_local.py が .api.local.json に書く)")

    @staticmethod
    def _policy_document(region: str, account: str) -> iam.PolicyDocument:
        """`infra/iam/policy.json` の `<ACCOUNT_ID>` と東京のリージョンを、このスタックの値に置き換えたもの。
        SSM だけは東京固定(大阪・us-west-2 の Lambda も東京の SSM を読む)。ワイルドカードは policy.json にある分だけ。"""
        table_arns = [f"arn:aws:dynamodb:{region}:{account}:table/{name}" for name in c.TABLE_KEYS]
        param = c.SECRET_PARAM.lstrip("/")
        return iam.PolicyDocument(statements=[
            iam.PolicyStatement(
                sid="Logs",
                actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[f"arn:aws:logs:{region}:{account}:log-group:{c.LOG_GROUP_PREFIX}*:*"],
            ),
            iam.PolicyStatement(
                sid="DynamoDB",
                actions=["dynamodb:PutItem", "dynamodb:BatchWriteItem", "dynamodb:DeleteItem",
                         "dynamodb:Query", "dynamodb:GetItem", "dynamodb:BatchGetItem", "dynamodb:Scan"],
                resources=table_arns,
            ),
            iam.PolicyStatement(
                sid="ApiKey",
                actions=["ssm:GetParameter"],
                resources=[f"arn:aws:ssm:{c.HOME_REGION}:{account}:parameter/{param}"],
            ),
            iam.PolicyStatement(
                sid="Bedrock",
                actions=["bedrock:InvokeModel"],
                resources=[
                    f"arn:aws:bedrock:{region}:{account}:inference-profile/{c.BEDROCK_INFERENCE_PROFILE}",
                    f"arn:aws:bedrock:*::foundation-model/{c.BEDROCK_FOUNDATION_MODEL}",
                    f"arn:aws:bedrock:::foundation-model/{c.BEDROCK_FOUNDATION_MODEL}",
                ],
            ),
        ])
