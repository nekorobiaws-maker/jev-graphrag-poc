#!/usr/bin/env python3
"""CDK でデプロイした API の URL とキーを `.api.local.json`(権限 600・git 除外)に書く。

    ./.venv/bin/python infra/cdk/write_api_local.py      # リポジトリ直下の .venv(boto3 入り)で実行
    JEV_REGION=ap-northeast-3 ./.venv/bin/python infra/cdk/write_api_local.py   # 大阪のスタック(--region でも可)

- リージョン: `--region` → 環境変数 `JEV_REGION` → 既定 東京。API があるのは `config.FULL_REGIONS`(東京・大阪)だけ
- そのリージョンのスタック(東京 `JevGraphragPoc`、大阪 `JevGraphragPocApNortheast3`)の出力から
  URL・API の ID・API キーの ID を読む
- `.api.local.json` は 1 つだけなので、別のリージョンで実行すると上書きされる
- キーの値は `apigateway get-api-key --include-value` で取り、ファイルに書くだけ。**表示しない**
- 書く形式は `infra/deploy.py create-api` と同じ(`ui/server.py` がそのまま読める)
- 最初に `common.check_account()` で AWS アカウントを確かめる(環境変数 `JEV_AWS_ACCOUNT_ID`)"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
for _p in (str(HERE), str(PROJECT_ROOT / "lib")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import config as c  # noqa: E402
from common import REGION, boto_config, check_account, error_code  # noqa: E402

API_LOCAL_PATH = PROJECT_ROOT / ".api.local.json"


def stack_outputs(session, stack_name: str, region: str = c.HOME_REGION) -> dict[str, str]:
    cfn = session.client("cloudformation", region_name=region, config=boto_config(max_attempts=3))
    stacks = cfn.describe_stacks(StackName=stack_name).get("Stacks") or []
    if not stacks:
        raise RuntimeError(f"スタック {stack_name} が見つかりません")
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs") or []}


def write_api_local(url: str, api_id: str, key_value: str, path: Path = API_LOCAL_PATH,
                    region: str = c.HOME_REGION) -> Path:
    data = {"url": url, "api_key": key_value, "api_id": api_id, "stage": c.API_STAGE, "region": region}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.chmod(path, 0o600)            # 既にあったファイルでも 600 にそろえる
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", choices=list(c.FULL_REGIONS), default=None,
                    help=f"スタックのリージョン。省略時は環境変数 JEV_REGION(既定 {c.HOME_REGION})")
    args = ap.parse_args(argv)
    region = args.region or REGION
    if region not in c.FULL_REGIONS:        # AWS を呼ぶ前に止める(us-west-2 には API が無い)
        print(f"!! {region} には API がありません(API があるのは {', '.join(c.FULL_REGIONS)})")
        return 1

    import boto3

    session = boto3.Session(region_name=region)
    check_account(session, region)
    stack_name = c.STACK_NAMES[region]
    try:
        outputs = stack_outputs(session, stack_name, region)
    except Exception as exc:  # noqa: BLE001
        print(f"!! スタック {stack_name} の出力を読めませんでした: {type(exc).__name__} / {error_code(exc) or '(コード無し)'}")
        return 1
    missing = [k for k in (c.OUTPUT_API_URL, c.OUTPUT_API_ID, c.OUTPUT_API_KEY_ID) if not outputs.get(k)]
    if missing:
        print(f"!! スタックの出力に {missing} がありません")
        return 1
    url, api_id, key_id = outputs[c.OUTPUT_API_URL], outputs[c.OUTPUT_API_ID], outputs[c.OUTPUT_API_KEY_ID]

    apigw = session.client("apigateway", region_name=region, config=boto_config(max_attempts=3))
    try:
        value = apigw.get_api_key(apiKey=key_id, includeValue=True).get("value")
    except Exception as exc:  # noqa: BLE001
        print(f"!! API キーを読めませんでした: {type(exc).__name__} / {error_code(exc) or '(コード無し)'}")
        return 1
    if not value:
        print("!! API キーの値が取れませんでした")
        return 1
    path = write_api_local(url, api_id, value, region=region)
    print(f"URL : {url}")
    print(f"{path.name} に URL とキーを書きました(権限 600。キーの値は表示しません)")
    print("使用量プランの反映には数十秒かかることがあります(直後は 403 になる場合あり)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
