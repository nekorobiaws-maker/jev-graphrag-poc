#!/usr/bin/env python3
"""Jev の疎通確認(1 リクエスト、1 セント未満)。

- サーバが答えた `response.model` が `jev-1.13.0` か(違えば終了コード 1)
- usage が取れ、呼び出しレコードが `results/cache/jev.jsonl` に残るか

実行: `./.venv/bin/python scripts/00_smoke_jev.py`"""

from __future__ import annotations

import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent.parent / "lib"
_lib = str(LIB_DIR)
if _lib in sys.path:
    sys.path.remove(_lib)
sys.path.insert(0, _lib)   # 同名モジュールが他所にあっても lib/ を優先させる

from budget import Budget  # noqa: E402
from common import (  # noqa: E402
    JEV_MODEL,
    append_jsonl,
    cache_path,
    ensure_dirs,
    load_api_key,
    utc_now_iso,
)
from jev_client import JevClient  # noqa: E402
from jev_questions import noul_value  # noqa: E402
from ratelimit import TokenBucket  # noqa: E402

LABEL = "smoke_jev"
N_CALLS = 1
EST_USD = 0.001   # 前段判定 2 問 × 数百トークン × $0.042/1M。実際は 1 セントの 1/100 程度

STATE = {"question": "工藤新一は現在、誰の家に居候し、どこを拠点に事件を解決しているか？"}

QUESTIONS: dict[str, dict] = {
    "in_scope": {
        "type": "noul",
        "instructions": (
            "この質問は、漫画・アニメ作品（ONE PIECE、名探偵コナン、名探偵プリキュア!）の"
            "登場人物・設定についての質問か"
        ),
        "criteria": {
            "true": "作品の登場人物・設定・出来事など、作品の内容についての質問",
            "false": "作品と無関係な質問",
        },
    },
    "node_shinichi": {
        "type": "noul",
        "instructions": (
            "質問に『工藤新一』（別名: 新一／種別: キャラクター／高校生探偵）そのものが登場するか"
        ),
        "criteria": {
            "true": "質問がこのキャラクター本人に言及している",
            "false": "言及していない、または同名・部分一致の別物",
        },
    },
}


def main() -> int:
    ensure_dirs()
    budget = Budget()
    budget.announce(N_CALLS, EST_USD, LABEL)
    budget.check(EST_USD, LABEL)

    # キャッシュは使わない(疎通確認なので毎回実リクエストを投げる)。記録は下で追記する
    with JevClient(api_key=load_api_key(), bucket=TokenBucket(1000)) as client:
        try:
            record = client.judge(STATE, QUESTIONS, use_cache=False)
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status", None)
            print(f"[{LABEL}] 呼び出し失敗: {type(exc).__name__} status={status}", flush=True)
            return 2

    record["ts"] = utc_now_iso()
    out_path = cache_path("jev")
    append_jsonl(out_path, record)

    served_model = (record.get("response") or {}).get("model")
    model_ok = served_model == JEV_MODEL
    usage = record.get("usage") or {}

    print(f"[{LABEL}] 結果", flush=True)
    print(f"  response.model : {served_model} -> {'OK' if model_ok else 'NG'} (期待値 {JEV_MODEL})")
    for qid in QUESTIONS:
        value = noul_value(record, qid)
        shown = "取得できず" if value is None else f"{value:.4f}"
        print(f"  {qid:<14} : {shown}")
    print(f"  input_tokens   : {usage.get('input_tokens')}")
    print(f"  output_tokens  : {usage.get('output_tokens')}")
    print(f"  latency_ms     : {record.get('latency_ms', 0.0):.1f}")
    print(f"  retries        : {record.get('retries')}")
    print(f"  記録先          : {out_path}")
    print(f"  累計費用        : ${Budget().spent():.6f}", flush=True)

    if not model_ok:
        print(f"[{LABEL}] モデルの版が {JEV_MODEL} ではありません。以降のステップは止めてください", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
