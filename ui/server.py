#!/usr/bin/env python3
"""検索画面のローカルサーバー(Python 標準ライブラリだけ)。

    ./.venv/bin/python ui/server.py          # http://127.0.0.1:8765 を開く
    ./.venv/bin/python ui/server.py --demo   # API を呼ばず、results/query/ の保存結果を返す

- `127.0.0.1` だけで待ち受け、Host / Origin も確かめる
- `GET /api/master` はマスターを DynamoDB から読み、読めなければ同梱の `data/master_<版>.json`
- `POST /api/query` は `.api.local.json` の URL に `x-api-key` を付けて転送する。API キーはブラウザに渡さない
- 応答の呼び出しレコードを `results/cache/` に追記して予算に数え、上限を超えそうなら転送しない"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
LIB_DIR = PROJECT_ROOT / "lib"
_lib = str(LIB_DIR)
if _lib in sys.path:
    sys.path.remove(_lib)
sys.path.insert(0, _lib)   # 同名モジュールが他所にあっても lib/ を優先させる

from budget import Budget, BudgetExceeded, cost_of  # noqa: E402
from common import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_GEN_MODEL,
    JEV_MODEL,
    PRICES,
    RESULTS_DIR,
    append_jsonl,
    cache_path,
    dataset_paths,
)

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
API_LOCAL_PATH = PROJECT_ROOT / ".api.local.json"
FORWARD_TIMEOUT_SEC = 35.0
MAX_BODY_BYTES = 4096
PER_QUERY_GUARD_USD = 0.05
ALLOWED_QUERY_KEYS = frozenset({"question", "mode"})
UI_MODES = ("adaptive", "fixed1", "fixed2")

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/graph_model.js": ("graph_model.js", "text/javascript; charset=utf-8"),
}
_MASTER_FILE, _CHUNKS_FILE = dataset_paths()
DATA_FILES = {
    "/api/master": _MASTER_FILE.name,     # master_<版>.json(JEV_DATASET で切り替え)
    "/api/chunks": _CHUNKS_FILE.name,
}


def bundled_master_loader(data_dir: Path) -> Callable[[], tuple[dict, dict]]:
    """同梱ファイルだけを読むマスターの読み込み(AWS に触れない。デモ用の既定)。"""
    from master import MasterCache

    cache = MasterCache(Path(data_dir) / DATA_FILES["/api/master"])
    return cache.bundled


def dynamodb_master_loader(data_dir: Path, store_factory: Callable[[], Any] | None = None,
                           log: Callable[[str], None] | None = None) -> Callable[[], tuple[dict, dict]]:
    """DynamoDB から読み、読めなければ同梱ファイルを返すマスターの読み込み。
    store は最初に使うときに作る(認証情報が無ければ以後も同梱ファイル)。"""
    from master import MasterCache

    log = log or (lambda msg: sys.stderr.write(msg + "\n"))
    cache = MasterCache(Path(data_dir) / DATA_FILES["/api/master"], log=log)
    box: dict[str, Any] = {"store": None}

    def make_store() -> Any:
        if store_factory is not None:
            return store_factory()
        from graph_store import GraphStore
        return GraphStore()

    def load() -> tuple[dict, dict]:
        try:
            if box["store"] is None:
                box["store"] = make_store()
            return cache.get(box["store"])
        except Exception as exc:  # noqa: BLE001
            log(f"[ui] DynamoDB のマスターを読めませんでした({type(exc).__name__})。同梱ファイルを使います")
            master, info = cache.bundled()
            return master, {**info, "error": type(exc).__name__}

    return load


class ApiConfigMissing(RuntimeError):
    """`.api.local.json` が無い・壊れている。"""


def load_api_config(path: Path) -> dict:
    """URL とキーを読む。**キーの値はこの dict の外に出さない**。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ApiConfigMissing(f"{Path(path).name} がありません(deploy.py create-api で作ります)") from None
    except (OSError, json.JSONDecodeError):
        raise ApiConfigMissing(f"{Path(path).name} が読めません") from None
    url, key = data.get("url"), data.get("api_key")
    if not isinstance(url, str) or not url.startswith("https://") or not isinstance(key, str) or not key:
        raise ApiConfigMissing(f"{Path(path).name} に url / api_key がありません")
    return {"url": url, "api_key": key}


def record_costs(records: Any, cache_dir: Path | None) -> dict:
    """呼び出しレコードをキャッシュ(jev.jsonl / gen.jsonl)に追記し、費用を返す。"""
    totals = {"jev": 0.0, "gen": 0.0}
    n = 0
    unknown = 0
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        kind = "jev" if record.get("kind") == "jev" else "gen"
        append_jsonl(cache_path(kind, cache_dir), dict(record))
        n += 1
        try:
            totals[kind] += cost_of(dict(record))
        except ValueError:                      # 単価が未登録のモデル(数えられないので件数だけ)
            unknown += 1
    return {"jev_usd": totals["jev"], "gen_usd": totals["gen"], "total_usd": totals["jev"] + totals["gen"],
            "n_records": n, "unpriced_records": unknown}


def estimate_costs(records: Any, tokens: Any) -> dict:
    """デモモード用の費用の目安(キャッシュには書かない)。レコードがあれば `cost_of()` で、無ければ
    トレースの `tokens`(jev_input / bedrock_input / bedrock_output)と PRICES(Jev・既定の生成モデル)で出す。
    戻り値: `{jev_usd, gen_usd, total_usd, basis: "records" | "tokens" | "none"}`"""
    totals = {"jev": 0.0, "gen": 0.0}
    priced = 0
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        try:
            totals["jev" if record.get("kind") == "jev" else "gen"] += cost_of(dict(record))
            priced += 1
        except ValueError:
            continue
    basis = "records" if priced else "none"
    if not priced and isinstance(tokens, Mapping):
        def num(key: str) -> float:
            value = tokens.get(key)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
        jev_p, gen_p = PRICES[JEV_MODEL], PRICES[DEFAULT_GEN_MODEL]
        totals["jev"] = num("jev_input") * jev_p["input"] / 1e6
        totals["gen"] = num("bedrock_input") * gen_p["input"] / 1e6 + num("bedrock_output") * gen_p["output"] / 1e6
        basis = "tokens"
    return {"jev_usd": totals["jev"], "gen_usd": totals["gen"], "total_usd": totals["jev"] + totals["gen"],
            "basis": basis}


def validate_query(body: Any) -> dict:
    """画面から来た body を確かめる(Lambda 側でも確かめるが、無駄な転送をしないため)。"""
    if not isinstance(body, Mapping):
        raise ValueError("JSON オブジェクトを送ってください")
    unknown = sorted(set(body) - ALLOWED_QUERY_KEYS)
    if unknown:
        raise ValueError(f"受け付けないキーがあります: {unknown}")
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("質問を入力してください")
    if len(question.strip()) > 200:
        raise ValueError("質問は 200 文字までです")
    mode = body.get("mode", "adaptive")
    if mode not in UI_MODES:
        raise ValueError(f"mode は {list(UI_MODES)} のどれかです")
    return {"question": question.strip(), "mode": mode}


def forward(url: str, api_key: str, payload: Mapping[str, Any], *,
            timeout: float = FORWARD_TIMEOUT_SEC,
            opener: Callable[..., Any] = urllib.request.urlopen) -> tuple[int, bytes]:
    """API Gateway に転送する。戻り値は (ステータス, 本文)。例外の本文は返さない(種類だけ)。"""
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "x-api-key": api_key, "Accept": "application/json"})
    try:
        with opener(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() or b""
    except (socket.timeout, TimeoutError):
        return HTTPStatus.GATEWAY_TIMEOUT, json.dumps(
            {"message": f"API の応答が {timeout:.0f} 秒で返りませんでした"}, ensure_ascii=False).encode("utf-8")
    except urllib.error.URLError as exc:
        reason = type(exc.reason).__name__ if not isinstance(exc.reason, str) else "URLError"
        return HTTPStatus.BAD_GATEWAY, json.dumps(
            {"message": f"API に接続できませんでした({reason})"}, ensure_ascii=False).encode("utf-8")


def demo_response(demo_dir: Path, mode: str, question: str) -> dict | None:
    """過去の実測(`results/query/*.json` の runs[mode].body)から、**質問文が完全に同じ**ものの最新 1 件。
    無ければ None(別の質問の結果を取り違えて見せないよう、フォールバックはしない)。"""
    files = sorted(Path(demo_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        body = ((saved.get("runs") or {}).get(mode) or {}).get("body")
        if not isinstance(body, dict) or not body.get("ok"):
            continue
        if saved.get("question") == question:
            return {**body, "_demo_source": path.name}
    return None


def make_handler(*, api_local_path: Path = API_LOCAL_PATH, cache_dir: Path | None = None,
                 data_dir: Path = DATA_DIR, ui_dir: Path = UI_DIR,
                 opener: Callable[..., Any] = urllib.request.urlopen,
                 budget_factory: Callable[[], Budget] | None = None,
                 demo_dir: Path | None = None,
                 master_loader: Callable[[], tuple[dict, dict]] | None = None) -> type[BaseHTTPRequestHandler]:
    """設定を閉じ込めたハンドラークラスを作る。"""
    budget_factory = budget_factory or (lambda: Budget(cache_dir or CACHE_DIR))
    master_loader = master_loader or bundled_master_loader(data_dir)

    class Handler(BaseHTTPRequestHandler):
        server_version = "jev-graphrag-ui/1"

        # ---- 共通
        def _allowed_hosts(self) -> set[str]:
            port = self.server.server_address[1]
            return {f"127.0.0.1:{port}", f"localhost:{port}"}

        def _host_ok(self) -> bool:
            return (self.headers.get("Host") or "") in self._allowed_hosts()

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            return origin in {f"http://{h}" for h in self._allowed_hosts()}

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def log_message(self, fmt: str, *args: Any) -> None:     # パスとステータスだけ(キーは出ない)
            sys.stderr.write(f"[ui] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")

        # ---- GET
        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"message": "Host が違います"})
            path = self.path.split("?", 1)[0]
            if path in STATIC:
                name, ctype = STATIC[path]
                try:
                    return self._send(HTTPStatus.OK, (ui_dir / name).read_bytes(), ctype)
                except FileNotFoundError:
                    return self._json(HTTPStatus.NOT_FOUND, {"message": f"{name} がありません"})
            if path == "/api/master":
                try:
                    master, info = master_loader()
                except FileNotFoundError:
                    return self._json(HTTPStatus.NOT_FOUND, {"message": f"{DATA_FILES[path]} がありません"})
                except ValueError as exc:
                    return self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                                      {"message": f"マスターの形が違います: {exc}"})
                return self._json(HTTPStatus.OK, {**master, "_master": info})
            if path in DATA_FILES:
                try:
                    raw = (Path(data_dir) / DATA_FILES[path]).read_bytes()
                except FileNotFoundError:
                    return self._json(HTTPStatus.NOT_FOUND, {"message": f"{DATA_FILES[path]} がありません"})
                return self._send(HTTPStatus.OK, raw, "application/json; charset=utf-8")
            if path == "/api/questions":
                try:
                    qs = json.loads((Path(data_dir) / "questions_v2.json").read_text(encoding="utf-8"))
                except FileNotFoundError:
                    qs = []
                keep = ("id", "type", "question", "answer", "required_chunks")
                return self._json(HTTPStatus.OK, [{k: q.get(k) for k in keep} for q in qs])
            if path == "/api/budget":
                b = budget_factory()
                return self._json(HTTPStatus.OK, {"spent_usd": b.spent(), "limit_usd": b.limit_usd})
            return self._json(HTTPStatus.NOT_FOUND, {"message": "ありません"})

        # ---- POST
        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok() or not self._origin_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"message": "このサーバーの画面以外からは呼べません"})
            if self.path.split("?", 1)[0] != "/api/query":
                return self._json(HTTPStatus.NOT_FOUND, {"message": "ありません"})
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"message": "JSON で送ってください"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._json(HTTPStatus.BAD_REQUEST, {"message": "本文の長さが不正です"})
            try:
                payload = validate_query(json.loads(self.rfile.read(length)))
            except (ValueError, UnicodeDecodeError) as exc:
                msg = str(exc) if not isinstance(exc, json.JSONDecodeError) else "JSON が読めません"
                return self._json(HTTPStatus.BAD_REQUEST, {"message": msg})
            if demo_dir is not None:
                data = demo_response(demo_dir, payload["mode"], payload["question"])
                if data is None:
                    return self._json(HTTPStatus.NOT_FOUND, {
                        "message": f"この質問のデモ用データはありません(mode {payload['mode']})。"
                                   "デモモードでは results/query に保存済みの質問だけ表示できます"})
                cost_est = estimate_costs(data.pop("records", None), data.get("tokens"))
                b = budget_factory()
                data["_local"] = {"upstream_status": 200, "wall_ms": 0.0, "demo": True,
                                  "cost": {"jev_usd": 0.0, "gen_usd": 0.0, "total_usd": 0.0, "n_records": 0},
                                  "cost_est": cost_est,
                                  "spent_usd": b.spent(), "limit_usd": b.limit_usd,
                                  "requested_mode": payload["mode"]}
                return self._json(HTTPStatus.OK, data)
            try:
                conf = load_api_config(api_local_path)
            except ApiConfigMissing as exc:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"message": str(exc)})
            budget = budget_factory()
            try:
                budget.check(PER_QUERY_GUARD_USD, "ui")
            except BudgetExceeded as exc:
                return self._json(HTTPStatus.PAYMENT_REQUIRED, {"message": str(exc)})

            started = time.perf_counter()
            status, raw = forward(conf["url"], conf["api_key"], payload, opener=opener)
            wall_ms = (time.perf_counter() - started) * 1000.0
            try:
                data = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {"message": f"API の応答が JSON ではありません({len(raw)} bytes)"}
            if not isinstance(data, dict):
                data = {"message": "API の応答の形が想定外です"}
            # 課金レコードを先に残す(失敗応答でも課金ぶんは数える)
            cost = record_costs(data.pop("records", None), cache_dir)
            data["_local"] = {"upstream_status": status, "wall_ms": wall_ms, "cost": cost,
                              "spent_usd": budget_factory().spent(), "limit_usd": budget.limit_usd,
                              "requested_mode": payload["mode"]}
            return self._json(status if 200 <= status < 600 else HTTPStatus.BAD_GATEWAY, data)

    return Handler


def make_server(port: int = DEFAULT_PORT, **kwargs: Any) -> ThreadingHTTPServer:
    """127.0.0.1 だけで待ち受けるサーバー(port=0 なら空いている番号)。"""
    return ThreadingHTTPServer((HOST, port), make_handler(**kwargs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="検索画面のローカルサーバー(127.0.0.1 のみ)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--demo", action="store_true",
                        help="API を呼ばず results/query/*.json の過去の結果を返す(費用ゼロ。画面確認用)")
    args = parser.parse_args(argv)
    if args.demo:
        server = make_server(args.port, demo_dir=RESULTS_DIR / "query")
        print(f"デモモード: API は呼びません(results/query の保存結果を返します)")
        print(f"http://{HOST}:{server.server_address[1]} を開いてください(Ctrl-C で終了)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
        finally:
            server.server_close()
        return 0
    try:
        conf = load_api_config(API_LOCAL_PATH)
        print(f"API: {conf['url']}(キーは .api.local.json から。表示しません)")
        mode = API_LOCAL_PATH.stat().st_mode & 0o777
        if mode != 0o600:
            print(f"!! {API_LOCAL_PATH.name} の権限が {oct(mode)} です。chmod 600 にしてください")
    except ApiConfigMissing as exc:
        print(f"!! {exc}。画面は開けますが、検索はできません")
    server = make_server(args.port, master_loader=dynamodb_master_loader(DATA_DIR))
    print("マスター: DynamoDB(jev-graphrag-graph の PK=MASTER)から読みます。読めなければ同梱ファイル")
    print(f"http://{HOST}:{server.server_address[1]} を開いてください(Ctrl-C で終了)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
