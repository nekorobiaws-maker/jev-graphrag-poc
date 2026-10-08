#!/usr/bin/env python3
"""マスターを DynamoDB(`jev-graphrag-graph` の `PK=MASTER`)に入れる。

    ./.venv/bin/python scripts/60_put_master.py --status                         # 今の版を見るだけ
    ./.venv/bin/python scripts/60_put_master.py --file data/master_v3.json       # 差分を見て Enter で書く
    ./.venv/bin/python scripts/60_put_master.py --file data/master_v3.json --yes

Lambda は呼び出しごとに版を見て読み直すので、デプロイし直さずに反映される。
ただしグラフは古いマスターで判定したままなので、エンティティやエッジ種別を変えたら登録(`20_invoke_ingest.py --reset`)をやり直す。
書き込みは 新しい行を put → 消えた行を delete → 最後に META の順。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

LIB_DIR = Path(__file__).resolve().parent.parent / "lib"
_lib = str(LIB_DIR)
if _lib in sys.path:
    sys.path.remove(_lib)
sys.path.insert(0, _lib)   # 同名モジュールが他所にあっても lib/ を優先させる

from common import REGION, boto_config, check_account, utc_now_iso, require_home_region  # noqa: E402
from master import (  # noqa: E402
    MASTER_PATH,
    MasterInconsistent,
    file_sha256,
    load_master,
    load_master_from_store,
    master_version,
    normalize_edge_type,
)

LABEL = "put master"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="マスターを DynamoDB(PK=MASTER)に入れる")
    parser.add_argument("--file", type=Path, default=MASTER_PATH,
                        help=f"入れるマスターの JSON(既定 {MASTER_PATH.relative_to(MASTER_PATH.parent.parent)})")
    parser.add_argument("--status", action="store_true", help="DynamoDB の今の版と手元の差分を見るだけ(書かない)")
    parser.add_argument("--yes", action="store_true", help="実行前の Enter 確認を飛ばす")
    return parser.parse_args(argv)


# ============================================================== 差分(純粋関数)

def _changed_fields(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    return [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]


def diff_masters(old: Mapping[str, Any] | None, new: Mapping[str, Any]) -> dict:
    """旧(DynamoDB。無ければ None)→ 新(ファイル)の差分。"""
    old = old or {"entities": [], "edge_types": [], "node_types": []}
    o_ent = {e["id"]: e for e in old["entities"]}
    n_ent = {e["id"]: e for e in new["entities"]}
    o_nt = {t["name"]: t for t in old["node_types"]}
    n_nt = {t["name"]: t for t in new["node_types"]}
    o_et = [normalize_edge_type(e) for e in old["edge_types"]]
    n_et = [normalize_edge_type(e) for e in new["edge_types"]]
    return {
        "entities": {
            "added": [i for i in n_ent if i not in o_ent],
            "removed": [i for i in o_ent if i not in n_ent],
            "changed": [{"id": i, "fields": _changed_fields(o_ent[i], n_ent[i])}
                        for i in n_ent if i in o_ent and o_ent[i] != n_ent[i]],
        },
        "edge_types": {
            "added": [{"index": i, "name": e["name"]} for i, e in enumerate(n_et) if i >= len(o_et)],
            "removed": [{"index": i, "name": e["name"]} for i, e in enumerate(o_et) if i >= len(n_et)],
            "changed": [{"index": i, "fields": _changed_fields(o_et[i], n_et[i]),
                         "old_name": o_et[i]["name"], "new_name": n_et[i]["name"]}
                        for i in range(min(len(o_et), len(n_et))) if o_et[i] != n_et[i]],
        },
        "node_types": {
            "added": [n for n in n_nt if n not in o_nt],
            "removed": [n for n in o_nt if n not in n_nt],
            "changed": [{"name": n, "fields": _changed_fields(o_nt[n], n_nt[n])}
                        for n in n_nt if n in o_nt and o_nt[n] != n_nt[n]],
        },
    }


def is_empty_diff(diff: Mapping[str, Any]) -> bool:
    return not any(v for part in diff.values() for v in part.values())


def format_diff(diff: Mapping[str, Any], new: Mapping[str, Any], limit: int = 30) -> list[str]:
    names = {e["id"]: e["name"] for e in new["entities"]}
    lines: list[str] = []

    def shown(items: list[str]) -> str:
        more = "" if len(items) <= limit else f" …ほか {len(items) - limit} 件"
        return ", ".join(items[:limit]) + more

    ent = diff["entities"]
    lines.append(f"  エンティティ : 追加 {len(ent['added'])} / 削除 {len(ent['removed'])} / 変更 {len(ent['changed'])}")
    if ent["added"]:
        added = ["{}({})".format(i, names.get(i, i)) for i in ent["added"]]
        lines.append(f"    + {shown(added)}")
    if ent["removed"]:
        lines.append(f"    - {shown(ent['removed'])}")
    if ent["changed"]:
        changed = ["{}[{}]".format(c["id"], ",".join(c["fields"])) for c in ent["changed"]]
        lines.append(f"    ~ {shown(changed)}")
    et = diff["edge_types"]
    lines.append(f"  エッジ種別   : 追加 {len(et['added'])} / 削除 {len(et['removed'])} / 変更 {len(et['changed'])}")
    sym_new = {i for i, e in enumerate(new["edge_types"]) if e.get("symmetric") is True}
    for a in et["added"]:
        lines.append(f"    + [{a['index']}] {a['name']}{'(対称)' if a['index'] in sym_new else ''}")
    for r in et["removed"]:
        lines.append(f"    - [{r['index']}] {r['name']}")
    for c in et["changed"]:
        rename = f" {c['old_name']} → {c['new_name']}" if c["old_name"] != c["new_name"] else f" {c['new_name']}"
        sym = ""
        if "symmetric" in c["fields"]:
            sym = " 対称: なし → あり" if c["index"] in sym_new else " 対称: あり → なし"
        lines.append(f"    ~ [{c['index']}]{rename}[{','.join(c['fields'])}]{sym}")
    nt = diff["node_types"]
    lines.append(f"  ノード種別   : 追加 {len(nt['added'])} / 削除 {len(nt['removed'])} / 変更 {len(nt['changed'])}")
    for n in nt["added"]:
        lines.append(f"    + {n}")
    for n in nt["removed"]:
        lines.append(f"    - {n}")
    for c in nt["changed"]:
        lines.append(f"    ~ {c['name']}[{','.join(c['fields'])}]")
    return lines


def confirm() -> bool:
    try:
        input("Enter で書き込み、Ctrl-C で中止: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return True


# ============================================================== 本体

def read_current(store: Any) -> tuple[dict | None, dict | None, str | None]:
    """DynamoDB の今のマスター。`(master, meta, 注意)`。無ければ (None, None, None)。
    中身と META が合わないときは、行から組み立てた中身で差分を出す。"""
    from graph_store import master_from_items

    try:
        loaded = load_master_from_store(store)
    except MasterInconsistent as exc:
        items = store.query_master_items()
        return master_from_items(items), store.get_master_meta(), f"{exc}(今回の書き込みで直ります)"
    if loaded is None:
        return None, None, None
    return loaded[0], loaded[1], None


REINGEST_COMMAND = "./.venv/bin/python scripts/20_invoke_ingest.py --reset"


def ingest_warnings(diff: Mapping[str, Any], old: Mapping[str, Any] | None, new: Mapping[str, Any]) -> list[str]:
    """マスターを入れ替えたあと、グラフ(登場・関係の行)の再登録が要るかの注意書き(純粋関数)。"""
    if old is None:
        return ["  注意: DynamoDB にマスターが無かったので、今のグラフは同梱の master.json で登録したものです。"
                f"手元と違う版で登録していたなら、グラフの再登録が要ります({REINGEST_COMMAND})"]
    lines: list[str] = []
    et, ent = diff["edge_types"], diff["entities"]
    gone = sorted({e["name"] for e in old["edge_types"]} - {e["name"] for e in new["edge_types"]})
    if gone:
        lines.append(f"  注意: 関係 {'、'.join(gone)} は手元のマスターにありません。登録済みのグラフのその行は、"
                     "検索でテンプレートが引けず黙って飛ばされます")
    if any("symmetric" in c["fields"] for c in et["changed"]):
        lines.append("  注意: 対称(symmetric)を変えた関係は、再登録するまで古い形(OUT/IN か SYM)の行のままです")
    if et["added"] or et["removed"] or et["changed"] or ent["removed"]:
        lines.append("  注意: エッジ種別の変更(添字・名前・テンプレート・対称)やエンティティの削除があるので、"
                     f"グラフの再登録が必要です: {REINGEST_COMMAND}")
    elif ent["added"] or ent["changed"]:
        lines.append("  注意: 新しい・変えたエンティティをグラフに反映するには再登録が要ります"
                     f"({REINGEST_COMMAND})")
    return lines


def run(store: Any, path: Path, *, status_only: bool, yes: bool, ask=confirm,
        now=utc_now_iso) -> int:
    """`check_account()` の後の本体(DynamoDB の store は引数で受け取る)。"""
    from graph_store import master_meta_item

    new = load_master(path)
    version = master_version(new)
    old, meta, note = read_current(store)
    print(f"[{LABEL}] 手元     : {path}(版 {version[:8]}、エンティティ {len(new['entities'])} / "
          f"エッジ種別 {len(new['edge_types'])} / ノード種別 {len(new['node_types'])})")
    if meta is None:
        print(f"[{LABEL}] DynamoDB : マスターなし(Lambda は同梱の master.json を使っている)")
    else:
        print(f"[{LABEL}] DynamoDB : 版 {meta['version'][:8]}、エンティティ {meta['entities']} / "
              f"エッジ種別 {meta['edge_types']} / ノード種別 {meta['node_types']}(更新 {meta['updated_at']})")
    if note:
        print(f"[{LABEL}] 注意     : {note}")
    diff = diff_masters(old, new)
    same = meta is not None and meta["version"] == version and note is None
    print(f"[{LABEL}] 差分(DynamoDB → 手元)")
    for line in format_diff(diff, new):
        print(line)
    for line in ingest_warnings(diff, old, new):
        print(line)
    if status_only:
        print(f"[{LABEL}] {'一致しています' if same else '手元と違います'}(--status なので書きません)")
        return 0
    if same:
        print(f"[{LABEL}] 同じ版なので書きません")
        return 0
    if not yes and not ask():
        print(f"[{LABEL}] 中止しました(何も書いていません)", flush=True)
        return 1
    meta_item = master_meta_item(version=version, entities=len(new["entities"]),
                                 node_types=len(new["node_types"]), edge_types=len(new["edge_types"]),
                                 updated_at=now(), source_sha256=file_sha256(path))
    result = store.write_master(new, meta_item)
    print(f"[{LABEL}] 書きました: put {result['put']} 行(META 込み)/ delete {result['deleted']} 行")
    loaded = load_master_from_store(store)
    if loaded is None or loaded[1]["version"] != version:
        print(f"[{LABEL}] !! 読み戻した版が一致しません", flush=True)
        return 2
    print(f"[{LABEL}] 読み戻し OK(版 {version[:8]})。Lambda は次の呼び出しから新しい版を使います")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_master(args.file)                      # AWS に触れる前に形を確かめる

    import boto3

    session = boto3.Session(region_name=REGION)
    require_home_region(__file__.rsplit('/', 1)[-1])  # 東京専用(JEV_REGION が東京以外なら止める)
    check_account(session)                      # JEV_AWS_ACCOUNT_ID と違えばここで例外

    from graph_store import GraphStore

    store = GraphStore(session.resource("dynamodb", region_name=REGION, config=boto_config(max_attempts=5)))
    return run(store, args.file, status_only=args.status, yes=args.yes)


if __name__ == "__main__":
    sys.exit(main())
