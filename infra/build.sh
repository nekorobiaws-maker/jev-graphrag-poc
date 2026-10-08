#!/usr/bin/env bash
# Lambda の zip を作る。AWS は呼ばない。
#
#   bash infra/build.sh            # ingest と query の両方(= all)
#   bash infra/build.sh ingest     # build/ingest.zip だけ
#   bash infra/build.sh query      # build/query.zip だけ
#
# zip のルートに平らに置くもの:
#   - typesafe-sdk とその依存(Lambda python3.13 / arm64 用のホイール)
#   - lib/*.py、lambdas/<target>/handler.py、master.json、chunks.json
#     (データは JEV_DATASET の版。zip の中では版名なしの名前にする)
#   - query だけ: .venv の botocore の DynamoDB API 定義(botocore_data/。Lambda は AWS_DATA_PATH で読む)
#     検証時と同じ zip にするため残している(本手法は SearchVectors を使わない)
# boto3 / botocore はランタイム同梱のものを使うので入れない。
# Lambda は root 以外のユーザーで動くので、ファイル 644・ディレクトリ 755 にそろえてから zip にする。
set -euo pipefail

TARGET="${1:-all}"
case "$TARGET" in
  ingest) TARGETS=(ingest) ;;
  query)  TARGETS=(query) ;;
  all)    TARGETS=(ingest query) ;;
  *) echo "!! 引数は ingest / query / all のどれかです: $TARGET" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
BUILD="$ROOT/build"
PKG="$BUILD/pkg"

if [[ ! -x "$PY" ]]; then
  echo "!! $PY がありません(python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt)" >&2
  exit 1
fi

# どの版を入れるかは lib/common.py の dataset_paths() に決めさせ、中身も確かめてから入れる
DATA_INFO="$(cd "$ROOT/lib" && "$PY" - <<'PYEOF'
from common import DATASET, dataset_paths
from master import load_chunks, load_master
m, c = dataset_paths()
master, chunks = load_master(m), load_chunks(c)
print(DATASET, m, c, len(master["entities"]), len(chunks))
PYEOF
)"
read -r DATASET MASTER_SRC CHUNKS_SRC N_ENT N_CHUNKS <<< "$DATA_INFO"
echo "== データ: $DATASET(エンティティ $N_ENT 件 / チャンク $N_CHUNKS 件)"
echo "   $MASTER_SRC"
echo "   $CHUNKS_SRC"

# 手元と同じ版の SDK を入れる
SDK_VERSION="$("$PY" -c 'import importlib.metadata as m; print(m.version("typesafe-sdk"))')"
echo "== typesafe-sdk==$SDK_VERSION を manylinux2014_aarch64 / cp313 向けに取得"

rm -rf "$PKG"
mkdir -p "$PKG"

"$PY" -m pip install --quiet --disable-pip-version-check \
  --platform manylinux2014_aarch64 --only-binary=:all: \
  --python-version 3.13 --implementation cp \
  -t "$PKG" "typesafe-sdk==$SDK_VERSION"

# boto3 / botocore はランタイム同梱を使う。依存で紛れ込んでいたら止める
for name in boto3 botocore; do
  if [[ -e "$PKG/$name" ]]; then
    echo "!! $name が依存に紛れ込んでいます。ランタイム同梱版とぶつかるので止めます" >&2
    exit 1
  fi
done

# lib/*.py と同名のトップレベルが依存側にあると、平らに置いたときに片方が隠れる
for src in "$ROOT"/lib/*.py; do
  mod="$(basename "$src" .py)"
  if [[ -e "$PKG/$mod" || -e "$PKG/$mod.py" ]]; then
    echo "!! lib/$mod.py と同名のモジュールが依存側にあります: $PKG/$mod" >&2
    exit 1
  fi
done

build_one() {
T="$1"
STAGE="$BUILD/stage-$T"
ZIP="$BUILD/$T.zip"
rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"
echo
echo "######## $T"
echo "== 作業ディレクトリに集める"
cp -R "$PKG"/. "$STAGE"/
cp "$ROOT"/lib/*.py "$STAGE"/
cp "$ROOT/lambdas/$T/handler.py" "$STAGE/handler.py"
cp "$MASTER_SRC" "$STAGE/master.json"
cp "$CHUNKS_SRC" "$STAGE/chunks.json"
if [[ "$T" == "query" ]]; then
  echo "== DynamoDB の API 定義(.venv の botocore)を botocore_data/ に入れる"
  "$PY" - "$STAGE/botocore_data" <<'PYEOF'
import gzip, json, os, shutil, sys
import botocore

src = os.path.join(os.path.dirname(botocore.__file__), "data", "dynamodb", "2012-08-10")
dst = os.path.join(sys.argv[1], "dynamodb", "2012-08-10")
os.makedirs(dst, exist_ok=True)
for name in sorted(os.listdir(src)):
    path = os.path.join(src, name)
    if name.endswith(".json.gz"):
        with gzip.open(path, "rb") as f_in, open(os.path.join(dst, name[:-3]), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    elif name.endswith(".json"):
        shutil.copyfile(path, os.path.join(dst, name))
service = json.load(open(os.path.join(dst, "service-2.json"), encoding="utf-8"))
rules = open(os.path.join(dst, "endpoint-rule-set-1.json"), encoding="utf-8").read()
if "SearchVectors" not in service.get("operations", {}):
    sys.exit(f"!! botocore {botocore.__version__} の service-2.json に SearchVectors がありません")
if "IsSearchOperation" not in rules:
    sys.exit("!! endpoint-rule-set-1.json に IsSearchOperation(検索用エンドポイントの振り分け)がありません")
total = sum(os.path.getsize(os.path.join(dst, n)) for n in os.listdir(dst))
print(f"  botocore {botocore.__version__}: {', '.join(sorted(os.listdir(dst)))}(計 {total:,} bytes)")
PYEOF
fi
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGE" -name '*.pyc' -delete
find "$STAGE" -type d -exec chmod 755 {} +
find "$STAGE" -type f -exec chmod 644 {} +

echo "== zip を作る: $ZIP"
(cd "$STAGE" && zip -q -r -X "$ZIP" .)

echo
echo "== zip のルートに置いたファイル(依存のディレクトリは下にまとめて表示)"
"$PY" - "$ZIP" "$T" "$MASTER_SRC" "$CHUNKS_SRC" <<'PYEOF'
import hashlib, sys, zipfile
from collections import Counter

zf = zipfile.ZipFile(sys.argv[1])
names = [i.filename for i in zf.infolist() if not i.is_dir()]
root_files = sorted(n for n in names if "/" not in n)
dirs = Counter(n.split("/", 1)[0] for n in names if "/" in n)
for n in root_files:
    info = zf.getinfo(n)
    mode = (info.external_attr >> 16) & 0o777
    print(f"  {n:<28} {info.file_size:>8,} bytes  {oct(mode)}")
print("  -- 依存(トップレベルのディレクトリ: ファイル数)")
for d, c in sorted(dirs.items()):
    print(f"  {d + '/':<40} {c:>5}")
print(f"  合計 {len(names):,} ファイル")

target = sys.argv[2]
required = {"handler.py", "master.json", "chunks.json", "common.py", "graph_store.py",
            "aws_secrets.py", "jev_client.py", "jev_questions.py", "master.py", "ratelimit.py",
            "cache.py"}
required |= {"ingest": {"ingest_core.py"},
             "query": {"query_core.py", "bedrock_llm.py", "entry_dictionary.py", "budget.py"}}[target]
missing = sorted(required - set(root_files))
if missing:
    sys.exit(f"!! zip のルートに足りないファイルがあります: {missing}")
marker = f'FUNCTION_NAME = "jev-graphrag-{target}"'
if marker not in zf.read("handler.py").decode("utf-8"):
    sys.exit(f"!! handler.py が {target} 用ではありません({marker} が見つからない)")
unreadable = [n for n in names if ((zf.getinfo(n).external_attr >> 16) & 0o004) == 0]
if unreadable:
    sys.exit(f"!! 他ユーザーが読めないファイルがあります: {unreadable[:5]}")
for bad in ("boto3/", "botocore/"):
    if any(n.startswith(bad) for n in names):
        sys.exit(f"!! {bad} が入っています")
data_files = sorted(n for n in names if n.startswith("botocore_data/"))
if target == "query":
    want = "botocore_data/dynamodb/2012-08-10/service-2.json"
    if want not in data_files or "SearchVectors" not in zf.read(want).decode("utf-8"):
        sys.exit(f"!! {want} が無いか、SearchVectors がありません")
    if "botocore_data/dynamodb/2012-08-10/endpoint-rule-set-1.json" not in data_files:
        sys.exit("!! endpoint-rule-set-1.json がありません(検索用エンドポイントに振り分けられない)")
    extra = [n for n in data_files if not n.startswith("botocore_data/dynamodb/2012-08-10/")]
    if extra:
        sys.exit(f"!! botocore_data に DynamoDB 以外が入っています: {extra[:5]}")
    print(f"  botocore_data: {len(data_files)} ファイル・{sum(zf.getinfo(n).compress_size for n in data_files):,} bytes(圧縮後)")
elif data_files:
    sys.exit(f"!! {target} に botocore_data は要りません")
for inner, src in (("master.json", sys.argv[3]), ("chunks.json", sys.argv[4])):
    got = hashlib.sha256(zf.read(inner)).hexdigest()
    want = hashlib.sha256(open(src, "rb").read()).hexdigest()
    if got != want:
        sys.exit(f"!! zip の {inner} が {src} と違います")
    print(f"  {inner} = {src.rsplit('/', 1)[-1]}(sha256 {got[:12]}…)")
print(f"  必須ファイル・{target} 用ハンドラー・読み取り権限・boto3 不在・データの版・API 定義: OK")
PYEOF

echo
echo "== pydantic_core の C 拡張が aarch64 用か"
SO_LIST="$(find "$STAGE/pydantic_core" -name '_pydantic_core*.so' 2>/dev/null || true)"
if [[ -z "$SO_LIST" ]]; then
  echo "!! pydantic_core の .so が見つかりません" >&2
  exit 1
fi
status=0
while IFS= read -r so; do
  desc="$(file -b "$so")"
  echo "  $(basename "$so")"
  echo "    $desc"
  if [[ "$(basename "$so")" != *cpython-313-aarch64-linux-gnu* ]] || [[ "$desc" != *aarch64* ]]; then
    echo "!! aarch64 / cp313 用ではありません" >&2
    status=1
  fi
done <<< "$SO_LIST"
[[ $status -eq 0 ]] || exit 1
echo "  OK(ELF / aarch64 / cpython-313)"

echo
ls -l "$ZIP"
echo "== $T: できました(データ $DATASET)。次は: ./.venv/bin/python infra/deploy.py deploy-lambdas --only $T"
}

for t in "${TARGETS[@]}"; do
  build_one "$t"
done
