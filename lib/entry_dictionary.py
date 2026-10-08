#!/usr/bin/env python3
"""マスターの正式名・別名の辞書で、文中に出てくるエンティティの候補を拾う(Jev を使わない)。

検索の前段判定と、登録のノード判定の 1 段目で使う。辞書はマスターから組み立てる。
照合の前に質問文と語の両方を `normalize()` でそろえる(NFKC・ひらがな→カタカナ・小文字化・記号の除去)。
部分文字列として含まれる語のエンティティをすべて候補にし、同名の別物の見分けは 2 段目の Jev に任せる。"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

NAKAGURO = "・"
MIN_TERM_CHARS = 2           # 検索の前段判定(query)の既定。1 文字の語は入れない
INGEST_MIN_TERM_CHARS = 1    # 登録のノード判定。1 文字の別名も拾う

_HIRA_START, _HIRA_END = 0x3041, 0x3096   # ぁ〜ゖ
_HIRA_ITER = {0x309D: 0x30FD, 0x309E: 0x30FE}   # ゝゞ → ヽヾ
_KATA_OFFSET = 0x60
_SPACES = re.compile(r"\s+")

# 同じマスターのオブジェクトなら使い回す(Lambda のウォームスタート用)。最小文字数ごとに 1 つずつ持つ
# (query の 2 と ingest の 1 が交互に来ても組み立て直さない)
_cache: dict[int, tuple[Any, dict]] = {}


def _to_katakana(ch: str) -> str:
    code = ord(ch)
    if _HIRA_START <= code <= _HIRA_END:
        return chr(code + _KATA_OFFSET)
    if code in _HIRA_ITER:
        return chr(_HIRA_ITER[code])
    return ch


def _is_removed_symbol(ch: str) -> bool:
    if ch == NAKAGURO:
        return False
    return unicodedata.category(ch)[0] in ("P", "S")


def normalize(text: str) -> str:
    """照合用の正規化(質問文にも辞書の語にも同じものをかける)。"""
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        if ch.isspace():
            out.append(" ")
            continue
        if _is_removed_symbol(ch):
            continue
        out.append(_to_katakana(ch).lower())
    return _SPACES.sub(" ", "".join(out)).strip()


def _check_min_chars(min_chars: int) -> int:
    if isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 1:
        raise ValueError(f"min_chars は 1 以上の整数です: {min_chars!r}")
    return min_chars


def build_dictionary(master: Mapping[str, Any],
                     min_chars: int = MIN_TERM_CHARS) -> dict[str, list[tuple[str, str]]]:
    """`{正規化した語: [(entity_id, マスターに書かれた元の語)]}`。マスターの並び順を保つ。"""
    _check_min_chars(min_chars)
    dictionary: dict[str, list[tuple[str, str]]] = {}
    for entity in master["entities"]:
        for surface in [entity["name"], *(entity.get("aliases") or [])]:
            if not isinstance(surface, str):
                continue
            term = normalize(surface)
            if not term or len(term) < min_chars:
                continue
            pairs = dictionary.setdefault(term, [])
            if (entity["id"], surface) not in pairs:
                pairs.append((entity["id"], surface))
    return dictionary


def get_dictionary(master: Mapping[str, Any],
                   min_chars: int = MIN_TERM_CHARS) -> dict[str, list[tuple[str, str]]]:
    """`build_dictionary()` のキャッシュ付き版。マスターのオブジェクトが変わったら組み立て直す
    (最小文字数ごとに別々にキャッシュする)。"""
    _check_min_chars(min_chars)
    cached = _cache.get(min_chars)
    if cached is None or cached[0] is not master:
        cached = (master, build_dictionary(master, min_chars))
        _cache[min_chars] = cached
    return cached[1]


def find_candidates(question: str, master: Mapping[str, Any],
                    min_chars: int = MIN_TERM_CHARS) -> list[dict]:
    """質問文(登録ではチャンク本文)に辞書の語が部分文字列として含まれるエンティティを全部。
    `[{id, matched_terms}]`(マスターの並び順。matched_terms はマスターに書かれた元の語)。
    `min_chars` は辞書に入れる語の最小文字数(既定 2。登録のノード判定は `INGEST_MIN_TERM_CHARS`=1)。"""
    text = normalize(question)
    hits: dict[str, list[str]] = {}
    for term, pairs in get_dictionary(master, min_chars).items():
        if term in text:
            for eid, surface in pairs:
                terms = hits.setdefault(eid, [])
                if surface not in terms:
                    terms.append(surface)
    order = [e["id"] for e in master["entities"]]
    return [{"id": eid, "matched_terms": hits[eid]} for eid in order if eid in hits]
