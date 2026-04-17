#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Idiom Parallel Corpus (v2 - split literal vs meaning)

Input:  Fijian Idioms - Copy.txt (and/or any *.txt in folder that match)
Output (same folder):
  - idiom_parallel_fj2en__ALL_SPLIT.csv   (two rows per idiom when possible)
  - idiom_parallel_en2fj__ALL_SPLIT.csv
  - idiom_parallel_both__ALL_SPLIT.csv

Schema (merge-friendly):
  domain, subdomain, source_id, source_doc, source_lang, target_lang, direction, source_text, target_text

Defaults:
  - lowercase
  - keep punctuation
  - remove digits
"""

from __future__ import annotations
import re
from pathlib import Path
import unicodedata as ud
import pandas as pd

MASTER_COLS = [
    "domain","subdomain","source_id","source_doc",
    "source_lang","target_lang","direction",
    "source_text","target_text"
]

MOJIBAKE_MAP = {
    "â€œ": "“",
    "â€": "”",
    "â€˜": "‘",
    "â€™": "’",
    "â€“": "–",
    "â€”": "—",
    "â€¦": "…",
    "Â": "",
}

WS_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\d+")

def fix_mojibake(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\ufeff","")
    for k,v in MOJIBAKE_MAP.items():
        s = s.replace(k,v)
    return ud.normalize("NFKC", s)

def normalize_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()

def clean_text(s: str) -> str:
    s = fix_mojibake(s)
    s = normalize_ws(s)
    s = DIGIT_RE.sub("", s)
    s = normalize_ws(s)
    s = s.lower()
    s = normalize_ws(s)
    return s

def split_english_fields(raw: str) -> tuple[str|None, str|None]:
    """
    Try to split EN target into (literal, meaning).
    Handles patterns like:
      "literal: ... meaning: ..."
      "literal - ...; meaning - ..."
      "literal translation: ... / meaning: ..."
      "... | ..."  (fallback: treat first part literal, second meaning)
    If no split possible: returns (raw, None)
    """
    t = fix_mojibake(raw).strip()
    if not t:
        return (None, None)

    # labeled patterns
    m = re.search(r"(literal(?:\s+translation)?\s*[:\-])", t, flags=re.IGNORECASE)
    n = re.search(r"(meaning\s*[:\-])", t, flags=re.IGNORECASE)

    if m and n and m.start() < n.start():
        lit = t[m.end():n.start()].strip(" ;/|,-")
        mea = t[n.end():].strip(" ;/|,-")
        return (lit or None, mea or None)

    # common unlabeled separators
    for sep in [" | ", " / ", " ; ", " – ", " - "]:
        if sep in t:
            a,b = t.split(sep, 1)
            a = a.strip()
            b = b.strip()
            # only accept split if both look non-trivial
            if len(a) >= 3 and len(b) >= 3:
                return (a, b)

    return (t, None)

def read_lines(p: Path) -> list[str]:
    raw = p.read_text(encoding="utf-8", errors="ignore")
    raw = fix_mojibake(raw)
    lines = []
    for ln in raw.splitlines():
        ln = normalize_ws(ln)
        if ln:
            lines.append(ln)
    return lines

def extract_pairs_from_lines(lines: list[str]) -> list[dict]:
    """
    Best-effort extraction: looks for blocks containing a Fijian idiom
    with English literal/meaning nearby.

    Because formats differ, this extractor is conservative:
    - it looks for lines that contain BOTH a Fijian-looking segment and an English segment split by tab/colon/dash.
    - if your file uses a strict template, adjust the regex here once and it will become perfect.
    """
    rows = []
    # This pattern catches common "fj - en" or "fj: en" lines
    fj_en_pat = re.compile(r"^(?P<fj>.+?)(?:\s*[:\-–]\s+|\t+)(?P<en>.+)$")

    idx = 0
    for ln in lines:
        m = fj_en_pat.match(ln)
        if not m:
            continue
        fj_raw = m.group("fj")
        en_raw = m.group("en")

        fj = clean_text(fj_raw)
        en = fix_mojibake(en_raw).strip()

        if not fj or not en:
            continue

        literal, meaning = split_english_fields(en)

        idx += 1
        base_id = f"idiom|idioms|{idx:06d}"

        # literal row (always if present)
        if literal:
            rows.append({
                "domain":"idiom",
                "subdomain":"idiom_literal",
                "source_id": base_id + "|literal",
                "source_doc":"(merged)",
                "source_lang":"fj",
                "target_lang":"en",
                "direction":"fj->en",
                "source_text": fj,
                "target_text": clean_text(literal),
            })

        # meaning row (only if present)
        if meaning:
            rows.append({
                "domain":"idiom",
                "subdomain":"idiom_meaning",
                "source_id": base_id + "|meaning",
                "source_doc":"(merged)",
                "source_lang":"fj",
                "target_lang":"en",
                "direction":"fj->en",
                "source_text": fj,
                "target_text": clean_text(meaning),
            })

    return rows

def main():
    folder = Path(__file__).resolve().parent
    txts = sorted(folder.glob("*.txt"))
    if not txts:
        raise SystemExit("No .txt found in the script folder.")

    all_rows = []
    for p in txts:
        lines = read_lines(p)
        rows = extract_pairs_from_lines(lines)
        # update source_doc for each row
        for r in rows:
            r["source_doc"] = p.name
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No idiom pairs extracted. If format is different, paste 10 sample lines and I'll tune the regex.")

    fj2en = pd.DataFrame(all_rows)[MASTER_COLS]

    # Build reverse direction (EN->FJ) while preserving label via subdomain
    en2fj = fj2en.copy()
    en2fj["source_lang"] = "en"
    en2fj["target_lang"] = "fj"
    en2fj["direction"] = "en->fj"
    en2fj = en2fj.rename(columns={"source_text":"target_text_tmp","target_text":"source_text"})
    en2fj = en2fj.rename(columns={"target_text_tmp":"target_text"})  # swap
    en2fj = en2fj[MASTER_COLS]

    # Deduplicate within each direction
    fj2en = fj2en.drop_duplicates(subset=["direction","subdomain","source_text","target_text"]).reset_index(drop=True)
    en2fj = en2fj.drop_duplicates(subset=["direction","subdomain","source_text","target_text"]).reset_index(drop=True)

    both = pd.concat([fj2en, en2fj], ignore_index=True)

    out_fj2en = folder / "idiom_parallel_fj2en__ALL_SPLIT.csv"
    out_en2fj = folder / "idiom_parallel_en2fj__ALL_SPLIT.csv"
    out_both = folder / "idiom_parallel_both__ALL_SPLIT.csv"

    fj2en.to_csv(out_fj2en, index=False, encoding="utf-8")
    en2fj.to_csv(out_en2fj, index=False, encoding="utf-8")
    both.to_csv(out_both, index=False, encoding="utf-8")

    print("DONE")
    print(f"wrote: {out_fj2en.name} rows={len(fj2en):,}")
    print(f"wrote: {out_en2fj.name} rows={len(en2fj):,}")
    print(f"wrote: {out_both.name} rows={len(both):,}")

if __name__ == "__main__":
    main()
