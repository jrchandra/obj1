#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build iTaukei (Fijian) Idioms Parallel Corpus from TXT

Input:  Put this script in the SAME folder as your idioms .txt file(s)
        (e.g., "Fijian Idioms - Copy.txt"). It will read all *.txt in the folder.

Output (same folder):
  - idiom_parallel_fj2en__ALL.csv
  - idiom_parallel_en2fj__ALL.csv
  - idiom_parallel_both__ALL.csv

Schema (merge-friendly across domains):
  domain, subdomain, source_id, source_doc,
  source_lang, target_lang, direction,
  source_text, target_text

Extraction:
  - source_text: the idiom phrase in iTaukei (before the first period in each numbered item)
  - target_text (literal): extracted after "literally translated as/translated as/translation:"
  - target_text (meaning): extracted after "it means/meaning:/refers to/it implies/idiom indicates..."

Cleaning defaults:
  - lowercase
  - keep punctuation
  - remove digits
  - normalize whitespace
  - fix common mojibake (â€™ etc.)
"""

from __future__ import annotations
from pathlib import Path
import re
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

LIT_RE = re.compile(r"(literally translated as|literarily translated as|translated as|translation:)", re.IGNORECASE)
MEAN_RE = re.compile(
    r"(it means|meaning:|meaning\s*-\s*|meaning\s+as\s+expressed[^:]*:|"
    r"this refers to|this idiom indicates|idiom indicates|it implies|refers to)",
    re.IGNORECASE
)

def fix_mojibake(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\ufeff", "")
    for k, v in MOJIBAKE_MAP.items():
        s = s.replace(k, v)
    return ud.normalize("NFKC", s)

def normalize_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()

def clean_text(s: str) -> str:
    s = fix_mojibake(s)
    s = normalize_ws(s)
    s = DIGIT_RE.sub("", s)       # remove numbers only
    s = normalize_ws(s)
    return s.lower()

def clean_literal(lit: str) -> str:
    lit = normalize_ws(fix_mojibake(lit))
    # cut if literal line accidentally includes meaning
    lit = re.split(r"\b(meaning|it means|it implies|this implies)\b", lit, flags=re.IGNORECASE)[0]
    return lit.strip().strip('“”"\'').strip()

def parse_numbered_blocks(text: str) -> list[str]:
    """
    Split text into numbered blocks: 1. ... / 2. ... etc.
    """
    pat = re.compile(r"(?m)^\s*(\d+)\.\s+")
    ms = list(pat.finditer(text))
    blocks = []
    for i, m in enumerate(ms):
        start = m.start()
        end = ms[i+1].start() if i+1 < len(ms) else len(text)
        blocks.append(text[start:end].strip())
    return blocks

def parse_block(block: str) -> tuple[str|None, str|None, str|None]:
    """
    Returns (fj_idiom, en_literal, en_meaning) best-effort.
    """
    b = re.sub(r"^\s*\d+\.\s*", "", block).strip()
    if "." in b:
        fj, rest = b.split(".", 1)
        fj = fj.strip()
        rest = rest.strip()
    else:
        fj = b.strip()
        rest = ""

    lit = None
    meaning = None

    m = LIT_RE.search(rest)
    if m:
        after = rest[m.end():].strip().lstrip(" :,-")
        p = after.find(".")
        lit_raw = after[:p].strip() if p != -1 else after.strip()
        lit = clean_literal(lit_raw)

    m2 = MEAN_RE.search(rest)
    if m2:
        after = rest[m2.end():].strip().lstrip(" :,-")
        meaning = after.strip()

    return fj, lit, meaning

def build_from_file(txt_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    text = fix_mojibake(text)

    blocks = parse_numbered_blocks(text)

    rows_fj2en = []
    for idx, blk in enumerate(blocks, start=1):
        fj, lit, meaning = parse_block(blk)
        if not fj:
            continue
        fjc = clean_text(fj)
        if lit:
            litc = clean_text(lit)
            if fjc and litc:
                rows_fj2en.append({
                    "domain": "idiom",
                    "subdomain": "idiom_literal",
                    "source_id": f"idiom|{idx:04d}|lit",
                    "source_doc": txt_path.name,
                    "source_lang": "fj",
                    "target_lang": "en",
                    "direction": "fj->en",
                    "source_text": fjc,
                    "target_text": litc,
                })
        if meaning:
            meanc = clean_text(meaning)
            if fjc and meanc:
                rows_fj2en.append({
                    "domain": "idiom",
                    "subdomain": "idiom_meaning",
                    "source_id": f"idiom|{idx:04d}|mean",
                    "source_doc": txt_path.name,
                    "source_lang": "fj",
                    "target_lang": "en",
                    "direction": "fj->en",
                    "source_text": fjc,
                    "target_text": meanc,
                })

    df_fj2en = pd.DataFrame(rows_fj2en)[MASTER_COLS] if rows_fj2en else pd.DataFrame(columns=MASTER_COLS)

    # Build en->fj by swapping
    if len(df_fj2en) > 0:
        df_en2fj = df_fj2en.copy()
        df_en2fj["source_lang"] = "en"
        df_en2fj["target_lang"] = "fj"
        df_en2fj["direction"] = "en->fj"
        df_en2fj = df_en2fj.rename(columns={"source_text": "target_text_tmp", "target_text": "source_text"})
        df_en2fj["target_text"] = df_en2fj["target_text_tmp"]
        df_en2fj = df_en2fj.drop(columns=["target_text_tmp"])
        df_en2fj = df_en2fj[MASTER_COLS]
    else:
        df_en2fj = pd.DataFrame(columns=MASTER_COLS)

    # dedupe
    df_fj2en = df_fj2en.drop_duplicates(subset=["direction","subdomain","source_text","target_text"]).reset_index(drop=True)
    df_en2fj = df_en2fj.drop_duplicates(subset=["direction","subdomain","source_text","target_text"]).reset_index(drop=True)
    df_both = pd.concat([df_fj2en, df_en2fj], ignore_index=True)

    return df_fj2en, df_en2fj, df_both

def main():
    folder = Path(__file__).resolve().parent
    txts = sorted(folder.glob("*.txt"))
    if not txts:
        print("No .txt files found in this folder. Put your idioms .txt next to this script.")
        raise SystemExit(1)

    all_fj2en = []
    all_en2fj = []
    for p in txts:
        fj2en, en2fj, _ = build_from_file(p)
        if len(fj2en) > 0:
            all_fj2en.append(fj2en)
        if len(en2fj) > 0:
            all_en2fj.append(en2fj)

    df_fj2en = pd.concat(all_fj2en, ignore_index=True) if all_fj2en else pd.DataFrame(columns=MASTER_COLS)
    df_en2fj = pd.concat(all_en2fj, ignore_index=True) if all_en2fj else pd.DataFrame(columns=MASTER_COLS)

    df_fj2en = df_fj2en.drop_duplicates(subset=["direction","subdomain","source_text","target_text"]).reset_index(drop=True)
    df_en2fj = df_en2fj.drop_duplicates(subset=["direction","subdomain","source_text","target_text"]).reset_index(drop=True)
    df_both = pd.concat([df_fj2en, df_en2fj], ignore_index=True)

    (folder / "idiom_parallel_fj2en__ALL.csv").write_text(df_fj2en.to_csv(index=False), encoding="utf-8")
    (folder / "idiom_parallel_en2fj__ALL.csv").write_text(df_en2fj.to_csv(index=False), encoding="utf-8")
    (folder / "idiom_parallel_both__ALL.csv").write_text(df_both.to_csv(index=False), encoding="utf-8")

    print("DONE")
    print(f"fj2en rows: {len(df_fj2en):,}")
    print(f"en2fj rows: {len(df_en2fj):,}")
    print(f"both rows:  {len(df_both):,}")

if __name__ == "__main__":
    main()
