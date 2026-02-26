#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build parallel corpus from ONE file: definition.xlsx

Outputs (same folder):
  - definition_parallel_fj2en__ALL.csv
  - definition_parallel_en2fj__ALL.csv
  - definition_parallel_both__ALL.csv
  - definition_build_summary.txt

Schema (merge-friendly across domains):
  domain, subdomain, source_id, source_doc,
  source_lang, target_lang, direction, source_text, target_text

Cleaning rules:
  - lowercase
  - keep punctuation
  - remove digits in source_text / target_text
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Tuple, List, Dict
import unicodedata as ud

import pandas as pd

MASTER_COLS = [
    "domain","subdomain","source_id","source_doc",
    "source_lang","target_lang","direction",
    "source_text","target_text"
]

WS_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\d+")

MOJIBAKE_MAP = {
    "â€œ": "“", "â€": "”", "â€˜": "‘", "â€™": "’",
    "â€“": "–", "â€”": "—", "â€¦": "…", "Â": ""
}

EN_STOP = {
    "the","and","or","to","of","in","on","for","with","without","from",
    "is","are","was","were","be","been","being","a","an","this","that",
    "i","you","he","she","it","we","they","my","your","our","their",
    "as","by","at","into","about","over","under","between"
}

def fix_mojibake(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\ufeff", "")
    for k, v in MOJIBAKE_MAP.items():
        s = s.replace(k, v)
    return ud.normalize("NFKC", s)

def normalize_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()

def clean_text(s: str, lowercase: bool = True, remove_digits: bool = True) -> str:
    s = fix_mojibake(s)
    s = normalize_ws(s)
    if remove_digits:
        s = DIGIT_RE.sub("", s)
        s = normalize_ws(s)
    if lowercase:
        s = s.lower()
        s = normalize_ws(s)
    # treat "nan" as empty
    if s == "nan":
        return ""
    return s

def looks_like_header_row(row: pd.Series) -> bool:
    joined = " ".join(clean_text(x, lowercase=True, remove_digits=False) for x in row.astype(str).tolist())
    if not joined:
        return False
    hits = sum(kw in joined for kw in ["english", "fijian", "itaukei", "definition", "meaning", "word"])
    return hits >= 2

def english_score(s: str) -> float:
    """
    Heuristic: English stopword hits + ascii-letter ratio
    """
    s0 = clean_text(s, lowercase=True, remove_digits=False)
    if not s0:
        return 0.0
    toks = re.findall(r"[a-z']+", s0)
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in EN_STOP)
    ascii_letters = sum(1 for ch in s0 if "a" <= ch <= "z")
    ratio = ascii_letters / max(1, len(s0))
    return hits + ratio

def pick_best_text_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Pick (fj_col, en_col) by:
    - choosing columns with lots of non-empty text
    - choosing English column by higher english_score
    """
    cand = []
    for c in df.columns:
        ser = df[c].astype(str).map(lambda x: clean_text(x, lowercase=False, remove_digits=False))
        ser = ser[ser.map(bool)]
        if len(ser) < 30:
            continue
        sample = ser.sample(min(200, len(ser)), random_state=11)
        avg_en = float(sample.map(english_score).mean())
        nonempty = int(len(ser))
        avg_len = float(sample.map(len).mean())
        cand.append((c, nonempty, avg_en, avg_len))

    if len(cand) < 2:
        raise ValueError("Could not find at least 2 usable text columns in the sheet.")

    # Prefer big non-empty columns first
    cand.sort(key=lambda x: (x[1], x[3]), reverse=True)
    top = cand[:6]

    # choose best pair by distinct english scores and coverage
    best = None
    best_val = -1.0
    for i in range(len(top)):
        for j in range(i+1, len(top)):
            c1, n1, e1, l1 = top[i]
            c2, n2, e2, l2 = top[j]
            # encourage both populated + different english-likeness
            val = (min(n1, n2) / 100.0) + abs(e1 - e2)
            if val > best_val:
                best_val = val
                best = (c1, c2, e1, e2)

    assert best is not None
    c1, c2, e1, e2 = best
    if e1 >= e2:
        en_col, fj_col = c1, c2
    else:
        en_col, fj_col = c2, c1
    return fj_col, en_col

def extract_pairs_from_sheet(df: pd.DataFrame, sheet_name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    # drop fully empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    # remove obvious header row if present in first few rows
    for r in range(min(5, len(df))):
        if looks_like_header_row(df.iloc[r]):
            df = df.drop(index=r).reset_index(drop=True)
            break

    fj_col, en_col = pick_best_text_cols(df)

    fj = df[fj_col].astype(str).map(lambda x: clean_text(x, lowercase=True, remove_digits=True))
    en = df[en_col].astype(str).map(lambda x: clean_text(x, lowercase=True, remove_digits=True))

    out = pd.DataFrame({"fj": fj, "en": en})
    out = out[(out["fj"].map(bool)) & (out["en"].map(bool))].copy()

    before = len(out)
    out = out.drop_duplicates(subset=["fj", "en"]).reset_index(drop=True)
    dedup = before - len(out)

    stats = {
        "sheet": sheet_name,
        "picked_fj_col": str(fj_col),
        "picked_en_col": str(en_col),
        "pairs": int(len(out)),
        "dedup_removed": int(dedup),
    }
    return out, stats

def main():
    folder = Path(__file__).resolve().parent
    xlsx_path = folder / "definition.xlsx"
    if not xlsx_path.exists():
        raise SystemExit("definition.xlsx not found in this folder. Put it next to this script.")

    xls = pd.ExcelFile(xlsx_path)
    all_pairs: List[pd.DataFrame] = []
    summary: List[str] = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

        try:
            pairs, st = extract_pairs_from_sheet(df, sheet)
        except Exception as e:
            summary.append(f"[FAIL] sheet={sheet}: {e}")
            continue

        if len(pairs) == 0:
            summary.append(f"[OK] sheet={sheet}: pairs=0 (picked fj={st['picked_fj_col']} en={st['picked_en_col']})")
            continue

        pairs["source_doc"] = f"definition.xlsx::{sheet}"
        all_pairs.append(pairs)

        summary.append(
            f"[OK] sheet={sheet}: pairs={st['pairs']:,} dedup_removed={st['dedup_removed']:,} "
            f"(picked fj={st['picked_fj_col']} en={st['picked_en_col']})"
        )

    if not all_pairs:
        raise SystemExit("No pairs extracted from definition.xlsx. If it’s a special layout, share a screenshot of 10 rows.")

    df = pd.concat(all_pairs, ignore_index=True).drop_duplicates(subset=["fj","en"]).reset_index(drop=True)

    # stable ids
    df["source_id"] = df.index.map(lambda i: f"definition|{i+1:06d}")

    # Build parallel outputs
    fj2en = pd.DataFrame({
        "domain": "definition",
        "subdomain": "definition",
        "source_id": df["source_id"],
        "source_doc": df["source_doc"],
        "source_lang": "fj",
        "target_lang": "en",
        "direction": "fj->en",
        "source_text": df["fj"],
        "target_text": df["en"],
    })[MASTER_COLS]

    en2fj = pd.DataFrame({
        "domain": "definition",
        "subdomain": "definition",
        "source_id": df["source_id"],
        "source_doc": df["source_doc"],
        "source_lang": "en",
        "target_lang": "fj",
        "direction": "en->fj",
        "source_text": df["en"],
        "target_text": df["fj"],
    })[MASTER_COLS]

    # De-dupe within direction
    fj2en = fj2en.drop_duplicates(subset=["direction","source_text","target_text"]).reset_index(drop=True)
    en2fj = en2fj.drop_duplicates(subset=["direction","source_text","target_text"]).reset_index(drop=True)
    both = pd.concat([fj2en, en2fj], ignore_index=True)

    out_fj2en = folder / "definition_parallel_fj2en__ALL.csv"
    out_en2fj = folder / "definition_parallel_en2fj__ALL.csv"
    out_both  = folder / "definition_parallel_both__ALL.csv"
    out_sum   = folder / "definition_build_summary.txt"

    fj2en.to_csv(out_fj2en, index=False, encoding="utf-8")
    en2fj.to_csv(out_en2fj, index=False, encoding="utf-8")
    both.to_csv(out_both, index=False, encoding="utf-8")
    out_sum.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("DONE")
    print(f"wrote: {out_fj2en.name} rows={len(fj2en):,}")
    print(f"wrote: {out_en2fj.name} rows={len(en2fj):,}")
    print(f"wrote: {out_both.name}  rows={len(both):,}")
    print(f"wrote: {out_sum.name}")

if __name__ == "__main__":
    main()
