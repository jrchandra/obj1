#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build CLEAN conversational parallel corpus from original Ministry TXT files.

Reads: all *.txt in the same folder as this script
Outputs (same folder):
  - conversational_parallel_en2fj__CLEAN.csv
  - conversational_parallel_fj2en__CLEAN.csv
  - conversational_trilingual__CLEAN.csv
  - conversational_build_summary.txt

Schema (merge-friendly across domains):
  domain, subdomain, source_id, source_doc, source_lang, target_lang, direction, source_text, target_text

Rules:
  - lowercase
  - keep punctuation
  - remove digits
  - normalize whitespace
  - remove obvious instructional/noise lines
  - deduplicate pairs (within direction)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import unicodedata as ud

import pandas as pd


# -------------------------
# Normalization / cleaning
# -------------------------
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
    s = str(s).replace("\ufeff", "")
    for k, v in MOJIBAKE_MAP.items():
        s = s.replace(k, v)
    return ud.normalize("NFKC", s)

def normalize_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()

def clean_text(s: str) -> str:
    """
    Keep punctuation, remove digits, lowercase, keep lexical forms intact.
    """
    s = fix_mojibake(s)
    s = normalize_ws(s)
    s = DIGIT_RE.sub("", s)       # remove numbers only
    s = normalize_ws(s)
    s = s.lower()
    s = normalize_ws(s)
    return s


# -------------------------
# Noise / instruction filter
# -------------------------
NOISE_RE = re.compile(
    r"(produced\s+by|copyright|acknowledg|introduction|learning\s+outcome|"
    r"lesson\s*\d+|activity\s*\d+|matching|cross\s*word|true\s+or\s+false|"
    r"draw\s+a\s+line|circle\s+the\s+correct|fill\s+in\s+the\s+blanks|"
    r"write\s+the\s+correct|reference|table\s+of\s+contents|contents\b|"
    r"teacher\s+to\s+record|answers\b|instructions\b|worksheet\b|"
    r"suva,\s*fiji|ministry\s+of\s+education|page\s+no\.?)",
    re.IGNORECASE
)

DIVIDER_RE = re.compile(r"^\s*[-_=•.]{3,}\s*$", re.IGNORECASE)
PAGE_RE = re.compile(r"^\s*\d+\s*$")  # page numbers

def is_noise_line(raw: str) -> bool:
    s = fix_mojibake(raw).strip()
    if not s:
        return True
    if PAGE_RE.match(s):
        return True
    if DIVIDER_RE.match(s):
        return True
    if NOISE_RE.search(s):
        return True
    # very low alphabetic content (artifacts)
    letters = sum(ch.isalpha() for ch in s)
    if len(s) > 12 and (letters / max(1, len(s)) < 0.25):
        return True
    return False


# -------------------------
# Language hints (only to distinguish FJ vs HI in the 2nd+3rd lines)
# We DON'T try hard to detect English — in these books, English is usually line1.
# -------------------------
FJ_HINTS = {
    "na","ni","sa","au","o","kei","e","tiko","noqu","kequ","vinaka","bula","moce",
    "qasenivuli","itokani","qai","ena","sega","me","ia","qo","qori","vei","vosa",
    "dua","rua","tolu","va","lima","ono","vitu","walu","ciwa","tini"
}
HI_HINTS = {
    "hum","hai","ke","naam","tum","aap","kaise","aur","mein","mei","hamar","iske",
    "baccho","mummy","papa","namaste","bye","shukriya","dhanyavaad","kya","kyon",
    "ka","ki","se","par","bahut"
}

WORD_RE = re.compile(r"[a-zA-Z’']+")

def score_hint(s: str, vocab: set) -> int:
    toks = [t.lower() for t in WORD_RE.findall(fix_mojibake(s))]
    return sum(1 for t in toks if t in vocab)

def classify_fj_hi(line: str) -> str:
    fj = score_hint(line, FJ_HINTS)
    hi = score_hint(line, HI_HINTS)
    if fj > hi and fj > 0:
        return "fj"
    if hi > fj and hi > 0:
        return "hi"
    # tie/unknown
    return "unk"


# -------------------------
# Extraction logic
# -------------------------
def read_clean_lines(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = fix_mojibake(raw)
    lines: List[str] = []
    for ln in raw.splitlines():
        if is_noise_line(ln):
            continue
        t = normalize_ws(ln)
        if t:
            lines.append(t)
    return lines

def looks_like_trailing_synonym(line: str) -> bool:
    """
    Many lessons show:
      english
      fijian
      hindi
      english_synonym (optional)
    We'll drop this 4th line when it's very short and alphabetic.
    """
    t = normalize_ws(line).lower()
    if not t:
        return False
    if len(t) <= 12 and len(t.split()) <= 2 and sum(ch.isalpha() for ch in t) >= 3:
        return True
    return False

def extract_triples(lines: List[str]) -> List[Tuple[str, str, str]]:
    """
    Sliding window:
      a,b,c = lines[i], lines[i+1], lines[i+2]
    If b looks FJ and c looks HI -> (a,b,c)
    If b looks HI and c looks FJ -> (a,c,b)
    Otherwise shift by 1.

    This is robust when the book is "stacked triples" (most of it).
    """
    triples: List[Tuple[str, str, str]] = []
    i = 0
    while i + 2 < len(lines):
        a, b, c = lines[i], lines[i+1], lines[i+2]

        # Clean now (so checks reflect final form)
        a2, b2, c2 = clean_text(a), clean_text(b), clean_text(c)
        if not a2 or not b2:
            i += 1
            continue

        b_lang = classify_fj_hi(b2)
        c_lang = classify_fj_hi(c2)

        if b_lang == "fj" and c_lang == "hi":
            triples.append((a2, b2, c2))
            i += 3
            if i < len(lines) and looks_like_trailing_synonym(lines[i]):
                i += 1
            continue

        if b_lang == "hi" and c_lang == "fj":
            triples.append((a2, c2, b2))
            i += 3
            if i < len(lines) and looks_like_trailing_synonym(lines[i]):
                i += 1
            continue

        i += 1

    return triples

def subdomain_from_filename(name: str) -> str:
    """
    Years 1 - 3_... -> years_1_3
    """
    m = re.search(r"Years?\s*([0-9]+)\s*-\s*([0-9]+)", name, flags=re.IGNORECASE)
    if m:
        return f"years_{m.group(1)}_{m.group(2)}"
    base = re.sub(r"\W+", "_", name.lower()).strip("_")
    return base[:40]


# -------------------------
# Build outputs
# -------------------------
MASTER_COLS = [
    "domain","subdomain","source_id","source_doc",
    "source_lang","target_lang","direction","source_text","target_text"
]

def main():
    folder = Path(__file__).resolve().parent
    txt_files = sorted(folder.glob("*.txt"))

    if not txt_files:
        print("No .txt files found. Put the Years*.txt files in the same folder as this script.")
        sys.exit(1)

    tri_rows: List[Dict[str, str]] = []
    summary: List[str] = []

    total_found = 0
    for f in txt_files:
        lines = read_clean_lines(f)
        triples = extract_triples(lines)
        sub = subdomain_from_filename(f.name)

        kept = 0
        for idx, (en, fj, hi) in enumerate(triples, start=1):
            # extra hard filters to keep it "Excel-clean"
            if len(en) > 300 or len(fj) > 300:   # drop paragraph-like artifacts
                continue
            if is_noise_line(en) or is_noise_line(fj):
                continue
            kept += 1
            tri_rows.append({
                "domain": "conversational",
                "subdomain": sub,
                "source_id": f"conversational|{sub}|{kept:05d}",
                "source_doc": f.name,
                "en_text": en,
                "fj_text": fj,
                "hindi_text": hi,
            })

        total_found += kept
        summary.append(f"{f.name}: lines={len(lines):,} triples_extracted={len(triples):,} kept={kept:,}")

    if not tri_rows:
        print("No triples extracted. If your TXT formatting differs, paste 30 lines from the middle of one file.")
        sys.exit(2)

    df_tri = pd.DataFrame(tri_rows)

    # Deduplicate trilingual on (en,fj) to remove repeats across files
    before = len(df_tri)
    df_tri = df_tri.drop_duplicates(subset=["en_text","fj_text"], keep="first").reset_index(drop=True)
    dedup_tri = before - len(df_tri)

    # Build parallel
    df_en2fj = pd.DataFrame({
        "domain": df_tri["domain"],
        "subdomain": df_tri["subdomain"],
        "source_id": df_tri["source_id"],
        "source_doc": df_tri["source_doc"],
        "source_lang": "en",
        "target_lang": "fj",
        "direction": "en->fj",
        "source_text": df_tri["en_text"],
        "target_text": df_tri["fj_text"],
    })[MASTER_COLS]

    df_fj2en = pd.DataFrame({
        "domain": df_tri["domain"],
        "subdomain": df_tri["subdomain"],
        "source_id": df_tri["source_id"],
        "source_doc": df_tri["source_doc"],
        "source_lang": "fj",
        "target_lang": "en",
        "direction": "fj->en",
        "source_text": df_tri["fj_text"],
        "target_text": df_tri["en_text"],
    })[MASTER_COLS]

    # Deduplicate within each direction on (source_text,target_text)
    before_en = len(df_en2fj)
    df_en2fj = df_en2fj.drop_duplicates(subset=["direction","source_text","target_text"], keep="first").reset_index(drop=True)
    dedup_en = before_en - len(df_en2fj)

    before_fj = len(df_fj2en)
    df_fj2en = df_fj2en.drop_duplicates(subset=["direction","source_text","target_text"], keep="first").reset_index(drop=True)
    dedup_fj = before_fj - len(df_fj2en)

    # Write outputs
    out_en2fj = folder / "conversational_parallel_en2fj__CLEAN.csv"
    out_fj2en = folder / "conversational_parallel_fj2en__CLEAN.csv"
    out_tri = folder / "conversational_trilingual__CLEAN.csv"
    out_sum = folder / "conversational_build_summary.txt"

    df_en2fj.to_csv(out_en2fj, index=False, encoding="utf-8")
    df_fj2en.to_csv(out_fj2en, index=False, encoding="utf-8")
    df_tri.to_csv(out_tri, index=False, encoding="utf-8")

    out_sum.write_text(
        "\n".join(summary)
        + "\n\n"
        + f"trilingual_rows_written: {len(df_tri):,} (dedup_removed={dedup_tri:,})\n"
        + f"en2fj_rows_written: {len(df_en2fj):,} (dedup_removed={dedup_en:,})\n"
        + f"fj2en_rows_written: {len(df_fj2en):,} (dedup_removed={dedup_fj:,})\n",
        encoding="utf-8"
    )

    print("DONE")
    print(f"wrote: {out_en2fj.name}  rows={len(df_en2fj):,}")
    print(f"wrote: {out_fj2en.name}  rows={len(df_fj2en):,}")
    print(f"wrote: {out_tri.name}     rows={len(df_tri):,}")
    print(f"wrote: {out_sum.name}")

if __name__ == "__main__":
    main()
