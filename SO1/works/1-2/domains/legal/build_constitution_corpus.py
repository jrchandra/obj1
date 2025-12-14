#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Constitution Parallel Corpus (EN <-> iTaukei) from two TXT files.

Inputs (same folder):
  - constitution - eng.txt
  - constitution - itaukei.txt

Outputs (same folder):
  - constitution_parallel_en2fj__CLAUSE.csv
  - constitution_parallel_fj2en__CLAUSE.csv
  - constitution_parallel_both__CLAUSE.csv
  - constitution_build_summary.txt

Schema (merge-friendly across domains):
  domain, subdomain, source_id, source_doc, source_lang, target_lang, direction, source_text, target_text

Text policy:
  - lowercase
  - keep punctuation
  - remove digits FROM source_text/target_text (numbers retained in source_id for traceability)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple
import unicodedata as ud

import pandas as pd

MASTER_COLS = [
    "domain","subdomain","source_id","source_doc",
    "source_lang","target_lang","direction",
    "source_text","target_text"
]

# ---------- normalization ----------
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
DIGITS_RE = re.compile(r"\d+")

def fix_mojibake(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\ufeff", "")
    for k, v in MOJIBAKE_MAP.items():
        s = s.replace(k, v)
    return ud.normalize("NFKC", s)

def normalize_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()

def clean_text_keep_punct_drop_digits(s: str) -> str:
    s = fix_mojibake(s)
    s = normalize_ws(s)
    s = DIGITS_RE.sub("", s)
    s = normalize_ws(s)
    s = s.lower()
    s = normalize_ws(s)
    return s

# ---------- heuristics / parsing ----------
# We try to build keys like:
#   "preamble"
#   "1"
#   "2(1)"
#   "2(3)"
#   "10(1)(a)"
#   "13(1)(a)(ii)"
#
# This works because both EN and iTaukei versions share the same numbering structure.
SEC_RE = re.compile(r"^\s*(?P<sec>\d+)\.\s+(?P<rest>.+)\s*$")
SEC_DASH_RE = re.compile(r"^\s*(?P<sec>\d+)\s*[—-]\s*\((?P<sub>\d+)\)\s*(?P<rest>.+)\s*$")
SUB_RE = re.compile(r"^\s*\((?P<sub>\d+)\)\s+(?P<rest>.+)\s*$")
LETTER_RE = re.compile(r"^\s*\((?P<let>[a-z])\)\s+(?P<rest>.+)\s*$", re.IGNORECASE)
ROMAN_RE = re.compile(r"^\s*\((?P<rom>i{1,4}|v|vi{0,3}|ix|x)\)\s+(?P<rest>.+)\s*$", re.IGNORECASE)

# headings we ignore (contents pages etc.)
IGNORE_RE = re.compile(
    r"^(constitution|yavunivakavulewa|contents|lewena|chapter|wase|part|tabana)\b",
    re.IGNORECASE
)

def is_noise_line(line: str) -> bool:
    s = fix_mojibake(line).strip()
    if not s:
        return True
    if IGNORE_RE.match(s):
        # headings are useful sometimes, but they are often duplicated in front matter.
        # We'll handle PREAMBLE explicitly and otherwise ignore these headings for the corpus rows.
        return True
    # page numbers / roman numerals alone
    if re.fullmatch(r"[ivx]+", s.strip().lower()):
        return True
    if re.fullmatch(r"\d+", s.strip()):
        return True
    # repeated underscores / dividers
    if re.fullmatch(r"[_\-=•.]{3,}", s.strip()):
        return True
    return False

def read_lines(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = fix_mojibake(raw)
    lines = [ln.rstrip("\n\r") for ln in raw.splitlines()]
    return lines

def parse_constitution(lines: List[str]) -> Dict[str, str]:
    """
    Returns mapping: clause_key -> clause_text
    clause_key examples: "1", "2(1)", "10(1)(a)(ii)", "preamble"
    """
    out: Dict[str, str] = {}

    cur_sec: str | None = None
    cur_sub: str | None = None
    cur_let: str | None = None
    cur_rom: str | None = None

    # Detect PREAMBLE section explicitly
    in_preamble = False
    preamble_buf: List[str] = []

    def flush_preamble():
        nonlocal preamble_buf
        if preamble_buf:
            out["preamble"] = normalize_ws(" ".join(preamble_buf))
            preamble_buf = []

    # Buffer per key so wrapped lines append
    cur_key: str | None = None
    buf: List[str] = []

    def flush_current():
        nonlocal cur_key, buf
        if cur_key and buf:
            out[cur_key] = normalize_ws(" ".join(buf))
        cur_key = None
        buf = []

    for raw in lines:
        line = fix_mojibake(raw).rstrip()

        # preamble markers (both languages)
        if re.match(r"^\s*preamble\s*$", line, re.IGNORECASE) or re.match(r"^\s*ikau\s*$", line, re.IGNORECASE):
            # entering preamble region (front matter also has "CONTENTS", but we ignore those above)
            in_preamble = True
            flush_current()
            continue

        # Heuristic end of preamble: first numbered section line
        if in_preamble and (SEC_RE.match(line) or SEC_DASH_RE.match(line)):
            in_preamble = False
            flush_preamble()
            # Continue parsing this line as normal (do not skip)

        if in_preamble:
            if not is_noise_line(line):
                preamble_buf.append(line.strip())
            continue

        # Skip noise
        if is_noise_line(line):
            continue

        # New top-level section "1. ..."
        m = SEC_RE.match(line)
        if m:
            flush_current()
            cur_sec = m.group("sec")
            cur_sub = None
            cur_let = None
            cur_rom = None
            cur_key = cur_sec
            buf = [m.group("rest")]
            continue

        # "2.—(1) ..." style
        m = SEC_DASH_RE.match(line)
        if m:
            flush_current()
            cur_sec = m.group("sec")
            cur_sub = m.group("sub")
            cur_let = None
            cur_rom = None
            cur_key = f"{cur_sec}({cur_sub})"
            buf = [m.group("rest")]
            continue

        # Subsection "(2) ..." (assumes under current section)
        m = SUB_RE.match(line)
        if m and cur_sec is not None:
            flush_current()
            cur_sub = m.group("sub")
            cur_let = None
            cur_rom = None
            cur_key = f"{cur_sec}({cur_sub})"
            buf = [m.group("rest")]
            continue

        # Lettered "(a) ..."
        m = LETTER_RE.match(line)
        if m and cur_sec is not None:
            flush_current()
            cur_let = m.group("let").lower()
            cur_rom = None
            if cur_sub is not None:
                cur_key = f"{cur_sec}({cur_sub})({cur_let})"
            else:
                cur_key = f"{cur_sec}({cur_let})"
            buf = [m.group("rest")]
            continue

        # Roman "(i) ..." under lettered
        m = ROMAN_RE.match(line)
        if m and cur_sec is not None and (cur_sub is not None or cur_let is not None):
            flush_current()
            cur_rom = m.group("rom").lower()
            parts = [cur_sec]
            if cur_sub is not None:
                parts.append(f"({cur_sub})")
            if cur_let is not None:
                parts.append(f"({cur_let})")
            parts.append(f"({cur_rom})")
            cur_key = "".join(parts)
            buf = [m.group("rest")]
            continue

        # Continuation line: append to current buffer if we have a key
        if cur_key:
            buf.append(line.strip())
        else:
            # If we have no key (rare), ignore
            pass

    # flush at end
    flush_current()
    flush_preamble()
    return out

def build_rows(en_map: Dict[str, str], fj_map: Dict[str, str], en_doc: str, fj_doc: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    keys = sorted(set(en_map.keys()) & set(fj_map.keys()), key=lambda k: (k!="preamble", k))
    summary = [
        f"en_keys={len(en_map):,}",
        f"fj_keys={len(fj_map):,}",
        f"aligned_keys={len(keys):,}"
    ]

    base = []
    for k in keys:
        en_txt = clean_text_keep_punct_drop_digits(en_map[k])
        fj_txt = clean_text_keep_punct_drop_digits(fj_map[k])

        # drop if either becomes empty after digit removal
        if not en_txt or not fj_txt:
            continue

        base.append({
            "ref": k,
            "en": en_txt,
            "fj": fj_txt,
        })

    df = pd.DataFrame(base)

    fj2en = pd.DataFrame({
        "domain": "legal",
        "subdomain": "constitution",
        "source_id": df["ref"].map(lambda r: f"constitution|{r}"),
        "source_doc": fj_doc,
        "source_lang": "fj",
        "target_lang": "en",
        "direction": "fj->en",
        "source_text": df["fj"],
        "target_text": df["en"],
    })[MASTER_COLS]

    en2fj = pd.DataFrame({
        "domain": "legal",
        "subdomain": "constitution",
        "source_id": df["ref"].map(lambda r: f"constitution|{r}"),
        "source_doc": en_doc,
        "source_lang": "en",
        "target_lang": "fj",
        "direction": "en->fj",
        "source_text": df["en"],
        "target_text": df["fj"],
    })[MASTER_COLS]

    # de-dup within direction
    fj2en = fj2en.drop_duplicates(subset=["direction","source_text","target_text"]).reset_index(drop=True)
    en2fj = en2fj.drop_duplicates(subset=["direction","source_text","target_text"]).reset_index(drop=True)
    both = pd.concat([en2fj, fj2en], ignore_index=True)

    summary.append(f"rows_written_en2fj={len(en2fj):,}")
    summary.append(f"rows_written_fj2en={len(fj2en):,}")
    summary.append(f"rows_written_both={len(both):,}")

    return en2fj, fj2en, both, summary

def main():
    folder = Path(__file__).resolve().parent

    en_path = folder / "constitution - eng.txt"
    fj_path = folder / "constitution - itaukei.txt"

    if not en_path.exists() or not fj_path.exists():
        raise SystemExit("Put BOTH files in the same folder as this script:\n  - constitution - eng.txt\n  - constitution - itaukei.txt")

    en_lines = read_lines(en_path)
    fj_lines = read_lines(fj_path)

    en_map = parse_constitution(en_lines)
    fj_map = parse_constitution(fj_lines)

    en2fj, fj2en, both, summary = build_rows(en_map, fj_map, en_path.name, fj_path.name)

    out_en2fj = folder / "constitution_parallel_en2fj__CLAUSE.csv"
    out_fj2en = folder / "constitution_parallel_fj2en__CLAUSE.csv"
    out_both = folder / "constitution_parallel_both__CLAUSE.csv"
    out_sum  = folder / "constitution_build_summary.txt"

    en2fj.to_csv(out_en2fj, index=False, encoding="utf-8")
    fj2en.to_csv(out_fj2en, index=False, encoding="utf-8")
    both.to_csv(out_both, index=False, encoding="utf-8")
    out_sum.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("DONE")
    print(f"wrote: {out_en2fj.name} rows={len(en2fj):,}")
    print(f"wrote: {out_fj2en.name} rows={len(fj2en):,}")
    print(f"wrote: {out_both.name} rows={len(both):,}")
    print(f"wrote: {out_sum.name}")

if __name__ == "__main__":
    main()
