#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build MEDICAL parallel corpus from OCR'd TXT files (English + iTaukei).

Inputs:  all *.txt in this script folder
Outputs:
  - medical_parallel_en2fj__TXT.csv
  - medical_parallel_fj2en__TXT.csv
  - medical_parallel_both__TXT.csv
  - medical_txt_build_summary.txt

Rules:
  - lowercase
  - keep punctuation
  - remove digits
  - line-level alignment by position (min(len(en_lines), len(fj_lines)))
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import unicodedata as ud

import pandas as pd

MASTER_COLS = [
    "domain","subdomain","source_id","source_doc",
    "source_lang","target_lang","direction","source_text","target_text"
]

WS_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\d+")
# keep punctuation; remove only digits (per your requirement)

MOJIBAKE_MAP = {
    "â€œ": "“", "â€": "”", "â€˜": "‘", "â€™": "’",
    "â€“": "–", "â€”": "—", "â€¦": "…", "Â": ""
}

# Filename cues for pairing
EN_CUES = ["english", "_en", "-en", "eng", "rgb"]
FJ_CUES = ["fijian", "itaukei", "iTAUKEI".lower(), "fiji", "_fj", "-fj"]

def norm_filename(name: str) -> str:
    n = name.lower()
    n = n.replace(" ", "-")
    n = re.sub(r"[^a-z0-9\-_\.]+", "", n)
    return n

def base_key(name: str) -> str:
    """
    Convert a filename into a pairing key by stripping language cues.
    Example:
      "dementia-and-support-english.txt" -> "dementia-and-support"
      "dementia-and-support-fiji.txt"    -> "dementia-and-support"
    """
    n = norm_filename(name)
    n = n.replace(".txt", "")
    # remove language cues tokens
    for cue in EN_CUES + FJ_CUES:
        n = n.replace(cue, "")
    # cleanup
    n = re.sub(r"[-_]+", "-", n).strip("-_")
    return n

def fix_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\ufeff", "")
    for k, v in MOJIBAKE_MAP.items():
        s = s.replace(k, v)
    s = ud.normalize("NFKC", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def clean_text(s: str) -> str:
    s = fix_text(s)
    s = DIGIT_RE.sub("", s)          # exclude numbers
    s = WS_RE.sub(" ", s).strip()
    s = s.lower()                    # fully lowercase
    # keep punctuation as-is (no stripping)
    # treat literal "nan" as empty (defensive)
    if s == "nan":
        return ""
    return s

def is_noise_line(s: str) -> bool:
    """
    Light noise filter:
    - empty
    - extremely short (<=2 chars) after cleaning
    - mostly punctuation
    """
    if not s:
        return True
    if len(s) <= 2:
        return True
    # if line is mostly punctuation/symbols
    letters = sum(ch.isalpha() for ch in s)
    if letters == 0 and len(s) < 12:
        return True
    return False

def read_clean_lines(p: Path) -> List[str]:
    raw = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out: List[str] = []
    for ln in raw:
        ln2 = clean_text(ln)
        if is_noise_line(ln2):
            continue
        out.append(ln2)
    return out

def detect_lang_side(filename: str) -> str:
    n = filename.lower()
    if any(cue in n for cue in EN_CUES):
        return "en"
    if any(cue in n for cue in FJ_CUES):
        return "fj"
    # unknown: return "" and we’ll try pairing by best match
    return ""

def pair_files(txt_files: List[Path]) -> List[Tuple[Path, Path, str]]:
    """
    Return list of (en_file, fj_file, subdomain_key)
    Uses:
      - language cues in filename
      - shared base_key
    """
    buckets: Dict[str, Dict[str, List[Path]]] = {}
    for f in txt_files:
        key = base_key(f.name)
        side = detect_lang_side(f.name)  # en / fj / ""
        if key not in buckets:
            buckets[key] = {"en": [], "fj": [], "unk": []}
        if side == "en":
            buckets[key]["en"].append(f)
        elif side == "fj":
            buckets[key]["fj"].append(f)
        else:
            buckets[key]["unk"].append(f)

    pairs: List[Tuple[Path, Path, str]] = []
    for key, grp in buckets.items():
        en_list = grp["en"]
        fj_list = grp["fj"]

        # if ambiguous, try to resolve with unknowns
        # (rare; your filenames are pretty clear)
        if not en_list and grp["unk"]:
            en_list = grp["unk"][:]
        if not fj_list and grp["unk"]:
            fj_list = grp["unk"][:]

        if not en_list or not fj_list:
            # cannot pair this key
            continue

        # choose first if multiple
        en_file = sorted(en_list, key=lambda p: p.name)[0]
        fj_file = sorted(fj_list, key=lambda p: p.name)[0]
        pairs.append((en_file, fj_file, key if key else "medical_txt"))

    # de-duplicate exact file pairs
    uniq = []
    seen = set()
    for enf, fjf, k in pairs:
        sig = (enf.name, fjf.name)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append((enf, fjf, k))
    return uniq

def build_parallel_rows(en_file: Path, fj_file: Path, subdomain: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    en_lines = read_clean_lines(en_file)
    fj_lines = read_clean_lines(fj_file)

    n = min(len(en_lines), len(fj_lines))
    en_lines = en_lines[:n]
    fj_lines = fj_lines[:n]

    if n == 0:
        return (
            pd.DataFrame(columns=MASTER_COLS),
            pd.DataFrame(columns=MASTER_COLS),
            pd.DataFrame(columns=MASTER_COLS),
            f"[EMPTY] {en_file.name} <-> {fj_file.name}"
        )

    source_doc = f"{en_file.name}||{fj_file.name}"

    # stable ids: medical|<subdomain>|000001 etc.
    ids = [f"medical|{subdomain}|{i+1:06d}" for i in range(n)]

    en2fj = pd.DataFrame({
        "domain": "medical",
        "subdomain": subdomain,
        "source_id": ids,
        "source_doc": source_doc,
        "source_lang": "en",
        "target_lang": "fj",
        "direction": "en->fj",
        "source_text": en_lines,
        "target_text": fj_lines,
    })[MASTER_COLS]

    fj2en = pd.DataFrame({
        "domain": "medical",
        "subdomain": subdomain,
        "source_id": ids,
        "source_doc": source_doc,
        "source_lang": "fj",
        "target_lang": "en",
        "direction": "fj->en",
        "source_text": fj_lines,
        "target_text": en_lines,
    })[MASTER_COLS]

    both = pd.concat([en2fj, fj2en], ignore_index=True)

    msg = f"[OK] {subdomain}: aligned_pairs={n:,} (en_lines={len(read_clean_lines(en_file)):,} fj_lines={len(read_clean_lines(fj_file)):,})"
    return en2fj, fj2en, both, msg

def main():
    folder = Path(__file__).resolve().parent
    txt_files = sorted(folder.glob("*.txt"))

    if not txt_files:
        raise SystemExit("No .txt files found in this folder.")

    pairs = pair_files(txt_files)

    if not pairs:
        raise SystemExit(
            "Could not auto-pair any EN/FJ txt files. "
            "Rename files to include 'english' and 'itaukei'/'fijian' cues."
        )

    all_en2fj = []
    all_fj2en = []
    all_both = []
    summary_lines = []
    summary_lines.append("MEDICAL TXT CORPUS BUILD SUMMARY")
    summary_lines.append(f"folder: {str(folder)}")
    summary_lines.append(f"txt_files_found: {len(txt_files)}")
    summary_lines.append(f"paired_sets_found: {len(pairs)}")
    summary_lines.append("")

    for enf, fjf, sub in pairs:
        en2fj, fj2en, both, msg = build_parallel_rows(enf, fjf, subdomain=sub)
        summary_lines.append(msg)
        if not en2fj.empty:
            all_en2fj.append(en2fj)
            all_fj2en.append(fj2en)
            all_both.append(both)
        else:
            summary_lines.append(f"  -> skipped empty pair for {sub}")

    if not all_both:
        raise SystemExit("All paired sets produced zero aligned rows after cleaning.")

    df_en2fj = pd.concat(all_en2fj, ignore_index=True).drop_duplicates(subset=["direction","source_text","target_text"])
    df_fj2en = pd.concat(all_fj2en, ignore_index=True).drop_duplicates(subset=["direction","source_text","target_text"])
    df_both  = pd.concat(all_both,  ignore_index=True).drop_duplicates(subset=["direction","source_text","target_text"])

    df_en2fj.to_csv(folder / "medical_parallel_en2fj__TXT.csv", index=False, encoding="utf-8")
    df_fj2en.to_csv(folder / "medical_parallel_fj2en__TXT.csv", index=False, encoding="utf-8")
    df_both.to_csv(folder / "medical_parallel_both__TXT.csv", index=False, encoding="utf-8")

    (folder / "medical_txt_build_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("DONE")
    print(f"medical_parallel_en2fj__TXT.csv rows={len(df_en2fj):,}")
    print(f"medical_parallel_fj2en__TXT.csv rows={len(df_fj2en):,}")
    print(f"medical_parallel_both__TXT.csv  rows={len(df_both):,}")
    print("see: medical_txt_build_summary.txt")

if __name__ == "__main__":
    main()
