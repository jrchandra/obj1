#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine all domain corpora into one master corpus + auto-populate sentence_type fields.

Inputs:
  - All .csv / .tsv in the input folder (default: script folder)
  - Optional .xlsx / .xls (if present)

Outputs:
  - master_parallel_corpus__COMBINED.csv
  - master_parallel_corpus__COMBINED.xlsx (optional if --xlsx)
  - master_corpus_build_report.txt
  - master_corpus_stats_by_domain.csv

Adds columns (auto-populated):
  - sentence_type (coarse)
  - sentence_type_fine (fine)

Assumptions:
  - Your per-domain files already contain at least:
      source_text, target_text
    and preferably:
      domain, subdomain, direction, source_lang, target_lang, source_id, source_doc

  - Text is often lowercase already, punctuation kept, digits removed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

# -------------------------
# Required / Standard columns
# -------------------------
BASE_COLS = [
    "domain", "subdomain", "source_id", "source_doc",
    "source_lang", "target_lang", "direction",
    "source_text", "target_text"
]

NEW_COLS = ["sentence_type", "sentence_type_fine"]

# If some files use alternate column names, map them here:
ALIASES = {
    "src_text": "source_text",
    "tgt_text": "target_text",
    "src": "source_text",
    "tgt": "target_text",
    "lang_source": "source_lang",
    "lang_target": "target_lang",
    "doc": "source_doc",
    "id": "source_id",
}

# -------------------------
# Utility: normalization
# -------------------------
WS_RE = re.compile(r"\s+")

def norm_ws(s: str) -> str:
    return WS_RE.sub(" ", str(s)).strip()

def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s

# -------------------------
# Sentence type classification (coarse + fine)
# -------------------------

# Basic cues (EN + FJ/iTaukei)
RE_QUESTION = re.compile(r"\?\s*$")
RE_BULLET = re.compile(r"^\s*(?:[-•–—]|\(?[a-z]\)|\(?\d+\)|[a-z]\.)\s+")
RE_SECTION = re.compile(r"^\s*(?:section|clause|article|part|chapter)\b", re.IGNORECASE)

# Warning/prohibition cues
WARN_CUES_EN = ["warning", "caution", "do not", "don't", "never", "avoid", "must not"]
WARN_CUES_FJ = ["kua ni", "kakua ni", "kakua", "kakua ni cakava"]  # common prohibition cues

# Instruction cues
INST_CUES_EN = ["please", "use", "keep", "wash", "stay", "call", "go to", "wear", "follow", "take", "seek"]
INST_CUES_FJ = ["yalovinaka", "mo", "tiko", "cakava", "murika", "vakayagataka", "qarauna", "lako"]

# Definition cues
DEF_CUES_EN = ["means", "is defined as", "refers to", "definition", "is a", "are"]
DEF_CUES_FJ = ["kena ibalebale", "e kena ibalebale", "baleta ni", "sa dua na", "sa"]

# Greeting/dialogue cues
GREET_CUES_EN = ["hello", "hi", "good morning", "good afternoon", "good evening", "thanks", "thank you"]
GREET_CUES_FJ = ["bula", "ni sa bula", "yadra", "moce", "vinaka", "vinaka vakalevu", "tavale"]

# Symptom/health cues (fine labeling)
SYMPTOM_CUES_EN = ["symptom", "symptoms", "fever", "cough", "pain", "tired", "fatigue", "headache", "sore"]
SYMPTOM_CUES_FJ = ["ivakatakilakila", "moca", "katakata", "mavoa", "maco", "rerevaki", "cati"]

def classify_sentence_domain_aware(text: str, domain: str) -> tuple[str, str]:
    """
    Returns:
      sentence_type (coarse): short | medium | long
      sentence_type_fine (domain-specific)
    """

    t = norm_ws(text).lower()
    n = len(t)

    # -------- coarse length --------
    if n <= 45:
        coarse = "short"
    elif n <= 120:
        coarse = "medium"
    else:
        coarse = "long"

    if not t:
        return coarse, "unknown"

    # ================= MEDICAL =================
    if domain == "medical":
        if any(x in t for x in ["symptom", "fever", "cough", "pain", "ivakatakilakila", "katakata", "moca"]):
            return coarse, "symptom"
        if any(x in t for x in ["wash", "clean", "protect", "prevent", "savata", "qarauna", "maroroya"]):
            return coarse, "prevention"
        if any(x in t for x in ["take", "use", "seek", "call", "lako", "vakayagataka"]):
            return coarse, "instruction"
        if any(x in t for x in ["do not", "must not", "never", "kakua", "kua ni"]):
            return coarse, "warning"
        if any(x in t for x in ["treatment", "medicine", "doctor", "veiqaravi"]):
            return coarse, "treatment"
        return coarse, "information"

    # ================= LEGAL =================
    if domain == "legal":
        if any(x in t for x in ["means", "defined as", "kena ibalebale"]):
            return coarse, "definition"
        if any(x in t for x in ["has the right", "right to", "dodonu me"]):
            return coarse, "right"
        if any(x in t for x in ["must", "shall", "dodonu me"]):
            return coarse, "obligation"
        if any(x in t for x in ["must not", "shall not", "e sega ni tara"]):
            return coarse, "prohibition"
        return coarse, "legal_clause"

    # ================= RELIGION =================
    if domain == "religion":
        if any(x in t for x in ["the lord", "god said", "sa kaya na kalou"]):
            return coarse, "narrative"
        if any(x in t for x in ["thou shalt", "mo", "kakua"]):
            return coarse, "command"
        if any(x in t for x in ["blessed", "promise", "vosavakalou"]):
            return coarse, "promise"
        if any(x in t for x in ["curse", "destroy", "punishment"]):
            return coarse, "warning"
        return coarse, "doctrine"

    # ================= CONVERSATIONAL =================
    if domain == "conversational":
        if t.startswith(("hello", "hi", "bula", "ni sa bula")):
            return coarse, "greeting"
        if t.endswith("?"):
            return coarse, "question"
        if any(x in t for x in ["please", "can you", "yalovinaka"]):
            return coarse, "request"
        return coarse, "statement"

    # ================= IDIOM =================
    if domain == "idiom":
        if any(x in t for x in ["means", "meaning", "ibalebale"]):
            return coarse, "idiom_meaning"
        return coarse, "idiom_literal"

    # ================= DEFINITION =================
    if domain == "definition":
        if len(t.split()) <= 2:
            return coarse, "headword"
        if any(x in t for x in ["means", "refers to", "kena ibalebale"]):
            return coarse, "definition"
        return coarse, "example"

    return coarse, "statement"


# -------------------------
# File loading + standardization
# -------------------------

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names
    rename = {}
    for c in df.columns:
        c2 = str(c).strip()
        ckey = c2.lower()
        if ckey in ALIASES:
            rename[c] = ALIASES[ckey]
    if rename:
        df = df.rename(columns=rename)

    # ensure all base cols exist
    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = ""

    # ensure required text cols exist
    if "source_text" not in df.columns:
        df["source_text"] = ""
    if "target_text" not in df.columns:
        df["target_text"] = ""

    # normalize text columns to strings
    df["source_text"] = df["source_text"].map(safe_str).map(norm_ws)
    df["target_text"] = df["target_text"].map(safe_str).map(norm_ws)

    # normalize other columns
    for c in ["domain","subdomain","source_id","source_doc","source_lang","target_lang","direction"]:
        df[c] = df[c].map(safe_str).map(norm_ws)

    return df

def infer_domain_from_filename(filename: str) -> str:
    f = filename.lower()
    if "genesis" in f or "bible" in f or "religion" in f:
        return "religion"
    if "constitution" in f or "legal" in f:
        return "legal"
    if "medical" in f or "dementia" in f or "screening" in f or "isolation" in f or "anxiety" in f or "depression" in f:
        return "medical"
    if "idiom" in f:
        return "idiom"
    if "conversational" in f:
        return "conversational"
    if "definition" in f:
        return "definition"
    return ""

def load_any_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        # read all sheets and concat
        xls = pd.ExcelFile(path)
        parts = []
        for sh in xls.sheet_names:
            d = pd.read_excel(path, sheet_name=sh, dtype=str)
            d["__sheet__"] = sh
            parts.append(d)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    raise ValueError(f"Unsupported file type: {path.suffix}")

def autopopulate_missing_meta(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    # domain/subdomain defaults
    inferred_domain = infer_domain_from_filename(filename)

    if (df["domain"] == "").all() and inferred_domain:
        df["domain"] = inferred_domain
    else:
        # fill blanks only
        df.loc[df["domain"] == "", "domain"] = inferred_domain or "unknown"

    # subdomain: prefer existing; otherwise from filename stem
    stem = Path(filename).stem
    stem = re.sub(r"__.*$", "", stem)  # drop double-underscore suffixes
    df.loc[df["subdomain"] == "", "subdomain"] = stem.lower()

    # source_doc: fill blanks with filename (and sheet if exists)
    if "__sheet__" in df.columns:
        df.loc[df["source_doc"] == "", "source_doc"] = df["__sheet__"].map(lambda sh: f"{Path(filename).name}::{sh}")
    else:
        df.loc[df["source_doc"] == "", "source_doc"] = Path(filename).name

    # source_id: if missing, create stable row id (file-based)
    if (df["source_id"] == "").all():
        df = df.reset_index(drop=True)
        df["source_id"] = [f"{df.loc[i,'domain']}|{df.loc[i,'subdomain']}|{i+1:07d}" for i in range(len(df))]
    else:
        # fill blanks only
        blanks = df["source_id"] == ""
        if blanks.any():
            start = 1
            for i in df.index[blanks]:
                df.at[i, "source_id"] = f"{df.at[i,'domain']}|{df.at[i,'subdomain']}|missing|{start:07d}"
                start += 1

    return df

def add_sentence_types(df: pd.DataFrame) -> pd.DataFrame:
    # Choose which text to classify from: usually source_text
    # If source_text empty, fallback to target_text
    def choose_text(row):
        s = row.get("source_text", "")
        if s:
            return s
        return row.get("target_text", "")

    types = df.apply(
    lambda r: classify_sentence_domain_aware(
        choose_text(r),
        r.get("domain", "").lower()
    ),
    axis=1
)

    df["sentence_type"] = [t[0] for t in types]
    df["sentence_type_fine"] = [t[1] for t in types]
    return df

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=".", help="Folder containing per-domain corpus files")
    ap.add_argument("--out", type=str, default="master_parallel_corpus__COMBINED.csv", help="Output CSV name")
    ap.add_argument("--xlsx", action="store_true", help="Also export combined XLSX")
    ap.add_argument("--include", type=str, default="csv,tsv,xlsx", help="Comma list of types to include: csv,tsv,xlsx")
    ap.add_argument("--exclude", type=str, default="master_parallel_corpus__COMBINED", help="Exclude files whose name contains this substring")
    args = ap.parse_args()

    folder = Path(args.input).resolve()
    if not folder.exists():
        raise SystemExit(f"Input folder does not exist: {folder}")

    include = {x.strip().lower() for x in args.include.split(",") if x.strip()}
    ex_sub = args.exclude.lower().strip()

    # collect files
    candidates: List[Path] = []
    if "csv" in include:
        candidates += list(folder.glob("*.csv"))
    if "tsv" in include:
        candidates += list(folder.glob("*.tsv")) + list(folder.glob("*.txt"))
    if "xlsx" in include:
        candidates += list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))

    # filter excluded
    files = [p for p in sorted(set(candidates)) if ex_sub not in p.name.lower()]
    if not files:
        raise SystemExit("No input files found. Put your per-domain CSVs in the folder or adjust --input/--include.")

    report_lines = []
    frames = []

    for p in files:
        try:
            df = load_any_table(p)
            if df is None or df.empty:
                report_lines.append(f"[SKIP empty] {p.name}")
                continue

            df = standardize_columns(df)
            df = autopopulate_missing_meta(df, p.name)

            # Drop rows that have no usable text
            df = df[(df["source_text"].map(bool)) & (df["target_text"].map(bool))].copy()
            if df.empty:
                report_lines.append(f"[SKIP no-text] {p.name}")
                continue

            df = add_sentence_types(df)

            # keep only base + new cols (and optionally keep extras? we drop extras to standardize)
            df = df[BASE_COLS + NEW_COLS].copy()

            frames.append(df)
            report_lines.append(f"[OK] {p.name}: rows={len(df):,}")

        except Exception as e:
            report_lines.append(f"[FAIL] {p.name}: {e}")

    if not frames:
        raise SystemExit("No rows were combined (all files empty/failed). Check master_corpus_build_report.txt after running.")

    combined = pd.concat(frames, ignore_index=True)

    # final de-dupe (direction + text pair)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["direction","source_text","target_text"]).reset_index(drop=True)
    dedup_removed = before - len(combined)

    # stats by domain/subdomain/direction
    stats = (combined
             .groupby(["domain","subdomain","direction"], dropna=False)
             .size()
             .reset_index(name="rows")
             .sort_values(["domain","subdomain","direction"]))

    out_csv = folder / args.out
    combined.to_csv(out_csv, index=False, encoding="utf-8")

    stats_csv = folder / "master_corpus_stats_by_domain.csv"
    stats.to_csv(stats_csv, index=False, encoding="utf-8")

    report = folder / "master_corpus_build_report.txt"
    header = [
        "MASTER CORPUS BUILD REPORT",
        f"input_folder: {folder}",
        f"files_seen: {len(files)}",
        f"combined_rows_before_dedup: {before:,}",
        f"dedup_removed: {dedup_removed:,}",
        f"final_rows: {len(combined):,}",
        "",
        "FILES:",
    ]
    report.write_text("\n".join(header + report_lines) + "\n", encoding="utf-8")

    if args.xlsx:
        out_xlsx = folder / "master_parallel_corpus__COMBINED.xlsx"
        combined.to_excel(out_xlsx, index=False)

    print("DONE")
    print(f"combined CSV: {out_csv.name} rows={len(combined):,}")
    print(f"stats: {stats_csv.name}")
    print(f"report: {report.name}")
    if args.xlsx:
        print("also wrote: master_parallel_corpus__COMBINED.xlsx")

if __name__ == "__main__":
    main()
