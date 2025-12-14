#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST quality audit for a combined parallel corpus.

Input: master_parallel_corpus__COMBINED.csv
Outputs: quality_reports_fast/
  - quality_flags_per_row.csv
  - quality_summary_overall.csv
  - quality_summary_by_domain.csv
  - flagged_examples_top.csv

Checks:
  - empty / too short
  - duplicate pairs
  - suspicious glued tokens (e.g., "inthe", "andthe")
  - mojibake/encoding artifacts (â€œ â€™ etc.)
  - non-printable chars
  - digit presence (should be minimal if removed)
  - length ratio anomalies (source vs target)
  - light language plausibility (English stopwords vs Fijian cues)
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
from collections import Counter
import pandas as pd

WS_RE = re.compile(r"\s+")
MOJIBAKE = ["â€œ","â€","â€™","â€˜","Â","â€“","â€”","â€¦"]
GLUE_PAT = re.compile(r"\b(?:inthe|andthe|onthe|ofthe|tothe|fromthe|letthere|becomea|therewas|andthere)\b")
NONPRINT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

EN_STOP = set("""
the and to of in on for with without from is are was were be been being a an this that
i you he she it we they my your our their as by at into about over under between
""".split())

FJ_CUES = set("""
na e me mo ni kei ka ga tiko sega vakalevu vinaka yalovinaka
""".split())

def norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("nan", "")
    s = WS_RE.sub(" ", s).strip()
    return s

def token_words(s: str):
    return re.findall(r"[a-zA-Z']+", s.lower())

def english_score(s: str) -> float:
    toks = token_words(s)
    if not toks: return 0.0
    return sum(1 for t in toks if t in EN_STOP) / max(1, len(toks))

def fijian_score(s: str) -> float:
    toks = token_words(s)
    if not toks: return 0.0
    return sum(1 for t in toks if t in FJ_CUES) / max(1, len(toks))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="master_parallel_corpus__COMBINED.csv")
    ap.add_argument("--outdir", default="quality_reports_fast")
    ap.add_argument("--min_chars", type=int, default=6)
    ap.add_argument("--ratio_hi", type=float, default=3.0, help="flag if max(len)/min(len) exceeds this")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    for c in ["domain","direction","source_text","target_text"]:
        if c not in df.columns:
            df[c] = ""

    df["domain"] = df["domain"].map(lambda x: norm(x) if norm(x) else "unknown")
    df["direction"] = df["direction"].map(lambda x: norm(x) if norm(x) else "unknown")
    df["source_text"] = df["source_text"].map(norm)
    df["target_text"] = df["target_text"].map(norm)

    # features
    df["src_len"] = df["source_text"].map(len)
    df["tgt_len"] = df["target_text"].map(len)

    # avoid div-by-zero
    def ratio(a,b):
        a = max(1, a); b = max(1, b)
        return max(a,b) / min(a,b)

    df["len_ratio"] = [ratio(a,b) for a,b in zip(df["src_len"], df["tgt_len"])]

    # row flags
    flags = []

    # duplicate signature
    sig = (df["direction"] + "||" + df["source_text"] + "||" + df["target_text"])
    dup_mask = sig.duplicated(keep="first")
    df["flag_duplicate"] = dup_mask

    for i, row in df.iterrows():
        src = row["source_text"]
        tgt = row["target_text"]
        row_flags = []

        if row["src_len"] < args.min_chars or row["tgt_len"] < args.min_chars:
            row_flags.append("too_short_or_empty")

        if any(m in src for m in MOJIBAKE) or any(m in tgt for m in MOJIBAKE):
            row_flags.append("mojibake")

        if GLUE_PAT.search(src) or GLUE_PAT.search(tgt):
            row_flags.append("glued_tokens")

        if NONPRINT.search(src) or NONPRINT.search(tgt):
            row_flags.append("non_printable")

        # digits (should be low if removed)
        if re.search(r"\d", src) or re.search(r"\d", tgt):
            row_flags.append("has_digits")

        # length ratio
        if row["len_ratio"] >= args.ratio_hi:
            row_flags.append("length_ratio_high")

        # light language plausibility (weak heuristic)
        # If direction says en->fj but english_score on target is higher than source, suspicious
        es_src = english_score(src)
        es_tgt = english_score(tgt)
        fs_src = fijian_score(src)
        fs_tgt = fijian_score(tgt)

        if "en->fj" in row["direction"]:
            if es_tgt > es_src and es_tgt > 0.10:
                row_flags.append("lang_suspect_target_english")
        if "fj->en" in row["direction"]:
            if es_src > es_tgt and es_src > 0.10:
                row_flags.append("lang_suspect_source_english")

        # For dictionary/definition: headwords may look "englishy" or "fijiany" — don’t over-flag
        if row["domain"] in ["definition","dictionary"]:
            row_flags = [f for f in row_flags if not f.startswith("lang_suspect")]

        flags.append("|".join(row_flags) if row_flags else "")

        df.at[i, "english_score_src"] = es_src
        df.at[i, "english_score_tgt"] = es_tgt
        df.at[i, "fijian_score_src"] = fs_src
        df.at[i, "fijian_score_tgt"] = fs_tgt

    df["quality_flags"] = flags
    df["is_flagged"] = df["quality_flags"].map(bool)

    # summaries
    overall = pd.DataFrame([{
        "rows_total": len(df),
        "rows_flagged": int(df["is_flagged"].sum()),
        "flagged_pct": round(df["is_flagged"].mean()*100, 2),
        "duplicate_pct": round(df["flag_duplicate"].mean()*100, 2),
        "avg_len_ratio": round(df["len_ratio"].mean(), 3),
        "p90_len_ratio": round(df["len_ratio"].quantile(0.90), 3),
        "p95_len_ratio": round(df["len_ratio"].quantile(0.95), 3),
    }])

    by_domain = (df.groupby("domain")
                   .agg(rows=("domain","size"),
                        flagged=("is_flagged","sum"),
                        flagged_pct=("is_flagged", lambda x: round(x.mean()*100,2)),
                        dup_pct=("flag_duplicate", lambda x: round(x.mean()*100,2)),
                        p90_len_ratio=("len_ratio", lambda x: round(x.quantile(0.90),3)))
                   .reset_index()
                   .sort_values("rows", ascending=False))

    # top flagged examples
    flagged = df[df["is_flagged"]].copy()
    # score severity by number of flags
    flagged["num_flags"] = flagged["quality_flags"].map(lambda s: 0 if not s else len(s.split("|")))
    flagged = flagged.sort_values(["num_flags","len_ratio"], ascending=[False, False]).head(200)

    out_cols = ["domain","direction","source_text","target_text","quality_flags","len_ratio","flag_duplicate",
                "english_score_src","english_score_tgt","fijian_score_src","fijian_score_tgt"]
    df[out_cols].to_csv(outdir / "quality_flags_per_row.csv", index=False, encoding="utf-8")
    overall.to_csv(outdir / "quality_summary_overall.csv", index=False, encoding="utf-8")
    by_domain.to_csv(outdir / "quality_summary_by_domain.csv", index=False, encoding="utf-8")
    flagged[out_cols + ["num_flags"]].to_csv(outdir / "flagged_examples_top.csv", index=False, encoding="utf-8")

    print("DONE (fast audit)")
    print(f"wrote: {outdir}")

if __name__ == "__main__":
    main()
