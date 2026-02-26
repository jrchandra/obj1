#!/usr/bin/env python3
"""
fill_metrics_columns.py

Auto-fills translation quality metric columns for each system output column
against the reference column.

Works on your dataset format like:
- ref
- chatgpt, gemini, google_translate, microsoft_translate

Creates/fills these metrics per system:
- bleu__<system>
- chrf__<system>
- meteor__<system>
- len_ratio__<system>
- edit_dist__<system>
- exact_match__<system>

Also creates placeholders (left as NaN) for:
- ter__<system>, bertscore_f1__<system>, comet__<system>, bleurt__<system>

Requirements:
  pip install pandas numpy nltk

First-time NLTK downloads (automatic in script):
  wordnet, omw-1.4

Usage:
  python fill_metrics_columns.py \
      --input dataset_with_system_outputs.csv \
      --output dataset_with_system_outputs__metrics_filled.csv

If you want to overwrite input:
  python fill_metrics_columns.py --input in.csv --output in.csv --overwrite
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


# -----------------------------
# Text utilities
# -----------------------------
def norm_text(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simple_tokenize(s: str):
    s = norm_text(s)
    toks = []
    for t in s.split():
        t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
        if t:
            toks.append(t)
    return toks


# -----------------------------
# Metrics
# -----------------------------
_smooth = SmoothingFunction().method1


def sent_bleu(ref: str, hyp: str):
    rt = simple_tokenize(ref)
    ht = simple_tokenize(hyp)
    if not rt or not ht:
        return np.nan
    try:
        return float(sentence_bleu([rt], ht, smoothing_function=_smooth))
    except Exception:
        return np.nan


def levenshtein(a: str, b: str) -> int:
    a = norm_text(a)
    b = norm_text(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # O(min(n,m)) space
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def chr_ngrams(s: str, n: int):
    s = norm_text(s).replace(" ", "")
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def chrf(ref: str, hyp: str, n: int = 6, beta: float = 2.0):
    ref = norm_text(ref)
    hyp = norm_text(hyp)
    if not ref or not hyp:
        return np.nan

    total_prec = 0.0
    total_rec = 0.0
    count = 0

    for k in range(1, n + 1):
        r = chr_ngrams(ref, k)
        h = chr_ngrams(hyp, k)
        if not r or not h:
            continue

        r_counts = {}
        for g in r:
            r_counts[g] = r_counts.get(g, 0) + 1

        h_counts = {}
        for g in h:
            h_counts[g] = h_counts.get(g, 0) + 1

        matches = 0
        for g, c in h_counts.items():
            matches += min(c, r_counts.get(g, 0))

        prec = matches / max(1, len(h))
        rec = matches / max(1, len(r))

        total_prec += prec
        total_rec += rec
        count += 1

    if count == 0:
        return np.nan

    P = total_prec / count
    R = total_rec / count
    if P == 0 and R == 0:
        return 0.0

    b2 = beta * beta
    return float((1 + b2) * P * R / (b2 * P + R))


def meteor(ref: str, hyp: str):
    rt = simple_tokenize(ref)
    ht = simple_tokenize(hyp)
    if not rt or not ht:
        return np.nan
    try:
        # meteor_score accepts token lists
        return float(meteor_score([rt], ht))
    except Exception:
        return np.nan


def length_ratio(ref: str, hyp: str):
    rt = simple_tokenize(ref)
    ht = simple_tokenize(hyp)
    if not rt or not ht:
        return np.nan
    return float(len(ht) / len(rt))


def exact_match(ref: str, hyp: str):
    if ref is None or hyp is None:
        return np.nan
    return float(norm_text(ref) == norm_text(hyp))


# -----------------------------
# Main
# -----------------------------
def ensure_nltk():
    # METEOR may require these corpora in many NLTK setups
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--ref_col", default="ref", help="Reference column name (default: ref)")
    parser.add_argument(
        "--systems",
        default="chatgpt,gemini,google_translate,microsoft_translate",
        help="Comma-separated system columns",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow output to overwrite input")
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output

    if (os.path.abspath(in_path) == os.path.abspath(out_path)) and not args.overwrite:
        print("ERROR: input and output paths are the same. Use --overwrite to allow this.", file=sys.stderr)
        sys.exit(1)

    ensure_nltk()

    df = pd.read_csv(in_path)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    ref_col = args.ref_col

    if ref_col not in df.columns:
        print(f"ERROR: reference column '{ref_col}' not found in CSV.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    for s in systems:
        if s not in df.columns:
            print(f"ERROR: system column '{s}' not found in CSV.", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    # Create columns (filled or placeholder)
    compute_metrics = {
        "bleu": sent_bleu,
        "chrf": chrf,
        "meteor": meteor,
        "len_ratio": length_ratio,
        "edit_dist": lambda r, h: float(levenshtein(r, h)) if (r is not None and h is not None) else np.nan,
        "exact_match": exact_match,
    }
    placeholder_metrics = ["ter", "bertscore_f1", "comet", "bleurt"]

    # Ensure columns exist
    for m in list(compute_metrics.keys()) + placeholder_metrics:
        for syscol in systems:
            col = f"{m}__{syscol}"
            if col not in df.columns:
                df[col] = np.nan

    # Fill computed metrics
    refs = df[ref_col].astype("string")
    for syscol in systems:
        hyps = df[syscol].astype("string")
        for m, fn in compute_metrics.items():
            col = f"{m}__{syscol}"
            df[col] = [fn(r, h) for r, h in zip(refs, hyps)]

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("Filled metrics:", ", ".join(compute_metrics.keys()))
    print("Placeholders kept NaN:", ", ".join(placeholder_metrics))


if __name__ == "__main__":
    main()
