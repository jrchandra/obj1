#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create stratified manual evaluation sample + annotation template.

Outputs: manual_eval_pack/
  - manual_eval_sample.csv (rows to annotate)
  - annotation_template.csv (blank columns to fill)
  - sampling_summary.csv

Annotators can fill:
  - adequacy (0-2)
  - fluency (0-2)
  - meaning_preserved (yes/no)
  - issues (comma list)
  - notes
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="master_parallel_corpus__COMBINED.csv")
    ap.add_argument("--outdir", default="manual_eval_pack")
    ap.add_argument("--n_per_stratum", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    for c in ["domain","direction","sentence_type_fine","source_id","source_text","target_text","source_doc"]:
        if c not in df.columns:
            df[c] = ""

    df = df[(df["source_text"].astype(str).str.strip() != "") & (df["target_text"].astype(str).str.strip() != "")].copy()
    df["domain"] = df["domain"].replace("", "unknown")
    df["direction"] = df["direction"].replace("", "unknown")
    df["sentence_type_fine"] = df["sentence_type_fine"].replace("", "unknown")

    strata_cols = ["domain","direction","sentence_type_fine"]
    rng = np.random.default_rng(args.seed)

    samples = []
    summary_rows = []

    for key, g in df.groupby(strata_cols):
        n = min(args.n_per_stratum, len(g))
        if n <= 0:
            continue
        idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
        samples.append(df.loc[idx])
        summary_rows.append({
            "domain": key[0],
            "direction": key[1],
            "sentence_type_fine": key[2],
            "available_rows": len(g),
            "sampled_rows": n
        })

    sample_df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=df.columns)

    keep_cols = ["domain","direction","sentence_type_fine","source_id","source_doc","source_text","target_text"]
    sample_df = sample_df[keep_cols].copy()

    # Annotation template
    template = sample_df.copy()
    template["adequacy_0_2"] = ""
    template["fluency_0_2"] = ""
    template["meaning_preserved_yes_no"] = ""
    template["issues"] = ""  # e.g., "wrong_term,missing,extra,ocr_noise,grammar"
    template["notes"] = ""

    sample_df.to_csv(outdir / "manual_eval_sample.csv", index=False, encoding="utf-8")
    template.to_csv(outdir / "annotation_template.csv", index=False, encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(outdir / "sampling_summary.csv", index=False, encoding="utf-8")

    print("DONE (manual eval pack)")
    print(f"wrote: {outdir}")

if __name__ == "__main__":
    main()
