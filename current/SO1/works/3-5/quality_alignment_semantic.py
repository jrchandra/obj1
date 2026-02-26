#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic alignment scoring for parallel corpus pairs.

Computes cosine similarity between embeddings of source_text and target_text using a multilingual model.

Outputs: quality_reports_semantic/
  - semantic_scores_per_row.csv
  - semantic_summary_by_domain.csv
  - low_score_examples.csv

Note: This does NOT "prove" perfect translation, but strongly flags likely misalignment.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="master_parallel_corpus__COMBINED.csv")
    ap.add_argument("--outdir", default="quality_reports_semantic")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--low_thresh", type=float, default=0.35, help="flag if cosine < this threshold")
    ap.add_argument("--max_rows", type=int, default=0, help="0=all; otherwise limit for quick test")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Missing input: {in_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    for c in ["domain","direction","source_text","target_text","source_id"]:
        if c not in df.columns:
            df[c] = ""

    df = df[(df["source_text"].astype(str).str.strip() != "") & (df["target_text"].astype(str).str.strip() != "")].copy()
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    model = SentenceTransformer(args.model)

    src = df["source_text"].astype(str).tolist()
    tgt = df["target_text"].astype(str).tolist()

    emb_src = model.encode(src, batch_size=args.batch, show_progress_bar=True, normalize_embeddings=True)
    emb_tgt = model.encode(tgt, batch_size=args.batch, show_progress_bar=True, normalize_embeddings=True)

    # cosine similarity of corresponding rows
    scores = (emb_src * emb_tgt).sum(axis=1)  # since normalized
    df["semantic_cosine"] = scores

    df["semantic_flag_low"] = df["semantic_cosine"] < args.low_thresh

    # summaries
    by_domain = (df.groupby("domain")
                   .agg(rows=("domain","size"),
                        mean_cosine=("semantic_cosine","mean"),
                        p10=("semantic_cosine", lambda x: x.quantile(0.10)),
                        p25=("semantic_cosine", lambda x: x.quantile(0.25)),
                        low_flagged=("semantic_flag_low","sum"),
                        low_pct=("semantic_flag_low", lambda x: round(x.mean()*100,2)))
                   .reset_index()
                   .sort_values("rows", ascending=False))

    # examples
    low = df[df["semantic_flag_low"]].copy()
    low = low.sort_values("semantic_cosine", ascending=True).head(300)

    out_cols = ["domain","direction","source_id","source_text","target_text","semantic_cosine","semantic_flag_low"]
    df[out_cols].to_csv(outdir / "semantic_scores_per_row.csv", index=False, encoding="utf-8")
    by_domain.to_csv(outdir / "semantic_summary_by_domain.csv", index=False, encoding="utf-8")
    low[out_cols].to_csv(outdir / "low_score_examples.csv", index=False, encoding="utf-8")

    print("DONE (semantic alignment scoring)")
    print(f"wrote: {outdir}")

if __name__ == "__main__":
    main()
