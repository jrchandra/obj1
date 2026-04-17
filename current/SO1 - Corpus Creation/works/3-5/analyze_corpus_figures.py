#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate paper-ready figures (PNGs) from the combined corpus.

Input:
  master_parallel_corpus__COMBINED.csv

Outputs:
  corpus_figures/
    - fig_domain_distribution.png
    - fig_direction_distribution.png
    - fig_sentence_type_by_domain_stacked.png
    - fig_sentence_type_fine_top_by_domain.png
    - fig_src_token_hist.png
    - fig_tgt_token_hist.png
    - fig_src_token_box_by_domain.png
    - fig_tgt_token_box_by_domain.png

Install:
  pip install pandas matplotlib
Run:
  python analyze_corpus_figures.py --input master_parallel_corpus__COMBINED.csv
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

TOKEN_RE = re.compile(r"[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF']+")

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def tokenize(s: str):
    s = normalize_ws(s).lower()
    return TOKEN_RE.findall(s)

def add_len(df: pd.DataFrame) -> pd.DataFrame:
    df["src_tokens"] = df["source_text"].astype(str).map(lambda x: len(tokenize(x)))
    df["tgt_tokens"] = df["target_text"].astype(str).map(lambda x: len(tokenize(x)))
    return df

def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="master_parallel_corpus__COMBINED.csv")
    ap.add_argument("--outdir", type=str, default="corpus_figures")
    ap.add_argument("--topk_fine", type=int, default=8, help="Top fine types to show per domain")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)

    # ensure columns
    for c in ["domain","direction","sentence_type","sentence_type_fine","source_text","target_text"]:
        if c not in df.columns:
            df[c] = ""

    df["domain"] = df["domain"].astype(str).map(lambda x: normalize_ws(x) if normalize_ws(x) else "unknown")
    df["direction"] = df["direction"].astype(str).map(lambda x: normalize_ws(x) if normalize_ws(x) else "unknown")
    df["sentence_type"] = df["sentence_type"].astype(str).map(lambda x: normalize_ws(x) if normalize_ws(x) else "unknown")
    df["sentence_type_fine"] = df["sentence_type_fine"].astype(str).map(lambda x: normalize_ws(x) if normalize_ws(x) else "unknown")
    df["source_text"] = df["source_text"].astype(str).map(normalize_ws)
    df["target_text"] = df["target_text"].astype(str).map(normalize_ws)

    df = df[(df["source_text"] != "") & (df["target_text"] != "")].copy()
    df = add_len(df)

    # -------------------------
    # Fig 1: Domain distribution
    # -------------------------
    dom_counts = df["domain"].value_counts().sort_values(ascending=False)
    plt.figure()
    dom_counts.plot(kind="bar")
    plt.title("domain distribution (rows)")
    plt.xlabel("domain")
    plt.ylabel("rows")
    savefig(outdir / "fig_domain_distribution.png")

    # -------------------------
    # Fig 2: Direction distribution
    # -------------------------
    dir_counts = df["direction"].value_counts().sort_values(ascending=False)
    plt.figure()
    dir_counts.plot(kind="bar")
    plt.title("direction distribution (rows)")
    plt.xlabel("direction")
    plt.ylabel("rows")
    savefig(outdir / "fig_direction_distribution.png")

    # -------------------------
    # Fig 3: Stacked sentence_type by domain
    # -------------------------
    pivot = pd.pivot_table(
        df, index="domain", columns="sentence_type", values="source_text",
        aggfunc="count", fill_value=0
    )
    pivot = pivot.loc[dom_counts.index]  # order domains
    plt.figure()
    pivot.plot(kind="bar", stacked=True)
    plt.title("sentence_type by domain (stacked counts)")
    plt.xlabel("domain")
    plt.ylabel("rows")
    plt.legend(title="sentence_type", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(outdir / "fig_sentence_type_by_domain_stacked.png")

    # -------------------------
    # Fig 4: sentence_type_fine (top-K per domain)
    # (simple: show only types that are globally common)
    # -------------------------
    fine_counts = df["sentence_type_fine"].value_counts()
    top_global = fine_counts.head(20).index.tolist()
    df_fine = df[df["sentence_type_fine"].isin(top_global)].copy()
    pivot_fine = pd.pivot_table(
        df_fine, index="domain", columns="sentence_type_fine", values="source_text",
        aggfunc="count", fill_value=0
    )
    pivot_fine = pivot_fine.loc[dom_counts.index]
    plt.figure()
    pivot_fine.plot(kind="bar", stacked=True)
    plt.title("sentence_type_fine by domain (top global types, stacked)")
    plt.xlabel("domain")
    plt.ylabel("rows")
    plt.legend(title="sentence_type_fine", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(outdir / "fig_sentence_type_fine_top_by_domain.png")

    # -------------------------
    # Fig 5: Source token histogram
    # -------------------------
    plt.figure()
    plt.hist(df["src_tokens"], bins=40)
    plt.title("source sentence length (tokens) histogram")
    plt.xlabel("tokens")
    plt.ylabel("count")
    savefig(outdir / "fig_src_token_hist.png")

    # -------------------------
    # Fig 6: Target token histogram
    # -------------------------
    plt.figure()
    plt.hist(df["tgt_tokens"], bins=40)
    plt.title("target sentence length (tokens) histogram")
    plt.xlabel("tokens")
    plt.ylabel("count")
    savefig(outdir / "fig_tgt_token_hist.png")

    # -------------------------
    # Fig 7/8: Boxplots by domain (tokens)
    # -------------------------
    # Keep top 10 domains to avoid unreadable plots
    top_domains = dom_counts.head(10).index.tolist()
    dtop = df[df["domain"].isin(top_domains)].copy()

    # source
    plt.figure()
    data = [dtop.loc[dtop["domain"] == dom, "src_tokens"].tolist() for dom in top_domains]
    plt.boxplot(data, labels=top_domains, vert=True)
    plt.title("source tokens by domain (boxplot)")
    plt.xlabel("domain")
    plt.ylabel("tokens")
    plt.xticks(rotation=45, ha="right")
    savefig(outdir / "fig_src_token_box_by_domain.png")

    # target
    plt.figure()
    data = [dtop.loc[dtop["domain"] == dom, "tgt_tokens"].tolist() for dom in top_domains]
    plt.boxplot(data, labels=top_domains, vert=True)
    plt.title("target tokens by domain (boxplot)")
    plt.xlabel("domain")
    plt.ylabel("tokens")
    plt.xticks(rotation=45, ha="right")
    savefig(outdir / "fig_tgt_token_box_by_domain.png")

    print("DONE (figures)")
    print(f"wrote folder: {outdir}")

if __name__ == "__main__":
    main()
