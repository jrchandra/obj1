#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate paper-ready numeric tables from a combined parallel corpus CSV.

Input (default):
  master_parallel_corpus__COMBINED.csv

Outputs:
  corpus_tables/
    - table_overall_summary.csv
    - table_by_domain.csv
    - table_by_domain_direction.csv
    - table_sentence_type_by_domain.csv
    - table_sentence_type_fine_by_domain.csv
    - table_length_stats_by_domain.csv
    - table_top_tokens_by_domain_source.csv
    - table_missingness_report.csv
    - corpus_analysis_report.txt

Install:
  pip install pandas
Run:
  python analyze_corpus_tables.py --input master_parallel_corpus__COMBINED.csv
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from collections import Counter

import pandas as pd

REQ_TEXT_COLS = ["source_text", "target_text"]
META_COLS = ["domain", "subdomain", "direction", "source_lang", "target_lang", "source_id", "source_doc"]
TYPE_COLS = ["sentence_type", "sentence_type_fine"]

TOKEN_RE = re.compile(r"[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF']+")  # latin letters + apostrophe

def safe_col(df: pd.DataFrame, col: str, default="") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df))
    s = df[col].astype(str)
    s = s.replace("nan", "")
    return s

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def tokenize(s: str):
    s = normalize_ws(s).lower()
    return TOKEN_RE.findall(s)

def add_length_cols(df: pd.DataFrame) -> pd.DataFrame:
    src = safe_col(df, "source_text")
    tgt = safe_col(df, "target_text")
    df["src_chars"] = src.map(lambda x: len(normalize_ws(x)))
    df["tgt_chars"] = tgt.map(lambda x: len(normalize_ws(x)))
    df["src_tokens"] = src.map(lambda x: len(tokenize(x)))
    df["tgt_tokens"] = tgt.map(lambda x: len(tokenize(x)))
    return df

def pct(series: pd.Series) -> pd.Series:
    total = series.sum()
    if total == 0:
        return series * 0
    return (series / total) * 100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="master_parallel_corpus__COMBINED.csv")
    ap.add_argument("--outdir", type=str, default="corpus_tables")
    ap.add_argument("--topk", type=int, default=30, help="top tokens per domain (source_text)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
    # Ensure key columns exist
    for c in META_COLS + TYPE_COLS + REQ_TEXT_COLS:
        if c not in df.columns:
            df[c] = ""

    # Drop rows with empty text (paper analysis should reflect usable pairs)
    df["source_text"] = df["source_text"].map(normalize_ws)
    df["target_text"] = df["target_text"].map(normalize_ws)
    df = df[(df["source_text"] != "") & (df["target_text"] != "")].copy()

    df = add_length_cols(df)

    # ---------------------------
    # Table: overall summary
    # ---------------------------
    total_rows = len(df)
    domains = df["domain"].replace("", "unknown")
    directions = df["direction"].replace("", "unknown")

    overall = pd.DataFrame([{
        "total_rows": total_rows,
        "num_domains": domains.nunique(),
        "num_subdomains": df["subdomain"].replace("", "unknown").nunique(),
        "num_directions": directions.nunique(),
        "avg_src_tokens": round(df["src_tokens"].mean(), 2) if total_rows else 0,
        "avg_tgt_tokens": round(df["tgt_tokens"].mean(), 2) if total_rows else 0,
        "avg_src_chars": round(df["src_chars"].mean(), 2) if total_rows else 0,
        "avg_tgt_chars": round(df["tgt_chars"].mean(), 2) if total_rows else 0,
    }])
    overall.to_csv(outdir / "table_overall_summary.csv", index=False)

    # ---------------------------
    # Table: by domain
    # ---------------------------
    by_domain = (df.assign(domain=domains)
                 .groupby("domain", dropna=False)
                 .size()
                 .rename("rows")
                 .reset_index())
    by_domain["pct"] = pct(by_domain["rows"]).round(2)
    by_domain = by_domain.sort_values("rows", ascending=False)
    by_domain.to_csv(outdir / "table_by_domain.csv", index=False)

    # ---------------------------
    # Table: by domain + direction
    # ---------------------------
    by_dom_dir = (df.assign(domain=domains, direction=directions)
                  .groupby(["domain", "direction"], dropna=False)
                  .size()
                  .rename("rows")
                  .reset_index())
    by_dom_dir["pct_within_domain"] = (by_dom_dir.groupby("domain")["rows"]
                                       .transform(lambda x: (x / x.sum() * 100) if x.sum() else 0)).round(2)
    by_dom_dir = by_dom_dir.sort_values(["domain", "rows"], ascending=[True, False])
    by_dom_dir.to_csv(outdir / "table_by_domain_direction.csv", index=False)

    # ---------------------------
    # Table: sentence_type by domain
    # ---------------------------
    if df["sentence_type"].str.len().sum() > 0:
        t = df["sentence_type"].replace("", "unknown")
        by_type = (df.assign(domain=domains, sentence_type=t)
                   .groupby(["domain", "sentence_type"], dropna=False)
                   .size().rename("rows").reset_index())
        by_type["pct_within_domain"] = (by_type.groupby("domain")["rows"]
                                        .transform(lambda x: (x / x.sum() * 100) if x.sum() else 0)).round(2)
        by_type.to_csv(outdir / "table_sentence_type_by_domain.csv", index=False)
    else:
        pd.DataFrame(columns=["domain","sentence_type","rows","pct_within_domain"])\
          .to_csv(outdir / "table_sentence_type_by_domain.csv", index=False)

    # ---------------------------
    # Table: sentence_type_fine by domain
    # ---------------------------
    if df["sentence_type_fine"].str.len().sum() > 0:
        tf = df["sentence_type_fine"].replace("", "unknown")
        by_type_fine = (df.assign(domain=domains, sentence_type_fine=tf)
                        .groupby(["domain", "sentence_type_fine"], dropna=False)
                        .size().rename("rows").reset_index())
        by_type_fine["pct_within_domain"] = (by_type_fine.groupby("domain")["rows"]
                                             .transform(lambda x: (x / x.sum() * 100) if x.sum() else 0)).round(2)
        by_type_fine = by_type_fine.sort_values(["domain","rows"], ascending=[True, False])
        by_type_fine.to_csv(outdir / "table_sentence_type_fine_by_domain.csv", index=False)
    else:
        pd.DataFrame(columns=["domain","sentence_type_fine","rows","pct_within_domain"])\
          .to_csv(outdir / "table_sentence_type_fine_by_domain.csv", index=False)

    # ---------------------------
    # Table: length stats by domain
    # ---------------------------
    length_stats = (df.assign(domain=domains)
                    .groupby("domain", dropna=False)
                    .agg(
                        rows=("domain","size"),
                        src_tokens_mean=("src_tokens","mean"),
                        src_tokens_p50=("src_tokens","median"),
                        src_tokens_p90=("src_tokens", lambda x: x.quantile(0.90)),
                        tgt_tokens_mean=("tgt_tokens","mean"),
                        tgt_tokens_p50=("tgt_tokens","median"),
                        tgt_tokens_p90=("tgt_tokens", lambda x: x.quantile(0.90)),
                        src_chars_mean=("src_chars","mean"),
                        tgt_chars_mean=("tgt_chars","mean"),
                    )
                    .reset_index())
    # round nicely
    for c in length_stats.columns:
        if c.endswith(("mean","p50","p90")):
            length_stats[c] = length_stats[c].round(2)
    length_stats.to_csv(outdir / "table_length_stats_by_domain.csv", index=False)

    # ---------------------------
    # Table: top tokens by domain (source side)
    # ---------------------------
    topk = args.topk
    rows_out = []
    for dom, g in df.assign(domain=domains).groupby("domain"):
        counter = Counter()
        for s in g["source_text"]:
            counter.update(tokenize(s))
        for tok, cnt in counter.most_common(topk):
            rows_out.append({"domain": dom, "token": tok, "count": cnt})
    pd.DataFrame(rows_out).to_csv(outdir / "table_top_tokens_by_domain_source.csv", index=False)

    # ---------------------------
    # Missingness report
    # ---------------------------
    miss = []
    for c in META_COLS + TYPE_COLS + REQ_TEXT_COLS:
        miss.append({
            "column": c,
            "missing_rows": int((df[c].astype(str).map(normalize_ws) == "").sum()),
            "missing_pct": round(((df[c].astype(str).map(normalize_ws) == "").mean() * 100), 2)
        })
    pd.DataFrame(miss).to_csv(outdir / "table_missingness_report.csv", index=False)

    # ---------------------------
    # Plain text report for quick writing
    # ---------------------------
    report_lines = []
    report_lines.append("CORPUS ANALYSIS REPORT")
    report_lines.append(f"input: {in_path.name}")
    report_lines.append(f"usable_rows: {total_rows:,}")
    report_lines.append(f"domains: {domains.nunique():,}")
    report_lines.append(f"subdomains: {df['subdomain'].replace('', 'unknown').nunique():,}")
    report_lines.append(f"directions: {directions.nunique():,}")
    report_lines.append("")
    report_lines.append("Top domains by rows:")
    for _, r in by_domain.head(10).iterrows():
        report_lines.append(f"  - {r['domain']}: {int(r['rows']):,} ({r['pct']}%)")
    report_lines.append("")
    report_lines.append("Direction totals:")
    dir_totals = df.assign(direction=directions).groupby("direction").size().sort_values(ascending=False)
    for k, v in dir_totals.items():
        report_lines.append(f"  - {k}: {int(v):,}")
    report_lines.append("")
    report_lines.append("Length (tokens) overall:")
    report_lines.append(f"  - src mean={df['src_tokens'].mean():.2f}, median={df['src_tokens'].median():.2f}, p90={df['src_tokens'].quantile(0.90):.2f}")
    report_lines.append(f"  - tgt mean={df['tgt_tokens'].mean():.2f}, median={df['tgt_tokens'].median():.2f}, p90={df['tgt_tokens'].quantile(0.90):.2f}")

    (outdir / "corpus_analysis_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("DONE (tables)")
    print(f"wrote folder: {outdir}")

if __name__ == "__main__":
    main()
