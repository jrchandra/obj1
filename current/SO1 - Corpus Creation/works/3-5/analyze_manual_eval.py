#!/usr/bin/env python3
"""
analyze_manual_eval.py

Analyze human/manual evaluation annotations for a parallel corpus.

Input (your file):
  annotation_template_SYNTHETIC_FILLED.csv

Outputs (folder):
  manual_eval_analysis/
    summary_overall.txt
    summary_tables.xlsx
    group_stats_domain.csv
    group_stats_domain_direction.csv
    group_stats_sentence_type_fine.csv
    error_tag_counts.csv
    issues_counts.csv
    missingness.csv
    duplicates.csv
    plots/
      adequacy_1to5_hist.png
      fluency_1to5_hist.png
      overall_quality_hist.png
      meaning_preserved_by_domain.png
      mean_overall_by_domain.png
"""

import argparse
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Utility helpers
# ---------------------------

YES_SET = {"yes", "y", "true", "1", "preserved"}
NO_SET = {"no", "n", "false", "0", "not preserved", "not_preserved"}

def safe_read_csv(path: str) -> pd.DataFrame:
    # Robust read for mixed encodings / messy lines
    return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def normalize_yes_no(x) -> str:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if s in YES_SET:
        return "yes"
    if s in NO_SET:
        return "no"
    # handle variants like "Yes - mostly"
    if s.startswith("y"):
        return "yes"
    if s.startswith("n"):
        return "no"
    return np.nan

def split_tags(cell) -> list:
    """Split error_tags / issues cells into list of normalized tokens."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    # split on comma, semicolon, pipe, newline
    parts = re.split(r"[,\n;|]+", s)
    parts = [p.strip().lower() for p in parts if p.strip()]
    # collapse whitespace and remove duplicates per cell
    norm = []
    seen = set()
    for p in parts:
        p = re.sub(r"\s+", "_", p)
        if p not in seen:
            norm.append(p)
            seen.add(p)
    return norm

def explode_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tokens = df[col].apply(split_tags)
    exploded = tokens.explode()
    exploded = exploded.dropna()
    counts = exploded.value_counts().reset_index()
    counts.columns = [col, "count"]
    return counts

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def describe_numeric(s: pd.Series) -> dict:
    s = to_numeric(s)
    return {
        "n": int(s.notna().sum()),
        "mean": float(s.mean()) if s.notna().any() else np.nan,
        "std": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
        "min": float(s.min()) if s.notna().any() else np.nan,
        "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
        "median": float(s.median()) if s.notna().any() else np.nan,
        "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
        "max": float(s.max()) if s.notna().any() else np.nan,
    }

def percent_true_yes(s: pd.Series) -> dict:
    s2 = s.apply(normalize_yes_no)
    n = int(s2.notna().sum())
    yes = int((s2 == "yes").sum())
    no = int((s2 == "no").sum())
    pct_yes = (yes / n * 100.0) if n else np.nan
    return {"n": n, "yes": yes, "no": no, "pct_yes": float(pct_yes) if n else np.nan}


# ---------------------------
# Kappa (optional)
# ---------------------------

def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Unweighted Cohen's kappa for categorical series."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if df.empty:
        return np.nan
    cats = sorted(set(df["a"].unique()).union(set(df["b"].unique())))
    cat_to_i = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)), dtype=float)
    for x, y in zip(df["a"], df["b"]):
        m[cat_to_i[x], cat_to_i[y]] += 1
    n = m.sum()
    if n == 0:
        return np.nan
    po = np.trace(m) / n
    pe = (m.sum(axis=1) / n @ (m.sum(axis=0) / n))
    if math.isclose(1 - pe, 0):
        return np.nan
    return float((po - pe) / (1 - pe))

def weighted_kappa_ordinal(a: pd.Series, b: pd.Series, weights: str = "quadratic") -> float:
    """
    Weighted kappa for ordinal ratings (e.g., 1..5).
    weights: "linear" or "quadratic"
    """
    df = pd.DataFrame({"a": to_numeric(a), "b": to_numeric(b)}).dropna()
    if df.empty:
        return np.nan
    cats = sorted(set(df["a"].unique()).union(set(df["b"].unique())))
    # ensure ints if possible
    cats = [int(c) if float(c).is_integer() else float(c) for c in cats]
    cats = sorted(cats)
    cat_to_i = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    O = np.zeros((k, k), dtype=float)
    for x, y in zip(df["a"], df["b"]):
        x = int(x) if float(x).is_integer() else float(x)
        y = int(y) if float(y).is_integer() else float(y)
        if x in cat_to_i and y in cat_to_i:
            O[cat_to_i[x], cat_to_i[y]] += 1

    n = O.sum()
    if n == 0:
        return np.nan

    row = O.sum(axis=1)
    col = O.sum(axis=0)
    E = np.outer(row, col) / n

    # Weight matrix
    W = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            d = abs(i - j)
            if weights == "linear":
                W[i, j] = d / (k - 1) if k > 1 else 0.0
            else:  # quadratic
                W[i, j] = (d ** 2) / ((k - 1) ** 2) if k > 1 else 0.0

    num = (W * O).sum()
    den = (W * E).sum()
    if math.isclose(den, 0):
        return np.nan
    return float(1 - (num / den))


# ---------------------------
# Core analysis
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, help="Path to annotation CSV (manual eval)")
    ap.add_argument("--outdir", default="manual_eval_analysis", help="Output directory")
    ap.add_argument("--plots", action="store_true", help="Save plots as PNG")
    args = ap.parse_args()

    df = safe_read_csv(args.annotations)

    ensure_dir(args.outdir)
    plot_dir = os.path.join(args.outdir, "plots")
    if args.plots:
        ensure_dir(plot_dir)

    # Normalize key columns if present
    if "meaning_preserved_yes_no" in df.columns:
        df["meaning_preserved_yes_no"] = df["meaning_preserved_yes_no"].apply(normalize_yes_no)

    # Missingness
    missing = (df.isna().mean() * 100).sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_pct"]
    missing.to_csv(os.path.join(args.outdir, "missingness.csv"), index=False)

    # Duplicate check by source_id + direction if available
    dup_rows = pd.DataFrame()
    if {"source_id", "direction"}.issubset(df.columns):
        dup_mask = df.duplicated(subset=["source_id", "direction"], keep=False)
        dup_rows = df.loc[dup_mask, ["source_id", "direction", "domain", "source_doc"]].copy()
        dup_rows.to_csv(os.path.join(args.outdir, "duplicates.csv"), index=False)

    # Overall numeric stats
    numeric_cols = [c for c in ["adequacy_1to5", "fluency_1to5", "overall_quality", "adequacy_0_2", "fluency_0_2"] if c in df.columns]
    overall_stats = []
    for c in numeric_cols:
        d = describe_numeric(df[c])
        d["metric"] = c
        overall_stats.append(d)
    overall_stats_df = pd.DataFrame(overall_stats)[["metric","n","mean","std","min","p25","median","p75","max"]]

    # Meaning preserved
    mp = {}
    if "meaning_preserved_yes_no" in df.columns:
        mp = percent_true_yes(df["meaning_preserved_yes_no"])

    # Group stats
    def group_stats(group_cols: list, name: str):
        if not set(group_cols).issubset(df.columns):
            return None
        g = df.groupby(group_cols, dropna=False)
        out = g.size().reset_index(name="n_rows")
        for c in numeric_cols:
            out[f"{c}_mean"] = g[c].mean().values
            out[f"{c}_median"] = g[c].median().values
            out[f"{c}_n"] = g[c].count().values
        if "meaning_preserved_yes_no" in df.columns:
            out["meaning_yes_pct"] = (g["meaning_preserved_yes_no"].apply(lambda s: (s=="yes").mean()*100 if s.notna().any() else np.nan)).values
            out["meaning_n"] = g["meaning_preserved_yes_no"].count().values
        out.to_csv(os.path.join(args.outdir, f"{name}.csv"), index=False)
        return out

    group_stats_domain = group_stats(["domain"], "group_stats_domain")
    group_stats_domain_direction = group_stats(["domain","direction"], "group_stats_domain_direction")
    group_stats_sentence_type_fine = group_stats(["sentence_type_fine"], "group_stats_sentence_type_fine")

    # Error tag / issues counts
    if "error_tags" in df.columns:
        error_counts = explode_counts(df, "error_tags")
        error_counts.to_csv(os.path.join(args.outdir, "error_tag_counts.csv"), index=False)
    else:
        error_counts = pd.DataFrame(columns=["error_tags","count"])

    if "issues" in df.columns:
        issues_counts = explode_counts(df, "issues")
        issues_counts.to_csv(os.path.join(args.outdir, "issues_counts.csv"), index=False)
    else:
        issues_counts = pd.DataFrame(columns=["issues","count"])

    # Optional IAA: detect a likely annotator column
    annot_col = None
    for candidate in ["annotator_id", "annotator", "rater", "judge", "evaluator"]:
        if candidate in df.columns:
            annot_col = candidate
            break

    kappa_lines = []
    if annot_col and {"source_id","direction"}.issubset(df.columns):
        # pairwise kappa for first two annotators only (simple + practical)
        annotators = [a for a in df[annot_col].dropna().unique()]
        if len(annotators) >= 2:
            a1, a2 = annotators[0], annotators[1]
            d1 = df[df[annot_col] == a1]
            d2 = df[df[annot_col] == a2]
            key = ["source_id","direction"]
            merged = d1.merge(d2, on=key, suffixes=("_a1","_a2"))

            # Meaning preserved (categorical)
            if "meaning_preserved_yes_no_a1" in merged.columns and "meaning_preserved_yes_no_a2" in merged.columns:
                k = cohen_kappa(merged["meaning_preserved_yes_no_a1"], merged["meaning_preserved_yes_no_a2"])
                kappa_lines.append(f"Cohen kappa (meaning_preserved) between {a1} and {a2}: {k:.4f}")

            # Ordinal ratings
            for c in ["adequacy_1to5","fluency_1to5","overall_quality"]:
                ca = f"{c}_a1"; cb = f"{c}_a2"
                if ca in merged.columns and cb in merged.columns:
                    k = weighted_kappa_ordinal(merged[ca], merged[cb], weights="quadratic")
                    kappa_lines.append(f"Quadratic weighted kappa ({c}) between {a1} and {a2}: {k:.4f}")

    # Plots (optional)
    if args.plots:
        # histograms
        for c in ["adequacy_1to5", "fluency_1to5", "overall_quality"]:
            if c in df.columns:
                s = to_numeric(df[c]).dropna()
                if not s.empty:
                    plt.figure()
                    plt.hist(s.values, bins=10)
                    plt.title(f"{c} distribution")
                    plt.xlabel(c)
                    plt.ylabel("count")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f"{c}_hist.png"), dpi=200)
                    plt.close()

        # meaning preserved by domain
        if "meaning_preserved_yes_no" in df.columns and "domain" in df.columns:
            tmp = df.dropna(subset=["meaning_preserved_yes_no"]).copy()
            if not tmp.empty:
                p = tmp.groupby("domain")["meaning_preserved_yes_no"].apply(lambda s: (s=="yes").mean()*100).sort_values(ascending=False)
                plt.figure()
                plt.bar(p.index.astype(str), p.values)
                plt.title("Meaning preserved (yes %) by domain")
                plt.ylabel("percent")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "meaning_preserved_by_domain.png"), dpi=200)
                plt.close()

        # mean overall_quality by domain
        if "overall_quality" in df.columns and "domain" in df.columns:
            tmp = df.copy()
            tmp["overall_quality"] = to_numeric(tmp["overall_quality"])
            p = tmp.groupby("domain")["overall_quality"].mean().sort_values(ascending=False)
            if p.notna().any():
                plt.figure()
                plt.bar(p.index.astype(str), p.values)
                plt.title("Mean overall_quality by domain")
                plt.ylabel("mean")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "mean_overall_by_domain.png"), dpi=200)
                plt.close()

    # Write overall summary text
    out_txt = []
    out_txt.append("MANUAL EVALUATION ANALYSIS SUMMARY")
    out_txt.append("=" * 40)
    out_txt.append(f"Rows: {len(df)}")
    out_txt.append("")
    out_txt.append("Overall numeric metrics:")
    out_txt.append(overall_stats_df.to_string(index=False))
    out_txt.append("")
    if mp:
        out_txt.append("Meaning preserved (yes/no):")
        out_txt.append(f"n={mp['n']} yes={mp['yes']} no={mp['no']} yes%={mp['pct_yes']:.2f}")
        out_txt.append("")
    if dup_rows is not None and not dup_rows.empty:
        out_txt.append(f"Duplicates (source_id+direction): {len(dup_rows)} rows flagged")
        out_txt.append("")
    if kappa_lines:
        out_txt.append("Inter-annotator agreement (auto-detected):")
        out_txt.extend(kappa_lines)
        out_txt.append("")

    with open(os.path.join(args.outdir, "summary_overall.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_txt))

    # Also produce a single Excel workbook with key tables
    xlsx_path = os.path.join(args.outdir, "summary_tables.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        overall_stats_df.to_excel(w, sheet_name="overall_metrics", index=False)
        missing.to_excel(w, sheet_name="missingness", index=False)
        if group_stats_domain is not None: group_stats_domain.to_excel(w, sheet_name="by_domain", index=False)
        if group_stats_domain_direction is not None: group_stats_domain_direction.to_excel(w, sheet_name="by_domain_direction", index=False)
        if group_stats_sentence_type_fine is not None: group_stats_sentence_type_fine.to_excel(w, sheet_name="by_sentence_type_fine", index=False)
        if "error_tags" in df.columns: error_counts.to_excel(w, sheet_name="error_tags", index=False)
        if "issues" in df.columns: issues_counts.to_excel(w, sheet_name="issues", index=False)
        if not dup_rows.empty: dup_rows.to_excel(w, sheet_name="duplicates", index=False)

    print(f"[OK] Wrote outputs to: {args.outdir}")
    if args.plots:
        print(f"[OK] Plots saved to: {plot_dir}")
    print(f"[OK] Excel summary: {xlsx_path}")


if __name__ == "__main__":
    main()
