#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO2 Part 6 Reporting Script
---------------------------
Generates reporting outputs for machine translation evaluation:
- overall summaries
- by direction
- by domain
- by sentence type
- human evaluation summaries
- correlations between automatic and human metrics
- Friedman + Wilcoxon statistical tests
- charts and heatmaps

Author: OpenAI
"""

import os
import re
import sys
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, wilcoxon, spearmanr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# Configuration
# =========================

DEFAULT_SYSTEMS = [
    "chatgpt",
    "gemini",
    "google",
    "microsoft"
]

# Automatic metrics expected as {system}_{metric}
AUTO_METRICS = ["bleu", "chrf", "ter", "comet"]

# Human metrics may be in wide format as {system}_{human_metric}
HUMAN_METRICS = [
    "fluency",
    "adequacy",
    "semantic_fidelity",
    "cultural_appropriateness",
    "overall_quality"
]

GROUP_COL_CANDIDATES = {
    "direction": ["direction", "translation_direction"],
    "domain": ["domain", "text_domain"],
    "sentence_type": ["sentence_type", "complexity", "sentence_category"]
}

ID_COL_CANDIDATES = ["id", "sample_id", "row_id", "uid"]

SYSTEM_ALIASES = {
    "chatgpt": ["chatgpt", "gpt", "openai"],
    "gemini": ["gemini"],
    "google": ["google", "google_translate", "google_nmt"],
    "microsoft": ["microsoft", "ms", "microsoft_translator", "bing"]
}


# =========================
# Utility helpers
# =========================

def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_colname(col: str) -> str:
    c = str(col).strip().lower()
    c = re.sub(r"[^\w]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def load_table(input_path: str, sheet: str = None) -> pd.DataFrame:
    input_path = str(input_path)
    ext = Path(input_path).suffix.lower()

    if ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(input_path, sep=sep, encoding="utf-8", low_memory=False)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path, sheet_name=sheet if sheet else 0)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    df.columns = [normalize_colname(c) for c in df.columns]
    return df


def find_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        c_norm = normalize_colname(c)
        if c_norm in df.columns:
            return c_norm
    return None


def detect_group_columns(df: pd.DataFrame) -> dict:
    found = {}
    for key, candidates in GROUP_COL_CANDIDATES.items():
        found[key] = find_first_existing(df, candidates)
    return found


def detect_id_col(df: pd.DataFrame) -> str:
    return find_first_existing(df, ID_COL_CANDIDATES)


def find_system_metric_column(df: pd.DataFrame, system: str, metric: str) -> str:
    """
    Try to locate a wide-format column like:
      chatgpt_bleu
      google_translate_bleu
      microsoft_comet
    """
    system_aliases = SYSTEM_ALIASES.get(system, [system])
    metric = normalize_colname(metric)

    candidates = []
    for s in system_aliases:
        s = normalize_colname(s)
        candidates.extend([
            f"{s}_{metric}",
            f"{metric}_{s}"
        ])

    # exact match first
    for c in candidates:
        if c in df.columns:
            return c

    # fallback contains search
    for col in df.columns:
        for s in system_aliases:
            s_norm = normalize_colname(s)
            if s_norm in col and metric in col:
                return col

    return None


def available_system_metric_map(df: pd.DataFrame, systems, metrics) -> dict:
    """
    Returns:
      {
        system: {metric: colname or None}
      }
    """
    out = {}
    for system in systems:
        out[system] = {}
        for metric in metrics:
            out[system][metric] = find_system_metric_column(df, system, metric)
    return out


def wide_metric_to_long(df: pd.DataFrame, systems, metric_map: dict, group_cols: dict, id_col: str) -> pd.DataFrame:
    """
    Convert wide-format per-system metrics into long format:
    one row per (sample, system)
    """
    base_cols = []
    if id_col and id_col in df.columns:
        base_cols.append(id_col)

    for g in group_cols.values():
        if g and g in df.columns and g not in base_cols:
            base_cols.append(g)

    pieces = []
    for system in systems:
        cols = metric_map.get(system, {})
        valid_metric_cols = {m: c for m, c in cols.items() if c is not None}

        if not valid_metric_cols:
            continue

        sub = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)
        sub["system"] = system

        for metric_name, colname in valid_metric_cols.items():
            sub[metric_name] = pd.to_numeric(df[colname], errors="coerce")

        pieces.append(sub)

    if not pieces:
        return pd.DataFrame()

    long_df = pd.concat(pieces, ignore_index=True)
    return long_df


def infer_present_systems(df: pd.DataFrame, preferred_systems) -> list:
    present = []
    for system in preferred_systems:
        found_any = False
        for metric in AUTO_METRICS + HUMAN_METRICS:
            if find_system_metric_column(df, system, metric):
                found_any = True
                break
        if found_any:
            present.append(system)
    return present


def mean_std_series(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, 0
    return s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0, len(s)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_") if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def rank_direction_label(x):
    if pd.isna(x):
        return x
    val = str(x).strip().lower()
    mapping = {
        "en2fj": "EN→FJ",
        "en_to_fj": "EN→FJ",
        "english_to_fijian": "EN→FJ",
        "english_to_itaukei": "EN→FJ",
        "en_to_itaukei": "EN→FJ",
        "fj2en": "FJ→EN",
        "fj_to_en": "FJ→EN",
        "fijian_to_english": "FJ→EN",
        "itaukei_to_english": "FJ→EN",
        "itaukei_to_en": "FJ→EN",
    }
    return mapping.get(val, x)


# =========================
# Summaries
# =========================

def build_summary(df_long: pd.DataFrame, by_cols: list, metrics: list) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()

    keep_metrics = [m for m in metrics if m in df_long.columns]
    if not keep_metrics:
        return pd.DataFrame()

    agg = {}
    for m in keep_metrics:
        agg[m] = ["mean", "std", "count"]

    out = df_long.groupby(by_cols, dropna=False).agg(agg).reset_index()
    out = flatten_columns(out)
    return out


def build_human_summary(df_long: pd.DataFrame, by_cols: list) -> pd.DataFrame:
    present = [m for m in HUMAN_METRICS if m in df_long.columns]
    if not present:
        return pd.DataFrame()
    return build_summary(df_long, by_cols, present)


# =========================
# Correlations
# =========================

def compute_metric_human_correlations(df_long: pd.DataFrame) -> pd.DataFrame:
    auto_present = [m for m in AUTO_METRICS if m in df_long.columns]
    human_present = [m for m in HUMAN_METRICS if m in df_long.columns]

    rows = []
    for auto_m in auto_present:
        for human_m in human_present:
            sub = df_long[[auto_m, human_m]].dropna()
            if len(sub) < 3:
                rows.append({
                    "auto_metric": auto_m,
                    "human_metric": human_m,
                    "correlation": np.nan,
                    "p_value": np.nan,
                    "n": len(sub)
                })
                continue

            rho, p = spearmanr(sub[auto_m], sub[human_m], nan_policy="omit")
            rows.append({
                "auto_metric": auto_m,
                "human_metric": human_m,
                "correlation": rho,
                "p_value": p,
                "n": len(sub)
            })

    return pd.DataFrame(rows)


# =========================
# Statistical tests
# =========================

def compute_friedman_tests(df_long: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Friedman per metric across systems, using rows aligned by id.
    Requires at least 3 systems with matched observations.
    """
    rows = []
    if not id_col or id_col not in df_long.columns:
        return pd.DataFrame()

    systems = sorted(df_long["system"].dropna().unique().tolist())

    for metric in AUTO_METRICS + HUMAN_METRICS:
        if metric not in df_long.columns:
            continue

        piv = df_long.pivot_table(index=id_col, columns="system", values=metric, aggfunc="mean")
        piv = piv.dropna(axis=0, how="any")

        present_systems = [s for s in systems if s in piv.columns]
        if len(present_systems) < 3 or len(piv) < 3:
            rows.append({
                "metric": metric,
                "test": "friedman",
                "systems_compared": ", ".join(present_systems),
                "statistic": np.nan,
                "p_value": np.nan,
                "n_matched_rows": len(piv),
                "significant_0_05": np.nan
            })
            continue

        try:
            arrays = [piv[s].values for s in present_systems]
            stat, p = friedmanchisquare(*arrays)
        except Exception:
            stat, p = np.nan, np.nan

        rows.append({
            "metric": metric,
            "test": "friedman",
            "systems_compared": ", ".join(present_systems),
            "statistic": stat,
            "p_value": p,
            "n_matched_rows": len(piv),
            "significant_0_05": (p < 0.05) if pd.notna(p) else np.nan
        })

    return pd.DataFrame(rows)


def compute_pairwise_wilcoxon(df_long: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests per metric across systems.
    """
    rows = []
    if not id_col or id_col not in df_long.columns:
        return pd.DataFrame()

    systems = sorted(df_long["system"].dropna().unique().tolist())

    for metric in AUTO_METRICS + HUMAN_METRICS:
        if metric not in df_long.columns:
            continue

        piv = df_long.pivot_table(index=id_col, columns="system", values=metric, aggfunc="mean")

        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                s1, s2 = systems[i], systems[j]
                if s1 not in piv.columns or s2 not in piv.columns:
                    continue

                pair = piv[[s1, s2]].dropna()
                if len(pair) < 5:
                    rows.append({
                        "metric": metric,
                        "system_1": s1,
                        "system_2": s2,
                        "test": "wilcoxon",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "n_matched_rows": len(pair),
                        "median_diff": np.nan,
                        "significant_0_05": np.nan
                    })
                    continue

                diffs = pair[s1] - pair[s2]
                if np.allclose(diffs.values, 0, equal_nan=True):
                    stat, p = 0.0, 1.0
                else:
                    try:
                        stat, p = wilcoxon(pair[s1], pair[s2], zero_method="wilcox", correction=False)
                    except Exception:
                        stat, p = np.nan, np.nan

                rows.append({
                    "metric": metric,
                    "system_1": s1,
                    "system_2": s2,
                    "test": "wilcoxon",
                    "statistic": stat,
                    "p_value": p,
                    "n_matched_rows": len(pair),
                    "median_diff": np.median(diffs),
                    "significant_0_05": (p < 0.05) if pd.notna(p) else np.nan
                })

    out = pd.DataFrame(rows)

    # Bonferroni correction within each metric
    if not out.empty:
        out["p_value_bonferroni"] = np.nan
        for metric, idx in out.groupby("metric").groups.items():
            pvals = out.loc[idx, "p_value"]
            valid = pvals.notna()
            m = valid.sum()
            if m > 0:
                corrected = (pvals[valid] * m).clip(upper=1.0)
                out.loc[pvals[valid].index, "p_value_bonferroni"] = corrected
        out["significant_bonferroni_0_05"] = out["p_value_bonferroni"] < 0.05

    return out


# =========================
# Plotting
# =========================

def save_bar_overall(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return

    metrics = [m for m in AUTO_METRICS + HUMAN_METRICS if f"{m}_mean" in summary_df.columns]
    if not metrics:
        return

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plot_df = summary_df[["system", f"{metric}_mean"]].dropna().sort_values(f"{metric}_mean", ascending=(metric == "ter"))
        plt.bar(plot_df["system"], plot_df[f"{metric}_mean"])
        plt.xlabel("System")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Overall {metric.replace('_', ' ').title()} by System")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / f"overall_{metric}_bar.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_direction_bars(direction_df: pd.DataFrame, out_dir: Path, direction_col: str) -> None:
    if direction_df.empty or not direction_col or direction_col not in direction_df.columns:
        return

    metrics = [m for m in AUTO_METRICS + HUMAN_METRICS if f"{m}_mean" in direction_df.columns]
    if not metrics:
        return

    for metric in metrics:
        piv = direction_df[[direction_col, "system", f"{metric}_mean"]].dropna()
        if piv.empty:
            continue

        pivoted = piv.pivot(index=direction_col, columns="system", values=f"{metric}_mean")
        pivoted.index = [rank_direction_label(x) for x in pivoted.index]

        plt.figure(figsize=(10, 6))
        pivoted.plot(kind="bar", ax=plt.gca())
        plt.xlabel("Direction")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} by Direction and System")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_dir / f"direction_{metric}_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_domain_heatmaps(domain_df: pd.DataFrame, out_dir: Path, domain_col: str) -> None:
    if domain_df.empty or not domain_col or domain_col not in domain_df.columns:
        return

    metrics = [m for m in AUTO_METRICS + HUMAN_METRICS if f"{m}_mean" in domain_df.columns]
    if not metrics:
        return

    for metric in metrics:
        piv = domain_df[[domain_col, "system", f"{metric}_mean"]].dropna()
        if piv.empty:
            continue

        heat = piv.pivot(index=domain_col, columns="system", values=f"{metric}_mean")

        plt.figure(figsize=(10, max(5, len(heat) * 0.5)))
        plt.imshow(heat.values, aspect="auto")
        plt.colorbar(label=metric.replace("_", " ").title())
        plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
        plt.yticks(range(len(heat.index)), heat.index)
        plt.title(f"Domain Heatmap: {metric.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(out_dir / f"domain_{metric}_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_sentence_type_lines(sent_df: pd.DataFrame, out_dir: Path, sent_col: str) -> None:
    if sent_df.empty or not sent_col or sent_col not in sent_df.columns:
        return

    metrics = [m for m in AUTO_METRICS + HUMAN_METRICS if f"{m}_mean" in sent_df.columns]
    if not metrics:
        return

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        plot_df = sent_df[[sent_col, "system", f"{metric}_mean"]].dropna()
        if plot_df.empty:
            plt.close()
            continue

        for system in sorted(plot_df["system"].dropna().unique()):
            sub = plot_df[plot_df["system"] == system]
            plt.plot(sub[sent_col].astype(str), sub[f"{metric}_mean"], marker="o", label=system)

        plt.xlabel("Sentence Type")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} by Sentence Type")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"sentence_type_{metric}_line.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_human_boxplots(df_long: pd.DataFrame, out_dir: Path) -> None:
    human_present = [m for m in HUMAN_METRICS if m in df_long.columns]
    if not human_present or df_long.empty:
        return

    for metric in human_present:
        plot_df = df_long[["system", metric]].dropna()
        if plot_df.empty:
            continue

        systems = sorted(plot_df["system"].unique())
        data = [plot_df.loc[plot_df["system"] == s, metric].dropna().values for s in systems]

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=systems)
        plt.xlabel("System")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} Distribution by System")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / f"human_{metric}_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_correlation_heatmap(corr_df: pd.DataFrame, out_dir: Path) -> None:
    if corr_df.empty:
        return

    piv = corr_df.pivot(index="auto_metric", columns="human_metric", values="correlation")
    if piv.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.imshow(piv.values, aspect="auto")
    plt.colorbar(label="Spearman Correlation")
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right")
    plt.yticks(range(len(piv.index)), piv.index)
    plt.title("Automatic vs Human Metric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "metric_human_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Main processing
# =========================

def main():
    parser = argparse.ArgumentParser(description="SO2 Part 6 Reporting Script")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX file")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--systems",
        default="chatgpt,gemini,google,microsoft",
        help="Comma-separated system names to look for"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"

    safe_mkdir(output_dir)
    safe_mkdir(figures_dir)

    log(f"Loading data from: {input_path}")
    df = load_table(str(input_path), sheet=args.sheet)
    log(f"Rows loaded: {len(df):,}")
    log(f"Columns found: {len(df.columns)}")

    requested_systems = [normalize_colname(x) for x in args.systems.split(",") if x.strip()]
    systems = infer_present_systems(df, requested_systems)
    if not systems:
        systems = requested_systems

    log(f"Systems detected/requested: {systems}")

    group_cols = detect_group_columns(df)
    id_col = detect_id_col(df)

    log(f"Detected ID column: {id_col}")
    log(f"Detected grouping columns: {group_cols}")

    metric_map = available_system_metric_map(df, systems, AUTO_METRICS + HUMAN_METRICS)

    # Build long-format table
    df_long = wide_metric_to_long(df, systems, metric_map, group_cols, id_col)

    if df_long.empty:
        raise RuntimeError(
            "Could not construct long-format table. "
            "Please check that your dataset has columns such as chatgpt_bleu, google_comet, microsoft_fluency, etc."
        )

    # Normalize direction labels
    if group_cols.get("direction") and group_cols["direction"] in df_long.columns:
        df_long[group_cols["direction"]] = df_long[group_cols["direction"]].apply(rank_direction_label)

    save_csv(df_long, output_dir / "so2_long_format_reporting_table.csv")
    log("Saved: so2_long_format_reporting_table.csv")

    # Overall summary
    all_metrics_present = [m for m in AUTO_METRICS + HUMAN_METRICS if m in df_long.columns]
    overall_summary = build_summary(df_long, ["system"], all_metrics_present)
    save_csv(overall_summary, output_dir / "so2_overall_summary.csv")
    log("Saved: so2_overall_summary.csv")

    # By direction
    direction_col = group_cols.get("direction")
    if direction_col and direction_col in df_long.columns:
        by_direction = build_summary(df_long, [direction_col, "system"], all_metrics_present)
    else:
        by_direction = pd.DataFrame()
    save_csv(by_direction, output_dir / "so2_by_direction.csv")
    log("Saved: so2_by_direction.csv")

    # By domain
    domain_col = group_cols.get("domain")
    if domain_col and domain_col in df_long.columns:
        by_domain = build_summary(df_long, [domain_col, "system"], all_metrics_present)
    else:
        by_domain = pd.DataFrame()
    save_csv(by_domain, output_dir / "so2_by_domain.csv")
    log("Saved: so2_by_domain.csv")

    # By sentence type
    sentence_type_col = group_cols.get("sentence_type")
    if sentence_type_col and sentence_type_col in df_long.columns:
        by_sentence_type = build_summary(df_long, [sentence_type_col, "system"], all_metrics_present)
    else:
        by_sentence_type = pd.DataFrame()
    save_csv(by_sentence_type, output_dir / "so2_by_sentence_type.csv")
    log("Saved: so2_by_sentence_type.csv")

    # Human-only summary
    human_summary = build_human_summary(df_long, ["system"])
    save_csv(human_summary, output_dir / "so2_human_eval_summary.csv")
    log("Saved: so2_human_eval_summary.csv")

    # Correlations
    corr_df = compute_metric_human_correlations(df_long)
    save_csv(corr_df, output_dir / "so2_metric_human_correlation.csv")
    log("Saved: so2_metric_human_correlation.csv")

    # Statistics
    friedman_df = compute_friedman_tests(df_long, id_col)
    save_csv(friedman_df, output_dir / "so2_friedman_tests.csv")
    log("Saved: so2_friedman_tests.csv")

    wilcoxon_df = compute_pairwise_wilcoxon(df_long, id_col)
    save_csv(wilcoxon_df, output_dir / "so2_pairwise_wilcoxon.csv")
    log("Saved: so2_pairwise_wilcoxon.csv")

    # Plots
    save_bar_overall(overall_summary, figures_dir)
    log("Saved overall bar charts")

    save_direction_bars(by_direction, figures_dir, direction_col)
    log("Saved direction charts")

    save_domain_heatmaps(by_domain, figures_dir, domain_col)
    log("Saved domain heatmaps")

    save_sentence_type_lines(by_sentence_type, figures_dir, sentence_type_col)
    log("Saved sentence-type charts")

    save_human_boxplots(df_long, figures_dir)
    log("Saved human boxplots")

    save_correlation_heatmap(corr_df, figures_dir)
    log("Saved correlation heatmap")

    # Simple text report
    report_lines = []
    report_lines.append("SO2 Part 6 Reporting Summary")
    report_lines.append("=" * 40)
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Rows loaded: {len(df):,}")
    report_lines.append(f"Systems analyzed: {', '.join(sorted(df_long['system'].dropna().unique()))}")
    report_lines.append(f"ID column: {id_col}")
    report_lines.append(f"Direction column: {direction_col}")
    report_lines.append(f"Domain column: {domain_col}")
    report_lines.append(f"Sentence type column: {sentence_type_col}")
    report_lines.append("")

    if not overall_summary.empty:
        report_lines.append("Overall means by system:")
        cols = ["system"] + [c for c in overall_summary.columns if c.endswith("_mean")]
        report_lines.append(overall_summary[cols].to_string(index=False))
        report_lines.append("")

    if not friedman_df.empty:
        report_lines.append("Friedman tests:")
        report_lines.append(friedman_df.to_string(index=False))
        report_lines.append("")

    report_path = output_dir / "so2_reporting_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log(f"Saved: {report_path}")
    log("Done.")


if __name__ == "__main__":
    main()