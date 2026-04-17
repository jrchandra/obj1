#!/usr/bin/env python3
"""
Step 5: Performance Quantification
- Computes per-system averages for automatic metrics (BLEU/chrF/TER/COMET/etc. if present)
- Computes per-system mean human scores (adequacy/fluency/cohesion) by mapping blind human rows
  back to system outputs in the metrics file.

Inputs:
  --metrics_csv  dataset_with_system_outputs__metrics_filled_all_with_bluert_and_rouge_plus_llm_judge.csv
  --human_csv    human_eval_BLIND_SYNTHETIC_FILLED.csv

Outputs (in --out_dir):
  - auto_metrics_summary_overall.csv
  - auto_metrics_summary_by_domain.csv
  - auto_metrics_summary_by_direction.csv
  - auto_metrics_summary_by_domain_direction.csv
  - human_mapped_rows.csv
  - human_summary_overall.csv
  - human_summary_by_domain.csv
  - human_summary_by_direction.csv
  - human_summary_by_domain_direction.csv
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ID_COLS = ["domain", "direction", "sentence_type_fine", "source_id", "source_doc", "source_text"]


def norm_text(x: str) -> str:
    """Normalize text for robust matching."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    x = str(x)
    x = x.strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def discover_systems_and_metrics(metrics_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Detect systems and metric prefixes from columns like:
      bleu__chatgpt, chrf__gemini, comet__google_translate, etc.
    Returns (systems, metrics)
    """
    systems = set()
    metrics = set()
    pat = re.compile(r"^(.+?)__(.+)$")
    for c in metrics_df.columns:
        m = pat.match(c)
        if not m:
            continue
        metric, system = m.group(1), m.group(2)
        # Heuristic: metric cols are numeric; system cols exist as raw outputs too.
        metrics.add(metric)
        systems.add(system)

    # Only keep "systems" that also appear as raw output columns if possible
    raw_output_systems = [s for s in systems if s in metrics_df.columns]
    if raw_output_systems:
        systems = set(raw_output_systems)

    return sorted(systems), sorted(metrics)


def summarize_auto_metrics(metrics_df: pd.DataFrame, systems: List[str], metrics: List[str],
                          group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Summarize automatic metrics. Produces mean/std/count for each (group, system, metric).
    """
    rows = []
    group_cols = group_cols or []

    for system in systems:
        for metric in metrics:
            col = f"{metric}__{system}"
            if col not in metrics_df.columns:
                continue
            sub = metrics_df[group_cols + [col]].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            if group_cols:
                g = sub.groupby(group_cols, dropna=False)[col]
                agg = g.agg(["count", "mean", "std"]).reset_index()
                for _, r in agg.iterrows():
                    rows.append({
                        **{gc: r[gc] for gc in group_cols},
                        "system": system,
                        "metric": metric,
                        "n": int(r["count"]),
                        "mean": float(r["mean"]) if pd.notna(r["mean"]) else np.nan,
                        "std": float(r["std"]) if pd.notna(r["std"]) else np.nan,
                    })
            else:
                rows.append({
                    "system": system,
                    "metric": metric,
                    "n": int(sub[col].count()),
                    "mean": float(sub[col].mean()) if sub[col].count() else np.nan,
                    "std": float(sub[col].std()) if sub[col].count() else np.nan,
                })

    out = pd.DataFrame(rows)
    # nice ordering
    if group_cols:
        return out.sort_values(group_cols + ["metric", "system"]).reset_index(drop=True)
    return out.sort_values(["metric", "system"]).reset_index(drop=True)


def map_blind_human_to_system(metrics_df: pd.DataFrame, human_df: pd.DataFrame,
                             systems: List[str]) -> pd.DataFrame:
    """
    Map human_df rows (blind) to a system by comparing system_output against each system column
    in metrics_df for the same (domain, direction, sentence_type_fine, source_id).

    Strategy:
      1) Exact match (normalized)
      2) If no match, try "contains" match (rare)
      3) Else mark as UNMAPPED

    Returns human_df with added columns:
      - system (mapped)
      - match_type (EXACT / CONTAINS / UNMAPPED / AMBIGUOUS)
    """
    # Build lookup: key -> dict(system -> normalized output)
    key_cols = ["domain", "direction", "sentence_type_fine", "source_id"]
    needed_cols = set(key_cols + systems)
    missing = [c for c in needed_cols if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Metrics CSV missing required columns for mapping: {missing}")

    msub = metrics_df[key_cols + systems].copy()

    # Normalize all system outputs
    for s in systems:
        msub[s] = msub[s].map(norm_text)

    # Create dict: key -> {system: text}
    lookup: Dict[Tuple, Dict[str, str]] = {}
    for row in msub.itertuples(index=False):
        key = tuple(getattr(row, c) for c in key_cols)
        lookup[key] = {s: getattr(row, s) for s in systems}

    out = human_df.copy()
    out["system_output_norm"] = out["system_output"].map(norm_text)

    mapped_systems = []
    match_types = []

    for row in out.itertuples(index=False):
        key = tuple(getattr(row, c) for c in key_cols)
        sys_out = getattr(row, "system_output_norm")
        candidates = lookup.get(key)

        if not candidates:
            mapped_systems.append("UNMAPPED")
            match_types.append("NO_KEY")
            continue

        # EXACT matches
        exact = [s for s, txt in candidates.items() if txt and txt == sys_out]
        if len(exact) == 1:
            mapped_systems.append(exact[0])
            match_types.append("EXACT")
            continue
        if len(exact) > 1:
            mapped_systems.append("AMBIGUOUS")
            match_types.append("AMBIGUOUS_EXACT")
            continue

        # CONTAINS (fallback)
        contains = [s for s, txt in candidates.items() if txt and (sys_out in txt or txt in sys_out)]
        if len(contains) == 1:
            mapped_systems.append(contains[0])
            match_types.append("CONTAINS")
            continue
        if len(contains) > 1:
            mapped_systems.append("AMBIGUOUS")
            match_types.append("AMBIGUOUS_CONTAINS")
            continue

        mapped_systems.append("UNMAPPED")
        match_types.append("UNMAPPED")

    out["system"] = mapped_systems
    out["match_type"] = match_types

    return out


def summarize_human(human_mapped_df: pd.DataFrame, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Summarize human scores per system, optionally grouped by domain/direction.
    """
    group_cols = group_cols or []
    df = human_mapped_df.copy()

    # Keep only mapped rows
    df = df[df["system"].isin(["UNMAPPED", "AMBIGUOUS"]) == False].copy()

    # Ensure numeric
    for c in ["adequacy_1to5", "fluency_1to5", "cohesion_discourse_optional"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_cols = [c for c in ["adequacy_1to5", "fluency_1to5", "cohesion_discourse_optional"] if c in df.columns]

    gcols = group_cols + ["system"]
    if not gcols:
        gcols = ["system"]

    g = df.groupby(gcols, dropna=False)

    rows = []
    for name, sub in g:
        if isinstance(name, tuple):
            key_vals = dict(zip(gcols, name))
        else:
            key_vals = {gcols[0]: name}

        row = {**key_vals, "n": int(len(sub))}
        for c in agg_cols:
            row[f"{c}_mean"] = float(sub[c].mean()) if sub[c].count() else np.nan
            row[f"{c}_std"] = float(sub[c].std()) if sub[c].count() else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = group_cols + ["system"] if group_cols else ["system"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--human_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    metrics_df = pd.read_csv(args.metrics_csv)
    human_df = pd.read_csv(args.human_csv)

    systems, metrics = discover_systems_and_metrics(metrics_df)
    if not systems:
        raise ValueError("Could not detect systems. Expected columns like bleu__chatgpt AND a raw output column chatgpt.")
    if not metrics:
        raise ValueError("Could not detect metric columns (expected pattern metric__system).")

    # --- AUTO METRICS SUMMARIES ---
    auto_overall = summarize_auto_metrics(metrics_df, systems, metrics, group_cols=None)
    auto_by_domain = summarize_auto_metrics(metrics_df, systems, metrics, group_cols=["domain"])
    auto_by_direction = summarize_auto_metrics(metrics_df, systems, metrics, group_cols=["direction"])
    auto_by_dom_dir = summarize_auto_metrics(metrics_df, systems, metrics, group_cols=["domain", "direction"])

    auto_overall.to_csv(os.path.join(args.out_dir, "auto_metrics_summary_overall.csv"), index=False)
    auto_by_domain.to_csv(os.path.join(args.out_dir, "auto_metrics_summary_by_domain.csv"), index=False)
    auto_by_direction.to_csv(os.path.join(args.out_dir, "auto_metrics_summary_by_direction.csv"), index=False)
    auto_by_dom_dir.to_csv(os.path.join(args.out_dir, "auto_metrics_summary_by_domain_direction.csv"), index=False)

    # --- HUMAN MAPPING + SUMMARIES ---
    human_mapped = map_blind_human_to_system(metrics_df, human_df, systems)
    human_mapped.to_csv(os.path.join(args.out_dir, "human_mapped_rows.csv"), index=False)

    # Quality check stats
    map_stats = human_mapped["match_type"].value_counts(dropna=False)
    map_stats.to_csv(os.path.join(args.out_dir, "human_mapping_matchtype_counts.csv"))

    human_overall = summarize_human(human_mapped, group_cols=None)
    human_by_domain = summarize_human(human_mapped, group_cols=["domain"])
    human_by_direction = summarize_human(human_mapped, group_cols=["direction"])
    human_by_dom_dir = summarize_human(human_mapped, group_cols=["domain", "direction"])

    human_overall.to_csv(os.path.join(args.out_dir, "human_summary_overall.csv"), index=False)
    human_by_domain.to_csv(os.path.join(args.out_dir, "human_summary_by_domain.csv"), index=False)
    human_by_direction.to_csv(os.path.join(args.out_dir, "human_summary_by_direction.csv"), index=False)
    human_by_dom_dir.to_csv(os.path.join(args.out_dir, "human_summary_by_domain_direction.csv"), index=False)

    print("DONE.")
    print(f"Detected systems: {systems}")
    print(f"Detected metrics: {metrics[:15]}{'...' if len(metrics) > 15 else ''}")
    print("Wrote outputs to:", args.out_dir)
    print("Human mapping match_type counts:\n", map_stats.to_string())


if __name__ == "__main__":
    main()
