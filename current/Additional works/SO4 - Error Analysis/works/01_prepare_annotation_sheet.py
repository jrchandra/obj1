import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pick_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}. Found: {list(df.columns)}")
    return None


def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8", engine="python")
    if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError("Unsupported input. Use .csv or .xlsx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="SO2 dataset .csv or .xlsx")
    ap.add_argument("--taxonomy", required=True, help="taxonomy/error_taxonomy.json")
    ap.add_argument("--outfile", required=True, help="Output .xlsx annotation sheet")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for blinding/shuffle")
    ap.add_argument("--sample_n", type=int, default=0, help="If >0, sample N segments before expanding systems")
    ap.add_argument("--id_col", default="", help="Optional: name of ID column if your dataset already has one")
    args = ap.parse_args()

    random.seed(args.seed)

    df = load_table(args.infile)
    df.columns = [c.strip() for c in df.columns]

    # Try to auto-detect typical SO2 columns (adjust candidates to your dataset headers)
    col_id = args.id_col.strip() or _pick_column(df, ["id", "seg_id", "segment_id", "row_id"], required=False)
    col_domain = _pick_column(df, ["domain", "subdomain"], required=False)
    col_sent_type = _pick_column(df, ["sentence_type", "sent_type"], required=False)
    col_direction = _pick_column(df, ["direction", "lang_dir", "translation_direction"], required=False)

    col_source = _pick_column(df, ["source", "source_text", "src", "english", "en"], required=True)
    col_ref = _pick_column(df, ["reference", "human_ref", "ref", "target_ref", "gold"], required=True)

    # Detect system output columns (common patterns)
    # You can also explicitly rename your dataset columns to match this pattern if needed.
    sys_cols = []
    for c in df.columns:
        cl = c.lower()
        if cl in {col_source.lower(), col_ref.lower()}:
            continue
        if col_id and cl == col_id.lower():
            continue
        if col_domain and cl == col_domain.lower():
            continue
        if col_sent_type and cl == col_sent_type.lower():
            continue
        if col_direction and cl == col_direction.lower():
            continue

        # heuristics: columns that look like model/system outputs
        if any(k in cl for k in ["gpt", "llm", "nmt", "google", "microsoft", "gemini", "model", "system", "output", "hyp"]):
            sys_cols.append(c)

    if not sys_cols:
        raise ValueError(
            "Could not auto-detect system output columns. "
            "Rename system output columns to include keywords like 'gpt', 'google', 'nmt', 'output', 'hyp', etc."
        )

    if args.sample_n and args.sample_n > 0:
        df = df.sample(n=min(args.sample_n, len(df)), random_state=args.seed).reset_index(drop=True)

    # Build blinded mapping
    sys_cols_sorted = sorted(sys_cols)
    masked_ids = [f"S{idx+1:02d}" for idx in range(len(sys_cols_sorted))]
    random.shuffle(masked_ids)

    mapping = dict(zip(sys_cols_sorted, masked_ids))

    with open(args.taxonomy, "r", encoding="utf-8") as f:
        tax = json.load(f)
    err_types = tax["error_types"]
    sev_levels = tax["severity_levels"]

    # Expand to long format: one row per (segment, system)
    rows = []
    for i, r in df.iterrows():
        seg_id = r[col_id] if col_id else f"SEG{i+1:06d}"
        base = {
            "seg_id": seg_id,
            "domain": r[col_domain] if col_domain else "",
            "sentence_type": r[col_sent_type] if col_sent_type else "",
            "direction": r[col_direction] if col_direction else "",
            "source": _norm(r[col_source]),
            "reference": _norm(r[col_ref]),
        }
        for syscol in sys_cols_sorted:
            out = _norm(r[syscol])
            rows.append({
                **base,
                "system_masked": mapping[syscol],     # what annotators see
                "system_true": syscol,                 # kept for later (remove for annotators if desired)
                "hypothesis": out,

                # Annotation fields
                "error_present": "",                   # Y/N
                "error_types": "",                     # multi-label: e.g., LEXICAL;CULTURAL
                "severity": "",                        # MINOR/MAJOR/CRITICAL
                "rationale": "",                       # free text
            })

    ann = pd.DataFrame(rows)

    # Shuffle to reduce bias and system clustering
    ann = ann.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Add data validations hint rows (not strict Excel validation, but guidance)
    guidance = pd.DataFrame([{
        "seg_id": "GUIDE",
        "domain": "",
        "sentence_type": "",
        "direction": "",
        "source": "Fill error_present as Y/N. If Y, choose one or more error_types separated by ';' and severity.",
        "reference": f"Allowed error_types: {', '.join(err_types)}",
        "system_masked": "",
        "system_true": "",
        "hypothesis": f"Allowed severity: {', '.join(sev_levels)}",
        "error_present": "",
        "error_types": "",
        "severity": "",
        "rationale": ""
    }])

    ann_out = pd.concat([guidance, ann], ignore_index=True)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    ann_out.to_excel(args.outfile, index=False)

    # Also export the mapping so you can unblind later (KEEP PRIVATE)
    map_path = os.path.splitext(args.outfile)[0] + "__SYSTEM_MAPPING.csv"
    pd.DataFrame([{"system_true": k, "system_masked": v} for k, v in mapping.items()]).to_csv(map_path, index=False)

    print(f"[OK] Wrote annotation sheet: {args.outfile}")
    print(f"[OK] Wrote private system mapping: {map_path}")
    print(f"[INFO] Detected system columns: {sys_cols_sorted}")


if __name__ == "__main__":
    main()
