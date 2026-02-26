import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer


DEFAULT_SYSTEMS = ["chatgpt", "gemini", "google_translate", "microsoft_translate"]


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    # Keep it simple + language-agnostic (works for iTaukei too)
    s = " ".join(s.split())
    return s


def ensure_rouge_cols(df: pd.DataFrame, systems: list[str]) -> pd.DataFrame:
    variants = ["rouge1", "rouge2", "rougeL"]
    stats = ["p", "r", "f1"]
    for v in variants:
        for st in stats:
            for s in systems:
                col = f"{v}_{st}__{s}"
                if col not in df.columns:
                    df[col] = np.nan
    return df


def compute_rouge_pair(scorer: rouge_scorer.RougeScorer, ref: str, hyp: str):
    # returns dict like {'rouge1': Score(p, r, fmeasure), ...}
    return scorer.score(target=ref, prediction=hyp)


def fill_rouge(df: pd.DataFrame, systems: list[str], ref_col: str, overwrite: bool) -> pd.DataFrame:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    for syscol in systems:
        # Determine if we need to fill anything
        probe_col = f"rouge1_f1__{syscol}"
        if probe_col not in df.columns:
            df[probe_col] = np.nan

        if overwrite:
            mask = df[syscol].notna() & df[ref_col].notna()
        else:
            mask = df[probe_col].isna() & df[syscol].notna() & df[ref_col].notna()

        idxs = df.index[mask].tolist()
        print(f"[INFO] ROUGE for {syscol}: filling {len(idxs)} rows ({'overwrite' if overwrite else 'missing-only'})")
        if not idxs:
            continue

        for i in tqdm(idxs, desc=f"ROUGE {syscol}", unit="row"):
            ref = norm_text(df.at[i, ref_col])
            hyp = norm_text(df.at[i, syscol])

            if not ref or not hyp:
                continue

            scores = compute_rouge_pair(scorer, ref, hyp)

            for v in ["rouge1", "rouge2", "rougeL"]:
                df.at[i, f"{v}_p__{syscol}"] = float(scores[v].precision)
                df.at[i, f"{v}_r__{syscol}"] = float(scores[v].recall)
                df.at[i, f"{v}_f1__{syscol}"] = float(scores[v].fmeasure)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV")
    ap.add_argument("--ref_col", default="ref", help="Reference column name (default: ref)")
    ap.add_argument("--systems", default="chatgpt,gemini,google_translate,microsoft_translate",
                    help="Comma-separated system columns")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if ROUGE already present")
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    df = pd.read_csv(args.inp)
    print(f"[INFO] Loaded {len(df)} rows from {os.path.basename(args.inp)}")

    df = ensure_rouge_cols(df, systems)
    df = fill_rouge(df, systems=systems, ref_col=args.ref_col, overwrite=args.overwrite)

    df.to_csv(args.out, index=False)
    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()
