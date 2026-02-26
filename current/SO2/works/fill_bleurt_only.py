#!/usr/bin/env python3
"""
Fill ONLY BLEURT columns in an existing dataset CSV.

Expected columns:
- ref (reference translation)
- chatgpt, gemini, google_translate, microsoft_translate (system outputs)
- bleurt__chatgpt, bleurt__gemini, bleurt__google_translate, bleurt__microsoft_translate (will be filled)

Usage:
  python fill_bleurt_only.py ^
    --in_csv dataset_with_system_outputs__with_metrics_columns.csv ^
    --out_csv dataset_with_system_outputs__metrics_filled.csv ^
    --ref_col ref ^
    --model lucadiliello/BLEURT-20-D12 ^
    --batch_size 16
"""

import argparse
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ref_col", default="ref")
    ap.add_argument("--systems", default="chatgpt,gemini,google_translate,microsoft_translate")
    ap.add_argument("--model", default="lucadiliello/BLEURT-20-D12",
                    help="BLEURT checkpoint on Hugging Face (PyTorch port).")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--overwrite", action="store_true",
                    help="If set, recompute even if bleurt__* already has a value.")
    args = ap.parse_args()

    try:
        import torch
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
    except Exception as e:
        print("[FATAL] Missing deps for BLEURT-PyTorch.")
        print("Install with:")
        print("  pip install -U torch transformers tqdm pandas numpy")
        print("  pip install -U bleurt-pytorch")
        print("or:")
        print("  pip install -U git+https://github.com/lucadiliello/bleurt-pytorch.git")
        raise

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    df = pd.read_csv(args.in_csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.in_csv}")

    # Ensure BLEURT columns exist
    for sysname in systems:
        col = f"bleurt__{sysname}"
        if col not in df.columns:
            df[col] = np.nan

    # Load model/tokenizer
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA requested but not available. Falling back to CPU.")

    print(f"[INFO] Loading BLEURT model: {args.model} on {device.type}")
    tokenizer = BleurtTokenizer.from_pretrained(args.model)
    model = BleurtForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    def to_str(x):
        if pd.isna(x):
            return ""
        return str(x)

    @torch.no_grad()
    def score_batches(refs, hyps):
        """Return BLEURT scores (float) for aligned refs/hyps."""
        scores = []
        for i in range(0, len(refs), args.batch_size):
            r_batch = refs[i:i+args.batch_size]
            h_batch = hyps[i:i+args.batch_size]
            enc = tokenizer(
                r_batch,
                h_batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            # For BLEURT regression, logits is shape [B, 1] or [B]
            logits = out.logits.squeeze(-1).detach().cpu().numpy()
            scores.extend([float(x) for x in logits])
        return scores

    ref_col = args.ref_col
    if ref_col not in df.columns:
        raise ValueError(f"[FATAL] ref_col '{ref_col}' not found in CSV.")

    for sysname in systems:
        if sysname not in df.columns:
            print(f"[WARN] System column '{sysname}' not found. Skipping.")
            continue

        outcol = f"bleurt__{sysname}"

        # Decide which rows need filling
        if args.overwrite:
            mask = df[sysname].notna() & df[ref_col].notna()
        else:
            mask = df[outcol].isna() & df[sysname].notna() & df[ref_col].notna()

        idxs = df.index[mask].tolist()
        if not idxs:
            print(f"[INFO] {outcol}: nothing to fill.")
            continue

        refs = [to_str(df.at[i, ref_col]) for i in idxs]
        hyps = [to_str(df.at[i, sysname]) for i in idxs]

        print(f"[INFO] Filling {outcol}: {len(idxs)} pairs...")
        # Optional progress: chunk scoring with visible tqdm
        all_scores = []
        for start in tqdm(range(0, len(refs), args.batch_size), desc=f"BLEURT for {sysname}"):
            r_batch = refs[start:start+args.batch_size]
            h_batch = hyps[start:start+args.batch_size]
            all_scores.extend(score_batches(r_batch, h_batch))

        # Write back
        for i, sc in zip(idxs, all_scores):
            df.at[i, outcol] = sc

        print(f"[OK] {outcol} filled.")

    df.to_csv(args.out_csv, index=False)
    print(f"[DONE] Saved -> {args.out_csv}")

if __name__ == "__main__":
    main()
