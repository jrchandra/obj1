#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fill missing MT evaluation metrics columns:
- TER (0..1; lower is better)   -> ter__<system>
- BERTScore F1 (0..1; higher)  -> bertscore_f1__<system>
- COMET (higher)               -> comet__<system>
- BLEURT (higher)              -> bleurt__<system>

Input expects columns:
  - ref (reference translation)
  - chatgpt, gemini, google_translate, microsoft_translate (system outputs)
and metric columns already present (will be overwritten if --overwrite).

Fixes BERTScore baseline error by defaulting rescale_with_baseline=False.
"""

import argparse
import os
import sys
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

SYSTEMS_DEFAULT = ["chatgpt", "gemini", "google_translate", "microsoft_translate"]
REF_COL_DEFAULT = "ref"


def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


# -----------------------
# TER (needs sacrebleu)
# -----------------------
def fill_ter(df, systems, ref_col, overwrite=False):
    try:
        import sacrebleu
    except ImportError:
        print("[SKIP] sacrebleu not installed, skipping TER. Install: pip install sacrebleu")
        return df

    print("[INFO] Filling TER with sacrebleu (TER/1.0). Stored as 0..1 (lower is better).")
    for syscol in systems:
        out_col = f"ter__{syscol}"
        if out_col not in df.columns:
            df[out_col] = np.nan

        refs = [norm_str(r) for r in df[ref_col].tolist()]
        hyps = [norm_str(h) for h in df[syscol].tolist()]

        vals = []
        for r, h in tqdm(list(zip(refs, hyps)), desc=f"TER for {syscol}", total=len(refs)):
            if (not overwrite) and (not pd.isna(df.loc[len(vals), out_col])):
                # shouldn't happen reliably row-wise; keep simple overwrite logic below
                pass
            if not r or not h:
                vals.append(np.nan)
                continue
            try:
                ter = sacrebleu.metrics.TER().sentence_score(h, [r]).score
                # sacrebleu returns 0..100; convert to 0..1
                vals.append(float(ter) / 100.0)
            except Exception:
                vals.append(np.nan)

        df[out_col] = vals

    print("[OK] TER filled.")
    return df


# -----------------------
# BERTScore (needs bert-score + torch)
# -----------------------
def fill_bertscore(df, systems, ref_col, overwrite=False,
                   model_type="xlm-roberta-large",
                   device=None,
                   batch_size=16,
                   rescale_with_baseline=False,
                   lang=None):
    """
    Key fix for your error:
      - if rescale_with_baseline=True -> must provide lang.
    For mixed EN/FJ, rescaling is not recommended -> default False.
    """
    try:
        from bert_score import score as bertscore_score
    except ImportError:
        print("[SKIP] bert-score not installed, skipping BERTScore. Install: pip install bert-score")
        return df

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if rescale_with_baseline and not lang:
        raise ValueError("BERTScore: rescale_with_baseline=True requires lang (e.g., lang='en'). "
                         "For mixed EN/FJ, use rescale_with_baseline=False (recommended).")

    print(f"[INFO] Filling BERTScore F1 using {model_type} on {device} "
          f"(rescale_with_baseline={rescale_with_baseline}, lang={lang}).")

    for syscol in systems:
        out_col = f"bertscore_f1__{syscol}"
        if out_col not in df.columns:
            df[out_col] = np.nan

        # Only compute rows that need it (unless overwrite)
        refs = df[ref_col].map(norm_str).tolist()
        hyps = df[syscol].map(norm_str).tolist()

        need_idx = []
        for i in range(len(df)):
            if not refs[i] or not hyps[i]:
                continue
            if overwrite or pd.isna(df.at[i, out_col]):
                need_idx.append(i)

        if not need_idx:
            print(f"[OK] BERTScore F1 already filled for {syscol}.")
            continue

        r_need = [refs[i] for i in need_idx]
        h_need = [hyps[i] for i in need_idx]

        # Compute in batches
        f1_all = []
        for start in tqdm(range(0, len(need_idx), batch_size),
                          desc=f"BERTScore F1 for {syscol}", total=math.ceil(len(need_idx) / batch_size)):
            r_batch = r_need[start:start+batch_size]
            h_batch = h_need[start:start+batch_size]

            P, R, F1 = bertscore_score(
                cands=h_batch,
                refs=r_batch,
                model_type=model_type,
                device=device,
                batch_size=batch_size,
                verbose=False,
                rescale_with_baseline=rescale_with_baseline,
                lang=lang
            )
            f1_all.extend([float(x) for x in F1])

        # Write back
        for j, i in enumerate(need_idx):
            df.at[i, out_col] = f1_all[j]

        print(f"[OK] BERTScore F1 filled for {syscol}.")
    return df


# -----------------------
# COMET (needs comet)
# -----------------------
def fill_comet(df, systems, ref_col, overwrite=False, model_name="Unbabel/wmt22-comet-da"):
    """
    Requires: pip install unbabel-comet
    Downloads model on first run.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("[SKIP] unbabel-comet not installed, skipping COMET. Install: pip install unbabel-comet")
        return df

    print(f"[INFO] Filling COMET using model: {model_name} (will download if missing).")
    ckpt_path = download_model(model_name)
    model = load_from_checkpoint(ckpt_path)

    for syscol in systems:
        out_col = f"comet__{syscol}"
        if out_col not in df.columns:
            df[out_col] = np.nan

        refs = df[ref_col].map(norm_str).tolist()
        hyps = df[syscol].map(norm_str).tolist()
        srcs = df.get("source_text", pd.Series([""] * len(df))).map(norm_str).tolist()

        need_idx = []
        samples = []
        for i in range(len(df)):
            if not refs[i] or not hyps[i]:
                continue
            if overwrite or pd.isna(df.at[i, out_col]):
                need_idx.append(i)
                samples.append({"src": srcs[i], "mt": hyps[i], "ref": refs[i]})

        if not need_idx:
            print(f"[OK] COMET already filled for {syscol}.")
            continue

        # Predict
        # gpus defaults based on availability; COMET handles it internally
        preds = model.predict(samples, batch_size=16, gpus=1 if model.device.type == "cuda" else 0)
        scores = preds.scores

        for j, i in enumerate(need_idx):
            df.at[i, out_col] = float(scores[j])

        print(f"[OK] COMET filled for {syscol}.")
    return df


# -----------------------
# BLEURT (HF transformers)
# -----------------------
def fill_bleurt(df, systems, ref_col, overwrite=False,
                model_name="lucadiliello/BLEURT-20-D12",
                batch_size=16):
    """
    BLEURT via HuggingFace:
      pip install transformers sentencepiece torch
    Model can be switched; this is a commonly used port.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("[SKIP] transformers/torch not installed, skipping BLEURT. Install: pip install transformers torch sentencepiece")
        return df

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Filling BLEURT using HF model {model_name} on {device} (downloads on first run).")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    def score_batch(refs, hyps):
        # BLEURT expects (reference, candidate) pairs
        enc = tokenizer(
            refs, hyps,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
            # logits shape: (batch, 1) or (batch,)
            logits = out.logits.squeeze(-1).detach().cpu().numpy()
        return logits.tolist()

    for syscol in systems:
        out_col = f"bleurt__{syscol}"
        if out_col not in df.columns:
            df[out_col] = np.nan

        refs = df[ref_col].map(norm_str).tolist()
        hyps = df[syscol].map(norm_str).tolist()

        need_idx = []
        r_need, h_need = [], []
        for i in range(len(df)):
            if not refs[i] or not hyps[i]:
                continue
            if overwrite or pd.isna(df.at[i, out_col]):
                need_idx.append(i)
                r_need.append(refs[i])
                h_need.append(hyps[i])

        if not need_idx:
            print(f"[OK] BLEURT already filled for {syscol}.")
            continue

        scores_all = []
        for start in tqdm(range(0, len(need_idx), batch_size),
                          desc=f"BLEURT for {syscol}", total=math.ceil(len(need_idx) / batch_size)):
            rb = r_need[start:start+batch_size]
            hb = h_need[start:start+batch_size]
            scores_all.extend(score_batch(rb, hb))

        for j, i in enumerate(need_idx):
            df.at[i, out_col] = float(scores_all[j])

        print(f"[OK] BLEURT filled for {syscol}.")
    return df


def ensure_cols(df, systems):
    # If columns are missing, add them so the output is consistent
    metric_bases = ["ter", "bertscore_f1", "comet", "bleurt"]
    for m in metric_bases:
        for s in systems:
            col = f"{m}__{s}"
            if col not in df.columns:
                df[col] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV")
    ap.add_argument("--output", required=True, help="Output CSV")
    ap.add_argument("--ref_col", default=REF_COL_DEFAULT, help="Reference column name (default: ref)")
    ap.add_argument("--systems", default=",".join(SYSTEMS_DEFAULT),
                    help="Comma-separated system columns (default: chatgpt,gemini,google_translate,microsoft_translate)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing metric values")

    # BERTScore config
    ap.add_argument("--bertscore_model", default="xlm-roberta-large")
    ap.add_argument("--bertscore_batch", type=int, default=16)
    ap.add_argument("--bertscore_rescale", action="store_true",
                    help="Enable baseline rescaling (NOT recommended for mixed EN/FJ). Requires --bertscore_lang.")
    ap.add_argument("--bertscore_lang", default=None, help="Language code for BERTScore rescaling (e.g., en).")

    # COMET config
    ap.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")

    # BLEURT config
    ap.add_argument("--bleurt_model", default="lucadiliello/BLEURT-20-D12")
    ap.add_argument("--bleurt_batch", type=int, default=16)

    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    df = pd.read_csv(args.input)
    print(f"[INFO] Loaded {len(df)} rows from {os.path.basename(args.input)}")

    df = ensure_cols(df, systems)

    # Fill metrics
    print("[INFO] Filling metrics...")
    df = fill_ter(df, systems, args.ref_col, overwrite=args.overwrite)

    df = fill_bertscore(
        df, systems, args.ref_col, overwrite=args.overwrite,
        model_type=args.bertscore_model,
        batch_size=args.bertscore_batch,
        rescale_with_baseline=args.bertscore_rescale,
        lang=args.bertscore_lang
    )

    df = fill_comet(df, systems, args.ref_col, overwrite=args.overwrite, model_name=args.comet_model)

    df = fill_bleurt(
        df, systems, args.ref_col, overwrite=args.overwrite,
        model_name=args.bleurt_model,
        batch_size=args.bleurt_batch
    )

    df.to_csv(args.output, index=False)
    print(f"[DONE] Saved: {args.output}")


if __name__ == "__main__":
    main()
