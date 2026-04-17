"""
Fill TER, BERTScore_F1, COMET, BLEURT columns in a dataset that already has:
- ref column (default: "ref")
- system output columns (default: chatgpt, gemini, google_translate, microsoft_translate)

It writes metrics into columns named:
  ter__<system>
  bertscore_f1__<system>
  comet__<system>
  bleurt__<system>

Designed to be robust on Windows (COMET download symlink issue).
"""

import os
import re
import sys
import glob
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_SYSTEMS = ["chatgpt", "gemini", "google_translate", "microsoft_translate"]
METRIC_COLS = ["ter", "bertscore_f1", "comet", "bleurt"]


def ensure_metric_columns(df: pd.DataFrame, systems: List[str]) -> pd.DataFrame:
    for m in METRIC_COLS:
        for s in systems:
            col = f"{m}__{s}"
            if col not in df.columns:
                df[col] = np.nan
    return df


def norm_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


# -------------------------
# TER (needs sacrebleu)
# -------------------------
def fill_ter(
    df: pd.DataFrame,
    systems: List[str],
    ref_col: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    try:
        from sacrebleu.metrics import TER
    except Exception as e:
        print("[WARN] sacrebleu not installed. TER will remain NaN.")
        print("       Install: pip install sacrebleu")
        return df

    ter_metric = TER()

    for syscol in systems:
        outcol = f"ter__{syscol}"
        if (not overwrite) and df[outcol].notna().any():
            # still fill missing
            pass

        refs = df[ref_col].astype("string").fillna("")
        hyps = df[syscol].astype("string").fillna("")

        scores = []
        print(f"[INFO] TER for {syscol}: computing {len(df)} pairs...")
        for r, h, cur in tqdm(zip(refs, hyps, df[outcol]), total=len(df)):
            if (not overwrite) and (pd.notna(cur)):
                scores.append(cur)
                continue
            r = norm_str(r)
            h = norm_str(h)
            if not r or not h:
                scores.append(np.nan)
                continue
            # sacrebleu TER returns score in percentage (0..100) normally; convert to 0..1
            sc = ter_metric.sentence_score(h, [r]).score
            scores.append(float(sc) / 100.0)

        df[outcol] = scores

    print("[OK] TER filled (stored as 0..1).")
    return df


# -------------------------
# BERTScore F1 (needs bert-score)
# -------------------------
def fill_bertscore(
    df: pd.DataFrame,
    systems: List[str],
    ref_col: str,
    overwrite: bool = False,
    model_type: str = "xlm-roberta-large",
    rescale_with_baseline: bool = False,
    lang: Optional[str] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    try:
        from bert_score import score as bertscore_score
    except Exception:
        print("[WARN] bert-score not installed. BERTScore will remain NaN.")
        print("       Install: pip install bert-score")
        return df

    # IMPORTANT: Your error came from rescale_with_baseline=True without lang.
    if rescale_with_baseline and not lang:
        raise ValueError(
            "BERTScore rescale_with_baseline=True requires --bertscore_lang (e.g., en). "
            "Or run without --bertscore_rescale."
        )

    for syscol in systems:
        outcol = f"bertscore_f1__{syscol}"
        refs = df[ref_col].astype("string").fillna("")
        hyps = df[syscol].astype("string").fillna("")

        # Build index list of rows to compute
        idx = []
        preds = []
        golds = []
        for i, (r, h, cur) in enumerate(zip(refs, hyps, df[outcol])):
            if (not overwrite) and pd.notna(cur):
                continue
            r = norm_str(r)
            h = norm_str(h)
            if not r or not h:
                continue
            idx.append(i)
            preds.append(h)
            golds.append(r)

        if not idx:
            print(f"[INFO] BERTScore({model_type}) for {syscol}: nothing to fill.")
            continue

        print(f"[INFO] BERTScore({model_type}) for {syscol}: computing {len(idx)} pairs...")
        # Compute in one go (bertscore internally batches)
        P, R, F1 = bertscore_score(
            preds,
            golds,
            model_type=model_type,
            lang=lang,  # only needed when rescaling with baseline
            rescale_with_baseline=rescale_with_baseline,
            batch_size=batch_size,
            verbose=True,
        )
        F1 = F1.cpu().numpy().astype(float)

        # Write back
        for k, row_i in enumerate(idx):
            df.at[row_i, outcol] = F1[k]

    print("[OK] BERTScore F1 filled.")
    return df


# -------------------------
# COMET (needs unbabel-comet + HF download)
# -------------------------
def _download_comet_repo_no_symlinks(model_id: str, cache_dir: str) -> str:
    """
    Downloads the HF repo to a local folder with symlinks disabled (Windows-safe),
    returns local path.
    """
    from huggingface_hub import snapshot_download

    local_dir = os.path.join(cache_dir, "hf_models", re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id))
    os.makedirs(local_dir, exist_ok=True)

    repo_path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,   # key for WinError 1314
        resume_download=True,
    )
    return repo_path


def _find_ckpt(repo_path: str) -> str:
    # COMET HF repos typically contain a .ckpt somewhere
    ckpts = glob.glob(os.path.join(repo_path, "**", "*.ckpt"), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found inside downloaded repo: {repo_path}")
    # pick the largest ckpt
    ckpts = sorted(ckpts, key=lambda p: os.path.getsize(p), reverse=True)
    return ckpts[0]


def fill_comet(
    df: pd.DataFrame,
    systems: List[str],
    ref_col: str,
    overwrite: bool = False,
    model_id: str = "Unbabel/wmt22-comet-da",
    cache_dir: str = ".comet_cache",
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Uses reference-based COMET: needs source, mt, ref triplets.
    Your CSV doesn't appear to have the *source* sentence separate from ref,
    BUT you do have source_text and ref.

    We'll use:
      src = source_text
      mt  = system output
      ref = ref
    """
    try:
        from comet import load_from_checkpoint
    except Exception:
        print("[WARN] unbabel-comet not installed. COMET will remain NaN.")
        print("       Install: pip install unbabel-comet")
        return df

    # Download without symlinks (Windows-friendly)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        repo_path = _download_comet_repo_no_symlinks(model_id, cache_dir=cache_dir)
        ckpt_path = _find_ckpt(repo_path)
    except Exception as e:
        print("[ERROR] Could not download/load COMET model.")
        print("        Root cause:", repr(e))
        print("        Fix options:")
        print("        1) Upgrade COMET:  pip install -U unbabel-comet")
        print("        2) Ensure HuggingFace Hub works without symlinks (script already disables them).")
        print("        3) If you are offline or behind a proxy, pre-download the model on a machine with access.")
        return df

    model = load_from_checkpoint(ckpt_path)
    model.eval()

    if "source_text" not in df.columns:
        print("[WARN] No 'source_text' column found; COMET needs source. COMET will remain NaN.")
        return df

    for syscol in systems:
        outcol = f"comet__{syscol}"

        rows = []
        row_ids = []

        for i, (src, mt, ref, cur) in enumerate(
            zip(df["source_text"], df[syscol], df[ref_col], df[outcol])
        ):
            if (not overwrite) and pd.notna(cur):
                continue
            src = norm_str(src)
            mt = norm_str(mt)
            ref = norm_str(ref)
            if not src or not mt or not ref:
                continue
            rows.append({"src": src, "mt": mt, "ref": ref})
            row_ids.append(i)

        if not row_ids:
            print(f"[INFO] COMET({model_id}) for {syscol}: nothing to fill.")
            continue

        print(f"[INFO] COMET({model_id}) for {syscol}: scoring {len(row_ids)} triplets...")
        # model.predict expects list of dicts
        outputs = model.predict(rows, batch_size=batch_size, gpus=0)
        scores = outputs.scores  # list of floats

        for k, i in enumerate(row_ids):
            df.at[i, outcol] = float(scores[k])

    print("[OK] COMET filled.")
    return df


# -------------------------
# BLEURT (optional/heavy deps)
# -------------------------
def fill_bleurt(
    df: pd.DataFrame,
    systems: List[str],
    ref_col: str,
    overwrite: bool = False,
    checkpoint: str = "bleurt-base-128",
) -> pd.DataFrame:
    """
    BLEURT is often the hardest to set up on Windows (tensorflow + bleurt).
    This function fills it if available; otherwise keeps NaN and prints instructions.

    Recommended install (often easiest):
      pip install tensorflow
      pip install bleurt
    Checkpoint files may need manual download depending on the bleurt package version.
    """
    try:
        from bleurt import score as bleurt_score
    except Exception:
        print("[WARN] BLEURT not available. BLEURT will remain NaN.")
        print("       Install (may be heavy): pip install tensorflow bleurt")
        return df

    # BLEURT scorer needs a local checkpoint folder for many installs.
    # Some distributions include it; others require download.
    try:
        scorer = bleurt_score.BleurtScorer(checkpoint)
    except Exception as e:
        print("[ERROR] BLEURT installed but checkpoint could not be loaded:", repr(e))
        print("        You likely need to download a BLEURT checkpoint locally and pass its path.")
        print("        Example: --bleurt_checkpoint C:\\path\\to\\bleurt-base-128")
        return df

    for syscol in systems:
        outcol = f"bleurt__{syscol}"
        refs = df[ref_col].astype("string").fillna("")
        hyps = df[syscol].astype("string").fillna("")

        idx, preds, golds = [], [], []
        for i, (r, h, cur) in enumerate(zip(refs, hyps, df[outcol])):
            if (not overwrite) and pd.notna(cur):
                continue
            r = norm_str(r)
            h = norm_str(h)
            if not r or not h:
                continue
            idx.append(i)
            golds.append(r)
            preds.append(h)

        if not idx:
            print(f"[INFO] BLEURT({checkpoint}) for {syscol}: nothing to fill.")
            continue

        print(f"[INFO] BLEURT({checkpoint}) for {syscol}: computing {len(idx)} pairs...")
        scores = scorer.score(references=golds, candidates=preds)
        for k, i in enumerate(idx):
            df.at[i, outcol] = float(scores[k])

    print("[OK] BLEURT filled.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV (your dataset with system outputs + metric columns).")
    ap.add_argument("--output", required=True, help="Output CSV path.")
    ap.add_argument("--ref_col", default="ref", help="Reference column name (default: ref).")
    ap.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS), help="Comma-separated system columns.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing metric values (default: only fill missing).")

    # BERTScore
    ap.add_argument("--bertscore_model", default="xlm-roberta-large")
    ap.add_argument("--bertscore_batch", type=int, default=32)
    ap.add_argument("--bertscore_rescale", action="store_true",
                    help="If set, uses rescale_with_baseline=True (requires --bertscore_lang).")
    ap.add_argument("--bertscore_lang", default=None, help="Language code used when rescaling with baseline (e.g., en).")

    # COMET
    ap.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--comet_cache", default=".comet_cache")
    ap.add_argument("--comet_batch", type=int, default=16)

    # BLEURT
    ap.add_argument("--bleurt_checkpoint", default="bleurt-base-128")

    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    df = pd.read_csv(args.input)

    print(f"[INFO] Loaded {len(df)} rows from {os.path.basename(args.input)}")
    df = ensure_metric_columns(df, systems)

    # Fill each metric
    print("[INFO] Filling metrics...")
    df = fill_ter(df, systems, args.ref_col, overwrite=args.overwrite)
    df = fill_bertscore(
        df, systems, args.ref_col,
        overwrite=args.overwrite,
        model_type=args.bertscore_model,
        rescale_with_baseline=args.bertscore_rescale,
        lang=args.bertscore_lang,
        batch_size=args.bertscore_batch,
    )
    df = fill_comet(
        df, systems, args.ref_col,
        overwrite=args.overwrite,
        model_id=args.comet_model,
        cache_dir=args.comet_cache,
        batch_size=args.comet_batch,
    )
    df = fill_bleurt(
        df, systems, args.ref_col,
        overwrite=args.overwrite,
        checkpoint=args.bleurt_checkpoint,
    )

    df.to_csv(args.output, index=False)
    print(f"[OK] Saved: {args.output}")


if __name__ == "__main__":
    main()
