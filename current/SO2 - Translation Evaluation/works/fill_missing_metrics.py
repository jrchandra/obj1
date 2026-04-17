import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ----------------------------
# CONFIG
# ----------------------------
INPUT_CSV = "dataset_with_system_outputs__metrics_filled.csv"
OUTPUT_CSV = "dataset_with_system_outputs__with_all_metrics_filled.csv"

REF_COL = "ref"
SYSTEM_COLS = ["chatgpt", "gemini", "google_translate", "microsoft_translate"]

# You can change these if you want
BERTSCORE_MODEL = "xlm-roberta-large"   # good multilingual default
COMET_MODEL = "Unbabel/wmt22-comet-da"  # strong general COMET model
BLEURT_CHECKPOINT = "bleurt-base-128"   # light-ish BLEURT checkpoint

# ----------------------------
# HELPERS
# ----------------------------
def _norm(x: str) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()

def ensure_column(df: pd.DataFrame, col: str):
    if col not in df.columns:
        df[col] = np.nan

def ensure_metric_columns(df: pd.DataFrame, metric_name: str):
    for syscol in SYSTEM_COLS:
        ensure_column(df, f"{metric_name}__{syscol}")

def has_nonempty_pair(ref: str, hyp: str) -> bool:
    return bool(_norm(ref)) and bool(_norm(hyp))

# ----------------------------
# TER (via sacrebleu)
# ----------------------------
def fill_ter(df: pd.DataFrame) -> bool:
    try:
        import sacrebleu
    except Exception as e:
        print(f"[SKIP] TER: sacrebleu not available ({e}). Install: pip install sacrebleu")
        return False

    ensure_metric_columns(df, "ter")

    refs = df[REF_COL].astype("string").fillna("")
    for syscol in SYSTEM_COLS:
        hyps = df[syscol].astype("string").fillna("")
        out = []
        for r, h in tqdm(zip(refs, hyps), total=len(df), desc=f"TER for {syscol}"):
            r = _norm(r)
            h = _norm(h)
            if not has_nonempty_pair(r, h):
                out.append(np.nan)
                continue
            # sentence-level TER via corpus_ter on one example
            score = sacrebleu.corpus_ter([h], [[r]]).score  # % TER (0..100)
            out.append(float(score) / 100.0)  # store 0..1
        df[f"ter__{syscol}"] = out

    print("[OK] TER filled (stored as 0..1).")
    return True

# ----------------------------
# BERTScore F1 (via bert-score)
# ----------------------------
def fill_bertscore(df: pd.DataFrame) -> bool:
    try:
        from bert_score import score as bertscore_score
    except Exception as e:
        print(f"[SKIP] BERTScore: bert-score not available ({e}). Install: pip install bert-score")
        return False

    ensure_metric_columns(df, "bertscore_f1")

    refs_all = df[REF_COL].astype("string").fillna("").map(_norm).tolist()

    for syscol in SYSTEM_COLS:
        hyps_all = df[syscol].astype("string").fillna("").map(_norm).tolist()

        # Only compute for valid pairs, keep alignment
        idx = [i for i, (r, h) in enumerate(zip(refs_all, hyps_all)) if has_nonempty_pair(r, h)]
        if not idx:
            df[f"bertscore_f1__{syscol}"] = np.nan
            continue

        refs = [refs_all[i] for i in idx]
        hyps = [hyps_all[i] for i in idx]

        print(f"[INFO] BERTScore({BERTSCORE_MODEL}) for {syscol}: computing {len(idx)} pairs...")
        P, R, F1 = bertscore_score(
            cands=hyps,
            refs=refs,
            model_type=BERTSCORE_MODEL,
            lang=None,          # multilingual model handles it
            verbose=False,
            rescale_with_baseline=True
        )

        # Fill back into full-length vector
        full = [np.nan] * len(df)
        f1_list = F1.cpu().numpy().tolist()
        for j, i in enumerate(idx):
            full[i] = float(f1_list[j])

        df[f"bertscore_f1__{syscol}"] = full

    print("[OK] BERTScore F1 filled (0..1).")
    return True

# ----------------------------
# COMET (via unbabel-comet)
# ----------------------------
def fill_comet(df: pd.DataFrame) -> bool:
    try:
        from comet import download_model, load_from_checkpoint
    except Exception as e:
        print(f"[SKIP] COMET: unbabel-comet not available ({e}). Install: pip install unbabel-comet")
        return False

    ensure_metric_columns(df, "comet")

    print(f"[INFO] Download/load COMET model: {COMET_MODEL}")
    ckpt_path = download_model(COMET_MODEL)
    model = load_from_checkpoint(ckpt_path)

    # COMET expects: {"src":..., "mt":..., "ref":...}
    # Your CSV has source_text and ref (reference translation). That’s enough.
    # If your direction is en->fj, source_text is EN, ref is FJ. Works.
    # If direction were fj->en, source_text would be FJ, ref would be EN. Works too.

    if "source_text" not in df.columns:
        print("[SKIP] COMET: missing source_text column (needed for src).")
        return False

    src_all = df["source_text"].astype("string").fillna("").map(_norm).tolist()
    ref_all = df[REF_COL].astype("string").fillna("").map(_norm).tolist()

    for syscol in SYSTEM_COLS:
        mt_all = df[syscol].astype("string").fillna("").map(_norm).tolist()

        data = []
        idx = []
        for i, (src, mt, ref) in enumerate(zip(src_all, mt_all, ref_all)):
            if not (_norm(src) and _norm(mt) and _norm(ref)):
                continue
            data.append({"src": src, "mt": mt, "ref": ref})
            idx.append(i)

        full = [np.nan] * len(df)
        if not data:
            df[f"comet__{syscol}"] = full
            continue

        print(f"[INFO] COMET for {syscol}: scoring {len(data)} triples...")
        # model.predict returns a dict with "scores"
        pred = model.predict(data, batch_size=8, gpus=1 if model.device.type == "cuda" else 0)
        scores = pred["scores"]

        for j, i in enumerate(idx):
            full[i] = float(scores[j])

        df[f"comet__{syscol}"] = full

    print("[OK] COMET filled (higher is better; typical range ~0..1 but can vary).")
    return True

# ----------------------------
# BLEURT (via bleurt)
# ----------------------------
def fill_bleurt(df: pd.DataFrame) -> bool:
    try:
        from bleurt import score as bleurt_score
    except Exception as e:
        print(f"[SKIP] BLEURT: bleurt not available ({e}). Install: pip install bleurt")
        return False

    ensure_metric_columns(df, "bleurt")

    refs_all = df[REF_COL].astype("string").fillna("").map(_norm).tolist()

    # BLEURT scorer downloads / loads checkpoint
    print(f"[INFO] Load BLEURT checkpoint: {BLEURT_CHECKPOINT}")
    scorer = bleurt_score.BleurtScorer(BLEURT_CHECKPOINT)

    for syscol in SYSTEM_COLS:
        hyps_all = df[syscol].astype("string").fillna("").map(_norm).tolist()

        idx = [i for i, (r, h) in enumerate(zip(refs_all, hyps_all)) if has_nonempty_pair(r, h)]
        full = [np.nan] * len(df)

        if not idx:
            df[f"bleurt__{syscol}"] = full
            continue

        refs = [refs_all[i] for i in idx]
        hyps = [hyps_all[i] for i in idx]

        print(f"[INFO] BLEURT for {syscol}: scoring {len(idx)} pairs...")
        # scores list length = len(refs)
        scores = scorer.score(references=refs, candidates=hyps, batch_size=16)

        for j, i in enumerate(idx):
            full[i] = float(scores[j])

        df[f"bleurt__{syscol}"] = full

    print("[OK] BLEURT filled (score scale depends on checkpoint; higher is better).")
    return True

# ----------------------------
# MAIN
# ----------------------------
def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"INPUT_CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # Ensure target columns exist even if we skip metrics
    ensure_metric_columns(df, "ter")
    ensure_metric_columns(df, "bertscore_f1")
    ensure_metric_columns(df, "comet")
    ensure_metric_columns(df, "bleurt")

    # Fill what we can
    print(f"[INFO] Loaded {len(df)} rows from {INPUT_CSV}")
    print("[INFO] Filling metrics...")

    fill_ter(df)
    fill_bertscore(df)
    fill_comet(df)
    fill_bleurt(df)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
