import argparse
import os
import pandas as pd
import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)

def run_st_cosine(df, ref_col, hyp_col,
                  model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    refs = df[ref_col].fillna("").astype(str).tolist()
    hyps = df[hyp_col].fillna("").astype(str).tolist()

    emb_ref = model.encode(refs, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    emb_hyp = model.encode(hyps, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    return cosine_sim(emb_ref, emb_hyp)

def run_bertscore(df, ref_col, hyp_col, lang="en"):
    from bert_score import score
    refs = df[ref_col].fillna("").astype(str).tolist()
    hyps = df[hyp_col].fillna("").astype(str).tolist()
    P, R, F1 = score(hyps, refs, lang=lang, rescale_with_baseline=True, verbose=True)
    return F1.numpy()

def run_comet(df, src_col, ref_col, hyp_col, model_id="Unbabel/wmt22-comet-da"):
    from comet import download_model, load_from_checkpoint
    model_path = download_model(model_id)
    model = load_from_checkpoint(model_path)

    data = [{"src": s, "mt": mt, "ref": r}
            for s, mt, r in zip(df[src_col].fillna("").astype(str),
                                df[hyp_col].fillna("").astype(str),
                                df[ref_col].fillna("").astype(str))]

    preds = model.predict(data, batch_size=8, gpus=1 if os.environ.get("CUDA_VISIBLE_DEVICES") else 0)
    return np.array(preds.scores, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)

    # UPDATED DEFAULTS FOR YOUR SO2 CSV
    ap.add_argument("--src_col", default="source_text")
    ap.add_argument("--ref_col", default="ref")
    ap.add_argument("--systems", default="chatgpt,gemini,google_translate,microsoft_translate")

    ap.add_argument("--metric_st", action="store_true")
    ap.add_argument("--metric_bertscore", action="store_true")
    ap.add_argument("--bertscore_lang", default="en")
    ap.add_argument("--metric_comet", action="store_true")
    ap.add_argument("--comet_model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--st_model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    systems = [c.strip() for c in args.systems.split(",") if c.strip()]

    missing = [c for c in [args.src_col, args.ref_col] + systems if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    for syscol in systems:
        if args.metric_st:
            df[f"{syscol}__st_cos"] = run_st_cosine(df, args.ref_col, syscol, model_name=args.st_model)

        if args.metric_bertscore:
            df[f"{syscol}__bertscore_f1"] = run_bertscore(df, args.ref_col, syscol, lang=args.bertscore_lang)

        if args.metric_comet:
            df[f"{syscol}__comet"] = run_comet(df, args.src_col, args.ref_col, syscol, model_id=args.comet_model)

    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()
