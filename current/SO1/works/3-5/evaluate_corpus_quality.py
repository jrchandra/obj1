import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

MASTER_FILE_XLSX = "master_parallel_corpus__COMBINED.xlsx"
MASTER_FILE_CSV  = "master_parallel_corpus__COMBINED.csv"

OUTPUT_DIR = "corpus_quality_eval"

# ------------------ LOAD DATA -------------------
def load_corpus():
    if Path(MASTER_FILE_XLSX).exists():
        df = pd.read_excel(MASTER_FILE_XLSX)
    elif Path(MASTER_FILE_CSV).exists():
        df = pd.read_csv(MASTER_FILE_CSV)
    else:
        raise FileNotFoundError("Master corpus file not found.")
    df["source_text"] = df["source_text"].astype(str)
    df["target_text"] = df["target_text"].astype(str)
    return df

# ------------------ LANGUAGE HEURISTICS -------------------
def fj_char_ratio(text):
    """
    Approx. Fijian-specific characters: 'q', 'drau' patterns, 'g̱', 'c' pronounced /ð/, etc.
    Heuristic only.
    """
    text = text.lower()
    count = sum(c in "qcđḡ" for c in text)
    return count / max(len(text), 1)

def en_char_ratio(text):
    """
    Rough heuristic: high frequency of th, sh, tion, ing etc.
    """
    text = text.lower()
    score = 0
    score += text.count("th")
    score += text.count("ing")
    score += text.count("tion")
    return score / max(len(text.split()), 1)

def detect_noise(text):
    """
    Flags: too short, numbers-only, symbols-only, mixed language indicators.
    """
    if len(text.strip()) == 0:
        return True
    if re.fullmatch(r"[0-9]+", text.strip()):
        return True
    if len(text.split()) <= 1:
        return True
    return False

# ------------------ ALIGNMENT QUALITY -------------------
def compute_alignment_similarity(df):
    """Cosine similarity score between EN and FJ vectors (correlated pairs)."""
    texts = df["source_text"].tolist() + df["target_text"].tolist()
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
    tfidf = vectorizer.fit_transform(texts)

    src_vec = tfidf[: len(df)]
    tgt_vec = tfidf[len(df):]

    sims = cosine_similarity(src_vec, tgt_vec).diagonal()
    df["alignment_similarity"] = sims
    return df

# ------------------ MAIN EVALUATION PIPELINE -------------------
def evaluate():
    df = load_corpus()
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # --- Language score heuristics ---
    df["fj_ratio_src"] = df["source_text"].apply(fj_char_ratio)
    df["fj_ratio_tgt"] = df["target_text"].apply(fj_char_ratio)

    df["en_ratio_src"] = df["source_text"].apply(en_char_ratio)
    df["en_ratio_tgt"] = df["target_text"].apply(en_char_ratio)

    df["src_noise"] = df["source_text"].apply(detect_noise)
    df["tgt_noise"] = df["target_text"].apply(detect_noise)

    # --- Alignment ---
    df = compute_alignment_similarity(df)

    # Save intermediate results
    df.to_csv(f"{OUTPUT_DIR}/quality_raw.csv", index=False)

    # --- Summaries ---
    summary = {}
    summary["total_segments"] = len(df)
    summary["src_noise_percent"] = df["src_noise"].mean() * 100
    summary["tgt_noise_percent"] = df["tgt_noise"].mean() * 100
    summary["alignment_mean"]  = df["alignment_similarity"].mean()
    summary["alignment_median"] = df["alignment_similarity"].median()
    summary["alignment_low_quality_pct"] = ((df["alignment_similarity"] < 0.1).mean() * 100)

    # Language identity correctness
    summary["likely_english_source_pct"] = (df["en_ratio_src"] > df["fj_ratio_src"]).mean() * 100
    summary["likely_fijian_target_pct"] = (df["fj_ratio_tgt"] > df["en_ratio_tgt"]).mean() * 100

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{OUTPUT_DIR}/quality_summary.csv", index=False)

    print("\n======= CORPUS QUALITY SUMMARY =======")
    print(summary_df)

    print("\nResults saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    evaluate()
