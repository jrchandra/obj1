import argparse
import pandas as pd
import numpy as np
import re

def load_terms(path):
    terms = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                terms.append(t)
    return terms

def build_pattern_string(terms):
    escaped = []
    for t in terms:
        t = t.strip()
        if not t:
            continue
        # escape special chars, allow flexible whitespace for multiword phrases
        s = re.escape(t).replace(r"\ ", r"\s+")
        escaped.append(s)
    if not escaped:
        raise ValueError("No usable lexicon terms found.")
    # non-capturing group
    return r"(?:%s)" % "|".join(escaped)

def normalize_to_risk(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([0.0]*len(series), index=series.index)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi - lo < 1e-12:
        norm = pd.Series([0.0]*len(series), index=series.index)
    else:
        norm = (s - lo) / (hi - lo)
        norm = norm.fillna(0.0)

    # convert to "risk": low quality => higher risk
    return (1 - norm) if higher_is_better else norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--lexicon_txt", required=True)
    ap.add_argument("--out_ranked_csv", default="so3_cultural_risk_ranked.csv")
    ap.add_argument("--out_summary_csv", default="so3_cultural_risk_summary.csv")
    ap.add_argument("--systems", default="chatgpt,gemini,google_translate,microsoft_translate")
    ap.add_argument("--src_col", default="source_text")
    ap.add_argument("--ref_col", default="ref")
    ap.add_argument("--id_col", default=None)
    args
