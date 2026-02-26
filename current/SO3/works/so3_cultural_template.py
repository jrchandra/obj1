import argparse
import pandas as pd
import numpy as np
import random
import regex as re

def load_lexicon(path):
    # one term per line (can be multiword)
    terms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                terms.append(t)
    return terms

def build_pattern(terms):
    # escape terms; match as words where possible
    escaped = [re.escape(t) for t in terms]
    # allow whitespace normalization
    escaped = [t.replace(r"\ ", r"\s+") for t in escaped]
    return re.compile(r"(" + "|".join(escaped) + r")", re.I)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ref_col", default="reference_text")
    ap.add_argument("--src_col", default="source_text")
    ap.add_argument(
    "--systems",
    default="chatgpt,gemini,google_translate,microsoft_translate",
    help="Comma-separated system output columns"
)

    ap.add_argument("--lexicon_txt", required=True, help="Terms for culture/idioms/kinship/honorifics etc.")
    ap.add_argument("--sample_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.in_csv)
    systems = [c.strip() for c in args.systems.split(",") if c.strip()]

    for c in [args.src_col, args.ref_col] + systems:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    terms = load_lexicon(args.lexicon_txt)
    if not terms:
        raise ValueError("Lexicon is empty. Put at least a few terms in the txt file.")
    pat = build_pattern(terms)

    # filter: match in source or reference (you can choose only source if preferred)
    mask = df[args.src_col].fillna("").astype(str).str.contains(pat) | df[args.ref_col].fillna("").astype(str).str.contains(pat)
    sub = df[mask].copy()

    if len(sub) == 0:
        raise ValueError("No rows matched lexicon. Expand your lexicon or change matching logic.")

    # sample
    sub = sub.sample(n=min(args.sample_n, len(sub)), random_state=args.seed).reset_index(drop=True)

    # build blinded candidates
    cand_labels = [f"cand_{chr(ord('A') + i)}" for i in range(len(systems))]
    rows = []
    for i, row in sub.iterrows():
        sys_vals = [(s, str(row[s]) if pd.notna(row[s]) else "") for s in systems]
        random.shuffle(sys_vals)

        out = {
            "item_id": i + 1,
            "source_text": row[args.src_col],
            "reference_text": row[args.ref_col],
        }

        # store shuffled candidates + hidden key
        key = {}
        for lab, (sysname, text) in zip(cand_labels, sys_vals):
            out[lab] = text
            key[lab] = sysname

        # rubric fields (fill by humans)
        for lab in cand_labels:
            out[f"{lab}__idiom_handling_score_0_4"] = ""
            out[f"{lab}__cultural_appropriateness_0_4"] = ""
            out[f"{lab}__pragmatic_tone_register_0_4"] = ""
            out[f"{lab}__notes"] = ""

        # keep key for later decoding (you can remove this column before giving to annotators)
        out["_BLIND_KEY_JSON"] = str(key)

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] Wrote: {args.out_csv}")
    print("IMPORTANT: remove _BLIND_KEY_JSON before sharing with annotators.")

if __name__ == "__main__":
    main()
