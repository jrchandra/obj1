import argparse
import pandas as pd
import numpy as np
import random
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
        s = re.escape(t).replace(r"\ ", r"\s+")
        escaped.append(s)
    if not escaped:
        raise ValueError("Lexicon has no usable terms.")
    return r"(?:%s)" % "|".join(escaped)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--key_csv", default="so3_cultural_blind_key.csv")

    ap.add_argument("--src_col", default="source_text")
    ap.add_argument("--ref_col", default="ref")
    ap.add_argument("--systems", default="chatgpt,gemini,google_translate,microsoft_translate")

    ap.add_argument("--lexicon_txt", required=True)
    ap.add_argument("--sample_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--match_in", choices=["src", "ref", "both"], default="both")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.in_csv, encoding="utf-8", low_memory=False)
    systems = [c.strip() for c in args.systems.split(",") if c.strip()]

    needed = [args.src_col, args.ref_col] + systems
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    terms = load_terms(args.lexicon_txt)
    pattern = build_pattern_string(terms)

    src_series = df[args.src_col].fillna("").astype(str)
    ref_series = df[args.ref_col].fillna("").astype(str)

    if args.match_in == "src":
        mask = src_series.str.contains(pattern, regex=True, flags=re.IGNORECASE)
    elif args.match_in == "ref":
        mask = ref_series.str.contains(pattern, regex=True, flags=re.IGNORECASE)
    else:
        mask = (
            src_series.str.contains(pattern, regex=True, flags=re.IGNORECASE) |
            ref_series.str.contains(pattern, regex=True, flags=re.IGNORECASE)
        )

    sub = df[mask].copy()
    if len(sub) == 0:
        print("[FAIL] No rows matched lexicon.")
        print("Try: --match_in src (often ref is English) and expand lexicon variants.")
        raise SystemExit(2)

    sub = sub.sample(n=min(args.sample_n, len(sub)), random_state=args.seed).reset_index(drop=True)

    cand_labels = [f"cand_{chr(ord('A') + i)}" for i in range(len(systems))]
    rows, key_rows = [], []

    for i, row in sub.iterrows():
        sys_vals = [(s, str(row[s]) if pd.notna(row[s]) else "") for s in systems]
        random.shuffle(sys_vals)

        out = {
            "item_id": i + 1,
            "source_text": row[args.src_col],
            "ref": row[args.ref_col],
        }

        for gc in ["domain", "direction", "sentence_type_fine", "source_id", "source_doc"]:
            if gc in sub.columns:
                out[gc] = row.get(gc, "")

        key_map = {"item_id": i + 1}
        for lab, (sysname, text) in zip(cand_labels, sys_vals):
            out[lab] = text
            key_map[lab] = sysname

        for lab in cand_labels:
            out[f"{lab}__semantic_fidelity_0_4"] = ""
            out[f"{lab}__cultural_adequacy_0_4"] = ""
            out[f"{lab}__fluency_0_4"] = ""
            out[f"{lab}__error_tags"] = ""
            out[f"{lab}__notes"] = ""

        rows.append(out)
        key_rows.append(key_map)

    pd.DataFrame(rows).to_csv(args.out_csv, index=False, encoding="utf-8")
    pd.DataFrame(key_rows).to_csv(args.key_csv, index=False, encoding="utf-8")

    print(f"[OK] Template for annotators: {args.out_csv}")
    print(f"[OK] Private blind key: {args.key_csv}")

if __name__ == "__main__":
    main()
