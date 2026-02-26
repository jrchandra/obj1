import argparse
import pandas as pd
import regex as re

NEG_EN = re.compile(r"\b(no|not|never|none|nothing|n't|cannot|can't|won't|don't|doesn't|didn't|without)\b", re.I)
NUM_RE = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)(?!\w)")

# lightweight named-entity-ish heuristic
NE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}|[A-Z]{2,}|[A-Za-z]+-\d+|\d+-[A-Za-z]+|[A-Za-z]+\d+)\b")

def extract(pattern, text):
    if text is None:
        return []
    return pattern.findall(str(text))

def set_overlap(a, b):
    a = set(map(str, a))
    b = set(map(str, b))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def negation_flip_flag(ref, hyp):
    ref_has = bool(NEG_EN.search(str(ref or "")))
    hyp_has = bool(NEG_EN.search(str(hyp or "")))
    return int(ref_has != hyp_has)

def numerals_mismatch(ref, hyp):
    return 1.0 - set_overlap(extract(NUM_RE, ref), extract(NUM_RE, hyp))

def ne_mismatch(ref, hyp):
    return 1.0 - set_overlap(extract(NE_RE, ref), extract(NE_RE, hyp))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)

    # UPDATED DEFAULTS FOR YOUR SO2 CSV
    ap.add_argument("--ref_col", default="ref")
    ap.add_argument("--systems", default="chatgpt,gemini,google_translate,microsoft_translate")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    systems = [c.strip() for c in args.systems.split(",") if c.strip()]

    for c in [args.ref_col] + systems:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    for syscol in systems:
        df[f"{syscol}__neg_flip_flag"] = [
            negation_flip_flag(r, h) for r, h in zip(df[args.ref_col], df[syscol])
        ]
        df[f"{syscol}__numeral_mismatch"] = [
            numerals_mismatch(r, h) for r, h in zip(df[args.ref_col], df[syscol])
        ]
        df[f"{syscol}__ne_mismatch"] = [
            ne_mismatch(r, h) for r, h in zip(df[args.ref_col], df[syscol])
        ]

        df[f"{syscol}__needs_review_flag"] = (
            (df[f"{syscol}__neg_flip_flag"] == 1) |
            (df[f"{syscol}__numeral_mismatch"] >= 0.5) |
            (df[f"{syscol}__ne_mismatch"] >= 0.5)
        ).astype(int)

    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()
