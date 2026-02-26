import os
import json
import pandas as pd
from tqdm import tqdm

from openai import OpenAI

# -------------------------
# CONFIG
# -------------------------
INPUT_CSV  = "dataset_with_system_outputs__metrics_filled_all_with_bluert_and_rouge.csv"  # or your current file
OUTPUT_CSV = "dataset_with_system_outputs__metrics_filled_all_with_bluert_and_rouge_plus_GEMBA.csv"

MODEL = "gpt-4.1-mini"   # choose what you use in your account

REF_COL = "ref"
SRC_COL = "source_text"
SYSTEM_COLS = ["chatgpt", "gemini", "google_translate", "microsoft_translate"]

USE_REFERENCE = True   # set False for reference-free GEMBA-style judging

client = OpenAI()


def to_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def ensure_cols(df):
    for sys in SYSTEM_COLS:
        for c in [f"gemba_score__{sys}", f"gemba_error_tags__{sys}", f"gemba_rationale__{sys}"]:
            if c not in df.columns:
                df[c] = pd.NA
    return df


JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 100},
        "error_tags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "no_error",
                    "mistranslation",
                    "omission",
                    "addition",
                    "grammar",
                    "fluency",
                    "terminology",
                    "style_register",
                    "punctuation",
                    "named_entity",
                    "word_order",
                    "inconsistency",
                    "other"
                ]
            }
        },
        "rationale": {"type": "string"}
    },
    "required": ["score", "error_tags", "rationale"]
}


def judge_one(source, reference, hypothesis):
    if USE_REFERENCE:
        prompt = (
            "You are an expert machine translation evaluator.\n"
            "Score the hypothesis translation from 0 to 100 where 100 is perfect.\n"
            "Use the source and the reference to judge meaning preservation and fluency.\n"
            "Return JSON that matches the schema.\n\n"
            f"SOURCE:\n{source}\n\n"
            f"REFERENCE:\n{reference}\n\n"
            f"HYPOTHESIS:\n{hypothesis}\n"
        )
    else:
        prompt = (
            "You are an expert machine translation evaluator.\n"
            "Score the hypothesis translation from 0 to 100 where 100 is perfect.\n"
            "Judge adequacy vs the source and fluency of the hypothesis. No reference is provided.\n"
            "Return JSON that matches the schema.\n\n"
            f"SOURCE:\n{source}\n\n"
            f"HYPOTHESIS:\n{hypothesis}\n"
        )

    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        text={
            "format": {
                "name": "gemba_mt_judge",
                "type": "json_schema",
                "schema": JUDGE_SCHEMA,
                "strict": True
            }
        }
    )

    # Responses API: safest extraction is to parse the first output text
    # (When strict schema is used, the text is valid JSON)
    txt = resp.output_text
    return json.loads(txt)


def main():
    df = pd.read_csv(INPUT_CSV)
    df = ensure_cols(df)

    for sys in SYSTEM_COLS:
        score_col = f"gemba_score__{sys}"
        tags_col  = f"gemba_error_tags__{sys}"
        rat_col   = f"gemba_rationale__{sys}"

        print(f"[INFO] GEMBA judging for {sys}...")

        for i in tqdm(range(len(df))):
            if pd.notna(df.at[i, score_col]) and pd.notna(df.at[i, tags_col]) and pd.notna(df.at[i, rat_col]):
                continue

            source = to_str(df.at[i, SRC_COL])
            ref    = to_str(df.at[i, REF_COL])
            hyp    = to_str(df.at[i, sys])

            if not source or (USE_REFERENCE and not ref) or not hyp:
                continue

            try:
                out = judge_one(source, ref, hyp)
                df.at[i, score_col] = float(out["score"])
                df.at[i, tags_col]  = json.dumps(out["error_tags"], ensure_ascii=False)
                df.at[i, rat_col]   = out["rationale"]
            except Exception as e:
                df.at[i, rat_col] = f"JUDGE_ERROR: {type(e).__name__}: {e}"

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
