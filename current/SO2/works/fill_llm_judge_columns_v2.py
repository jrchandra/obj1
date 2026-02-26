import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ---------- CONFIG ----------
INPUT_CSV  = "dataset_with_system_outputs__metrics_filled_all_with_bluert_and_rouge.csv"
OUTPUT_CSV = "dataset_with_system_outputs__metrics_filled_all_with_bluert_and_rouge_plus_llm_judge.csv"

REF_COL = "ref"
SRC_COL = "source_text"
SYSTEMS = ["chatgpt", "gemini", "google_translate", "microsoft_translate"]

MODEL = "gpt-4o-mini"  # good cost/quality tradeoff
OVERWRITE = False

# Judge rubric: keep it consistent across systems
JUDGE_PROMPT = """You are an expert machine translation evaluator.
Given:
- source text (English)
- reference translation (Fijian)
- system translation (Fijian)

Score the system translation on 4 criteria from 1 to 5:
1) adequacy: meaning preserved vs source
2) fluency: grammatical/natural Fijian
3) terminology: correct key terms/proper nouns
4) overall: overall quality

Also output:
- error_tags: list of short tags (e.g., "omission","addition","mistranslation","grammar","named_entity","style","hallucination")
- rationale: 1-3 short sentences (no long essay)

Return ONLY valid JSON matching the schema.
"""

# ---------- Helpers ----------
def ensure_col(df, col):
    if col not in df.columns:
        df[col] = np.nan

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def main():
    client = OpenAI()  # uses OPENAI_API_KEY env var

    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Loaded {len(df)} rows from {INPUT_CSV}")

    # Create columns per system
    for sys in SYSTEMS:
        ensure_col(df, f"llm_adequacy__{sys}")
        ensure_col(df, f"llm_fluency__{sys}")
        ensure_col(df, f"llm_terminology__{sys}")
        ensure_col(df, f"llm_overall__{sys}")
        ensure_col(df, f"llm_error_tags__{sys}")
        ensure_col(df, f"llm_rationale__{sys}")

    # JSON schema for structured output via Responses API text.format
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "adequacy": {"type": "integer", "minimum": 1, "maximum": 5},
            "fluency": {"type": "integer", "minimum": 1, "maximum": 5},
            "terminology": {"type": "integer", "minimum": 1, "maximum": 5},
            "overall": {"type": "integer", "minimum": 1, "maximum": 5},
            "error_tags": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"}
        },
        "required": ["adequacy","fluency","terminology","overall","error_tags","rationale"]
    }

    srcs = df[SRC_COL].apply(safe_str).tolist()
    refs = df[REF_COL].apply(safe_str).tolist()

    for sys in SYSTEMS:
        print(f"[INFO] Judging system: {sys}")
        hyps = df[sys].apply(safe_str).tolist()

        for i in tqdm(range(len(df)), desc=f"LLM-judge {sys}"):
            # skip if already filled (unless overwrite)
            if (not OVERWRITE) and (not pd.isna(df.at[i, f"llm_overall__{sys}"])):
                continue

            src = srcs[i]; ref = refs[i]; hyp = hyps[i]
            if not src or not ref or not hyp:
                continue

            user_input = (
                f"SOURCE (EN): {src}\n"
                f"REFERENCE (FJ): {ref}\n"
                f"SYSTEM (FJ): {hyp}\n"
            )

            try:
                resp = client.responses.create(
                    model=MODEL,
                    input=[
                        {"role": "system", "content": JUDGE_PROMPT},
                        {"role": "user", "content": user_input},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "mt_judgement",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                )

                # The SDK provides a helper in many versions; safest is to parse output_text
                raw = resp.output_text
                data = json.loads(raw)

                df.at[i, f"llm_adequacy__{sys}"] = int(data["adequacy"])
                df.at[i, f"llm_fluency__{sys}"] = int(data["fluency"])
                df.at[i, f"llm_terminology__{sys}"] = int(data["terminology"])
                df.at[i, f"llm_overall__{sys}"] = int(data["overall"])
                df.at[i, f"llm_error_tags__{sys}"] = "|".join(data["error_tags"])
                df.at[i, f"llm_rationale__{sys}"] = data["rationale"]

            except Exception as e:
                df.at[i, f"llm_rationale__{sys}"] = f"JUDGE_ERROR: {type(e).__name__}: {e}"

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
