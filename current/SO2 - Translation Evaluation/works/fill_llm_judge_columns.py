import os
import json
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

SYSTEMS_DEFAULT = ["chatgpt", "gemini", "google_translate", "microsoft_translate"]

JUDGE_SCHEMA = {
  "name": "mt_quality_judgment",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "adequacy": {"type": "number", "minimum": 0, "maximum": 100},
      "fluency":  {"type": "number", "minimum": 0, "maximum": 100},
      "overall":  {"type": "number", "minimum": 0, "maximum": 100},
      "error_tags": {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 10
      },
      "rationale": {"type": "string", "maxLength": 280}
    },
    "required": ["adequacy", "fluency", "overall", "error_tags", "rationale"]
  },
  "strict": True
}

SYSTEM_PROMPT = """You are an MT evaluation judge.
Score candidate translation vs reference for meaning and fluency.
Return ONLY the JSON that matches the provided schema.
Be consistent and conservative.
"""

def norm(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()

def call_judge(client: OpenAI, model: str, source: str, reference: str, candidate: str, direction: str):
    user_prompt = f"""Direction: {direction}

SOURCE:
{source}

REFERENCE:
{reference}

CANDIDATE:
{candidate}

Evaluate CANDIDATE against REFERENCE (and SOURCE for adequacy).
"""
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
    )
    # Responses API returns structured output as text in output[0].content[0].text
    txt = resp.output[0].content[0].text
    return json.loads(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")  # cheap+good judge; change if you want
    ap.add_argument("--ref_col", default="ref")
    ap.add_argument("--src_col", default="source_text")
    ap.add_argument("--direction_col", default="direction")
    ap.add_argument("--systems", default=",".join(SYSTEMS_DEFAULT))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.2, help="Small delay to reduce rate-limit risk")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment first.")

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    df = pd.read_csv(args.in_csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.in_csv}")

    # Create columns per system
    for syscol in systems:
        for metric in ["llm_adequacy", "llm_fluency", "llm_overall", "llm_error_tags", "llm_rationale"]:
            col = f"{metric}__{syscol}"
            if col not in df.columns:
                df[col] = np.nan

    client = OpenAI()

    for syscol in systems:
        cols = {
            "a": f"llm_adequacy__{syscol}",
            "f": f"llm_fluency__{syscol}",
            "o": f"llm_overall__{syscol}",
            "t": f"llm_error_tags__{syscol}",
            "r": f"llm_rationale__{syscol}",
        }

        if args.overwrite:
            need = np.ones(len(df), dtype=bool)
        else:
            need = df[cols["o"]].isna().to_numpy()

        idxs = np.where(need)[0]
        if len(idxs) == 0:
            print(f"[OK] Judge columns already filled for {syscol}. Skipping.")
            continue

        print(f"[INFO] Judging {len(idxs)} rows for {syscol} using {args.model}...")

        for i in tqdm(idxs, desc=f"LLM-judge for {syscol}"):
            source = norm(df.at[i, args.src_col])
            reference = norm(df.at[i, args.ref_col])
            candidate = norm(df.at[i, syscol])
            direction = norm(df.at[i, args.direction_col])

            if not reference or not candidate:
                continue

            # retry loop (simple)
            for attempt in range(1, 4):
                try:
                    j = call_judge(client, args.model, source, reference, candidate, direction)
                    df.at[i, cols["a"]] = float(j["adequacy"])
                    df.at[i, cols["f"]] = float(j["fluency"])
                    df.at[i, cols["o"]] = float(j["overall"])
                    df.at[i, cols["t"]] = json.dumps(j["error_tags"], ensure_ascii=False)
                    df.at[i, cols["r"]] = j["rationale"]
                    break
                except Exception as e:
                    if attempt == 3:
                        df.at[i, cols["r"]] = f"JUDGE_ERROR: {type(e).__name__}: {e}"
                    else:
                        time.sleep(1.5 * attempt)

            time.sleep(args.sleep)

        print(f"[OK] Filled LLM-judge columns for {syscol}")

    df.to_csv(args.out_csv, index=False)
    print(f"[DONE] Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()
