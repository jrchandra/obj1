import os
import re
import time
import json
import hashlib
from typing import Dict, Tuple, Optional

import pandas as pd
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# --------- OpenAI (ChatGPT) ----------
from openai import OpenAI

# --------- Gemini ----------
from google import genai

# --------- Google Cloud Translate v3 ----------
from google.cloud import translate_v3


load_dotenv()


def normalize_direction(direction: str) -> Optional[Tuple[str, str]]:
    """
    Accepts: "en->fj", "fj->en", "en-fj", "EN to FJ", etc.
    Returns: (src_lang, tgt_lang) like ("en","fj") or None if cannot parse.
    """
    if not isinstance(direction, str):
        return None
    d = direction.strip().lower()

    # common variants
    d = d.replace("→", "->").replace("to", "->")
    d = re.sub(r"\s+", "", d)

    # en->fj, fj->en
    m = re.match(r"^([a-z]{2,3})->([a-z]{2,3})$", d)
    if not m:
        # en-fj
        m = re.match(r"^([a-z]{2,3})-([a-z]{2,3})$", d)
    if not m:
        return None

    return m.group(1), m.group(2)


def stable_key(text: str, src: str, tgt: str) -> str:
    raw = f"{src}|{tgt}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


# ------------------- TRANSLATORS -------------------

class ChatGPTTranslator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def translate(self, text: str, src: str, tgt: str) -> str:
        # Keep it strict: "translation only"
        prompt = (
            f"Translate from {src} to {tgt}.\n"
            f"Return ONLY the translation text, no commentary.\n\n"
            f"Text:\n{text}"
        )
        # Responses API (recommended in docs)
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return (resp.output_text or "").strip()


class GeminiTranslator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def translate(self, text: str, src: str, tgt: str) -> str:
        prompt = (
            f"Translate from {src} to {tgt}.\n"
            f"Return ONLY the translation text, no commentary.\n\n"
            f"Text:\n{text}"
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return (resp.text or "").strip()


class GoogleNMTTranslator:
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        if not self.project_id:
            raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT_ID")
        self.client = translate_v3.TranslationServiceClient()
        self.parent = f"projects/{self.project_id}/locations/global"

    def translate(self, text: str, src: str, tgt: str) -> str:
        request = translate_v3.TranslateTextRequest(
            parent=self.parent,
            contents=[text],
            mime_type="text/plain",
            source_language_code=src,
            target_language_code=tgt,
        )
        resp = self.client.translate_text(request=request)
        if not resp.translations:
            return ""
        return (resp.translations[0].translated_text or "").strip()


class MicrosoftTranslator:
    def __init__(self):
        self.key = os.getenv("AZURE_TRANSLATOR_KEY")
        self.region = os.getenv("AZURE_TRANSLATOR_REGION")
        self.endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
        if not self.key or not self.region:
            raise RuntimeError("Missing AZURE_TRANSLATOR_KEY or AZURE_TRANSLATOR_REGION")

    def translate(self, text: str, src: str, tgt: str) -> str:
        url = f"{self.endpoint}/translate"
        params = {
            "api-version": "3.0",
            "from": src,
            "to": tgt
        }
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Ocp-Apim-Subscription-Region": self.region,
            "Content-type": "application/json"
        }
        body = [{"text": text}]
        r = requests.post(url, params=params, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        # expected shape: [{"translations":[{"text":"...","to":"fj"}]}]
        return (data[0]["translations"][0]["text"] or "").strip()


# ------------------- MAIN PIPELINE -------------------

def populate_translations(
    input_csv: str,
    output_csv: str,
    direction_col: str = "direction",
    source_col: str = "source_text",
    overwrite: bool = False,
    sleep_s: float = 0.0,
):
    df = pd.read_csv(input_csv)

    required = [direction_col, source_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Ensure output columns exist
    out_cols = ["chatgpt", "gemini", "google_translate", "microsoft_translate"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = ""

    # Init translators
    chatgpt = ChatGPTTranslator()
    gemini = GeminiTranslator()
    google_nmt = GoogleNMTTranslator()
    msft = MicrosoftTranslator()

    # Simple in-run cache to avoid re-translating duplicates
    cache: Dict[str, Dict[str, str]] = {}

    for i in tqdm(range(len(df)), desc="Translating"):
        direction = df.at[i, direction_col]
        text = df.at[i, source_col]

        if not isinstance(text, str) or not text.strip():
            continue

        pair = normalize_direction(str(direction))
        if not pair:
            # cannot parse direction; skip row
            continue

        src, tgt = pair
        key = stable_key(text, src, tgt)

        if key not in cache:
            cache[key] = {}

        # Decide per-system whether to translate
        def need(colname: str) -> bool:
            if overwrite:
                return True
            current = df.at[i, colname]
            return not (isinstance(current, str) and current.strip())

        try:
            if need("chatgpt"):
                if "chatgpt" not in cache[key]:
                    cache[key]["chatgpt"] = chatgpt.translate(text, src, tgt)
                df.at[i, "chatgpt"] = cache[key]["chatgpt"]

            if need("gemini"):
                if "gemini" not in cache[key]:
                    cache[key]["gemini"] = gemini.translate(text, src, tgt)
                df.at[i, "gemini"] = cache[key]["gemini"]

            if need("google_translate"):
                if "google_translate" not in cache[key]:
                    cache[key]["google_translate"] = google_nmt.translate(text, src, tgt)
                df.at[i, "google_translate"] = cache[key]["google_translate"]

            if need("microsoft_translate"):
                if "microsoft_translate" not in cache[key]:
                    cache[key]["microsoft_translate"] = msft.translate(text, src, tgt)
                df.at[i, "microsoft_translate"] = cache[key]["microsoft_translate"]

        except Exception as e:
            # mark error but keep going
            df.at[i, "notes"] = (str(df.at[i, "notes"]) if "notes" in df.columns else "") + f" | TRANSLATION_ERROR: {e}"

        if sleep_s > 0:
            time.sleep(sleep_s)

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    # Example usage:
    # python populate_translations.py
    populate_translations(
        input_csv="dataset.csv",
        output_csv="dataset_with_system_outputs.csv",
        direction_col="direction",
        source_col="source_text",
        overwrite=False,
        sleep_s=0.1,  # small pause to reduce rate-limit risk
    )
