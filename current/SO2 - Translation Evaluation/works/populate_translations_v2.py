import os
import re
import time
import math
import random
import hashlib
from typing import Dict, Tuple, Optional, Callable

import pandas as pd
import requests
from tqdm import tqdm
from dotenv import load_dotenv

from openai import OpenAI
from google import genai
from google.cloud import translate_v3

load_dotenv()

# ---------------- Utilities ----------------

def normalize_direction(direction: str) -> Optional[Tuple[str, str]]:
    if not isinstance(direction, str):
        return None
    d = direction.strip().lower().replace("→", "->")
    d = d.replace(" to ", "->").replace("to", "->")
    d = re.sub(r"\s+", "", d)

    m = re.match(r"^([a-z]{2,3})->([a-z]{2,3})$", d) or re.match(r"^([a-z]{2,3})-([a-z]{2,3})$", d)
    if not m:
        return None
    return m.group(1), m.group(2)

def stable_key(text: str, src: str, tgt: str) -> str:
    raw = f"{src}|{tgt}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()

def retry(
    fn: Callable[[], str],
    label: str,
    max_retries: int = 8,
    base_sleep: float = 1.0,
    max_sleep: float = 60.0,
):
    """
    Exponential backoff with jitter.
    Retries for typical rate-limit / transient failures.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            is_rate = ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("rate" in msg.lower())
            is_transient = any(x in msg for x in ["502", "503", "504", "timeout", "temporarily", "Unavailable"])

            if attempt >= max_retries or not (is_rate or is_transient):
                raise

            sleep = min(max_sleep, base_sleep * (2 ** attempt))
            sleep = sleep * (0.7 + random.random() * 0.6)  # jitter 0.7x..1.3x
            print(f"[{label}] retry {attempt+1}/{max_retries} after {sleep:.1f}s due to: {msg[:160]}")
            time.sleep(sleep)

# ---------------- Translators ----------------

class ChatGPTTranslator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def translate(self, text: str, src: str, tgt: str) -> str:
        prompt = (
            f"Translate from {src} to {tgt}.\n"
            f"Return ONLY the translation text, no commentary.\n\n"
            f"Text:\n{text}"
        )
        resp = self.client.responses.create(model=self.model, input=prompt)
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
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        return (resp.text or "").strip()


class GoogleNMTTranslator:
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        if not self.project_id:
            raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT_ID")

        # Validate project id shape early (your error suggests invalid characters)
        if not re.match(r"^[a-z0-9.\-:]+$", self.project_id):
            raise RuntimeError(
                f"GOOGLE_CLOUD_PROJECT_ID looks invalid: '{self.project_id}'. "
                "Use Google Cloud Console 'Project ID' (usually lowercase with hyphens)."
            )

        self.client = translate_v3.TranslationServiceClient()
        self.parent = f"projects/{self.project_id}/locations/global"

    def translate(self, text: str, src: str, tgt: str) -> str:
        req = translate_v3.TranslateTextRequest(
            parent=self.parent,
            contents=[text],
            mime_type="text/plain",
            source_language_code=src,
            target_language_code=tgt,
        )
        resp = self.client.translate_text(request=req)
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
        params = {"api-version": "3.0", "from": src, "to": tgt}
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Ocp-Apim-Subscription-Region": self.region,
            "Content-type": "application/json"
        }
        body = [{"text": text}]
        r = requests.post(url, params=params, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data[0]["translations"][0]["text"] or "").strip()

# ---------------- Main ----------------

def populate_translations(
    input_csv: str,
    output_csv: str,
    direction_col: str = "direction",
    source_col: str = "source_text",
    overwrite: bool = False,
    save_every: int = 25,
    throttle_s: float = 0.2,
):
    df = pd.read_csv(input_csv)

    for c in [direction_col, source_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    out_cols = ["chatgpt", "gemini", "google_translate", "microsoft_translate"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = ""

    if "notes" not in df.columns:
        df["notes"] = ""

    chatgpt = ChatGPTTranslator()
    gemini = GeminiTranslator()
    google_nmt = GoogleNMTTranslator()
    msft = MicrosoftTranslator()

    cache: Dict[str, Dict[str, str]] = {}

    def need(i: int, col: str) -> bool:
        if overwrite:
            return True
        v = df.at[i, col]
        return not (isinstance(v, str) and v.strip())

    for i in tqdm(range(len(df)), desc="Translating"):
        text = df.at[i, source_col]
        if not isinstance(text, str) or not text.strip():
            continue

        pair = normalize_direction(str(df.at[i, direction_col]))
        if not pair:
            df.at[i, "notes"] = (str(df.at[i, "notes"]) or "") + " | BAD_DIRECTION"
            continue

        src, tgt = pair
        key = stable_key(text, src, tgt)
        cache.setdefault(key, {})

        try:
            if need(i, "chatgpt"):
                if "chatgpt" not in cache[key]:
                    cache[key]["chatgpt"] = retry(lambda: chatgpt.translate(text, src, tgt), "openai")
                df.at[i, "chatgpt"] = cache[key]["chatgpt"]

            if need(i, "gemini"):
                if "gemini" not in cache[key]:
                    cache[key]["gemini"] = retry(lambda: gemini.translate(text, src, tgt), "gemini")
                df.at[i, "gemini"] = cache[key]["gemini"]

            if need(i, "google_translate"):
                if "google_translate" not in cache[key]:
                    cache[key]["google_translate"] = retry(lambda: google_nmt.translate(text, src, tgt), "gcp_translate")
                df.at[i, "google_translate"] = cache[key]["google_translate"]

            if need(i, "microsoft_translate"):
                if "microsoft_translate" not in cache[key]:
                    cache[key]["microsoft_translate"] = retry(lambda: msft.translate(text, src, tgt), "ms_translate")
                df.at[i, "microsoft_translate"] = cache[key]["microsoft_translate"]

        except Exception as e:
            df.at[i, "notes"] = (str(df.at[i, "notes"]) or "") + f" | TRANSLATION_ERROR: {e}"

        if throttle_s > 0:
            time.sleep(throttle_s)

        if save_every and (i + 1) % save_every == 0:
            df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    populate_translations(
        input_csv="dataset.csv",
        output_csv="dataset_with_system_outputs.csv",
        overwrite=False,
        save_every=25,
        throttle_s=0.25,
    )
