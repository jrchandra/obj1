import pandas as pd
from pathlib import Path
import re

MASTER_COLS = [
    "domain","subdomain","source_id","source_doc",
    "source_lang","target_lang","direction",
    "source_text","target_text"
]

def clean(s):
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def read_lines(p):
    return [clean(x) for x in p.read_text(encoding="utf-8").splitlines() if len(x.strip()) >= 8]

def build_pair(en_txt, fj_txt, subdomain):
    en = read_lines(en_txt)
    fj = read_lines(fj_txt)
    n = min(len(en), len(fj))

    rows = []
    for i in range(n):
        rows.append({
            "id": f"{subdomain}|{i+1:05d}",
            "en": en[i],
            "fj": fj[i]
        })

    df = pd.DataFrame(rows)

    en2fj = pd.DataFrame({
        "domain": "medical",
        "subdomain": subdomain,
        "source_id": df["id"],
        "source_doc": en_txt.name,
        "source_lang": "en",
        "target_lang": "fj",
        "direction": "en->fj",
        "source_text": df["en"],
        "target_text": df["fj"],
    })[MASTER_COLS]

    fj2en = pd.DataFrame({
        "domain": "medical",
        "subdomain": subdomain,
        "source_id": df["id"],
        "source_doc": fj_txt.name,
        "source_lang": "fj",
        "target_lang": "en",
        "direction": "fj->en",
        "source_text": df["fj"],
        "target_text": df["en"],
    })[MASTER_COLS]

    return en2fj, fj2en

def main():
    folder = Path(__file__).resolve().parent

    pairs = [
        ("Dementia-and-support-english.txt", "Dementia-and-support-Fiji.txt", "dementia_support"),
        ("Screening-Clinic-A2-Poster-English.txt", "Screening-Clinic-A2-Poster-iTaukei.txt", "covid_screening"),
    ]

    all_en2fj = []
    all_fj2en = []

    for en_name, fj_name, sub in pairs:
        en_p = folder / en_name
        fj_p = folder / fj_name
        if not en_p.exists() or not fj_p.exists():
            print(f"SKIP missing {sub}")
            continue

        en2fj, fj2en = build_pair(en_p, fj_p, sub)
        all_en2fj.append(en2fj)
        all_fj2en.append(fj2en)

    df_en2fj = pd.concat(all_en2fj, ignore_index=True)
    df_fj2en = pd.concat(all_fj2en, ignore_index=True)
    df_both = pd.concat([df_en2fj, df_fj2en], ignore_index=True)

    df_en2fj.to_csv("medical_parallel_en2fj__OCR.csv", index=False)
    df_fj2en.to_csv("medical_parallel_fj2en__OCR.csv", index=False)
    df_both.to_csv("medical_parallel_both__OCR.csv", index=False)

    print("DONE")

if __name__ == "__main__":
    main()
