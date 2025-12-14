#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Medical Parallel Corpus from paired EN/iTaukei PDFs (v3 - poster-friendly).

Key improvements:
- Uses pdfplumber.extract_words() with x/y coordinates (much better for posters).
- Rebuilds lines in reading order, then groups into blocks.
- Aligns by position (index) first (works when translations are not lexically similar),
  optionally refines with fuzzy matching.

Install:
  pip install pandas pdfplumber rapidfuzz
Run:
  python build_medical_pdf_corpus_v3.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Dict
import unicodedata as ud

import pandas as pd
import pdfplumber
from rapidfuzz import fuzz

MASTER_COLS = [
    "domain","subdomain","source_id","source_doc",
    "source_lang","target_lang","direction","source_text","target_text"
]

# -----------------------
# Cleaning (your preferences)
# -----------------------
MOJIBAKE_MAP = {
    "â€œ": "“",
    "â€": "”",
    "â€˜": "‘",
    "â€™": "’",
    "â€“": "–",
    "â€”": "—",
    "â€¦": "…",
    "Â": "",
}
WS_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\d+")

def fix_mojibake(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\ufeff", "")
    for k, v in MOJIBAKE_MAP.items():
        s = s.replace(k, v)
    return ud.normalize("NFKC", s)

def normalize_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()

def clean_text_keep_punct_lower_no_digits(s: str) -> str:
    s = fix_mojibake(s)
    s = normalize_ws(s)
    s = DIGIT_RE.sub("", s)   # exclude numbers
    s = normalize_ws(s)
    s = s.lower()
    s = normalize_ws(s)
    return s

def is_junk_segment(s: str) -> bool:
    if not s:
        return True
    if len(s) < 8:
        return True
    # very low alphabetic ratio -> often artifacts
    letters = sum(ch.isalpha() for ch in s)
    if len(s) > 25 and (letters / max(1, len(s)) < 0.15):
        return True
    return False


# -----------------------
# Poster-friendly extraction: words -> lines -> blocks
# -----------------------
def words_to_lines(words: List[dict], y_tol: float = 3.0) -> List[str]:
    """
    Group words into lines by their 'top' y coordinate.
    """
    if not words:
        return []

    # sort top-to-bottom, then left-to-right
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))

    lines = []
    cur = []
    cur_y = None

    for w in words:
        txt = (w.get("text") or "").strip()
        if not txt:
            continue
        y = float(w["top"])
        if cur_y is None:
            cur_y = y
            cur = [w]
            continue

        if abs(y - cur_y) <= y_tol:
            cur.append(w)
        else:
            # flush line
            cur = sorted(cur, key=lambda ww: ww["x0"])
            line = " ".join((ww.get("text") or "").strip() for ww in cur).strip()
            line = normalize_ws(fix_mojibake(line))
            if line:
                lines.append(line)
            # start new line
            cur_y = y
            cur = [w]

    # last
    if cur:
        cur = sorted(cur, key=lambda ww: ww["x0"])
        line = " ".join((ww.get("text") or "").strip() for ww in cur).strip()
        line = normalize_ws(fix_mojibake(line))
        if line:
            lines.append(line)

    # remove obvious duplicates caused by layered text
    deduped = []
    seen = set()
    for ln in lines:
        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ln)
    return deduped


def lines_to_blocks(lines: List[str], max_block_chars: int = 220) -> List[str]:
    """
    Merge consecutive short lines into blocks until size threshold.
    Helps when posters break sentences into many lines.
    """
    blocks = []
    buf = ""

    for ln in lines:
        ln = normalize_ws(ln)
        if not ln:
            continue

        # if looks like a heading, flush buffer first
        if ln.isupper() and len(ln) < 80:
            if buf:
                blocks.append(buf.strip())
                buf = ""
            blocks.append(ln)
            continue

        if not buf:
            buf = ln
        elif len(buf) + 1 + len(ln) <= max_block_chars:
            buf = f"{buf} {ln}"
        else:
            blocks.append(buf.strip())
            buf = ln

    if buf:
        blocks.append(buf.strip())

    # final cleaning + filtering
    cleaned = []
    for b in blocks:
        c = clean_text_keep_punct_lower_no_digits(b)
        if is_junk_segment(c):
            continue
        cleaned.append(c)

    return cleaned


def extract_pdf_segments_v3(pdf_path: Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Returns (segments, stats)
    stats helps you debug extraction.
    """
    total_words = 0
    total_lines = 0
    total_blocks = 0

    all_blocks: List[str] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            # Method A: word extraction (best for posters)
            words = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=True,          # important
                extra_attrs=["fontname", "size"]
            ) or []
            total_words += len(words)

            lines = words_to_lines(words, y_tol=3.0)
            total_lines += len(lines)

            blocks = lines_to_blocks(lines, max_block_chars=220)
            total_blocks += len(blocks)

            all_blocks.extend(blocks)

            # Method B fallback if word extraction yields nothing
            if len(words) == 0:
                txt = page.extract_text() or ""
                txt = fix_mojibake(txt)
                raw_lines = [normalize_ws(x) for x in txt.splitlines() if normalize_ws(x)]
                if raw_lines:
                    all_blocks.extend(lines_to_blocks(raw_lines, max_block_chars=220))

    # dedupe blocks
    out = []
    seen = set()
    for s in all_blocks:
        k = s.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s)

    stats = {
        "pages": 0,  # filled below
        "words": total_words,
        "lines": total_lines,
        "blocks": total_blocks,
        "unique_blocks": len(out),
    }
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            stats["pages"] = len(pdf.pages)
    except Exception:
        pass

    return out, stats


# -----------------------
# Alignment that actually works for translations:
# position-first, optional fuzzy refinement
# -----------------------
def align_by_position(en_segs: List[str], fj_segs: List[str]) -> List[Tuple[str, str, int]]:
    """
    Pair segments by index up to min length.
    Score is a weak indicator (not lexical similarity across languages),
    so we compute a *length-shape score* just for logging.
    """
    m = min(len(en_segs), len(fj_segs))
    aligned = []
    for i in range(m):
        en = en_segs[i]
        fj = fj_segs[i]
        # cheap "shape" score: closer lengths => higher
        le, lf = len(en), len(fj)
        score = int(100 * (1 - (abs(le - lf) / max(le, lf, 1))))
        aligned.append((en, fj, score))
    return aligned

def refine_with_fuzzy_window(en_segs: List[str], fj_segs: List[str], window: int = 6) -> List[Tuple[str, str, int]]:
    """
    Optional refinement:
    For each EN seg i, find best FJ seg within +/- window by token_set_ratio
    on lightly normalized strings (still not great cross-lingually, but helps when
    there are shared words like names, numbers removed, etc.).
    """
    aligned = []
    used = set()

    def norm(s: str) -> str:
        # keep punctuation but simplify for fuzz
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    fj_norm = [norm(x) for x in fj_segs]

    for i, en in enumerate(en_segs):
        lo = max(0, i - window)
        hi = min(len(fj_segs), i + window + 1)

        best = (-1, None)
        en_n = norm(en)

        for j in range(lo, hi):
            if j in used:
                continue
            sc = fuzz.token_set_ratio(en_n, fj_norm[j])
            if sc > best[0]:
                best = (sc, j)

        if best[1] is not None:
            used.add(best[1])
            aligned.append((en, fj_segs[best[1]], int(best[0])))

    return aligned


def build_parallel_rows_safe(
    aligned: List[Tuple[str, str, int]],
    subdomain: str,
    source_doc_en: str,
    source_doc_fj: str,
    base_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not aligned:
        empty = pd.DataFrame(columns=MASTER_COLS)
        return empty.copy(), empty.copy(), empty.copy()

    rows = []
    for i, (en, fj, score) in enumerate(aligned, start=1):
        sid = f"{base_id}|{i:05d}|s{score}"
        rows.append({"source_id": sid, "en": en, "fj": fj})

    df = pd.DataFrame(rows)

    en2fj = pd.DataFrame({
        "domain": "medical",
        "subdomain": subdomain,
        "source_id": df["source_id"],
        "source_doc": source_doc_en,
        "source_lang": "en",
        "target_lang": "fj",
        "direction": "en->fj",
        "source_text": df["en"],
        "target_text": df["fj"],
    })[MASTER_COLS]

    fj2en = pd.DataFrame({
        "domain": "medical",
        "subdomain": subdomain,
        "source_id": df["source_id"],
        "source_doc": source_doc_fj,
        "source_lang": "fj",
        "target_lang": "en",
        "direction": "fj->en",
        "source_text": df["fj"],
        "target_text": df["en"],
    })[MASTER_COLS]

    both = pd.concat([en2fj, fj2en], ignore_index=True)
    return en2fj, fj2en, both


def main():
    folder = Path(__file__).resolve().parent

    pairs = [
        ("Dementia-and-support-english.pdf", "Dementia-and-support-Fiji.pdf", "dementia_support", "medical|dementia_support"),
        ("Screening-Clinic-A2-Poster-English.pdf", "Screening-Clinic-A2-Poster-iTaukei.pdf", "covid_screening_clinic", "medical|covid_screening_clinic"),
    ]

    summary = []
    all_en2fj = []
    all_fj2en = []
    all_both = []

    for en_name, fj_name, subdomain, base_id in pairs:
        en_path = folder / en_name
        fj_path = folder / fj_name

        if not en_path.exists() or not fj_path.exists():
            summary.append(f"[SKIP] missing pair: {en_name} <-> {fj_name}")
            continue

        en_segs, en_stats = extract_pdf_segments_v3(en_path)
        fj_segs, fj_stats = extract_pdf_segments_v3(fj_path)

        summary.append(f"[PAIR] {subdomain}")
        summary.append(f"  EN: pages={en_stats['pages']} words={en_stats['words']} lines={en_stats['lines']} unique_blocks={en_stats['unique_blocks']}")
        summary.append(f"  FJ: pages={fj_stats['pages']} words={fj_stats['words']} lines={fj_stats['lines']} unique_blocks={fj_stats['unique_blocks']}")

        # If almost nothing extracted, log clearly
        if len(en_segs) < 3 or len(fj_segs) < 3:
            summary.append("  WARN: very low extraction — PDF may be image-based; consider OCR source PDFs.")
            aligned = []
        else:
            # primary: position alignment (works best for translations)
            aligned_pos = align_by_position(en_segs, fj_segs)

            # optional refinement (choose better of pos vs refined by count + avg score)
            aligned_fuzzy = refine_with_fuzzy_window(en_segs, fj_segs, window=6)

            def avg(a): return sum(x[2] for x in a) / max(1, len(a))
            aligned = aligned_pos
            if len(aligned_fuzzy) > len(aligned_pos) or (len(aligned_fuzzy) == len(aligned_pos) and avg(aligned_fuzzy) > avg(aligned_pos)):
                aligned = aligned_fuzzy
                summary.append(f"  ALIGN: used fuzzy-window (pairs={len(aligned):,}, avg_score={avg(aligned):.1f})")
            else:
                summary.append(f"  ALIGN: used position (pairs={len(aligned):,}, avg_score={avg(aligned):.1f})")

        en2fj, fj2en, both = build_parallel_rows_safe(
            aligned=aligned,
            subdomain=subdomain,
            source_doc_en=en_path.name,
            source_doc_fj=fj_path.name,
            base_id=base_id
        )
        all_en2fj.append(en2fj)
        all_fj2en.append(fj2en)
        all_both.append(both)

    # Always write outputs
    df_en2fj = pd.concat(all_en2fj, ignore_index=True) if all_en2fj else pd.DataFrame(columns=MASTER_COLS)
    df_fj2en = pd.concat(all_fj2en, ignore_index=True) if all_fj2en else pd.DataFrame(columns=MASTER_COLS)
    df_both  = pd.concat([df_en2fj, df_fj2en], ignore_index=True)

    out_en2fj = folder / "medical_parallel_en2fj__ALL.csv"
    out_fj2en = folder / "medical_parallel_fj2en__ALL.csv"
    out_both  = folder / "medical_parallel_both__ALL.csv"
    out_sum   = folder / "medical_build_summary.txt"

    df_en2fj.to_csv(out_en2fj, index=False, encoding="utf-8")
    df_fj2en.to_csv(out_fj2en, index=False, encoding="utf-8")
    df_both.to_csv(out_both, index=False, encoding="utf-8")
    out_sum.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("DONE")
    print(f"wrote: {out_en2fj.name} rows={len(df_en2fj):,}")
    print(f"wrote: {out_fj2en.name} rows={len(df_fj2en):,}")
    print(f"wrote: {out_both.name}  rows={len(df_both):,}")
    print(f"wrote: {out_sum.name}")

if __name__ == "__main__":
    main()
