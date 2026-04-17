#!/usr/bin/env python3
"""
build_dictionary_corpus.py

Creates a clean, merge-friendly parallel corpus from the provided Fijian→English dictionary TXT files.

Outputs (in the same folder you run this from):
  - dictionary_parallel_fj2en__ALL.csv
  - dictionary_parallel_en2fj__ALL.csv
  - dictionary_parallel_both__ALL.csv

Default behavior:
  - lowercases both sides (for easier computation)
  - keeps punctuation
  - removes digits (page numbers, numbering) from text fields
  - preserves original lexical forms as much as possible (no stemming/lemmatization)

Usage:
  python build_dictionary_corpus.py
  python build_dictionary_corpus.py --inputs Fijian-English_Dictionary.txt Fijian_English_Dictionary_THE_LATE_DAVID.txt
  python build_dictionary_corpus.py --no-lowercase
  python build_dictionary_corpus.py --keep-numbers
"""

from __future__ import annotations
import argparse
import csv
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict

STOP_HEADS = {
    "the","and","or","in","see","to","a","an","of","for","with","on","at","by","from","this","that","these","those",
    "pronounced","example","examples","page","chapter","lesson","activity","list","match","write","draw","label","correct",
    "postfixed","prefixed","suffix","prefix","possessive","pronoun","pronouns","subject","object","termination","terminations","irreg","irregular"
}

FIELDNAMES = ["domain","sentence_type","direction","source_text","target_text","alignment_method","quality_flags","notes","source_doc"]

def read_text_robust(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1", errors="replace")

def fix_mojibake(s: str) -> str:
    repl = {
        "â€œ": '"', "â€": '"', "â€˜": "'", "â€™": "'", "â€\"": "-", "â€“": "-", "â€”": "-",
        "\u00a0": " ",
    }
    for a,b in repl.items():
        s = s.replace(a,b)
    return unicodedata.normalize("NFKC", s)

def remove_page_markers(text: str) -> str:
    return re.sub(r"-{10,}\s*Page\s*\d+.*?\n", "\n", text)

def dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

def strip_page_artifacts(lines: List[str]) -> List[str]:
    out=[]
    for ln in lines:
        if re.search(r"-{10,}\s*Page\s*\d+", ln):
            continue
        if re.fullmatch(r"\s*\d+\s*", ln):  # bare page number
            continue
        if re.search(r"FIJIAN\s+[-–]\s+ENGLISH\s+DICTIONARY", ln, re.I):
            continue
        out.append(ln)
    return out

def clean_field(text: str, lowercase: bool, drop_numbers: bool) -> str:
    text = fix_mojibake(text)
    text = text.replace("\u200b","")
    text = re.sub(r"\s+", " ", text).strip()
    if drop_numbers:
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text

def extract_gloss(entry_text: str, head: str) -> str:
    t = entry_text.strip()
    t = re.sub(r"^\*?\s*"+re.escape(head)+r"\s*,?\s*", "", t, flags=re.I)
    return t.strip()

def parse_gatty_basic(text: str) -> List[Tuple[str,str]]:
    """
    Heuristic parser for R. Gatty-style OCR text (Fijian–English Dictionary).
    Detects entry starts by: leading token(s) then either 2+ spaces, parenthesis, or POS marker.
    """
    text = fix_mojibake(remove_page_markers(text))
    text = dehyphenate(text)
    lines = text.replace("\r\n","\n").replace("\r","\n").split("\n")
    # start at the "A" section if present
    start_idx=0
    for i,ln in enumerate(lines):
        if re.fullmatch(r"\s*A\s*", ln):
            start_idx=i
            break
    lines = strip_page_artifacts(lines[start_idx:])
    lines = [ln.rstrip() for ln in lines]
    start_re = re.compile(r"^\s*([A-Za-z][A-Za-z'’\- ]{0,60}?)(?:\s{2,}|\s+\(|\s+(?:n\.|v\.|adj\.|adv\.|prep\.|pron\.|interj\.|conj\.)\b)", re.I)

    entries=[]
    cur=[]
    cur_head=None
    for ln in lines:
        if not ln.strip():
            if cur:
                cur.append("")
            continue
        if re.fullmatch(r"\s*[A-Z]\s*", ln.strip()):
            continue
        m=start_re.match(ln)
        if m and not ln.lstrip().lower().startswith(("idiom:", "syn.", "see ")):
            head=m.group(1).strip()
            if cur_head and cur:
                entries.append((cur_head, " ".join(cur).strip()))
            cur_head=head
            cur=[ln.strip()]
        else:
            if cur_head:
                cur.append(ln.strip())
    if cur_head and cur:
        entries.append((cur_head, " ".join(cur).strip()))
    return entries

def parse_david_v2(text: str) -> List[Tuple[str,str]]:
    """
    Heuristic parser for older scanned dictionary text (often has asterisks, lots of blank lines).
    Treats a line as an entry start if it begins with optional '*' + headword and contains POS marker
    or classic forms like '-na' / 'v. intr.' / 'v. tr.'.
    """
    text = fix_mojibake(remove_page_markers(text))
    text = dehyphenate(text)
    lines = text.replace("\r\n","\n").replace("\r","\n").split("\n")
    lines = [ln.strip() for ln in lines]

    head_re = re.compile(r"^\*?([A-Za-z][A-Za-z'\-]*)(?:,|\s)\s*(.*)$")
    pos_re = re.compile(r"\b(?:v\.|n\.|a\.|adj\.|adv\.|pron\.)\b", re.I)

    entries=[]
    cur_head=None
    cur=[]
    for ln in lines:
        if not ln:
            if cur_head:
                cur.append("")
            continue
        m=head_re.match(ln)
        is_head=False
        if m:
            head=m.group(1)
            if pos_re.search(ln) or re.search(r"-na\b", ln) or re.search(r"\bv\.\s*(?:intr|tr)\b", ln, re.I):
                is_head=True
        if is_head:
            if cur_head and cur:
                entries.append((cur_head, " ".join(cur).strip()))
            cur_head=m.group(1)
            cur=[ln]
        else:
            if cur_head:
                cur.append(ln)
    if cur_head and cur:
        entries.append((cur_head, " ".join(cur).strip()))
    return entries

def filter_entries(entries: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    out=[]
    for head,txt in entries:
        h=head.strip().strip(",")
        if re.fullmatch(r"[A-Z]", h):
            continue
        if h.lower() in STOP_HEADS:
            continue
        # drop obvious phonology note
        if h == "A" and "pronounced" in txt.lower():
            continue
        if h.lower() == "the" and "poss." in txt.lower():
            continue
        out.append((h,txt))
    return out

def make_rows(entries: List[Tuple[str,str]], source_doc: str, alignment_method: str,
              lowercase: bool, drop_numbers: bool) -> List[Dict[str,str]]:
    rows=[]
    for head, full in entries:
        gloss = extract_gloss(full, head)

        qflags=[]
        if re.search(r"\bIdiom:", full, re.I): qflags.append("has_idiom")
        if re.search(r"\bSyn\.", full): qflags.append("has_syn")
        if re.search(r"\bSee\b", full): qflags.append("has_see")
        if "?" in full: qflags.append("has_question")
        if re.search(r"\b(Eng\.|Anglicism)\b", full): qflags.append("has_anglicism")

        src = clean_field(head, lowercase=lowercase, drop_numbers=drop_numbers)
        tgt = clean_field(gloss, lowercase=lowercase, drop_numbers=drop_numbers)
        if not src or not tgt:
            continue

        rows.append({
            "domain":"dictionary",
            "sentence_type":"lexeme_gloss",
            "direction":"fj_to_en",
            "source_text": src,
            "target_text": tgt,
            "alignment_method": alignment_method,
            "quality_flags":";".join(qflags),
            "notes":"",
            "source_doc": source_doc
        })
    return rows

def invert_rows(rows: List[Dict[str,str]]) -> List[Dict[str,str]]:
    inv=[]
    for r in rows:
        inv.append({
            **r,
            "source_text": r["target_text"],
            "target_text": r["source_text"],
            "direction": "en_to_fj",
            "alignment_method": r["alignment_method"] + "+invert",
            "quality_flags": (r["quality_flags"] + (";" if r["quality_flags"] else "") + "inverted")
        })
    return inv

def write_csv(path: Path, rows: List[Dict[str,str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in FIELDNAMES})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=None, help="Input .txt files. If omitted, uses all *.txt in current folder.")
    ap.add_argument("--no-lowercase", action="store_true", help="Do not lowercase text.")
    ap.add_argument("--keep-numbers", action="store_true", help="Keep digits in text fields.")
    args = ap.parse_args()

    cwd = Path(".").resolve()
    if args.inputs:
        input_paths = [Path(p).resolve() for p in args.inputs]
    else:
        input_paths = sorted([p for p in cwd.glob("*.txt")])

    if not input_paths:
        raise SystemExit("No .txt files found. Put your dictionary .txt files in this folder or pass --inputs ...")

    lowercase = not args.no_lowercase
    drop_numbers = not args.keep_numbers

    all_rows: List[Dict[str,str]] = []

    for p in input_paths:
        raw = read_text_robust(p)
        # choose parser
        if re.search(r"R\s*\.?\s*GATTY|FIJIAN\s+[-–]\s+ENGLISH\s+DICTIONARY", raw, re.I):
            parsed = filter_entries(parse_gatty_basic(raw))
            rows = make_rows(parsed, p.name, "parse_gatty_basic", lowercase, drop_numbers)
        else:
            parsed = filter_entries(parse_david_v2(raw))
            rows = make_rows(parsed, p.name, "parse_david_v2", lowercase, drop_numbers)
        all_rows.extend(rows)

    fj2en = all_rows
    en2fj = invert_rows(fj2en)
    both = fj2en + en2fj

    write_csv(cwd/"dictionary_parallel_fj2en__ALL.csv", fj2en)
    write_csv(cwd/"dictionary_parallel_en2fj__ALL.csv", en2fj)
    write_csv(cwd/"dictionary_parallel_both__ALL.csv", both)

    print(f"Done. Rows: fj2en={len(fj2en):,}  en2fj={len(en2fj):,}  both={len(both):,}")
    print("Outputs written to current folder.")

if __name__ == "__main__":
    main()
