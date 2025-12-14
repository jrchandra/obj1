import os
import re
import unicodedata as ud
from typing import Dict, List, Tuple
import pandas as pd

# ---------------------------
# FILES (same folder as script)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EN_FILE = os.path.join(BASE_DIR, "genesis-en.txt")
FJ_FILE = os.path.join(BASE_DIR, "genesis-fj.txt")

# ---------------------------
# CLEANING RULES (your spec)
# - lowercase
# - keep punctuation
# - remove digits anywhere in text fields
# ---------------------------
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

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def fix_mojibake(s: str) -> str:
    if s is None:
        return ""
    for k, v in MOJIBAKE_MAP.items():
        s = s.replace(k, v)
    s = ud.normalize("NFKC", s)
    s = re.sub(r"\s+([”’])", r"\1", s)
    s = re.sub(r"([“‘])\s+", r"\1", s)
    return s

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def light_punct_spacing(s: str) -> str:
    s = re.sub(r"([\,\.\;\:\?\!])(?!\s|$)", r"\1 ", s)
    s = re.sub(r"([\,\.\;\:\?\!])\s{2,}", r"\1 ", s)
    s = re.sub(r"”(?=\w)", "” ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def fix_glues_en(s: str) -> str:
    t = s
    replacements = [
        (r"\binthe\b", "in the"),
        (r"\bandtheearth\b", "and the earth"),
        (r"\bonthesurface\b", "on the surface"),
        (r"\bofthedeep\b", "of the deep"),
        (r"\bofthewaters\b", "of the waters"),
        (r"\bandthedarkness\b", "and the darkness"),
        (r"\bandthere\b", "and there"),
        (r"\bwasmorning\b", "was morning"),
        (r"\bthefirst\b", "the first"),
        (r"\bletthere\b", "let there"),
        (r"\bbeanexpanse\b", "be an expanse"),
        (r"\bthemiddle\b", "the middle"),
        (r"\bletitdivide\b", "let it divide"),
        (r"\bthefishofthesea\b", "the fish of the sea"),
        (r"\bcreepingthing\b", "creeping thing"),
        (r"\boverevery\b", "over every"),
        (r"\bon theearth\b", "on the earth"),
        (r"\bgodcalled\b", "god called"),
        (r"\bgodsaid\b", "god said"),
        (r"\bgodsaw\b", "god saw"),
        (r"\bgodblessed\b", "god blessed"),
        (r"\bgodmade\b", "god made"),
    ]
    for pat, rep in replacements:
        t = re.sub(pat, rep, t)

    t = re.sub(r"\b(and|of|on|in|to|for|with|from|let)(the|there|it|be|an|a)\b", r"\1 \2", t)
    t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t)
    return t

def fix_glues_fj(s: str) -> str:
    t = s
    t = re.sub(r"\bk\s*a\s*l\s*o\s*u\b", "kalou", t, flags=re.IGNORECASE)
    t = re.sub(r"\by\s*a\s*l\s*o\b", "yalo", t, flags=re.IGNORECASE)
    t = re.sub(r"\bn\s+a\b", "na", t, flags=re.IGNORECASE)
    t = re.sub(r"\be\s+n\s+a\b", "ena", t, flags=re.IGNORECASE)
    t = re.sub(r"\bm\s+e\b", "me", t, flags=re.IGNORECASE)
    t = re.sub(r"\bs\s+a\b", "sa", t, flags=re.IGNORECASE)
    t = re.sub(r"\bi\s+a\b", "ia", t, flags=re.IGNORECASE)
    t = re.sub(r"\bq\s+a\b", "qa", t, flags=re.IGNORECASE)
    t = re.sub(r"\bd\s+e\s+l\s+a\b", "dela", t, flags=re.IGNORECASE)
    t = re.sub(r"(\w)-\s+(\w)", r"\1\2", t)
    return t

def remove_digits_anywhere(s: str) -> str:
    return re.sub(r"\d+", "", s)

def refine_en(s: str) -> str:
    s = fix_mojibake(s)
    s = fix_glues_en(s)
    s = normalize_ws(s)
    s = remove_digits_anywhere(s)
    s = s.lower()
    s = light_punct_spacing(s)
    return normalize_ws(s)

def refine_fj(s: str) -> str:
    s = fix_mojibake(s)
    s = fix_glues_fj(s)
    s = normalize_ws(s)
    s = remove_digits_anywhere(s)
    s = s.lower()
    s = light_punct_spacing(s)
    return normalize_ws(s)

# ---------------------------
# PREPROCESS: remove page junk + bare headers
# ---------------------------
def preprocess_raw(raw: str, lang: str) -> str:
    raw = raw.replace("\ufeff", "")
    lines = raw.splitlines()
    out = []
    book_headers = {"en": {"genesis"}, "fj": {"vakatekivu"}}
    for line in lines:
        s = line.strip().lower()
        if re.match(r"^-{3,}\s*page.*-{3,}$", s):
            continue
        if re.match(r"^-{3,}$", s):
            continue
        if s in book_headers.get(lang, set()):
            continue
        out.append(line)
    return "\n".join(out)

# ---------------------------
# SPLIT INTO CHAPTER BLOCKS
# ---------------------------
CHAPTER_RE = re.compile(r"^\s*chapter\s+(\d+)\b", flags=re.IGNORECASE)

def split_into_chapters(raw: str) -> Dict[int, str]:
    lines = raw.splitlines()
    chapters: Dict[int, List[str]] = {}
    current = 1
    chapters[current] = []
    saw_header = False

    for line in lines:
        m = CHAPTER_RE.match(line.strip())
        if m:
            saw_header = True
            current = int(m.group(1))
            chapters.setdefault(current, [])
            continue
        chapters.setdefault(current, []).append(line)

    if not saw_header:
        return {1: raw}

    return {ch: "\n".join(block) for ch, block in chapters.items()}

# ---------------------------
# VERSE PARSING (ROBUST)
# Handles BOTH styles within a chapter:
#   - "1 in the beginning..."
#   - "1:1 in the beginning..."   (chapter:verse)
# Also handles multiple verses on one line.
# ---------------------------
# token matches either:
#  A) chapter:verse (e.g., 1:1)
#  B) verse number (e.g., 1)
TOKEN_RE = re.compile(
    r"(?:(?<=\s)|^)(?:(\d{1,3})\s*:\s*(\d{1,3})|(\d{1,3}))(?=\s*[A-Za-z(“‘\"'])"
)

def parse_chapter_block(chapter_text: str, chapter_num: int) -> List[Tuple[int, int, str]]:
    txt = normalize_ws(fix_mojibake(chapter_text))
    if not txt:
        return []

    matches = list(TOKEN_RE.finditer(txt))
    if not matches:
        return [(chapter_num, 1, txt)]

    # Build segments
    segs: List[Tuple[int, int, str]] = []

    # Leading text before first token: if any, assign verse 1
    lead = txt[:matches[0].start()].strip()
    if lead:
        segs.append((chapter_num, 1, lead))

    for i, m in enumerate(matches):
        ch = m.group(1)
        vs = m.group(2)
        v_only = m.group(3)

        # Determine verse number for THIS chapter block
        if ch is not None and vs is not None:
            # token is chapter:verse
            ch_n = int(ch)
            v_n = int(vs)
            # only accept if chapter matches current block
            if ch_n != chapter_num:
                continue
        else:
            v_n = int(v_only)

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        seg = txt[start:end].strip()
        if seg:
            segs.append((chapter_num, v_n, seg))

    # If verse 1 still missing for chapter 1, attempt fallback:
    # take everything before first explicit verse marker (even if first marker = 2)
    present = {v for (_, v, _) in segs}
    if chapter_num == 1 and 1 not in present:
        first = matches[0]
        fallback = txt[:first.start()].strip()
        if fallback:
            segs.append((chapter_num, 1, fallback))

    # Deduplicate by verse: keep longest
    best: Dict[int, str] = {}
    for _, v, seg in segs:
        seg_n = normalize_ws(seg)
        if v not in best or len(seg_n) > len(best[v]):
            best[v] = seg_n

    return [(chapter_num, v, best[v]) for v in sorted(best.keys())]

def parse_verses_from_txt(raw: str, language: str) -> pd.DataFrame:
    chapters = split_into_chapters(raw)
    recs: List[Tuple[int, int, str]] = []
    for ch in sorted(chapters.keys()):
        recs.extend(parse_chapter_block(chapters[ch], ch))

    df = pd.DataFrame(recs, columns=["chapter", "verse", "text"])
    df["book"] = "genesis"
    if language == "en":
        df.rename(columns={"text": "en_text"}, inplace=True)
    else:
        df.rename(columns={"text": "fj_text"}, inplace=True)
    return df

# ---------------------------
# MAIN: VERSE-ONLY OUTPUTS
# (combination-friendly schema)
# ---------------------------
def main():
    if not os.path.exists(EN_FILE) or not os.path.exists(FJ_FILE):
        print("Missing input files. Put these in the same folder as the script:")
        print(" - genesis-en.txt")
        print(" - genesis-fj.txt")
        return

    en_raw = preprocess_raw(read_txt(EN_FILE), "en")
    fj_raw = preprocess_raw(read_txt(FJ_FILE), "fj")

    en_df = parse_verses_from_txt(en_raw, "en")
    fj_df = parse_verses_from_txt(fj_raw, "fj")

    print(f"parsed en verses: {len(en_df)}")
    print(f"parsed fj verses: {len(fj_df)}")

    merged = pd.merge(en_df, fj_df, on=["book", "chapter", "verse"], how="inner")
    print(f"aligned merged verses: {len(merged)}")

    # Apply refinements
    merged["en_text"] = merged["en_text"].map(refine_en)
    merged["fj_text"] = merged["fj_text"].map(refine_fj)

    # Combination-friendly schema
    # - domain/subdomain make merging with other domains easy later
    # - source_id helps dedupe when you concatenate datasets
    base_cols = ["domain", "subdomain", "source_id", "book", "chapter", "verse",
                 "source_lang", "target_lang", "direction", "source_text", "target_text"]

    # EN -> FJ
    en2fj = pd.DataFrame({
        "domain": "bible",
        "subdomain": "genesis",
        "source_id": merged.apply(lambda r: f"bible|genesis|{int(r['chapter']):03d}|{int(r['verse']):03d}", axis=1),
        "book": merged["book"],
        "chapter": merged["chapter"].astype(int),
        "verse": merged["verse"].astype(int),
        "source_lang": "en",
        "target_lang": "fj",
        "direction": "en->fj",
        "source_text": merged["en_text"],
        "target_text": merged["fj_text"],
    })[base_cols]

    # FJ -> EN
    fj2en = pd.DataFrame({
        "domain": "bible",
        "subdomain": "genesis",
        "source_id": merged.apply(lambda r: f"bible|genesis|{int(r['chapter']):03d}|{int(r['verse']):03d}", axis=1),
        "book": merged["book"],
        "chapter": merged["chapter"].astype(int),
        "verse": merged["verse"].astype(int),
        "source_lang": "fj",
        "target_lang": "en",
        "direction": "fj->en",
        "source_text": merged["fj_text"],
        "target_text": merged["en_text"],
    })[base_cols]

    out_en2fj = os.path.join(BASE_DIR, "bible_genesis_verse__en2fj.csv")
    out_fj2en = os.path.join(BASE_DIR, "bible_genesis_verse__fj2en.csv")

    en2fj.to_csv(out_en2fj, index=False, encoding="utf-8")
    fj2en.to_csv(out_fj2en, index=False, encoding="utf-8")

    # Quick sanity: confirm gen 1:1 exists
    chk = en2fj[(en2fj["chapter"] == 1) & (en2fj["verse"] == 1)]
    print("genesis 1:1 rows:", len(chk))
    print("done:")
    print(" -", out_en2fj)
    print(" -", out_fj2en)

if __name__ == "__main__":
    main()
