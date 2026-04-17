import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import re
import unicodedata as ud

# UPDATE if your Tesseract path is different
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MOJIBAKE_MAP = {
    "â€œ": "“", "â€": "”", "â€™": "’", "â€˜": "‘",
    "â€“": "–", "â€”": "—", "Â": ""
}

def fix_text(s: str) -> str:
    for k, v in MOJIBAKE_MAP.items():
        s = s.replace(k, v)
    s = ud.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def ocr_pdf(pdf_path: Path):
    images = convert_from_path(pdf_path, dpi=300)
    lines = []
    for img in images:
        txt = pytesseract.image_to_string(img, lang="eng")
        for ln in txt.splitlines():
            ln = fix_text(ln)
            if len(ln) >= 5:
                lines.append(ln)
    return lines

def main():
    folder = Path(__file__).resolve().parent
    pdfs = list(folder.glob("*.pdf"))

    for pdf in pdfs:
        print(f"OCR → {pdf.name}")
        lines = ocr_pdf(pdf)
        out = pdf.with_suffix(".txt")
        out.write_text("\n".join(lines), encoding="utf-8")

    print("DONE")

if __name__ == "__main__":
    main()
