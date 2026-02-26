from pathlib import Path

inp = Path("idiom_parallel_fj2en__ALL_SPLIT.csv")
out = Path("idiom_parallel_fj2en__ALL_SPLIT__UTF8.csv")

raw = inp.read_bytes()

# Most likely cp1252 (because of 0x91). Use errors="strict" to confirm, or "replace" to force.
text = raw.decode("cp1252")  # change to latin-1 if needed
out.write_text(text, encoding="utf-8")

print("Wrote:", out)
