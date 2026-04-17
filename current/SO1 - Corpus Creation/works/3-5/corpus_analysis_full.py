import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

MASTER = "master_parallel_corpus__COMBINED.xlsx"
OUTDIR = "analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------
# Load corpus
# ----------------------
print("Loading corpus...")
df = pd.read_excel(MASTER)

# Normalise fields
df["domain"] = df["domain"].astype(str).str.strip().str.lower()
df["sentence_type"] = df["sentence_type"].astype(str).str.strip().str.lower()
df["direction"] = df["direction"].astype(str).str.strip()

# Remove empty rows
df = df[(df["source_text"].astype(str).str.strip() != "") &
        (df["target_text"].astype(str).str.strip() != "")]

# Token lengths
df["src_len"] = df["source_text"].astype(str).str.split().map(len)
df["tgt_len"] = df["target_text"].astype(str).str.split().map(len)

# ----------------------
# TABLES
# ----------------------
print("Generating tables...")

# 1. Domain counts
domain_counts = df["domain"].value_counts().rename_axis("domain").reset_index(name="count")
domain_counts.to_csv(f"{OUTDIR}/table_domain_counts.csv", index=False)

# 2. Sentence type counts
stype_counts = df["sentence_type"].value_counts().rename_axis("sentence_type").reset_index(name="count")
stype_counts.to_csv(f"{OUTDIR}/table_sentence_type_counts.csv", index=False)

# 3. Direction counts
dir_counts = df["direction"].value_counts().rename_axis("direction").reset_index(name="count")
dir_counts.to_csv(f"{OUTDIR}/table_direction_counts.csv", index=False)

# 4. Domain × sentence type
dom_stype = pd.crosstab(df["domain"], df["sentence_type"])
dom_stype.to_csv(f"{OUTDIR}/table_domain_sentence_type.csv")

# 5. Domain × direction
dom_dir = pd.crosstab(df["domain"], df["direction"])
dom_dir.to_csv(f"{OUTDIR}/table_domain_direction.csv")

# ----------------------
# PLOTS
# ----------------------
print("Generating plots...")

plt.figure(figsize=(10,6))
domain_counts.plot(kind="bar", x="domain", y="count", legend=False)
plt.title("Corpus Distribution by Domain")
plt.ylabel("Number of pairs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/plot_domain_distribution.png")
plt.close()

plt.figure(figsize=(10,6))
stype_counts.plot(kind="bar", x="sentence_type", y="count", legend=False)
plt.title("Corpus Distribution by Sentence Type")
plt.ylabel("Number of pairs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/plot_sentence_type_distribution.png")
plt.close()

plt.figure(figsize=(6,5))
dir_counts.plot(kind="bar", x="direction", y="count", legend=False)
plt.title("Corpus Distribution by Direction")
plt.ylabel("Number of pairs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/plot_direction_distribution.png")
plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(dom_dir, annot=True, fmt="d", cmap="Blues")
plt.title("Domain × Direction Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/plot_domain_direction_heatmap.png")
plt.close()

# Sentence length histograms
plt.figure(figsize=(10,6))
plt.hist(df["src_len"], bins=50)
plt.title("Source Sentence Length Distribution (tokens)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/plot_source_length_hist.png")
plt.close()

plt.figure(figsize=(10,6))
plt.hist(df["tgt_len"], bins=50)
plt.title("Target Sentence Length Distribution (tokens)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/plot_target_length_hist.png")
plt.close()

# ----------------------
# STATISTICS SUMMARY
# ----------------------
stats = {
    "total_pairs": len(df),
    "avg_src_len": df["src_len"].mean(),
    "median_src_len": df["src_len"].median(),
    "avg_tgt_len": df["tgt_len"].mean(),
    "median_tgt_len": df["tgt_len"].median(),
    "std_src_len": df["src_len"].std(),
    "std_tgt_len": df["tgt_len"].std(),
    "min_src_len": df["src_len"].min(),
    "min_tgt_len": df["tgt_len"].min(),
    "max_src_len": df["src_len"].max(),
    "max_tgt_len": df["tgt_len"].max(),
}

stats_df = pd.DataFrame([stats])
stats_df.to_csv(f"{OUTDIR}/corpus_stats_summary.csv", index=False)

print("All analysis outputs saved to:", OUTDIR)
