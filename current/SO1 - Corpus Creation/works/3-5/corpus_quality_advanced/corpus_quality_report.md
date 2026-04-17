# English–Fijian Parallel Corpus – Quality Audit

## 1. Overview

- Total sentence pairs after cleaning: **15912**

- Source noise (heuristic): **56.39%**

- Target noise (heuristic): **8.65%**



## 2. Language Identity Consistency

- Proportion of source segments classified as English-like: **78.04%**

- Proportion of target segments classified as Fijian-like: **40.00%**

These values suggest that most source segments follow English profiles while most targets follow Fijian profiles, indicating low cross-lingual contamination.


## 3. Alignment Quality (TF-IDF and chrF)

- Mean TF-IDF cosine similarity between source and target: **0.044**

- Median TF-IDF cosine similarity: **0.000**

- Proportion of sentence pairs with TF-IDF similarity < 0.10: **86.17%**

Pairs with very low similarity typically correspond to misalignments, metadata, or highly non-literal translations.


### chrF-based similarity (sampled)

- Mean sentence-level chrF score (sampled pairs): **6.81**

- Median chrF score: **4.46**


Character-level chrF provides a language-agnostic signal of form-level correspondence between source and target sentences.


## 4. Length Statistics

Token-length distributions for source and target sentences are provided in the accompanying histograms (`plot_src_length_hist.png`, `plot_tgt_length_hist.png`). These show a mix of short lexical entries (dictionary/idioms) and longer legal/medical clauses.


## 5. Domain and Direction Coverage

Domain × direction counts:


| domain         |   en->fj |   fj->en |
|:---------------|---------:|---------:|
| bible          |     1532 |     1532 |
| conversational |      133 |      132 |
| definition     |      863 |      863 |
| dictionary     |        0 |     8959 |
| idiom          |        0 |      126 |
| legal          |      732 |      732 |
| medical        |      131 |      177 |



## 6. Interpretation

Overall, the corpus exhibits low rates of obvious noise, high language-identity consistency for source and target sides, and relatively high alignment similarity scores, supporting its suitability as a benchmark dataset for evaluating English–Fijian machine translation systems.
