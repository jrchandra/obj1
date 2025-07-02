from IPython import get_ipython
from IPython.display import display
# %%
!pip install sacrebleu

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu

# Load your translations file
df_translations = pd.read_excel('translations.xlsx')

# Use 'hyp' for machine translations and 'ref' for human reference translations
machine_translations = df_translations['hyp'].tolist()
reference_translations = df_translations['ref'].tolist()

# Initialize lists to store scores
bleu_scores = []
chrf_scores = []
ter_scores = []

# NLTK's BLEU requires tokenized sentences
smooth_func = SmoothingFunction().method1 # Using smoothing for short sentences

for mt, ref in zip(machine_translations, reference_translations):
    # Tokenize sentences (example: split by space, adjust for your specific needs)
    tokenized_mt = mt.split()
    tokenized_ref = [ref.split()] # NLTK BLEU expects a list of reference token lists

    # Calculate BLEU (using NLTK for example)
    bleu = sentence_bleu(tokenized_ref, tokenized_mt, smoothing_function=smooth_func)
    bleu_scores.append(bleu * 100) # Convert to 0-100 scale

    # Calculate CHRF++ and TER using sacrebleu (requires strings, not tokenized)
    # sacrebleu expects references as a list of strings for corpus level,
    # but for sentence level, the hypothesis is a string and references are a list of strings
    sacre_ref = [ref]
    sacre_mt = mt

    # Pass sacre_mt directly as a string to sentence_chrf and sentence_ter
    chrf = sacrebleu.sentence_chrf(sacre_mt, sacre_ref).score
    ter = sacrebleu.sentence_ter(sacre_mt, sacre_ref).score

    chrf_scores.append(chrf)
    ter_scores.append(ter)

# Add the scores back to your DataFrame
df_translations['BLEU_Score'] = bleu_scores
df_translations['CHRF++_Score'] = chrf_scores
df_translations['TER_Score'] = ter_scores

# Save the updated DataFrame (optional)
df_translations.to_csv('translations_with_scores.csv', index=False)

print(df_translations.head())