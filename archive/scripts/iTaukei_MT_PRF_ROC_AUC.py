import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# Load datasets
auto = pd.read_excel("automatic_evaluation.xlsx", sheet_name="translations_with_scores")
human = pd.read_excel("human_evaluation.xlsx", sheet_name="sheet1")

# Clean column names
def clean_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

auto = clean_cols(auto)
human = clean_cols(human)

# Merge datasets
merge_keys = [k for k in ['translation','sentence_type','system','src','ref','hyp'] if k in auto.columns and k in human.columns]
merged = pd.merge(auto, human, on=merge_keys, suffixes=('_auto','_human'), how='inner').reset_index(drop=True)

# Compute token-level precision, recall, F1
def prf_scores(ref, hyp):
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    tp = len([t for t in hyp_tokens if t in ref_tokens])
    fp = len([t for t in hyp_tokens if t not in ref_tokens])
    fn = len([t for t in ref_tokens if t not in hyp_tokens])
    prec = tp / (tp + fp) if (tp+fp)>0 else 0
    rec = tp / (tp + fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return prec, rec, f1

merged['precision'], merged['recall'], merged['f1'] = zip(*merged.apply(lambda r: prf_scores(str(r['ref']), str(r['hyp'])), axis=1))

# Aggregate per system & translation
system_prf = merged.groupby(['system','translation'])[['precision','recall','f1']].mean().reset_index()

# ROC-AUC analysis (Adequacy as binary ground truth)
merged['adequacy_binary'] = (merged['adequacy'] >= 3).astype(int)

roc_results = []
metrics_to_test = [c for c in merged.columns if any(k in c for k in ['bleu','chrf','ter'])]

plt.figure(figsize=(6,6))
for m in metrics_to_test:
    if merged[m].nunique() > 1:
        fpr, tpr, _ = roc_curve(merged['adequacy_binary'], merged[m])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{m} (AUC={roc_auc:.2f})")
        roc_results.append({'metric':m,'AUC':roc_auc})

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Automatic metrics predicting adequacy≥3")
plt.legend()

# Save outputs
out_dir = "analysis_outputs"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir,"roc_curves_metrics.png"))
plt.close()

roc_results_df = pd.DataFrame(roc_results)

# Save dataframes
system_prf.to_csv(os.path.join(out_dir,"system_precision_recall_f1.csv"), index=False)
roc_results_df.to_csv(os.path.join(out_dir,"roc_auc_results.csv"), index=False)
