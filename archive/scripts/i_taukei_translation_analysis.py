# iTaukei translation analysis

"""
Instructions
------------
1. Place your uploaded Excel files in the same folder as this script, or update the `AUTO_EVAL_PATH` and `HUMAN_EVAL_PATH` variables.
   Expected filenames used by this script (change if needed):
     - automatic_evaluation.xlsx (sheet name: translations_with_scores)
     - human_evaluation.xlsx (sheet name: sheet1)

2. Recommended Python environment (install using pip):
   pip install pandas numpy matplotlib scipy statsmodels scikit-posthocs nltk openpyxl

   Optional (faster Levenshtein):
   pip install python-Levenshtein

3. Run this script in a Jupyter notebook cell-by-cell or as a script. It will create an `analysis_outputs/` folder with CSVs and PNG figures.

4. If you have per-annotator human ratings (one column per rater), rename that file to `human_annotators.xlsx` and update the code where indicated.

What this script does
---------------------
- Loads automatic and human evaluation spreadsheets
- Merges them on robust matching keys (attempts src/ref/hyp matching; falls back to group-index merge)
- Computes automatic and additional metrics (BLEU, normalized Levenshtein ratio)
- Descriptive statistics and plots (boxplots)
- Correlation (Pearson & Spearman) between auto and human metrics
- Repeated-measures testing: Friedman test + post-hoc pairwise Wilcoxon with Holm correction
- Effect sizes: paired Cohen's d and Cliff's delta
- (Optional) Nemenyi post-hoc using scikit-posthocs if installed
- Regression (OLS) predicting human adequacy from automatic metrics (with option to standardize predictors)
- (Optional) Inter-annotator agreement (Fleiss' kappa / Krippendorff) if you provide per-annotator data, otherwise a simulated 3-rater example is available
- Saves a set of CSVs and PNG figures in ./analysis_outputs/

"""
# -------------------------------
# CONFIG
# -------------------------------
AUTO_EVAL_PATH = "automatic_evaluation.xlsx"  # change if needed
HUMAN_EVAL_PATH = "human_evaluation.xlsx"    # change if needed
# If you have a separate file containing per-annotator ratings (columns: src, rater1, rater2, rater3, ...)
HUMAN_ANNOTATORS_PATH = None  # e.g. "human_annotators.xlsx" (set to None to skip)

OUT_DIR = "analysis_outputs"
RANDOM_SEED = 42

# -------------------------------
# IMPORTS
# -------------------------------
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# statistics
from scipy import stats
import statsmodels.api as sm

# optional imports
try:
    import scikit_posthocs as sp
    _has_sp = True
except Exception:
    _has_sp = False

# nltk BLEU
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _has_nltk = True
except Exception:
    _has_nltk = False

# leptvenshtein optional
try:
    import Levenshtein as _levmod
    _has_lev = True
except Exception:
    _has_lev = False

# -------------------------------
# HELPERS
# -------------------------------

def ensure_outdir(path=OUT_DIR):
    os.makedirs(path, exist_ok=True)


def clean_colnames(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    colmap = {c: c.strip().lower().replace(' ', '_').replace('++','pp').replace('.','') for c in df.columns}
    return df.rename(columns=colmap)


def levenshtein_ratio(s1, s2):
    if pd.isna(s1) or pd.isna(s2):
        return np.nan
    s1 = str(s1)
    s2 = str(s2)
    if _has_lev:
        try:
            return _levmod.ratio(s1, s2)
        except Exception:
            pass
    # fallback DP
    la, lb = len(s1), len(s2)
    if la == 0 and lb == 0:
        return 1.0
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): dp[i][0] = i
    for j in range(lb+1): dp[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if s1[i-1]==s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    edit = dp[la][lb]
    return 1 - edit / max(la, lb)


def sentence_bleu_safe(ref, hyp):
    if not _has_nltk:
        return np.nan
    try:
        ref_tokens = ref.strip('"').split()
        hyp_tokens = hyp.strip('"').split()
        if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)
    except Exception:
        return np.nan


def cohens_d_paired(a, b):
    d = np.array(a) - np.array(b)
    md = np.mean(d)
    sd = np.std(d, ddof=1)
    if sd == 0:
        return np.nan
    return md / sd


def cliffs_delta(a, b):
    # a and b are 1-D arrays
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a); m = len(b)
    gt = 0; lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    delta = (gt - lt) / (n*m)
    ad = abs(delta)
    if ad < 0.147:
        mag = 'negligible'
    elif ad < 0.33:
        mag = 'small'
    elif ad < 0.474:
        mag = 'medium'
    else:
        mag = 'large'
    return delta, mag


def fleiss_kappa(counts):
    # counts: N x k matrix where row i has counts of categories for item i across raters
    counts = np.asarray(counts, dtype=float)
    N, k = counts.shape
    n = counts.sum(axis=1)[0]  # raters per item (assumes balanced)
    p_j = counts.sum(axis=0) / (N*n)
    P_i = (np.sum(counts**2, axis=1) - n) / (n*(n-1))
    P_bar = P_i.mean()
    P_e = np.sum(p_j**2)
    if (1 - P_e) == 0:
        return np.nan
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa

# -------------------------------
# LOAD DATA
# -------------------------------
ensure_outdir(OUT_DIR)
np.random.seed(RANDOM_SEED)

print('Loading Excel files...')
if not os.path.exists(AUTO_EVAL_PATH) or not os.path.exists(HUMAN_EVAL_PATH):
    raise FileNotFoundError(f"Please place {AUTO_EVAL_PATH} and {HUMAN_EVAL_PATH} in this folder or update paths.")

auto_sheets = pd.read_excel(AUTO_EVAL_PATH, sheet_name=None)
human_sheets = pd.read_excel(HUMAN_EVAL_PATH, sheet_name=None)
print('Found sheets:')
print('Automatic:', list(auto_sheets.keys()))
print('Human:', list(human_sheets.keys()))

# heuristics: find likely sheets
auto_df = None
for name, df in auto_sheets.items():
    cols = [c.lower() for c in df.columns]
    if any('bleu' in c for c in cols) or any('chrf' in c for c in cols):
        auto_df = df.copy()
        auto_sheet_name = name
        break
if auto_df is None:
    auto_sheet_name, auto_df = list(auto_sheets.items())[0]

human_df = None
for name, df in human_sheets.items():
    cols = [c.lower() for c in df.columns]
    if any('adequacy' in c for c in cols) or any('fluency' in c for c in cols):
        human_df = df.copy()
        human_sheet_name = name
        break
if human_df is None:
    human_sheet_name, human_df = list(human_sheets.items())[0]

print(f"Using automatic sheet: {auto_sheet_name}, shape: {auto_df.shape}")
print(f"Using human sheet: {human_sheet_name}, shape: {human_df.shape}")

# normalize column names
auto = clean_colnames(auto_df)
human = clean_colnames(human_df)

print('Auto columns:', list(auto.columns))
print('Human columns:', list(human.columns))

# Ensure numeric metrics
for c in auto.columns:
    if 'bleu' in c or 'chrf' in c or 'ter' in c:
        try:
            auto[c] = pd.to_numeric(auto[c], errors='coerce')
        except Exception:
            pass
for c in human.columns:
    if c in ['adequacy','fluency','cohesion']:
        human[c] = pd.to_numeric(human[c], errors='coerce')

# Attempt merge on detailed keys
common_keys = [k for k in ['translation','sentence_type','system','src','ref','hyp'] if k in auto.columns and k in human.columns]
if set(['translation','system','src','ref','hyp']).issubset(set(auto.columns)) and set(['translation','system','src','ref','hyp']).issubset(set(human.columns)):
    merged = pd.merge(auto, human, on=['translation','sentence_type','system','src','ref','hyp'], suffixes=('_auto','_human'), how='inner')
else:
    # fallback: merge on translation+system and group index
    auto = auto.copy(); human = human.copy()
    if 'translation' in auto.columns and 'system' in auto.columns:
        auto['_rowg'] = auto.groupby(['translation','system']).cumcount()
    else:
        auto['_rowg'] = auto.index
    if 'translation' in human.columns and 'system' in human.columns:
        human['_rowg'] = human.groupby(['translation','system']).cumcount()
    else:
        human['_rowg'] = human.index
    merged = pd.merge(auto, human, on=['translation','system','_rowg'], suffixes=('_auto','_human'), how='inner')

print('Merged shape:', merged.shape)

# -------------------------------
# COMPUTE EXTRA METRICS
# -------------------------------
print('Computing sentence-level BLEU and Levenshtein ratio...')
merged['sent_bleu_nltk'] = merged.apply(lambda r: sentence_bleu_safe(r.get('ref', ''), r.get('hyp', '')), axis=1)
merged['levenshtein_ratio'] = merged.apply(lambda r: levenshtein_ratio(r.get('ref', ''), r.get('hyp', '')), axis=1)

# Save merged extended
merged.to_csv(os.path.join(OUT_DIR, 'merged_extended.csv'), index=False)

# -------------------------------
# DESCRIPTIVE STATS & PLOTS
# -------------------------------
print('Generating descriptive tables and boxplots...')
# identify metrics
auto_metrics = [c for c in merged.columns if any(k in c for k in ['bleu','chrf','ter'])]
human_metrics = [c for c in ['adequacy','fluency','cohesion'] if c in merged.columns]

group_cols = ['translation','system'] if 'translation' in merged.columns else ['system']

auto_summary = merged.groupby(group_cols)[auto_metrics].agg(['count','mean','std','min','max'])
human_summary = merged.groupby(group_cols)[human_metrics].agg(['count','mean','std','min','max'])

auto_summary.to_csv(os.path.join(OUT_DIR, 'auto_summary_by_system_translation.csv'))
human_summary.to_csv(os.path.join(OUT_DIR, 'human_summary_by_system_translation.csv'))

# boxplots
import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})

for metric in auto_metrics:
    plt.figure(figsize=(8,4))
    merged.boxplot(column=metric, by='system', grid=False)
    plt.title(f'{metric} by system')
    plt.suptitle('')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'boxplot_{metric}.png'))
    plt.close()

for metric in human_metrics:
    plt.figure(figsize=(8,4))
    merged.boxplot(column=metric, by='system', grid=False)
    plt.title(f'{metric} by system')
    plt.suptitle('')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'boxplot_{metric}.png'))
    plt.close()

# -------------------------------
# CORRELATIONS
# -------------------------------
print('Computing correlations auto <-> human...')
cor_rows = []
for a in auto_metrics:
    for h in human_metrics:
        sub = merged[[a,h]].dropna()
        if len(sub) >= 3:
            pr, pp = stats.pearsonr(sub[a], sub[h])
            sr, sp = stats.spearmanr(sub[a], sub[h])
        else:
            pr = pp = sr = sp = np.nan
        cor_rows.append({'auto_metric': a, 'human_metric': h, 'n': len(sub), 'pearson_r': pr, 'pearson_p': pp, 'spearman_r': sr, 'spearman_p': sp})
cor_df = pd.DataFrame(cor_rows)
cor_df.to_csv(os.path.join(OUT_DIR, 'correlation_auto_vs_human.csv'), index=False)

# -------------------------------
# REPEATED-MEASURES: FRIEDMAN + PAIRWISE WILCOXON + EFFECT SIZES
# -------------------------------
print('Running Friedman tests and pairwise Wilcoxon with effect sizes...')

pairwise_results = []
# pick pair_key for pivoting
pair_key = None
for k in ['src','ref','hyp','_rowg']:
    if k in merged.columns:
        pair_key = k
        break
if pair_key is None:
    raise ValueError('No pairing key found for repeated-measures tests; ensure src/ref/hyp or _rowg exists')

systems = sorted(merged['system'].unique())

for metric in auto_metrics + human_metrics:
    pivot = merged.pivot_table(index=pair_key, columns='system', values=metric)
    pivot_drop = pivot.dropna(axis=0, how='any')
    n_segments = pivot_drop.shape[0]
    fried_stat = fried_p = np.nan
    if pivot_drop.shape[1] > 2 and n_segments >= 5:
        try:
            fried_stat, fried_p = stats.friedmanchisquare(*[pivot_drop[c] for c in pivot_drop.columns])
        except Exception:
            fried_stat, fried_p = np.nan, np.nan
    # pairwise
    from itertools import combinations
    pairs = list(combinations(pivot_drop.columns, 2))
    pvals = []
    pr_list = []
    for a,b in pairs:
        xa = pivot_drop[a].values
        xb = pivot_drop[b].values
        if len(xa) < 2:
            stat_w = p_w = np.nan
        else:
            try:
                stat_w, p_w = stats.wilcoxon(xa, xb)
            except Exception:
                stat_w = p_w = np.nan
        d = cohens_d_paired(xa, xb)
        cd, mag = cliffs_delta(xa, xb)
        pr_list.append({'metric': metric, 'pair': f'{a} vs {b}', 'n': len(xa), 'friedman_stat': fried_stat, 'friedman_p': fried_p, 'wilcoxon_stat': stat_w, 'wilcoxon_p': p_w, 'cohens_d_paired': d, 'cliffs_delta': cd, 'cliffs_mag': mag})
        pvals.append(p_w if not pd.isna(p_w) else 1.0)
    # Holm correction if any pvals
    if pvals:
        reject, p_adj, _, _ = statsmodels.stats.multitest.multipletests(pvals, alpha=0.05, method='holm')
        for i, row in enumerate(pr_list):
            row['wilcoxon_p_adj_holm'] = p_adj[i]
            row['reject_holm'] = bool(reject[i])
    pairwise_results.extend(pr_list)

pairwise_df = pd.DataFrame(pairwise_results)
pairwise_df.to_csv(os.path.join(OUT_DIR, 'pairwise_wilcoxon_effects.csv'), index=False)

# Optional Nemenyi (if scikit-posthocs available)
if _has_sp:
    try:
        for metric in auto_metrics + human_metrics:
            pivot = merged.pivot_table(index=pair_key, columns='system', values=metric)
            pivot_drop = pivot.dropna(axis=0, how='any')
            if pivot_drop.shape[0] >= 2 and pivot_drop.shape[1] > 2:
                nemenyi = sp.posthoc_nemenyi_friedman(pivot_drop.values)
                # save matrix
                pd.DataFrame(nemenyi, index=pivot_drop.columns, columns=pivot_drop.columns).to_csv(os.path.join(OUT_DIR, f'nemenyi_{metric}.csv'))
    except Exception as e:
        print('Nemenyi failed:', e)

# -------------------------------
# REGRESSION: predict adequacy from automatic metrics
# -------------------------------
print('Running OLS regression predicting adequacy from automatic metrics...')
reg_metrics = [c for c in ['bleu_score','BLEU_Score','chrfpp_score','CHRF++_Score','ter_score','TER_Score','sent_bleu_nltk','levenshtein_ratio'] if c in merged.columns]
# normalize names
reg_metrics = list(dict.fromkeys(reg_metrics))
print('Regression predictors:', reg_metrics)
reg_df = merged[['adequacy'] + reg_metrics].dropna()
if len(reg_df) < 3:
    print('Insufficient data for regression. Skipping regression step.')
else:
    X = reg_df[reg_metrics]
    # option: standardize predictors
    scaler = None
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        X = pd.DataFrame(Xs, columns=reg_metrics)
    except Exception:
        pass
    X = sm.add_constant(X)
    y = reg_df['adequacy']
    model = sm.OLS(y, X).fit()
    with open(os.path.join(OUT_DIR, 'regression_summary.txt'), 'w') as f:
        f.write(model.summary().as_text())
    print(model.summary())

# -------------------------------
# INTER-ANNOTATOR AGREEMENT
# -------------------------------
print('Computing inter-annotator agreement (if per-annotator file provided).')
if HUMAN_ANNOTATORS_PATH and os.path.exists(HUMAN_ANNOTATORS_PATH):
    ann_sheets = pd.read_excel(HUMAN_ANNOTATORS_PATH, sheet_name=None)
    # heuristics: pick first sheet
    ann_df = list(ann_sheets.values())[0]
    # expect one column per rater (plus an id column like src)
    # user should provide proper format. We'll attempt Fleiss' kappa.
    raters_cols = [c for c in ann_df.columns if c.lower().startswith('r') or c.lower().startswith('rater')]
    if len(raters_cols) >= 2:
        counts = []
        categories = sorted(set(sum([list(ann_df[c].dropna().unique()) for c in raters_cols], [])))
        for _, row in ann_df.iterrows():
            counts.append([sum(row[r] == cat for r in raters_cols) for cat in categories])
        kappa = fleiss_kappa(counts)
        with open(os.path.join(OUT_DIR, 'inter_annotator_agreement.txt'), 'w') as f:
            f.write(f'Fleiss kappa (categories {categories}): {kappa}\n')
        print('Fleiss kappa:', kappa)
    else:
        print('Per-annotator file found but could not detect rater columns. Expected columns: r1, r2, r3...')
else:
    # simulate 3 raters from the mean ratings (for illustration only)
    print('No per-annotator file provided. Simulating 3 raters from mean ratings (for demonstration only).')
    iaa = {}
    for metric in human_metrics:
        base = merged[metric].fillna(3).astype(float).values
        n = len(base)
        r = 3
        # add small random noise and clip to 1..5
        rng = np.random.RandomState(RANDOM_SEED)
        raters = []
        for i in range(r):
            noise = rng.choice([-1,0,1], size=n, p=[0.2,0.6,0.2])
            sim = np.clip(np.round(base + noise), 1, 5).astype(int)
            raters.append(sim)
        counts = []
        categories = [1,2,3,4,5]
        for i in range(n):
            rowcounts = [sum(raters[j][i] == c for j in range(r)) for c in categories]
            counts.append(rowcounts)
        kappa = fleiss_kappa(counts)
        iaa[metric] = {'fleiss_kappa': kappa, 'n_raters': r, 'n_items': n}
    with open(os.path.join(OUT_DIR, 'inter_annotator_simulated.txt'), 'w') as f:
        f.write(str(iaa))
    print('Simulated IAA results saved.')

# -------------------------------
# SAVE SUMMARY CSVs
# -------------------------------
print('Saving summary CSVs...')
cor_df.to_csv(os.path.join(OUT_DIR, 'correlation_auto_vs_human.csv'), index=False)
pairwise_df.to_csv(os.path.join(OUT_DIR, 'pairwise_wilcoxon_effects.csv'), index=False)

print('All outputs saved to folder:', OUT_DIR)
print('Files:')
for f in os.listdir(OUT_DIR):
    print('-', f)

print('\nDone. Open the analysis_outputs folder to retrieve CSVs and PNGs.')

