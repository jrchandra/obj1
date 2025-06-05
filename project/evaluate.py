import pandas as pd
import sacrebleu
from comet import download_model, load_from_checkpoint
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIGURATION ===
TRANSLATION_FILE = "data/translations.xlsx"
HUMAN_EVAL_FILE = "data/human_evaluation.xlsx"
OUTPUT_FILE = "outputs/evaluation_results.xlsx"
CHART_DIR = "outputs/metric_charts"

os.makedirs(CHART_DIR, exist_ok=True)

# === Load Data ===
df_trans = pd.read_excel(TRANSLATION_FILE)
systems = df_trans['system'].unique()

# === Load COMET Model ===
print("🔁 Loading COMET model...")
model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

# === Collect Metrics ===
results = []

for system in systems:
    sys_data = df_trans[df_trans['system'] == system]
    references = [[ref] for ref in sys_data["ref"].tolist()]
    hypotheses = sys_data["hyp"].tolist()

    # Automatic metrics
    bleu = sacrebleu.corpus_bleu(hypotheses, references).score
    chrf = sacrebleu.corpus_chrf(hypotheses, references).score
    ter = sacrebleu.corpus_ter(hypotheses, references).score

    # COMET
    comet_data = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(sys_data["src"], sys_data["hyp"], sys_data["ref"])
    ]
    comet_scores = comet_model.predict(comet_data, batch_size=8, gpus=1 if comet_model.on_gpu else 0)["scores"]
    comet = sum(comet_scores) / len(comet_scores)

    # Human evaluation (if available)
    fluency = adequacy = cohesion = None
    if os.path.exists(HUMAN_EVAL_FILE):
        df_human = pd.read_excel(HUMAN_EVAL_FILE)
        df_human_sys = df_human[df_human["system"] == system]
        if not df_human_sys.empty:
            fluency = df_human_sys["fluency"].mean()
            adequacy = df_human_sys["adequacy"].mean()
            cohesion = df_human_sys["cohesion"].mean()

    results.append({
        "System": system,
        "BLEU": bleu,
        "CHRF++": chrf,
        "TER": ter,
        "COMET": comet,
        "Fluency (human)": fluency,
        "Adequacy (human)": adequacy,
        "Cohesion (human)": cohesion,
    })

df_results = pd.DataFrame(results)
df_results.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Results saved to {OUTPUT_FILE}")

# === Visualization ===
sns.set(style="whitegrid")

def plot_metrics(df, columns, title, filename):
    df_melted = df.melt(id_vars="System", value_vars=columns, var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="System")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, filename))
    plt.close()

plot_metrics(df_results, ["BLEU", "CHRF++", "TER", "COMET"], "Automatic Evaluation Metrics", "automatic_metrics.png")
plot_metrics(df_results, ["Fluency (human)", "Adequacy (human)", "Cohesion (human)"], "Human Evaluation Metrics", "human_metrics.png")
print("📊 Visualizations saved to", CHART_DIR)
