# translation_evaluation.py

# Requirements:
# Uninstall existing potentially incompatible versions
# Use --force-reinstall to ensure fresh installation even if versions match
# Include any potentially relevant dependencies like Pillow or other image libraries
!pip uninstall -y unbabel-comet torch torchvision torchaudio pytorch-lightning numpy Pillow sacrebleu rouge-score scipy transformers scikit-learn

# Install core scientific libraries first, together to ensure compatibility
# Installing numpy and scipy together to potentially resolve dependencies
!pip install --upgrade numpy scipy

# Reinstall torch with cuda support (adjust if you don't have a GPU or specific cuda version)
# Adding --no-cache-dir can help avoid using potentially corrupted cached wheels
# Keep this as is, as torch installation seems separate from the numpy/scipy issue
!pip install torch torchvision torchio --index-url https://download.pytorch.org/pytorch-nightly/cu118 --no-cache-dir

# Install transformers and scikit-learn next, which depend on numpy/scipy
# This order might help in resolving compatible versions
!pip install transformers scikit-learn

# Install pytorch-lightning and unbabel-comet
!pip install pytorch-lightning unbabel-comet
!pip install sacrebleu
!pip install rouge-score


import sacrebleu
from comet import download_model, load_from_checkpoint

from rouge_score import rouge_scorer

def evaluate_rouge(references, hypotheses, rouge_types=['rouge1', 'rouge2', 'rougeL']):
  scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
  # rouge_scorer expects single strings, so iterate if inputs are lists
  if isinstance(references, list) and isinstance(hypotheses, list):
      # Assuming references and hypotheses are lists of single strings for simplicity
      # The original code only scored the first pair.
      # If you have multiple pairs, you would need to loop here.
      scores = scorer.score(references[0], hypotheses[0])
  else:
      scores = scorer.score(references, hypotheses)
  return scores

def evaluate_bleu(references, hypotheses):
    # sacrebleu expects hypotheses as a list of strings and references as a list of lists of strings
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def evaluate_chrf(references, hypotheses):
    # sacrebleu expects hypotheses as a list of strings and references as a list of lists of strings
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return chrf.score

def evaluate_ter(references, hypotheses):
    # sacrebleu expects hypotheses as a list of strings and references as a list of lists of strings
    ter = sacrebleu.corpus_ter(hypotheses, [references])
    return ter.score

def evaluate_comet(references, hypotheses, sources):
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, hypotheses, references)]
    # Adjust gpus based on your system's GPU availability
    # Check if model.hparams has 'gpus' attribute before accessing it
    # Use num_gpus=1 if you are sure you have a GPU, otherwise 0
    # If running on CPU, ensure gpus=0 or remove the argument if predict defaults to CPU
    num_gpus = 0 # Default to CPU
    if hasattr(model, 'hparams') and hasattr(model.hparams, 'gpus') and model.hparams.gpus > 0:
        # Attempt to use GPU if available and specified in model hparams
        # You might need to adjust this based on actual GPU availability and setup
        num_gpus = 1 # Assuming a single GPU is available

    score = model.predict(data, batch_size=8, gpus=num_gpus)
    return score.system_score

# Example usage:
sources = ["Put the pillow in the pillowcase"]
references = ["Vakawaqana na ilokoloko"]
hypotheses = ["Biuta na ulunivanua ena kena cove?"]
print("BLEU:", evaluate_bleu(references, hypotheses))
print("CHRF++:", evaluate_chrf(references, hypotheses))
print("TER:", evaluate_ter(references, hypotheses))
print("COMET:", evaluate_comet(references, hypotheses, sources))
print("ROUGE:", evaluate_rouge(references, hypotheses))