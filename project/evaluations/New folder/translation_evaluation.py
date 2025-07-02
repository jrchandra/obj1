# translation_evaluation.py

# Requirements:
# pip install sacrebleu
# pip install unbabel-comet

import sacrebleu
from comet import download_model, load_from_checkpoint

def evaluate_bleu(references, hypotheses):
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def evaluate_chrf(references, hypotheses):
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return chrf.score

def evaluate_ter(references, hypotheses):
    ter = sacrebleu.corpus_ter(hypotheses, [references])
    return ter.score

def evaluate_comet(references, hypotheses, sources):
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, hypotheses, references)]
    score = model.predict(data, batch_size=8, gpus=1 if model.hparams.gpus > 0 else 0)
    return score.system_score

# Example usage:
# sources = ["This is a test.", "How are you?"]
# references = ["Oqo e dua na vakatovolea.", "Vakacava o iko?"]
# hypotheses = ["Oqo e dua na vakatovolea.", "O iko vakacava?"]
# print("BLEU:", evaluate_bleu(references, hypotheses))
# print("CHRF++:", evaluate_chrf(references, hypotheses))
# print("TER:", evaluate_ter(references, hypotheses))
# print("COMET:", evaluate_comet(references, hypotheses, sources))
