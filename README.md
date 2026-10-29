# Explainable Hate Speech Detection on HateXplain

NLP / deep learning project: classify social-media posts as **Hate speech**, **Offensive** or **Normal**, and check whether the model's explanations agree with what human annotators marked as the reason for their label.

- **Dataset:** [HateXplain](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain) (Gab + Twitter posts, 3 annotators per post, word-level human rationales). 15,383 train / 1,922 validation / 1,924 test. Label = majority vote.
- **Models:** TF-IDF + Logistic Regression (baseline) → fine-tuned XLM-RoBERTa base.
- **Explainability:** SHAP, compared against human rationales (overlap, comprehensiveness, sufficiency).

## Pipeline

```
HateXplain (HF parquet, pinned revision)
        │  src/data.py   join tokens, majority-vote label
        ▼
data/processed/*.csv  (id, text, label, rationales, annotator_labels, label_name)
        │
        ├──► TF-IDF (1-2 grams, 50k) ─► Logistic Regression ─► SHAP LinearExplainer
        │                                   (notebooks 02, 03)
        │
        └──► XLM-R tokenizer (max_len 128) ─► XLM-R base + 3-class head ─► test metrics
                                              (src/train_xlmr.py)           confusion matrix
                                                         │
                                                         ▼
                                  explanations vs human rationales (planned)
                                                         ▼
                                              Streamlit demo (planned)
```

## Results so far

| Model | Split | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| TF-IDF + LR | validation | 0.650 | 0.631 | Offensive is the weakest class (F1 0.48, recall 0.41) |
| TF-IDF + LR | test | run notebook 02 | | writes `results/tables/baseline_metrics_test.csv` |
| XLM-RoBERTa base | validation / test | run `src/train_xlmr.py` | | writes `results/tables/xlmr_metrics.csv` |

SHAP on the baseline (validation posts that have human rationales): mean rationale coverage 0.76 on 53 posts; mean comprehensiveness 0.22 and sufficiency gap 0.13 on 30 random posts. These use a top-15 / top-5 word cut-off and will be replaced by a stricter metric in the next step.

## Repo layout

```
notebooks/   01 exploration · 02 baseline · 03 SHAP explanations
src/         config.py (paths, constants) · data.py · metrics.py · xlmr.py
             train_xlmr.py · evaluate.py
results/     tables/ and figures/ (committed) · logs/ (ignored)
data/        processed CSVs (ignored; rebuilt with python -m src.data)
models/      weights and checkpoints (ignored)
```

## Setup

Use Python 3.12 (PyTorch support for 3.14 is not reliable yet).

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

`requirements-lock.txt` is the full pinned environment from the original Mac setup.

## How to run

```bash
# 1. Build the processed CSVs (needs internet, once)
python -m src.data

# 2. Baseline + SHAP: run notebooks/02_Baseline_Classifier.ipynb, then 03_SHAP_Explanations.ipynb

# 3. XLM-RoBERTa. First a quick check that everything works:
python -m src.train_xlmr --smoke

# Full training from Terminal (survives Jupyter kernel restarts; keeps the Mac awake):
caffeinate -i nohup python -m src.train_xlmr > results/logs/xlmr_train.log 2>&1 &
tail -f results/logs/xlmr_train.log

# If it stops, continue from the last checkpoint:
python -m src.train_xlmr --resume

# Re-evaluate the saved model on the test set without retraining:
python -m src.evaluate
```

Training settings: `FacebookAI/xlm-roberta-base`, max length 128, learning rate 2e-5, batch size 4 with gradient accumulation 4 (effective 16), 3 epochs, weight decay 0.01, best epoch chosen by validation macro-F1, seed 42. Runs on Apple MPS, CUDA or CPU automatically.

## Roadmap

- [x] Data exploration and processed splits
- [x] TF-IDF + LR baseline, SHAP explanations
- [ ] XLM-RoBERTa fine-tuning and test-set evaluation
- [ ] Error analysis (Hate vs Offensive confusion)
- [ ] Explanations for XLM-R vs human rationales (token-level F1 / AUPRC, comprehensiveness, sufficiency)
- [ ] Streamlit demo
- [ ] Final report

## Limitations

HateXplain posts are English and from 2019-2020 Gab/Twitter, annotator agreement on Hate vs Offensive is imperfect, and models can over-flag identity terms (for example, "muslim" or "gay" in neutral posts). Results should not be used for real moderation without further bias evaluation.
