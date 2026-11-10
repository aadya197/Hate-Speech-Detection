# Explainable Hate Speech Detection on HateXplain

NLP / deep learning project: classify social-media posts as **Hate speech**, **Offensive** or **Normal**, and check whether the model's explanations agree with what human annotators marked as the reason for their label.

- **Dataset:** [HateXplain](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain) (Gab + Twitter posts, 3 annotators per post, word-level human rationales). 15,383 train / 1,922 validation / 1,924 test. Label = majority vote.
- **Models:** TF-IDF + Logistic Regression (baseline) → fine-tuned BERT-base → fine-tuned XLM-RoBERTa base.
- **Explainability:** LIME token attributions, compared against human rationales.

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

## Results

### Model Comparison (test set, 1 924 posts)

| Model | Accuracy | Macro-F1 | F1 Hate | F1 Normal | F1 Offensive |
|---|---|---|---|---|---|
| XLM-R (GPU) | 0.693 | 0.680 | 0.773 | 0.742 | 0.524 |
| BERT-base | 0.689 | 0.678 | 0.774 | 0.730 | 0.531 |
| TF-IDF + LogReg | 0.669 | 0.648 | 0.727 | 0.729 | 0.486 |

### Statistical Significance (XLM-R vs TF-IDF baseline, paired bootstrap n=1 924)

| Metric | Diff | 95 % CI | P(better) |
|---|---|---|---|
| Macro-F1 | +0.032 | [+0.011, +0.053] | 0.997 |
| Accuracy | +0.024 | [+0.004, +0.044] | 0.988 |

XLM-R significantly outperforms the baseline on both metrics (p < 0.01).

### Error Analysis (XLM-R GPU, top confusion pairs)

| True | Predicted | Count | Share of errors |
|---|---|---|---|
| Normal | Offensive | 152 | 25.8 % |
| Offensive | Normal | 151 | 25.6 % |
| Offensive | Hate speech | 124 | 21.0 % |
| Hate speech | Offensive | 68 | 11.5 % |
| Normal | Hate speech | 58 | 9.8 % |

Accuracy by annotator agreement: **81.0 %** when all 3 annotators agree · **57.4 %** on split 2-1 votes — confirming that hard cases are genuinely ambiguous even for humans.

SHAP on the baseline (validation posts that have human rationales): mean rationale coverage 0.76 on 53 posts; mean comprehensiveness 0.22 and sufficiency gap 0.13 on 30 random posts.

## Repo layout

```
notebooks/   01 exploration · 02 baseline · 03 SHAP explanations · 04 XLM-R · 05 Colab GPU · 06 error analysis
src/         config.py · data.py · metrics.py · xlmr.py · train_xlmr.py · evaluate.py · analysis.py · explain.py
results/     tables/ and figures/ (committed) · logs/ (ignored)
data/        processed CSVs (ignored; rebuilt with python -m src.data)
models/      weights and checkpoints (ignored)
app.py       Streamlit demo
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
- [x] TF-IDF + LR baseline, SHAP/LIME explanations
- [x] BERT-base fine-tuning and test-set evaluation
- [x] XLM-RoBERTa fine-tuning and test-set evaluation
- [x] Error analysis (Hate vs Offensive confusion, agreement breakdown)
- [x] Explainability module (LIME token attributions vs human rationales)
- [x] Streamlit demo (`app.py`)
- [ ] Final report

## Limitations

HateXplain posts are English and from 2019-2020 Gab/Twitter, annotator agreement on Hate vs Offensive is imperfect, and models can over-flag identity terms (for example, "muslim" or "gay" in neutral posts). Results should not be used for real moderation without further bias evaluation.
