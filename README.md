# Explainable Hate Speech Detection on HateXplain

Deep-learning project: classify social-media posts as **Hate speech**, **Offensive** or **Normal**, using our own architecture (**HateNet**) whose attention is trained to agree with the words human annotators marked as the reason for their label. We compare it with a classical baseline and standard fine-tuned transformers, run an ablation study, and test whether the explanations are faithful.

- **Dataset:** [HateXplain](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain) (Gab + Twitter posts, 3 annotators per post, word-level human rationales). 15,383 train / 1,922 validation / 1,924 test. Label = majority vote.
- **Models:** TF-IDF + Logistic Regression (baseline) · fine-tuned BERT-base · fine-tuned XLM-RoBERTa base · **HateNet** (BERT + BiLSTM + rationale-guided attention).
- **Explainability:** word-level occlusion (works for every model) and HateNet's built-in attention, both compared with human rationales (plausibility) and tested with comprehensiveness / sufficiency (faithfulness).

## HateNet architecture

```
tokens ─► BERT encoder ─► BiLSTM (128 per direction) ─► attention pooling ─► dropout 0.3 ─► linear ─► 3 classes
                                                              │
                                  extra loss: attention weights ≈ human-marked words
                          total loss = cross-entropy + 1.0 × attention loss (posts that have rationales)
```

The attention pooling is a learned weighted average over tokens. During training, an auxiliary loss pulls the attention distribution towards the words the annotators marked, so the attention doubles as an explanation with a single forward pass (occlusion needs one pass per word). Code: `src/custom_model.py`, `src/train_custom.py`.

## Pipeline

```
HateXplain (HF parquet, pinned revision)
        │  src/data.py
        ▼
data/processed/*.csv  (id, text, label, rationales, annotator_labels, label_name)
        │
        ├──► TF-IDF + Logistic Regression                       (notebook 02)
        ├──► BERT / XLM-R fine-tuning (HF Trainer)              (src/train_xlmr.py)
        └──► HateNet + ablation, 3 seeds                        (src/train_custom.py, src/run_ablation.py)
                         │
                         ▼
   comparison · significance · error analysis                   (src/analysis.py, src/ablation_report.py)
   explainability (occlusion, attention)                        (src/explain.py, src/hn_faithfulness.py)
                         ▼
                 Streamlit demo (app.py)
```

## Results

### Model comparison (test set, 1,924 posts)

| Model | Seeds | Accuracy | Macro-F1 | F1 Hate | F1 Normal | F1 Offensive |
|---|---|---|---|---|---|---|
| **HateNet, 6 frozen BERT layers** | 3 | 0.703 | **0.693 ± 0.004** | 0.774 | 0.746 | 0.560 |
| HateNet full | 3 | 0.695 | 0.687 ± 0.005 | 0.771 | 0.739 | 0.552 |
| Plain BERT head (same pipeline) | 3 | 0.678 | 0.669 ± 0.010 | 0.757 | 0.724 | 0.525 |
| XLM-R (HF Trainer) | 1 | 0.693 | 0.680 | 0.773 | 0.742 | 0.524 |
| BERT-base (HF Trainer) | 1 | 0.689 | 0.678 | 0.774 | 0.730 | 0.531 |
| TF-IDF + LogReg | 1 | 0.669 | 0.648 | 0.727 | 0.729 | 0.486 |

HateNet and plain-head rows are mean ± standard deviation over seeds 42, 1, 2; per-class F1 is the mean over seeds. BERT and XLM-R were trained once with the standard recipe, so their gap to HateNet is suggestive, not proven.

### Ablation (one change at a time, seed 42)

| Variant | Macro-F1 | Accuracy | Attention AUPRC |
|---|---|---|---|
| Plain BERT head ([CLS]) | 0.671 | 0.677 | – |
| + attention pooling | 0.672 | 0.689 | 0.79 |
| + BiLSTM | 0.677 | 0.688 | 0.73 |
| + rationale-guided attention (full) | 0.693 | 0.704 | 0.92 |
| full + class weights | 0.681 | 0.692 | 0.92 |
| full + 6 frozen layers | 0.698 | 0.707 | 0.91 |

Attention and BiLSTM alone barely change macro-F1; the rationale loss gives about +2 points and makes attention far closer to human rationales. Class weights did not help. Freezing the lower six BERT layers costs nothing and cuts training time from about 7.5 to 4.6 minutes per run on a Colab T4. Attention AUPRC = average precision of the attention weights against the human rationale.

### Statistical significance (paired bootstrap, 2,000 resamples, 95% interval on macro-F1 difference)

| Comparison | Diff | 95% CI |
|---|---|---|
| XLM-R vs TF-IDF baseline | +0.032 | [+0.011, +0.053] |
| HateNet full vs TF-IDF baseline | +0.045 | [+0.023, +0.067] |
| HateNet (6 frozen) vs TF-IDF baseline | +0.050 | [+0.029, +0.071] |
| HateNet full vs plain head, seed 42 | +0.022 | [+0.004, +0.039] |
| HateNet full vs plain head, seed 1 | +0.006 | [−0.011, +0.021] |
| HateNet full vs plain head, seed 2 | +0.027 | [+0.009, +0.045] |
| HateNet (6 frozen) vs BERT (HF Trainer) | +0.019 | [+0.002, +0.037] |

The improvement over the baseline is clear. Over the plain head it is consistent in direction but not significant for every seed (seed 1). Three seeds is a small sample.

### Error analysis (XLM-R, top confusion pairs)

| True | Predicted | Count | Share of errors |
|---|---|---|---|
| Normal | Offensive | 152 | 25.8% |
| Offensive | Normal | 151 | 25.6% |
| Offensive | Hate speech | 124 | 21.0% |
| Hate speech | Offensive | 68 | 11.5% |
| Normal | Hate speech | 58 | 9.8% |

Accuracy by annotator agreement: **81.0%** when all 3 annotators agree · **57.4%** on split 2-1 votes, so many errors reflect genuinely ambiguous labels. Offensive is the hardest class for every model.

### Explainability (300 test posts with human rationales)

Plausibility = AUPRC of word scores against the human rationale. Comprehensiveness = drop in predicted-class probability when the top 20% words are removed (higher is better). Sufficiency = drop when only those words are kept (lower is better).

| Word ranking | Plausibility | Comprehensiveness | Sufficiency |
|---|---|---|---|
| **HateNet attention** (1 forward pass) | 0.926 | 0.557 | 0.009 |
| Occlusion on HateNet | 0.716 | 0.565 | −0.059 |
| Occlusion on BERT | 0.703 | 0.578 | −0.051 |
| Occlusion on XLM-R | 0.679 | 0.640 | −0.016 |
| Occlusion on TF-IDF baseline | 0.650 | 0.368 | −0.235 |
| Random words | 0.277–0.287 | 0.08–0.16 | 0.17–0.47 |

HateNet's attention matches human rationales much better than occlusion does, and it is as faithful as occlusion: removing its top words hurts the prediction as much as removing the occlusion words, and keeping only them preserves it. Occlusion rows for BERT, XLM-R and the baseline come from `src/explain.py`; the attention row comes from `src/hn_faithfulness.py` (one HateNet model, seed 42). Note that attention was trained on the human rationales, so its high plausibility is expected; the comprehensiveness and sufficiency numbers are what show it is not decoration.

## Repo layout

```
notebooks/   01 exploration · 02 baseline · 03 SHAP (baseline) · 04 XLM-R · 05 Colab GPU · 06 comparison and error analysis
src/         config.py · data.py · metrics.py · xlmr.py · train_xlmr.py · evaluate.py · analysis.py · explain.py
             custom_model.py · train_custom.py · run_ablation.py · ablation_report.py · hn_faithfulness.py
results/     tables/ and figures/ (committed) · logs/ (ignored)
data/        processed CSVs (ignored; rebuilt with python -m src.data)
models/      weights and checkpoints (ignored)
app.py       Streamlit demo
```

## Setup

Use Python 3.12 or 3.13 (PyTorch support for 3.14 is not reliable yet).

```bash
python3.13 -m venv .venv313
source .venv313/bin/activate
pip install -r requirements.txt
```

`requirements-lock.txt` is the full pinned environment from the original Mac setup.

## How to run

Training runs need a GPU (Colab T4 is enough, notebook 05 sets it up). Long runs on the Mac's MPS backend stalled silently, so HateNet training uses CUDA or CPU only.

```bash
# 1. Build the processed CSVs (needs internet, once)
python -m src.data

# 2. Baseline: notebooks/02_Baseline_Classifier.ipynb

# 3. BERT / XLM-R (GPU)
python -m src.train_xlmr --model_name bert-base-uncased --tag bert --epochs 4 --train_bs 16 --grad_accum 1 --fp16
python -m src.train_xlmr --tag xlmr_gpu --epochs 4 --train_bs 16 --grad_accum 1 --warmup_ratio 0.1 --fp16

# 4. HateNet: quick check, then the whole ablation (about 1 hour on a T4) and the extra seeds (about 40 min)
python -m src.train_custom --smoke
python -m src.run_ablation --stage design
python -m src.run_ablation --stage seeds        # cls and full, seeds 1 and 2
python -m src.train_custom --tag hn_full_freeze6_s1 --fp16 --epochs 4 --head bilstm_attn --rationale_lambda 1.0 --freeze_layers 6 --seed 1   # likewise seed 2

# 5. Comparison, significance, ablation report
python -m src.evaluate --model models/bert_best --tag bert
python -m src.analysis
python -m src.ablation_report

# 6. Explainability
python -m src.explain --model baseline --tag baseline
python -m src.explain --model models/bert_best --tag bert --max_posts 300
python -m src.train_custom --tag hn_full --fp16 --epochs 4 --head bilstm_attn --rationale_lambda 1.0 --save_model
python -m src.hn_faithfulness --tag hn_full --max_posts 300

# 7. Demo
streamlit run app.py
```

Common settings: max length 128, batch size 16, weight decay 0.01, AdamW with linear schedule, best epoch chosen by validation macro-F1, fp16, seed 42 unless stated. BERT/XLM-R: learning rate 2e-5, 4 epochs (XLM-R with 10% warmup). HateNet: BERT at 2e-5, new layers at 1e-3, 10% warmup, gradient clipping 1.0, dropout 0.3, 4 epochs.

## Roadmap

- [x] Data exploration and processed splits
- [x] TF-IDF + LR baseline
- [x] BERT-base and XLM-RoBERTa fine-tuning, test-set evaluation
- [x] Statistical significance and error analysis
- [x] Occlusion explainability (plausibility and faithfulness)
- [x] HateNet architecture, ablation study and 3-seed evaluation
- [x] Faithfulness of HateNet attention
- [x] Streamlit demo (`app.py`, BERT and XLM-R)
- [ ] HateNet in the demo app
- [ ] Final report

## Limitations

HateXplain posts are English and from 2019-2020 Gab/Twitter, annotator agreement on Hate vs Offensive is imperfect (half of the test posts are split 2-1), and models can over-flag identity terms (for example "muslim" or "gay" in neutral posts). Seeds: three for HateNet and the plain head, one for the BERT, XLM-R and baseline references. HateNet's faithfulness was tested on one model and 300 posts. Results should not be used for real moderation without further bias evaluation.
