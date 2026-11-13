"""Streamlit demo: type a post, see the predicted label and which words drove the decision.

Run from the project folder:
    streamlit run app.py
"""
import html
from pathlib import Path

import numpy as np
import streamlit as st

from src.config import LABEL_LIST, MODELS_DIR, SEED
from src.explain import explain_post, load_predictor
from src.hn_infer import load_hatenet

MAX_WORDS = 60

EXAMPLES = [
    "have a great day everyone, see you all tomorrow",
    "i hate mondays they ruin my whole week",
    "you are such an idiot, shut up and go away",
    "this new policy is a disaster and the people behind it should be ashamed",
]

st.set_page_config(page_title="Explainable Hate Speech Detection", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(spec):
    """Returns (predict, attention). attention is None for models without built-in attention."""
    if spec.startswith("hatenet:"):
        return load_hatenet(spec.split(":", 1)[1])
    return load_predictor(spec), None


def available_models():
    models = {}
    if (MODELS_DIR / "hn_full_best.pt").exists():
        models["HateNet (ours: BERT + BiLSTM + rationale-guided attention)"] = "hatenet:hn_full"
    if (MODELS_DIR / "bert_best").exists():
        models["BERT (fine-tuned)"] = str(MODELS_DIR / "bert_best")
    if (MODELS_DIR / "xlmr_gpu_best").exists():
        models["XLM-RoBERTa (fine-tuned)"] = str(MODELS_DIR / "xlmr_gpu_best")
    if (MODELS_DIR / "baseline_tfidf_lr.joblib").exists():
        models["TF-IDF + Logistic Regression (baseline)"] = "baseline"
    return models


def heatmap(words, scores):
    top = max(float(np.max(scores)), 1e-9)
    spans = []
    for w, s in zip(words, scores):
        a = max(0.0, float(s)) / top
        spans.append(
            f"<span style='background:rgba(220,50,50,{a:.2f});padding:2px 4px;border-radius:4px;'>"
            f"{html.escape(w)}</span>"
        )
    return "<div style='line-height:2.1;font-size:1.1rem'>" + " ".join(spans) + "</div>"


st.title("Explainable Hate Speech Detection")
st.caption(
    "Fine-tuned on HateXplain (Gab and Twitter posts). Research demo: it can be wrong, "
    "especially on sarcasm, slang and posts about identity groups. Not for real moderation."
)

models = available_models()
if not models:
    st.error("No trained model found in models/. Train one first (see README) or run notebook 02.")
    st.stop()

choice = st.selectbox("Model", list(models))
use_attention = False
if models[choice].startswith("hatenet:"):
    use_attention = st.radio(
        "Explanation", ["Built-in attention (1 pass)", "Occlusion (word-by-word, slower)"], horizontal=True
    ).startswith("Built-in")

if "text" not in st.session_state:
    st.session_state["text"] = EXAMPLES[0]
cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    if col.button(f"Example {i + 1}", use_container_width=True):
        st.session_state["text"] = ex

text = st.text_area("Post", key="text", height=100)

if st.button("Analyse", type="primary"):
    words = text.split()[:MAX_WORDS]
    if not words:
        st.warning("Please type a post first.")
    else:
        predictor, attention = get_predictor(models[choice])
        probs = predictor([" ".join(words)])[0]
        if use_attention and attention is not None:
            c, p0, scores = int(probs.argmax()), float(probs.max()), attention(words)
        else:
            c, p0, scores = explain_post(predictor, words, np.random.default_rng(SEED))

        left, right = st.columns([1, 2])
        left.metric("Prediction", LABEL_LIST[c], f"{p0:.0%} confident", delta_color="off")
        right.bar_chart({name: float(p) for name, p in zip(LABEL_LIST, probs)}, horizontal=True)

        st.subheader("Why this prediction?")
        st.markdown(heatmap(words, scores), unsafe_allow_html=True)
        if use_attention:
            st.caption(
                "Red = the model's attention. HateNet's attention was trained to match the words human "
                "annotators marked, and tests show that removing these words really changes the prediction."
            )
        else:
            st.caption(
                "Red = words the model relied on: removing a word lowers the probability of the predicted "
                "label. The darker the red, the bigger the drop."
            )
        if len(text.split()) > MAX_WORDS:
            st.info(f"Only the first {MAX_WORDS} words were analysed.")
