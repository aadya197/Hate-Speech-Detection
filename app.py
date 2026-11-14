"""Streamlit demo: type a post, see the predicted label and which words drove the decision.

Run from the project folder:
    streamlit run app.py
"""
import html

import numpy as np
import streamlit as st

from src.config import LABEL_LIST, MODELS_DIR, SEED
from src.explain import explain_post, load_predictor
from src.hn_infer import load_hatenet

MAX_WORDS = 60
COLORS = {"Hate speech": "#C0392B", "Offensive": "#D97706", "Normal": "#2E8B57"}

# (button title, post, note shown under the post). Notes only state results we have checked.
EXAMPLES = [
    ("Friendly message", "have a wonderful day my friend", "A harmless post."),
    ("Direct insult", "i can't believe how worthless you are, you complete moron",
     "HateNet calls this Offensive and its attention lands on the insult."),
    ("Neutral, mentions a group", "my neighbour is muslim and very kind",
     "Neutral sentences like this are not flagged in our identity-term check."),
    ("Known failure", "stop being so stupid, nobody likes you",
     "Annotators would call this Offensive, but HateNet says Normal. Attention and occlusion disagree here."),
]

st.set_page_config(page_title="Explainable Hate Speech Detection", page_icon="🛡️", layout="centered")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 820px;}
    .badge {display:inline-block; padding:6px 18px; border-radius:999px; color:white; font-weight:700;
            font-size:1.5rem; letter-spacing:.3px;}
    .conf {opacity:.75; margin-left:12px; font-size:1.05rem;}
    .prob-row {display:flex; align-items:center; gap:10px; margin:5px 0; font-size:.95rem;}
    .prob-name {width:95px;}
    .prob-track {flex:1; height:14px; background:rgba(128,128,128,.25); border-radius:7px; overflow:hidden;}
    .prob-fill {height:100%; border-radius:7px;}
    .prob-val {width:48px; text-align:right; opacity:.8;}
    .legend {height:10px; border-radius:5px; width:220px;
             background:linear-gradient(90deg, rgba(220,50,50,0), rgba(220,50,50,1));}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(spec):
    """Returns (predict, attention). attention is None for models without built-in attention."""
    if spec.startswith("hatenet:"):
        return load_hatenet(spec.split(":", 1)[1])
    return load_predictor(spec), None


def available_models():
    models = {}
    if (MODELS_DIR / "hn_full_best.pt").exists():
        models["HateNet (ours)"] = "hatenet:hn_full"
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
            f"<span style='background:rgba(220,50,50,{a:.2f});padding:3px 6px;border-radius:5px;'>"
            f"{html.escape(w)}</span>"
        )
    return "<div style='line-height:2.3;font-size:1.15rem'>" + " ".join(spans) + "</div>"


def prob_bars(probs):
    rows = []
    for name, p in zip(LABEL_LIST, probs):
        rows.append(
            f"<div class='prob-row'><div class='prob-name'>{name}</div>"
            f"<div class='prob-track'><div class='prob-fill' style='width:{p * 100:.0f}%;background:{COLORS[name]}'></div></div>"
            f"<div class='prob-val'>{p:.0%}</div></div>"
        )
    return "".join(rows)


def set_example(text):
    st.session_state["text"] = text
    st.session_state["analysed"] = True


def start_analysis():
    st.session_state["analysed"] = True


# ---------- page ----------
st.title("🛡️ Explainable Hate Speech Detection")
st.caption("Classifies a post as Hate speech, Offensive or Normal and highlights the words behind the decision. "
           "Trained on HateXplain (Gab and Twitter, 2019-20). Research demo, not for real moderation.")

models = available_models()
if not models:
    st.error("No trained model found in models/. Train one first (see README) or run notebook 02.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Model", list(models))
    use_attention = False
    if models[choice].startswith("hatenet:"):
        mode = st.radio("Explanation", ["Built-in attention", "Occlusion (word by word)"],
                        help="Attention comes from one forward pass. Occlusion removes each word in turn and "
                             "measures how much the prediction drops; it is slower but works for any model.")
        use_attention = mode.startswith("Built-in")
    st.divider()
    st.markdown("**How to read the colours**")
    st.markdown("<div class='legend'></div><div style='display:flex;justify-content:space-between;width:220px;"
                "font-size:.8rem;opacity:.7'><span>less</span><span>more important</span></div>",
                unsafe_allow_html=True)
    with st.expander("About HateNet"):
        st.markdown(
            "BERT + BiLSTM + attention pooling. During training the attention is pushed towards the words "
            "human annotators marked, so it doubles as an explanation. On 300 test posts removing its top "
            "words hurts the prediction about as much as removing the words found by occlusion."
        )

st.markdown("##### Try an example")
cols = st.columns(len(EXAMPLES))
for i, (col, (title, text, _)) in enumerate(zip(cols, EXAMPLES)):
    col.button(title, key=f"ex{i}", use_container_width=True, on_click=set_example, args=(text,))

if "text" not in st.session_state:
    st.session_state["text"] = ""
text = st.text_area("Post", key="text", height=100, placeholder="Type or paste a post (English)...")
st.button("Analyse", type="primary", on_click=start_analysis)

note = next((n for _, t, n in EXAMPLES if t == text), None)
if note:
    st.caption(f"ℹ️ {note}")

if st.session_state.get("analysed"):
    words = text.split()[:MAX_WORDS]
    if not words:
        st.warning("Please type a post first.")
    else:
        predictor, attention = get_predictor(models[choice])
        with st.spinner("Explaining..."):
            probs = predictor([" ".join(words)])[0]
            if use_attention and attention is not None:
                c, p0, scores = int(probs.argmax()), float(probs.max()), attention(words)
            else:
                c, p0, scores = explain_post(predictor, words, np.random.default_rng(SEED))

        st.divider()
        label = LABEL_LIST[c]
        st.markdown(
            f"<span class='badge' style='background:{COLORS[label]}'>{label}</span>"
            f"<span class='conf'>{p0:.0%} confident</span>",
            unsafe_allow_html=True,
        )
        if p0 < 0.5:
            st.info("Low confidence: the model is unsure, treat this prediction with care.")
        st.markdown(prob_bars(probs), unsafe_allow_html=True)

        st.subheader("Why this prediction?")
        st.markdown(heatmap(words, scores), unsafe_allow_html=True)
        order = np.argsort(-np.asarray(scores))[:3]
        top_words = [words[i].strip(".,!?;:\"'") for i in order if scores[i] > 0]
        if top_words:
            st.markdown("**Most influential words:** " + ", ".join(f"`{w}`" for w in top_words))
        if use_attention:
            st.caption("Red = the model's attention. HateNet's attention was trained to match the words human "
                       "annotators marked, and tests show that removing these words really changes the prediction.")
        else:
            st.caption("Red = words the model relied on: removing a word lowers the probability of the predicted "
                       "label. The darker the red, the bigger the drop.")
        if len(text.split()) > MAX_WORDS:
            st.info(f"Only the first {MAX_WORDS} words were analysed.")
