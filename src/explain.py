"""Explainability: do the model's important words match what human annotators marked?

Method: word-level occlusion. For every word we remove it and measure how much the predicted
class probability drops. This works for ANY model (TF-IDF baseline, BERT, XLM-R) and lines up
exactly with the HateXplain words, so the comparison between models is fair.

    python -m src.explain --model baseline --tag baseline
    python -m src.explain --model models/bert_best --tag bert --max_posts 300

Metrics (per post, then averaged), on test posts whose gold label is Hate or Offensive:
  Plausibility  (agreement with human rationales)
    auprc      average precision of the word scores against the human rationale words
    topk_f1    overlap of the top-k words with the human words, k = size of the human rationale
  Faithfulness  (does the model really rely on those words?)
    comp       comprehensiveness: probability drop when the top 20% words are REMOVED (higher = better)
    suff       sufficiency: probability drop when ONLY the top 20% words are KEPT (lower = better)
  Every metric is also computed for random word scores ("rand_*") so you can see what chance looks like.

Outputs: results/tables/explain_<tag>_posts.csv, explain_summary.csv,
         results/figures/explainability_comparison.png, results/explain_examples_<tag>.html
"""
import argparse
import html
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.config import FIGURES_DIR, LABEL_LIST, MODELS_DIR, RESULTS_DIR, SEED, TABLES_DIR, ensure_dirs  # noqa: E402
from src.data import consensus_mask, load_processed, parse_rationales  # noqa: E402

TOP_FRACTION = 0.2


def load_predictor(spec: str):
    """Return f(list_of_texts) -> array (n, 3) of class probabilities."""
    if spec == "baseline":
        import joblib

        model = joblib.load(MODELS_DIR / "baseline_tfidf_lr.joblib")
        return lambda texts: model.predict_proba(list(texts))

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(spec)
    model = AutoModelForSequenceClassification.from_pretrained(spec).to(device).eval()
    print("Explaining with", spec, "on", device)

    def predict(texts, batch=64):
        texts, out = list(texts), []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = tok(texts[i:i + batch], truncation=True, max_length=128, padding=True, return_tensors="pt")
                logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
                out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    return predict


def drop_words(words, idx):
    idx = set(idx)
    return " ".join(w for i, w in enumerate(words) if i not in idx)


def keep_words(words, idx):
    return " ".join(words[i] for i in sorted(idx))


def explain_post(predict, words, rng):
    """Occlusion scores for the predicted class, plus faithfulness numbers."""
    n = len(words)
    probs = predict([" ".join(words)] + [drop_words(words, [i]) for i in range(n)])
    c = int(probs[0].argmax())
    p0 = probs[0, c]
    scores = p0 - probs[1:, c]
    return c, p0, scores


def faithfulness(predict, words, c, p0, scores, rng):
    n = len(words)
    k = max(1, math.ceil(TOP_FRACTION * n))
    jitter = lambda s: s + 1e-9 * rng.random(n)  # break ties randomly, not by position
    top = np.argsort(-jitter(scores))[:k]
    rand = rng.permutation(n)[:k]
    p = predict([drop_words(words, top), keep_words(words, top), drop_words(words, rand), keep_words(words, rand)])[:, c]
    return {"comp": p0 - p[0], "suff": p0 - p[1], "rand_comp": p0 - p[2], "rand_suff": p0 - p[3]}


def select_posts(max_posts):
    test = load_processed("test")
    rows = []
    for _, r in test.iterrows():
        if int(r["label"]) == 1:  # Normal posts have no rationale
            continue
        words = str(r["text"]).split()
        mask = consensus_mask(parse_rationales(r["rationales"]), words)
        if mask.sum() > 0 and len(mask) == len(words):
            rows.append((r["id"], int(r["label"]), words, mask))
    rng = np.random.default_rng(SEED)
    if max_posts and len(rows) > max_posts:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), max_posts, replace=False))]
    return rows


def run(model_spec, tag, max_posts):
    ensure_dirs()
    predict = load_predictor(model_spec)
    rng = np.random.default_rng(SEED)
    posts, results, store = select_posts(max_posts), [], []
    print(f"Explaining {len(posts)} test posts with human rationales")
    for n_done, (pid, label, words, mask) in enumerate(posts, 1):
        c, p0, scores = explain_post(predict, words, rng)
        jittered = scores + 1e-9 * rng.random(len(words))
        r = int(mask.sum())
        top_r = np.argsort(-jittered)[:r]
        rand_scores = rng.random(len(words))
        row = {
            "id": pid, "label": label, "pred": c, "n_words": len(words), "n_rationale": r,
            "auprc": average_precision_score(mask, jittered),
            "topk_f1": mask[top_r].sum() / r,
            "rand_auprc": average_precision_score(mask, rand_scores),
            "rand_topk_f1": mask[np.argsort(-rand_scores)[:r]].sum() / r,
            "base_rate": r / len(words),
        }
        row.update(faithfulness(predict, words, c, p0, scores, rng))
        results.append(row)
        store.append((pid, label, c, words, mask, scores))
        if n_done % 50 == 0:
            print(f"  {n_done}/{len(posts)}")
    df = pd.DataFrame(results)
    df.to_csv(TABLES_DIR / f"explain_{tag}_posts.csv", index=False)

    metrics = ["auprc", "topk_f1", "comp", "suff", "rand_auprc", "rand_topk_f1", "rand_comp", "rand_suff", "base_rate"]
    summary = {"tag": tag, "posts": len(df), **{m: df[m].mean() for m in metrics}}
    summary.update({f"{m}_sem": df[m].std() / math.sqrt(len(df)) for m in ("auprc", "comp", "suff")})
    path = TABLES_DIR / "explain_summary.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[old["tag"] != tag]
    pd.concat([old, pd.DataFrame([summary])], ignore_index=True).to_csv(path, index=False)
    for k, v in summary.items():
        print(f"  {k:14s} {v:.3f}" if isinstance(v, float) else f"  {k:14s} {v}")

    write_html(tag, store)
    plot_summary()


def write_html(tag, store, n=8):
    """Word heatmap: red = important for the model, bold + underline = marked by human annotators."""
    parts = ["<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:900px;margin:20px auto'>",
             f"<h2>{html.escape(tag)}: model attention vs human rationales</h2>",
             "<p>Red background = word the model relied on (occlusion). <b><u>Bold underline</u></b> = marked by the annotators.</p>"]
    for pid, label, c, words, mask, scores in store[:n]:
        top = max(scores.max(), 1e-9)
        spans = []
        for w, m, s in zip(words, mask, scores):
            a = max(0.0, s) / top
            style = f"background:rgba(220,50,50,{a:.2f});padding:1px 3px;border-radius:3px;"
            if m:
                style += "font-weight:bold;text-decoration:underline;"
            spans.append(f"<span style='{style}'>{html.escape(w)}</span>")
        parts.append(f"<p><small>true: {LABEL_LIST[label]} | predicted: {LABEL_LIST[c]}</small><br>{' '.join(spans)}</p>")
    parts.append("</body></html>")
    (RESULTS_DIR / f"explain_examples_{tag}.html").write_text("\n".join(parts), encoding="utf-8")


def plot_summary():
    path = TABLES_DIR / "explain_summary.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [("auprc", "rand_auprc", "Plausibility: AUPRC\n(higher = closer to humans)"),
             ("comp", "rand_comp", "Comprehensiveness\n(higher = relies on these words)"),
             ("suff", "rand_suff", "Sufficiency\n(lower = these words are enough)")]
    for ax, (m, rm, title) in zip(axes, specs):
        names = list(s["tag"]) + ["random"]
        vals = list(s[m]) + [s[rm].mean()]
        ax.bar(names, vals, color=["#4C78A8"] * len(s) + ["#BBBBBB"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "explainability_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="'baseline' or path of a fine-tuned model folder")
    p.add_argument("--tag", required=True)
    p.add_argument("--max_posts", type=int, default=300)
    a = p.parse_args(argv)
    run(a.model, a.tag, a.max_posts)


if __name__ == "__main__":
    main()
