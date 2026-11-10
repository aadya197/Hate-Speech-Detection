"""Model comparison and error analysis (needs the *_test_predictions.csv files).

    python -m src.analysis                 # compare all models found, analyse the best one
    python -m src.analysis --best bert     # analyse a specific model

Outputs (results/tables and results/figures):
    model_comparison.csv / .png        accuracy, macro-F1 and per-class F1 of every model
    confusion_matrices_all.png         normalised confusion matrices side by side
    error_confusion_pairs.csv          which true -> predicted mistakes are most common
    error_by_annotator_agreement.csv   accuracy when annotators agreed vs. split 2-1
    error_by_length.csv                accuracy by post length
    error_examples.csv                 most confident mistakes (good slide material)
"""
import argparse
import ast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

from src.config import FIGURES_DIR, LABEL_LIST, MODELS_DIR, TABLES_DIR, ensure_dirs  # noqa: E402
from src.data import load_processed  # noqa: E402

# tag -> name shown in tables and slides
MODEL_NAMES = {
    "baseline": "TF-IDF + LogReg",
    "bert": "BERT-base",
    "xlmr_gpu": "XLM-R (GPU)",
    "xlmr": "XLM-R (Mac run)",
}


def make_baseline_predictions() -> None:
    """Create baseline_test_predictions.csv from the model saved by notebook 02."""
    import joblib

    path = MODELS_DIR / "baseline_tfidf_lr.joblib"
    out = TABLES_DIR / "baseline_test_predictions.csv"
    if out.exists() or not path.exists():
        return
    model = joblib.load(path)
    test = load_processed("test")
    probs = model.predict_proba(test["text"])
    df = pd.DataFrame(
        {"id": test["id"], "label": test["label"].astype(int), "pred": probs.argmax(axis=1)}
    )
    for i, name in enumerate(LABEL_LIST):
        df[f"p_{name}"] = probs[:, i]
    df.to_csv(out, index=False)
    print("Created", out)


def load_preds(tag):
    path = TABLES_DIR / f"{tag}_test_predictions.csv"
    return pd.read_csv(path) if path.exists() else None


def scores(df) -> dict:
    y, p = df["label"], df["pred"]
    row = {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
    }
    for i, name in enumerate(LABEL_LIST):
        row[f"f1_{name}"] = f1_score(y, p, labels=[i], average=None)[0]
    return row


def fallback_scores(tag):
    """Models without a predictions file (the Mac XLM-R run): use the saved metric tables."""
    m, c = TABLES_DIR / f"{tag}_metrics.csv", TABLES_DIR / f"{tag}_per_class_test.csv"
    if not (m.exists() and c.exists()):
        return None
    mm = pd.read_csv(m)
    mm = mm[mm["split"] == "test"].iloc[0]
    cc = pd.read_csv(c).set_index("Class")["F1-score"]
    row = {"accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"]}
    for name in LABEL_LIST:
        row[f"f1_{name}"] = cc[name]
    return row


def compare_models(preds: dict) -> pd.DataFrame:
    rows = []
    for tag, name in MODEL_NAMES.items():
        r = scores(preds[tag]) if tag in preds else fallback_scores(tag)
        if r:
            rows.append({"model": name, "tag": tag, **r})
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    table.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    cols = ["macro_f1"] + [f"f1_{n}" for n in LABEL_LIST]
    labels = ["Macro-F1"] + LABEL_LIST
    x = np.arange(len(cols))
    w = 0.8 / max(len(table), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, r in table.iterrows():
        ax.bar(x + i * w, [r[c] for c in cols], w, label=r["model"])
    ax.set_xticks(x + w * (len(table) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score (test set)")
    ax.set_title("Model comparison on the HateXplain test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return table


def plot_confusions(preds: dict) -> None:
    tags = [t for t in MODEL_NAMES if t in preds]
    if not tags:
        return
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 4.3), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        d = preds[tag]
        cm = confusion_matrix(d["label"], d["pred"], labels=[0, 1, 2], normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(LABEL_LIST, rotation=20); ax.set_yticklabels(LABEL_LIST)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(MODEL_NAMES[tag])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices_all.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def error_analysis(tag: str, pred: pd.DataFrame) -> None:
    test = load_processed("test")
    df = test.merge(pred[["id", "pred"] + [c for c in pred.columns if c.startswith("p_")]], on="id")
    df["correct"] = df["pred"] == df["label"]
    df["true_name"] = df["label"].map(dict(enumerate(LABEL_LIST)))
    df["pred_name"] = df["pred"].map(dict(enumerate(LABEL_LIST)))
    df["confidence"] = df[[c for c in df.columns if c.startswith("p_")]].max(axis=1)

    # 1. Most common mistakes
    wrong = df[~df["correct"]]
    pairs = (wrong.groupby(["true_name", "pred_name"]).size().rename("count").reset_index()
             .sort_values("count", ascending=False))
    pairs["share_of_errors"] = (pairs["count"] / len(wrong)).round(3)
    pairs.to_csv(TABLES_DIR / "error_confusion_pairs.csv", index=False)

    # 2. Does annotator disagreement explain the errors?
    def top_votes(v):
        votes = ast.literal_eval(v) if isinstance(v, str) else list(v)
        return max(votes.count(k) for k in set(votes))

    df["agreement"] = df["annotator_labels"].map(top_votes).map({3: "all 3 agree", 2: "split 2-1"}).fillna("other")
    agree = df.groupby("agreement")["correct"].agg(posts="size", accuracy="mean").reset_index()
    agree.to_csv(TABLES_DIR / "error_by_annotator_agreement.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(agree["agreement"], agree["accuracy"], color=["#4C78A8", "#F58518", "#999999"][: len(agree)])
    for i, r in agree.iterrows():
        ax.text(i, r["accuracy"] + 0.01, f"{r['accuracy']:.2f}\n(n={int(r['posts'])})", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
    ax.set_title(f"{MODEL_NAMES[tag]}: accuracy vs annotator agreement")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_by_agreement.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Post length
    df["n_words"] = df["text"].astype(str).str.split().str.len()
    df["length_bin"] = pd.cut(df["n_words"], [0, 10, 20, 30, 1000], labels=["1-10", "11-20", "21-30", "31+"])
    (df.groupby("length_bin", observed=True)["correct"].agg(posts="size", accuracy="mean").reset_index()
     .to_csv(TABLES_DIR / "error_by_length.csv", index=False))

    # 4. Most confident mistakes, from the three most common error types
    ex = []
    for _, r in pairs.head(3).iterrows():
        sub = wrong[(wrong["true_name"] == r["true_name"]) & (wrong["pred_name"] == r["pred_name"])]
        ex.append(sub.sort_values("confidence", ascending=False).head(10))
    if ex:
        pd.concat(ex)[["id", "text", "true_name", "pred_name", "confidence", "annotator_labels"]].round(3).to_csv(
            TABLES_DIR / "error_examples.csv", index=False)

    print(f"\n=== Error analysis for {MODEL_NAMES[tag]} (test accuracy {df['correct'].mean():.3f}) ===")
    print(pairs.head(5).to_string(index=False))
    print("\nAccuracy by annotator agreement:")
    print(agree.to_string(index=False))


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    """Is model A really better than model B? Resample the SAME test posts many times and look
    at the difference in macro-F1 / accuracy. If the 95% interval excludes 0, the gain is real."""
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    y, pa, pb = m["label_a"].values, m["pred_a"].values, m["pred_b"].values
    rng = np.random.default_rng(seed)
    d_f1, d_acc = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d_f1.append(f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro"))
        d_acc.append((pa[i] == y[i]).mean() - (pb[i] == y[i]).mean())
    out = {"posts": len(y)}
    for name, d, point in (
        ("macro_f1", d_f1, f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")),
        ("accuracy", d_acc, (pa == y).mean() - (pb == y).mean()),
    ):
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[f"{name}_diff"], out[f"{name}_ci_low"], out[f"{name}_ci_high"] = point, lo, hi
        out[f"{name}_p_better"] = float((np.array(d) > 0).mean())
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", default=None, help="tag to analyse (default: highest macro-F1 with predictions)")
    args = parser.parse_args(argv)
    ensure_dirs()
    make_baseline_predictions()
    preds = {t: d for t in MODEL_NAMES if (d := load_preds(t)) is not None}
    table = compare_models(preds)
    print(table.round(3).to_string(index=False))
    plot_confusions(preds)
    candidates = [t for t in table["tag"] if t in preds]
    best = args.best or (candidates[0] if candidates else None)
    if best is None:
        print("No *_test_predictions.csv found yet. Train a model first.")
        return
    error_analysis(best, preds[best])
    if "baseline" in preds and best != "baseline":
        r = paired_bootstrap(preds[best], preds["baseline"])
        r = {"model": MODEL_NAMES[best], "vs": MODEL_NAMES["baseline"], **r}
        pd.DataFrame([r]).to_csv(TABLES_DIR / "significance_vs_baseline.csv", index=False)
        print(f"\n=== {MODEL_NAMES[best]} vs {MODEL_NAMES['baseline']} (paired bootstrap, {r['posts']} test posts) ===")
        for k in ("macro_f1", "accuracy"):
            print(f"{k}: diff {r[k + '_diff']:+.3f}  95% CI [{r[k + '_ci_low']:+.3f}, {r[k + '_ci_high']:+.3f}]"
                  f"  P(better) = {r[k + '_p_better']:.3f}")
    print("\nFiles written to", TABLES_DIR, "and", FIGURES_DIR)


if __name__ == "__main__":
    main()
