"""Metrics and plots shared by the XLM-R training and evaluation scripts."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import LABEL_LIST


def compute_metrics(eval_pred) -> dict:
    """Hugging Face Trainer callback: macro metrics (macro-F1 selects the best epoch)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def per_class_table(labels, preds) -> pd.DataFrame:
    """Same layout as results/tables/baseline_metrics.csv so models are comparable."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], zero_division=0
    )
    return pd.DataFrame(
        {
            "Class": LABEL_LIST,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Support": support,
        }
    )


def save_confusion_matrix(labels, preds, title, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_LIST).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_predictions(split, logits, path, limit=None) -> None:
    """Save per-post predictions (id, true label, predicted label, class probabilities).
    Needed for error analysis. Row order matches data/processed/hatexplain_<split>.csv."""
    from src.data import load_processed

    df = load_processed(split)
    if limit:
        df = df.head(limit)
    logits = np.asarray(logits, dtype=float)
    assert len(df) == len(logits), f"{len(df)} posts but {len(logits)} predictions"
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    out = pd.DataFrame(
        {"id": df["id"].values, "label": df["label"].astype(int).values, "pred": logits.argmax(axis=1)}
    )
    for i, name in enumerate(LABEL_LIST):
        out[f"p_{name}"] = probs[:, i]
    out.to_csv(path, index=False)
