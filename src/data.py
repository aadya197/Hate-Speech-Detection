"""HateXplain loading, label building and rationale helpers.

Build the processed CSVs once (needs internet):
    python -m src.data
"""
import ast
from collections import Counter

import numpy as np
import pandas as pd

from src.config import (
    HATEXPLAIN_BASE_URL,
    LABEL_NAMES,
    PROCESSED_DIR,
    ensure_dirs,
)

SPLITS = {"train": "train", "validation": "validation", "test": "test"}


def majority_label(labels):
    """Majority vote of the annotator labels (HateXplain has no ties in practice)."""
    return Counter(labels).most_common(1)[0][0]


def prepare_split(hf_split) -> pd.DataFrame:
    rows = []
    for ex in hf_split:
        rows.append(
            {
                "id": ex["id"],
                "text": " ".join(ex["post_tokens"]),
                "label": majority_label(ex["annotators"]["label"]),
                "rationales": ex["rationales"],
                "annotator_labels": ex["annotators"]["label"],
            }
        )
    df = pd.DataFrame(rows)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    return df


def build_processed_csvs() -> None:
    """Download HateXplain (pinned revision) and write the three processed CSVs."""
    from datasets import load_dataset  # imported here so other helpers work without it

    ensure_dirs()
    data_files = {
        name: f"{HATEXPLAIN_BASE_URL}hatexplain-{name}.parquet" for name in SPLITS
    }
    dataset = load_dataset("parquet", data_files=data_files)
    for name in SPLITS:
        df = prepare_split(dataset[name])
        path = PROCESSED_DIR / f"hatexplain_{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name}: {df.shape} -> {path}")


def load_processed(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"hatexplain_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data` first to build it."
        )
    return pd.read_csv(path)


def parse_rationales(value):
    """CSV stores rationales as a string; turn it back into a list of 0/1 masks."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return value if value is not None else []


def consensus_mask(rationales, tokens) -> np.ndarray:
    """Tokens selected by MORE THAN HALF of the annotators.

    One definition used everywhere (3 annotators -> at least 2 votes).
    Returns an all-zero mask for Normal posts, which have no rationales.
    """
    masks = [m for m in rationales if len(m) == len(tokens)]
    if not masks:
        return np.zeros(len(tokens), dtype=int)
    votes = np.asarray(masks, dtype=int).sum(axis=0)
    return (votes > len(masks) / 2).astype(int)


if __name__ == "__main__":
    build_processed_csvs()
