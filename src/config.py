"""Shared paths and constants. Every path is relative to the repo root,
so the project runs the same on a Mac, Colab or any other machine."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

MODELS_DIR = PROJECT_DIR / "models"

SEED = 42
MAX_LENGTH = 128

# HateXplain labels (majority vote of the 3 annotators)
LABEL_NAMES = {0: "Hate speech", 1: "Normal", 2: "Offensive"}
LABEL_LIST = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]
NUM_LABELS = len(LABEL_NAMES)

# Pinned HuggingFace dataset revision (same one used in notebook 01)
HATEXPLAIN_REVISION = "87e4e5d596ea66919447cb1e1bd6228a1c5e23e5"
HATEXPLAIN_BASE_URL = (
    "https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain/resolve/"
    f"{HATEXPLAIN_REVISION}/plain_text/"
)

XLMR_MODEL_NAME = "FacebookAI/xlm-roberta-base"


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
