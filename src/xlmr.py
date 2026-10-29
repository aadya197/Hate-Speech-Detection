"""Helpers for fine-tuning / evaluating XLM-RoBERTa on HateXplain."""
import inspect
import os

# Must be set before transformers / huggingface_hub are imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # avoids the xet download problem
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for ops MPS lacks

import torch  # noqa: E402
from datasets import Dataset, DatasetDict  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import LABEL_NAMES, MAX_LENGTH, NUM_LABELS  # noqa: E402
from src.data import load_processed  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer(model_name_or_path):
    return AutoTokenizer.from_pretrained(model_name_or_path)


def load_model(model_name_or_path):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=NUM_LABELS,
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )


def build_tokenized_datasets(tokenizer, max_length=MAX_LENGTH, limit=None) -> DatasetDict:
    """Load the processed CSVs and tokenize every split. `limit` is for smoke tests."""
    splits = {}
    for name in ("train", "validation", "test"):
        df = load_processed(name)[["text", "label"]].copy()
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
        if limit:
            df = df.head(limit)
        splits[name] = Dataset.from_pandas(df, preserve_index=False)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return DatasetDict(splits).map(tokenize, batched=True)


def _supported(cls, name) -> bool:
    return name in inspect.signature(cls.__init__).parameters


def make_training_args(output_dir, eval_mode="epoch", **overrides) -> TrainingArguments:
    """Settings used in the project chat. Handles the eval_strategy / evaluation_strategy
    rename between transformers versions."""
    kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # effective batch size 16
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,  # keep disk usage down (each checkpoint is ~1 GB)
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
        dataloader_pin_memory=False,  # silences the MPS pin_memory warning
    )
    kwargs.update(overrides)
    strategy_key = "eval_strategy" if _supported(TrainingArguments, "eval_strategy") else "evaluation_strategy"
    kwargs[strategy_key] = eval_mode
    return TrainingArguments(**kwargs)


def make_trainer(model, args, tokenizer, train_ds=None, eval_ds=None) -> Trainer:
    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    # processing_class replaced tokenizer in newer transformers versions
    kwargs["processing_class" if _supported(Trainer, "processing_class") else "tokenizer"] = tokenizer
    return Trainer(**kwargs)
