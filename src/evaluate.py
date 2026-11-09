"""Evaluate a saved fine-tuned model on the test set (no retraining).

    python -m src.evaluate                      # uses models/xlmr_best
    python -m src.evaluate --model models/xlmr_best --split validation
"""
import argparse

import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, TABLES_DIR, ensure_dirs
from src.metrics import per_class_table, save_confusion_matrix, save_predictions
from src.xlmr import (
    build_tokenized_datasets,
    load_model,
    load_tokenizer,
    make_trainer,
    make_training_args,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODELS_DIR / "xlmr_best"))
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--tag", default="xlmr")
    args = parser.parse_args()

    ensure_dirs()
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model)
    data = build_tokenized_datasets(tokenizer)
    training_args = make_training_args(
        MODELS_DIR / "eval_tmp", eval_mode="no", load_best_model_at_end=False, save_strategy="no"
    )
    trainer = make_trainer(model, training_args, tokenizer)

    pred = trainer.predict(data[args.split])
    labels, preds = pred.label_ids, pred.predictions.argmax(-1)
    table = per_class_table(labels, preds)
    table.to_csv(TABLES_DIR / f"{args.tag}_per_class_{args.split}.csv", index=False)
    save_predictions(args.split, pred.predictions, TABLES_DIR / f"{args.tag}_{args.split}_predictions.csv")
    save_confusion_matrix(
        labels, preds, f"XLM-RoBERTa: {args.split.title()} Confusion Matrix",
        FIGURES_DIR / f"{args.tag}_confusion_matrix_{args.split}.png",
    )
    m = pred.metrics
    pd.DataFrame(
        [{"split": args.split, **{k.replace("test_", ""): v for k, v in m.items() if k.startswith("test_") and k.split("_", 1)[1] in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}}]
    ).to_csv(TABLES_DIR / f"{args.tag}_metrics_{args.split}.csv", index=False)
    print(f"{args.split}: macro-F1 = {m['test_macro_f1']:.4f}, accuracy = {m['test_accuracy']:.4f}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
