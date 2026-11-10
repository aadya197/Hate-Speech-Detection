"""Fine-tune a transformer (XLM-RoBERTa by default, or BERT etc.) on HateXplain.

Quick check that everything works (a couple of minutes):
    python -m src.train_xlmr --smoke

On a GPU (Colab / Kaggle), about 30-45 minutes for 4 epochs:
    python -m src.train_xlmr --model_name bert-base-uncased --tag bert \
        --epochs 4 --train_bs 16 --grad_accum 1 --fp16

On the Mac (slow, and long runs can silently stall on MPS; prefer a GPU):
    caffeinate -i nohup python -m src.train_xlmr > results/logs/xlmr_train.log 2>&1 &

If it gets interrupted, continue from the last checkpoint:
    python -m src.train_xlmr --resume
"""
import argparse

import pandas as pd

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    SEED,
    TABLES_DIR,
    XLMR_MODEL_NAME,
    ensure_dirs,
)
from transformers import TrainerCallback

from src.metrics import per_class_table, save_confusion_matrix, save_predictions
from src.xlmr import (
    build_tokenized_datasets,
    get_device,
    load_model,
    load_tokenizer,
    make_trainer,
    make_training_args,
)


class StallWatch(TrainerCallback):
    """Warn loudly if two evaluations give exactly the same metrics: the model is not learning
    (this happened on the Mac's MPS backend: epoch 1 and 2 were identical)."""

    def __init__(self):
        self.last = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        cur = {k: round(v, 6) for k, v in (metrics or {}).items() if k in ("eval_loss", "eval_macro_f1")}
        if self.last is not None and cur == self.last:
            print("\n" + "!" * 70)
            print("WARNING: evaluation metrics are IDENTICAL to the previous epoch.")
            print("The model is probably not learning. Stop and check the setup.")
            print("!" * 70 + "\n")
        self.last = cur


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=XLMR_MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bs", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="fraction of steps for learning-rate warmup (0.1 helps XLM-R)")
    parser.add_argument("--fp16", action="store_true", help="mixed precision (CUDA GPUs only)")
    parser.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    parser.add_argument("--tag", default="xlmr", help="prefix for output files")
    args = parser.parse_args()

    if args.smoke:
        args.tag += "_smoke"  # never overwrite real results with smoke-test output
    ensure_dirs()
    from transformers import set_seed

    set_seed(SEED)
    print("Device:", get_device())

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name)
    data = build_tokenized_datasets(tokenizer, limit=64 if args.smoke else None)
    print({k: len(v) for k, v in data.items()})

    ckpt_dir = MODELS_DIR / f"{args.tag}_checkpoints"
    training_args = make_training_args(
        ckpt_dir,
        num_train_epochs=1 if args.smoke else args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=1 if args.smoke else 100,
        fp16=args.fp16,
        warmup_ratio=args.warmup_ratio,
    )
    trainer = make_trainer(model, training_args, tokenizer, data["train"], data["validation"])
    trainer.add_callback(StallWatch())

    train_result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    print("\nTraining completed!")
    print(train_result.metrics)

    # Save the best model (selected by validation macro-F1) so it can be reloaded later.
    best_dir = MODELS_DIR / f"{args.tag}_best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print("Saved best model to", best_dir)

    # Training curve data for the report
    pd.DataFrame(trainer.state.log_history).to_csv(TABLES_DIR / f"{args.tag}_training_log.csv", index=False)

    # Final numbers: validation (used for model selection) and test (report this one)
    rows = []
    for split in ("validation", "test"):
        pred = trainer.predict(data[split])
        labels, preds = pred.label_ids, pred.predictions.argmax(-1)
        m = pred.metrics
        save_predictions(
            split, pred.predictions, TABLES_DIR / f"{args.tag}_{split}_predictions.csv",
            limit=64 if args.smoke else None,
        )
        rows.append(
            {
                "split": split,
                "accuracy": m["test_accuracy"],
                "macro_precision": m["test_macro_precision"],
                "macro_recall": m["test_macro_recall"],
                "macro_f1": m["test_macro_f1"],
            }
        )
        print(f"\n{split}: macro-F1 = {m['test_macro_f1']:.4f}, accuracy = {m['test_accuracy']:.4f}")
        if split == "test":
            table = per_class_table(labels, preds)
            table.to_csv(TABLES_DIR / f"{args.tag}_per_class_test.csv", index=False)
            save_confusion_matrix(
                labels, preds, f"{args.model_name.split('/')[-1]}: Test Confusion Matrix",
                FIGURES_DIR / f"{args.tag}_confusion_matrix_test.png",
            )
            print(table.to_string(index=False))
    pd.DataFrame(rows).to_csv(TABLES_DIR / f"{args.tag}_metrics.csv", index=False)
    print("\nResults written to", TABLES_DIR)


if __name__ == "__main__":
    main()
