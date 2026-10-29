"""Fine-tune XLM-RoBERTa on HateXplain from the Terminal (survives notebook kernel restarts).

Quick check that everything works (a couple of minutes):
    python -m src.train_xlmr --smoke

Full run (keep the Mac awake and log to a file):
    caffeinate -i nohup python -m src.train_xlmr > results/logs/xlmr_train.log 2>&1 &
    tail -f results/logs/xlmr_train.log

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
from src.metrics import per_class_table, save_confusion_matrix
from src.xlmr import (
    build_tokenized_datasets,
    get_device,
    load_model,
    load_tokenizer,
    make_trainer,
    make_training_args,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=XLMR_MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bs", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
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
    )
    trainer = make_trainer(model, training_args, tokenizer, data["train"], data["validation"])

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
                labels, preds, "XLM-RoBERTa: Test Confusion Matrix",
                FIGURES_DIR / f"{args.tag}_confusion_matrix_test.png",
            )
            print(table.to_string(index=False))
    pd.DataFrame(rows).to_csv(TABLES_DIR / f"{args.tag}_metrics.csv", index=False)
    print("\nResults written to", TABLES_DIR)


if __name__ == "__main__":
    main()
