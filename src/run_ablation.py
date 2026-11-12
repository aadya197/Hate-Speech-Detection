"""Run the whole ablation study, one run after another (skips runs that are already in ablation.csv).

    python -m src.run_ablation --stage design     # 5 runs, about 1 hour on a Colab T4
    python -m src.run_ablation --stage seeds      # 4 extra runs (2 more seeds of cls and full)
    python -m src.run_ablation --stage design --dry   # just print the commands
"""
import argparse
import subprocess
import sys

import pandas as pd

from src.config import TABLES_DIR

COMMON = ["--fp16", "--epochs", "4"]
STAGES = {
    "design": [  # one change at a time, so every difference can be explained
        ("hn_cls", ["--head", "cls"]),                                             # control: plain BERT head
        ("hn_attn", ["--head", "attn"]),                                           # + attention pooling
        ("hn_bilstm", ["--head", "bilstm_attn"]),                                  # + BiLSTM
        ("hn_full", ["--head", "bilstm_attn", "--rationale_lambda", "1.0"]),       # + rationale-guided attention
        ("hn_full_cw", ["--head", "bilstm_attn", "--rationale_lambda", "1.0", "--class_weights"]),
        ("hn_full_freeze6", ["--head", "bilstm_attn", "--rationale_lambda", "1.0", "--freeze_layers", "6"]),
    ],
    "seeds": [
        (f"{t}_s{s}", ["--head", h, "--seed", str(s)] + (["--rationale_lambda", "1.0"] if t == "hn_full" else []))
        for s in (1, 2) for t, h in (("hn_cls", "cls"), ("hn_full", "bilstm_attn"))
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=list(STAGES))
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    path = TABLES_DIR / "ablation.csv"
    done = set(pd.read_csv(path)["tag"]) if path.exists() else set()
    for tag, extra in STAGES[a.stage]:
        if tag in done:
            print("skip (already done):", tag)
            continue
        cmd = [sys.executable, "-m", "src.train_custom", "--tag", tag, *COMMON, *extra]
        print("\n>>>", " ".join(cmd), flush=True)
        if not a.dry:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
