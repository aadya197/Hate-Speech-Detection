"""Summarise the HateNet ablation: tables, charts and significance tests.

    python -m src.ablation_report

Needs results/tables/ablation.csv and the hn_*_test_predictions.csv files from src.train_custom.
Writes: ablation_summary.csv, hatenet_significance.csv, hatenet_per_class.csv,
        figures/ablation_chart.png, figures/training_curves.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src.analysis import paired_bootstrap  # noqa: E402
from src.config import FIGURES_DIR, LABEL_LIST, TABLES_DIR  # noqa: E402

ORDER = [("hn_cls", "Plain BERT head"), ("hn_attn", "+ attention"), ("hn_bilstm", "+ BiLSTM"),
         ("hn_full", "+ rationale loss (full)"), ("hn_full_cw", "full + class weights"),
         ("hn_full_freeze6", "full + 6 frozen layers")]
SEEDED = {"hn_cls", "hn_full", "hn_full_freeze6"}


def base_tag(t):
    return t.rsplit("_s", 1)[0] if t.rsplit("_s", 1)[-1].isdigit() else t


def main():
    ab = pd.read_csv(TABLES_DIR / "ablation.csv")
    ab["config"] = ab["tag"].map(base_tag)
    rows = []
    for cfg, name in ORDER:
        d = ab[ab["config"] == cfg]
        rows.append({"config": cfg, "name": name, "runs": len(d),
                     "f1_mean": d["test_macro_f1"].mean(), "f1_std": d["test_macro_f1"].std(ddof=1) if len(d) > 1 else np.nan,
                     "acc_mean": d["test_accuracy"].mean(), "acc_std": d["test_accuracy"].std(ddof=1) if len(d) > 1 else np.nan,
                     "attn_auprc": d["attention_auprc"].mean(), "minutes": d["train_minutes"].mean()})
    summ = pd.DataFrame(rows)
    summ.to_csv(TABLES_DIR / "ablation_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))

    # chart: bars = mean, dots = individual seeds
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(ORDER))
    ax.bar(x, summ["f1_mean"], color=["#9AA3B5"] * 3 + ["#0E7C86", "#D97706", "#0E7C86"], width=0.6)
    for i, (cfg, _) in enumerate(ORDER):
        v = ab[ab["config"] == cfg]["test_macro_f1"].values
        ax.scatter([i] * len(v), v, color="#172033", s=18, zorder=3)
        ax.text(i, 0.642, f"{summ.f1_mean[i]:.3f}", ha="center", fontsize=9, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in ORDER], rotation=20, ha="right")
    ax.set_ylim(0.64, 0.71)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Ablation (bar = mean, dots = individual seeds)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ablation_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # training curves (seed 42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for cfg, name in (("hn_cls", "Plain BERT head"), ("hn_full", "HateNet full"), ("hn_full_freeze6", "HateNet, 6 frozen")):
        log = pd.read_csv(TABLES_DIR / f"{cfg}_training_log.csv")
        if cfg == "hn_cls":
            axes[0].plot(log["epoch"], log["train_loss"], marker="o", label=name)
        axes[1].plot(log["epoch"], log["val_macro_f1"], marker="o", label=name)
    axes[0].set_title("Training loss (plain head)"); axes[1].set_title("Validation macro-F1")
    for a in axes:
        a.set_xlabel("epoch"); a.set_xticks([1, 2, 3, 4])
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # significance: pooled over seeds (each seed compared to the plain head of the SAME seed), plus vs baseline
    def preds(tag):
        return pd.read_csv(TABLES_DIR / f"{tag}_test_predictions.csv")

    suffix = ["", "_s1", "_s2"]
    sig = []
    for cfg in ("hn_full", "hn_full_freeze6"):
        for s in suffix:
            r = paired_bootstrap(preds(cfg + s), preds("hn_cls" + s))
            sig.append({"model": cfg + s, "vs": "hn_cls" + s, **r})
        r = paired_bootstrap(preds(cfg), preds("baseline"))
        sig.append({"model": cfg, "vs": "baseline", **r})
        r = paired_bootstrap(preds(cfg), preds("bert"))
        sig.append({"model": cfg, "vs": "bert (HF Trainer)", **r})
    sig = pd.DataFrame(sig)
    sig.to_csv(TABLES_DIR / "hatenet_significance.csv", index=False)
    print(sig[["model", "vs", "macro_f1_diff", "macro_f1_ci_low", "macro_f1_ci_high", "macro_f1_p_better"]].round(3).to_string(index=False))

    # per-class F1 (mean over seeds)
    pc = []
    for cfg, name in ORDER:
        tags = [cfg + s for s in suffix] if cfg in SEEDED else [cfg]
        f = np.mean([f1_score(preds(t)["label"], preds(t)["pred"], average=None) for t in tags], axis=0)
        pc.append({"name": name, **dict(zip(LABEL_LIST, f))})
    pd.DataFrame(pc).to_csv(TABLES_DIR / "hatenet_per_class.csv", index=False)
    print(pd.DataFrame(pc).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
