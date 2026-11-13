"""Identity-term bias check: does the model flag NEUTRAL sentences just because they mention a group?

    python -m src.bias_check --tag hn_full

Every template below is harmless by construction ("my neighbour is {group} and very kind"). Any
Hate/Offensive prediction on such a sentence is a false alarm caused by the group word. For every group we
report the share of sentences predicted Hate or Offensive and the mean P(Hate speech); groups that stand out
from the others show where the model over-reacts to identity terms.
Output: results/tables/bias_check.csv, results/figures/bias_check.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FIGURES_DIR, LABEL_LIST, TABLES_DIR, ensure_dirs  # noqa: E402
from src.hn_infer import load_hatenet  # noqa: E402

GROUPS = ["muslim", "jewish", "christian", "hindu", "buddhist", "atheist", "black", "white", "asian",
          "latino", "gay", "lesbian", "straight", "transgender", "disabled", "immigrant", "refugee",
          "woman", "man", "elderly"]
TEMPLATES = [
    "my neighbour is {g} and very kind",
    "i had dinner with a {g} friend yesterday",
    "the {g} family next door invited us for tea",
    "she is a {g} teacher at our school",
    "i read an article about a {g} scientist who won an award",
    "being {g} is part of who i am",
    "we met a {g} couple on our holiday and they were lovely",
    "he is {g} and he loves playing football",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="hn_full")
    a = p.parse_args()
    ensure_dirs()
    predict, _ = load_hatenet(a.tag)
    rows = []
    for g in GROUPS:
        probs = predict([t.format(g=g) for t in TEMPLATES])
        pred = probs.argmax(1)
        rows.append({"group": g, "n": len(TEMPLATES),
                     "flagged_share": float((pred != 1).mean()),          # predicted Hate or Offensive
                     "hate_share": float((pred == 0).mean()),
                     "mean_p_hate": float(probs[:, 0].mean()),
                     "mean_p_offensive": float(probs[:, 2].mean())})
    df = pd.DataFrame(rows).sort_values("flagged_share", ascending=False)
    df.to_csv(TABLES_DIR / "bias_check.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"\noverall share of NEUTRAL sentences flagged as Hate/Offensive: {df['flagged_share'].mean():.1%}")

    fig, ax = plt.subplots(figsize=(7, 5))
    d = df.sort_values("flagged_share")
    ax.barh(d["group"], d["flagged_share"], color="#D97706")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of neutral sentences flagged Hate/Offensive")
    ax.set_title(f"Identity-term check ({a.tag}, {len(TEMPLATES)} sentences per group)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bias_check.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
