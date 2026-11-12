"""Is HateNet's attention FAITHFUL, not just plausible?

    python -m src.hn_faithfulness --tag hn_full --max_posts 300

Loads models/<tag>_best.pt (train with --save_model), uses the same 300 test posts as src.explain, and for
three ways of ranking words (HateNet attention, occlusion on HateNet, random) measures
  plausibility  AUPRC against the human rationale (word level)
  comp          probability drop when the top 20% words are removed  (higher = model relies on them)
  suff          probability drop when ONLY the top 20% words are kept (lower = they are enough)
Output: results/tables/hn_faithfulness.csv
"""
import argparse
import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer

from src.config import MAX_LENGTH, MODELS_DIR, SEED, TABLES_DIR
from src.custom_model import HateNet
from src.explain import TOP_FRACTION, drop_words, keep_words, select_posts


def load(tag, device):
    ck = torch.load(MODELS_DIR / f"{tag}_best.pt", map_location="cpu")
    a = ck["args"]
    model = HateNet(a["encoder"], a["head"], rationale_lambda=a["rationale_lambda"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), AutoTokenizer.from_pretrained(a["encoder"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="hn_full")
    p.add_argument("--max_posts", type=int, default=300)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load(a.tag, device)

    @torch.no_grad()
    def run(texts, want_att=False, bs=64):
        probs, atts = [], []
        for i in range(0, len(texts), bs):
            enc = tok([t.split() or ["."] for t in texts[i:i + bs]], is_split_into_words=True, truncation=True,
                      max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            out = model(ids, mask)
            probs.append(torch.softmax(out["logits"].float(), -1).cpu().numpy())
            if want_att:
                for j in range(ids.shape[0]):
                    atts.append((out["attention"][j].float().cpu().numpy(), enc.word_ids(j)))
        return np.concatenate(probs), atts

    rng = np.random.default_rng(SEED)
    rows = []
    for pid, label, words, mask in select_posts(a.max_posts):
        n = len(words)
        text = " ".join(words)
        probs, atts = run([text], want_att=True)
        c, p0 = int(probs[0].argmax()), float(probs[0].max())
        att, wid = atts[0]
        att_w = np.zeros(n)  # word score = sum of attention of its word pieces
        for t, w in enumerate(wid):
            if w is not None and w < n:
                att_w[w] += att[t]
        occ = p0 - run([drop_words(words, [i]) for i in range(n)])[0][:, c]
        scores = {"attention": att_w, "occlusion": occ, "random": rng.random(n)}
        k = max(1, math.ceil(TOP_FRACTION * n))
        for name, s in scores.items():
            s = s + 1e-9 * rng.random(n)
            top = np.argsort(-s)[:k]
            pr = run([drop_words(words, top), keep_words(words, top)])[0][:, c]
            rows.append({"id": pid, "method": name, "auprc": average_precision_score(mask, s),
                         "comp": p0 - pr[0], "suff": p0 - pr[1]})
    df = pd.DataFrame(rows)
    summ = df.groupby("method")[["auprc", "comp", "suff"]].mean().reset_index()
    summ["posts"] = df["id"].nunique()
    summ.to_csv(TABLES_DIR / "hn_faithfulness.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
