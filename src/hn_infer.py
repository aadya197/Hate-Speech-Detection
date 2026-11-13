"""Load a trained HateNet (models/<tag>_best.pt) for inference: probabilities and attention per word."""
import numpy as np
import torch
from transformers import AutoTokenizer

from src.config import MAX_LENGTH, MODELS_DIR
from src.custom_model import HateNet


def load_hatenet(tag="hn_full", device=None):
    """Return (predict, attention). predict(texts) -> (n, 3) probabilities (same interface as src.explain);
    attention(words) -> one score per word (sum of the attention of its word pieces, sums to 1)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(MODELS_DIR / f"{tag}_best.pt", map_location="cpu")
    a = ck["args"]
    model = HateNet(a["encoder"], a["head"], rationale_lambda=a["rationale_lambda"])
    model.load_state_dict(ck["state"])
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(a["encoder"])

    def encode(batch_words):
        enc = tok(batch_words, is_split_into_words=True, truncation=True, max_length=MAX_LENGTH,
                  padding=True, return_tensors="pt")
        return enc, enc["input_ids"].to(device), enc["attention_mask"].to(device)

    @torch.no_grad()
    def predict(texts, batch=64):
        texts, out = list(texts), []
        for i in range(0, len(texts), batch):
            _, ids, mask = encode([t.split() or ["."] for t in texts[i:i + batch]])
            out.append(torch.softmax(model(ids, mask)["logits"].float(), -1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def attention(words):
        enc, ids, mask = encode([list(words)])
        att = model(ids, mask)["attention"]
        scores = np.zeros(len(words))
        if att is None:  # plain [CLS] head has no attention
            return scores
        for t, w in enumerate(enc.word_ids(0)):
            if w is not None and w < len(words):
                scores[w] += float(att[0, t])
        return scores

    return predict, attention
