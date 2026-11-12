"""Train HateNet (our own architecture) and run ablations.

    python -m src.train_custom --smoke                                   # 1 minute check
    python -m src.train_custom --tag hn_full --head bilstm_attn --rationale_lambda 1.0 --fp16
    python -m src.train_custom --tag hn_cls --head cls --fp16            # ablation: plain BERT head
    python -m src.train_custom --tag hn_full_s1 --seed 1 ...             # other seeds

Writes the same files as src.train_xlmr (<tag>_test_predictions.csv, <tag>_metrics.csv, ...) plus
<tag>_attention_eval.csv when the head has attention. Results are appended to results/tables/ablation.csv.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.config import FIGURES_DIR, MAX_LENGTH, MODELS_DIR, TABLES_DIR, ensure_dirs
from src.custom_model import HateNet
from src.data import consensus_mask, load_processed, parse_rationales
from src.metrics import compute_metrics, per_class_table, save_confusion_matrix, save_predictions


def encode(df, tok, max_len, with_rationale=True):
    """Tokenize by words so the human word-level rationale can be mapped onto sub-word tokens."""
    items = []
    for _, r in df.iterrows():
        words = str(r["text"]).split() or ["."]
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=max_len)
        rat = [0] * len(enc["input_ids"])
        if with_rationale and int(r["label"]) != 1:
            m = consensus_mask(parse_rationales(r["rationales"]), words)
            if len(m) == len(words):
                rat = [int(m[w]) if w is not None else 0 for w in enc.word_ids()]
        items.append({"input_ids": enc["input_ids"], "label": int(r["label"]), "rationale": rat})
    return items


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    rat = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L], mask[i, :L], rat[i, :L] = torch.tensor(b["input_ids"]), 1, torch.tensor(b["rationale"])
    return {"input_ids": ids, "attention_mask": mask, "rationale": rat,
            "labels": torch.tensor([b["label"] for b in batch])}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, atts = [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out = model(b["input_ids"], b["attention_mask"])
        logits.append(out["logits"].float().cpu().numpy())
        if out["attention"] is not None:
            atts.append(out["attention"].float().cpu().numpy())
    return np.concatenate(logits), atts


def attention_plausibility(model, data, device, pad_id, batch_size=64):
    """AUPRC of the attention weights against the human rationale (higher = attention looks like humans)."""
    from sklearn.metrics import average_precision_score

    rows = [d for d in data if sum(d["rationale"]) > 0]
    if not rows or model.head == "cls":
        return float("nan")
    _, atts = predict(model, DataLoader(rows, batch_size=batch_size, collate_fn=lambda b: collate(b, pad_id)), device)
    scores, i = [], 0
    for a in atts:
        for row in a:
            r = rows[i]["rationale"]
            if 0 < sum(r) < len(r):
                scores.append(average_precision_score(r, row[: len(r)]))
            i += 1
    return float(np.mean(scores))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="hn_full")
    p.add_argument("--encoder", default="bert-base-uncased")
    p.add_argument("--head", default="bilstm_attn", choices=["cls", "attn", "bilstm_attn"])
    p.add_argument("--rationale_lambda", type=float, default=0.0)
    p.add_argument("--class_weights", action="store_true")
    p.add_argument("--freeze_layers", type=int, default=0)
    p.add_argument("--max_len", type=int, default=MAX_LENGTH)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=1e-3, help="learning rate of the new layers (LSTM/attention/classifier)")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--save_model", action="store_true", help="save the best weights to models/<tag>_best.pt")
    a = p.parse_args(argv)
    if a.smoke:
        a.tag += "_smoke"
    ensure_dirs()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # MPS avoided: it stalled before
    print("Device:", device, "|", vars(a))

    tok = AutoTokenizer.from_pretrained(a.encoder)
    lim = 64 if a.smoke else None
    data = {s: encode(load_processed(s).head(lim) if lim else load_processed(s), tok, a.max_len)
            for s in ("train", "validation", "test")}
    pad = tok.pad_token_id
    mk = lambda d, shuffle: DataLoader(d, batch_size=a.bs if shuffle else 64, shuffle=shuffle,
                                       collate_fn=lambda b: collate(b, pad))
    train_dl, val_dl, test_dl = mk(data["train"], True), mk(data["validation"], False), mk(data["test"], False)

    cw = None
    if a.class_weights:
        counts = np.bincount([d["label"] for d in data["train"]], minlength=3)
        cw = (counts.sum() / (3 * counts)).tolist()
        print("class weights:", np.round(cw, 3))
    model = HateNet(a.encoder, a.head, freeze_layers=a.freeze_layers, class_weights=cw,
                    rationale_lambda=a.rationale_lambda).to(device)

    enc_params = [q for q in model.encoder.parameters() if q.requires_grad]
    enc_ids = {id(q) for q in enc_params}
    new_params = [q for q in model.parameters() if q.requires_grad and id(q) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": new_params, "lr": a.head_lr}],
                            weight_decay=0.01)
    total = len(train_dl) * (1 if a.smoke else a.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup_ratio * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=a.fp16 and device.type == "cuda")

    best, best_state, log = -1.0, None, []
    t0 = time.time()
    for epoch in range(1, (1 if a.smoke else a.epochs) + 1):
        model.train()
        run = 0.0
        for step, b in enumerate(train_dl, 1):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=a.fp16 and device.type == "cuda"):
                loss = model(**b)["loss"]
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {run / step:.4f}", flush=True)
        vl, _ = predict(model, val_dl, device)
        vm = compute_metrics((vl, np.array([d["label"] for d in data["validation"]])))
        log.append({"epoch": epoch, "train_loss": run / len(train_dl), **{f"val_{k}": v for k, v in vm.items()}})
        print(f"epoch {epoch}: train_loss {run / len(train_dl):.4f} val_macro_f1 {vm['macro_f1']:.4f} "
              f"val_acc {vm['accuracy']:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pd.DataFrame(log).to_csv(TABLES_DIR / f"{a.tag}_training_log.csv", index=False)
    if a.save_model:
        torch.save({"state": best_state, "args": vars(a)}, MODELS_DIR / f"{a.tag}_best.pt")

    rows = []
    for split, dl in (("validation", val_dl), ("test", test_dl)):
        lg, _ = predict(model, dl, device)
        labels = np.array([d["label"] for d in data[split]])
        m = compute_metrics((lg, labels))
        save_predictions(split, lg, TABLES_DIR / f"{a.tag}_{split}_predictions.csv", limit=lim)
        rows.append({"split": split, **m})
        if split == "test":
            per_class_table(labels, lg.argmax(1)).to_csv(TABLES_DIR / f"{a.tag}_per_class_test.csv", index=False)
            save_confusion_matrix(labels, lg.argmax(1), f"{a.tag}: Test Confusion Matrix",
                                  FIGURES_DIR / f"{a.tag}_confusion_matrix_test.png")
            test_m = m
    pd.DataFrame(rows).to_csv(TABLES_DIR / f"{a.tag}_metrics.csv", index=False)
    att_auprc = attention_plausibility(model, data["test"], device, pad)
    print(f"\nTEST macro-F1 {test_m['macro_f1']:.4f}  acc {test_m['accuracy']:.4f}  attention AUPRC {att_auprc:.3f}")

    path = TABLES_DIR / "ablation.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[old["tag"] != a.tag]
    row = {"tag": a.tag, "head": a.head, "rationale_lambda": a.rationale_lambda, "class_weights": a.class_weights,
           "freeze_layers": a.freeze_layers, "max_len": a.max_len, "seed": a.seed, "epochs": a.epochs,
           "val_macro_f1": best, "test_macro_f1": test_m["macro_f1"], "test_accuracy": test_m["accuracy"],
           "attention_auprc": att_auprc, "params_M": sum(q.numel() for q in model.parameters()) / 1e6,
           "train_minutes": (time.time() - t0) / 60}
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


if __name__ == "__main__":
    main()
