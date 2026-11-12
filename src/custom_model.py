"""HateNet: our own architecture on top of a pretrained transformer encoder.

    tokens -> encoder (BERT) -> [optional BiLSTM] -> additive attention pooling -> dropout -> linear -> 3 classes

Design choices (each one can be switched off for the ablation study):
  head="cls"           plain BERT: classifier on the [CLS] vector (the standard fine-tuning baseline)
  head="attn"          attention pooling over all tokens (learned weighted average)
  head="bilstm_attn"   BiLSTM on the encoder output, then attention pooling
  rationale_lambda>0   auxiliary loss: push the attention weights towards the words that human
                       annotators marked (HateXplain rationales). The attention then doubles as a
                       built-in explanation, with no extra occlusion passes.
  class_weights        inverse-frequency weights in the cross-entropy loss
  freeze_layers=k      freeze embeddings and the first k encoder layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class HateNet(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased", head="bilstm_attn", num_labels=3,
                 lstm_hidden=128, dropout=0.3, freeze_layers=0, class_weights=None,
                 rationale_lambda=0.0, encoder=None):
        super().__init__()
        assert head in ("cls", "attn", "bilstm_attn")
        self.head, self.rationale_lambda = head, rationale_lambda
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(encoder_name)
        dim = self.encoder.config.hidden_size
        if head == "bilstm_attn":
            self.lstm = nn.LSTM(dim, lstm_hidden, batch_first=True, bidirectional=True)
            dim = 2 * lstm_hidden
        if head != "cls":
            self.att = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1, bias=False))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, num_labels)
        self.register_buffer("class_weights", None if class_weights is None else torch.tensor(class_weights, dtype=torch.float))
        if freeze_layers:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
            for layer in self.encoder.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None, rationale=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        att = None
        if self.head == "cls":
            pooled = h[:, 0]
        else:
            if self.head == "bilstm_attn":
                h, _ = self.lstm(h)
            scores = self.att(h).squeeze(-1).masked_fill(attention_mask == 0, -1e4)
            att = torch.softmax(scores.float(), dim=-1)
            pooled = torch.bmm(att.unsqueeze(1).to(h.dtype), h).squeeze(1)
        logits = self.out(self.drop(pooled))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float(), labels, weight=self.class_weights)
            if self.rationale_lambda > 0 and att is not None and rationale is not None:
                has = rationale.sum(1) > 0  # only posts that have human rationales
                if has.any():
                    target = rationale[has].float()
                    target = target / target.sum(1, keepdim=True)
                    att_loss = -(target * torch.log(att[has] + 1e-9)).sum(1).mean()
                    loss = loss + self.rationale_lambda * att_loss
        return {"loss": loss, "logits": logits, "attention": att}
