"""Tier D: fine-tuned DistilBERT, wrapped to look like the sklearn pipelines.

Needs the optional deps: pip install -r requirements-bert.txt
"""

import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging

logging.set_verbosity_error()


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BertClassifier:
    """fit(frame, y) / predict_proba(frame) on the frame's `text` column."""

    # 3 epochs at 512 tokens beat 2 at 256 (most postings exceed 256 tokens); batch 16 fits a 16 GB laptop.
    def __init__(self, model_name="distilbert-base-uncased", epochs=3, max_len=512, batch_size=16, lr=3e-5, seed=0):
        self.model_name = model_name
        self.epochs = epochs
        self.max_len = max_len
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    def _encode(self, X):
        texts = X["text"].tolist()
        return self.tokenizer(texts, truncation=True, max_length=self.max_len)["input_ids"]

    def _batch(self, ids, rows, device):
        width = max(len(ids[i]) for i in rows)
        input_ids = torch.full((len(rows), width), self.tokenizer.pad_token_id)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for r, i in enumerate(rows):
            input_ids[r, :len(ids[i])] = torch.tensor(ids[i])
            mask[r, :len(ids[i])] = 1
        return input_ids.to(device), mask.to(device)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        device = _device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(device)
        ids, y = self._encode(X), np.asarray(y)
        # Same idea as class_weight="balanced" in the other tiers.
        weight = torch.tensor(len(y) / (2 * np.bincount(y, minlength=2)), dtype=torch.float, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * int(np.ceil(len(ids) / self.batch_size))
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
        self.model.train()
        start_time = time.time()
        for epoch in range(self.epochs):
            order = rng.permutation(len(ids))
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start + self.batch_size]
                input_ids, mask = self._batch(ids, rows, device)
                logits = self.model(input_ids=input_ids, attention_mask=mask).logits
                loss = loss_fn(logits, torch.tensor(y[rows], device=device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            print(f"  epoch {epoch + 1}/{self.epochs} done, {time.time() - start_time:.0f}s", flush=True)
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        device = _device()
        self.model.to(device).eval()
        ids = self._encode(X)
        order = np.argsort([len(i) for i in ids])  # similar lengths together -> less padding
        proba = np.empty((len(ids), 2))
        for start in range(0, len(order), self.batch_size * 2):
            rows = order[start:start + self.batch_size * 2]
            input_ids, mask = self._batch(ids, rows, device)
            logits = self.model(input_ids=input_ids, attention_mask=mask).logits
            proba[rows] = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return proba

    def __getstate__(self):
        # Pickle on cpu so the artifact loads on machines without this one's GPU.
        if hasattr(self, "model"):
            self.model.cpu()
        return self.__dict__
