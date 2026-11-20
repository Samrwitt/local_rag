"""Embeddings from the MiniLM snapshot already in the Hugging Face cache.

sentence-transformers is not installed. The model weights are, and so are
torch and transformers, so we mean-pool the last hidden state ourselves.
That is the same pooling MiniLM was trained with.
"""

from __future__ import annotations

import threading

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

_lock = threading.Lock()
_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    with _lock:
        if _model is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
            model.to("cpu")
            model.eval()
            _tokenizer = tokenizer
            _model = model
    return _tokenizer, _model


def embed_texts(texts: list[str], batch_size: int = 8) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    tokenizer, model = _load()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
