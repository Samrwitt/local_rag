"""Cross-encoder reranking. Optional — skipped when weights are unavailable."""

from __future__ import annotations

import os
import threading

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from local_rag.settings import ENABLE_RERANKER, RERANKER_ID

_lock = threading.Lock()
_tokenizer = None
_model = None
_load_error: str | None = None


def available() -> bool:
    return ENABLE_RERANKER and _load_error is None


def _load():
    global _tokenizer, _model, _load_error
    if not ENABLE_RERANKER:
        _load_error = "reranker disabled in settings"
        return None, None
    if _model is not None:
        return _tokenizer, _model
    if _load_error:
        return None, None
    with _lock:
        if _model is not None:
            return _tokenizer, _model
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(RERANKER_ID, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(RERANKER_ID, local_files_only=True)
            model.to("cpu")
            model.eval()
            _tokenizer = tokenizer
            _model = model
            return _tokenizer, _model
        except Exception as exc:  # noqa: BLE001 — optional component
            _load_error = str(exc)
            return None, None


def rerank(question: str, candidates: list[dict], text_key: str = "text", top_k: int = 8) -> list[dict]:
    """Score (question, candidate) pairs. Falls back to input order if unavailable."""
    if not candidates:
        return []
    tokenizer, model = _load()
    if tokenizer is None or model is None:
        for cand in candidates:
            cand.setdefault("rerank_score", None)
            cand.setdefault("rerank_skipped", _load_error or "disabled")
        return candidates[:top_k]

    import torch

    pairs_a = [question] * len(candidates)
    pairs_b = [str(c.get(text_key) or c.get("question") or "")[:800] for c in candidates]
    scores = []
    batch = 8
    with torch.no_grad():
        for start in range(0, len(pairs_a), batch):
            tokens = tokenizer(
                pairs_a[start : start + batch],
                pairs_b[start : start + batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            logits = model(**tokens).logits
            if logits.shape[-1] == 1:
                batch_scores = logits.squeeze(-1)
            else:
                batch_scores = logits[:, -1]
            scores.extend(batch_scores.cpu().tolist())

    ranked = []
    for cand, score in zip(candidates, scores):
        item = dict(cand)
        item["rerank_score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
