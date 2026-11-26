"""Shared settings that must not import torch."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KB_PATH = DATA_DIR / "kb" / "curated_qa.json"
EVAL_PATH = DATA_DIR / "eval" / "sample_questions.json"
SYNONYM_PATH = DATA_DIR / "kb" / "synonyms.json"

# Document index was built with the English MiniLM snapshot cached on this machine.
# Keep it as the active encoder until a reindex with a new model is completed.
ACTIVE_MODEL_ID = os.environ.get(
    "LOCAL_RAG_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
MODEL_ID = ACTIVE_MODEL_ID

# Recommended baseline for Amharic/English once reindexed.
MULTILINGUAL_BASELINE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Candidate encoders to benchmark (download only when explicitly requested).
EMBEDDING_CANDIDATES = {
    "minilm_en": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "purpose": "Current index / English baseline",
        "dim": 384,
    },
    "minilm_multilingual": {
        "id": MULTILINGUAL_BASELINE,
        "purpose": "Multilingual baseline for Am/En",
        "dim": 384,
    },
    "bge_m3": {
        "id": "BAAI/bge-m3",
        "purpose": "Multilingual retrieval (dense)",
        "dim": 1024,
    },
    "e5_multilingual": {
        "id": "intfloat/multilingual-e5-base",
        "purpose": "Query-to-passage retrieval",
        "dim": 768,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "roberta_amharic": {
        "id": "rasyosef/roberta-base-amharic",
        "purpose": "Amharic-oriented encoder (experimental pooling)",
        "dim": 768,
        "experimental": True,
    },
}

RERANKER_CANDIDATES = {
    "bge_reranker_v2_m3": "BAAI/bge-reranker-v2-m3",
    "gte_multilingual": "Alibaba-NLP/gte-multilingual-reranker-base",
}

RETRIEVE_CANDIDATES = 30
RERANK_TOP_K = 8
RRF_K = 60
MIN_CONFIDENCE = 0.42
ENABLE_RERANKER = os.environ.get("LOCAL_RAG_ENABLE_RERANKER", "0") == "1"
RERANKER_ID = os.environ.get(
    "LOCAL_RAG_RERANKER",
    RERANKER_CANDIDATES["bge_reranker_v2_m3"],
)

ACCEPTANCE_TARGETS = {
    "recall_at_5": 0.95,
    "top1_accuracy": 0.90,
    "intent_accuracy": 0.90,
    "ood_detection": 0.95,
    "source_traceability": 1.0,
}
