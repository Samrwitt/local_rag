"""BM25, keyword, and character n-gram lexical retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from local_rag.preprocess import char_ngrams, tokenize


@dataclass
class LexDoc:
    doc_id: str
    text: str
    meta: dict


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[LexDoc] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.df: Counter = Counter()
        self.N = 0
        self._ngram_sets: list[set[str]] = []

    def build(self, docs: list[LexDoc]) -> None:
        self.docs = docs
        self.doc_tokens = [tokenize(doc.text) for doc in docs]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.N = len(docs)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.df = Counter()
        for toks in self.doc_tokens:
            self.df.update(set(toks))
        self._ngram_sets = [set(char_ngrams(doc.text, 3)) for doc in docs]

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], query_ngrams: list[str], k: int = 30) -> list[tuple[str, float, dict]]:
        if not self.docs or not query_tokens:
            return []
        scores = []
        q_ng = set(query_ngrams)
        for idx, (doc, toks, length) in enumerate(zip(self.docs, self.doc_tokens, self.doc_len)):
            tf = Counter(toks)
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                idf = self._idf(term)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * length / max(self.avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1)) / denom
            # Character n-gram Jaccard for misspellings / Amharic variation.
            if q_ng and self._ngram_sets[idx]:
                inter = len(q_ng & self._ngram_sets[idx])
                union = len(q_ng | self._ngram_sets[idx])
                score += 0.35 * (inter / union if union else 0.0)
            # Exact phrase bump
            blob = " ".join(toks)
            if len(query_tokens) >= 2 and " ".join(query_tokens) in blob:
                score += 1.5
            if score > 0:
                scores.append((doc.doc_id, score, doc.meta))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]


_TOKEN_RE = re.compile(r"[\w\u1200-\u137F]+", re.UNICODE)
