"""Confidence scoring and safety validation — no generative LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass

EMERGENCY_PATTERNS = [
    r"\b(emergency|faint|fainting|heavy bleeding|suicidal|overdose)\b",
    r"(ድንገተኛ|ከባድ ህመም|ብዙ ደም|ራስን መግደል)",
]


@dataclass
class ValidationResult:
    ok: bool
    confidence: float
    reason: str
    abstain: bool = False
    emergency: bool = False
    source_ids: list[str] | None = None


def score_candidate(candidate: dict, query_lang: str) -> float:
    """Map retrieval signals into a 0–1 confidence."""
    # Document passages: dense similarity is the reliable signal. Raw BM25 on
    # long PDFs over-fires on common verbs like "write".
    if candidate.get("source_type") == "document":
        dense = max(0.0, float(candidate.get("dense_score") or 0.0))
        legacy = float(candidate.get("legacy_score") or 0.0)
        bm25 = min(1.0, float(candidate.get("bm25_score") or 0.0) / 20.0)
        name_overlap = float(candidate.get("name_overlap") or 0.0)
        conf = 0.55 * dense + 0.35 * min(1.0, legacy) + 0.10 * bm25
        if name_overlap >= 0.5:
            conf = max(conf, 0.42 + 0.4 * name_overlap)
        if candidate.get("rerank_score") is not None:
            conf = 0.7 * conf + 0.3 * (1 / (1 + pow(2.718281828, -float(candidate["rerank_score"]))))
        return max(0.0, min(1.0, conf))

    parts = []
    if candidate.get("rerank_score") is not None:
        parts.append(1 / (1 + pow(2.718281828, -float(candidate["rerank_score"]))))
    if candidate.get("rrf_score") is not None:
        parts.append(min(1.0, float(candidate["rrf_score"]) * 8))
    if candidate.get("dense_score") is not None:
        parts.append(max(0.0, float(candidate["dense_score"])))
    if candidate.get("bm25_score") is not None:
        parts.append(min(1.0, float(candidate["bm25_score"]) / 8.0))
    if not parts:
        return 0.0
    conf = sum(parts) / len(parts)
    conf = min(1.0, conf + 0.08)  # curated KB bump
    if query_lang in {"am", "en"} and candidate.get("lang") == query_lang:
        conf = min(1.0, conf + 0.03)
    return conf


def detect_emergency(question: str) -> bool:
    blob = question or ""
    return any(re.search(pat, blob, re.I) for pat in EMERGENCY_PATTERNS)


def validate_answer(
    question: str,
    candidates: list[dict],
    *,
    min_confidence: float = 0.42,
) -> ValidationResult:
    if detect_emergency(question):
        emergency = next((c for c in candidates if (c.get("safety") or {}).get("emergency")), None)
        if emergency:
            return ValidationResult(
                ok=True,
                confidence=1.0,
                reason="emergency_escalation",
                emergency=True,
                source_ids=[emergency.get("source_id") or emergency.get("id")],
            )
        return ValidationResult(
            ok=False,
            confidence=0.0,
            reason="emergency_no_protocol",
            abstain=True,
            emergency=True,
        )

    if not candidates:
        return ValidationResult(ok=False, confidence=0.0, reason="no_candidates", abstain=True)

    ranked = sorted(candidates, key=lambda c: c.get("confidence", 0.0), reverse=True)
    best = ranked[0]
    conf = float(best.get("confidence") or 0.0)
    source_ids = [c.get("source_id") or c.get("id") for c in ranked[:3] if c.get("source_id") or c.get("id")]

    if (best.get("safety") or {}).get("force_return"):
        return ValidationResult(ok=True, confidence=conf, reason="forced_safety", source_ids=source_ids)

    if conf < min_confidence:
        return ValidationResult(
            ok=False,
            confidence=conf,
            reason="low_confidence",
            abstain=True,
            source_ids=source_ids,
        )

    # Margin check: top should clearly beat #2 for KB answers.
    if len(ranked) > 1 and best.get("source_type") == "kb":
        gap = conf - float(ranked[1].get("confidence") or 0.0)
        if gap < 0.03 and conf < 0.7:
            return ValidationResult(
                ok=False,
                confidence=conf,
                reason="ambiguous_top2",
                abstain=True,
                source_ids=source_ids,
            )

    return ValidationResult(ok=True, confidence=conf, reason="verified", source_ids=source_ids)


def pick_answer_text(candidate: dict, language: str) -> str:
    if language == "am":
        return candidate.get("answer_am") or candidate.get("answer_en") or candidate.get("text") or ""
    if language == "en":
        return candidate.get("answer_en") or candidate.get("answer_am") or candidate.get("text") or ""
    return (
        candidate.get("answer_am")
        or candidate.get("answer_en")
        or candidate.get("text")
        or ""
    )
