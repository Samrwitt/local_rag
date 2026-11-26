"""Query preprocessing: normalize, detect language, light spelling repair."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from local_rag.settings import SYNONYM_PATH

ETHIOPIC_RE = re.compile(r"[\u1200-\u137F]")
LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass
class PreparedQuery:
    raw: str
    normalized: str
    language: str  # am | en | mixed | unknown
    tokens: list[str] = field(default_factory=list)
    char_ngrams: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)
    intent_hints: list[str] = field(default_factory=list)


def detect_language(text: str) -> str:
    eth = len(ETHIOPIC_RE.findall(text or ""))
    lat = len(LATIN_RE.findall(text or ""))
    if eth and lat:
        return "mixed"
    if eth:
        return "am"
    if lat:
        return "en"
    return "unknown"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Collapse Latin punctuation noise; keep Ethiopic letters intact.
    text = re.sub(r"[^\w\s\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF？?!.،,;:/\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u1200-\u137F]+", text.lower(), flags=re.UNICODE)


def char_ngrams(text: str, n: int = 3) -> list[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i : i + n] for i in range(len(compact) - n + 1)]


def _load_synonyms() -> dict[str, list[str]]:
    path = Path(SYNONYM_PATH)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k).lower(): [str(x).lower() for x in v] for k, v in data.items()}


def expand_synonyms(tokens: list[str]) -> list[str]:
    syn = _load_synonyms()
    out = list(tokens)
    seen = set(tokens)
    for token in tokens:
        for alt in syn.get(token, []):
            if alt not in seen:
                out.append(alt)
                seen.add(alt)
    return out


def prepare_query(question: str, intent_hints: list[str] | None = None) -> PreparedQuery:
    raw = (question or "").strip()
    normalized = normalize_text(raw)
    language = detect_language(raw)
    tokens = tokenize(normalized)
    grams = char_ngrams(normalized, 3)
    expanded = expand_synonyms(tokens)
    return PreparedQuery(
        raw=raw,
        normalized=normalized,
        language=language,
        tokens=tokens,
        char_ngrams=grams,
        expanded_terms=expanded,
        intent_hints=list(intent_hints or []),
    )
