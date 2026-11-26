"""Curated question-answer knowledge base (approved answers, no generation)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from local_rag.lexical import BM25Index, LexDoc
from local_rag.preprocess import normalize_text
from local_rag.settings import KB_PATH

_lock = threading.Lock()
_entries: list[dict] = []
_question_rows: list[dict] = []  # one row per paraphrased question
_bm25: BM25Index | None = None
_vectors: np.ndarray | None = None
_loaded_mtime: float | None = None


def _default_kb() -> list[dict]:
    return [
        {
            "id": "menstrual_001",
            "category": "menstrual_health",
            "intent": "menstrual_pain",
            "questions": {
                "am": [
                    "የወር አበባ ህመም ለምን ይከሰታል?",
                    "በወር አበባ ጊዜ ሆዴ ለምን ያመኛል?",
                    "የወር አበባ ህመም መንስኤው ምንድነው?",
                ],
                "en": [
                    "What causes menstrual pain?",
                    "Why do I have cramps during my period?",
                    "What causes period pain?",
                ],
            },
            "answer_am": (
                "የወር አበባ ህመም በተለምዶ ከማህፀን ጡንቻ መኮማተር ጋር ይያያዛል። "
                "ፕሮስታግላንዲን የተባሉ ኬሚካሎች ማህፀኑ እንዲኮማተር ያደርጋሉ፤ ይህም ህመም ሊያስከትል ይችላል። "
                "ይህ አጭር መረጃ ነው እንጂ ህክምና አይደለም። ከባድ ህመም፣ ትኩሳት ወይም ያልተለመደ ደም መፍሰስ ካለ የጤና ባለሙያ ያማክሩ።"
            ),
            "answer_en": (
                "Menstrual pain is often linked to uterine muscle contractions. "
                "Chemicals called prostaglandins make the uterus contract, which can cause cramps. "
                "This is general information, not medical advice. Seek care for severe pain, fever, or unusual bleeding."
            ),
            "source_id": "verified_medical_reference",
            "safety": {"domain": "health", "abstain_if_emergency": True},
        },
        {
            "id": "menstrual_002",
            "category": "menstrual_health",
            "intent": "pain_management",
            "questions": {
                "am": [
                    "እንዴት ማስታገስ እችላለሁ?",
                    "የወር አበባ ህመምን እንዴት ማርገብ እችላለሁ?",
                    "ለወር አበባ ህመም ምን ማድረግ እችላለሁ?",
                ],
                "en": [
                    "How can I relieve menstrual pain?",
                    "How can menstrual pain be relieved?",
                    "What helps with period cramps?",
                ],
            },
            "answer_am": (
                "ብዙ ሰዎች ሙቅ ጨርቅ፣ እረፍት፣ ቀላል እንቅስቃሴ እና በፋርማሲ የሚገኙ የህመም ማስታገሻዎችን ይጠቀማሉ። "
                "መድሃኒት ከመውሰድዎ በፊት መመሪያውን ያንብቡ ወይም የጤና ባለሙያ ያማክሩ። "
                "ህመሙ ከባድ ከሆነ ወይም የዕለት ተግባርዎን የሚያስተጓጉል ከሆነ ህክምና ይጠይቁ።"
            ),
            "answer_en": (
                "Common approaches include a warm compress, rest, light movement, and over-the-counter pain relief when appropriate. "
                "Read labels or ask a clinician before taking medicine. "
                "Seek care if pain is severe or disrupts daily life."
            ),
            "source_id": "verified_medical_reference",
            "safety": {"domain": "health", "abstain_if_emergency": True},
        },
        {
            "id": "safety_emergency_001",
            "category": "safety",
            "intent": "emergency_escalation",
            "questions": {
                "am": [
                    "ድንገተኛ የሆድ ህመም አለብኝ",
                    "ብዙ ደም እየፈሰሰ ነው",
                ],
                "en": [
                    "I have sudden severe abdominal pain",
                    "I am bleeding heavily and feel faint",
                ],
            },
            "answer_am": (
                "ይህ አስቸኳይ ሊሆን ይችላል። ባለሙያ የጤና እርዳታ ወዲያውኑ ይጠይቁ ወይም ወደ ቅርብ የጤና ተቋም ይሂዱ። "
                "ይህ ስርዓት ድንገተኛ ህክምና አይሰጥም።"
            ),
            "answer_en": (
                "This may be an emergency. Seek professional medical help immediately or go to the nearest clinic or hospital. "
                "This system cannot provide emergency care."
            ),
            "source_id": "safety_policy",
            "safety": {"domain": "health", "emergency": True, "force_return": True},
        },
    ]


def ensure_kb_file() -> Path:
    path = Path(KB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(_default_kb(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_entries(force: bool = False) -> list[dict]:
    global _entries, _question_rows, _bm25, _vectors, _loaded_mtime
    path = ensure_kb_file()
    mtime = path.stat().st_mtime
    with _lock:
        if not force and _entries and _loaded_mtime == mtime:
            return _entries
        entries = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for entry in entries:
            for lang, questions in (entry.get("questions") or {}).items():
                for q in questions:
                    rows.append(
                        {
                            "row_id": f"{entry['id']}::{lang}::{normalize_text(q)[:48]}",
                            "entry_id": entry["id"],
                            "lang": lang,
                            "question": q,
                            "question_norm": normalize_text(q),
                            "intent": entry.get("intent"),
                            "category": entry.get("category"),
                            "answer_am": entry.get("answer_am"),
                            "answer_en": entry.get("answer_en"),
                            "source_id": entry.get("source_id"),
                            "safety": entry.get("safety") or {},
                            "entry": entry,
                        }
                    )
        docs = [LexDoc(doc_id=row["row_id"], text=row["question_norm"], meta=row) for row in rows]
        index = BM25Index()
        index.build(docs)
        _entries = entries
        _question_rows = rows
        _bm25 = index
        _vectors = None  # rebuilt lazily
        _loaded_mtime = mtime
        return _entries


def question_rows() -> list[dict]:
    load_entries()
    return _question_rows


def bm25_index() -> BM25Index:
    load_entries()
    assert _bm25 is not None
    return _bm25


def dense_vectors() -> np.ndarray:
    """Embed each curated question once; rebuild if KB changed."""
    global _vectors
    load_entries()
    with _lock:
        if _vectors is not None and len(_vectors) == len(_question_rows):
            return _vectors
        from local_rag.embedder import embed_texts

        texts = [row["question"] for row in _question_rows]
        _vectors = embed_texts(texts) if texts else np.zeros((0, 384), dtype=np.float32)
        return _vectors


def entry_by_id(entry_id: str) -> dict | None:
    load_entries()
    for entry in _entries:
        if entry["id"] == entry_id:
            return entry
    return None


def list_intents() -> list[str]:
    load_entries()
    return sorted({e.get("intent") for e in _entries if e.get("intent")})
