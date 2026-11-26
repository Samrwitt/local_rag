"""Deterministic conversation manager — session memory, intent, follow-ups."""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field

from local_rag import kb
from local_rag.preprocess import normalize_text, prepare_query

_lock = threading.Lock()
_sessions: dict[str, "Session"] = {}

FOLLOWUP_SHORT = re.compile(
    r"^(how|what|why|when|where|can|could|እንዴት|ምን|ለምን|መቼ)",
    re.I,
)


@dataclass
class Session:
    session_id: str
    previous_intent: str | None = None
    previous_category: str | None = None
    previous_entry_id: str | None = None
    previous_language: str | None = None
    turns: list[dict] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


def get_or_create(session_id: str | None = None) -> Session:
    with _lock:
        if session_id and session_id in _sessions:
            session = _sessions[session_id]
            session.updated_at = time.time()
            return session
        sid = session_id or str(uuid.uuid4())
        session = Session(session_id=sid)
        _sessions[sid] = session
        return session


def classify_intent(question: str) -> str | None:
    """Keyword / curated-question overlap intent classifier (no LLM)."""
    kb.load_entries()
    prepared = prepare_query(question)
    q_tokens = set(prepared.expanded_terms)
    best_intent = None
    best_score = 0.0
    for row in kb.question_rows():
        row_tokens = set(normalize_text(row["question"]).split())
        if not row_tokens:
            continue
        overlap = len(q_tokens & row_tokens) / len(row_tokens)
        if overlap > best_score:
            best_score = overlap
            best_intent = row.get("intent")
    return best_intent if best_score >= 0.34 else None


def resolve_followup(question: str, session: Session) -> dict:
    """Expand short follow-ups using previous topic templates."""
    prepared = prepare_query(question)
    intent = classify_intent(question)
    resolved = prepared.raw
    used_context = False

    short = len(prepared.tokens) <= 6 or bool(FOLLOWUP_SHORT.search(prepared.normalized))
    if short and session.previous_intent and not intent:
        # Template: previous menstrual pain + "how can I relieve" style follow-up.
        if session.previous_intent == "menstrual_pain":
            if any(tok in prepared.normalized for tok in ["relieve", "help", "stop", "ማስታገስ", "ማርገብ", "ምን ማድረግ"]):
                intent = "pain_management"
                resolved = (
                    "የወር አበባ ህመምን እንዴት ማርገብ እችላለሁ?"
                    if prepared.language == "am"
                    else "How can menstrual pain be relieved?"
                )
                used_context = True
        if not used_context and session.previous_category:
            resolved = f"{session.previous_category}: {prepared.raw}"
            used_context = True
            intent = intent or session.previous_intent

    if intent is None:
        intent = session.previous_intent

    return {
        "session_id": session.session_id,
        "raw_question": prepared.raw,
        "resolved_question": resolved,
        "language": prepared.language,
        "intent": intent,
        "used_context": used_context,
        "previous_intent": session.previous_intent,
        "previous_entry_id": session.previous_entry_id,
    }


def remember(session: Session, *, intent: str | None, category: str | None, entry_id: str | None, language: str | None, question: str, answer_preview: str) -> None:
    session.previous_intent = intent or session.previous_intent
    session.previous_category = category or session.previous_category
    session.previous_entry_id = entry_id or session.previous_entry_id
    session.previous_language = language or session.previous_language
    session.turns.append(
        {
            "question": question,
            "intent": intent,
            "entry_id": entry_id,
            "answer_preview": (answer_preview or "")[:200],
            "ts": time.time(),
        }
    )
    session.turns = session.turns[-20:]
    session.updated_at = time.time()
