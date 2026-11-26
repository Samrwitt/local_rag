"""Multi-stage retrieval pipeline (no generative LLM).

Stages:
1. Query preprocessing
2. Conversational resolve (optional session)
3. Dense semantic search (KB + documents)
4. Lexical BM25 / n-grams
5. Hybrid RRF
6. Cross-encoder rerank (optional)
7. Confidence + safety validation
8. Verified answer selection
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from local_rag import confidence, conversation, kb, store
from local_rag.hybrid import attach_meta, reciprocal_rank_fusion
from local_rag.lexical import BM25Index, LexDoc
from local_rag.preprocess import prepare_query
from local_rag.rerank import rerank
from local_rag.settings import (
    MIN_CONFIDENCE,
    RERANK_TOP_K,
    RETRIEVE_CANDIDATES,
    RRF_K,
)


def _doc_candidates(question: str, prepared, k: int) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[tuple[str, float]], dict[str, dict]]:
    """Dense + lexical + legacy hybrid over indexed document chunks."""
    from local_rag.embedder import embed_query

    meta_by_id: dict[str, dict] = {}
    dense_ranking: list[tuple[str, float]] = []
    lexical_ranking: list[tuple[str, float]] = []
    legacy_ranking: list[tuple[str, float]] = []

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, path, ordinal, text, kind, captured_at, country, city, roles, vector FROM chunks"
        ).fetchall()
    if not rows:
        return dense_ranking, lexical_ranking, legacy_ranking, meta_by_id

    qvec = embed_query(prepared.normalized or question).astype(np.float32)
    dense_scored = []
    lex_docs = []
    for row in rows:
        doc_id = f"doc:{row['id']}"
        text = row["text"] or ""
        meta = {
            "id": doc_id,
            "source_type": "document",
            "path": row["path"],
            "name": Path(row["path"]).name,
            "ordinal": row["ordinal"],
            "text": text,
            "question": text[:240],
            "kind": row["kind"],
            "captured_at": row["captured_at"],
            "country": row["country"],
            "city": row["city"],
            "roles": row["roles"],
            "source_id": row["path"],
            "answer_en": text,
            "answer_am": text,
            "lang": prepared.language,
        }
        meta_by_id[doc_id] = meta
        blob = row["vector"]
        if blob:
            vec = np.frombuffer(blob, dtype=np.float32)
            cos = float(vec @ qvec)
            dense_scored.append((doc_id, cos))
            meta["dense_score"] = cos
        lex_docs.append(LexDoc(doc_id=doc_id, text=text, meta=meta))

    dense_scored.sort(key=lambda item: item[1], reverse=True)
    dense_ranking = dense_scored[:k]

    bm25 = BM25Index()
    bm25.build(lex_docs)
    lex_hits = bm25.search(prepared.expanded_terms, prepared.char_ngrams, k=k)
    for doc_id, score, _meta in lex_hits:
        lexical_ranking.append((doc_id, score))
        if doc_id in meta_by_id:
            meta_by_id[doc_id]["bm25_score"] = score

    # Legacy hybrid ranker (filename + rare terms) as an extra RRF list.
    legacy_hits = store.search(question, qvec, k=k)
    path_to_ids: dict[str, list[str]] = {}
    for doc_id, meta in meta_by_id.items():
        path_to_ids.setdefault(meta["path"], []).append(doc_id)
    for hit in legacy_hits:
        for doc_id in path_to_ids.get(hit["path"], [])[:1]:
            legacy_ranking.append((doc_id, float(hit["score"])))
            meta_by_id[doc_id]["legacy_score"] = float(hit["score"])

    return dense_ranking, lexical_ranking, legacy_ranking, meta_by_id


def _kb_candidates(prepared, k: int) -> tuple[list[tuple[str, float]], list[tuple[str, float]], dict[str, dict]]:
    rows = kb.question_rows()
    meta_by_id: dict[str, dict] = {}
    for row in rows:
        meta_by_id[row["row_id"]] = {
            "id": row["row_id"],
            "source_type": "kb",
            "entry_id": row["entry_id"],
            "question": row["question"],
            "text": row["question"],
            "lang": row["lang"],
            "intent": row["intent"],
            "category": row["category"],
            "answer_am": row["answer_am"],
            "answer_en": row["answer_en"],
            "source_id": row["source_id"],
            "safety": row["safety"],
        }

    # Dense over curated questions
    dense_ranking: list[tuple[str, float]] = []
    vectors = kb.dense_vectors()
    if len(vectors) and rows:
        from local_rag.embedder import embed_query

        qvec = embed_query(prepared.normalized or prepared.raw).astype(np.float32)
        sims = vectors @ qvec
        order = np.argsort(-sims)[:k]
        for idx in order:
            row = rows[int(idx)]
            score = float(sims[int(idx)])
            dense_ranking.append((row["row_id"], score))
            meta_by_id[row["row_id"]]["dense_score"] = score

    # Lexical BM25 over curated questions
    lexical_ranking: list[tuple[str, float]] = []
    for doc_id, score, _meta in kb.bm25_index().search(prepared.expanded_terms, prepared.char_ngrams, k=k):
        lexical_ranking.append((doc_id, score))
        if doc_id in meta_by_id:
            meta_by_id[doc_id]["bm25_score"] = score

    return dense_ranking, lexical_ranking, meta_by_id


def retrieve_verified(
    question: str,
    *,
    session_id: str | None = None,
    k: int = 5,
    prefer_kb: bool = True,
    kind: str | None = None,
) -> dict:
    session = conversation.get_or_create(session_id)
    resolved = conversation.resolve_followup(question, session)
    prepared = prepare_query(resolved["resolved_question"], intent_hints=[resolved["intent"]] if resolved.get("intent") else None)
    kind_filter = kind if kind and kind != "any" else None

    stages = {
        "preprocess": {
            "language": prepared.language,
            "normalized": prepared.normalized,
            "tokens": prepared.tokens,
            "expanded_terms": prepared.expanded_terms,
            "kind_filter": kind_filter,
        },
        "conversation": resolved,
    }

    kb_dense, kb_lex, kb_meta = _kb_candidates(prepared, RETRIEVE_CANDIDATES)
    doc_dense, doc_lex, doc_legacy, doc_meta = _doc_candidates(
        resolved["resolved_question"], prepared, RETRIEVE_CANDIDATES
    )
    if kind_filter:
        # Restrict document pool to the requested media type (e.g. image).
        keep_ids = {doc_id for doc_id, meta in doc_meta.items() if meta.get("kind") == kind_filter}
        doc_dense = [item for item in doc_dense if item[0] in keep_ids]
        doc_lex = [item for item in doc_lex if item[0] in keep_ids]
        doc_legacy = [item for item in doc_legacy if item[0] in keep_ids]
        doc_meta = {doc_id: meta for doc_id, meta in doc_meta.items() if doc_id in keep_ids}
        # Media filters are about the user's files, not the curated KB.
        prefer_kb = False
        kb_dense, kb_lex, kb_meta = [], [], {}

    meta_by_id = {}
    meta_by_id.update(doc_meta)
    meta_by_id.update(kb_meta)  # KB wins on id collision (none expected)

    fileish = bool(kind_filter) or any(
        term in prepared.normalized
        for term in (
            "file", "document", "pdf", "photo", "photos", "picture", "pictures",
            "image", "images", "png", "jpg", "jpeg", "screenshot",
            "video", "invoice", "receipt", "indexed", "wrote", "write",
            "my notes", "folder", "tour", "ፋይል", "ሰነድ", "ፎቶ", "ምስል",
        )
    )

    rankings = []
    if fileish:
        # File questions: documents first; KB only as weak tie-break.
        if doc_legacy:
            rankings.append(doc_legacy)
        if doc_dense:
            rankings.append(doc_dense)
        if doc_lex:
            rankings.append(doc_lex)
        if prefer_kb and kb_dense:
            rankings.append(kb_dense[:10])
        if prefer_kb and kb_lex:
            rankings.append(kb_lex[:10])
    else:
        if prefer_kb and kb_dense:
            rankings.append(kb_dense)
        if prefer_kb and kb_lex:
            rankings.append(kb_lex)
        if doc_dense:
            rankings.append(doc_dense)
        if doc_lex:
            rankings.append(doc_lex)

    fused = reciprocal_rank_fusion(rankings, k=RRF_K) if rankings else []
    candidates = attach_meta(fused, meta_by_id, limit=RETRIEVE_CANDIDATES)

    # Intent soft filter: if conversation resolved an intent, lift matching KB rows.
    if resolved.get("intent"):
        for cand in candidates:
            if cand.get("intent") == resolved["intent"]:
                cand["rrf_score"] = float(cand.get("rrf_score") or 0) + 0.02
        candidates.sort(key=lambda c: c.get("rrf_score") or 0, reverse=True)

    reranked = rerank(
        resolved["resolved_question"],
        candidates,
        text_key="question",
        top_k=max(RERANK_TOP_K, k),
    )

    for cand in reranked:
        cand["confidence"] = confidence.score_candidate(cand, prepared.language)

    # Emergency curated answers jump the queue.
    if confidence.detect_emergency(question):
        for row in kb.question_rows():
            if (row.get("safety") or {}).get("emergency"):
                emergency = {
                    **{k: row[k] for k in row if k != "entry"},
                    "id": row["row_id"],
                    "source_type": "kb",
                    "text": row["question"],
                    "confidence": 1.0,
                    "rrf_score": 1.0,
                }
                reranked = [emergency] + [c for c in reranked if c.get("entry_id") != row["entry_id"]]
                break

    # Curated KB is primary for topical Q&A. Documents win for fileish questions.
    ranked = sorted(reranked, key=lambda c: c.get("confidence") or 0.0, reverse=True)
    kb_hits = [c for c in ranked if c.get("source_type") == "kb"]
    doc_hits = [c for c in ranked if c.get("source_type") == "document"]
    kb_best = kb_hits[0] if kb_hits else None
    doc_best = doc_hits[0] if doc_hits else None

    if fileish:
        if doc_best and (doc_best.get("confidence") or 0) >= 0.48:
            selected_pool = doc_hits
        else:
            selected_pool = []
    elif kb_best and (kb_best.get("confidence") or 0) >= MIN_CONFIDENCE:
        selected_pool = kb_hits
    else:
        selected_pool = []

    validation = confidence.validate_answer(
        question,
        selected_pool,
        min_confidence=MIN_CONFIDENCE,
    )
    stages["hybrid_count"] = len(candidates)
    stages["rerank_count"] = len(reranked)
    stages["validation"] = {
        "ok": validation.ok,
        "confidence": validation.confidence,
        "reason": validation.reason,
        "abstain": validation.abstain,
        "emergency": validation.emergency,
        "source_ids": validation.source_ids,
    }

    answer = None
    sources = (selected_pool or ranked)[:k]
    if validation.ok and selected_pool:
        best = selected_pool[0]
        answer = confidence.pick_answer_text(best, prepared.language)
        conversation.remember(
            session,
            intent=best.get("intent") or resolved.get("intent"),
            category=best.get("category"),
            entry_id=best.get("entry_id"),
            language=prepared.language,
            question=question,
            answer_preview=answer,
        )
    else:
        answer = (
            "I cannot tell from the verified knowledge base or indexed files."
            if prepared.language != "am"
            else "ከተረጋገጠው እውቀት ወይም ከተመዘገቡ ፋይሎች መመለስ አልቻልኩም።"
        )
        if validation.emergency:
            answer = (
                "This may need urgent medical care. Please contact a clinician or emergency services."
                if prepared.language != "am"
                else "ይህ አስቸኳይ የጤና እርዳታ ሊፈልግ ይችላል። የጤና ባለሙያ ወይም ድንገተኛ አገልግሎት ያነጋግሩ።"
            )
        conversation.remember(
            session,
            intent=resolved.get("intent"),
            category=None,
            entry_id=None,
            language=prepared.language,
            question=question,
            answer_preview=answer,
        )

    return {
        "question": question,
        "resolved_question": resolved["resolved_question"],
        "session_id": session.session_id,
        "language": prepared.language,
        "intent": resolved.get("intent"),
        "answer": answer,
        "abstained": validation.abstain or not validation.ok,
        "confidence": validation.confidence,
        "sources": sources,
        "stages": stages,
        "llm_used": False,
    }
