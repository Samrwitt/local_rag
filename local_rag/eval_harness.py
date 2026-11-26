"""Evaluate retrieval quality before any generative LLM is added."""

from __future__ import annotations

import json
from pathlib import Path

from local_rag.retrieval import retrieve_verified
from local_rag.settings import ACCEPTANCE_TARGETS, EVAL_PATH


def load_eval_set(path: Path | None = None) -> dict:
    path = path or Path(EVAL_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(path: Path | None = None) -> dict:
    data = load_eval_set(path)
    items = data.get("items") or []
    targets = data.get("targets") or ACCEPTANCE_TARGETS

    top1_ok = 0
    recall5_ok = 0
    intent_ok = 0
    intent_total = 0
    ood_ok = 0
    ood_total = 0
    traced = 0
    answered = 0
    details = []

    for item in items:
        session_id = None
        if item.get("context_question"):
            warm = retrieve_verified(item["context_question"], session_id=None, k=5)
            session_id = warm["session_id"]
        result = retrieve_verified(item["question"], session_id=session_id, k=5)
        sources = result.get("sources") or []
        entry_ids = [s.get("entry_id") for s in sources if s.get("entry_id")]
        expected = item.get("expected_entry_id")
        ood = bool(item.get("ood"))

        top1 = bool(sources and sources[0].get("entry_id") == expected) if expected else False
        recall5 = expected in entry_ids if expected else False
        if ood:
            ood_total += 1
            # OOD success = abstain / no KB hit
            success = bool(result.get("abstained")) or not any(s.get("source_type") == "kb" for s in sources[:1])
            if expected is None and result.get("abstained"):
                success = True
            if success:
                ood_ok += 1
        else:
            if top1:
                top1_ok += 1
            if recall5 or (item.get("emergency") and not result.get("abstained")):
                recall5_ok += 1
            answered += 1

        if item.get("expected_intent"):
            intent_total += 1
            if result.get("intent") == item["expected_intent"] or (
                sources and sources[0].get("intent") == item["expected_intent"]
            ):
                intent_ok += 1

        source_ids = result.get("stages", {}).get("validation", {}).get("source_ids") or []
        if source_ids or result.get("abstained"):
            traced += 1

        details.append(
            {
                "id": item["id"],
                "top1": top1,
                "recall5": recall5,
                "intent": result.get("intent"),
                "abstained": result.get("abstained"),
                "confidence": result.get("confidence"),
                "answer_preview": (result.get("answer") or "")[:160],
            }
        )

    n = max(answered, 1)
    metrics = {
        "n_items": len(items),
        "top1_accuracy": top1_ok / n,
        "recall_at_5": recall5_ok / n,
        "intent_accuracy": (intent_ok / intent_total) if intent_total else None,
        "ood_detection": (ood_ok / ood_total) if ood_total else None,
        "source_traceability": traced / max(len(items), 1),
    }
    return {
        "metrics": metrics,
        "targets": targets,
        "passes": {
            key: (metrics[key] is not None and metrics[key] >= targets.get(key, 0))
            for key in targets
            if key in metrics
        },
        "details": details,
    }
