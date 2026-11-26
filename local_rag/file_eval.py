"""Real-file search evaluation (Precision@k, MRR) against the local index."""

from __future__ import annotations

import json
import time
from pathlib import Path

from local_rag.retrieval import retrieve_verified
from local_rag.settings import DATA_DIR

DEFAULT_PATH = DATA_DIR / "eval" / "file_search_questions.json"


def _name_match(hit: dict, substrings: list[str]) -> bool:
    name = (hit.get("name") or "") + " " + (hit.get("path") or "")
    name_l = name.lower()
    return any(s.lower() in name_l for s in substrings)


def run_file_eval(path: Path | None = None) -> dict:
    data = json.loads((path or DEFAULT_PATH).read_text(encoding="utf-8"))
    items = data.get("items") or []
    details = []
    precisions = []
    mrrs = []
    abstain_ok = 0
    abstain_total = 0
    latencies = []

    for item in items:
        t0 = time.perf_counter()
        result = retrieve_verified(
            item["question"],
            k=10,
            kind=item.get("kind"),
            country=item.get("country"),
            year=item.get("year"),
            city=item.get("city"),
        )
        latencies.append(time.perf_counter() - t0)
        sources = [s for s in (result.get("sources") or []) if s.get("source_type") != "kb"]
        top5 = sources[:5]
        expected = item.get("expected_name_substrings") or []
        kinds = set(item.get("expected_kinds") or [])

        if item.get("ood"):
            abstain_total += 1
            # Success: abstain, or no confidently wrong Italy-tagged trip dump.
            ok = bool(result.get("abstained")) or len(sources) == 0
            if not ok:
                # Also accept if none of the top hits claim Italy.
                ok = not any((s.get("country") or "").lower() == "italy" for s in top5)
            if ok:
                abstain_ok += 1
            details.append(
                {
                    "id": item["id"],
                    "ood": True,
                    "ok": ok,
                    "abstained": result.get("abstained"),
                    "n_sources": len(sources),
                    "top": [s.get("name") for s in top5],
                    "latency_s": latencies[-1],
                }
            )
            continue

        expected_country = (item.get("expected_country") or "").lower()

        def is_relevant(hit: dict) -> bool:
            name_ok = _name_match(hit, expected) if expected else False
            kind_ok = (not kinds) or (hit.get("kind") in kinds)
            country_ok = (not expected_country) or (hit.get("country") or "").lower() == expected_country
            if expected and kinds:
                return bool(name_ok and kind_ok)
            if expected and expected_country:
                return bool(name_ok or country_ok)
            if expected:
                return bool(name_ok)
            if kinds:
                return bool(kind_ok)
            return bool(country_ok)

        relevant_flags = [is_relevant(hit) for hit in top5]
        precision = (sum(relevant_flags) / len(relevant_flags)) if relevant_flags else 0.0
        precisions.append(precision)

        rr = 0.0
        for rank, hit in enumerate(sources[:10], start=1):
            if is_relevant(hit):
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        details.append(
            {
                "id": item["id"],
                "precision_at_5": precision,
                "rr": rr,
                "abstained": result.get("abstained"),
                "confidence": result.get("confidence"),
                "top": [
                    {
                        "name": s.get("name"),
                        "kind": s.get("kind"),
                        "country": s.get("country"),
                        "reasons": s.get("match_reasons"),
                    }
                    for s in top5
                ],
                "answer_preview": (result.get("answer") or "")[:220],
                "latency_s": latencies[-1],
            }
        )

    metrics = {
        "n_items": len(items),
        "precision_at_5": (sum(precisions) / len(precisions)) if precisions else None,
        "mrr": (sum(mrrs) / len(mrrs)) if mrrs else None,
        "abstain_accuracy": (abstain_ok / abstain_total) if abstain_total else None,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "p95_latency_s": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else None,
    }
    targets = data.get("targets") or {}
    return {
        "metrics": metrics,
        "targets": targets,
        "passes": {
            key: metrics.get(key) is not None and metrics[key] >= targets[key]
            for key in targets
            if key in metrics
        },
        "note": data.get("note"),
        "details": details,
    }
