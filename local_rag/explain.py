"""Human-readable match explanations (no technical jargon for end users)."""

from __future__ import annotations

from pathlib import Path

from local_rag.place import parse_trip_query


def explain_hit(question: str, hit: dict) -> list[str]:
    """Why this result matches — short, plain-language reasons."""
    reasons: list[str] = []
    hints = parse_trip_query(question)
    name = hit.get("name") or (Path(hit["path"]).name if hit.get("path") else "This file")
    text = (hit.get("text") or hit.get("question") or "").lower()
    path_l = (hit.get("path") or name).lower()
    q_l = (question or "").lower()

    country = hit.get("country")
    city = hit.get("city")
    year = (hit.get("captured_at") or "")[:4]
    roles = [r for r in (hit.get("roles") or "").split(",") if r]
    kind = hit.get("kind") or hit.get("source_type")

    if hints.get("country") and country and str(country).lower() == str(hints["country"]).lower():
        reasons.append(f"Tagged as {country}")
    elif hints.get("country") and hints["country"].lower() in path_l:
        reasons.append(f"Filename mentions {hints['country']}")

    if hints.get("city") and city and str(city).lower() == str(hints["city"]).lower():
        reasons.append(f"City: {city}")

    if hints.get("year") and year == str(hints["year"]):
        reasons.append(f"Dated {year}")
    elif hints.get("year") and str(hints["year"]) in path_l:
        reasons.append(f"Year {hints['year']} in the filename")

    if hints.get("trip") and roles:
        trip_roles = [r for r in roles if r in {"receipt", "hotel", "booking", "memo"}]
        if trip_roles:
            reasons.append("Looks like travel paperwork (" + ", ".join(trip_roles) + ")")

    if kind == "image":
        reasons.append("Photo")
    elif kind == "video":
        reasons.append("Video (from file details only)")
    elif kind in {"pdf", "docx", "pptx", "xlsx", "text"}:
        reasons.append("Document")

    # Keyword overlap for distinctive terms
    for token in ("samrawit", "addis", "ethiopia", "invoice", "receipt", "cv", "hotel", "dubai"):
        if token in q_l and (token in text or token in path_l):
            reasons.append(f"Mentions “{token}”")

    if hit.get("source_type") == "related":
        for r in hit.get("match_reasons") or []:
            reasons.append(r)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            out.append(reason)
    if not out:
        out.append("Possibly related to your search")
    return out[:6]


def attach_explanations(question: str, hits: list[dict]) -> list[dict]:
    for hit in hits:
        hit["match_reasons"] = explain_hit(question, hit)
    return hits
