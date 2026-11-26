"""Confidence-based cross-file relationship discovery.

Links files that likely belong to the same trip, event, or subject.
Never groups solely by year — requires at least one stronger signal.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from local_rag import store


@dataclass
class Relation:
    path_a: str
    path_b: str
    score: float
    reasons: list[str]
    relation_type: str  # trip | event | subject | duplicate


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00").split("+")[0])
    except ValueError:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if t not in {"the", "and", "for", "with"}}


def _pair_score(a: dict, b: dict) -> Relation | None:
    if a["path"] == b["path"]:
        return None
    # Stable order
    if a["path"] > b["path"]:
        a, b = b, a

    score = 0.0
    reasons: list[str] = []
    strong = 0

    # Same content preview / near-duplicate name
    name_a = (a.get("name") or "").lower()
    name_b = (b.get("name") or "").lower()
    stem_a = Path(name_a).stem
    stem_b = Path(name_b).stem
    if stem_a and stem_a == stem_b:
        score += 0.55
        strong += 1
        reasons.append("same filename stem")
    elif stem_a and stem_b and (stem_a in stem_b or stem_b in stem_a) and min(len(stem_a), len(stem_b)) >= 8:
        score += 0.25
        reasons.append("similar filenames")

    country_a = (a.get("country") or "").strip()
    country_b = (b.get("country") or "").strip()
    city_a = (a.get("city") or "").strip()
    city_b = (b.get("city") or "").strip()

    same_country = bool(country_a and country_a == country_b)
    same_city = bool(city_a and city_a == city_b)

    ta = _parse_time(a.get("captured_at"))
    tb = _parse_time(b.get("captured_at"))
    days = None
    if ta and tb:
        days = abs((ta - tb).total_seconds()) / 86400.0

    # Geographic proximity from GPS
    if a.get("lat") is not None and b.get("lat") is not None and a.get("lon") is not None and b.get("lon") is not None:
        dist = _haversine_km(float(a["lat"]), float(a["lon"]), float(b["lat"]), float(b["lon"]))
        if dist <= 5:
            score += 0.45
            strong += 1
            reasons.append(f"GPS within {dist:.1f} km")
        elif dist <= 50:
            score += 0.2
            reasons.append(f"GPS within {dist:.0f} km")

    # Shared city/country only count as strong with time proximity or name similarity.
    close_in_time = days is not None and days <= 21
    if same_city and close_in_time:
        score += 0.35
        strong += 1
        reasons.append(f"same city ({city_a})")
    elif same_city:
        score += 0.12
        reasons.append(f"same city ({city_a})")
    elif same_country and close_in_time:
        score += 0.18
        reasons.append(f"same country ({country_a})")
    elif same_country:
        score += 0.06
        reasons.append(f"same country ({country_a})")

    # Temporal proximity — only counts with another signal
    if days is not None:
        if days <= 3:
            score += 0.3
            reasons.append(f"within {days:.0f} days")
        elif days <= 14:
            score += 0.18
            reasons.append(f"within {days:.0f} days")
        elif days <= 45:
            score += 0.08
            reasons.append(f"within {days:.0f} days")

    # Shared folder (parent path) — weak alone
    parent_a = str(Path(a["path"]).parent)
    parent_b = str(Path(b["path"]).parent)
    if parent_a == parent_b:
        score += 0.12
        reasons.append("same folder")
    elif Path(parent_a).parent == Path(parent_b).parent and parent_a != parent_b:
        score += 0.04
        reasons.append("sibling folders")

    # Shared roles need place + time; roles alone over-fire on academic PDFs.
    roles_a = {r for r in (a.get("roles") or "").split(",") if r}
    roles_b = {r for r in (b.get("roles") or "").split(",") if r}
    shared_roles = roles_a & roles_b
    trip_roles = {"receipt", "hotel", "booking", "memo"}
    if shared_roles & trip_roles and same_city and close_in_time:
        score += 0.22
        strong += 1
        reasons.append("shared trip roles: " + ", ".join(sorted(shared_roles & trip_roles)))
    elif (roles_a & trip_roles) and (roles_b & trip_roles) and same_city and close_in_time:
        score += 0.2
        strong += 1
        reasons.append("complementary trip documents")

    # Preview overlap only counts with another geographic/temporal/role signal.
    # Metadata cards share boilerplate words and otherwise create false links.
    tok_a = _tokens(a.get("preview") or "")
    tok_b = _tokens(b.get("preview") or "")
    preview_overlap = 0.0
    if tok_a and tok_b:
        preview_overlap = len(tok_a & tok_b) / max(1, min(len(tok_a), len(tok_b)))
        if preview_overlap >= 0.45 and (same_city or same_country or (days is not None and days <= 14) or shared_roles):
            score += min(0.2, preview_overlap * 0.3)
            reasons.append(f"preview overlap {preview_overlap:.0%}")

    # Reject year-only / country-only / folder-only weak links
    year_only = (
        days is not None
        and ta is not None
        and tb is not None
        and ta.year == tb.year
        and days > 45
        and not same_city
        and strong == 0
        and not (shared_roles & trip_roles)
    )
    if year_only:
        return None
    folder_only = strong == 0 and not same_city and not (shared_roles & trip_roles) and not (
        days is not None and days <= 14 and same_country
    )
    if folder_only and score < 0.7:
        return None

    # Require either a strong signal or a combined score above threshold
    if strong == 0 and score < 0.55:
        return None
    if score < 0.5:
        return None

    relation_type = "trip" if (same_country or same_city) and (days is not None and days <= 45 or shared_roles & trip_roles) else "subject"
    if "same filename stem" in reasons:
        relation_type = "duplicate"
    return Relation(
        path_a=a["path"],
        path_b=b["path"],
        score=min(1.0, score),
        reasons=reasons,
        relation_type=relation_type,
    )


def discover_relations(max_files: int = 800, min_score: float = 0.5) -> list[Relation]:
    """Scan indexed files and persist pairwise relationships above threshold."""
    files = store.list_files()
    if len(files) > max_files:
        # Prefer files that already have place/time/roles signals.
        files = sorted(
            files,
            key=lambda f: (
                1 if f.get("country") else 0,
                1 if f.get("roles") else 0,
                1 if f.get("captured_at") else 0,
            ),
            reverse=True,
        )[:max_files]

    relations: list[Relation] = []
    for i, a in enumerate(files):
        for b in files[i + 1 :]:
            rel = _pair_score(a, b)
            if rel and rel.score >= min_score:
                relations.append(rel)

    store.replace_relations(
        [
            {
                "path_a": r.path_a,
                "path_b": r.path_b,
                "score": r.score,
                "reasons": "; ".join(r.reasons),
                "relation_type": r.relation_type,
            }
            for r in relations
        ]
    )
    return relations


def related_paths(path: str, limit: int = 8) -> list[dict]:
    return store.relations_for(path, limit=limit)


def expand_with_relations(hits: list[dict], limit_extra: int = 4) -> list[dict]:
    """Attach related files for top hits without flooding the result list."""
    seen = {h.get("path") for h in hits if h.get("path")}
    extras = []
    for hit in hits[:3]:
        path = hit.get("path")
        if not path:
            continue
        for rel in related_paths(path, limit=4):
            other = rel["path_b"] if rel["path_a"] == path else rel["path_a"]
            if other in seen:
                continue
            seen.add(other)
            extras.append(
                {
                    "path": other,
                    "name": Path(other).name,
                    "source_type": "related",
                    "relation_score": rel["score"],
                    "relation_type": rel["relation_type"],
                    "match_reasons": [f"Related to {Path(path).name}: {rel['reasons']}"],
                    "text": f"Related file ({rel['relation_type']}, score {rel['score']:.2f}). {rel['reasons']}",
                    "confidence": min(0.7, float(rel["score"])),
                }
            )
            if len(extras) >= limit_extra:
                return hits + extras
    return hits + extras


def group_summary(files: list[dict]) -> dict[str, list[str]]:
    """Bucket paths by country+year for debugging / UI."""
    groups: dict[str, list[str]] = defaultdict(list)
    for f in files:
        country = f.get("country") or "unknown"
        year = (f.get("captured_at") or "")[:4] or "unknown"
        key = f"{country}:{year}"
        groups[key].append(f.get("path") or "")
    return dict(groups)
