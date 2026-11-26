"""Evidence-backed attributes for enrichment (observed vs inferred)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AttributeEvidence:
    """One inferred or observed attribute with provenance."""

    name: str
    value: Any
    method: str
    confidence: float
    observed: bool
    evidence: str = ""
    alternatives: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def pack_attributes(items: list[AttributeEvidence]) -> dict:
    """Serialize attributes keyed by name for storage."""
    return {item.name: item.to_dict() for item in items}


def attribute(
    name: str,
    value: Any,
    *,
    method: str,
    confidence: float,
    observed: bool,
    evidence: str = "",
    alternatives: list[Any] | None = None,
) -> AttributeEvidence:
    return AttributeEvidence(
        name=name,
        value=value,
        method=method,
        confidence=max(0.0, min(1.0, float(confidence))),
        observed=observed,
        evidence=evidence[:500],
        alternatives=list(alternatives or []),
    )
