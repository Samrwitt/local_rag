"""Reciprocal Rank Fusion and ranking utilities."""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine ranked lists with RRF. Each item is (doc_id, score_ignored_for_rank)."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def attach_meta(
    fused: list[tuple[str, float]],
    meta_by_id: dict[str, dict],
    limit: int,
) -> list[dict]:
    out = []
    for doc_id, score in fused[:limit]:
        meta = dict(meta_by_id.get(doc_id) or {})
        meta["id"] = doc_id
        meta["rrf_score"] = score
        out.append(meta)
    return out
