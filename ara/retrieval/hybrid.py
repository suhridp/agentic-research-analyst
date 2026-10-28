"""Hybrid retrieval by Reciprocal Rank Fusion (Cormack et al., 2009).

RRF fuses ranks, not scores, so BM25's unbounded scores and cosine similarities never need
calibrating against each other: score(p) = sum over retrievers of 1 / (k_rrf + rank_r(p)).
"""
from __future__ import annotations

from .base import Hit


class HybridRRF:
    def __init__(self, retrievers: list, depth: int = 50, k_rrf: int = 60):
        self.retrievers, self.depth, self.k_rrf = retrievers, depth, k_rrf

    def search(self, query: str, k: int = 5) -> list[Hit]:
        fused: dict[str, float] = {}
        by_id = {}
        for r in self.retrievers:
            for rank, h in enumerate(r.search(query, max(self.depth, k)), 1):
                fused[h.passage.id] = fused.get(h.passage.id, 0.0) + 1.0 / (self.k_rrf + rank)
                by_id[h.passage.id] = h.passage
        order = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        return [Hit(by_id[i], fused[i]) for i in order]

    def describe(self) -> dict:
        return {"kind": "hybrid_rrf", "k_rrf": self.k_rrf, "depth": self.depth,
                "components": [r.describe() for r in self.retrievers]}
