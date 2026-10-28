"""Okapi BM25 over passage index_text, with an inverted index so query cost scales with postings."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from ..corpus import Passage
from ..text import tokens
from .base import Hit


class BM25:
    def __init__(self, passages: list[Passage], k1: float = 1.2, b: float = 0.75):
        self.passages, self.k1, self.b = passages, k1, b
        tfs = [Counter(tokens(p.index_text)) for p in passages]
        self.doclen = np.array([sum(t.values()) for t in tfs], dtype=float)
        avg = self.doclen.mean() if len(passages) else 1.0
        self.norm = k1 * (1 - b + b * self.doclen / max(avg, 1e-9))
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(tfs):
            for w, f in t.items():
                self.postings[w].append((i, f))
        n = len(passages)
        self.idf = {w: math.log(1 + (n - len(ps) + 0.5) / (len(ps) + 0.5)) for w, ps in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.passages))
        for w in set(tokens(query)):
            for i, f in self.postings.get(w, ()):
                s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.norm[i])
        return s

    def search(self, query: str, k: int = 5) -> list[Hit]:
        s = self.scores(query)
        nz = np.flatnonzero(s)
        top = nz[np.argsort(-s[nz], kind="stable")][:k]
        return [Hit(self.passages[i], float(s[i])) for i in top]

    def describe(self) -> dict:
        return {"kind": "bm25", "k1": self.k1, "b": self.b}
