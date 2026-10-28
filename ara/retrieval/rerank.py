"""Second-stage rerankers: rescore the top `depth` first-stage candidates and keep the best k.

  * CrossEncoderReranker - sentence-transformers CrossEncoder (default ms-marco-MiniLM-L-6-v2).
  * LLMReranker          - asks the LLM to grade each candidate 0-3 in one batched call.
  * LexicalReranker      - query-term and bigram coverage. Offline stand-in, not a learned model.
"""
from __future__ import annotations

import warnings

from ..text import tokens
from .base import Hit


class LexicalReranker:
    name = "lexical"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = tokens(query)
        qs, qb = set(q), set(zip(q, q[1:]))
        out = []
        for t in texts:
            tt = tokens(t)
            ts, tb = set(tt), set(zip(tt, tt[1:]))
            cov = len(qs & ts) / max(len(qs), 1)
            big = len(qb & tb) / max(len(qb), 1) if qb else 0.0
            out.append(cov + 0.5 * big)
        return out


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # optional dependency
        self.model_name = model
        self.model = CrossEncoder(model)

    @property
    def name(self) -> str:
        return f"cross-encoder:{self.model_name}"

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


class LLMReranker:
    SYS = ("You grade how useful each passage is for answering a research question. "
           "3 = directly answers it, 2 = partially, 1 = related background, 0 = irrelevant. "
           'Return ONLY JSON: {"<passage number>": <grade>, ...} covering every passage.')

    def __init__(self, llm):
        if llm is None:
            raise ValueError("LLM reranker needs an LLM")
        self.llm = llm

    @property
    def name(self) -> str:
        return f"llm:{self.llm.name}"

    def score(self, query: str, texts: list[str]) -> list[float]:
        listing = "\n\n".join(f"[{i}] {t[:1200]}" for i, t in enumerate(texts))
        out = self.llm.complete_json(self.SYS, f"Question: {query}\n\nPassages:\n{listing}", 400,
                                     validate=lambda o: isinstance(o, dict), tag="rerank")
        out = out or {}
        # ties broken by first-stage order: tiny decreasing epsilon
        return [float(out.get(str(i), 0) or 0) - i * 1e-6 for i in range(len(texts))]


def make_reranker(spec: str, llm=None):
    if spec == "lexical":
        return LexicalReranker()
    if spec == "llm":
        return LLMReranker(llm)
    if spec.startswith("cross-encoder"):
        return CrossEncoderReranker(spec.split(":", 1)[1]) if ":" in spec else CrossEncoderReranker()
    if spec == "auto":
        try:
            return CrossEncoderReranker()
        except Exception as e:
            warnings.warn(f"cross-encoder unavailable ({type(e).__name__}); using lexical reranker", stacklevel=2)
            return LexicalReranker()
    raise ValueError(f"unknown reranker: {spec}")


class RerankedRetriever:
    def __init__(self, first_stage, reranker, depth: int = 30):
        self.first, self.reranker, self.depth = first_stage, reranker, depth

    def search(self, query: str, k: int = 5) -> list[Hit]:
        cands = self.first.search(query, max(self.depth, k))
        if not cands:
            return []
        scores = self.reranker.score(query, [h.passage.index_text for h in cands])
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], i))[:k]
        return [Hit(cands[i].passage, scores[i]) for i in order]

    def describe(self) -> dict:
        return {"kind": "reranked", "reranker": self.reranker.name, "depth": self.depth,
                "first_stage": self.first.describe(),
                "neural_reranker": self.reranker.name.startswith(("cross-encoder", "llm"))}
