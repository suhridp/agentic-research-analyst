"""Retrieval: BM25, dense, hybrid (RRF) and reranking behind one interface.

    retriever = build_retriever({"kind": "hybrid", "encoder": "auto", "rerank": "auto"}, corpus)
    hits = retriever.search("query", k=5)        # list[Hit]
    retriever.describe()                          # which backends ACTUALLY ran - recorded with results

"auto" picks the neural backend when its package is installed and otherwise falls back to an
offline stand-in. The fallback is never silent: `describe()` names it, and the eval runner writes it
into every results file, so an LSA run can't be mistaken for a sentence-transformer run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..corpus import Corpus
from .base import Hit, Retriever
from .bm25 import BM25
from .dense import DenseRetriever, make_encoder
from .hybrid import HybridRRF
from .rerank import RerankedRetriever, make_reranker

__all__ = ["Hit", "Retriever", "BM25", "DenseRetriever", "HybridRRF", "RerankedRetriever", "build_retriever"]

_CACHE: dict[tuple[str, str], Retriever] = {}


def build_retriever(cfg: dict[str, Any] | str, corpus: Corpus, *, llm=None,
                    cache_dir: str | Path = ".index_cache") -> Retriever:
    if isinstance(cfg, str):
        cfg = {"kind": cfg}
    key = (corpus.fingerprint, json.dumps(cfg, sort_keys=True))
    if key in _CACHE and cfg.get("rerank") != "llm":
        return _CACHE[key]
    kind = cfg.get("kind", "bm25")
    if kind == "bm25":
        r: Retriever = BM25(corpus.passages, k1=cfg.get("k1", 1.2), b=cfg.get("b", 0.75))
    elif kind in ("dense", "hybrid"):
        enc = make_encoder(cfg.get("encoder", "auto"))
        dense = DenseRetriever(corpus, enc, cache_dir=cache_dir)
        r = dense if kind == "dense" else HybridRRF(
            [BM25(corpus.passages), dense], depth=cfg.get("fusion_depth", 50), k_rrf=cfg.get("k_rrf", 60))
    else:
        raise ValueError(f"unknown retriever kind: {kind}")
    if cfg.get("rerank"):
        r = RerankedRetriever(r, make_reranker(cfg["rerank"], llm=llm), depth=cfg.get("rerank_depth", 30))
    _CACHE[key] = r
    return r
