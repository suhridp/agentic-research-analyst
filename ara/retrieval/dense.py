"""Dense retrieval with a pluggable encoder and vector store.

Encoders
  * SBERTEncoder - sentence-transformers (default BAAI/bge-small-en-v1.5, which wants a query prefix).
  * LSAEncoder   - TF-IDF + truncated SVD (latent semantic analysis). NOT a neural model: it is an
                   offline stand-in so the dense/hybrid code paths run and are tested on machines
                   without torch. Never report LSA numbers as "dense retrieval".
Vector store
  * FAISS IndexFlatIP when faiss is installed, else exact numpy inner product (fine up to ~100k passages).
Passage embeddings are cached on disk keyed by corpus fingerprint + encoder name.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np

from ..corpus import Corpus
from .base import Hit


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class SBERTEncoder:
    PREFIXES = {"bge": ("Represent this sentence for searching relevant passages: ", ""),
                "e5": ("query: ", "passage: ")}

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model_name = model
        self.model = SentenceTransformer(model)
        self.batch_size = batch_size
        fam = next((f for f in self.PREFIXES if f in model.lower()), None)
        self.q_prefix, self.p_prefix = self.PREFIXES.get(fam, ("", ""))

    @property
    def name(self) -> str:
        return f"sbert:{self.model_name}"

    def fit(self, texts: list[str]) -> None:
        pass

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([self.p_prefix + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=len(texts) > 500).astype("float32")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([self.q_prefix + t for t in texts], normalize_embeddings=True).astype("float32")


class LSAEncoder:
    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim, self.seed = dim, seed
        self.vec = self.svd = None

    @property
    def name(self) -> str:
        return f"lsa:{self.dim}"

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        from ..text import tokens
        self.vec = TfidfVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None,
                                   ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        X = self.vec.fit_transform(texts)
        dim = max(1, min(self.dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(dim, random_state=self.seed).fit(X)

    def _enc(self, texts: list[str]) -> np.ndarray:
        if self.vec is None:
            raise RuntimeError("LSAEncoder.fit must be called first")
        return _l2norm(self.svd.transform(self.vec.transform(texts))).astype("float32")

    encode_passages = encode_queries = _enc


def make_encoder(spec: str = "auto"):
    """'auto' | 'lsa' | 'sbert' | 'sbert:<hf model id>'"""
    if spec == "lsa":
        return LSAEncoder()
    if spec.startswith("sbert"):
        model = spec.split(":", 1)[1] if ":" in spec else "BAAI/bge-small-en-v1.5"
        return SBERTEncoder(model)
    if spec == "auto":
        try:
            return SBERTEncoder()
        except Exception as e:  # ImportError, or model download failure offline
            warnings.warn(f"sentence-transformers unavailable ({type(e).__name__}); using LSA stand-in", stacklevel=2)
            return LSAEncoder()
    raise ValueError(f"unknown encoder: {spec}")


class DenseRetriever:
    def __init__(self, corpus: Corpus, encoder, cache_dir: str | Path | None = ".index_cache"):
        self.passages = corpus.passages
        self.encoder = encoder
        texts = [p.index_text for p in self.passages]
        if isinstance(encoder, LSAEncoder):
            encoder.fit(texts)  # cheap; LSA is corpus-specific so never cached across corpora
            self.emb = encoder.encode_passages(texts)
        else:
            self.emb = self._cached(corpus.fingerprint, texts, cache_dir)
        try:
            import faiss  # optional
            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)
            self.store = "faiss"
        except ImportError:
            self.index, self.store = None, "numpy"

    def _cached(self, fp: str, texts: list[str], cache_dir) -> np.ndarray:
        if not cache_dir:
            return self.encoder.encode_passages(texts)
        h = hashlib.sha256(f"{fp}|{self.encoder.name}".encode()).hexdigest()[:20]
        f = Path(cache_dir) / f"emb_{h}.npy"
        if f.exists():
            emb = np.load(f)
            if emb.shape[0] == len(texts):
                return emb
        emb = self.encoder.encode_passages(texts)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, emb)
        return emb

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self.passages:
            return []
        q = self.encoder.encode_queries([query])
        k = min(k, len(self.passages))
        if self.index is not None:
            scores, idx = self.index.search(q, k)
            return [Hit(self.passages[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]
        s = self.emb @ q[0]
        top = np.argsort(-s, kind="stable")[:k]
        return [Hit(self.passages[i], float(s[i])) for i in top]

    def describe(self) -> dict:
        return {"kind": "dense", "encoder": self.encoder.name, "store": self.store,
                "neural": not isinstance(self.encoder, LSAEncoder)}
