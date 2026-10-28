"""Neural backends. Skipped unless sentence-transformers is installed (CI job `neural-backends`)."""
import pytest

st = pytest.importorskip("sentence_transformers")

from ara.retrieval import build_retriever  # noqa: E402


def test_sbert_dense_retrieval_is_neural_and_finds_gold(corpus, tmp_path):
    r = build_retriever({"kind": "dense", "encoder": "sbert:BAAI/bge-small-en-v1.5"}, corpus, cache_dir=tmp_path)
    assert r.describe()["neural"] is True
    ids = [h.passage.id for h in r.search("How accurate was DoS detection on synthetic data?", 3)]
    assert "syn_dos#1" in ids
    # embeddings are cached on disk and reused
    assert list(tmp_path.glob("emb_*.npy"))


def test_cross_encoder_rerank(corpus, tmp_path):
    r = build_retriever({"kind": "bm25", "rerank": "cross-encoder"}, corpus, cache_dir=tmp_path)
    assert r.describe()["neural_reranker"] is True
    top = r.search("anomaly detection false positive rate on spoofed reports", 1)[0]
    assert top.passage.id == "syn_defenses#1"
