import warnings

from ara.corpus import Passage
from ara.retrieval import BM25, HybridRRF, build_retriever
from ara.retrieval.base import Hit
from ara.retrieval.rerank import LexicalReranker, RerankedRetriever


def test_bm25_ranks_gold_first(corpus):
    top = BM25(corpus.passages).search("gradient boosted classifier DoS floods accuracy", 3)
    assert top[0].passage.id == "syn_dos#1"


def test_bm25_returns_nothing_for_unknown_terms(corpus):
    assert BM25(corpus.passages).search("terahertz quantum blockchain", 5) == []


def test_dense_lsa_fallback_is_labelled_non_neural(corpus):
    r = build_retriever({"kind": "dense", "encoder": "lsa"}, corpus, cache_dir=None)
    d = r.describe()
    assert d["neural"] is False and d["encoder"].startswith("lsa")
    assert r.search("spoofed measurement reports rogue cell", 3)


def test_auto_encoder_never_silently_claims_neural(corpus):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = build_retriever({"kind": "hybrid", "encoder": "auto"}, corpus, cache_dir=None)
    comps = r.describe()["components"]
    dense = next(c for c in comps if c["kind"] == "dense")
    try:
        import sentence_transformers  # noqa: F401
        assert dense["neural"] is True
    except ImportError:
        assert dense["neural"] is False


class _Fixed:
    def __init__(self, ids):
        self.ids = ids

    def search(self, q, k=5):
        return [Hit(Passage("d", int(i), f"text {i}"), 1.0) for i in self.ids[:k]]

    def describe(self):
        return {"kind": "fixed"}


def test_rrf_rewards_agreement_between_retrievers():
    fused = HybridRRF([_Fixed(["1", "2", "3"]), _Fixed(["3", "1", "4"])], depth=3).search("q", 4)
    assert [h.passage.id for h in fused][:2] == ["d#1", "d#3"]  # top in both lists beat single-list hits


def test_reranker_reorders_first_stage(corpus):
    class Rev:
        def search(self, q, k=5):
            hits = BM25(corpus.passages).search(q, k)
            return list(reversed(hits))

        def describe(self):
            return {"kind": "rev"}
    rr = RerankedRetriever(Rev(), LexicalReranker(), depth=10)
    assert rr.search("anomaly detection flagged spoofed reports false positive rate", 1)[0].passage.id == "syn_defenses#1"
