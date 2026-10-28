"""End-to-end pipeline tests, offline and with a fake LLM that hallucinates in realistic ways.
The fake-LLM tests check the MECHANISM (safeguards catch planted errors); they are not results."""
from ara.agents import ABSTAIN, cite_ids
from ara.pipeline import build_system
from eval.judge import LexicalJudge
from eval.metrics import answer_units

Q_DOS = "How does DoS detection accuracy on synthetic data compare with accuracy on real operator traffic?"


def test_offline_agentic_answers_with_valid_citations(corpus):
    rep = build_system({"name": "a"}, corpus).answer(Q_DOS)
    assert not rep.abstained and rep.claims
    assert all(i in corpus.by_id for i in cite_ids(rep.answer))
    assert {"syn_dos#1", "syn_dos_replication#1"} <= set(rep.cited)


def test_offline_agentic_abstains_with_gap(corpus):
    rep = build_system({"name": "a"}, corpus).answer("What is the capital of France?")
    assert rep.abstained and rep.answer == ABSTAIN and rep.gaps


def test_offline_rag_baseline(corpus):
    rep = build_system({"name": "r", "kind": "rag"}, corpus).answer(Q_DOS)
    assert rep.claims and all(c.verdict == "unchecked" for c in rep.claims)


def test_unknown_system_key_is_rejected(corpus):
    import pytest
    with pytest.raises(ValueError, match="unknown keys"):
        build_system({"name": "x", "verifyer": "none"}, corpus)


def _unsupported(rep, corpus):
    units, _ = answer_units(rep.answer)
    j = LexicalJudge()
    return [t for t, ids in units if not j.supported(t, [corpus.by_id[i].text for i in ids])]


def test_safeguards_catch_planted_hallucinations(corpus, fake_llm):
    full = build_system({"name": "full"}, corpus, fake_llm).answer(Q_DOS)
    rejected = {c.text for c in full.rejected}
    assert any("99" in t for t in rejected), "number filter should reject the invented number"
    assert any("improved" in t for t in rejected), "LLM verifier should reject the flipped direction"
    assert "50 percent" not in full.answer, "critic should drop the synthesizer's invented statistic"
    assert full.llm_calls_saved_by_filter >= 1
    assert _unsupported(full, corpus) == []


def test_ablations_let_planted_errors_through(corpus, fake_llm):
    no_ver = build_system({"name": "nv", "verifier": "none"}, corpus, fake_llm).answer(Q_DOS)
    assert any("99" in c.text for c in no_ver.claims), "without the verifier the invented number survives"
    no_critic = build_system({"name": "nc", "critic": False}, corpus, fake_llm).answer(Q_DOS)
    assert "50 percent" in no_critic.answer, "without the critic the invented statistic survives"
    rag = build_system({"name": "rag", "kind": "rag"}, corpus, fake_llm).answer(Q_DOS)
    assert _unsupported(rag, corpus), "the unverified baseline keeps its hallucination"


def test_retry_round_rewrites_queries_for_empty_subquestions(corpus, fake_llm):
    from common.tracing import Tracer
    tr = Tracer()
    with tr.activate():
        build_system({"name": "a", "max_rounds": 2}, corpus, fake_llm).answer("Which 6G terahertz band was used?")
    rounds = {e["inputs"].get("round") for e in tr.events if e.get("step") == "retrieve"}
    assert rounds == {1, 2}


def test_unanswerable_subquestion_is_reported_as_gap(corpus):
    import json

    from conftest import hallucinating_responder

    from ara import agents
    from common.llm import ScriptedLLM

    def responder(system, prompt):
        if system == agents.SYS_PLAN:
            return json.dumps([Q_DOS, "Which 6G terahertz band was used in the testbed?"])
        return hallucinating_responder(system, prompt)
    rep = build_system({"name": "a"}, corpus, ScriptedLLM(responder)).answer(Q_DOS + " Which THz band was used?")
    assert not rep.abstained
    assert rep.gaps == ["Which 6G terahertz band was used in the testbed?"]
    assert "Evidence gaps" in rep.answer
