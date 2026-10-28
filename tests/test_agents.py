import json

from ara import agents
from ara.agents import Claim, VerifierConfig, cite_ids, critique, lexical_support, verify
from ara.corpus import Passage
from common.llm import ScriptedLLM

P = Passage("dos", 1, "A gradient boosted classifier detected DoS floods with 97 percent accuracy on a synthetic dataset.")


def test_cite_id_forms():
    assert cite_ids("x [a#1][b#2] y [c#3, d#4]. [not a cite] [e#x]") == ["a#1", "b#2", "c#3", "d#4"]


def test_fabricated_number_is_unsupported():
    assert lexical_support(P.text, P.text) == 1.0
    assert lexical_support(P.text.replace("97", "99"), P.text) == 0.0


def test_number_filter_rejects_before_llm_call():
    calls = []
    llm = ScriptedLLM(lambda s, p: calls.append(1) or '{"verdict": "supported", "quote": "x"}')
    c = verify(Claim(P.text.replace("97", "99"), P.id), P, llm)
    assert c.verdict == "not_supported" and c.stage == "lexical" and not calls


def test_lexical_negation_check():
    c = verify(Claim("The classifier did not detect DoS floods with 97 percent accuracy.", P.id), P, None)
    assert c.verdict == "not_supported" and "negation" in c.rationale
    off = verify(Claim("The classifier did not detect DoS floods with 97 percent accuracy.", P.id), P, None,
                 VerifierConfig(mode="lexical", negation_check=False))
    assert off.verdict == "supported"  # the documented weakness the stress test measures


def test_llm_verifier_requires_real_quote():
    good = ScriptedLLM(lambda s, p: json.dumps({"verdict": "supported", "quote": "detected DoS floods with 97 percent accuracy"}))
    assert verify(Claim("DoS floods were detected with 97 percent accuracy.", P.id), P, good).verdict == "supported"
    made_up = ScriptedLLM(lambda s, p: json.dumps({"verdict": "supported", "quote": "accuracy was verified in a live network"}))
    c = verify(Claim("DoS floods were detected with 97 percent accuracy.", P.id), P, made_up)
    assert c.verdict == "not_supported" and "quote not found" in c.rationale


def test_llm_verifier_fails_closed_on_garbage():
    c = verify(Claim("DoS floods were detected with 97 percent accuracy.", P.id), P, ScriptedLLM(lambda s, p: "sure!"))
    assert c.verdict == "not_supported" and "fail closed" in c.rationale


def test_verifier_none_marks_unchecked():
    c = verify(Claim("anything 12", P.id), P, None, VerifierConfig(mode="none"))
    assert c.verdict == "unchecked" and c.verified


def test_critic_sentence_level():
    claims = [Claim("Accuracy was 97 percent.", "a#0", verdict="supported"),
              Claim("Something unverified.", "b#1", verdict="not_supported")]
    report = ("Accuracy was 97 percent [a#0]. Accuracy improved by 50 percent [a#0]. "
              "This is uncited. Bad cite [b#1]. Not covered: deployment cost.")
    cleaned, issues = critique(report, claims)
    assert cleaned == "Accuracy was 97 percent [a#0]. Not covered: deployment cost."
    assert any("50" in i for i in issues) and any("uncited" in i for i in issues) and any("b#1" in i for i in issues)


def test_critic_handles_citation_after_period_and_bullets():
    claims = [Claim("x 3", "a#0", verdict="supported")]
    cleaned, issues = critique("- It was 3 units. [a#0]\n# Heading", claims)
    assert cleaned.startswith("- It was 3 units. [a#0]") and not issues


def test_critic_falls_back_to_claim_list():
    claims = [Claim("Fact one.", "a#0", verdict="supported")]
    cleaned, issues = critique("No citations at all.", claims)
    assert cleaned == "- Fact one. [a#0]" and issues


def test_planner_heuristic_and_llm_validation():
    assert agents.plan("What is X and how does Y work?")[0] == "What is X and how does Y work?"
    bad = ScriptedLLM(lambda s, p: '{"not": "a list"}')
    assert agents.plan("Single question here?", bad) == ["Single question here?"]  # falls back after repair


def test_extractor_drops_claims_citing_unknown_passages():
    llm = ScriptedLLM(lambda s, p: json.dumps([{"claim": "ok", "source": "dos#1"}, {"claim": "x", "source": "zzz#9"}]))
    out = agents.extract("q", [P], llm)
    assert [c.source for c in out] == ["dos#1"]
