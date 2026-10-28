import csv
import json
from pathlib import Path

import pytest
from conftest import ROOT

from eval import failures, judge_agreement, retrieval_eval, run, validate, verifier_stress
from eval.metrics import answer_units
from eval.questions import Question, load_questions, resolve_gold


def test_question_schema_rejects_inconsistent_items(tmp_path):
    with pytest.raises(ValueError):
        Question(id="x", question="What is this?", type="unanswerable", evidence=[{"doc": "d", "quote": "something long"}])
    with pytest.raises(ValueError):
        Question(id="x", question="What is this?", type="factual")
    p = tmp_path / "q.jsonl"
    p.write_text('{"id": "a", "question": "Q long enough?", "type": "unanswerable"}\n' * 2)
    with pytest.raises(ValueError, match="duplicate id"):
        load_questions(p)


def test_gold_resolves_from_quotes(corpus):
    qs = load_questions(ROOT / "data/synthetic_questions.jsonl")
    for q in qs:
        groups, missing = resolve_gold(q, corpus)
        assert not missing and len(groups) == len(q.evidence)


def test_answer_units_ignores_gaps_and_abstention():
    units, uncited = answer_units("A fact [d#1]. Another. [d#2]\nStray sentence.\n\nEvidence gaps (x):\n- something")
    assert units == [("A fact.", ["d#1"]), ("Another.", ["d#2"])] and uncited == 1
    assert answer_units("Insufficient verified evidence in the collection to answer this question.") == ([], 0)
    units, _ = answer_units("- 18 percent fewer failures [d#1].\n2. 7 ms added [d#2].")
    assert [u[0] for u in units] == ["18 percent fewer failures.", "7 ms added."]


def test_smoke_run_end_to_end(tmp_path):
    out = run.main([str(ROOT / "configs/smoke.yaml"), "--out", str(tmp_path), "--workers", "2"])
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["systems"]) == {"bm25_rag", "hybrid_rerank_rag", "agentic_full", "no_planner",
                                       "no_verifier", "no_critic"}
    assert all(s["errors"] == 0 for s in summary["systems"].values())
    assert "agentic_full" not in summary["comparisons"]["completeness"]
    rows = (out / "per_question.jsonl").read_text().splitlines()
    assert len(rows) == 6 * 11 and len(list((out / "traces").iterdir())) == 6 * 11
    meta = json.loads((out / "meta.json").read_text())
    assert meta["corpus"]["fingerprint"] and meta["environment"]["python"]
    assert "lexical proxy" in (out / "summary.md").read_text()
    # failure analysis and judge-agreement tooling run on the output
    failures.main([str(out)])
    assert (out / "failures.md").exists()


def test_judge_agreement_roundtrip(tmp_path, monkeypatch):
    out = run.main([str(ROOT / "configs/smoke.yaml"), "--out", str(tmp_path / "r"), "--systems", "agentic_full"])
    monkeypatch.setattr(judge_agreement, "LABEL_DIR", tmp_path / "labels")
    judge_agreement.export(out, 6, 0)
    path = tmp_path / "labels/judge_sample.csv"
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["human_label"] = "1"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    res = judge_agreement.score()
    assert res["llm_judge"]["n"] == len(rows)


def test_test_split_requires_confirmation(tmp_path):
    with pytest.raises(SystemExit, match="confirm-test"):
        run.main([str(ROOT / "configs/smoke.yaml"), "--split", "test", "--out", str(tmp_path)])
    run.main([str(ROOT / "configs/smoke.yaml"), "--split", "test", "--confirm-test", "--out", str(tmp_path),
              "--systems", "agentic_full"])
    assert (tmp_path / "TEST_RUNS.log").read_text().count("\n") == 1


def test_retrieval_eval_and_stress_and_validate(tmp_path):
    out = retrieval_eval.main([str(ROOT / "configs/retrieval_smoke.yaml"), "--out", str(tmp_path)])
    s = json.loads((out / "summary.json").read_text())
    assert s["retrievers"]["bm25"]["recall@5"]["n"] > 0
    out = verifier_stress.main(["--corpus", str(ROOT / "data/synthetic"), "--n", "20", "--out", str(tmp_path)])
    res = json.loads((out / "summary.json").read_text())["results"]
    v1 = res["lexical_v1 (numbers only)"]
    assert v1["original"]["rate"] == 1.0 and v1["number_change"]["rate"] == 0.0
    assert validate.main(["--questions", str(ROOT / "data/synthetic_questions.jsonl"),
                          "--corpus", str(ROOT / "data/synthetic")]) == 0


def test_validate_flags_unresolved_quotes(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"id": "a", "question": "Some question?", "type": "factual",
                             "evidence": [{"doc": "syn_dos", "quote": "this sentence is not in the paper"}]}) + "\n")
    assert validate.main(["--questions", str(p), "--corpus", str(Path(ROOT / "data/synthetic"))]) == 1


def test_runner_llm_mode_with_fake_model(tmp_path, monkeypatch):
    """Runner LLM path: parallel workers, per-question cost metering, traces with LLM calls, and the
    paired comparison showing the planted hallucinations are caught only by the full system."""
    from conftest import hallucinating_responder

    from common.llm import Price, ScriptedLLM

    fake = ScriptedLLM(hallucinating_responder, model="fake", price=Price(1.0, 2.0))
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())
    cfg["llm"] = {"provider": "fake", "model": "fake"}
    cfg_path = tmp_path / "fake.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(run, "build_llm", lambda cfg, **kw: fake if cfg else None)
    out = run.main([str(cfg_path), "--out", str(tmp_path), "--workers", "4",
                    "--systems", "bm25_rag,agentic_full,no_verifier,no_critic"])
    rows = [json.loads(line) for line in (out / "per_question.jsonl").read_text().splitlines()]
    assert all("error" not in r for r in rows)
    full = [r for r in rows if r["system"] == "agentic_full"]
    assert all(r["llm_calls"] > 0 and r["cost_usd"] > 0 for r in full)
    assert sum(r["llm_calls"] for r in rows) == fake.usage.calls  # nothing double-counted across threads
    s = json.loads((out / "summary.json").read_text())["systems"]
    assert s["agentic_full"]["unsupported_rate"]["mean"] == 0.0
    assert s["bm25_rag"]["unsupported_rate"]["mean"] > 0 and s["no_critic"]["unsupported_rate"]["mean"] > 0
    trace = (out / "traces" / f"{full[0]['qid']}__agentic_full.jsonl").read_text()
    assert '"kind": "llm_call"' in trace and '"tag": "verify"' in trace
