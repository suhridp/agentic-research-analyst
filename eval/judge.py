"""Evaluation judges: is an answer sentence supported by the passages it cites, and does an answer
cover a gold fact?

The judge is deliberately separate from the pipeline's verifier (different prompt, and ideally a
different model - the runner warns if they match), otherwise the agentic system is graded by the same
component that filtered its claims. Judge numbers are only trustworthy after measuring agreement with
hand labels: see eval/judge_agreement.py.
"""
from __future__ import annotations

from ara.agents import lexical_support
from ara.text import numbers, tokens

SYS_SUPPORT = """You are an exacting fact-checking evaluator. Decide whether the EVIDENCE fully supports the
STATEMENT: every fact, number, unit, comparison direction and qualifier in the statement must be stated
in or directly entailed by the evidence. Hedged evidence ("may", "in simulation") does not support an
unhedged statement. Return ONLY JSON: {"label": "supported"|"not_supported", "reason": "<one sentence>"}"""

SYS_FACTS = """You check whether an ANSWER states each reference FACT (paraphrase is fine; numbers must match).
Return ONLY a JSON list of true/false, one per fact, in order."""


class LexicalJudge:
    """Offline proxy. Under-credits paraphrase; use only for plumbing checks, never as a headline number."""
    name = "lexical"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def supported(self, statement: str, evidence: list[str]) -> bool:
        return lexical_support(statement, " ".join(evidence)) >= self.threshold

    def covers(self, answer: str, facts: list[str]) -> list[bool]:
        a_tok, a_num = set(tokens(answer)), numbers(answer)
        out = []
        for f in facts:
            ft = set(tokens(f))
            out.append(numbers(f) <= a_num and len(ft & a_tok) / max(len(ft), 1) >= 0.6)
        return out

    def describe(self) -> dict:
        return {"kind": "lexical", "threshold": self.threshold}


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm

    @property
    def name(self) -> str:
        return f"llm:{self.llm.name}"

    def supported(self, statement: str, evidence: list[str]) -> bool:
        ev = "\n\n".join(f"<evidence>\n{e}\n</evidence>" for e in evidence)
        out = self.llm.complete_json(SYS_SUPPORT, f"{ev}\n\nSTATEMENT: {statement}", 200, tag="judge_support",
                                     validate=lambda o: isinstance(o, dict) and "label" in o)
        return bool(out) and str(out.get("label", "")).lower() == "supported"

    def covers(self, answer: str, facts: list[str]) -> list[bool]:
        if not facts:
            return []
        listing = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
        out = self.llm.complete_json(SYS_FACTS, f"ANSWER:\n{answer}\n\nFACTS:\n{listing}", 200, tag="judge_facts",
                                     validate=lambda o: isinstance(o, list) and len(o) == len(facts))
        return [bool(x) for x in out] if out else [False] * len(facts)

    def describe(self) -> dict:
        return {"kind": "llm", **self.llm.describe()}
