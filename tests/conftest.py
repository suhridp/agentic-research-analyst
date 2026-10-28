from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ara import agents  # noqa: E402
from ara.corpus import load_corpus  # noqa: E402
from common.llm import ScriptedLLM  # noqa: E402

SYN = ROOT / "data" / "synthetic"


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(SYN)


def _passages(prompt: str) -> list[tuple[str, str]]:
    return re.findall(r"^\[(\S+#\d+)\] (.+)$", prompt, flags=re.M)


def hallucinating_responder(system: str, prompt: str) -> str:
    """A fake LLM with realistic failure modes, used to test that the safeguards catch them:
    the extractor faithfully copies one sentence per passage but also invents a number for the
    first passage and flips a direction word for the second; the synthesizer adds an unsupported
    statistic; the verifier is a careful (string-matching) checker that quotes its evidence."""
    if system == agents.SYS_PLAN:
        q = prompt.strip()
        return json.dumps([q])
    if system == agents.SYS_EXTRACT:
        out = []
        for i, (pid, text) in enumerate(_passages(prompt)):
            first = re.split(r"(?<=\.)\s", text)[0]
            out.append({"claim": first, "source": pid})
            if i == 0 and re.search(r"\d", first):
                out.append({"claim": re.sub(r"\d+", "99", first, count=1), "source": pid})
            if i == 1:
                flipped = re.sub(r"\b(reduced|reduces|increased|dropped|detected|flagged|relies|are|is)\b",
                                 "improved", first, count=1)
                if flipped != first:
                    out.append({"claim": flipped, "source": pid})
        return json.dumps(out)
    if system == agents.SYS_VERIFY:
        passage = prompt.split("Passage:\n", 1)[1].split("\n\nClaim:\n")[0]
        claim = prompt.split("\n\nClaim:\n", 1)[1].strip()
        ok = claim.rstrip(".") in passage
        return json.dumps({"verdict": "supported" if ok else "not_supported", "quote": claim if ok else ""})
    if system == agents.SYS_SYNTH:
        lines = re.findall(r"^- (.+) \[(\S+#\d+)\]$", prompt, flags=re.M)
        body = " ".join(f"{c.rstrip('.')} [{s}]." for c, s in lines)
        return body + f" Overall this improves detection by 50 percent [{lines[0][1]}]." if lines else body
    if system == agents.SYS_COMPLETE:
        return "[]"
    if system == agents.SYS_REWRITE:
        return prompt.split("Sub-question:")[-1].strip()
    if "Answer the question using only the passages" in system:  # RAG baseline
        ps = _passages(prompt)
        if not ps:
            return agents.ABSTAIN
        pid, text = ps[0]
        first = re.split(r"(?<=\.)\s", text)[0]
        return f"{first} [{pid}]. It is 99 percent accurate in every deployment [{pid}]."
    return "[]"


@pytest.fixture
def fake_llm():
    return ScriptedLLM(hallucinating_responder)
