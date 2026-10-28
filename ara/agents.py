"""The agents: planner, extractor, verifier, synthesizer, critic.

Each agent uses an LLM when one is given and a deterministic heuristic otherwise, so the whole
pipeline runs offline for tests and CI. The heuristics are plumbing, not a research result: the
meaningful comparisons run in LLM mode.

Two invariants make verification trustworthy:
  1. Every claim carries exactly one passage id, and the verifier sees ONLY that claim and that
     passage, so surrounding context can't talk it into accepting a claim.
  2. The critic is deterministic: a report sentence survives only if every citation points at a
     verified claim and every number in it appears in those claims.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .corpus import Passage
from .text import has_negation, normalize, numbers, sentences, tokens

PROMPT_VERSION = "2026-10-v2"  # bump when any prompt changes; recorded with results

SYS_PLAN = """You plan research over a fixed document collection about wireless networks and security.
Break the user's question into 1-4 focused sub-questions that can each be answered from a single passage.
Keep the user's terms (numbers, named methods, metrics). Don't add sub-questions the user didn't ask.
Return ONLY a JSON list of strings."""

SYS_EXTRACT = """You extract evidence. Using ONLY the passages given, list atomic claims (one fact each) that help
answer the question. Each claim must be supported by a single passage; copy numbers, units and qualifiers
exactly ("may", "in simulation", "on a synthetic dataset"). Never combine passages or add outside knowledge.
If no passage helps, return [].
Return ONLY JSON: [{"claim": "...", "source": "<passage id exactly as given>"}]"""

SYS_VERIFY = """You check one claim against one passage. Answer "supported" only if the passage alone entails
every part of the claim, including each number, unit, direction of change and qualifier.
"partially_supported" if some part is missing or stronger than the passage; "not_supported" otherwise.
Return ONLY JSON: {"verdict": "supported"|"partially_supported"|"not_supported",
"quote": "<shortest exact span from the passage that supports the claim, or empty>"}"""

SYS_SYNTH = """Write a concise research answer from the verified claims only.
Rules: every sentence ends with the id(s) of the claims it uses, like [doc#3] or [doc#3][doc#7].
Do not add facts, numbers or interpretations that aren't in the claims. If sources disagree, say so and
cite both. If the claims answer only part of the question, say which part is not covered (without citation,
starting the sentence with "Not covered:")."""

SYS_REWRITE = """Rewrite the research sub-question as a keyword search query (8-15 words) with useful synonyms
and technical terms. Return ONLY the query text."""

SYS_COMPLETE = """Compare a research question with the answer written for it. List the parts of the question
the answer does not address. Don't judge correctness. Return ONLY a JSON list of short strings ([] if complete)."""

_CITE_BLOCK = re.compile(r"\[([^\[\]]+)\]")
_CITE_ID = re.compile(r"^\S+#\d+$")


def cite_ids(text: str) -> list[str]:
    """Citation ids in order: handles [a#1], [a#1][b#2] and [a#1, b#2]."""
    out = []
    for block in _CITE_BLOCK.findall(text):
        for part in re.split(r"[,;]\s*", block):
            part = part.strip()
            if _CITE_ID.match(part):
                out.append(part)
    return out


def strip_cites(text: str) -> str:
    return re.sub(r"\s*\[[^\[\]]*#\d+[^\[\]]*\]", "", text).strip()


@dataclass
class Claim:
    text: str
    source: str
    subq: str = ""
    verdict: str | None = None      # supported | partially_supported | not_supported | unchecked
    score: float = 0.0
    stage: str = ""                 # lexical | llm | none
    rationale: str = ""

    @property
    def verified(self) -> bool:
        return self.verdict in ("supported", "unchecked")

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------- planner
def plan(question: str, llm=None) -> list[str]:
    if llm:
        out = llm.complete_json(SYS_PLAN, question, 300, tag="plan",
                                validate=lambda o: isinstance(o, list) and all(isinstance(x, str) for x in o))
        if out:
            subs = [s.strip() for s in out if s.strip()][:4]
            if subs:
                return subs
    # heuristic: split conjunctions; always keep the full question as the first sub-question
    parts = [p.strip() for p in re.split(r"\band\b|;|,|\bvs\.?\b|\bversus\b", question.rstrip("?"))
             if len(tokens(p)) >= 2]
    return [question] + parts if len(parts) > 1 else [question]


def rewrite_query(subq: str, question: str, llm=None) -> str:
    if llm:
        q = llm.complete(SYS_REWRITE, f"Overall question: {question}\nSub-question: {subq}", 80, tag="rewrite").strip()
        if 2 <= len(q.split()) <= 40:
            return q
    extra = [t for t in tokens(question) if t not in tokens(subq)]
    return f"{subq} {' '.join(extra)}".strip()


# ------------------------------------------------------------------------------------- extractor
def extract(subq: str, passages: list[Passage], llm=None, max_claims: int = 8) -> list[Claim]:
    if not passages:
        return []
    if llm:
        ctx = "\n\n".join(f"[{p.id}] {p.text}" for p in passages)
        out = llm.complete_json(SYS_EXTRACT, f"Question: {subq}\n\nPassages:\n{ctx}", 1200, tag="extract",
                                validate=lambda o: isinstance(o, list))
        valid = {p.id for p in passages}
        claims = []
        for c in out or []:
            if not (isinstance(c, dict) and isinstance(c.get("claim"), str)):
                continue
            src = str(c.get("source", "")).strip("[]")
            if src in valid:
                claims.append(Claim(c["claim"].strip(), src, subq))
        return claims[:max_claims]
    q, claims = set(tokens(subq)), []
    for p in passages:
        best = max(sentences(p.text), key=lambda s: len(q & set(tokens(s))), default="")
        if len(q & set(tokens(best))) >= 2:
            claims.append(Claim(best, p.id, subq))
    return claims[:max_claims]


# -------------------------------------------------------------------------------------- verifier
@dataclass
class VerifierConfig:
    mode: str = "auto"          # auto (llm if available else lexical) | lexical | llm | none
    threshold: float = 0.8      # lexical coverage needed to accept
    number_filter: bool = True  # cheap first pass: reject if the claim has a number the passage lacks
    negation_check: bool = True # lexical mode only: reject if negation differs from the best-matching sentence
    quote_check: bool = True    # llm mode: the verifier's supporting quote must really be in the passage


def lexical_support(claim: str, passage: str) -> float:
    """Content-word coverage of the claim by the passage, zeroed if any claim number is missing."""
    if not numbers(claim) <= numbers(passage):
        return 0.0
    ct, pt = set(tokens(claim)), set(tokens(passage))
    return len(ct & pt) / max(len(ct), 1)


def _best_sentence(claim: str, passage: str) -> str:
    c = set(tokens(claim))
    return max(sentences(passage) or [passage], key=lambda s: len(c & set(tokens(s))))


def _quote_in(quote: str, passage: str) -> bool:
    q, p = normalize(quote).lower().strip(" .\"'"), normalize(passage).lower()
    if not q:
        return False
    if q in p:
        return True
    qt = tokens(q)  # tolerate minor punctuation/whitespace differences
    return bool(qt) and len(set(qt) & set(tokens(p))) / len(set(qt)) >= 0.9


def verify(claim: Claim, passage: Passage, llm=None, cfg: VerifierConfig | None = None) -> Claim:
    """Fill in claim.verdict/score/stage/rationale. The verifier sees only this claim and this passage."""
    cfg = cfg or VerifierConfig()
    mode = cfg.mode if cfg.mode != "auto" else ("llm" if llm else "lexical")
    if mode == "none":
        claim.verdict, claim.stage = "unchecked", "none"
        return claim
    missing = numbers(claim.text) - numbers(passage.text)
    if cfg.number_filter and missing:
        claim.verdict, claim.score, claim.stage = "not_supported", 0.0, "lexical"
        claim.rationale = f"number(s) not in passage: {sorted(missing)}"
        return claim
    if mode == "lexical" or llm is None:
        s = lexical_support(claim.text, passage.text)
        claim.score, claim.stage = s, "lexical"
        if cfg.negation_check and has_negation(claim.text) != has_negation(_best_sentence(claim.text, passage.text)):
            claim.verdict, claim.rationale = "not_supported", "negation mismatch"
        elif s >= cfg.threshold:
            claim.verdict, claim.rationale = "supported", f"coverage {s:.2f}"
        else:
            claim.verdict, claim.rationale = "not_supported", f"coverage {s:.2f} < {cfg.threshold}"
        return claim
    out = llm.complete_json(SYS_VERIFY, f"Passage:\n{passage.text}\n\nClaim:\n{claim.text}", 300, tag="verify",
                            validate=lambda o: isinstance(o, dict) and "verdict" in o)
    claim.stage = "llm"
    if not out:
        claim.verdict, claim.rationale = "not_supported", "verifier output unparseable (fail closed)"
        return claim
    verdict = str(out.get("verdict", "")).strip().lower()
    quote = str(out.get("quote", "") or "")
    if verdict == "supported" and cfg.quote_check and not _quote_in(quote, passage.text):
        claim.verdict, claim.rationale = "not_supported", "verifier quote not found in passage (fail closed)"
    else:
        claim.verdict = verdict if verdict in ("supported", "partially_supported", "not_supported") else "not_supported"
        claim.rationale = quote[:300]
    claim.score = 1.0 if claim.verdict == "supported" else 0.0
    return claim


# ----------------------------------------------------------------------------------- synthesizer
def synthesize(question: str, claims: list[Claim], llm=None) -> str:
    if not claims:
        return ABSTAIN
    if llm:
        ctx = "\n".join(f"- {c.text} [{c.source}]" for c in claims)
        prompt = f"Question: {question}\n\nVerified claims:\n{ctx}"
        return llm.complete(SYS_SYNTH, prompt, 1200, tag="synthesize").strip()
    return claim_list(claims)


def claim_list(claims: list[Claim]) -> str:
    seen, lines = set(), []
    for c in claims:
        if (c.text, c.source) not in seen:
            seen.add((c.text, c.source))
            lines.append(f"- {c.text} [{c.source}]")
    return "\n".join(lines)


ABSTAIN = "Insufficient verified evidence in the collection to answer this question."


# ---------------------------------------------------------------------------------------- critic
def critique(report: str, claims: list[Claim]) -> tuple[str, list[str]]:
    """Deterministic sentence-level check. Returns (cleaned report, issues).
    A sentence is kept only if it cites >=1 verified claim, cites nothing unverified, and every number in
    it appears in the cited claims. "Not covered:" sentences (stated gaps) are kept without citations."""
    ok: dict[str, set[str]] = {}
    for c in claims:
        if c.verified:
            ok.setdefault(c.source, set()).update(numbers(c.text))
    issues, kept_lines = [], []
    for line in report.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            kept_lines.append(line)
            continue
        m = re.match(r"^(\s*(?:[-*]|\d+\.)\s+)", line)
        prefix = m.group(1) if m else ""
        body = line[len(prefix):]
        sents: list[str] = []
        for s in sentences(body):  # glue a citation-only fragment back onto its sentence
            if sents and not strip_cites(s):
                sents[-1] += " " + s
            else:
                sents.append(s)
        keep = []
        for s in sents:
            ids = cite_ids(s)
            if not ids:
                if s.lower().startswith(("not covered", "the collection does not", "no evidence")) and not numbers(s):
                    keep.append(s)
                    continue
                issues.append(f"uncited sentence dropped: {s[:80]}")
                continue
            bad = [i for i in ids if i not in ok]
            if bad:
                issues.append(f"citation not backed by a verified claim: {', '.join(bad)}")
                continue
            extra = numbers(strip_cites(s)) - set().union(*(ok[i] for i in ids))
            if extra:
                issues.append(f"number(s) {sorted(extra)} not in cited claims: {s[:80]}")
                continue
            keep.append(s)
        if keep:
            kept_lines.append(prefix + " ".join(keep))
    cleaned = "\n".join(kept_lines).strip()
    if claims and not cite_ids(cleaned):
        issues.append("no valid cited sentence survived; falling back to verified-claim list")
        cleaned = claim_list([c for c in claims if c.verified])
    return cleaned, issues


def completeness_gaps(question: str, report: str, llm=None) -> list[str]:
    if not llm:
        return []
    out = llm.complete_json(SYS_COMPLETE, f"Question: {question}\n\nAnswer:\n{report}", 300, tag="completeness",
                            validate=lambda o: isinstance(o, list))
    return [str(x) for x in (out or []) if str(x).strip()][:5]
