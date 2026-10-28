"""The two system families compared in the evaluation, both built from config so ablations are one line.

  rag:      retrieve top-k for the question -> answer once with citations (no verification)
  agentic:  plan -> retrieve per sub-question -> extract claims -> verify each claim against its passage
            -> (retry with rewritten queries for sub-questions that found nothing)
            -> synthesize from verified claims -> deterministic critic -> evidence gaps / abstention

Ablations (see configs/): planner=false, verifier=none, critic=false, retrieval kind, k.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from common import tracing

from . import agents
from .agents import ABSTAIN, Claim, VerifierConfig
from .corpus import Corpus
from .retrieval import Retriever


@dataclass
class Report:
    question: str
    system: str
    answer: str
    claims: list[Claim]                 # claims the answer rests on (verified ones for agentic)
    rejected: list[Claim] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)  # passage ids, in first-seen order
    subquestions: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    abstained: bool = False
    latency_s: float = 0.0
    llm_calls_saved_by_filter: int = 0

    @property
    def cited(self) -> list[str]:
        return list(dict.fromkeys(agents.cite_ids(self.answer)))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cited"] = self.cited
        return d


@dataclass
class SystemConfig:
    name: str
    kind: str = "agentic"                       # agentic | rag
    retriever: dict[str, Any] = field(default_factory=lambda: {"kind": "bm25"})
    k: int = 5
    planner: bool = True
    verifier: str = "auto"                      # auto | lexical | llm | none
    critic: bool = True
    completeness_check: bool = True             # LLM-only; part of the critic
    max_rounds: int = 2
    rewrite_queries: bool = True
    max_claims_per_subq: int = 8
    verifier_opts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> SystemConfig:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"system {d.get('name')}: unknown keys {sorted(unknown)}")
        return cls(**d)


class System:
    def __init__(self, cfg: SystemConfig, corpus: Corpus, retriever: Retriever, llm=None):
        self.cfg, self.corpus, self.retriever, self.llm = cfg, corpus, retriever, llm
        self.vcfg = VerifierConfig(mode=cfg.verifier, **cfg.verifier_opts)

    def describe(self) -> dict:
        return {**asdict(self.cfg), "retriever_backend": self.retriever.describe(),
                "llm": self.llm.describe() if self.llm else None, "prompt_version": agents.PROMPT_VERSION}

    def answer(self, question: str) -> Report:
        t0 = time.perf_counter()
        rep = self._rag(question) if self.cfg.kind == "rag" else self._agentic(question)
        rep.latency_s = time.perf_counter() - t0
        tracing.event("report", system=self.cfg.name, abstained=rep.abstained, n_claims=len(rep.claims),
                      n_rejected=len(rep.rejected), issues=rep.issues, gaps=rep.gaps)
        return rep

    # ------------------------------------------------------------------------------- baseline
    def _rag(self, q: str) -> Report:
        with tracing.span("retrieve", query=q, k=self.cfg.k) as s:
            hits = self.retriever.search(q, self.cfg.k)
            s["ids"] = [h.passage.id for h in hits]
        passages = [h.passage for h in hits]
        ids = [p.id for p in passages]
        if self.llm:
            ctx = "\n\n".join(f"[{p.id}] {p.text}" for p in passages)
            with tracing.span("answer"):
                ans = self.llm.complete(
                    "Answer the question using only the passages. End every sentence with the id of the passage "
                    "it comes from, like [doc#3]. If the passages don't answer it, reply exactly: "
                    f"{ABSTAIN}", f"Question: {q}\n\nPassages:\n{ctx}", 1000, tag="rag_answer").strip()
            claims = [Claim(agents.strip_cites(sent), cid, q, verdict="unchecked", stage="none")
                      for line in ans.splitlines() for sent in agents.sentences(line)
                      for cid in agents.cite_ids(sent)]
        else:
            claims = agents.extract(q, passages)
            for c in claims:
                c.verdict, c.stage = "unchecked", "none"
            ans = agents.synthesize(q, claims)
        abstained = not claims or ABSTAIN.lower()[:30] in ans.lower()
        return Report(q, self.cfg.name, ans, claims, retrieved=ids, subquestions=[q], abstained=abstained)

    # -------------------------------------------------------------------------------- agentic
    def _agentic(self, q: str) -> Report:
        cfg, llm = self.cfg, self.llm
        with tracing.span("plan", enabled=cfg.planner) as s:
            subs = agents.plan(q, llm) if cfg.planner else [q]
            s["subquestions"] = subs
        retrieved: dict[str, Any] = {}
        verified: list[Claim] = []
        rejected: list[Claim] = []
        seen: set[tuple[str, str]] = set()
        found = {s: 0 for s in subs}
        saved = 0  # claims the cheap number filter rejected before an LLM verifier call
        eff_mode = self.vcfg.mode if self.vcfg.mode != "auto" else ("llm" if llm else "lexical")
        for rnd in range(cfg.max_rounds):
            todo = subs if rnd == 0 else [s for s in subs if found[s] == 0]
            if not todo:
                break
            for sq in todo:
                query = sq if rnd == 0 or not cfg.rewrite_queries else agents.rewrite_query(sq, q, llm)
                k = cfg.k * (rnd + 1)
                with tracing.span("retrieve", round=rnd + 1, query=query, k=k) as s:
                    hits = self.retriever.search(query, k)
                    s["ids"] = [h.passage.id for h in hits]
                passages = [h.passage for h in hits]
                for p in passages:
                    retrieved.setdefault(p.id, p)
                fresh = [p for p in passages if rnd == 0 or p.id not in {c.source for c in verified + rejected}]
                with tracing.span("extract", subq=sq, n_passages=len(fresh)) as s:
                    claims = agents.extract(sq, fresh, llm, cfg.max_claims_per_subq)
                    s["n_claims"] = len(claims)
                for c in claims:
                    if (c.text, c.source) in seen:
                        continue
                    seen.add((c.text, c.source))
                    agents.verify(c, retrieved[c.source], llm, self.vcfg)
                    saved += eff_mode == "llm" and c.stage == "lexical"
                    (verified if c.verified else rejected).append(c)
                    found[sq] += c.verified
                tracing.event("verify", subq=sq, round=rnd + 1,
                              verdicts=[{"claim": c.text[:120], "source": c.source, "verdict": c.verdict,
                                         "stage": c.stage, "why": c.rationale[:120]} for c in claims])
        # Sub-questions from the heuristic planner are text fragments, so only LLM-planned ones are
        # reported as evidence gaps.
        gaps = [s for s in subs if found[s] == 0] if len(subs) > 1 and llm and cfg.planner else []
        if not verified:
            return Report(q, cfg.name, ABSTAIN, [], rejected, list(retrieved), subs, gaps or [q],
                          abstained=True, llm_calls_saved_by_filter=saved)
        with tracing.span("synthesize", n_claims=len(verified)):
            answer = agents.synthesize(q, verified, llm)
        issues: list[str] = []
        if cfg.critic:
            with tracing.span("critic") as s:
                answer, issues = agents.critique(answer, verified)
                if cfg.completeness_check and llm:
                    gaps += [g for g in agents.completeness_gaps(q, answer, llm) if g not in gaps]
                s["issues"] = issues
        if gaps:
            answer += ("\n\nEvidence gaps (nothing verifiable found in the collection):\n"
                       + "\n".join(f"- {g}" for g in gaps))
        used = set(agents.cite_ids(answer))
        kept = [c for c in verified if c.source in used] or verified
        return Report(q, cfg.name, answer, kept, rejected, list(retrieved), subs, gaps, issues,
                      llm_calls_saved_by_filter=saved)


def build_system(cfg: dict | SystemConfig, corpus: Corpus, llm=None, cache_dir: str = ".index_cache") -> System:
    from .retrieval import build_retriever
    sc = cfg if isinstance(cfg, SystemConfig) else SystemConfig.from_dict(cfg)
    return System(sc, corpus, build_retriever(sc.retriever, corpus, llm=llm, cache_dir=cache_dir), llm)
