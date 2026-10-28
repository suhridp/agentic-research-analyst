"""HTTP API for the research analyst.

    uvicorn api.main:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "How do spoofed measurement reports affect handover?"}' | jq

Configured by environment variables so the same image runs offline or with an LLM:
    ARA_CONFIG   experiment config whose corpus/llm/systems to serve (default configs/smoke.yaml)
    ARA_SYSTEM   default system name (default: the config's `reference`)
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ara import __version__
from ara.corpus import load_corpus
from ara.pipeline import build_system
from common.config import load_yaml
from common.llm import BudgetExceeded, LLMError, UsageMeter, build_llm
from eval.run import _prices, _systems

app = FastAPI(title="Agentic Research Analyst", version=__version__)


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    system: str | None = None


class ClaimOut(BaseModel):
    text: str
    source: str
    verdict: str | None
    stage: str
    rationale: str


class AskResponse(BaseModel):
    question: str
    system: str
    answer: str
    abstained: bool
    claims: list[ClaimOut]
    rejected: list[ClaimOut]
    gaps: list[str]
    sources: dict[str, dict]
    latency_s: float
    llm_calls: int
    cost_usd: float


@lru_cache(maxsize=1)
def _state():
    cfg = load_yaml(os.getenv("ARA_CONFIG", "configs/smoke.yaml"))
    c = cfg.get("corpus", {"path": "data/corpus"})
    corpus = load_corpus(c["path"], c.get("max_words", 160), c.get("overlap", 1))
    llm = build_llm(cfg.get("llm"), prices=_prices(cfg))
    systems = {s["name"]: build_system(s, corpus, llm) for s in _systems(cfg)}
    default = os.getenv("ARA_SYSTEM") or cfg.get("reference") or next(iter(systems))
    return corpus, systems, default


@app.get("/health")
def health():
    corpus, systems, default = _state()
    return {"status": "ok", "version": __version__, "documents": len(corpus.documents),
            "passages": len(corpus.passages), "systems": list(systems), "default_system": default}


@app.get("/systems")
def systems():
    _, systems, _ = _state()
    return {name: s.describe() for name, s in systems.items()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    corpus, systems, default = _state()
    name = req.system or default
    if name not in systems:
        raise HTTPException(404, f"unknown system '{name}'; options: {list(systems)}")
    t0 = time.perf_counter()
    try:
        with UsageMeter() as m:
            rep = systems[name].answer(req.question)
    except BudgetExceeded as e:
        raise HTTPException(429, str(e)) from e
    except LLMError as e:
        raise HTTPException(502, f"LLM provider error: {e}") from e

    def co(c):
        return ClaimOut(text=c.text, source=c.source, verdict=c.verdict, stage=c.stage, rationale=c.rationale)
    sources = {pid: {"doc": corpus.by_id[pid].doc, "section": corpus.by_id[pid].section,
                     "text": corpus.by_id[pid].text, **corpus.doc_meta(corpus.by_id[pid].doc)}
               for pid in rep.cited if pid in corpus.by_id}
    return AskResponse(question=req.question, system=name, answer=rep.answer, abstained=rep.abstained,
                       claims=[co(c) for c in rep.claims], rejected=[co(c) for c in rep.rejected], gaps=rep.gaps,
                       sources=sources, latency_s=round(time.perf_counter() - t0, 3), llm_calls=m.usage.calls,
                       cost_usd=round(m.usage.cost_usd, 6))
