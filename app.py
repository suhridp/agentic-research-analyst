"""Streamlit demo: ask a question, see the answer, every claim's verification verdict, the evidence,
what was rejected and why, and the agent trace. Optionally compare against the RAG baseline.

    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from ara.corpus import load_corpus
from ara.pipeline import build_system
from common.config import load_yaml
from common.llm import UsageMeter, build_llm
from common.tracing import Tracer
from eval.run import _prices, _systems

st.set_page_config(page_title="Agentic Research Analyst", layout="wide")


@st.cache_resource
def load(config_path: str):
    cfg = load_yaml(config_path)
    c = cfg.get("corpus", {"path": "data/corpus"})
    corpus = load_corpus(c["path"], c.get("max_words", 160), c.get("overlap", 1))
    llm = build_llm(cfg.get("llm"), prices=_prices(cfg))
    return cfg, corpus, {s["name"]: build_system(s, corpus, llm) for s in _systems(cfg)}, llm


st.sidebar.title("Settings")
config_path = st.sidebar.text_input("Experiment config", os.getenv("ARA_CONFIG", "configs/smoke.yaml"))
cfg, corpus, systems, llm = load(config_path)
names = list(systems)
ref = cfg.get("reference") if cfg.get("reference") in names else names[0]
main_sys = st.sidebar.selectbox("System", names, index=names.index(ref))
compare = st.sidebar.selectbox("Compare with", ["(none)"] + [n for n in names if n != main_sys],
                               index=1 if len(names) > 1 else 0)
st.sidebar.caption(f"{len(corpus.documents)} documents, {len(corpus.passages)} passages - "
                   f"LLM: {llm.name if llm else 'none (offline heuristics)'}")

st.title("Agentic Research Analyst")
st.caption("Claim-level citations. Every claim is checked against the one passage it cites; unsupported claims "
           "are dropped and the system abstains when nothing verifies.")
q = st.text_input("Research question", placeholder="How do spoofed measurement reports affect predictive handover?")


def show(name: str):
    tr = Tracer(name)
    with tr.activate(), UsageMeter() as m:
        rep = systems[name].answer(q)
    st.subheader(f"`{name}`")
    (st.warning if rep.abstained else st.markdown)(rep.answer)
    st.caption(f"{rep.latency_s:.2f}s | {m.usage.calls} LLM calls | ${m.usage.cost_usd:.4f} | "
               f"{len(rep.claims)} claims kept | {len(rep.rejected)} rejected")
    if rep.claims:
        with st.expander("Evidence for each cited claim", expanded=True):
            for c in rep.claims:
                p = corpus.by_id.get(c.source)
                st.markdown(f"**[{c.source}]** {c.text}  \n_verdict: {c.verdict} ({c.stage})_")
                if p:
                    st.caption(f"{corpus.doc_meta(p.doc).get('title', p.doc)} - {p.section}: {p.text}")
    if rep.rejected:
        with st.expander(f"Rejected claims ({len(rep.rejected)})"):
            for c in rep.rejected:
                st.markdown(f"- ~~{c.text}~~ [{c.source}] - {c.verdict} via {c.stage}: {c.rationale}")
    with st.expander("Agent trace"):
        st.json([e for e in tr.events if e["kind"] != "llm_call"] or [{"note": "no steps"}])


if q:
    cols = st.columns(2) if compare != "(none)" else [st.container()]
    with cols[0]:
        show(main_sys)
    if compare != "(none)":
        with cols[1]:
            show(compare)
