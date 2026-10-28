"""Per-question metrics and their aggregation with uncertainty.

All systems are graded on what they actually OUTPUT: each answer sentence that carries citations is
checked against the passages it cites. That treats the RAG baseline (whose sentences are its claims)
and the agentic system (whose synthesizer may paraphrase verified claims) identically.

Definitions (per question):
  cited_sentences     answer sentences with >=1 citation (the "Evidence gaps" block is excluded)
  unsupported         cited sentences the judge says their cited passages do not support
  uncited             answer sentences with no citation (excluding explicit "Not covered:" statements)
  gold_retrieved      fraction of gold evidence items with >=1 matching passage among retrieved
  gold_cited          fraction of gold evidence items with >=1 matching passage cited in the answer
  completeness        fraction of gold facts the judge finds stated in the answer
  abstained           system declined to answer
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from ara.agents import ABSTAIN, cite_ids, strip_cites
from ara.text import sentences
from common.stats import bootstrap_ci, holm, paired_bootstrap, percentile, ratio_bootstrap_ci, wilson

GAP_MARKER = "Evidence gaps"


def answer_units(answer: str) -> tuple[list[tuple[str, list[str]]], int]:
    """-> ([(sentence without citations, cited ids)], number of uncited sentences)."""
    body = answer.split(GAP_MARKER)[0]
    if body.strip().startswith(ABSTAIN[:30]):
        return [], 0
    units, uncited = [], 0
    for line in body.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+", "", line).strip()  # bullet / list marker only
        if not line or line.startswith("#"):
            continue
        sents: list[str] = []
        for s in sentences(line):
            if sents and not strip_cites(s):
                sents[-1] += " " + s
            else:
                sents.append(s)
        for s in sents:
            ids = cite_ids(s)
            if ids:
                units.append((strip_cites(s), list(dict.fromkeys(ids))))
            elif not s.lower().startswith(("not covered", "the collection does not", "no evidence")):
                uncited += 1
    return units, uncited


def score_question(q, rep, gold_groups: list[set[str]], corpus, judge) -> dict:
    units, uncited = answer_units(rep.answer)
    unsupported, details = 0, []
    for text, ids in units:
        ev = [corpus.by_id[i].text for i in ids if i in corpus.by_id]
        ok = bool(ev) and len(ev) == len(ids) and judge.supported(text, ev)  # unknown id => unsupported
        unsupported += not ok
        details.append({"sentence": text, "cited": ids, "supported": ok})
    retrieved, cited = set(rep.retrieved), set(rep.cited)
    m = {
        "qid": q.id, "type": q.type, "answerable": q.answerable, "abstained": rep.abstained,
        "cited_sentences": len(units), "unsupported": unsupported, "uncited": uncited,
        "unsupported_rate": unsupported / len(units) if units else None,
        "gold_retrieved": None, "gold_cited": None, "completeness": None,
        "abstention_correct": rep.abstained == (not q.answerable),
        "n_claims": len(rep.claims), "n_rejected": len(rep.rejected), "critic_issues": len(rep.issues),
        "latency_s": rep.latency_s, "sentence_judgements": details,
    }
    if q.answerable and gold_groups:
        m["gold_retrieved"] = sum(bool(g & retrieved) for g in gold_groups) / len(gold_groups)
        m["gold_cited"] = sum(bool(g & cited) for g in gold_groups) / len(gold_groups)
    if q.answerable and q.facts:
        m["completeness"] = 0.0 if rep.abstained else sum(judge.covers(rep.answer, q.facts)) / len(q.facts)
    return m


def _vals(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]
    lat = _vals(ok, "latency_s")
    s = {
        "n": len(rows), "errors": len(rows) - len(ok),
        "unsupported_rate": ratio_bootstrap_ci([r["unsupported"] for r in ok], [r["cited_sentences"] for r in ok]),
        "uncited_per_answer": bootstrap_ci(_vals(ok, "uncited")),
        "gold_retrieved": bootstrap_ci(_vals(ans, "gold_retrieved")),
        "gold_cited": bootstrap_ci(_vals(ans, "gold_cited")),
        "completeness": bootstrap_ci(_vals(ans, "completeness")),
        "abstention_on_unanswerable": _prop(sum(r["abstained"] for r in unans), len(unans)),
        "false_abstention_on_answerable": _prop(sum(r["abstained"] for r in ans), len(ans)),
        "latency_p50_s": percentile(lat, .5) if lat else None,
        "latency_p95_s": percentile(lat, .95) if lat else None,
        "mean_cost_usd": _mean(_vals(ok, "cost_usd")),
        "mean_llm_calls": _mean(_vals(ok, "llm_calls")),
        "mean_tokens": _mean([r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in ok]),
    }
    by_type = defaultdict(list)
    for r in ok:
        by_type[r["type"]].append(r)
    s["by_type"] = {t: {"n": len(v),
                        "completeness": _mean(_vals(v, "completeness")),
                        "unsupported_rate": _safe_div(sum(r["unsupported"] for r in v), sum(r["cited_sentences"] for r in v)),
                        "abstained": _mean([float(r["abstained"]) for r in v])}
                    for t, v in sorted(by_type.items())}
    return s


def compare(rows_by_system: dict[str, list[dict]], reference: str) -> dict:
    """Paired comparisons of every system against the reference, per question, Holm-adjusted per metric."""
    if reference not in rows_by_system:
        return {}
    ref = {r["qid"]: r for r in rows_by_system[reference] if "error" not in r}
    metrics = {
        "unsupported_rate": lambda r: r["unsupported_rate"],
        "completeness": lambda r: r["completeness"],
        "gold_cited": lambda r: r["gold_cited"],
        "abstention_correct": lambda r: float(r["abstention_correct"]),
    }
    out: dict = {}
    for metric, f in metrics.items():
        res = {}
        for name, rows in rows_by_system.items():
            if name == reference:
                continue
            other = {r["qid"]: r for r in rows if "error" not in r}
            qids = sorted(set(ref) & set(other))
            a = [f(ref[q]) for q in qids]
            b = [f(other[q]) for q in qids]
            res[name] = paired_bootstrap(a, b)
        adj = holm({k: v["p"] for k, v in res.items()})
        for k in res:
            res[k]["p_holm"] = adj[k]
        out[metric] = res
    return out


def _prop(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"mean": k / n if n else float("nan"), "lo": lo, "hi": hi, "n": n}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _safe_div(a, b):
    return a / b if b else None


def fmt_ci(d: dict | None, pct: bool = True) -> str:
    if not d or d.get("n", 0) == 0 or d["mean"] is None or (isinstance(d["mean"], float) and math.isnan(d["mean"])):
        return "-"
    k = 100 if pct else 1
    return f"{d['mean'] * k:.1f} [{d['lo'] * k:.1f}, {d['hi'] * k:.1f}]"
