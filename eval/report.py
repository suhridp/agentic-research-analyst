"""Render results as markdown tables (summary.md) for the README and write-up."""
from __future__ import annotations

import math

from .metrics import fmt_ci


def _f(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x * 100:.1f}" if pct else f"{x:.{nd}f}"


def _backend(desc: dict) -> str:
    rb = desc.get("retriever_backend", {})

    def walk(d):
        k = d.get("kind")
        if k == "reranked":
            return f"{walk(d['first_stage'])} + rerank({d['reranker']})"
        if k == "hybrid_rrf":
            return "RRF(" + ", ".join(walk(c) for c in d["components"]) + ")"
        if k == "dense":
            return f"dense({d['encoder']})"
        return k or "?"
    return walk(rb)


def render_summary(summary: dict, meta: dict) -> str:
    m = summary["meta"]
    lines = [f"# {m['name']} ({m['split']} split, {m['n_questions']} questions)", ""]
    if summary.get("incomplete"):
        lines += [f"> **INCOMPLETE RUN:** {summary['incomplete']}", ""]
    c = m["corpus"]
    lines += [f"Corpus: `{c['path']}` - {c['documents']} documents, {c['passages']} passages "
              f"(fingerprint `{c['fingerprint']}`). Judge: `{m['judge'].get('model', m['judge']['kind'])}`.", ""]
    if m["judge"]["kind"] == "lexical":
        lines += ["> Judge is the **lexical proxy**: support and completeness numbers are plumbing checks, "
                  "not results. Use an LLM judge validated against hand labels for reported numbers.", ""]
    lines += ["Values are % with 95% bootstrap CIs over questions (abstention: Wilson).", "",
              "| System | Unsupported sentences | Completeness | Gold evidence cited | Gold evidence retrieved "
              "| Abstains on unanswerable | False abstention | p50 latency (s) | LLM calls / q | Cost / q ($) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary["systems"].items():
        lines.append(f"| `{name}` | {fmt_ci(s['unsupported_rate'])} | {fmt_ci(s['completeness'])} | "
                     f"{fmt_ci(s['gold_cited'])} | {fmt_ci(s['gold_retrieved'])} | "
                     f"{fmt_ci(s['abstention_on_unanswerable'])} | {fmt_ci(s['false_abstention_on_answerable'])} | "
                     f"{_f(s['latency_p50_s'])} | {_f(s['mean_llm_calls'], 1)} | {_f(s['mean_cost_usd'], 4)} |")
    errs = {n: s["errors"] for n, s in summary["systems"].items() if s["errors"]}
    if errs:
        lines += ["", f"Errors (questions that failed and are excluded): {errs}"]

    comps = summary.get("comparisons") or {}
    ref = summary.get("reference")
    if comps and ref:
        lines += ["", f"## Paired differences vs `{ref}` (reference minus system, percentage points)", "",
                  "Negative unsupported-rate difference = reference makes fewer unsupported statements. "
                  "p = paired sign-flip permutation test, Holm-adjusted within each metric.", "",
                  "| System | Δ Unsupported | Δ Completeness | Δ Gold cited | Δ Abstention correct |", "|---|---|---|---|---|"]
        others = sorted({n for d in comps.values() for n in d})
        for n in others:
            cells = []
            for metric in ("unsupported_rate", "completeness", "gold_cited", "abstention_correct"):
                d = comps.get(metric, {}).get(n)
                if not d or d["n"] == 0 or math.isnan(d["diff"]):
                    cells.append("-")
                else:
                    cells.append(f"{d['diff'] * 100:+.1f} [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}] p={d['p_holm']:.3f} (n={d['n']})")
            lines.append(f"| `{n}` | " + " | ".join(cells) + " |")

    lines += ["", "## By question type (completeness % / unsupported % / abstained %)", "",
              "| System | " + " | ".join(sorted({t for s in summary["systems"].values() for t in s["by_type"]})) + " |"]
    types = sorted({t for s in summary["systems"].values() for t in s["by_type"]})
    lines.append("|---|" + "---|" * len(types))
    for name, s in summary["systems"].items():
        cells = []
        for t in types:
            b = s["by_type"].get(t)
            cells.append("-" if not b else f"{_f(b['completeness'], pct=True)} / {_f(b['unsupported_rate'], pct=True)} / "
                                           f"{_f(b['abstained'], pct=True)} (n={b['n']})")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += ["", "## Systems", "", "| System | Kind | Retrieval (as actually run) | Planner | Verifier | Critic |",
              "|---|---|---|---|---|---|"]
    for name, d in meta["systems"].items():
        lines.append(f"| `{name}` | {d['kind']} | {_backend(d)} | {d['planner'] if d['kind'] == 'agentic' else '-'} | "
                     f"{d['verifier'] if d['kind'] == 'agentic' else '-'} | {d['critic'] if d['kind'] == 'agentic' else '-'} |")
    llms = {d["llm"]["model"] for d in meta["systems"].values() if d.get("llm")}
    lines += ["", f"LLM: {', '.join(sorted(llms)) or 'none (offline heuristics)'}. "
              f"Prompt version: {next(iter(meta['systems'].values()))['prompt_version']}. "
              f"Git: `{(meta['environment']['git']['commit'] or 'n/a')[:10]}`"
              f"{' (dirty)' if meta['environment']['git']['dirty'] else ''}."]
    return "\n".join(lines) + "\n"
