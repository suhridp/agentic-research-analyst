"""Pull failure cases out of a run into a markdown file for hand-coding (failure analysis is part of
the deliverable, not an appendix).

    python -m eval.failures results/main_comparison/<stamp> [--system agentic_full] [--max 15]

Sections: unsupported statements that reached the answer, answers to unanswerable questions, false
abstentions, gold evidence never retrieved, gold retrieved but not cited, and verifier rejections of
claims that cite gold evidence (possible false rejects). Each case has a `Category:` line; code it
with the taxonomy in docs/EVALUATION.md, then count categories for the write-up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--system")
    ap.add_argument("--max", type=int, default=15, help="cases per section")
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in (a.run / "per_question.jsonl").read_text(encoding="utf-8").splitlines()]
    qtext = {}
    if (a.run / "config.yaml").exists():
        import yaml

        from .questions import load_questions
        cfg = yaml.safe_load((a.run / "config.yaml").read_text())
        qtext = {q.id: q for q in load_questions(cfg["questions"])}
    if a.system:
        rows = [r for r in rows if r.get("system") == a.system]
    sections: dict[str, list[str]] = {k: [] for k in (
        "Unsupported statements in the answer", "Answered an unanswerable question", "False abstention",
        "Gold evidence never retrieved", "Gold evidence retrieved but not cited",
        "Verifier rejected a claim citing gold evidence (possible false reject)", "Errors")}

    def head(r):
        q = qtext.get(r["qid"])
        return f"### {r['qid']} - `{r['system']}` ({r['type']})\n**Q:** {q.question if q else '?'}\n"

    for r in rows:
        if "error" in r:
            sections["Errors"].append(f"{head(r)}\n`{r['error']}`\n")
            continue
        rep = r["report"]
        gold = [set(g) for g in r.get("gold_groups", [])]
        for j in r.get("sentence_judgements", []):
            if not j["supported"]:
                sections["Unsupported statements in the answer"].append(
                    f"{head(r)}**Statement:** {j['sentence']}\n**Cited:** {', '.join(j['cited'])}\n\nCategory: ____\n")
        if not r["answerable"] and not r["abstained"]:
            sections["Answered an unanswerable question"].append(f"{head(r)}**Answer:**\n\n{rep['answer'][:800]}\n\nCategory: ____\n")
        if r["answerable"] and r["abstained"]:
            sections["False abstention"].append(
                f"{head(r)}Sub-questions: {rep['subquestions']}\nRejected claims: {len(rep['rejected'])}\n"
                + "".join(f"- ({c['verdict']}, {c['stage']}: {c['rationale'][:80]}) {c['text'][:150]} [{c['source']}]\n"
                          for c in rep["rejected"][:5]) + "\nCategory: ____\n")
        if gold:
            retrieved, cited = set(rep["retrieved"]), set(rep["cited"])
            missed = [sorted(g) for g in gold if not g & retrieved]
            if missed:
                sections["Gold evidence never retrieved"].append(f"{head(r)}Missing: {missed}\n\nCategory: ____\n")
            uncited = [sorted(g) for g in gold if g & retrieved and not g & cited]
            if uncited and not r["abstained"]:
                sections["Gold evidence retrieved but not cited"].append(f"{head(r)}Not cited: {uncited}\n\nCategory: ____\n")
            gold_ids = set().union(*gold)
            for c in rep["rejected"]:
                if c["source"] in gold_ids:
                    sections["Verifier rejected a claim citing gold evidence (possible false reject)"].append(
                        f"{head(r)}**Claim:** {c['text']}\n**Source:** {c['source']} - {c['verdict']} via {c['stage']}: "
                        f"{c['rationale'][:150]}\n\nCategory: ____\n")
    out = [f"# Failure cases: {a.run}", "", "Counts: " + ", ".join(f"{k}: {len(v)}" for k, v in sections.items()), ""]
    for k, v in sections.items():
        if v:
            out += [f"## {k} ({len(v)})", ""] + v[:a.max] + ([f"_... {len(v) - a.max} more_", ""] if len(v) > a.max else [])
    path = a.run / "failures.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
