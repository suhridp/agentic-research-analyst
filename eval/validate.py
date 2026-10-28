"""Check a question set before spending money on it.

    python -m eval.validate --questions data/questions.jsonl --corpus data/corpus

Fails (exit 1) on schema errors or evidence quotes that don't resolve to any passage; prints the type
and split distribution against the plan's targets (~100 questions, ~15% unanswerable, 20 held out).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from ara.corpus import load_corpus

from .questions import load_questions, resolve_gold


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--corpus", default="data/corpus")
    a = ap.parse_args(argv)
    try:
        qs = load_questions(a.questions)
    except ValueError as e:
        print(e)
        return 1
    corpus = load_corpus(a.corpus)
    unresolved = []
    for q in qs:
        unresolved += resolve_gold(q, corpus)[1]
        for ev in q.evidence:
            if ev.doc not in {d.doc_id for d in corpus.documents}:
                unresolved.append(f"{q.id}: unknown doc '{ev.doc}'")
    types, splits = Counter(q.type for q in qs), Counter(q.split for q in qs)
    dup = [t for t, c in Counter(q.question.strip().lower() for q in qs).items() if c > 1]
    n = len(qs)
    print(f"{n} questions | splits {dict(splits)} | types {dict(types)}")
    print(f"unanswerable: {types['unanswerable'] / n:.0%} (target ~15%) | test: {splits['test']} (target 20)")
    print(f"documents cited by evidence: {len({e.doc for q in qs for e in q.evidence})} of {len(corpus.documents)}")
    no_facts = [q.id for q in qs if q.answerable and not q.facts]
    if no_facts:
        print(f"note: {len(no_facts)} answerable question(s) without `facts` (completeness not scored): {no_facts[:8]}")
    if dup:
        print(f"duplicate question text: {dup[:5]}")
    if unresolved:
        print(f"{len(unresolved)} unresolved evidence quote(s):", *unresolved, sep="\n  ")
        return 1
    print("OK: all evidence quotes resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
