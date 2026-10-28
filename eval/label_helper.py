"""Speed up gold labelling: for each draft question, show the top candidate passages from BM25 and
dense retrieval (union, so labels aren't biased toward one retriever) in a review file.

    python -m eval.label_helper --draft data/questions_draft.jsonl --out data/label_review.md

Draft line: {"id": "q001", "question": "...", "type": "factual|numeric|comparison|gap|unanswerable"}
Read the passages, then write the final line into data/questions.jsonl with `evidence` quotes copied
VERBATIM from the passage text, `facts`, and `split`. Also search the PDFs yourself: gold must not be
limited to what the retrievers happen to find, or Recall is overstated.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from ara.corpus import load_corpus
from ara.retrieval import build_retriever


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", default="data/label_review.md")
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args(argv)
    corpus = load_corpus(a.corpus)
    bm25 = build_retriever({"kind": "bm25"}, corpus)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dense = build_retriever({"kind": "dense", "encoder": "auto"}, corpus)
    lines = ["# Label review", "", "Copy evidence quotes verbatim (8+ characters) from the passage text.", ""]
    for raw in Path(a.draft).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        q = json.loads(raw)
        lines += [f"## {q.get('id', '?')} ({q.get('type', '?')}): {q['question']}", ""]
        seen = set()
        for src, r in (("bm25", bm25), ("dense", dense)):
            for h in r.search(q["question"], a.k):
                if h.passage.id in seen:
                    continue
                seen.add(h.passage.id)
                p = h.passage
                lines.append(f"- **{p.id}** ({src}; {p.section or 'no section'}): {p.text}")
        lines += ["", "evidence: ____  facts: ____  split: ____", ""]
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
