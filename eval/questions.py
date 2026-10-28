"""Question-set schema, loading and gold resolution.

Gold evidence is stored as QUOTES, not passage ids:

    {"id": "q017", "type": "numeric", "split": "dev",
     "question": "...",
     "evidence": [{"doc": "2301.01234", "quote": "reduced handover failures by 18 percent"}],
     "facts": ["18 percent fewer handover failures"],          # used for answer completeness
     "reference": "optional free-text reference answer"}

Passage ids change whenever chunking changes; quotes don't. At eval time each evidence item is
resolved to the set of passages that contain it, so re-chunking never silently invalidates labels.
Unanswerable questions have no evidence and no facts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

QType = Literal["factual", "numeric", "comparison", "gap", "unanswerable"]


class Evidence(BaseModel):
    doc: str
    quote: str = Field(min_length=8)


class Question(BaseModel):
    id: str
    question: str = Field(min_length=8)
    type: QType
    split: Literal["dev", "test"] = "dev"
    evidence: list[Evidence] = []
    facts: list[str] = []
    reference: str = ""
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"

    @model_validator(mode="after")
    def _consistent(self):
        if self.type == "unanswerable" and (self.evidence or self.facts):
            raise ValueError(f"{self.id}: unanswerable questions must have no evidence/facts")
        if self.type != "unanswerable" and not self.evidence:
            raise ValueError(f"{self.id}: answerable questions need at least one evidence quote")
        return self


def load_questions(path: str | Path, split: str | None = None) -> list[Question]:
    qs, errors = [], []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            qs.append(Question.model_validate(json.loads(line)))
        except Exception as e:  # collect all problems, then fail once
            errors.append(f"line {n}: {e}")
    ids = [q.id for q in qs]
    errors += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    if errors:
        raise ValueError(f"{path}: invalid question set:\n  " + "\n  ".join(errors))
    return [q for q in qs if split in (None, "all") or q.split == split]


def resolve_gold(q: Question, corpus) -> tuple[list[set[str]], list[str]]:
    """-> (one set of passage ids per evidence item, list of unresolved quotes)."""
    groups, missing = [], []
    for ev in q.evidence:
        ids = set(corpus.find_quote(ev.doc, ev.quote))
        if ids:
            groups.append(ids)
        else:
            missing.append(f"{q.id}: [{ev.doc}] {ev.quote[:70]}")
    return groups, missing
