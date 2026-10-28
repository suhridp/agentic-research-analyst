"""Documents, passages and section-aware chunking.

Supported inputs in the corpus folder:
  * <id>.json  - written by scripts/ingest.py: {"doc_id","title","url","license","sections":[{"heading","text"}]}
  * <id>.txt / <id>.md - plain text; markdown '#' lines become section headings, the first one the title.

Chunking packs whole sentences into passages of at most `max_words`, never crossing a section
boundary, with `overlap` sentences carried over so a fact split across a boundary stays retrievable.
Each passage keeps its title and section heading; retrieval indexes `index_text` (header + text)
while verification only ever sees the raw `text`.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .text import normalize, sentences


@dataclass(frozen=True)
class Passage:
    doc: str
    idx: int
    text: str
    title: str = ""
    section: str = ""

    @property
    def id(self) -> str:
        return f"{self.doc}#{self.idx}"

    @property
    def index_text(self) -> str:
        head = " | ".join(x for x in (self.title, self.section) if x)
        return f"{head}\n{self.text}" if head else self.text

    def to_dict(self) -> dict:
        return {"id": self.id, "doc": self.doc, "section": self.section, "text": self.text}


@dataclass
class Document:
    doc_id: str
    title: str
    sections: list[tuple[str, str]]  # (heading, text)
    meta: dict = field(default_factory=dict)


def _from_markdown(stem: str, raw: str) -> Document:
    title, heading, buf, sections = "", "", [], []
    for line in raw.splitlines():
        m = re.match(r"^\s*#{1,6}\s+(.*)$", line)
        if m:
            if buf:
                sections.append((heading, " ".join(buf)))
                buf = []
            if not title:
                title = m.group(1).strip()
            else:
                heading = m.group(1).strip()
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((heading, " ".join(buf)))
    return Document(stem, title or stem, sections)


def load_documents(folder: str | Path) -> list[Document]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"corpus folder not found: {folder}")
    docs: list[Document] = []
    for p in sorted(folder.iterdir()):
        if p.name.startswith(("_", ".")):
            continue
        if p.suffix == ".json":
            d = json.loads(p.read_text(encoding="utf-8"))
            secs = [(s.get("heading", ""), s["text"]) for s in d["sections"] if s.get("text", "").strip()]
            meta = {k: v for k, v in d.items() if k not in ("sections", "doc_id", "title")}
            docs.append(Document(d.get("doc_id", p.stem), d.get("title", p.stem), secs, meta))
        elif p.suffix in (".txt", ".md"):
            docs.append(_from_markdown(p.stem, p.read_text(encoding="utf-8")))
    ids = [d.doc_id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate doc ids in {folder}: {sorted(dupes)}")
    return docs


def chunk(doc: Document, max_words: int = 160, overlap: int = 1) -> list[Passage]:
    out: list[Passage] = []
    for heading, text in doc.sections:
        sents = sentences(text)
        i = 0
        while i < len(sents):
            j, n = i, 0
            while j < len(sents) and (n == 0 or n + len(sents[j].split()) <= max_words):
                n += len(sents[j].split())
                j += 1
            out.append(Passage(doc.doc_id, len(out), " ".join(sents[i:j]), doc.title, heading))
            if j >= len(sents):
                break
            i = max(j - overlap, i + 1)
    return out


@dataclass
class Corpus:
    documents: list[Document]
    passages: list[Passage]
    by_id: dict[str, Passage]
    params: dict

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256(json.dumps(self.params, sort_keys=True).encode())
        for p in self.passages:
            h.update(p.id.encode())
            h.update(p.index_text.encode())
        return h.hexdigest()[:16]

    def doc_meta(self, doc_id: str) -> dict:
        for d in self.documents:
            if d.doc_id == doc_id:
                return {"title": d.title, **d.meta}
        return {}

    def find_quote(self, doc: str, quote: str) -> list[str]:
        """Passage ids in `doc` whose text contains `quote` (whitespace/case-insensitive).
        A quote straddling a chunk boundary maps to every passage that contains a long enough piece of it."""
        q = normalize(quote).lower()
        cands = [p for p in self.passages if p.doc == doc]
        hits = [p.id for p in cands if q in normalize(p.text).lower()]
        if hits or len(q) < 40:
            return hits
        # straddling quote: accept passages containing the first or last 60% of it
        cut = int(len(q) * 0.6)
        head, tail = q[:cut], q[-cut:]
        return [p.id for p in cands if head in normalize(p.text).lower() or tail in normalize(p.text).lower()]


def load_corpus(folder: str | Path, max_words: int = 160, overlap: int = 1) -> Corpus:
    docs = load_documents(folder)
    passages = [p for d in docs for p in chunk(d, max_words, overlap)]
    return Corpus(docs, passages, {p.id: p for p in passages},
                  {"folder": str(folder), "max_words": max_words, "overlap": overlap})
