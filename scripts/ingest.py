"""Convert PDFs in data/papers/ into clean, sectioned JSON documents in data/corpus/.

    python scripts/ingest.py [--src data/papers] [--dst data/corpus]

What cleaning does, and why:
  * column-aware extraction: two-column papers are read left column then right column. The gutter is
    found from word coordinates (pdfplumber). Plain pdftotext interleaves columns line by line when
    the gutter is narrow ("measure- The proposed detector ..."), silently corrupting passages.
  * repeated short lines (running headers/footers), bare page numbers and the arXiv margin stamp are removed.
  * hyphenated line breaks are re-joined, wrapped lines merged into paragraphs, ligatures normalized.
  * numbered/Roman/all-caps headings become section boundaries, so chunks never straddle sections.
  * everything from References/Bibliography onwards is dropped (citations would pollute retrieval).
  * numeric citation markers like [12] or [3]-[5] are removed; lines that are mostly symbols (tables,
    equations) are dropped.
A per-document log (_ingest_log.csv) flags suspicious extractions: very short text (scanned PDF?),
no detected sections, or no references heading found.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HEADING = re.compile(
    r"^(?:(?:[IVX]{1,5}|[A-H]|\d{1,2}(?:\.\d{1,2}){0,2})[.)]?\s+)?"
    r"(Abstract|Introduction|Related Work|Background|Preliminaries|System Model|Problem Formulation|"
    r"Methodology|Methods?|Approach|Proposed [A-Za-z ]+|Experimental Setup|Experiments?|Evaluation|"
    r"Simulation(?: Results| Setup)?|Results(?: and Discussion)?|Performance Evaluation|Discussion|"
    r"Limitations|Threat Model|Attack Model|Security Analysis|Conclusions?(?: and Future Work)?|Future Work)\b.{0,60}$",
    re.I)
NUMBERED = re.compile(r"^(?:[IVX]{1,5}|\d{1,2}(?:\.\d{1,2}){0,2})[.)]?\s+[A-Z][A-Za-z0-9 ,:&/\-()]{2,70}$")
END = re.compile(r"^(?:[IVX]+\.?\s+|\d+\.?\s+)?(References|Bibliography|Acknowledg(?:e)?ments?)\s*$", re.I)
ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s*\[[\w.\-]+\]\s*\d{1,2}\s+\w{3}\s+\d{4}")
CITATION = re.compile(r"\s?\[\d+(?:\s*[,–\-]\s*\d+)*\](?:\s*[–\-]\s*\[\d+\])?")


def pdf_to_text(pdf: Path) -> list[str]:
    """-> list of page texts in reading order. Uses the column-aware extractor when pdfplumber is
    installed, else pdftotext (reading-order mode; can interleave columns with narrow gutters)."""
    try:
        import pdfplumber  # noqa: F401
        return extract_columns(pdf)
    except ImportError:
        pass
    if not shutil.which("pdftotext"):
        sys.exit("need pdfplumber (pip install pdfplumber) or pdftotext (poppler-utils)")
    raw = subprocess.run(["pdftotext", "-enc", "UTF-8", str(pdf), "-"], capture_output=True, text=True,
                         timeout=120).stdout
    return raw.split("\f")


def find_gutter(words: list[dict], width: float) -> float | None:
    """x of the gap between two text columns, or None for a single-column page. Scans the middle
    third of the page for the x-position crossed by the fewest words; it is a gutter only if almost
    no words cross it and there is substantial text on both sides."""
    if len(words) < 40:
        return None
    best_x, best_cross = None, None
    for x in range(int(width * 0.35), int(width * 0.65)):
        cross = sum(1 for w in words if w["x0"] < x < w["x1"])
        if best_cross is None or cross < best_cross:
            best_x, best_cross = x, cross
    left = sum(1 for w in words if w["x1"] <= best_x)
    right = sum(1 for w in words if w["x0"] >= best_x)
    if best_cross <= 0.03 * len(words) and min(left, right) >= 0.2 * len(words):
        return float(best_x)
    return None


def _lines(words: list[dict], tol: float = 3.0) -> list[tuple[float, float, str]]:
    """Group words into lines -> [(top, bottom, text)] sorted top to bottom."""
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if rows and abs(rows[-1][0]["top"] - w["top"]) <= tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [(min(w["top"] for w in r), max(w["bottom"] for w in r),
             " ".join(w["text"] for w in sorted(r, key=lambda w: w["x0"]))) for r in rows]


def _with_breaks(lines: list[tuple[float, float, str]]) -> list[str]:
    """Insert a blank line where the vertical gap is clearly larger than normal line spacing."""
    if not lines:
        return []
    gaps = sorted(b[0] - a[0] for a, b in zip(lines, lines[1:]) if b[0] > a[0])
    typical = gaps[len(gaps) // 2] if gaps else 12.0
    out = [lines[0][2]]
    for a, b in zip(lines, lines[1:]):
        if b[0] - a[0] > 1.45 * typical:
            out.append("")
        out.append(b[2])
    return out


def extract_columns(pdf: Path) -> list[str]:
    import pdfplumber
    pages = []
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            words = page.extract_words(x_tolerance=1.5, y_tolerance=3, keep_blank_chars=False)
            g = find_gutter(words, float(page.width))
            if g is None:
                pages.append("\n".join(_with_breaks(_lines(words))))
                continue
            # Full-width rows (title, wide figures/tables, page-wide abstracts) cut the page into bands;
            # inside each band read the left column, then the right column.
            spanning = [w for w in words if w["x0"] < g < w["x1"]]
            span_rows = _lines(spanning)
            cuts = [r[0] for r in span_rows] + [float("inf")]
            out, prev = [], float("-inf")
            for cut, row in zip(cuts, span_rows + [None]):
                band = [w for w in words if prev <= w["top"] < cut and not (w["x0"] < g < w["x1"])]
                left = [w for w in band if w["x1"] <= g]
                right = [w for w in band if w["x0"] >= g]
                for col in (left, right):
                    if col:
                        out += _with_breaks(_lines(col)) + [""]
                if row is not None:
                    full = [w for w in words if abs(w["top"] - row[0]) <= 3]
                    out += [" ".join(w["text"] for w in sorted(full, key=lambda w: w["x0"])), ""]
                    prev = row[1]
            pages.append("\n".join(out))
    return pages


def _alpha_ratio(s: str) -> float:
    s = s.replace(" ", "")
    return sum(c.isalpha() for c in s) / max(len(s), 1)


def clean_pages(pages: list[str]) -> list[str]:
    """-> cleaned lines with headers/footers/page numbers removed."""
    norm = [[ln.strip() for ln in p.splitlines()] for p in pages]
    counts = Counter(ln for p in norm for ln in set(p) if 0 < len(ln) < 90)
    repeated = {ln for ln, c in counts.items() if len(pages) >= 4 and c >= max(3, 0.3 * len(pages))}
    lines: list[str] = []
    for p in norm:
        for ln in p:
            ln = ARXIV_STAMP.sub("", ln).strip()
            if ln in repeated or re.fullmatch(r"\d{1,4}", ln) or re.fullmatch(r"(?i)page \d+( of \d+)?", ln):
                continue
            lines.append(ln)
        lines.append("")  # page break = paragraph break
    return lines


def is_heading(ln: str) -> bool:
    if len(ln) > 90 or ln.endswith((".", ",", ";")) and not re.match(r"^[IVX\d]+\.\s", ln):
        return False
    if HEADING.match(ln):
        return True
    if NUMBERED.match(ln) and len(ln.split()) <= 9:
        return True
    letters = re.sub(r"[^A-Za-z]", "", ln)
    return len(letters) >= 4 and ln.isupper() and len(ln.split()) <= 7 and _alpha_ratio(ln) > 0.8


def structure(lines: list[str]) -> tuple[list[dict], dict]:
    sections: list[dict] = []
    heading, para, buf = "Front matter", [], []
    flags = {"references_found": False}

    def flush_para():
        if para:
            txt = " ".join(para)
            txt = re.sub(r"(\w)- (\w)", r"\1\2", txt)  # de-hyphenate across wrapped lines
            buf.append(txt)
            para.clear()

    def flush_section():
        flush_para()
        text = CITATION.sub("", " ".join(buf))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) >= 15 or (not sections and len(text.split()) >= 5):
            sections.append({"heading": heading, "text": text})
        elif text and sections:  # short subsection: keep its text with the parent section
            sections[-1]["text"] += " " + text
        buf.clear()

    for ln in lines:
        if END.match(ln):
            flags["references_found"] = True
            break
        if not ln:
            flush_para()
            continue
        if is_heading(ln):
            flush_section()
            heading = re.sub(r"^(?:[IVX]{1,5}|[A-H]|\d{1,2}(?:\.\d{1,2}){0,2})[.)]?\s+", "", ln).strip().title()
            continue
        if len(ln) < 40 and _alpha_ratio(ln) < 0.5:  # table cells, equation fragments
            continue
        para.append(ln)
    flush_section()
    return sections, flags


def ingest_pdf(pdf: Path, meta: dict) -> tuple[dict, dict]:
    pages = pdf_to_text(pdf)
    sections, flags = structure(clean_pages(pages))
    words = sum(len(s["text"].split()) for s in sections)
    warn = []
    if words < 1500:
        warn.append("short text (scanned PDF or extraction failure?)")
    if len(sections) < 3:
        warn.append("few sections detected")
    if not flags["references_found"]:
        warn.append("no references heading (tail may contain citations)")
    doc = {"doc_id": meta.get("doc_id") or pdf.stem, "title": meta.get("title") or pdf.stem,
           "url": meta.get("url", ""), "license": meta.get("license", ""),
           "redistributable": meta.get("redistributable", ""), "sections": sections}
    return doc, {"file": pdf.name, "pages": len(pages), "sections": len(sections), "words": words,
                 "warnings": "; ".join(warn)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/papers"))
    ap.add_argument("--dst", type=Path, default=Path("data/corpus"))
    a = ap.parse_args(argv)
    meta = {}
    src_csv = a.src / "SOURCES.csv"
    if src_csv.exists():
        meta = {r["file"]: r for r in csv.DictReader(src_csv.open(encoding="utf-8"))}
    else:
        print(f"WARNING: {src_csv} missing - no titles or licence provenance will be recorded", file=sys.stderr)
    pdfs = sorted(a.src.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs in {a.src}")
    a.dst.mkdir(parents=True, exist_ok=True)
    log = []
    for pdf in pdfs:
        doc, row = ingest_pdf(pdf, meta.get(pdf.name, {}))
        (a.dst / f"{doc['doc_id']}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        log.append(row)
        flag = f"  ! {row['warnings']}" if row["warnings"] else ""
        print(f"{pdf.name}: {row['pages']} pages, {row['sections']} sections, {row['words']} words{flag}")
    with (a.dst / "_ingest_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"{len(log)} documents -> {a.dst} ({sum(1 for r in log if r['warnings'])} with warnings; see _ingest_log.csv)")


if __name__ == "__main__":
    main()
