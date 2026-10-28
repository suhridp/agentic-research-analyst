"""Build the real corpus from arXiv, with licence provenance (milestone 1).

Step 1 - find candidates (writes data/papers/candidates.csv with each paper's licence):
    python scripts/fetch_arxiv.py search --max-per-query 40
Step 2 - curate: copy the rows you want into data/papers/manifest.csv and set include=1.
         Aim for 40-60 papers; read titles/abstracts, drop off-topic ones, and prefer CC licences.
Step 3 - download exactly the manifest (pinned versions, so the corpus is reproducible):
    python scripts/fetch_arxiv.py download
Then: python scripts/ingest.py

Licences come from arXiv's OAI-PMH metadata. Only CC BY / CC BY-SA / CC0 papers are marked
redistributable=1; the repo never commits PDFs or extracted text (see .gitignore), only the manifest,
so anyone can rebuild the identical corpus. Requests are spaced >= 3 s apart per arXiv's API terms.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://export.arxiv.org/api/query"
OAI = "https://export.arxiv.org/oai2"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}
REDISTRIBUTABLE = re.compile(r"creativecommons\.org/(licenses/(by|by-sa)/|publicdomain/zero/)")
DEFAULT_QUERIES = [
    'abs:handover AND (abs:5G OR abs:6G) AND (abs:prediction OR abs:learning)',
    'abs:"conditional handover"',
    'abs:"mobility management" AND (abs:5G OR abs:6G)',
    'abs:handover AND (abs:attack OR abs:security OR abs:spoofing OR abs:adversarial)',
    'abs:"fake base station" OR abs:"rogue base station"',
    'abs:5G AND abs:"denial of service" AND abs:detection',
    'abs:"network slicing" AND abs:security',
]
FIELDS = ["arxiv_id", "title", "authors", "published", "primary_category", "license", "redistributable",
          "abs_url", "pdf_url", "query", "include", "notes"]
_last = [0.0]


def _get(url: str, timeout: float = 60) -> bytes:
    wait = 3.1 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    ua = f"ara-research-analyst/0.2 (mailto:{os.getenv('ARXIV_CONTACT', 'unknown')})"
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                _last[0] = time.time()
                return r.read()
        except Exception as e:  # network hiccup or 503 from arXiv: back off and retry
            _last[0] = time.time()
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} after error: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def parse_atom(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    out = []
    for e in root.findall("a:entry", NS):
        abs_url = e.findtext("a:id", "", NS).strip()
        aid = abs_url.rsplit("/abs/", 1)[-1]
        pdf = next((link.get("href") for link in e.findall("a:link", NS) if link.get("title") == "pdf"),
                   f"https://arxiv.org/pdf/{aid}")
        prim = e.find("arxiv:primary_category", NS)
        out.append({"arxiv_id": aid,
                    "title": " ".join(e.findtext("a:title", "", NS).split()),
                    "authors": "; ".join(a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)),
                    "published": e.findtext("a:published", "", NS)[:10],
                    "primary_category": prim.get("term") if prim is not None else "",
                    "abs_url": abs_url, "pdf_url": pdf})
    return out


def parse_oai_license(xml: bytes) -> str:
    root = ET.fromstring(xml)
    lic = root.find(".//ax:license", NS)
    return (lic.text or "").strip() if lic is not None else ""


def license_of(arxiv_id: str) -> str:
    base = re.sub(r"v\d+$", "", arxiv_id)
    q = urllib.parse.urlencode({"verb": "GetRecord", "identifier": f"oai:arXiv.org:{base}", "metadataPrefix": "arXiv"})
    try:
        return parse_oai_license(_get(f"{OAI}?{q}")) or "arxiv-default (non-exclusive distribution licence)"
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def search(queries: list[str], per_query: int, out: Path) -> None:
    seen: dict[str, dict] = {}
    for q in queries:
        url = f"{API}?" + urllib.parse.urlencode({"search_query": q, "start": 0, "max_results": per_query,
                                                  "sortBy": "relevance"})
        entries = parse_atom(_get(url))
        print(f"{len(entries):3d} results  {q}")
        for e in entries:
            seen.setdefault(re.sub(r"v\d+$", "", e["arxiv_id"]), {**e, "query": q})
    print(f"{len(seen)} unique papers; fetching licences (3 s per request)...")
    for i, e in enumerate(seen.values(), 1):
        e["license"] = license_of(e["arxiv_id"])
        e["redistributable"] = int(bool(REDISTRIBUTABLE.search(e["license"])))
        e["include"], e["notes"] = "", ""
        if i % 10 == 0:
            print(f"  {i}/{len(seen)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(seen.values())
    n_cc = sum(e["redistributable"] for e in seen.values())
    print(f"wrote {out}: {len(seen)} candidates, {n_cc} under CC BY/BY-SA/CC0")


def safe_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def download(manifest: Path, dest: Path) -> None:
    rows = [r for r in csv.DictReader(manifest.open(encoding="utf-8")) if r.get("include", "").strip() == "1"]
    if not rows:
        sys.exit(f"{manifest}: no rows with include=1")
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    for r in rows:
        f = dest / f"{safe_name(r['arxiv_id'])}.pdf"
        if f.exists() and f.read_bytes()[:4] == b"%PDF":
            ok += 1
            continue
        url = r["pdf_url"] if r["pdf_url"].endswith(r["arxiv_id"]) else f"https://arxiv.org/pdf/{r['arxiv_id']}"
        data = _get(url, timeout=120)
        if data[:4] != b"%PDF":
            print(f"  ! {r['arxiv_id']}: not a PDF, skipped", file=sys.stderr)
            continue
        f.write_bytes(data)
        ok += 1
        print(f"  {r['arxiv_id']}  {len(data) // 1024} KB  {r['title'][:60]}")
    with (dest / "SOURCES.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["file", "doc_id", "title", "authors", "url", "license", "redistributable"])
        w.writeheader()
        for r in rows:
            w.writerow({"file": f"{safe_name(r['arxiv_id'])}.pdf", "doc_id": safe_name(r["arxiv_id"]),
                        "title": r["title"], "authors": r["authors"], "url": r["abs_url"],
                        "license": r["license"], "redistributable": r["redistributable"]})
    print(f"{ok}/{len(rows)} PDFs in {dest}; provenance in {dest / 'SOURCES.csv'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--query", action="append", help="arXiv query (repeatable); defaults to the built-in topic list")
    s.add_argument("--max-per-query", type=int, default=40)
    s.add_argument("--out", type=Path, default=Path("data/papers/candidates.csv"))
    d = sub.add_parser("download")
    d.add_argument("--manifest", type=Path, default=Path("data/papers/manifest.csv"))
    d.add_argument("--dest", type=Path, default=Path("data/papers"))
    a = ap.parse_args(argv)
    if a.cmd == "search":
        search(a.query or DEFAULT_QUERIES, a.max_per_query, a.out)
    else:
        download(a.manifest, a.dest)


if __name__ == "__main__":
    main()
