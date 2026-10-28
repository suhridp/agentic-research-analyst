import shutil
from pathlib import Path

import fetch_arxiv
import ingest
import pytest
from conftest import ROOT

FIX = ROOT / "tests" / "fixtures"


def test_structure_drops_references_and_citations():
    lines = ["I. INTRODUCTION", "Handover prediction matters [3], [4]-[6] for mobility in dense", "networks today.",
             "", "II. RESULTS", "The detector reduced attacks by 35 percent compared with the baseline approach.",
             "x = 0.3 + y", "References", "[1] A. Author, a paper."]
    sections, flags = ingest.structure(lines)
    assert [s["heading"] for s in sections] == ["Introduction", "Results"] or sections[0]["heading"] == "Introduction"
    text = " ".join(s["text"] for s in sections)
    assert "[3]" not in text and "A. Author" not in text and flags["references_found"]


def test_headers_footers_and_page_numbers_removed():
    pages = [f"IEEE TRANSACTIONS ON NETWORKS\nBody text on page {i} that is unique.\n{i + 1}" for i in range(5)]
    lines = ingest.clean_pages(pages)
    assert "IEEE TRANSACTIONS ON NETWORKS" not in lines and "3" not in lines
    assert any("page 2" in ln for ln in lines)


@pytest.mark.skipif(not (shutil.which("pdftotext") or __import__("importlib").util.find_spec("pdfplumber")),
                    reason="needs pdfplumber or pdftotext")
def test_two_column_pdf_is_read_in_column_order():
    doc, row = ingest.ingest_pdf(FIX / "two_column_paper.pdf", {"doc_id": "fx", "title": "Fixture"})
    text = " ".join(s["text"] for s in doc["sections"])
    assert "measurement reports. Our detector flags 92 percent" in text  # no column interleaving
    assert "The proposed detector reduced successful spoofing attacks by 35 percent" in text
    assert "[3]" not in text and "A. Author" not in text
    heads = [s["heading"] for s in doc["sections"]]
    assert {"Introduction", "System Model", "Results", "Conclusion"} <= set(heads)


def test_arxiv_parsers():
    e = fetch_arxiv.parse_atom((FIX / "arxiv_atom.xml").read_bytes())[0]
    assert e["arxiv_id"] == "2301.01234v2" and e["title"] == "Learning-Based Handover Prediction for 5G"
    assert e["authors"] == "Ada Lovelace; Alan Turing" and e["primary_category"] == "cs.NI"
    lic = fetch_arxiv.parse_oai_license((FIX / "arxiv_oai.xml").read_bytes())
    assert lic.endswith("/by/4.0/") and fetch_arxiv.REDISTRIBUTABLE.search(lic)
    assert not fetch_arxiv.REDISTRIBUTABLE.search("http://creativecommons.org/licenses/by-nc-nd/4.0/")


def test_api_roundtrip(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ARA_CONFIG", str(Path(ROOT / "configs/smoke.yaml")))
    monkeypatch.chdir(ROOT)
    from api import main
    main._state.cache_clear()
    c = TestClient(main.app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/ask", json={"question": "What did anomaly detection flag on spoofed measurement reports?"}).json()
    assert r["claims"] and all(pid in r["sources"] for pid in {cl["source"] for cl in r["claims"]})
    assert c.post("/ask", json={"question": "Something here?", "system": "nope"}).status_code == 404
