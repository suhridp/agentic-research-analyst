import json

import pytest

from ara.corpus import Document, chunk, load_corpus
from ara.text import has_negation, numbers, sentences, tokens


def test_numbers_canonical_and_ignore_identifiers():
    assert numbers("1,000 flows at 3.50 ms, 97% and three cells") == {"1000", "3.5", "97", "3"}
    assert numbers("5G cores and 3GPP Release") == set()


def test_sentence_split_survives_abbreviations_and_decimals():
    s = sentences("As in Fig. 3 of Smith et al. 2021, e.g. at 3.5 GHz, it works. Next one.")
    assert s == ["As in Fig. 3 of Smith et al. 2021, e.g. at 3.5 GHz, it works.", "Next one."]


def test_tokens_stem_plurals_and_drop_stopwords():
    assert tokens("The handover failures of networks") == ["handover", "failure", "network"]
    assert tokens("data-driven inter-RAT at 3.5 GHz") == ["data", "driven", "inter", "rat", "3.5", "ghz"]


def test_negation_detection():
    assert has_negation("The model did not reduce failures")
    assert has_negation("It can't be detected")
    assert not has_negation("The model reduced failures")


def test_chunking_respects_sections_and_budget():
    long = " ".join(f"Sentence number {i} is here." for i in range(60))
    doc = Document("d", "Title", [("Intro", long), ("Results", "Short results section here.")])
    ps = chunk(doc, max_words=40, overlap=1)
    assert all(len(p.text.split()) <= 40 for p in ps)
    assert ps[-1].section == "Results" and ps[-1].text == "Short results section here."
    assert not any("Short results" in p.text for p in ps if p.section == "Intro")
    # overlap: consecutive chunks in a section share a sentence
    assert ps[0].text.split(". ")[-1].rstrip(".") in ps[1].text
    assert [p.idx for p in ps] == list(range(len(ps)))


def test_index_text_has_header_but_text_is_raw(corpus):
    p = corpus.passages[0]
    assert p.title in p.index_text and p.title not in p.text


def test_find_quote_whitespace_and_case_insensitive(corpus):
    assert corpus.find_quote("syn_dos", "DETECTED DoS   floods with 97 percent accuracy") == ["syn_dos#1"]
    assert corpus.find_quote("syn_dos", "not in this document at all, really") == []


def test_json_documents_and_duplicate_ids(tmp_path):
    d = {"doc_id": "x1", "title": "T", "license": "CC BY", "sections": [{"heading": "A", "text": "One fact here. Two."}]}
    (tmp_path / "x1.json").write_text(json.dumps(d))
    c = load_corpus(tmp_path)
    assert c.passages[0].id == "x1#0" and c.doc_meta("x1")["license"] == "CC BY"
    (tmp_path / "x1.md").write_text("# T\nduplicate")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(tmp_path)


def test_fingerprint_changes_with_chunking():
    from conftest import SYN
    assert load_corpus(SYN, max_words=160).fingerprint != load_corpus(SYN, max_words=20).fingerprint
