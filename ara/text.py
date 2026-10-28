"""Text utilities shared by retrieval, verification and evaluation."""
from __future__ import annotations

import re
import unicodedata

STOP = frozenset("""a an the of and or to in on for with is are was were be been being by as at from that this these
those it its we our us they their them can could which what when where who whom how do does did not no nor than
then there here also such into onto over under via per using use used based between within without about after
before during while if else so may might will would should shall has have had having i you he she his her""".split())
# "not", "no", "without" are stopwords for retrieval but matter for verification (see NEGATION).

NEGATION = frozenset({"not", "no", "never", "none", "neither", "nor", "without", "cannot", "fails", "fail",
                      "failed", "unable", "lack", "lacks", "absent", "n't"})

# "one" and "zero" are left out on purpose: as words they are too often not quantities.
_NUM_WORDS = {w: str(i) for i, w in enumerate("_ _ two three four five six seven eight nine ten eleven twelve "
                                               "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
                                               "twenty".split()) if w != "_"}
_NUM_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "hundred": "100", "thousand": "1000",
                   "half": "0.5"})

_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|et al|etc|vs|cf|Fig|Figs|Eq|Eqs|Sec|Ref|Refs|Tab|No|approx|resp|"
                     r"Dr|Prof|Inc|Ltd|Jr|Sr|St)\.|\b[A-Z]\.)$")
_WORD = re.compile(r"[a-z0-9]+(?:\.\d+)*")  # hyphenated terms split: "data-driven" -> data, driven
# A number not glued to letters, so "5G", "3GPP", "x2" and ids are not treated as quantities.
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)")


def normalize(text: str) -> str:
    """Unicode-normalize, fix ligatures and odd dashes/spaces from PDF extraction, collapse whitespace."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("−", "-").replace("–", "-").replace("—", " - ").replace(" ", " ")
    return re.sub(r"\s+", " ", t).strip()


def _stem(w: str) -> str:
    if w[0].isdigit() or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def words(text: str) -> list[str]:
    """Lowercased word tokens, no stopword removal (used for negation and n-gram checks)."""
    return _WORD.findall(normalize(text).lower())


def tokens(text: str) -> list[str]:
    """Content tokens for retrieval and overlap: lowercased, stopwords removed, light plural stemming."""
    return [_stem(w) for w in words(text) if w not in STOP and w not in NEGATION]


def numbers(text: str) -> set[str]:
    """Numbers in canonical form: '1,000' -> '1000', '3.50' -> '3.5', 'three' -> '3'."""
    out = set()
    for m in NUM.findall(normalize(text)):
        v = m.replace(",", "").lstrip("+")
        if "." in v:
            v = v.rstrip("0").rstrip(".")
        out.add(v)
    for w in words(text):
        if w in _NUM_WORDS:
            out.add(_NUM_WORDS[w])
    return out


def sentences(text: str) -> list[str]:
    """Sentence split that survives 'e.g.', 'et al.', 'Fig. 3', initials and decimals."""
    text = normalize(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])", text)
    out: list[str] = []
    for p in pieces:
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return [s.strip() for s in out if s.strip()]


def has_negation(text: str) -> bool:
    ws = words(text.replace("n't", " not"))
    return any(w in NEGATION for w in ws)
