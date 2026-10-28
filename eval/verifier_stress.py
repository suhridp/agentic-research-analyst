"""Verifier stress test: how often does a verifier accept claims that are subtly WRONG?

Needs no hand labels, so it runs on any corpus. Real sentences are sampled from the corpus and turned
into claims; each claim is checked against the passage the sentence came from:

  original          the sentence itself                             -> should be ACCEPTED
  paraphrase        faithful LLM rewording (needs --llm)            -> should be ACCEPTED
  number_change     one number altered                              -> should be rejected
  negation          meaning negated                                 -> should be rejected
  direction_flip    increase<->decrease, higher<->lower, ...        -> should be rejected
  hedge_removal     may/can/suggests -> will/always/proves          -> should be rejected
  qualifier_drop    "simulated", "synthetic", "up to", ... removed  -> should be rejected
  term_swap         one technical term replaced by a term from another document -> should be rejected

Reported per verifier: accept rate per perturbation with a Wilson 95% CI (claims are independent
samples here, so Wilson is appropriate). Labels are produced automatically; examples.csv lets you
spot-check that each perturbation really changed the meaning.

    python -m eval.verifier_stress --corpus data/corpus --n 200
    python -m eval.verifier_stress --corpus data/corpus --n 200 --llm-config configs/main.yaml
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict

from ara.agents import Claim, VerifierConfig, verify
from ara.corpus import load_corpus
from ara.text import NUM, has_negation, numbers, sentences, tokens
from common.config import load_yaml, run_dir, write_json
from common.llm import build_llm
from common.stats import wilson

FLIPS = [("increase", "decrease"), ("increased", "decreased"), ("increases", "decreases"), ("higher", "lower"),
         ("more", "fewer"), ("improves", "degrades"), ("improved", "degraded"), ("reduced", "increased"),
         ("reduces", "increases"), ("above", "below"), ("faster", "slower"), ("larger", "smaller"),
         ("better", "worse"), ("outperforms", "underperforms"), ("outperformed", "underperformed"),
         ("gain", "loss"), ("maximize", "minimize"), ("before", "after"), ("enables", "prevents")]
HEDGES = [(r"\bmay\b", "will"), (r"\bmight\b", "will"), (r"\bcan\b", "always"), (r"\bcould\b", "will"),
          (r"\bsuggests?\b", "proves"), (r"\boften\b", "always"), (r"\bsome\b", "all"), (r"\btypically\b", "always"),
          (r"\blikely\b", "certainly")]
QUALIFIERS = [r"\bin (?:the )?simulat(?:ed|ion)s?(?: scenario| study)?\b", r"\bsimulated\b", r"\bsynthetic\b",
              r"\bup to\b", r"\bapproximately\b", r"\baround\b", r"\bon average\b", r"\bunder [a-z\- ]+ conditions\b",
              r"\bin some cases\b", r"\bpartially\b"]
AUX = re.compile(r"\b(is|are|was|were|can|may|does|did|has|have|had|will|could|should|might)\b", re.I)


def _keep_case(src: str, dst: str) -> str:
    return dst.capitalize() if src[:1].isupper() else dst


def perturb_number(s: str, rng: random.Random) -> str | None:
    ms = list(NUM.finditer(s))
    if not ms:
        return None
    m = rng.choice(ms)
    v = m.group().replace(",", "")
    try:
        x = float(v)
    except ValueError:
        return None
    if "." in v:
        dec = len(v.split(".")[1])
        new = f"{x * rng.choice([0.5, 1.5, 2]) + 10 ** -dec:.{dec}f}"
    else:
        new = str(int(x) + rng.choice([1, 2, 5, 10]) if x < 20 else int(round(x * rng.choice([0.6, 1.4]))))
    if new in numbers(s):
        return None
    return s[:m.start()] + new + s[m.end():]


def perturb_negation(s: str, rng: random.Random) -> str | None:
    if has_negation(s):  # inserting "not" into an already-negated sentence gives an ambiguous label
        return None
    m = AUX.search(s)
    if m and not re.match(r"\s+not\b", s[m.end():]):
        return s[:m.end()] + " not" + s[m.end():]
    return None


def perturb_flip(s: str, rng: random.Random) -> str | None:
    pairs = [(a, b) for a, b in FLIPS] + [(b, a) for a, b in FLIPS]
    rng.shuffle(pairs)
    for a, b in pairs:
        m = re.search(rf"\b{a}\b", s, re.I)
        if m:
            return s[:m.start()] + _keep_case(m.group(), b) + s[m.end():]
    return None


def perturb_hedge(s: str, rng: random.Random) -> str | None:
    for pat, rep in HEDGES:
        m = re.search(pat, s, re.I)
        if m:
            return s[:m.start()] + _keep_case(m.group(), rep) + s[m.end():]
    return None


def perturb_qualifier(s: str, rng: random.Random) -> str | None:
    for pat in QUALIFIERS:
        if re.search(pat, s, re.I):
            out = re.sub(r"\s{2,}", " ", re.sub(pat, "", s, count=1, flags=re.I)).replace(" ,", ",").replace(" .", ".")
            return out.strip()
    return None


def make_term_swapper(corpus):
    df: dict[str, set[str]] = defaultdict(set)
    for p in corpus.passages:
        for t in set(tokens(p.text)):
            if len(t) >= 5 and not t[0].isdigit():
                df[t].add(p.doc)

    def swap(s: str, rng: random.Random, passage) -> str | None:
        ptoks = set(tokens(passage.text))
        cands = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{4,}", s) if w.lower() in df and len(df[w.lower()]) <= 3]
        if not cands:
            return None
        w = rng.choice(cands)
        pool = sorted(t for t, docs in df.items() if passage.doc not in docs and t not in ptoks)
        if not pool:
            return None
        return re.sub(rf"\b{re.escape(w)}\b", rng.choice(pool), s, count=1)
    return swap


SYS_PARA = ("Paraphrase the sentence faithfully: keep every fact, number, unit and qualifier, change the wording. "
            "Return only the paraphrase.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--n", type=int, default=200, help="sentences sampled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm-config", help="YAML whose `llm:` block configures the LLM verifier")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    corpus = load_corpus(a.corpus)
    pool = [(s, p) for p in corpus.passages for s in sentences(p.text) if 8 <= len(s.split()) <= 60]
    rng.shuffle(pool)
    pool = pool[:a.n]
    swap = make_term_swapper(corpus)
    perturbers = {"number_change": perturb_number, "negation": perturb_negation, "direction_flip": perturb_flip,
                  "hedge_removal": perturb_hedge, "qualifier_drop": perturb_qualifier}
    llm = None
    if a.llm_config:
        cfg = load_yaml(a.llm_config)
        prices = load_yaml(cfg["prices"]).get("models", {}) if isinstance(cfg.get("prices"), str) else {}
        llm = build_llm(cfg.get("llm"), prices=prices)
    items = []  # (kind, claim_text, passage, original)
    for s, p in pool:
        items.append(("original", s, p, s))
        for kind, f in perturbers.items():
            out = f(s, rng)
            if out and out != s:
                items.append((kind, out, p, s))
        out = swap(s, rng, p)
        if out and out != s:
            items.append(("term_swap", out, p, s))
        if llm:
            para = llm.complete(SYS_PARA, s, 200, tag="paraphrase").strip()
            if para and para != s:
                items.append(("paraphrase", para, p, s))

    verifiers = {
        "lexical_v1 (numbers only)": (None, VerifierConfig(mode="lexical", negation_check=False)),
        "lexical_v2 (+negation)": (None, VerifierConfig(mode="lexical")),
    }
    if llm:
        verifiers[f"llm ({llm.model}) + number filter"] = (llm, VerifierConfig(mode="llm"))
    accepted: dict[str, dict[str, list[int]]] = {v: defaultdict(list) for v in verifiers}
    out = run_dir(a.out, "verifier_stress")
    with (out / "examples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "claim", "original", "passage_id", *verifiers])
        for kind, text, p, orig in items:
            row = [kind, text, orig, p.id]
            for vname, (vllm, vcfg) in verifiers.items():
                c = verify(Claim(text, p.id), p, vllm, vcfg)
                ok = int(c.verdict == "supported")
                accepted[vname][kind].append(ok)
                row.append(ok)
            w.writerow(row)
    kinds = ["original", "paraphrase", "number_change", "negation", "direction_flip", "hedge_removal",
             "qualifier_drop", "term_swap"]
    kinds = [k for k in kinds if any(accepted[v][k] for v in verifiers)]
    summary = {v: {k: {"accepted": sum(x), "n": len(x), "rate": sum(x) / len(x), "ci": wilson(sum(x), len(x))}
                   for k, x in d.items() if x} for v, d in accepted.items()}
    write_json(out / "summary.json", {"corpus": corpus.fingerprint, "n_sentences": len(pool), "seed": a.seed,
                                      "results": summary})
    head = "| Claim type | should be | n | " + " | ".join(verifiers) + " |"
    lines = [f"# Verifier stress test ({len(pool)} sentences from `{a.corpus}`, seed {a.seed})", "",
             "Accept rate % [Wilson 95% CI]. Higher is better for originals/paraphrases; for every other row the "
             "number is a **false-accept rate** (lower is better).", "", head, "|---|---|---|" + "---|" * len(verifiers)]
    for k in kinds:
        n = len(next(iter(accepted.values()))[k])
        cells = []
        for v in verifiers:
            r = summary[v].get(k)
            cells.append("-" if not r else f"{r['rate'] * 100:.0f} [{r['ci'][0] * 100:.0f}, {r['ci'][1] * 100:.0f}]")
        lines.append(f"| {k} | {'accepted' if k in ('original', 'paraphrase') else 'rejected'} | {n} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    print(md)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
