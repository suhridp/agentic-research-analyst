"""Retrieval-only comparison (milestone 3): no LLM needed, so it is cheap to iterate on.

    python -m eval.retrieval_eval configs/retrieval.yaml [--split dev]

For each retriever config: Recall@k for each k in `ks` (fraction of gold evidence items with a matching
passage in the top k) and MRR@10, with bootstrap CIs, plus paired differences against the first
retriever listed (the baseline). Answerable questions only.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from ara.corpus import load_corpus
from ara.retrieval import build_retriever
from common.config import environment, load_yaml, run_dir, write_json
from common.stats import bootstrap_ci, holm, paired_bootstrap

from .questions import load_questions, resolve_gold
from .run import guard_test_split


def recall_at(ranked: list[str], groups: list[set[str]], k: int) -> float:
    top = set(ranked[:k])
    return sum(bool(g & top) for g in groups) / len(groups)


def mrr(ranked: list[str], groups: list[set[str]], k: int = 10) -> float:
    gold = set().union(*groups)
    for i, pid in enumerate(ranked[:k], 1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    split = a.split or cfg.get("split", "dev")
    guard_test_split(split, a.confirm_test, Path(a.out), cfg["name"])
    cc = cfg.get("corpus", {"path": "data/corpus"})
    corpus = load_corpus(cc["path"], cc.get("max_words", 160), cc.get("overlap", 1))
    qs = [q for q in load_questions(cfg["questions"], split) if q.answerable]
    gold = {q.id: resolve_gold(q, corpus)[0] for q in qs}
    qs = [q for q in qs if gold[q.id]]
    if not qs:
        sys.exit("no answerable questions with resolvable evidence")
    ks = cfg.get("ks", [1, 3, 5, 10])
    per: dict[str, dict[str, list[float]]] = {}
    backends = {}
    for rc in cfg["retrievers"]:
        name = rc["name"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = build_retriever({k: v for k, v in rc.items() if k != "name"}, corpus,
                                cache_dir=cfg.get("index_cache", ".index_cache"))
        for x in w:
            print(f"NOTE ({name}): {x.message}", file=sys.stderr)
        backends[name] = r.describe()
        per[name] = {f"recall@{k}": [] for k in ks} | {"mrr@10": []}
        for q in qs:
            ranked = [h.passage.id for h in r.search(q.question, max(ks + [10]))]
            for k in ks:
                per[name][f"recall@{k}"].append(recall_at(ranked, gold[q.id], k))
            per[name]["mrr@10"].append(mrr(ranked, gold[q.id]))
    base = cfg["retrievers"][0]["name"]
    summary = {"split": split, "n_questions": len(qs), "baseline": base,
               "retrievers": {n: {m: bootstrap_ci(v) for m, v in d.items()} for n, d in per.items()},
               "vs_baseline": {}}
    for m in per[base]:
        res = {n: paired_bootstrap(per[n][m], per[base][m]) for n in per if n != base}
        adj = holm({n: v["p"] for n, v in res.items()})
        summary["vs_baseline"][m] = {n: {**v, "p_holm": adj[n]} for n, v in res.items()}
    out = run_dir(a.out, cfg["name"])
    write_json(out / "summary.json", summary)
    write_json(out / "meta.json", {"backends": backends, "corpus_fingerprint": corpus.fingerprint,
                                   "environment": environment()})
    metrics = list(per[base])
    lines = [f"# {cfg['name']} ({split}, {len(qs)} answerable questions)", "",
             "| Retriever | Backend | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for n, d in summary["retrievers"].items():
        lines.append(f"| `{n}` | {_short(backends[n])} | " + " | ".join(
            f"{d[m]['mean'] * 100:.1f} [{d[m]['lo'] * 100:.1f}, {d[m]['hi'] * 100:.1f}]" for m in metrics) + " |")
    lines += ["", f"Paired difference vs `{base}` (pp, Holm-adjusted p):", "",
              "| Retriever | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for n in per:
        if n == base:
            continue
        lines.append(f"| `{n}` | " + " | ".join(
            f"{summary['vs_baseline'][m][n]['diff'] * 100:+.1f} (p={summary['vs_baseline'][m][n]['p_holm']:.3f})"
            for m in metrics) + " |")
    md = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(md)
    print(md)
    print(f"wrote {out}")
    return out


def _short(d: dict) -> str:
    if d["kind"] == "reranked":
        return f"{_short(d['first_stage'])} + {d['reranker']}"
    if d["kind"] == "hybrid_rrf":
        return "RRF(" + ", ".join(_short(c) for c in d["components"]) + ")"
    if d["kind"] == "dense":
        return d["encoder"]
    return d["kind"]


if __name__ == "__main__":
    main()
