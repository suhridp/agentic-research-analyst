"""Config-driven experiment runner: every system in the config answers every question; results,
traces, provenance and a markdown summary go to results/<name>/<UTC timestamp>/.

    python -m eval.run configs/smoke.yaml                     # offline, synthetic, seconds
    python -m eval.run configs/main.yaml --split dev          # real corpus, LLM mode
    python -m eval.run configs/main.yaml --split test --confirm-test   # once, at the very end

Robustness: a failing question is recorded and the run continues; hitting the spend cap stops the run
cleanly and still writes partial results marked incomplete.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ara.corpus import load_corpus
from ara.pipeline import build_system
from common.config import environment, load_yaml, run_dir, write_json
from common.llm import BudgetExceeded, UsageMeter, build_llm
from common.tracing import Tracer

from .judge import LexicalJudge, LLMJudge
from .metrics import aggregate, compare, score_question
from .questions import load_questions, resolve_gold
from .report import render_summary


def _prices(cfg: dict) -> dict:
    p = cfg.get("prices")
    if isinstance(p, str):
        return (load_yaml(p) or {}).get("models", {}) if Path(p).exists() else {}
    return p or {}


def _systems(cfg: dict) -> list[dict]:
    defaults = cfg.get("defaults", {})
    out = []
    for s in cfg["systems"]:
        merged = {**defaults, **s}
        if "retriever" in defaults and "retriever" in s:
            merged["retriever"] = {**defaults["retriever"], **s["retriever"]}
        out.append(merged)
    names = [s["name"] for s in out]
    if len(set(names)) != len(names):
        raise ValueError("system names must be unique")
    return out


def guard_test_split(split: str, confirmed: bool, root: Path, name: str) -> None:
    if split != "test":
        return
    if not confirmed:
        sys.exit("Refusing to run on the held-out test split without --confirm-test. "
                 "Tune on dev; touch test once, at the end.")
    log = root / "TEST_RUNS.log"
    prior = log.read_text().count("\n") if log.exists() else 0
    if prior:
        print(f"WARNING: the test split has been evaluated {prior} time(s) before (see {log}). "
              "Report every test run, not the best one.", file=sys.stderr)
    root.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{name}\n")


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--split", help="override config split: dev | test | all")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--limit", type=int, help="first N questions only (debugging)")
    ap.add_argument("--systems", help="comma-separated subset of system names")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    cfg = load_yaml(a.config)
    split = a.split or cfg.get("split", "dev")
    root = Path(a.out)
    guard_test_split(split, a.confirm_test, root, cfg["name"])

    ccfg = cfg.get("corpus", {"path": "data/corpus"})
    corpus = load_corpus(ccfg["path"], ccfg.get("max_words", 160), ccfg.get("overlap", 1))
    questions = load_questions(cfg["questions"], split)
    if a.limit:
        questions = questions[:a.limit]
    if not questions:
        sys.exit(f"no questions for split={split}")
    gold, unresolved = {}, []
    for q in questions:
        gold[q.id], miss = resolve_gold(q, corpus)
        unresolved += miss
    if unresolved:
        print(f"WARNING: {len(unresolved)} evidence quote(s) not found in corpus (run `python -m eval.validate`):",
              *unresolved[:10], sep="\n  ", file=sys.stderr)

    prices = _prices(cfg)
    llm = build_llm(cfg.get("llm"), prices=prices)
    jcfg = cfg.get("judge", "lexical")
    judge = LexicalJudge() if jcfg in ("lexical", None) else LLMJudge(build_llm(jcfg, prices=prices))
    if llm and isinstance(judge, LLMJudge) and judge.llm.model == llm.model:
        warnings.warn("judge uses the same model as the system under test; prefer a different model", stacklevel=1)

    sys_cfgs = _systems(cfg)
    if a.systems:
        keep = set(a.systems.split(","))
        sys_cfgs = [s for s in sys_cfgs if s["name"] in keep]
    out = run_dir(root, cfg["name"])
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "split": split}, sort_keys=False))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        systems = [build_system(s, corpus, llm, cache_dir=cfg.get("index_cache", ".index_cache")) for s in sys_cfgs]
    for w in caught:
        print(f"NOTE: {w.message}", file=sys.stderr)
    if cfg.get("require_neural"):
        for s in systems:
            if '"neural": false' in json.dumps(s.describe()["retriever_backend"]):
                sys.exit(f"{s.cfg.name}: require_neural is set but a non-neural fallback was loaded "
                         "(pip install -r requirements-ml.txt)")

    meta = {"name": cfg["name"], "split": split, "n_questions": len(questions),
            "corpus": {"path": ccfg["path"], "documents": len(corpus.documents), "passages": len(corpus.passages),
                       "fingerprint": corpus.fingerprint},
            "judge": judge.describe(), "systems": {s.cfg.name: s.describe() for s in systems},
            "unresolved_quotes": unresolved, "environment": environment()}
    write_json(out / "meta.json", meta)

    workers = a.workers or cfg.get("workers", 1 if llm is None else 4)
    rows_by_system: dict[str, list[dict]] = {}
    incomplete = None
    t_start = time.perf_counter()
    for system in systems:
        name = system.cfg.name

        def run_one(q, system=system, name=name):
            tr, err, row = Tracer(f"{q.id}/{name}"), None, None
            with tr.activate():
                with UsageMeter() as meter:  # system cost only; judge calls happen outside the meter
                    try:
                        rep = system.answer(q.question)
                    except BudgetExceeded:
                        raise
                    except Exception as e:  # keep going; the error is part of the results
                        err = f"{type(e).__name__}: {e}"
                if err is None:
                    try:
                        row = score_question(q, rep, gold[q.id], corpus, judge)
                        row["report"] = rep.to_dict()
                        row["gold_groups"] = [sorted(g) for g in gold[q.id]]
                    except BudgetExceeded:
                        raise
                    except Exception as e:
                        err = f"scoring failed: {type(e).__name__}: {e}"
            if err is not None:
                row = {"qid": q.id, "type": q.type, "answerable": q.answerable, "error": err}
            u = meter.usage
            row.update({"system": name, "llm_calls": u.calls, "cache_hits": u.cache_hits,
                        "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                        "cost_usd": round(u.cost_usd, 6)})
            tr.dump(out / "traces" / f"{q.id}__{name}.jsonl")
            return row

        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                rows = list(ex.map(run_one, questions))
        except BudgetExceeded as e:
            incomplete = str(e)
            print(f"STOPPED: {e}", file=sys.stderr)
            break
        rows_by_system[name] = rows
        print(f"{name}: {len(rows)} questions, {sum('error' in r for r in rows)} errors")

    with (out / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for rows in rows_by_system.values():
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summary = {"meta": {k: meta[k] for k in ("name", "split", "n_questions", "corpus", "judge")},
               "incomplete": incomplete, "wall_time_s": round(time.perf_counter() - t_start, 2),
               "systems": {n: aggregate(r) for n, r in rows_by_system.items()},
               "comparisons": compare(rows_by_system, cfg.get("reference", next(iter(rows_by_system), ""))),
               "reference": cfg.get("reference"), "llm_usage_total": llm.usage.__dict__ if llm else None,
               "judge_usage_total": judge.llm.usage.__dict__ if isinstance(judge, LLMJudge) else None}
    write_json(out / "summary.json", summary)
    md = render_summary(summary, meta)
    (out / "summary.md").write_text(md, encoding="utf-8")
    (root / cfg["name"] / "LATEST").write_text(out.name + "\n")
    print(md)
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    main()
