"""Validate the LLM judge against hand labels (milestone 4). Without this, judge-based numbers
aren't trustworthy.

  1. Export a blind sample from a finished run (judge labels go to a separate key file so they can't
     anchor you while labelling; the sample is stratified so both labels are represented):
        python -m eval.judge_agreement export results/main_comparison/<stamp> --n 30
  2. Open data/labels/judge_sample.csv and fill `human_label` with 1 (supported) or 0 (not supported),
     reading only `sentence` and `evidence`. See docs/EVALUATION.md for the labelling rules.
  3. Score:
        python -m eval.judge_agreement score
     -> accuracy, Cohen's kappa and confusion matrix for the LLM judge AND the lexical proxy.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from ara.corpus import load_corpus
from common.config import load_yaml, write_json
from common.stats import cohen_kappa, wilson

from .judge import LexicalJudge

LABEL_DIR = Path("data/labels")


def export(run: Path, n: int, seed: int) -> None:
    cfg = load_yaml(run / "config.yaml")
    corpus = load_corpus(cfg["corpus"]["path"], cfg["corpus"].get("max_words", 160), cfg["corpus"].get("overlap", 1))
    items = []
    for line in (run / "per_question.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for j in r.get("sentence_judgements", []):
            if all(i in corpus.by_id for i in j["cited"]):
                items.append({"system": r["system"], "qid": r["qid"], "sentence": j["sentence"],
                              "cited": " ".join(j["cited"]),
                              "evidence": "\n---\n".join(corpus.by_id[i].text for i in j["cited"]),
                              "judge": int(j["supported"])})
    rng = random.Random(seed)
    pos = [x for x in items if x["judge"]]
    neg = [x for x in items if not x["judge"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    take = neg[:n // 2] + pos[:n - min(len(neg), n // 2)]
    take = (take + [x for x in pos + neg if x not in take])[:n]
    rng.shuffle(take)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    with (LABEL_DIR / "judge_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "sentence", "evidence", "human_label", "notes"])
        for i, x in enumerate(take):
            w.writerow([i, x["sentence"], x["evidence"], "", ""])
    write_json(LABEL_DIR / "judge_sample.key.json",
               {"run": str(run), "judge": load_yaml(run / "config.yaml").get("judge"),
                "items": [{"item": i, **{k: x[k] for k in ("system", "qid", "cited", "judge")}} for i, x in enumerate(take)]})
    print(f"wrote {LABEL_DIR / 'judge_sample.csv'} ({len(take)} items; {sum(x['judge'] for x in take)} judged supported)")


def score() -> dict:
    key = json.loads((LABEL_DIR / "judge_sample.key.json").read_text())["items"]
    rows = list(csv.DictReader((LABEL_DIR / "judge_sample.csv").open(encoding="utf-8")))
    lex = LexicalJudge()
    human, llm_j, lex_j = [], [], []
    for r, k in zip(rows, key):
        if r["human_label"].strip() not in ("0", "1"):
            continue
        human.append(int(r["human_label"]))
        llm_j.append(k["judge"])
        lex_j.append(int(lex.supported(r["sentence"], r["evidence"].split("\n---\n"))))
    if not human:
        raise SystemExit("no human labels yet: fill the human_label column with 0/1")

    def stats(pred):
        agree = sum(h == p for h, p in zip(human, pred))
        tp = sum(h and p for h, p in zip(human, pred))
        fp = sum((not h) and p for h, p in zip(human, pred))
        fn = sum(h and (not p) for h, p in zip(human, pred))
        tn = len(human) - tp - fp - fn
        return {"n": len(human), "accuracy": agree / len(human), "accuracy_ci": wilson(agree, len(human)),
                "kappa": cohen_kappa(human, pred),
                "confusion": {"human1_judge1": tp, "human0_judge1": fp, "human1_judge0": fn, "human0_judge0": tn},
                "false_accept_rate": fp / max(fp + tn, 1)}
    res = {"llm_judge": stats(llm_j), "lexical_proxy": stats(lex_j)}
    write_json(LABEL_DIR / "judge_agreement.json", res)
    for name, s in res.items():
        print(f"{name:14s} n={s['n']} accuracy={s['accuracy']:.2f} "
              f"[{s['accuracy_ci'][0]:.2f}, {s['accuracy_ci'][1]:.2f}] kappa={s['kappa']:.2f} "
              f"false-accept={s['false_accept_rate']:.2f} {s['confusion']}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("run", type=Path)
    e.add_argument("--n", type=int, default=30)
    e.add_argument("--seed", type=int, default=0)
    sub.add_parser("score")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        export(a.run, a.n, a.seed)
    else:
        score()


if __name__ == "__main__":
    main()
