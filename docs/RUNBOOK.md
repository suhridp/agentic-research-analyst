# Runbook: from this repo to finished results

The ordered steps that take the project from "In development" to done. Time estimates assume about
15 hours per week.

## 0. Setup (30 min)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-ml.txt -r requirements-serve.txt
make test && make smoke          # should pass offline
export ANTHROPIC_API_KEY=...      # or configure openai_compat for a local model
export ARXIV_CONTACT=you@example.com
```

Fill in `configs/prices.yaml` from the provider's pricing page, so that cost-per-question and the
spend cap work in dollars. Without prices, only the token cap (`max_tokens`) applies.

## 1. Corpus (milestone 1, ~1 day)

```bash
python scripts/fetch_arxiv.py search --max-per-query 40      # ~10 min (arXiv rate limit)
```

Open `data/papers/candidates.csv`. Copy 40-60 relevant rows to `data/papers/manifest.csv` and set
`include=1`. Prefer `redistributable=1`, cover all sub-topics (prediction, conditional handover,
attacks on handover, rogue base stations, DoS, slicing security), and note why each was chosen.

```bash
make corpus                                                    # download + ingest
```

Check `data/corpus/_ingest_log.csv`. Open 3-4 flagged documents and one unflagged one and read the
JSON text. If a paper is mostly tables or a scan, drop it from the manifest.
**Acceptance:** 40+ documents, licence log (`data/papers/SOURCES.csv`), clean text.

## 2. Question set (milestone 2, 3-5 days, the slowest part)

Follow `docs/EVALUATION.md` section 1. Draft questions into `data/questions_draft.jsonl`, then:

```bash
python -m eval.label_helper --draft data/questions_draft.jsonl   # writes data/label_review.md
make validate
```

Write 40 first, run everything end to end (`dev_cheap.yaml`), then grow to 100.
**Acceptance:** `make validate` passes; 100 questions, 20 test, ~15 unanswerable.

## 3. Retrieval (milestone 3, half a day)

```bash
make retrieval
```

Copy the run's `summary.md` into `results/reported/retrieval.md`. Pick the setup with the best
Recall@5 on dev and put it in `configs/main.yaml` `defaults.retriever`.
**Acceptance:** hybrid + reranker beats BM25 on Recall@5 with a paired CI, or the negative result is documented.

## 4. Verifier (milestone 4, 1 day)

```bash
make stress LLM=configs/main.yaml           # adds the LLM verifier column
```

This is the first real result: does the LLM verifier close the gap on direction flips, hedges and
qualifiers, and how often does it reject faithful paraphrases? Spot-check 20 rows of `examples.csv`
to confirm the automatic labels are right.

## 5. Main comparison (milestone 5, 1-2 days)

```bash
python -m eval.run configs/dev_cheap.yaml   # iterate cheaply first
make eval                                    # then the real models, dev split
python -m eval.judge_agreement export results/main_comparison/$(cat results/main_comparison/LATEST)
# label 30 items (about 1 hour)
python -m eval.judge_agreement score
make failures                                # hand-code with the taxonomy
make eval-test                               # once
```

Rough cost: (6 systems × 100 questions × ~8 LLM calls) plus judge calls. Run `dev_cheap` and
`--limit 10` first to measure the real cost per question before the full run. Caching makes
reruns free.

## 6. Write-up (milestone 6, 2 days)

Replace "Results so far" in the README with:
1. the main table (unsupported rate, completeness, gold cited, abstention, false abstention,
   latency, cost) with CIs, and the paired-difference table;
2. the retrieval table;
3. the verifier stress table with the LLM column;
4. judge agreement (κ, false-accept rate);
5. failure analysis: category counts and 2-3 quoted examples;
6. limitations, updated with what you learned.

Move "Status: In development" to "Done" only when every box in the plan's section 9 is ticked.

## Résumé wording (only after step 6)

Template. Fill the brackets only with measured numbers:

> Built an agentic research assistant (hybrid retrieval, claim-level verification, abstention) over
> [N] arXiv papers; on a [100]-question hand-labelled benchmark, verification cut unsupported
> statements from [X]% to [Y]% (paired 95% CI [a, b]) at a cost of [Z] pp completeness and [k]×
> latency; LLM judge validated against human labels (κ = [κ]).

Until then: "Agentic research assistant with evidence verification (in development): built the
pipeline and evaluation harness with paired statistical testing; real-corpus evaluation in progress."
