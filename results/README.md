# Results

Each run writes `results/<experiment>/<UTC timestamp>/` containing:

| File | Contents |
|---|---|
| `config.yaml` | the exact config that ran (with the split) |
| `meta.json` | git commit, package versions, corpus fingerprint, judge, and the retrieval backends that actually ran |
| `per_question.jsonl` | one row per (question, system): metrics, the full report, sentence-level judgements, cost |
| `traces/` | one JSONL trace per (question, system): every step and LLM call |
| `summary.json` / `summary.md` | aggregates with 95% CIs, paired comparisons, per-type breakdown |
| `failures.md` | written by `python -m eval.failures <run>` |

Run outputs are git-ignored. Copy the runs you report into `results/reported/` and commit them.
`TEST_RUNS.log` records every evaluation on the held-out test split.
