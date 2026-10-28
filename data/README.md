# Data

| Path | What | Committed? |
|---|---|---|
| `synthetic/` | 6 short **synthetic** documents written for tests and the smoke run. Not real papers, and not evidence for anything | yes |
| `synthetic_questions.jsonl` | 14 questions over the synthetic fixture | yes |
| `papers/manifest.csv` | the curated arXiv papers (pinned versions), with licences | yes |
| `papers/SOURCES.csv` | provenance written by `fetch_arxiv.py download` | yes |
| `papers/*.pdf`, `corpus/*.json` | downloaded PDFs and extracted text, rebuilt with `make corpus` | no (licences vary) |
| `questions.jsonl` | the hand-written evaluation set (see `docs/EVALUATION.md`) | yes |
| `questions_draft.example.jsonl` | format example for drafting questions | yes |
| `labels/` | human labels for judge validation | yes |
