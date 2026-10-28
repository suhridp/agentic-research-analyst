# Evaluation protocol

The rules for building the question set, labelling, running and reporting. Decide them before
looking at results; changing them afterwards is how evaluations get fooled.

## 1. Question set (`data/questions.jsonl`)

Target: about **100 questions**, of which **20 are held out** (`"split": "test"`) and **about 15% are unanswerable**.
Start with 40 and grow (gold labelling takes a few days of careful work).

| Type | Share | What it tests | Example shape |
|---|---|---|---|
| `factual` | 25% | single-fact lookup | "What does X assume about Y?" |
| `numeric` | 25% | exact numbers, units, settings | "By how much did X reduce Y in setting Z?" |
| `comparison` | 20% | multi-document synthesis, disagreement | "How do A and B differ in their reported Y?" |
| `gap` | 15% | knowing what the collection doesn't cover | "Which of X, Y, Z is not evaluated in any paper?" |
| `unanswerable` | 15% | abstention | in-domain and plausible, but not in the corpus |

Line format:

```json
{"id": "q017", "type": "numeric", "split": "dev",
 "question": "By how much did the LSTM predictor reduce handover failures in the urban scenario?",
 "evidence": [{"doc": "2301.01234v2", "quote": "reduced handover failures by 18% compared to"}],
 "facts": ["18% fewer handover failures than the A3-event baseline (urban scenario)"],
 "reference": "optional reference answer"}
```

Rules:

1. **Write the question from the paper, then find all the evidence.** Search the PDF text yourself
   (`grep -i` in `data/corpus/*.json`), not only `label_helper.py`'s retrieved passages. If gold only
   ever comes from what the retriever found, Recall is inflated.
2. **Quotes are verbatim**, at least 8 characters, ideally one clause, copied from the extracted text
   (not the PDF, since hyphenation can differ). `make validate` must pass.
3. **Facts are atomic**: one checkable statement each, with the number and unit. Completeness is the
   fraction stated in the answer.
4. **Unanswerable questions must be tempting**: same vocabulary as the corpus, about a detail the
   papers don't report (energy cost, a band that wasn't used, a dataset that doesn't exist).
   "Capital of France" tests nothing.
5. **Use no particular system's output to write questions**, and don't edit questions after seeing results.
6. **Split before running anything:** pick the 20 test questions at random (stratified by type) on day one.

## 2. Judge validation (milestone 4)

Judge-based numbers are reported only alongside judge agreement:

```bash
python -m eval.judge_agreement export results/main_comparison/<run> --n 30
# fill human_label (1/0) in data/labels/judge_sample.csv
python -m eval.judge_agreement score
```

Labelling rules for "supported" (1):
- Every fact, number, unit, comparison direction and qualifier in the sentence is stated in, or
  directly entailed by, the evidence.
- A hedged source ("may", "in simulation", "on a synthetic dataset") does **not** support an unhedged sentence.
- Correct but too general still counts as supported. Missing detail is completeness, not support.
- When unsure, label 0 and write a note. Count the notes.

Report the judge's accuracy, Cohen's κ and its **false-accept rate** next to the lexical proxy's. If
κ < 0.6, fix the judge prompt on *other* items and re-measure. Don't tune on the 30 labelled items.

## 3. Running

1. `make retrieval`: choose the retrieval setup on **dev**. Recall@5 is the acceptance metric.
2. `make stress LLM=configs/main.yaml`: verifier false-accept rates, including the LLM verifier.
3. `python -m eval.run configs/dev_cheap.yaml`: iterate on prompts cheaply. Bump `PROMPT_VERSION`
   when a prompt changes.
4. `make eval`: the main comparison on dev. Freeze the configuration after this.
5. Judge agreement (section 2).
6. `make eval-test` **once**. If you have to rerun (a bug, say), report both runs. `TEST_RUNS.log` records them.

## 4. Reporting rules

- Every rate has a 95% CI, and every system difference is paired, with a Holm-adjusted p.
- Always show **completeness next to the unsupported rate**, and **false abstention next to
  abstention**. A system that abstains on everything has a 0% unsupported rate and is useless.
- Report cost and latency per question for every system.
- State n. With 20 test questions, a 10-point difference is often not significant. Say so rather
  than calling it a trend.
- A null result (verification doesn't help much) is a valid finding when measured properly. Report it.
- Name the data: real corpus, number of papers, number of questions, and who wrote the questions.

## 5. Failure taxonomy (for `failures.md`)

Code each case with one primary category. Count them in the write-up and include 2-3 examples.

| Code | Category | Typical cause |
|---|---|---|
| R1 | Gold not retrieved | vocabulary mismatch, acronym, table-only fact |
| R2 | Retrieved but not extracted | extractor skipped it; claim limit |
| V1 | Verifier false reject | paraphrase, unit conversion, number written as words |
| V2 | Verifier false accept | overreach, dropped hedge, wrong direction |
| S1 | Synthesis drift | synthesizer merged claims into something stronger |
| C1 | Critic over-pruning | correct sentence dropped (e.g. computed difference "a 13-point drop") |
| A1 | Missed abstention | tangential evidence treated as an answer |
| A2 | False abstention | evidence exists but is spread across passages |
| E1 | Extraction/ingest error | broken PDF text, table, equation |
| J1 | Judge error | judge label wrong on inspection |
