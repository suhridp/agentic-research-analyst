# Design notes

Why each component is the way it is. The plan's rule is that you should be able to explain every
design decision without notes; this file is the notes. Each section ends with the question an
interviewer is likely to ask.

## 1. Shared foundation (`common/`)

**LLM wrapper.** Three providers behind one interface: Anthropic, any OpenAI-compatible endpoint
(OpenAI, or a local open-weight model through Ollama or vLLM), and `ScriptedLLM`, a deterministic
fake used in tests. Every call goes through `BaseLLM.complete`, which adds:

- a **disk cache** keyed by provider, model, temperature, system prompt, prompt and max_tokens.
  Reruns are free, and results stay reproducible even if the provider changes behaviour later.
  Writes are atomic (temp file plus `os.replace`), so parallel workers never read a half-written entry.
- **retries** with exponential backoff and jitter on 429/5xx/529 and network errors, honouring
  `Retry-After`. 4xx errors are not retried, because retrying a bad request wastes money.
- **cost accounting and a spend cap.** Prices live in `configs/prices.yaml`, not in code, because
  they change. The cap is checked before every call; if it trips, the runner writes partial results
  marked incomplete.
- **`complete_json`**, which makes exactly one repair attempt and then returns `None`. Callers must
  handle failure explicitly, and in the verifier, failure means *reject* (fail closed).

*Q: Why not use the provider SDKs?* Stdlib HTTP keeps the dependency surface small and makes the
retry and cache behaviour explicit and testable (see `tests/test_llm_stats.py`, which fakes HTTP 429
and 400 responses). Swapping in an SDK would mean changing one method per provider.

**Per-question cost with a shared client.** `UsageMeter` is thread-local, so when questions run in
parallel on one client, each question's cost is still attributed correctly. A test checks this
with four threads.

**Tracing.** `Tracer.activate()` makes a tracer current on the thread. Components deep in the call
stack, including the LLM client, log to it without it being passed around. Each question/system
pair gets one JSONL file of steps, verdicts and LLM calls (prompt and output included).

**Statistics.**
- The bootstrap resamples **questions**, not claims. Claims from one answer are correlated, so
  treating them as independent would make intervals too narrow.
- System comparisons are **paired**: same questions, difference per question. That is much more
  sensitive than checking whether two CIs overlap.
- The p-value comes from a sign-flip permutation test (exact for n ≤ 16), with Holm adjustment
  across the systems compared against the reference.

*Q: Why Holm rather than Bonferroni?* It controls the same family-wise error rate and is uniformly
more powerful.

## 2. Corpus and chunking (`ara/corpus.py`, `scripts/`)

- **Ingestion is column-aware.** Plain `pdftotext` interleaves the two columns of IEEE-style papers
  line by line when the gutter is narrow. `scripts/ingest.py` finds the gutter from word coordinates
  (pdfplumber), reads the left column then the right, and treats full-width rows (titles, wide
  tables) as band breaks. `tests/fixtures/two_column_paper.pdf` is a regression test.
- **Cleaning:** running headers and footers (lines repeated on ≥30% of pages), page numbers, the
  arXiv margin stamp, numeric citation markers and everything after "References" are removed,
  because citation lists would pollute retrieval.
- **Chunking** packs whole sentences up to `max_words` (160) with a one-sentence overlap and never
  crosses a section boundary. The sentence splitter survives "e.g.", "et al.", "Fig. 3" and decimals.
- **Retrieval sees the header, verification doesn't.** `index_text` prepends title and section, so
  a passage that says only "it reduced failures by 18%" is still findable by topic. The verifier
  sees only `text`, so a title can't lend support to a claim.
- **Gold labels are quotes.** Passage IDs change whenever chunking changes; quotes don't.
  `Corpus.find_quote` resolves each evidence quote to every passage that contains it, including
  quotes that straddle a chunk boundary.

*Q: What happens to your labels if you change the chunk size?* Nothing. They are re-resolved at eval
time, and `make validate` fails if a quote no longer resolves.

## 3. Retrieval (`ara/retrieval/`)

- **BM25** with an inverted index, so a query only touches the postings of its terms.
- **Dense** uses `BAAI/bge-small-en-v1.5` by default, with its query instruction prefix applied.
  Passage embeddings are cached on disk by corpus fingerprint plus model. FAISS is used when
  installed, otherwise exact numpy search (fine at this scale).
- **Hybrid** uses Reciprocal Rank Fusion, which fuses ranks rather than scores. BM25 scores are
  unbounded and cosine scores aren't, so score fusion would need calibration that RRF avoids.
- **Reranking** uses a cross-encoder (`ms-marco-MiniLM-L-6-v2`) over the top 30 first-stage hits.
  An LLM reranker is also available.
- **Fallbacks are labelled.** Without sentence-transformers, `auto` falls back to LSA (TF-IDF plus
  SVD) and a lexical reranker, so the code paths still run in CI. `describe()` reports what actually
  ran, every results table prints it, and `require_neural: true` turns a fallback into an error.

*Q: What if hybrid doesn't beat BM25?* That is a plausible outcome on a technical corpus full of
acronyms, and the retrieval eval reports it with a paired test either way. The plan's acceptance
criterion explicitly allows a documented negative result.

## 4. Agents (`ara/agents.py`) and pipeline (`ara/pipeline.py`)

| Agent | LLM mode | Offline heuristic | Failure handling |
|---|---|---|---|
| Planner | 1-4 sub-questions, JSON | split on conjunctions | invalid JSON → repair once → fall back to `[question]` |
| Extractor | atomic claims, one passage id each | best-overlap sentence per passage | claims citing unknown passage ids are dropped |
| Verifier | number filter, then LLM verdict + quote | coverage ≥ 0.8 + number + negation checks | unparseable output or a quote not in the passage → reject |
| Synthesizer | prose with `[doc#i]` after every sentence | bullet list of claims | |
| Critic | deterministic per-sentence check (+ LLM completeness check) | same | drops failing sentences; falls back to the claim list if none survive |

- **Retry loop.** Only sub-questions with zero verified claims are retried, with an LLM-rewritten
  query and double the k. Passages already judged aren't re-extracted.
- **Abstention** is a first-class outcome, not an error: no verified claims means a fixed abstention
  message. Partial answers list their **evidence gaps** (LLM-planned sub-questions that found nothing).
- **Ablations are config, not code.** `planner: false`, `verifier: none`, `critic: false` and the
  retrieval settings are fields of `SystemConfig`. Unknown keys raise an error, so a typo can't
  silently run the wrong system.
- **Prompts are versioned** (`PROMPT_VERSION`), and the version is recorded with every result.

*Q: Isn't the verifier just another LLM that can be wrong?* Yes. That is why (1) it sees one
passage only, (2) it must quote evidence that is then checked by string matching, (3) the cheap
deterministic number filter runs first, and (4) its false-accept rate is measured directly by the
stress test rather than assumed.

*Q: Why grade answer sentences rather than the extracted claims?* Because the synthesizer can
paraphrase or add things after verification. Users read the answer, so the answer is what's graded,
and the same rule applies to the baseline.

## 5. Evaluation (`eval/`)

- `run.py` runs every system on every question in parallel. Each question records an error instead
  of crashing the run. Outputs: resolved config, provenance (git commit, package versions, backends
  that actually ran, corpus fingerprint), per-question rows, traces, `summary.json` and `summary.md`.
- **The judge is separate from the verifier,** and the runner warns if both use the same model.
  Judge validity is measured, not assumed (`judge_agreement.py`).
- **Test-split discipline:** `--confirm-test` is required and every run is appended to
  `results/TEST_RUNS.log`.
- **System cost excludes judge cost.** Scoring happens outside the system's usage meter.

## 6. Decision log

| Decision | Choice | Why / when to revisit |
|---|---|---|
| Orchestration | Plain Python | The plan's rule is to port to LangGraph once the pipeline stabilises, and to list it on a résumé only after that |
| Default LLM | `claude-sonnet-5-5`; judge `claude-opus-5-5` | Judge differs from the system under test. A cheap config (`dev_cheap.yaml`) covers prompt iteration |
| Open-weight model | via `openai_compat` provider | Lets the main result be replicated on a local model |
| Temperature | 0 plus cache | Reproducibility. Trials with seeds matter more for Project 2 |
| Corpus redistribution | Commit the manifest, never the PDFs or text | Licences vary. Anyone can rebuild the identical corpus from pinned arXiv versions |
