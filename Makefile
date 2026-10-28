PY ?= python

.PHONY: install install-ml install-all test lint smoke stress retrieval validate eval eval-test failures api demo corpus

install:        ## core + dev tools (offline mode)
	$(PY) -m pip install -r requirements-dev.txt
install-ml:     ## neural retrieval backends (sentence-transformers, FAISS)
	$(PY) -m pip install -r requirements-ml.txt
install-all: install install-ml
	$(PY) -m pip install -r requirements-serve.txt

test:           ## offline test suite (~5 s, no API key)
	$(PY) -m pytest -q
lint:
	ruff check .
smoke:          ## every system end to end on the synthetic fixture (plumbing check, not a result)
	$(PY) -m eval.run configs/smoke.yaml

corpus:         ## download the curated papers in data/papers/manifest.csv and extract text
	$(PY) scripts/fetch_arxiv.py download
	$(PY) scripts/ingest.py
validate:       ## check the question set: schema, quotes resolve, type/split balance
	$(PY) -m eval.validate --questions data/questions.jsonl --corpus data/corpus
retrieval:      ## milestone 3: BM25 vs dense vs hybrid vs reranked (no LLM)
	$(PY) -m eval.retrieval_eval configs/retrieval.yaml
stress:         ## verifier false-accept rates on perturbed claims (add LLM=configs/main.yaml for the LLM verifier)
	$(PY) -m eval.verifier_stress --corpus data/corpus --n 200 $(if $(LLM),--llm-config $(LLM),)
eval:           ## milestone 5: all systems and ablations, dev split
	$(PY) -m eval.run configs/main.yaml
eval-test:      ## the held-out test split - run ONCE, at the end
	$(PY) -m eval.run configs/main.yaml --split test --confirm-test
failures:       ## failure cases from the latest main run, for hand-coding
	$(PY) -m eval.failures results/main_comparison/$$(cat results/main_comparison/LATEST)

api:
	uvicorn api.main:app --port 8000
demo:
	streamlit run app.py
