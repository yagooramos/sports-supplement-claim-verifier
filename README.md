# Sports Supplement Claim Verifier

## Overview

This repository is a conservative and auditable claim-verification system for sports-supplement claims.

The current main repo now combines:

- the validated deterministic baseline from the principal repository

The canonical schema remains centered on:

- `ingredient`
- `claim_type`
- `outcome_target`

## Current Functional Scope

The repository now supports:

1. text-only claim verification
2. image-only verification through OCR claim extraction
3. multimodal verification using text + image together
4. deterministic parsing, retrieval, and reasoning
5. bounded supervised ML prediction of `claim_type`
6. optional local LLM assistance through Ollama for parser fallback and explanation generation
7. offline genetic optimization of retrieval parameters, with fixed optimized settings loaded at runtime

The deterministic reasoning path is still the authoritative core.

## Current Status

Validated snapshot on April 21, 2026:

- retrieval benchmark: `20/20`
- retrieval MRR with optimized retriever: `0.975`
- reasoning benchmark: `16/16`
- Streamlit app: runnable
- OCR image flow: integrated
- ML classifier dataset and metrics: integrated
- local LLM support: optional and auto-detected
- genetic retriever optimization: integrated offline and versioned

## Architecture

### Deterministic Core

- `scripts/claim_parser_v1.py`
- `scripts/lexical_retriever_v1.py`
- `scripts/reasoning_v1.py`
- `scripts/pipeline.py`

### Retrieval Optimization

- `scripts/optimize_retriever_ga.py`
- versioned retriever artifact in `models/retriever_optimized_config.json`
- offline genetic search over `k1`, `b`, and field weights for:
  - `fragment_text`
  - `retrieval_keywords`
  - `ingredient`
  - `claim_type`
  - `outcome_target`

### Vision And OCR

- `scripts/ocr_claim_extractor.py`
- `scripts/vision_v1.py`
- image test asset in `data/test_images/`

### Machine Learning

- `scripts/claim_type_classifier.py`
- labeled dataset in `data/ml/claim_type_dataset.csv`
- reference metrics in `models/claim_type_metrics.json`

### Optional LLM Layer

- `scripts/llm_adapter.py`
- local Ollama integration only when available

### UI

- `app.py`

## Repository Layout

- `app.py`: main Streamlit interface
- `data/sources/`: canonical source tables
- `data/annotations/`: canonical evidence annotations
- `data/benchmarks/`: retrieval and reasoning benchmarks
- `data/config/`: parser rules
- `data/ml/`: claim-type classifier dataset
- `data/test_images/`: synthetic vision test asset
- `models/`: classifier metrics and retriever optimization artifacts
- `scripts/`: deterministic, OCR, ML, LLM, and pipeline modules
- `docs/`: technical notes and source inventory

## Data Snapshot

Current repository snapshot:

- `18` lexicon rows
- `10` matrix-scope rows
- `15` evidence fragments
- `20` retrieval benchmark queries
- `16` reasoning benchmark cases
- `345` claim-type classification examples

The canonical evidence corpus is still curated and compact.
The ML dataset is an auxiliary project dataset, not a replacement for the canonical reasoning corpus.
The retriever optimization artifact is also auxiliary: it improves ranking while keeping the deterministic verdict path intact.

## Run

Install requirements:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python -m streamlit run app.py
```

On Windows in this repo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Benchmarks

Run deterministic repository benchmarks:

```bash
python -m scripts.evaluate_baseline
```

Inspect baseline retrieval quality before optimization:

```bash
python -m scripts.optimize_retriever_ga baseline
```

Run offline genetic optimization and save the selected retriever config:

```bash
python -m scripts.optimize_retriever_ga optimize
```

Compare baseline retrieval against the saved optimized config:

```bash
python -m scripts.optimize_retriever_ga compare
```

Train the classifier:

```bash
python -m scripts.claim_type_classifier train
```

Predict claim type:

```bash
python -m scripts.claim_type_classifier predict --claim "creatine boosts strength"
```

Generate or inspect the test image:

```bash
python -m scripts.vision_v1 generate-test
python -m scripts.vision_v1 extract --image data/test_images/creatine_label.png
```


## Notes On Genetic Optimization

The genetic algorithm is an offline optimization layer for the lexical retriever only.

- it does not run inside the normal pipeline
- it does not change the deterministic reasoning rules
- it only selects fixed retrieval parameters that are then loaded from disk at runtime

The current saved artifact improves the retrieval benchmark from:

- baseline `hit@5 = 0.95`, `MRR = 0.8833`
- optimized `hit@5 = 1.0`, `MRR = 0.975`


