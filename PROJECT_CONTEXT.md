## 1. Project Summary

- Project name: `Sports Supplement Claim Verifier`
- Current stage: runnable multimodal repository with offline-optimized retrieval
- Primary objective: verify one supplement claim at a time with a conservative and auditable decision path
- Current interface: Streamlit
- Current repository shape: deterministic core plus bounded OCR, ML, optional LLM, and offline retriever optimization extensions

## 2. Current Execution Flow

The current expected flow is:

1. receive a claim as text, image, or both
2. if an image is present, run OCR and structured vision extraction
3. construct the effective claim used downstream
4. parse the claim into `ingredient`, `claim_type`, and `outcome_target`
5. retrieve candidate evidence fragments
6. evaluate scope, coverage, support, and limitations
7. produce the deterministic verdict
8. optionally attach classifier output and optional local LLM assistance output

The deterministic parser, optimized retriever configuration, and reasoner remain authoritative in runtime.

## 3. Included Components

### Deterministic Core

- `scripts/claim_parser_v1.py`
- `scripts/lexical_retriever_v1.py`
- `scripts/reasoning_v1.py`
- `scripts/pipeline.py`
- `scripts/utils.py`

### OCR And Vision

- `scripts/ocr_claim_extractor.py`
- `scripts/vision_v1.py`
- `data/test_images/creatine_label.png`

### Machine Learning

- `scripts/claim_type_classifier.py`
- `data/ml/claim_type_dataset.csv`
- `models/claim_type_metrics.json`

### Offline Retrieval Optimization

- `scripts/optimize_retriever_ga.py`
- `models/retriever_optimized_config.json`
- genetic search over BM25 `k1`, `b`, and per-field weights

### Optional LLM Support

- `scripts/llm_adapter.py`
- local Ollama only when available

### Validation

- `scripts/evaluate_baseline.py`
- `data/benchmarks/retrieval_eval_queries.csv`
- `data/benchmarks/reasoning_eval_cases.csv`


## 4. Current Data Snapshot

- canonical lexicon rows: `18`
- matrix scope rows: `10`
- canonical evidence fragments: `15`
- retrieval benchmark queries: `20`
- reasoning benchmark cases: `16`
- classifier dataset rows: `345`

## 5. Current Validation Snapshot

Validated on April 21, 2026:

- retrieval benchmark: `20/20`
- retrieval MRR with optimized config: `0.975`
- reasoning benchmark: `16/16`
- app startup: passed

## 6. Academic Scope

The merged repository now spans:

- `Speech and Natural Language Processing`
  Through parsing, retrieval, OCR text extraction, normalization, and schema mapping.

- `Intelligent Systems`
  Through explicit deterministic reasoning, conservative verdict logic, and offline genetic optimization of retriever parameters.

- `Advanced Machine Learning`
  Through the bounded TF-IDF + Logistic Regression `claim_type` classifier.

- `Computer Vision`
  Through OCR-assisted text extraction from supplement images.

## 7. Constraints

- preserve the deterministic verdict path as authoritative
- treat OCR, classifier, LLM, and GA layers as support layers unless explicitly re-scoped
- keep docs aligned with actual code and benchmark state
- do not silently turn the project into an opaque end-to-end ML system
- prefer portability and inspectability over sophistication for its own sake

The retriever optimization specifically must remain:

- offline only
- versioned as a saved config artifact
- separate from the runtime reasoning logic

## 8. Open Decisions

- whether classifier output should remain advisory or feed parsing decisions
- whether local LLM fallback should remain optional
- whether the next UI step should focus on corpus browsing or benchmark inspection
- whether the retrieval benchmark should be expanded beyond the current compact canonical set
- whether multilingual claim handling should become a next-stage feature

## 9. Practical Commands

Run app:

```bash
python -m streamlit run app.py
```

Run deterministic benchmarks:

```bash
python -m scripts.evaluate_baseline
```

Run the retriever GA baseline report:

```bash
python -m scripts.optimize_retriever_ga baseline
```

Optimize retriever parameters offline:

```bash
python -m scripts.optimize_retriever_ga optimize
```

Compare baseline and optimized retrieval:

```bash
python -m scripts.optimize_retriever_ga compare
```

Train classifier:

```bash
python -m scripts.claim_type_classifier train
```

Inspect test image:

```bash
python -m scripts.vision_v1 extract --image data/test_images/creatine_label.png
```
