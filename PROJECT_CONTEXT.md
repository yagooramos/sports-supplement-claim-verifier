# Project Context For Another Programming Agent

This file describes the current repository after merging the main repo with the more advanced worktree found in `.claude/worktrees/lucid-jones`.

## 1. Project Summary

- Project name: `Sports Supplement Claim Verifier`
- Current stage: runnable multimodal repository baseline
- Primary objective: verify one supplement claim at a time with a conservative and auditable decision path
- Current interface: Streamlit
- Current repository shape: deterministic core plus bounded OCR, ML, and optional LLM extensions

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

The deterministic parser, retriever, and reasoner remain authoritative.

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

### Optional LLM Support

- `scripts/llm_adapter.py`
- local Ollama only when available

### Validation

- `scripts/evaluate_baseline.py`
- `data/benchmarks/retrieval_eval_queries.csv`
- `data/benchmarks/reasoning_eval_cases.csv`

## 4. Important Merge Decision

The advanced `.claude` worktree included a Tesseract/OpenCV-based OCR path.
The main repository does not use that exact implementation.

Instead, the merged repo keeps the current `RapidOCR`-based OCR stack because:

- it already worked locally in this environment
- it avoided the external system dependency on Tesseract
- it was easier to validate immediately

So the merge is feature-level, not a literal file-for-file replacement.

## 5. Current Data Snapshot

- canonical lexicon rows: `18`
- matrix scope rows: `10`
- canonical evidence fragments: `15`
- retrieval benchmark queries: `20`
- reasoning benchmark cases: `16`
- classifier dataset rows: `345`

## 6. Current Validation Snapshot

Validated on April 14, 2026:

- retrieval benchmark: `20/20`
- reasoning benchmark: `16/16`
- app startup: passed

## 7. Academic Scope

The merged repository now spans:

- `Speech and Natural Language Processing`
  Through parsing, retrieval, OCR text extraction, normalization, and schema mapping.

- `Intelligent Systems`
  Through explicit deterministic reasoning and conservative verdict logic.

- `Advanced Machine Learning`
  Through the bounded TF-IDF + Logistic Regression `claim_type` classifier.

- `Computer Vision`
  Through OCR-assisted text extraction from supplement images.

## 8. Constraints

- preserve the deterministic verdict path as authoritative
- treat OCR, classifier, and LLM layers as support layers unless explicitly re-scoped
- keep docs aligned with actual code and benchmark state
- do not silently turn the project into an opaque end-to-end ML system
- prefer portability and inspectability over sophistication for its own sake

## 9. Open Decisions

- whether classifier output should remain advisory or feed parsing decisions
- whether local LLM fallback should remain optional
- whether the next UI step should focus on corpus browsing or benchmark inspection
- whether multilingual claim handling should become a next-stage feature

## 10. Practical Commands

Run app:

```bash
python -m streamlit run app.py
```

Run deterministic benchmarks:

```bash
python -m scripts.evaluate_baseline
```

Train classifier:

```bash
python -m scripts.claim_type_classifier train
```

Inspect test image:

```bash
python -m scripts.vision_v1 extract --image data/test_images/creatine_label.png
```
