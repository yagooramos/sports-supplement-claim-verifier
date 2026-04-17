# Documentation Notes

This folder keeps the technical notes concise while the repository root remains focused on runnable project files.

## Scripts

### Deterministic Core

- `scripts/claim_parser_v1.py`: shallow schema-oriented claim parser
- `scripts/lexical_retriever_v1.py`: BM25 lexical retrieval over the evidence corpus
- `scripts/reasoning_v1.py`: deterministic verdict logic
- `scripts/pipeline.py`: unified text, image, and multimodal orchestration layer

### OCR And Vision

- `scripts/ocr_claim_extractor.py`: OCR preprocessing and claim ranking from uploaded images
- `scripts/vision_v1.py`: structured image extraction wrapper used by the main pipeline

### ML

- `scripts/claim_type_classifier.py`: TF-IDF + Logistic Regression classifier for `claim_type`

### Optional LLM

- `scripts/llm_adapter.py`: optional Ollama-based parser fallback, OCR claim selection, and explanation generation

### Evaluation And Helpers

- `scripts/evaluate_baseline.py`: retrieval and reasoning benchmark runner
- `scripts/utils.py`: shared normalization and phrase-matching helpers
- `scripts/__init__.py`: package marker

## Data

### Canonical Deterministic Data

- `data/sources/lexicon.csv`
- `data/sources/matrix_scope.csv`
- `data/annotations/evidence_fragments.csv`
- `data/config/claim_parser_rules.json`

### Benchmarks

- `data/benchmarks/retrieval_eval_queries.csv`
- `data/benchmarks/reasoning_eval_cases.csv`

### ML Data

- `data/ml/claim_type_dataset.csv`: auxiliary labeled dataset for supervised claim-type classification
- `models/claim_type_metrics.json`: reference classifier metrics

### Vision Assets

- `data/test_images/creatine_label.png`: synthetic test image for OCR and pipeline checks

## Current Snapshot

- deterministic retrieval benchmark: `20/20`
- deterministic reasoning benchmark: `16/16`
- classifier dataset size: `345`

## Main Commands

```bash
python -m streamlit run app.py
python -m scripts.evaluate_baseline
python -m scripts.claim_type_classifier train
python -m scripts.vision_v1 extract --image data/test_images/creatine_label.png
```
