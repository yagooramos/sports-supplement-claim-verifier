# Documentation Notes

This folder contains lightweight supporting documentation for the project.

Its role is to keep the repository root clean while preserving technical clarity.

## Scripts

### Operational

- `scripts/lexical_retriever_v1.py`: retrieves candidate evidence fragments from the canonical corpus
- `scripts/reasoning_v1.py`: produces conservative reasoning outcomes from structured claim and retrieval inputs

### Support

- `scripts/claim_parser_v1.py`: extracts a shallow structured claim representation from raw text
- `scripts/pipeline.py`: orchestrates parsing, retrieval, and reasoning for one claim
- `scripts/utils.py`: shared text normalization, tokenization, and phrase-matching helpers

### Machine Learning

- `scripts/claim_type_classifier.py`: supervised claim-type classifier (TF-IDF + Logistic Regression). Trains on `data/ml/claim_type_dataset.csv`, evaluates with stratified 5-fold CV, saves model and metrics to `models/`.

### Computer Vision

- `scripts/vision_v1.py`: extracts text from supplement product images using OpenCV preprocessing (grayscale, Gaussian blur, adaptive thresholding) and Tesseract OCR. Produces structured output: `detected_text`, `detected_ingredient`, `detected_claims`, `detected_dose`, `vision_confidence`, `vision_notes`. Requires Tesseract OCR system install.

### Pipeline and Application

- `scripts/pipeline.py`: unified orchestrator connecting vision, parser, retriever, and reasoner. Supports text-only, image-only, and multimodal input modes. CLI: `python scripts/pipeline.py --claim "..." [--image path]`.
- `app.py`: Streamlit web interface. Run with `streamlit run app.py`.

### Configuration

- `data/config/claim_parser_rules.json`: parser rules and domain-specific phrase lists used by `claim_parser_v1.py`

## Data

### Sources

- `data/sources/lexicon.csv`: canonical lexicon for ingredients, claims, and outcomes
- `data/sources/matrix_scope.csv`: canonical matrix of supported project tuples

### Annotations

- `data/annotations/evidence_fragments.csv`: canonical evidence fragments used by retrieval

### Benchmarks

- `data/benchmarks/retrieval_eval_queries.csv`: minimal retrieval benchmark
- `data/benchmarks/reasoning_eval_cases.csv`: minimal reasoning benchmark

### ML Data

- `data/ml/claim_type_dataset.csv`: labeled dataset for claim-type classification (181 examples, 4 classes). Sources: `matrix_scope` example claims, `reasoning_eval` cases, ISSN position stand abstracts from PubMed (`pubmed_derived`), and diverse domain-vocabulary augmentations. Each row records its provenance in the `source` column.

### ML Artifacts

- `models/claim_type_metrics.json`: cross-validation and training metrics from the classifier
- `models/claim_type_model.joblib`: trained model binary (gitignored, reproducible via `python scripts/claim_type_classifier.py train`)

### Test Images

- `data/test_images/creatine_label.png`: synthetic supplement label image for testing the vision pipeline. Generated with `python scripts/vision_v1.py generate-test`.

## Additional Notes

- `docs/dataset_sources.md`: overview of the external sources used to gather and justify the dataset
- `app.py`: Streamlit entrypoint for the single-claim verification interface
- `PROJECT_CONTEXT.md`: full handoff briefing for another programming agent
