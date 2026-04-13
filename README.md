# Project

## Final Goal

The goal of this project is to build a conservative and auditable claim-verification system for sports-supplement claims.

The project is built around a canonical tabular corpus with three core fields:

- `ingredient`
- `claim_type`
- `outcome_target`

On top of that corpus, the system is organized around two operational layers:

- `retrieval`
- `reasoning`

This first project state is intentionally narrow. It prioritizes:

- traceability
- explicit structure
- conservative decisions
- minimal reproducible evaluation

## Why This Design

The current design is deliberately simple.

The project does not start from a fully learned end-to-end model. Instead, it starts from:

- a structured corpus
- a lexical retrieval layer
- a deterministic reasoning layer

This choice was made for three reasons:

- it keeps the system interpretable
- it makes the project easier to validate and defend
- it creates a strong baseline that later extensions can be compared against

## What The Project Uses

The current version uses:

- a canonical corpus of source tables and annotated evidence fragments
- lexical retrieval to recover candidate evidence
- deterministic reasoning to decide scope, coverage, and support
- shallow claim parsing as a support step for raw text inputs
- computer vision for extracting text from supplement product images
- a Streamlit interface for interactive claim verification

It does not currently depend on:

- semantic retrieval
- neural ranking
- end-to-end machine learning
- chatbot-style free-form generation
- paid external APIs

## Operational Flow

The system supports three input modes:

1. **Text only**: raw claim text goes directly to the parser.
2. **Image only**: `vision_v1.py` extracts text from a supplement label image, builds a synthetic claim, then continues through the pipeline.
3. **Multimodal** (text + image): vision enriches the user-provided text with information extracted from the image.

In all cases, the downstream flow is the same:

1. `claim_parser_v1.py` maps the claim into the project schema.
2. `lexical_retriever_v1.py` retrieves candidate evidence fragments from the canonical corpus.
3. `reasoning_v1.py` consumes structured claim information and retrieved evidence to produce a conservative outcome.

`pipeline.py` orchestrates all components. `app.py` provides the Streamlit interface.

## Academic Alignment

This project supports three academic domains:

- `Speech and Natural Language Processing`
  Through claim parsing, lexical retrieval, text normalization, schema alignment, and corpus-driven claim handling.

- `Intelligent Systems`
  Through explicit knowledge representation, rule-based reasoning, and auditable deterministic decisions.

- `Advanced Machine Learning`
  Through a supervised text classification component (TF-IDF + Logistic Regression) that predicts `claim_type` from raw claim text. This is a bounded, transparent ML extension that complements the deterministic baseline without replacing it.

- `Computer Vision`
  Through a vision module that extracts text from supplement product images using OpenCV preprocessing (grayscale conversion, Gaussian blur, adaptive thresholding) and Tesseract OCR. The extracted text is transformed into structured input for the existing pipeline.

## Repository Structure

- `data/sources/`: canonical source tables
- `data/annotations/`: canonical evidence annotations
- `data/benchmarks/`: minimal retrieval and reasoning benchmark files
- `data/config/`: parser configuration and domain rules
- `data/ml/`: labeled dataset for the ML classifier
- `data/test_images/`: synthetic test images for the vision module
- `scripts/`: core operational, support, ML, and CV scripts
- `models/`: saved ML metrics (model binary is gitignored and reproducible via training)
- `docs/`: concise technical notes and script/data summaries
- `app.py`: Streamlit application entry point
- `ROADMAP.md`: current status, completed checks, and next steps

## Scripts

### Operational

- `scripts/lexical_retriever_v1.py`: lexical retrieval over the canonical evidence corpus
- `scripts/reasoning_v1.py`: deterministic reasoning layer over structured inputs

### Support

- `scripts/claim_parser_v1.py`: shallow parser for mapping raw claims into the project schema
- `scripts/utils.py`: shared normalization and text-matching helpers

The parser was simplified by moving most domain-specific rule lists into `data/config/claim_parser_rules.json`.

### Machine Learning

- `scripts/claim_type_classifier.py`: supervised claim-type classifier (TF-IDF + Logistic Regression)

### Computer Vision

- `scripts/vision_v1.py`: supplement label text extraction (OpenCV + Tesseract OCR)

### Pipeline and Application

- `scripts/pipeline.py`: unified pipeline orchestrator connecting vision, parser, retriever, and reasoner
- `app.py`: Streamlit web interface supporting text, image, and multimodal input

## Data

### Canonical Sources

- `data/sources/lexicon.csv`
- `data/sources/matrix_scope.csv`

### Canonical Annotations

- `data/annotations/evidence_fragments.csv`

### Minimal Benchmarks

- `data/benchmarks/retrieval_eval_queries.csv`
- `data/benchmarks/reasoning_eval_cases.csv`

### ML Dataset

- `data/ml/claim_type_dataset.csv`

### ML Metrics

- `models/claim_type_metrics.json`

## Minimal Viable Documentation

The repository keeps documentation intentionally short:

- `README.md`
  Main project entry point: goal, design choices, academic alignment, structure, and core components.

- `ROADMAP.md`
  Current status, completed checks, and next steps.

- `docs/README.md`
  Short technical map of scripts, data files, and configuration.

- `docs/dataset_sources.md`
  Overview of the external sources used to build the dataset.

## Steps To Follow

The current project path is:

1. Keep the corpus and schema stable and explicit.
2. Validate the retrieval layer on the minimal benchmark.
3. Validate the reasoning layer on the minimal benchmark.
4. Evaluate and extend the ML classifier as needed.
5. Extend the vision module with layout analysis or learned text detection.
