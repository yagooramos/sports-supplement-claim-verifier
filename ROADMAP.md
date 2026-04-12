# Roadmap

## Current Goal

Build a minimal and coherent repository around the project core:

- canonical corpus
- retrieval
- reasoning
- supervised ML classifier (bounded extension)

This project state covers:

- `Speech and Natural Language Processing`
- `Intelligent Systems`
- `Advanced Machine Learning`

## Completed

- [x] Define a minimal repository structure
- [x] Rescue the minimal canonical data needed for the project core
- [x] Rescue the minimal script set needed for parsing, retrieval, and reasoning
- [x] Consolidate scripts, data, and dataset-source notes into the new project structure
- [x] Translate the rescued minimal dataset into English
- [x] Simplify the parser by moving domain rules into `data/config/claim_parser_rules.json`
- [x] Add concise project documentation for scripts, data, and dataset sources
- [x] Prune non-essential files from the minimal baseline

## ML Extension

- [x] Choose a bounded supervised learning task (claim-type classification)
- [x] Build labeled dataset from corpus, PubMed ISSN position stands, and diverse augmentations (181 examples)
- [x] Implement TF-IDF + Logistic Regression classifier (`scripts/claim_type_classifier.py`)
- [x] Evaluate with stratified 5-fold cross-validation (accuracy: ~0.89, F1 macro: ~0.88)
- [x] Save reproducible metrics (`models/claim_type_metrics.json`)
- [x] Document the ML component in README.md, ROADMAP.md, and docs/README.md

## Next Steps

- [ ] Review the rescued files and remove anything still non-essential
- [ ] Decide the first executable project flow
- [ ] Add minimal run instructions for the core pipeline
- [ ] Decide whether the parser should remain in the minimal baseline or stay documented as optional support
- [ ] Evaluate whether to integrate ML predictions into the reasoning layer

## Notes

This roadmap tracks project progress.
It should stay short and operational.
