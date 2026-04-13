# Project

## Project Goal

The goal of this project is to build a conservative and auditable claim-verification system for sports-supplement claims.

The project is organized around a canonical tabular corpus with three core fields:

- `ingredient`
- `claim_type`
- `outcome_target`

## Final Objective

The current baseline verifies one claim at a time and exposes the full decision flow visually:

1. parse the claim into the project schema
2. retrieve candidate evidence fragments
3. apply deterministic reasoning to produce a conservative result

The baseline prioritizes:

- traceability
- explicit structure
- conservative decisions
- minimal reproducible evaluation

## Why This Architecture

The project starts from:

- a structured corpus
- a lexical retrieval layer
- a deterministic reasoning layer

This design was chosen because it:

- keeps the system interpretable
- makes the verification process easy to inspect
- provides a strong baseline for future project work

## What The Project Uses

The current version uses:

- a canonical corpus of source tables and annotated evidence fragments
- lexical retrieval to recover candidate evidence
- deterministic reasoning to decide scope, coverage, and support
- shallow claim parsing as a support step for raw text inputs
- a Streamlit frontend as the main user-facing interface

It does not currently use:

- semantic retrieval
- neural ranking
- end-to-end machine learning
- chatbot-style free-form generation

## Operational Layers

The project has two operational layers:

- `retrieval`
- `reasoning`

Claim parsing is included as a support component that prepares raw text for the two main layers.

## Academic Alignment

This baseline mainly supports:

- `Speech and Natural Language Processing`
  Through claim parsing, lexical retrieval, text normalization, schema alignment, and corpus-driven claim handling.

- `Intelligent Systems`
  Through explicit knowledge representation, rule-based reasoning, and auditable deterministic decisions.

This baseline is not yet the machine-learning-focused stage of the project.
Its role is to establish the structured and reproducible system that later ML work can build on or be compared against.

## Project Context For Another Agent

If this repository is handed to another programming agent, use `PROJECT_CONTEXT.md` as the full briefing document.

That document records:

- the current project scope
- the academic role of each subject area already identified in the repository
- the expected deliverables and constraints for this baseline
- the current implementation status
- the open decisions that remain before a later ML extension

## Repository Structure

- `app.py`: Streamlit entrypoint for the single-claim verifier
- `data/sources/`: canonical source tables
- `data/annotations/`: canonical evidence annotations
- `data/benchmarks/`: minimal retrieval and reasoning benchmark files
- `data/config/`: parser configuration and domain rules
- `scripts/`: operational logic and support modules
- `docs/`: concise technical notes and dataset-source summaries
- `ROADMAP.md`: current project status and next steps

## Main Components

### Operational

- `scripts/lexical_retriever_v1.py`: lexical retrieval over the canonical evidence corpus
- `scripts/reasoning_v1.py`: deterministic reasoning over structured claim and retrieval inputs

### Support

- `scripts/claim_parser_v1.py`: shallow parser for mapping raw claims into the project schema
- `scripts/pipeline.py`: thin orchestration layer that runs parsing, retrieval, and reasoning end to end
- `scripts/utils.py`: shared normalization and text-matching helpers

The parser keeps its logic in code while its domain-specific rule lists live in `data/config/claim_parser_rules.json`.

## Minimal Data Layout

### Canonical Sources

- `data/sources/lexicon.csv`
- `data/sources/matrix_scope.csv`

### Canonical Annotations

- `data/annotations/evidence_fragments.csv`

### Minimal Benchmarks

- `data/benchmarks/retrieval_eval_queries.csv`
- `data/benchmarks/reasoning_eval_cases.csv`

## Minimal Viable Documentation

- `README.md`
  Main project entry point: goal, architecture, academic alignment, structure, and run instructions.

- `ROADMAP.md`
  Current status and next steps.

- `PROJECT_CONTEXT.md`
  Full reusable briefing for another programming agent, including academic framing, scope, deliverables, status, and constraints.

- `docs/README.md`
  Short technical map of scripts, data files, and configuration.

- `docs/dataset_sources.md`
  Overview of the external sources used to build the dataset.

## Official Run Command

Run the baseline app from the repository root:

```bash
streamlit run app.py
```
