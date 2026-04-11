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

It does not currently depend on:

- semantic retrieval
- neural ranking
- end-to-end machine learning
- chatbot-style free-form generation

## Operational Flow

The current flow is:

1. A raw claim can be mapped into the project schema with `claim_parser_v1.py`.
2. `lexical_retriever_v1.py` retrieves candidate evidence fragments from the canonical corpus.
3. `reasoning_v1.py` consumes structured claim information and retrieved evidence to produce a conservative outcome.

The two operational layers are still:

- `retrieval`
- `reasoning`

The parser is treated as a support component, not as a separate top-level layer.

## Academic Alignment

This first version mainly supports:

- `Speech and Natural Language Processing`
  Through claim parsing, lexical retrieval, text normalization, schema alignment, and corpus-driven claim handling.

- `Intelligent Systems`
  Through explicit knowledge representation, rule-based reasoning, and auditable deterministic decisions.

This version does not yet position `Advanced Machine Learning` as a core contribution.
Instead, it establishes the structured baseline that a later ML-oriented extension can build on.

## Repository Structure

- `data/sources/`: canonical source tables
- `data/annotations/`: canonical evidence annotations
- `data/benchmarks/`: minimal retrieval and reasoning benchmark files
- `data/config/`: parser configuration and domain rules
- `scripts/`: core operational and support scripts
- `docs/`: concise technical notes and script/data summaries
- `ROADMAP.md`: current status, completed checks, and next steps

## Scripts

### Operational

- `scripts/lexical_retriever_v1.py`: lexical retrieval over the canonical evidence corpus
- `scripts/reasoning_v1.py`: deterministic reasoning layer over structured inputs

### Support

- `scripts/claim_parser_v1.py`: shallow parser for mapping raw claims into the project schema
- `scripts/utils.py`: shared normalization and text-matching helpers

The parser was simplified by moving most domain-specific rule lists into `data/config/claim_parser_rules.json`.

## Data

### Canonical Sources

- `data/sources/lexicon.csv`
- `data/sources/matrix_scope.csv`

### Canonical Annotations

- `data/annotations/evidence_fragments.csv`

### Minimal Benchmarks

- `data/benchmarks/retrieval_eval_queries.csv`
- `data/benchmarks/reasoning_eval_cases.csv`

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
4. Decide whether the parser remains part of the minimal baseline or stays as support only.
5. Define the next extension path, especially for a future machine-learning component.
