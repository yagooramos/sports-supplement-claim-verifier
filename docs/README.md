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

## Additional Notes

- `docs/dataset_sources.md`: overview of the external sources used to gather and justify the dataset
- `app.py`: Streamlit entrypoint for the single-claim verification interface
- `PROJECT_CONTEXT.md`: full handoff briefing for another programming agent
