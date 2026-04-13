# Project Context For Another Programming Agent

This document is the full repository briefing for any programming agent that needs to continue the work.

It is intentionally more explicit than `README.md`.
`README.md` explains the baseline.
This file explains the academic context, the current scope, the expected deliverables, and the constraints that should shape future changes.

## 1. General Project Context

- Project name: `Sports Supplement Claim Verifier`
- Current repository goal: build a conservative, auditable baseline for verifying sports-supplement claims
- Current project stage: minimal runnable baseline
- Current interface: single-page Streamlit application
- Baseline objective: verify one claim at a time and expose the full decision path
- Immediate academic purpose: provide a defendable and inspectable project core before any later machine-learning extension

## 2. Executive Summary

The project aims to verify claims about sports supplements with a conservative pipeline instead of a black-box end-to-end model.
The current baseline works with a canonical tabular corpus and three schema fields: `ingredient`, `claim_type`, and `outcome_target`.
From a raw claim, the system parses the text into that schema, retrieves candidate evidence fragments from the curated corpus, and applies deterministic reasoning to produce a verdict.
The current repository is designed to be interpretable, easy to defend in an academic setting, and reproducible enough to serve as the reference baseline for later extensions.

## 3. Academic Scope By Subject

The repository currently supports three academic areas, but only two are active in the present baseline.

### 3.1 Speech and Natural Language Processing

- Current role in the project:
  claim parsing, schema alignment, text normalization, lexical retrieval, and corpus-driven handling of claim text
- Current baseline contribution:
  raw claims can be mapped into a structured representation and matched against the evidence corpus with transparent lexical methods
- Evidence in the repository:
  `scripts/claim_parser_v1.py`, `scripts/lexical_retriever_v1.py`, parser rules in `data/config/claim_parser_rules.json`, and retrieval benchmarks in `data/benchmarks/retrieval_eval_queries.csv`
- Deliverable expectation at the current stage:
  a clear explanation of how claims are normalized, parsed, and matched to evidence
- Evaluation logic that should be defendable:
  explainable preprocessing, explicit schema mapping, auditable retrieval behavior, and minimal benchmark-based validation

### 3.2 Intelligent Systems

- Current role in the project:
  explicit knowledge representation and deterministic rule-based reasoning over structured claim and evidence data
- Current baseline contribution:
  the system produces conservative verdicts using reasoning rules rather than generative output
- Evidence in the repository:
  `scripts/reasoning_v1.py`, the matrix scope file, annotated evidence fragments, and reasoning benchmarks in `data/benchmarks/reasoning_eval_cases.csv`
- Deliverable expectation at the current stage:
  a defendable reasoning layer with transparent scope checks, support checks, and limitation handling
- Evaluation logic that should be defendable:
  traceable verdict rules, conservative handling of uncertainty, and explicit reason codes

### 3.3 Advanced Machine Learning

- Current role in the project:
  not part of the active baseline
- Current status:
  postponed intentionally
- Why it is postponed:
  the repository is first establishing a structured, reproducible, and auditable baseline that later ML work can improve on or compare against
- Constraint for future agents:
  do not reposition this baseline as an ML-heavy project unless the user explicitly asks for that extension

## 4. Current Functional Scope

### Included in the baseline

- accept one sports-supplement claim as raw text
- parse the claim into the project schema
- retrieve candidate evidence fragments from the canonical corpus
- reason over retrieved evidence and corpus coverage
- return a conservative verdict with explanation fields
- expose the result through a Streamlit interface

### Explicitly not included yet

- semantic retrieval
- neural reranking
- end-to-end ML classification
- free-form chatbot interaction
- broad multi-claim workflows
- production deployment concerns beyond local execution

## 5. End-to-End Flow

The expected baseline flow is:

1. receive one raw claim
2. parse it into `ingredient`, `claim_type`, and `outcome_target` when possible
3. retrieve candidate evidence fragments from the canonical evidence corpus
4. build the retrieval bundle for reasoning
5. evaluate scope, coverage, support, and limitations
6. return the verdict, reason code, explanation, and supporting fragment identifiers
7. show the decision flow in Streamlit

## 6. Technical Architecture

### Frontend

- Technology: Streamlit
- Entry point: `app.py`
- Interface style: single-page baseline for one claim at a time
- User-facing purpose: make the pipeline easy to inspect and demo

### Backend Logic

- Language: Python
- Current orchestration layer: `scripts/pipeline.py`
- Core modules:
  - `scripts/claim_parser_v1.py`
  - `scripts/lexical_retriever_v1.py`
  - `scripts/reasoning_v1.py`
  - `scripts/utils.py`

### Data Layer

- `data/sources/lexicon.csv`
- `data/sources/matrix_scope.csv`
- `data/annotations/evidence_fragments.csv`
- `data/config/claim_parser_rules.json`
- `data/benchmarks/retrieval_eval_queries.csv`
- `data/benchmarks/reasoning_eval_cases.csv`

## 7. Deliverables Expected From The Current Baseline

At this repository stage, another agent should preserve or improve these deliverables:

- runnable code for the single-claim verifier
- a clear `README.md`
- concise technical notes in `docs/`
- a roadmap of current status and next decisions
- a baseline that can be demonstrated locally with `streamlit run app.py`
- enough structure and traceability to justify the project academically

## 8. Documentation Expectations

When updating the repository, documentation should preserve both technical clarity and academic framing.

Minimum documentation expectations:

- `README.md` should explain the project goal, architecture, subject alignment, structure, and run instructions
- `PROJECT_CONTEXT.md` should remain the handoff document for another agent
- `ROADMAP.md` should stay short and operational
- `docs/README.md` should document scripts, data files, and configuration
- `docs/dataset_sources.md` should justify where the dataset was assembled from

## 9. Current Repository Status

This is the current known state of the baseline:

- the repository has a canonical minimal data layout
- parsing, retrieval, and reasoning modules are present
- a small orchestration layer exists in `scripts/pipeline.py`
- a Streamlit interface exists in `app.py`
- `streamlit` is listed in `requirements.txt`
- the baseline is locally runnable
- the project is still intentionally narrow and conservative

## 10. Known Open Decisions

These questions are still open and should not be silently decided by another agent:

- whether the parser should stay visible as a public baseline component or remain framed as internal support
- what the next post-baseline step should be: benchmark inspection, corpus browsing, or ML-oriented extension work
- how the later machine-learning stage should be introduced without weakening the interpretability of the baseline

## 11. Constraints Another Agent Should Respect

- preserve the conservative and auditable character of the project
- do not replace the current reasoning layer with opaque generation without explicit approval
- do not present the current baseline as a full ML system
- keep the repository coherent and easy to defend academically
- prefer explicit and reproducible structure over sophistication for its own sake
- avoid adding non-essential files or environment artifacts to version control

## 12. Current Run Instructions

From the repository root:

```bash
streamlit run app.py
```

## 13. If More Academic Metadata Is Needed

Some course-specific metadata is still unknown in the repository itself, such as:

- official subject names in the user's language
- professor names
- exact grading rubrics
- exact report or presentation requirements
- final submission format

If the user provides that information later, it should be added here instead of being left implicit.
