#!/usr/bin/env python3
"""
Small orchestration layer for the project baseline.

This module keeps the end-to-end claim verification flow in one place so it can
be reused by the Streamlit app and by direct checks.
"""

from __future__ import annotations

from pathlib import Path

import claim_parser_v1
import lexical_retriever_v1
import reasoning_v1


DEFAULT_MATRIX_SCOPE_CSV = Path("data/sources/matrix_scope.csv")
DEFAULT_LEXICON_CSV = Path("data/sources/lexicon.csv")
DEFAULT_FRAGMENTS_CSV = Path("data/annotations/evidence_fragments.csv")


def enrich_candidates(
    results: list[lexical_retriever_v1.SearchResult],
    fragment_lookup: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    enriched = []
    for result in results:
        fragment_row = dict(fragment_lookup.get(result.fragment_id, {}))
        fragment_row.update(
            {
                "rank": result.rank,
                "score": result.score,
                "fragment_id": result.fragment_id,
                "doc_id": result.doc_id,
                "matrix_id": result.matrix_id,
                "ingredient": result.ingredient,
                "claim_type": result.claim_type,
                "outcome_target": result.outcome_target,
                "supports_claim": result.supports_claim,
                "support_strength": result.support_strength,
                "fragment_text": result.fragment_text,
            }
        )
        enriched.append(fragment_row)
    return enriched


def build_fragment_lookup(fragments_csv: str | Path = DEFAULT_FRAGMENTS_CSV) -> dict[str, dict[str, object]]:
    fragments_df = lexical_retriever_v1.load_fragments(fragments_csv).fillna("")
    lookup = {}
    for row in fragments_df.to_dict(orient="records"):
        fragment_id = str(row.get("fragment_id", "")).strip()
        if fragment_id:
            lookup[fragment_id] = row
    return lookup


def run_claim_verification(
    claim_text: str,
    matrix_scope_csv: str | Path = DEFAULT_MATRIX_SCOPE_CSV,
    lexicon_csv: str | Path = DEFAULT_LEXICON_CSV,
    fragments_csv: str | Path = DEFAULT_FRAGMENTS_CSV,
    top_k: int = 5,
) -> dict[str, object]:
    parser = claim_parser_v1.build_parser(matrix_scope_csv, lexicon_csv)
    claim_parse = parser.parse_claim(claim_text)

    fragments_df = lexical_retriever_v1.load_fragments(fragments_csv).fillna("")
    fragment_lookup = build_fragment_lookup(fragments_csv)
    retriever = lexical_retriever_v1.BM25Retriever(fragments_df.to_dict(orient="records"))
    retrieval_results = retriever.search(claim_text, top_k=top_k)
    retrieved_candidates = enrich_candidates(retrieval_results, fragment_lookup)

    retrieval_bundle = {
        "retrieval_model": "v1",
        "query_text": claim_text,
        "retrieved_candidates": retrieved_candidates,
    }
    corpus_coverage = reasoning_v1.build_corpus_coverage(matrix_scope_csv, fragments_csv)
    reasoning_result = reasoning_v1.evaluate_claim(claim_parse, retrieval_bundle, corpus_coverage)

    return {
        "claim_text": claim_text,
        "claim_parse": claim_parse,
        "retrieval_results": [result.__dict__ for result in retrieval_results],
        "retrieved_candidates": retrieved_candidates,
        "retrieval_bundle": retrieval_bundle,
        "reasoning_result": reasoning_result,
    }
