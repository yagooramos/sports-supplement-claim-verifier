#!/usr/bin/env python3
"""
Unified pipeline orchestrator for claim verification.

Supports:
- text-only input
- image-only input
- multimodal input (text + image)

The deterministic parser + retrieval + reasoning path remains the
authoritative core. OCR, classifier hints, and optional LLM helpers
are secondary support layers.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

try:
    from . import claim_parser_v1, lexical_retriever_v1, reasoning_v1, vision_v1
    from .claim_type_classifier import classifier_available, predict_claim_type
    from .llm_adapter import (
        extract_claim_fields as llm_extract_claim_fields,
        extract_claim_from_ocr as llm_extract_claim_from_ocr,
        generate_explanation as llm_generate_explanation,
        is_available as llm_is_available,
    )
except ImportError:
    import claim_parser_v1
    import lexical_retriever_v1
    import reasoning_v1
    import vision_v1
    from claim_type_classifier import classifier_available, predict_claim_type
    from llm_adapter import (
        extract_claim_fields as llm_extract_claim_fields,
        extract_claim_from_ocr as llm_extract_claim_from_ocr,
        generate_explanation as llm_generate_explanation,
        is_available as llm_is_available,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX_SCOPE_CSV = REPO_ROOT / "data" / "sources" / "matrix_scope.csv"
DEFAULT_LEXICON_CSV = REPO_ROOT / "data" / "sources" / "lexicon.csv"
DEFAULT_FRAGMENTS_CSV = REPO_ROOT / "data" / "annotations" / "evidence_fragments.csv"


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


class Pipeline:
    def __init__(
        self,
        matrix_scope_csv: str | Path = DEFAULT_MATRIX_SCOPE_CSV,
        lexicon_csv: str | Path = DEFAULT_LEXICON_CSV,
        fragments_csv: str | Path = DEFAULT_FRAGMENTS_CSV,
        use_llm: bool = True,
    ):
        self.matrix_scope_csv = Path(matrix_scope_csv)
        self.lexicon_csv = Path(lexicon_csv)
        self.fragments_csv = Path(fragments_csv)

        self.parser = claim_parser_v1.build_parser(self.matrix_scope_csv, self.lexicon_csv)
        fragments_df = lexical_retriever_v1.load_fragments(self.fragments_csv).fillna("")
        self.fragment_lookup = build_fragment_lookup(self.fragments_csv)
        self.retriever = lexical_retriever_v1.BM25Retriever(fragments_df.to_dict(orient="records"))
        self.corpus_coverage = reasoning_v1.build_corpus_coverage(self.matrix_scope_csv, self.fragments_csv)

        self.use_llm = bool(use_llm and llm_is_available())

    def _determine_effective_claim(
        self,
        claim_text: str | None,
        vision_result: dict[str, object] | None,
    ) -> tuple[str, dict[str, object] | None]:
        llm_ocr_extraction = None
        cleaned_claim = str(claim_text or "").strip()

        if cleaned_claim and vision_result:
            ingredient = str(vision_result.get("detected_ingredient", "")).replace("_", " ").strip()
            effective_claim = cleaned_claim
            if ingredient and ingredient.lower() not in cleaned_claim.lower():
                effective_claim = f"{ingredient}: {cleaned_claim}"
            return effective_claim, llm_ocr_extraction

        if vision_result:
            effective_claim = vision_v1.build_claim_from_vision(vision_result)
            if not effective_claim and self.use_llm:
                ocr_text = str(vision_result.get("detected_text", "")).strip()
                if ocr_text:
                    llm_ocr_extraction = llm_extract_claim_from_ocr(ocr_text)
                    if llm_ocr_extraction and llm_ocr_extraction.get("claim"):
                        ingredient = str(llm_ocr_extraction.get("ingredient", "")).strip()
                        claim = str(llm_ocr_extraction.get("claim", "")).strip()
                        if ingredient and ingredient.lower() not in claim.lower():
                            effective_claim = f"{ingredient} {claim}".strip()
                        else:
                            effective_claim = claim
            return effective_claim, llm_ocr_extraction

        return cleaned_claim, llm_ocr_extraction

    def run(
        self,
        claim_text: str | None = None,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        top_k: int = 5,
    ) -> dict[str, object]:
        vision_result = None
        input_mode = "text"
        if image_path is not None or image_bytes is not None:
            vision_result = vision_v1.extract_from_image(image_path=image_path, image_bytes=image_bytes)
            input_mode = "multimodal" if str(claim_text or "").strip() else "image"

        effective_claim, llm_ocr_extraction = self._determine_effective_claim(claim_text, vision_result)
        if not str(effective_claim).strip():
            return {
                "input_mode": input_mode,
                "vision_result": vision_result,
                "effective_claim": "",
                "claim_parse": None,
                "parse_result": None,
                "retrieval_results": [],
                "retrieved_candidates": [],
                "retrieval_bundle": {"retrieved_candidates": []},
                "reasoning_result": {
                    "verdict": "not_evaluable",
                    "reason_code": "no_input",
                    "explanation": "No claim text could be determined from the provided input.",
                },
                "claim_type_prediction": None,
                "llm_used": False,
                "llm_extraction": None,
                "llm_ocr_extraction": llm_ocr_extraction,
                "llm_explanation": "",
            }

        parse_result = self.parser.parse_claim(effective_claim)
        llm_used = False
        llm_extraction = None

        parse_status = str(parse_result.get("parse_status", "")).strip()
        if self.use_llm and parse_status in {"not_parseable", "partially_parseable"}:
            llm_extraction = llm_extract_claim_fields(effective_claim)
            if llm_extraction and llm_extraction.get("ingredient"):
                llm_used = True
                parse_result["ingredient"] = str(llm_extraction.get("ingredient", "")).strip()
                parse_result["claim_type"] = str(llm_extraction.get("claim_type", "")).strip()
                parse_result["outcome_target"] = str(llm_extraction.get("outcome_target", "")).strip()
                parse_result["parse_status"] = (
                    "fully_parseable"
                    if parse_result["claim_type"] and parse_result["outcome_target"]
                    else "partially_parseable"
                )
                parse_result["notes"] = (
                    f"LLM fallback ({llm_extraction.get('confidence', '?')}): "
                    f"{llm_extraction.get('reasoning', '')}"
                ).strip()

        search_results = self.retriever.search(effective_claim, top_k=top_k)
        retrieval_results = [asdict(result) for result in search_results]
        retrieved_candidates = enrich_candidates(search_results, self.fragment_lookup)
        retrieval_bundle = {
            "retrieval_model": "v1",
            "query_text": effective_claim,
            "retrieved_candidates": retrieved_candidates,
        }

        reasoning_result = reasoning_v1.evaluate_claim(parse_result, retrieval_bundle, self.corpus_coverage)
        claim_type_prediction = predict_claim_type(effective_claim) if classifier_available() else None
        llm_explanation = ""
        if self.use_llm and reasoning_result.get("verdict"):
            llm_explanation = llm_generate_explanation(effective_claim, reasoning_result, retrieved_candidates)

        return {
            "input_mode": input_mode,
            "vision_result": vision_result,
            "effective_claim": effective_claim,
            "claim_parse": parse_result,
            "parse_result": parse_result,
            "retrieval_results": retrieval_results,
            "retrieved_candidates": retrieved_candidates,
            "retrieval_bundle": retrieval_bundle,
            "reasoning_result": reasoning_result,
            "claim_type_prediction": claim_type_prediction,
            "llm_used": llm_used,
            "llm_extraction": llm_extraction,
            "llm_ocr_extraction": llm_ocr_extraction,
            "llm_explanation": llm_explanation,
        }


def run_claim_verification(
    claim_text: str,
    matrix_scope_csv: str | Path = DEFAULT_MATRIX_SCOPE_CSV,
    lexicon_csv: str | Path = DEFAULT_LEXICON_CSV,
    fragments_csv: str | Path = DEFAULT_FRAGMENTS_CSV,
    top_k: int = 5,
) -> dict[str, object]:
    pipeline = Pipeline(
        matrix_scope_csv=matrix_scope_csv,
        lexicon_csv=lexicon_csv,
        fragments_csv=fragments_csv,
        use_llm=False,
    )
    return pipeline.run(claim_text=claim_text, top_k=top_k)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the full claim verification pipeline.")
    parser.add_argument("--claim", default=None, help="Raw claim text")
    parser.add_argument("--image", default=None, help="Path to supplement image")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval results")
    args = parser.parse_args()

    if not args.claim and not args.image:
        parser.error("Provide at least --claim or --image (or both).")

    pipeline = Pipeline()
    result = pipeline.run(claim_text=args.claim, image_path=args.image, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
