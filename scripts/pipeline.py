#!/usr/bin/env python3
"""
Unified pipeline orchestrator for claim verification.

Connects all project components into a single callable flow:
  vision (optional) -> parser -> LLM fallback (optional) -> retriever -> reasoner

Supports three input modes:
  1. text-only:  raw claim text goes directly to parser
  2. image-only: vision extracts text, builds a synthetic claim
  3. multimodal: vision enriches user-provided text

When the deterministic parser fails (not_parseable or partially_parseable),
the pipeline can optionally use a local LLM via Ollama to attempt
structured extraction. The deterministic baseline is always tried first.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROJECT_ROOT = SCRIPT_DIR.parent

from claim_parser_v1 import build_parser
from lexical_retriever_v1 import BM25Retriever, load_fragments
from reasoning_v1 import build_corpus_coverage, evaluate_claim
from vision_v1 import extract_from_image, build_claim_from_vision, tesseract_available
from llm_adapter import (
    is_available as llm_is_available,
    extract_claim_fields as llm_extract_claim_fields,
    extract_claim_from_ocr as llm_extract_claim_from_ocr,
    generate_explanation as llm_generate_explanation,
)

# Default data paths
MATRIX_SCOPE_CSV = PROJECT_ROOT / "data" / "sources" / "matrix_scope.csv"
LEXICON_CSV = PROJECT_ROOT / "data" / "sources" / "lexicon.csv"
FRAGMENTS_CSV = PROJECT_ROOT / "data" / "annotations" / "evidence_fragments.csv"


class Pipeline:
    """Loads all components once and exposes a single run() method."""

    def __init__(
        self,
        matrix_scope_csv: str | Path = MATRIX_SCOPE_CSV,
        lexicon_csv: str | Path = LEXICON_CSV,
        fragments_csv: str | Path = FRAGMENTS_CSV,
        use_llm: bool = True,
    ):
        self.parser = build_parser(matrix_scope_csv, lexicon_csv)

        fragments_df = load_fragments(fragments_csv)
        self.retriever = BM25Retriever(fragments_df.to_dict(orient="records"))

        self.corpus_coverage = build_corpus_coverage(matrix_scope_csv, fragments_csv)

        self.use_llm = use_llm and llm_is_available()

    def run(
        self,
        claim_text: str | None = None,
        image_path: str | Path | None = None,
        top_k: int = 5,
    ) -> dict[str, object]:
        """Run the full verification pipeline.

        Returns a dict with: input_mode, vision_result (if image),
        effective_claim, parse_result, retrieval_results, reasoning_result.
        """
        vision_result = None
        input_mode = "text"

        # --- Step 1: Vision extraction (if image provided) ---
        if image_path is not None:
            vision_result = extract_from_image(image_path)
            input_mode = "multimodal" if claim_text else "image"

        # --- Step 2: Determine effective claim text ---
        llm_ocr_extraction = None
        if claim_text and vision_result:
            # Multimodal: prefer user text but enrich with vision if it
            # found an ingredient the user text does not mention
            vision_claim = build_claim_from_vision(vision_result)
            effective_claim = claim_text
            if vision_claim and vision_result.get("detected_ingredient"):
                ingredient = str(vision_result["detected_ingredient"]).replace("_", " ")
                if ingredient.lower() not in claim_text.lower():
                    effective_claim = f"{ingredient}: {claim_text}"
        elif vision_result:
            # Image-only: use LLM to intelligently select the claim from
            # raw OCR text instead of relying on heuristic matching
            ocr_text = str(vision_result.get("detected_text", ""))
            if self.use_llm and ocr_text.strip():
                llm_ocr_extraction = llm_extract_claim_from_ocr(ocr_text)
                if llm_ocr_extraction and llm_ocr_extraction.get("claim"):
                    claim = llm_ocr_extraction["claim"]
                    ingr = llm_ocr_extraction.get("ingredient", "")
                    if ingr and ingr.lower() not in claim.lower():
                        effective_claim = f"{ingr} {claim}"
                    else:
                        effective_claim = claim
                else:
                    # LLM failed or unavailable, fall back to heuristic
                    effective_claim = build_claim_from_vision(vision_result)
            else:
                effective_claim = build_claim_from_vision(vision_result)
        else:
            effective_claim = claim_text or ""

        if not effective_claim.strip():
            return {
                "input_mode": input_mode,
                "vision_result": vision_result,
                "effective_claim": "",
                "parse_result": None,
                "retrieval_results": [],
                "reasoning_result": {
                    "verdict": "not_evaluable",
                    "reason_code": "no_input",
                    "explanation": "No claim text could be determined from the input.",
                },
            }

        # --- Step 3: Parse (deterministic) ---
        parse_result = self.parser.parse_claim(effective_claim)
        llm_used = False
        llm_extraction = None

        # --- Step 3b: LLM fallback if parser failed ---
        parse_status = str(parse_result.get("parse_status", ""))
        if self.use_llm and parse_status in ("not_parseable", "partially_parseable"):
            llm_extraction = llm_extract_claim_fields(effective_claim)
            if llm_extraction and llm_extraction.get("ingredient"):
                llm_used = True
                parse_result["ingredient"] = llm_extraction["ingredient"]
                parse_result["claim_type"] = llm_extraction.get("claim_type", "")
                parse_result["outcome_target"] = llm_extraction.get("outcome_target", "")
                if llm_extraction.get("claim_type") and llm_extraction.get("outcome_target"):
                    parse_result["parse_status"] = "fully_parseable"
                else:
                    parse_result["parse_status"] = "partially_parseable"
                parse_result["notes"] = (
                    f"LLM fallback ({llm_extraction.get('confidence', '?')}): "
                    f"{llm_extraction.get('reasoning', '')}"
                )

        # --- Step 4: Retrieve ---
        search_results = self.retriever.search(effective_claim, top_k=top_k)
        retrieval_results = [asdict(r) for r in search_results]
        retrieval_bundle = {"retrieved_candidates": retrieval_results}

        # --- Step 5: Reason (deterministic) ---
        reasoning_result = evaluate_claim(
            parse_result, retrieval_bundle, self.corpus_coverage,
        )

        # --- Step 6: LLM explanation (optional) ---
        llm_explanation = ""
        if self.use_llm and reasoning_result.get("verdict"):
            llm_explanation = llm_generate_explanation(
                effective_claim, reasoning_result, retrieval_results,
            )

        return {
            "input_mode": input_mode,
            "vision_result": vision_result,
            "effective_claim": effective_claim,
            "parse_result": parse_result,
            "retrieval_results": retrieval_results,
            "reasoning_result": reasoning_result,
            "llm_used": llm_used,
            "llm_extraction": llm_extraction,
            "llm_ocr_extraction": llm_ocr_extraction,
            "llm_explanation": llm_explanation,
        }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run the full claim verification pipeline.")
    ap.add_argument("--claim", default=None, help="Raw claim text")
    ap.add_argument("--image", default=None, help="Path to supplement image")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k retrieval results")
    args = ap.parse_args()

    if not args.claim and not args.image:
        ap.error("Provide at least --claim or --image (or both).")

    pipeline = Pipeline()
    result = pipeline.run(claim_text=args.claim, image_path=args.image, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
