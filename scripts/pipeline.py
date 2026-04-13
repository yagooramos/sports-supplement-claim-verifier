#!/usr/bin/env python3
"""
Unified pipeline orchestrator for claim verification.

Connects all project components into a single callable flow:
  vision (optional) -> parser -> retriever -> reasoner

Supports three input modes:
  1. text-only:  raw claim text goes directly to parser
  2. image-only: vision extracts text, builds a synthetic claim
  3. multimodal: vision enriches user-provided text

This module does not add new logic. It only wires existing components.
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
    ):
        self.parser = build_parser(matrix_scope_csv, lexicon_csv)

        fragments_df = load_fragments(fragments_csv)
        self.retriever = BM25Retriever(fragments_df.to_dict(orient="records"))

        self.corpus_coverage = build_corpus_coverage(matrix_scope_csv, fragments_csv)

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
            # Image-only: build a synthetic claim from vision output
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

        # --- Step 3: Parse ---
        parse_result = self.parser.parse_claim(effective_claim)

        # --- Step 4: Retrieve ---
        search_results = self.retriever.search(effective_claim, top_k=top_k)
        retrieval_results = [asdict(r) for r in search_results]
        retrieval_bundle = {"retrieved_candidates": retrieval_results}

        # --- Step 5: Reason ---
        reasoning_result = evaluate_claim(
            parse_result, retrieval_bundle, self.corpus_coverage,
        )

        return {
            "input_mode": input_mode,
            "vision_result": vision_result,
            "effective_claim": effective_claim,
            "parse_result": parse_result,
            "retrieval_results": retrieval_results,
            "reasoning_result": reasoning_result,
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
