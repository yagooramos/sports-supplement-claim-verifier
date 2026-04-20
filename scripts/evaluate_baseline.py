#!/usr/bin/env python3
"""
Run the repository's minimal retrieval and reasoning benchmarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from . import lexical_retriever_v1, pipeline
    from .utils import split_pipe_values
except ImportError:
    import lexical_retriever_v1
    import pipeline
    from utils import split_pipe_values


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRAGMENTS_CSV = REPO_ROOT / "data" / "annotations" / "evidence_fragments.csv"
DEFAULT_RETRIEVAL_BENCHMARK_CSV = REPO_ROOT / "data" / "benchmarks" / "retrieval_eval_queries.csv"
DEFAULT_REASONING_BENCHMARK_CSV = REPO_ROOT / "data" / "benchmarks" / "reasoning_eval_cases.csv"
DEFAULT_RETRIEVER_CONFIG_PATH = lexical_retriever_v1.DEFAULT_RETRIEVER_CONFIG_PATH


def _meaningful(value: object) -> str:
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _same_fragment_set(expected_value: str, actual_value: str) -> bool:
    expected_ids = split_pipe_values(expected_value)
    actual_ids = split_pipe_values(actual_value)
    return sorted(expected_ids) == sorted(actual_ids)


def evaluate_retrieval(
    fragments_csv: str | Path = DEFAULT_FRAGMENTS_CSV,
    benchmark_csv: str | Path = DEFAULT_RETRIEVAL_BENCHMARK_CSV,
    retriever_config_path: str | Path = DEFAULT_RETRIEVER_CONFIG_PATH,
    top_k: int = 5,
) -> dict[str, object]:
    benchmark_df = pd.read_csv(benchmark_csv).fillna("")
    fragments_df = lexical_retriever_v1.load_fragments(fragments_csv)
    retriever_config = lexical_retriever_v1.load_retriever_config(retriever_config_path)
    retriever = lexical_retriever_v1.BM25Retriever(fragments_df.to_dict(orient="records"), config=retriever_config)

    query_results = []
    hit_count = 0
    reciprocal_rank_sum = 0.0
    for row in benchmark_df.to_dict(orient="records"):
        query_text = str(row.get("query_text", "")).strip()
        expected_ids = split_pipe_values(row.get("expected_fragment_ids", ""))
        results = retriever.search(query_text, top_k=top_k)
        returned_ids = [result.fragment_id for result in results]
        hit = all(expected_id in returned_ids for expected_id in expected_ids)
        first_relevant_rank = next(
            (index for index, fragment_id in enumerate(returned_ids, start=1) if fragment_id in expected_ids),
            None,
        )
        reciprocal_rank = round(1.0 / first_relevant_rank, 4) if first_relevant_rank else 0.0
        hit_count += int(hit)
        reciprocal_rank_sum += reciprocal_rank
        query_results.append(
            {
                "query_id": str(row.get("query_id", "")).strip(),
                "query_text": query_text,
                "expected_fragment_ids": expected_ids,
                "returned_fragment_ids": returned_ids,
                "hit": hit,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    total = len(query_results)
    return {
        "metric": f"all_expected_in_top_{top_k}",
        "passed": hit_count,
        "total": total,
        "accuracy": round(hit_count / total, 4) if total else 0.0,
        "mrr": round(reciprocal_rank_sum / total, 4) if total else 0.0,
        "queries": query_results,
    }


def evaluate_reasoning(
    benchmark_csv: str | Path = DEFAULT_REASONING_BENCHMARK_CSV,
) -> dict[str, object]:
    benchmark_df = pd.read_csv(benchmark_csv).fillna("")
    case_results = []
    pass_count = 0
    compared_fields = [
        ("expected_parse_status", lambda result: _meaningful(result["claim_parse"].get("parse_status", ""))),
        ("expected_ingredient", lambda result: _meaningful(result["claim_parse"].get("ingredient", ""))),
        ("expected_claim_type", lambda result: _meaningful(result["claim_parse"].get("claim_type", ""))),
        ("expected_outcome_target", lambda result: _meaningful(result["claim_parse"].get("outcome_target", ""))),
        ("expected_matrix_id", lambda result: _meaningful(result["reasoning_result"].get("matched_matrix_id", ""))),
        ("expected_scope_status", lambda result: _meaningful(result["reasoning_result"].get("scope_status", ""))),
        ("expected_coverage_status", lambda result: _meaningful(result["reasoning_result"].get("coverage_status", ""))),
        ("expected_verdict", lambda result: _meaningful(result["reasoning_result"].get("verdict", ""))),
        ("expected_reason_code", lambda result: _meaningful(result["reasoning_result"].get("reason_code", ""))),
        (
            "expected_supporting_fragment_ids",
            lambda result: "|".join(result["reasoning_result"].get("supporting_fragment_ids", [])),
        ),
        (
            "expected_limiting_fragment_ids",
            lambda result: "|".join(result["reasoning_result"].get("limiting_fragment_ids", [])),
        ),
    ]

    for row in benchmark_df.to_dict(orient="records"):
        claim_text = str(row.get("claim_text", "")).strip()
        result = pipeline.run_claim_verification(claim_text)
        mismatches = []
        for expected_field, actual_fn in compared_fields:
            expected_value = _meaningful(row.get(expected_field, ""))
            actual_value = _meaningful(actual_fn(result))
            if expected_field in {"expected_supporting_fragment_ids", "expected_limiting_fragment_ids"}:
                matches = _same_fragment_set(expected_value, actual_value)
            else:
                matches = expected_value == actual_value
            if not matches:
                mismatches.append(
                    {
                        "field": expected_field,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
        passed = not mismatches
        pass_count += int(passed)
        case_results.append(
            {
                "case_id": str(row.get("case_id", "")).strip(),
                "claim_text": claim_text,
                "passed": passed,
                "mismatches": mismatches,
            }
        )

    total = len(case_results)
    return {
        "metric": "exact_match_on_expected_fields",
        "passed": pass_count,
        "total": total,
        "accuracy": round(pass_count / total, 4) if total else 0.0,
        "cases": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal repository benchmarks.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval benchmark.")
    parser.add_argument("--json", action="store_true", help="Print full JSON output.")
    parser.add_argument(
        "--retriever-config",
        default=str(DEFAULT_RETRIEVER_CONFIG_PATH),
        help="Path to retriever config JSON. Missing files fall back to baseline defaults.",
    )
    args = parser.parse_args()

    retrieval_report = evaluate_retrieval(top_k=args.top_k, retriever_config_path=args.retriever_config)
    reasoning_report = evaluate_reasoning()

    summary = {
        "retrieval": {
            "metric": retrieval_report["metric"],
            "passed": retrieval_report["passed"],
            "total": retrieval_report["total"],
            "accuracy": retrieval_report["accuracy"],
            "mrr": retrieval_report["mrr"],
        },
        "reasoning": {
            "metric": reasoning_report["metric"],
            "passed": reasoning_report["passed"],
            "total": reasoning_report["total"],
            "accuracy": reasoning_report["accuracy"],
        },
    }
    print(json.dumps(summary, indent=2))

    if args.json:
        print(json.dumps({"retrieval": retrieval_report, "reasoning": reasoning_report}, indent=2))


if __name__ == "__main__":
    main()
