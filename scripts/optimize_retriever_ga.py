#!/usr/bin/env python3
"""
Offline genetic optimization for the lexical retriever.

This script tunes BM25 parameters and per-field weights against the
retrieval benchmark. It is intentionally offline: runtime code only
loads the selected config from disk and never runs the optimizer.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import pandas as pd

try:
    from . import lexical_retriever_v1
    from .utils import split_pipe_values
except ImportError:
    import lexical_retriever_v1
    from utils import split_pipe_values


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRAGMENTS_CSV = REPO_ROOT / "data" / "annotations" / "evidence_fragments.csv"
DEFAULT_RETRIEVAL_BENCHMARK_CSV = REPO_ROOT / "data" / "benchmarks" / "retrieval_eval_queries.csv"
DEFAULT_OUTPUT_PATH = lexical_retriever_v1.DEFAULT_RETRIEVER_CONFIG_PATH
FIELD_BOUNDS = {
    "fragment_text": (0.5, 3.5),
    "retrieval_keywords": (0.5, 3.5),
    "ingredient": (0.0, 2.5),
    "claim_type": (0.0, 2.5),
    "outcome_target": (0.0, 2.5),
}
PARAM_BOUNDS = {
    "k1": (0.6, 2.2),
    "b": (0.1, 0.95),
}


def build_individual(rng: random.Random) -> dict[str, object]:
    field_weights = {
        field: round(rng.uniform(*bounds), 4)
        for field, bounds in FIELD_BOUNDS.items()
    }
    return {
        "k1": round(rng.uniform(*PARAM_BOUNDS["k1"]), 4),
        "b": round(rng.uniform(*PARAM_BOUNDS["b"]), 4),
        "field_weights": field_weights,
    }


def clip_individual(individual: dict[str, object]) -> dict[str, object]:
    clipped = deepcopy(individual)
    clipped["k1"] = round(
        min(PARAM_BOUNDS["k1"][1], max(PARAM_BOUNDS["k1"][0], float(clipped["k1"]))),
        4,
    )
    clipped["b"] = round(
        min(PARAM_BOUNDS["b"][1], max(PARAM_BOUNDS["b"][0], float(clipped["b"]))),
        4,
    )
    field_weights = {}
    raw_weights = clipped.get("field_weights", {})
    for field, bounds in FIELD_BOUNDS.items():
        value = float(raw_weights.get(field, lexical_retriever_v1.BASELINE_RETRIEVER_CONFIG["field_weights"][field]))
        field_weights[field] = round(min(bounds[1], max(bounds[0], value)), 4)
    clipped["field_weights"] = field_weights
    return clipped


def evaluate_config(
    config: dict[str, object],
    benchmark_rows: list[dict[str, object]],
    fragments_rows: list[dict[str, object]],
    top_k: int,
) -> dict[str, object]:
    normalized = lexical_retriever_v1.normalize_retriever_config(config)
    retriever = lexical_retriever_v1.BM25Retriever(fragments_rows, config=normalized)

    hit_count = 0
    reciprocal_rank_sum = 0.0
    query_reports = []
    for row in benchmark_rows:
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
        query_reports.append(
            {
                "query_id": str(row.get("query_id", "")).strip(),
                "returned_fragment_ids": returned_ids,
                "hit": hit,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    total = len(benchmark_rows)
    hit_at_k = round(hit_count / total, 4) if total else 0.0
    mrr = round(reciprocal_rank_sum / total, 4) if total else 0.0
    l1_distance = round(
        sum(abs(float(normalized["field_weights"][field]) - 1.0) for field in lexical_retriever_v1.RETRIEVAL_FIELDS),
        4,
    )
    return {
        "config": normalized,
        "metrics": {
            "hit_at_k": hit_at_k,
            "mrr": mrr,
            "top_k": top_k,
            "queries": total,
        },
        "stability_penalty": l1_distance,
        "query_reports": query_reports,
    }


def score_candidate(
    report: dict[str, object],
    baseline_hit_at_k: float,
) -> float:
    metrics = report["metrics"]
    hit_gap = max(0.0, baseline_hit_at_k - float(metrics["hit_at_k"]))
    stability_penalty = float(report["stability_penalty"])
    return (
        float(metrics["mrr"]) * 1000.0
        - hit_gap * 10000.0
        - stability_penalty
    )


def choose_better(
    left: dict[str, object],
    right: dict[str, object],
    baseline_hit_at_k: float,
) -> dict[str, object]:
    left_hit = float(left["metrics"]["hit_at_k"])
    right_hit = float(right["metrics"]["hit_at_k"])
    if left_hit >= baseline_hit_at_k and right_hit < baseline_hit_at_k:
        return left
    if right_hit >= baseline_hit_at_k and left_hit < baseline_hit_at_k:
        return right

    left_score = score_candidate(left, baseline_hit_at_k)
    right_score = score_candidate(right, baseline_hit_at_k)
    if right_score > left_score:
        return right
    if left_score > right_score:
        return left

    left_mrr = float(left["metrics"]["mrr"])
    right_mrr = float(right["metrics"]["mrr"])
    if right_mrr > left_mrr:
        return right
    if left_mrr > right_mrr:
        return left

    if float(right["stability_penalty"]) < float(left["stability_penalty"]):
        return right
    return left


def tournament_select(
    population_reports: list[dict[str, object]],
    rng: random.Random,
    baseline_hit_at_k: float,
    size: int = 3,
) -> dict[str, object]:
    contenders = rng.sample(population_reports, k=min(size, len(population_reports)))
    winner = contenders[0]
    for contender in contenders[1:]:
        winner = choose_better(winner, contender, baseline_hit_at_k)
    return winner


def crossover(
    parent_a: dict[str, object],
    parent_b: dict[str, object],
    rng: random.Random,
) -> dict[str, object]:
    alpha = rng.uniform(0.3, 0.7)
    child = {
        "k1": round(alpha * float(parent_a["k1"]) + (1 - alpha) * float(parent_b["k1"]), 4),
        "b": round(alpha * float(parent_a["b"]) + (1 - alpha) * float(parent_b["b"]), 4),
        "field_weights": {},
    }
    for field in lexical_retriever_v1.RETRIEVAL_FIELDS:
        child["field_weights"][field] = round(
            alpha * float(parent_a["field_weights"][field]) + (1 - alpha) * float(parent_b["field_weights"][field]),
            4,
        )
    return clip_individual(child)


def mutate(
    individual: dict[str, object],
    rng: random.Random,
    mutation_rate: float,
) -> dict[str, object]:
    mutated = deepcopy(individual)
    if rng.random() < mutation_rate:
        mutated["k1"] = round(float(mutated["k1"]) + rng.uniform(-0.25, 0.25), 4)
    if rng.random() < mutation_rate:
        mutated["b"] = round(float(mutated["b"]) + rng.uniform(-0.12, 0.12), 4)
    for field in lexical_retriever_v1.RETRIEVAL_FIELDS:
        if rng.random() < mutation_rate:
            mutated["field_weights"][field] = round(
                float(mutated["field_weights"][field]) + rng.uniform(-0.45, 0.45),
                4,
            )
    return clip_individual(mutated)


def optimize(
    benchmark_rows: list[dict[str, object]],
    fragments_rows: list[dict[str, object]],
    top_k: int,
    population_size: int,
    generations: int,
    mutation_rate: float,
    elite_count: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    baseline_config = lexical_retriever_v1.default_retriever_config()
    baseline_report = evaluate_config(baseline_config, benchmark_rows, fragments_rows, top_k=top_k)
    baseline_hit_at_k = float(baseline_report["metrics"]["hit_at_k"])

    population = [baseline_config]
    while len(population) < population_size:
        population.append(build_individual(rng))

    best_report = baseline_report
    generation_summaries = []
    for generation in range(generations):
        population_reports = [
            evaluate_config(candidate, benchmark_rows, fragments_rows, top_k=top_k)
            for candidate in population
        ]
        population_reports.sort(
            key=lambda report: score_candidate(report, baseline_hit_at_k),
            reverse=True,
        )

        current_best = population_reports[0]
        best_report = choose_better(best_report, current_best, baseline_hit_at_k)
        generation_summaries.append(
            {
                "generation": generation,
                "best_hit_at_k": current_best["metrics"]["hit_at_k"],
                "best_mrr": current_best["metrics"]["mrr"],
            }
        )

        next_population = [report["config"] for report in population_reports[:elite_count]]
        while len(next_population) < population_size:
            parent_a = tournament_select(population_reports, rng, baseline_hit_at_k)["config"]
            parent_b = tournament_select(population_reports, rng, baseline_hit_at_k)["config"]
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, rng, mutation_rate=mutation_rate)
            next_population.append(child)
        population = next_population

    selected_report = choose_better(baseline_report, best_report, baseline_hit_at_k)
    return {
        "artifact_type": "retriever_optimization_result",
        "algorithm": "genetic_search_v1",
        "selected_config": selected_report["config"],
        "baseline_metrics": baseline_report["metrics"],
        "optimized_metrics": selected_report["metrics"],
        "search_settings": {
            "population_size": population_size,
            "generations": generations,
            "mutation_rate": mutation_rate,
            "elite_count": elite_count,
            "seed": seed,
        },
        "generation_summaries": generation_summaries,
    }


def _load_rows(
    fragments_csv: str | Path,
    benchmark_csv: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fragments_df = lexical_retriever_v1.load_fragments(fragments_csv).fillna("")
    benchmark_df = pd.read_csv(benchmark_csv).fillna("")
    return fragments_df.to_dict(orient="records"), benchmark_df.to_dict(orient="records")


def _print_summary(label: str, report: dict[str, object]) -> None:
    metrics = report["metrics"]
    print(
        json.dumps(
            {
                "label": label,
                "hit_at_k": metrics["hit_at_k"],
                "mrr": metrics["mrr"],
                "top_k": metrics["top_k"],
                "queries": metrics["queries"],
            },
            indent=2,
        )
    )


def cmd_baseline(args: argparse.Namespace) -> None:
    fragments_rows, benchmark_rows = _load_rows(args.fragments_csv, args.benchmark_csv)
    baseline_report = evaluate_config(
        lexical_retriever_v1.default_retriever_config(),
        benchmark_rows,
        fragments_rows,
        top_k=args.top_k,
    )
    _print_summary("baseline", baseline_report)


def cmd_compare(args: argparse.Namespace) -> None:
    fragments_rows, benchmark_rows = _load_rows(args.fragments_csv, args.benchmark_csv)
    baseline_report = evaluate_config(
        lexical_retriever_v1.default_retriever_config(),
        benchmark_rows,
        fragments_rows,
        top_k=args.top_k,
    )
    optimized_config = lexical_retriever_v1.load_retriever_config(args.config)
    optimized_report = evaluate_config(optimized_config, benchmark_rows, fragments_rows, top_k=args.top_k)
    print(
        json.dumps(
            {
                "baseline": baseline_report["metrics"],
                "optimized": optimized_report["metrics"],
                "selected_config": optimized_report["config"],
            },
            indent=2,
        )
    )


def cmd_optimize(args: argparse.Namespace) -> None:
    fragments_rows, benchmark_rows = _load_rows(args.fragments_csv, args.benchmark_csv)
    artifact = optimize(
        benchmark_rows=benchmark_rows,
        fragments_rows=fragments_rows,
        top_k=args.top_k,
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        elite_count=args.elite_count,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)

    print(json.dumps(artifact, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genetic optimization for retriever ranking.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("baseline", "compare", "optimize"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--fragments-csv", default=str(DEFAULT_FRAGMENTS_CSV))
        subparser.add_argument("--benchmark-csv", default=str(DEFAULT_RETRIEVAL_BENCHMARK_CSV))
        subparser.add_argument("--top-k", type=int, default=5)
        if name == "compare":
            subparser.add_argument("--config", default=str(DEFAULT_OUTPUT_PATH))
        if name == "optimize":
            subparser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
            subparser.add_argument("--population-size", type=int, default=24)
            subparser.add_argument("--generations", type=int, default=18)
            subparser.add_argument("--mutation-rate", type=float, default=0.35)
            subparser.add_argument("--elite-count", type=int, default=4)
            subparser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "baseline":
        cmd_baseline(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
