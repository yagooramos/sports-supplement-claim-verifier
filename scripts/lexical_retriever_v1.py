#!/usr/bin/env python3
"""
Lexical retriever v1 for the project corpus.

What it does
------------
- Loads evidence fragments from a CSV
- Builds a lexical index over evidence text plus structured metadata fields
- Scores with configurable field-weighted BM25
- Supports optional filtering by ingredient, claim_type, matrix_id, outcome_target
- Returns top-k results with scores

Usage
-----
Run from the project root:
python scripts/lexical_retriever_v1.py --csv data/annotations/evidence_fragments.csv --query "creatine strength"
python scripts/lexical_retriever_v1.py --csv data/annotations/evidence_fragments.csv --query "reduce fatigue caffeine" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .utils import normalize_text, tokenize  # re-exported for backward compatibility
except ImportError:
    from utils import normalize_text, tokenize  # re-exported for backward compatibility


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RETRIEVER_CONFIG_PATH = PROJECT_ROOT / "models" / "retriever_optimized_config.json"
RETRIEVAL_FIELDS = [
    "fragment_text",
    "retrieval_keywords",
    "ingredient",
    "claim_type",
    "outcome_target",
]
BASELINE_RETRIEVER_CONFIG = {
    "k1": 1.2,
    "b": 0.75,
    "field_weights": {
        "fragment_text": 1.0,
        "retrieval_keywords": 1.0,
        "ingredient": 1.0,
        "claim_type": 1.0,
        "outcome_target": 1.0,
    },
}


@dataclass
class SearchResult:
    rank: int
    score: float
    fragment_id: str
    doc_id: str
    matrix_id: str
    ingredient: str
    claim_type: str
    outcome_target: str
    supports_claim: str
    support_strength: str
    fragment_text: str


def default_retriever_config() -> dict[str, object]:
    return {
        "k1": float(BASELINE_RETRIEVER_CONFIG["k1"]),
        "b": float(BASELINE_RETRIEVER_CONFIG["b"]),
        "field_weights": dict(BASELINE_RETRIEVER_CONFIG["field_weights"]),
    }


def normalize_retriever_config(config: dict[str, object] | None = None) -> dict[str, object]:
    normalized = default_retriever_config()
    if not config:
        return normalized

    payload = dict(config)
    if isinstance(payload.get("selected_config"), dict):
        payload = dict(payload["selected_config"])

    try:
        normalized["k1"] = float(payload.get("k1", normalized["k1"]))
    except (TypeError, ValueError):
        normalized["k1"] = float(BASELINE_RETRIEVER_CONFIG["k1"])
    normalized["k1"] = max(0.1, normalized["k1"])

    try:
        normalized["b"] = float(payload.get("b", normalized["b"]))
    except (TypeError, ValueError):
        normalized["b"] = float(BASELINE_RETRIEVER_CONFIG["b"])
    normalized["b"] = min(1.0, max(0.0, normalized["b"]))

    raw_weights = payload.get("field_weights", {})
    field_weights = dict(normalized["field_weights"])
    if isinstance(raw_weights, dict):
        for field in RETRIEVAL_FIELDS:
            try:
                field_weights[field] = max(0.0, float(raw_weights.get(field, field_weights[field])))
            except (TypeError, ValueError):
                continue
    normalized["field_weights"] = field_weights
    return normalized


def load_retriever_config(config_path: str | Path = DEFAULT_RETRIEVER_CONFIG_PATH) -> dict[str, object]:
    path = Path(config_path)
    if not path.exists():
        return default_retriever_config()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return normalize_retriever_config(payload)


class BM25Retriever:
    def __init__(
        self,
        rows: list[dict],
        k1: float | None = None,
        b: float | None = None,
        field_weights: dict[str, float] | None = None,
        config: dict[str, object] | None = None,
    ):
        self.rows = rows
        config_payload = dict(config or {})
        if k1 is not None:
            config_payload["k1"] = k1
        if b is not None:
            config_payload["b"] = b
        if field_weights is not None:
            config_payload["field_weights"] = field_weights
        self.config = normalize_retriever_config(config_payload)
        self.k1 = float(self.config["k1"])
        self.b = float(self.config["b"])
        self.field_weights = dict(self.config["field_weights"])

        self.field_doc_tokens = {
            field: [self._field_to_tokens(r, field) for r in rows] for field in RETRIEVAL_FIELDS
        }
        self.field_doc_lens = {
            field: [len(tokens) for tokens in self.field_doc_tokens[field]] for field in RETRIEVAL_FIELDS
        }
        self.field_avgdl = {
            field: sum(lengths) / max(1, len(lengths)) for field, lengths in self.field_doc_lens.items()
        }
        self.field_term_doc_freq = {
            field: self._build_doc_freq(self.field_doc_tokens[field]) for field in RETRIEVAL_FIELDS
        }
        self.N = len(self.rows)

    @staticmethod
    def _field_to_tokens(row: dict, field: str) -> list[str]:
        value = str(row.get(field, ""))
        if field == "retrieval_keywords":
            value = value.replace("|", " ")
        elif field in {"claim_type", "outcome_target", "ingredient"}:
            value = value.replace("_", " ")
        return tokenize(value)

    def _build_doc_freq(self, field_tokens: list[list[str]]) -> dict[str, int]:
        df = Counter()
        for toks in field_tokens:
            df.update(set(toks))
        return dict(df)

    def idf(self, term: str, field: str) -> float:
        n = self.field_term_doc_freq[field].get(term, 0)
        # BM25 IDF with +1 safeguard
        return math.log(1 + ((self.N - n + 0.5) / (n + 0.5)))

    def _score_field(self, doc_index: int, field: str, q_terms: list[str]) -> float:
        weight = float(self.field_weights.get(field, 0.0))
        if weight <= 0:
            return 0.0

        tf = Counter(self.field_doc_tokens[field][doc_index])
        if not tf:
            return 0.0

        dl = self.field_doc_lens[field][doc_index]
        avgdl = max(self.field_avgdl.get(field, 0.0), 1e-9)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            term_frequency = tf[term]
            idf = self.idf(term, field)
            denom = term_frequency + self.k1 * (1 - self.b + self.b * (dl / avgdl))
            score += weight * idf * ((term_frequency * (self.k1 + 1)) / denom)
        return score

    def _row_passes_filters(self, row: dict, filters: dict[str, str] | None) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if value is None or value == "":
                continue
            if str(row.get(key, "")).strip().lower() != str(value).strip().lower():
                return False
        return True

    def search(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[SearchResult]:
        q_terms = tokenize(query)
        scored = []

        for i, row in enumerate(self.rows):
            if not self._row_passes_filters(row, filters):
                continue

            score = sum(self._score_field(i, field, q_terms) for field in RETRIEVAL_FIELDS)

            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, row) in enumerate(scored[:top_k], start=1):
            results.append(
                SearchResult(
                    rank=rank,
                    score=round(score, 4),
                    fragment_id=str(row.get("fragment_id", "")),
                    doc_id=str(row.get("doc_id", "")),
                    matrix_id=str(row.get("matrix_id", "")),
                    ingredient=str(row.get("ingredient", "")),
                    claim_type=str(row.get("claim_type", "")),
                    outcome_target=str(row.get("outcome_target", "")),
                    supports_claim=str(row.get("supports_claim", "")),
                    support_strength=str(row.get("support_strength", "")),
                    fragment_text=str(row.get("fragment_text", "")),
                )
            )
        return results


def load_fragments(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "fragment_id", "doc_id", "matrix_id", "ingredient", "claim_type",
        "outcome_target", "fragment_text", "supports_claim", "support_strength",
        "retrieval_keywords",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.fillna("")
    return df


def print_results(results: Iterable[SearchResult]) -> None:
    results = list(results)
    if not results:
        print("No results found.")
        return

    for r in results:
        print("=" * 80)
        print(f"rank={r.rank} score={r.score}")
        print(f"fragment_id={r.fragment_id} doc_id={r.doc_id} matrix_id={r.matrix_id}")
        print(f"ingredient={r.ingredient} claim_type={r.claim_type} outcome_target={r.outcome_target}")
        print(f"supports_claim={r.supports_claim} support_strength={r.support_strength}")
        print(f"fragment_text={r.fragment_text}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BM25 lexical retriever v1 for the project corpus."
    )
    parser.add_argument("--csv", required=True, help="Path to evidence fragments CSV")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k results to return")
    parser.add_argument("--ingredient", default="", help="Optional exact filter")
    parser.add_argument("--claim-type", default="", help="Optional exact filter")
    parser.add_argument("--matrix-id", default="", help="Optional exact filter")
    parser.add_argument("--outcome-target", default="", help="Optional exact filter")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_RETRIEVER_CONFIG_PATH),
        help="Path to retriever config JSON. Falls back to baseline defaults if missing.",
    )
    args = parser.parse_args()

    df = load_fragments(args.csv)
    retriever = BM25Retriever(df.to_dict(orient="records"), config=load_retriever_config(args.config))
    filters = {
        "ingredient": args.ingredient,
        "claim_type": args.claim_type,
        "matrix_id": args.matrix_id,
        "outcome_target": args.outcome_target,
    }
    results = retriever.search(args.query, top_k=args.top_k, filters=filters)
    print_results(results)


if __name__ == "__main__":
    main()
