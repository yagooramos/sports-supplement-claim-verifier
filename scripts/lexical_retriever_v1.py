#!/usr/bin/env python3
"""
Lexical retriever v1 for the project corpus.

What it does
------------
- Loads evidence fragments from a CSV
- Builds a lexical index over fragment_text + retrieval_keywords + metadata
- Scores with BM25 (default k1=1.2, b=0.75)
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


class BM25Retriever:
    def __init__(self, rows: list[dict], k1: float = 1.2, b: float = 0.75):
        self.rows = rows
        self.k1 = k1
        self.b = b
        self.doc_tokens = [self._row_to_tokens(r) for r in rows]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_doc_freq = self._build_doc_freq()
        self.N = len(self.rows)

    @staticmethod
    def _row_to_tokens(row: dict) -> list[str]:
        # Weight lexical retrieval toward actual evidence text and retrieval keywords,
        # but include metadata so simple claim-like queries can still hit.
        combined = " ".join(
            [
                str(row.get("fragment_text", "")),
                str(row.get("retrieval_keywords", "")),
                str(row.get("ingredient", "")),
                str(row.get("claim_type", "")),
                str(row.get("outcome_target", "")),
                str(row.get("matrix_id", "")),
            ]
        )
        return tokenize(combined)

    def _build_doc_freq(self) -> dict[str, int]:
        df = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        return dict(df)

    def idf(self, term: str) -> float:
        n = self.term_doc_freq.get(term, 0)
        # BM25 IDF with +1 safeguard
        return math.log(1 + ((self.N - n + 0.5) / (n + 0.5)))

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

            tf = Counter(self.doc_tokens[i])
            dl = self.doc_lens[i]
            score = 0.0

            for term in q_terms:
                if term not in tf:
                    continue
                f = tf[term]
                idf = self.idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                score += idf * ((f * (self.k1 + 1)) / denom)

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
    args = parser.parse_args()

    df = load_fragments(args.csv)
    retriever = BM25Retriever(df.to_dict(orient="records"))
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
