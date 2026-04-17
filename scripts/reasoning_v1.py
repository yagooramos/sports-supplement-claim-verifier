#!/usr/bin/env python3
"""
Deterministic reasoning layer for reasoning v1.

The reasoner consumes:
- one structured claim parse
- one enriched top-5 retrieval bundle
- one corpus coverage map

It does not parse raw text and it does not run retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

try:
    from .utils import (
        normalize_text,
        split_pipe_values,
        text_has_phrase as has_phrase,
        unique_preserving_order,
    )
except ImportError:
    from utils import (
        normalize_text,
        split_pipe_values,
        text_has_phrase as has_phrase,
        unique_preserving_order,
    )


EXAGGERATION_CUES = [
    "huge",
    "massive",
    "guarantee",
    "guaranteed",
    "guarantees",
    "fast",
    "immediate",
    "immediately",
    "almost immediately",
    "overnight",
    "extreme",
    "unstoppable",
    "for everyone",
    "always",
    "any sport",
    "build muscle fast",
    "burn fat fast",
]
UNIVERSAL_CUES = ["for everyone", "always", "any sport"]


def has_negated_phrase(text: str, phrase: str) -> bool:
    phrase_pattern = re.escape(normalize_text(phrase)).replace(r"\ ", r"[-\s]+")
    negation_prefixes = [
        r"not",
        r"no",
        r"never",
        r"does\s+not",
        r"doesn't",
        r"do\s+not",
        r"don't",
    ]
    for prefix in negation_prefixes:
        pattern = re.compile(rf"(?<!\w){prefix}(?:\s+\w+){{0,3}}\s+{phrase_pattern}(?!\w)")
        if pattern.search(text):
            return True
    return False


def is_meaningful_text(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return normalize_text(text) not in {"not specified in this extracted fragment", "not standardized in this extracted fragment", "mixed"}


def build_corpus_coverage(
    matrix_scope_csv: str | Path,
    fragments_csv: str | Path,
) -> dict[str, dict[str, object]]:
    matrix_scope_df = pd.read_csv(matrix_scope_csv).fillna("")
    fragments_df = pd.read_csv(fragments_csv).fillna("")

    fragments_by_matrix: dict[str, list[str]] = {}
    for row in fragments_df.to_dict(orient="records"):
        matrix_id = str(row.get("matrix_id", "")).strip()
        fragment_id = str(row.get("fragment_id", "")).strip()
        if not matrix_id or not fragment_id:
            continue
        fragments_by_matrix.setdefault(matrix_id, []).append(fragment_id)

    coverage = {}
    for row in matrix_scope_df.to_dict(orient="records"):
        matrix_id = str(row.get("matrix_id", "")).strip()
        fragment_ids = unique_preserving_order(fragments_by_matrix.get(matrix_id, []))
        coverage[matrix_id] = {
            "matrix_id": matrix_id,
            "ingredient": str(row.get("ingredient", "")).strip(),
            "claim_type": str(row.get("claim_type", "")).strip(),
            "outcome_target": str(row.get("outcome_target", "")).strip(),
            "scope_status": str(row.get("scope_status", "")).strip(),
            "fragment_ids": fragment_ids,
            "coverage_status": "covered" if fragment_ids else "coverage_gap",
        }
    return coverage


def resolve_matrix_id(
    claim_parse: dict[str, object],
    corpus_coverage: dict[str, dict[str, object]],
) -> str:
    ingredient = str(claim_parse.get("ingredient", "")).strip()
    claim_type = str(claim_parse.get("claim_type", "")).strip()
    outcome_target = str(claim_parse.get("outcome_target", "")).strip()
    for matrix_id, row in corpus_coverage.items():
        if (
            ingredient == str(row.get("ingredient", "")).strip()
            and claim_type == str(row.get("claim_type", "")).strip()
            and outcome_target == str(row.get("outcome_target", "")).strip()
        ):
            return matrix_id
    return ""


def candidate_blob(candidate: dict[str, object]) -> str:
    combined = " ".join(
        [
            str(candidate.get("fragment_text", "")),
            str(candidate.get("retrieval_keywords", "")),
            str(candidate.get("conditions_or_limits", "")),
            str(candidate.get("annotation_notes", "")),
            str(candidate.get("dose_context", "")),
            str(candidate.get("population_context", "")),
        ]
    )
    return normalize_text(combined)
def exact_top5_candidates(
    retrieval_bundle: dict[str, object],
    matched_matrix_id: str,
) -> list[dict[str, object]]:
    candidates = retrieval_bundle.get("retrieved_candidates", [])
    exact_candidates = []
    for candidate in candidates[:5]:
        if str(candidate.get("matrix_id", "")).strip() == matched_matrix_id:
            exact_candidates.append(candidate)
    return exact_candidates


def evidence_family_is_partial(matched_matrix_id: str, exact_candidates: list[dict[str, object]]) -> bool:
    if matched_matrix_id in {"M03", "M07", "M08"}:
        return True
    return any(
        str(candidate.get("supports_claim", "")).strip().lower() == "partial"
        or str(candidate.get("support_strength", "")).strip().lower() == "weak"
        for candidate in exact_candidates
    )


def claim_states_key_limit(claim_parse: dict[str, object], matched_matrix_id: str) -> bool:
    claim_text = normalize_text(claim_parse.get("claim_text_normalized", claim_parse.get("claim_text_raw", "")))
    dose_context = str(claim_parse.get("dose_context", "")).strip()
    duration_context = str(claim_parse.get("duration_context", "")).strip()

    if matched_matrix_id == "M06":
        return "4-6 g/day" in dose_context and bool(duration_context)
    if matched_matrix_id == "M05":
        return "moderate doses" in dose_context or "3-6 mg/kg" in dose_context
    if matched_matrix_id == "M08":
        plateau_phrases = [
            "not always better",
            "does not keep rising",
            "doesnt keep rising",
            "does not keep adding benefit",
            "doesnt keep adding benefit",
            "stops increasing",
            "benefit plateaus",
            "benefits plateau",
            "plateaus",
        ]
        has_plateau_phrase = any(has_phrase(claim_text, phrase) for phrase in plateau_phrases)
        has_limit_dose = "1.6 g/kg/day" in dose_context or has_phrase(claim_text, "1.6 g/kg/day") or has_phrase(
            claim_text, "1.6 g/kg per day"
        )
        if has_phrase(claim_text, "not always better"):
            return True
        return has_plateau_phrase and has_limit_dose
    return False


def dose_condition_not_met(
    claim_parse: dict[str, object],
    matched_matrix_id: str,
    exact_candidates: list[dict[str, object]],
) -> bool:
    claim_text = normalize_text(claim_parse.get("claim_text_normalized", claim_parse.get("claim_text_raw", "")))
    if matched_matrix_id == "M06":
        has_loading_requirement = any("4 6 g day" in candidate_blob(candidate) for candidate in exact_candidates)
        if has_loading_requirement and not claim_states_key_limit(claim_parse, matched_matrix_id):
            return True
        if any(
            has_phrase(claim_text, phrase)
            for phrase in [
                "without loading period",
                "without loading phase",
                "without loading weeks",
                "skip the loading weeks",
                "without buildup weeks",
                "without build up weeks",
                "skip the buildup weeks",
                "skip the build up weeks",
            ]
        ):
            return True
    if matched_matrix_id == "M05":
        has_moderate_dose_requirement = any(
            "moderate doses" in str(candidate.get("dose_context", "")).lower()
            or "3-6 mg/kg" in str(candidate.get("dose_context", ""))
            for candidate in exact_candidates
        )
        if any(
            has_phrase(claim_text, phrase)
            for phrase in [
                "any dose",
                "at any dose",
                "no matter the dose",
                "regardless of dose",
                "whatever the dose",
            ]
        ):
            return True
        if has_moderate_dose_requirement and not claim_states_key_limit(claim_parse, matched_matrix_id):
            return True
    if matched_matrix_id == "M08":
        if "always better" in claim_text and "not always better" not in claim_text:
            return True
    return False


def marketing_exaggeration(
    claim_parse: dict[str, object],
    matched_matrix_id: str,
    exact_candidates: list[dict[str, object]],
) -> bool:
    claim_text = normalize_text(claim_parse.get("claim_text_normalized", claim_parse.get("claim_text_raw", "")))
    active_cues = list(EXAGGERATION_CUES)
    if has_phrase(claim_text, "not always better"):
        active_cues = [cue for cue in active_cues if cue != "always"]
    active_cues = [cue for cue in active_cues if not has_negated_phrase(claim_text, cue)]

    has_exaggeration = any(has_phrase(claim_text, cue) for cue in active_cues)
    if not has_exaggeration:
        return False
    if matched_matrix_id in {"M03", "M07", "M08"}:
        return True
    if matched_matrix_id == "M04" and any(has_phrase(claim_text, cue) for cue in ["extreme", "unstoppable"]):
        return True
    if matched_matrix_id in {"M01", "M02", "M05", "M06"} and any(
        has_phrase(claim_text, cue) for cue in ["guaranteed", "for everyone", "always"]
    ):
        return True
    return False


def limitation_penalty(candidate: dict[str, object]) -> int:
    blob = candidate_blob(candidate)
    annotation_notes = normalize_text(candidate.get("annotation_notes", ""))
    penalty = 0
    penalty_phrases = [
        "limitation fragment",
        "no additional benefit",
        "benefit plateaus",
        "benefit plateau",
        "plateau",
        "response varies",
        "indirect evidence",
        "effect is small",
        "equivocal",
        "mixed",
        "does not justify",
    ]
    for phrase in penalty_phrases:
        if has_phrase(blob, phrase) or has_phrase(annotation_notes, phrase):
            penalty += 4
    return penalty


def looks_like_limiting_fragment(candidate: dict[str, object]) -> bool:
    blob = candidate_blob(candidate)
    annotation_notes = normalize_text(candidate.get("annotation_notes", ""))
    return any(
        has_phrase(blob, phrase) or has_phrase(annotation_notes, phrase)
        for phrase in [
            "limitation fragment",
            "no additional benefit",
            "response varies",
            "useful for constraining claims",
        ]
    )


def support_score(candidate: dict[str, object], claim_text: str, prefer_non_limiting: bool = False) -> int:
    score = 0
    supports_claim = str(candidate.get("supports_claim", "")).strip().lower()
    support_strength = str(candidate.get("support_strength", "")).strip().lower()
    blob = candidate_blob(candidate)

    if supports_claim == "yes":
        score += 20
    elif supports_claim == "partial":
        score += 10

    if support_strength == "strong":
        score += 6
    elif support_strength == "moderate":
        score += 4
    elif support_strength == "weak":
        score += 2

    for phrase in [
        "strength",
        "fatigue",
        "endurance",
        "recovery",
        "lean mass",
        "small",
        "exercise capacity",
        "moderate",
    ]:
        if has_phrase(claim_text, phrase) and has_phrase(blob, phrase):
            score += 2

    if prefer_non_limiting and (
        has_phrase(blob, "3 6 mg kg")
        or has_phrase(blob, "4 6 g day")
        or has_phrase(blob, "1 6 g kg day")
    ):
        score -= 1
        score -= limitation_penalty(candidate)
    return score


def select_primary_support_fragment(
    exact_candidates: list[dict[str, object]],
    claim_text: str,
    prefer_non_limiting: bool = False,
) -> list[str]:
    if not exact_candidates:
        return []
    candidate_pool = exact_candidates
    if prefer_non_limiting:
        non_limiting_candidates = [candidate for candidate in exact_candidates if not looks_like_limiting_fragment(candidate)]
        if non_limiting_candidates:
            candidate_pool = non_limiting_candidates
    ranked = sorted(
        candidate_pool,
        key=lambda candidate: (
            -support_score(candidate, claim_text, prefer_non_limiting=prefer_non_limiting),
            int(candidate.get("rank", 9999)),
        ),
    )
    fragment_id = str(ranked[0].get("fragment_id", "")).strip()
    return [fragment_id] if fragment_id else []


def select_partial_support_fragments(
    matched_matrix_id: str,
    exact_candidates: list[dict[str, object]],
    claim_text: str,
) -> list[str]:
    fragment_ids = []
    partial_candidates = []
    for candidate in exact_candidates:
        supports_claim = str(candidate.get("supports_claim", "")).strip().lower()
        support_strength = str(candidate.get("support_strength", "")).strip().lower()
        if supports_claim == "partial" or support_strength == "weak":
            partial_candidates.append(candidate)
            fragment_ids.append(str(candidate.get("fragment_id", "")).strip())

    if matched_matrix_id == "M03" and has_phrase(claim_text, "small"):
        ranked = sorted(
            partial_candidates,
            key=lambda candidate: (
                -support_score(candidate, claim_text, prefer_non_limiting=False),
                int(candidate.get("rank", 9999)),
            ),
        )
        if ranked:
            fragment_id = str(ranked[0].get("fragment_id", "")).strip()
            return [fragment_id] if fragment_id else []

    if fragment_ids:
        return unique_preserving_order(fragment_ids)
    return select_primary_support_fragment(
        exact_candidates,
        claim_text,
        prefer_non_limiting=False,
    )


def select_limiting_fragments(
    matched_matrix_id: str,
    exact_candidates: list[dict[str, object]],
    exclude_fragment_ids: list[str] | None = None,
) -> list[str]:
    exclude_fragment_ids = set(exclude_fragment_ids or [])
    preferred_phrases = {
        "M03": ["small", "partial", "no broad claims"],
        "M05": ["moderate", "3 6 mg kg", "response varies"],
        "M06": ["4 6 g day", "2 4 weeks", "not universal"],
        "M07": ["mixed", "equivocal", "indirect"],
        "M08": ["1 6 g kg day", "no additional benefit", "plateau"],
    }.get(matched_matrix_id, [])

    fragment_ids = []
    for candidate in exact_candidates:
        fragment_id = str(candidate.get("fragment_id", "")).strip()
        if not fragment_id or fragment_id in exclude_fragment_ids:
            continue
        blob = candidate_blob(candidate)
        if any(has_phrase(blob, phrase) for phrase in preferred_phrases):
            fragment_ids.append(fragment_id)

    if fragment_ids:
        return unique_preserving_order(fragment_ids)

    for candidate in exact_candidates:
        fragment_id = str(candidate.get("fragment_id", "")).strip()
        if not fragment_id or fragment_id in exclude_fragment_ids:
            continue
        if is_meaningful_text(candidate.get("dose_context")) or is_meaningful_text(candidate.get("conditions_or_limits")):
            fragment_ids.append(fragment_id)

    return unique_preserving_order(fragment_ids)


def conditions_to_state(
    matched_matrix_id: str,
    claim_parse: dict[str, object],
    exact_candidates: list[dict[str, object]],
) -> list[str]:
    claim_text = normalize_text(claim_parse.get("claim_text_normalized", claim_parse.get("claim_text_raw", "")))
    conditions = []
    if matched_matrix_id == "M03":
        conditions.append("effect is small")
    if matched_matrix_id == "M05":
        conditions.append("moderate doses")
    if matched_matrix_id == "M06":
        conditions.append("4-6 g/day for at least 2-4 weeks")
        if any(has_phrase(claim_text, cue) for cue in UNIVERSAL_CUES):
            conditions.append("not universal across all sports")
    if matched_matrix_id == "M07":
        conditions.append("evidence is mixed and partly indirect")
    if matched_matrix_id == "M08":
        conditions.append("benefit plateaus beyond about 1.6 g/kg/day total protein")
    return unique_preserving_order(conditions)


def base_result(parse_status: str) -> dict[str, object]:
    return {
        "parse_status": parse_status,
        "scope_status": "not_applicable",
        "coverage_status": "not_applicable",
        "verdict": "",
        "reason_code": "",
        "matched_matrix_id": "",
        "supporting_fragment_ids": [],
        "limiting_fragment_ids": [],
        "conditions_to_state": [],
        "explanation": "",
    }


def evaluate_claim(
    claim_parse: dict[str, object],
    retrieval_bundle: dict[str, object],
    corpus_coverage: dict[str, dict[str, object]],
) -> dict[str, object]:
    parse_status = str(claim_parse.get("parse_status", "not_parseable")).strip()
    result = base_result(parse_status)

    if parse_status != "fully_parseable":
        result["verdict"] = "not_evaluable"
        result["reason_code"] = "claim_not_parseable"
        result["explanation"] = "Parser could not resolve one stable canonical target."
        return result

    matched_matrix_id = resolve_matrix_id(claim_parse, corpus_coverage)
    if not matched_matrix_id:
        result["scope_status"] = "out_of_scope"
        result["verdict"] = "not_evaluable"
        result["reason_code"] = "claim_outside_scope"
        result["explanation"] = "Parsed tuple is outside the current canonical matrix."
        return result

    result["scope_status"] = "in_scope"
    result["matched_matrix_id"] = matched_matrix_id

    coverage_row = corpus_coverage[matched_matrix_id]
    coverage_status = str(coverage_row.get("coverage_status", "coverage_gap")).strip()
    result["coverage_status"] = coverage_status

    if coverage_status == "coverage_gap":
        result["verdict"] = "insufficient_evidence"
        result["reason_code"] = "insufficient_corpus_support"
        result["explanation"] = f"{matched_matrix_id} is in scope but has no canonical fragments in the current corpus."
        return result

    exact_candidates = exact_top5_candidates(retrieval_bundle, matched_matrix_id)
    if not exact_candidates:
        result["verdict"] = "insufficient_evidence"
        result["reason_code"] = "insufficient_corpus_support"
        result["explanation"] = (
            f"{matched_matrix_id} is covered in the corpus, but the retrieved top-5 contained no exact-matrix evidence."
        )
        return result

    claim_text = normalize_text(claim_parse.get("claim_text_normalized", claim_parse.get("claim_text_raw", "")))
    partial_family = evidence_family_is_partial(matched_matrix_id, exact_candidates)

    if dose_condition_not_met(claim_parse, matched_matrix_id, exact_candidates):
        limiting_fragment_ids = select_limiting_fragments(matched_matrix_id, exact_candidates)
        result["verdict"] = "potentially_misleading"
        result["reason_code"] = "dose_condition_not_met"
        result["limiting_fragment_ids"] = limiting_fragment_ids
        result["conditions_to_state"] = conditions_to_state(matched_matrix_id, claim_parse, exact_candidates)
        result["explanation"] = (
            f"{matched_matrix_id} has exact evidence, but the claim omits or contradicts a critical dose or duration condition."
        )
        return result

    if marketing_exaggeration(claim_parse, matched_matrix_id, exact_candidates):
        limiting_fragment_ids = select_limiting_fragments(matched_matrix_id, exact_candidates)
        result["verdict"] = "potentially_misleading"
        result["reason_code"] = "marketing_exaggeration"
        result["limiting_fragment_ids"] = limiting_fragment_ids
        result["conditions_to_state"] = conditions_to_state(matched_matrix_id, claim_parse, exact_candidates)
        result["explanation"] = (
            f"{matched_matrix_id} has only limited exact support, and the claim overstates magnitude, certainty, or speed."
        )
        return result

    if claim_states_key_limit(claim_parse, matched_matrix_id):
        supporting_fragment_ids = select_primary_support_fragment(
            exact_candidates,
            claim_text,
            prefer_non_limiting=True,
        )
        limiting_fragment_ids = select_limiting_fragments(
            matched_matrix_id,
            exact_candidates,
            exclude_fragment_ids=supporting_fragment_ids,
        )
        result["supporting_fragment_ids"] = supporting_fragment_ids
        result["limiting_fragment_ids"] = limiting_fragment_ids
        result["conditions_to_state"] = conditions_to_state(matched_matrix_id, claim_parse, exact_candidates)
        result["verdict"] = "partially_backed" if partial_family else "backed"
        result["reason_code"] = "support_with_limitations"
        result["explanation"] = (
            f"{matched_matrix_id} is supported when the claim keeps the key limitation stated in the exact evidence."
        )
        return result

    if partial_family:
        supporting_fragment_ids = select_partial_support_fragments(
            matched_matrix_id,
            exact_candidates,
            claim_text,
        )
        result["supporting_fragment_ids"] = supporting_fragment_ids
        result["conditions_to_state"] = conditions_to_state(matched_matrix_id, claim_parse, exact_candidates)
        result["verdict"] = "partially_backed"
        result["reason_code"] = "partial_or_indirect_support"
        result["explanation"] = f"{matched_matrix_id} has exact evidence, but the support is partial or indirect."
        return result

    supporting_fragment_ids = select_primary_support_fragment(exact_candidates, claim_text)
    result["supporting_fragment_ids"] = supporting_fragment_ids
    result["verdict"] = "backed"
    result["reason_code"] = "direct_strong_support"
    result["explanation"] = f"{matched_matrix_id} has exact top-5 evidence with direct support."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reasoning v1 on one JSON payload.")
    parser.add_argument("--claim-parse", required=True, help="JSON string for the parsed claim")
    parser.add_argument("--retrieval-bundle", required=True, help="JSON string for the retrieval bundle")
    parser.add_argument(
        "--matrix-scope",
        default="data/sources/matrix_scope.csv",
        help="Path to matrix scope CSV",
    )
    parser.add_argument(
        "--fragments",
        default="data/annotations/evidence_fragments.csv",
        help="Path to fragments CSV",
    )
    args = parser.parse_args()

    claim_parse = json.loads(args.claim_parse)
    retrieval_bundle = json.loads(args.retrieval_bundle)
    corpus_coverage = build_corpus_coverage(args.matrix_scope, args.fragments)
    result = evaluate_claim(claim_parse, retrieval_bundle, corpus_coverage)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
