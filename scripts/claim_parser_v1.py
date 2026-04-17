#!/usr/bin/env python3
"""
Shallow deterministic claim parser for reasoning v1.

This parser is intentionally narrow. It supports the documented reasoning v1
case set and close lexical variants. It prefers abstention over false
precision.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
try:
    from .utils import (
        normalize_text,
        split_pipe_values,
        text_has_any_phrase,
        text_has_phrase,
        unique_preserving_order,
    )
except ImportError:
    from utils import (
        normalize_text,
        split_pipe_values,
        text_has_any_phrase,
        text_has_phrase,
        unique_preserving_order,
    )

DEFAULT_RULES_PATH = SCRIPT_DIR.parent / "data" / "config" / "claim_parser_rules.json"


def load_parser_rules(config_path: str | Path = DEFAULT_RULES_PATH) -> dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


PARSER_RULES = load_parser_rules()
NEGATION_CUES = PARSER_RULES["NEGATION_CUES"]
NEGATION_OPTIONAL_FILLERS = PARSER_RULES["NEGATION_OPTIONAL_FILLERS"]
GENERIC_POSITIVE_CUES = PARSER_RULES["GENERIC_POSITIVE_CUES"]
EXAGGERATION_CUES = PARSER_RULES["EXAGGERATION_CUES"]
CAUTIOUS_CUES = PARSER_RULES["CAUTIOUS_CUES"]
CONJUNCTION_CUES = PARSER_RULES["CONJUNCTION_CUES"]
OUT_OF_SCOPE_INGREDIENT_CUES = PARSER_RULES["OUT_OF_SCOPE_INGREDIENT_CUES"]
OUT_OF_SCOPE_TUPLE_RULES = PARSER_RULES["OUT_OF_SCOPE_TUPLE_RULES"]
MATRIX_HINTS = PARSER_RULES["MATRIX_HINTS"]
MATRIX_HINT_WEIGHTS = PARSER_RULES["MATRIX_HINT_WEIGHTS"]


class ClaimParserV1:
    def __init__(self, matrix_scope_df: pd.DataFrame, lexicon_df: pd.DataFrame):
        self.matrix_rows = matrix_scope_df.fillna("").to_dict(orient="records")
        self.lexicon_rows = lexicon_df.fillna("").to_dict(orient="records")
        self.matrix_by_id = {
            str(row["matrix_id"]).strip(): row for row in self.matrix_rows if str(row.get("matrix_id", "")).strip()
        }
        self.matrices_by_ingredient: dict[str, list[dict]] = {}
        for row in self.matrix_rows:
            ingredient = str(row.get("ingredient", "")).strip()
            if ingredient:
                self.matrices_by_ingredient.setdefault(ingredient, []).append(row)

        self.ingredient_aliases = self._build_ingredient_aliases()
        self.negatable_phrases = self._build_negatable_phrases()

    def _build_ingredient_aliases(self) -> dict[str, list[str]]:
        aliases: dict[str, list[str]] = {}
        for row in self.lexicon_rows:
            if str(row.get("term_type", "")).strip() != "ingredient":
                continue
            canonical = str(row.get("canonical_form", "")).strip()
            surface = str(row.get("surface_form", "")).strip()
            if not canonical:
                continue
            aliases.setdefault(canonical, []).append(canonical)
            if surface:
                aliases[canonical].append(surface)

        for canonical, values in aliases.items():
            aliases[canonical] = unique_preserving_order([normalize_text(value) for value in values if value])
        return aliases

    def _build_negatable_phrases(self) -> list[str]:
        phrases = []
        for matrix_id, hints in MATRIX_HINTS.items():
            phrases.extend(hints)
            matrix_row = self.matrix_by_id.get(matrix_id, {})
            phrases.extend(split_pipe_values(matrix_row.get("example_claims", "")))
        for aliases in self.ingredient_aliases.values():
            phrases.extend(aliases)
        phrases.extend(["muscle growth", "endurance performance", "explosive power"])
        normalized = [normalize_text(phrase) for phrase in phrases if phrase]
        normalized = [phrase for phrase in normalized if phrase]
        normalized.sort(key=len, reverse=True)
        return unique_preserving_order(normalized)

    def _match_ingredients(self, text: str) -> list[str]:
        matches = []
        for canonical, aliases in self.ingredient_aliases.items():
            if any(alias and text_has_phrase(text, alias) for alias in aliases):
                matches.append(canonical)
        for raw_phrase, canonical in OUT_OF_SCOPE_INGREDIENT_CUES.items():
            if text_has_phrase(text, raw_phrase):
                matches.append(canonical)
        return unique_preserving_order(matches)

    def _extract_dose_context(self, text: str) -> str:
        values = []
        if re.search(r"\b4\s*(?:to|-)?\s*6\s*g(?:ram)?s?\b", text) or "4 6 g day" in text:
            values.append("4-6 g/day")
        if (
            "moderate dose" in text
            or "moderate doses" in text
            or re.search(
                r"\bmoderate(?:\s+\w+)?\s+(?:amount|amounts|intake|serving|servings|range)\b",
                text,
            )
        ):
            values.append("moderate doses")
        if re.search(r"\b3\s*(?:to|-)?\s*6\s*mg\s*kg\b", text):
            values.append("3-6 mg/kg")
        if re.search(r"\b1(?:\.|\s*)6\s*g\s*(?:per\s*)?kg\b", text):
            values.append("1.6 g/kg/day")
        return "; ".join(unique_preserving_order(values))

    def _extract_duration_context(self, text: str) -> str:
        values = []
        if "2 4 weeks" in text or re.search(r"\b2\s*(?:to|-)?\s*4\s*weeks\b", text):
            values.append("2-4 weeks")
        if re.search(r"\b(?:a\s+)?few\s+weeks\b", text) or re.search(r"\bseveral\s+weeks\b", text):
            values.append("2-4 weeks")
        if re.search(r"\bcouple\s+of\s+weeks\b", text):
            values.append("2-4 weeks")
        if "at least 2 weeks" in text:
            values.append("at least 2 weeks")
        if (
            "loading period" in text
            or "loading phase" in text
            or "loading weeks" in text
            or "buildup weeks" in text
            or "build up weeks" in text
        ):
            values.append("loading period")
        return "; ".join(unique_preserving_order(values))

    def _extract_population_context(self, text: str) -> str:
        values = []
        if "for everyone" in text:
            values.append("for everyone")
        if "any sport" in text:
            values.append("any sport")
        if "trained adults" in text or "resistance trained adults" in text:
            values.append("trained adults")
        if "healthy adults" in text:
            values.append("healthy adults")
        if "trained athletes" in text:
            values.append("trained athletes")
        if "resistance trained athletes" in text:
            values.append("resistance-trained athletes")
        return "; ".join(unique_preserving_order(values))

    def _extract_modifiers(self, text: str) -> list[str]:
        modifiers = []
        for phrase in EXAGGERATION_CUES + CAUTIOUS_CUES:
            if text_has_phrase(text, phrase):
                modifiers.append(phrase)
        return unique_preserving_order(modifiers)

    def _strip_negated_cues(self, text: str) -> tuple[str, list[str]]:
        working_text = f" {text} "
        negated = []
        filler_pattern = r"(?:\s+(?:" + "|".join(re.escape(token) for token in NEGATION_OPTIONAL_FILLERS) + r")){0,3}"
        for cue in NEGATION_CUES:
            cue_norm = normalize_text(cue)
            for phrase in self.negatable_phrases:
                target = f" {cue_norm} {phrase} "
                if target in working_text:
                    working_text = working_text.replace(target, " ")
                    negated.append(phrase)
                    continue
                phrase_pattern = re.escape(phrase).replace(r"\ ", r"[-\s]+")
                pattern = re.compile(rf"(?<!\w){re.escape(cue_norm)}{filler_pattern}\s+{phrase_pattern}(?!\w)")
                if pattern.search(working_text):
                    working_text = pattern.sub(" ", working_text)
                    negated.append(phrase)
        working_text = re.sub(r"\s+", " ", working_text).strip()
        return working_text, unique_preserving_order(negated)

    def _matrix_matches(self, text: str, matrix_row: dict) -> list[tuple[str, int]]:
        matrix_id = str(matrix_row.get("matrix_id", "")).strip()
        weighted_hints = dict(MATRIX_HINT_WEIGHTS.get(matrix_id, {}))
        for phrase in MATRIX_HINTS.get(matrix_id, []):
            weighted_hints.setdefault(phrase, 2)
        for phrase in split_pipe_values(matrix_row.get("example_claims", "")):
            weighted_hints.setdefault(phrase, 3)
        matches = []
        for phrase, weight in weighted_hints.items():
            if text_has_phrase(text, phrase):
                matches.append((phrase, int(weight)))
        return matches

    def _matrix_score(self, text: str, matrix_row: dict) -> int:
        return sum(weight for _, weight in self._matrix_matches(text, matrix_row))

    def _is_contextual_overlap_match(
        self,
        matrix_id: str,
        matches: list[tuple[str, int]],
        positive_text: str,
    ) -> bool:
        phrases = [phrase for phrase, _ in matches]
        if matrix_id == "M01" and phrases == ["strength"]:
            return text_has_any_phrase(
                positive_text,
                [
                    "strength first",
                    "strength first program",
                    "strength-first",
                    "strength-first program",
                    "strength focused",
                    "strength focused training",
                    "strength focused training phase",
                    "strength focused phase",
                    "strength focused block",
                    "strength focused program",
                ],
            )
        if matrix_id == "M01" and phrases == ["resistance training"]:
            return text_has_any_phrase(
                positive_text,
                [
                    "resistance training cycle",
                    "resistance-training cycle",
                    "strength first program",
                    "strength-first program",
                    "strength first",
                    "strength focused",
                    "strength-focused",
                ],
            )
        return False

    def _has_multi_outcome_coordination(
        self,
        positive_text: str,
        matrix_scores: dict[str, int],
    ) -> bool:
        positive_matrices = [matrix_id for matrix_id, score in matrix_scores.items() if score > 0]
        if len(positive_matrices) < 2:
            return False
        if not text_has_any_phrase(positive_text, CONJUNCTION_CUES):
            return False
        return any(score >= 3 for score in matrix_scores.values())

    def _resolve_in_scope_matrix(
        self,
        ingredient: str,
        positive_text: str,
    ) -> tuple[str, str]:
        matrix_rows = self.matrices_by_ingredient.get(ingredient, [])
        if not matrix_rows:
            return "", ""

        if len(matrix_rows) == 1:
            has_matrix_specific_cue = self._matrix_score(positive_text, matrix_rows[0]) > 0
            has_generic_positive_cue = any(text_has_phrase(positive_text, cue) for cue in GENERIC_POSITIVE_CUES)
            if has_matrix_specific_cue or has_generic_positive_cue:
                if not has_matrix_specific_cue:
                    return "", "recognized ingredient but missing stable target cues"
                return str(matrix_rows[0]["matrix_id"]), ""
            return "", "recognized ingredient but missing stable target cues"

        positive_matrices = []
        matrix_scores = {}
        matrix_matches = {}
        for matrix_row in matrix_rows:
            matrix_id = str(matrix_row["matrix_id"])
            matches = self._matrix_matches(positive_text, matrix_row)
            matrix_matches[matrix_id] = matches
            score = sum(weight for _, weight in matches)
            matrix_scores[matrix_id] = score
            if score > 0:
                positive_matrices.append(matrix_id)

        if self._has_multi_outcome_coordination(positive_text, matrix_scores):
            strong_positive = [matrix_id for matrix_id, score in matrix_scores.items() if score >= 3]
            return "", f"multiple positive matrix cues remain: {', '.join(strong_positive)}"

        if len(positive_matrices) == 1:
            return positive_matrices[0], ""
        if len(positive_matrices) > 1:
            if self._has_multi_outcome_coordination(positive_text, matrix_scores):
                return "", f"multiple positive matrix cues remain: {', '.join(positive_matrices)}"
            ranked = sorted(
                ((matrix_id, score) for matrix_id, score in matrix_scores.items() if score > 0),
                key=lambda item: (-item[1], item[0]),
            )
            top_matrix, top_score = ranked[0]
            second_matrix, second_score = ranked[1]
            second_matches = matrix_matches.get(second_matrix, [])
            second_strong_match = any(weight >= 3 for _, weight in second_matches)
            contextual_second_match = self._is_contextual_overlap_match(second_matrix, second_matches, positive_text)
            if contextual_second_match:
                second_strong_match = False
                if top_score > second_score:
                    return top_matrix, f"resolved dominant matrix from overlapping cues: {top_matrix}"
            if top_score >= second_score + 2 and not second_strong_match:
                return top_matrix, f"resolved dominant matrix from overlapping cues: {top_matrix}"
            return "", f"multiple positive matrix cues remain: {', '.join(positive_matrices)}"
        return "", "recognized ingredient but no stable single matrix"

    def _resolve_out_of_scope_tuple(self, ingredient: str, positive_text: str) -> tuple[str, str, str]:
        rule = OUT_OF_SCOPE_TUPLE_RULES.get(ingredient)
        if not rule:
            return "", "", ""
        required_phrase_groups = rule.get("required_phrase_groups", [])
        if required_phrase_groups:
            for phrase_group in required_phrase_groups:
                if not any(text_has_phrase(positive_text, phrase) for phrase in phrase_group):
                    return "", "", ""
        return ingredient, str(rule["claim_type"]), str(rule["outcome_target"])

    def parse_claim(self, claim_text: str) -> dict[str, object]:
        claim_text_raw = str(claim_text or "").strip()
        claim_text_normalized = normalize_text(claim_text_raw)
        positive_text, negated_cues = self._strip_negated_cues(claim_text_normalized)
        ingredient_matches = self._match_ingredients(positive_text or claim_text_normalized)

        parse = {
            "claim_text_raw": claim_text_raw,
            "claim_text_normalized": claim_text_normalized,
            "ingredient": "",
            "claim_type": "",
            "outcome_target": "",
            "dose_context": self._extract_dose_context(claim_text_normalized),
            "duration_context": self._extract_duration_context(claim_text_normalized),
            "population_context": self._extract_population_context(claim_text_normalized),
            "claim_modifiers": self._extract_modifiers(claim_text_normalized),
            "negated_cues": negated_cues,
            "parse_status": "not_parseable",
            "notes": "",
        }

        notes = []
        if negated_cues:
            notes.append(f"excluded cues: {', '.join(negated_cues)}")

        if len(ingredient_matches) > 1:
            parse["notes"] = "multiple ingredient cues detected"
            return parse

        if not ingredient_matches:
            parse["notes"] = "no stable ingredient cue detected"
            return parse

        ingredient = ingredient_matches[0]
        parse["ingredient"] = ingredient

        if ingredient in self.matrices_by_ingredient:
            matrix_id, matrix_note = self._resolve_in_scope_matrix(ingredient, positive_text)
            if matrix_note:
                notes.append(matrix_note)

            if matrix_id:
                matrix_row = self.matrix_by_id[matrix_id]
                parse["claim_type"] = str(matrix_row.get("claim_type", "")).strip()
                parse["outcome_target"] = str(matrix_row.get("outcome_target", "")).strip()
                parse["parse_status"] = "fully_parseable"
            else:
                if "multiple positive matrix cues remain" in matrix_note:
                    parse["parse_status"] = "not_parseable"
                else:
                    parse["parse_status"] = "partially_parseable"
        else:
            out_ingredient, claim_type, outcome_target = self._resolve_out_of_scope_tuple(
                ingredient,
                positive_text,
            )
            if out_ingredient and claim_type and outcome_target:
                parse["ingredient"] = out_ingredient
                parse["claim_type"] = claim_type
                parse["outcome_target"] = outcome_target
                parse["parse_status"] = "fully_parseable"
            else:
                parse["parse_status"] = "partially_parseable"
                notes.append("ingredient recognized but no supported out-of-scope tuple rule matched")

        parse["notes"] = "; ".join(unique_preserving_order(notes))
        return parse


def build_parser(matrix_scope_csv: str | Path, lexicon_csv: str | Path) -> ClaimParserV1:
    matrix_scope_df = pd.read_csv(matrix_scope_csv).fillna("")
    lexicon_df = pd.read_csv(lexicon_csv).fillna("")
    return ClaimParserV1(matrix_scope_df, lexicon_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shallow deterministic claim parser for reasoning v1.")
    parser.add_argument("--claim", required=True, help="Raw claim text")
    parser.add_argument(
        "--matrix-scope",
        default="data/sources/matrix_scope.csv",
        help="Path to matrix scope CSV",
    )
    parser.add_argument(
        "--lexicon",
        default="data/sources/lexicon.csv",
        help="Path to lexicon CSV",
    )
    args = parser.parse_args()

    claim_parser = build_parser(args.matrix_scope, args.lexicon)
    result = claim_parser.parse_claim(args.claim)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
