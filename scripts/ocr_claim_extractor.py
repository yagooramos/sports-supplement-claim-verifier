#!/usr/bin/env python3
"""
OCR helpers for extracting the most likely supplement claim from an image.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

try:
    from . import claim_parser_v1
except ImportError:
    import claim_parser_v1

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX_SCOPE_CSV = REPO_ROOT / "data" / "sources" / "matrix_scope.csv"
DEFAULT_LEXICON_CSV = REPO_ROOT / "data" / "sources" / "lexicon.csv"

CLAIM_VERBS = (
    "increase",
    "increases",
    "improve",
    "improves",
    "improved",
    "boost",
    "boosts",
    "enhance",
    "enhances",
    "support",
    "supports",
    "reduce",
    "reduces",
    "build",
    "builds",
    "help",
    "helps",
)


@dataclass
class OCRLine:
    text: str
    confidence: float


def build_ocr_engine() -> RapidOCR:
    return RapidOCR()


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _line_has_claim_shape(text: str) -> bool:
    lowered = text.lower()
    return any(verb in lowered for verb in CLAIM_VERBS)


def _normalize_for_score(text: str) -> str:
    return re.sub(r"[^a-z0-9\s/%.-]", " ", text.lower())


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Mild preprocessing improves OCR on screenshots and labels without
    # changing the content semantics.
    grayscale = ImageOps.grayscale(image)
    autocontrast = ImageOps.autocontrast(grayscale)
    scaled = autocontrast.resize((autocontrast.width * 2, autocontrast.height * 2))
    return np.array(scaled)


def extract_ocr_lines(image_bytes: bytes, engine: RapidOCR) -> list[OCRLine]:
    prepared_image = _preprocess_image(image_bytes)
    result, _ = engine(prepared_image)
    if not result:
        return []

    lines: list[OCRLine] = []
    for item in result:
        if len(item) < 3:
            continue
        raw_text = _clean_text(str(item[1]))
        if not raw_text:
            continue
        try:
            confidence = float(item[2])
        except (TypeError, ValueError):
            confidence = 0.0
        lines.append(OCRLine(text=raw_text, confidence=confidence))
    return lines


def _candidate_windows(lines: list[OCRLine]) -> list[tuple[str, float, int]]:
    candidates: list[tuple[str, float, int]] = []
    for idx in range(len(lines)):
        for size in (1, 2, 3):
            subset = lines[idx : idx + size]
            if len(subset) != size:
                continue
            text = _clean_text(" ".join(line.text for line in subset))
            if len(text) < 12:
                continue
            confidence = sum(line.confidence for line in subset) / len(subset)
            candidates.append((text, confidence, size))
    deduped: list[tuple[str, float, int]] = []
    seen: set[str] = set()
    for text, confidence, size in candidates:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((text, confidence, size))
    return deduped


def _score_candidate(
    candidate_text: str,
    confidence: float,
    line_count: int,
    parser: claim_parser_v1.ClaimParserV1,
) -> tuple[float, dict[str, object]]:
    parse = parser.parse_claim(candidate_text)
    status = str(parse.get("parse_status", "")).strip()
    base = {
        "fully_parseable": 100.0,
        "partially_parseable": 55.0,
        "not_parseable": 0.0,
    }.get(status, 0.0)

    normalized = _normalize_for_score(candidate_text)
    words = [word for word in normalized.split() if word]
    word_count = len(words)

    if 4 <= word_count <= 18:
        base += 10.0
    elif word_count > 24:
        base -= 8.0

    base -= (line_count - 1) * 8.0

    if _line_has_claim_shape(candidate_text):
        base += 8.0

    if str(parse.get("ingredient", "")).strip():
        base += 12.0
    if str(parse.get("claim_type", "")).strip():
        base += 10.0
    if str(parse.get("outcome_target", "")).strip():
        base += 10.0

    if candidate_text.endswith((".", "!", "?")):
        base += 2.0

    uppercase_tokens = [token for token in candidate_text.split() if len(token) >= 6 and token.isupper()]
    if uppercase_tokens:
        base -= min(12.0, len(uppercase_tokens) * 4.0)

    if re.search(r"(https?://|www\.|@\w+)", candidate_text.lower()):
        base -= 15.0

    base += confidence * 10.0
    return base, parse


def extract_claim_from_image(
    image_bytes: bytes,
    engine: RapidOCR,
    parser: claim_parser_v1.ClaimParserV1,
) -> dict[str, object]:
    lines = extract_ocr_lines(image_bytes, engine)
    line_payload = [{"text": line.text, "confidence": round(line.confidence, 3)} for line in lines]
    if not lines:
        return {
            "claim_text": "",
            "ocr_lines": [],
            "parse_preview": {},
            "status": "no_text_detected",
        }

    ranked_candidates: list[tuple[float, str, dict[str, object], float]] = []
    for candidate_text, confidence, line_count in _candidate_windows(lines):
        score, parse = _score_candidate(candidate_text, confidence, line_count, parser)
        ranked_candidates.append((score, candidate_text, parse, confidence))

    ranked_candidates.sort(key=lambda item: item[0], reverse=True)
    if not ranked_candidates:
        fallback_text = max(lines, key=lambda line: (len(line.text), line.confidence)).text
        return {
            "claim_text": fallback_text,
            "ocr_lines": line_payload,
            "parse_preview": parser.parse_claim(fallback_text),
            "status": "fallback_longest_line",
        }

    _, best_text, best_parse, _ = ranked_candidates[0]
    return {
        "claim_text": best_text,
        "ocr_lines": line_payload,
        "parse_preview": best_parse,
        "status": "claim_extracted",
    }


def build_default_parser() -> claim_parser_v1.ClaimParserV1:
    return claim_parser_v1.build_parser(DEFAULT_MATRIX_SCOPE_CSV, DEFAULT_LEXICON_CSV)
