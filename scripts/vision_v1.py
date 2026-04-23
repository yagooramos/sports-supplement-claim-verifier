#!/usr/bin/env python3
"""
Computer Vision module v1: supplement label text extraction.

Vision task
-----------
Extract structured text information from supplement product images
(labels, packaging, marketing material) and transform it into input
compatible with the existing claim verification pipeline.

CV techniques used (maps to Unit I: Digital Image Fundamentals)
- Image loading and color space conversion (BGR to grayscale)
- Gaussian blur for noise reduction
- Adaptive thresholding for binarization
- OCR text extraction via Tesseract

Input:  an image file (photo of a supplement label or package)
Output: a VisionResult dict with detected_text, detected_ingredient,
        detected_claims, detected_dose, vision_confidence, vision_notes

Limitations
-----------
- Depends on Tesseract OCR being installed on the system
- Works best on clear, front-facing label photos with readable text
- Heuristic extraction relies on the project lexicon; unknown
  ingredients or phrasings will not be detected
- No layout analysis or region detection in v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import normalize_text, text_has_phrase, split_pipe_values

PROJECT_ROOT = SCRIPT_DIR.parent
LEXICON_PATH = PROJECT_ROOT / "data" / "sources" / "lexicon.csv"
MATRIX_SCOPE_PATH = PROJECT_ROOT / "data" / "sources" / "matrix_scope.csv"

# ---------------------------------------------------------------------------
# Tesseract availability check
# ---------------------------------------------------------------------------

_TESSERACT_AVAILABLE = False
try:
    import pytesseract

    # On Windows, Tesseract may not be on PATH after install
    if sys.platform == "win32":
        _win_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if _win_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(_win_path)

    pytesseract.get_tesseract_version()
    _TESSERACT_AVAILABLE = True
except Exception:
    pass


def tesseract_available() -> bool:
    return _TESSERACT_AVAILABLE


# ---------------------------------------------------------------------------
# Domain vocabulary (loaded once from project data)
# ---------------------------------------------------------------------------

def _load_ingredient_surface_forms() -> dict[str, list[str]]:
    """Load ingredient surface forms from the project lexicon."""
    import pandas as pd
    if not LEXICON_PATH.exists():
        return {}
    df = pd.read_csv(LEXICON_PATH).fillna("")
    aliases: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        if str(row.get("term_type", "")).strip() != "ingredient":
            continue
        canonical = str(row.get("canonical_form", "")).strip()
        surface = str(row.get("surface_form", "")).strip()
        if canonical:
            aliases.setdefault(canonical, []).append(normalize_text(canonical))
            if surface:
                aliases[canonical].append(normalize_text(surface))
    return aliases


def _load_claim_phrases() -> dict[str, list[str]]:
    """Load example claim phrases per claim_type from matrix_scope."""
    import pandas as pd
    if not MATRIX_SCOPE_PATH.exists():
        return {}
    df = pd.read_csv(MATRIX_SCOPE_PATH).fillna("")
    phrases: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        claim_type = str(row.get("claim_type", "")).strip()
        examples = split_pipe_values(row.get("example_claims", ""))
        if claim_type and examples:
            phrases.setdefault(claim_type, []).extend(
                [normalize_text(ex) for ex in examples]
            )
    return phrases


_INGREDIENT_ALIASES: dict[str, list[str]] | None = None
_CLAIM_PHRASES: dict[str, list[str]] | None = None


def _get_ingredient_aliases() -> dict[str, list[str]]:
    global _INGREDIENT_ALIASES
    if _INGREDIENT_ALIASES is None:
        _INGREDIENT_ALIASES = _load_ingredient_surface_forms()
    return _INGREDIENT_ALIASES


def _get_claim_phrases() -> dict[str, list[str]]:
    global _CLAIM_PHRASES
    if _CLAIM_PHRASES is None:
        _CLAIM_PHRASES = _load_claim_phrases()
    return _CLAIM_PHRASES


# ---------------------------------------------------------------------------
# Image preprocessing (OpenCV - Unit I fundamentals)
# ---------------------------------------------------------------------------

def preprocess_for_ocr(
    image: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Apply CV preprocessing to improve OCR quality.

    Steps applied:
    1. Grayscale conversion (color space conversion)
    2. Gaussian blur 3x3 (noise reduction)
    3. Adaptive Gaussian thresholding (binarization)

    Returns the processed image and a list of applied step names.
    """
    steps: list[str] = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steps.append("grayscale_conversion")

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    steps.append("gaussian_blur_3x3")

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )
    steps.append("adaptive_threshold")

    return thresh, steps


def load_image(image_path: str | Path) -> np.ndarray | None:
    """Load an image from disk using OpenCV."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return img


def load_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    """Load an image from in-memory bytes using OpenCV."""
    if not image_bytes:
        return None
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    if image_array.size == 0:
        return None
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


# ---------------------------------------------------------------------------
# OCR text extraction
# ---------------------------------------------------------------------------

def extract_text_ocr(processed_image: np.ndarray) -> str:
    """Run Tesseract OCR on a preprocessed grayscale/binary image."""
    if not tesseract_available():
        return ""
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(processed_image)
    text = pytesseract.image_to_string(pil_img, lang="eng")
    return text.strip()


# ---------------------------------------------------------------------------
# Heuristic extraction from OCR text
# ---------------------------------------------------------------------------

def detect_ingredients(text: str) -> list[str]:
    """Match OCR text against the project ingredient lexicon."""
    normalized = normalize_text(text)
    aliases = _get_ingredient_aliases()
    found: list[str] = []
    for canonical, surface_forms in aliases.items():
        for form in surface_forms:
            if form and text_has_phrase(normalized, form):
                found.append(canonical)
                break
    return found


def detect_claims(text: str) -> list[str]:
    """Find claim-like phrases in OCR text using project vocabulary."""
    normalized = normalize_text(text)
    claim_phrases = _get_claim_phrases()
    found: list[str] = []
    for _claim_type, phrases in claim_phrases.items():
        for phrase in phrases:
            if phrase and text_has_phrase(normalized, phrase):
                found.append(phrase)
    return found


def detect_dose(text: str) -> str:
    """Extract dose patterns from OCR text using regex."""
    normalized = normalize_text(text)
    doses: list[str] = []
    # Match patterns like "5g", "200mg", "3-6 mg/kg", "5 g per serving"
    patterns = [
        (r"\b(\d+(?:\.\d+)?)\s*mg\b", "mg"),
        (r"\b(\d+(?:\.\d+)?)\s*g\b", "g"),
        (r"\b(\d+)\s*(?:to|-)\s*(\d+)\s*mg\b", "mg_range"),
        (r"\b(\d+)\s*(?:to|-)\s*(\d+)\s*g\b", "g_range"),
    ]
    for pattern, unit in patterns:
        for match in re.finditer(pattern, normalized):
            if unit == "mg_range":
                doses.append(f"{match.group(1)}-{match.group(2)} mg")
            elif unit == "g_range":
                doses.append(f"{match.group(1)}-{match.group(2)} g")
            else:
                doses.append(f"{match.group(1)} {unit}")
    return "; ".join(doses) if doses else ""


def assess_confidence(
    detected_text: str,
    ingredients: list[str],
    claims: list[str],
) -> str:
    """Assess extraction confidence based on what was found."""
    if not detected_text.strip():
        return "none"
    if ingredients and claims:
        return "high"
    if ingredients or claims:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_from_image(
    image_path: str | Path | None = None,
    image_bytes: bytes | None = None,
) -> dict[str, object]:
    """Full vision pipeline: load -> preprocess -> OCR -> extract.

    Returns a dict with:
        detected_text, detected_ingredient, detected_claims,
        detected_dose, vision_confidence, vision_notes,
        preprocessing_steps
    """
    result = {
        "detected_text": "",
        "detected_ingredient": "",
        "detected_claims": [],
        "detected_dose": "",
        "vision_confidence": "none",
        "vision_notes": "",
        "preprocessing_steps": [],
    }

    if image_bytes is not None:
        image = load_image_bytes(image_bytes)
        image_source = "uploaded image bytes"
    else:
        image = load_image(image_path) if image_path is not None else None
        image_source = str(image_path) if image_path is not None else "no image source"

    if image is None:
        result["vision_notes"] = f"Could not load image: {image_source}"
        return result

    processed, steps = preprocess_for_ocr(image)
    result["preprocessing_steps"] = steps

    if not tesseract_available():
        result["vision_notes"] = (
            "Tesseract OCR is not installed. Preprocessing was applied "
            "but text extraction requires Tesseract. "
            "Install from: https://github.com/tesseract-ocr/tesseract"
        )
        return result

    raw_text = extract_text_ocr(processed)
    result["detected_text"] = raw_text

    if not raw_text.strip():
        result["vision_notes"] = "OCR produced no readable text from the image."
        return result

    ingredients = detect_ingredients(raw_text)
    claims = detect_claims(raw_text)
    dose = detect_dose(raw_text)

    result["detected_ingredient"] = ingredients[0] if ingredients else ""
    result["detected_claims"] = claims
    result["detected_dose"] = dose
    result["vision_confidence"] = assess_confidence(raw_text, ingredients, claims)

    notes_parts: list[str] = []
    if ingredients:
        notes_parts.append(f"ingredient(s) detected: {', '.join(ingredients)}")
    else:
        notes_parts.append("no known ingredient detected in OCR text")
    if claims:
        notes_parts.append(f"{len(claims)} claim phrase(s) matched")
    else:
        notes_parts.append("no known claim phrases matched")
    if dose:
        notes_parts.append(f"dose detected: {dose}")
    result["vision_notes"] = "; ".join(notes_parts)

    return result


def build_claim_from_vision(vision_result: dict[str, object]) -> str:
    """Construct a synthetic claim string from vision extraction output.

    This produces a text string that can be fed directly into the
    existing claim parser, bridging the CV output to the NLP pipeline.
    """
    parts: list[str] = []
    ingredient = str(vision_result.get("detected_ingredient", "")).strip()
    claims = vision_result.get("detected_claims", [])
    dose = str(vision_result.get("detected_dose", "")).strip()

    if ingredient:
        canonical_name = ingredient.replace("_", " ")
        if claims:
            parts.append(f"{canonical_name} {claims[0]}")
        elif dose:
            parts.append(f"{canonical_name} at {dose}")
        else:
            parts.append(canonical_name)
    elif claims:
        parts.append(claims[0])

    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Synthetic test image generator
# ---------------------------------------------------------------------------

def generate_test_image(output_path: str | Path) -> None:
    """Create a synthetic supplement label image for testing.

    Draws text on a white background simulating a product label.
    Useful for validating the vision pipeline without real photos.
    """
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Draw a border
    cv2.rectangle(img, (10, 10), (590, 390), (0, 0, 0), 2)

    # Product name
    cv2.putText(
        img, "CREATINE MONOHYDRATE",
        (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2,
    )

    # Separator line
    cv2.line(img, (40, 90), (560, 90), (150, 150, 150), 1)

    # Marketing claim
    cv2.putText(
        img, "Increases Strength",
        (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2,
    )
    cv2.putText(
        img, "Supports Performance",
        (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2,
    )

    # Dose info
    cv2.putText(
        img, "5g per serving",
        (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1,
    )

    # Supplement facts header
    cv2.putText(
        img, "Supplement Facts",
        (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1,
    )
    cv2.putText(
        img, "Creatine Monohydrate 5000mg",
        (40, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vision v1: extract supplement information from images",
    )
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser(
        "extract", help="Extract information from a supplement image",
    )
    extract_parser.add_argument("--image", required=True, help="Path to image file")

    gen_parser = subparsers.add_parser(
        "generate-test", help="Generate a synthetic test image",
    )
    gen_parser.add_argument(
        "--output",
        default="data/test_images/creatine_label.png",
        help="Output path for the test image",
    )

    args = parser.parse_args()

    if args.command == "extract":
        result = extract_from_image(args.image)
        print(json.dumps(result, indent=2, ensure_ascii=True))
        claim = build_claim_from_vision(result)
        if claim:
            print(f"\nSynthetic claim for pipeline: {claim}")
    elif args.command == "generate-test":
        generate_test_image(args.output)
        print(f"Test image saved to: {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
