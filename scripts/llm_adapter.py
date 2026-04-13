#!/usr/bin/env python3
"""
Local LLM adapter for parser fallback and explanation generation.

Connects to Ollama (local LLM server) to provide:
1. Structured claim extraction when the deterministic parser fails
2. Natural-language explanation of verification results
3. Intelligent claim selection from raw OCR text (vision pipeline)

This is an optional enhancement. If Ollama is not running or the
model is unavailable, the system falls back to the deterministic
baseline with no change in behavior.

Requirements:
    - Ollama installed and running (https://ollama.com)
    - A model pulled (e.g. ollama pull qwen2.5:3b)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Valid schema values for validation
VALID_INGREDIENTS = [
    "creatine_monohydrate", "caffeine", "beta_alanine",
    "whey_protein", "bcaa", "l_carnitine",
    "citrulline_malate", "vitamin_d", "fish_oil", "hmb",
    "sodium_bicarbonate", "nitrate_beetroot", "ashwagandha", "taurine",
]
VALID_CLAIM_TYPES = [
    "performance", "body_composition", "energy_fatigue", "recovery",
    "cognitive", "health",
]
VALID_OUTCOME_TARGETS = [
    "strength", "power_high_intensity_performance", "lean_mass_gain",
    "perceived_energy_fatigue_reduction", "endurance_exercise_performance",
    "exercise_capacity_high_intensity_tolerance", "post_exercise_recovery",
    "muscle_soreness_recovery", "fat_mass_reduction",
    "reaction_time_focus", "blood_flow_oxygen_delivery", "time_to_exhaustion",
    "immune_function", "inflammation_reduction", "muscle_protein_synthesis",
]


# ---------------------------------------------------------------------------
# Ollama availability
# ---------------------------------------------------------------------------

def is_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            model_names = [m.get("name", "") for m in data.get("models", [])]
            return any(model in name for name in model_names)
    except Exception:
        return False


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to Ollama and return the response text."""
    import urllib.request
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        return data.get("response", "")


# ---------------------------------------------------------------------------
# Structured claim extraction (parser fallback)
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are a sports supplement claim analyzer. Extract structured fields from the claim below.

VALID VALUES (use exactly these strings):
- ingredient: {ingredients}
- claim_type: {claim_types}
- outcome_target: {outcome_targets}

If the claim does not match any valid value, use an empty string "".

Claim: "{claim_text}"

Respond ONLY with a JSON object, nothing else:
{{"ingredient": "...", "claim_type": "...", "outcome_target": "...", "confidence": "high|medium|low", "reasoning": "one sentence explaining your choice"}}"""


def extract_claim_fields(
    claim_text: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, str] | None:
    """Use the LLM to extract structured claim fields.

    Returns a dict with ingredient, claim_type, outcome_target,
    confidence, and reasoning. Returns None on failure.
    """
    prompt = _EXTRACTION_PROMPT.format(
        ingredients=", ".join(VALID_INGREDIENTS),
        claim_types=", ".join(VALID_CLAIM_TYPES),
        outcome_targets=", ".join(VALID_OUTCOME_TARGETS),
        claim_text=claim_text,
    )

    try:
        raw = _call_ollama(prompt, model)
    except Exception:
        return None

    return _parse_extraction_response(raw)


def _parse_extraction_response(raw: str) -> dict[str, str] | None:
    """Parse and validate the LLM JSON response."""
    # Try to find JSON in the response
    json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not json_match:
        return None

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    ingredient = str(parsed.get("ingredient", "")).strip()
    claim_type = str(parsed.get("claim_type", "")).strip()
    outcome_target = str(parsed.get("outcome_target", "")).strip()

    # Validate against schema
    if ingredient and ingredient not in VALID_INGREDIENTS:
        ingredient = ""
    if claim_type and claim_type not in VALID_CLAIM_TYPES:
        claim_type = ""
    if outcome_target and outcome_target not in VALID_OUTCOME_TARGETS:
        outcome_target = ""

    if not ingredient and not claim_type:
        return None

    return {
        "ingredient": ingredient,
        "claim_type": claim_type,
        "outcome_target": outcome_target,
        "confidence": str(parsed.get("confidence", "low")).strip(),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


# ---------------------------------------------------------------------------
# OCR claim selection (vision pipeline)
# ---------------------------------------------------------------------------

_OCR_CLAIM_PROMPT = """You are analyzing raw OCR text extracted from a sports supplement product label.

The text contains a mix of: product name, ingredient lists, serving sizes, marketing claims, supplement facts, and other label information.

Your task: identify and extract ONLY the health or performance claim(s) from this text. A claim is a statement about what the supplement does (e.g. "increases strength", "supports muscle recovery", "boosts energy").

Do NOT include: product names, ingredient lists, dosage amounts, supplement facts, or general label text.

OCR text:
---
{ocr_text}
---

Respond ONLY with a JSON object:
{{"claim": "the single most specific health/performance claim found", "ingredient": "the supplement ingredient mentioned", "all_claims": ["list", "of", "all", "claims", "found"]}}

If no clear health/performance claim is found, respond with:
{{"claim": "", "ingredient": "", "all_claims": []}}"""


def extract_claim_from_ocr(
    ocr_text: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, object] | None:
    """Use the LLM to select the actual claim from raw OCR text.

    Returns a dict with 'claim' (the best claim sentence),
    'ingredient', and 'all_claims'. Returns None on failure.
    """
    if not ocr_text.strip():
        return None

    prompt = _OCR_CLAIM_PROMPT.format(ocr_text=ocr_text)

    try:
        raw = _call_ollama(prompt, model)
    except Exception:
        return None

    # Parse JSON from response
    json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not json_match:
        return None

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    claim = str(parsed.get("claim", "")).strip()
    ingredient = str(parsed.get("ingredient", "")).strip()
    all_claims = parsed.get("all_claims", [])
    if isinstance(all_claims, list):
        all_claims = [str(c).strip() for c in all_claims if str(c).strip()]
    else:
        all_claims = []

    if not claim and not all_claims:
        return None

    return {
        "claim": claim,
        "ingredient": ingredient,
        "all_claims": all_claims,
    }


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------

_EXPLANATION_PROMPT = """You are explaining a supplement claim verification result to a user.

Claim: "{claim_text}"
Verdict: {verdict}
Reason: {reason_code}
Matched ingredient: {ingredient}
Evidence summary: {evidence_summary}
Conditions: {conditions}

Write a clear 2-3 sentence explanation of why this claim received this verdict. Be factual and concise. Do not add information beyond what is provided."""


def generate_explanation(
    claim_text: str,
    reasoning_result: dict[str, object],
    retrieval_results: list[dict[str, object]],
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a natural-language explanation of the verification result."""
    verdict = reasoning_result.get("verdict", "")
    reason_code = reasoning_result.get("reason_code", "")
    matched_matrix_id = reasoning_result.get("matched_matrix_id", "")
    conditions = reasoning_result.get("conditions_to_state", [])
    explanation = reasoning_result.get("explanation", "")

    # Build evidence summary from top retrieval results
    evidence_parts = []
    for r in retrieval_results[:3]:
        supports = r.get("supports_claim", "")
        strength = r.get("support_strength", "")
        text = r.get("fragment_text", "")
        if text:
            short = text[:120] + "..." if len(text) > 120 else text
            evidence_parts.append(f"[{supports}/{strength}] {short}")
    evidence_summary = " | ".join(evidence_parts) if evidence_parts else "No evidence retrieved."

    prompt = _EXPLANATION_PROMPT.format(
        claim_text=claim_text,
        verdict=verdict,
        reason_code=reason_code,
        ingredient=matched_matrix_id,
        evidence_summary=evidence_summary,
        conditions="; ".join(conditions) if conditions else "none",
    )

    try:
        return _call_ollama(prompt, model).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Test LLM adapter functions")
    ap.add_argument("--check", action="store_true", help="Check if Ollama is available")
    ap.add_argument("--extract", type=str, help="Extract claim fields from text")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    args = ap.parse_args()

    if args.check:
        available = is_available(args.model)
        print(f"Ollama available: {available} (model: {args.model})")
        return

    if args.extract:
        result = extract_claim_fields(args.extract, args.model)
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("Extraction failed or returned no valid fields.")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
