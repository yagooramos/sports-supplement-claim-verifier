#!/usr/bin/env python3
"""
Shared utility functions used across the project scripts.

This module centralizes text normalization, tokenization, and small helpers
used across claim parsing, retrieval, and reasoning.

All other scripts should import these from here. The lexical retriever
re-exports normalize_text and tokenize for backward compatibility.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: object) -> str:
    """Lowercase, strip accents, collapse whitespace, keep only [a-z0-9 ]."""
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s/_-]", " ", text)
    text = text.replace("/", " ").replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split normalized text into non-empty tokens."""
    return [tok for tok in normalize_text(text).split() if tok]


def unique_preserving_order(values: list[str]) -> list[str]:
    """Deduplicate a list of strings preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def split_pipe_values(raw_value: object) -> list[str]:
    """Split a pipe-delimited string into a list of non-empty stripped values."""
    if raw_value is None:
        return []
    values = [value.strip() for value in str(raw_value).split("|")]
    return [value for value in values if value]


def join_pipe(values: list[str]) -> str:
    """Join non-empty values with pipe delimiter."""
    return "|".join([value for value in values if value])


def text_has_phrase(text: str, phrase: str) -> bool:
    """Check whether `phrase` appears as a whole-word match in normalized `text`."""
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return False
    pattern = rf"(?<!\w){re.escape(phrase_norm)}(?!\w)"
    return bool(re.search(pattern, text))


def text_has_any_phrase(text: str, phrases: list[str]) -> bool:
    """Check whether any of `phrases` appears as a whole-word match in `text`."""
    return any(text_has_phrase(text, phrase) for phrase in phrases)
