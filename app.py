#!/usr/bin/env python3
"""
Single-page Streamlit interface for the project baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline import run_claim_verification


# ---------------------------------------------------------------------------
# Verdict display helpers
# ---------------------------------------------------------------------------

VERDICT_CONFIG: dict[str, dict[str, str]] = {
    "backed": {
        "label": "Supported",
        "icon": "✅",
        "color": "success",
        "description": "The claim is supported by evidence in our database.",
    },
    "partially_backed": {
        "label": "Partially Supported",
        "icon": "⚠️",
        "color": "warning",
        "description": "The claim has partial or indirect support. Some important conditions or limitations apply.",
    },
    "potentially_misleading": {
        "label": "Potentially Misleading",
        "icon": "🚫",
        "color": "error",
        "description": "The claim overstates the evidence or omits a critical condition.",
    },
    "insufficient_evidence": {
        "label": "Insufficient Evidence",
        "icon": "❓",
        "color": "info",
        "description": "The supplement and claim type are in scope, but the current evidence base is too limited to reach a verdict.",
    },
    "not_evaluable__claim_outside_scope": {
        "label": "Supplement or Claim Not in Our Database",
        "icon": "🔍",
        "color": "info",
        "description": "This supplement or claim type is not covered in our current database.",
    },
    "not_evaluable__claim_not_parseable": {
        "label": "Claim Could Not Be Parsed",
        "icon": "❓",
        "color": "warning",
        "description": "The claim could not be interpreted. Try rephrasing it more specifically.",
    },
}

_RENDER_FN = {
    "success": st.success,
    "warning": st.warning,
    "error": st.error,
    "info": st.info,
}

REASON_CODE_LABELS: dict[str, str] = {
    "direct_strong_support": "Direct strong support",
    "support_with_limitations": "Support with limitations",
    "partial_or_indirect_support": "Partial or indirect support",
    "dose_condition_not_met": "Required dose/duration condition not stated",
    "marketing_exaggeration": "Marketing exaggeration detected",
    "insufficient_corpus_support": "Insufficient corpus support",
    "claim_outside_scope": "Outside database scope",
    "claim_not_parseable": "Claim not parseable",
}


def verdict_key(reasoning_result: dict[str, object]) -> str:
    verdict = str(reasoning_result.get("verdict", "")).strip()
    reason_code = str(reasoning_result.get("reason_code", "")).strip()
    if verdict == "not_evaluable":
        return f"not_evaluable__{reason_code}"
    return verdict


def render_verdict(reasoning_result: dict[str, object]) -> None:
    key = verdict_key(reasoning_result)
    cfg = VERDICT_CONFIG.get(key, {
        "label": str(reasoning_result.get("verdict", "Unknown")),
        "icon": "❓",
        "color": "info",
        "description": "",
    })
    render_fn = _RENDER_FN.get(cfg["color"], st.info)
    render_fn(f"**{cfg['icon']} {cfg['label']}** — {cfg['description']}")


def to_label(text: object) -> str:
    value = str(text or "").strip()
    return value if value else "-"


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Sports Supplement Claim Verifier",
        page_icon=":mag:",
        layout="wide",
    )

    st.title("Sports Supplement Claim Verifier")
    st.caption(
        "A conservative baseline for single-claim verification built on lexical retrieval and deterministic reasoning."
    )

    default_claim = "Creatine increases strength during resistance training."
    claim_text = st.text_area(
        "Claim",
        value=default_claim,
        height=110,
        placeholder="Enter one sports-supplement claim.",
    )

    run_clicked = st.button("Run verification", type="primary", use_container_width=False)
    if not run_clicked:
        st.info("Enter one claim and run the baseline verifier.")
        return

    if not claim_text.strip():
        st.warning("Please enter a claim before running the verifier.")
        return

    with st.spinner("Running parser, retrieval, and reasoning..."):
        result = run_claim_verification(claim_text.strip())

    claim_parse = result["claim_parse"]
    reasoning_result = result["reasoning_result"]
    retrieved_candidates = result["retrieved_candidates"]

    # --- Verdict banner (full width, top) ---
    st.divider()
    st.subheader("Verdict")
    render_verdict(reasoning_result)

    reason_code_raw = str(reasoning_result.get("reason_code", "")).strip()
    reason_label = REASON_CODE_LABELS.get(reason_code_raw, reason_code_raw) if reason_code_raw else "-"
    explanation = to_label(reasoning_result.get("explanation"))

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.markdown(f"**Reason code:** {reason_label}")
        st.markdown(f"**Matrix ID matched:** {to_label(reasoning_result.get('matched_matrix_id'))}")
        st.markdown(f"**Scope status:** {to_label(reasoning_result.get('scope_status'))}")
        st.markdown(f"**Coverage status:** {to_label(reasoning_result.get('coverage_status'))}")
    with col_r2:
        st.markdown(f"**Explanation:** {explanation}")

        conditions = reasoning_result.get("conditions_to_state", [])
        if conditions:
            st.markdown("**Conditions to state:**")
            for c in conditions:
                st.markdown(f"- {c}")

        supporting = ", ".join(reasoning_result.get("supporting_fragment_ids", []))
        limiting = ", ".join(reasoning_result.get("limiting_fragment_ids", []))
        if supporting:
            st.markdown(f"**Supporting fragment IDs:** {supporting}")
        if limiting:
            st.markdown(f"**Limiting fragment IDs:** {limiting}")

    # --- Parsed Claim ---
    st.divider()
    st.subheader("Parsed Claim")
    st.json(claim_parse, expanded=False)

    # --- Retrieved Evidence ---
    st.divider()
    st.subheader("Retrieved Evidence")
    if not retrieved_candidates:
        st.info("No retrieval results were returned for this claim.")
    else:
        display_rows = []
        for candidate in retrieved_candidates:
            display_rows.append(
                {
                    "rank": candidate.get("rank"),
                    "fragment_id": candidate.get("fragment_id"),
                    "matrix_id": candidate.get("matrix_id"),
                    "ingredient": candidate.get("ingredient"),
                    "claim_type": candidate.get("claim_type"),
                    "outcome_target": candidate.get("outcome_target"),
                    "supports_claim": candidate.get("supports_claim"),
                    "support_strength": candidate.get("support_strength"),
                    "fragment_text": candidate.get("fragment_text"),
                }
            )
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
