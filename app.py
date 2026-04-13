#!/usr/bin/env python3
"""
Streamlit application for the sports supplement claim verifier.

Provides three input modes:
  1. Text claim input (original baseline flow)
  2. Image upload (CV-based label extraction)
  3. Both text and image (multimodal fusion)

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import Pipeline
from vision_v1 import tesseract_available
from llm_adapter import is_available as llm_is_available

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Supplement Claim Verifier",
    layout="wide",
)

st.title("Sports Supplement Claim Verifier")

# ---------------------------------------------------------------------------
# Load pipeline (cached so it only loads once)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pipeline() -> Pipeline:
    return Pipeline()

pipeline = load_pipeline()

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

st.header("Input")

col_text, col_image = st.columns(2)

with col_text:
    st.subheader("Claim Text")
    claim_text = st.text_area(
        "Enter a supplement claim to verify:",
        placeholder="e.g. Creatine increases strength during resistance training.",
        height=100,
        label_visibility="collapsed",
    )

with col_image:
    st.subheader("Product Image (optional)")
    if not tesseract_available():
        st.info(
            "Tesseract OCR is not installed. Image upload will show "
            "preprocessing results but cannot extract text. "
            "Install from: https://github.com/tesseract-ocr/tesseract",
            icon="ℹ️",
        )
    uploaded_file = st.file_uploader(
        "Upload a supplement label image:",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed",
    )

run_button = st.button("Verify Claim", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if run_button:
    if not claim_text.strip() and uploaded_file is None:
        st.warning("Please enter a claim or upload an image.")
        st.stop()

    # Save uploaded image to a temp path
    image_path = None
    if uploaded_file is not None:
        tmp_path = Path("_tmp_uploaded_image.png")
        tmp_path.write_bytes(uploaded_file.getvalue())
        image_path = str(tmp_path)

    with st.spinner("Running verification pipeline..."):
        result = pipeline.run(
            claim_text=claim_text.strip() if claim_text.strip() else None,
            image_path=image_path,
        )

    # Clean up temp file
    if image_path:
        Path(image_path).unlink(missing_ok=True)

    # --- Display results ---

    st.divider()

    # Input mode and LLM status badges
    mode = result.get("input_mode", "text")
    mode_labels = {"text": "Text Only", "image": "Image Only", "multimodal": "Text + Image"}
    llm_used = result.get("llm_used", False)
    badges = f"Input mode: **{mode_labels.get(mode, mode)}**"
    if llm_used:
        badges += " | LLM fallback: **active**"
    st.caption(badges)

    # --- Vision results (if image was used) ---
    vision = result.get("vision_result")
    if vision:
        st.header("Vision Extraction")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
        with vcol2:
            confidence = vision.get("vision_confidence", "none")
            conf_colors = {"high": "green", "medium": "orange", "low": "red", "none": "red"}
            st.metric("Confidence", confidence)
            st.text(f"Ingredient: {vision.get('detected_ingredient', '-') or '-'}")
            st.text(f"Dose: {vision.get('detected_dose', '-') or '-'}")

            claims_list = vision.get("detected_claims", [])
            if claims_list:
                st.text(f"Claims: {', '.join(claims_list)}")
            else:
                st.text("Claims: -")

            st.text(f"Notes: {vision.get('vision_notes', '')}")

        with st.expander("Raw OCR text"):
            st.code(vision.get("detected_text", "(no text extracted)"))
        with st.expander("Preprocessing steps"):
            steps = vision.get("preprocessing_steps", [])
            st.write(steps if steps else "None")

        # Show LLM claim selection if used
        llm_ocr = result.get("llm_ocr_extraction")
        if llm_ocr:
            st.subheader("LLM Claim Selection")
            st.success(f"Selected claim: **{llm_ocr.get('claim', '-')}**")
            all_claims = llm_ocr.get("all_claims", [])
            if all_claims:
                st.caption(f"All claims found: {', '.join(all_claims)}")

    # --- Effective claim ---
    effective = result.get("effective_claim", "")
    if effective:
        st.header("Effective Claim")
        st.info(effective)

    # --- Reasoning result ---
    reasoning = result.get("reasoning_result", {})
    if reasoning:
        st.header("Verification Result")

        verdict = reasoning.get("verdict", "")
        verdict_colors = {
            "backed": "success",
            "partially_backed": "info",
            "potentially_misleading": "warning",
            "not_evaluable": "error",
            "insufficient_evidence": "error",
        }
        verdict_type = verdict_colors.get(verdict, "info")

        # Show verdict prominently
        if verdict_type == "success":
            st.success(f"Verdict: **{verdict}**")
        elif verdict_type == "warning":
            st.warning(f"Verdict: **{verdict}**")
        elif verdict_type == "error":
            st.error(f"Verdict: **{verdict}**")
        else:
            st.info(f"Verdict: **{verdict}**")

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.text(f"Reason: {reasoning.get('reason_code', '-')}")
            st.text(f"Scope: {reasoning.get('scope_status', '-')}")
            st.text(f"Coverage: {reasoning.get('coverage_status', '-')}")
            st.text(f"Matrix ID: {reasoning.get('matched_matrix_id', '-') or '-'}")
        with rcol2:
            supporting = reasoning.get("supporting_fragment_ids", [])
            limiting = reasoning.get("limiting_fragment_ids", [])
            conditions = reasoning.get("conditions_to_state", [])
            st.text(f"Supporting: {', '.join(supporting) if supporting else '-'}")
            st.text(f"Limiting: {', '.join(limiting) if limiting else '-'}")
            if conditions:
                st.text(f"Conditions: {'; '.join(conditions)}")

        st.caption(reasoning.get("explanation", ""))

    # --- LLM explanation (if available) ---
    llm_explanation = result.get("llm_explanation", "")
    if llm_explanation:
        st.header("LLM Explanation")
        st.markdown(llm_explanation)
        st.caption("Generated by local LLM (Ollama). The deterministic verdict above is authoritative.")

    # --- LLM extraction details (if used) ---
    llm_extraction = result.get("llm_extraction")
    if llm_extraction:
        with st.expander("LLM extraction details"):
            st.json(llm_extraction)

    # --- Parse details ---
    parse = result.get("parse_result")
    if parse:
        with st.expander("Parse details"):
            st.json(parse)

    # --- Retrieval details ---
    retrieval = result.get("retrieval_results", [])
    if retrieval:
        with st.expander(f"Retrieval results ({len(retrieval)} fragments)"):
            for r in retrieval:
                st.markdown(
                    f"**{r.get('fragment_id', '?')}** "
                    f"(score={r.get('score', 0)}, "
                    f"supports={r.get('supports_claim', '?')}, "
                    f"strength={r.get('support_strength', '?')})"
                )
                st.caption(r.get("fragment_text", ""))
