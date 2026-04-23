#!/usr/bin/env python3
"""
Streamlit application for the sports supplement claim verifier.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from scripts.claim_type_classifier import classifier_available
from scripts.llm_adapter import is_available as llm_is_available
from scripts.ocr_claim_extractor import build_default_parser, build_ocr_engine, extract_claim_from_image
from scripts.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parent


VERDICT_CONFIG: dict[str, dict[str, str]] = {
    "backed": {
        "label": "Supported",
        "icon": "[OK]",
        "color": "success",
        "description": "The claim is supported by evidence in our database.",
    },
    "partially_backed": {
        "label": "Partially Supported",
        "icon": "[!]",
        "color": "warning",
        "description": "The claim has partial or indirect support. Some important conditions or limitations apply.",
    },
    "potentially_misleading": {
        "label": "Potentially Misleading",
        "icon": "[X]",
        "color": "error",
        "description": "The claim overstates the evidence or omits a critical condition.",
    },
    "insufficient_evidence": {
        "label": "Insufficient Evidence",
        "icon": "[?]",
        "color": "info",
        "description": "The current evidence base is too limited to reach a verdict.",
    },
    "not_evaluable__claim_outside_scope": {
        "label": "Supplement or Claim Not in Our Database",
        "icon": "[SCOPE]",
        "color": "info",
        "description": "This supplement or claim type is not covered in the current canonical database.",
    },
    "not_evaluable__claim_not_parseable": {
        "label": "Claim Could Not Be Parsed",
        "icon": "[?]",
        "color": "warning",
        "description": "The claim could not be interpreted reliably.",
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
    "dose_condition_not_met": "Required dose or duration condition not stated",
    "marketing_exaggeration": "Marketing exaggeration detected",
    "insufficient_corpus_support": "Insufficient corpus support",
    "claim_outside_scope": "Outside database scope",
    "claim_not_parseable": "Claim not parseable",
    "no_input": "No usable claim extracted",
}

MODE_LABELS = {
    "text": "Text Only",
    "image": "Image Only",
    "multimodal": "Text + Image",
}

EXAMPLE_CLAIMS: dict[str, list[str]] = {
    "-- pick an example --": [],
    "Backed claims": [
        "Creatine increases strength during resistance training.",
        "Caffeine reduces perceived fatigue during exercise.",
        "Beta-alanine at 4 to 6 g per day for at least 2 weeks can support high-intensity exercise capacity.",
    ],
    "Partially backed claims": [
        "Creatine may support small gains in lean mass during training.",
        "Whey helps you recover faster after lifting.",
        "Whey can support lean mass gains during resistance training, but more is not always better.",
    ],
    "Potentially misleading claims": [
        "Creatine guarantees strength gains for everyone.",
        "Beta-alanine works for any sport even without a loading period.",
        "Caffeine improves endurance performance.",
    ],
    "Out-of-scope / unparseable": [
        "Electrolytes improve hydration performance.",
        "This supplement makes you unstoppable.",
    ],
}


@st.cache_resource
def get_ocr_engine():
    return build_ocr_engine()


@st.cache_resource
def get_claim_parser():
    return build_default_parser()


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline()


def verdict_key(reasoning_result: dict[str, object]) -> str:
    verdict = str(reasoning_result.get("verdict", "")).strip()
    reason_code = str(reasoning_result.get("reason_code", "")).strip()
    if verdict == "not_evaluable":
        return f"not_evaluable__{reason_code}"
    return verdict


def render_verdict(reasoning_result: dict[str, object]) -> None:
    key = verdict_key(reasoning_result)
    cfg = VERDICT_CONFIG.get(
        key,
        {
            "label": str(reasoning_result.get("verdict", "Unknown")),
            "icon": "[?]",
            "color": "info",
            "description": "",
        },
    )
    render_fn = _RENDER_FN.get(cfg["color"], st.info)
    render_fn(f"**{cfg['icon']} {cfg['label']}** - {cfg['description']}")


def to_label(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "-"


def load_coverage_table() -> pd.DataFrame:
    matrix_path = ROOT_DIR / "data" / "sources" / "matrix_scope.csv"
    fragments_path = ROOT_DIR / "data" / "annotations" / "evidence_fragments.csv"
    try:
        matrix_df = pd.read_csv(matrix_path).fillna("")
        fragments_df = pd.read_csv(fragments_path).fillna("")
    except Exception:
        return pd.DataFrame()

    fragment_counts = fragments_df.groupby("matrix_id").size().reset_index(name="fragments")
    merged = matrix_df.merge(fragment_counts, on="matrix_id", how="left").fillna({"fragments": 0})
    merged["fragments"] = merged["fragments"].astype(int)

    display = merged[["matrix_id", "ingredient", "claim_type", "outcome_target", "fragments"]].copy()
    display.columns = ["ID", "Ingredient", "Claim type", "Outcome", "Fragments"]
    display["Ingredient"] = display["Ingredient"].str.replace("_", " ").str.title()
    display["Claim type"] = display["Claim type"].str.replace("_", " ")
    display["Outcome"] = display["Outcome"].str.replace("_", " ")
    return display


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Corpus coverage")
        st.caption("Supplement-claim pairs currently covered by the canonical corpus.")
        coverage_df = load_coverage_table()
        if coverage_df.empty:
            st.warning("Could not load the coverage table.")
        else:
            st.dataframe(coverage_df, hide_index=True, use_container_width=True)

        st.divider()
        with st.expander("How it works", expanded=False):
            st.markdown(
                """
1. **OCR and claim extraction** convert an uploaded label or screenshot into candidate claim text.
2. **Parser** maps the claim to the project schema: `ingredient`, `claim_type`, and `outcome_target`.
3. **Retrieval** finds the top evidence fragments in the canonical corpus.
4. **Reasoning** produces the authoritative verdict from scope, coverage, and evidence strength.
5. **Classifier and local LLM** remain optional support layers and never replace the deterministic verdict.
                """
            )


def ensure_default_state() -> None:
    st.session_state.setdefault("claim_text", "Creatine increases strength during resistance training.")
    st.session_state.setdefault("pending_claim_text", "")
    st.session_state.setdefault("ocr_image_hash", "")
    st.session_state.setdefault("ocr_result", None)
    st.session_state.setdefault("example_group", "-- pick an example --")
    st.session_state.setdefault("selected_example", "")
    st.session_state.setdefault("last_example_claim", "")


def render_example_selector() -> None:
    st.caption("Quick demo examples")
    group = st.selectbox("Try an example", options=list(EXAMPLE_CLAIMS.keys()), key="example_group")
    options = EXAMPLE_CLAIMS.get(group, [])
    if not options:
        st.session_state["selected_example"] = ""
        return

    current = st.session_state.get("selected_example", "")
    if current not in options:
        st.session_state["selected_example"] = options[0]

    st.radio("Select a claim", options=options, key="selected_example", label_visibility="collapsed")
    selected_example = str(st.session_state.get("selected_example", "")).strip()
    if selected_example and selected_example != st.session_state.get("last_example_claim"):
        st.session_state["claim_text"] = selected_example
        st.session_state["last_example_claim"] = selected_example


def process_uploaded_claim_image(uploaded_image) -> bytes | None:
    if uploaded_image is None:
        st.info("No image uploaded. You can still run text-only verification.")
        return None

    image_bytes = uploaded_image.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    try:
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption=uploaded_image.name, use_container_width=True)
    except Exception as exc:
        st.error(f"The uploaded image could not be opened: {exc}")
        return None

    if st.session_state.get("ocr_image_hash") != image_hash:
        with st.spinner("Extracting claim text from the image..."):
            ocr_result = extract_claim_from_image(
                image_bytes=image_bytes,
                engine=get_ocr_engine(),
                parser=get_claim_parser(),
            )
        st.session_state["ocr_image_hash"] = image_hash
        st.session_state["ocr_result"] = ocr_result
        extracted_claim = str(ocr_result.get("claim_text", "")).strip()
        if extracted_claim:
            st.session_state["pending_claim_text"] = extracted_claim
            st.rerun()

    ocr_result = st.session_state.get("ocr_result") or {}
    extracted_claim = str(ocr_result.get("claim_text", "")).strip()
    if extracted_claim:
        st.success("Claim extracted from the image. The claim field has been prefilled.")
        st.markdown(f"**Extracted claim:** {extracted_claim}")
    else:
        st.warning("No stable claim was extracted automatically. You can still type or edit the claim manually.")

    with st.expander("OCR details", expanded=False):
        st.json(ocr_result, expanded=False)

    return image_bytes


def render_classifier_panel(prediction: dict[str, object] | None) -> None:
    st.subheader("ML Claim-Type Prediction")
    if not prediction:
        st.info("The classifier is not available for this run.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted type", to_label(prediction.get("predicted_claim_type")))
        st.metric("Confidence", to_label(prediction.get("confidence")))
    with col2:
        probabilities = prediction.get("probabilities", {})
        if probabilities:
            st.json(probabilities, expanded=False)


def main() -> None:
    st.set_page_config(page_title="Sports Supplement Claim Verifier", page_icon=":mag:", layout="wide")
    ensure_default_state()
    pending_claim_text = str(st.session_state.get("pending_claim_text", "")).strip()
    if pending_claim_text:
        st.session_state["claim_text"] = pending_claim_text
        st.session_state["pending_claim_text"] = ""
    render_sidebar()

    st.title("Sports Supplement Claim Verifier")
    st.caption(
        "Conservative verification pipeline with text input, image OCR, bounded ML classification, and optional local LLM support."
    )

    feature_bits = [
        "OCR: on",
        f"Classifier: {'on' if classifier_available() else 'off'}",
        f"Local LLM: {'on' if llm_is_available() else 'off'}",
    ]
    st.caption(" | ".join(feature_bits))

    st.header("Input")
    col_text, col_image = st.columns(2)

    with col_text:
        st.subheader("Claim Text")
        render_example_selector()
        st.text_area(
            "Claim",
            key="claim_text",
            height=120,
            placeholder="Enter one sports-supplement claim.",
            help="If you also upload an image, the pipeline can combine both inputs.",
        )

    with col_image:
        st.subheader("Claim Image")
        uploaded_image = st.file_uploader(
            "Add an image",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            help="Upload a label, ad, or screenshot. OCR will attempt to extract the claim.",
        )
        image_bytes = process_uploaded_claim_image(uploaded_image)

    run_clicked = st.button("Run verification", type="primary", use_container_width=True)
    if not run_clicked:
        st.info("Provide claim text, an image, or both, then run verification.")
        return

    claim_text = str(st.session_state.get("claim_text", "")).strip()
    if not claim_text and image_bytes is None:
        st.warning("Please enter a claim or upload an image before running the verifier.")
        return

    with st.spinner("Running full verification pipeline..."):
        try:
            result = get_pipeline().run(
                claim_text=claim_text or None,
                image_bytes=image_bytes,
            )
        except Exception as exc:
            st.error(f"The verifier failed while processing the input: {exc}")
            return

    st.divider()
    st.caption(
        f"Input mode: **{MODE_LABELS.get(result.get('input_mode', 'text'), result.get('input_mode', 'text'))}**"
        f" | LLM fallback used: **{'yes' if result.get('llm_used') else 'no'}**"
    )

    effective_claim = to_label(result.get("effective_claim"))
    st.subheader("Effective Claim")
    st.info(effective_claim)

    vision_result = result.get("vision_result")
    uploaded_ocr_result = st.session_state.get("ocr_result") or {}
    extracted_image_claim = str(uploaded_ocr_result.get("claim_text", "")).strip()
    has_vision_matches = bool(
        vision_result
        and (
            str(vision_result.get("detected_ingredient", "")).strip()
            or vision_result.get("detected_claims")
            or str(vision_result.get("detected_dose", "")).strip()
        )
    )
    if vision_result or extracted_image_claim:
        st.divider()
        st.subheader("Image Analysis")
        if extracted_image_claim:
            st.success(f"**Extracted claim:** {extracted_image_claim}")

        if vision_result and has_vision_matches:
            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.markdown(f"**Ingredient:** {to_label(vision_result.get('detected_ingredient'))}")
                detected_claims = vision_result.get("detected_claims", [])
                st.markdown(f"**Detected claims:** {', '.join(detected_claims) if detected_claims else '-'}")
                st.markdown(f"**Detected dose:** {to_label(vision_result.get('detected_dose'))}")
            with vcol2:
                st.markdown(f"**Vision confidence:** {to_label(vision_result.get('vision_confidence'))}")
                st.markdown(f"**Notes:** {to_label(vision_result.get('vision_notes'))}")
                st.markdown(
                    f"**Preprocessing:** {', '.join(vision_result.get('preprocessing_steps', [])) or '-'}"
                )
        details_payload = {"ocr_claim_extraction": uploaded_ocr_result, "vision_result": vision_result}
        with st.expander("Image extraction details", expanded=False):
            st.json(details_payload, expanded=False)

    st.divider()
    render_classifier_panel(result.get("claim_type_prediction"))

    reasoning_result = result.get("reasoning_result", {})
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
            for condition in conditions:
                st.markdown(f"- {condition}")
        supporting = ", ".join(reasoning_result.get("supporting_fragment_ids", []))
        limiting = ", ".join(reasoning_result.get("limiting_fragment_ids", []))
        if supporting:
            st.markdown(f"**Supporting fragment IDs:** {supporting}")
        if limiting:
            st.markdown(f"**Limiting fragment IDs:** {limiting}")

    if result.get("llm_explanation"):
        st.divider()
        st.subheader("Local LLM Explanation")
        st.markdown(str(result["llm_explanation"]))
        st.caption("The deterministic verdict remains authoritative. This explanation is optional local assistance.")

    st.divider()
    st.subheader("Parsed Claim")
    st.json(result.get("claim_parse"), expanded=False)

    if result.get("llm_extraction"):
        with st.expander("LLM extraction details", expanded=False):
            st.json(result["llm_extraction"], expanded=False)
    if result.get("llm_ocr_extraction"):
        with st.expander("LLM OCR claim selection", expanded=False):
            st.json(result["llm_ocr_extraction"], expanded=False)

    st.divider()
    st.subheader("Retrieved Evidence")
    retrieved_candidates = result.get("retrieved_candidates", [])
    if not retrieved_candidates:
        st.info("No retrieval results were returned for this claim.")
    else:
        display_rows = []
        for candidate in retrieved_candidates:
            display_rows.append(
                {
                    "rank": candidate.get("rank"),
                    "score": candidate.get("score"),
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
