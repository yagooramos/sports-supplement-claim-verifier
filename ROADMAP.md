# Roadmap

## Current Goal

Keep the repository coherent, runnable, multimodal, and benchmarked:

- canonical corpus
- deterministic parser, retriever, and reasoner
- OCR-assisted image input
- multimodal pipeline
- bounded ML classifier
- optional local LLM support

## Completed

- [x] Stabilize the canonical repository structure
- [x] Validate deterministic retrieval and reasoning
- [x] Add OCR-based claim extraction
- [x] Add a package-safe `scripts/` layout
- [x] Add benchmark runner for deterministic evaluation
- [x] Merge the useful advanced components from `.claude/worktrees/lucid-jones`
- [x] Integrate the claim-type classifier dataset and metrics
- [x] Integrate optional local LLM adapter support
- [x] Add structured multimodal pipeline support
- [x] Add grouped example-claim shortcuts in the Streamlit interface
- [x] Add corpus coverage sidebar with fragment counts
- [x] Add a sidebar "How it works" explainer
- [x] Refresh main documentation to match the merged repository state

## Current Validation Snapshot

- Retrieval benchmark: `20/20`
- Reasoning benchmark: `16/16`
- Streamlit startup check: passed

## Next Steps

- [ ] Validate the merged multimodal flow with real supplement photos, not only clean screenshots and the synthetic test image
- [ ] Decide whether classifier predictions should remain informational or influence downstream parsing
- [ ] Decide whether local LLM fallback should stay optional support or become a formal deliverable
- [ ] Add compact benchmark and corpus inspection views in the app

## Notes

The deterministic verdict path remains the authoritative project core.
