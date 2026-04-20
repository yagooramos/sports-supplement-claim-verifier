# Roadmap

## Current Goal

Keep the repository coherent, runnable, multimodal, and benchmarked:

- canonical corpus
- deterministic parser, retriever, and reasoner
- offline-optimized retriever configuration
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
- [x] Integrate the claim-type classifier dataset and metrics
- [x] Integrate optional local LLM adapter support
- [x] Add structured multimodal pipeline support
- [x] Add grouped example-claim shortcuts in the Streamlit interface
- [x] Add corpus coverage sidebar with fragment counts
- [x] Add a sidebar "How it works" explainer
- [x] Refresh main documentation to match the merged repository state
- [x] Add offline genetic optimization for retrieval parameters
- [x] Version the selected retriever config for runtime use
- [x] Extend evaluation output with retrieval MRR

## Current Validation Snapshot

- Retrieval benchmark: `20/20`
- Retrieval MRR with optimized config: `0.975`
- Reasoning benchmark: `16/16`
- Streamlit startup check: passed

## Next Steps

- [ ] Validate the merged multimodal flow with real supplement photos, not only clean screenshots and the synthetic test image
- [ ] Decide whether classifier predictions should remain informational or influence downstream parsing
- [ ] Decide whether local LLM fallback should stay optional support or become a formal deliverable
- [ ] Expand the retrieval benchmark beyond the current 20-query compact set
- [ ] Add compact benchmark and corpus inspection views in the app

## Notes

The deterministic verdict path remains the authoritative project core.
The genetic algorithm is an offline support tool for retriever tuning and does not run in the normal pipeline.
