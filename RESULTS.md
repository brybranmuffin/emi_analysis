# Training results status

This repository is configured for the continued-pretraining and EMI verification pipeline described in the handoff.

## Current status

- The speech truncation bug is explicitly guarded against by chunking rather than truncating.
- BERT is configured for a 256-token context window.
- GPT-2 is configured for a 512-token context window.
- Held-out speech IDs are meant to be kept fixed before any processing and stored under `data/heldout/`.
- Verification checkpoints are written to the canonical CSV schema:
  - `model`
  - `step`
  - `congressional_loss`
  - `general_loss`
  - `mean_seed_drift`

## Missing runtime artifacts

Actual pipeline outputs such as measured token counts, deduplication breakdowns, held-out IDs, adapted checkpoints, and final EMI correlations must be produced by running the data pipeline and training scripts on the actual congressional corpus.

This repository includes the scripts and configuration to do that, but the raw speech corpus and cluster runtime environment are not present in the workspace.

## Expected final reports

When the full run completes, this file should be updated with:

- token counts before and after the chunking fix
- dedup counts by era
- final EMI Pearson and Spearman correlations for BERT and GPT-2
- any stop conditions or anomalies that were encountered
