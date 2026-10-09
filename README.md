# EMI Continued-Pretraining Project

This repository is set up for the congressional-speech continued pretraining workflow described in the project handoff. The main bug fixed here is the speech truncation bug: training previously cut speeches at 256 tokens for BERT and 512 tokens for GPT-2, dropping all later tokens from the gradient signal. The rebuilt pipeline chunks each speech into non-overlapping windows, packs short chunks when needed, and validates token conservation before the full run.

## Core requirements from the handoff

- Do not truncate speeches before the model max length is reached. Instead, split them into windows and keep every token.
- BERT uses a 256-token context window; GPT-2 uses a 512-token window.
- Training must be run on a held-out speech split created before any data processing.
- Each checkpoint must record:
  - model
  - step
  - congressional loss
  - general loss
  - mean seed drift
- For BERT, evaluate held-out congressional loss using MLM loss; for GPT-2, use perplexity.
- General-domain loss should remain roughly flat; a sharp rise is a stopping condition.
- Mean seed drift should remain small and not keep rising.
- For the final decision, correlate adapted-model EMI against the off-the-shelf baseline using Pearson and Spearman.

## Project layout

- `src/emi_analysis/`: pipeline and training utilities
- `configs/`: model-specific training configs
- `slurm/`: batch submission scripts for HPC clusters
- `scripts/`: validation and run helpers
- `data/heldout/`: held-out IDs and generated artifacts
- `docs/`: notes and verification requirements

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m emi_analysis.train --config configs/bert_continued_pretraining.yaml
python -m emi_analysis.train --config configs/gpt2_continued_pretraining.yaml
```

For Slurm jobs, use the scripts in `slurm/`.

## Slurm

The included batch scripts assume the project is mounted on a cluster filesystem and a Python environment is available in the workspace. They submit a job with the correct model config, output logging, and checkpointing behavior.

## Validation

Before running the full corpus, the project validates on a 10k-speech subset:

1. Check that total output tokens approximately match total input tokens after deduplication and minimum-length drops.
2. Ensure no chunk exceeds the model max length.
3. Verify that held-out speech IDs never appear in the training set.
4. Inspect attention masks for boundary blocking when packing is enabled.

## Notes

This repository is intentionally structured as a runnable starter for the handoff workflow. Actual dataset paths, cluster names, and full model checkpoints should be filled in for your specific compute environment.
