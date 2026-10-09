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

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download the dataset archive and cache the Hugging Face models locally if they are not already present:

```bash
python -m emi_analysis.download_assets
```

Run a smoke test before the full training run:

```bash
python scripts/run_smoke_test.py
```

Run a small local training job directly with the repo config files:

```bash
python -m emi_analysis.train --config configs/bert_continued_pretraining.json --max-steps 2 --checkpoint-every 1
python -m emi_analysis.train --config configs/gpt2_continued_pretraining.json --max-steps 2 --checkpoint-every 1
```

## Quest (SLURM)

On Quest, submit each model job from the repo root:

```bash
cd /path/to/emi_analysis
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m emi_analysis.download_assets
sbatch slurm/bert_train.sh
sbatch slurm/gpt2_train.sh
```

The job scripts use a single A100 GPU and write logs to `logs/` while saving checkpoints under `checkpoints/bert` and `checkpoints/gpt2`.

If you want to inspect the queued jobs or output:

```bash
squeue -u $USER
ls -R logs
ls -R checkpoints
```

## Slurm

The included batch scripts assume the project is mounted on a cluster filesystem and a Python environment is available in the workspace. Each script prepares `logs/`, creates the cache directory, downloads data/model assets if needed, and launches the continued-pretraining job with the matching JSON config.

## Validation

Before running the full corpus, the project validates on a 10k-speech subset:

1. Check that total output tokens approximately match total input tokens after deduplication and minimum-length drops.
2. Ensure no chunk exceeds the model max length.
3. Verify that held-out speech IDs never appear in the training set.
4. Inspect attention masks for boundary blocking when packing is enabled.

## Notes

This repository is intentionally structured as a runnable starter for the handoff workflow. Actual dataset paths, cluster names, and full model checkpoints should be filled in for your specific compute environment.
