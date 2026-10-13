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

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Inputs you must download

The Zenodo archive fetched by `python -m emi_analysis.download_assets` contains only the EMI paper's aggregate results. It has **no speeches**. The EMI README points to the [OSF repository](https://doi.org/10.17605/OSF.IO/Z6UTW) for the large files. From there you need:

- **The speech corpus**, e.g. `combined_congress1879_till_2022.csv.gzip` (the EMI scripts also use a `..._filtered_nonprocedural.csv.gzip` variant). Expected columns: `speech_id`, `text`, and `congress` or `year`. Other names can be mapped with `--id-column`, `--text-column`, `--congress-column` and `--year-column`.
- **The seed lexicons** `evidence_lexicon.csv` and `intuition_lexicon.csv` (columns `evidence_keywords` / `intuition_keywords`). Put them in `data/lexicons/`. Training refuses to start without them, because seed drift is a required checkpoint metric.

## Workflow

1. **Prepare data** (CPU). This holds out speeches first, then deduplicates the training side and writes everything `train.py` reads:

   ```bash
   python -m emi_analysis.prepare_data --input data/raw/osf/<corpus>.csv.gzip
   # on Quest:
   sbatch --export=ALL,CORPUS=data/raw/osf/<corpus>.csv.gzip slurm/prepare_data.sh
   ```

   Outputs: `data/train.jsonl`, `data/heldout/{speech_ids.txt,congressional.jsonl,general.jsonl}`, `data/processed/{dedup_report.csv,prep_summary.json,prepare_data.log}`. The held-out ID file is **reused** on later runs so the split stays fixed (`--regenerate-holdout` overrides this). The general-domain sample is 2,000 WikiText-103 paragraphs by default (`--general-source`).

2. **Smoke test** (a few minutes; runs the real models on a synthetic corpus under `data/smoke/` and `checkpoints/smoke/`):

   ```bash
   python scripts/run_smoke_test.py
   ```

3. **Validate on 10k speeches** before the full run (handoff Step 2):

   ```bash
   # a random 10k speeches with 500 held out, written under data/val10k/ (the full run's data/ is untouched)
   sbatch --export=ALL,CORPUS=data/raw/osf/combined_congress1879_till_2022_filtered_nonprocedural.csv.gzip \
     slurm/prepare_data.sh --sample 10000 --holdout-count 500 --out-root data/val10k
   # once that finishes: 300 GPU steps per model on that subset; checkpoints go to checkpoints/val10k/<model>/
   sbatch slurm/bert_train.sh --data-root data/val10k --max-steps 300
   sbatch slurm/gpt2_train.sh --data-root data/val10k --max-steps 300
   ```

   Check in each log: the prep summary's `held-out IDs in training set: 0`, train's `Held-out leakage check ... OK`, the TOKEN ACCOUNTING block (tokens in = tokens out + tokens dropped; max chunk length within the limit), falling congressional loss across the checkpoints, and the steps/s on the A100. Steps/s lets you estimate the full run's duration before submitting it. Packing is not used (chunks are batched by length), so there are no cross-speech attention masks to inspect. EMI on the adapted models is not computed by this repo.

4. **Full training** (one A100 each; extra arguments after the script name are passed to `train.py`):

   ```bash
   sbatch slurm/bert_train.sh
   sbatch slurm/gpt2_train.sh
   ```

   Pass `--legacy-truncation` to reproduce the old truncating loader for comparison.

## Reading a training run

Each run gets its own folder, `checkpoints/<model>/run-job<SLURM_JOB_ID>/`, so reruns never mix with older checkpoints:

| File | Contents |
| --- | --- |
| `train.log` | Same lines as `logs/<model>_<job>.out` |
| `run_config.json` | Resolved config, environment (GPU, git commit), token accounting, step schedule |
| `metrics.csv` | One row for step 0 (unadapted model) and one per checkpoint: `model, step, congressional_loss, general_loss, mean_seed_drift`, then `max_seed_drift, loss_metric, epoch, train_loss, tokens_seen, elapsed_seconds, flags, checkpoint` |
| `checkpoint-<step>/` | Model and tokenizer, plus `checkpoint_info.json` with that checkpoint's metrics |
| `run_summary.json` | `completed`, `stopped_on_flag` or `failed` (with the error) |

The log is organised as follows:

- **Header block.** Host, GPU, precision, git commit, Python, job ID.
- **TOKEN ACCOUNTING.** Tokens in, tokens out in chunks, tokens dropped as chunks under `min_chunk_tokens`, and the longest chunk. The run aborts if in ≠ out + dropped, or if any chunk exceeds `max_length`.
- **TRAINING PLAN.** Total optimizer steps, the checkpoint steps (every 10%), learning-rate schedule and stop thresholds.
- **BASELINE.** Held-out losses of the unadapted model (step 0).
- **Progress lines** every `log_every` steps: step, % done, epoch, train loss, learning rate, throughput, elapsed time and ETA. If the ETA exceeds the job's remaining SLURM time, a warning is logged early in the run.
- **CHECKPOINT k/10 blocks.** Where the checkpoint was saved, held-out congressional and general loss with the % change from step 0, seed drift mean/max, and any flags.

`congressional_loss`/`general_loss` are MLM loss for BERT and perplexity for GPT-2 (`loss_metric` column). Compare each model only to its own step 0.

**Stop conditions.** A flag fires when general loss is more than `general_loss_max_rel_increase` (default 20%) above step 0, or when mean seed drift exceeds `seed_drift_threshold` (default 0.10) and is still rising. On a flag, the run logs `STOP CONDITION`, saves that checkpoint and exits with code 3, so `sacct` shows the job as FAILED. The handoff does not quantify "sharp" or "large", so these defaults are starting points. Set `stop_on_flag: false` to log flags without stopping.
