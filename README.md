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
   sbatch slurm/modernbert_train.sh --data-root data/val10k --max-steps 300
   sbatch slurm/qwen3_train.sh --data-root data/val10k --max-steps 300
   ```

   Check in each log: the prep summary's `held-out IDs in training set: 0`, train's `Held-out leakage check ... OK`, the TOKEN ACCOUNTING block (tokens in = tokens out + tokens dropped; max chunk length within the limit), falling congressional loss across the checkpoints, and the steps/s on the GPU. Steps/s lets you estimate the full run's duration before submitting it. For ModernBERT and Qwen3, also check the PACKING block (fill rate, and tokens in packs = tokens in chunks). EMI on the adapted models is not computed by this repo.

4. **Full training** (one GPU each; extra arguments after the script name are passed to `train.py`):

   ```bash
   sbatch slurm/bert_train.sh
   sbatch slurm/gpt2_train.sh
   sbatch slurm/modernbert_train.sh
   sbatch slurm/qwen3_train.sh
   ```

   Pass `--legacy-truncation` to reproduce the old truncating loader for comparison (BERT/GPT-2).

## Models

| Model | Objective | Input | Per optimizer step | Epochs |
| --- | --- | --- | --- | --- |
| `bert-base-uncased` | masked LM, 15% | chunks of 256 tokens, batched by length | 32 chunks (~5.6k tokens) | 2.5 |
| `gpt2` | causal LM | chunks of 512 tokens, batched by length | 16 chunks (~3.8k tokens) | 1.5 |
| `answerdotai/ModernBERT-base` | masked LM, 30% (its pretraining rate) | packed sequences of 2,048 tokens | 3 packs (~6k tokens) | 1.25 |
| `Qwen/Qwen3-0.6B-Base` | causal LM | packed sequences of 2,048 tokens | 2 packs (~4k tokens) | 1.0 |

All four use the same learning rate (5e-5, linear decay, no warmup), data, held-out sets and verification. Weights are always loaded and trained in float32, with bf16 used only for the GPU computation (autocast). Qwen3's checkpoint is stored in bf16 and would otherwise be trained in bf16, where many small updates round away. ModernBERT's 1.25 epochs at 30% masking give the same number of masked-token predictions per token as BERT's 2.5 epochs at 15%.

**Packing (ModernBERT, Qwen3).** These models read far longer inputs than a typical speech, so several speech chunks are packed into each 2,048-token sequence (best-fit, ~99% real tokens). Each chunk behaves exactly as if it were alone:
- its position ids restart at 0;
- attention never crosses into another chunk, in ModernBERT's global and sliding-window layers as well as Qwen3's causal layers;
- for Qwen3, the first token of a chunk is never predicted from the end of the previous one.

`tests/test_packing.py` proves this. A chunk's outputs inside a pack match its outputs alone, and changing its neighbour changes nothing; the real ModernBERT-base and Qwen3-0.6B-Base give identical outputs, a difference of exactly 0. The PACKING block in the log reports the packs, the fill rate, and a check that no chunk or token was lost.

## Clean-text check (is the general-domain rise real?)

The general-domain set is raw WikiText, which spaces out punctuation ("film , television", "player 's"). Byte-level tokenizers (GPT-2, ModernBERT, Qwen3) see that spacing, so part of a measured general-domain rise can be formatting rather than forgetting. The check scores the same paragraphs raw and detokenized, for the unadapted model and the final checkpoint, using the training evaluation code. The raw scores therefore reproduce the training log.

```bash
sbatch slurm/clean_text_check.sh                          # all four models (latest completed run of each)
sbatch slurm/clean_text_check.sh qwen3 --all-checkpoints  # one model, every checkpoint
python -m emi_analysis.clean_text_check --model gpt2 --run-dir checkpoints/gpt2/run-job9843505
```

Each model gets `results/clean_text_check/<model>_<run>.csv`, with one row per checkpoint and text set, and a `.json` summary. The summary holds the rise on raw and clean text and the share of the raw loss rise explained by formatting. BERT is a control: its tokenizer ignores the spacing, so its raw and clean scores are identical.

## Seed-axis check and the adaptation table

Training's seed-drift number measures how far each seed word's embedding moved. A shift shared by all embeddings inflates it without changing what EMI uses. `emi_analysis.seed_axis_check` re-embeds the seed words at step 0 and at every checkpoint and measures:
- **axis cosine:** does the evidence-minus-intuition direction still point the same way?
- **relational stability:** do the seed words keep the same pairwise-similarity pattern?
- **seed order along the axis**;
- **pole separation:** is each word still closer to its own leave-one-out pole?

It then joins these with congressional loss and clean general-domain loss into one table per checkpoint, `results/seed_axis_check/<model>_<run>.csv`. Each checkpoint gets a provisional verdict: `stable`, `watch` or `moved`. The recomputed raw drift is checked against the training log.

```bash
sbatch slurm/adaptation_check.sh               # all four models: clean-text check on every checkpoint, then seed-axis check
sbatch slurm/adaptation_check.sh qwen3         # one model
```

## Reading a training run

Each run gets its own folder, `checkpoints/<model>/run-job<SLURM_JOB_ID>/` (or `checkpoints/<data-root name>/<model>/...` with `--data-root`), so reruns never mix with older checkpoints:

| File | Contents |
| --- | --- |
| `train.log` | Same lines as `logs/<model>_<job>.out` |
| `progress.json` | Live snapshot, rewritten at every progress line and evaluation: step, epoch, ETA, recent train loss, latest checkpoint, baseline and latest evaluation, flags so far |
| `run_config.json` | Resolved config, environment (GPU, git commit), token accounting, step schedule including epoch-end steps |
| `metrics.csv` | One row for step 0 (unadapted model), one per 10% checkpoint and one per epoch end: `model, step, congressional_loss, general_loss, mean_seed_drift`, then `max_seed_drift, loss_metric, epoch, train_loss, tokens_seen, elapsed_seconds, flags, checkpoint, event` |
| `checkpoint-<step>/` | Model and tokenizer, plus `checkpoint_info.json` with that evaluation's metrics |
| `run_summary.json` | `completed`, `cancelled` (scancel or time limit), `stopped_on_flag` or `failed` (with the error); flags raised; latest checkpoint |

### Checking on a run, and stopping one

```bash
python -m emi_analysis.status                      # latest run in every folder under checkpoints/
python -m emi_analysis.status checkpoints/val10k   # just the validation runs
```

For each run this prints the status, progress, epoch, ETA and latest checkpoint. It also prints a table of every evaluation so far, with the change from step 0 for congressional and general loss, seed drift, train loss and any flags. That is everything needed to decide whether to stop. To stop a run, `scancel <jobid>` (the status output prints the exact command). The run catches the signal, writes `run_summary.json` with status `cancelled` and exits with code 143. Everything up to the latest checkpoint is kept, and that checkpoint is a usable model. A run cut off by the SLURM time limit is handled the same way.

### The log

- **Header block.** Host, GPU, precision, git commit, Python, job ID.
- **TOKEN ACCOUNTING.** Tokens in, tokens out in chunks, tokens dropped as chunks under `min_chunk_tokens`, and the longest chunk. The run aborts if in ≠ out + dropped, or if any chunk exceeds `max_length`.
- **TRAINING PLAN.** Total optimizer steps, checkpoint steps (every 10%), epoch-end steps, learning-rate schedule, flag thresholds and whether a flag stops the run.
- **BASELINE.** Held-out losses of the unadapted model (step 0).
- **Progress lines** every `log_every` steps: step, % done, epoch, train loss, learning rate, throughput, elapsed time and ETA. If the ETA exceeds the job's remaining SLURM time, a warning is logged early in the run.
- **CHECKPOINT k/10 and END OF EPOCH n blocks.** Where the checkpoint was saved, held-out congressional and general loss with the % change from step 0, seed drift mean/max, train loss since the previous evaluation (and over the whole epoch at an epoch end), and any flags. An epoch end that falls between 10% checkpoints is evaluated and checkpointed as well.

`congressional_loss`/`general_loss` are MLM loss for BERT and perplexity for GPT-2 (`loss_metric` column). Compare each model only to its own step 0.

Warnings, flags and stop reasons are also written to `logs/<model>_<job>.err`, so that file holds everything worth looking at. It also contains a harmless transformers "LOAD REPORT" about unused pretrained weights.

**Flags.** A flag fires when general loss is more than `general_loss_max_rel_increase` (default 20%) above step 0, or when mean seed drift exceeds `seed_drift_threshold` (default 0.10) and is still rising. The handoff does not quantify "sharp" or "large", so these thresholds are starting points. By default a flag is only recorded (`FLAG at step ...` in the log and `.err`, the `flags` column, `flags_raised` in the summary) and training continues (`stop_on_flag: false` in both configs). Pass `--stop-on-flag` to make a flag end the run instead. It then saves that checkpoint and exits with code 3, which SLURM reports as FAILED.

**Exit codes.** 0 = completed, 1 = error (traceback in `.err`), 3 = stopped on a flag (`--stop-on-flag`), 143 = cancelled by `scancel` or the time limit.
