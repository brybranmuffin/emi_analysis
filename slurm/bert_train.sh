#!/bin/bash
#SBATCH --job-name=emi-bert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/bert_%j.out
#SBATCH --error=logs/bert_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/.."
mkdir -p logs checkpoints/bert .cache/huggingface

source .venv/bin/activate || true
python -m emi_analysis.download_assets
python -m emi_analysis.train --config configs/bert_continued_pretraining.json --max-steps 5 --checkpoint-every 2
