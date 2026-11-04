#!/bin/bash
#SBATCH --job-name=emi-sae
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=160G
#SBATCH --output=logs/train_sae_%j.out
#SBATCH --error=logs/train_sae_%j.err
#SBATCH --partition=gengpu
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Train a TopK SAE on one layer (see emi_analysis/train_sae.py). The layer comes from the raw scoring's primary layer.
#   sbatch slurm/train_sae.sh --model bert --kind token --activations activations/bert/run-job4757959/step350847/corpus \
#     --layer-from results/emi/raw/bert/run-job4757959/step350847/meta.json --seed 0
#   ... --kind speech (trains on the stored speech vectors; needs ~2 bytes x hidden x 8.4M of memory)

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/.."
mkdir -p logs .cache/huggingface

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "WARNING: no .venv in $PWD; using $(command -v python)" >&2
fi
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')"
echo "python $(command -v python) ($(python --version 2>&1)); git $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi unavailable"
python -m emi_analysis.train_sae --require-gpu "$@"
