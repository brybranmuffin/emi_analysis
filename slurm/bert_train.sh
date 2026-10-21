#!/bin/bash
#SBATCH --job-name=emi-bert
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/bert_%j.out
#SBATCH --error=logs/bert_%j.err
#SBATCH --partition=gengpu
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Full continued-pretraining run. Needs the outputs of slurm/prepare_data.sh and the seed lexicons
# in data/lexicons/. Progress, token accounting and every checkpoint are logged to logs/bert_<job>.out
# and checkpoints/bert/run-job<job>/. Exit code 3 means a stop condition fired (see the log).

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/.."
mkdir -p logs checkpoints/bert .cache/huggingface

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

python -m emi_analysis.train --config configs/bert_continued_pretraining.json --require-gpu "$@"
