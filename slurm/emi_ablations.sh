#!/bin/bash
#SBATCH --job-name=emi-ablations
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/emi_ablations_%j.out
#SBATCH --error=logs/emi_ablations_%j.err
#SBATCH --partition=short
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Held-out ablations for one scores folder (see emi_analysis/emi_ablations.py). --compare adds other folders
# (other representations, the base model, the second SAE seed, SAE sweep configs).
#   sbatch slurm/emi_ablations.sh --scores results/emi/raw/bert/run-job4757959/step350847 \
#     --compare results/emi/raw/bert/base/step0

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
python -m emi_analysis.emi_ablations "$@"
