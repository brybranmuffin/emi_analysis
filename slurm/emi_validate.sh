#!/bin/bash
#SBATCH --job-name=emi-validate
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/emi_validate_%j.out
#SBATCH --error=logs/emi_validate_%j.err
#SBATCH --partition=short
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Validation against the W2V series and the Prolific ratings (see emi_analysis/emi_validate.py); run last.
#   sbatch slurm/emi_validate.sh score --scores results/emi/raw/bert/run-job4757959/step350847 \
#     --excerpt-activations activations/bert/run-job4757959/step350847/prolific \
#     --prolific data/raw/extracted/saroyehun-EvidenceMinusIntuition-4200ef3/data/prolific_ratings_US_congress_congressw2v.csv \
#     --w2v data/raw/extracted/saroyehun-EvidenceMinusIntuition-4200ef3/data/congress_EMI_w2v_bootstrap_CIs.csv

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
python -m emi_analysis.emi_validate "$@"
