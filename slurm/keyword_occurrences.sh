#!/bin/bash
#SBATCH --job-name=emi-occurrences
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/occurrences_%j.out
#SBATCH --error=logs/occurrences_%j.err
#SBATCH --partition=short
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Era-balanced sample of the dictionary keywords as they occur in the speeches (CPU; once, shared by all models).
#   sbatch slurm/keyword_occurrences.sh --input data/train.jsonl --input data/heldout/congressional.jsonl
# Output: emi/occurrences/corpus/ (see emi_analysis/keyword_occurrences.py).

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
python -m emi_analysis.keyword_occurrences --num-proc "${SLURM_CPUS_PER_TASK:-1}" "$@"
