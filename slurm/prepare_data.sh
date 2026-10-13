#!/bin/bash
#SBATCH --job-name=emi-prep
#SBATCH --account=e32706
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#SBATCH --partition=short
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Builds data/train.jsonl and the fixed held-out sets (CPU only). Usage:
#   sbatch --export=ALL,CORPUS=data/raw/osf/<corpus>.csv.gzip slurm/prepare_data.sh
# Extra prepare_data flags go after the script name, e.g. for the 10k validation subset:
#   sbatch --export=ALL,CORPUS=... slurm/prepare_data.sh --sample 10000 --holdout-count 500 --out-root data/val10k

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/.."
mkdir -p logs .cache/huggingface
: "${CORPUS:?Set CORPUS to the speech corpus file: sbatch --export=ALL,CORPUS=<path> slurm/prepare_data.sh}"

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "WARNING: no .venv in $PWD; using $(command -v python)" >&2
fi
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')"
echo "python $(command -v python) ($(python --version 2>&1)); git $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

python -m emi_analysis.prepare_data --input "$CORPUS" "$@"
