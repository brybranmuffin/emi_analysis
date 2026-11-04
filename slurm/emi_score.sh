#!/bin/bash
#SBATCH --job-name=emi-score
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/emi_score_%j.out
#SBATCH --error=logs/emi_score_%j.err
#SBATCH --partition=normal
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Contextual EMI for every speech, one model and representation (see emi_analysis/emi_score.py). Raw runs on CPU.
#   sbatch slurm/emi_score.sh --model bert --representation raw \
#     --activations activations/bert/run-job4757959/step350847/corpus \
#     --keywords emi/keywords/bert/run-job4757959/step350847/corpus --occurrences emi/occurrences/corpus
# The speech-SAE representation encodes 8.4M vectors: give it a GPU on the command line:
#   sbatch --partition=gengpu --gres=gpu:h100:1 slurm/emi_score.sh --model bert --representation speech_sae --sae <sae.pt> ...

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
python -m emi_analysis.emi_score "$@"
