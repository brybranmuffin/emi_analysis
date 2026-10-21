#!/bin/bash
#SBATCH --job-name=emi-adaptcheck
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/adaptcheck_%j.out
#SBATCH --error=logs/adaptcheck_%j.err
#SBATCH --partition=gengpu
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Adaptation check (evaluation only), per model: (1) clean-text check on every checkpoint, then (2) seed-axis check,
# which joins congressional loss, clean general-domain loss and axis-level drift into one table per checkpoint.
# Results: results/clean_text_check/ and results/seed_axis_check/.
#   sbatch slurm/adaptation_check.sh               # all four models (latest completed run of each)
#   sbatch slurm/adaptation_check.sh qwen3 gpt2    # selected models

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/.."
mkdir -p logs results/clean_text_check results/seed_axis_check .cache/huggingface

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

models=()
flags=()
for arg in "$@"; do
  if [[ "$arg" == --* ]]; then flags+=("$arg"); else models+=("$arg"); fi
done
[[ ${#models[@]} -eq 0 ]] && models=(bert gpt2 modernbert qwen3)

status=0
for model in "${models[@]}"; do
  echo "=== adaptation check: $model ==="
  if python -m emi_analysis.clean_text_check --model "$model" --require-gpu --all-checkpoints "${flags[@]+"${flags[@]}"}"; then
    python -m emi_analysis.seed_axis_check --model "$model" --require-gpu "${flags[@]+"${flags[@]}"}" || status=1
  else
    status=1
  fi
done
exit $status
