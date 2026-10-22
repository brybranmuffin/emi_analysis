#!/bin/bash
#SBATCH --job-name=emi-activations
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/activations_%j.out
#SBATCH --error=logs/activations_%j.err
#SBATCH --partition=gengpu
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Speech-pooled activations for every layer at a model's peak checkpoint (see emi_analysis/collect_activations.py).
# One model per job when passing --run-dir. Re-submitting the same command resumes an interrupted collection.
#   sbatch slurm/collect_activations.sh bert --run-dir checkpoints/bert-continued/run-job4757959         # held-out set
#   sbatch slurm/collect_activations.sh gpt2 --run-dir checkpoints/gpt2/run-job9843505 \
#     --input data/train.jsonl --input data/heldout/congressional.jsonl --tag corpus                     # full corpus

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/.."
mkdir -p logs activations .cache/huggingface

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
args=()
for arg in "$@"; do
  case "$arg" in
    bert|gpt2|modernbert|qwen3) models+=("$arg") ;;
    *) args+=("$arg") ;;
  esac
done
[[ ${#models[@]} -eq 0 ]] && { echo "Name at least one model: bert, gpt2, modernbert or qwen3" >&2; exit 2; }

status=0
for model in "${models[@]}"; do
  echo "=== activation collection: $model ==="
  python -m emi_analysis.collect_activations --model "$model" --require-gpu "${args[@]+"${args[@]}"}" || status=1
done
exit $status
