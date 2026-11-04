#!/bin/bash
#SBATCH --job-name=emi-keywords
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/keywords_%j.out
#SBATCH --error=logs/keywords_%j.err
#SBATCH --partition=gengpu
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# In-context hidden states of the sampled keyword occurrences for one model (all layers), plus each keyword alone.
# Add --base for the unadapted model; add --sae <sae.pt> once an SAE exists (stores its features per occurrence).
# Precision defaults to strict fp32 for ModernBERT and bf16 for the others.
#   sbatch slurm/keyword_activations.sh bert --run-dir checkpoints/bert-continued/run-job4757959 \
#     --occurrences emi/occurrences/corpus

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

models=()
args=()
for arg in "$@"; do
  case "$arg" in
    bert|gpt2|modernbert|qwen3) models+=("$arg") ;;
    *) args+=("$arg") ;;
  esac
done
[[ ${#models[@]} -eq 0 ]] && { echo "Name a model: bert, gpt2, modernbert or qwen3" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi unavailable"
status=0
for model in "${models[@]}"; do
  echo "=== keyword activations: $model ==="
  python -m emi_analysis.keyword_activations --model "$model" --require-gpu "${args[@]+"${args[@]}"}" || status=1
done
exit $status
