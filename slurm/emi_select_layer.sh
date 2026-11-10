#!/bin/bash
#SBATCH --job-name=emi-select-layer
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/emi_select_layer_%j.out
#SBATCH --error=logs/emi_select_layer_%j.err
#SBATCH --partition=short
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Re-choose the primary layer of finished emi_score results by leave-one-keyword-out speech AUC
# (see emi_analysis/emi_select_layer.py). Updates meta.json and layer_selection.csv; scores are not recomputed.
#   sbatch slurm/emi_select_layer.sh results/emi/raw/bert/run-job4757959/step350847 \
#     results/emi/raw/gpt2/run-job9843505/step759917 results/emi/raw/modernbert/run-job4757963/step316779 \
#     results/emi/raw/qwen3/run-job4757993/step481486

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/.."
mkdir -p logs

export PYTHONUNBUFFERED=1

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "WARNING: no .venv in $PWD; using $(command -v python)" >&2
fi
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')"
echo "python $(command -v python) ($(python --version 2>&1)); git $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
python -m emi_analysis.emi_select_layer "$@"
