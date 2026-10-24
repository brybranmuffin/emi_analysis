#!/bin/bash
#SBATCH --job-name=emi-verify
#SBATCH --account=p53299
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#SBATCH --partition=gengpu
#SBATCH --mail-user=bettencourt@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Verify collected activations (see emi_analysis/verify_activations.py). With no arguments, every collection under
# activations/ is checked and their speech orders compared. Exit code 1 means a check failed (details in the log).
#   sbatch slurm/verify_activations.sh
#   sbatch slurm/verify_activations.sh activations/bert/run-job4757959/step350847/corpus --spot-check 100

set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/.."
mkdir -p logs

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "WARNING: no .venv in $PWD; using $(command -v python)" >&2
fi
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')"

if [[ $# -eq 0 || "$1" == --* ]]; then
  mapfile -t dirs < <(find activations -mindepth 5 -maxdepth 5 -name meta.json -exec dirname {} \; | sort)
  [[ ${#dirs[@]} -eq 0 ]] && { echo "No collections found under activations/" >&2; exit 2; }
  set -- "${dirs[@]}" "$@"
fi
echo "verifying: $*"
python -m emi_analysis.verify_activations "$@" --require-gpu --report "logs/verify_${SLURM_JOB_ID:-local}.json"
