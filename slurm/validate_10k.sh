#!/bin/bash
#SBATCH --job-name=emi-validate
#SBATCH --account=e32706
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/validate_%j.out
#SBATCH --error=logs/validate_%j.err
#SBATCH --partition=gengpu
#SBATCH --mail-user=bettencourt@u.northwestern.edu


set -euo pipefail

cd "$SLURM_SUBMIT_DIR" || cd "$(dirname "$0")/.."
mkdir -p logs

source .venv/bin/activate || true
python scripts/validate_10k.py --input data/train.jsonl --max-length 256
