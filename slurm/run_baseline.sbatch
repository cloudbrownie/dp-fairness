#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=1:00:00

# Clean LR baseline: runs all four fairness interventions (unmitigated,
# reweighing, expgrad, threshold) on the prepared ACS data. expgrad
# uses a STRATIFIED 20k subsample (min 200 per RAC1P group) for its
# oracle calls - matches full-train-156k on the clean data within 3
# decimal places at ~7x lower compute, and matches the sampling scheme
# used in the grid runs so the clean/private comparison is like-for-like.
#
# Expected wall time: ~15-20 min (expgrad dominates at stratified-20k).

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs data/results

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

exec python train_baseline.py \
  --data-dir data/raw \
  --output-dir data/results \
  --output-name baseline.csv \
  --expgrad-subsample 20000 \
  --min-per-group 200 \
  --seed 42
