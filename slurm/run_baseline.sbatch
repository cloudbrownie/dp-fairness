#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00

# Runs all four fairness interventions (unmitigated, reweighting, expgrad, threshold)
# on the prepared ACS data. expgrad runs on the full training set (no subsample)
# to guarantee all 9 RAC1P groups are represented in every oracle call.
# Expected wall time: ~1.5-2 hrs (expgrad dominates); 4:00:00 is a safe buffer.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs results

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

# Let BLAS use all allocated CPUs. expgrad's oracle calls are sequential so
# multi-threaded BLAS per call is faster than single-threaded with no parallelism.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

exec python train_baseline.py \
  --data-dir data/raw \
  --output-dir results \
  --expgrad-subsample 0 \
  --seed 42
