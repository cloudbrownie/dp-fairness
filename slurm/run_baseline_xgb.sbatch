#!/bin/bash
#SBATCH --job-name=baseline_xgb
#SBATCH --output=logs/baseline_xgb_%j.out
#SBATCH --error=logs/baseline_xgb_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00

# Clean XGB baseline: all four fairness interventions on prepared ACS data.
# expgrad uses a STRATIFIED 20k subsample (min 200 per RAC1P group), matching
# the LR baseline and both grid scripts. Writes data/results/baseline_xgb.csv.
#
# Estimated wall time:
#   unmitigated  ~5-15s
#   reweighing  ~5-15s
#   expgrad      ~3-6 min  (subsample=20k, n_estimators=300)
#   threshold    ~15-30s
# Total: ~5-8 min; 0:30:00 has adequate headroom.

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

exec python train_baseline_xgb.py \
  --data-dir   data/raw \
  --output-dir data/results \
  --output-name baseline_xgb.csv \
  --expgrad-subsample 20000 \
  --min-per-group 200 \
  --seed 42
