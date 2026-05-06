#!/bin/bash
#SBATCH --job-name=grid_xgb_pb
#SBATCH --output=logs/grid_xgb_pb_%A_%a.out
#SBATCH --error=logs/grid_xgb_pb_%A_%a.err
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# XGB grid on PrivBayes-synthesized data. One array task = one (epsilon, seed)
# pair. Ten cells per task: unmitigated, reweighing (DP),
# expgrad-{dp,eo} x {uniform,stratified}, threshold-{dp,eo} x {naive,honest}.
# Reuses train_grid_xgb.py unchanged; only input/output directories change
# vs run_grid_xgb_array.sbatch.
#
# Task index layout: TASK_ID = EPS_IDX * N_SEEDS + SEED
# EPS_IDX: 0=eps1, 1=eps2, 2=eps4, 3=eps8

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs data/results/grid_xgb_pb

EPS_LIST=(1 2 4 8)
N_SEEDS=10

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
EPS_IDX=$((TASK_ID / N_SEEDS))
SYNTH_SEED=$((TASK_ID % N_SEEDS))
EPS="${EPS_LIST[${EPS_IDX}]}"

echo "task=${TASK_ID}  eps=${EPS}  synth_seed=${SYNTH_SEED}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

# XGBoost uses n_jobs=4 internally; BLAS threading is less critical here but
# set consistently with the LR grid for reproducibility across nodes.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

exec python train_grid_xgb.py \
  --data-dir    data/raw \
  --synth-dir   data/synth_pb \
  --output-dir  data/results/grid_xgb_pb \
  --epsilon     "${EPS}" \
  --synth-seed  "${SYNTH_SEED}" \
  --expgrad-subsample 20000 \
  --min-per-group 200 \
  --seed        42
