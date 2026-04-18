#!/bin/bash
#SBATCH --job-name=grid
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err
#SBATCH --array=0-39
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:30:00

# One array task = one (epsilon, seed) pair, all four interventions.
# Array size 0-19: eps in {1,2,4,8} x seed in {0..4} (5 seeds).
# Change to 0-39 for all 10 seeds.
#
# Task index layout: TASK_ID = EPS_IDX * 5 + SEED  (for 5 seeds)
# EPS_IDX: 0=eps1, 1=eps2, 2=eps4, 3=eps8

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs data/results/grid

EPS_LIST=(1 2 4 8)
N_SEEDS=10   # seeds 0..4; change to 10 and array to 0-39 for full grid

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
EPS_IDX=$((TASK_ID / N_SEEDS))
SYNTH_SEED=$((TASK_ID % N_SEEDS))
EPS="${EPS_LIST[${EPS_IDX}]}"

echo "task=${TASK_ID}  eps=${EPS}  synth_seed=${SYNTH_SEED}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

exec python train_grid.py \
  --data-dir    data/raw \
  --synth-dir   data/synth \
  --output-dir  data/results/grid \
  --epsilon     "${EPS}" \
  --synth-seed  "${SYNTH_SEED}" \
  --expgrad-subsample 20000 \
  --seed        42
