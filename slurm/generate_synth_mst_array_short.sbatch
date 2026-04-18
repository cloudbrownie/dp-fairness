#!/bin/bash
#SBATCH --job-name=mst_synth
#SBATCH --output=logs/mst_synth_%A_%a.out
#SBATCH --error=logs/mst_synth_%A_%a.err
#SBATCH --array=0-39
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:10:00

# One array task = one (epsilon, seed). 40 tasks: eps in {1,2,4,8} × seed in 0..9

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs data/synth

EPS_LIST=(1 2 4 8)
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
EPS_IDX=$((TASK_ID / 10))
SEED=$((TASK_ID % 10))
EPS="${EPS_LIST[${EPS_IDX}]}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NJOBS="${SLURM_CPUS_PER_TASK:-1}"

exec python generate_synth_mst.py \
  --only-epsilon "${EPS}" \
  --only-seed "${SEED}" \
  --output-dir data/synth \
  --format parquet \
  --n-jobs "${NJOBS}"
