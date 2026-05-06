#!/bin/bash
#SBATCH --job-name=pb_synth
#SBATCH --output=logs/pb_synth_%A_%a.out
#SBATCH --error=logs/pb_synth_%A_%a.err
#SBATCH --array=0-39
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00

# One array task = one (epsilon, seed). 40 tasks: eps in {1,2,4,8} × seed in 0..9
# Mirrors generate_synth_mst_array.sbatch (PrivBayes is closer to MST in cost
# than AIM). Tune --cpus-per-task / --mem / --time if jobs OOM or timeout.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs data/synth_pb

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

NJOBS="${SLURM_CPUS_PER_TASK:-2}"

exec python generate_synth_pb.py \
  --only-epsilon "${EPS}" \
  --only-seed "${SEED}" \
  --output-dir data/synth_pb \
  --format parquet \
  --n-jobs "${NJOBS}"
