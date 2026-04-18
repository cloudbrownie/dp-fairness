#!/bin/bash
#SBATCH --job-name=mst_synth_all
#SBATCH --output=logs/mst_synth_serial_%j.out
#SBATCH --error=logs/mst_synth_serial_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=YOUR_PARTITION

# Single job: all 40 (epsilon, seed) runs back-to-back on one node.
# Wall time ~= 40 × (one MST fit+generate). Easiest to operate; longest clock time.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs data/synth

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NJOBS="${SLURM_CPUS_PER_TASK:-16}"

exec python generate_synth_mst.py \
  --output-dir data/synth \
  --format parquet \
  --n-jobs "${NJOBS}"
