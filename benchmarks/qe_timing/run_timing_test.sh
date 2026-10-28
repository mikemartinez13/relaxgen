#!/bin/bash
# Run one task of the QE timing test on any Slurm cluster.
#
#   cd <relaxgen repo>
#   sbatch --array=0-1 <this cluster's partition/GPU/CPU flags> \
#     benchmarks/qe_timing/run_timing_test.sh RUN_DIR/relax_manifest.json
#
# GPU: one GPU and one task, e.g. --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=60G.
# CPU: one rank per core, e.g. -N2 --ntasks=84 --ntasks-per-node=42, with NPOOL set.
# Each array element is one test structure (see README.md).

#SBATCH --job-name=qe_timing
#SBATCH --time=12:00:00
#SBATCH --output=qe_timing-%A_%a.out

set -euo pipefail
[[ $# -eq 1 && -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo "Usage: sbatch --array=0-1 [flags] $0 RUN_DIR/relax_manifest.json" >&2; exit 2; }

# ---- Set these for the cluster (or export them before sbatch) ----------------
BACKEND=${BACKEND:-a100}            # a100, b200, h100 or cpu
QE_EXECUTABLE=${QE_EXECUTABLE:?Set QE_EXECUTABLE to pw.x. A GPU build must sit in a folder named qe-7.5-a100, qe-7.5-b200 or qe-7.5-h100.}
PYTHON_BIN=${PYTHON_BIN:-python}    # python of the snb environment (conda_envs/snb.yml)
NPOOL=${NPOOL:-1}                   # CPU only: k-point pools; must divide the number of ranks
# Load the compiler, MPI and CUDA libraries the QE build needs, for example:
#   module load <compiler> <mpi> <cuda>
# A CPU build of your own works too: point QE_EXECUTABLE at its pw.x and load its MPI.
# -----------------------------------------------------------------------------

PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}
MANIFEST=$(realpath "$1")
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
# QE's scratch files. Multi-node CPU jobs need a folder every node can see.
export QE_TMP_BASE_DIR=${QE_TMP_BASE_DIR:-${TMPDIR:-/tmp}}

if [[ "${BACKEND}" == cpu ]]; then
  export OMP_NUM_THREADS=1
  DEVICE="cpu:${SLURM_JOB_NODELIST}:${SLURM_NTASKS}ranks:${NPOOL}pools"
else
  export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
  DEVICE="${SLURM_NTASKS}x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi
JOB_END=$(squeue -h -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" -o %e)

echo "[INFO] Timing task ${SLURM_ARRAY_TASK_ID} on ${DEVICE}; job ends ${JOB_END}."
exec "${PYTHON_BIN}" -u -m relaxgen.dft.dft_relaxation_campaign run-task \
  --manifest "${MANIFEST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --backend "${BACKEND}" \
  --allocated-device "${DEVICE}" \
  --qe-executable "${QE_EXECUTABLE}" \
  --npool "${NPOOL}" \
  --deadline "${JOB_END}"
