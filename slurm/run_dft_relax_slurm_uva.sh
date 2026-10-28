#!/bin/bash
# Run one segment of a resumable BFGS/QE-socket relaxation on 1-2 A100/B200 GPUs.
#
#   sbatch --array=0 slurm/run_dft_relax_slurm_uva.sh RELAX_MANIFEST.json
#   sbatch --array=0 --gres=gpu:2 --ntasks=2 --mem=100G slurm/run_dft_relax_slurm_uva.sh ...
#
# One MPI rank and one GPU per k-point pool.  The segment stops before the job's
# time limit and the next job resumes from the saved trajectory and BFGS state.
# 12 h segments match the common job length on the GPU nodes, so they backfill.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|b200"
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --job-name=dft_relax
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=FAIL

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "[ERROR] Submit this launcher with sbatch." >&2; exit 2; }
[[ $# -eq 1 ]] || { echo "Usage: sbatch $0 RELAX_MANIFEST.json" >&2; exit 2; }
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[ERROR] This launcher requires a Slurm array index." >&2; exit 2; }

PROJECT_ROOT=${PROJECT_ROOT:-/home/gqs7zr/ai4mat/relaxgen}
PYTHON_BIN=${PYTHON_BIN:-/home/gqs7zr/.conda/envs/snb/bin/python}
QE_ROOT=${QE_ROOT:-/scratch/gqs7zr/ai4mat/qe}
MANIFEST=$(realpath "$1")
[[ -r "${MANIFEST}" ]] || { echo "[ERROR] Manifest is not readable: ${MANIFEST}" >&2; exit 2; }

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#GPU_NAMES[@]} == SLURM_NTASKS )) || {
  echo "[ERROR] Expected one GPU per task: ${#GPU_NAMES[@]} GPUs, ${SLURM_NTASKS} tasks." >&2; exit 2; }
case "${GPU_NAMES[0],,}" in
  *a100*) GPU_TYPE=a100 ;;
  *b200*) GPU_TYPE=b200 ;;
  *) echo "[ERROR] Only A100 and B200 GPUs are allowed; allocated ${GPU_NAMES[0]}." >&2; exit 2 ;;
esac
QE_EXECUTABLE=${QE_ROOT}/qe-7.5-${GPU_TYPE}/bin/pw.x
[[ -x "${QE_EXECUTABLE}" ]] || { echo "[ERROR] QE executable is unavailable: ${QE_EXECUTABLE}" >&2; exit 2; }

module purge
# NVHPC 25.11 modules were retired with the 202606 software stack.
source "${PROJECT_ROOT}/slurm/qe_gpu_runtime_uva.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export QE_TMP_BASE_DIR=${QE_TMP_BASE_DIR:-${TMPDIR:-/tmp}}
cd "${PROJECT_ROOT}"

# Query this element: for the array's last element SLURM_JOB_ID equals the array ID,
# which squeue expands to every element (one end time per line).
JOB_END=$(squeue -h -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" -o %e)
command=(
  "${PYTHON_BIN}" -u -m relaxgen.dft.dft_relaxation_campaign run-task
  --manifest "${MANIFEST}"
  --task-index "${SLURM_ARRAY_TASK_ID}"
  --backend "${GPU_TYPE}"
  --allocated-device "${SLURM_NTASKS}x ${GPU_NAMES[0]}"
  --qe-executable "${QE_EXECUTABLE}"
  --deadline "${JOB_END}"
)
# Under qe_dispatch.py the worker claims the unit first, so a twin that starts
# second exits without running.
if [[ -n "${DISPATCH_DIR:-}" ]]; then
  command=(
    "${PYTHON_BIN}" -u -m relaxgen.dispatch.qe_dispatch worker
    --dispatch-dir "${DISPATCH_DIR}"
    --task-index "${SLURM_ARRAY_TASK_ID}"
    --backend "${DISPATCH_BACKEND:?DISPATCH_BACKEND must accompany DISPATCH_DIR}"
    -- "${command[@]}"
  )
fi

echo "[INFO] Relaxation task ${SLURM_ARRAY_TASK_ID} on ${SLURM_NTASKS}x ${GPU_NAMES[0]}; job ends ${JOB_END}."
printf '[INFO] Command:'; printf ' %q' "${command[@]}"; printf '\n'
exec "${command[@]}"
