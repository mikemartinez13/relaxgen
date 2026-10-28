#!/bin/bash
# Run one QE SCF single point of the uMLIP-to-DFT handoff threshold experiment.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|b200"
#SBATCH --cpus-per-task=4
#SBATCH --mem=44G
#SBATCH --time=03:00:00
#SBATCH --job-name=handoff_qe_scf
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=FAIL

# A from-scratch production SCF takes ~47 min on an A100; 3 h leaves margin
# while staying short enough to backfill on a congested queue.

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "[ERROR] Submit this launcher with sbatch." >&2; exit 2; }
[[ $# -eq 1 ]] || { echo "Usage: sbatch $0 SCF_MANIFEST.json" >&2; exit 2; }
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[ERROR] This launcher requires a Slurm array index." >&2; exit 2; }

PROJECT_ROOT=${PROJECT_ROOT:-/home/gqs7zr/ai4mat/relaxgen}
PYTHON_BIN=${PYTHON_BIN:-/home/gqs7zr/.conda/envs/snb/bin/python}
QE_ROOT=${QE_ROOT:-/scratch/gqs7zr/ai4mat/qe}
MANIFEST=$(realpath "$1")

[[ -r "${MANIFEST}" ]] || { echo "[ERROR] Manifest is not readable: ${MANIFEST}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "[ERROR] Python is not executable: ${PYTHON_BIN}" >&2; exit 2; }

command -v nvidia-smi >/dev/null 2>&1 || { echo "[ERROR] nvidia-smi is unavailable." >&2; exit 2; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#GPU_NAMES[@]} == 1 )) || { echo "[ERROR] Expected exactly one allocated GPU." >&2; exit 2; }
GPU_NAME=${GPU_NAMES[0]}
case "${GPU_NAME,,}" in
  *a100*) GPU_TYPE=a100 ;;
  *b200*) GPU_TYPE=b200 ;;
  *) echo "[ERROR] Only A100 and B200 GPUs are allowed; allocated ${GPU_NAME}." >&2; exit 2 ;;
esac
QE_EXECUTABLE=${QE_ROOT}/qe-7.5-${GPU_TYPE}/bin/pw.x
[[ -x "${QE_EXECUTABLE}" ]] || { echo "[ERROR] QE executable is unavailable: ${QE_EXECUTABLE}" >&2; exit 2; }

module purge
# NVHPC 25.11 modules were retired with the 202606 software stack.
source "${PROJECT_ROOT}/slurm/qe_gpu_runtime_uva.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

command=(
  "${PYTHON_BIN}" -u -m relaxgen.prerelax.umlip_dft_handoff_threshold run-task
  --manifest "${MANIFEST}"
  --task-index "${SLURM_ARRAY_TASK_ID}"
  --gpu-type "${GPU_TYPE}"
  --allocated-gpu-name "${GPU_NAME}"
  --qe-executable "${QE_EXECUTABLE}"
)
[[ "${OVERWRITE:-0}" == 1 ]] && command+=(--overwrite)
# Under qe_dispatch.py the worker claims the unit first, so a GPU/CPU twin
# that starts second exits without running.
if [[ -n "${DISPATCH_DIR:-}" ]]; then
  command=(
    "${PYTHON_BIN}" -u -m relaxgen.dispatch.qe_dispatch worker
    --dispatch-dir "${DISPATCH_DIR}"
    --task-index "${SLURM_ARRAY_TASK_ID}"
    --backend "${DISPATCH_BACKEND:?DISPATCH_BACKEND must accompany DISPATCH_DIR}"
    -- "${command[@]}"
  )
fi

echo "[INFO] Starting handoff SCF task ${SLURM_ARRAY_TASK_ID} on ${GPU_NAME}."
printf '[INFO] Command:'; printf ' %q' "${command[@]}"; printf '\n'
exec "${command[@]}"
