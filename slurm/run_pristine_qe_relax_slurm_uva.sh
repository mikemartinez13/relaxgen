#!/bin/bash
# Prepare and run exactly one fixed-cell pristine QE relaxation.
# The allocated A40/A100/B200 is detected at runtime and matched to the
# corresponding QE 7.5 GPU build under QE_ROOT.
#
# Usage:
#   sbatch slurm/run_pristine_qe_relax_slurm_uva.sh PRISTINE.extxyz WORK_DIR
#   sbatch --constraint=a100 slurm/run_pristine_qe_relax_slurm_uva.sh PRISTINE.extxyz WORK_DIR

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100|b200"
#SBATCH --cpus-per-task=4
#SBATCH --mem=44G
#SBATCH --time=24:00:00
#SBATCH --job-name=bi2te3_pristine_qe
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/gqs7zr/ai4mat/runs/%x-%j.out
#SBATCH --error=/scratch/gqs7zr/ai4mat/runs/%x-%j.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "[ERROR] Submit this launcher with sbatch." >&2
  exit 2
}
[[ $# -eq 2 ]] || {
  echo "Usage: sbatch $0 PRISTINE.extxyz WORK_DIR" >&2
  exit 2
}

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-${PWD}}
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -r "${SUBMIT_DIR}/relaxgen/structures/bi2te3_defect_generation.py" ]]; then
    PROJECT_ROOT=${SUBMIT_DIR}
  elif [[ -r "/home/gqs7zr/ai4mat/relaxgen/relaxgen/structures/bi2te3_defect_generation.py" ]]; then
    PROJECT_ROOT=/home/gqs7zr/ai4mat/relaxgen
  else
    echo "[ERROR] Cannot locate the relaxgen checkout." >&2
    echo "[ERROR] Submit from the repository root or export PROJECT_ROOT explicitly." >&2
    exit 2
  fi
fi
PROJECT_ROOT=$(realpath "${PROJECT_ROOT}")
SNB_PYTHON=${SNB_PYTHON:-/home/gqs7zr/.conda/envs/snb/bin/python}
PREPARE_SCRIPT=${PREPARE_SCRIPT:-${PROJECT_ROOT}/relaxgen/structures/bi2te3_defect_generation.py}
PSEUDO_DIR=${PSEUDO_DIR:-${PROJECT_ROOT}/pseudos}
QE_ROOT=${QE_ROOT:-/scratch/gqs7zr/ai4mat/qe}
QE_MPI_PROCS=${QE_MPI_PROCS:-${SLURM_NTASKS:-1}}
QE_NPOOL=${QE_NPOOL:-${QE_MPI_PROCS}}
QE_NDIAG=${QE_NDIAG:-1}
QE_GPUS_PER_TASK=${QE_GPUS_PER_TASK:-1}

PRISTINE_PATH=$1
WORK_DIR=$2
[[ "${PRISTINE_PATH}" == /* ]] || PRISTINE_PATH="${SUBMIT_DIR}/${PRISTINE_PATH}"
[[ "${WORK_DIR}" == /* ]] || WORK_DIR="${SUBMIT_DIR}/${WORK_DIR}"
[[ -r "${PRISTINE_PATH}" ]] || {
  echo "[ERROR] Pristine extxyz is not readable: ${PRISTINE_PATH}" >&2
  exit 2
}
[[ -x "${SNB_PYTHON}" ]] || {
  echo "[ERROR] snb Python is not executable: ${SNB_PYTHON}" >&2
  exit 2
}
[[ -r "${PREPARE_SCRIPT}" ]] || {
  echo "[ERROR] QE preparation script is not readable: ${PREPARE_SCRIPT}" >&2
  exit 2
}
[[ -d "${PSEUDO_DIR}" ]] || {
  echo "[ERROR] QE pseudopotential directory is absent: ${PSEUDO_DIR}" >&2
  exit 2
}
PRISTINE_PATH=$(realpath "${PRISTINE_PATH}")
mkdir -p "${WORK_DIR}"
WORK_DIR=$(realpath "${WORK_DIR}")
QE_INPUT_PATH=${WORK_DIR}/pristine_relax.pwi
QE_OUTPUT_PATH=${WORK_DIR}/pristine_relax.pwo
QE_STATE_DIR=${WORK_DIR}/state/${SLURM_JOB_ID}
QE_PREPARE_MANIFEST=${WORK_DIR}/qe_prepare_manifest.json

for output_path in "${QE_INPUT_PATH}" "${QE_OUTPUT_PATH}" "${QE_PREPARE_MANIFEST}"; do
  [[ ! -e "${output_path}" || "${OVERWRITE:-0}" == 1 ]] || {
    echo "[ERROR] Refusing to overwrite ${output_path}; submit with OVERWRITE=1 if intended." >&2
    exit 2
  }
done

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "[ERROR] nvidia-smi is required to select the matching QE GPU build." >&2
  exit 2
}
if ! ALLOCATED_GPU_OUTPUT=$(
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
); then
  echo "[ERROR] nvidia-smi could not identify the allocated GPU." >&2
  exit 2
fi
[[ -n "${ALLOCATED_GPU_OUTPUT}" ]] || {
  echo "[ERROR] nvidia-smi did not report an allocated GPU." >&2
  exit 2
}
mapfile -t ALLOCATED_GPU_NAMES <<< "${ALLOCATED_GPU_OUTPUT}"

declare -A DETECTED_GPU_TYPES=()
for gpu_name in "${ALLOCATED_GPU_NAMES[@]}"; do
  gpu_name_normalized=${gpu_name,,}
  case "${gpu_name_normalized}" in
    *a40*)  detected_gpu_type=a40 ;;
    *a100*) detected_gpu_type=a100 ;;
    *b200*) detected_gpu_type=b200 ;;
    *)
      echo "[ERROR] No supported QE build for allocated GPU: ${gpu_name}" >&2
      echo "[ERROR] Expected an A40, A100, or B200 allocation." >&2
      exit 2
      ;;
  esac
  DETECTED_GPU_TYPES["${detected_gpu_type}"]=1
done
(( ${#DETECTED_GPU_TYPES[@]} == 1 )) || {
  echo "[ERROR] QE requires a homogeneous GPU allocation; detected: ${!DETECTED_GPU_TYPES[*]}" >&2
  exit 2
}
QE_GPU_TYPE=${!DETECTED_GPU_TYPES[*]}
ALLOCATED_GPU_NAME=${ALLOCATED_GPU_NAMES[0]}
QE_HOME=${QE_HOME:-${QE_ROOT}/qe-7.5-${QE_GPU_TYPE}}

[[ -x "${QE_HOME}/bin/pw.x" ]] || {
  echo "[ERROR] Matching QE ${QE_GPU_TYPE} executable is unavailable: ${QE_HOME}/bin/pw.x" >&2
  exit 2
}
module purge
# NVHPC 25.11 modules were retired with the 202606 software stack.
source "${PROJECT_ROOT}/slurm/qe_gpu_runtime_uva.sh"

prepare_command=(
  "${SNB_PYTHON}"
  "${PREPARE_SCRIPT}"
  prepare-qe
  --inferred "${PRISTINE_PATH}"
  --output-dir "${WORK_DIR}"
  --pseudo-dir "${PSEUDO_DIR}"
  --qe-outdir "${QE_STATE_DIR}"
)
[[ "${OVERWRITE:-0}" == 1 ]] && prepare_command+=(--overwrite)

echo "[INFO] Michael, preparing the precise fixed-cell QE input from ${PRISTINE_PATH}."
printf '[INFO] Preparation command:'
printf ' %q' "${prepare_command[@]}"
printf '\n'
"${prepare_command[@]}"
[[ -r "${QE_INPUT_PATH}" ]] || {
  echo "[ERROR] QE preparation did not create ${QE_INPUT_PATH}" >&2
  exit 2
}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
GPU_REQUEST="${QE_GPU_TYPE}:${QE_GPUS_PER_TASK}"
command=(
  env -u CUDA_VISIBLE_DEVICES
  srun
  "--ntasks=${QE_MPI_PROCS}"
  "--gpus-per-task=${GPU_REQUEST}"
  "--gpu-bind=verbose,per_task:${QE_GPUS_PER_TASK}"
  "${QE_HOME}/bin/pw.x"
  -nk "${QE_NPOOL}"
  -nd "${QE_NDIAG}"
  -in "${QE_INPUT_PATH}"
)

echo "[INFO] Michael, starting the prepared pristine QE relaxation at $(date)."
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Pristine structure: ${PRISTINE_PATH}"
echo "[INFO] Working directory: ${WORK_DIR}"
echo "[INFO] QE input: ${QE_INPUT_PATH}"
echo "[INFO] QE output: ${QE_OUTPUT_PATH}"
echo "[INFO] QE state: ${QE_STATE_DIR}"
echo "[INFO] Allocated GPU: ${ALLOCATED_GPU_NAME}"
echo "[INFO] Selected QE GPU type: ${QE_GPU_TYPE}"
echo "[INFO] QE executable: ${QE_HOME}/bin/pw.x"
printf '[INFO] Command:'
printf ' %q' "${command[@]}"
printf '\n'

"${command[@]}" >"${QE_OUTPUT_PATH}" 2>&1

grep -q "JOB DONE" "${QE_OUTPUT_PATH}" || {
  echo "[ERROR] QE exited without a JOB DONE marker: ${QE_OUTPUT_PATH}" >&2
  exit 1
}
echo "[INFO] Michael, QE finished at $(date). Validate the output with the documented validate-qe command."
printf '[INFO] Validation command: %q %q validate-qe --inferred %q --qe-output %q --output-dir %q\n' \
  "${SNB_PYTHON}" "${PREPARE_SCRIPT}" "${PRISTINE_PATH}" "${QE_OUTPUT_PATH}" \
  "${WORK_DIR}/validated"
