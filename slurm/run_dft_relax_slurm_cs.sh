#!/bin/bash
# Run one segment of a resumable BFGS/QE-socket relaxation on one GPU of the UVA CS cluster.
#
#   sbatch --array=0 slurm/run_dft_relax_slurm_cs.sh RELAX_MANIFEST.json
#
# The GPUs a relaxation may use are the Slurm features in CS_GPU_FEATURES
# (relaxgen/dispatch/clusters.py); qe_dispatch passes them as --constraint, and
# the default below matches them.  The QE build follows the GPU this job gets:
# H100 -> qe-7.5-h100, A100 -> qe-7.5-a100.  The segment stops before the job's
# time limit and the next job resumes from the saved trajectory and BFGS state;
# 24 h fits a whole Plan A relaxation on either GPU.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100_94gb|a100_80gb"
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --job-name=dft_relax
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=FAIL

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "[ERROR] Submit this launcher with sbatch." >&2; exit 2; }
[[ $# -eq 1 ]] || { echo "Usage: sbatch $0 RELAX_MANIFEST.json" >&2; exit 2; }
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[ERROR] This launcher requires a Slurm array index." >&2; exit 2; }

PROJECT_ROOT=${PROJECT_ROOT:-/u/gqs7zr/relaxgen}
PYTHON_BIN=${PYTHON_BIN:-/u/gqs7zr/.conda/envs/snb/bin/python}
QE_ROOT=${QE_ROOT:-/bigtemp/gqs7zr/qe_builds}
MANIFEST=$(realpath "$1")
[[ -r "${MANIFEST}" ]] || { echo "[ERROR] Manifest is not readable: ${MANIFEST}" >&2; exit 2; }

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
(( ${#GPU_NAMES[@]} == SLURM_NTASKS )) || {
  echo "[ERROR] Expected one GPU per task: ${#GPU_NAMES[@]} GPUs, ${SLURM_NTASKS} tasks." >&2; exit 2; }
case "${GPU_NAMES[0],,}" in
  *h100*) GPU_TYPE=h100 ;;
  *a100*) GPU_TYPE=a100 ;;
  *) echo "[ERROR] Only H100 and A100 GPUs have QE builds; allocated ${GPU_NAMES[0]}." >&2; exit 2 ;;
esac
QE_EXECUTABLE=${QE_ROOT}/qe-7.5-${GPU_TYPE}/bin/pw.x
[[ -x "${QE_EXECUTABLE}" ]] || { echo "[ERROR] QE executable is unavailable: ${QE_EXECUTABLE}" >&2; exit 2; }

# The builds use NVHPC 26.1 and the HPC-X MPI it ships.  HPC-X has no Slurm
# PMI-2 support, so srun cannot start it here: the runner uses its mpirun.
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module purge
module load nvhpc/26.1-CUDA-13.1.0
HPCX_INIT=${EBROOTNVHPC}/Linux_x86_64/26.1/comm_libs/13.1/hpcx/hpcx-2.25.1/hpcx-init.sh
[[ -r "${HPCX_INIT}" ]] || { echo "[ERROR] HPC-X is missing: ${HPCX_INIT}" >&2; exit 2; }
set +u
source "${HPCX_INIT}"
hpcx_load
set -u
export QE_GPU_LAUNCHER=mpirun
export UCX_WARN_UNUSED_ENV_VARS=n
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
