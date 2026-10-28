#!/bin/bash
# Run one QE SCF task from a handoff-style manifest on CPUs with the cluster QE 7.5 build.
#
#   sbatch --array=0 slurm/run_qe_scf_cpu_slurm_uva.sh SCF_MANIFEST.json
#   NPOOL=20 sbatch -p interactive --time=12:00:00 --array=0 ... (quick checks)
#
# The cell has 112 irreducible k-points and no symmetry, so k-point pools give
# near-linear scaling; NPOOL must divide the MPI rank count.  Each rank gets
# 16 GB: 4 GB/rank ran out of memory, so plan at most 42 ranks per 768 GB Afton
# node and 20 per 384 GB Rivanna node.

#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=qe_scf_cpu
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=FAIL

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "[ERROR] Submit this launcher with sbatch." >&2; exit 2; }
[[ $# -eq 1 ]] || { echo "Usage: sbatch $0 SCF_MANIFEST.json" >&2; exit 2; }
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[ERROR] This launcher requires a Slurm array index." >&2; exit 2; }

PROJECT_ROOT=${PROJECT_ROOT:-/home/gqs7zr/ai4mat/relaxgen}
PYTHON_BIN=${PYTHON_BIN:-/home/gqs7zr/.conda/envs/snb/bin/python}
NPOOL=${NPOOL:-10}
MANIFEST=$(realpath "$1")

[[ -r "${MANIFEST}" ]] || { echo "[ERROR] Manifest is not readable: ${MANIFEST}" >&2; exit 2; }
(( SLURM_NTASKS % NPOOL == 0 )) || { echo "[ERROR] NPOOL=${NPOOL} does not divide ${SLURM_NTASKS} ranks." >&2; exit 2; }

# The QE module only becomes visible after its compiler and MPI are loaded,
# one at a time.
module purge
module load intel-compilers/2025.3.2
module load impi/2021.17.2
module load quantumespresso/7.5
QE_EXECUTABLE=$(command -v pw.x)
# Intel MPI's Slurm bootstrap launches its per-node proxies with srun, which
# aborts when the batch environment carries more than one of
# SLURM_MEM_PER_{CPU,GPU,NODE}.  Single-node jobs fork their ranks directly;
# multi-node jobs (e.g. 2 x 84 ranks on the parallel partition) keep the Slurm
# bootstrap with only the per-CPU memory specification exported.
if (( ${SLURM_JOB_NUM_NODES:-1} > 1 )); then
  export I_MPI_HYDRA_BOOTSTRAP=slurm
  if [[ -n "${SLURM_MEM_PER_CPU:-}" ]]; then
    unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
  fi
else
  export I_MPI_HYDRA_BOOTSTRAP=fork
fi
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

command=(
  "${PYTHON_BIN}" -u -m relaxgen.prerelax.umlip_dft_handoff_threshold run-task
  --manifest "${MANIFEST}"
  --task-index "${SLURM_ARRAY_TASK_ID}"
  --backend cpu
  --allocated-gpu-name "cpu:${SLURM_JOB_NODELIST}:${SLURM_NTASKS}ranks:${NPOOL}pools"
  --qe-executable "${QE_EXECUTABLE}"
  --npool "${NPOOL}"
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

echo "[INFO] Starting CPU SCF task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: ${SLURM_NTASKS} ranks over ${SLURM_JOB_NUM_NODES:-1} node(s), ${NPOOL} pools, bootstrap ${I_MPI_HYDRA_BOOTSTRAP}."
printf '[INFO] Command:'; printf ' %q' "${command[@]}"; printf '\n'
exec "${command[@]}"
