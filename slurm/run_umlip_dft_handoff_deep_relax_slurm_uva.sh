#!/bin/bash
# Re-relax one handoff-cohort candidate with EqV3/FIRE2 to a 1e-4 eV/Angstrom ML force target.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="b200|a100|a6000|a40|rtxpro6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=handoff_eqv3_deep
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=FAIL

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "[ERROR] Submit this launcher with sbatch." >&2; exit 2; }
[[ $# -eq 1 ]] || { echo "Usage: sbatch $0 COHORT.json" >&2; exit 2; }
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "[ERROR] This launcher requires a Slurm array index." >&2; exit 2; }

PROJECT_ROOT=${PROJECT_ROOT:-/home/gqs7zr/ai4mat/relaxgen}
PYTHON_BIN=${PYTHON_BIN:-/home/gqs7zr/.conda/envs/fairchemv1/bin/python}
COHORT=$(realpath "$1")
[[ -r "${COHORT}" ]] || { echo "[ERROR] Cohort is not readable: ${COHORT}" >&2; exit 2; }

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gqs7zr-xdg-cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gqs7zr-matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

exec "${PYTHON_BIN}" -u -m relaxgen.prerelax.umlip_dft_handoff_threshold deep-relax \
  --cohort "${COHORT}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda \
  --fmax "${DEEP_FMAX:-1e-4}" \
  --max-steps "${DEEP_MAX_STEPS:-5000}"
