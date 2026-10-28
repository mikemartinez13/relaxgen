#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="b200|a100|a6000|a40|rtxpro6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-187%8
#SBATCH --job-name=bi2te3_eqv3_tight
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/gqs7zr/ai4mat/runs/%x-%A_%a.out
#SBATCH --error=/scratch/gqs7zr/ai4mat/runs/%x-%A_%a.err

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit this script with sbatch." >&2; exit 1; }

PROJECT_ROOT=${PROJECT_ROOT:-/home/gqs7zr/ai4mat/relaxgen}
PYTHON_BIN=${PYTHON_BIN:-/home/gqs7zr/.conda/envs/fairchemv1/bin/python}
INPUT_DIR=${1:?"Usage: sbatch $0 INPUT_DIR OUTPUT_DIR [CHECKPOINT] [CONFIG_YML]"}
OUTPUT_DIR=${2:?"Usage: sbatch $0 INPUT_DIR OUTPUT_DIR [CHECKPOINT] [CONFIG_YML]"}
CKPT_DIR=${RELAXGEN_CKPT_DIR:-${PROJECT_ROOT}/ft_ckpts}
CHECKPOINT=${3:-${CKPT_DIR}/eqv3/best_by_forces_mae.pt}
CONFIG_YML=${4:-${CKPT_DIR}/eqv3/config.yml}

[[ -d "${INPUT_DIR}" ]] || { echo "Candidate directory not found: ${INPUT_DIR}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "EqV3 checkpoint not found: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${CONFIG_YML}" ]] || { echo "EqV3 config not found: ${CONFIG_YML}" >&2; exit 2; }

mkdir -p "${OUTPUT_DIR}" /scratch/gqs7zr/ai4mat/runs
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gqs7zr-xdg-cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gqs7zr-matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

selector=(--task-index "${SLURM_ARRAY_TASK_ID}")
if [[ -n "${CONFIG_ID:-}" ]]; then
  selector=(--config-id "${CONFIG_ID}")
fi

exec "${PYTHON_BIN}" -u -m relaxgen.prerelax.relax_reconstruction_candidates_eqv3 relax \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "${selector[@]}" \
  --checkpoint "${CHECKPOINT}" \
  --config-yml "${CONFIG_YML}" \
  --device cuda \
  --fmax 0.001 \
  --max-steps 600 \
  --maxstep 0.10 \
  --warning-min-distance 1.8 \
  --hard-min-distance 1.2
