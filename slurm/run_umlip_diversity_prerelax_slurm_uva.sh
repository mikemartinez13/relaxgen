#!/bin/bash
# BFGS pre-relaxation of the diversity-experiment starts with EqV3 and UMA.
# Each model runs in its own environment: EqV3 needs the FairChem-v1 tree on
# PYTHONPATH, UMA needs FairChem-v2 without it.
#
#   sbatch slurm/run_umlip_diversity_prerelax_slurm_uva.sh OUTPUT_DIR
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="b200|a100|a6000|a40|rtxpro6000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=diversity_prerelax
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit this script with sbatch." >&2; exit 1; }
OUTPUT_DIR=${1:?"Usage: sbatch $0 OUTPUT_DIR"}
PROJECT_ROOT=${PROJECT_ROOT:-/home/gqs7zr/ai4mat/relaxgen}
EQV3_PYTHON=${EQV3_PYTHON:-/home/gqs7zr/.conda/envs/fairchemv1/bin/python}
UMA_PYTHON=${UMA_PYTHON:-/home/gqs7zr/.conda/envs/fairchemv2/bin/python}

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gqs7zr-xdg-cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gqs7zr-matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${EQV3_PYTHON}" -u -m experiments.umlip_dft_diversity prerelax --output-dir "${OUTPUT_DIR}" --model eqv3
env -u PYTHONPATH "${UMA_PYTHON}" -u -m experiments.umlip_dft_diversity prerelax --output-dir "${OUTPUT_DIR}" --model uma
