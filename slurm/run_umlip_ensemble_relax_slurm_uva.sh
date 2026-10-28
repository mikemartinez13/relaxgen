#!/bin/bash
# Stage A of the uMLIP ensemble experiment: array element i relaxes shard
# (i % NUM_SHARDS) of the candidates with model MODELS[i / NUM_SHARDS].  Runs on
# MIG slices, which have their own per-user GPU limit.  EqV3 and MACE need the
# FairChem-v1 tree on PYTHONPATH; UMA needs FairChem-v2 without it.
#
#   sbatch --array=0-79%16 slurm/run_umlip_ensemble_relax_slurm_uva.sh OUTPUT_DIR
#SBATCH --partition=gpu-mig
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=umlip_ensemble
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=FAIL

set -euo pipefail
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Submit this script as a Slurm array." >&2; exit 1; }
OUTPUT_DIR=${1:?"Usage: sbatch --array=... $0 OUTPUT_DIR"}
NUM_SHARDS=${NUM_SHARDS:-20}
MODELS=(eqv3 mace_ft uma mace_omat)
MODEL=${MODELS[$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))]}
SHARD=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))
PROJECT_ROOT=${PROJECT_ROOT:-/home/gqs7zr/ai4mat/relaxgen}

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gqs7zr-xdg-cache"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gqs7zr-matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"
args=(-m relaxgen.prerelax.umlip_ensemble_diversity relax --output-dir "${OUTPUT_DIR}" --model "${MODEL}"
      --shard "${SHARD}" --num-shards "${NUM_SHARDS}" ${MAX_CANDIDATES:+--max-candidates "${MAX_CANDIDATES}"})
echo "[INFO] ${MODEL} shard ${SHARD}/${NUM_SHARDS} on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [[ "${MODEL}" == "uma" ]]; then
  env -u PYTHONPATH /home/gqs7zr/.conda/envs/fairchemv2/bin/python -u "${args[@]}"
else
  PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" EQUIFORMER_V3_ROOT="${EQUIFORMER_V3_ROOT:-/home/gqs7zr/ai4mat/ConstrDiff_Material/models/equiformer_v3}" \
    /home/gqs7zr/.conda/envs/fairchemv1/bin/python -u "${args[@]}"
fi
