#!/bin/bash
# Submit the prepared uMLIP-to-QE campaign as one A100-or-B200 Slurm array.

set -euo pipefail
[[ $# -ge 1 && $# -le 3 ]] || {
  echo "Usage: $0 CAMPAIGN_MANIFEST.json [MAX_CONCURRENT] [ARRAY_SPEC]" >&2
  exit 2
}

PROJECT_ROOT=${PROJECT_ROOT:-/home/gqs7zr/ai4mat/relaxgen}
PYTHON_BIN=${PYTHON_BIN:-/home/gqs7zr/.conda/envs/snb/bin/python}
LAUNCHER=${PROJECT_ROOT}/slurm/run_qe_umlip_campaign_slurm_uva.sh
MANIFEST=$(realpath "$1")
MAX_CONCURRENT=${2:-8}

[[ -r "${MANIFEST}" ]] || { echo "[ERROR] Manifest is not readable: ${MANIFEST}" >&2; exit 2; }
[[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] MAX_CONCURRENT must be a positive integer." >&2; exit 2; }

TASK_COUNT=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_count"])' "${MANIFEST}")
(( TASK_COUNT > 0 )) || { echo "[ERROR] Manifest contains no tasks." >&2; exit 2; }
LAST_INDEX=$((TASK_COUNT - 1))
ARRAY_SPEC=${3:-0-${LAST_INDEX}}
CAMPAIGN_DIR=$(dirname "${MANIFEST}")
SUBMISSION_DIR=${CAMPAIGN_DIR}/submission
SLURM_LOG_DIR=${CAMPAIGN_DIR}/slurm
SUBMISSION_FILE=${SUBMISSION_DIR}/submitted_jobs.tsv
mkdir -p "${SUBMISSION_DIR}" "${SLURM_LOG_DIR}"

[[ "${ARRAY_SPEC}" =~ ^[0-9,-]+$ ]] || {
  echo "[ERROR] ARRAY_SPEC must contain only indices, commas, and ranges." >&2
  exit 2
}

JOB_ID=$(sbatch --parsable \
  --array="${ARRAY_SPEC}%${MAX_CONCURRENT}" \
  --output="${SLURM_LOG_DIR}/%x-%A_%a.out" \
  --error="${SLURM_LOG_DIR}/%x-%A_%a.err" \
  "${LAUNCHER}" "${MANIFEST}")
JOB_ID=${JOB_ID%%;*}

if [[ ! -s "${SUBMISSION_FILE}" ]]; then
  printf 'submitted_at\tjob_id\tmanifest_task_count\tarray_spec\tmax_concurrent\tallowed_gpus\n' >"${SUBMISSION_FILE}"
fi
printf '%s\t%s\t%s\t%s\t%s\ta100|b200\n' \
  "$(date --iso-8601=seconds)" "${JOB_ID}" "${TASK_COUNT}" "${ARRAY_SPEC}" "${MAX_CONCURRENT}" \
  >>"${SUBMISSION_FILE}"

echo "QE campaign submitted."
echo "job_id=${JOB_ID}"
echo "array=${ARRAY_SPEC}%${MAX_CONCURRENT}"
echo "allowed_gpus=a100|b200"
echo "submission_record=${SUBMISSION_FILE}"
