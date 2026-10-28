#!/bin/bash
# Submit one stage of the uMLIP-to-DFT handoff threshold experiment as a Slurm array.
#
#   deep-relax COHORT.json        one EqV3 deep relaxation per cohort entry
#   scf        SCF_MANIFEST.json  one QE SCF single point per selected frame (A100|B200)

set -euo pipefail
[[ $# -ge 2 && $# -le 4 ]] || {
  echo "Usage: $0 {deep-relax|scf} COHORT_OR_MANIFEST.json [MAX_CONCURRENT] [ARRAY_SPEC]" >&2
  exit 2
}

PROJECT_ROOT=${PROJECT_ROOT:-/home/gqs7zr/ai4mat/relaxgen}
PYTHON_BIN=${PYTHON_BIN:-/home/gqs7zr/.conda/envs/snb/bin/python}
STAGE=$1
INPUT=$(realpath "$2")
[[ -r "${INPUT}" ]] || { echo "[ERROR] Input is not readable: ${INPUT}" >&2; exit 2; }

case "${STAGE}" in
  deep-relax)
    LAUNCHER=${PROJECT_ROOT}/slurm/run_umlip_dft_handoff_deep_relax_slurm_uva.sh
    COUNT_KEY=cohort_size
    DEFAULT_CONCURRENT=20
    ;;
  scf)
    LAUNCHER=${PROJECT_ROOT}/slurm/run_umlip_dft_handoff_scf_slurm_uva.sh
    COUNT_KEY=task_count
    DEFAULT_CONCURRENT=32
    ;;
  *) echo "[ERROR] Unknown stage ${STAGE}; expected deep-relax or scf." >&2; exit 2 ;;
esac
MAX_CONCURRENT=${3:-${DEFAULT_CONCURRENT}}
[[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] MAX_CONCURRENT must be a positive integer." >&2; exit 2; }

COUNT=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "${INPUT}" "${COUNT_KEY}")
(( COUNT > 0 )) || { echo "[ERROR] ${INPUT} contains no tasks." >&2; exit 2; }
ARRAY_SPEC=${4:-0-$((COUNT - 1))}
[[ "${ARRAY_SPEC}" =~ ^[0-9,-]+$ ]] || {
  echo "[ERROR] ARRAY_SPEC must contain only indices, commas, and ranges." >&2
  exit 2
}

EXPERIMENT_DIR=$(dirname "${INPUT}")
SUBMISSION_DIR=${EXPERIMENT_DIR}/submission
SLURM_LOG_DIR=${EXPERIMENT_DIR}/slurm/${STAGE}
SUBMISSION_FILE=${SUBMISSION_DIR}/submitted_jobs.tsv
mkdir -p "${SUBMISSION_DIR}" "${SLURM_LOG_DIR}"

JOB_ID=$(sbatch --parsable \
  --array="${ARRAY_SPEC}%${MAX_CONCURRENT}" \
  --output="${SLURM_LOG_DIR}/%x-%A_%a.out" \
  --error="${SLURM_LOG_DIR}/%x-%A_%a.err" \
  "${LAUNCHER}" "${INPUT}")
JOB_ID=${JOB_ID%%;*}

if [[ ! -s "${SUBMISSION_FILE}" ]]; then
  printf 'submitted_at\tstage\tjob_id\tinput\ttask_count\tarray_spec\tmax_concurrent\n' >"${SUBMISSION_FILE}"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date --iso-8601=seconds)" "${STAGE}" "${JOB_ID}" "${INPUT}" "${COUNT}" "${ARRAY_SPEC}" "${MAX_CONCURRENT}" \
  >>"${SUBMISSION_FILE}"

echo "Handoff ${STAGE} submitted."
echo "job_id=${JOB_ID}"
echo "array=${ARRAY_SPEC}%${MAX_CONCURRENT}"
echo "submission_record=${SUBMISSION_FILE}"
