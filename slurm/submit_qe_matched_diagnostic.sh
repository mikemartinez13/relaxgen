#!/bin/bash
# Submit the three matched arrays for 12 uMLIP-pre-relaxed endpoints.

set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 EXPERIMENT_MANIFEST.json" >&2; exit 2; }

PROJECT_ROOT=${PROJECT_ROOT:-/home/gqs7zr/ai4mat/relaxgen}
LAUNCHER=${PROJECT_ROOT}/slurm/run_qe_matched_diagnostic_slurm_uva.sh
MANIFEST=$(realpath "$1")
EXPERIMENT_DIR=$(dirname "${MANIFEST}")
SUBMISSION_DIR=${EXPERIMENT_DIR}/submission
SLURM_LOG_DIR=${EXPERIMENT_DIR}/slurm
mkdir -p "${SUBMISSION_DIR}" "${SLURM_LOG_DIR}"

SUBMITTED_AT=$(date --iso-8601=seconds)
SUBMISSION_FILE=${SUBMISSION_DIR}/submitted_jobs.tsv
ESTIMATE_FILE=${SUBMISSION_DIR}/initial_queue_estimates.tsv
[[ -e "${SUBMISSION_FILE}" ]] || printf 'submitted_at\tjob_id\tjob_name\tprofile\ttask_indices\n' >"${SUBMISSION_FILE}"
[[ -e "${ESTIMATE_FILE}" ]] || printf 'observed_at\tjob_id\tsubmit_time\testimated_start\treason\n' >"${ESTIMATE_FILE}"

submit_group() {
  local profile=$1 indices=$2 job_name=$3
  local job_id queue_snapshot
  if awk -F '\t' -v name="${job_name}" 'NR > 1 && $3 == name { found = 1 } END { exit !found }' "${SUBMISSION_FILE}"; then
    echo "[SKIPPED] ${job_name} is already recorded in ${SUBMISSION_FILE}."
    return
  fi
  job_id=$(sbatch --parsable \
    --constraint="${profile}" \
    --array="${indices}" \
    --job-name="${job_name}" \
    --output="${SLURM_LOG_DIR}/%x-%A_%a.out" \
    --error="${SLURM_LOG_DIR}/%x-%A_%a.err" \
    "${LAUNCHER}" "${MANIFEST}")
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\t%s\n' "${SUBMITTED_AT}" "${job_id}" "${job_name}" "${profile}" "${indices}" >>"${SUBMISSION_FILE}"
  queue_snapshot=$(squeue --start -j "${job_id}" -h -o '%V|%S|%R' 2>/dev/null || true)
  while IFS='|' read -r submit_time estimate reason; do
    [[ -n "${submit_time}" ]] || continue
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${job_id}" "${submit_time}" "${estimate}" "${reason}" >>"${ESTIMATE_FILE}"
  done <<<"${queue_snapshot}"
  echo "[SUBMITTED] ${job_name}: ${job_id} (${profile}, tasks ${indices})"
}

submit_group b200 0-3 qe_diag_b200_ml
submit_group a100 4-7 qe_diag_a100_ml
submit_group a40 8-11 qe_diag_a40_ml

echo "All 12 uMLIP-to-DFT diagnostic tasks were submitted."
echo "Submission record: ${SUBMISSION_FILE}"
echo "Initial queue estimates: ${ESTIMATE_FILE}"
