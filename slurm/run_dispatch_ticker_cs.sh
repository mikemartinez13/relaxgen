#!/bin/bash
# Run qe_dispatch ticks every few minutes on the UVA CS cluster.
#
#   sbatch --job-name=tick_<campaign> slurm/run_dispatch_ticker_cs.sh DISPATCH_DIR [MINUTES]
#
# scrontab is disabled on the CS cluster, so this one-core job on the 20-day
# nolim partition does what the Rivanna scrontab entry does.  At start it
# queues its successor (same name, --dependency=singleton), so ticks continue
# after the time limit.  It exits, without a successor, once every unit is
# done, failed or withdrawn, or when DISPATCH_DIR/STOP_TICKER exists.  Use a job
# name other than the campaign's, so the controller does not count this job.

#SBATCH --partition=nolim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=qd_ticker
#SBATCH --mail-user=gqs7zr@virginia.edu
#SBATCH --mail-type=FAIL

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "[ERROR] Submit this launcher with sbatch." >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || { echo "Usage: sbatch --job-name=NAME $0 DISPATCH_DIR [MINUTES]" >&2; exit 2; }

PROJECT_ROOT=${PROJECT_ROOT:-/u/gqs7zr/relaxgen}
PYTHON_BIN=${PYTHON_BIN:-/u/gqs7zr/.conda/envs/snb/bin/python}
DISPATCH_DIR=$(realpath "$1")
MINUTES=${2:-10}
[[ -r "${DISPATCH_DIR}/config.json" ]] || { echo "[ERROR] No config.json in ${DISPATCH_DIR}" >&2; exit 2; }
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

campaign_finished() {
  "${PYTHON_BIN}" - "${DISPATCH_DIR}/log.jsonl" <<'EOF'
import json, sys
from pathlib import Path
lines = Path(sys.argv[1]).read_text().splitlines() if Path(sys.argv[1]).exists() else []
last = json.loads(lines[-1]) if lines else None
finished = last and not last["dry_run"] and last["done"] + last["failed"] + last["withdrawn"] == last["units"]
sys.exit(0 if finished else 1)
EOF
}

# Keep exactly one successor queued behind this job.
if [[ -z "$(squeue -h -u "${USER}" -n "${SLURM_JOB_NAME}" -t PENDING -o %i)" ]]; then
  sbatch --parsable --dependency=singleton --job-name="${SLURM_JOB_NAME}" \
    --output="${DISPATCH_DIR}/slurm/%x-%j.out" "${PROJECT_ROOT}/slurm/run_dispatch_ticker_cs.sh" \
    "${DISPATCH_DIR}" "${MINUTES}"
fi

while true; do
  if [[ -e "${DISPATCH_DIR}/STOP_TICKER" ]] || campaign_finished; then
    echo "[INFO] $(date -Is) Stopping: campaign finished or STOP_TICKER present."
    scancel -u "${USER}" -n "${SLURM_JOB_NAME}" -t PENDING || true
    exit 0
  fi
  "${PYTHON_BIN}" -m relaxgen.dispatch.qe_dispatch tick --dispatch-dir "${DISPATCH_DIR}" > /dev/null \
    || echo "[WARN] $(date -Is) tick failed"
  "${PYTHON_BIN}" -m relaxgen.dispatch.qe_dispatch status --dispatch-dir "${DISPATCH_DIR}" | head -1 || true
  sleep $(( MINUTES * 60 ))
done
