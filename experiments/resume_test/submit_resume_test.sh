#!/bin/bash
# Resume-discontinuity test for relaxgen/dft/dft_relaxation_campaign.py.
# Run from anywhere: bash experiments/resume_test/submit_resume_test.sh
#
# Four VBi_001 starts, production electronic settings at the Gamma point. The
# manifests predate the recompute_on_resume option, so every resume recomputes
# the stored geometry: that recomputation is the discontinuity measurement.
#
#   continuous_a / continuous_b  one uninterrupted A100 job each (A vs B = run-to-run noise
#                                on identical hardware)
#   gpu_resumes                  12 chained 20-min A100 segments (resume on the same hardware)
#   mixed_resumes                12 chained segments cycling A100 -> B200 -> CPU -> A100 -> CPU -> B200,
#                                twice: every directed hand-off between the three backends
# Array element i of each segment waits for element i of the previous one
# (aftercorr), so every structure advances independently.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
T=${T:-/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/13_dft_relax_resume_test}
GPU=slurm/run_dft_relax_slurm_uva.sh
CPU=slurm/run_dft_relax_cpu_slurm_uva.sh
SEGMENTS=12
ROTATION=(a100 b200 cpu a100 cpu b200)
RECORD=${T}/submitted_jobs.tsv
[[ -s "${RECORD}" ]] || printf 'submitted_at\tarm\tsegment\tbackend\tjob_id\n' >"${RECORD}"

submit() {  # arm segment backend dependency
  local arm=$1 segment=$2 backend=$3 dependency=$4
  local args=(--parsable --array=0-3 --job-name="resume_${arm}" \
    --output="${T}/${arm}/slurm/%x-%A_%a.out" --error="${T}/${arm}/slurm/%x-%A_%a.err")
  [[ -n "${dependency}" ]] && args+=(--dependency="aftercorr:${dependency}")
  case "${backend}" in
    a100|b200) args+=(--constraint="${backend}" --time=00:20:00 "${GPU}") ;;
    a100-long) args+=(--constraint=a100 --time=06:00:00 "${GPU}") ;;
    cpu) args+=(-p standard -N1 -C afton --ntasks=42 --mem-per-cpu=16G --time=00:30:00
                --export=ALL,NPOOL=1 "${CPU}") ;;
  esac
  local job
  job=$(sbatch "${args[@]}" "${T}/${arm}/relax_manifest.json")
  job=${job%%;*}
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${arm}" "${segment}" "${backend}" "${job}" >>"${RECORD}"
  echo "${job}"
}

for arm in continuous_a continuous_b; do
  submit "${arm}" 1 a100-long "" >/dev/null
done

previous=""
for segment in $(seq 1 "${SEGMENTS}"); do
  previous=$(submit gpu_resumes "${segment}" a100 "${previous}")
done

previous=""
for segment in $(seq 1 "${SEGMENTS}"); do
  backend=${ROTATION[$(( (segment - 1) % ${#ROTATION[@]} ))]}
  previous=$(submit mixed_resumes "${segment}" "${backend}" "${previous}")
done
column -t -s $'\t' "${RECORD}"
