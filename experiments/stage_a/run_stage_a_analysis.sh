#!/bin/bash
# Stage A analysis (symmetry-aware RMSD) and Stage B selection, as run on 2026-10-23.
# Run from anywhere: bash experiments/stage_a/run_stage_a_analysis.sh
set -e
PY=/home/gqs7zr/.conda/envs/snb/bin/python
E=${E:-/scratch/gqs7zr/relaxgen/umlip_ensemble}
OUT=${OUT:-$E/deep_sym}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONWARNINGS=ignore
echo "== stage A analyze (symmetry-aware)"; $PY -m relaxgen.prerelax.umlip_ensemble_diversity analyze --output-dir $E --workers 24 2>&1 | grep -v Warning
echo "== matrices"; $PY -m experiments.stage_a.endpoint_rmsd_matrices $OUT
echo "== metrics"; $PY -m experiments.stage_a.endpoint_metrics $OUT
echo "== stage B select"; $PY -m experiments.umlip_ensemble_stage_b select --output-dir $E/stage_b --workers 24 --overwrite
