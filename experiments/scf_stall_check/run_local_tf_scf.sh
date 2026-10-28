#!/bin/bash
# Threshold task 228 (6x6x6 SCF that stalled at ~1e-7 Ry for 75 iterations) with
# the ladder's first fallback: local-TF mixing, beta 0.1, ndim 12, 120 iterations.
# Submit from the folder that holds scf.pwi:  sbatch <relaxgen>/experiments/scf_stall_check/run_local_tf_scf.sh
#SBATCH --job-name=task228_local_tf
#SBATCH --partition=gpu
#SBATCH --constraint=a100|b200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=44G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
module purge
source "${RELAXGEN_ROOT:-/home/gqs7zr/ai4mat/relaxgen}/slurm/qe_gpu_runtime_uva.sh"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "${GPU}" in *B200*) TYPE=b200 ;; *) TYPE=a100 ;; esac
echo "GPU: ${GPU}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd "${SLURM_SUBMIT_DIR}"
mkdir -p qe_state
srun --ntasks=1 --cpus-per-task "${SLURM_CPUS_PER_TASK}" --cpu-bind=none \
  /scratch/gqs7zr/ai4mat/qe/qe-7.5-${TYPE}/bin/pw.x -nk 1 -nd 1 -in scf.pwi > scf.pwo
grep -a "convergence has been achieved\|convergence NOT achieved\|^!" scf.pwo
