# Runtime environment for the qe-7.5-{a40,a100,b200} GPU builds.  Source it
# after ``module purge`` in a launcher.
#
# The builds were compiled with NVHPC 25.11 + OpenMPI 5.0.7.  When /apps moved
# to the 202606 software stack (which only ships nvhpc/26.1) those modules
# disappeared.  The 202512 stack is still on disk, but its modulefiles cannot
# be reused because they hardcode /apps paths, so this reproduces what
# ``module load nvhpc/25.11 nvompi/25.11_5.0.7`` set.  Remove it once QE is
# rebuilt against nvhpc/26.1.

QE_GPU_OLD_STACK=${QE_GPU_OLD_STACK:-/sfs/weka/applications/202512_build/software/standard}
QE_GPU_NVHPC_LIB=${QE_GPU_OLD_STACK}/core/nvhpc/25.11/Linux_x86_64/25.11/compilers/lib
QE_GPU_OMPI_ROOT=${QE_GPU_OLD_STACK}/compiler/nvhpc/25.11/openmpi/5.0.7
QE_GPU_CUDA_LIB=${QE_GPU_OLD_STACK}/core/cuda/13.0.2/targets/x86_64-linux/lib
QE_GPU_PMIX_LIB=/opt/pmix/5.0.7/lib

for qe_gpu_dir in "${QE_GPU_NVHPC_LIB}" "${QE_GPU_OMPI_ROOT}/lib" "${QE_GPU_CUDA_LIB}" "${QE_GPU_PMIX_LIB}"; do
  [[ -d "${qe_gpu_dir}" ]] || {
    echo "[ERROR] QE GPU runtime directory is missing: ${qe_gpu_dir}" >&2
    echo "[ERROR] The 202512 software stack may have been removed; rebuild QE with nvhpc/26.1." >&2
    exit 2
  }
done

# OpenMPI was installed under the old /apps prefix; OPAL_PREFIX relocates it so
# it finds its MCA components.
export OPAL_PREFIX=${QE_GPU_OMPI_ROOT}
export PATH="${QE_GPU_OMPI_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${QE_GPU_NVHPC_LIB}:${QE_GPU_OMPI_ROOT}/lib:${QE_GPU_CUDA_LIB}:${QE_GPU_PMIX_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SLURM_CPU_BIND_TYPE=cores
export SLURM_MPI_TYPE=pmix
export OMPI_MCA_btl=^openib
export UCX_WARN_UNUSED_ENV_VARS=n
unset qe_gpu_dir
