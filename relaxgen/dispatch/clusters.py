"""Per-cluster backends for ``qe_dispatch``: which hardware a campaign may use, and how.

Each cluster has its own backend lists and Slurm launchers; the controller logic
in ``qe_dispatch.py`` is shared.  ``qe_dispatch init`` writes the lists of the
cluster it runs on into ``config.json`` (``--cluster`` overrides the detection),
so a running campaign keeps its backends even if this file changes.

* ``rivanna``: UVA Research Computing.  A100/B200 GPUs plus CPU backends on
  ``parallel`` and ``standard``, with launchers ``slurm/*_uva.sh``.
* ``cs``: UVA Computer Science.  Relaxations only, on the GPUs in
  ``CS_GPU_FEATURES``; the launcher is ``slurm/run_dft_relax_slurm_cs.sh``.
  CPU nodes are about 0.43x an Afton core per rank and would add ~3% to the
  GPUs, so there is no CPU backend (``docs/cost_analysis/README.md``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

# Slurm ClusterName -> profile.  Rivanna's ClusterName is not recorded yet, so it
# is recognized by its shared storage instead.
SLURM_CLUSTER_NAMES = {"cs": "cs"}
RIVANNA_MARKER = Path("/sfs/weka")


def detect_cluster() -> str:
    try:
        output = subprocess.run(["scontrol", "show", "config"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        output = ""
    name = next((line.split("=", 1)[1].strip() for line in output.splitlines()
                 if line.split("=", 1)[0].strip() == "ClusterName"), "")
    if name in SLURM_CLUSTER_NAMES:
        return SLURM_CLUSTER_NAMES[name]
    if RIVANNA_MARKER.is_dir():
        return "rivanna"
    raise SystemExit(f"Unknown cluster (Slurm ClusterName {name!r}); pass --cluster {{{','.join(PROFILES)}}}.")


# --------------------------------------------------------------------------
# Rivanna
# --------------------------------------------------------------------------

PILOT_BACKEND_CHECK = Path(
    "/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/10_qe_backend_check_fullnode"
)


def backend_check_summary(manifest_dir: Path) -> Path:
    """The full-node CPU backend check: the nearest one above the campaign, else the pilot's."""

    for directory in (manifest_dir, *manifest_dir.parents):
        check = directory / "10_qe_backend_check_fullnode"
        if check.is_dir():
            return check / "backend_check_summary.json"
    return PILOT_BACKEND_CHECK / "backend_check_summary.json"


def default_backends(manifest_dir: Path) -> list[dict[str, Any]]:
    """Backends for single-point manifests; runtimes are seeded from measured runs."""

    fullnode_check = backend_check_summary(manifest_dir)
    return [
        {
            "name": "gpu",
            "kind": "gpu",
            "launcher": "slurm/run_umlip_dft_handoff_scf_slurm_uva.sh",
            "sbatch_args": [],
            "env": {},
            "max_jobs": 32,
            "cores": 0,
            "wait_partitions": ["gpu-a100", "gpu-b200"],
            # A100 ~48 min, B200 ~29 min per production SCF (2026-10-22).
            "runtime_prior_seconds": 2600,
            "wait_prior_seconds": 900,
            "check": None,
        },
        {
            "name": "cpu-parallel-2x42",
            "kind": "cpu",
            "launcher": "slurm/run_qe_scf_cpu_slurm_uva.sh",
            "sbatch_args": ["-p", "parallel", "-N2", "--ntasks=84", "--ntasks-per-node=42",
                            "--mem-per-cpu=16G", "--time=06:00:00"],
            "env": {"NPOOL": "28"},
            "max_jobs": 8,
            "cores": 84,
            "wait_partitions": ["parallel"],
            "runtime_prior_seconds": None,
            "wait_prior_seconds": 6 * 3600,
            "check": {"summary": fullnode_check.as_posix(), "task_index": 0},
        },
        {
            "name": "cpu-standard-42",
            "kind": "cpu",
            "launcher": "slurm/run_qe_scf_cpu_slurm_uva.sh",
            "sbatch_args": ["-p", "standard", "-N1", "-C", "afton", "--ntasks=42",
                            "--mem-per-cpu=16G", "--time=08:00:00"],
            "env": {"NPOOL": "14"},
            "max_jobs": 8,
            "cores": 42,
            "wait_partitions": ["standard"],
            "runtime_prior_seconds": None,
            "wait_prior_seconds": 24 * 3600,
            "check": {"summary": fullnode_check.as_posix(), "task_index": 1},
        },
    ]


def relax_backends(manifest_dir: Path) -> list[dict[str, Any]]:
    """Backends for resumable relaxations; 1- and 2-GPU requests race for each unit."""

    fullnode_check = default_backends(manifest_dir)[1]["check"]
    gpu = {
        "kind": "gpu",
        "launcher": "slurm/run_dft_relax_slurm_uva.sh",
        "env": {},
        "cores": 0,
        "wait_partitions": ["gpu-a100", "gpu-b200"],
        # Median of 1-GPU 12 h queue probes, 2026-10-24.
        "wait_prior_seconds": 3.4 * 3600,
        "check": None,
    }
    return [
        # Step priors are slow-case (from-scratch) production SCFs: B200 29 min,
        # A100 46 min on one GPU (2026-10-22); two GPUs assumed ~1.8x faster.
        {**gpu, "name": "gpu-1x", "sbatch_args": [], "max_jobs": 10,
         "runtime_prior_seconds": 12 * 3600, "segment_seconds": 12 * 3600,
         "step_prior_seconds": 2800},
        {**gpu, "name": "gpu-2x", "sbatch_args": ["--gres=gpu:2", "--ntasks=2", "--mem=100G"],
         "max_jobs": 5, "runtime_prior_seconds": 12 * 3600, "segment_seconds": 12 * 3600,
         "step_prior_seconds": 1600},
        {
            "name": "cpu-parallel-2x42",
            "kind": "cpu",
            "launcher": "slurm/run_dft_relax_cpu_slurm_uva.sh",
            "sbatch_args": ["-p", "parallel", "-N2", "--ntasks=84", "--ntasks-per-node=42",
                            "--mem-per-cpu=16G", "--time=24:00:00"],
            "env": {"NPOOL": "28"},
            "max_jobs": 5,
            "cores": 84,
            "wait_partitions": ["parallel"],
            "runtime_prior_seconds": 24 * 3600,
            "segment_seconds": 24 * 3600,
            # p90 of Stage B's measured steps (2026-10-24); the backend check's SCF
            # wall time is used first when there is one.
            "step_prior_seconds": 80 * 60,
            # 2-node probes waited a median 4-7 min (p90 ~1.5 h) for any limit, 2026-10-24.
            "wait_prior_seconds": 15 * 60,
            "check": fullnode_check,
        },
    ]


# --------------------------------------------------------------------------
# CS cluster
# --------------------------------------------------------------------------

# Slurm features of the GPUs relaxations may run on.  The launcher picks the QE
# build from the GPU it gets (H100 -> qe-7.5-h100, A100 -> qe-7.5-a100).  The
# A100 40GB nodes are PCIe cards, 1.6x slower than the 80GB ones (2026-10-29).
CS_GPU_FEATURES = ["h100_94gb", "a100_80gb"]


def cs_relax_backends(manifest_dir: Path) -> list[dict[str, Any]]:
    """One GPU per relaxation; 24 h segments fit a whole Plan A relaxation."""

    return [
        {
            "name": "gpu-1x",
            "kind": "gpu",
            "launcher": "slurm/run_dft_relax_slurm_cs.sh",
            "sbatch_args": [f"--constraint={'|'.join(CS_GPU_FEATURES)}"],
            "env": {},
            # 12 GPUs match CS_GPU_FEATURES; a few more queued jobs take a GPU as soon
            # as one frees up.  The per-user limit is 40 GPUs.
            "max_jobs": 16,
            "cores": 0,
            "wait_partitions": ["gpu"],
            # From-scratch SCF on the slower of the two, A100 80GB 51 min
            # (H100 NVL 28 min), from the 2026-10-29 timing test.
            "step_prior_seconds": 3100,
            "runtime_prior_seconds": 24 * 3600,
            "segment_seconds": 24 * 3600,
            # Median wait of the six timing-test GPU jobs, 2026-10-28.
            "wait_prior_seconds": 2 * 3600,
            "check": None,
        },
    ]


def cs_scf_backends(manifest_dir: Path) -> list[dict[str, Any]]:
    raise SystemExit("The cs profile has no single-point (scf) backends; use --workload relax.")


PROFILES: dict[str, dict[str, Callable[[Path], list[dict[str, Any]]]]] = {
    "rivanna": {"scf": default_backends, "relax": relax_backends},
    "cs": {"scf": cs_scf_backends, "relax": cs_relax_backends},
}


def backends_for(cluster: str, workload: str, manifest_dir: Path) -> list[dict[str, Any]]:
    return PROFILES[cluster][workload](manifest_dir)
