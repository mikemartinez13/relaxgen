#!/usr/bin/env python3
"""Resumable fixed-cell DFT relaxations: ASE BFGS driving QE over its i-PI socket.

Each manifest task relaxes one structure with the repository's precise QE
protocol (``evaluation.baseline_eval_utils``, the same calculator builder used
by ``inference.surrogate_inference.run_dft_relaxation``) until every atomic
force is below ``fmax``.  ASE keeps one ``pw.x`` alive over a UNIX socket, so
each ionic step starts from the previous wavefunctions instead of from scratch.

A job relaxes until the structure converges or its Slurm wall-clock budget runs
low.  The trajectory and the BFGS state are written after every step, so the
next job, on any backend, resumes from the last evaluated geometry.  A segment
that ends unconverged exits 0 and leaves ``scientifically_converged`` false;
``qe_dispatch.py`` then simply schedules the unit again.

An SCF that does not converge within ``electron_maxstep`` iterations (occupations
flipping between near-degenerate defect levels at the Fermi energy) stops
``pw.x``.  The stall is a property of the geometry and recurs on any backend, so
the relaxation is abandoned: the task fails with ``failure_kind =
"scf_not_converged"`` and is not retried.  Changing the electronic settings or
the step would leave the production energy surface or the BFGS path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relaxgen.common.baseline_eval_utils import build_qe_calculator  # noqa: E402
from relaxgen.common.qe_scratch import temporary_qe_outdir  # noqa: E402

from relaxgen.structures.bi2te3_defect_generation import formula_counts, geometry_sha256, write_extxyz_float64
from relaxgen.dft.qe_matched_diagnostic import DEFAULT_PSEUDO_DIR, read_json, utc_now, write_json_atomic
# Every production (6x6x6) SCF so far converged within 22 iterations, so a cap of
# 60 stops a stall after ~2.5 A100 hours instead of ~8 at QE's 200.
ELECTRON_MAXSTEP = 60
SCF_NOT_CONVERGED_MARKER = "convergence NOT achieved"


class ScfNotConverged(RuntimeError):
    """``pw.x`` stopped because an SCF exhausted ``electron_maxstep``."""


def scf_stalled(qe_output: Path) -> bool:
    if not qe_output.exists():
        return False
    return SCF_NOT_CONVERGED_MARKER in qe_output.read_text(errors="replace")


def read_structure_spec(spec: str) -> Atoms:
    """``path`` or ``path@frame`` (e.g. a uMLIP handoff frame in a trajectory)."""

    path, _, frame = spec.partition("@")
    atoms = ase_read(path, index=int(frame) if frame else 0)
    atoms.set_pbc(True)
    return Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=np.asarray(atoms.positions, dtype=np.float64),
        cell=np.asarray(atoms.cell.array, dtype=np.float64),
        pbc=True,
    )


def step_from_stored_forces(optimizer: BFGS, atoms: Atoms, stored: Atoms, live_calculator: Any) -> None:
    """Take the BFGS step a continuous run would take from ``stored`` without a new SCF.

    ``atoms`` sits at the stored geometry and ``optimizer`` holds the restored
    BFGS state, so the step from the stored forces lands exactly where the
    uninterrupted relaxation would have gone next.  The first SCF of the
    resumed segment is then at that new geometry.
    """

    if not np.array_equal(atoms.positions, stored.positions):
        raise ValueError("Resume geometry differs from the stored frame.")
    atoms.calc = SinglePointCalculator(
        atoms, energy=stored.get_potential_energy(), forces=stored.get_forces()
    )
    optimizer.step()
    atoms.calc = live_calculator


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "relax_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --overwrite.")
    labels = args.labels or [Path(spec.partition("@")[0]).stem for spec in args.structures]
    if len(labels) != len(args.structures) or len(set(labels)) != len(labels):
        raise ValueError("--labels must be unique and match --structures one to one.")

    tasks = []
    for task_index, (spec, label) in enumerate(zip(args.structures, labels)):
        atoms = read_structure_spec(spec)
        task_dir = output_dir / "tasks" / f"{task_index:04d}_{label}"
        input_path = task_dir / "input.extxyz"
        write_extxyz_float64(input_path, atoms)
        tasks.append(
            {
                "task_index": task_index,
                "task_id": f"{task_index:04d}_{label}",
                "label": label,
                "task_artifact_type": "dft_bfgs_relaxation",
                "resource_profile": "a100_or_b200_or_h100_or_cpu",
                "source_structure": spec,
                "source_geometry_sha256": geometry_sha256(atoms),
                "formula_counts": formula_counts(atoms),
                "natoms": len(atoms),
                "task_dir": task_dir.as_posix(),
                "input_structure": input_path.as_posix(),
                "trajectory": (task_dir / "trajectory.traj").as_posix(),
                "optimizer_state": (task_dir / "bfgs_state.json").as_posix(),
                "optimizer_log": (task_dir / "optimizer.log").as_posix(),
                "relaxed_structure": (task_dir / "relaxed.extxyz").as_posix(),
                "status": (task_dir / "status.json").as_posix(),
            }
        )
    manifest = {
        "artifact_type": "dft_bfgs_relaxation_campaign",
        "prepared_utc": utc_now(),
        "output_dir": output_dir.as_posix(),
        "task_count": len(tasks),
        "protocol": {
            "calculator": "relaxgen.common.baseline_eval_utils.build_qe_calculator over the QE i-PI socket",
            "calculation_mode": args.calculation_mode,
            "pseudo_dir": args.pseudo_dir.expanduser().resolve().as_posix(),
            "kpts": list(args.kpts),
            "fixed_cell": True,
            "optimizer": "ase.optimize.BFGS",
            "maxstep_angstrom": args.maxstep,
            "fmax_ev_per_angstrom": args.fmax,
            "fmax_definition": "maximum per-atom force-vector norm (ASE convergence test)",
            "max_steps": args.max_steps,
            # Recomputing the stored geometry on resume measures restart
            # discontinuities (resume tests); production steps from stored forces.
            "recompute_on_resume": args.recompute_on_resume,
            # An SCF still unconverged after this many iterations ends the relaxation.
            "electron_maxstep": args.electron_maxstep,
        },
        "tasks": tasks,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def _deadline_unix(value: str | None) -> float:
    if not value:
        return math.inf
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value).timestamp()


def _launch_command(args: argparse.Namespace) -> str:
    executable = args.qe_executable.expanduser().resolve()
    ranks = int(os.environ.get("SLURM_NTASKS", "1"))
    if args.backend == "cpu":
        # Intel MPI; the launcher selects its bootstrap for one or more nodes.
        launcher = ["mpirun", "-np", str(ranks)]
        pools = args.npool
    else:
        # One rank and one GPU per k-point pool; the step inherits the job's GPUs.
        launcher = ["srun", f"--ntasks={ranks}", "--cpus-per-task",
                    os.environ.get("SLURM_CPUS_PER_TASK", "1"), "--cpu-bind=none"]
        pools = ranks
    return shlex.join([*launcher, executable.as_posix(), "-nk", str(pools), "-nd", "1"])


def load_frames(trajectory_path: Path) -> list[Atoms]:
    if trajectory_path.exists() and trajectory_path.stat().st_size > 0:
        with Trajectory(trajectory_path) as reader:
            return list(reader)
    return []


def _relax_segment(
    args: argparse.Namespace,
    task: dict[str, Any],
    status: dict[str, Any],
    status_path: Path,
    deadline: float,
    job_step_seconds: list[float],
) -> bool:
    """Run one ``pw.x`` session from the last stored frame; return whether it stopped for budget.

    Raises ``ScfNotConverged`` when QE stops on an unconverged SCF.
    """

    protocol = status["protocol"]
    trajectory_path = Path(task["trajectory"])
    frames = load_frames(trajectory_path)
    atoms = frames[-1].copy() if frames else read_structure_spec(task["input_structure"])
    atoms.calc = None
    steps_before = len(frames)
    if steps_before >= int(protocol["max_steps"]) + 1:
        status["failure_kind"] = "max_steps"
        raise RuntimeError(f"{task['task_id']} reached max_steps without converging.")

    step_seconds: list[float] = []
    segment = {
        "backend": args.backend,
        "allocated_device": args.allocated_device,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "qe_executable": args.qe_executable.expanduser().resolve().as_posix(),
        "qe_command": _launch_command(args),
        "started_utc": utc_now(),
        "frames_before": steps_before,
        "resumed": bool(frames),
        "step_wall_seconds": step_seconds,
    }
    status["segments"].append(segment)
    write_json_atomic(status_path, status)

    fmax_target = task_fmax(task, protocol)
    directory = Path(task["task_dir"]) / "qe_work" / f"segment_{len(status['segments']):03d}"
    stopped_for_budget = False
    try:
        with temporary_qe_outdir(prefix=f"relax_{task['task_index']:04d}_") as qe_outdir:
            base = build_qe_calculator(
                command=segment["qe_command"],
                pseudo_dir=protocol["pseudo_dir"],
                kpts=tuple(protocol["kpts"]),
                directory=directory,
                qe_outdir=qe_outdir,
                calculation_mode=protocol["calculation_mode"],
            )
            input_data = base.parameters["input_data"]
            # The socket keeps wavefunctions in memory between steps; a relaxation's
            # saved wavefunctions previously occupied ~28 GB.
            input_data["control"]["disk_io"] = "nowf"
            # Manifests written before the cap existed get it too.
            input_data["electrons"]["electron_maxstep"] = int(protocol.get("electron_maxstep", ELECTRON_MAXSTEP))
            unixsocket = f"relax_{os.environ.get('SLURM_JOB_ID', os.getpid())}_{task['task_index']}_{len(status['segments'])}"
            with base.socketio(unixsocket=unixsocket) as calc, Trajectory(trajectory_path, "a") as writer:
                atoms.calc = calc
                last_written = frames[-1].positions.copy() if frames else None
                frames_written = len(frames)

                def record() -> None:
                    # Resuming re-evaluates the last frame; keep each geometry once.
                    nonlocal last_written, frames_written
                    if last_written is not None and np.array_equal(atoms.positions, last_written):
                        return
                    writer.write(atoms)
                    last_written = atoms.positions.copy()
                    frames_written += 1

                optimizer = BFGS(
                    atoms,
                    restart=task["optimizer_state"],
                    logfile=task["optimizer_log"],
                    maxstep=float(protocol["maxstep_angstrom"]),
                )
                optimizer.attach(record, interval=1)
                # Manifests written before this option always recomputed on resume.
                if frames and not protocol.get("recompute_on_resume", True):
                    step_from_stored_forces(optimizer, atoms, frames[-1], calc)
                    segment["resumed_from_stored_forces"] = True
                remaining = int(protocol["max_steps"]) + 1 - steps_before
                last = time.time()
                for converged in optimizer.irun(fmax=fmax_target, steps=remaining):
                    now = time.time()
                    step_seconds.append(now - last)
                    job_step_seconds.append(now - last)
                    last = now
                    if frames and "resume_check" not in segment and not segment.get("resumed_from_stored_forces"):
                        # The first evaluation of a resumed segment recomputes the last
                        # stored geometry from a fresh SCF; any difference is a restart
                        # discontinuity rather than optimizer motion.  Resuming from stored
                        # forces evaluates the next step instead, so there is nothing to check.
                        stored = frames[-1]
                        force_delta = np.linalg.norm(atoms.get_forces() - stored.get_forces(), axis=1)
                        segment["resume_check"] = {
                            "positions_identical": bool(np.array_equal(atoms.positions, stored.positions)),
                            "energy_difference_mev": 1.0e3 * float(
                                atoms.get_potential_energy() - stored.get_potential_energy()
                            ),
                            "max_atom_force_difference_ev_per_angstrom": float(force_delta.max()),
                            "rms_force_difference_ev_per_angstrom": float(np.sqrt(np.mean(force_delta**2))),
                            "stored_frame_backend": status["segments"][-2]["backend"]
                            if len(status["segments"]) > 1 else None,
                        }
                    forces = np.linalg.norm(atoms.get_forces(), axis=1)
                    status.update(
                        {
                            "current_step": steps_before + len(step_seconds),
                            # Read by qe_dispatch.py to detect segments that make no progress.
                            "trajectory_frames": frames_written,
                            "current_max_force_ev_per_angstrom": float(forces.max()),
                            "current_energy_ev": float(atoms.get_potential_energy()),
                            "updated_utc": utc_now(),
                        }
                    )
                    write_json_atomic(status_path, status)
                    if converged:
                        break
                    # Stop while one more step (the slowest so far) and a margin still fit.
                    if now + 1.3 * max(job_step_seconds) + 300 > deadline:
                        stopped_for_budget = True
                        break
    except Exception as error:
        segment.update({"finished_utc": utc_now(), "frames_after": len(load_frames(trajectory_path))})
        if scf_stalled(directory / "espresso.pwo"):
            segment["outcome"] = "scf_not_converged"
            raise ScfNotConverged(
                f"{task['task_id']}: SCF not converged within "
                f"{input_data['electrons']['electron_maxstep']} iterations after trajectory frame "
                f"{segment['frames_after']}; relaxation abandoned."
            ) from error
        segment["outcome"] = "error"
        raise
    segment.update(
        {
            "finished_utc": utc_now(),
            "frames_after": len(load_frames(trajectory_path)),
            "stopped_for_budget": stopped_for_budget,
            "outcome": "stopped_for_budget" if stopped_for_budget else "finished",
        }
    )
    return stopped_for_budget


def task_fmax(task: dict[str, Any], protocol: dict[str, Any]) -> float:
    """The task's force target: its own (set by ``tighten``) or the manifest's."""

    return float(task.get("fmax_ev_per_angstrom", protocol["fmax_ev_per_angstrom"]))


def tighten(args: argparse.Namespace) -> list[int]:
    """Lower the force target of converged tasks so that they resume and relax further.

    The trajectory and BFGS state are kept, so the relaxation continues from its last
    frame.  Tasks that are not converged yet are left alone: a running segment has
    already read its target, and a later ``tighten`` catches them once they finish.
    """

    manifest_path = args.manifest.expanduser().resolve()
    manifest = read_json(manifest_path)
    reopened = []
    for task in manifest["tasks"]:
        if int(task["task_index"]) not in set(args.task_indices):
            continue
        status_path = Path(task["status"])
        status = read_json(status_path) if status_path.exists() else {}
        if not status.get("scientifically_converged"):
            print(f"[LEFT] {task['task_id']} is not converged yet.")
            continue
        task["fmax_ev_per_angstrom"] = args.fmax
        if float(status["final_max_force_ev_per_angstrom"]) < args.fmax:
            print(f"[LEFT] {task['task_id']} already has fmax below {args.fmax}.")
            continue
        status.update({"scientifically_converged": False, "execution_status": "segment_completed",
                       "fmax_ev_per_angstrom": args.fmax})
        write_json_atomic(status_path, status)
        reopened.append(int(task["task_index"]))
        print(f"[REOPENED] {task['task_id']}: fmax {status['final_max_force_ev_per_angstrom']:.4f} -> target {args.fmax}")
    write_json_atomic(manifest_path, manifest)
    return reopened


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.manifest.expanduser().resolve())
    protocol = manifest["protocol"]
    task = next(t for t in manifest["tasks"] if int(t["task_index"]) == args.task_index)
    if args.backend not in task["resource_profile"].split("_or_"):
        raise ValueError(f"Task {args.task_index} does not allow backend {args.backend}.")
    if args.backend != "cpu" and f"qe-7.5-{args.backend}" not in args.qe_executable.as_posix():
        raise ValueError(f"QE executable does not visibly match {args.backend}: {args.qe_executable}")

    status_path = Path(task["status"])
    previous = read_json(status_path) if status_path.exists() else {}
    if previous.get("scientifically_converged") is True:
        print(f"[SKIPPED] {task['task_id']} is already relaxed.")
        return previous
    if previous.get("failure_kind") == "scf_not_converged":
        print(f"[SKIPPED] {task['task_id']} was abandoned after an SCF stall.")
        return previous
    status = {
        **task,
        "protocol": protocol,
        "execution_status": "running",
        "scientifically_converged": False,
        "segments": list(previous.get("segments", [])),
    }

    deadline = _deadline_unix(args.deadline)
    job_step_seconds: list[float] = []
    trajectory_path = Path(task["trajectory"])
    fmax_target = task_fmax(task, protocol)
    try:
        try:
            stopped_for_budget = _relax_segment(args, task, status, status_path, deadline, job_step_seconds)
        except ScfNotConverged:
            status["failure_kind"] = "scf_not_converged"
            raise

        frames = load_frames(trajectory_path)
        if not frames:
            raise RuntimeError(f"{task['task_id']} has no evaluated frame.")
        final = frames[-1]
        force_norms = np.linalg.norm(final.get_forces(), axis=1)
        converged = bool(force_norms.max() < fmax_target)
        status.update(
            {
                "execution_status": "completed" if converged else "segment_completed",
                "scientifically_converged": converged,
                "final_energy_ev": float(final.get_potential_energy()),
                "final_max_force_ev_per_angstrom": float(force_norms.max()),
                "final_mean_force_ev_per_angstrom": float(force_norms.mean()),
                "dft_singlepoint_count": len(frames),
                "trajectory_frames": len(frames),
                "bfgs_steps": max(len(frames) - 1, 0),
            }
        )
        if converged:
            relaxed = final.copy()
            relaxed.calc = None
            write_extxyz_float64(Path(task["relaxed_structure"]), relaxed)
            status["relaxed_geometry_sha256"] = geometry_sha256(relaxed)
        elif not stopped_for_budget:
            status["execution_status"] = "failed"
            status["failure_kind"] = "max_steps"
            status["failure_reason"] = "BFGS stopped before convergence (max_steps reached)."
        write_json_atomic(status_path, status)
        segment_steps = status["segments"][-1]["step_wall_seconds"]
        print(
            f"[{status['execution_status'].upper()}] {task['task_id']}: "
            f"{len(frames)} SCFs, fmax={force_norms.max():.5f} eV/A, "
            f"job steps={len(job_step_seconds)}, "
            f"median step={np.median(segment_steps) / 60 if segment_steps else 0:.1f} min"
        )
        if status["execution_status"] == "failed":
            raise RuntimeError(status["failure_reason"])
        return status
    except Exception as error:
        status["execution_status"] = "failed"
        status.setdefault("failure_kind", "error")
        status.setdefault("failure_reason", f"{type(error).__name__}: {error}")
        if status["segments"]:
            status["segments"][-1].setdefault("finished_utc", utc_now())
        write_json_atomic(status_path, status)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Write a relaxation manifest for a set of structures.")
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--structures", nargs="+", required=True, help="path or path@frame")
    prep.add_argument("--labels", nargs="+")
    prep.add_argument("--calculation-mode", default="precise", choices=("ultrafast", "coarse", "precise"))
    prep.add_argument("--pseudo-dir", type=Path, default=DEFAULT_PSEUDO_DIR)
    prep.add_argument("--kpts", nargs=3, type=int, default=(6, 6, 6))
    prep.add_argument("--fmax", type=float, default=0.01)
    prep.add_argument("--maxstep", type=float, default=0.10)
    prep.add_argument("--max-steps", type=int, default=400)
    prep.add_argument("--electron-maxstep", type=int, default=ELECTRON_MAXSTEP)
    prep.add_argument(
        "--recompute-on-resume",
        action="store_true",
        help="Re-run the SCF of the stored geometry on resume (restart-discontinuity tests).",
    )
    prep.add_argument("--overwrite", action="store_true")

    tight = sub.add_parser("tighten", help="Lower converged tasks' force target so they relax further.")
    tight.add_argument("--manifest", type=Path, required=True)
    tight.add_argument("--task-indices", nargs="+", type=int, required=True)
    tight.add_argument("--fmax", type=float, required=True)

    run = sub.add_parser("run-task", help="Run one relaxation segment inside an allocation.")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--backend", choices=("a100", "b200", "h100", "cpu"), required=True)
    run.add_argument("--allocated-device", required=True)
    run.add_argument("--qe-executable", type=Path, required=True)
    run.add_argument("--npool", type=int, default=1, help="k-point pools (CPU backend).")
    run.add_argument("--deadline", help="Job end time (ISO or Unix seconds); stop before it.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        manifest = prepare(args)
        print(f"task_count={manifest['task_count']}")
        print(f"manifest={Path(manifest['output_dir']) / 'relax_manifest.json'}")
    elif args.command == "tighten":
        tighten(args)
    else:
        run_task(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
