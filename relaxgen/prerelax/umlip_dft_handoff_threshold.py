#!/usr/bin/env python3
"""Locate the uMLIP force target beyond which tighter pre-relaxation stops helping DFT.

A fixed cohort of Bi2Te3 reconstruction candidates is re-relaxed with the
production EqV3/FIRE2 protocol, except that the ML force target is tightened
to 1e-4 eV/Angstrom.  From each trajectory the first frame below every ML-fmax
checkpoint is selected.  That frame is exactly where a production relaxation
run with that ``fmax`` would have stopped and handed off to DFT.  Every
selected frame then receives one fixed-cell QE SCF single point with the
repository's precise PBEsol protocol.

The summary relates the ML force target to the DFT maximum force, the DFT
energy, ML/DFT force agreement, and, where a converged QE relaxation of the
same candidate exists, the energy and RMSD remaining to the DFT minimum.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read as ase_read
from ase.io import write as ase_write

from relaxgen.structures.bi2te3_defect_generation import (
    QE_PSEUDOPOTENTIALS,
    file_sha256,
    formula_counts,
    geometry_sha256,
    precise_qe_input_data,
    read_single_structure,
    rewrite_qe_geometry_float64,
    write_extxyz_float64,
)
from relaxgen.dft.qe_matched_diagnostic import (
    DEFAULT_PSEUDO_DIR,
    _frames_with_results,
    _parse_qe_timers,
    _read_live_timing,
    read_json,
    run_task,
    utc_now,
    write_json_atomic,
)
from relaxgen.prerelax import relax_reconstruction_candidates_eqv3 as eqv3_relax
PILOT_ROOT = Path("/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot")
DEFAULT_UMLIP_DIR = PILOT_ROOT / "05_eqv3_tight_relaxations"
DEFAULT_OUTPUT_DIR = PILOT_ROOT / "09_umlip_dft_handoff_threshold"
DEFAULT_REFERENCE_DIRS = (
    PILOT_ROOT / "08_qe_relaxations_historical_convergence",
    PILOT_ROOT / "07_qe_relaxations_tight_umlip",
    PILOT_ROOT / "06_qe_matched_diagnostic",
)

# Production FIRE2 stops at the first frame with fmax below the target, so each
# checkpoint is one possible uMLIP-to-DFT handoff point.
ML_FMAX_CHECKPOINTS = (
    0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5.0e-4, 2.0e-4, 1.0e-4
)

# Cases that already have QE relaxations from their uMLIP endpoints, so the
# remaining energy and distance to the DFT minimum can be measured directly.
NAMED_CASES = (
    ("VTe", "VTe_001", "unperturbed", "matched diagnostic: easy single vacancy"),
    ("3Bi_Te", "3Bi_Te_002", "center_01_TeA_000_contract_40pct",
     "matched diagnostic: aggressive 3Bi_Te contraction"),
    ("3VTe", "3VTe_002", "center_01_TeA_000_contract_40pct",
     "matched diagnostic: structurally diverse 3VTe"),
    ("3VBi", "3VBi_001", "center_03_Bi_010_contract_40pct",
     "matched diagnostic: structurally diverse 3VBi"),
    ("2Bi_Te", "2Bi_Te_001", "center_01_TeA_000_contract_20pct",
     "tight QE reference: 2Bi_Te contraction"),
    ("2Bi_Te", "2Bi_Te_001", "center_02_TeA_003_contract_40pct",
     "tight QE reference: 2Bi_Te contraction"),
)
# One seeded rattle per defect family spans every family in the catalog.
COHORT_FAMILIES = (
    "2Bi_Te", "2Te_Bi", "2VBi", "2VTe", "3Bi_Te", "3Te_Bi", "3VBi",
    "3VTe", "Bi_Te", "Te_Bi", "VBi", "VBi_VTe", "VTe", "antisite_pair",
)
FAMILY_RATTLE_CANDIDATE = "rattle_seed_101"


def _cohort_specs() -> list[tuple[str, str, str, str]]:
    specs = list(NAMED_CASES)
    specs.extend(
        (family, f"{family}_001", FAMILY_RATTLE_CANDIDATE, "family coverage: seeded rattle")
        for family in COHORT_FAMILIES
    )
    keys = [spec[:3] for spec in specs]
    if len(set(keys)) != len(keys):
        raise RuntimeError("The cohort specification contains duplicate candidates.")
    return specs


def _catalog_record(source_structure: Path, candidate_id: str) -> dict[str, Any]:
    summary = read_json(source_structure.parent / "candidate_summary.json")
    catalog = Path(summary["candidate_catalog"])
    matches = [
        record
        for record in (json.loads(line) for line in catalog.read_text().splitlines())
        if record["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {candidate_id!r} in {catalog}, found {len(matches)}.")
    return matches[0]


def prepare_cohort(args: argparse.Namespace) -> dict[str, Any]:
    umlip_dir = args.umlip_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cohort_path = output_dir / "cohort.json"
    if cohort_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {cohort_path}; pass --overwrite.")

    entries = []
    for index, (family, config_id, candidate_id, role) in enumerate(_cohort_specs()):
        production_status_path = umlip_dir / family / config_id / candidate_id / "status.json"
        production = read_json(production_status_path)
        source = Path(production["source_candidate_structure"])
        record = _catalog_record(source, candidate_id)
        atoms = read_single_structure(source)
        if geometry_sha256(atoms) != record["geometry_sha256"]:
            raise ValueError(f"Candidate geometry hash mismatch for {source}.")
        entries.append(
            {
                "cohort_index": index,
                "defect_family": family,
                "defect_config_id": config_id,
                "candidate_id": candidate_id,
                "candidate_type": record["candidate_type"],
                "role": role,
                "source_candidate_structure": source.as_posix(),
                "source_candidate_geometry_sha256": record["geometry_sha256"],
                "candidate_record": record,
                "production_umlip_status": production_status_path.as_posix(),
                "production_umlip_converged": production.get("converged"),
                "production_umlip_optimizer_steps": production.get("optimizer_steps"),
                "deep_relax_dir": (
                    output_dir / "eqv3_deep" / family / config_id / candidate_id
                ).as_posix(),
                "formula_counts": formula_counts(atoms),
                "natoms": len(atoms),
            }
        )
    cohort = {
        "artifact_type": "umlip_dft_handoff_threshold_cohort",
        "prepared_utc": utc_now(),
        "production_umlip_root": umlip_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "cohort_size": len(entries),
        "ml_fmax_checkpoints_ev_per_angstrom": list(ML_FMAX_CHECKPOINTS),
        "entries": entries,
    }
    write_json_atomic(cohort_path, cohort)
    return cohort


def deep_relax(args: argparse.Namespace) -> dict[str, Any]:
    cohort_path = args.cohort.expanduser().resolve()
    cohort = read_json(cohort_path)
    matches = [entry for entry in cohort["entries"] if entry["cohort_index"] == args.index]
    if len(matches) != 1:
        raise IndexError(f"Cohort index {args.index} is not in {cohort_path}.")
    entry = matches[0]
    if args.fmax <= 0 or args.max_steps < 1 or args.maxstep <= 0:
        raise ValueError("--fmax, --max-steps, and --maxstep must be positive.")

    deep_root = Path(cohort["output_dir"]) / "eqv3_deep"
    model = eqv3_relax.model_provenance(args.checkpoint, args.config_yml, deep_root)
    # Identical keys and values to the production run except the force target
    # and step budget, so the two protocols differ only in how far they go.
    settings = {
        "optimizer": "FIRE2",
        "fixed_cell": True,
        "target_fmax_ev_per_angstrom": float(args.fmax),
        "max_steps": int(args.max_steps),
        "maxstep_angstrom": float(args.maxstep),
        "warning_min_distance_angstrom": float(args.warning_min_distance),
        "hard_min_distance_angstrom": float(args.hard_min_distance),
        "seed": int(args.seed),
    }
    signature = eqv3_relax.run_signature(model, settings)
    runner = eqv3_relax.load_eqv3(args.checkpoint, args.config_yml, args.device, args.seed)
    result = eqv3_relax.relax_candidate(
        entry["candidate_record"],
        Path(entry["source_candidate_structure"]),
        Path(entry["deep_relax_dir"]),
        runner.calculator,
        model,
        settings,
        signature,
    )
    final = result.get("final", {})
    print(
        f"[{result['status'].upper()}] {entry['defect_config_id']}/{entry['candidate_id']} "
        f"converged={result.get('converged')} steps={result.get('optimizer_steps')} "
        f"final_ml_fmax={final.get('max_force_ev_per_angstrom')}",
        flush=True,
    )
    if result["status"] != "completed":
        raise RuntimeError(result.get("failure_reason", "deep relaxation failed"))
    return result


def import_umlip_runs(args: argparse.Namespace) -> dict[str, Any]:
    """Stand another model's relaxations of the same cohort in for the EqV3 deep runs.

    ``umlip_ensemble_diversity.py`` (Stage A) relaxed every candidate with each
    model (BFGS, ML fmax 1e-3).  Each cohort entry's run is rewritten as a
    completed deep relaxation (extxyz frames carrying ML fmax, energy, and
    forces) under ``--output-dir``, with a cohort file that points there, so
    ``prepare-scf`` and ``summarize`` run unchanged on that model.
    """

    from ase.io.trajectory import Trajectory

    source = read_json(args.source_cohort.expanduser().resolve())
    ensemble_dir = args.ensemble_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cohort_path = output_dir / "cohort.json"
    if cohort_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {cohort_path}; pass --overwrite.")

    entries = []
    for entry in source["entries"]:
        run_dir = output_dir / f"{args.model}_runs" / entry["defect_family"] / entry["defect_config_id"]
        deep_dir = run_dir / entry["candidate_id"]
        deep_dir.mkdir(parents=True, exist_ok=True)
        record = ensemble_dir / args.model / entry["defect_family"] / entry["defect_config_id"]
        run_path = record / f"{entry['candidate_id']}.json"
        run = read_json(run_path) if run_path.is_file() else None
        if run is None or run["unsafe_geometry"]:
            status = {"status": "failed", "failure_reason": "no run" if run is None else "unsafe geometry"}
        else:
            frames = []
            with Trajectory(run["trajectory"]) as reader:
                for step, frame in enumerate(reader):
                    forces = frame.get_forces()
                    frame.info.update(
                        ml_fmax_ev_per_angstrom=float(np.linalg.norm(forces, axis=1).max()),
                        ml_energy_ev=float(frame.get_potential_energy()),
                        optimizer="BFGS",
                        optimizer_step=step,
                    )
                    frames.append(frame)
            trajectory = deep_dir / "trajectory.extxyz"
            ase_write(trajectory, frames, format="extxyz")
            status = {"status": "completed", "converged": run["converged"], "trajectory": trajectory.as_posix(),
                      "source_run": run_path.as_posix(), "umlip_model": args.model}
        write_json_atomic(deep_dir / "status.json", status)
        entries.append({**entry, "deep_relax_dir": deep_dir.as_posix(), "umlip_model": args.model})
    cohort = {**source, "prepared_utc": utc_now(), "output_dir": output_dir.as_posix(),
              "umlip_model": args.model, "source_cohort": args.source_cohort.as_posix(),
              "umlip_ensemble_dir": ensemble_dir.as_posix(), "entries": entries}
    write_json_atomic(cohort_path, cohort)
    return cohort


def merge_manifests(args: argparse.Namespace) -> dict[str, Any]:
    """One SCF manifest over several, for a single ``qe_dispatch.py`` campaign.

    Tasks are renumbered; each keeps its own task directory and status file, so
    ``summarize`` still runs on the per-model manifests.
    """

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite.")
    manifests = [read_json(path.expanduser().resolve()) for path in args.manifests]
    protocols = {json.dumps(m["protocol"], sort_keys=True) for m in manifests}
    if len(protocols) != 1:
        raise ValueError("Manifests use different QE protocols.")
    tasks = []
    for path, manifest in zip(args.manifests, manifests):
        for task in manifest["tasks"]:
            tasks.append({**task, "task_index": len(tasks), "source_manifest": path.as_posix(),
                          "source_task_index": task["task_index"]})
    merged = {"artifact_type": "umlip_dft_handoff_scf_campaign", "prepared_utc": utc_now(),
              "output_dir": output.parent.as_posix(), "source_manifests": [p.as_posix() for p in args.manifests],
              "protocol": manifests[0]["protocol"], "task_count": len(tasks), "tasks": tasks}
    write_json_atomic(output, merged)
    return merged


def _trajectory_frames(path: Path) -> list[Atoms]:
    frames = ase_read(path, index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]
    for frame in frames:
        frame.set_pbc((True, True, True))
    return frames


def select_handoff_frames(
    ml_fmax: Sequence[float], checkpoints: Sequence[float]
) -> tuple[list[dict[str, Any]], list[float]]:
    """Map checkpoints to the frame where ``FIRE2.run(fmax=t)`` would stop.

    Returns one record per distinct frame (in trajectory order) and the
    checkpoints the trajectory never reached.  The final frame is always
    included as the deepest available handoff.  Unrelaxed ShakeNBreak inputs
    can carry ML forces of order 100 eV/Angstrom, so frame 0 is only used when
    it already satisfies a checkpoint.
    """

    values = np.asarray(ml_fmax, dtype=np.float64)
    labels: dict[int, list[str]] = defaultdict(list)
    thresholds: dict[int, list[float]] = defaultdict(list)
    unreached = []
    for threshold in sorted(checkpoints, reverse=True):
        # ASE's convergence test is a strict ``fmax < target``.
        below = np.flatnonzero(values < float(threshold))
        if below.size == 0:
            unreached.append(float(threshold))
            continue
        frame = int(below[0])
        labels[frame].append(f"fmax<{threshold:g}")
        thresholds[frame].append(float(threshold))
    labels[len(values) - 1].append("final")
    records = [
        {
            "frame_index": frame,
            "checkpoint_labels": labels[frame],
            "ml_fmax_checkpoints_ev_per_angstrom": thresholds[frame],
        }
        for frame in sorted(labels)
    ]
    return records, unreached


def _prepare_scf_input(
    atoms: Atoms,
    path: Path,
    state_dir: Path,
    pseudo_dir: Path,
    prefix: str,
    *,
    kpts: tuple[int, int, int],
    ecutwfc_ry: float,
    ecutrho_ry: float,
) -> None:
    input_data = precise_qe_input_data(ecutwfc_ry=ecutwfc_ry, ecutrho_ry=ecutrho_ry)
    input_data["control"].update(
        calculation="scf",
        restart_mode="from_scratch",
        prefix=prefix,
        outdir=state_dir.as_posix(),
        pseudo_dir=pseudo_dir.as_posix(),
        tprnfor=True,
        # A relaxation's saved wavefunctions occupied ~28 GB; single points
        # are never restarted, so only the XML data file is kept.
        disk_io="nowf",
    )
    ase_write(
        path,
        atoms,
        format="espresso-in",
        input_data=input_data,
        pseudopotentials=QE_PSEUDOPOTENTIALS,
        kpts=kpts,
    )
    rewrite_qe_geometry_float64(path, atoms)


def prepare_scf(args: argparse.Namespace) -> dict[str, Any]:
    cohort_path = args.cohort.expanduser().resolve()
    cohort = read_json(cohort_path)
    output_dir = Path(cohort["output_dir"])
    pseudo_dir = args.pseudo_dir.expanduser().resolve()
    manifest_path = output_dir / "scf_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --overwrite.")
    for filename in QE_PSEUDOPOTENTIALS.values():
        if not (pseudo_dir / filename).is_file():
            raise FileNotFoundError(pseudo_dir / filename)
    checkpoints = tuple(sorted({float(value) for value in args.checkpoints}, reverse=True))

    tasks = []
    candidates = []
    excluded = []
    for entry in cohort["entries"]:
        deep_dir = Path(entry["deep_relax_dir"])
        status_path = deep_dir / "status.json"
        status = read_json(status_path) if status_path.is_file() else {}
        if status.get("status") != "completed":
            excluded.append(
                {
                    "cohort_index": entry["cohort_index"],
                    "candidate": f"{entry['defect_config_id']}/{entry['candidate_id']}",
                    "reason": status.get("failure_reason", "deep relaxation not completed"),
                }
            )
            continue
        trajectory_path = Path(status["trajectory"])
        frames = _trajectory_frames(trajectory_path)
        ml_fmax = [float(frame.info["ml_fmax_ev_per_angstrom"]) for frame in frames]
        selections, unreached = select_handoff_frames(ml_fmax, checkpoints)
        candidates.append(
            {
                "cohort_index": entry["cohort_index"],
                "defect_family": entry["defect_family"],
                "defect_config_id": entry["defect_config_id"],
                "candidate_id": entry["candidate_id"],
                "deep_relax_converged": status.get("converged"),
                "deep_relax_final_ml_fmax_ev_per_angstrom": ml_fmax[-1],
                "trajectory": trajectory_path.as_posix(),
                "trajectory_frame_count": len(frames),
                "unreached_checkpoints_ev_per_angstrom": unreached,
                "selected_frame_indices": [row["frame_index"] for row in selections],
            }
        )
        for selection in selections:
            frame_index = selection["frame_index"]
            frame = frames[frame_index]
            atoms = Atoms(
                symbols=frame.get_chemical_symbols(),
                positions=np.asarray(frame.positions, dtype=np.float64),
                cell=np.asarray(frame.cell.array, dtype=np.float64),
                pbc=True,
            )
            task_index = len(tasks)
            task_dir = output_dir / "scf_tasks" / f"{task_index:04d}"
            state_dir = task_dir / "qe_state"
            input_path = task_dir / "input.extxyz"
            qe_input = task_dir / "scf.pwi"
            state_dir.mkdir(parents=True, exist_ok=True)
            write_extxyz_float64(input_path, atoms)
            _prepare_scf_input(
                atoms,
                qe_input,
                state_dir,
                pseudo_dir,
                prefix=f"handoff_{task_index:04d}",
                kpts=tuple(args.kpts),
                ecutwfc_ry=args.ecutwfc,
                ecutrho_ry=args.ecutrho,
            )
            label = selection["checkpoint_labels"][-1].replace("<", "_lt_")
            tasks.append(
                {
                    "task_index": task_index,
                    "task_id": (
                        f"{task_index:04d}_{entry['defect_config_id']}_"
                        f"{entry['candidate_id']}_frame{frame_index:05d}_{label}"
                    ),
                    "task_artifact_type": "umlip_dft_handoff_scf",
                    "case_id": f"{entry['defect_config_id']}/{entry['candidate_id']}",
                    "role": entry["role"],
                    "cohort_index": entry["cohort_index"],
                    "defect_family": entry["defect_family"],
                    "defect_config_id": entry["defect_config_id"],
                    "candidate_id": entry["candidate_id"],
                    "candidate_type": entry["candidate_type"],
                    "frame_index": frame_index,
                    "trajectory_frame_count": len(frames),
                    "optimizer": frame.info.get("optimizer"),
                    "optimizer_step": int(frame.info.get("optimizer_step", frame_index)),
                    "ml_fmax_ev_per_angstrom": ml_fmax[frame_index],
                    "ml_energy_ev": float(frame.info["ml_energy_ev"]),
                    "checkpoint_labels": selection["checkpoint_labels"],
                    "ml_fmax_checkpoints_ev_per_angstrom": selection[
                        "ml_fmax_checkpoints_ev_per_angstrom"
                    ],
                    "resource_profile": "a100_or_b200_or_h100_or_cpu",
                    "source_trajectory": trajectory_path.as_posix(),
                    "source_geometry_sha256": geometry_sha256(atoms),
                    "formula_counts": formula_counts(atoms),
                    "natoms": len(atoms),
                    "task_dir": task_dir.as_posix(),
                    "input_structure": input_path.as_posix(),
                    "qe_input": qe_input.as_posix(),
                    "qe_input_sha256": file_sha256(qe_input),
                    "qe_output": (task_dir / "scf.pwo").as_posix(),
                    "qe_stderr": (task_dir / "scf.err").as_posix(),
                    "timing_events": (task_dir / "timing_events.jsonl").as_posix(),
                    "qe_state_dir": state_dir.as_posix(),
                    "status": (task_dir / "status.json").as_posix(),
                }
            )
    if not tasks:
        raise RuntimeError("No completed deep relaxations were available.")

    protocol = {
        "calculation": "scf",
        "fixed_cell": True,
        "spin_polarized": False,
        "pseudopotentials": dict(QE_PSEUDOPOTENTIALS),
        "pseudo_dir": pseudo_dir.as_posix(),
        "kpts": list(args.kpts),
        "ecutwfc_ry": args.ecutwfc,
        "ecutrho_ry": args.ecutrho,
        "occupations": "smearing",
        "smearing": "mv",
        "degauss_ry": 0.001,
        "scf_conv_thr_ry": 1.0e-9,
        "disk_io": "nowf",
        "geometry_dtype": "float64",
        "qe_geometry_float_significant_digits": 17,
        "allowed_backends": ["a100", "b200", "h100", "cpu"],
    }
    manifest = {
        "artifact_type": "umlip_dft_handoff_scf_campaign",
        "execution_status": "prepared_not_submitted",
        "prepared_utc": utc_now(),
        "cohort": cohort_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "ml_fmax_checkpoints_ev_per_angstrom": list(checkpoints),
        "task_count": len(tasks),
        "candidates": candidates,
        "excluded_candidates": excluded,
        "protocol": protocol,
        "tasks": tasks,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def _parse_scf_result(task: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    input_atoms = read_single_structure(Path(task["input_structure"]))
    output_path = Path(task["qe_output"])
    text = output_path.read_text(encoding="utf-8", errors="ignore")
    frames = _frames_with_results(output_path)
    if len(frames) != 1:
        raise ValueError(f"An SCF single point produced {len(frames)} force blocks.")
    result = frames[0]
    energy = float(result.get_potential_energy())
    forces = np.asarray(result.get_forces(), dtype=np.float64)
    force_norms = np.linalg.norm(forces, axis=1)
    iterations = [
        int(value)
        for value in re.findall(
            r"convergence has been achieved in\s+(\d+)\s+iterations", text, re.I
        )
    ]
    job_done = "JOB DONE" in text
    electronic_converged = bool(iterations) and "convergence NOT achieved" not in text
    symbols_unchanged = result.get_chemical_symbols() == input_atoms.get_chemical_symbols()
    cell_unchanged = bool(
        np.allclose(result.cell.array, input_atoms.cell.array, atol=1.0e-6, rtol=0.0)
    )
    # QE echoes SCF positions in alat units to 7 decimals (~1e-6 Angstrom).
    position_shift, _ = find_mic(
        np.asarray(result.positions - input_atoms.positions, dtype=np.float64),
        input_atoms.cell,
        pbc=True,
    )
    positions_unchanged = bool(np.linalg.norm(position_shift, axis=1).max() <= 1.0e-4)
    singlepoint_path = Path(task["task_dir"]) / "dft_singlepoint.extxyz"
    ase_write(singlepoint_path, result, format="extxyz")
    return {
        "job_done": job_done,
        "electronic_converged": electronic_converged,
        "scientifically_converged": bool(
            job_done
            and electronic_converged
            and symbols_unchanged
            and cell_unchanged
            and positions_unchanged
            and np.isfinite(forces).all()
        ),
        "dft_energy_ev": energy,
        "final_energy_ev": energy,
        "final_max_force_ev_per_angstrom": float(force_norms.max()),
        "final_mean_force_ev_per_angstrom": float(force_norms.mean()),
        "final_rms_force_ev_per_angstrom": float(np.sqrt(np.mean(force_norms**2))),
        "singlepoint_evaluation_count": 1,
        "electronic_iteration_counts": iterations,
        "electronic_iteration_sum": sum(iterations),
        "qe_footer_timing": _parse_qe_timers(text),
        "live_timing": _read_live_timing(Path(task["timing_events"])),
        "cell_unchanged": cell_unchanged,
        "positions_unchanged": positions_unchanged,
        "species_and_order_unchanged": symbols_unchanged,
        "dft_singlepoint": singlepoint_path.as_posix(),
    }


def prepare_backend_check(args: argparse.Namespace) -> dict[str, Any]:
    """Recompute the first SCF of a finished GPU QE run on another backend.

    The reference run's first force block is an SCF at its input geometry
    with the same electronic protocol, so the comparison isolates the build.
    Each copy is an independent task, so several MPI layouts can be timed.
    """

    reference_status_path = args.reference_status.expanduser().resolve()
    reference = read_json(reference_status_path)
    output_dir = args.output_dir.expanduser().resolve()
    pseudo_dir = args.pseudo_dir.expanduser().resolve()
    manifest_path = output_dir / "scf_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --overwrite.")
    atoms = read_single_structure(Path(reference["input_structure"]))
    reference_frame = _frames_with_results(Path(reference["qe_output"]))[0]
    if not np.allclose(reference_frame.positions, atoms.positions, atol=1.0e-4):
        raise ValueError("The reference run's first SCF is not at its input geometry.")

    tasks = []
    for task_index in range(args.copies):
        task_dir = output_dir / "scf_tasks" / f"{task_index:04d}"
        state_dir = task_dir / "qe_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        input_path = task_dir / "input.extxyz"
        qe_input = task_dir / "scf.pwi"
        write_extxyz_float64(input_path, atoms)
        _prepare_scf_input(
            atoms,
            qe_input,
            state_dir,
            pseudo_dir,
            prefix=f"backend_check_{task_index:04d}",
            kpts=tuple(args.kpts),
            ecutwfc_ry=args.ecutwfc,
            ecutrho_ry=args.ecutrho,
        )
        tasks.append(
            {
                "task_index": task_index,
                "task_id": f"{task_index:04d}_backend_check_{reference.get('case_id', 'reference')}",
                "task_artifact_type": "qe_backend_check_scf",
                "case_id": reference.get("case_id"),
                "candidate_id": reference.get("candidate_id"),
                "resource_profile": "a100_or_b200_or_h100_or_cpu",
                "source_geometry_sha256": geometry_sha256(atoms),
                "task_dir": task_dir.as_posix(),
                "input_structure": input_path.as_posix(),
                "qe_input": qe_input.as_posix(),
                "qe_input_sha256": file_sha256(qe_input),
                "qe_output": (task_dir / "scf.pwo").as_posix(),
                "qe_stderr": (task_dir / "scf.err").as_posix(),
                "timing_events": (task_dir / "timing_events.jsonl").as_posix(),
                "qe_state_dir": state_dir.as_posix(),
                "status": (task_dir / "status.json").as_posix(),
            }
        )
    reference_forces = np.asarray(reference_frame.get_forces(), dtype=np.float64)
    manifest = {
        "artifact_type": "qe_backend_check",
        "prepared_utc": utc_now(),
        "output_dir": output_dir.as_posix(),
        "task_count": len(tasks),
        "reference": {
            "status": reference_status_path.as_posix(),
            "qe_output": reference["qe_output"],
            "backend": reference.get("resource_profile"),
            "gpu_name": reference.get("runtime", {}).get("allocated_gpu_name"),
            "first_scf_energy_ev": float(reference_frame.get_potential_energy()),
            "first_scf_forces_ev_per_angstrom": reference_forces.tolist(),
            "first_scf_wall_seconds": (
                reference.get("live_timing", {})
                .get("evaluations", [{}])[0]
                .get("observed_scf_through_force_wall_seconds")
            ),
        },
        "protocol": {
            "calculation": "scf",
            "pseudopotentials": dict(QE_PSEUDOPOTENTIALS),
            "kpts": list(args.kpts),
            "ecutwfc_ry": args.ecutwfc,
            "ecutrho_ry": args.ecutrho,
            "smearing": "mv",
            "degauss_ry": 0.001,
            "scf_conv_thr_ry": 1.0e-9,
        },
        "tasks": tasks,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def compare_backend_check(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = read_json(manifest_path)
    reference = manifest["reference"]
    reference_forces = np.asarray(reference["first_scf_forces_ev_per_angstrom"])
    rows = []
    for task in manifest["tasks"]:
        status_path = Path(task["status"])
        status = read_json(status_path) if status_path.is_file() else {}
        row = {
            "task_index": task["task_index"],
            "execution_status": status.get("execution_status", "pending"),
            "backend": status.get("resource_profile"),
            "qe_command": " ".join(status.get("runtime", {}).get("qe_command", [])[-8:]),
        }
        if status.get("scientifically_converged") is True:
            result = ase_read(status["dft_singlepoint"], index=0, format="extxyz")
            delta = np.asarray(result.get_forces(), dtype=np.float64) - reference_forces
            evaluations = status.get("live_timing", {}).get("evaluations", [{}])
            row.update(
                {
                    "energy_difference_mev": 1.0e3
                    * (status["dft_energy_ev"] - reference["first_scf_energy_ev"]),
                    "max_atom_force_difference_ev_per_angstrom": float(
                        np.linalg.norm(delta, axis=1).max()
                    ),
                    "rms_force_difference_ev_per_angstrom": float(
                        np.sqrt(np.mean(np.sum(delta**2, axis=1)))
                    ),
                    "electronic_iterations": sum(status.get("electronic_iteration_counts", [])),
                    "qe_wall_seconds": status.get("runtime", {}).get("qe_wall_time_seconds"),
                    "scf_through_force_wall_seconds": evaluations[0].get(
                        "observed_scf_through_force_wall_seconds"
                    )
                    if evaluations
                    else None,
                }
            )
        rows.append(row)
    summary = {
        "artifact_type": "qe_backend_check_summary",
        "summarized_utc": utc_now(),
        "reference_backend": reference["backend"],
        "reference_gpu_name": reference["gpu_name"],
        "reference_first_scf_wall_seconds": reference["first_scf_wall_seconds"],
        "tasks": rows,
    }
    write_json_atomic(Path(manifest["output_dir"]) / "backend_check_summary.json", summary)
    print(
        f"Reference: {reference['gpu_name']} first SCF "
        f"{reference['first_scf_wall_seconds']} s"
    )
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return rows


def _mic_rmsd(left: Atoms, right: Atoms) -> float:
    if left.get_chemical_symbols() != right.get_chemical_symbols():
        raise ValueError("MIC RMSD requires identical atom ordering and species.")
    displacements, _ = find_mic(
        np.asarray(left.positions - right.positions, dtype=np.float64),
        left.cell,
        pbc=True,
    )
    return float(np.sqrt(np.mean(np.sum(displacements**2, axis=1))))


def _same_electronic_protocol(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("pseudopotentials", "kpts", "ecutwfc_ry", "ecutrho_ry", "smearing", "degauss_ry")
    return all(left.get(key) == right.get(key) for key in keys)


def collect_references(
    reference_dirs: Sequence[Path], protocol: dict[str, Any], max_fmax: float
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Best converged QE relaxation per candidate under the same electronic protocol."""

    references: dict[tuple[str, str, str], dict[str, Any]] = {}
    for root in reference_dirs:
        manifests = sorted(root.glob("*manifest.json"))
        if not manifests:
            continue
        reference_protocol = read_json(manifests[0]).get("protocol", {})
        if not _same_electronic_protocol(protocol, reference_protocol):
            continue
        for status_path in sorted(root.glob("tasks/*/status.json")):
            status = read_json(status_path)
            fmax = status.get("final_max_force_ev_per_angstrom")
            relaxed = status.get("relaxed_structure")
            if not (
                status.get("job_done") is True
                and status.get("cell_unchanged") is True
                and fmax is not None
                and float(fmax) <= max_fmax
                and relaxed
                and Path(relaxed).is_file()
            ):
                continue
            key = (
                status["defect_family"],
                status["defect_config_id"],
                status["candidate_id"],
            )
            if key not in references or float(fmax) < references[key]["fmax"]:
                references[key] = {
                    "fmax": float(fmax),
                    "energy_ev": float(status["final_energy_ev"]),
                    "structure": relaxed,
                    "status": status_path.as_posix(),
                }
    return references


def _force_agreement(ml_forces: np.ndarray, dft_forces: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(dft_forces - ml_forces, axis=1)
    ml_norm = np.linalg.norm(ml_forces, axis=1)
    dft_norm = np.linalg.norm(dft_forces, axis=1)
    # Directions of near-zero forces are noise; only resolved atoms count.
    resolved = (ml_norm > 1.0e-3) & (dft_norm > 1.0e-3)
    angles = np.degrees(
        np.arccos(
            np.clip(
                np.sum(ml_forces[resolved] * dft_forces[resolved], axis=1)
                / (ml_norm[resolved] * dft_norm[resolved]),
                -1.0,
                1.0,
            )
        )
    )
    return {
        "force_error_max_ev_per_angstrom": float(error.max()),
        "force_error_rms_ev_per_angstrom": float(np.sqrt(np.mean(error**2))),
        "median_ml_dft_angle_deg": float(np.median(angles)) if angles.size else math.nan,
        "resolved_angle_atom_count": int(resolved.sum()),
    }


def _first_within(rows: Sequence[dict[str, Any]], key: str, bound: float) -> float | None:
    """Largest ML checkpoint (in relaxation order) whose ``key`` is within ``bound``."""

    for row in rows:
        if row["ml_fmax_checkpoint_ev_per_angstrom"] is not None and row[key] <= bound:
            return row["ml_fmax_checkpoint_ev_per_angstrom"]
    return None


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)])
    if finite.size == 0:
        return {"count": 0, "median": None, "q25": None, "q75": None}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = read_json(manifest_path)
    output_dir = Path(manifest["output_dir"])
    references = collect_references(
        [path.expanduser().resolve() for path in args.reference_dirs],
        manifest["protocol"],
        args.reference_max_fmax,
    )
    trajectories: dict[str, list[Atoms]] = {}
    rows = []
    counts = defaultdict(int)
    for task in manifest["tasks"]:
        status_path = Path(task["status"])
        status = read_json(status_path) if status_path.is_file() else {}
        state = status.get("execution_status", "pending")
        counts[state] += 1
        if status.get("scientifically_converged") is not True:
            continue
        if task["source_trajectory"] not in trajectories:
            trajectories[task["source_trajectory"]] = _trajectory_frames(
                Path(task["source_trajectory"])
            )
        ml_frame = trajectories[task["source_trajectory"]][task["frame_index"]]
        dft = ase_read(status["dft_singlepoint"], index=0, format="extxyz")
        agreement = _force_agreement(
            np.asarray(ml_frame.get_forces(), dtype=np.float64),
            np.asarray(dft.get_forces(), dtype=np.float64),
        )
        key = (task["defect_family"], task["defect_config_id"], task["candidate_id"])
        reference = references.get(key)
        atoms = read_single_structure(Path(task["input_structure"]))
        checkpoints = task["ml_fmax_checkpoints_ev_per_angstrom"]
        rows.append(
            {
                "case_id": task["case_id"],
                "defect_family": task["defect_family"],
                "candidate_type": task["candidate_type"],
                "task_index": task["task_index"],
                "frame_index": task["frame_index"],
                "optimizer_step": task["optimizer_step"],
                "checkpoint_labels": "|".join(task["checkpoint_labels"]),
                # The loosest checkpoint served by this frame is the one a
                # production run targeting it would stop on first.
                "ml_fmax_checkpoint_ev_per_angstrom": max(checkpoints) if checkpoints else None,
                "ml_fmax_ev_per_angstrom": task["ml_fmax_ev_per_angstrom"],
                "ml_energy_ev": task["ml_energy_ev"],
                "dft_energy_ev": status["dft_energy_ev"],
                "dft_fmax_ev_per_angstrom": status["final_max_force_ev_per_angstrom"],
                "dft_mean_force_ev_per_angstrom": status["final_mean_force_ev_per_angstrom"],
                "dft_rms_force_ev_per_angstrom": status["final_rms_force_ev_per_angstrom"],
                **agreement,
                "electronic_iterations": sum(status.get("electronic_iteration_counts", [])),
                "qe_wall_seconds": status.get("runtime", {}).get("qe_wall_time_seconds"),
                "gpu_type": status.get("resource_profile"),
                "reference_dft_fmax_ev_per_angstrom": reference["fmax"] if reference else None,
                "energy_above_dft_minimum_mev": (
                    1.0e3 * (status["dft_energy_ev"] - reference["energy_ev"])
                    if reference
                    else None
                ),
                "rmsd_to_dft_minimum_angstrom": (
                    _mic_rmsd(atoms, read_single_structure(Path(reference["structure"])))
                    if reference
                    else None
                ),
            }
        )

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    candidate_summaries = []
    for case_id, case_rows in sorted(by_case.items()):
        case_rows.sort(key=lambda row: row["frame_index"])
        minimum_energy = min(row["dft_energy_ev"] for row in case_rows)
        minimum_fmax = min(row["dft_fmax_ev_per_angstrom"] for row in case_rows)
        for row in case_rows:
            row["dft_energy_above_case_minimum_mev"] = 1.0e3 * (
                row["dft_energy_ev"] - minimum_energy
            )
            row["ml_energy_drop_from_first_handoff_mev"] = 1.0e3 * (
                case_rows[0]["ml_energy_ev"] - row["ml_energy_ev"]
            )
            row["dft_energy_drop_from_first_handoff_mev"] = 1.0e3 * (
                case_rows[0]["dft_energy_ev"] - row["dft_energy_ev"]
            )
        best_fmax = min(case_rows, key=lambda row: row["dft_fmax_ev_per_angstrom"])
        best_energy = min(case_rows, key=lambda row: row["dft_energy_ev"])
        candidate_summaries.append(
            {
                "case_id": case_id,
                "defect_family": case_rows[0]["defect_family"],
                "completed_singlepoints": len(case_rows),
                "expected_singlepoints": sum(
                    task["case_id"] == case_id for task in manifest["tasks"]
                ),
                "minimum_dft_fmax_ev_per_angstrom": minimum_fmax,
                "ml_fmax_at_minimum_dft_fmax": best_fmax["ml_fmax_ev_per_angstrom"],
                "ml_fmax_at_minimum_dft_energy": best_energy["ml_fmax_ev_per_angstrom"],
                "dft_fmax_saturation_checkpoint_ev_per_angstrom": _first_within(
                    case_rows,
                    "dft_fmax_ev_per_angstrom",
                    args.fmax_saturation_ratio * minimum_fmax,
                ),
                "dft_energy_saturation_checkpoint_ev_per_angstrom": _first_within(
                    case_rows,
                    "dft_energy_above_case_minimum_mev",
                    args.energy_saturation_mev,
                ),
                "has_dft_minimum_reference": case_rows[0]["energy_above_dft_minimum_mev"]
                is not None,
            }
        )

    per_checkpoint = []
    for threshold in [*manifest["ml_fmax_checkpoints_ev_per_angstrom"], "final"]:
        if threshold == "final":
            selected = [row for row in rows if threshold in row["checkpoint_labels"].split("|")]
        else:
            label = f"fmax<{float(threshold):g}"
            selected = [row for row in rows if label in row["checkpoint_labels"].split("|")]
        per_checkpoint.append(
            {
                "checkpoint": threshold,
                "dft_fmax_ev_per_angstrom": _quantiles(
                    [row["dft_fmax_ev_per_angstrom"] for row in selected]
                ),
                "dft_energy_above_case_minimum_mev": _quantiles(
                    [row["dft_energy_above_case_minimum_mev"] for row in selected]
                ),
                "energy_above_dft_minimum_mev": _quantiles(
                    [row["energy_above_dft_minimum_mev"] for row in selected]
                ),
                "rmsd_to_dft_minimum_angstrom": _quantiles(
                    [row["rmsd_to_dft_minimum_angstrom"] for row in selected]
                ),
                "force_error_rms_ev_per_angstrom": _quantiles(
                    [row["force_error_rms_ev_per_angstrom"] for row in selected]
                ),
            }
        )

    summary = {
        "artifact_type": "umlip_dft_handoff_threshold_summary",
        "summarized_utc": utc_now(),
        "manifest": manifest_path.as_posix(),
        "task_count": manifest["task_count"],
        "task_states": dict(sorted(counts.items())),
        "completed_singlepoints": len(rows),
        "saturation_rules": {
            "dft_fmax": f"first checkpoint with DFT fmax <= {args.fmax_saturation_ratio:g} x case minimum",
            "dft_energy": f"first checkpoint within {args.energy_saturation_mev:g} meV/cell of case minimum",
        },
        "references_found": len(references),
        "per_checkpoint": per_checkpoint,
        "candidates": candidate_summaries,
    }
    write_json_atomic(output_dir / "handoff_summary.json", summary)
    if rows:
        columns = list(rows[0].keys())
        with (output_dir / "handoff_singlepoints.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (row["case_id"], row["frame_index"])))
        _plot(rows, per_checkpoint, output_dir / "handoff_threshold.png")

    lines = [
        f"Completed SCF single points: {len(rows)}/{manifest['task_count']} "
        f"(states: {dict(sorted(counts.items()))})",
        "",
        "checkpoint      n   DFT fmax median [q25, q75] eV/A     dE above case min median meV",
    ]
    for record in per_checkpoint:
        fmax = record["dft_fmax_ev_per_angstrom"]
        energy = record["dft_energy_above_case_minimum_mev"]
        if not fmax["count"]:
            continue
        name = record["checkpoint"]
        name = name if isinstance(name, str) else f"fmax<{name:g}"
        lines.append(
            f"{name:14s} {fmax['count']:3d}   {fmax['median']:.4f} "
            f"[{fmax['q25']:.4f}, {fmax['q75']:.4f}]            {energy['median']:.2f}"
        )
    lines += ["", "case                                           fmax-saturation  energy-saturation  best-fmax-at-ML"]
    for record in candidate_summaries:
        lines.append(
            f"{record['case_id']:46s} "
            f"{str(record['dft_fmax_saturation_checkpoint_ev_per_angstrom']):16s} "
            f"{str(record['dft_energy_saturation_checkpoint_ev_per_angstrom']):18s} "
            f"{record['ml_fmax_at_minimum_dft_fmax']:.2e}"
        )
    text_path = output_dir / "handoff_summary.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(text_path.read_text(encoding="utf-8"), end="")
    return summary


def _plot(
    rows: Sequence[dict[str, Any]], per_checkpoint: Sequence[dict[str, Any]], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accent, context, ink, muted, grid = "#2a78d6", "#c3c2b7", "#0b0b0b", "#898781", "#e1e0d9"
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor="#fcfcfb", sharex=True)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    numeric = [
        record for record in per_checkpoint if not isinstance(record["checkpoint"], str)
    ]
    panels = (
        (axes[0], "dft_fmax_ev_per_angstrom", "DFT max force (eV/Å)"),
        (axes[1], "dft_energy_above_case_minimum_mev", "DFT energy above case minimum (meV/cell)"),
    )
    for axis, key, ylabel in panels:
        axis.set_facecolor("#fcfcfb")
        for case_rows in by_case.values():
            case_rows = sorted(case_rows, key=lambda row: row["frame_index"])
            axis.plot(
                [row["ml_fmax_ev_per_angstrom"] for row in case_rows],
                [max(row[key], 1.0e-3) for row in case_rows],
                color=context,
                linewidth=1.0,
                marker="o",
                markersize=2.5,
                zorder=1,
            )
        medians = [
            (float(record["checkpoint"]), record[key]["median"])
            for record in numeric
            if record[key]["count"]
        ]
        if medians:
            axis.plot(
                [x for x, _ in medians],
                [max(y, 1.0e-3) for _, y in medians],
                color=accent,
                linewidth=2.0,
                marker="o",
                markersize=8,
                markeredgecolor="#fcfcfb",
                markeredgewidth=2.0,
                zorder=3,
                label="median across cases at each checkpoint",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("uMLIP max force at handoff (eV/Å) → tighter pre-relaxation", color=ink)
        axis.set_ylabel(ylabel, color=ink)
        axis.grid(True, which="major", color=grid, linewidth=0.6)
        axis.tick_params(colors=muted)
        for spine in axis.spines.values():
            spine.set_color(grid)
    lower, upper = 1.0e-4, 1.0
    axes[0].plot([upper, lower], [upper, lower], color=muted, linewidth=1.0, linestyle="--", zorder=2)
    axes[0].text(2.0e-4, 2.6e-4, "DFT = uMLIP", color=muted, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower left")
    axes[0].set_title("Does tighter uMLIP relaxation lower the DFT force?", color=ink, fontsize=10, loc="left")
    axes[1].set_title("Does it lower the DFT energy? (floored at 1e-3 meV)", color=ink, fontsize=10, loc="left")
    figure.text(0.01, 0.005, "Gray: one line per candidate.", color=muted, fontsize=8)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cohort = subparsers.add_parser("prepare-cohort", help="Freeze the candidate cohort.")
    cohort.add_argument("--umlip-dir", type=Path, default=DEFAULT_UMLIP_DIR)
    cohort.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    cohort.add_argument("--overwrite", action="store_true")

    deep = subparsers.add_parser("deep-relax", help="Re-relax one cohort entry with EqV3/FIRE2.")
    deep.add_argument("--cohort", type=Path, required=True)
    deep.add_argument("--index", type=int, required=True)
    deep.add_argument("--checkpoint", type=Path, default=eqv3_relax.DEFAULT_CHECKPOINT)
    deep.add_argument("--config-yml", type=Path, default=eqv3_relax.DEFAULT_CONFIG)
    deep.add_argument("--device", default="cuda")
    deep.add_argument("--seed", type=int, default=42)
    deep.add_argument("--fmax", type=float, default=1.0e-4)
    deep.add_argument("--max-steps", type=int, default=5000)
    deep.add_argument("--maxstep", type=float, default=0.10)
    deep.add_argument("--warning-min-distance", type=float, default=1.8)
    deep.add_argument("--hard-min-distance", type=float, default=1.2)

    scf = subparsers.add_parser("prepare-scf", help="Select handoff frames and write QE SCF inputs.")
    scf.add_argument("--cohort", type=Path, required=True)
    scf.add_argument("--pseudo-dir", type=Path, default=DEFAULT_PSEUDO_DIR)
    scf.add_argument("--checkpoints", type=float, nargs="+", default=ML_FMAX_CHECKPOINTS)
    scf.add_argument("--kpts", nargs=3, type=int, default=(6, 6, 6))
    scf.add_argument("--ecutwfc", type=float, default=60.0)
    scf.add_argument("--ecutrho", type=float, default=480.0)
    scf.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run-task", help="Run one allocated QE SCF task.")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument(
        "--backend", "--gpu-type", dest="gpu_type", choices=("a100", "b200", "h100", "cpu"), required=True
    )
    run.add_argument("--allocated-gpu-name", required=True)
    run.add_argument("--qe-executable", type=Path, required=True)
    run.add_argument("--npool", type=int, default=1)
    run.add_argument("--ndiag", type=int, default=1)
    run.add_argument("--overwrite", action="store_true")

    check = subparsers.add_parser(
        "prepare-backend-check", help="Recompute a GPU run's first SCF on another backend."
    )
    check.add_argument("--reference-status", type=Path, required=True)
    check.add_argument("--output-dir", type=Path, required=True)
    check.add_argument("--copies", type=int, default=1)
    check.add_argument("--pseudo-dir", type=Path, default=DEFAULT_PSEUDO_DIR)
    check.add_argument("--kpts", nargs=3, type=int, default=(6, 6, 6))
    check.add_argument("--ecutwfc", type=float, default=60.0)
    check.add_argument("--ecutrho", type=float, default=480.0)
    check.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser(
        "compare-backend-check", help="Compare backend-check SCFs with the GPU reference."
    )
    compare.add_argument("--manifest", type=Path, required=True)

    imported = subparsers.add_parser(
        "import-umlip-runs", help="Use another model's Stage A relaxations of the cohort as deep runs."
    )
    imported.add_argument("--source-cohort", type=Path, default=DEFAULT_OUTPUT_DIR / "cohort.json")
    imported.add_argument("--ensemble-dir", type=Path, required=True)
    imported.add_argument("--model", required=True)
    imported.add_argument("--output-dir", type=Path, required=True)
    imported.add_argument("--overwrite", action="store_true")

    merge = subparsers.add_parser("merge-manifests", help="Merge SCF manifests for one dispatch campaign.")
    merge.add_argument("--manifests", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--overwrite", action="store_true")

    summary = subparsers.add_parser("summarize", help="Aggregate finished single points.")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--reference-dirs", type=Path, nargs="*", default=DEFAULT_REFERENCE_DIRS)
    summary.add_argument("--reference-max-fmax", type=float, default=0.02)
    summary.add_argument("--fmax-saturation-ratio", type=float, default=1.1)
    summary.add_argument("--energy-saturation-mev", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-cohort":
        cohort = prepare_cohort(args)
        print(f"cohort_size={cohort['cohort_size']}")
        print(f"cohort={Path(cohort['output_dir']) / 'cohort.json'}")
    elif args.command == "deep-relax":
        deep_relax(args)
    elif args.command == "prepare-scf":
        manifest = prepare_scf(args)
        print(f"task_count={manifest['task_count']}")
        print(f"excluded_candidates={len(manifest['excluded_candidates'])}")
        print(f"manifest={Path(manifest['output_dir']) / 'scf_manifest.json'}")
    elif args.command == "import-umlip-runs":
        cohort = import_umlip_runs(args)
        print(f"cohort={Path(cohort['output_dir']) / 'cohort.json'}")
    elif args.command == "merge-manifests":
        merged = merge_manifests(args)
        print(f"task_count={merged['task_count']} manifest={args.output}")
    elif args.command == "run-task":
        run_task(args, result_parser=_parse_scf_result)
    elif args.command == "prepare-backend-check":
        manifest = prepare_backend_check(args)
        print(f"task_count={manifest['task_count']}")
        print(f"manifest={Path(manifest['output_dir']) / 'scf_manifest.json'}")
    elif args.command == "compare-backend-check":
        compare_backend_check(args)
    else:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
