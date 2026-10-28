#!/usr/bin/env python3
"""Prepare, run, and summarize the 12-run matched Bi2Te3 QE diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.units import Bohr, Rydberg

from relaxgen.structures.bi2te3_defect_generation import (
    QE_PSEUDOPOTENTIALS,
    file_sha256,
    formula_counts,
    geometry_sha256,
    precise_qe_input_data,
    read_single_structure,
    rewrite_qe_geometry_float64,
    write_extxyz_float64,
    write_json,
)
DEFAULT_UMLIP_DIR = Path(
    "/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/"
    "05_eqv3_fire2_preview"
)
DEFAULT_OUTPUT_DIR = Path(
    "/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/"
    "06_qe_matched_diagnostic"
)
DEFAULT_PSEUDO_DIR = Path(__file__).resolve().parents[2] / "pseudos"

# These were selected from the audited 2,124-candidate uMLIP cohort.  The two
# multi-vacancy endpoints are locally distinct from their configuration's
# unperturbed endpoint, rather than merely having dramatic recipe names.
SELECTED_CASES = (
    {
        "case_id": "easy_vte",
        "role": "easy single-vacancy endpoint",
        "defect_family": "VTe",
        "defect_config_id": "VTe_001",
        "candidate_id": "unperturbed",
    },
    {
        "case_id": "aggressive_3bi_te",
        "role": "aggressive 3Bi_Te contraction",
        "defect_family": "3Bi_Te",
        "defect_config_id": "3Bi_Te_002",
        "candidate_id": "center_01_TeA_000_contract_40pct",
    },
    {
        "case_id": "diverse_3vte",
        "role": "structurally diverse 3VTe endpoint",
        "defect_family": "3VTe",
        "defect_config_id": "3VTe_002",
        "candidate_id": "center_01_TeA_000_contract_40pct",
    },
    {
        "case_id": "diverse_3vbi",
        "role": "structurally diverse 3VBi endpoint",
        "defect_family": "3VBi",
        "defect_config_id": "3VBi_001",
        "candidate_id": "center_03_Bi_010_contract_40pct",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _mic_rmsd(left: Atoms, right: Atoms, mask: np.ndarray | None = None) -> float:
    if left.get_chemical_symbols() != right.get_chemical_symbols():
        raise ValueError("MIC RMSD requires identical atom ordering and species.")
    if not np.allclose(left.cell.array, right.cell.array, atol=1.0e-10, rtol=0.0):
        raise ValueError("MIC RMSD requires matching cells.")
    displacements, _ = find_mic(
        np.asarray(left.positions - right.positions, dtype=np.float64),
        left.cell,
        pbc=True,
    )
    if mask is not None:
        displacements = displacements[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(np.mean(np.sum(displacements**2, axis=1))))


def _local_mask(atoms: Atoms, centers: np.ndarray, radius: float = 5.0) -> np.ndarray:
    fractional = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=np.float64)
    delta = fractional[:, None, :] - centers[None, :, :]
    delta -= np.round(delta)
    distances = np.linalg.norm(delta @ atoms.cell.array, axis=2)
    return np.asarray(distances.min(axis=1) <= float(radius), dtype=bool)


def _selection_diversity(
    umlip_dir: Path,
    case: dict[str, str],
    selected_atoms: Atoms,
) -> dict[str, Any] | None:
    family = case["defect_family"]
    config_id = case["defect_config_id"]
    candidate_root = umlip_dir / family / config_id
    summary_path = next(
        (
            path
            for path in (
                candidate_root / "candidate_summary.json",
                Path(read_json(candidate_root / case["candidate_id"] / "status.json")[
                    "source_candidate_structure"
                ]).parent
                / "candidate_summary.json",
            )
            if path.is_file()
        ),
        None,
    )
    if summary_path is None:
        return None
    centers = np.asarray(
        [
            center["fractional_coordinates"]
            for center in read_json(summary_path)["defect_centers"]
        ],
        dtype=np.float64,
    )
    endpoints: list[tuple[str, Atoms]] = []
    for status_path in sorted(candidate_root.glob("*/status.json")):
        status = read_json(status_path)
        if status.get("status") == "completed" and status.get("converged") is True:
            endpoints.append(
                (
                    str(status["candidate_id"]),
                    read_single_structure(Path(status["relaxed_structure"])),
                )
            )
    if not endpoints:
        return None
    unperturbed = next((atoms for name, atoms in endpoints if name == "unperturbed"), None)
    if unperturbed is None:
        return None
    mask = _local_mask(selected_atoms, centers)
    other_rmsds = [
        _mic_rmsd(selected_atoms, atoms, mask)
        for name, atoms in endpoints
        if name != case["candidate_id"]
    ]
    return {
        "metric": "ordered-atom MIC RMSD within 5.0 A of any defect center",
        "local_atom_count": int(mask.sum()),
        "local_rmsd_to_unperturbed_angstrom": _mic_rmsd(
            selected_atoms, unperturbed, mask
        ),
        "all_atom_rmsd_to_unperturbed_angstrom": _mic_rmsd(
            selected_atoms, unperturbed
        ),
        "nearest_other_endpoint_local_rmsd_angstrom": min(other_rmsds),
        "mean_other_endpoint_local_rmsd_angstrom": float(np.mean(other_rmsds)),
        "converged_endpoint_count_in_configuration": len(endpoints),
    }


def _prepare_qe_input(
    atoms: Atoms,
    path: Path,
    state_dir: Path,
    pseudo_dir: Path,
    prefix: str,
    *,
    kpts: tuple[int, int, int],
    ecutwfc_ry: float,
    ecutrho_ry: float,
    force_max: float,
    max_steps: int,
) -> None:
    input_data = precise_qe_input_data(
        ecutwfc_ry=ecutwfc_ry, ecutrho_ry=ecutrho_ry
    )
    input_data["control"].update(
        calculation="relax",
        restart_mode="from_scratch",
        prefix=prefix,
        outdir=state_dir.as_posix(),
        pseudo_dir=pseudo_dir.as_posix(),
        tprnfor=True,
        tstress=True,
        nstep=int(max_steps),
        forc_conv_thr=float(force_max) / (Rydberg / Bohr),
    )
    input_data["ions"] = {"ion_dynamics": "bfgs"}
    ase_write(
        path,
        atoms,
        format="espresso-in",
        input_data=input_data,
        pseudopotentials=QE_PSEUDOPOTENTIALS,
        kpts=kpts,
    )
    rewrite_qe_geometry_float64(path, atoms)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    umlip_dir = args.umlip_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pseudo_dir = args.pseudo_dir.expanduser().resolve()
    manifest_path = output_dir / "experiment_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --overwrite.")
    if not umlip_dir.is_dir():
        raise FileNotFoundError(umlip_dir)
    for filename in QE_PSEUDOPOTENTIALS.values():
        if not (pseudo_dir / filename).is_file():
            raise FileNotFoundError(pseudo_dir / filename)

    selected: list[dict[str, Any]] = []
    for raw_case in SELECTED_CASES:
        case = dict(raw_case)
        status_path = (
            umlip_dir
            / case["defect_family"]
            / case["defect_config_id"]
            / case["candidate_id"]
            / "status.json"
        )
        status = read_json(status_path)
        if status.get("status") != "completed" or status.get("converged") is not True:
            raise ValueError(f"Selected uMLIP candidate is not converged: {status_path}")
        endpoint_path = Path(status["relaxed_structure"]).resolve()
        original_path = Path(status["source_candidate_structure"]).resolve()
        endpoint = read_single_structure(endpoint_path)
        original = read_single_structure(original_path)
        candidate_summary_path = original_path.parent / "candidate_summary.json"
        candidate_summary = read_json(candidate_summary_path)
        if endpoint.get_chemical_symbols() != original.get_chemical_symbols():
            raise ValueError(f"Species/order changed for {case['case_id']}.")
        if not np.array_equal(endpoint.cell.array, original.cell.array):
            raise ValueError(f"Cell changed for {case['case_id']}.")
        case.update(
            {
                "status_manifest": status_path.as_posix(),
                "umlip_endpoint": endpoint_path.as_posix(),
                "umlip_endpoint_file_sha256": file_sha256(endpoint_path),
                "umlip_endpoint_geometry_sha256": geometry_sha256(endpoint),
                "pre_umlip_candidate_structure": original_path.as_posix(),
                "pre_umlip_candidate_file_sha256": file_sha256(original_path),
                "pre_umlip_candidate_geometry_sha256": geometry_sha256(original),
                "candidate_summary": candidate_summary_path.as_posix(),
                "defect_centers": candidate_summary["defect_centers"],
                "formula_counts": formula_counts(endpoint),
                "natoms": len(endpoint),
                "umlip": {
                    "initial": status["initial"],
                    "final": status["final"],
                    "optimizer_steps": status["optimizer_steps"],
                    "final_optimizer": status.get("final_optimizer", "FIRE2"),
                    "fallback_used": status.get("fallback_used", False),
                    "model": status["model"],
                },
                "selection_diversity": _selection_diversity(
                    umlip_dir, case, endpoint
                ),
            }
        )
        selected.append(case)

    task_specs = [
        (resource, case)
        for resource in ("b200", "a100", "a40")
        for case in selected
    ]

    tasks: list[dict[str, Any]] = []
    for task_index, (resource, case) in enumerate(task_specs):
        source_path = Path(case["umlip_endpoint"])
        atoms = read_single_structure(source_path)
        task_id = f"{task_index:02d}_{case['case_id']}_umlip_endpoint_{resource}"
        task_dir = output_dir / "tasks" / task_id
        input_path = task_dir / "input.extxyz"
        pwi_path = task_dir / "relax.pwi"
        state_dir = task_dir / "qe_state"
        task_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        write_extxyz_float64(input_path, atoms)
        _prepare_qe_input(
            atoms,
            pwi_path,
            state_dir,
            pseudo_dir,
            prefix=f"diag_{task_index:02d}_{case['case_id']}",
            kpts=tuple(args.kpts),
            ecutwfc_ry=args.ecutwfc,
            ecutrho_ry=args.ecutrho,
            force_max=args.fmax,
            max_steps=args.max_steps,
        )
        tasks.append(
            {
                "task_index": task_index,
                "task_id": task_id,
                "case_id": case["case_id"],
                "role": case["role"],
                "defect_family": case["defect_family"],
                "defect_config_id": case["defect_config_id"],
                "candidate_id": case["candidate_id"],
                "source_kind": "umlip_endpoint",
                "resource_profile": resource,
                "source_structure": source_path.as_posix(),
                "source_file_sha256": file_sha256(source_path),
                "source_geometry_sha256": geometry_sha256(atoms),
                "task_dir": task_dir.as_posix(),
                "input_structure": input_path.as_posix(),
                "qe_input": pwi_path.as_posix(),
                "qe_input_sha256": file_sha256(pwi_path),
                "qe_output": (task_dir / "relax.pwo").as_posix(),
                "qe_stderr": (task_dir / "relax.err").as_posix(),
                "timing_events": (task_dir / "timing_events.jsonl").as_posix(),
                "qe_state_dir": state_dir.as_posix(),
                "status": (task_dir / "status.json").as_posix(),
            }
        )

    manifest = {
        "artifact_type": "bi2te3_matched_qe_relaxation_diagnostic",
        "execution_status": "prepared_not_submitted",
        "prepared_utc": utc_now(),
        "umlip_root": umlip_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "selected_cases": selected,
        "task_count": len(tasks),
        "task_design": {
            "umlip_endpoints": {"b200": 4, "a100": 4, "a40": 4},
            "direct_dft_controls": 0,
            "pre_umlip_candidates_run_with_dft": False,
        },
        "protocol": {
            "calculation": "relax",
            "ion_dynamics": "bfgs",
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
            "force_max_ev_per_angstrom": args.fmax,
            "max_ionic_steps": args.max_steps,
            "geometry_dtype": "float64",
            "qe_geometry_float_significant_digits": 17,
            "timing_instrumentation": {
                "live_events": [
                    "scf_start",
                    "electronic_converged",
                    "total_force",
                    "bfgs_converged",
                ],
                "qe_footer_routines": [
                    "init_run",
                    "electrons",
                    "forces",
                    "stress",
                    "PWSCF",
                ],
            },
        },
        "tasks": tasks,
    }
    write_json(manifest_path, manifest)
    return manifest


def _task_from_manifest(manifest: dict[str, Any], task_index: int) -> dict[str, Any]:
    matches = [task for task in manifest["tasks"] if task["task_index"] == task_index]
    if len(matches) != 1:
        raise ValueError(f"Expected one task with index {task_index}, found {len(matches)}.")
    return matches[0]


def _slurm_timestamps(job_id: str | None) -> dict[str, Any]:
    if not job_id or shutil.which("scontrol") is None:
        return {}
    completed = subprocess.run(
        ["scontrol", "show", "job", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = completed.stdout.strip()
    values = dict(re.findall(r"(SubmitTime|EligibleTime|StartTime|EndTime)=([^ ]+)", raw))
    return {"scontrol_at_task_start": raw, **values}


def _frames_with_results(path: Path) -> list[Atoms]:
    frames = ase_read(path, index=":", format="espresso-out")
    if not isinstance(frames, list):
        frames = [frames]
    usable: list[Atoms] = []
    for frame in frames:
        try:
            energy = float(frame.get_potential_energy())
            forces = np.asarray(frame.get_forces(), dtype=np.float64)
        except Exception:
            continue
        if np.isfinite(energy) and forces.shape == (len(frame), 3) and np.isfinite(forces).all():
            usable.append(frame)
    if not usable:
        raise ValueError(f"No frames with energy and forces were readable from {path}.")
    return usable


def _qe_time_to_seconds(value: str) -> float:
    match = re.fullmatch(
        r"(?:(?P<days>\d+)d)?(?:(?P<hours>\d+)h)?"
        r"(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
        value.strip(),
    )
    if match is None or not any(match.groupdict().values()):
        raise ValueError(f"Unrecognized QE duration: {value!r}")
    parts = {key: float(raw or 0.0) for key, raw in match.groupdict().items()}
    return (
        parts["days"] * 86400.0
        + parts["hours"] * 3600.0
        + parts["minutes"] * 60.0
        + parts["seconds"]
    )


def _parse_qe_timers(text: str) -> dict[str, Any]:
    routines: dict[str, Any] = {}
    for name in ("init_run", "electrons", "forces", "stress"):
        matches = re.findall(
            rf"^\s*{re.escape(name)}\s*:\s*(\S+)\s+CPU\s+(\S+)\s+WALL\s+"
            r"\(\s*(\d+)\s+calls\)",
            text,
            re.MULTILINE,
        )
        if not matches:
            continue
        cpu, wall, calls_text = matches[-1]
        calls = int(calls_text)
        wall_seconds = _qe_time_to_seconds(wall)
        routines[name] = {
            "cpu_seconds": _qe_time_to_seconds(cpu),
            "wall_seconds": wall_seconds,
            "calls": calls,
            "wall_seconds_per_call": wall_seconds / calls if calls else None,
        }
    total_matches = re.findall(
        r"^\s*PWSCF\s*:\s*(\S+)\s+CPU\s+(\S+)\s+WALL\s*$",
        text,
        re.MULTILINE,
    )
    return {
        "routines": routines,
        "pwscf_total_cpu_seconds": (
            _qe_time_to_seconds(total_matches[-1][0]) if total_matches else None
        ),
        "pwscf_total_wall_seconds": (
            _qe_time_to_seconds(total_matches[-1][1]) if total_matches else None
        ),
    }


def _read_live_timing(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"timing_events": [], "evaluations": []}
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    evaluations: dict[int, dict[str, Any]] = {}
    for event in events:
        index = event.get("singlepoint_index")
        if index is None:
            continue
        record = evaluations.setdefault(int(index), {"singlepoint_index": int(index)})
        event_type = str(event["event"])
        record[f"{event_type}_seconds_since_qe_start"] = event[
            "seconds_since_qe_start"
        ]
        if "electronic_iterations" in event:
            record["electronic_iterations"] = event["electronic_iterations"]
    for record in evaluations.values():
        start = record.get("scf_start_seconds_since_qe_start")
        convergence = record.get("electronic_converged_seconds_since_qe_start")
        force = record.get("total_force_seconds_since_qe_start")
        if start is not None and convergence is not None:
            record["observed_scf_solve_wall_seconds"] = convergence - start
        if start is not None and force is not None:
            record["observed_scf_through_force_wall_seconds"] = force - start
    return {
        "timing_events": events,
        "evaluations": [evaluations[index] for index in sorted(evaluations)],
    }


def _parse_qe_result(task: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    input_atoms = read_single_structure(Path(task["input_structure"]))
    output_path = Path(task["qe_output"])
    text = output_path.read_text(encoding="utf-8", errors="ignore")
    frames = _frames_with_results(output_path)
    final = frames[-1]
    energy = float(final.get_potential_energy())
    forces = np.asarray(final.get_forces(), dtype=np.float64)
    force_norms = np.linalg.norm(forces, axis=1)
    final_fmax = float(force_norms.max())
    symbols_unchanged = final.get_chemical_symbols() == input_atoms.get_chemical_symbols()
    cell_unchanged = bool(
        np.allclose(final.cell.array, input_atoms.cell.array, atol=1.0e-6, rtol=0.0)
    )
    convergence_iterations = [
        int(value)
        for value in re.findall(
            r"convergence has been achieved in\s+(\d+)\s+iterations", text, re.I
        )
    ]
    bfgs_match = re.search(
        r"bfgs converged in\s+(\d+)\s+scf cycles and\s+(\d+)\s+bfgs steps",
        text,
        re.I,
    )
    scf_start_count = len(
        re.findall(
            r"^\s*Self-consistent Calculation\s*$", text, re.MULTILINE | re.I
        )
    )
    force_block_count = len(re.findall(r"Forces acting on atoms", text, re.I))
    reported_scf_cycles = int(bfgs_match.group(1)) if bfgs_match else None
    singlepoint_count = reported_scf_cycles or force_block_count or len(frames)
    qe_timers = _parse_qe_timers(text)
    task_dir = Path(task["task_dir"])
    trajectory_path = task_dir / "trajectory.extxyz"
    relaxed_path = task_dir / "relaxed.extxyz"
    ase_write(trajectory_path, frames, format="extxyz")
    ase_write(relaxed_path, final, format="extxyz")
    qe_component_target = float(
        protocol.get(
            "qe_force_component_max_ev_per_angstrom",
            protocol["force_max_ev_per_angstrom"],
        )
    )
    target = float(
        protocol.get(
            "acceptance_max_atom_force_ev_per_angstrom",
            protocol["force_max_ev_per_angstrom"],
        )
    )
    job_done = "JOB DONE" in text
    ionic_converged = (
        "bfgs converged in" in text.lower()
        or "end of bfgs geometry optimization" in text.lower()
    )
    scientifically_converged = bool(
        job_done
        and ionic_converged
        and final_fmax <= target
        and symbols_unchanged
        and cell_unchanged
    )
    return {
        "job_done": job_done,
        "ionic_converged": ionic_converged,
        "scientifically_converged": scientifically_converged,
        "final_energy_ev": energy,
        "final_max_force_ev_per_angstrom": final_fmax,
        "final_mean_force_ev_per_angstrom": float(force_norms.mean()),
        "qe_force_component_target_ev_per_angstrom": qe_component_target,
        "target_fmax_ev_per_angstrom": target,
        "qe_force_evaluation_count": len(frames),
        "ionic_steps_estimate": max(len(frames) - 1, 0),
        "singlepoint_evaluation_count": singlepoint_count,
        "scf_start_marker_count": scf_start_count,
        "force_block_count": force_block_count,
        "reported_scf_cycles": reported_scf_cycles,
        "reported_bfgs_steps": int(bfgs_match.group(2)) if bfgs_match else None,
        "electronic_iteration_counts": convergence_iterations,
        "electronic_iteration_sum": sum(convergence_iterations),
        "qe_footer_timing": qe_timers,
        "live_timing": _read_live_timing(Path(task["timing_events"])),
        "cell_unchanged": cell_unchanged,
        "maximum_cell_component_change_angstrom": float(
            np.max(np.abs(final.cell.array - input_atoms.cell.array))
        ),
        "species_and_order_unchanged": symbols_unchanged,
        "trajectory": trajectory_path.as_posix(),
        "trajectory_frame_count": len(frames),
        "relaxed_structure": relaxed_path.as_posix(),
        "relaxed_geometry_sha256": geometry_sha256(final),
    }


def _human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "not available"
    seconds = float(seconds)
    hours, remainder = divmod(seconds, 3600.0)
    minutes, remainder = divmod(remainder, 60.0)
    if hours >= 1.0:
        return f"{int(hours)}h {int(minutes):02d}m {remainder:05.2f}s"
    if minutes >= 1.0:
        return f"{int(minutes)}m {remainder:05.2f}s"
    return f"{remainder:.2f}s"


def _human_distribution(values: Sequence[float], *, duration: bool = False) -> str:
    numbers = np.asarray(values, dtype=np.float64)
    if not numbers.size:
        return "not available"
    formatter = _human_duration if duration else lambda value: f"{value:.2f}"
    return (
        f"mean {formatter(float(numbers.mean()))}; "
        f"median {formatter(float(np.median(numbers)))}; "
        f"range {formatter(float(numbers.min()))} to "
        f"{formatter(float(numbers.max()))}"
    )


def _print_task_statistics(status: dict[str, Any]) -> None:
    runtime = status.get("runtime", {})
    full_wall = runtime.get("qe_wall_time_seconds")
    singlepoints = status.get("singlepoint_evaluation_count")
    if singlepoints is None:
        live_path = status.get("timing_events") or runtime.get("timing_events")
        live = _read_live_timing(Path(live_path)) if live_path else {"evaluations": []}
        singlepoints = len(live["evaluations"])
    else:
        live = status.get("live_timing", {"evaluations": []})
    singlepoints = int(singlepoints or 0)
    effective = float(full_wall) / singlepoints if full_wall and singlepoints else None

    timers = status.get("qe_footer_timing", {})
    routines = timers.get("routines", {})
    electrons = routines.get("electrons", {})
    forces = routines.get("forces", {})
    iterations = [float(value) for value in status.get("electronic_iteration_counts", [])]
    live_evaluations = live.get("evaluations", [])
    live_scf = [
        float(record["observed_scf_solve_wall_seconds"])
        for record in live_evaluations
        if record.get("observed_scf_solve_wall_seconds") is not None
    ]
    live_through_force = [
        float(record["observed_scf_through_force_wall_seconds"])
        for record in live_evaluations
        if record.get("observed_scf_through_force_wall_seconds") is not None
    ]
    convergence = status.get("scientifically_converged")
    if convergence is True:
        result = "CONVERGED"
    elif convergence is False:
        result = "NOT CONVERGED"
    else:
        result = str(status.get("execution_status", "unknown")).upper()
    validation_status = status.get("validation_status")
    if validation_status is None:
        if convergence is True:
            validation_status = "passed"
        elif convergence is False:
            validation_status = "failed"
        else:
            validation_status = "not available"

    print()
    print("QE relaxation statistics")
    print("------------------------")
    print(f"Task:                         {status.get('task_id', 'not available')}")
    print(f"Case:                         {status.get('case_id', 'not available')}")
    print(f"Candidate:                    {status.get('candidate_id', 'not available')}")
    print(f"GPU profile:                  {status.get('resource_profile', 'not available')}")
    print(f"Allocated GPU:                {runtime.get('allocated_gpu_name', 'not available')}")
    print(f"Result:                       {result}")
    print(f"Validation status:            {validation_status}")
    print(f"Full QE process wall time:    {_human_duration(full_wall)}")
    print(
        "QE footer PWSCF wall time:   "
        f"{_human_duration(timers.get('pwscf_total_wall_seconds'))}"
    )
    print(f"Single-point evaluations:     {singlepoints}")
    print(f"BFGS ionic steps:             {status.get('reported_bfgs_steps', 'not available')}")
    print(f"Wall time / single point:     {_human_duration(effective)}")
    print(f"Electronic iterations total: {int(sum(iterations)) if iterations else 'not available'}")
    print(f"Iterations / single point:    {_human_distribution(iterations)}")
    print(
        "QE electrons wall total:    "
        f"{_human_duration(electrons.get('wall_seconds'))}"
    )
    print(
        "QE electrons wall / call:   "
        f"{_human_duration(electrons.get('wall_seconds_per_call'))}"
    )
    print(
        "QE forces wall total:        "
        f"{_human_duration(forces.get('wall_seconds'))}"
    )
    print(
        "Observed SCF solve times:    "
        f"{_human_distribution(live_scf, duration=True)}"
    )
    print(
        "Observed SCF-through-force:  "
        f"{_human_distribution(live_through_force, duration=True)}"
    )
    final_fmax = status.get("final_max_force_ev_per_angstrom")
    qe_component_target = status.get("qe_force_component_target_ev_per_angstrom")
    acceptance_target = status.get("target_fmax_ev_per_angstrom")
    print(
        "QE component force target:    "
        + (
            f"{float(qe_component_target):.6g} eV/Angstrom"
            if qe_component_target is not None
            else "not available"
        )
    )
    print(
        "Accepted max atomic force:    "
        + (
            f"{float(acceptance_target):.6g} eV/Angstrom"
            if acceptance_target is not None
            else "not available"
        )
    )
    print(
        "Final maximum force:         "
        + (
            f"{float(final_fmax):.6g} eV/Angstrom"
            if final_fmax is not None
            else "not available"
        )
    )
    print(f"Detailed timing events:       {status.get('timing_events', runtime.get('timing_events', 'not available'))}")
    print(f"Status file:                  {status.get('status', 'not available')}")
    if status.get("failure_reason"):
        print(f"Failure:                      {status['failure_reason']}")
    print("------------------------")
    print()


def run_task(
    args: argparse.Namespace,
    result_parser: Callable[
        [dict[str, Any], dict[str, Any]], dict[str, Any]
    ] = _parse_qe_result,
) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = read_json(manifest_path)
    task = _task_from_manifest(manifest, args.task_index)
    requested_profile = str(task["resource_profile"])
    allowed_gpus = requested_profile.split("_or_")
    if args.gpu_type not in allowed_gpus:
        raise ValueError(
            f"Task {args.task_index} requires {requested_profile}, not {args.gpu_type}."
        )
    actual_gpu = str(args.gpu_type)
    qe_executable = args.qe_executable.expanduser().resolve()
    if not qe_executable.is_file() or not os.access(qe_executable, os.X_OK):
        raise FileNotFoundError(qe_executable)
    if actual_gpu != "cpu" and f"qe-7.5-{actual_gpu}" not in qe_executable.as_posix():
        raise ValueError(
            f"QE executable does not visibly match {actual_gpu}: {qe_executable}"
        )

    status_path = Path(task["status"])
    if status_path.is_file() and not args.overwrite:
        previous = read_json(status_path)
        if previous.get("scientifically_converged") is True:
            print(f"[SKIPPED] Task {args.task_index} is already converged.")
            _print_task_statistics(previous)
            return previous
        raise FileExistsError(f"Refusing to overwrite {status_path}; pass --overwrite.")

    job_id = os.environ.get("SLURM_JOB_ID")
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    slurm_query_id = (
        f"{array_job_id}_{array_task_id}" if array_job_id and array_task_id else job_id
    )
    runtime = {
        "slurm_job_id": job_id,
        "slurm_array_job_id": array_job_id,
        "slurm_array_task_id": array_task_id,
        "slurm_query_job_id": slurm_query_id,
        "requested_gpu_profile": requested_profile,
        "allocated_gpu_type": actual_gpu,
        "allocated_gpu_name": args.allocated_gpu_name,
        "qe_executable": qe_executable.as_posix(),
        "qe_executable_sha256": file_sha256(qe_executable),
        "task_started_utc": utc_now(),
        "task_started_unix": time.time(),
        **_slurm_timestamps(slurm_query_id),
    }
    status = {
        **task,
        "artifact_type": task.get(
            "task_artifact_type", "matched_qe_diagnostic_task"
        ),
        "resource_profile_request": requested_profile,
        "resource_profile": actual_gpu,
        "execution_status": "running",
        "runtime": runtime,
    }
    write_json_atomic(status_path, status)

    if actual_gpu == "cpu":
        # The cluster's CPU build uses Intel MPI, whose mpirun bootstraps
        # through Slurm; one rank per allocated task.
        launcher = ["mpirun", "-np", os.environ.get("SLURM_NTASKS", "1")]
    else:
        # The step inherits the job's GPUs.  Requesting them again per task
        # makes Slurm bind to GPU-local cores, and the step fails ("Unable to
        # satisfy cpu bind request") whenever the job was allocated other
        # cores.  One MPI rank per k-point pool; QE maps ranks to the GPUs.
        launcher = [
            "srun",
            f"--ntasks={int(args.npool)}",
            "--cpus-per-task",
            os.environ.get("SLURM_CPUS_PER_TASK", "1"),
            "--cpu-bind=none",
        ]
    command = [
        *launcher,
        qe_executable.as_posix(),
        "-nk",
        str(args.npool),
        "-nd",
        str(args.ndiag),
        "-in",
        task["qe_input"],
    ]
    environment = os.environ.copy()
    output_path = Path(task["qe_output"])
    stderr_path = Path(task["qe_stderr"])
    timing_path = Path(task["timing_events"])
    start = time.perf_counter()
    try:
        singlepoint_index = 0
        with (
            output_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
            timing_path.open("w", encoding="utf-8") as timing,
        ):
            process = subprocess.Popen(
                command,
                cwd=task["task_dir"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("QE stdout pipe was not created.")
            for line in process.stdout:
                stdout.write(line)
                # Flush every line so progress is visible while a long SCF runs.
                stdout.flush()
                event: dict[str, Any] | None = None
                if re.search(
                    r"^\s*Self-consistent Calculation\s*$", line, re.I
                ):
                    singlepoint_index += 1
                    event = {
                        "event": "scf_start",
                        "singlepoint_index": singlepoint_index,
                    }
                else:
                    convergence = re.search(
                        r"convergence has been achieved in\s+(\d+)\s+iterations",
                        line,
                        re.I,
                    )
                    if convergence:
                        event = {
                            "event": "electronic_converged",
                            "singlepoint_index": singlepoint_index,
                            "electronic_iterations": int(convergence.group(1)),
                        }
                    elif re.search(r"^\s*Total force\s*=", line, re.I):
                        event = {
                            "event": "total_force",
                            "singlepoint_index": singlepoint_index,
                        }
                    elif re.search(r"bfgs converged in", line, re.I):
                        event = {
                            "event": "bfgs_converged",
                            "singlepoint_index": singlepoint_index,
                        }
                if event is not None:
                    event.update(
                        {
                            "seconds_since_qe_start": time.perf_counter() - start,
                            "observed_utc": utc_now(),
                        }
                    )
                    timing.write(json.dumps(event, sort_keys=True) + "\n")
                    timing.flush()
            return_code = process.wait()
        wall_time = time.perf_counter() - start
        status["runtime"].update(
            {
                "qe_command": command,
                "qe_return_code": return_code,
                "qe_wall_time_seconds": wall_time,
                "timing_events": timing_path.as_posix(),
                "task_finished_utc": utc_now(),
                "task_finished_unix": time.time(),
            }
        )
        status["execution_status"] = "qe_exited"
        if return_code != 0:
            raise RuntimeError(f"QE returned exit code {return_code}.")
        status.update(result_parser(task, manifest["protocol"]))
        if not status["scientifically_converged"]:
            status["validation_status"] = "failed"
            raise RuntimeError(
                "QE exited but did not pass the task's JOB DONE, convergence, force, "
                "cell, and species checks."
            )
        status["execution_status"] = "completed"
        status["validation_status"] = "passed"
        write_json_atomic(status_path, status)
        _print_task_statistics(status)
        return status
    except Exception as error:
        status["execution_status"] = "failed"
        status["failure_reason"] = f"{type(error).__name__}: {error}"
        status["runtime"].setdefault("qe_wall_time_seconds", time.perf_counter() - start)
        status["runtime"].setdefault("task_finished_utc", utc_now())
        status["runtime"].setdefault("task_finished_unix", time.time())
        if output_path.is_file():
            try:
                status.update(result_parser(task, manifest["protocol"]))
            except Exception as parse_error:
                status["parse_failure_reason"] = f"{type(parse_error).__name__}: {parse_error}"
        write_json_atomic(status_path, status)
        _print_task_statistics(status)
        raise


def _sacct_rows(job_ids: Sequence[str]) -> list[dict[str, str]]:
    if not job_ids or shutil.which("sacct") is None:
        return []
    fields = [
        "JobIDRaw",
        "JobName",
        "State",
        "Submit",
        "Eligible",
        "Start",
        "End",
        "ElapsedRaw",
        "TimelimitRaw",
        "AllocTRES",
    ]
    completed = subprocess.run(
        [
            "sacct",
            "-j",
            ",".join(sorted(set(job_ids))),
            f"--format={','.join(fields)}",
            "-n",
            "-P",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        values = line.split("|")
        if len(values) >= len(fields):
            row = dict(zip(fields, values))
            if "." not in row["JobIDRaw"]:
                rows.append(row)
    return rows


def _slurm_duration_seconds(start: str, end: str) -> float | None:
    if not start or not end or start in {"Unknown", "None"} or end in {"Unknown", "None"}:
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except ValueError:
        return None


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    numbers = np.asarray(values, dtype=np.float64)
    if not numbers.size:
        return {"count": 0}
    return {
        "count": int(numbers.size),
        "minimum": float(numbers.min()),
        "median": float(np.median(numbers)),
        "mean": float(numbers.mean()),
        "maximum": float(numbers.max()),
        "total": float(numbers.sum()),
    }


def _timing_group(statuses: Sequence[dict[str, Any]]) -> dict[str, Any]:
    job_wall = [float(status["runtime"]["qe_wall_time_seconds"]) for status in statuses]
    singlepoint_counts = [int(status["singlepoint_evaluation_count"]) for status in statuses]
    effective_per_singlepoint = [
        wall / count
        for wall, count in zip(job_wall, singlepoint_counts)
        if count > 0
    ]
    ionic_steps = [
        int(
            status["reported_bfgs_steps"]
            if status.get("reported_bfgs_steps") is not None
            else status["ionic_steps_estimate"]
        )
        for status in statuses
    ]
    electronic_iterations = [
        int(value)
        for status in statuses
        for value in status.get("electronic_iteration_counts", [])
    ]
    live_scf = [
        float(evaluation["observed_scf_solve_wall_seconds"])
        for status in statuses
        for evaluation in status.get("live_timing", {}).get("evaluations", [])
        if evaluation.get("observed_scf_solve_wall_seconds") is not None
    ]
    live_through_force = [
        float(evaluation["observed_scf_through_force_wall_seconds"])
        for status in statuses
        for evaluation in status.get("live_timing", {}).get("evaluations", [])
        if evaluation.get("observed_scf_through_force_wall_seconds") is not None
    ]
    electron_records = [
        status.get("qe_footer_timing", {}).get("routines", {}).get("electrons")
        for status in statuses
    ]
    electron_records = [record for record in electron_records if record is not None]
    electron_wall = [float(record["wall_seconds"]) for record in electron_records]
    electron_calls = [int(record["calls"]) for record in electron_records]
    total_singlepoints = sum(singlepoint_counts)
    total_electron_calls = sum(electron_calls)
    return {
        "job_count": len(statuses),
        "full_relaxation_wall_seconds": _numeric_summary(job_wall),
        "total_gpu_hours": sum(job_wall) / 3600.0,
        "singlepoint_evaluations_per_job": _numeric_summary(singlepoint_counts),
        "total_singlepoint_evaluations": total_singlepoints,
        "effective_job_wall_seconds_per_singlepoint": _numeric_summary(
            effective_per_singlepoint
        ),
        "weighted_job_wall_seconds_per_singlepoint": (
            sum(job_wall) / total_singlepoints if total_singlepoints else None
        ),
        "bfgs_ionic_steps_per_job": _numeric_summary(ionic_steps),
        "electronic_iterations_per_singlepoint": _numeric_summary(
            electronic_iterations
        ),
        "qe_electrons_wall_seconds": _numeric_summary(electron_wall),
        "qe_electrons_call_count": sum(electron_calls),
        "weighted_qe_electrons_wall_seconds_per_call": (
            sum(electron_wall) / total_electron_calls
            if total_electron_calls
            else None
        ),
        "live_observed_scf_solve_wall_seconds": _numeric_summary(live_scf),
        "live_observed_scf_through_force_wall_seconds": _numeric_summary(
            live_through_force
        ),
    }


def _timing_summary(statuses: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "definition": {
            "singlepoint_evaluation": (
                "one self-consistent QE energy/force evaluation at a fixed ionic geometry"
            ),
            "full_relaxation_wall_seconds": "runner wall time from pw.x launch to exit",
            "effective_job_wall_seconds_per_singlepoint": (
                "full pw.x wall time divided by the number of QE SCF/force evaluations"
            ),
            "qe_electrons_wall_seconds_per_call": (
                "authoritative aggregate electrons timer from the QE footer divided by calls"
            ),
            "live_observed_scf_solve_wall_seconds": (
                "wall time between observed Self-consistent Calculation and convergence lines"
            ),
        },
        "overall": _timing_group(statuses),
        "by_resource_profile": {
            profile: _timing_group(
                [status for status in statuses if status["resource_profile"] == profile]
            )
            for profile in ("b200", "a100", "a40")
        },
        "jobs": [
            {
                "task_index": status["task_index"],
                "case_id": status["case_id"],
                "resource_profile": status["resource_profile"],
                "full_relaxation_wall_seconds": status["runtime"][
                    "qe_wall_time_seconds"
                ],
                "singlepoint_evaluation_count": status[
                    "singlepoint_evaluation_count"
                ],
                "reported_bfgs_steps": status.get("reported_bfgs_steps"),
                "electronic_iteration_counts": status.get(
                    "electronic_iteration_counts", []
                ),
                "qe_footer_timing": status.get("qe_footer_timing", {}),
                "live_timing": status.get("live_timing", {}),
            }
            for status in sorted(statuses, key=lambda status: status["task_index"])
        ],
    }


def _recover_duration_parser_failure(
    status: dict[str, Any], task: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    failure = str(status.get("failure_reason", ""))
    if "Unrecognized QE duration" not in failure:
        return status
    output_path = Path(task["qe_output"])
    if not output_path.is_file() or "JOB DONE" not in output_path.read_text(
        encoding="utf-8", errors="ignore"
    ):
        return status
    status.update(_parse_qe_result(task, protocol))
    status["execution_status"] = "completed"
    status["validation_status"] = (
        "passed"
        if status["scientifically_converged"]
        else "qe_converged_external_fmax_threshold_missed"
    )
    status["recovered_postprocessing_utc"] = utc_now()
    status["recovered_from_parser_failure"] = failure
    status.pop("failure_reason", None)
    status.pop("parse_failure_reason", None)
    write_json_atomic(Path(task["status"]), status)
    return status


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.manifest.expanduser().resolve())
    statuses: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        path = Path(task["status"])
        if path.is_file():
            status = _recover_duration_parser_failure(
                read_json(path), task, manifest["protocol"]
            )
            statuses.append(status)
    job_ids = [
        str(status.get("runtime", {}).get("slurm_query_job_id"))
        for status in statuses
        if status.get("runtime", {}).get("slurm_query_job_id")
    ]
    accounting = _sacct_rows(job_ids)
    for row in accounting:
        row["submit_to_eligible_seconds"] = _slurm_duration_seconds(
            row.get("Submit", ""), row.get("Eligible", "")
        )
        row["eligible_to_start_seconds"] = _slurm_duration_seconds(
            row.get("Eligible", ""), row.get("Start", "")
        )
        row["submit_to_start_seconds"] = _slurm_duration_seconds(
            row.get("Submit", ""), row.get("Start", "")
        )
        row["start_to_end_seconds"] = _slurm_duration_seconds(
            row.get("Start", ""), row.get("End", "")
        )

    completed = [s for s in statuses if s.get("scientifically_converged") is True]
    comparisons: dict[str, Any] = {}
    for case in manifest["selected_cases"]:
        case_id = case["case_id"]
        case_statuses = [s for s in completed if s["case_id"] == case_id]
        by_key = {(s["source_kind"], s["resource_profile"]): s for s in case_statuses}
        gpu_runs = [
            by_key.get(("umlip_endpoint", resource))
            for resource in ("b200", "a100", "a40")
        ]
        gpu_runs = [run for run in gpu_runs if run is not None]
        record: dict[str, Any] = {}
        if len(gpu_runs) >= 2:
            reference = next(
                (run for run in gpu_runs if run["resource_profile"] == "b200"),
                gpu_runs[0],
            )
            reference_atoms = read_single_structure(Path(reference["relaxed_structure"]))
            centers = np.asarray(
                [
                    center["fractional_coordinates"]
                    for center in case["defect_centers"]
                ],
                dtype=np.float64,
            )
            local_mask = _local_mask(reference_atoms, centers)
            record["cross_gpu_vs_b200"] = {
                run["resource_profile"]: {
                    "wall_time_ratio_to_b200": (
                        run["runtime"]["qe_wall_time_seconds"]
                        / reference["runtime"]["qe_wall_time_seconds"]
                    ),
                    "final_energy_difference_ev": (
                        run["final_energy_ev"] - reference["final_energy_ev"]
                    ),
                    "final_all_atom_mic_rmsd_angstrom": _mic_rmsd(
                        read_single_structure(Path(run["relaxed_structure"])),
                        reference_atoms,
                    ),
                    "final_local_5A_mic_rmsd_angstrom": _mic_rmsd(
                        read_single_structure(Path(run["relaxed_structure"])),
                        reference_atoms,
                        local_mask,
                    ),
                    "ionic_steps": run["ionic_steps_estimate"],
                    "singlepoint_evaluation_count": run[
                        "singlepoint_evaluation_count"
                    ],
                    "full_relaxation_wall_seconds": run["runtime"][
                        "qe_wall_time_seconds"
                    ],
                    "effective_wall_seconds_per_singlepoint": (
                        run["runtime"]["qe_wall_time_seconds"]
                        / run["singlepoint_evaluation_count"]
                    ),
                    "electronic_iteration_sum": run["electronic_iteration_sum"],
                }
                for run in gpu_runs
            }
        comparisons[case_id] = record

    summary = {
        "artifact_type": "bi2te3_matched_qe_diagnostic_summary",
        "generated_utc": utc_now(),
        "expected_task_count": manifest["task_count"],
        "status_count": len(statuses),
        "scientifically_converged_count": len(completed),
        "failed_or_unconverged_count": len(statuses) - len(completed),
        "missing_task_count": manifest["task_count"] - len(statuses),
        "timing": _timing_summary(completed),
        "comparisons": comparisons,
        "slurm_accounting": accounting,
    }
    output_path = Path(manifest["output_dir"]) / "diagnostic_summary.json"
    write_json(output_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Freeze cases and write 12 QE inputs.")
    prepare_parser.add_argument("--umlip-dir", type=Path, default=DEFAULT_UMLIP_DIR)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--pseudo-dir", type=Path, default=DEFAULT_PSEUDO_DIR)
    prepare_parser.add_argument("--kpts", nargs=3, type=int, default=(6, 6, 6))
    prepare_parser.add_argument("--ecutwfc", type=float, default=60.0)
    prepare_parser.add_argument("--ecutrho", type=float, default=480.0)
    prepare_parser.add_argument("--fmax", type=float, default=0.01)
    prepare_parser.add_argument("--max-steps", type=int, default=100)
    prepare_parser.add_argument("--overwrite", action="store_true")

    run_parser = subparsers.add_parser("run-task", help="Run one allocated QE task.")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--task-index", type=int, required=True)
    run_parser.add_argument("--gpu-type", choices=("a40", "a100", "b200"), required=True)
    run_parser.add_argument("--allocated-gpu-name", required=True)
    run_parser.add_argument("--qe-executable", type=Path, required=True)
    run_parser.add_argument("--npool", type=int, default=1)
    run_parser.add_argument("--ndiag", type=int, default=1)
    run_parser.add_argument("--overwrite", action="store_true")

    summary_parser = subparsers.add_parser("summarize", help="Validate and compare finished runs.")
    summary_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
        print("Matched QE diagnostic preparation completed successfully.")
        print(f"task_count={result['task_count']}")
        print(f"manifest={Path(result['output_dir']) / 'experiment_manifest.json'}")
    elif args.command == "run-task":
        run_task(args)
    elif args.command == "summarize":
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
