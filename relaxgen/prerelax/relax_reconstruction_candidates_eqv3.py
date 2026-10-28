#!/usr/bin/env python3
"""Fixed-cell EqV3/FIRE2 pre-relaxation for reconstruction candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.optimize.fire import FIRE
from ase.optimize.fire2 import FIRE2

from relaxgen import CKPT_DIR
from relaxgen.structures.bi2te3_defect_generation import (
    file_sha256,
    geometry_sha256,
    read_single_structure,
    write_extxyz_float64,
)
DEFAULT_CHECKPOINT = CKPT_DIR / "eqv3" / "best_by_forces_mae.pt"
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.with_name("config.yml")


class GeometrySafetyError(RuntimeError):
    """An optimizer step produced a geometry that must not be retained."""


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def configuration_summaries(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("*/*/candidate_summary.json"))
    if not paths:
        raise FileNotFoundError(
            f"No candidate summaries found under {input_dir}; expected FAMILY/CONFIG/."
        )
    return paths


def select_configuration(
    paths: Sequence[Path],
    *,
    task_index: int | None,
    config_id: str | None,
) -> tuple[int, Path]:
    if (task_index is None) == (config_id is None):
        raise ValueError("Specify exactly one of --task-index or --config-id.")
    if task_index is not None:
        if not 0 <= task_index < len(paths):
            raise IndexError(
                f"Task index {task_index} is outside 0..{len(paths) - 1}."
            )
        return task_index, paths[task_index]
    matches = [path for path in paths if path.parent.name == config_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one configuration {config_id!r}, found {len(matches)}.")
    path = matches[0]
    return paths.index(path), path


def model_provenance(
    checkpoint: Path, config_yml: Path | None, output_dir: Path
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    config_yml = config_yml.expanduser().resolve() if config_yml else None
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if config_yml is not None and not config_yml.is_file():
        raise FileNotFoundError(config_yml)

    cache_path = output_dir / "eqv3_model.json"
    stat = checkpoint.stat()
    config_hash = file_sha256(config_yml) if config_yml else None
    if cache_path.is_file():
        cached = read_json(cache_path)
        if (
            cached.get("checkpoint_path") == checkpoint.as_posix()
            and cached.get("checkpoint_size_bytes") == stat.st_size
            and cached.get("checkpoint_mtime_ns") == stat.st_mtime_ns
            and cached.get("config_yml")
            == (config_yml.as_posix() if config_yml else None)
            and cached.get("config_yml_sha256") == config_hash
        ):
            return cached

    provenance = {
        "model": "equiformer_v3",
        "checkpoint_path": checkpoint.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_size_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "config_yml": config_yml.as_posix() if config_yml else None,
        "config_yml_sha256": config_hash,
    }
    write_json_atomic(cache_path, provenance)
    return provenance


def run_signature(model: dict[str, Any], settings: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"model": model, "settings": settings}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def minimum_distance(atoms: Atoms) -> float:
    distances = atoms.get_all_distances(mic=True)
    return float(np.min(distances[np.triu_indices(len(atoms), k=1)]))


def evaluate(atoms: Atoms) -> dict[str, Any]:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise GeometrySafetyError("EqV3 returned a non-finite energy or force.")
    force_norms = np.linalg.norm(forces, axis=1)
    return {
        "energy_ev": energy,
        "forces": forces,
        "max_force_ev_per_angstrom": float(force_norms.max()),
        "mean_force_ev_per_angstrom": float(force_norms.mean()),
        "minimum_distance_angstrom": minimum_distance(atoms),
    }


def validate_geometry(
    atoms: Atoms,
    reference: Atoms,
    *,
    hard_min_distance: float,
) -> float:
    if len(atoms) != len(reference):
        raise ValueError("Relaxation changed the atom count.")
    if atoms.get_chemical_symbols() != reference.get_chemical_symbols():
        raise ValueError("Relaxation changed atom ordering or species.")
    if not np.array_equal(atoms.cell.array, reference.cell.array):
        raise ValueError("Relaxation changed the fixed cell.")
    if not np.isfinite(atoms.positions).all():
        raise GeometrySafetyError("Relaxation produced non-finite positions.")
    distance = minimum_distance(atoms)
    if not np.isfinite(distance):
        raise GeometrySafetyError("Relaxation produced a non-finite distance.")
    if distance < hard_min_distance:
        raise GeometrySafetyError(
            f"Minimum distance {distance:.6g} Å fell below the hard limit "
            f"{hard_min_distance:.6g} Å."
        )
    return distance


def append_ml_frame(
    path: Path,
    atoms: Atoms,
    evaluation: dict[str, Any],
    *,
    optimizer_name: str,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = " ".join(f"{value:.17g}" for value in atoms.cell.array.reshape(-1))
    pbc = " ".join("T" if value else "F" for value in atoms.pbc)
    lines = [
        str(len(atoms)),
        (
            f'Lattice="{cell}" Properties=species:S:1:pos:R:3:forces:R:3 '
            f'ml_energy_ev={evaluation["energy_ev"]:.17g} '
            f'ml_fmax_ev_per_angstrom={evaluation["max_force_ev_per_angstrom"]:.17g} '
            f'optimizer={optimizer_name} optimizer_step={step} pbc="{pbc}"'
        ),
    ]
    for symbol, position, force in zip(
        atoms.get_chemical_symbols(), atoms.positions, evaluation["forces"]
    ):
        values = [*position, *force]
        lines.append(f"{symbol} " + " ".join(f"{value:.17g}" for value in values))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def relax_candidate(
    candidate: dict[str, Any],
    source_path: Path,
    candidate_dir: Path,
    calculator: Any,
    model: dict[str, Any],
    settings: dict[str, Any],
    signature: str,
) -> dict[str, Any]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    status_path = candidate_dir / "status.json"
    metrics_path = candidate_dir / "metrics.json"
    relaxed_path = candidate_dir / "relaxed.extxyz"
    trajectory_path = candidate_dir / "trajectory.extxyz"
    log_path = candidate_dir / "optimizer.log"
    fallback_log_path = candidate_dir / "fallback_optimizer.log"
    input_path = candidate_dir / "input.extxyz"

    if status_path.is_file():
        previous = read_json(status_path)
        required_completed_outputs = [
            relaxed_path,
            trajectory_path,
            log_path,
            metrics_path,
            input_path,
        ]
        if previous.get("fallback_used", False):
            required_completed_outputs.append(fallback_log_path)
        if previous.get("status") == "completed" and all(
            path.is_file() for path in required_completed_outputs
        ):
            if previous.get("run_signature") != signature:
                raise ValueError(
                    f"Completed output uses different settings: {candidate_dir}. "
                    "Use a new output directory."
                )
            return {**previous, "skipped_existing": True}

    for path in (
        metrics_path,
        relaxed_path,
        trajectory_path,
        log_path,
        fallback_log_path,
        input_path,
    ):
        if path.exists():
            path.unlink()

    started = time.time()
    optimizer_attempts: list[dict[str, Any]] = []
    fallback_used = False
    fallback_reason: str | None = None
    base_status = {
        "artifact_type": "eqv3_fire2_candidate_relaxation",
        "status": "running",
        "defect_family": candidate["defect_family"],
        "defect_config_id": candidate["defect_config_id"],
        "candidate_id": candidate["candidate_id"],
        "candidate_type": candidate["candidate_type"],
        "candidate_parameters": candidate.get("parameters", {}),
        "source_candidate_structure": source_path.as_posix(),
        "source_candidate_geometry_sha256": candidate["geometry_sha256"],
        "model": model,
        "settings": settings,
        "run_signature": signature,
        "started_unix_time": started,
        "surrogate_relaxation_attempted": True,
        "surrogate_relaxed": False,
        "dft_labeled": False,
        "fallback_policy": {
            "trigger": "geometry_safety_failure",
            "optimizer": "FIRE",
            "restart_from": "last_safe_geometry",
            "target_fmax_ev_per_angstrom": settings[
                "target_fmax_ev_per_angstrom"
            ],
            "maxstep_angstrom": settings["maxstep_angstrom"],
            "downhill_check": True,
        },
    }
    write_json_atomic(status_path, base_status)

    try:
        reference = read_single_structure(source_path)
        if geometry_sha256(reference) != candidate["geometry_sha256"]:
            raise ValueError("Candidate geometry hash does not match its manifest.")
        write_extxyz_float64(input_path, reference)
        atoms = reference.copy()
        atoms.calc = calculator
        initial = evaluate(atoms)
        validate_geometry(
            atoms,
            reference,
            hard_min_distance=float(settings["hard_min_distance_angstrom"]),
        )

        trajectory_path.touch()
        trajectory_frames = 0
        last_geometry_hash: str | None = None
        minimum_distance_seen = initial["minimum_distance_angstrom"]
        warning_frame_count = 0
        last_safe_positions = atoms.positions.copy()

        def observe(optimizer: Any = None, optimizer_name: str = "FIRE2") -> None:
            nonlocal trajectory_frames, last_geometry_hash
            nonlocal minimum_distance_seen, warning_frame_count
            nonlocal last_safe_positions
            current_hash = geometry_sha256(atoms)
            if current_hash == last_geometry_hash:
                return
            current = evaluate(atoms)
            distance = validate_geometry(
                atoms,
                reference,
                hard_min_distance=float(settings["hard_min_distance_angstrom"]),
            )
            minimum_distance_seen = min(minimum_distance_seen, distance)
            if distance < float(settings["warning_min_distance_angstrom"]):
                warning_frame_count += 1
            last_safe_positions = atoms.positions.copy()
            step = int(optimizer.nsteps) if optimizer is not None else 0
            append_ml_frame(
                trajectory_path,
                atoms,
                current,
                optimizer_name=optimizer_name,
                step=step,
            )
            last_geometry_hash = current_hash
            trajectory_frames += 1

        observe(optimizer_name="FIRE2")
        primary_optimizer = FIRE2(
            atoms,
            logfile=log_path.as_posix(),
            maxstep=float(settings["maxstep_angstrom"]),
        )
        primary_optimizer.attach(
            lambda: observe(primary_optimizer, "FIRE2"), interval=1
        )
        final_optimizer: Any = primary_optimizer
        try:
            primary_optimizer.run(
                fmax=float(settings["target_fmax_ev_per_angstrom"]),
                steps=int(settings["max_steps"]),
            )
            optimizer_attempts.append(
                {
                    "optimizer": "FIRE2",
                    "status": "completed",
                    "steps": int(primary_optimizer.nsteps),
                    "log": log_path.as_posix(),
                }
            )
        except GeometrySafetyError as exc:
            fallback_used = True
            fallback_reason = f"{type(exc).__name__}: {exc}"
            optimizer_attempts.append(
                {
                    "optimizer": "FIRE2",
                    "status": "geometry_safety_failure",
                    "steps": int(primary_optimizer.nsteps),
                    "reason": fallback_reason,
                    "log": log_path.as_posix(),
                }
            )

            atoms.set_positions(last_safe_positions)
            validate_geometry(
                atoms,
                reference,
                hard_min_distance=float(settings["hard_min_distance_angstrom"]),
            )
            fallback_start_geometry_sha256 = geometry_sha256(atoms)
            fallback_optimizer = FIRE(
                atoms,
                logfile=fallback_log_path.as_posix(),
                maxstep=float(settings["maxstep_angstrom"]),
                downhill_check=True,
            )
            fallback_optimizer.attach(
                lambda: observe(fallback_optimizer, "FIRE"), interval=1
            )
            final_optimizer = fallback_optimizer
            try:
                fallback_optimizer.run(
                    fmax=float(settings["target_fmax_ev_per_angstrom"]),
                    steps=int(settings["max_steps"]),
                )
                optimizer_attempts.append(
                    {
                        "optimizer": "FIRE",
                        "status": "completed",
                        "steps": int(fallback_optimizer.nsteps),
                        "downhill_check": True,
                        "restart_geometry_sha256": (
                            fallback_start_geometry_sha256
                        ),
                        "log": fallback_log_path.as_posix(),
                    }
                )
            except Exception as fallback_exc:
                optimizer_attempts.append(
                    {
                        "optimizer": "FIRE",
                        "status": "failed",
                        "steps": int(fallback_optimizer.nsteps),
                        "downhill_check": True,
                        "reason": (
                            f"{type(fallback_exc).__name__}: {fallback_exc}"
                        ),
                        "log": fallback_log_path.as_posix(),
                    }
                )
                raise
        final = evaluate(atoms)
        validate_geometry(
            atoms,
            reference,
            hard_min_distance=float(settings["hard_min_distance_angstrom"]),
        )
        observe(
            final_optimizer,
            "FIRE" if fallback_used else "FIRE2",
        )
        primary_steps = int(primary_optimizer.nsteps)
        fallback_steps = (
            int(final_optimizer.nsteps) if fallback_used else 0
        )
        steps = primary_steps + fallback_steps
        converged = (
            final["max_force_ev_per_angstrom"]
            <= float(settings["target_fmax_ev_per_angstrom"])
        )
        write_extxyz_float64(relaxed_path, atoms)

        finished = time.time()
        metrics = {
            **base_status,
            "status": "completed",
            "finished_unix_time": finished,
            "wall_time_seconds": finished - started,
            "initial": {key: value for key, value in initial.items() if key != "forces"},
            "final": {key: value for key, value in final.items() if key != "forces"},
            "optimizer_steps": steps,
            "primary_optimizer_steps": primary_steps,
            "fallback_optimizer_steps": fallback_steps,
            "final_optimizer": "FIRE" if fallback_used else "FIRE2",
            "optimizer_attempts": optimizer_attempts,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "trajectory_frame_count": trajectory_frames,
            "converged": converged,
            "reached_max_steps": (
                int(final_optimizer.nsteps) >= int(settings["max_steps"])
                and not converged
            ),
            "minimum_distance_seen_angstrom": minimum_distance_seen,
            "minimum_distance_warning_frame_count": warning_frame_count,
            "cell_unchanged": np.array_equal(atoms.cell.array, reference.cell.array),
            "species_and_order_unchanged": (
                atoms.get_chemical_symbols() == reference.get_chemical_symbols()
            ),
            "relaxed_structure": relaxed_path.as_posix(),
            "relaxed_geometry_sha256": geometry_sha256(atoms),
            "trajectory": trajectory_path.as_posix(),
            "optimizer_log": log_path.as_posix(),
            "fallback_optimizer_log": (
                fallback_log_path.as_posix() if fallback_used else None
            ),
            "surrogate_relaxed": converged,
        }
        write_json_atomic(metrics_path, metrics)
        write_json_atomic(status_path, metrics)
        return metrics
    except Exception as exc:
        finished = time.time()
        failure = {
            **base_status,
            "status": "failed",
            "finished_unix_time": finished,
            "wall_time_seconds": finished - started,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "converged": False,
            "optimizer_attempts": optimizer_attempts,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        }
        write_json_atomic(metrics_path, failure)
        write_json_atomic(status_path, failure)
        return failure


def load_eqv3(checkpoint: Path, config_yml: Path | None, device: str, seed: int) -> Any:
    import torch

    if device.lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this process.")
    from relaxgen.prerelax.equiformer_relaxation import EquiformerRelaxer

    return EquiformerRelaxer(
        checkpoint_path=checkpoint,
        config_yml=config_yml,
        device=device,
        seed=seed,
        quiet_model_load=True,
    )


def run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = configuration_summaries(input_dir)
    task_index, summary_path = select_configuration(
        paths, task_index=args.task_index, config_id=args.config_id
    )
    config_summary = read_json(summary_path)
    candidate_catalog = Path(config_summary["candidate_catalog"])
    candidates = [
        json.loads(line) for line in candidate_catalog.read_text().splitlines()
    ]
    if args.max_candidates is not None:
        if args.max_candidates < 1:
            raise ValueError("--max-candidates must be positive.")
        candidates = candidates[: args.max_candidates]

    if args.fmax <= 0:
        raise ValueError("--fmax must be positive.")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive.")
    if args.maxstep <= 0:
        raise ValueError("--maxstep must be positive.")
    if args.hard_min_distance <= 0:
        raise ValueError("--hard-min-distance must be positive.")
    if args.warning_min_distance < args.hard_min_distance:
        raise ValueError(
            "--warning-min-distance must be greater than or equal to "
            "--hard-min-distance."
        )

    model = model_provenance(args.checkpoint, args.config_yml, output_dir)
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
    signature = run_signature(model, settings)
    family = str(config_summary["defect_family"])
    config_id = str(config_summary["defect_config_id"])
    config_output = output_dir / family / config_id

    pending = []
    statuses = []
    for candidate in candidates:
        candidate_dir = config_output / str(candidate["candidate_id"])
        status_path = candidate_dir / "status.json"
        if status_path.is_file():
            status = read_json(status_path)
            required_outputs = [
                candidate_dir / "relaxed.extxyz",
                candidate_dir / "trajectory.extxyz",
                candidate_dir / "optimizer.log",
                candidate_dir / "metrics.json",
                candidate_dir / "input.extxyz",
            ]
            if status.get("fallback_used", False):
                required_outputs.append(candidate_dir / "fallback_optimizer.log")
            if status.get("status") == "completed" and all(
                path.is_file() for path in required_outputs
            ):
                if status.get("run_signature") != signature:
                    raise ValueError(
                        f"Existing output uses different settings: {candidate_dir}."
                    )
                statuses.append({**status, "skipped_existing": True})
                continue
        pending.append(candidate)

    runner = None
    if pending:
        runner = load_eqv3(args.checkpoint, args.config_yml, args.device, args.seed)
        for candidate in pending:
            source_path = candidate_catalog.parent / f"{candidate['candidate_id']}.extxyz"
            if not source_path.is_file():
                source_path = Path(candidate["candidate_structure"])
            result = relax_candidate(
                candidate,
                source_path,
                config_output / str(candidate["candidate_id"]),
                runner.calculator,
                model,
                settings,
                signature,
            )
            statuses.append(result)
            print(
                f"[{result['status'].upper()}] {config_id}/{candidate['candidate_id']}",
                flush=True,
            )

    task_summary = {
        "artifact_type": "eqv3_fire2_configuration_relaxation",
        "task_index": task_index,
        "configuration_count_in_input": len(paths),
        "defect_family": family,
        "defect_config_id": config_id,
        "candidate_count_selected": len(candidates),
        "completed": sum(item["status"] == "completed" for item in statuses),
        "converged": sum(
            item["status"] == "completed" and item.get("converged", False)
            for item in statuses
        ),
        "failed": sum(item["status"] == "failed" for item in statuses),
        "fallback_used": sum(item.get("fallback_used", False) for item in statuses),
        "skipped_existing": sum(item.get("skipped_existing", False) for item in statuses),
        "model": model,
        "settings": settings,
        "run_signature": signature,
    }
    write_json_atomic(config_output / "configuration_relaxation_summary.json", task_summary)
    if task_summary["failed"]:
        raise RuntimeError(
            f"{task_summary['failed']} candidates failed for {config_id}; rerun the task "
            "after inspecting their status.json files."
        )
    return task_summary


def summarize(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    expected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for summary_path in configuration_summaries(input_dir):
        config = read_json(summary_path)
        catalog = Path(config["candidate_catalog"])
        for line in catalog.read_text().splitlines():
            candidate = json.loads(line)
            key = (
                str(candidate["defect_family"]),
                str(candidate["defect_config_id"]),
                str(candidate["candidate_id"]),
            )
            expected[key] = candidate

    statuses: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in output_dir.glob("*/*/*/status.json"):
        status = read_json(path)
        key = (
            str(status["defect_family"]),
            str(status["defect_config_id"]),
            str(status["candidate_id"]),
        )
        statuses[key] = status

    completed = [item for item in statuses.values() if item["status"] == "completed"]
    converged = [item for item in completed if item.get("converged", False)]
    fallback = [item for item in completed if item.get("fallback_used", False)]
    failed = [item for item in statuses.values() if item["status"] == "failed"]
    max_steps = [item for item in completed if item.get("reached_max_steps", False)]

    def distribution(field: str) -> dict[str, float | None]:
        values = np.asarray([float(item[field]) for item in completed], dtype=np.float64)
        if not values.size:
            return {"median": None, "q05": None, "q95": None}
        return {
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        }

    by_candidate_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_defect_count: dict[str, Counter[str]] = defaultdict(Counter)
    for key, candidate in expected.items():
        status = statuses.get(key, {"status": "missing"})["status"]
        by_candidate_type[str(candidate["candidate_type"])][status] += 1
        by_defect_count[str(candidate["defect_count"])][status] += 1

    summary = {
        "artifact_type": "eqv3_fire2_relaxation_summary",
        "input_candidate_root": input_dir.as_posix(),
        "output_root": output_dir.as_posix(),
        "input_candidates": len(expected),
        "status_files": len(statuses),
        "completed": len(completed),
        "converged": len(converged),
        "fallback_used": len(fallback),
        "hit_max_steps": len(max_steps),
        "failed": len(failed),
        "missing": len(set(expected) - set(statuses)),
        "optimizer_steps": distribution("optimizer_steps"),
        "wall_time_seconds": distribution("wall_time_seconds"),
        "initial_fmax_ev_per_angstrom": distribution_from_nested(
            completed, "initial", "max_force_ev_per_angstrom"
        ),
        "final_fmax_ev_per_angstrom": distribution_from_nested(
            completed, "final", "max_force_ev_per_angstrom"
        ),
        "by_candidate_type": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_candidate_type.items())
        },
        "by_defect_count": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_defect_count.items())
        },
    }
    write_json_atomic(output_dir / "relaxation_summary.json", summary)
    return summary


def distribution_from_nested(
    records: Sequence[dict[str, Any]], outer: str, inner: str
) -> dict[str, float | None]:
    values = np.asarray(
        [float(record[outer][inner]) for record in records], dtype=np.float64
    )
    if not values.size:
        return {"median": None, "q05": None, "q95": None}
    return {
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Print the stable array-task map.")
    list_parser.add_argument("--input-dir", type=Path, required=True)

    relax = subparsers.add_parser("relax", help="Relax one defect configuration.")
    relax.add_argument("--input-dir", type=Path, required=True)
    relax.add_argument("--output-dir", type=Path, required=True)
    selector = relax.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task-index", type=int)
    selector.add_argument("--config-id")
    relax.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    relax.add_argument("--config-yml", type=Path, default=DEFAULT_CONFIG)
    relax.add_argument("--device", default="cuda")
    relax.add_argument("--seed", type=int, default=42)
    relax.add_argument("--fmax", type=float, default=0.001)
    relax.add_argument("--max-steps", type=int, default=600)
    relax.add_argument("--maxstep", type=float, default=0.10)
    relax.add_argument("--warning-min-distance", type=float, default=1.8)
    relax.add_argument("--hard-min-distance", type=float, default=1.2)
    relax.add_argument("--max-candidates", type=int)

    summary = subparsers.add_parser("summarize", help="Aggregate relaxation status.")
    summary.add_argument("--input-dir", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        for index, path in enumerate(
            configuration_summaries(args.input_dir.expanduser().resolve())
        ):
            print(f"{index}\t{path.parent.parent.name}\t{path.parent.name}")
        return 0
    if args.command == "relax":
        result = run_configuration(args)
        print("Michael, EqV3/FIRE2 configuration relaxation completed.")
        for key in (
            "task_index",
            "defect_config_id",
            "completed",
            "converged",
            "failed",
            "fallback_used",
            "skipped_existing",
        ):
            print(f"{key}={result[key]}")
        return 0
    result = summarize(args.input_dir, args.output_dir)
    print("Michael, EqV3/FIRE2 relaxation summary completed.")
    for key in (
        "input_candidates",
        "completed",
        "converged",
        "fallback_used",
        "hit_max_steps",
        "failed",
        "missing",
    ):
        print(f"{key}={result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
