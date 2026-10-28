#!/usr/bin/env python3
"""Prepare, run, and summarize fixed-cell QE relaxations of uMLIP endpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from relaxgen.structures.bi2te3_defect_generation import (
    QE_PSEUDOPOTENTIALS,
    file_sha256,
    formula_counts,
    geometry_sha256,
    read_single_structure,
    write_extxyz_float64,
)
from relaxgen.dft.qe_matched_diagnostic import (
    DEFAULT_PSEUDO_DIR,
    _prepare_qe_input,
    _timing_group,
    read_json,
    run_task,
    utc_now,
    write_json_atomic,
)
DEFAULT_UMLIP_DIR = Path(
    "/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/"
    "05_eqv3_tight_relaxations"
)
DEFAULT_OUTPUT_DIR = Path(
    "/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/"
    "07_qe_relaxations_tight_umlip"
)


def collect_endpoints(umlip_dir: Path, max_umlip_fmax: float) -> tuple[list[dict], dict]:
    selected = []
    excluded = Counter()
    status_paths = sorted(umlip_dir.glob("*/*/*/status.json"))
    for status_path in status_paths:
        status = read_json(status_path)
        if status.get("status") != "completed":
            excluded["not_completed"] += 1
            continue
        if status.get("converged") is not True:
            excluded["umlip_not_converged"] += 1
            continue
        final_fmax = float(status["final"]["max_force_ev_per_angstrom"])
        if final_fmax > max_umlip_fmax:
            excluded["above_umlip_fmax"] += 1
            continue
        endpoint = Path(status["relaxed_structure"]).expanduser().resolve()
        if not endpoint.is_file():
            excluded["missing_relaxed_structure"] += 1
            continue
        selected.append(
            {
                "status_path": status_path.resolve(),
                "status": status,
                "endpoint": endpoint,
                "final_fmax": final_fmax,
            }
        )
    selected.sort(
        key=lambda row: (
            row["status"]["defect_family"],
            row["status"]["defect_config_id"],
            row["status"]["candidate_id"],
        )
    )
    audit = {
        "status_file_count": len(status_paths),
        "selected_count": len(selected),
        "excluded_count": sum(excluded.values()),
        "excluded_by_reason": dict(sorted(excluded.items())),
    }
    return selected, audit


def prepare(args: argparse.Namespace) -> dict:
    umlip_dir = args.umlip_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pseudo_dir = args.pseudo_dir.expanduser().resolve()
    manifest_path = output_dir / "campaign_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --overwrite.")
    if not umlip_dir.is_dir():
        raise FileNotFoundError(umlip_dir)
    for filename in QE_PSEUDOPOTENTIALS.values():
        if not (pseudo_dir / filename).is_file():
            raise FileNotFoundError(pseudo_dir / filename)

    selected, selection_audit = collect_endpoints(umlip_dir, args.max_umlip_fmax)
    if not selected:
        raise RuntimeError("No uMLIP endpoints passed the selection criteria.")
    acceptance_fmax = (
        args.dft_fmax if args.acceptance_fmax is None else args.acceptance_fmax
    )

    tasks = []
    family_counts = Counter()
    for task_index, row in enumerate(selected):
        status = row["status"]
        atoms = read_single_structure(row["endpoint"])
        family = status["defect_family"]
        config = status["defect_config_id"]
        candidate = status["candidate_id"]
        family_counts[family] += 1

        task_dir = output_dir / "tasks" / f"{task_index:04d}"
        state_dir = task_dir / "qe_state"
        input_path = task_dir / "input.extxyz"
        qe_input = task_dir / "relax.pwi"
        task_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        write_extxyz_float64(input_path, atoms)
        _prepare_qe_input(
            atoms,
            qe_input,
            state_dir,
            pseudo_dir,
            prefix=f"bi2te3_{task_index:04d}",
            kpts=tuple(args.kpts),
            ecutwfc_ry=args.ecutwfc,
            ecutrho_ry=args.ecutrho,
            force_max=args.dft_fmax,
            max_steps=args.max_steps,
        )

        task_id = f"{task_index:04d}_{family}_{config}_{candidate}"
        tasks.append(
            {
                "task_index": task_index,
                "task_id": task_id,
                "task_artifact_type": "bi2te3_umlip_to_qe_relaxation",
                "case_id": config,
                "role": "production uMLIP endpoint refinement",
                "defect_family": family,
                "defect_config_id": config,
                "candidate_id": candidate,
                "candidate_type": status["candidate_type"],
                "source_kind": "umlip_endpoint",
                "resource_profile": "a100_or_b200",
                "source_structure": row["endpoint"].as_posix(),
                "source_file_sha256": file_sha256(row["endpoint"]),
                "source_geometry_sha256": geometry_sha256(atoms),
                "source_umlip_status": row["status_path"].as_posix(),
                "source_umlip_final_fmax_ev_per_angstrom": row["final_fmax"],
                "source_umlip_energy_ev": float(status["final"]["energy_ev"]),
                "formula_counts": formula_counts(atoms),
                "natoms": len(atoms),
                "task_dir": task_dir.as_posix(),
                "input_structure": input_path.as_posix(),
                "qe_input": qe_input.as_posix(),
                "qe_input_sha256": file_sha256(qe_input),
                "qe_output": (task_dir / "relax.pwo").as_posix(),
                "qe_stderr": (task_dir / "relax.err").as_posix(),
                "timing_events": (task_dir / "timing_events.jsonl").as_posix(),
                "qe_state_dir": state_dir.as_posix(),
                "status": (task_dir / "status.json").as_posix(),
            }
        )

    protocol = {
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
        "force_max_ev_per_angstrom": args.dft_fmax,
        "qe_force_component_max_ev_per_angstrom": args.dft_fmax,
        "acceptance_max_atom_force_ev_per_angstrom": acceptance_fmax,
        "max_ionic_steps": args.max_steps,
        "geometry_dtype": "float64",
        "qe_geometry_float_significant_digits": 17,
        "allowed_gpu_types": ["a100", "b200"],
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
    }
    manifest = {
        "artifact_type": "bi2te3_umlip_to_qe_relaxation_campaign",
        "execution_status": "prepared_not_submitted",
        "prepared_utc": utc_now(),
        "umlip_root": umlip_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "task_count": len(tasks),
        "selection": {
            "require_completed": True,
            "require_umlip_converged": True,
            "maximum_umlip_fmax_ev_per_angstrom": args.max_umlip_fmax,
            **selection_audit,
            "selected_by_defect_family": dict(sorted(family_counts.items())),
        },
        "protocol": protocol,
        "tasks": tasks,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def summarize(manifest_path: Path) -> dict:
    manifest = read_json(manifest_path.expanduser().resolve())
    statuses = []
    missing = []
    for task in manifest["tasks"]:
        status_path = Path(task["status"])
        if status_path.is_file():
            statuses.append(read_json(status_path))
        else:
            missing.append(task["task_index"])

    converged = [row for row in statuses if row.get("scientifically_converged") is True]
    failed = [row for row in statuses if row.get("execution_status") == "failed"]
    running = [row for row in statuses if row.get("execution_status") == "running"]
    by_family = defaultdict(lambda: {"expected": 0, "converged": 0, "failed": 0})
    for task in manifest["tasks"]:
        by_family[task["defect_family"]]["expected"] += 1
    for row in converged:
        by_family[row["defect_family"]]["converged"] += 1
    for row in failed:
        by_family[row["defect_family"]]["failed"] += 1

    timing_by_gpu = {}
    for gpu in ("b200", "a100"):
        rows = [row for row in converged if row.get("resource_profile") == gpu]
        timing_by_gpu[gpu] = _timing_group(rows) if rows else {"job_count": 0}

    summary = {
        "artifact_type": "bi2te3_umlip_to_qe_relaxation_summary",
        "generated_utc": utc_now(),
        "expected_task_count": manifest["task_count"],
        "status_count": len(statuses),
        "scientifically_converged_count": len(converged),
        "failed_count": len(failed),
        "running_status_count": len(running),
        "missing_status_count": len(missing),
        "by_defect_family": dict(sorted(by_family.items())),
        "timing_by_gpu": timing_by_gpu,
    }
    output_dir = Path(manifest["output_dir"])
    write_json_atomic(output_dir / "campaign_summary.json", summary)
    lines = [
        "QE relaxation campaign summary",
        "",
        f"Expected tasks:              {summary['expected_task_count']}",
        f"Status files:                {summary['status_count']}",
        f"Scientifically converged:    {summary['scientifically_converged_count']}",
        f"Failed:                      {summary['failed_count']}",
        f"Currently marked running:    {summary['running_status_count']}",
        f"Missing status:              {summary['missing_status_count']}",
        "",
        "Defect-family progress",
    ]
    for family, counts in summary["by_defect_family"].items():
        lines.append(
            f"{family}: {counts['converged']}/{counts['expected']} converged; "
            f"{counts['failed']} failed"
        )
    human_path = output_dir / "campaign_summary.txt"
    human_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(human_path.read_text(encoding="utf-8"), end="")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--umlip-dir", type=Path, default=DEFAULT_UMLIP_DIR)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--pseudo-dir", type=Path, default=DEFAULT_PSEUDO_DIR)
    prepare_parser.add_argument("--max-umlip-fmax", type=float, default=0.001)
    prepare_parser.add_argument("--kpts", nargs=3, type=int, default=(6, 6, 6))
    prepare_parser.add_argument("--ecutwfc", type=float, default=60.0)
    prepare_parser.add_argument("--ecutrho", type=float, default=480.0)
    prepare_parser.add_argument("--dft-fmax", type=float, default=0.01)
    prepare_parser.add_argument(
        "--acceptance-fmax",
        type=float,
        default=None,
        help=(
            "Maximum final per-atom force-vector norm in eV/Angstrom. "
            "Defaults to --dft-fmax."
        ),
    )
    prepare_parser.add_argument("--max-steps", type=int, default=100)
    prepare_parser.add_argument("--overwrite", action="store_true")

    run_parser = subparsers.add_parser("run-task")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--task-index", type=int, required=True)
    run_parser.add_argument("--gpu-type", choices=("a100", "b200"), required=True)
    run_parser.add_argument("--allocated-gpu-name", required=True)
    run_parser.add_argument("--qe-executable", type=Path, required=True)
    run_parser.add_argument("--npool", type=int, default=1)
    run_parser.add_argument("--ndiag", type=int, default=1)
    run_parser.add_argument("--overwrite", action="store_true")

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        if (
            args.max_umlip_fmax <= 0
            or args.dft_fmax <= 0
            or (args.acceptance_fmax is not None and args.acceptance_fmax <= 0)
            or args.max_steps <= 0
        ):
            raise ValueError("Force thresholds and --max-steps must be positive.")
        manifest = prepare(args)
        print("QE campaign preparation completed successfully.")
        print(f"selected_endpoints={manifest['task_count']}")
        print(f"excluded_endpoints={manifest['selection']['excluded_count']}")
        print(f"manifest={Path(manifest['output_dir']) / 'campaign_manifest.json'}")
    elif args.command == "run-task":
        run_task(args)
    else:
        summarize(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
