#!/usr/bin/env python3
"""Does uMLIP pre-relaxation narrow the DFT minima a defect can reach?

One defect configuration (VBi_001), five ShakeNBreak-style starts, three arms:

* ``dftonly``: QE relaxation straight from the start, twice per start.  Two
  identical relaxations drift apart after ~30 BFGS steps (SCF noise amplified
  on a flat landscape), so the replicate pair measures the noise floor that any
  difference between arms has to exceed.
* ``eqv3``: Bi2Te3-finetuned EqV3, then QE from the handoff frame.
* ``uma``: UMA ``uma-s-1p2`` (``omat`` task, PBE-trained; QE here is PBEsol),
  then QE from the handoff frame.

Every optimizer is ASE BFGS with ``maxstep = 0.1`` Angstrom.  The uMLIP runs go to
ML fmax 1e-3 eV/Angstrom; their handoff frame is the first one below 0.02 eV/Angstrom,
where the EqV3 handoff experiment found tighter ML relaxation stops lowering
the DFT force.  QE relaxations use the production protocol (6x6x6 k-points) to
fmax 1e-3 eV/Angstrom with ``dft_relaxation_campaign.py`` under ``qe_dispatch.py``;
the frame where each first drops below 0.01 eV/Angstrom is also analysed.

Subcommands, in order: ``prepare-starts``, ``prerelax`` (once per model, in that
model's Python environment), ``prepare-dft``, ``analyze``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CANDIDATE_DIR = Path(
    "/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/"
    "04_reconstruction_candidates_preview_no_pure_expand40/VBi/VBi_001"
)
STARTS = {
    "unperturbed": "unperturbed",
    "dimer": "center_01_Bi_000_dimer",
    "contract40": "center_01_Bi_000_contract_40pct",
    "expand40_rattle1404": "center_01_Bi_000_expand_40pct_rattle_seed_1404",
    "rattle202": "rattle_seed_202",
}
MODELS = ("eqv3", "uma")
UMA_MODEL = "uma-s-1p2"
UMA_TASK = "omat"
ML_FMAX = 1.0e-3
HANDOFF_FMAX = 0.02
DFT_FMAX = 1.0e-3
DFT_REPORT_FMAX = 0.01
RMSD_THRESHOLDS = (0.05, 0.1, 0.2)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def fmax_of(atoms) -> float:
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


def prepare_starts(args: argparse.Namespace) -> None:
    from relaxgen.structures.bi2te3_defect_generation import geometry_sha256, read_single_structure, write_extxyz_float64

    out = args.output_dir.expanduser().resolve()
    starts = {}
    for label, stem in STARTS.items():
        source = CANDIDATE_DIR / f"{stem}.extxyz"
        atoms = read_single_structure(source)
        target = out / "starts" / f"{label}.extxyz"
        write_extxyz_float64(target, atoms)
        starts[label] = {"source": source.as_posix(), "path": target.as_posix(),
                         "geometry_sha256": geometry_sha256(atoms), "natoms": len(atoms)}
    write_json(out / "starts.json", {"defect_config_id": "VBi_001", "starts": starts})
    print(f"Wrote {len(starts)} starts to {out / 'starts'}")


def load_calculator(model: str, device: str):
    if model == "eqv3":
        from relaxgen.prerelax.relax_reconstruction_candidates_eqv3 import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_eqv3

        return load_eqv3(DEFAULT_CHECKPOINT, DEFAULT_CONFIG, device, 42).calculator, {
            "checkpoint": DEFAULT_CHECKPOINT.as_posix()
        }
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(UMA_MODEL, device=device, seed=42)
    return FAIRChemCalculator(predictor, task_name=UMA_TASK), {"model": UMA_MODEL, "task": UMA_TASK}


def prerelax(args: argparse.Namespace) -> None:
    out = args.output_dir.expanduser().resolve()
    starts = json.loads((out / "starts.json").read_text())["starts"]
    calculator, model_info = load_calculator(args.model, args.device)
    model_dir = out / f"umlip_{args.model}"
    runs = {}
    for label, start in starts.items():
        run_dir = model_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        atoms = ase_read(start["path"])
        atoms.calc = calculator
        history: list[float] = []
        trajectory_path = run_dir / "trajectory.traj"
        with Trajectory(trajectory_path, "w") as writer:

            def record() -> None:
                history.append(fmax_of(atoms))
                writer.write(atoms)

            optimizer = BFGS(atoms, maxstep=args.maxstep, logfile=str(run_dir / "optimizer.log"))
            optimizer.attach(record, interval=1)
            started = time.time()
            converged = bool(optimizer.run(fmax=args.fmax, steps=args.max_steps))
        handoff = next((i for i, value in enumerate(history) if value < args.handoff_fmax), None)
        distances = atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)]
        runs[label] = {
            "trajectory": trajectory_path.as_posix(),
            "frames": len(history),
            "converged": converged,
            "final_fmax": history[-1],
            "final_energy_ev": float(atoms.get_potential_energy()),
            "handoff_frame": handoff,
            "handoff_fmax": history[handoff] if handoff is not None else None,
            "final_min_distance_angstrom": float(distances.min()),
            "wall_seconds": time.time() - started,
        }
        print(f"[{args.model}] {label}: {len(history)} frames, handoff frame {handoff}, "
              f"final fmax {history[-1]:.2e}", flush=True)
    write_json(model_dir / "prerelax_summary.json", {
        "model": args.model, **model_info, "optimizer": "ase.optimize.BFGS", "maxstep_angstrom": args.maxstep,
        "fmax_ev_per_angstrom": args.fmax, "handoff_fmax_ev_per_angstrom": args.handoff_fmax, "runs": runs,
    })


def prepare_dft(args: argparse.Namespace) -> None:
    from relaxgen.dft import dft_relaxation_campaign as campaign

    out = args.output_dir.expanduser().resolve()
    starts = json.loads((out / "starts.json").read_text())["starts"]
    # The DFT-only arm comes first so the controller's GPU jobs start it first.
    tasks = [
        {"arm": "dftonly", "start": label, "replicate": replicate, "structure": start["path"]}
        for replicate in range(1, args.replicates + 1)
        for label, start in starts.items()
    ]
    for model in MODELS:
        summary = json.loads((out / f"umlip_{model}" / "prerelax_summary.json").read_text())
        for label in starts:
            run = summary["runs"][label]
            if run["handoff_frame"] is None:
                raise RuntimeError(f"{model} {label} never reached ML fmax {summary['handoff_fmax_ev_per_angstrom']}.")
            tasks.append({"arm": model, "start": label, "replicate": 1,
                          "structure": f"{run['trajectory']}@{run['handoff_frame']}"})
    for task in tasks:
        suffix = f"_r{task['replicate']}" if task["arm"] == "dftonly" else ""
        task["label"] = f"{task['arm']}_{task['start']}{suffix}"

    dft_dir = out / "dft"
    campaign.prepare(campaign.build_parser().parse_args([
        "prepare", "--output-dir", str(dft_dir),
        "--structures", *[t["structure"] for t in tasks], "--labels", *[t["label"] for t in tasks],
        "--kpts", "6", "6", "6", "--fmax", str(DFT_FMAX), "--max-steps", str(args.max_steps),
        *(["--overwrite"] if args.overwrite else []),
    ]))
    write_json(out / "experiment.json", {
        "dft_manifest": (dft_dir / "relax_manifest.json").as_posix(),
        "dft_fmax_ev_per_angstrom": DFT_FMAX, "report_fmax_ev_per_angstrom": DFT_REPORT_FMAX,
        "tasks": [{"task_index": i, **task} for i, task in enumerate(tasks)],
    })
    print(f"Prepared {len(tasks)} QE relaxations in {dft_dir}")


def _load_run(task: dict[str, Any], manifest_task: dict[str, Any]) -> dict[str, Any]:
    status_path = Path(manifest_task["status"])
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    frames = campaign_frames(Path(manifest_task["trajectory"]))
    fmax = [fmax_of(f) for f in frames]
    report = next((i for i, value in enumerate(fmax) if value < DFT_REPORT_FMAX), None)
    return {
        **task,
        "execution_status": status.get("execution_status", "pending"),
        "failure_kind": status.get("failure_kind"),
        "converged": status.get("scientifically_converged") is True,
        "scf_count": len(frames),
        "current_fmax": fmax[-1] if fmax else None,
        "report_frame": report,
        "frames": frames,
    }


def campaign_frames(path: Path) -> list:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with Trajectory(path) as reader:
        return list(reader)


def _clusters(structures: list, energies: list[float]) -> dict[str, Any]:
    from relaxgen.analysis.analyze_umlip_endpoint_diversity import complete_linkage_labels, matched_mic_displacements

    count = len(structures)
    matrix = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            d = matched_mic_displacements(structures[i], structures[j])
            matrix[i, j] = matrix[j, i] = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    result: dict[str, Any] = {"count": count, "energies_ev": energies}
    if count:
        result["energy_spread_mev"] = 1.0e3 * (max(energies) - min(energies))
        result["clusters"] = {
            str(t): int(len(set(complete_linkage_labels(matrix, t).tolist()))) for t in RMSD_THRESHOLDS
        }
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_dir.expanduser().resolve()
    experiment = json.loads((out / "experiment.json").read_text())
    manifest = json.loads(Path(experiment["dft_manifest"]).read_text())
    by_index = {int(t["task_index"]): t for t in manifest["tasks"]}
    runs = [_load_run(task, by_index[task["task_index"]]) for task in experiment["tasks"]]

    report: dict[str, Any] = {"runs": [], "arms": {}, "replicate_noise": [], "cross_arm": []}
    for run in runs:
        report["runs"].append({k: v for k, v in run.items() if k != "frames"})

    finals = {}
    for arm in ("dftonly", "eqv3", "uma"):
        members = [r for r in runs if r["arm"] == arm]
        done = [r for r in members if r["converged"]]
        # One replicate per start, so the DFT-only arm is compared on equal footing.
        primary = [r for r in done if r["replicate"] == 1]
        reached = [r for r in members if r["report_frame"] is not None and r["replicate"] == 1]
        report["arms"][arm] = {
            "relaxations": len(members),
            "converged": len(done),
            "abandoned_scf_stall": sum(r["failure_kind"] == "scf_not_converged" for r in members),
            "scf_count_median": float(np.median([r["scf_count"] for r in done])) if done else None,
            "endpoints_fmax_1e-3": _clusters([r["frames"][-1] for r in primary],
                                             [float(r["frames"][-1].get_potential_energy()) for r in primary]),
            "frames_fmax_1e-2": _clusters([r["frames"][r["report_frame"]] for r in reached],
                                          [float(r["frames"][r["report_frame"]].get_potential_energy())
                                           for r in reached]),
        }
        finals[arm] = {r["start"]: r["frames"][-1] for r in primary}

    from relaxgen.analysis.analyze_umlip_endpoint_diversity import matched_mic_displacements

    def rmsd(a, b) -> float:
        d = matched_mic_displacements(a, b)
        return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))

    for start in STARTS:
        pair = [r for r in runs if r["arm"] == "dftonly" and r["start"] == start and r["converged"]]
        if len(pair) == 2:
            a, b = pair[0]["frames"][-1], pair[1]["frames"][-1]
            report["replicate_noise"].append({
                "start": start, "endpoint_rmsd_angstrom": rmsd(a, b),
                "energy_difference_mev": 1.0e3 * float(a.get_potential_energy() - b.get_potential_energy()),
            })
        for arm in ("eqv3", "uma"):
            if start in finals["dftonly"] and start in finals[arm]:
                a, b = finals[arm][start], finals["dftonly"][start]
                report["cross_arm"].append({
                    "start": start, "arm": arm, "endpoint_rmsd_to_dftonly_angstrom": rmsd(a, b),
                    "energy_minus_dftonly_mev": 1.0e3 * float(a.get_potential_energy() - b.get_potential_energy()),
                })
    write_json(out / "diversity_report.json", report)

    print(f"{'arm':8s} {'done':>6s} {'stalls':>6s} {'SCFs':>6s}  endpoints: n, clusters@{RMSD_THRESHOLDS} A, E spread")
    for arm, entry in report["arms"].items():
        end = entry["endpoints_fmax_1e-3"]
        print(f"{arm:8s} {entry['converged']:>3d}/{entry['relaxations']:<2d} {entry['abandoned_scf_stall']:>6d} "
              f"{entry['scf_count_median'] or float('nan'):>6.0f}  {end['count']}, {end.get('clusters')}, "
              f"{end.get('energy_spread_mev', float('nan')):.1f} meV")
    for row in report["replicate_noise"]:
        print(f"replicates {row['start']}: RMSD {row['endpoint_rmsd_angstrom']:.3f} A, dE {row['energy_difference_mev']:+.2f} meV")
    for row in report["cross_arm"]:
        print(f"{row['arm']} vs dftonly {row['start']}: RMSD {row['endpoint_rmsd_to_dftonly_angstrom']:.3f} A, "
              f"dE {row['energy_minus_dftonly_mev']:+.2f} meV")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-starts", "prerelax", "prepare-dft", "analyze"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "prerelax":
            command.add_argument("--model", choices=MODELS, required=True)
            command.add_argument("--device", default="cuda")
            command.add_argument("--fmax", type=float, default=ML_FMAX)
            command.add_argument("--handoff-fmax", type=float, default=HANDOFF_FMAX)
            command.add_argument("--maxstep", type=float, default=0.10)
            command.add_argument("--max-steps", type=int, default=3000)
        if name == "prepare-dft":
            command.add_argument("--replicates", type=int, default=2)
            command.add_argument("--max-steps", type=int, default=400)
            command.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    {"prepare-starts": prepare_starts, "prerelax": prerelax, "prepare-dft": prepare_dft, "analyze": analyze}[
        args.command
    ](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
