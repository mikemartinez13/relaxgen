#!/usr/bin/env python3
"""Stage A: do different uMLIPs relax the same starts into different minima?

Every reconstruction candidate (2124 starts over 188 defect configurations) is
relaxed independently by each model, with identical settings: ASE BFGS,
``maxstep`` 0.1 Angstrom, fixed cell, ML fmax 1e-3 eV/Angstrom.  A relaxation that
brings two atoms closer than ``--hard-min-distance`` is stopped and flagged.

Models:

* ``eqv3``: EquiformerV3 fine-tuned on Bi2Te3 (the production pre-relaxer);
* ``mace_ft``: MACE-OMAT-medium fine-tuned on Bi2Te3;
* ``uma``: UMA ``uma-s-1p2``, ``omat`` task (PBE-trained; the cells and QE are PBEsol);
* ``mace_omat``: the MACE-OMAT-medium foundation model, the fine-tune's parent.

``analyze`` asks whether pooling models adds minima: per start, the
inter-model endpoint RMSD; per configuration, the number of distinct endpoints
(complete linkage on permutation-matched MIC RMSD, minimised over lattice
symmetry operations) found by
each model alone, by the fine-tuned pair, and by all models together; and
whether models agree on the energy ranking of a configuration's endpoints.
Each model's handoff frame (first frame with ML fmax < 0.02 eV/Angstrom) is
recorded for Stage B, which DFT-relaxes the starts where the models disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relaxgen import CKPT_DIR  # noqa: E402

EQV3_RUNS = Path("/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/05_eqv3_tight_relaxations")
MACE_FT = CKPT_DIR / "mace_ft" / "bi2te3_mace_omat_medium_seed42_resume_epoch17_run-42.model"
MACE_OMAT = Path("/scratch/gqs7zr/ai4mat/mace_finetune/mace-omat-0-medium.model")
MODELS = ("eqv3", "mace_ft", "uma", "mace_omat")
FINE_TUNED = ("eqv3", "mace_ft")
HANDOFF_FMAX = 0.02
RMSD_THRESHOLDS = (0.05, 0.1, 0.2)
SYMMETRY_CHECK_FROM = 0.05


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> None:
    """Freeze the candidate list from the EqV3/FIRE2 campaign's inputs."""

    candidates = []
    for status_path in sorted(args.eqv3_runs.glob("*/*/*/status.json")):
        status = json.loads(status_path.read_text())
        candidates.append({
            "defect_family": status["defect_family"],
            "defect_config_id": status["defect_config_id"],
            "candidate_id": status["candidate_id"],
            "candidate_type": status.get("candidate_type"),
            "structure": status["source_candidate_structure"],
        })
    for index, candidate in enumerate(candidates):
        candidate["index"] = index
    out = args.output_dir.expanduser().resolve()
    write_json(out / "candidates.json", {"source": args.eqv3_runs.as_posix(), "candidates": candidates})
    print(f"{len(candidates)} candidates over {len({c['defect_config_id'] for c in candidates})} configurations")


def load_calculator(model: str, device: str):
    if model == "eqv3":
        from relaxgen.prerelax.relax_reconstruction_candidates_eqv3 import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_eqv3

        return load_eqv3(DEFAULT_CHECKPOINT, DEFAULT_CONFIG, device, 42).calculator, DEFAULT_CHECKPOINT.as_posix()
    if model in ("mace_ft", "mace_omat"):
        from mace.calculators import MACECalculator

        path = MACE_FT if model == "mace_ft" else MACE_OMAT
        return MACECalculator(model_paths=str(path), device=device, default_dtype="float64"), path.as_posix()
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=device, seed=42)
    return FAIRChemCalculator(predictor, task_name="omat"), "uma-s-1p2/omat"


def run_path(out: Path, model: str, candidate: dict[str, Any]) -> Path:
    return out / model / candidate["defect_family"] / candidate["defect_config_id"] / candidate["candidate_id"]


def min_distance(atoms) -> float:
    return float(atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)].min())


def relax(args: argparse.Namespace) -> None:
    out = args.output_dir.expanduser().resolve()
    candidates = json.loads((out / "candidates.json").read_text())["candidates"]
    mine = candidates[args.shard :: args.num_shards][: args.max_candidates]
    calculator, model_source = load_calculator(args.model, args.device)

    class Unsafe(Exception):
        pass

    for candidate in mine:
        base = run_path(out, args.model, candidate)
        if base.with_suffix(".json").exists():
            continue
        base.parent.mkdir(parents=True, exist_ok=True)
        atoms = ase_read(candidate["structure"])
        atoms.calc = calculator
        fmax_history: list[float] = []
        closest = [np.inf]
        with Trajectory(base.with_suffix(".traj"), "w") as writer:

            def record() -> None:
                fmax_history.append(float(np.linalg.norm(atoms.get_forces(), axis=1).max()))
                closest[0] = min(closest[0], min_distance(atoms))
                writer.write(atoms)
                if closest[0] < args.hard_min_distance:
                    raise Unsafe()

            optimizer = BFGS(atoms, maxstep=args.maxstep, logfile=None)
            optimizer.attach(record, interval=1)
            started = time.time()
            unsafe = False
            try:
                converged = bool(optimizer.run(fmax=args.fmax, steps=args.max_steps))
            except Unsafe:
                converged, unsafe = False, True
        handoff = next((i for i, value in enumerate(fmax_history) if value < HANDOFF_FMAX), None)
        write_json(base.with_suffix(".json"), {
            **candidate, "model": args.model, "model_source": model_source,
            "trajectory": base.with_suffix(".traj").as_posix(), "frames": len(fmax_history),
            "converged": converged, "unsafe_geometry": unsafe, "final_fmax": fmax_history[-1],
            "final_energy_ev": float(atoms.get_potential_energy()), "min_distance_seen": closest[0],
            "handoff_frame": handoff, "wall_seconds": time.time() - started,
        })
        print(f"[{args.model}] {candidate['defect_config_id']}/{candidate['candidate_id']}: "
              f"{len(fmax_history)} frames, fmax {fmax_history[-1]:.1e}{' UNSAFE' if unsafe else ''}", flush=True)


def endpoint_rmsd(left, right) -> float:
    """Atom-matched MIC RMSD, minimised over the lattice's symmetry operations.

    Translation-only matching puts the same defect structure at a symmetry-equivalent
    site or orientation 1-2 Angstrom away, so pairs above ``SYMMETRY_CHECK_FROM``
    are also matched with pymatgen's StructureMatcher (rotations included).
    """

    from relaxgen.analysis.analyze_umlip_endpoint_diversity import matched_mic_displacements
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor

    d = matched_mic_displacements(left, right)
    rmsd = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    if rmsd <= SYMMETRY_CHECK_FROM:
        return rmsd
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False, scale=False,
                               attempt_supercell=False)
    a, b = AseAtomsAdaptor.get_structure(left), AseAtomsAdaptor.get_structure(right)
    match = matcher.get_rms_dist(a, b)
    if match is None:
        return rmsd
    # StructureMatcher reports RMS displacement in units of (volume / sites)^(1/3).
    return min(rmsd, float(match[0]) * (a.volume / len(a)) ** (1.0 / 3.0))


def _rmsd_matrix(structures: list) -> np.ndarray:
    count = len(structures)
    matrix = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            matrix[i, j] = matrix[j, i] = endpoint_rmsd(structures[i], structures[j])
    return matrix


def _analyze_config(item: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Cluster one configuration's endpoints across models and candidates."""

    from relaxgen.analysis.analyze_umlip_endpoint_diversity import complete_linkage_labels

    config, runs = item
    usable = [r for r in runs if r["converged"] and not r["unsafe_geometry"]]
    endpoints = []
    for run in usable:
        with Trajectory(run["trajectory"]) as reader:
            endpoints.append(reader[-1])
    matrix = _rmsd_matrix(endpoints) if endpoints else np.zeros((0, 0))
    models = [r["model"] for r in usable]
    result: dict[str, Any] = {"config": config, "family": runs[0]["defect_family"], "endpoints": len(usable),
                              "unsafe": sum(r["unsafe_geometry"] for r in runs),
                              "unconverged": sum(not r["converged"] and not r["unsafe_geometry"] for r in runs),
                              "clusters": {}, "novel_clusters": {}, "start_disagreement": []}

    def count(subset: Sequence[str], threshold: float) -> int:
        idx = [i for i, m in enumerate(models) if m in subset]
        if not idx:
            return 0
        return int(len(set(complete_linkage_labels(matrix[np.ix_(idx, idx)], threshold).tolist())))

    for threshold in RMSD_THRESHOLDS:
        key = str(threshold)
        result["clusters"][key] = {
            **{m: count([m], threshold) for m in MODELS},
            "fine_tuned": count(FINE_TUNED, threshold),
            "all": count(MODELS, threshold),
        }
        # Clusters of the pooled endpoints that contain only one model's endpoints.
        if len(endpoints) > 1:
            labels = complete_linkage_labels(matrix, threshold)
            members = defaultdict(set)
            for label, model in zip(labels.tolist(), models):
                members[label].add(model)
            result["novel_clusters"][key] = {
                m: sum(1 for owners in members.values() if owners == {m}) for m in MODELS
            }

    # Per start: how far apart the models' endpoints are.
    by_start: dict[str, dict[str, int]] = defaultdict(dict)
    for i, run in enumerate(usable):
        by_start[run["candidate_id"]][run["model"]] = i
    for candidate, index in by_start.items():
        pairs = {f"{a}|{b}": float(matrix[index[a], index[b]])
                 for n, a in enumerate(MODELS) for b in MODELS[n + 1:] if a in index and b in index}
        result["start_disagreement"].append({"candidate_id": candidate, "pair_rmsd": pairs})

    # Energy-ranking agreement between models over the starts they all converged.
    # Models without runs (e.g. one left out of the campaign) are skipped.
    ranks = {
        model: {r["candidate_id"]: r["final_energy_ev"] for r in usable if r["model"] == model}
        for model in MODELS
    }
    present = [model for model in MODELS if ranks[model]]
    common = sorted(set.intersection(*(set(ranks[m]) for m in present))) if present else []
    if len(present) > 1 and len(common) >= 4:
        from scipy.stats import spearmanr

        result["energy_rank_spearman"] = {
            f"{a}|{b}": float(spearmanr([ranks[a][c] for c in common], [ranks[b][c] for c in common])[0])
            for n, a in enumerate(present) for b in present[n + 1:]
        }
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    from multiprocessing import Pool

    out = args.output_dir.expanduser().resolve()
    runs = [json.loads(p.read_text()) for model in MODELS for p in sorted((out / model).glob("*/*/*.json"))]
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_config[run["defect_config_id"]].append(run)
    with Pool(args.workers) as pool:
        configs = pool.map(_analyze_config, sorted(by_config.items()), chunksize=1)

    summary: dict[str, Any] = {"runs": len(runs), "configs": len(configs), "per_model": {}, "thresholds": {}}
    for model in MODELS:
        mine = [r for r in runs if r["model"] == model]
        summary["per_model"][model] = {
            "runs": len(mine), "converged": sum(r["converged"] for r in mine),
            "unsafe": sum(r["unsafe_geometry"] for r in mine),
            "median_frames": float(np.median([r["frames"] for r in mine])) if mine else None,
        }
    pair_values: dict[str, list[float]] = defaultdict(list)
    for config in configs:
        for start in config["start_disagreement"]:
            for pair, value in start["pair_rmsd"].items():
                pair_values[pair].append(value)
    summary["start_pair_rmsd"] = {
        pair: {"n": len(v), "median": float(np.median(v)), "frac_above_0.1": float(np.mean(np.array(v) > 0.1))}
        for pair, v in pair_values.items()
    }
    for threshold in RMSD_THRESHOLDS:
        key = str(threshold)
        totals = {name: sum(c["clusters"][key][name] for c in configs) for name in (*MODELS, "fine_tuned", "all")}
        novel = {m: sum(c["novel_clusters"].get(key, {}).get(m, 0) for c in configs) for m in MODELS}
        summary["thresholds"][key] = {"total_clusters": totals, "clusters_only_this_model_found": novel}
    spearman: dict[str, list[float]] = defaultdict(list)
    for config in configs:
        for pair, value in (config.get("energy_rank_spearman") or {}).items():
            if np.isfinite(value):
                spearman[pair].append(value)
    summary["energy_rank_spearman_median"] = {pair: float(np.median(v)) for pair, v in spearman.items()}
    write_json(out / "ensemble_diversity_summary.json", summary)
    write_json(out / "ensemble_diversity_configs.json", configs)

    print(f"{summary['runs']} relaxations over {summary['configs']} configurations")
    for model, entry in summary["per_model"].items():
        print(f"  {model:10s} runs={entry['runs']} converged={entry['converged']} unsafe={entry['unsafe']} "
              f"median frames={entry['median_frames']}")
    print("Per-start endpoint RMSD between models (median, fraction > 0.1 A):")
    for pair, entry in summary["start_pair_rmsd"].items():
        print(f"  {pair:22s} {entry['median']:.3f} A  {entry['frac_above_0.1']:.0%}")
    for key, entry in summary["thresholds"].items():
        print(f"Distinct endpoints at {key} A: {entry['total_clusters']}")
        print(f"  clusters found by only one model: {entry['clusters_only_this_model_found']}")
    print(f"Energy-ranking Spearman (median over configurations): {summary['energy_rank_spearman_median']}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "relax", "analyze"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--eqv3-runs", type=Path, default=EQV3_RUNS)
        if name == "relax":
            command.add_argument("--model", choices=MODELS, required=True)
            command.add_argument("--shard", type=int, default=0)
            command.add_argument("--num-shards", type=int, default=1)
            command.add_argument("--max-candidates", type=int)
            command.add_argument("--device", default="cuda")
            command.add_argument("--fmax", type=float, default=1.0e-3)
            command.add_argument("--maxstep", type=float, default=0.10)
            command.add_argument("--max-steps", type=int, default=1500)
            command.add_argument("--hard-min-distance", type=float, default=1.2)
        if name == "analyze":
            command.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    {"prepare": prepare, "relax": relax, "analyze": analyze}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
