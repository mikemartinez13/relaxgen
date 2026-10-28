#!/usr/bin/env python3
"""Stage B: are the minima that only one uMLIP finds real DFT minima?

Stage A (``umlip_ensemble_diversity.py``) relaxed every start with each model.
About one start in five ends more than 0.2 Angstrom apart depending on the model.
Stage B DFT-relaxes each model's handoff frame (first ML frame with fmax < 0.02
eV/Angstrom) for a sample of those starts and asks whether the DFT endpoints stay
distinct or merge.

``select`` picks:

* disagreement starts: in configurations where some start's model endpoints are
  more than ``--disagree-rmsd`` apart, the most disagreeing start, spread over
  defect families (round-robin) up to ``--disagreement-configs``;
* control starts: one start per configuration where every model ends within
  ``--agree-rmsd`` (the PBE-trained UMA sits ~0.08 Angstrom off the PBEsol-trained
  models even in shared minima), up to ``--control-configs`` other configurations.

Each selected start contributes one DFT relaxation per model, written as a
``dft_relaxation_campaign.py`` manifest at the production protocol with
``--fmax`` 0.05: merging versus staying distinct and the energy ordering of
distinct endpoints are settled long before the force tail.  ``qe_dispatch.py``
runs the manifest.

``analyze`` compares, per start, the DFT endpoints of the model arms: pairs more
than ``--distinct-rmsd`` apart stay distinct, others merged.  Merged pairs whose
energies still differ by more than ``--split-mev`` sit on a soft mode that fmax
0.05 does not resolve; with ``--tighten`` both arms resume to ``--tighten-fmax``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relaxgen.prerelax.umlip_ensemble_diversity import MODELS, endpoint_rmsd, write_json  # noqa: E402

DEFAULT_ENSEMBLE_DIR = Path("/scratch/gqs7zr/relaxgen/umlip_ensemble")


def min_distance(atoms) -> float:
    return float(atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)].min())


def load_runs(ensemble_dir: Path) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    """(config, candidate) -> model -> Stage A run record, for converged, safe runs."""

    starts: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for model in MODELS:
        for path in sorted((ensemble_dir / model).glob("*/*/*.json")):
            run = json.loads(path.read_text())
            if run["converged"] and not run["unsafe_geometry"]:
                starts[run["defect_config_id"], run["candidate_id"]][model] = run
    return starts


def _start_disagreement(item: tuple[tuple[str, str], dict[str, dict[str, Any]]]) -> dict[str, Any]:
    (config, candidate), runs = item
    ends = {}
    for model, run in runs.items():
        with Trajectory(run["trajectory"]) as reader:
            ends[model] = reader[-1]
    pairs = {f"{a}|{b}": endpoint_rmsd(ends[a], ends[b]) for a, b in combinations(sorted(ends), 2)}
    return {"config": config, "candidate_id": candidate, "family": next(iter(runs.values()))["defect_family"],
            "pair_rmsd": pairs, "max_pair_rmsd": max(pairs.values())}


def handoff(run: dict[str, Any], min_dist: float) -> dict[str, Any]:
    """The run's handoff frame, moved later if it has atoms closer than ``min_dist``."""

    frames = ase_read(run["trajectory"], index=":")
    first = run["handoff_frame"] if run["handoff_frame"] is not None else len(frames) - 1
    for index in range(first, len(frames)):
        closest = min_distance(frames[index])
        if closest >= min_dist:
            fmax = float(np.linalg.norm(frames[index].get_forces(), axis=1).max())
            return {"frame": index, "ml_fmax": fmax, "min_distance": closest}
    raise ValueError(f"{run['trajectory']}: no frame from {first} on with min distance >= {min_dist} A")


def round_robin(by_family: dict[str, list[dict[str, Any]]], count: int) -> list[dict[str, Any]]:
    queues = {family: list(items) for family, items in sorted(by_family.items())}
    picked: list[dict[str, Any]] = []
    while len(picked) < count and any(queues.values()):
        for family in sorted(queues):
            if queues[family] and len(picked) < count:
                picked.append(queues[family].pop(0))
    return picked


def select(args: argparse.Namespace) -> dict[str, Any]:
    from relaxgen.dft import dft_relaxation_campaign as campaign

    ensemble_dir = args.ensemble_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    runs = load_runs(ensemble_dir)
    models = sorted({m for per_start in runs.values() for m in per_start}, key=MODELS.index)
    complete = {key: per_start for key, per_start in runs.items() if set(per_start) == set(models)}
    with Pool(args.workers) as pool:
        starts = pool.map(_start_disagreement, sorted(complete.items()), chunksize=16)

    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for start in starts:
        by_config[start["config"]].append(start)

    disagreement: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for config, items in by_config.items():
        worst = max(items, key=lambda s: s["max_pair_rmsd"])
        if worst["max_pair_rmsd"] > args.disagree_rmsd:
            disagreement[worst["family"]].append(worst)
    for items in disagreement.values():
        items.sort(key=lambda s: -s["max_pair_rmsd"])
    chosen = round_robin(disagreement, args.disagreement_configs)

    taken = {s["config"] for s in chosen}
    agreeing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for config, items in sorted(by_config.items()):
        if config in taken:
            continue
        agree = [s for s in items if s["max_pair_rmsd"] < args.agree_rmsd]
        if agree:
            agree.sort(key=lambda s: (s["candidate_id"] != "unperturbed", s["max_pair_rmsd"]))
            agreeing[agree[0]["family"]].append(agree[0])
    controls = round_robin(agreeing, args.control_configs)

    tasks = []
    for role, group in (("disagreement", chosen), ("control", controls)):
        for start in group:
            for model in models:
                run = complete[start["config"], start["candidate_id"]][model]
                frame = handoff(run, args.min_distance)
                tasks.append({
                    "role": role, "defect_family": start["family"], "defect_config_id": start["config"],
                    "candidate_id": start["candidate_id"], "model": model,
                    "label": f"{start['config']}__{start['candidate_id']}__{model}",
                    "structure": f"{run['trajectory']}@{frame['frame']}", "handoff": frame,
                    "ml_endpoint_energy_ev": run["final_energy_ev"], "ml_pair_rmsd": start["pair_rmsd"],
                    "ml_trajectory": run["trajectory"],
                })

    campaign.prepare(campaign.build_parser().parse_args([
        "prepare", "--output-dir", str(out),
        "--structures", *[t["structure"] for t in tasks], "--labels", *[t["label"] for t in tasks],
        "--kpts", "6", "6", "6", "--fmax", str(args.fmax), "--max-steps", str(args.max_steps),
        *(["--overwrite"] if args.overwrite else []),
    ]))
    selection = {
        "ensemble_dir": ensemble_dir.as_posix(), "models": models,
        "disagree_rmsd_angstrom": args.disagree_rmsd, "agree_rmsd_angstrom": args.agree_rmsd,
        "dft_fmax_ev_per_angstrom": args.fmax, "min_distance_angstrom": args.min_distance,
        "configs_with_disagreement": sum(len(v) for v in disagreement.values()),
        "tasks": [{"task_index": i, **t} for i, t in enumerate(tasks)],
    }
    write_json(out / "stage_b.json", selection)
    print(f"{selection['configs_with_disagreement']} configurations disagree by > {args.disagree_rmsd} A; "
          f"selected {len(chosen)} disagreement and {len(controls)} control starts "
          f"({len(tasks)} DFT relaxations) in {out}")
    for start in chosen + controls:
        pairs = " ".join(f"{k}={v:.2f}" for k, v in start["pair_rmsd"].items())
        print(f"  {start['config']:22s} {start['candidate_id']:36s} {pairs}")
    return selection


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_dir.expanduser().resolve()
    selection = json.loads((out / "stage_b.json").read_text())
    manifest = json.loads((out / "relax_manifest.json").read_text())
    status_by_index = {t["task_index"]: Path(t["status"]) for t in manifest["tasks"]}

    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for task in selection["tasks"]:
        status_path = status_by_index[task["task_index"]]
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        groups[task["role"], task["defect_config_id"], task["candidate_id"]][task["model"]] = {
            **task, "dft": status,
        }

    rows, pending = [], 0
    for (role, config, candidate), arms in sorted(groups.items()):
        done = {m: a for m, a in arms.items() if a["dft"].get("scientifically_converged")}
        pending += len(arms) - len(done)
        ends = {}
        for model, arm in done.items():
            with Trajectory(arm["dft"]["trajectory"]) as reader:
                ends[model] = reader[-1]
        for a, b in combinations(sorted(ends), 2):
            dft_rmsd = endpoint_rmsd(ends[a], ends[b])
            rows.append({
                "role": role, "config": config, "candidate_id": candidate, "pair": f"{a}|{b}",
                "ml_rmsd": done[a]["ml_pair_rmsd"].get(f"{a}|{b}"), "dft_rmsd": dft_rmsd,
                "dft_energy_difference_mev": 1.0e3 * (done[a]["dft"]["final_energy_ev"]
                                                      - done[b]["dft"]["final_energy_ev"]),
                "distinct": dft_rmsd > args.distinct_rmsd,
                "task_indices": [done[a]["task_index"], done[b]["task_index"]],
                "fmax": [done[a]["dft"]["final_max_force_ev_per_angstrom"],
                         done[b]["dft"]["final_max_force_ev_per_angstrom"]],
            })

    summary: dict[str, Any] = {"pending_or_failed_relaxations": pending, "pairs": rows}
    print(f"{pending} relaxations not finished or failed")
    for role in ("disagreement", "control"):
        role_rows = [r for r in rows if r["role"] == role]
        if role == "disagreement":
            role_rows = [r for r in role_rows if (r["ml_rmsd"] or 0) > selection["disagree_rmsd_angstrom"]]
        distinct = sum(r["distinct"] for r in role_rows)
        summary[role] = {"pairs": len(role_rows), "distinct": distinct}
        label = "model pairs > %.2f A apart in ML" % selection["disagree_rmsd_angstrom"] if role == "disagreement" \
            else "agreeing model pairs"
        print(f"{role}: {len(role_rows)} {label}; after DFT {distinct} stay > {args.distinct_rmsd} A apart, "
              f"{len(role_rows) - distinct} merge")
        for r in role_rows:
            print(f"  {r['config']:22s} {r['candidate_id'][:30]:30s} {r['pair']:14s} ML {r['ml_rmsd'] or 0:.3f} A"
                  f" -> DFT {r['dft_rmsd']:.3f} A, dE {r['dft_energy_difference_mev']:+8.1f} meV"
                  f"{'  distinct' if r['distinct'] else ''}")
    # Same structure within the RMSD cut but a clear energy gap: at fmax 0.05 a soft mode
    # can leave ~100 meV unrelaxed, so these arms go on to a tighter force target.
    split = [r for r in rows if not r["distinct"] and abs(r["dft_energy_difference_mev"]) > args.split_mev
             and max(r["fmax"]) >= args.tighten_fmax]
    summary["merged_with_energy_gap"] = split
    print(f"{len(split)} merged pairs differ by > {args.split_mev:.0f} meV")
    for r in split:
        print(f"  {r['config']:22s} {r['pair']:14s} DFT {r['dft_rmsd']:.3f} A, dE {r['dft_energy_difference_mev']:+.1f} meV,"
              f" tasks {r['task_indices']}")
    if split and args.tighten:
        from relaxgen.dft import dft_relaxation_campaign as campaign

        indices = sorted({i for r in split for i in r["task_indices"]})
        summary["tightened"] = campaign.tighten(campaign.build_parser().parse_args(
            ["tighten", "--manifest", str(out / "relax_manifest.json"), "--task-indices", *map(str, indices),
             "--fmax", str(args.tighten_fmax)]))
    write_json(out / "stage_b_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--output-dir", type=Path, required=True)
    sel.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    sel.add_argument("--disagree-rmsd", type=float, default=0.3)
    sel.add_argument("--agree-rmsd", type=float, default=0.12)
    sel.add_argument("--disagreement-configs", type=int, default=16)
    sel.add_argument("--control-configs", type=int, default=5)
    sel.add_argument("--min-distance", type=float, default=2.0)
    sel.add_argument("--fmax", type=float, default=0.05)
    sel.add_argument("--max-steps", type=int, default=200)
    sel.add_argument("--workers", type=int, default=8)
    sel.add_argument("--overwrite", action="store_true")
    ana = sub.add_parser("analyze")
    ana.add_argument("--output-dir", type=Path, required=True)
    ana.add_argument("--distinct-rmsd", type=float, default=0.2)
    ana.add_argument("--split-mev", type=float, default=10.0)
    ana.add_argument("--tighten-fmax", type=float, default=0.01)
    ana.add_argument("--tighten", action="store_true",
                     help="Reopen both arms of each merged pair with an energy gap at --tighten-fmax.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    {"select": select, "analyze": analyze}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
