#!/usr/bin/env python3
"""Build a production DFT relaxation batch from EqV3 handoff frames.

Batch 1: the unperturbed candidate of every defect configuration, plus
``rattle_seed_101`` of each family's ``<family>_001`` configuration (the
rattled cohort of the handoff-threshold experiment, so their handoff frames
already have DFT single points).  Each relaxation starts from the first frame
of the existing EqV3/FIRE2 trajectory with ML fmax below ``--handoff-fmax``,
where tighter ML relaxation stopped lowering the DFT force.  Frames closer than
``--min-distance`` (trajectories that tripped the EqV3 geometry guard keep
their unsafe frames) are never used.

The QE relaxations go through ``dft_relaxation_campaign.py`` with the
production protocol; ``qe_dispatch.py`` runs them.  Unperturbed candidates come
first in the manifest, so the controller starts them first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase.io import iread

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_EQV3_DIR = Path("/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/05_eqv3_tight_relaxations")


def handoff_frame(trajectory: Path, handoff_fmax: float, min_distance: float) -> dict[str, Any] | None:
    for index, atoms in enumerate(iread(trajectory, index=":")):
        fmax = float(atoms.info.get("ml_fmax_ev_per_angstrom", np.inf))
        if fmax >= handoff_fmax:
            continue
        distances = atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)]
        if distances.min() >= min_distance:
            return {"frame": index, "ml_fmax": fmax, "min_distance": float(distances.min()),
                    "optimizer_step": int(atoms.info.get("optimizer_step", index))}
    return None


def select_candidates(eqv3_dir: Path, rattled: str) -> list[dict[str, Any]]:
    statuses = sorted(eqv3_dir.glob("*/*/*/status.json"))
    unperturbed, extra = [], []
    for path in statuses:
        status = json.loads(path.read_text())
        family, config = status["defect_family"], status["defect_config_id"]
        if status["candidate_id"] == "unperturbed":
            unperturbed.append(status)
        elif status["candidate_id"] == rattled and config == f"{family}_001":
            extra.append(status)
    return unperturbed + extra


def prepare(args: argparse.Namespace) -> None:
    from relaxgen.dft import dft_relaxation_campaign as campaign

    out = args.output_dir.expanduser().resolve()
    selected, skipped = [], []
    for status in select_candidates(args.eqv3_dir.expanduser().resolve(), args.rattled_candidate):
        trajectory = Path(status["trajectory"])
        handoff = handoff_frame(trajectory, args.handoff_fmax, args.min_distance)
        record = {"defect_family": status["defect_family"], "defect_config_id": status["defect_config_id"],
                  "candidate_id": status["candidate_id"], "eqv3_converged": status["converged"],
                  "eqv3_fallback_used": bool(status.get("fallback_used")), "eqv3_trajectory": trajectory.as_posix()}
        if handoff is None:
            skipped.append({**record, "reason": f"no frame with ML fmax < {args.handoff_fmax} "
                                                f"and min distance >= {args.min_distance} A"})
            continue
        selected.append({**record, "handoff": handoff,
                         "label": f"{status['defect_config_id']}__{status['candidate_id']}",
                         "structure": f"{trajectory}@{handoff['frame']}"})

    campaign.prepare(campaign.build_parser().parse_args([
        "prepare", "--output-dir", str(out),
        "--structures", *[s["structure"] for s in selected], "--labels", *[s["label"] for s in selected],
        "--kpts", "6", "6", "6", "--fmax", str(args.fmax), "--max-steps", str(args.max_steps),
        *(["--overwrite"] if args.overwrite else []),
    ]))
    (out / "batch.json").write_text(json.dumps({
        "description": args.description, "eqv3_dir": args.eqv3_dir.as_posix(),
        "handoff_fmax_ev_per_angstrom": args.handoff_fmax, "min_distance_angstrom": args.min_distance,
        "dft_fmax_ev_per_angstrom": args.fmax,
        "tasks": [{"task_index": i, **s} for i, s in enumerate(selected)], "skipped": skipped,
    }, indent=2) + "\n")
    families = sorted({s["defect_family"] for s in selected})
    print(f"Prepared {len(selected)} relaxations ({len(families)} families) in {out}; skipped {len(skipped)}.")
    for entry in skipped:
        print(f"  skipped {entry['defect_config_id']} {entry['candidate_id']}: {entry['reason']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eqv3-dir", type=Path, default=DEFAULT_EQV3_DIR)
    parser.add_argument("--rattled-candidate", default="rattle_seed_101")
    parser.add_argument("--handoff-fmax", type=float, default=0.02)
    parser.add_argument("--min-distance", type=float, default=2.0)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--description", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    prepare(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
