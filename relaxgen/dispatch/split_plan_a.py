#!/usr/bin/env python3
"""Split the rest of Plan A between Rivanna and the CS cluster.

Plan A relaxes every ShakeNBreak candidate once with DFT, starting from its
EqV3 handoff frame (the same rule as batch 1, ``handoff_frame`` in
``prepare_relaxgen_batch.py``).  This script takes every candidate with a
handoff frame that is not already in batch 1 and splits them:

- ``--cs-count`` of them go to the CS cluster, drawn at random within each
  defect family (each family gives the same share, rounded to whole tasks).
  They are written to a bundle folder to copy over: one float64 extxyz per
  starting frame, ``labels.txt``, ``batch.json`` and ``prepare_cs.sh``, which
  writes the CS manifest with batch 1's protocol.
- The rest become the next Rivanna batch, prepared here.

Candidates whose EqV3 relaxation never reached the handoff force (its step
limit ran out) start from their lowest-force frame with safe distances, unless
that force is above ``--fallback-max-fmax`` (EqV3 likely failed).  They go to
Rivanna only, after the CS draw, so the CS share does not depend on them.

Both lists are shuffled with the same seed, so the dispatcher works through
every family at once and a partly finished campaign is a fair sample.

    python -m relaxgen.dispatch.split_plan_a \
      --exclude-manifest /scratch/gqs7zr/relaxgen/trajectories/batch01_eqv3_handoff/relax_manifest.json \
      --rivanna-dir /scratch/gqs7zr/relaxgen/trajectories/batch02_eqv3_handoff \
      --cs-bundle-dir /scratch/gqs7zr/relaxgen/trajectories/cs_batch01_eqv3_handoff_bundle
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase.io import iread

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relaxgen.dispatch.prepare_relaxgen_batch import DEFAULT_EQV3_DIR, handoff_frame

CS_PREPARE = """#!/bin/bash
# Write the CS-cluster relaxation manifest for this bundle with batch 1's protocol.
# Run from the relaxgen repository root, with the snb environment active:
#   bash {bundle_name}/prepare_cs.sh [OUTPUT_DIR]
# OUTPUT_DIR defaults to {campaign_name} next to this bundle.
set -euo pipefail
BUNDLE=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${{1:-$(dirname "${{BUNDLE}}")/{campaign_name}}}
mapfile -t LABELS < "${{BUNDLE}}/labels.txt"
STRUCTURES=("${{LABELS[@]/#/${{BUNDLE}}/structures/}}")
python -m relaxgen.dft.dft_relaxation_campaign prepare --output-dir "${{OUTPUT_DIR}}" \\
  --structures "${{STRUCTURES[@]/%/.extxyz}}" --labels "${{LABELS[@]}}" \\
  --kpts 6 6 6 --fmax {fmax} --max-steps {max_steps}
"""


def family_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` over families in proportion to ``counts`` (largest remainder)."""

    available = sum(counts.values())
    if not 0 <= total <= available:
        raise ValueError(f"Cannot draw {total} of {available} candidates.")
    exact = {family: total * count / available for family, count in counts.items()}
    quotas = {family: int(value) for family, value in exact.items()}
    by_remainder = sorted(counts, key=lambda family: (quotas[family] - exact[family], family))
    for family in by_remainder[: total - sum(quotas.values())]:
        quotas[family] += 1
    return quotas


def split(candidates: list[dict[str, Any]], cs_count: int, seed: int) -> tuple[list, list]:
    """Return (rivanna, cs) lists; each is shuffled, and they never share a label."""

    rng = np.random.default_rng(seed)
    by_family: dict[str, list] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda c: c["label"]):
        by_family[candidate["defect_family"]].append(candidate)
    quotas = family_quotas({family: len(items) for family, items in by_family.items()}, cs_count)
    rivanna, cs = [], []
    for family in sorted(by_family):
        items = by_family[family]
        chosen = set(rng.choice(len(items), size=quotas[family], replace=False).tolist())
        cs += [item for i, item in enumerate(items) if i in chosen]
        rivanna += [item for i, item in enumerate(items) if i not in chosen]
    return [rivanna[i] for i in rng.permutation(len(rivanna))], [cs[i] for i in rng.permutation(len(cs))]


def lowest_force_frame(trajectory: Path, min_distance: float) -> dict[str, Any] | None:
    """The frame with the lowest ML fmax among frames with no pair closer than ``min_distance``."""

    best = None
    for index, atoms in enumerate(iread(trajectory, index=":")):
        fmax = float(atoms.info.get("ml_fmax_ev_per_angstrom", np.inf))
        distance = float(atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)].min())
        if distance >= min_distance and (best is None or fmax < best["ml_fmax"]):
            best = {"frame": index, "ml_fmax": fmax, "min_distance": distance,
                    "optimizer_step": int(atoms.info.get("optimizer_step", index))}
    return best


def add_fallbacks(rivanna: list, skipped: list, max_fmax: float, min_distance: float, seed: int) -> tuple[list, list]:
    """Insert skipped candidates that have a usable lowest-force frame into ``rivanna`` at random places."""

    added, still_skipped = [], []
    for entry in skipped:
        frame = lowest_force_frame(Path(entry["eqv3_trajectory"]), min_distance)
        if frame is None or frame["ml_fmax"] > max_fmax:
            still_skipped.append({**entry, "reason": f"no frame with min distance >= {min_distance} A and "
                                                     f"ML fmax <= {max_fmax}"})
            continue
        candidate = {key: value for key, value in entry.items() if key != "reason"}
        added.append({**candidate, "handoff": {**frame, "rule": "lowest_force_fallback"},
                      "structure": f"{entry['eqv3_trajectory']}@{frame['frame']}"})
    # A separate generator, so the CS draw and Rivanna's order do not depend on this step.
    rng = np.random.default_rng([seed, 1])
    merged = list(rivanna)
    for candidate in added:
        merged.insert(int(rng.integers(len(merged) + 1)), candidate)
    return merged, still_skipped


def plan_a_candidates(eqv3_dir: Path, handoff_fmax: float, min_distance: float) -> tuple[list, list]:
    selected, skipped = [], []
    for path in sorted(eqv3_dir.glob("*/*/*/status.json")):
        status = json.loads(path.read_text())
        trajectory = Path(status["trajectory"])
        record = {"defect_family": status["defect_family"], "defect_config_id": status["defect_config_id"],
                  "candidate_id": status["candidate_id"], "eqv3_converged": status["converged"],
                  "eqv3_fallback_used": bool(status.get("fallback_used")), "eqv3_trajectory": trajectory.as_posix(),
                  "label": f"{status['defect_config_id']}__{status['candidate_id']}"}
        handoff = handoff_frame(trajectory, handoff_fmax, min_distance)
        if handoff is None:
            skipped.append({**record, "reason": f"no frame with ML fmax < {handoff_fmax} "
                                                f"and min distance >= {min_distance} A"})
        else:
            selected.append({**record, "handoff": handoff, "structure": f"{trajectory}@{handoff['frame']}"})
    return selected, skipped


def write_cs_bundle(bundle: Path, tasks: list[dict[str, Any]], record: dict[str, Any], args) -> None:
    from relaxgen.dft.dft_relaxation_campaign import read_structure_spec
    from relaxgen.structures.bi2te3_defect_generation import geometry_sha256, write_extxyz_float64

    from ase.io import read

    for task in tasks:
        atoms = read_structure_spec(task["structure"])
        path = bundle / "structures" / f"{task['label']}.extxyz"
        write_extxyz_float64(path, atoms)
        task["source_geometry_sha256"] = geometry_sha256(atoms)
        if geometry_sha256(read_structure_spec(path.as_posix())) != task["source_geometry_sha256"]:
            raise RuntimeError(f"{path} does not reproduce its source geometry.")
    (bundle / "labels.txt").write_text("".join(f"{task['label']}\n" for task in tasks))
    (bundle / "batch.json").write_text(json.dumps(
        {**record, "tasks": [{"task_index": i, **task} for i, task in enumerate(tasks)]}, indent=2) + "\n")
    script = bundle / "prepare_cs.sh"
    script.write_text(CS_PREPARE.format(bundle_name=bundle.name, campaign_name=args.cs_campaign_name,
                                        fmax=args.fmax, max_steps=args.max_steps))
    script.chmod(0o755)


def run(args: argparse.Namespace) -> None:
    from relaxgen.dft import dft_relaxation_campaign as campaign

    excluded = set()
    for manifest in args.exclude_manifest:
        excluded |= {task["label"] for task in json.loads(manifest.read_text())["tasks"]}
    selected, skipped = plan_a_candidates(args.eqv3_dir.expanduser().resolve(), args.handoff_fmax,
                                          args.min_distance)
    remaining = [candidate for candidate in selected if candidate["label"] not in excluded]
    rivanna, cs = split(remaining, args.cs_count, args.seed)
    fallback_labels = {entry["label"] for entry in skipped} - excluded
    rivanna, skipped = add_fallbacks(rivanna, [e for e in skipped if e["label"] in fallback_labels],
                                     args.fallback_max_fmax, args.min_distance, args.seed)

    bundle = args.cs_bundle_dir.expanduser().resolve()
    if bundle.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {bundle}; pass --overwrite.")
    record = {
        "description": args.description, "eqv3_dir": args.eqv3_dir.as_posix(),
        "excluded_manifests": [m.as_posix() for m in args.exclude_manifest], "excluded_count": len(excluded),
        "seed": args.seed, "handoff_fmax_ev_per_angstrom": args.handoff_fmax,
        "min_distance_angstrom": args.min_distance, "dft_fmax_ev_per_angstrom": args.fmax,
        "fallback_max_fmax_ev_per_angstrom": args.fallback_max_fmax,
        "fallback_count": sum(t["handoff"].get("rule") == "lowest_force_fallback" for t in rivanna),
        "rivanna_count": len(rivanna), "cs_count": len(cs), "skipped": skipped,
    }

    out = args.rivanna_dir.expanduser().resolve()
    campaign.prepare(campaign.build_parser().parse_args([
        "prepare", "--output-dir", str(out),
        "--structures", *[s["structure"] for s in rivanna], "--labels", *[s["label"] for s in rivanna],
        "--kpts", "6", "6", "6", "--fmax", str(args.fmax), "--max-steps", str(args.max_steps),
        *(["--overwrite"] if args.overwrite else []),
    ]))
    (out / "batch.json").write_text(json.dumps(
        {**record, "cluster": "rivanna", "tasks": [{"task_index": i, **s} for i, s in enumerate(rivanna)]},
        indent=2) + "\n")
    write_cs_bundle(bundle, cs, {**record, "cluster": "cs"}, args)

    print(f"{len(selected)} candidates have a handoff frame; {record['fallback_count']} more start from their "
          f"lowest-force frame (Rivanna); {len(skipped)} left out; {len(excluded)} excluded as already prepared.")
    print(f"Rivanna: {len(rivanna)} relaxations in {out}")
    print(f"CS cluster: {len(cs)} relaxations in {bundle}")
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for column, tasks in enumerate((rivanna, cs)):
        for task in tasks:
            counts[task["defect_family"]][column] += 1
    for family in sorted(counts):
        print(f"  {family:15s} Rivanna {counts[family][0]:4d}  CS {counts[family][1]:4d}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exclude-manifest", type=Path, nargs="+", required=True,
                        help="Relaxation manifests whose labels are already prepared (batch 1).")
    parser.add_argument("--rivanna-dir", type=Path, required=True)
    parser.add_argument("--cs-bundle-dir", type=Path, required=True)
    parser.add_argument("--cs-campaign-name", default="cs_batch01_eqv3_handoff",
                        help="Default name of the CS manifest folder, next to the bundle.")
    parser.add_argument("--cs-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--eqv3-dir", type=Path, default=DEFAULT_EQV3_DIR)
    parser.add_argument("--handoff-fmax", type=float, default=0.02)
    parser.add_argument("--min-distance", type=float, default=2.0)
    parser.add_argument("--fallback-max-fmax", type=float, default=5.0,
                        help="Highest ML fmax (eV/A) a candidate without a handoff frame may start from.")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--description", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
