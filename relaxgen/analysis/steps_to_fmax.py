#!/usr/bin/env python3
"""Count the DFT steps (SCFs) each relaxation needed to reach each force target.

Reads every trajectory of the DFT relaxation campaigns and records, per
relaxation, the first step at which the largest atomic force drops below each
threshold in ``THRESHOLDS``.  Relaxations that have not reached a threshold yet
get ``None`` for it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

THRESHOLDS = (0.05, 0.02, 0.01, 0.005, 0.001)
MODELS = ("mace_ft", "uma", "eqv3", "dftonly")
CAMPAIGNS = {
    "diversity": "/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/15_vbi001_umlip_dft_diversity/dft/relax_manifest.json",
    "batch1": "/scratch/gqs7zr/relaxgen/trajectories/batch01_eqv3_handoff/relax_manifest.json",
    "stage_b": "/scratch/gqs7zr/relaxgen/umlip_ensemble/stage_b/relax_manifest.json",
}


def model_of(label: str) -> str:
    """Which pre-relaxer a task started from, read from its label (batch 1 is all EqV3)."""
    for model in MODELS:
        if model in label:
            return model
    return "eqv3"


def scan(campaigns: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for campaign, manifest in campaigns.items():
        for task in json.loads(Path(manifest).read_text())["tasks"]:
            if task.get("withdrawn"):
                continue
            status = json.loads(Path(task["status"]).read_text()) if os.path.exists(task["status"]) else {}
            trajectory = status.get("trajectory")
            if not trajectory or not os.path.exists(trajectory):
                continue
            try:
                frames = read(trajectory, ":")
            except Exception as error:  # a trajectory being written right now
                print(f"skipping {trajectory}: {error}", file=sys.stderr)
                continue
            fmax = [float(np.linalg.norm(frame.get_forces(), axis=1).max()) for frame in frames]
            rows.append({
                "campaign": campaign,
                "label": task["label"],
                "model": model_of(task["label"]),
                "scf_count": len(fmax),
                "current_fmax": fmax[-1],
                "converged": status.get("scientifically_converged") is True,
                "first_below": {str(t): next((i + 1 for i, f in enumerate(fmax) if f < t), None) for t in THRESHOLDS},
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write.")
    args = parser.parse_args()
    rows = scan(CAMPAIGNS)
    args.output.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{len(rows)} trajectories -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
