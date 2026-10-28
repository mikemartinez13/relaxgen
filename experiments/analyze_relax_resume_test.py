#!/usr/bin/env python3
"""Detect restart discontinuities in resumable DFT relaxations.

Compares arms of ``13_dft_relax_resume_test`` (or any directory of
``dft_relaxation_campaign.py`` campaigns over the same structures):

* resume check: energy and force change when a resumed segment recomputes the
  last stored geometry (a pure restart discontinuity; it should be ~0);
* step continuity: the first BFGS step after each resume compared with the
  median of the three steps before it (a reset optimizer takes a different
  step);
* trajectory agreement with a continuous reference arm, against the noise
  floor between two continuous arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase.io.trajectory import Trajectory


def load_arm(arm_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((arm_dir / "relax_manifest.json").read_text())
    runs = {}
    for task in manifest["tasks"]:
        status_path = Path(task["status"])
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        trajectory = Path(task["trajectory"])
        frames = list(Trajectory(trajectory)) if trajectory.exists() and trajectory.stat().st_size else []
        runs[task["label"]] = {"status": status, "frames": frames}
    return runs


def step_sizes(frames: list) -> np.ndarray:
    """Largest atomic displacement of each BFGS step (frame k-1 -> k)."""

    return np.array(
        [np.linalg.norm(frames[k].positions - frames[k - 1].positions, axis=1).max() for k in range(1, len(frames))]
    )


def rmsd(a, b) -> float:
    return float(np.sqrt(np.mean(np.sum((a.positions - b.positions) ** 2, axis=1))))


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    status, frames = run["status"], run["frames"]
    energies = np.array([f.get_potential_energy() for f in frames]) if frames else np.array([])
    steps = step_sizes(frames) if len(frames) > 1 else np.array([])
    boundaries = []
    for segment in status.get("segments", [])[1:]:
        b = int(segment.get("frames_before") or 0)
        # Step b-1 -> b is the first one taken after this resume.
        if 4 <= b < len(frames):
            before = float(np.median(steps[b - 4 : b - 1]))
            boundaries.append(
                {
                    "frame": b,
                    "backends": f"{status['segments'][status['segments'].index(segment) - 1]['backend']}->{segment['backend']}",
                    "first_step_angstrom": float(steps[b - 1]),
                    "median_prior_step_angstrom": before,
                    "step_ratio": float(steps[b - 1] / before) if before > 0 else None,
                    "energy_change_mev": 1.0e3 * float(energies[b] - energies[b - 1]),
                    **{k: v for k, v in (segment.get("resume_check") or {}).items()},
                }
            )
    return {
        "execution_status": status.get("execution_status"),
        "converged": status.get("scientifically_converged"),
        "segments": len(status.get("segments", [])),
        "frames": len(frames),
        "final_energy_ev": float(energies[-1]) if len(energies) else None,
        "final_fmax": status.get("final_max_force_ev_per_angstrom"),
        "energy_increases": int((np.diff(energies) > 1.0e-6).sum()) if len(energies) > 1 else 0,
        "resumes": boundaries,
    }


def compare(run, reference) -> dict[str, Any]:
    a, b = run["frames"], reference["frames"]
    n = min(len(a), len(b))
    if not n:
        return {}
    per_frame = [rmsd(a[k], b[k]) for k in range(n)]
    diverged = next((k for k, value in enumerate(per_frame) if value > 1.0e-3), None)
    return {
        "frames_compared": n,
        "first_frame_rmsd_above_1e-3_A": diverged,
        "max_rmsd_over_common_frames_A": float(max(per_frame)),
        "final_structure_rmsd_A": rmsd(a[-1], b[-1]),
        "final_energy_difference_mev": 1.0e3 * float(a[-1].get_potential_energy() - b[-1].get_potential_energy()),
        "frame_count_difference": len(a) - len(b),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--reference", default="continuous_a")
    parser.add_argument("--replicate", default="continuous_b")
    args = parser.parse_args()

    arms = {p.name: load_arm(p) for p in sorted(args.test_dir.iterdir()) if (p / "relax_manifest.json").exists()}
    reference = arms[args.reference]
    report: dict[str, Any] = {"reference": args.reference, "arms": {}}
    for arm, runs in arms.items():
        report["arms"][arm] = {}
        for label, run in runs.items():
            entry = summarize_run(run)
            if arm != args.reference and label in reference:
                entry["vs_reference"] = compare(run, reference[label])
            report["arms"][arm][label] = entry
    (args.test_dir / "resume_test_report.json").write_text(json.dumps(report, indent=2) + "\n")

    for arm, runs in report["arms"].items():
        print(f"\n== {arm}")
        for label, entry in runs.items():
            ref = entry.get("vs_reference", {})
            print(
                f"  {label:12s} {entry['execution_status'] or 'pending':18s} segments={entry['segments']:2d} "
                f"frames={entry['frames']:3d} fmax={entry['final_fmax'] or float('nan'):.4f} "
                f"E-rises={entry['energy_increases']}  "
                + (f"vs ref: final RMSD={ref['final_structure_rmsd_A']:.2e} A dE={ref['final_energy_difference_mev']:+.3f} meV "
                   f"diverge@{ref['first_frame_rmsd_above_1e-3_A']}" if ref else "")
            )
            for r in entry["resumes"]:
                print(
                    f"      resume@{r['frame']:3d} {r['backends']:9s} recompute dE={r.get('energy_difference_mev', float('nan')):+.4f} meV "
                    f"max|dF|={r.get('max_atom_force_difference_ev_per_angstrom', float('nan')):.2e} eV/A "
                    f"step ratio={r['step_ratio'] if r['step_ratio'] is None else round(r['step_ratio'], 2)}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
