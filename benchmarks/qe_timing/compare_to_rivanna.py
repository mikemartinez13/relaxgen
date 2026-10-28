#!/usr/bin/env python3
"""Compare a QE timing run on another cluster with the Rivanna results.

Reads the relaxation manifest of the run (made by ``dft_relaxation_campaign
prepare`` from the structures in ``structures/``), and for each task prints the
device, the SCF times next to Rivanna's, and how far the energies and forces are
from Rivanna's.  The first SCF must agree within the same tolerances the
dispatch controller uses for a new backend: 0.1 meV in energy and 0.001 eV/Å in
every force component.

    python benchmarks/qe_timing/compare_to_rivanna.py RUN_DIR/relax_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

REFERENCE = Path(__file__).resolve().parent / "rivanna_reference.json"
ENERGY_TOLERANCE_MEV = 0.1
FORCE_TOLERANCE = 0.001


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="relax_manifest.json of the run to compare.")
    args = parser.parse_args()
    reference = json.loads(REFERENCE.read_text())["structures"]
    manifest = json.loads(args.manifest.read_text())
    all_agree = True
    for task in manifest["tasks"]:
        label = task["label"]
        if label not in reference:
            continue
        ref = reference[label]
        status_path = Path(task["status"])
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        trajectory = Path(status.get("trajectory", ""))
        if not status.get("segments") or not trajectory.is_file() or trajectory.stat().st_size == 0:
            print(f"{label}: no finished SCF yet")
            continue
        frames = read(trajectory, ":")
        steps = [s / 60 for segment in status["segments"] for s in segment.get("step_wall_seconds") or []]
        devices = sorted({segment.get("allocated_device", "?") for segment in status["segments"]})
        print(f"\n{label} ({len(frames)} SCFs on {', '.join(devices)})")

        later = f", later SCFs median {np.median(steps[1:]):.1f} min" if len(steps) > 1 else ""
        print(f"  this run: first SCF {steps[0]:.1f} min{later}")
        if "first_scf_minutes" in ref:
            for hardware, minutes in ref["first_scf_minutes"].items():
                print(f"  Rivanna {hardware}: first SCF {minutes:.1f} min")
        if "step_minutes" in ref:
            rivanna = ref["step_minutes"]
            print(f"  Rivanna {ref['device']}: first SCF {rivanna[0]:.1f} min, "
                  f"later SCFs median {np.median(rivanna[1:]):.1f} min")

        energy = frames[0].get_potential_energy()
        reference_energy = ref.get("first_scf_energy_ev", (ref.get("energies_ev") or [None])[0])
        energy_mev = abs(energy - reference_energy) * 1000
        force = np.abs(frames[0].get_forces() - np.array(ref["first_scf_forces_ev_per_angstrom"])).max()
        agree = energy_mev <= ENERGY_TOLERANCE_MEV and force <= FORCE_TOLERANCE
        all_agree &= agree
        print(f"  first SCF vs Rivanna: |dE| = {energy_mev:.4f} meV, max |dF| = {force:.2e} eV/Å "
              f"-> {'agrees' if agree else 'DOES NOT AGREE'}")
        if "energies_ev" in ref and len(frames) > 1:
            shared = min(len(frames), len(ref["energies_ev"]))
            drift = max(abs(frames[i].get_potential_energy() - ref["energies_ev"][i]) * 1000 for i in range(shared))
            print(f"  largest energy difference over the {shared} shared steps: {drift:.3f} meV")
    print("\nAll first SCFs agree with Rivanna." if all_agree else "\nSome first SCFs differ from Rivanna; check the QE build.")
    return 0 if all_agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
