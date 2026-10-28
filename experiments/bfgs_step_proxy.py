"""Count ASE BFGS/LBFGS steps from each VBi_001 start on the EqV3 surface.

Proxy for the DFT-only relaxation cost: same optimizer and settings that the
DFT runner will use (maxstep 0.1 A, stop when every atomic force < 0.01 eV/A),
evaluated on the fine-tuned EqV3 potential instead of QE.  ASE (L)BFGS makes
one force evaluation per step, so DFT SCFs = steps + 1.

Run on a GPU from the repo root: python -m experiments.bfgs_step_proxy [OUTPUT_DIR]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.optimize import BFGS, LBFGS

from relaxgen.prerelax.relax_reconstruction_candidates_eqv3 import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_eqv3
from relaxgen.structures.bi2te3_defect_generation import read_single_structure

CANDIDATES = Path(
    "/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/"
    "04_reconstruction_candidates_preview_no_pure_expand40/VBi/VBi_001"
)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/11_vbi001_bfgs_step_proxy"
)
THRESHOLDS = (1.0, 0.1, 0.05, 0.02, 0.01)

runner = load_eqv3(DEFAULT_CHECKPOINT, DEFAULT_CONFIG, "cuda", 42)
rows = []
for path in sorted(CANDIDATES.glob("*.extxyz")):
    for name, optimizer_class in (("BFGS", BFGS), ("LBFGS", LBFGS)):
        atoms = read_single_structure(path)
        atoms.calc = runner.calculator
        history: list[float] = []

        def observe() -> None:
            history.append(float(np.linalg.norm(atoms.get_forces(), axis=1).max()))

        optimizer = optimizer_class(atoms, maxstep=0.1, logfile=None)
        optimizer.attach(observe, interval=1)
        observe()
        start = time.time()
        converged = optimizer.run(fmax=0.01, steps=1000)
        row = {
            "candidate": path.stem,
            "optimizer": name,
            "converged": bool(converged),
            "steps": int(optimizer.nsteps),
            "force_evaluations": int(optimizer.nsteps) + 1,
            "steps_to_fmax_below": {
                str(t): next((i for i, f in enumerate(history) if f < t), None) for t in THRESHOLDS
            },
            "initial_fmax": history[0],
            "final_fmax": history[-1],
            "final_energy_ev": float(atoms.get_potential_energy()),
            "wall_seconds": time.time() - start,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

(OUT / "bfgs_step_proxy.json").write_text(json.dumps(rows, indent=2) + "\n")
