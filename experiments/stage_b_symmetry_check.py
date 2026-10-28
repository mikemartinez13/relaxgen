"""Compare Stage B's ML endpoint RMSD with a symmetry-aware RMSD (pymatgen StructureMatcher).

Run from the repo root: python -m experiments.stage_b_symmetry_check
"""
import json, sys
from collections import defaultdict
from itertools import combinations
import numpy as np
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
sel = json.load(open("/scratch/gqs7zr/relaxgen/umlip_ensemble/stage_b/stage_b.json"))
sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False, scale=False, attempt_supercell=False)
groups = defaultdict(dict)
for t in sel["tasks"]:
    groups[t["role"], t["defect_config_id"], t["candidate_id"]][t["model"]] = t
for (role, config, cand), arms in groups.items():
    ends = {}
    for m, t in arms.items():
        with Trajectory(t["ml_trajectory"]) as r:
            ends[m] = AseAtomsAdaptor.get_structure(r[-1])
    out = []
    for a, b in combinations(sorted(ends), 2):
        rms = sm.get_rms_dist(ends[a], ends[b])
        # rms is normalized by (V/n)^(1/3); convert to Angstrom
        scale = (ends[a].volume / len(ends[a])) ** (1 / 3)
        ml = arms[a]["ml_pair_rmsd"][f"{a}|{b}"]
        out.append(f"{a}|{b} ml={ml:.2f} sym={'none' if rms is None else f'{rms[0]*scale:.2f}'}")
    print(role[:4], config.ljust(13), cand[:34].ljust(34), " ".join(out), flush=True)
