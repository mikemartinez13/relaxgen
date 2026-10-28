"""Stage A: endpoint RMSD matrices for every defect configuration (input to endpoint_metrics).

Run from the repo root: python -m experiments.stage_a.endpoint_rmsd_matrices OUTPUT_DIR
"""
import json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from ase.io.trajectory import Trajectory
from relaxgen.prerelax import umlip_ensemble_diversity as ued

ROOT = Path("/scratch/gqs7zr/relaxgen/umlip_ensemble")
OUT = Path(sys.argv[1])
MODELS = ("eqv3", "mace_ft", "uma")

def load(config_dir):
    runs = []
    for model in MODELS:
        for p in sorted((ROOT / model).glob(f"*/{config_dir}/*.json")):
            r = json.loads(p.read_text()); r["model"] = model; runs.append(r)
    return runs

def one(config):
    runs = [r for r in load(config) if r["converged"] and not r["unsafe_geometry"]]
    ends = []
    for r in runs:
        with Trajectory(r["trajectory"]) as t:
            ends.append(t[-1])
    m = ued._rmsd_matrix(ends)
    return config, [(r["model"], r["candidate_id"], r["final_energy_ev"]) for r in runs], m

if __name__ == "__main__":
    configs = sorted({p.parent.name for p in (ROOT / "eqv3").glob("*/*/*.json")})
    with Pool(24) as pool:
        res = pool.map(one, configs)
    np.savez_compressed(OUT / "stagea_matrices.npz", **{c: m for c, _, m in res})
    json.dump({c: meta for c, meta, _ in res}, open(OUT / "stagea_meta.json", "w"))
    print("saved", len(res))
