"""Group the VBi_001 diversity campaign's DFT endpoints into basins and print their energies.

Run from the repo root: python -m experiments.diversity_basins EXPERIMENT_DIR
(EXPERIMENT_DIR is .../15_vbi001_umlip_dft_diversity)
"""
import json, sys, numpy as np
from experiments import umlip_dft_diversity as u
from relaxgen.prerelax.umlip_ensemble_diversity import endpoint_rmsd
D = sys.argv[1]
exp = json.load(open(f"{D}/experiment.json")); man = json.load(open(exp["dft_manifest"]))
by = {int(t["task_index"]): t for t in man["tasks"]}
runs = [u._load_run(t, by[t["task_index"]]) for t in exp["tasks"] if not by[t["task_index"]].get("withdrawn")]
runs = [r for r in runs if r["frames"]]
last = [r["frames"][-1] for r in runs]
E = np.array([f.get_potential_energy() for f in last]); E0 = E.min()
n = len(runs); M = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        M[i, j] = M[j, i] = endpoint_rmsd(last[i], last[j])
# greedy basin labels: same basin if RMSD < 0.03 A to the basin's first member
labels = [-1] * n; reps = []
for i in np.argsort(E):
    for b, rep in enumerate(reps):
        if M[i, rep] < 0.03:
            labels[i] = b; break
    else:
        labels[i] = len(reps); reps.append(i)
print(f"{'run':30s} {'done':5s} {'SCFs':>4s} {'fmax':>7s} {'E-Emin meV':>10s} basin")
for i in np.argsort(E):
    r = runs[i]
    print(f"{r['label']:30s} {str(r['converged']):5s} {r['scf_count']:4d} {r['current_fmax']:7.4f} {1e3*(E[i]-E0):10.1f} {labels[i]}")
print("basin reps RMSD matrix:"); np.set_printoptions(precision=3, suppress=True); print(M[np.ix_(reps, reps)])
