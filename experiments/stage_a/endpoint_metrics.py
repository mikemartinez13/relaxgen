"""Metrics on the saved Stage A RMSD matrices.

Run from the repo root: python -m experiments.stage_a.endpoint_metrics DIR (the endpoint_rmsd_matrices output)
"""
import json, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from relaxgen.analysis.analyze_umlip_endpoint_diversity import complete_linkage_labels

D = Path(sys.argv[1])
meta = json.load(open(D / "stagea_meta.json"))
mats = np.load(D / "stagea_matrices.npz")
MODELS = ("eqv3", "mace_ft", "uma")
PAIRS = list(combinations(MODELS, 2))
out = {}

# 1. Per-start endpoint RMSD between models.
pair_rmsd = defaultdict(list)
for c, rows in meta.items():
    m = mats[c]; idx = {(r[0], r[1]): i for i, r in enumerate(rows)}
    for cand in {r[1] for r in rows}:
        for a, b in PAIRS:
            if (a, cand) in idx and (b, cand) in idx:
                pair_rmsd[f"{a}|{b}"].append(m[idx[a, cand], idx[b, cand]])
edges = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 99]
print("Per-start endpoint RMSD between models")
print(f"{'pair':16s} {'n':>5s} {'p10':>6s} {'p50':>6s} {'p90':>6s}  " + " ".join(f"{lo}-{hi}" for lo, hi in zip(edges, edges[1:])))
for k, v in pair_rmsd.items():
    v = np.array(v); h = np.histogram(v, edges)[0] / len(v)
    print(f"{k:16s} {len(v):5d} {np.percentile(v,10):6.3f} {np.median(v):6.3f} {np.percentile(v,90):6.3f}  " + " ".join(f"{x:6.1%}" for x in h))
    out[k] = {"p10": float(np.percentile(v, 10)), "p50": float(np.median(v)), "hist": dict(zip(map(str, edges[:-1]), h.tolist()))}

# 2. Diversity gain per configuration over EqV3 alone and over the best single model.
for t in (0.1, 0.2, 0.3):
    gain_eqv3, gain_best, configs_gain, per_model, pooled = [], [], 0, defaultdict(int), 0
    for c, rows in meta.items():
        m = mats[c]; models = [r[0] for r in rows]
        def n(sub):
            ix = [i for i, x in enumerate(models) if x in sub]
            return len(set(complete_linkage_labels(m[np.ix_(ix, ix)], t).tolist())) if ix else 0
        single = {x: n([x]) for x in MODELS}; allc = n(MODELS)
        for x in MODELS: per_model[x] += single[x]
        pooled += allc
        gain_eqv3.append(allc - single["eqv3"]); gain_best.append(allc - max(single.values()))
        configs_gain += allc > single["eqv3"]
    print(f"\n@{t} A: per-model total {dict(per_model)}, pooled {pooled}; "
          f"pooled - eqv3 = {sum(gain_eqv3)} (+{sum(gain_eqv3)/per_model['eqv3']:.0%}), "
          f"configs where pooling adds a basin over eqv3: {configs_gain}/{len(meta)}, "
          f"pooled - best single = {sum(gain_best)}")
    out[f"gain_{t}"] = {"per_model": dict(per_model), "pooled": pooled, "configs_gain_over_eqv3": configs_gain}

# 3. Do the models agree on the lowest-energy basin, in configurations with several basins?
t = 0.2
agree = defaultdict(lambda: [0, 0]); concord = defaultdict(lambda: [0, 0])
for c, rows in meta.items():
    m = mats[c]; models = np.array([r[0] for r in rows]); E = np.array([r[2] for r in rows])
    cands = [r[1] for r in rows]
    labels = complete_linkage_labels(m, t)
    if len(set(labels.tolist())) < 2:
        continue
    gs = {}
    for x in MODELS:
        ix = np.flatnonzero(models == x)
        if len(ix): gs[x] = labels[ix[np.argmin(E[ix])]]
    for a, b in PAIRS:
        if a in gs and b in gs:
            agree[f"{a}|{b}"][0] += gs[a] == gs[b]; agree[f"{a}|{b}"][1] += 1
    # Pairwise ordering of distinct basins: for two starts whose endpoints differ by >0.2 A
    # under both models and >10 meV under both, does the sign of dE agree?
    pos = {(x, cd): i for i, (x, cd) in enumerate(zip(models, cands))}
    for a, b in PAIRS:
        common = sorted({cd for x, cd in pos if x == a} & {cd for x, cd in pos if x == b})
        for p, q in combinations(common, 2):
            ia, ja, ib, jb = pos[a, p], pos[a, q], pos[b, p], pos[b, q]
            if m[ia, ja] > t and m[ib, jb] > t:
                da, db = E[ia] - E[ja], E[ib] - E[jb]
                if abs(da) > 0.01 and abs(db) > 0.01:
                    concord[f"{a}|{b}"][0] += np.sign(da) == np.sign(db); concord[f"{a}|{b}"][1] += 1
print(f"\nLowest-energy basin agreement (configs with >=2 pooled basins at {t} A):")
for k, (y, n_) in agree.items(): print(f"  {k:16s} {y}/{n_} = {y/n_:.0%}")
print("Ordering agreement for pairs of distinct endpoints (>0.2 A, |dE|>10 meV in both models):")
for k, (y, n_) in concord.items(): print(f"  {k:16s} {y}/{n_} = {y/n_:.0%}")
out["ground_state_agreement"] = {k: v for k, v in agree.items()}; out["ordering_concordance"] = {k: v for k, v in concord.items()}
json.dump(out, open(D / "stagea_metrics.json", "w"), indent=1, default=int)
