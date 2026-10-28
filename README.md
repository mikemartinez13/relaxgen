# relaxgen

Generates DFT relaxation trajectories for Bi2Te3 point defects. The pipeline:

1. enumerate defects and build ShakeNBreak reconstruction candidates,
2. pre-relax each start with a uMLIP (EquiformerV3, MACE-ft or UMA),
3. hand off to Quantum ESPRESSO and relax to a DFT force target.

The trajectories are training data for the flow-matching relaxer in
`ConstrDiff_Material`. This code was split out of `ConstrDiff_Material/datagen/`
at commit `68e4e5e`.

## Layout

| Path | Contents |
|---|---|
| `relaxgen/structures/` | defect enumeration, ShakeNBreak candidates |
| `relaxgen/prerelax/` | EqV3 / MACE-ft / UMA relaxers, uMLIP-to-DFT handoff thresholds |
| `relaxgen/dft/` | resumable QE relaxation runner, QE diagnostics |
| `relaxgen/dispatch/` | `qe_dispatch` controller (GPU/CPU backends, scrontab ticks), queue-wait survey, batch preparation |
| `relaxgen/analysis/` | endpoint diversity (symmetry-aware RMSD), figures, SCFs per relaxation, cost analysis |
| `eqv3_fairchem/` | fairchem v1 fork with the EquiformerV3 model, needed to load the EqV3 checkpoint |
| `relaxgen/common/` | QE/MACE calculator helpers copied from ConstrDiff_Material |
| `experiments/` | one-off studies: VBi_001 diversity and its basins, Stage A metrics, Stage B and its symmetry check, resume test, SCF-stall check, BFGS step proxy |
| `slurm/` | Slurm launchers for UVA Afton/Rivanna |
| `pseudos/` | QE pseudopotentials (Bi, Te pbesol rrkjus) |
| `docs/BI2TE3_2VTE_PILOT.md` | campaign notes and results |
| `docs/cost_analysis/` | estimated SCF times, queue waits, SU and days on Rivanna |
| `benchmarks/qe_timing/` | QE timing test for other clusters, with Rivanna reference results |

## Running

Run modules from the repository root with `-m`, for example:

```bash
python -m relaxgen.dispatch.qe_dispatch tick --dispatch-dir DISPATCH_DIR
sbatch --array=0 slurm/run_dft_relax_cpu_slurm_uva.sh RELAX_MANIFEST.json
```

Conda environments (files in `conda_envs/`):

- `snb`: the controller, QE runners (QE itself comes from the cluster), ShakeNBreak, analysis.
- `fairchemv1`: EqV3 and MACE-ft.
- `fairchemv2`: UMA, run without `PYTHONPATH`.

```bash
conda env create -f conda_envs/snb.yml
conda env create -f conda_envs/fairchemv2.yml
conda env create -f conda_envs/fairchemv1.yml
conda activate fairchemv1
pip install --no-deps mace-torch==0.3.15 -e eqv3_fairchem
```

The pinned versions are what each env loaded on UVA, including packages it
picked up from `~/.local`. mace-torch is installed separately because it pins
`e3nn==0.4.4` while EqV3 needs e3nn 0.6.0. On UVA the `fairchemv1` env loads
EqV3 from the ConstrDiff_Material submodule, which is identical to `eqv3_fairchem/`.

Tests (run with the `snb` env):

```bash
python -m unittest discover -s tests -t .
```

## External dependencies

These still live outside this repository:

- MACE checkpoints: `ConstrDiff_Material/model_checkpoints/mace`. Override with `CONSTR_DIFF_ROOT`.
- EqV3 and MACE-ft fine-tuned checkpoints: `ft_ckpts/` (git-ignored; copy it with rsync). Override with `RELAXGEN_CKPT_DIR`.
- Campaign data: `/scratch/gqs7zr/relaxgen` and `/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot`.

Campaigns started before the split still run from `ConstrDiff_Material/datagen/`. Their dispatch configs name `datagen/...` launchers.
