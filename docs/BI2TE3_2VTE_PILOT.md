# Bi2Te3 pristine-to-2VTe pilot

Michael, this workflow deliberately stops after constructing the 14 unrelaxed,
symmetry-inequivalent Bi16Te22 vacancy-pair structures. It does not generate
ShakeNBreak distortions, run a uMLIP, or run QE automatically.

## Numerical and scientific contract

- The inferred and canonical geometries are represented as NumPy `float64`.
- Extended XYZ geometry is written with 17 significant digits and checked for
  exact `float64` round trips.
- The prepared QE geometry is also written with 17 significant digits.
- The QE job is `calculation='relax'` with BFGS ions and no cell dynamics, so
  the lattice is fixed.
- The QE validator allows harmless text/float serialization differences. It
  compares the cells numerically, performs a species-constrained MIC atom
  mapping, measures RMSD and maximum displacement, and requires recovery of
  space group R-3m (No. 166) with the 16 Bi, 16 TeA, and 8 TeB site orbits.
- Once those checks pass, the canonical file uses the exact inferred
  `float64` cell, the relaxed QE coordinates, and the inferred site ordering.

Do not convert crystallographic geometry to `float32`: `float32` has less
precision than `float64`. A later uMLIP calculator may use `float32` if that is
the model's native dtype, but that is outside this pilot.

## 0. Environment

The existing environment is ready on the current system:

```bash
conda activate snb
python --version
```

It currently supplies Python 3.11 plus ASE, pymatgen, spglib, SciPy,
ShakeNBreak, doped, and DScribe. If Michael needs to recreate it elsewhere:

```bash
conda create --name snb python=3.11
conda activate snb
python -m pip install \
  numpy==2.4.6 scipy==1.17.1 ase==3.29.0 \
  pymatgen==2026.5.4 spglib==2.7.0 \
  shakenbreak==3.4.4 doped==3.2.1 dscribe==2.1.2
```

Core stages A-H use ASE, NumPy, SciPy, pymatgen, and spglib. ShakeNBreak and
doped are intentionally not invoked yet.

## 1. Create the staged folders

Run these commands from the repository root:

```bash
PILOT_ROOT=/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot
SNB_PYTHON=/home/gqs7zr/.conda/envs/snb/bin/python

mkdir -p "${PILOT_ROOT}/00_inferred"
mkdir -p "${PILOT_ROOT}/01_pristine_qe/state"
mkdir -p "${PILOT_ROOT}/02_canonical"
mkdir -p "${PILOT_ROOT}/03_unrelaxed_2vte"
```

## 2. Export the inferred pristine parent

```bash
"${SNB_PYTHON}" relaxgen/structures/bi2te3_defect_generation.py export-inferred \
  --source-db /scratch/gqs7zr/ai4mat/equiformer_v3/data/all_bi2te3_defects.db \
  --output-dir "${PILOT_ROOT}/00_inferred"
```

This recovers one 40-site Bi16Te24 parent from the exact shared site grid in
the source database. The output includes the high-precision extxyz, a CIF for
inspection, and a manifest containing the source audit, species votes, hashes,
space group, and site orbits.

## 3. Optional: prepare the pristine QE input without running it

The Slurm launcher in Stage 4 now performs this preparation automatically.
Michael only needs this standalone command when he wants to inspect the `.pwi`
before submitting a job:

```bash
"${SNB_PYTHON}" relaxgen/structures/bi2te3_defect_generation.py prepare-qe \
  --inferred "${PILOT_ROOT}/00_inferred/pristine_bi16te24_inferred.extxyz" \
  --output-dir "${PILOT_ROOT}/01_pristine_qe_preview" \
  --pseudo-dir "${PWD}/model_checkpoints/pseudos" \
  --qe-outdir "${PILOT_ROOT}/01_pristine_qe_preview/state"
```

The generated protocol is the repository's precise PBEsol setup:

```text
Bi  Bi.pbesol-dn-rrkjus.UPF
Te  Te.pbesol-n-rrkjus_psl.1.0.0.v2.UPF
ecutwfc = 60 Ry
ecutrho = 480 Ry
k-points = 6 x 6 x 6
conv_thr = 1e-9
fmax = 0.01 eV/Angstrom
```

The prepare manifest is explicitly marked `prepared_not_executed`.

## 4. Michael runs the expensive QE job later

Do not execute this as part of local testing. When Michael is ready to use the
UVA GPU QE builds, submit:

```bash
sbatch slurm/run_pristine_qe_relax_slurm_uva.sh \
  "${PILOT_ROOT}/00_inferred/pristine_bi16te24_inferred.extxyz" \
  "${PILOT_ROOT}/01_pristine_qe"
```

With the currently exported pristine structure, Michael can run the exact
command below from the repository root:

```bash
sbatch slurm/run_pristine_qe_relax_slurm_uva.sh \
  /scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/00_inferred/pristine_bi16te24_inferred.extxyz \
  /scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/01_pristine_qe
```

The launcher generates the `.pwi` from the pristine extxyz inside the allocated
job and then executes it. It does not mutate the extxyz and refuses to
overwrite existing preparation or output artifacts by default. The working
directory receives:

```text
01_pristine_qe/
  pristine_relax.pwi
  pristine_relax.pwo
  qe_prepare_manifest.json
  state/<slurm_job_id>/
```

By default, Slurm may allocate an A40, A100, or B200. The launcher detects the
allocated GPU with `nvidia-smi` and selects the matching host build:

```text
NVIDIA A40   -> qe-7.5-a40/bin/pw.x   (GPU_ARCH=86)
NVIDIA A100  -> qe-7.5-a100/bin/pw.x  (GPU_ARCH=80)
NVIDIA B200  -> qe-7.5-b200/bin/pw.x  (GPU_ARCH=100)
```

These are GPU-specific QE 7.5 host builds, not separate Apptainer images. This
table is relative to `/scratch/gqs7zr/ai4mat/qe`. This keeps the QE version and
invocation consistent while matching the compiled GPU architecture. The
launcher fails before calculation if the GPU is unsupported, the allocation is
heterogeneous, or the matching executable is absent.

Michael can force a particular supported GPU at submission time:

```bash
sbatch --constraint=a40 \
  slurm/run_pristine_qe_relax_slurm_uva.sh PRISTINE.extxyz WORK_DIR

sbatch --constraint=a100 \
  slurm/run_pristine_qe_relax_slurm_uva.sh PRISTINE.extxyz WORK_DIR

sbatch --constraint=b200 \
  slurm/run_pristine_qe_relax_slurm_uva.sh PRISTINE.extxyz WORK_DIR
```

Without an explicit command-line constraint, the embedded
`a40|a100|b200` constraint permits Slurm to choose any supported model.

## Generator-only preview without waiting for QE

Michael can test the persistent defect creator immediately by treating the
inferred structure as a preview parent:

```bash
/home/gqs7zr/.conda/envs/snb/bin/python \
  relaxgen/structures/bi2te3_defect_generation.py generate-2vte-preview \
  --parent /scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/00_inferred/pristine_bi16te24_inferred.extxyz \
  --output-dir /scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/03_unrelaxed_2vte_preview
```

This executes the same 276-to-14 symmetry enumeration and exact site-deletion
logic as the production generator. Every preview manifest is marked
`parent_validation_status: unvalidated_inferred_preview`; these structures test
the generator but do not replace the later cohort constructed from the
validated QE parent.

## 5. Validate and freeze the canonical parent after QE finishes

Michael, only run this after the Slurm QE job reports successful completion:

```bash
"${SNB_PYTHON}" relaxgen/structures/bi2te3_defect_generation.py validate-qe \
  --inferred "${PILOT_ROOT}/00_inferred/pristine_bi16te24_inferred.extxyz" \
  --qe-output "${PILOT_ROOT}/01_pristine_qe/pristine_relax.pwo" \
  --output-dir "${PILOT_ROOT}/02_canonical"
```

The default gates are:

```text
JOB DONE and BFGS convergence present
final maximum force <= 0.01 eV/Angstrom
40 atoms and Bi16Te24 retained
maximum cell-component difference <= 1e-6 Angstrom
species-mapped RMSD <= 0.10 Angstrom
maximum mapped displacement <= 0.25 Angstrom
R-3m and expected site-orbit pattern recovered at symprec <= 0.1 Angstrom
```

Review `qe_validation_manifest.json` before continuing. In particular, inspect
the mapped RMSD, maximum displacement, accepted `symprec`, final force, and
atom-order mapping. The validator writes both the raw parsed QE endpoint and
the canonical ordered endpoint, so no information is discarded.

## 6. Enumerate and construct the unrelaxed 2VTe cohort

After Michael accepts the validation report:

```bash
"${SNB_PYTHON}" relaxgen/structures/bi2te3_defect_generation.py generate-2vte \
  --canonical "${PILOT_ROOT}/02_canonical/pristine_bi16te24_qe_relaxed.extxyz" \
  --site-manifest "${PILOT_ROOT}/02_canonical/bulk_sites.json" \
  --output-dir "${PILOT_ROOT}/03_unrelaxed_2vte"
```

The code generates all C(24,2) = 276 Te-site pairs, reduces them with the
actual parent symmetry operations, and then checks the pilot expectation of 14
inequivalent orbits. The expected class counts are 7 TeA-TeA, 4 TeA-TeB, and
3 TeB-TeB. Orbit multiplicities must sum back to 276.

Each output is constructed only by deleting the two selected parent sites.
Every retained atom is checked for exact position and ordering equality with
the canonical parent, and every cell is checked for exact equality.

Expected layout:

```text
03_unrelaxed_2vte/
  catalog/
    2VTe_configurations.jsonl
    2VTe_enumeration_summary.json
  defects/
    2VTe_001/
      defect.json
      unperturbed.extxyz
    ...
    2VTe_014/
      defect.json
      unperturbed.extxyz
```

## 7. Stop gate

Michael, stop here and inspect the 14 structures and their manifests. Do not
add ShakeNBreak-style perturbations until the canonical-parent validation and
all 14 exact deletion structures have been reviewed.

## 8. Generate the broader vacancy/antisite catalog (no ShakeNBreak)

The reviewed catalog is defined in
`datagen/bi2te3_defect_family_catalog.json`. Pymatgen supplies the 96
ideal-parent symmetry permutations. ASE holds and writes the canonical and
defective structures. The generator applies only typed `remove` and
`substitute` operations; it does not distort or relax any coordinates.

Michael can first run the two-family generalization gate:

```bash
"${SNB_PYTHON}" relaxgen/structures/bi2te3_defect_generation.py \
  generate-defect-catalog \
  --canonical "${PILOT_ROOT}/02_canonical/pristine_bi16te24_qe_relaxed.extxyz" \
  --site-manifest "${PILOT_ROOT}/02_canonical/bulk_sites.json" \
  --families 3VTe antisite_pair \
  --output-dir "${PILOT_ROOT}/03_unrelaxed_defect_gate"
```

This computes 2,408 raw configurations and writes 58 unrelaxed,
symmetry-inequivalent structures: 46 `3VTe` and 12 composition-preserving
`antisite_pair` structures.

To generate every enabled family in the controlled catalog, omit
`--families`:

```bash
"${SNB_PYTHON}" relaxgen/structures/bi2te3_defect_generation.py \
  generate-defect-catalog \
  --canonical "${PILOT_ROOT}/02_canonical/pristine_bi16te24_qe_relaxed.extxyz" \
  --site-manifest "${PILOT_ROOT}/02_canonical/bulk_sites.json" \
  --output-dir "${PILOT_ROOT}/03_unrelaxed_defect_catalog"
```

The current bounded catalog computes 6,808 raw configurations and 188
symmetry-inequivalent structures across 14 neutral families. These counts are
regression results, not inputs to the enumeration algorithm.

Michael can also exercise the complete generator immediately using the
inferred structure. All such artifacts are explicitly marked as an
unvalidated preview:

```bash
"${SNB_PYTHON}" relaxgen/structures/bi2te3_defect_generation.py \
  generate-defect-catalog-preview \
  --parent "${PILOT_ROOT}/00_inferred/pristine_bi16te24_inferred.extxyz" \
  --output-dir "${PILOT_ROOT}/03_unrelaxed_defect_catalog_preview"
```

The general output layout is:

```text
03_unrelaxed_defect_catalog/
  catalog/
    defect_generation_summary.json
    defect_configurations.jsonl
    3VTe_enumeration_summary.json
    3VTe_configurations.jsonl
    ...
  defects/
    3VTe/
      3VTe_001/
        defect.json
        unperturbed.extxyz
      ...
    antisite_pair/
      ...
```

Each structure is reconstructed afresh from stable pristine site IDs.
Vacancies omit selected sites; substitutions change only species. The cell and
all retained Cartesian coordinates remain exactly equal to the canonical
float64 parent. Interstitials, charge states, ShakeNBreak distortions, uMLIP
relaxations, and QE relaxations are outside this generation stage.

## 9. Generate reconstruction candidates with ShakeNBreak primitives

`relaxgen/structures/generate_reconstruction_candidates.py` consumes the unrelaxed defect
catalog. It uses native ShakeNBreak operations for isolated defects and a
controlled per-center recipe for complexes. Every center-specific distortion
starts from the same unperturbed structure; distortions are never applied
sequentially around multiple centers.

The fixed recipes are:

```text
one center:    12 candidates
  unperturbed + 3 rattles + 3 bond distortions
  + 4 distortion/rattle candidates + 1 dimer

two centers:   10 candidates
  unperturbed + 3 rattles + 3 independent distortions per center

three centers: 12 candidates
  unperturbed + 2 rattles + 3 independent distortions per center
```

Michael can first run the representative four-configuration gate:

```bash
"${SNB_PYTHON}" relaxgen/structures/generate_reconstruction_candidates.py \
  --input-dir "${PILOT_ROOT}/03_unrelaxed_defect_catalog_preview" \
  --output-dir "${PILOT_ROOT}/04_reconstruction_candidate_gate" \
  --config-ids VTe_001 2VTe_001 3VTe_001 antisite_pair_001
```

This writes 44 structures. To process the complete 188-configuration catalog,
omit the filters:

```bash
"${SNB_PYTHON}" relaxgen/structures/generate_reconstruction_candidates.py \
  --input-dir "${PILOT_ROOT}/03_unrelaxed_defect_catalog" \
  --output-dir "${PILOT_ROOT}/04_reconstruction_candidates"
```

The full current recipe writes 2,124 unrelaxed candidates. Candidate manifests
record the source defect, center, ShakeNBreak primitive, factor/seed, geometry
hash, minimum distance, and displacement statistics. Every candidate is marked
`relaxed: false` and `dft_labeled: false`.

This stage is CPU-only. Its outputs should be inspected before starting uMLIP
relaxation.

## 10. Fixed-cell EqV3/FIRE2 pre-relaxation

`relaxgen/prerelax/relax_reconstruction_candidates_eqv3.py` performs the uMLIP stage.
It loads the Bi2Te3-finetuned EqV3 checkpoint once per defect configuration,
evaluates every candidate before optimization, and then relaxes atomic
positions with ASE FIRE2:

```text
target fmax = 0.001 eV/Angstrom
max steps  = 600
max step   = 0.10 Angstrom
cell       = fixed
```

If FIRE2 produces a non-finite state or crosses the hard minimum-distance
guard, the candidate is reset to its last known-safe geometry and retried with
standard ASE FIRE using the same `fmax` and `maxstep` plus
`downhill_check=True`. This fallback is only used for geometry-safety failures;
reaching the step limit does not silently switch optimizers.

Every candidate is restartable independently. A successfully completed
candidate is skipped when the same task is rerun. An interrupted or failed
candidate restarts from its original reconstruction geometry. Outputs include
`input.extxyz`, `trajectory.extxyz`, `relaxed.extxyz`, `optimizer.log`,
`metrics.json`, and `status.json`. The trajectory contains EqV3 energies and
forces and identifies each frame as FIRE2 or FIRE. Fallback cases additionally
write `fallback_optimizer.log`, and their metrics record both optimizer
attempts. All of these artifacts are explicitly ML-only, not DFT-labeled data.

Michael can inspect the stable array mapping without a GPU:

```bash
/home/gqs7zr/.conda/envs/fairchemv1/bin/python \
  relaxgen/prerelax/relax_reconstruction_candidates_eqv3.py list \
  --input-dir "${PILOT_ROOT}/04_reconstruction_candidates_preview_no_pure_expand40"
```

Run one 10-15-candidate smoke configuration:

```bash
sbatch --array=0-0 \
  --export=ALL,CONFIG_ID=VTe_001 \
  slurm/run_reconstruction_umlip_relax_slurm_uva.sh \
  "${PILOT_ROOT}/04_reconstruction_candidates_preview_no_pure_expand40" \
  "${PILOT_ROOT}/05_eqv3_tight_relaxations"
```

After inspecting the smoke output, a diverse approximately 100-candidate gate
can use these ten configuration indices:

```bash
sbatch --array=28,42,100,112,158,160-162,174,176 \
  slurm/run_reconstruction_umlip_relax_slurm_uva.sh \
  "${PILOT_ROOT}/04_reconstruction_candidates_preview_no_pure_expand40" \
  "${PILOT_ROOT}/05_eqv3_tight_relaxations"
```

Run the complete 188-configuration array only after those gates pass:

```bash
sbatch --array=0-187%8 \
  slurm/run_reconstruction_umlip_relax_slurm_uva.sh \
  "${PILOT_ROOT}/04_reconstruction_candidates_preview_no_pure_expand40" \
  "${PILOT_ROOT}/05_eqv3_tight_relaxations"
```

Aggregate the current state at any time:

```bash
/home/gqs7zr/.conda/envs/fairchemv1/bin/python \
  relaxgen/prerelax/relax_reconstruction_candidates_eqv3.py summarize \
  --input-dir "${PILOT_ROOT}/04_reconstruction_candidates_preview_no_pure_expand40" \
  --output-dir "${PILOT_ROOT}/05_eqv3_tight_relaxations"
```

The summary reports completed, converged, fallback-used, max-step, failed, and
missing counts, plus force, step, and wall-time distributions broken down by
candidate type and defect multiplicity. These preview-parent relaxations
validate the machinery;
the final data-generation run should use candidates regenerated from the
QE-validated canonical parent.

## Focused non-QE tests

These tests reconstruct the inferred parent, prepare but do not execute QE,
exercise the structural-equivalence logic with a synthetic tiny displacement,
verify the 276-to-14 symmetry partition, and construct the 14 deletion-only
structures:

```bash
"${SNB_PYTHON}" -m unittest datagen.test_bi2te3_defect_generation -v
```

They never invoke `pw.x` and do not validate a real QE result.

## 11. Run the 12-job matched QE diagnostic

`relaxgen/dft/qe_matched_diagnostic.py` freezes four audited cases and prepares a
matched fixed-cell QE/BFGS experiment:

```text
4 uMLIP endpoints x (B200 + A100 + A40) = 12 relaxations
direct DFT from pre-uMLIP distortions    =  0 relaxations
```

The selected cases are:

```text
easy_vte          VTe_001/unperturbed
aggressive_3bi_te 3Bi_Te_002/center_01_TeA_000_contract_40pct
diverse_3vte      3VTe_002/center_01_TeA_000_contract_40pct
diverse_3vbi      3VBi_001/center_03_Bi_010_contract_40pct
```

Michael can prepare or reproduce the manifest without running QE:

```bash
"${SNB_PYTHON}" relaxgen/dft/qe_matched_diagnostic.py prepare \
  --umlip-dir "${PILOT_ROOT}/05_eqv3_tight_relaxations" \
  --output-dir "${PILOT_ROOT}/06_qe_matched_diagnostic"
```

This writes 12 independent QE inputs using the same PBEsol pseudopotentials,
60/480 Ry cutoffs, 6x6x6 k-point grid, 0.01 eV/Angstrom final-force target,
100-step limit, fixed cell, and 17-digit float64 input geometry as the pristine
calculation. It does not submit work.

Michael can submit all four matched arrays with one command:

```bash
slurm/submit_qe_matched_diagnostic.sh \
  "${PILOT_ROOT}/06_qe_matched_diagnostic/experiment_manifest.json"
```

The submitter requests four B200 endpoint tasks, four A100 endpoint tasks, and
four A40 endpoint tasks. Every QE calculation starts from a converged uMLIP
endpoint; no original distorted candidate is sent directly to DFT. It records job IDs and
the initial `squeue --start` estimates under `submission/`. Each running task
also records Slurm submit, eligibility, and start timestamps from `scontrol`.

After all tasks leave the queue, Michael can validate and summarize them:

```bash
"${SNB_PYTHON}" relaxgen/dft/qe_matched_diagnostic.py summarize \
  --manifest "${PILOT_ROOT}/06_qe_matched_diagnostic/experiment_manifest.json"
```

The summary requires explicit `JOB DONE`, BFGS convergence, final QE forces at
or below 0.01 eV/Angstrom, an unchanged cell, and unchanged ordered species.
It collects `sacct` submit/eligible/start/end timestamps, queue and run
durations, SCF and ionic-step counts, GPU runtime ratios, and cross-GPU endpoint
RMSDs. The
uMLIP structures and trajectories remain surrogate artifacts; only completed
QE outputs are DFT-labeled.

Each task also writes `timing_events.jsonl`. The runner timestamps every
observed `Self-consistent Calculation`, electronic-convergence line, and
`Total force` line while QE is running. The final status combines these live
events with QE's authoritative footer timers for `electrons`, `forces`,
`stress`, and total `PWSCF` time. The aggregate summary reports, overall and by
GPU type:

```text
full relaxation wall time per job
exact SCF/force (singlepoint) evaluation count
BFGS ionic-step count
electronic iterations per evaluation
full job wall time / evaluation
QE electrons wall time / call
live observed SCF and SCF-through-force durations
total GPU-hours
```

Here one "singlepoint evaluation" means one self-consistent QE energy/force
evaluation at a fixed ionic geometry inside BFGS. It is distinct from one
electronic iteration: one singlepoint normally contains many electronic
iterations.

At the end of each individual Slurm `.out` file, the task runner prints a
generic human-readable `QE relaxation statistics` block. It includes the GPU,
convergence result, total wall time, singlepoint count, BFGS steps, average
wall time per singlepoint, electronic-iteration distribution, QE footer
timers, observed SCF timing distribution, and final maximum force. The same
block is printed for failed or previously completed tasks using whatever
statistics are available.

## 12. uMLIP-to-DFT handoff threshold

`relaxgen/prerelax/umlip_dft_handoff_threshold.py` measures where tighter uMLIP
pre-relaxation stops helping DFT. The existing QE runs started from EqV3
endpoints at 0.001 eV/Angstrom still saw 0.05-0.16 eV/Angstrom of DFT force, so
the useful ML force target may be much looser than the production one.

The cohort is fixed at 20 candidates: the four matched-diagnostic cases, the
two `2Bi_Te_001` cases with tight QE references, and `rattle_seed_101` of
`<family>_001` for each of the 14 families. Each is re-relaxed with the
production EqV3/FIRE2 settings, except `fmax = 1e-4` and `max_steps = 5000`.
From every trajectory the first frame with ML fmax strictly below each
checkpoint

```text
0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001 5e-4 2e-4 1e-4 eV/Angstrom
```

is selected. That is exactly the frame where `FIRE2.run(fmax=t)` would stop
and hand off. Frames serving several checkpoints run once. The final frame is
always included. Every selected frame gets one fixed-cell QE SCF with the
production electronic protocol (60/480 Ry, 6x6x6, `conv_thr = 1e-9`) on an
A100 or B200, using `disk_io='nowf'`. Relaxation state directories were about
28 GB each.

```bash
OUT="${PILOT_ROOT}/09_umlip_dft_handoff_threshold"

"${SNB_PYTHON}" relaxgen/prerelax/umlip_dft_handoff_threshold.py prepare-cohort --output-dir "${OUT}"
slurm/submit_umlip_dft_handoff_threshold.sh deep-relax "${OUT}/cohort.json"

# after the deep relaxations finish
"${SNB_PYTHON}" relaxgen/prerelax/umlip_dft_handoff_threshold.py prepare-scf --cohort "${OUT}/cohort.json"
slurm/submit_umlip_dft_handoff_threshold.sh scf "${OUT}/scf_manifest.json"   # 32 concurrent by default

# any time, including with partial results
"${SNB_PYTHON}" relaxgen/prerelax/umlip_dft_handoff_threshold.py summarize --manifest "${OUT}/scf_manifest.json"
```

The QE step launched by `run_task` inherits the job's GPUs. It used to
request them again with `srun --gpus-per-task`. That made Slurm bind to
GPU-local cores, and the step failed with "Unable to satisfy cpu bind
request" whenever the job had been allocated other cores. This happened to
tasks 1-3 of the `07_qe_relaxations_tight_umlip` campaign. Do not use
`--gres-flags=enforce-binding` instead: on this cluster it makes A100/B200
requests unschedulable.

`summarize` writes:

- `handoff_singlepoints.csv`: one row per SCF. Columns include ML and DFT
  fmax, DFT energy, ML/DFT force error and median angle, and the energy above
  the per-case minimum.
- `handoff_summary.json` and `handoff_summary.txt`: per-checkpoint quantiles
  and per-case saturation checkpoints. The fmax saturation point is the first
  checkpoint whose DFT fmax is within 1.1x of the case minimum. The energy
  saturation point is the first checkpoint within 1 meV/cell of the case
  minimum.
- `handoff_threshold.png`: DFT force and energy against ML fmax.

When a converged QE relaxation of the same candidate exists in 06/07/08 under
the same electronic protocol, each row also reports the energy above, and the
MIC RMSD to, that DFT minimum. EqV3 energies are float32-quantized
(about 8 meV steps at this total energy), so ML energy differences are not
meaningful at the meV level. FIRE2 uses only forces.

### CPU backend and backend check

SCF manifests prepared by `umlip_dft_handoff_threshold.py` accept
`--backend a100|b200|cpu`. The CPU path uses the cluster build of the same QE
version:

```bash
module load intel-compilers/2025.3.2
module load impi/2021.17.2
module load quantumespresso/7.5
```

These modules must be loaded one at a time; QE only becomes visible after the
compiler and MPI are loaded.

`slurm/run_qe_scf_cpu_slurm_uva.sh` runs one task on one node. It defaults
to 40 ranks on `standard`, and `NPOOL` sets the k-point pools; it must divide
the rank count. The cell has 112 k-points, so pool parallelism scales well.
Intel MPI forks its ranks directly (`I_MPI_HYDRA_BOOTSTRAP=fork`). Its default
Slurm bootstrap calls srun, which aborts when both `SLURM_MEM_PER_CPU` and
`SLURM_MEM_PER_NODE` are set.

Per-user QOS limits are 1,500 CPUs on `standard` (7-day jobs, single node),
32 GPUs on `gpu`, and 24 CPUs / 216 GB / 12 h on `interactive`.

Before mixing CPU and GPU results, `prepare-backend-check` recomputes the
first SCF of a finished A100 run at its identical input geometry.
`compare-backend-check` then reports the energy and force differences and the
wall times:

```bash
CHECK="${PILOT_ROOT}/10_qe_backend_check"
"${SNB_PYTHON}" relaxgen/prerelax/umlip_dft_handoff_threshold.py prepare-backend-check \
  --reference-status "${PILOT_ROOT}/06_qe_matched_diagnostic/tasks/04_easy_vte_umlip_endpoint_a100/status.json" \
  --output-dir "${CHECK}" --copies 2
sbatch -p interactive --ntasks=24 --mem-per-cpu=8G --time=12:00:00 --array=0 \
  --export=ALL,NPOOL=12 slurm/run_qe_scf_cpu_slurm_uva.sh "${CHECK}/scf_manifest.json"
"${SNB_PYTHON}" relaxgen/prerelax/umlip_dft_handoff_threshold.py compare-backend-check \
  --manifest "${CHECK}/scf_manifest.json"
```

## 13. GPU/CPU dispatch controller

`relaxgen/dispatch/qe_dispatch.py` runs the tasks of an SCF manifest on whichever Slurm
backend is expected to finish them first:

- GPU: A100/B200.
- CPU on `parallel`: 2 x 84 ranks, 56 k-point pools.
- CPU on `standard`: one Afton node, 84 ranks, 28 k-point pools.

Estimates come from this user's real submit-to-start delays (`sacct`, last 3
days) and measured run times. On 2026-10-22 the median waits were about
4 min for GPU and 14 min for `standard`, while `sbatch --test-only` predicted
days.

```bash
D="${PILOT_ROOT}/<campaign>/dispatch"
"${SNB_PYTHON}" relaxgen/dispatch/qe_dispatch.py init --dispatch-dir "${D}" \
  --manifest "${PILOT_ROOT}/<campaign>/scf_manifest.json" --job-name qd_<campaign>
"${SNB_PYTHON}" relaxgen/dispatch/qe_dispatch.py tick --dispatch-dir "${D}" --dry-run   # plan only
"${SNB_PYTHON}" relaxgen/dispatch/qe_dispatch.py tick --dispatch-dir "${D}"             # submit/cancel
"${SNB_PYTHON}" relaxgen/dispatch/qe_dispatch.py status --dispatch-dir "${D}"
```

How it works:

- **Queueing.** GPU jobs are kept queued for the head of the remaining work. A
  tail unit goes to a CPU backend only if CPU wait plus run time beats its
  expected GPU completion, within `cpu_core_budget` and per-backend
  `max_jobs`.
- **Claims.** A unit can be queued on several backends at once. The first job
  to start atomically creates `claims/<task>.json` and runs; twins exit, and
  the next tick cancels any still queued. The running worker refreshes its
  claim every minute. A claim older than 15 min whose job is no longer running
  is released.
- **Retries.** A unit fails after 3 runs that do not converge.
- **CPU gating.** A CPU backend stays disabled until its backend check
  (`10_qe_backend_check_fullnode`) agrees with the A100 reference within
  0.1 meV/cell and 1e-3 eV/Angstrom. `tick` re-runs the comparison itself
  whenever a check task finishes.
- **Ticks.** Each worker runs a tick when it exits. For a steady cadence, add
  a `scrontab` entry that runs `tick` every 10 minutes.

The SCF launchers run under the worker whenever `DISPATCH_DIR` is set, and
are unchanged otherwise.

### Relaxations under the controller (`init --workload relax`)

Units are tasks of a `dft_relaxation_campaign.py` manifest. Backends:

- 1-GPU and 2-GPU A100/B200 jobs, each a 12 h segment. Each unit is queued on
  both, and whichever starts first runs.
- CPU on `parallel`: 2 x 42 ranks, 16 GB per rank, 24 h segments. Gated on the
  backend check.

A segment that stops unconverged at its time budget is progress, not a
failure. Two safeguards prevent a relaxation from looping without
advancing:

- **Minimum segment length.** A backend is used only if its segment length is
  at least `min_segment_steps` (3) slow-case step times. The step time is the
  90th percentile of that backend's measured steps; until 3 have been
  measured, the backend-check SCF time or the backend's prior stands in. This
  matters on CPU: a production SCF takes hours there, and each resume first
  recomputes the last geometry, so a segment shorter than about one SCF would
  never advance.
- **Zero-progress backoff.** Each worker records how many trajectory frames
  its segment added (`trajectory_frames` in `status.json`). It also catches
  Slurm's time-limit SIGTERM, so killed segments are recorded too. A unit is
  never resent to a backend where it added no frames. A backend with 2
  zero-progress segments within 48 h is disabled until 48 h after the last
  one. Zero-progress segments count as failed attempts (3 maximum).

### CPU-to-GPU hand-over

While a relaxation runs on a CPU backend, `tick` keeps a GPU job queued for
it. When that GPU job starts, it hands the unit over to itself:

1. The GPU job writes `preempt/<task>.json`.
2. The CPU worker checks for that file every 10 s and stops QE's whole process
   group, then releases the claim.
3. The GPU job claims the unit and resumes from the last stored step.

Only the SCF in progress on the CPU is lost, and nothing better was possible:
QE cannot carry its electronic state across machines, so the GPU's first SCF
starts from scratch whenever it takes over.

**Edge cases:**
- A hand-over is recorded as outcome `preempted`, which counts neither as a
  failure nor as a zero-progress segment.
- A claim not refreshed for `claim_stale_seconds` belongs to a dead CPU job;
  the GPU job removes it and claims the unit.
- A request file older than the CPU job (left by a GPU job that died while
  waiting) is ignored.

GPU jobs are not preempted; they run until their segment ends.

**When a relaxation goes to CPU (2026-10-24).** Because a GPU job takes over
on start, CPU time spent waiting is never wasted. So a waiting relaxation gets
a CPU job when the CPU job can start and run `min_segment_steps` steps before
the relaxation's expected GPU start:

```
cpu_wait + min_segment_steps x cpu_step  <  gpu_wait + gpu_runtime x (queue position // GPU slots)
```

SCF campaigns, and relaxations without hand-over, keep the older rule: a whole
CPU job must finish before the GPU job would. A GPU backend capped at 0 jobs
counts as off.

The priors used before a campaign has its own measurements:

| backend | prior | source |
| --- | --- | --- |
| `cpu-parallel-2x42` | queue wait 15 min, step 80 min | 2-node probes (median 4-7 min, p90 ~1.5 h, about the same for any limit from 2 h to 24 h); Stage B steps (p90) |
| GPU | queue wait 3.4 h | 1-GPU 12 h probes (median) |
| 1 node, 84 ranks (Afton) | (not a backend) | one 79.5 min SCF, but a 27 h queue wait |

Campaign caps for 2-node CPU jobs: diversity 8, batch 1 6, Stage B 10.

**Queue waits on `parallel` (all users, 2026-10-20 to 2026-10-23):**

| nodes | median wait | 90th percentile |
| --- | --- | --- |
| 2 | 8.6 h | 32 h |
| 3-4 | 15.5 h | 18 h |

GPU jobs and single `standard-afton` nodes typically start within minutes.

**Memory per rank.** QE estimates 1.9-2.2 GB per rank for the production SCF,
and the measured peak was about 3 GB/rank. The 16 GB/rank request makes a
42-rank job ask for most of a 768 GB node. `17_qe_backend_check_singlenode`
measures single-node layouts (84, 96 and 42 ranks) at 6 GB/rank.

**Multi-node scratch.** Multi-node CPU relaxations put the QE outdir on shared
scratch (`/scratch/$USER/qe_tmp/<job>`). A node-local `TMPDIR` exists only on
the first node.

### SCF stalls: abandon the relaxation

Some defect geometries put two near-degenerate levels at the Fermi energy.
With `degauss = 0.001` Ry their occupations flip from one SCF iteration to the
next, and the SCF accuracy stalls short of `conv_thr = 1e-9`. In driver mode
`pw.x` then stops ("convergence NOT achieved"), and ASE reports the closed
socket as `ConnectionResetError`.

- **Rate.** At Gamma only (resume test), 11 of about 1,150 SCFs stalled at
  1e-4 to 1e-5 Ry. At 6x6x6, 1 of 423 stalled, at about 1e-7 Ry (threshold
  task 228). Every other 6x6x6 SCF converged within 22 iterations.
- **Restarting does not help.** The stall recurs at the same geometry on every
  backend: `contract40` stalled near step 30 in every resume-test arm.
- **Other settings were tested and rejected.** In `14_scf_stall_ladder_check`,
  `local-TF` mixing (beta 0.1) cleared 0 of 4 stalls. Widening the smearing to
  0.005 Ry cleared 3 of 4, but changes the energy surface. Shrinking the BFGS
  step would bend the path. None of these are used.

`dft_relaxation_campaign.py` caps every SCF at `electron_maxstep = 60` (a
`prepare` option recorded in the protocol). That stops a stall after about
2.5 A100 hours at 6x6x6, instead of about 8 h at QE's 200.

**On a stall.** The relaxation is abandoned:

- `status.json` gets `execution_status = "failed"` and
  `failure_kind = "scf_not_converged"`;
- later jobs skip the task;
- `qe_dispatch.py` does not retry it on any backend;
- frames written before the stall stay in `trajectory.traj`, but the
  relaxation is excluded from the dataset.
