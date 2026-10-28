# QE timing test

A small test for measuring how fast QE runs on a new cluster, with results you can compare directly with Rivanna. It uses the same QE settings as the production relaxations, so the times can be compared with `docs/cost_analysis/README.md`.

`relaxgen/analysis/cost_analysis.py` only works on Rivanna, because it reads the Rivanna campaign folders and Slurm history. Use this test instead on other clusters.

## What is in this folder

- `structures/easy_vte.extxyz`: a 39-atom structure with a Te vacancy. On Rivanna, one SCF from scratch on this structure took:

  | Hardware | Minutes |
  |---|---|
  | A100 80GB | 46.0 |
  | CPU, 2 nodes, 84 ranks | 75.0 |
  | CPU, 1 node, 84 ranks | 79.6 |
  | CPU, 1 node, 96 ranks | 92.5 |
  | CPU, 1 node, 42 ranks | 150.1 |

- `structures/2Bi_Te_001__unperturbed.extxyz`: a 40-atom structure from batch 1. On Rivanna it relaxed to fmax 0.01 eV/Å in 15 SCFs on one A100 80GB. The first SCF took 48.4 min, and the later ones a median 29.5 min.
- `rivanna_reference.json`: Rivanna's energies, forces and step times for both structures.
- `run_timing_test.sh`: a Slurm launcher that works on any cluster once you set a few values at its top.
- `compare_to_rivanna.py`: prints a run's SCF times next to Rivanna's, and checks that the energies and forces match.

Everything the test needs is in the git repository, including the pseudopotentials in `pseudos/`. It does not use the machine-learning models, so you do not need `ft_ckpts/`.

## What you need on the new cluster

- The relaxgen repository (`git clone` or `git pull`).
- The `snb` conda environment: `conda env create -f conda_envs/snb.yml`.
- A QE 7.5 build: the cluster's own module or one you compiled. For a GPU build, put it in a folder named after the GPU: `qe-7.5-a100`, `qe-7.5-b200` or `qe-7.5-h100`, for example `.../qe-7.5-h100/bin/pw.x`. The runners check the folder name, so it has to be a real folder with that name, not a symlink to one.

## Running the test

Run these from the repository root. Pick a run folder on the new cluster's scratch space.

1. Prepare the two relaxations. `--max-steps 4` stops each after 5 SCFs, enough to time the first SCF and four later ones (about 2.5 hours on an A100). Leave it out to run `2Bi_Te_001` all the way to convergence, which takes 15 SCFs.

   ```bash
   RUN=/path/to/scratch/qe_timing_run
   python -m relaxgen.dft.dft_relaxation_campaign prepare --output-dir $RUN \
     --structures benchmarks/qe_timing/structures/easy_vte.extxyz \
                  benchmarks/qe_timing/structures/2Bi_Te_001__unperturbed.extxyz \
     --labels easy_vte 2Bi_Te_001__unperturbed --max-steps 4
   ```

2. Set the values at the top of `run_timing_test.sh` (or export them), then submit both structures as a two-element array:

   ```bash
   # GPU (use BACKEND=h100 and the qe-7.5-h100 build on H100s)
   BACKEND=a100 QE_EXECUTABLE=/path/to/qe-7.5-a100/bin/pw.x PYTHON_BIN=/path/to/envs/snb/bin/python \
   sbatch --array=0-1 -p <gpu partition> --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=60G \
     benchmarks/qe_timing/run_timing_test.sh $RUN/relax_manifest.json

   # CPU, 2 nodes x 42 ranks, as on Rivanna
   BACKEND=cpu NPOOL=28 QE_EXECUTABLE=/path/to/pw.x PYTHON_BIN=/path/to/envs/snb/bin/python \
   sbatch --array=0-1 -p <cpu partition> -N2 --ntasks=84 --ntasks-per-node=42 \
     benchmarks/qe_timing/run_timing_test.sh $RUN/relax_manifest.json
   ```

   With `--max-steps`, each task ends with "max_steps reached". That is expected.

3. Compare with Rivanna:

   ```bash
   python benchmarks/qe_timing/compare_to_rivanna.py $RUN/relax_manifest.json
   ```

   The first SCF of each structure should match Rivanna within 0.1 meV in energy and 0.001 eV/Å in every force component. If it does not, the QE build differs in a way that matters, and its results should not be mixed with Rivanna's.

To time another layout or GPU, prepare a new run folder and submit again with different Slurm flags.
