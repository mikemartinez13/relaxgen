# Estimated Cost Analysis

### Measured on UVA's Rivanna cluster (Afton nodes), October 20–28, 2026

This page estimates how long our DFT relaxations take and how much they cost on Rivanna, and how much faster Plan A finishes when the CS cluster takes part of it (see [CS cluster](#cs-cluster)). Every number comes from our own jobs: the relaxation campaigns, the handoff threshold runs and a set of small test jobs. The script that computes them is `relaxgen/analysis/cost_analysis.py`, and the raw numbers are in `cost_numbers.json` in this folder.

A few terms used below:

- **Relaxation:** one Quantum ESPRESSO (QE) run that moves the atoms of one defect structure until the forces are small.
- **SCF:** one step of a relaxation. QE solves for the electrons, computes the forces, and then the atoms move. A relaxation is a series of SCFs, so its cost is roughly the number of SCFs times the time per SCF.
- **fmax:** the largest force on any atom, in eV/Å. A relaxation is finished when fmax drops below a target, for example 0.01 eV/Å.
- **SU (service unit):** the cluster's billing currency. Every job is charged SU per hour, depending on the cores, memory and GPUs it reserves.

**Hardware we use.** For GPUs, we only use A100s (40 GB and 80 GB) and B200s. We also have a QE build for A40 GPUs, but we choose not to use it because A40s are too slow for QE. For CPUs, we run each job on 2 nodes with 42 MPI ranks per node (84 ranks in total) on the `parallel` partition. All QE jobs use QE 7.5 with the same settings, so a relaxation can move between a GPU and a CPU job and give the same result.

## SCF estimated times

Each job's first SCF starts from scratch. Later SCFs in the same job are faster, because QE starts from the electrons of the previous step. Most of a relaxation's SCFs are the faster kind.

| Hardware | SCF from scratch, median (number of SCFs) | SCF inside a relaxation, median / 90th percentile (number of SCFs) |
|---|---|---|
| B200 | 25 min (32) | 16 min / 25 min (399) |
| A100 80GB | 48 min (233) | 26 min / 35 min (384) |
| A100 40GB | 55 min (62) | 33 min / 44 min (214) |
| CPU, 2 nodes, 84 ranks | 81 min (511) | 51 min / 66 min (1980) |
| CPU, 1 node, 84 ranks | 80 min (1) | - |
| CPU, 1 node, 96 ranks | 93 min (1) | - |
| CPU, 1 node, 42 ranks | 133 min (13) | - |

The B200 is the fastest, about 1.6 times faster than an A100 80GB and 3 times faster than a CPU job. The single-node CPU rows were one-off tests. They were no faster than 2 nodes and had to wait longer in the queue (next section), so we use 2 nodes.

The number of SCFs a relaxation needs depends on how tight the force target is, and on which machine-learning model pre-relaxed the starting structure:

| DFT fmax target | Start from EqV3 | Start from MACE-ft | Start from UMA |
|---|---|---|---|
| 0.01 eV/Å | 15 | 22 | 26 |
| 0.005 eV/Å | 24 | 31 | 34 |
| 0.001 eV/Å | 40 | 47 | 50 |

Where these come from:

- **EqV3 to 0.01:** the median of 51 finished relaxations in batch 1.
- **Tighter targets:** going from 0.01 to 0.005 adds a median 9 SCFs, and going to 0.001 adds 25. These come mostly from the diversity campaign, the only one run to 0.001 so far.
- **MACE-ft and UMA:** their structures start further from the DFT answer. In matched runs from the same starting structure, they needed a median 7 (MACE-ft) and 10–11 (UMA) more SCFs than EqV3. Most of these runs were cases where the models disagreed, so these extra counts may be on the high side.

## Estimated queue time

The queue wait is the time between a job becoming ready to run and the job starting. This table covers every job our campaigns submitted since October 20.

| Where the job ran | Jobs | Median wait | Average wait | 90th percentile |
|---|---|---|---|---|
| B200 | 16 | 8 min | 6.1 h | 4.9 h |
| A100 80GB | 56 | 6.3 h | 17.3 h | 82.5 h |
| A100 40GB | 17 | 2.1 h | 16.6 h | 83.9 h |
| Any GPU (all GPU rows above) | 89 | 4.2 h | 15.2 h | 82.7 h |
| CPU, 2 nodes, 84 ranks | 521 | 1 min | 21 min | 54 min |
| CPU, 1 node, 42 ranks | 12 | 12.9 h | 13.4 h | 17.4 h |

CPU jobs on 2 nodes start almost right away. GPU jobs usually wait hours, and about one in ten waits more than three days. There are two reasons:

- **The GPU partitions are full.** Hundreds of jobs from other users are always waiting for them.
- **We have low priority.** Slurm moves users who have used little recently to the front of the queue. Our campaigns used a lot, so our jobs wait near the back.

The B200 row looks fast only because of how we submit. Each GPU job may run on any A100 or B200, and it lands on a B200 only when one happens to be free right away. When nothing is free the job waits, and it usually ends up on an A100 because there are about four times as many of them.

We also submitted small test jobs that do no work, only to measure the wait for each job shape. They show the same pattern:

| Job shape | Test jobs started | Median wait | 90th percentile |
|---|---|---|---|
| CPU, 2 nodes, 84 ranks | 217 | 2 min | 1.3 h |
| CPU, 1 node, 42 ranks | 18 | 43 min | 33.2 h |
| CPU, 1 node, 84 ranks | 35 | 1.3 h | 27.2 h |
| 1 GPU (A100 or B200) | 5 | 3.7 h | 56.4 h |

## SU cost / time in days cost

On Rivanna, 1 SU is Slurm's `billing` value divided by 10. Our lab account, `raiselab`, gets 10,000,000 SU per year (May 6, 2026 to May 6, 2027). On October 28 it had **6.14M SU left**. The whole lab shares this pool, and unused SU does not carry over to the next year. Our campaigns have used about 177,000 SU since October 20.

| Hardware | SU per hour | SU per SCF inside a relaxation | SU per SCF from scratch |
|---|---|---|---|
| B200 | 132 | 36 | 55 |
| A100 80GB | 55 | 24 | 44 |
| A100 40GB | 49 | 27 | 45 |
| CPU, 2 nodes, 84 ranks | 53 | 45 | 72 |
| CPU, 1 node, 42 ranks | 34 | - | 75 |

An SCF on an A100 costs about half as much as the same SCF on CPU. The B200 is the fastest but costs the most per hour. Its rate here comes from Slurm's billing. UVA's published price list suggests a lower rate, about 81 SU per hour, so B200 costs may be overstated.

One relaxation from an EqV3 start, run time only (no queue wait):

| DFT fmax target | SCFs | On one A100 80GB GPU | On CPU, 2 nodes, 84 ranks |
|---|---|---|---|
| 0.01 eV/Å | 15 | 6.9 h, 383 SU | 13.2 h, 700 SU |
| 0.005 eV/Å | 24 | 11.3 h, 620 SU | 20.7 h, 1104 SU |
| 0.001 eV/Å | 40 | 18.3 h, 1007 SU | 34.7 h, 1848 SU |

The tables below estimate the full campaigns. Stage A gave us 2,124 starting structures. Plan A relaxes each one once, from its EqV3 pre-relaxation. Plan B relaxes each one three times, from its EqV3, MACE-ft and UMA pre-relaxations, so 6,372 relaxations in total.

Each column is a different amount of hardware:

- **~3 GPUs running + 71 CPU jobs:** what we get today. Over the past week our campaigns had on average about 3 GPU jobs running. 71 is the most 2-node CPU jobs one user can run at once (6,000 cores).
- **32 GPU jobs + 71 CPU jobs:** our GPU limit is 32 jobs at once, but every job first waits in the queue. With the past week's average wait, 32 GPU jobs keep only about a third of those GPUs busy.
- **32 GPU jobs only:** the same, without CPU jobs.

Each cell gives a range of days. The lower end assumes each job waits the median time in the queue, and the upper end assumes the average wait, which includes the long waits of the past week. GPU numbers use the A100 80GB, since most of our GPU jobs run there.

**Plan A: EqV3 starts only (2,124 relaxations)**

| DFT fmax target | SCFs per relaxation | ~3 GPUs running + 71 CPU jobs | 32 GPU jobs + 71 CPU jobs | 32 GPU jobs only |
|---|---|---|---|---|
| 0.01 eV/Å | 15 | 15–16 days, 1.4M SU | 11–13 days, 1.3M SU | 31–61 days, 0.81M SU |
| 0.005 eV/Å | 24 | 24 days, 2.3M SU | 18–21 days, 2.0M SU | 54–115 days, 1.3M SU |
| 0.001 eV/Å | 40 | 40–41 days, 3.8M SU | 27–33 days, 3.3M SU | 74–134 days, 2.1M SU |

**Plan B: EqV3, MACE-ft and UMA starts (6,372 relaxations)**

| DFT fmax target | SCFs per relaxation (average) | ~3 GPUs running + 71 CPU jobs | 32 GPU jobs + 71 CPU jobs | 32 GPU jobs only |
|---|---|---|---|---|
| 0.01 eV/Å | 21 | 63–64 days, 6.0M SU | 43–52 days, 5.1M SU | 114–205 days, 3.4M SU |
| 0.005 eV/Å | 30 | 91–93 days, 8.6M SU | 64–79 days, 7.5M SU | 184–367 days, 4.9M SU |
| 0.001 eV/Å | 46 | 138–140 days, 13.0M SU | 92–112 days, 11.2M SU | 243–425 days, 7.3M SU |

What this means:

- **Plan A at 0.01 eV/Å** takes about two weeks and 1.4M SU. That fits our remaining SU comfortably.
- **Plan B at 0.01 eV/Å** takes about two months and 5–6M SU. That is nearly all the SU the lab has left until May 2027. Tighter targets in Plan B need more SU than we have, unless they run on GPUs only, which would take six months to a year.
- **CPU jobs do most of the work,** because they start right away. More GPU jobs save SU, because GPU SCFs are cheaper, but they save only a few days while GPU waits stay this long.

## CS cluster

### Measured on UVA's Computer Science cluster, October 28–29, 2026

The CS cluster does not bill SU, so jobs there cost nothing from the `raiselab` allocation. The numbers below come from the timing test in `benchmarks/qe_timing/` (5–6 SCFs per structure on each kind of hardware), not from production campaigns.

**Hardware we use.** We use only the 8 H100 NVL GPUs (`serval06-09`, 2 per node) and the 4 A100 80GB GPUs (`cheetah04`). We do not use:

- **The A100 40GB GPUs on `cheetah01`.** They are the PCIe version and took 76 min for an SCF from scratch, 1.6 times slower than the A100 80GB.
- **The other GPUs** (GeForce, Quadro, RTX A-series, A40, A16, P100). They are slow at the double-precision math QE needs, or have too little memory. QE estimates 36 GB per GPU for our structures.
- **CPU nodes.** A 28-rank job on one `bigcat` node took 524 min for an SCF from scratch and 270 min inside a relaxation, about 0.43 times the speed of an Afton core. With the per-user limit of 200 cores, CPU jobs would add about 3% to what the GPUs do.

| Hardware | SCF from scratch, median | SCF inside a relaxation, median | One relaxation, 15 SCFs |
|---|---|---|---|
| H100 NVL | 28 min | 18.5 min | 4.8 h |
| A100 80GB | 51 min | 33 min | 8.6 h |

The H100 is 1.6–1.8 times faster than an A100 80GB, close to Rivanna's B200. The CS cluster's A100 80GB is about 10% slower than Rivanna's. The relaxation times assume each relaxation runs as one job, since jobs on the CS cluster can last up to 4 days.

**Same results as Rivanna.** The first SCF of both test structures matched Rivanna within 0.01 meV in energy and 3×10⁻⁵ eV/Å in every force, on both GPUs and on the CPU build. Relaxations can move between the two clusters.

**Queue time.** The six GPU test jobs waited between 5 min and 5.3 h, with a median of 2.0 h. That is too few jobs for a reliable estimate. We share these GPUs with the rest of the department, so the tables below give two cases: all 12 GPUs running our jobs all the time, and half of them.

### Plan A on both clusters

Rivanna runs ~3 GPUs and 71 CPU jobs, as in the first column of the Plan A table above. It finishes about 5.8 relaxations per hour at 0.01 eV/Å. All 12 CS GPUs finish about 2.1 per hour.

| DFT fmax target | Rivanna alone | Both clusters, all 12 CS GPUs | Both clusters, half of the CS GPUs | Share done on the CS cluster | SU saved |
|---|---|---|---|---|---|
| 0.01 eV/Å | 15 days, 1.4M SU | 11 days | 13 days | 16–27% | 0.22–0.39M |
| 0.005 eV/Å | 24 days, 2.3M SU | 18 days | 20 days | 16–27% | 0.35–0.61M |
| 0.001 eV/Å | 40 days, 3.8M SU | 29 days | 34 days | 16–27% | 0.59–1.03M |

Each cluster must get the share it can finish in the same time, because a relaxation cannot move to the other cluster's queue by itself. At 0.01 eV/Å:

- **Rivanna:** the first ~1,724 relaxations, including the 200 already running.
- **CS cluster:** the last 400. That is 19% of Plan A, the CS share if we get about 63% of its 12 GPUs.

If the CS cluster gets all 12 GPUs, its 400 finish in 8 days and Rivanna's 1,724 in 12. If it gets only half, its 400 take 16 days. Check after about a week, and move relaxations that have not started from the slower cluster to the faster one. Relaxations that have started can move too, since both clusters give the same results and a relaxation resumes from its last step, but moving only unstarted ones is simpler.

### Running on the CS cluster

- Load `nvhpc/26.1-CUDA-13.1.0`, then the HPC-X MPI that comes with it: `source <nvhpc>/comm_libs/13.1/hpcx/hpcx-2.25.1/hpcx-init.sh && hpcx_load`.
- Set `QE_GPU_LAUNCHER=mpirun`. HPC-X cannot be started by `srun` on this cluster, because Slurm here only supports PMI-2.
- Ask for memory: `--mem=60G` for one GPU. Both the `gpu` and `cpu` partitions default to 256 MB per CPU.
- Pick the GPU with `--constraint=h100_94gb` (QE build `qe-7.5-h100`, `BACKEND=h100`) or `--constraint=a100_80gb` (`qe-7.5-a100`, `BACKEND=a100`).

## Updating these numbers

Run this from the repository root with the `snb` environment, then copy the printed tables into this page:

```bash
python -m relaxgen.analysis.cost_analysis --markdown --output docs/cost_analysis/cost_numbers.json
```

The numbers change as campaigns run and as the queue gets busier or quieter. The queue waits and SU rates are the most likely to change.
