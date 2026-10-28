# Estimated Cost Analysis

### Measured on UVA's Rivanna cluster (Afton nodes), October 20–28, 2026

This page estimates how long our DFT relaxations take and how much they cost on Rivanna. Every number comes from our own jobs: the relaxation campaigns, the handoff threshold runs and a set of small test jobs. The script that computes them is `relaxgen/analysis/cost_analysis.py`, and the raw numbers are in `cost_numbers.json` in this folder.

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

## Updating these numbers

Run this from the repository root with the `snb` environment, then copy the printed tables into this page:

```bash
python -m relaxgen.analysis.cost_analysis --markdown --output docs/cost_analysis/cost_numbers.json
```

The numbers change as campaigns run and as the queue gets busier or quieter. The queue waits and SU rates are the most likely to change.
