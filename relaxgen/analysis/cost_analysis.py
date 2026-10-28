#!/usr/bin/env python3
"""Estimate what the DFT relaxations cost on UVA's Rivanna cluster.

Collects four kinds of measurement and combines them into time and SU
estimates for the planned campaigns:

* SCF times per hardware type, from the status files of every QE job the
  campaigns ran (each job segment records its device and step wall times).
* Queue waits for the campaigns' own jobs, from ``sacct``, plus the zero-work
  probe jobs of the queue-wait survey.
* SU per hour per job shape, from ``sacct`` billing (1 SU = billing / 10 on
  Rivanna), and the allocation balance from ``allocations``.
* SCFs per relaxation to each force target, from the trajectories
  (see ``steps_to_fmax``).

``--markdown`` prints the tables used in ``docs/cost_analysis/README.md``.
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from relaxgen.analysis import steps_to_fmax
from relaxgen.dispatch.queue_wait_survey import _probe_log, parse_time

PILOT = Path("/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot")
RELAXGEN = Path("/scratch/gqs7zr/relaxgen")
SCF_MANIFESTS = [
    PILOT / "09_umlip_dft_handoff_threshold/scf_manifest.json",
    PILOT / "17_qe_backend_check_singlenode/scf_manifest.json",
    *(RELAXGEN / f"threshold_rerun/{model}/scf_manifest.json" for model in ("eqv3_bfgs", "mace_ft", "uma")),
]
PROBE_DIR = PILOT / "18_queue_wait_survey"
STAGE_A_DIR = RELAXGEN / "umlip_ensemble"
CAMPAIGN_JOBS = ("qd_relaxgen_b01", "qd_vbi_diversity", "qd_stage_b", "qd_thresh_rerun")
ACCOUNT = "raiselab"

HARDWARE = ("B200", "A100 80GB", "A100 40GB", "CPU, 2 nodes, 84 ranks",
            "CPU, 1 node, 84 ranks", "CPU, 1 node, 96 ranks", "CPU, 1 node, 42 ranks")
PARTITION_HARDWARE = {"gpu-b200": "B200", "gpu-a100-80": "A100 80GB", "gpu-a100-40": "A100 40GB",
                      "parallel": "CPU, 2 nodes, 84 ranks", "standard-afton": "CPU, 1 node, 42 ranks"}
GPU_HARDWARE = ("B200", "A100 80GB", "A100 40GB")
ANY_GPU = "Any GPU (all GPU rows above)"
PROBE_LABELS = {"parallel_2x42_6G": "CPU, 2 nodes, 84 ranks", "afton_1x42_6G": "CPU, 1 node, 42 ranks",
                "afton_1x84_6G": "CPU, 1 node, 84 ranks", "gpu_1x": "1 GPU (A100 or B200)"}

# Estimate settings. Each relaxation runs as jobs of at most SEGMENT_HOURS; a
# job starts a new SCF only if it should finish inside SEGMENT_FILL of that.
ESTIMATE_HARDWARE = {"gpu": "A100 80GB", "cpu": "CPU, 2 nodes, 84 ranks"}
SEGMENT_HOURS = {"gpu": 12, "cpu": 24}
SEGMENT_FILL = 0.9
GPU_LIMIT = 32          # per-user running GPUs (gpu QOS)
CPU_JOB_LIMIT = 71      # per-user cores on parallel (6,000) / 84 cores per job
FMAX_TARGETS = ("0.01", "0.005", "0.001")


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def load(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def hardware(device: str) -> str | None:
    """Map a recorded device ("1x NVIDIA B200", "cpu:<hosts>:84ranks:28pools") to a HARDWARE label."""
    for key, label in (("B200", "B200"), ("A100-SXM4-80GB", "A100 80GB"), ("A100-SXM4-40GB", "A100 40GB")):
        if key in device:
            return label
    match = re.match(r"cpu:(.+):(\d+)ranks", device)
    if match:
        hosts, ranks = match.groups()
        nodes = 2 if ("," in hosts or "[" in hosts) else 1
        return f"CPU, {nodes} node{'s' if nodes > 1 else ''}, {ranks} ranks"
    return None


def summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {"n": len(array), "median": float(np.median(array)), "mean": float(np.mean(array)),
            "p90": float(np.quantile(array, 0.9))}


# ----------------------------------------------------------------------------- SCF times

def scf_times() -> dict[str, dict[str, dict[str, float]]]:
    """Minutes per SCF. "scratch": the first SCF of a job (no saved electronic state);
    "running": later SCFs of a relaxation, which reuse the previous step's state."""
    scratch, running = defaultdict(list), defaultdict(list)
    for manifest in steps_to_fmax.CAMPAIGNS.values():
        for task in load(manifest).get("tasks", []):
            for segment in load(task["status"]).get("segments", []):
                label = hardware(segment.get("allocated_device") or "")
                steps = [seconds / 60 for seconds in segment.get("step_wall_seconds") or []]
                if label and steps:
                    scratch[label].append(steps[0])
                    running[label].extend(steps[1:])
    for manifest in SCF_MANIFESTS:
        for task in load(manifest).get("tasks", []):
            status = load(task["status"])
            runtime = status.get("runtime", {})
            label = hardware(str(runtime.get("allocated_gpu_name") or ""))
            if status.get("execution_status") == "completed" and label and runtime.get("qe_wall_time_seconds"):
                scratch[label].append(float(runtime["qe_wall_time_seconds"]) / 60)
    return {label: {"scratch": summary(scratch[label]), "running": summary(running[label])} for label in HARDWARE}


# ----------------------------------------------------------------------------- queue and billing

def campaign_jobs(since: str) -> list[dict[str, Any]]:
    output = run(["sacct", "-X", "-n", "-P", "-S", since, "-u", getpass.getuser(),
                  "--name=" + ",".join(CAMPAIGN_JOBS),
                  "--format=JobID,Partition,State,Eligible,Start,ElapsedRaw,AllocTRES"])
    jobs = []
    for line in output.splitlines():
        job_id, partition, state, eligible, start, elapsed, tres = line.split("|")
        billing = re.search(r"billing=(\d+)", tres)
        gpus = re.search(r"gres/gpu=(\d+)", tres)
        jobs.append({
            "job_id": job_id, "state": state, "partition": partition,
            "hardware": next((label for prefix, label in PARTITION_HARDWARE.items()
                              if partition.startswith(prefix) and "," not in partition), None),
            "eligible": parse_time(eligible), "start": parse_time(start),
            "elapsed_h": int(elapsed or 0) / 3600,
            "billing": int(billing.group(1)) if billing else None,
            "gpus": int(gpus.group(1)) if gpus else 0,
            "cpu": partition.startswith(("parallel", "standard")),
        })
    return jobs


def queue_waits(jobs: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    started, pending = defaultdict(list), defaultdict(list)
    for job in jobs:
        if job["start"] and job["eligible"] and job["start"] >= job["eligible"]:
            hours = (job["start"] - job["eligible"]).total_seconds() / 3600
            started[job["hardware"]].append(hours)
            if job["hardware"] in GPU_HARDWARE:
                started[ANY_GPU].append(hours)
        elif job["state"] == "PENDING" and job["eligible"]:
            pending["CPU" if job["cpu"] else "GPU"].append((now - job["eligible"]).total_seconds() / 3600)
    return {"started": {label: summary(started[label]) for label in (*HARDWARE, ANY_GPU) if started[label]},
            "pending": {kind: summary(ages) for kind, ages in pending.items()}}


def probe_waits() -> dict[str, dict[str, float]]:
    """Waits (hours) of the zero-work probe jobs, grouped by shape without the time limit."""
    probes = _probe_log(PROBE_DIR)
    if not probes:
        return {}
    output = run(["sacct", "-X", "-n", "-P", "-j", ",".join(p["job_id"] for p in probes),
                  "--format=JobID,Eligible,Start"])
    timing = {job_id: (parse_time(eligible), parse_time(start))
              for job_id, eligible, start in (line.split("|") for line in output.splitlines())}
    waits, pending = defaultdict(list), Counter()
    for probe in probes:
        family = re.sub(r"_\d+h$", "", probe["shape"])
        eligible, start = timing.get(probe["job_id"], (None, None))
        if eligible and start:
            waits[family].append((start - eligible).total_seconds() / 3600)
        else:
            pending[family] += 1
    return {family: {**summary(waits[family]), "still_pending": pending[family]}
            for family in sorted(set(waits) | set(pending))}


def su_rates(jobs: list[dict[str, Any]]) -> dict[str, float]:
    """SU per hour for each hardware: the most common billing of its started jobs, / 10."""
    billing = defaultdict(Counter)
    for job in jobs:
        if job["start"] and job["billing"] and job["hardware"]:
            billing[job["hardware"]][job["billing"]] += 1
    return {label: counts.most_common(1)[0][0] / 10 for label, counts in billing.items()}


def average_busy(jobs: list[dict[str, Any]], now: datetime, days: int = 7) -> dict[str, float]:
    """Average GPUs and 2-node CPU jobs the campaigns had running over the last ``days``."""
    window_start = now - timedelta(days=days)
    gpu_hours = cpu_job_hours = 0.0
    for job in jobs:
        if not job["start"]:
            continue
        end = min(job["start"] + timedelta(hours=job["elapsed_h"]), now)
        hours = max(0.0, (end - max(job["start"], window_start)).total_seconds() / 3600)
        gpu_hours += hours * job["gpus"]
        cpu_job_hours += hours if job["hardware"] == "CPU, 2 nodes, 84 ranks" else 0.0
    return {"days": days, "gpus": gpu_hours / (24 * days), "cpu_jobs": cpu_job_hours / (24 * days)}


def campaign_su(jobs: list[dict[str, Any]]) -> float:
    return sum(job["billing"] * job["elapsed_h"] / 10 for job in jobs if job["billing"])


def allocation() -> dict[str, Any]:
    for line in run(["allocations", "-a", ACCOUNT]).splitlines():
        fields = line.split()
        if fields and fields[-1] == "True" and len(fields) >= 8:
            return {"account": ACCOUNT, "start": fields[1], "end": fields[3],
                    "allocated_su": float(fields[4]), "remaining_su": float(fields[5])}
    return {}


# ----------------------------------------------------------------------------- SCFs per relaxation

def scfs_per_relaxation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """SCFs to each DFT force target for EqV3, MACE-ft and UMA starts.

    EqV3 to 0.01 is the median of finished batch 1 relaxations; the extra SCFs to
    reach 0.005 and 0.001 are medians over every relaxation that reached both
    targets.  MACE-ft and UMA starts are further from the DFT minimum: their extra
    SCFs are the median difference to fmax 0.05 against the EqV3 run from the
    same start (Stage B and the diversity campaign)."""
    base = [r["first_below"]["0.01"] for r in rows if r["campaign"] == "batch1" and r["first_below"]["0.01"]]
    increments = {}
    for target in ("0.005", "0.001"):
        increments[target] = [r["first_below"][target] - r["first_below"]["0.01"] for r in rows
                              if r["first_below"]["0.01"] and r["first_below"][target]]
    starts = defaultdict(dict)
    for r in rows:
        if r["campaign"] == "stage_b":
            starts["b:" + r["label"].rsplit("__", 1)[0]][r["model"]] = r
        elif r["campaign"] == "diversity":
            starts["d:" + r["label"].replace(r["model"] + "_", "", 1)][r["model"]] = r
    extra = {}
    for model in ("mace_ft", "uma"):
        extra[model] = [runs[model]["first_below"]["0.05"] - runs["eqv3"]["first_below"]["0.05"]
                        for runs in starts.values()
                        if model in runs and "eqv3" in runs
                        and runs[model]["first_below"]["0.05"] and runs["eqv3"]["first_below"]["0.05"]]
    eqv3 = {"0.01": float(np.median(base))}
    for target in ("0.005", "0.001"):
        eqv3[target] = eqv3["0.01"] + float(np.median(increments[target]))
    models = {"eqv3": eqv3}
    for model in ("mace_ft", "uma"):
        models[model] = {target: value + float(np.median(extra[model])) for target, value in eqv3.items()}
    return {"scfs": {m: {t: round(v) for t, v in targets.items()} for m, targets in models.items()},
            "inputs": {"eqv3_to_0.01": summary(base),
                       **{f"0.01_to_{t}": summary(v) for t, v in increments.items()},
                       **{f"{m}_extra": summary(v) for m, v in extra.items()}}}


# ----------------------------------------------------------------------------- estimate

def one_relaxation(kind: str, scfs: int, times: dict[str, Any]) -> tuple[float, int]:
    """Run hours and number of jobs for one relaxation of ``scfs`` SCFs on ``kind`` (gpu/cpu)."""
    entry = times[ESTIMATE_HARDWARE[kind]]
    first, later = entry["scratch"]["median"] / 60, entry["running"]["median"] / 60
    hours, jobs, left = 0.0, 0, scfs
    while left > 0:
        jobs += 1
        used = first
        hours += first
        left -= 1
        while left > 0 and used + later <= SEGMENT_HOURS[kind] * SEGMENT_FILL:
            used += later
            hours += later
            left -= 1
    return hours, jobs


def plan_estimate(count: int, scfs: int, scenario: dict[str, Any], times, rates, waits) -> dict[str, Any]:
    """Days and SU for ``count`` relaxations.

    Each of the scenario's GPU and CPU job slots runs one job after another, and
    every job waits in the queue first (the controller's job cap counts queued jobs).
    A scenario with ``gpu_running`` set gives GPUs actually running, so no GPU waits
    are added.  Days use the median wait (low end) and the mean wait (high end)."""
    per = {kind: one_relaxation(kind, scfs, times) for kind in ("gpu", "cpu")}
    result = {}
    for statistic in ("median", "mean"):
        rate = {}
        for kind in ("gpu", "cpu"):
            hours, jobs = per[kind]
            wait = 0.0 if kind == "gpu" and scenario.get("gpu_running") else waits[kind][statistic]
            if scenario[kind]:
                rate[kind] = scenario[kind] / (hours + jobs * wait)   # relaxations per hour
        total = sum(rate.values())
        result[f"days_{statistic}"] = count / total / 24
        if statistic == "median":
            result["su"] = count * sum(rate[k] / total * per[k][0] * rates[ESTIMATE_HARDWARE[k]] for k in rate)
    return result


# ----------------------------------------------------------------------------- markdown

def minutes(value: float) -> str:
    return f"{value:.0f} min"


def hours(value: float) -> str:
    return f"{value * 60:.0f} min" if value < 1 else f"{value:.1f} h"


def millions(value: float) -> str:
    return f"{value / 1e6:.2f}M" if value < 1e6 else f"{value / 1e6:.1f}M"


def days(low: float, high: float) -> str:
    return f"{low:.0f} days" if round(low) == round(high) else f"{low:.0f}–{high:.0f} days"


def markdown(result: dict[str, Any]) -> str:
    out = []
    times, waits, rates = result["scf_times"], result["queue"], result["su_per_hour"]
    out += ["### SCF times", "",
            "| Hardware | SCF from scratch, median (n) | SCF inside a relaxation, median / p90 (n) |",
            "|---|---|---|"]
    for label in HARDWARE:
        scratch, running = times[label]["scratch"], times[label]["running"]
        if not scratch["n"] and not running["n"]:
            continue
        first = f"{minutes(scratch['median'])} ({scratch['n']})" if scratch["n"] else "-"
        later = (f"{minutes(running['median'])} / {minutes(running['p90'])} ({running['n']})"
                 if running["n"] else "-")
        out.append(f"| {label} | {first} | {later} |")
    out += ["", "### SCFs per relaxation", "", "| DFT fmax | EqV3 start | MACE-ft start | UMA start |", "|---|---|---|---|"]
    scfs = result["scfs_per_relaxation"]["scfs"]
    for target in FMAX_TARGETS:
        out.append(f"| {target} eV/Å | {scfs['eqv3'][target]} | {scfs['mace_ft'][target]} | {scfs['uma'][target]} |")
    out += ["", "### Queue waits of campaign jobs", "", "| Started on | Jobs | Median wait | Average wait | 90th percentile |", "|---|---|---|---|---|"]
    for label, entry in waits["started"].items():
        out.append(f"| {label} | {entry['n']} | {hours(entry['median'])} | {hours(entry['mean'])} | {hours(entry['p90'])} |")
    for kind, entry in waits["pending"].items():
        out.append(f"| Still waiting ({kind}) | {entry['n']} | {hours(entry['median'])} so far | | |")
    out += ["", "### Probe jobs", "", "| Probe shape | Started | Median wait | 90th percentile | Still waiting |", "|---|---|---|---|---|"]
    for family, entry in result["probes"].items():
        median = hours(entry["median"]) if entry["n"] else "-"
        p90 = hours(entry["p90"]) if entry["n"] else "-"
        out.append(f"| {PROBE_LABELS.get(family, family)} | {entry['n']} | {median} | {p90} | {entry['still_pending']} |")
    out += ["", "### SU per hour and per SCF", "", "| Hardware | SU per hour | SU per SCF inside a relaxation | SU per SCF from scratch |", "|---|---|---|---|"]
    for label in HARDWARE:
        if label not in rates:
            continue
        rate, entry = rates[label], times[label]
        later = f"{rate * entry['running']['median'] / 60:.0f}" if entry["running"]["n"] else "-"
        first = f"{rate * entry['scratch']['median'] / 60:.0f}" if entry["scratch"]["n"] else "-"
        out.append(f"| {label} | {rate:.0f} | {later} | {first} |")
    out += ["", "### One relaxation", "", f"| DFT fmax | SCFs (EqV3 start) | On one {ESTIMATE_HARDWARE['gpu']} GPU | On {ESTIMATE_HARDWARE['cpu']} |", "|---|---|---|---|"]
    for entry in result["one_relaxation"]:
        gpu, cpu = entry["gpu"], entry["cpu"]
        out.append(f"| {entry['fmax']} eV/Å | {entry['scfs']} | {gpu['hours']:.1f} h, {gpu['su']:.0f} SU | "
                   f"{cpu['hours']:.1f} h, {cpu['su']:.0f} SU |")
    for plan in result["plans"]:
        out += ["", f"### {plan['name']}", "", "| DFT fmax | SCFs per relaxation | " +
                " | ".join(plan["scenarios"]) + " |", "|---|---|" + "---|" * len(plan["scenarios"])]
        for row in plan["rows"]:
            cells = [f"{days(c['days_median'], c['days_mean'])}, {millions(c['su'])} SU" for c in row["cells"]]
            out.append(f"| {row['fmax']} eV/Å | {row['scfs']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- main

def analyze(since: str) -> dict[str, Any]:
    now = datetime.now()
    times = scf_times()
    jobs = campaign_jobs(since)
    rates = su_rates(jobs)
    waits = queue_waits(jobs, now)
    busy = average_busy(jobs, now)
    steps = scfs_per_relaxation(steps_to_fmax.scan(steps_to_fmax.CAMPAIGNS))
    starts = len(list((STAGE_A_DIR / "eqv3").rglob("*.traj")))
    wait_hours = {"gpu": {q: waits["started"][ANY_GPU][q] for q in ("median", "mean")},
                  "cpu": {q: waits["started"][ESTIMATE_HARDWARE["cpu"]][q] for q in ("median", "mean")}}
    realistic_gpus = max(1, round(busy["gpus"]))
    scenarios = {f"~{realistic_gpus} GPUs running + {CPU_JOB_LIMIT} CPU jobs":
                     {"gpu": realistic_gpus, "cpu": CPU_JOB_LIMIT, "gpu_running": True},
                 f"{GPU_LIMIT} GPU jobs + {CPU_JOB_LIMIT} CPU jobs": {"gpu": GPU_LIMIT, "cpu": CPU_JOB_LIMIT},
                 f"{GPU_LIMIT} GPU jobs only": {"gpu": GPU_LIMIT, "cpu": 0}}
    one = []
    for target in FMAX_TARGETS:
        scfs = steps["scfs"]["eqv3"][target]
        entry = {"fmax": target, "scfs": scfs}
        for kind in ("gpu", "cpu"):
            run_hours, _ = one_relaxation(kind, scfs, times)
            entry[kind] = {"hours": run_hours, "su": run_hours * rates[ESTIMATE_HARDWARE[kind]]}
        one.append(entry)
    plans = []
    for name, count, models in ((f"Plan A: EqV3 starts only ({starts:,} relaxations)", starts, ("eqv3",)),
                                (f"Plan B: EqV3, MACE-ft and UMA starts ({3 * starts:,} relaxations)",
                                 3 * starts, ("eqv3", "mace_ft", "uma"))):
        rows = []
        for target in FMAX_TARGETS:
            scfs = round(np.mean([steps["scfs"][m][target] for m in models]))
            rows.append({"fmax": target, "scfs": scfs,
                         "cells": [plan_estimate(count, scfs, scenario, times, rates, wait_hours)
                                   for scenario in scenarios.values()]})
        plans.append({"name": name, "count": count, "scenarios": list(scenarios), "rows": rows})
    return {"generated": now.isoformat(timespec="seconds"), "sacct_since": since, "scf_times": times,
            "queue": waits, "probes": probe_waits(), "su_per_hour": rates, "campaign_su_so_far": campaign_su(jobs),
            "average_busy": busy, "allocation": allocation(), "scfs_per_relaxation": steps,
            "stage_a_starts": starts, "estimate_waits_h": wait_hours, "one_relaxation": one, "plans": plans}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2026-10-20", help="First day of campaign jobs to read from sacct.")
    parser.add_argument("--output", type=Path, help="Write all numbers to this JSON file.")
    parser.add_argument("--markdown", action="store_true", help="Print the README tables.")
    args = parser.parse_args()
    result = analyze(args.since)
    if args.output:
        args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(markdown(result) if args.markdown else json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
