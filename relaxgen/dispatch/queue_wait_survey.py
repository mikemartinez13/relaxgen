#!/usr/bin/env python3
"""Measure Slurm queue waits as a function of job shape and time limit.

The dispatch controller should choose a CPU segment length by expected
progress per wall hour, which needs the queue wait for each request shape and
time limit, not one number per partition.

* ``history``: every user's jobs from ``sacct`` binned by partition, node
  count, memory fraction, time limit and hour of day.  Waits run from
  eligibility to start.  Members of multi-element arrays are reported
  separately because array throttles and per-user limits inflate their waits.
* ``probe-round``: submit one zero-work job (``true``) per probe shape whose
  previous probe has started.  Slurm schedules on the request, not the payload,
  so a probe waits as long as a real segment of that shape would.  Run it from
  ``scrontab`` every 2 hours until ``--until``.
* ``probe-report``: waits per probe shape from ``sacct``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

NODE_MEMORY_GB = {"standard-afton": 768.0, "standard-rivanna": 384.0, "parallel": 768.0}
LIMIT_BINS_H = (1, 2, 4, 8, 12, 24)

# The request shapes of candidate CPU segments, plus one GPU control.
PROBE_SHAPES: dict[str, list[str]] = {
    **{
        f"afton_1x84_6G_{hours}h": ["-p", "standard", "-N1", "-C", "afton", "--ntasks=84",
                                    "--mem-per-cpu=6G", f"--time={hours:02d}:00:00"]
        for hours in (2, 4, 8, 12, 24)
    },
    # One node, 42 ranks at measured memory (~3 GB/rank used): fits a partly used node (added 2026-10-24).
    **{
        f"afton_1x42_6G_{hours}h": ["-p", "standard", "-N1", "-C", "afton", "--ntasks=42",
                                    "--mem-per-cpu=6G", f"--time={hours:02d}:00:00"]
        for hours in (4, 12, 24)
    },
    **{
        f"parallel_2x42_6G_{hours}h": ["-p", "parallel", "-N2", "--ntasks=84", "--ntasks-per-node=42",
                                       "--mem-per-cpu=6G", f"--time={hours:02d}:00:00"]
        for hours in (2, 3, 4, 6, 12, 24)
    },
    "gpu_1x_12h": ["-p", "gpu", "--gres=gpu:1", "-C", "a100|b200", "--ntasks=1", "--cpus-per-task=4",
                   "--mem=60G", "--time=12:00:00"],
}
PROBE_PREFIX = "qwprobe_"


def run(command: Sequence[str]) -> str:
    return subprocess.run(list(command), check=True, capture_output=True, text=True).stdout


def parse_time(value: str) -> datetime | None:
    if value in ("", "None", "Unknown"):
        return None
    return datetime.fromisoformat(value)


def limit_hours(value: str) -> float | None:
    if value in ("", "UNLIMITED", "Partition_Limit", "INVALID"):
        return None
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    parts = [int(part) for part in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return 24 * days + parts[0] + parts[1] / 60 + parts[2] / 3600


def memory_gb(value: str, ncpus: int, nnodes: int) -> float | None:
    """Total requested memory in GB (sacct ReqMem, with optional c/n suffix)."""

    if not value or value == "0":
        return None
    per = value[-1] if value[-1] in "cn" else ""
    number = value[:-1] if per else value
    units = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
    gb = float(number[:-1]) * units[number[-1]] if number[-1] in units else float(number) / 1024
    return gb * (ncpus if per == "c" else nnodes if per == "n" else 1)


def limit_bin(hours: float) -> str:
    low = 0
    for high in LIMIT_BINS_H:
        if hours <= high:
            return f"{low}-{high}h"
        low = high
    return f">{LIMIT_BINS_H[-1]}h"


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {"n": len(ordered), "median_min": statistics.median(ordered), "p75_min": pick(0.75),
            "p90_min": pick(0.9)}


def history(args: argparse.Namespace) -> dict[str, Any]:
    output = run(["sacct", "-a", "-X", "-n", "-P", "-S", f"now-{args.days}days", "-E", "now",
                  "--format=JobID,Partition,NNodes,NCPUS,ReqMem,Timelimit,Eligible,Start"])
    rows = []
    array_sizes: dict[str, int] = defaultdict(int)
    for line in output.splitlines():
        job_id, partition, nnodes, ncpus, reqmem, limit, eligible, start = line.split("|")
        partition = partition.split(",")[0]
        if not partition.startswith(("standard", "parallel", "gpu")):
            continue
        eligible_at, start_at, hours = parse_time(eligible), parse_time(start), limit_hours(limit)
        if eligible_at is None or start_at is None or hours is None or start_at < eligible_at:
            continue
        array_id = job_id.split("_")[0] if "_" in job_id else None
        if array_id:
            array_sizes[array_id] += 1
        nodes, cpus = int(nnodes), int(ncpus)
        total_gb = memory_gb(reqmem, cpus, nodes)
        node_gb = NODE_MEMORY_GB.get(partition)
        rows.append({
            "partition": partition, "nodes": nodes, "cpus": cpus, "limit_h": hours,
            "memory_fraction": total_gb / (node_gb * nodes) if total_gb and node_gb else None,
            "hour": eligible_at.hour, "wait_min": (start_at - eligible_at).total_seconds() / 60,
            "array_id": array_id,
        })
    for row in rows:
        row["multi_array"] = bool(row["array_id"]) and array_sizes[row["array_id"]] > 1

    def memory_bin(fraction: float | None) -> str:
        if fraction is None:
            return "mem ?"
        return "mem <1/4" if fraction < 0.25 else "mem 1/4-2/3" if fraction < 0.67 else "mem >=2/3"

    def nodes_bin(nodes: int) -> str:
        return "1 node" if nodes == 1 else "2 nodes" if nodes == 2 else "3-4 nodes" if nodes <= 4 else "5+ nodes"

    tables: dict[str, dict[str, list[float]]] = {"shape": defaultdict(list), "hour": defaultdict(list)}
    for row in rows:
        if row["multi_array"]:
            continue
        if row["partition"].startswith("gpu"):
            shape = f"{row['partition']} | {limit_bin(row['limit_h'])}"
        else:
            cpu_bin = "<=16 cpu" if row["cpus"] <= 16 else "17-48 cpu" if row["cpus"] <= 48 else ">48 cpu"
            shape = (f"{row['partition']} | {nodes_bin(row['nodes'])} | {cpu_bin} | "
                     f"{memory_bin(row['memory_fraction'])} | {limit_bin(row['limit_h'])}")
        tables["shape"][shape].append(row["wait_min"])
        if row["partition"] in ("standard-afton", "parallel") and row["cpus"] > 16:
            tables["hour"][f"{row['partition']} | eligible {row['hour']:02d}h"].append(row["wait_min"])
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "window_days": args.days,
        "jobs": len(rows),
        "excluded_multi_array_members": sum(row["multi_array"] for row in rows),
        **{name: {key: quantiles(values) for key, values in sorted(table.items()) if len(values) >= args.min_jobs}
           for name, table in tables.items()},
    }
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "history_report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"Queue waits, eligible to start; {report['jobs']} jobs over {args.days} days; "
             f"{report['excluded_multi_array_members']} multi-element array members excluded.", ""]
    for name, table in (("By shape", report["shape"]), ("By hour of day (CPU jobs > 16 cores)", report["hour"])):
        lines += [name, f"{'':74s} {'n':>6s} {'median':>8s} {'p75':>8s} {'p90':>8s}  (min)"]
        lines += [f"{key:74s} {q['n']:>6d} {q['median_min']:>8.0f} {q['p75_min']:>8.0f} {q['p90_min']:>8.0f}"
                  for key, q in table.items()]
        lines.append("")
    (out / "history_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    return report


def _probe_log(out: Path) -> list[dict[str, Any]]:
    log = out / "probes.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def probe_round(args: argparse.Namespace) -> None:
    out = args.output_dir.expanduser().resolve()
    if args.until and datetime.now() >= datetime.fromisoformat(args.until):
        print(f"Probe campaign ended at {args.until}; nothing submitted.")
        return
    pending = set(run(["squeue", "-h", "--me", "-t", "PENDING", "-o", "%j"]).split())
    (out / "slurm").mkdir(parents=True, exist_ok=True)
    with (out / "probes.jsonl").open("a", encoding="utf-8") as log:
        for shape, request in PROBE_SHAPES.items():
            name = PROBE_PREFIX + shape
            if name in pending:
                continue  # One probe per shape in the queue, so probes never compete with each other.
            job_id = run(["sbatch", "--parsable", f"--job-name={name}", "--output=/dev/null",
                          "--mail-type=NONE", *request, "--wrap", "true"]).strip().split(";")[0]
            log.write(json.dumps({"shape": shape, "job_id": job_id,
                                  "submitted": datetime.now().isoformat(timespec="seconds")}) + "\n")
            print(f"submitted {shape}: {job_id}")


def probe_report(args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_dir.expanduser().resolve()
    probes = _probe_log(out)
    if not probes:
        raise SystemExit("No probes submitted yet.")
    output = run(["sacct", "-X", "-n", "-P", "-j", ",".join(p["job_id"] for p in probes),
                  "--format=JobID,Eligible,Start,State"])
    timing = {}
    for line in output.splitlines():
        job_id, eligible, start, state = line.split("|")
        timing[job_id] = (parse_time(eligible), parse_time(start), state)
    waits: dict[str, list[float]] = defaultdict(list)
    pending: dict[str, int] = defaultdict(int)
    for probe in probes:
        eligible, start, _ = timing.get(probe["job_id"], (None, None, "?"))
        if eligible and start:
            waits[probe["shape"]].append((start - eligible).total_seconds() / 60)
        else:
            pending[probe["shape"]] += 1
    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "shapes": {shape: {**(quantiles(waits[shape]) if waits[shape] else {"n": 0}),
                                 "still_pending": pending[shape], "waits_min": waits[shape]}
                         for shape in PROBE_SHAPES}}
    (out / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'shape':24s} {'n':>4s} {'median':>8s} {'p75':>8s} {'p90':>8s} {'pending':>8s}  (min)")
    for shape, entry in report["shapes"].items():
        if entry["n"]:
            print(f"{shape:24s} {entry['n']:>4d} {entry['median_min']:>8.0f} {entry['p75_min']:>8.0f} "
                  f"{entry['p90_min']:>8.0f} {entry['still_pending']:>8d}")
        else:
            print(f"{shape:24s} {0:>4d} {'-':>8s} {'-':>8s} {'-':>8s} {entry['still_pending']:>8d}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("history", "probe-round", "probe-report"):
        command = sub.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "history":
            command.add_argument("--days", type=int, default=7)
            command.add_argument("--min-jobs", type=int, default=5)
        if name == "probe-round":
            command.add_argument("--until", help="ISO time after which rounds submit nothing.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    {"history": history, "probe-round": probe_round, "probe-report": probe_report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
