#!/usr/bin/env python3
"""Dispatch QE manifest tasks across GPU and CPU Slurm backends.

Every task of an SCF manifest (``umlip_dft_handoff_threshold.py`` layout) is
one unit.  ``tick`` keeps GPU jobs queued for the head of the remaining work
and sends tail units to a CPU backend only when that CPU backend is expected to
finish them sooner than the GPU queue would.  Estimates come from this user's
real Slurm submit-to-start delays and from measured run times, never from
``sbatch --test-only``.

A unit may be queued on several backends at once.  The job that starts first
atomically creates ``claims/<task>.json``; later twins exit immediately and
the next tick cancels them.  The running worker refreshes the claim's mtime,
so a claim whose job died goes stale and is released.  A CPU backend with a
``check`` entry stays disabled until its backend-check result agrees with the
GPU reference within ``check_tolerances``.

Relaxations hand over from CPU to GPU.  While a unit runs on a CPU backend,
``tick`` keeps a GPU job queued for it.  When that job starts it writes
``preempt/<task>.json``; the CPU worker stops QE within about 10 s and releases
the claim, and the GPU job resumes from the last stored step.  Only the SCF in
progress on the CPU is lost: the GPU's first SCF starts from scratch after any
hand-off anyway, and finishes no later than the CPU would have.

A manifest task with ``"withdrawn": true`` is no longer wanted: ``tick``
cancels all of its jobs, queued or running, and never submits it again.

Layout under ``--dispatch-dir``::

    config.json   backends and policy (written by ``init``)
    state.json    submissions per unit (written only by ``tick``)
    claims/       one file per running unit (written only by workers)
    preempt/      hand-over requests from a GPU job to the CPU job holding a unit
    events/       one file per worker outcome
    log.jsonl     one line per tick with its decisions
    slurm/        job stdout and stderr
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED"}
HISTORY_WINDOW = "now-3days"
CLAIM_REFRESH_SECONDS = 60
PREEMPT_POLL_SECONDS = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_slurm_time(value: str | None) -> datetime | None:
    if not value or value in {"Unknown", "None", "N/A"}:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Slurm access.  Everything that talks to the scheduler lives here so the tick
# logic can be tested against a fake.
# --------------------------------------------------------------------------


@dataclass
class JobInfo:
    key: str
    state: str
    submit: datetime | None
    start: datetime | None
    partition: str = ""


class SlurmCLI:
    def __init__(self, user: str | None = None) -> None:
        self.user = user or os.environ.get("USER", "")

    @staticmethod
    def _run(command: Sequence[str]) -> str:
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout

    def active_jobs(self, name: str) -> dict[str, JobInfo]:
        """Array elements of this user's jobs named ``name`` that are still queued or running."""

        output = self._run(
            ["squeue", "-r", "-h", "-u", self.user, "-n", name, "-o", "%i|%T|%V|%S|%P"]
        )
        jobs = {}
        for line in output.splitlines():
            key, state, submit, start, partition = line.split("|")
            jobs[key] = JobInfo(
                key, state, parse_slurm_time(submit),
                parse_slurm_time(start) if state != "PENDING" else None, partition,
            )
        return jobs

    def finished_jobs(self, keys: Sequence[str]) -> dict[str, JobInfo]:
        if not keys:
            return {}
        # sacct finds an array element that never started (listed as "123_[4]") only
        # by its job ID, so query the job IDs; each submission is a one-element array.
        job_ids = sorted({key.split("_")[0] for key in keys})
        output = self._run(
            ["sacct", "-n", "-P", "-X", "-j", ",".join(job_ids),
             "--format=JobID,State,Submit,Start,Partition"]
        )
        wanted = set(keys)
        jobs = {}
        for line in output.splitlines():
            key, state, submit, start, partition = line.split("|")
            key = re.sub(r"_\[(\d+)\]$", r"_\1", key)
            if key not in wanted:
                continue
            jobs[key] = JobInfo(
                key, state.split()[0], parse_slurm_time(submit), parse_slurm_time(start), partition
            )
        return jobs

    def recent_waits(self, partitions: Sequence[str]) -> list[float]:
        """Eligible-to-start delays (s) of this user's jobs started recently in ``partitions``.

        Measured from eligibility, not submission, so time spent waiting on a
        job dependency is not counted as queue wait.
        """

        output = self._run(
            ["sacct", "-n", "-P", "-X", "-u", self.user, "-S", HISTORY_WINDOW,
             "--format=Partition,Eligible,Start"]
        )
        waits = []
        for line in output.splitlines():
            partition, submit, start = line.split("|")
            submitted, started = parse_slurm_time(submit), parse_slurm_time(start)
            if submitted and started and any(partition.startswith(p) for p in partitions):
                waits.append((started - submitted).total_seconds())
        return waits

    def submit(self, command: Sequence[str]) -> str:
        return self._run(command).strip().split(";")[0]

    def cancel(self, keys: Sequence[str]) -> None:
        if keys:
            subprocess.run(["scancel", *keys], check=False)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


PILOT_BACKEND_CHECK = Path(
    "/sfs/weka/scratch/gqs7zr/ai4mat/results/bi2te3_2vte_pilot/10_qe_backend_check_fullnode"
)


def backend_check_summary(manifest_dir: Path) -> Path:
    """The full-node CPU backend check: the nearest one above the campaign, else the pilot's."""

    for directory in (manifest_dir, *manifest_dir.parents):
        check = directory / "10_qe_backend_check_fullnode"
        if check.is_dir():
            return check / "backend_check_summary.json"
    return PILOT_BACKEND_CHECK / "backend_check_summary.json"


def default_backends(manifest_dir: Path) -> list[dict[str, Any]]:
    """Backends for single-point manifests; runtimes are seeded from measured runs."""

    fullnode_check = backend_check_summary(manifest_dir)
    return [
        {
            "name": "gpu",
            "kind": "gpu",
            "launcher": "slurm/run_umlip_dft_handoff_scf_slurm_uva.sh",
            "sbatch_args": [],
            "env": {},
            "max_jobs": 32,
            "cores": 0,
            "wait_partitions": ["gpu-a100", "gpu-b200"],
            # A100 ~48 min, B200 ~29 min per production SCF (2026-10-22).
            "runtime_prior_seconds": 2600,
            "wait_prior_seconds": 900,
            "check": None,
        },
        {
            "name": "cpu-parallel-2x42",
            "kind": "cpu",
            "launcher": "slurm/run_qe_scf_cpu_slurm_uva.sh",
            "sbatch_args": ["-p", "parallel", "-N2", "--ntasks=84", "--ntasks-per-node=42",
                            "--mem-per-cpu=16G", "--time=06:00:00"],
            "env": {"NPOOL": "28"},
            "max_jobs": 8,
            "cores": 84,
            "wait_partitions": ["parallel"],
            "runtime_prior_seconds": None,
            "wait_prior_seconds": 6 * 3600,
            "check": {"summary": fullnode_check.as_posix(), "task_index": 0},
        },
        {
            "name": "cpu-standard-42",
            "kind": "cpu",
            "launcher": "slurm/run_qe_scf_cpu_slurm_uva.sh",
            "sbatch_args": ["-p", "standard", "-N1", "-C", "afton", "--ntasks=42",
                            "--mem-per-cpu=16G", "--time=08:00:00"],
            "env": {"NPOOL": "14"},
            "max_jobs": 8,
            "cores": 42,
            "wait_partitions": ["standard"],
            "runtime_prior_seconds": None,
            "wait_prior_seconds": 24 * 3600,
            "check": {"summary": fullnode_check.as_posix(), "task_index": 1},
        },
    ]


def relax_backends(manifest_dir: Path) -> list[dict[str, Any]]:
    """Backends for resumable relaxations; 1- and 2-GPU requests race for each unit."""

    fullnode_check = default_backends(manifest_dir)[1]["check"]
    gpu = {
        "kind": "gpu",
        "launcher": "slurm/run_dft_relax_slurm_uva.sh",
        "env": {},
        "cores": 0,
        "wait_partitions": ["gpu-a100", "gpu-b200"],
        # Median of 1-GPU 12 h queue probes, 2026-10-24.
        "wait_prior_seconds": 3.4 * 3600,
        "check": None,
    }
    return [
        # Step priors are slow-case (from-scratch) production SCFs: B200 29 min,
        # A100 46 min on one GPU (2026-10-22); two GPUs assumed ~1.8x faster.
        {**gpu, "name": "gpu-1x", "sbatch_args": [], "max_jobs": 10,
         "runtime_prior_seconds": 12 * 3600, "segment_seconds": 12 * 3600,
         "step_prior_seconds": 2800},
        {**gpu, "name": "gpu-2x", "sbatch_args": ["--gres=gpu:2", "--ntasks=2", "--mem=100G"],
         "max_jobs": 5, "runtime_prior_seconds": 12 * 3600, "segment_seconds": 12 * 3600,
         "step_prior_seconds": 1600},
        {
            "name": "cpu-parallel-2x42",
            "kind": "cpu",
            "launcher": "slurm/run_dft_relax_cpu_slurm_uva.sh",
            "sbatch_args": ["-p", "parallel", "-N2", "--ntasks=84", "--ntasks-per-node=42",
                            "--mem-per-cpu=16G", "--time=24:00:00"],
            "env": {"NPOOL": "28"},
            "max_jobs": 5,
            "cores": 84,
            "wait_partitions": ["parallel"],
            "runtime_prior_seconds": 24 * 3600,
            "segment_seconds": 24 * 3600,
            # p90 of Stage B's measured steps (2026-10-24); the backend check's SCF
            # wall time is used first when there is one.
            "step_prior_seconds": 80 * 60,
            # 2-node probes waited a median 4-7 min (p90 ~1.5 h) for any limit, 2026-10-24.
            "wait_prior_seconds": 15 * 60,
            "check": fullnode_check,
        },
    ]


def init(args: argparse.Namespace) -> dict[str, Any]:
    dispatch_dir = args.dispatch_dir.expanduser().resolve()
    config_path = dispatch_dir / "config.json"
    if config_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {config_path}; pass --overwrite.")
    manifest = args.manifest.expanduser().resolve()
    config = {
        "manifest": manifest.as_posix(),
        "job_name": args.job_name,
        "max_attempts": 3,
        # Relaxations: a backend whose segment cannot fit this many slow-case steps
        # is not used, and a backend with this many zero-progress segments inside
        # the window is backed off for the rest of the window.
        "min_segment_steps": 3,
        "zero_progress_backoff_count": 2,
        "zero_progress_window_seconds": 48 * 3600,
        "claim_stale_seconds": 900,
        "cpu_core_budget": args.cpu_core_budget,
        # A CPU backend is used only when GPU completion is this many times later.
        "cpu_preference_margin": 1.0,
        "check_tolerances": {"energy_mev": 0.1, "max_force_ev_per_angstrom": 1.0e-3},
        "workload": args.workload,
        # Relaxations running on CPU move to a GPU as soon as one of their GPU jobs starts.
        "handover_to_gpu": args.workload == "relax",
        "backends": (relax_backends if args.workload == "relax" else default_backends)(manifest.parent),
    }
    write_json_atomic(config_path, config)
    for sub in ("claims", "events", "slurm", "preempt"):
        (dispatch_dir / sub).mkdir(parents=True, exist_ok=True)
    return config


# --------------------------------------------------------------------------
# Unit and backend state
# --------------------------------------------------------------------------


def backend_kind(config: dict[str, Any], name: str | None) -> str | None:
    return next((b["kind"] for b in config["backends"] if b["name"] == name), None)


def handover_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("handover_to_gpu", config.get("workload") == "relax"))


def unit_done(task: dict[str, Any]) -> bool:
    status = read_json(Path(task["status"]), {}) or {}
    return status.get("scientifically_converged") is True


def refresh_check_summary(summary_path: Path) -> None:
    """Re-run ``compare-backend-check`` when a check task finished after the last summary."""

    manifest_path = summary_path.with_name("scf_manifest.json")
    manifest = read_json(manifest_path)
    if not manifest:
        return
    summary_mtime = summary_path.stat().st_mtime if summary_path.exists() else 0.0
    newest = max(
        (Path(task["status"]).stat().st_mtime for task in manifest["tasks"] if Path(task["status"]).exists()),
        default=0.0,
    )
    if newest <= summary_mtime:
        return
    from relaxgen.prerelax.umlip_dft_handoff_threshold import compare_backend_check

    with contextlib.redirect_stdout(io.StringIO()):
        compare_backend_check(argparse.Namespace(manifest=manifest_path))


def check_result(backend: dict[str, Any], tolerances: dict[str, float]) -> tuple[bool, str, float | None]:
    """Whether a backend may run production work, and its measured run time if known."""

    check = backend.get("check")
    if not check:
        return True, "no check required", None
    refresh_check_summary(Path(check["summary"]))
    summary = read_json(Path(check["summary"]))
    if not summary:
        return False, "backend check has not been summarized", None
    rows = [row for row in summary["tasks"] if row["task_index"] == check["task_index"]]
    if not rows or "energy_difference_mev" not in rows[0]:
        return False, "backend check task has not converged", None
    row = rows[0]
    energy = abs(float(row["energy_difference_mev"]))
    force = float(row["max_atom_force_difference_ev_per_angstrom"])
    wall = row.get("qe_wall_seconds")
    if energy > tolerances["energy_mev"] or force > tolerances["max_force_ev_per_angstrom"]:
        return False, f"check failed: |dE|={energy:.3g} meV, max|dF|={force:.3g} eV/A", wall
    return True, f"check passed: |dE|={energy:.3g} meV, max|dF|={force:.3g} eV/A", wall


def load_events(dispatch_dir: Path) -> dict[int, list[dict[str, Any]]]:
    events: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((dispatch_dir / "events").glob("*.json")):
        event = read_json(path)
        if event:
            events[int(event["task_index"])].append(event)
    return events


RUN_OUTCOMES = {"ran", "terminated"}
# Runner failures that repeat on any backend: the SCF stalls at the same geometry
# under every electronic setting, or BFGS used up its step budget.
TERMINAL_FAILURE_KINDS = {"scf_not_converged", "max_steps"}


def zero_progress(event: dict[str, Any]) -> bool:
    """A relaxation segment that ran but added no trajectory frame."""

    return event["outcome"] in RUN_OUTCOMES and event.get("progress_frames") is not None and event["progress_frames"] <= 0


def failed_attempt(event: dict[str, Any]) -> bool:
    """Crashes, time-limit kills, and segments without progress; not clean partial segments."""

    if event["outcome"] not in RUN_OUTCOMES or event.get("converged"):
        return False
    if zero_progress(event):
        return True
    if event.get("progress_frames"):
        # A relaxation that advanced and then stopped is resumable work, unless it crashed.
        return event["outcome"] == "ran" and event.get("return_code") not in (0, None)
    return event["outcome"] == "terminated" or event.get("return_code") != 0


def step_estimate(backend: dict[str, Any], events: dict[int, list[dict[str, Any]]], check_wall: float | None) -> tuple[float | None, str]:
    """Slow-case seconds per relaxation step on this backend (90th percentile when measured)."""

    steps = [
        float(seconds)
        for unit_events in events.values()
        for event in unit_events
        if event["backend"] == backend["name"]
        for seconds in event.get("step_seconds") or []
    ]
    if len(steps) >= 3:
        return float(sorted(steps)[int(0.9 * (len(steps) - 1))]), f"p90 of {len(steps)} measured steps"
    if check_wall:
        return float(check_wall), "backend-check SCF"
    prior = backend.get("step_prior_seconds")
    return (float(prior), "prior") if prior else (None, "none")


def relax_backend_gate(
    backend: dict[str, Any],
    config: dict[str, Any],
    events: dict[int, list[dict[str, Any]]],
    check_wall: float | None,
    now: datetime,
    estimate: dict[str, Any],
) -> tuple[bool, str]:
    step, source = step_estimate(backend, events, check_wall)
    estimate["step_seconds"], estimate["step_source"] = step, source
    if step is None:
        return False, "no step-time estimate for relaxations"
    needed = config["min_segment_steps"] * step
    if backend["segment_seconds"] < needed:
        return False, (
            f"segment {backend['segment_seconds'] / 3600:.1f} h < {config['min_segment_steps']} x "
            f"{step / 3600:.2f} h steps ({source})"
        )
    window_start = now.timestamp() - config["zero_progress_window_seconds"]
    stalls = [
        event for unit_events in events.values() for event in unit_events
        if event["backend"] == backend["name"] and zero_progress(event)
        and datetime.fromisoformat(event["finished_utc"]).timestamp() >= window_start
    ]
    if len(stalls) >= config["zero_progress_backoff_count"]:
        last = max(datetime.fromisoformat(e["finished_utc"]).timestamp() for e in stalls)
        until = datetime.fromtimestamp(last + config["zero_progress_window_seconds"])
        return False, f"backed off: {len(stalls)} zero-progress segments; retry after {until:%Y-%m-%d %H:%M}"
    return True, f"ok: {step / 60:.0f} min steps ({source})"


def estimate_backend(
    backend: dict[str, Any],
    slurm: SlurmCLI,
    events: dict[int, list[dict[str, Any]]],
    submissions: list[dict[str, Any]],
    now: datetime,
    check_wall: float | None,
) -> dict[str, Any]:
    runtimes = [
        float(event["wall_seconds"])
        for unit_events in events.values()
        for event in unit_events
        if event["backend"] == backend["name"] and event["outcome"] == "ran"
        and (event.get("converged") or (event.get("progress_frames") or 0) > 0)
    ]
    runtime = median(runtimes) if runtimes else (check_wall or backend.get("runtime_prior_seconds"))
    waits = slurm.recent_waits(backend["wait_partitions"])
    wait = median(waits) if waits else backend.get("wait_prior_seconds")
    pending_ages = [
        (now - parse_slurm_time(sub["submitted_at"])).total_seconds()
        for sub in submissions
        if sub["backend"] == backend["name"] and sub.get("state") == "PENDING"
    ]
    # Our own oldest pending job bounds the wait from below.
    if pending_ages and wait is not None:
        wait = max(wait, max(pending_ages))
    return {
        "runtime_seconds": runtime,
        "runtime_samples": len(runtimes),
        "wait_seconds": wait,
        "wait_samples": len(waits),
    }


# --------------------------------------------------------------------------
# Tick
# --------------------------------------------------------------------------


class TickLock:
    def __init__(self, path: Path, stale_seconds: float = 900) -> None:
        self.path, self.stale_seconds, self.held = path, stale_seconds, False

    def __enter__(self) -> "TickLock":
        try:
            if time.time() - self.path.stat().st_mtime > self.stale_seconds:
                self.path.unlink()
        except FileNotFoundError:
            pass
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"{socket.gethostname()}:{os.getpid()}\n".encode())
            os.close(descriptor)
            self.held = True
        except FileExistsError:
            self.held = False
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)


def tick(dispatch_dir: Path, slurm: SlurmCLI, *, dry_run: bool = False, now: datetime | None = None) -> dict[str, Any]:
    dispatch_dir = dispatch_dir.expanduser().resolve()
    with TickLock(dispatch_dir / "tick.lock") as lock:
        if not lock.held:
            return {"skipped": "another tick holds the lock"}
        return _tick(dispatch_dir, slurm, dry_run=dry_run, now=now or datetime.now())


def _tick(dispatch_dir: Path, slurm: SlurmCLI, *, dry_run: bool, now: datetime) -> dict[str, Any]:
    config = read_json(dispatch_dir / "config.json")
    manifest = read_json(Path(config["manifest"]))
    tasks = {int(task["task_index"]): task for task in manifest["tasks"]}
    state = read_json(dispatch_dir / "state.json", {"units": {}}) or {"units": {}}
    units: dict[str, dict[str, Any]] = state["units"]
    decisions: list[dict[str, Any]] = []

    # 1. Refresh submission states from Slurm.
    active = slurm.active_jobs(config["job_name"])
    open_keys = [
        sub["job_key"]
        for unit in units.values()
        for sub in unit["submissions"]
        if sub.get("state") in ACTIVE_STATES and sub["job_key"] not in active
    ]
    finished = slurm.finished_jobs(open_keys)
    for unit in units.values():
        for sub in unit["submissions"]:
            info = active.get(sub["job_key"]) or finished.get(sub["job_key"])
            if info is None:
                continue
            sub["state"] = info.state
            if info.start and info.state != "PENDING":
                sub["started_at"] = info.start.isoformat()

    events = load_events(dispatch_dir)
    claims_dir = dispatch_dir / "claims"

    # 2. Classify units; release claims whose worker stopped refreshing them.
    pool: list[int] = []
    # Units running on a CPU backend; each keeps a GPU job queued to take it over.
    on_cpu: list[int] = []
    counts: dict[str, int] = defaultdict(int)
    withdrawn: set[int] = set()
    for index in sorted(tasks):
        unit = units.setdefault(str(index), {"submissions": [], "failed": False})
        claim = claims_dir / f"{index}.json"
        if tasks[index].get("withdrawn"):
            withdrawn.add(index)
            counts["withdrawn"] += 1
        elif unit_done(tasks[index]):
            unit["done"] = True
            counts["done"] += 1
        elif claim.exists():
            holder = read_json(claim, {}) or {}
            holder_state = next(
                (sub.get("state") for sub in unit["submissions"] if sub["job_key"] == holder.get("job_key")),
                None,
            )
            stale = time.time() - claim.stat().st_mtime > config["claim_stale_seconds"]
            if stale and holder_state != "RUNNING":
                decisions.append({"action": "release_stale_claim", "task": index, "holder": holder})
                if not dry_run:
                    claim.unlink(missing_ok=True)
                pool.append(index)
            else:
                counts["running"] += 1
                if handover_enabled(config) and backend_kind(config, holder.get("backend")) == "cpu":
                    on_cpu.append(index)
        else:
            attempts = sum(1 for event in events.get(index, []) if failed_attempt(event))
            terminal = any(event.get("failure_kind") in TERMINAL_FAILURE_KINDS for event in events.get(index, []))
            if terminal or attempts >= config["max_attempts"]:
                unit["failed"] = True
                counts["failed"] += 1
            else:
                pool.append(index)
        unit["attempts"] = sum(1 for event in events.get(index, []) if event["outcome"] in RUN_OUTCOMES)

    # 3. Cancel queued twins of units that are done, failed, or claimed, except
    #    the GPU jobs waiting to take over units that run on CPU.  Cancel every
    #    job of a withdrawn unit, running or not.
    pool_set, on_cpu_set = set(pool), set(on_cpu)
    to_cancel = [
        sub["job_key"]
        for index, unit in ((int(i), u) for i, u in units.items())
        if index not in pool_set
        for sub in unit["submissions"]
        if (sub.get("state") in ACTIVE_STATES if index in withdrawn else sub.get("state") == "PENDING")
        and not (index in on_cpu_set and backend_kind(config, sub["backend"]) == "gpu")
    ]
    if to_cancel:
        decisions.append({"action": "cancel_twins", "jobs": to_cancel})
        if not dry_run:
            slurm.cancel(to_cancel)
            for unit in units.values():
                for sub in unit["submissions"]:
                    if sub["job_key"] in to_cancel:
                        sub["state"] = "CANCELLED"

    # 4. Backend availability and estimates.
    all_subs = [sub for unit in units.values() for sub in unit["submissions"]]
    backends = []
    for backend in config["backends"]:
        enabled, reason, check_wall = check_result(backend, config["check_tolerances"])
        estimate = estimate_backend(backend, slurm, events, all_subs, now, check_wall)
        if enabled and (estimate["runtime_seconds"] is None or estimate["wait_seconds"] is None):
            enabled, reason = False, "no runtime or wait estimate"
        if enabled and config.get("workload") == "relax":
            enabled, reason = relax_backend_gate(backend, config, events, check_wall, now, estimate)
        active_subs = [
            sub for sub in all_subs if sub["backend"] == backend["name"] and sub.get("state") in ACTIVE_STATES
        ]
        backends.append({**backend, **estimate, "enabled": enabled, "reason": reason, "active": len(active_subs)})

    # 5. Plan submissions.  GPU backends serve the head of the pool, then the
    #    units running on CPU; CPU backends take tail units whose expected GPU
    #    completion is later.
    def has_active(index: int, name: str) -> bool:
        return any(
            sub["backend"] == name and sub.get("state") in ACTIVE_STATES
            for sub in units[str(index)]["submissions"]
        )

    def stalled_on(index: int, name: str) -> bool:
        return any(
            event["backend"] == name and zero_progress(event) for event in events.get(index, [])
        )

    plan: list[tuple[int, dict[str, Any]]] = []
    # A GPU backend capped at zero jobs is switched off, not a queue CPU work waits behind.
    gpus = [b for b in backends if b["kind"] == "gpu" and b["enabled"] and b["max_jobs"] > 0]
    for backend in gpus:
        free = backend["max_jobs"] - backend["active"]
        for index in pool + on_cpu:
            if free <= 0:
                break
            if not has_active(index, backend["name"]) and not stalled_on(index, backend["name"]):
                plan.append((index, backend))
                free -= 1

    gpu_slots = sum(b["max_jobs"] for b in gpus)
    gpu_runtime = min((b["runtime_seconds"] for b in gpus), default=math.inf)
    gpu_wait = min((b["wait_seconds"] for b in gpus), default=math.inf)
    cpu_cores_used = sum(b["cores"] * b["active"] for b in backends if b["kind"] == "cpu")
    hand_over = handover_enabled(config) and config.get("workload") == "relax"
    cpus = sorted(
        (b for b in backends if b["kind"] == "cpu" and b["enabled"]),
        key=lambda b: b["wait_seconds"] + b["runtime_seconds"],
    )
    for position in range(len(pool) - 1, -1, -1):
        index = pool[position]
        if any(has_active(index, b["name"]) for b in cpus):
            continue
        rounds = position // max(gpu_slots, 1)
        gpu_start = gpu_wait + gpu_runtime * rounds if gpus else math.inf
        gpu_finish = gpu_start + gpu_runtime
        for backend in cpus:
            if stalled_on(index, backend["name"]):
                continue
            if hand_over:
                # A GPU job takes the relaxation over as soon as it starts, so CPU time
                # is never wasted: use CPU when it starts and runs a few steps first.
                worthwhile = (backend["wait_seconds"] + config["min_segment_steps"] * backend["step_seconds"]
                              ) * config["cpu_preference_margin"] < gpu_start
            else:
                cpu_finish = backend["wait_seconds"] + backend["runtime_seconds"]
                worthwhile = cpu_finish * config["cpu_preference_margin"] < gpu_finish
            queued = sum(1 for i, b in plan if b["name"] == backend["name"])
            if (
                worthwhile
                and backend["active"] + queued < backend["max_jobs"]
                and cpu_cores_used + backend["cores"] <= config["cpu_core_budget"]
            ):
                plan.append((index, backend))
                cpu_cores_used += backend["cores"]
                break

    # 6. Submit.
    for index, backend in plan:
        command = [
            "sbatch", "--parsable", f"--array={index}", f"--job-name={config['job_name']}",
            f"--output={dispatch_dir / 'slurm'}/%x-%A_%a.out",
            f"--error={dispatch_dir / 'slurm'}/%x-%A_%a.err",
            "--export=" + ",".join(
                ["ALL", f"DISPATCH_DIR={dispatch_dir}", f"DISPATCH_BACKEND={backend['name']}", "OVERWRITE=1"]
                + [f"{key}={value}" for key, value in backend.get("env", {}).items()]
            ),
            *backend["sbatch_args"],
            str(PROJECT_ROOT / backend["launcher"]),
            config["manifest"],
        ]
        decision = {"action": "submit", "task": index, "backend": backend["name"]}
        if not dry_run:
            job_id = slurm.submit(command)
            units[str(index)]["submissions"].append(
                {"backend": backend["name"], "job_key": f"{job_id}_{index}",
                 "submitted_at": now.isoformat(), "state": "PENDING"}
            )
            decision["job_key"] = f"{job_id}_{index}"
        decisions.append(decision)

    summary = {
        "tick_at": now.isoformat(),
        "dry_run": dry_run,
        "units": len(tasks),
        "done": counts["done"],
        "running": counts["running"],
        "running_on_cpu": len(on_cpu),
        "failed": counts["failed"],
        "withdrawn": counts["withdrawn"],
        "pool": len(pool),
        "backends": [
            {key: b.get(key) for key in ("name", "enabled", "reason", "active", "runtime_seconds",
                                         "runtime_samples", "wait_seconds", "wait_samples",
                                         "step_seconds", "step_source")}
            for b in backends
        ],
        "decisions": decisions,
    }
    if not dry_run:
        state["updated"] = now.isoformat()
        write_json_atomic(dispatch_dir / "state.json", state)
        with (dispatch_dir / "log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True, default=str) + "\n")
    return summary


# --------------------------------------------------------------------------
# Worker: claim, run the launcher's command, refresh the claim, report.
# --------------------------------------------------------------------------


def worker(args: argparse.Namespace, slurm: SlurmCLI | None = None) -> int:
    dispatch_dir = args.dispatch_dir.expanduser().resolve()
    config = read_json(dispatch_dir / "config.json")
    manifest = read_json(Path(config["manifest"]))
    task = next(t for t in manifest["tasks"] if int(t["task_index"]) == args.task_index)
    job_key = f"{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'local'))}_{args.task_index}"
    event_path = dispatch_dir / "events" / f"{args.task_index}__{job_key}.json"
    event = {"task_index": args.task_index, "backend": args.backend, "job_key": job_key,
             "host": socket.gethostname(), "started_utc": utc_now()}

    if task.get("withdrawn"):
        write_json_atomic(event_path, {**event, "outcome": "withdrawn"})
        return 0
    if unit_done(task):
        write_json_atomic(event_path, {**event, "outcome": "already_done"})
        return 0
    claim = dispatch_dir / "claims" / f"{args.task_index}.json"
    preempt = dispatch_dir / "preempt" / f"{args.task_index}.json"
    if not acquire_claim(claim, event):
        holder = read_json(claim, {}) or {}
        takes_over = (
            handover_enabled(config)
            and backend_kind(config, args.backend) == "gpu"
            and backend_kind(config, holder.get("backend")) == "cpu"
        )
        if not (takes_over and take_over_from_cpu(claim, preempt, event, config)):
            write_json_atomic(event_path, {**event, "outcome": "lost_claim", "holder": holder})
            print(f"[DISPATCH] Task {args.task_index} is claimed by another job; exiting.")
            return 0
        event["took_over_from"] = holder.get("job_key")
        print(f"[DISPATCH] Took task {args.task_index} over from CPU job {holder.get('job_key')}.")

    start = time.time()
    stop = threading.Event()
    preempted = threading.Event()
    child: subprocess.Popen | None = None

    def refresh_claim() -> None:
        # Refresh the claim every minute; look for a GPU job's hand-over request
        # every few seconds and stop the whole QE process group when one arrives.
        # A request older than this job was left by a GPU job that died waiting.
        refreshed = time.time()
        while not stop.wait(PREEMPT_POLL_SECONDS):
            if time.time() - refreshed >= CLAIM_REFRESH_SECONDS:
                refreshed = time.time()
                try:
                    os.utime(claim)
                except FileNotFoundError:
                    return
            try:
                requested = preempt.stat().st_mtime >= start
            except FileNotFoundError:
                requested = False
            if requested and child is not None and not preempted.is_set():
                preempted.set()
                print(f"[DISPATCH] Hand-over requested by {read_json(preempt, {}).get('by')}; stopping QE.")
                stop_process_group(child)

    # Slurm sends SIGTERM at the time limit; record the segment before dying.
    class Terminated(Exception):
        pass

    def on_sigterm(signum: int, frame: Any) -> None:
        raise Terminated()

    signal.signal(signal.SIGTERM, on_sigterm)
    status_path = Path(task["status"])
    before = read_json(status_path, {}) or {}
    return_code: int | None = None
    outcome = "ran"
    try:
        # Its own session, so a hand-over can stop mpirun and every rank at once.
        child = subprocess.Popen(args.child, start_new_session=True)
        threading.Thread(target=refresh_claim, daemon=True).start()
        return_code = child.wait()
    except Terminated:
        outcome = "terminated"
    finally:
        stop.set()
        if preempted.is_set():
            outcome = "preempted"
        converged = unit_done(task)
        after = read_json(status_path, {}) or {}
        progress = None
        if "trajectory_frames" in after:
            progress = int(after["trajectory_frames"]) - int(before.get("trajectory_frames", 0))
        new_segments = (after.get("segments") or [])[len(before.get("segments") or []):]
        step_seconds = [s for segment in new_segments for s in segment.get("step_wall_seconds") or []]
        write_json_atomic(
            event_path,
            {**event, "outcome": outcome, "return_code": return_code, "converged": converged,
             "progress_frames": progress, "step_seconds": step_seconds or None,
             "failure_kind": after.get("failure_kind") if after.get("execution_status") == "failed" else None,
             "wall_seconds": time.time() - start, "finished_utc": utc_now()},
        )
        claim.unlink(missing_ok=True)
    if outcome == "terminated":
        return 143
    if outcome == "preempted":
        # The GPU job now holds the unit; this CPU job is done.
        return 0
    try:
        tick(dispatch_dir, slurm or SlurmCLI())
    except Exception as error:  # A failed piggyback tick must not fail the task.
        print(f"[DISPATCH] Piggyback tick failed: {type(error).__name__}: {error}")
    return return_code


def acquire_claim(claim: Path, event: dict[str, Any]) -> bool:
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(descriptor, json.dumps(event).encode())
    os.close(descriptor)
    return True


def take_over_from_cpu(
    claim: Path, preempt: Path, event: dict[str, Any], config: dict[str, Any], poll_seconds: float = 5.0
) -> bool:
    """Ask the CPU job holding ``claim`` to stop, then claim the unit for this GPU job.

    A holder that stopped refreshing its claim is dead; its claim is removed.
    """

    write_json_atomic(preempt, {"by": event["job_key"], "backend": event["backend"], "requested_utc": utc_now()})
    stale = float(config["claim_stale_seconds"])
    deadline = time.time() + stale + 120
    try:
        while time.time() < deadline:
            with contextlib.suppress(FileNotFoundError):
                if time.time() - claim.stat().st_mtime > stale:
                    claim.unlink(missing_ok=True)
            if acquire_claim(claim, event):
                return True
            time.sleep(poll_seconds)
        return False
    finally:
        preempt.unlink(missing_ok=True)


def stop_process_group(process: subprocess.Popen, grace_seconds: float = 30.0) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


# --------------------------------------------------------------------------
# Status and CLI
# --------------------------------------------------------------------------


def status(dispatch_dir: Path) -> None:
    dispatch_dir = dispatch_dir.expanduser().resolve()
    lines = (dispatch_dir / "log.jsonl").read_text(encoding="utf-8").splitlines() if (dispatch_dir / "log.jsonl").exists() else []
    if not lines:
        print("No ticks recorded yet.")
        return
    last = json.loads(lines[-1])
    print(f"Last tick {last['tick_at']}: {last['done']}/{last['units']} done, "
          f"{last['running']} running, {last['pool']} waiting, {last['failed']} failed")
    for backend in last["backends"]:
        def hours(value: float | None) -> str:
            return "n/a" if value is None else f"{value / 3600:.2f} h"
        print(f"  {backend['name']:20s} {'on ' if backend['enabled'] else 'off'} "
              f"active={backend['active']:3d} wait={hours(backend['wait_seconds'])} "
              f"run={hours(backend['runtime_seconds'])}  ({backend['reason']})")
    submitted = [d for d in last["decisions"] if d["action"] == "submit"]
    print(f"  last tick submitted {len(submitted)} job(s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Write config.json for a manifest.")
    init_parser.add_argument("--dispatch-dir", type=Path, required=True)
    init_parser.add_argument("--manifest", type=Path, required=True)
    init_parser.add_argument("--job-name", required=True, help="Slurm job name shared by this campaign's jobs.")
    init_parser.add_argument("--cpu-core-budget", type=int, default=1000)
    init_parser.add_argument("--workload", choices=("scf", "relax"), default="scf")
    init_parser.add_argument("--overwrite", action="store_true")

    tick_parser = sub.add_parser("tick", help="Refresh state and submit or cancel jobs.")
    tick_parser.add_argument("--dispatch-dir", type=Path, required=True)
    tick_parser.add_argument("--dry-run", action="store_true")

    status_parser = sub.add_parser("status", help="Print the last tick's summary.")
    status_parser.add_argument("--dispatch-dir", type=Path, required=True)

    worker_parser = sub.add_parser("worker", help="Claim a unit and run COMMAND (used by launchers).")
    worker_parser.add_argument("--dispatch-dir", type=Path, required=True)
    worker_parser.add_argument("--task-index", type=int, required=True)
    worker_parser.add_argument("--backend", required=True)
    worker_parser.add_argument("child", nargs=argparse.REMAINDER, help="-- launcher command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        config = init(args)
        print(f"backends={[b['name'] for b in config['backends']]}")
    elif args.command == "tick":
        summary = tick(args.dispatch_dir, SlurmCLI(), dry_run=args.dry_run)
        print(json.dumps(summary, indent=2, default=str))
    elif args.command == "status":
        status(args.dispatch_dir)
    else:
        if args.child and args.child[0] == "--":
            args.child = args.child[1:]
        if not args.child:
            raise SystemExit("worker needs a command after --")
        return worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
