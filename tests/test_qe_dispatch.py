from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from relaxgen.dispatch import qe_dispatch
from relaxgen.dispatch.qe_dispatch import (
    JobInfo, failed_attempt, take_over_from_cpu, tick, worker, write_json_atomic, zero_progress,
)

NOW = datetime(2026, 10, 22, 12, 0, 0)


class FakeSlurm:
    def __init__(self, waits: dict[str, list[float]] | None = None) -> None:
        self.jobs: dict[str, JobInfo] = {}
        self.waits = waits or {}
        self.submitted: list[list[str]] = []
        self.cancelled: list[str] = []
        self.next_id = 1000

    def active_jobs(self, name: str) -> dict[str, JobInfo]:
        return {k: j for k, j in self.jobs.items() if j.state in {"PENDING", "RUNNING"}}

    def finished_jobs(self, keys):
        return {k: self.jobs[k] for k in keys if k in self.jobs}

    def recent_waits(self, partitions):
        return [w for p in partitions for w in self.waits.get(p, [])]

    def submit(self, command):
        self.next_id += 1
        self.submitted.append(list(command))
        index = next(arg for arg in command if arg.startswith("--array=")).split("=")[1]
        self.jobs[f"{self.next_id}_{index}"] = JobInfo(f"{self.next_id}_{index}", "PENDING", NOW, None)
        return str(self.next_id)

    def cancel(self, keys):
        self.cancelled.extend(keys)
        for key in keys:
            self.jobs[key].state = "CANCELLED"


def backend(name, kind, *, max_jobs, cores=0, runtime=3000.0, wait=600.0, check=None, partitions=None):
    return {
        "name": name, "kind": kind, "launcher": "slurm/run_qe_scf_cpu_slurm_uva.sh",
        "sbatch_args": [], "env": {}, "max_jobs": max_jobs, "cores": cores,
        "wait_partitions": partitions or [name], "runtime_prior_seconds": runtime,
        "wait_prior_seconds": wait, "check": check,
    }


class SlurmCLITest(unittest.TestCase):
    def test_array_element_cancelled_before_it_started_is_seen_as_finished(self):
        output = ("21910895_[387]|CANCELLED by 15986613|2026-10-24T10:00:00|None|standard-afton\n"
                  "21905996_389|COMPLETED|2026-10-24T09:00:00|2026-10-24T09:30:00|standard-afton\n")
        with mock.patch.object(qe_dispatch.SlurmCLI, "_run", return_value=output) as run:
            jobs = qe_dispatch.SlurmCLI("me").finished_jobs(["21910895_387", "21905996_389"])
        # "sacct -j 21910895_387" lists nothing for an element that never started.
        self.assertIn("21905996,21910895", run.call_args[0][0])
        self.assertEqual(jobs["21910895_387"].state, "CANCELLED")
        self.assertEqual(jobs["21905996_389"].state, "COMPLETED")


class DispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dispatch = root / "dispatch"
        tasks = []
        for index in range(10):
            task_dir = root / "tasks" / f"{index:04d}"
            task_dir.mkdir(parents=True)
            tasks.append({"task_index": index, "status": str(task_dir / "status.json")})
        self.tasks = tasks
        self.manifest = root / "scf_manifest.json"
        write_json_atomic(self.manifest, {"tasks": tasks})
        for sub in ("claims", "events", "slurm"):
            (self.dispatch / sub).mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def configure(self, backends, budget=1000):
        write_json_atomic(self.dispatch / "config.json", {
            "manifest": str(self.manifest), "job_name": "qd_test", "max_attempts": 2,
            "claim_stale_seconds": 900, "cpu_core_budget": budget, "cpu_preference_margin": 1.0,
            "check_tolerances": {"energy_mev": 0.1, "max_force_ev_per_angstrom": 1e-3},
            "backends": backends,
        })

    def submitted_backends(self, summary):
        return [(d["task"], d["backend"]) for d in summary["decisions"] if d["action"] == "submit"]

    def test_gpu_fills_head_and_unchecked_cpu_stays_off(self):
        check = {"summary": str(Path(self.tmp.name) / "missing" / "backend_check_summary.json"), "task_index": 0}
        self.configure([backend("gpu", "gpu", max_jobs=3),
                        backend("cpu", "cpu", max_jobs=4, cores=168, runtime=100, wait=0, check=check)])
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        self.assertEqual(self.submitted_backends(summary), [(0, "gpu"), (1, "gpu"), (2, "gpu")])
        cpu = next(b for b in summary["backends"] if b["name"] == "cpu")
        self.assertFalse(cpu["enabled"])

    def test_cpu_takes_tail_when_gpu_queue_is_slow(self):
        # GPU: 2 slots, 1 h runs, 10 h wait.  CPU: 2 h total, two 168-core jobs fit the budget.
        self.configure(
            [backend("gpu", "gpu", max_jobs=2, runtime=3600, wait=36000),
             backend("cpu", "cpu", max_jobs=5, cores=168, runtime=3600, wait=3600)],
            budget=400,
        )
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        cpu_tasks = sorted(t for t, b in self.submitted_backends(summary) if b == "cpu")
        self.assertEqual(cpu_tasks, [8, 9])
        gpu_tasks = sorted(t for t, b in self.submitted_backends(summary) if b == "gpu")
        self.assertEqual(gpu_tasks, [0, 1])

    def test_gpu_capped_at_zero_jobs_leaves_every_unit_to_cpu(self):
        self.configure(
            [backend("gpu", "gpu", max_jobs=0, runtime=3000, wait=600),
             backend("cpu", "cpu", max_jobs=10, cores=84, runtime=3600, wait=3600)],
        )
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        self.assertEqual(sorted(self.submitted_backends(summary)), [(i, "cpu") for i in range(10)])

    def test_real_waits_override_priors(self):
        self.configure([backend("gpu", "gpu", max_jobs=1, wait=99999, partitions=["gpu-a100"])])
        summary = tick(self.dispatch, FakeSlurm(waits={"gpu-a100": [60, 300, 600]}), now=NOW)
        self.assertEqual(summary["backends"][0]["wait_seconds"], 300)

    def test_twins_of_claimed_and_done_units_are_cancelled(self):
        self.configure([backend("gpu", "gpu", max_jobs=10)])
        slurm = FakeSlurm()
        tick(self.dispatch, slurm, now=NOW)
        (self.dispatch / "claims" / "0.json").write_text(json.dumps({"job_key": "x"}))
        write_json_atomic(Path(self.tasks[1]["status"]), {"scientifically_converged": True})
        summary = tick(self.dispatch, slurm, now=NOW + timedelta(minutes=5))
        cancelled = next(d for d in summary["decisions"] if d["action"] == "cancel_twins")["jobs"]
        self.assertEqual(sorted(k.split("_")[1] for k in cancelled), ["0", "1"])
        self.assertEqual(summary["done"], 1)
        self.assertEqual(summary["running"], 1)

    def test_withdrawn_units_lose_every_job_and_are_not_resubmitted(self):
        self.configure([backend("gpu", "gpu", max_jobs=10)])
        slurm = FakeSlurm()
        tick(self.dispatch, slurm, now=NOW)
        running = next(k for k in slurm.jobs if k.endswith("_3"))
        slurm.jobs[running].state = "RUNNING"
        (self.dispatch / "claims" / "3.json").write_text(json.dumps({"job_key": running, "backend": "gpu"}))
        self.tasks[3]["withdrawn"] = self.tasks[4]["withdrawn"] = True
        write_json_atomic(self.manifest, {"tasks": self.tasks})
        summary = tick(self.dispatch, slurm, now=NOW + timedelta(minutes=10))
        cancelled = next(d for d in summary["decisions"] if d["action"] == "cancel_twins")["jobs"]
        self.assertEqual(sorted(k.split("_")[1] for k in cancelled), ["3", "4"])
        self.assertEqual(summary["withdrawn"], 2)
        self.assertEqual(summary["running"], 0)
        summary = tick(self.dispatch, slurm, now=NOW + timedelta(minutes=20))
        self.assertNotIn(3, [t for t, _ in self.submitted_backends(summary)])
        self.assertNotIn(4, [t for t, _ in self.submitted_backends(summary)])

        marker = Path(self.tmp.name) / "ran"
        args = argparse.Namespace(dispatch_dir=self.dispatch, task_index=4, backend="gpu",
                                  child=["touch", str(marker)])
        self.assertEqual(worker(args, slurm=slurm), 0)
        self.assertFalse(marker.exists())

    def test_stale_claim_without_running_job_is_released(self):
        self.configure([backend("gpu", "gpu", max_jobs=0)])
        claim = self.dispatch / "claims" / "3.json"
        claim.write_text(json.dumps({"job_key": "gone_3"}))
        old = time.time() - 3600
        os.utime(claim, (old, old))
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        self.assertFalse(claim.exists())
        self.assertIn("release_stale_claim", [d["action"] for d in summary["decisions"]])

    def test_units_fail_after_max_attempts(self):
        self.configure([backend("gpu", "gpu", max_jobs=10)])
        for attempt in range(2):
            write_json_atomic(self.dispatch / "events" / f"4__j{attempt}_4.json",
                              {"task_index": 4, "backend": "gpu", "outcome": "ran", "converged": False,
                               "return_code": 1})
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        self.assertEqual(summary["failed"], 1)
        self.assertNotIn(4, [t for t, _ in self.submitted_backends(summary)])

    def test_unconverged_segment_that_exits_cleanly_is_not_a_failure(self):
        self.configure([backend("gpu", "gpu", max_jobs=10)])
        for attempt in range(3):
            write_json_atomic(self.dispatch / "events" / f"6__j{attempt}_6.json",
                              {"task_index": 6, "backend": "gpu", "outcome": "ran", "converged": False,
                               "return_code": 0})
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        self.assertEqual(summary["failed"], 0)
        self.assertIn(6, [t for t, _ in self.submitted_backends(summary)])

    def configure_relax(self, backends, **policy):
        self.configure(backends)
        config = json.loads((self.dispatch / "config.json").read_text())
        config.update({"workload": "relax", "min_segment_steps": 3, "zero_progress_backoff_count": 2,
                       "zero_progress_window_seconds": 48 * 3600, **policy})
        write_json_atomic(self.dispatch / "config.json", config)

    def relax_backend(self, name, kind="gpu", *, segment=12 * 3600, step_prior=1800.0, **kw):
        return {**backend(name, kind, **kw), "segment_seconds": segment, "step_prior_seconds": step_prior}

    def event(self, task, job, backend_name, *, progress, steps=(), outcome="ran", code=0, finished=None):
        write_json_atomic(self.dispatch / "events" / f"{task}__{job}_{task}.json", {
            "task_index": task, "backend": backend_name, "outcome": outcome, "return_code": code,
            "converged": False, "progress_frames": progress, "step_seconds": list(steps),
            "finished_utc": (finished or NOW).astimezone().isoformat(), "wall_seconds": 100.0,
        })

    def test_backend_whose_segment_cannot_fit_three_steps_is_not_used(self):
        # 2 h CPU segments against measured 1 h steps: 3 steps need 3 h.
        self.configure_relax([
            self.relax_backend("gpu-1x", max_jobs=2),
            self.relax_backend("cpu", "cpu", max_jobs=4, cores=84, segment=2 * 3600, step_prior=None,
                               runtime=3600, wait=0),
        ])
        self.event(0, "j1", "cpu", progress=2, steps=[3600, 3600, 3600])
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        cpu = next(b for b in summary["backends"] if b["name"] == "cpu")
        self.assertFalse(cpu["enabled"])
        self.assertIn("segment 2.0 h < 3 x", cpu["reason"])
        self.assertNotIn("cpu", [b for _, b in self.submitted_backends(summary)])

    def test_unit_is_not_resent_to_a_backend_where_it_made_no_progress(self):
        self.configure_relax([self.relax_backend("gpu-1x", max_jobs=10),
                              self.relax_backend("gpu-2x", max_jobs=10)])
        self.event(3, "j1", "gpu-1x", progress=0, outcome="terminated", code=None)
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        plan = self.submitted_backends(summary)
        self.assertNotIn((3, "gpu-1x"), plan)
        self.assertIn((3, "gpu-2x"), plan)

    def test_backend_backs_off_after_repeated_zero_progress(self):
        self.configure_relax([self.relax_backend("gpu-1x", max_jobs=10)])
        self.event(1, "j1", "gpu-1x", progress=0, finished=NOW - timedelta(hours=2))
        self.event(2, "j2", "gpu-1x", progress=0, finished=NOW - timedelta(hours=1))
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        gpu = summary["backends"][0]
        self.assertFalse(gpu["enabled"])
        self.assertIn("backed off: 2 zero-progress", gpu["reason"])
        # Outside the 48 h window the backend is usable again.
        later = tick(self.dispatch, FakeSlurm(), now=NOW + timedelta(hours=60))
        self.assertTrue(later["backends"][0]["enabled"])

    def test_partial_segments_are_progress_and_stalls_are_failures(self):
        self.configure_relax([self.relax_backend("gpu-1x", max_jobs=10)], max_attempts=2,
                             zero_progress_backoff_count=99)
        for job in range(4):
            self.event(7, f"p{job}", "gpu-1x", progress=5, outcome="terminated", code=None)
        self.event(8, "s1", "gpu-1x", progress=0)
        self.event(8, "s2", "gpu-2x", progress=0)
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        self.assertEqual(summary["failed"], 1)
        self.assertIn(7, [t for t, _ in self.submitted_backends(summary)])
        self.assertNotIn(8, [t for t, _ in self.submitted_backends(summary)])

    def test_worker_records_progress_from_status_frames(self):
        self.configure([backend("gpu", "gpu", max_jobs=0)])
        status = Path(self.tasks[2]["status"])
        write_json_atomic(status, {"trajectory_frames": 10})
        script = (f"import json; p='{status}'; s=json.load(open(p)); s['trajectory_frames']=14; "
                  "s['segments']=[{'step_wall_seconds':[60,30,30,30,30]}]; json.dump(s, open(p,'w'))")
        args = argparse.Namespace(dispatch_dir=self.dispatch, task_index=2, backend="gpu",
                                  child=[sys.executable, "-c", script])
        worker(args, slurm=FakeSlurm())
        event = json.loads(next((self.dispatch / "events").glob("2__*.json")).read_text())
        self.assertEqual(event["progress_frames"], 4)
        self.assertEqual(event["step_seconds"], [60, 30, 30, 30, 30])

    def test_terminal_runner_failure_is_not_retried(self):
        self.configure_relax([self.relax_backend("gpu-1x", max_jobs=10)])
        write_json_atomic(self.dispatch / "events" / "5__j1_5.json", {
            "task_index": 5, "backend": "gpu-1x", "outcome": "ran", "return_code": 1, "converged": False,
            "progress_frames": 12, "failure_kind": "scf_not_converged",
            "finished_utc": NOW.astimezone().isoformat(), "wall_seconds": 100.0,
        })
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        self.assertEqual(summary["failed"], 1)
        self.assertNotIn(5, [t for t, _ in self.submitted_backends(summary)])

    def test_worker_collects_steps_from_every_qe_session_and_the_failure_kind(self):
        self.configure([backend("gpu", "gpu", max_jobs=0)])
        status = Path(self.tasks[2]["status"])
        write_json_atomic(status, {"trajectory_frames": 10, "segments": [{"step_wall_seconds": [99]}]})
        script = (f"import json; p='{status}'; s=json.load(open(p)); s['trajectory_frames']=13; "
                  "s['segments']+=[{'step_wall_seconds':[60,30]},{'step_wall_seconds':[90]}]; "
                  "s['execution_status']='failed'; s['failure_kind']='scf_not_converged'; json.dump(s, open(p,'w'))")
        args = argparse.Namespace(dispatch_dir=self.dispatch, task_index=2, backend="gpu",
                                  child=[sys.executable, "-c", script])
        worker(args, slurm=FakeSlurm())
        event = json.loads(next((self.dispatch / "events").glob("2__*.json")).read_text())
        self.assertEqual(event["step_seconds"], [60, 30, 90])
        self.assertEqual(event["failure_kind"], "scf_not_converged")

    def configure_handover(self):
        self.configure_relax([
            self.relax_backend("gpu-1x", max_jobs=10),
            self.relax_backend("cpu", "cpu", max_jobs=4, cores=84, step_prior=None, runtime=3600, wait=0),
        ])

    def test_relaxations_wait_for_a_gpu_on_cpu_when_it_starts_much_sooner(self):
        # GPU: 10 h queue, all slots busy.  CPU: 6 min queue, 80 min steps, 24 h segments.
        # A whole CPU segment ends after the GPU one would, but the GPU takes over on start.
        self.configure_relax([
            self.relax_backend("gpu-1x", max_jobs=0, wait=36000),
            self.relax_backend("gpu-2x", max_jobs=1, wait=36000),
            self.relax_backend("cpu", "cpu", max_jobs=3, cores=84, segment=24 * 3600, step_prior=4800,
                               runtime=24 * 3600, wait=360),
        ], handover_to_gpu=True)
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        self.assertEqual(sorted(t for t, b in self.submitted_backends(summary) if b == "cpu"), [7, 8, 9])

    def test_relaxations_stay_queued_for_a_gpu_that_starts_soon(self):
        self.configure_relax([
            self.relax_backend("gpu-1x", max_jobs=1, wait=1800),
            self.relax_backend("cpu", "cpu", max_jobs=3, cores=84, segment=24 * 3600, step_prior=4800,
                               runtime=24 * 3600, wait=360),
        ], handover_to_gpu=True)
        summary = tick(self.dispatch, FakeSlurm(), now=NOW)
        # Task 0 gets the GPU in 0.5 h; the tail waits several 50 min GPU runs, long enough for CPU steps.
        self.assertNotIn((0, "cpu"), self.submitted_backends(summary))
        self.assertEqual(sorted(t for t, b in self.submitted_backends(summary) if b == "cpu"), [7, 8, 9])

    def test_unit_running_on_cpu_keeps_a_gpu_job_queued(self):
        self.configure_handover()
        (self.dispatch / "claims" / "9.json").write_text(json.dumps({"job_key": "c_9", "backend": "cpu"}))
        slurm = FakeSlurm()
        summary = tick(self.dispatch, slurm, now=NOW)
        self.assertEqual(summary["running_on_cpu"], 1)
        self.assertIn((9, "gpu-1x"), self.submitted_backends(summary))
        # A unit held by a GPU job loses its queued twins; the CPU-held unit keeps its GPU job.
        (self.dispatch / "claims" / "0.json").write_text(json.dumps({"job_key": "g_0", "backend": "gpu-1x"}))
        later = tick(self.dispatch, slurm, now=NOW + timedelta(minutes=10))
        cancelled = next(d for d in later["decisions"] if d["action"] == "cancel_twins")["jobs"]
        self.assertIn("0", [key.split("_")[1] for key in cancelled])
        self.assertNotIn("9", [key.split("_")[1] for key in cancelled])

    def test_gpu_job_takes_the_unit_once_the_cpu_job_lets_go(self):
        self.configure_handover()
        claim = self.dispatch / "claims" / "4.json"
        preempt = self.dispatch / "preempt" / "4.json"
        claim.write_text(json.dumps({"job_key": "c_4", "backend": "cpu"}))

        def cpu_worker():
            while not preempt.exists():
                time.sleep(0.05)
            claim.unlink()

        threading.Thread(target=cpu_worker, daemon=True).start()
        event = {"job_key": "g_4", "backend": "gpu-1x"}
        config = json.loads((self.dispatch / "config.json").read_text())
        self.assertTrue(take_over_from_cpu(claim, preempt, event, config, poll_seconds=0.05))
        self.assertEqual(json.loads(claim.read_text())["job_key"], "g_4")
        self.assertFalse(preempt.exists())

    def test_claim_of_a_dead_cpu_job_is_taken_without_waiting(self):
        self.configure_handover()
        claim = self.dispatch / "claims" / "4.json"
        claim.write_text(json.dumps({"job_key": "c_4", "backend": "cpu"}))
        old = time.time() - 3600
        os.utime(claim, (old, old))
        config = json.loads((self.dispatch / "config.json").read_text())
        start = time.time()
        self.assertTrue(take_over_from_cpu(claim, self.dispatch / "preempt" / "4.json",
                                           {"job_key": "g_4", "backend": "gpu-1x"}, config, poll_seconds=0.05))
        self.assertLess(time.time() - start, 5)

    def test_cpu_worker_stops_qe_on_a_hand_over_request(self):
        self.configure_handover()
        preempt = self.dispatch / "preempt" / "2.json"

        def request():
            time.sleep(0.5)
            write_json_atomic(preempt, {"by": "g_2"})

        threading.Thread(target=request, daemon=True).start()
        args = argparse.Namespace(dispatch_dir=self.dispatch, task_index=2, backend="cpu",
                                  child=[sys.executable, "-c", "import time; time.sleep(60)"])
        start = time.time()
        with mock.patch.object(qe_dispatch, "PREEMPT_POLL_SECONDS", 0.1):
            self.assertEqual(worker(args, slurm=FakeSlurm()), 0)
        self.assertLess(time.time() - start, 20)
        event = json.loads(next((self.dispatch / "events").glob("2__*.json")).read_text())
        self.assertEqual(event["outcome"], "preempted")
        self.assertFalse((self.dispatch / "claims" / "2.json").exists())
        # A hand-over is neither a failure nor a stall on the CPU backend.
        event["progress_frames"] = 0
        self.assertFalse(failed_attempt(event))
        self.assertFalse(zero_progress(event))

    def test_request_left_by_a_dead_gpu_job_is_ignored(self):
        self.configure_handover()
        preempt = self.dispatch / "preempt" / "2.json"
        write_json_atomic(preempt, {"by": "g_dead"})
        old = time.time() - 3600
        os.utime(preempt, (old, old))
        args = argparse.Namespace(dispatch_dir=self.dispatch, task_index=2, backend="cpu",
                                  child=[sys.executable, "-c", "import time; time.sleep(1)"])
        with mock.patch.object(qe_dispatch, "PREEMPT_POLL_SECONDS", 0.1):
            worker(args, slurm=FakeSlurm())
        event = json.loads(next((self.dispatch / "events").glob("2__*.json")).read_text())
        self.assertEqual(event["outcome"], "ran")

    def test_worker_claims_runs_and_releases(self):
        self.configure([backend("gpu", "gpu", max_jobs=0)])
        args = argparse.Namespace(dispatch_dir=self.dispatch, task_index=2, backend="gpu", child=["true"])
        self.assertEqual(worker(args, slurm=FakeSlurm()), 0)
        self.assertFalse((self.dispatch / "claims" / "2.json").exists())
        events = [json.loads(p.read_text()) for p in (self.dispatch / "events").glob("2__*.json")]
        self.assertEqual(events[0]["outcome"], "ran")

    def test_worker_that_loses_the_claim_does_not_run(self):
        self.configure([backend("gpu", "gpu", max_jobs=0)])
        (self.dispatch / "claims" / "5.json").write_text(json.dumps({"job_key": "other"}))
        marker = Path(self.tmp.name) / "ran"
        args = argparse.Namespace(dispatch_dir=self.dispatch, task_index=5, backend="cpu",
                                  child=["touch", str(marker)])
        self.assertEqual(worker(args, slurm=FakeSlurm()), 0)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
