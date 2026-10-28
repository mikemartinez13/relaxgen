from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

from relaxgen.dft import dft_relaxation_campaign as campaign
from relaxgen.dft.dft_relaxation_campaign import ScfNotConverged, scf_stalled, step_from_stored_forces


def cluster() -> Atoms:
    rng = np.random.default_rng(7)
    grid = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=float)
    return Atoms("Ar8", positions=1.12 * grid + rng.normal(0.0, 0.08, (8, 3)))


def relax(atoms: Atoms, optimizer: BFGS, steps: int, trajectory: Trajectory | None, positions: list) -> None:
    """Mirror the runner: record each evaluated geometry, stop after ``steps`` yields."""

    for count, _ in enumerate(optimizer.irun(fmax=1.0e-6, steps=200), start=1):
        if not positions or not np.array_equal(positions[-1], atoms.positions):
            positions.append(atoms.positions.copy())
            if trajectory is not None:
                trajectory.write(atoms)
        if count == steps:
            break


class ResumeEquivalenceTest(unittest.TestCase):
    def run_interrupted(self, recompute: bool) -> tuple[list, list]:
        continuous: list = []
        atoms = cluster()
        atoms.calc = LennardJones()
        relax(atoms, BFGS(atoms, maxstep=0.1, logfile=None), 14, None, continuous)

        resumed: list = []
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "bfgs_state.json"
            traj_path = Path(tmp) / "trajectory.traj"
            atoms = cluster()
            atoms.calc = LennardJones()
            with Trajectory(traj_path, "w") as trajectory:
                relax(atoms, BFGS(atoms, restart=str(state), maxstep=0.1, logfile=None), 6, trajectory, resumed)

            # A new job: fresh atoms from the last stored frame and the saved BFGS state.
            with Trajectory(traj_path) as reader:
                stored = reader[-1]
            atoms = stored.copy()
            atoms.calc = None
            optimizer = BFGS(atoms, restart=str(state), maxstep=0.1, logfile=None)
            if recompute:
                atoms.calc = LennardJones()
            else:
                step_from_stored_forces(optimizer, atoms, stored, LennardJones())
            relax(atoms, optimizer, 14 - len(resumed) + (1 if recompute else 0), None, resumed)
        return continuous, resumed

    def test_resume_from_stored_forces_reproduces_the_continuous_trajectory(self):
        continuous, resumed = self.run_interrupted(recompute=False)
        self.assertEqual(len(resumed), len(continuous))
        for left, right in zip(continuous, resumed):
            np.testing.assert_allclose(left, right, atol=1.0e-12)

    def test_resume_with_recompute_also_matches(self):
        continuous, resumed = self.run_interrupted(recompute=True)
        n = min(len(continuous), len(resumed))
        self.assertGreater(n, 8)
        for left, right in zip(continuous[:n], resumed[:n]):
            np.testing.assert_allclose(left, right, atol=1.0e-12)

    def test_mismatched_geometry_is_rejected(self):
        atoms = cluster()
        stored = atoms.copy()
        stored.positions[0, 0] += 0.01
        with self.assertRaises(ValueError):
            step_from_stored_forces(BFGS(atoms, logfile=None), atoms, stored, LennardJones())


class ScfStallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        structure = root / "start.extxyz"
        ase_write(structure, cluster())
        campaign.prepare(campaign.build_parser().parse_args(
            ["prepare", "--output-dir", str(root / "campaign"), "--structures", str(structure), "--labels", "lj"]
        ))
        self.manifest = root / "campaign" / "relax_manifest.json"
        self.task = campaign.read_json(self.manifest)["tasks"][0]
        self.args = SimpleNamespace(
            manifest=self.manifest, task_index=0, backend="cpu", allocated_device="test",
            qe_executable=Path("/bin/true"), npool=1, deadline=None,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def fake_segment(self, stall):
        """Stand-in for one pw.x session: append a frame, then stall or stop for budget."""

        calls = []

        def segment(args, task, status, status_path, deadline, job_step_seconds):
            calls.append(1)
            status["segments"].append({"step_wall_seconds": [1.0]})
            atoms = cluster()
            forces = np.zeros((len(atoms), 3))
            forces[0, 0] = 0.5
            atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=forces)
            with Trajectory(task["trajectory"], "a") as writer:
                writer.write(atoms)
            if stall:
                raise ScfNotConverged("stalled")
            return True

        return calls, segment

    def test_manifest_records_the_iteration_cap(self):
        self.assertEqual(campaign.read_json(self.manifest)["protocol"]["electron_maxstep"], 60)

    def test_stall_abandons_the_relaxation_for_good(self):
        calls, segment = self.fake_segment(stall=True)
        with mock.patch.object(campaign, "_relax_segment", side_effect=segment):
            with self.assertRaises(ScfNotConverged):
                campaign.run_task(self.args)
            status = campaign.read_json(Path(self.task["status"]))
            self.assertEqual(len(calls), 1)
            self.assertEqual(status["execution_status"], "failed")
            self.assertEqual(status["failure_kind"], "scf_not_converged")
            # A later job does not pick the trajectory up again.
            campaign.run_task(self.args)
        self.assertEqual(len(calls), 1)

    def test_budget_stop_is_still_resumable(self):
        calls, segment = self.fake_segment(stall=False)
        with mock.patch.object(campaign, "_relax_segment", side_effect=segment):
            status = campaign.run_task(self.args)
        self.assertEqual(status["execution_status"], "segment_completed")
        self.assertNotIn("failure_kind", status)

    def test_stall_marker_is_found_in_binary_tainted_output(self):
        output = Path(self.tmp.name) / "espresso.pwo"
        output.write_bytes(b"iteration # 60\x00\x00\n     convergence NOT achieved after 60 iterations: stopping\n")
        self.assertTrue(scf_stalled(output))
        output.write_bytes(b"     convergence has been achieved in  18 iterations\n")
        self.assertFalse(scf_stalled(output))
        self.assertFalse(scf_stalled(output.with_name("missing.pwo")))

class TightenTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        structure = root / "start.extxyz"
        ase_write(structure, cluster())
        campaign.prepare(campaign.build_parser().parse_args(
            ["prepare", "--output-dir", str(root / "campaign"), "--structures", str(structure), str(structure),
             "--labels", "a", "b", "--fmax", "0.05"]
        ))
        self.manifest = root / "campaign" / "relax_manifest.json"
        self.forces = [0.03, 0.005]
        self.calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def segment(self, args, task, status, status_path, deadline, job_step_seconds):
        status["segments"].append({"step_wall_seconds": [1.0]})
        atoms = cluster()
        forces = np.zeros((len(atoms), 3))
        forces[0, 0] = self.forces[self.calls]
        self.calls += 1
        atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=forces)
        with Trajectory(task["trajectory"], "a") as writer:
            writer.write(atoms)
        return False

    def run_once(self, task_index):
        args = SimpleNamespace(manifest=self.manifest, task_index=task_index, backend="cpu",
                               allocated_device="test", qe_executable=Path("/bin/true"), npool=1, deadline=None)
        with mock.patch.object(campaign, "_relax_segment", side_effect=self.segment):
            return campaign.run_task(args)

    def tighten(self, *indices):
        return campaign.tighten(campaign.build_parser().parse_args(
            ["tighten", "--manifest", str(self.manifest), "--task-indices", *map(str, indices), "--fmax", "0.01"]
        ))

    def test_converged_task_resumes_to_the_tighter_target(self):
        self.assertTrue(self.run_once(0)["scientifically_converged"])
        self.assertEqual(self.tighten(0, 1), [0])
        status = self.run_once(0)
        self.assertEqual(self.calls, 2)
        self.assertTrue(status["scientifically_converged"])
        self.assertAlmostEqual(status["final_max_force_ev_per_angstrom"], 0.005)
        # Task 1 never ran, so it kept the manifest's target.
        tasks = campaign.read_json(self.manifest)["tasks"]
        self.assertEqual(tasks[0]["fmax_ev_per_angstrom"], 0.01)
        self.assertNotIn("fmax_ev_per_angstrom", tasks[1])


class BackendTest(unittest.TestCase):
    def test_h100_needs_matching_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structure = root / "start.extxyz"
            ase_write(structure, cluster())
            campaign.prepare(campaign.build_parser().parse_args(
                ["prepare", "--output-dir", str(root / "campaign"), "--structures", str(structure), "--labels", "lj"]
            ))
            manifest = root / "campaign" / "relax_manifest.json"
            self.assertIn("h100", campaign.read_json(manifest)["tasks"][0]["resource_profile"].split("_or_"))
            args = SimpleNamespace(manifest=manifest, task_index=0, backend="h100", allocated_device="test",
                                   qe_executable=Path("/opt/qe-7.5-a100/bin/pw.x"), npool=1, deadline=None)
            with self.assertRaisesRegex(ValueError, "does not visibly match h100"):
                campaign.run_task(args)


if __name__ == "__main__":
    unittest.main()
