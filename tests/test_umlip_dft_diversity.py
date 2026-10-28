from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory

from experiments import umlip_dft_diversity as diversity


def frame(shift: float, fmax: float, energy: float) -> Atoms:
    atoms = Atoms("Ar4", positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], cell=[6, 6, 6], pbc=True)
    atoms.positions[0, 0] += shift
    forces = np.zeros((4, 3))
    forces[1, 1] = fmax
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    return atoms


def write_trajectory(path: Path, frames: list[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Trajectory(path, "w") as writer:
        for atoms in frames:
            writer.write(atoms)


class DiversityPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        starts = {}
        for label in diversity.STARTS:
            path = self.out / "starts" / f"{label}.extxyz"
            path.parent.mkdir(parents=True, exist_ok=True)
            ase_write(path, frame(0.0, 1.0, 0.0))
            starts[label] = {"path": path.as_posix()}
        (self.out / "starts.json").write_text(json.dumps({"starts": starts}))
        for model in diversity.MODELS:
            runs = {}
            for label in diversity.STARTS:
                trajectory = self.out / f"umlip_{model}" / label / "trajectory.traj"
                write_trajectory(trajectory, [frame(0.0, 0.5, 0.0), frame(0.01, 0.015, -1.0), frame(0.02, 5e-4, -1.1)])
                runs[label] = {"trajectory": trajectory.as_posix(), "handoff_frame": 1}
            summary = {"handoff_fmax_ev_per_angstrom": 0.02, "runs": runs}
            (self.out / f"umlip_{model}" / "prerelax_summary.json").write_text(json.dumps(summary))
        diversity.prepare_dft(SimpleNamespace(output_dir=self.out, replicates=2, max_steps=400, overwrite=False))
        self.experiment = json.loads((self.out / "experiment.json").read_text())
        self.manifest = json.loads(Path(self.experiment["dft_manifest"]).read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def test_dft_only_arm_comes_first_and_uses_the_handoff_frame(self):
        arms = [t["arm"] for t in self.experiment["tasks"]]
        self.assertEqual(arms, ["dftonly"] * 10 + ["eqv3"] * 5 + ["uma"] * 5)
        self.assertEqual(self.manifest["protocol"]["fmax_ev_per_angstrom"], 1e-3)
        self.assertEqual(self.manifest["protocol"]["kpts"], [6, 6, 6])
        eqv3 = self.manifest["tasks"][10]
        self.assertTrue(eqv3["source_structure"].endswith("trajectory.traj@1"))
        self.assertEqual(eqv3["label"], "eqv3_unperturbed")

    def test_analysis_counts_minima_and_replicate_noise(self):
        tasks = self.manifest["tasks"]
        # Replicates of "unperturbed" end 0.3 A apart; eqv3 reaches the r1 minimum.
        endings = {0: 0.0, 5: 0.3, 10: 0.0}
        for index, shift in endings.items():
            write_trajectory(Path(tasks[index]["trajectory"]),
                             [frame(0.0, 0.2, 0.0), frame(shift, 5e-3, -1.0), frame(shift, 5e-4, -1.0 - shift)])
            Path(tasks[index]["status"]).write_text(json.dumps({
                "execution_status": "completed", "scientifically_converged": True}))
        Path(tasks[1]["status"]).write_text(json.dumps({
            "execution_status": "failed", "failure_kind": "scf_not_converged"}))
        report = diversity.analyze(SimpleNamespace(output_dir=self.out))
        self.assertEqual(report["arms"]["dftonly"]["converged"], 2)
        self.assertEqual(report["arms"]["dftonly"]["abandoned_scf_stall"], 1)
        self.assertEqual(report["arms"]["dftonly"]["endpoints_fmax_1e-3"]["count"], 1)
        self.assertAlmostEqual(report["replicate_noise"][0]["energy_difference_mev"], 300.0)
        self.assertGreater(report["replicate_noise"][0]["endpoint_rmsd_angstrom"], 0.1)
        cross = report["cross_arm"][0]
        self.assertEqual((cross["arm"], cross["start"]), ("eqv3", "unperturbed"))
        self.assertAlmostEqual(cross["endpoint_rmsd_to_dftonly_angstrom"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
