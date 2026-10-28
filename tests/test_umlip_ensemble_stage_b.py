import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

from relaxgen.prerelax.umlip_ensemble_diversity import endpoint_rmsd
from experiments.umlip_ensemble_stage_b import handoff, round_robin


class StageBSelectionTest(unittest.TestCase):
    def test_round_robin_spreads_over_families(self):
        by_family = {"a": [1, 2, 3], "b": [4], "c": [5, 6]}
        self.assertEqual(round_robin(by_family, 5), [1, 4, 5, 2, 6])

    def test_handoff_skips_frames_with_atoms_too_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.traj"
            with Trajectory(path, "w") as writer:
                for gap in (1.5, 1.8, 2.5):
                    atoms = Atoms("Bi2", positions=[[0, 0, 0], [gap, 0, 0]], cell=[10, 10, 10], pbc=True)
                    atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.full((2, 3), 0.001))
                    writer.write(atoms)
            frame = handoff({"trajectory": str(path), "handoff_frame": 0}, min_dist=2.0)
            self.assertEqual(frame["frame"], 2)


class EndpointRmsdTest(unittest.TestCase):
    def test_mirror_images_of_one_distortion_are_the_same_endpoint(self):
        from ase.build import bulk

        base = bulk("Po", "sc", a=3.0).repeat((3, 3, 3))
        plus, minus = base.copy(), base.copy()
        plus.positions[13, 0] += 0.4
        minus.positions[13, 0] -= 0.4
        self.assertLess(endpoint_rmsd(plus, minus), 0.01)
        moved = base.copy()
        moved.positions[13, 0] += 0.8
        self.assertGreater(endpoint_rmsd(plus, moved), 0.05)


if __name__ == "__main__":
    unittest.main()
