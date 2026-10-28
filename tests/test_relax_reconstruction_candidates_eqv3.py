from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read as ase_read

from relaxgen.structures.bi2te3_defect_generation import (
    geometry_sha256,
    write_extxyz_float64,
    write_json,
)
from relaxgen.prerelax.relax_reconstruction_candidates_eqv3 import (
    relax_candidate,
    run_signature,
    summarize,
)


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, target: np.ndarray, stiffness: float = 2.0) -> None:
        super().__init__()
        self.target = np.asarray(target, dtype=np.float64)
        self.stiffness = float(stiffness)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = np.asarray(atoms.positions, dtype=np.float64) - self.target
        self.results = {
            "energy": 0.5 * self.stiffness * float(np.sum(displacement**2)),
            "forces": -self.stiffness * displacement,
        }


class EqV3CandidateRelaxationTest(unittest.TestCase):
    def test_fixed_cell_fire2_outputs_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "input" / "VTe" / "VTe_001"
            source_dir.mkdir(parents=True)
            source_path = source_dir / "unperturbed.extxyz"
            atoms = Atoms(
                symbols=["Bi", "Te"],
                positions=[[1.0, 1.0, 1.0], [4.2, 1.0, 1.0]],
                cell=np.diag([10.0, 10.0, 10.0]),
                pbc=True,
            )
            write_extxyz_float64(source_path, atoms)
            target = np.asarray([[1.2, 1.0, 1.0], [4.0, 1.0, 1.0]])
            candidate = {
                "defect_family": "VTe",
                "defect_config_id": "VTe_001",
                "candidate_id": "unperturbed",
                "candidate_type": "unperturbed",
                "defect_count": 1,
                "parameters": {},
                "candidate_structure": source_path.as_posix(),
                "geometry_sha256": geometry_sha256(atoms),
            }
            candidate_catalog = source_dir / "candidates.jsonl"
            candidate_catalog.write_text(json.dumps(candidate) + "\n")
            write_json(
                source_dir / "candidate_summary.json",
                {
                    "defect_family": "VTe",
                    "defect_config_id": "VTe_001",
                    "candidate_catalog": candidate_catalog.as_posix(),
                },
            )
            settings = {
                "optimizer": "FIRE2",
                "fixed_cell": True,
                "target_fmax_ev_per_angstrom": 0.01,
                "max_steps": 200,
                "maxstep_angstrom": 0.10,
                "warning_min_distance_angstrom": 1.8,
                "hard_min_distance_angstrom": 1.2,
                "seed": 42,
            }
            model = {
                "model": "test_harmonic",
                "checkpoint_path": "/tmp/test-checkpoint",
                "checkpoint_sha256": "test",
            }
            signature = run_signature(model, settings)
            output_dir = root / "output" / "VTe" / "VTe_001" / "unperturbed"
            result = relax_candidate(
                candidate,
                source_path,
                output_dir,
                HarmonicCalculator(target),
                model,
                settings,
                signature,
            )

            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["converged"])
            self.assertLess(
                result["final"]["max_force_ev_per_angstrom"],
                result["initial"]["max_force_ev_per_angstrom"],
            )
            self.assertTrue(result["cell_unchanged"])
            self.assertTrue(result["species_and_order_unchanged"])
            self.assertFalse(result["dft_labeled"])
            self.assertGreaterEqual(result["trajectory_frame_count"], 2)

            relaxed = ase_read(output_dir / "relaxed.extxyz", format="extxyz")
            self.assertTrue(np.array_equal(relaxed.cell.array, atoms.cell.array))
            trajectory = ase_read(
                output_dir / "trajectory.extxyz", index=":", format="extxyz"
            )
            self.assertEqual(len(trajectory), result["trajectory_frame_count"])

            (output_dir / "trajectory.extxyz").unlink()
            repaired = relax_candidate(
                candidate,
                source_path,
                output_dir,
                HarmonicCalculator(target),
                model,
                settings,
                signature,
            )
            self.assertNotIn("skipped_existing", repaired)

            restarted = relax_candidate(
                candidate,
                source_path,
                output_dir,
                HarmonicCalculator(target),
                model,
                settings,
                signature,
            )
            self.assertTrue(restarted["skipped_existing"])

            aggregate = summarize(root / "input", root / "output")
            self.assertEqual(aggregate["input_candidates"], 1)
            self.assertEqual(aggregate["completed"], 1)
            self.assertEqual(aggregate["converged"], 1)
            self.assertEqual(aggregate["failed"], 0)
            self.assertEqual(aggregate["missing"], 0)

    def test_fire_fallback_restarts_from_last_safe_geometry(self) -> None:
        class UnsafeFIRE2:
            def __init__(self, atoms, logfile, **kwargs):
                self.atoms = atoms
                self.nsteps = 0
                self.callback = None
                Path(logfile).write_text("FIRE2 safety failure\n")

            def attach(self, callback, interval=1):
                self.callback = callback

            def run(self, fmax, steps):
                self.nsteps = 1
                self.atoms.positions[1] = self.atoms.positions[0] + [0.5, 0.0, 0.0]
                self.callback()

        class SafeFIRE:
            used_downhill_check = False

            def __init__(self, atoms, logfile, downhill_check, **kwargs):
                self.atoms = atoms
                self.nsteps = 0
                self.callback = None
                SafeFIRE.used_downhill_check = downhill_check
                Path(logfile).write_text("FIRE fallback\n")

            def attach(self, callback, interval=1):
                self.callback = callback

            def run(self, fmax, steps):
                self.nsteps = 1
                self.atoms.set_positions(self.atoms.calc.target)
                self.callback()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.extxyz"
            atoms = Atoms(
                symbols=["Bi", "Te"],
                positions=[[1.0, 1.0, 1.0], [4.2, 1.0, 1.0]],
                cell=np.diag([10.0, 10.0, 10.0]),
                pbc=True,
            )
            write_extxyz_float64(source, atoms)
            candidate = {
                "defect_family": "VTe",
                "defect_config_id": "VTe_001",
                "candidate_id": "aggressive_candidate",
                "candidate_type": "isolated_distortion_and_rattle",
                "defect_count": 1,
                "parameters": {},
                "candidate_structure": source.as_posix(),
                "geometry_sha256": geometry_sha256(atoms),
            }
            settings = {
                "optimizer": "FIRE2",
                "fixed_cell": True,
                "target_fmax_ev_per_angstrom": 0.10,
                "max_steps": 300,
                "maxstep_angstrom": 0.10,
                "warning_min_distance_angstrom": 1.8,
                "hard_min_distance_angstrom": 1.2,
                "seed": 42,
            }
            model = {"model": "test_harmonic"}
            target = np.asarray([[1.2, 1.0, 1.0], [4.0, 1.0, 1.0]])

            with (
                patch(
                    "relaxgen.prerelax.relax_reconstruction_candidates_eqv3.FIRE2",
                    UnsafeFIRE2,
                ),
                patch(
                    "relaxgen.prerelax.relax_reconstruction_candidates_eqv3.FIRE",
                    SafeFIRE,
                ),
            ):
                result = relax_candidate(
                    candidate,
                    source,
                    root / "output",
                    HarmonicCalculator(target),
                    model,
                    settings,
                    run_signature(model, settings),
                )

            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["converged"])
            self.assertTrue(result["fallback_used"])
            self.assertTrue(SafeFIRE.used_downhill_check)
            self.assertEqual(result["final_optimizer"], "FIRE")
            self.assertEqual(
                [attempt["status"] for attempt in result["optimizer_attempts"]],
                ["geometry_safety_failure", "completed"],
            )
            self.assertEqual(
                result["optimizer_attempts"][1]["restart_geometry_sha256"],
                geometry_sha256(atoms),
            )
            self.assertTrue((root / "output" / "fallback_optimizer.log").is_file())


if __name__ == "__main__":
    unittest.main()
