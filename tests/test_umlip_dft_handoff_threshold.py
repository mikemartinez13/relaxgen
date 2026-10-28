from __future__ import annotations

import unittest

import numpy as np

from relaxgen.prerelax.umlip_dft_handoff_threshold import (
    _cohort_specs,
    _force_agreement,
    select_handoff_frames,
)


class HandoffFrameSelectionTest(unittest.TestCase):
    def test_each_checkpoint_maps_to_first_frame_strictly_below_it(self) -> None:
        # FIRE is not monotone: frame 3 dips below 0.1 before frame 4 rises again.
        ml_fmax = [2.0, 0.3, 0.15, 0.09, 0.12, 0.1, 0.04, 0.0009]
        records, unreached = select_handoff_frames(ml_fmax, (0.1, 0.05, 0.001, 1.0e-4))
        by_frame = {record["frame_index"]: record for record in records}

        self.assertEqual(sorted(by_frame), [3, 6, 7])
        self.assertEqual(by_frame[3]["checkpoint_labels"], ["fmax<0.1"])
        self.assertEqual(by_frame[6]["checkpoint_labels"], ["fmax<0.05"])
        self.assertEqual(by_frame[7]["checkpoint_labels"], ["fmax<0.001", "final"])
        self.assertEqual(unreached, [1.0e-4])

    def test_a_frame_serving_several_checkpoints_is_evaluated_once(self) -> None:
        records, unreached = select_handoff_frames([0.5, 0.004], (0.1, 0.01, 0.005))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["frame_index"], 1)
        self.assertEqual(
            records[0]["ml_fmax_checkpoints_ev_per_angstrom"], [0.1, 0.01, 0.005]
        )
        self.assertEqual(unreached, [])

    def test_threshold_equal_to_frame_fmax_is_not_reached(self) -> None:
        records, _ = select_handoff_frames([0.2, 0.1, 0.05], (0.1,))
        self.assertEqual(records[0]["frame_index"], 2)


class ForceAgreementTest(unittest.TestCase):
    def test_identical_forces_have_zero_error_and_angle(self) -> None:
        forces = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0e-5]])
        agreement = _force_agreement(forces, forces)
        self.assertEqual(agreement["force_error_max_ev_per_angstrom"], 0.0)
        self.assertAlmostEqual(agreement["median_ml_dft_angle_deg"], 0.0, places=5)
        # The near-zero third atom has no meaningful direction.
        self.assertEqual(agreement["resolved_angle_atom_count"], 2)

    def test_opposed_forces_are_180_degrees(self) -> None:
        ml = np.array([[0.1, 0.0, 0.0]])
        agreement = _force_agreement(ml, -ml)
        self.assertAlmostEqual(agreement["median_ml_dft_angle_deg"], 180.0)
        self.assertAlmostEqual(agreement["force_error_max_ev_per_angstrom"], 0.2)


class CohortTest(unittest.TestCase):
    def test_cohort_covers_every_family_without_duplicates(self) -> None:
        specs = _cohort_specs()
        self.assertEqual(len(specs), 20)
        families = {family for family, *_ in specs}
        self.assertEqual(len(families), 14)


if __name__ == "__main__":
    unittest.main()
