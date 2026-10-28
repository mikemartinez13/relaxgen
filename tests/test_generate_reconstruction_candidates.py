from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaxgen.structures.bi2te3_defect_generation import (
    DEFAULT_SOURCE_DB,
    generate_defect_catalog_preview,
    reconstruct_inferred_pristine,
    write_extxyz_float64,
)
from relaxgen.structures.generate_reconstruction_candidates import generate_candidate_dataset


class ReconstructionCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DEFAULT_SOURCE_DB.is_file():
            raise unittest.SkipTest(
                f"Bi2Te3 source database is unavailable: {DEFAULT_SOURCE_DB}"
            )
        cls.parent, _ = reconstruct_inferred_pristine(DEFAULT_SOURCE_DB)

    def test_isolated_and_complex_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_path = root / "pristine.extxyz"
            write_extxyz_float64(parent_path, self.parent)
            generate_defect_catalog_preview(
                parent_path,
                root / "defects",
                selected_families=["VTe", "2VTe"],
            )
            summary = generate_candidate_dataset(
                root / "defects",
                root / "candidates",
                config_ids=["VTe_001", "2VTe_001"],
            )

            self.assertEqual(summary["defect_configuration_count"], 2)
            self.assertEqual(summary["candidate_count"], 22)
            self.assertFalse(
                summary["scope"]["sequential_multi_center_distortions"]
            )

            isolated = json.loads(
                (
                    root
                    / "candidates"
                    / "VTe"
                    / "VTe_001"
                    / "candidate_summary.json"
                ).read_text()
            )
            complex_summary = json.loads(
                (
                    root
                    / "candidates"
                    / "2VTe"
                    / "2VTe_001"
                    / "candidate_summary.json"
                ).read_text()
            )
            self.assertEqual(isolated["candidate_count"], 12)
            self.assertEqual(complex_summary["candidate_count"], 10)
            self.assertEqual(isolated["candidate_type_counts"]["isolated_dimer"], 1)
            self.assertEqual(
                isolated["candidate_type_counts"][
                    "isolated_distortion_and_rattle"
                ],
                4,
            )
            self.assertNotIn(
                "isolated_distortion_and_rattle",
                complex_summary["candidate_type_counts"],
            )

            candidate_records = []
            for path in (root / "candidates").glob("*/*/candidates.jsonl"):
                candidate_records.extend(
                    json.loads(line) for line in path.read_text().splitlines()
                )
            self.assertEqual(len(candidate_records), 22)
            self.assertTrue(all(not record["relaxed"] for record in candidate_records))
            self.assertTrue(
                all(not record["dft_labeled"] for record in candidate_records)
            )
            self.assertTrue(
                all(
                    record["metrics"]["minimum_interatomic_distance_angstrom"]
                    >= 1.2
                    for record in candidate_records
                )
            )


if __name__ == "__main__":
    unittest.main()
