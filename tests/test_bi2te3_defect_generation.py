from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from relaxgen.structures.bi2te3_defect_generation import (
    DEFAULT_DEFECT_FAMILY_CATALOG,
    DEFAULT_SOURCE_DB,
    _operation_permutations,
    array_sha256,
    build_site_records,
    enumerate_defect_family,
    enumerate_2vte_pairs,
    formula_counts,
    generate_defect_catalog,
    generate_defect_catalog_preview,
    generate_2vte_preview,
    generate_2vte_structures,
    geometry_sha256,
    load_defect_family_catalog,
    prepare_qe_relaxation,
    reconstruct_inferred_pristine,
    validate_relaxed_atoms,
    write_extxyz_float64,
    write_json,
)


class Bi2Te3DefectGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DEFAULT_SOURCE_DB.is_file():
            raise unittest.SkipTest(
                f"Bi2Te3 source database is unavailable: {DEFAULT_SOURCE_DB}"
            )
        cls.inferred, cls.inferred_manifest = reconstruct_inferred_pristine(
            DEFAULT_SOURCE_DB
        )
        cls.site_records = build_site_records(cls.inferred)

    def test_inferred_parent_and_algorithmic_pair_enumeration(self) -> None:
        self.assertEqual(formula_counts(self.inferred), {"Bi": 16, "Te": 24})
        records, summary = enumerate_2vte_pairs(
            self.inferred,
            self.inferred,
            self.site_records,
        )
        self.assertEqual(summary["symmetry_operation_count"], 96)
        self.assertEqual(summary["raw_pair_count"], 276)
        self.assertEqual(summary["inequivalent_pair_count"], 14)
        self.assertEqual(summary["symmetry_orbit_multiplicity_sum"], 276)
        self.assertEqual(
            summary["category_counts"],
            {"TeA-TeA": 7, "TeA-TeB": 4, "TeB-TeB": 3},
        )
        self.assertEqual(len({record["defect_config_id"] for record in records}), 14)

    def test_full_typed_catalog_symmetry_counts_are_computed(self) -> None:
        families, _ = load_defect_family_catalog(DEFAULT_DEFECT_FAMILY_CATALOG)
        permutations = _operation_permutations(self.inferred, symprec=1.0e-3)
        expected = {
            "VTe": (24, 2),
            "2VTe": (276, 14),
            "3VTe": (2024, 46),
            "VBi": (16, 1),
            "2VBi": (120, 7),
            "3VBi": (560, 12),
            "Te_Bi": (16, 1),
            "2Te_Bi": (120, 7),
            "3Te_Bi": (560, 12),
            "Bi_Te": (24, 2),
            "2Bi_Te": (276, 14),
            "3Bi_Te": (2024, 46),
            "antisite_pair": (384, 12),
            "VBi_VTe": (384, 12),
        }
        observed: dict[str, tuple[int, int]] = {}
        for family in families:
            records, summary = enumerate_defect_family(
                self.inferred,
                self.inferred,
                self.site_records,
                family,
                permutations=permutations,
            )
            observed[family.name] = (
                summary["raw_configuration_count"],
                summary["inequivalent_configuration_count"],
            )
            self.assertEqual(
                summary["symmetry_orbit_multiplicity_sum"],
                summary["raw_configuration_count"],
            )
            self.assertEqual(len(records), summary["inequivalent_configuration_count"])
        self.assertEqual(observed, expected)
        self.assertEqual(sum(raw for raw, _ in observed.values()), 6808)
        self.assertEqual(sum(unique for _, unique in observed.values()), 188)

    def test_structural_equivalence_accepts_reordering_and_small_motion(self) -> None:
        translated = self.inferred.copy()
        shift = np.array([2.0e-4, -3.0e-4, 1.0e-4], dtype=np.float64)
        translated.positions = translated.positions + shift
        permutation = np.roll(np.arange(len(translated)), 7)
        reordered = translated[permutation]
        reordered.set_cell(
            np.asarray(self.inferred.cell.array, dtype=np.float64) + 1.0e-8,
            scale_atoms=False,
        )
        reordered.set_pbc((True, True, True))

        canonical, metrics = validate_relaxed_atoms(self.inferred, reordered)

        self.assertEqual(
            canonical.get_chemical_symbols(), self.inferred.get_chemical_symbols()
        )
        self.assertTrue(np.array_equal(canonical.cell.array, self.inferred.cell.array))
        self.assertLess(metrics["mapped_rmsd_angstrom"], 1.0e-3)
        self.assertLess(metrics["mapped_max_displacement_angstrom"], 1.0e-3)
        self.assertFalse(metrics["qe_output_order_preserved"])
        self.assertEqual(metrics["accepted_symprec_angstrom"], 1.0e-3)

    def test_prepare_only_qe_input_uses_fixed_cell_precise_protocol(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        pseudo_dir = project_root / "pseudos"
        if not pseudo_dir.is_dir():
            self.skipTest(f"QE pseudopotentials are unavailable: {pseudo_dir}")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inferred_path = root / "inferred.extxyz"
            write_extxyz_float64(inferred_path, self.inferred)
            manifest = prepare_qe_relaxation(
                inferred_path,
                root / "qe",
                pseudo_dir=pseudo_dir,
                qe_outdir=root / "qe_scratch",
            )
            text = Path(manifest["qe_input"]).read_text(encoding="utf-8")

        self.assertEqual(manifest["execution_status"], "prepared_not_executed")
        self.assertIn("calculation      = 'relax'", text)
        self.assertIn("ion_dynamics     = 'bfgs'", text)
        self.assertNotIn("cell_dynamics", text.lower())
        self.assertIn("ecutwfc          = 60.0", text)
        self.assertIn("ecutrho          = 480.0", text)
        self.assertIn("6 6 6  0 0 0", text)
        first_cell_line = " ".join(
            f"{float(value):.17g}" for value in self.inferred.cell.array[0]
        )
        self.assertIn(first_cell_line, text)

    def test_constructs_fourteen_untouched_bi16te22_structures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inferred_path = root / "inferred.extxyz"
            canonical_path = root / "canonical.extxyz"
            site_manifest_path = root / "bulk_sites.json"
            write_extxyz_float64(inferred_path, self.inferred)
            write_extxyz_float64(canonical_path, self.inferred)
            site_manifest = {
                "artifact_type": "canonical_bi16te24_bulk_sites",
                "inferred_structure": inferred_path.as_posix(),
                "inferred_geometry_sha256": geometry_sha256(self.inferred),
                "canonical_structure": canonical_path.as_posix(),
                "canonical_geometry_sha256": geometry_sha256(self.inferred),
                "canonical_cell_sha256": array_sha256(
                    self.inferred.cell.array, np.float64
                ),
                "sites": self.site_records,
            }
            write_json(site_manifest_path, site_manifest)

            summary = generate_2vte_structures(
                canonical_path,
                site_manifest_path,
                root / "generated",
            )
            defect_paths = sorted((root / "generated" / "defects").glob("*/unperturbed.extxyz"))
            self.assertEqual(summary["written_structure_count"], 14)
            self.assertEqual(len(defect_paths), 14)
            for defect_path in defect_paths:
                defect = ase_read(defect_path, index=0, format="extxyz")
                self.assertEqual(len(defect), 38)
                self.assertEqual(formula_counts(defect), {"Bi": 16, "Te": 22})
                self.assertTrue(
                    np.array_equal(defect.cell.array, self.inferred.cell.array)
                )

            catalog = root / "generated" / "catalog" / "2VTe_configurations.jsonl"
            rows = [json.loads(line) for line in catalog.read_text().splitlines()]
            self.assertEqual(sum(row["symmetry_orbit_multiplicity"] for row in rows), 276)
            self.assertTrue(
                all(row["construction_contract"]["distorted"] is False for row in rows)
            )

    def test_preview_generation_is_explicitly_unvalidated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_path = root / "inferred.extxyz"
            write_extxyz_float64(parent_path, self.inferred)
            summary = generate_2vte_preview(
                parent_path,
                root / "preview",
            )

            self.assertEqual(summary["written_structure_count"], 14)
            self.assertEqual(
                summary["parent_validation_status"],
                "unvalidated_inferred_preview",
            )
            manifest = json.loads(
                (
                    root
                    / "preview"
                    / "defects"
                    / "2VTe_001"
                    / "defect.json"
                ).read_text()
            )
            self.assertEqual(
                manifest["parent_validation_status"],
                "unvalidated_inferred_preview",
            )

    def test_generic_catalog_constructs_3vte_and_antisite_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inferred_path = root / "inferred.extxyz"
            canonical_path = root / "canonical.extxyz"
            site_manifest_path = root / "bulk_sites.json"
            write_extxyz_float64(inferred_path, self.inferred)
            write_extxyz_float64(canonical_path, self.inferred)
            write_json(
                site_manifest_path,
                {
                    "artifact_type": "canonical_bi16te24_bulk_sites",
                    "parent_validation_status": "unit_test_parent",
                    "inferred_structure": inferred_path.as_posix(),
                    "inferred_geometry_sha256": geometry_sha256(self.inferred),
                    "canonical_structure": canonical_path.as_posix(),
                    "canonical_geometry_sha256": geometry_sha256(self.inferred),
                    "canonical_cell_sha256": array_sha256(
                        self.inferred.cell.array, np.float64
                    ),
                    "sites": self.site_records,
                },
            )

            summary = generate_defect_catalog(
                canonical_path,
                site_manifest_path,
                root / "generated",
                selected_families=["3VTe", "antisite_pair"],
            )
            self.assertEqual(summary["total_raw_configuration_count"], 2408)
            self.assertEqual(summary["total_inequivalent_configuration_count"], 58)
            self.assertEqual(summary["written_structure_count"], 58)
            self.assertEqual(
                summary["formula_distribution"], {"Bi16Te21": 46, "Bi16Te24": 12}
            )

            vacancy_paths = sorted(
                (root / "generated" / "defects" / "3VTe").glob(
                    "*/unperturbed.extxyz"
                )
            )
            antisite_paths = sorted(
                (root / "generated" / "defects" / "antisite_pair").glob(
                    "*/unperturbed.extxyz"
                )
            )
            self.assertEqual(len(vacancy_paths), 46)
            self.assertEqual(len(antisite_paths), 12)
            for path in vacancy_paths:
                defect = ase_read(path, index=0, format="extxyz")
                self.assertEqual(formula_counts(defect), {"Bi": 16, "Te": 21})
                self.assertEqual(len(defect), 37)
                self.assertTrue(
                    np.array_equal(defect.cell.array, self.inferred.cell.array)
                )
            for path in antisite_paths:
                defect = ase_read(path, index=0, format="extxyz")
                self.assertEqual(formula_counts(defect), {"Bi": 16, "Te": 24})
                self.assertEqual(len(defect), 40)
                self.assertTrue(np.array_equal(defect.positions, self.inferred.positions))
                changed_species = sum(
                    left != right
                    for left, right in zip(
                        defect.get_chemical_symbols(),
                        self.inferred.get_chemical_symbols(),
                    )
                )
                self.assertEqual(changed_species, 2)

            first_manifest = json.loads(
                (
                    root
                    / "generated"
                    / "defects"
                    / "antisite_pair"
                    / "antisite_pair_001"
                    / "defect.json"
                ).read_text()
            )
            self.assertTrue(
                first_manifest["construction_contract"][
                    "constructed_from_stable_bulk_site_ids"
                ]
            )
            self.assertFalse(
                first_manifest["construction_contract"]["shakenbreak_applied"]
            )
            self.assertFalse(first_manifest["construction_contract"]["relaxed"])

    def test_generic_catalog_preview_preserves_preview_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_path = root / "inferred.extxyz"
            write_extxyz_float64(parent_path, self.inferred)
            summary = generate_defect_catalog_preview(
                parent_path,
                root / "preview",
                selected_families=["VTe", "VBi"],
            )
            self.assertEqual(summary["written_structure_count"], 3)
            self.assertEqual(
                summary["parent_validation_status"],
                "unvalidated_inferred_preview",
            )


if __name__ == "__main__":
    unittest.main()
