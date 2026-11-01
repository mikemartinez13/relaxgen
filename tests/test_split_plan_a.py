from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

from relaxgen.dispatch.split_plan_a import add_fallbacks, family_quotas, lowest_force_frame, split


def candidates(counts: dict[str, int]) -> list[dict]:
    return [{"defect_family": family, "label": f"{family}_{i:03d}"}
            for family, count in counts.items() for i in range(count)]


class SplitTest(unittest.TestCase):
    counts = {"3VTe": 505, "3Bi_Te": 478, "VBi": 10, "Te_Bi": 10, "2VBi": 62}

    def test_quotas_sum_to_the_total_in_proportion(self):
        quotas = family_quotas(self.counts, 200)
        self.assertEqual(sum(quotas.values()), 200)
        available = sum(self.counts.values())
        for family, count in self.counts.items():
            self.assertLess(abs(quotas[family] - 200 * count / available), 1)

    def test_split_is_exclusive_complete_and_seeded(self):
        rivanna, cs = split(candidates(self.counts), 200, seed=1)
        self.assertEqual(len(cs), 200)
        labels = [c["label"] for c in rivanna + cs]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(set(labels), {c["label"] for c in candidates(self.counts)})
        again_rivanna, again_cs = split(list(reversed(candidates(self.counts))), 200, seed=1)
        self.assertEqual([c["label"] for c in cs], [c["label"] for c in again_cs])
        self.assertEqual([c["label"] for c in rivanna], [c["label"] for c in again_rivanna])

    def test_too_many_requested(self):
        with self.assertRaises(ValueError):
            family_quotas({"a": 3}, 4)


class FallbackTest(unittest.TestCase):
    def trajectory(self, root: Path, name: str, frames: list[tuple[float, float]]) -> Path:
        """Two-atom frames given as (pair distance, ML fmax)."""

        path = root / f"{name}.extxyz"
        for distance, fmax in frames:
            atoms = Atoms("Bi2", positions=[[0, 0, 0], [distance, 0, 0]], cell=[10, 10, 10], pbc=True)
            atoms.info["ml_fmax_ev_per_angstrom"] = fmax
            ase_write(path, atoms, append=True)
        return path

    def test_lowest_safe_frame_and_the_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            usable = self.trajectory(root, "usable", [(3.0, 0.5), (1.5, 0.01), (3.0, 0.08), (3.0, 0.09)])
            failed = self.trajectory(root, "failed", [(3.0, 9.0), (3.0, 6.0)])
            self.assertEqual(lowest_force_frame(usable, 2.0)["frame"], 2)
            skipped = [{"label": name, "defect_family": "a", "eqv3_trajectory": str(path), "reason": "x"}
                       for name, path in (("usable", usable), ("failed", failed))]
            rivanna, still_skipped = add_fallbacks(candidates({"b": 5}), skipped, 5.0, 2.0, seed=1)
            self.assertEqual(len(rivanna), 6)
            added = [c for c in rivanna if c["label"] == "usable"][0]
            self.assertEqual(added["structure"], f"{usable}@2")
            self.assertEqual(added["handoff"]["rule"], "lowest_force_fallback")
            self.assertEqual([e["label"] for e in still_skipped], ["failed"])
            self.assertEqual([c["label"] for c in rivanna if c["label"] != "usable"],
                             [c["label"] for c in candidates({"b": 5})])


if __name__ == "__main__":
    unittest.main()
