from __future__ import annotations

import unittest

from relaxgen.dispatch.split_plan_a import family_quotas, split


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


if __name__ == "__main__":
    unittest.main()
