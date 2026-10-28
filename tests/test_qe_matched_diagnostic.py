from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from relaxgen.dft.qe_matched_diagnostic import (
    DEFAULT_PSEUDO_DIR,
    DEFAULT_UMLIP_DIR,
    _parse_qe_timers,
    _print_task_statistics,
    _qe_time_to_seconds,
    prepare,
)


class QEMatchedDiagnosticTest(unittest.TestCase):
    def test_task_statistics_are_human_readable_and_generic(self) -> None:
        status = {
            "task_id": "00_easy_vte_umlip_endpoint_b200",
            "case_id": "easy_vte",
            "candidate_id": "unperturbed",
            "resource_profile": "b200",
            "execution_status": "completed",
            "scientifically_converged": True,
            "singlepoint_evaluation_count": 2,
            "reported_bfgs_steps": 1,
            "electronic_iteration_counts": [16, 7],
            "final_max_force_ev_per_angstrom": 0.008,
            "timing_events": "/tmp/timing_events.jsonl",
            "status": "/tmp/status.json",
            "runtime": {
                "allocated_gpu_name": "NVIDIA B200",
                "qe_wall_time_seconds": 180.0,
            },
            "qe_footer_timing": {
                "pwscf_total_wall_seconds": 175.0,
                "routines": {
                    "electrons": {
                        "wall_seconds": 150.0,
                        "wall_seconds_per_call": 75.0,
                    },
                    "forces": {"wall_seconds": 10.0},
                },
            },
            "live_timing": {
                "evaluations": [
                    {
                        "observed_scf_solve_wall_seconds": 80.0,
                        "observed_scf_through_force_wall_seconds": 85.0,
                    },
                    {
                        "observed_scf_solve_wall_seconds": 70.0,
                        "observed_scf_through_force_wall_seconds": 75.0,
                    },
                ]
            },
        }
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            _print_task_statistics(status)
        text = stream.getvalue()
        self.assertIn("QE relaxation statistics", text)
        self.assertIn("Single-point evaluations:     2", text)
        self.assertIn("Wall time / single point:     1m 30.00s", text)
        self.assertIn("Observed SCF solve times:", text)
        self.assertNotIn("Michael", text)

    def test_parses_qe_footer_timing(self) -> None:
        text = """
     init_run     :     67.87s CPU     72.52s WALL (       1 calls)
     electrons    :   2348.83s CPU   2381.27s WALL (       4 calls)
     forces       :      8.84s CPU      8.93s WALL (       4 calls)
     stress       :     17.71s CPU     18.00s WALL (       4 calls)
     PWSCF        :  40m52.85s CPU  41m44.75s WALL
"""
        timers = _parse_qe_timers(text)
        self.assertEqual(_qe_time_to_seconds("1h2m3.5s"), 3723.5)
        self.assertEqual(_qe_time_to_seconds("7h56m"), 28560.0)
        self.assertEqual(timers["pwscf_total_wall_seconds"], 2504.75)
        self.assertEqual(timers["routines"]["electrons"]["calls"], 4)
        self.assertAlmostEqual(
            timers["routines"]["electrons"]["wall_seconds_per_call"],
            595.3175,
        )

    def test_prepares_exact_matched_task_matrix(self) -> None:
        if not DEFAULT_UMLIP_DIR.is_dir():
            self.skipTest(f"uMLIP results are unavailable: {DEFAULT_UMLIP_DIR}")
        if not DEFAULT_PSEUDO_DIR.is_dir():
            self.skipTest(f"QE pseudopotentials are unavailable: {DEFAULT_PSEUDO_DIR}")

        with tempfile.TemporaryDirectory() as temporary:
            manifest = prepare(
                argparse.Namespace(
                    umlip_dir=DEFAULT_UMLIP_DIR,
                    output_dir=Path(temporary),
                    pseudo_dir=DEFAULT_PSEUDO_DIR,
                    kpts=(6, 6, 6),
                    ecutwfc=60.0,
                    ecutrho=480.0,
                    fmax=0.01,
                    max_steps=100,
                    overwrite=False,
                )
            )

            self.assertEqual(manifest["task_count"], 12)
            self.assertEqual(len(manifest["selected_cases"]), 4)
            self.assertEqual(
                [(task["resource_profile"], task["source_kind"]) for task in manifest["tasks"]],
                [("b200", "umlip_endpoint")] * 4
                + [("a100", "umlip_endpoint")] * 4
                + [("a40", "umlip_endpoint")] * 4,
            )
            self.assertEqual(manifest["task_design"]["direct_dft_controls"], 0)
            self.assertFalse(
                manifest["task_design"]["pre_umlip_candidates_run_with_dft"]
            )
            self.assertEqual(manifest["protocol"]["force_max_ev_per_angstrom"], 0.01)
            self.assertTrue(manifest["protocol"]["fixed_cell"])
            self.assertIn(
                "electrons",
                manifest["protocol"]["timing_instrumentation"][
                    "qe_footer_routines"
                ],
            )
            for task in manifest["tasks"]:
                text = Path(task["qe_input"]).read_text(encoding="utf-8")
                self.assertIn("calculation      = 'relax'", text)
                self.assertIn("ion_dynamics     = 'bfgs'", text)
                self.assertIn("6 6 6  0 0 0", text)
                self.assertEqual(
                    Path(task["timing_events"]).name, "timing_events.jsonl"
                )

            by_case = {case["case_id"]: case for case in manifest["selected_cases"]}
            self.assertLess(by_case["easy_vte"]["umlip"]["optimizer_steps"], 10)
            self.assertGreater(
                by_case["aggressive_3bi_te"]["umlip"]["initial"][
                    "max_force_ev_per_angstrom"
                ],
                100.0,
            )
            self.assertGreater(
                by_case["diverse_3vte"]["selection_diversity"][
                    "nearest_other_endpoint_local_rmsd_angstrom"
                ],
                0.5,
            )
            self.assertGreater(
                by_case["diverse_3vbi"]["selection_diversity"][
                    "nearest_other_endpoint_local_rmsd_angstrom"
                ],
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
