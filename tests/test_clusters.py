from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from relaxgen.dispatch import clusters
from relaxgen.dispatch.qe_dispatch import PROJECT_ROOT, build_parser, init


def scontrol(cluster_name: str):
    output = f"ClusterName             = {cluster_name}\nSlurmctldHost[0]        = sched01\n"
    return mock.patch.object(clusters.subprocess, "run",
                             return_value=subprocess.CompletedProcess([], 0, stdout=output))


class DetectClusterTest(unittest.TestCase):
    def test_cs_is_detected_from_slurm_cluster_name(self):
        with scontrol("cs"):
            self.assertEqual(clusters.detect_cluster(), "cs")

    def test_rivanna_is_detected_from_its_shared_storage(self):
        with scontrol("unknown-name"), mock.patch.object(clusters, "RIVANNA_MARKER", Path(__file__).parent):
            self.assertEqual(clusters.detect_cluster(), "rivanna")

    def test_unknown_cluster_stops_instead_of_guessing(self):
        with scontrol("elsewhere"), mock.patch.object(clusters, "RIVANNA_MARKER", Path("/nonexistent")):
            with self.assertRaises(SystemExit):
                clusters.detect_cluster()


class ProfileTest(unittest.TestCase):
    def test_every_profile_launcher_exists(self):
        for cluster, workloads in clusters.PROFILES.items():
            for workload in workloads:
                try:
                    backends = clusters.backends_for(cluster, workload, Path("/nonexistent/campaign"))
                except SystemExit:
                    continue
                for backend in backends:
                    self.assertTrue((PROJECT_ROOT / backend["launcher"]).is_file(), (cluster, backend["name"]))

    def test_cs_relaxations_use_only_the_listed_gpus(self):
        (backend,) = clusters.backends_for("cs", "relax", Path("/nonexistent/campaign"))
        self.assertEqual(backend["kind"], "gpu")
        self.assertIn("--constraint=h100_94gb|a100_80gb", backend["sbatch_args"])
        launcher = (PROJECT_ROOT / backend["launcher"]).read_text()
        self.assertIn('#SBATCH --constraint="h100_94gb|a100_80gb"', launcher)
        self.assertLessEqual(backend["segment_seconds"], 4 * 24 * 3600)   # gpu partition limit
        # Queued relaxations give way to the user's other jobs by holding, not by --nice.
        self.assertEqual(backend["yield_to"]["features"], clusters.CS_GPU_FEATURES)
        self.assertFalse(any(arg.startswith("--nice") for arg in backend["sbatch_args"]))

    def test_cs_has_no_scf_backends(self):
        with self.assertRaises(SystemExit):
            clusters.backends_for("cs", "scf", Path("/nonexistent/campaign"))

    def test_rivanna_profile_keeps_its_backends(self):
        scf = [b["name"] for b in clusters.backends_for("rivanna", "scf", Path("/nonexistent/campaign"))]
        relax = clusters.backends_for("rivanna", "relax", Path("/nonexistent/campaign"))
        self.assertEqual(scf, ["gpu", "cpu-parallel-2x42", "cpu-standard-42"])
        self.assertEqual([b["name"] for b in relax], ["gpu-1x", "gpu-2x", "cpu-parallel-2x42"])
        self.assertTrue(all(b["launcher"].endswith("_uva.sh") for b in relax))
        for backend in relax:
            self.assertEqual("yield_to" in backend, backend["kind"] == "gpu")
            self.assertFalse(any(arg.startswith("--nice") for arg in backend["sbatch_args"]))


class InitTest(unittest.TestCase):
    def test_init_records_the_cluster_and_its_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "relax_manifest.json"
            manifest.write_text('{"tasks": []}')
            args = build_parser().parse_args(
                ["init", "--dispatch-dir", f"{tmp}/dispatch", "--manifest", str(manifest),
                 "--job-name", "qd_test", "--workload", "relax", "--cluster", "cs"])
            config = init(args)
        self.assertEqual(config["cluster"], "cs")
        self.assertEqual([b["launcher"] for b in config["backends"]], ["slurm/run_dft_relax_slurm_cs.sh"])


if __name__ == "__main__":
    unittest.main()
