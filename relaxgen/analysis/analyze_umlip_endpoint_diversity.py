#!/usr/bin/env python3
"""Measure structural diversity among relaxed candidates for each defect configuration."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "q05": None, "q95": None, "maximum": None}
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def center_fractional_coordinates(defect_manifest: dict) -> np.ndarray:
    centers = []
    for action in defect_manifest["typed_actions"]:
        centers.extend(action["canonical_fractional_coordinates"])
    return np.asarray(centers, dtype=float)


def local_mask(
    positions: np.ndarray, cell: np.ndarray, centers_frac: np.ndarray, cutoff: float
) -> np.ndarray:
    mask = np.zeros(len(positions), dtype=bool)
    for center_frac in centers_frac:
        vectors = positions - center_frac @ cell
        mic_vectors, _ = find_mic(vectors, cell, pbc=True)
        mask |= np.linalg.norm(mic_vectors, axis=1) <= cutoff
    return mask


def matched_mic_displacements(left, right) -> np.ndarray:
    left_symbols = np.asarray(left.get_chemical_symbols())
    right_symbols = np.asarray(right.get_chemical_symbols())
    if sorted(left_symbols) != sorted(right_symbols):
        raise ValueError("RMSD requires identical composition.")
    if not np.allclose(left.cell.array, right.cell.array, atol=1e-10, rtol=0.0):
        raise ValueError("RMSD requires matching cells.")

    left_positions = left.get_positions()
    right_positions = right.get_positions().copy()
    displacements = np.zeros_like(left_positions)

    # Remove any irrelevant collective translation while repeatedly updating the
    # optimal species-preserving assignment under periodic boundary conditions.
    for _ in range(4):
        for symbol in sorted(set(left_symbols)):
            left_indices = np.flatnonzero(left_symbols == symbol)
            right_indices = np.flatnonzero(right_symbols == symbol)
            vectors = (
                right_positions[right_indices][None, :, :]
                - left_positions[left_indices][:, None, :]
            )
            mic_vectors, _ = find_mic(
                vectors.reshape(-1, 3), left.cell, pbc=True
            )
            mic_vectors = mic_vectors.reshape(
                len(left_indices), len(right_indices), 3
            )
            costs = np.sum(mic_vectors * mic_vectors, axis=2)
            rows, columns = linear_sum_assignment(costs)
            displacements[left_indices[rows]] = mic_vectors[rows, columns]
        translation = np.mean(displacements, axis=0)
        right_positions -= translation
        if np.linalg.norm(translation) < 1e-10:
            break
    return displacements


def pairwise_rmsd(
    structures: list, centers_frac: np.ndarray, local_cutoff: float
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    count = len(structures)
    global_matrix = np.zeros((count, count), dtype=float)
    local_matrix = np.zeros((count, count), dtype=float)
    local_atom_counts = []
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            displacement = matched_mic_displacements(
                structures[left_index], structures[right_index]
            )
            squared = np.sum(displacement * displacement, axis=1)
            left = structures[left_index]
            left_positions = left.get_positions()
            right_positions_matched = left_positions + displacement
            mask = local_mask(
                left_positions, left.cell.array, centers_frac, local_cutoff
            ) | local_mask(
                right_positions_matched, left.cell.array, centers_frac, local_cutoff
            )
            if not np.any(mask):
                raise RuntimeError("No atoms found in a defect-local comparison region.")
            local_atom_counts.append(int(np.sum(mask)))
            global_value = float(np.sqrt(np.mean(squared)))
            local_value = float(np.sqrt(np.mean(squared[mask])))
            global_matrix[left_index, right_index] = global_value
            global_matrix[right_index, left_index] = global_value
            local_matrix[left_index, right_index] = local_value
            local_matrix[right_index, left_index] = local_value
    return global_matrix, local_matrix, local_atom_counts


def complete_linkage_labels(matrix: np.ndarray, threshold: float) -> np.ndarray:
    if len(matrix) == 1:
        return np.ones(1, dtype=int)
    tree = linkage(squareform(matrix, checks=False), method="complete")
    return fcluster(tree, threshold, criterion="distance")


def medoid_indices(matrix: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    medoids = {}
    for label in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == label)
        within = matrix[np.ix_(members, members)]
        medoids[int(label)] = int(members[np.argmin(np.mean(within, axis=1))])
    return medoids


def find_records(
    input_dir: Path, relaxation_dir: Path, converged_only: bool
) -> dict[str, list[dict]]:
    records = defaultdict(list)
    for status_path in sorted(relaxation_dir.glob("*/*/*/status.json")):
        status = load_json(status_path)
        if status.get("status") != "completed":
            continue
        if converged_only and not status.get("converged", False):
            continue
        family = status["defect_family"]
        config = status["defect_config_id"]
        candidate = status["candidate_id"]
        records[config].append(
            {
                "family": family,
                "config": config,
                "candidate": candidate,
                "candidate_type": status["candidate_type"],
                "converged": bool(status["converged"]),
                "reached_max_steps": bool(status["reached_max_steps"]),
                "energy_ev": float(status["final"]["energy_ev"]),
                "fmax_ev_per_angstrom": float(
                    status["final"]["max_force_ev_per_angstrom"]
                ),
                "structure": Path(status["relaxed_structure"]),
                "defect_manifest": None,
            }
        )

    for config_records in records.values():
        family = config_records[0]["family"]
        first_candidate = json.loads(
            (input_dir / family / config_records[0]["config"] / "candidates.jsonl")
            .read_text()
            .splitlines()[0]
        )
        manifest = Path(first_candidate["source_defect_manifest"])
        for record in config_records:
            record["defect_manifest"] = manifest
    return records


def analyze(
    input_dir: Path,
    relaxation_dir: Path,
    output_dir: Path,
    local_cutoff: float,
    thresholds: list[float],
    converged_only: bool,
) -> dict:
    by_config = find_records(input_dir, relaxation_dir, converged_only)
    if not by_config:
        raise RuntimeError(f"No completed endpoints found under {relaxation_dir}")

    family_rows = defaultdict(
        lambda: {
            "configuration_count": 0,
            "endpoint_count": 0,
            "converged_count": 0,
            "pairwise_local_rmsd_angstrom": [],
            "clusters": {str(value): 0 for value in thresholds},
            "multicluster_configurations": {str(value): 0 for value in thresholds},
        }
    )
    total_clusters = {str(value): 0 for value in thresholds}
    total_multicluster = {str(value): 0 for value in thresholds}
    all_local_distances = []
    all_global_distances = []
    recipe_novelty = defaultdict(lambda: {"count": 0, "outside_unperturbed_cluster": 0})
    configuration_results = []
    selection_threshold = min(thresholds, key=lambda value: abs(value - 0.10))

    for config, records in sorted(by_config.items()):
        records.sort(key=lambda row: row["candidate"])
        structures = [read(record["structure"], index=-1) for record in records]
        manifest = load_json(records[0]["defect_manifest"])
        global_matrix, local_matrix, local_atom_counts = pairwise_rmsd(
            structures, center_fractional_coordinates(manifest), local_cutoff
        )
        upper = np.triu_indices(len(records), k=1)
        local_values = local_matrix[upper].tolist()
        global_values = global_matrix[upper].tolist()
        all_local_distances.extend(local_values)
        all_global_distances.extend(global_values)

        cluster_counts = {}
        selected_labels = None
        for threshold in thresholds:
            labels = complete_linkage_labels(local_matrix, threshold)
            count = len(set(labels.tolist()))
            key = str(threshold)
            cluster_counts[key] = count
            total_clusters[key] += count
            if count > 1:
                total_multicluster[key] += 1
            if threshold == selection_threshold:
                selected_labels = labels

        assert selected_labels is not None
        unperturbed_indices = [
            index for index, record in enumerate(records) if record["candidate"] == "unperturbed"
        ]
        unperturbed_label = (
            int(selected_labels[unperturbed_indices[0]]) if unperturbed_indices else None
        )
        for index, record in enumerate(records):
            novelty = recipe_novelty[record["candidate_type"]]
            novelty["count"] += 1
            if unperturbed_label is not None and int(selected_labels[index]) != unperturbed_label:
                novelty["outside_unperturbed_cluster"] += 1

        medoids = medoid_indices(local_matrix, selected_labels)
        representatives = []
        for label, index in medoids.items():
            record = records[index]
            representatives.append(
                {
                    "cluster": label,
                    "candidate": record["candidate"],
                    "candidate_type": record["candidate_type"],
                    "converged": record["converged"],
                    "energy_ev": record["energy_ev"],
                    "structure": str(record["structure"]),
                }
            )

        family = records[0]["family"]
        family_row = family_rows[family]
        family_row["configuration_count"] += 1
        family_row["endpoint_count"] += len(records)
        family_row["converged_count"] += sum(row["converged"] for row in records)
        family_row["pairwise_local_rmsd_angstrom"].extend(local_values)
        for threshold in thresholds:
            key = str(threshold)
            family_row["clusters"][key] += cluster_counts[key]
            if cluster_counts[key] > 1:
                family_row["multicluster_configurations"][key] += 1

        configuration_results.append(
            {
                "defect_family": family,
                "defect_config_id": config,
                "endpoint_count": len(records),
                "converged_count": sum(row["converged"] for row in records),
                "local_atom_count": quantiles(local_atom_counts),
                "pairwise_global_rmsd_angstrom": quantiles(global_values),
                "pairwise_local_rmsd_angstrom": quantiles(local_values),
                "cluster_count_by_local_rmsd_threshold": cluster_counts,
                "representatives_at_selection_threshold": representatives,
            }
        )

    family_summary = {}
    for family, row in sorted(family_rows.items()):
        local_values = row.pop("pairwise_local_rmsd_angstrom")
        row["pairwise_local_rmsd_angstrom"] = quantiles(local_values)
        family_summary[family] = row

    endpoint_count = sum(len(records) for records in by_config.values())
    converged_count = sum(
        record["converged"] for records in by_config.values() for record in records
    )
    result = {
        "method": {
            "endpoint_filter": "strictly converged only" if converged_only else "all completed",
            "comparison_scope": "within each canonical defect configuration",
            "distance": (
                "translation-aligned, species-permutation-invariant MIC RMSD"
            ),
            "local_cutoff_angstrom": local_cutoff,
            "clustering": "complete-linkage agglomerative clustering",
            "thresholds_angstrom": thresholds,
            "selection_threshold_angstrom": selection_threshold,
            "note": (
                "Different defect configurations and compositions are intentionally not "
                "merged by RMSD. Cluster counts are sensitivity estimates, "
                "not DFT-confirmed basin identities."
            ),
        },
        "configuration_count": len(by_config),
        "endpoint_count": endpoint_count,
        "converged_count": int(converged_count),
        "hit_max_steps_count": endpoint_count - int(converged_count),
        "pairwise_global_rmsd_angstrom": quantiles(all_global_distances),
        "pairwise_local_rmsd_angstrom": quantiles(all_local_distances),
        "estimated_distinct_endpoints_by_threshold": total_clusters,
        "multicluster_configurations_by_threshold": total_multicluster,
        "family_summary": family_summary,
        "candidate_type_novelty_at_selection_threshold": dict(sorted(recipe_novelty.items())),
        "configuration_results": configuration_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diversity_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    write_human_summary(result, output_dir / "diversity_summary.txt")
    return result


def write_human_summary(result: dict, path: Path) -> None:
    local = result["pairwise_local_rmsd_angstrom"]
    lines = [
        "uMLIP endpoint diversity summary",
        "",
        f"Endpoint filter: {result['method']['endpoint_filter']}",
        f"Canonical defect configurations: {result['configuration_count']}",
        f"Relaxed endpoints: {result['endpoint_count']}",
        f"Strictly converged endpoints: {result['converged_count']}",
        f"Endpoints at maximum steps: {result['hit_max_steps_count']}",
        "",
        "Within-configuration pairwise local MIC RMSD",
        f"Median: {local['median']:.4f} A",
        f"95th percentile: {local['q95']:.4f} A",
        f"Maximum: {local['maximum']:.4f} A",
        "",
        "Complete-linkage endpoint counts",
    ]
    for threshold, count in result["estimated_distinct_endpoints_by_threshold"].items():
        multi = result["multicluster_configurations_by_threshold"][threshold]
        lines.append(
            f"Local RMSD <= {float(threshold):.2f} A: {count} clusters; "
            f"{multi}/{result['configuration_count']} configurations have multiple clusters"
        )

    threshold = str(result["method"]["selection_threshold_angstrom"])
    lines.extend(["", f"Family breakdown at {float(threshold):.2f} A local RMSD"])
    for family, row in result["family_summary"].items():
        lines.append(
            f"{family}: {row['endpoint_count']} endpoints, "
            f"{row['clusters'][threshold]} clusters, "
            f"{row['multicluster_configurations'][threshold]}/"
            f"{row['configuration_count']} multicluster configurations"
        )

    lines.extend(["", "Candidate-type novelty relative to the unperturbed endpoint cluster"])
    for candidate_type, row in result["candidate_type_novelty_at_selection_threshold"].items():
        fraction = row["outside_unperturbed_cluster"] / row["count"]
        lines.append(
            f"{candidate_type}: {row['outside_unperturbed_cluster']}/{row['count']} "
            f"({100.0 * fraction:.1f}%) outside"
        )
    lines.extend(["", result["method"]["note"]])
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--relaxation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-cutoff", type=float, default=5.0)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30]
    )
    parser.add_argument("--converged-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze(
        args.input_dir,
        args.relaxation_dir,
        args.output_dir,
        args.local_cutoff,
        sorted(set(args.thresholds)),
        args.converged_only,
    )
    print((args.output_dir / "diversity_summary.txt").read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
