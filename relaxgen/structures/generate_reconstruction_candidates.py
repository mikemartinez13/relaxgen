#!/usr/bin/env python3
"""Generate unrelaxed ShakeNBreak reconstruction candidates for Bi2Te3 defects.

The input is the unrelaxed defect catalog written by
``bi2te3_defect_generation.py``.  Isolated defects use ShakeNBreak bond
distortions, distortion-plus-rattle, and a dimer candidate.  Defect complexes
use global rattles plus independent single-center bond distortions.  A complex
is never distorted sequentially around multiple centers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from pymatgen.core import Structure
from shakenbreak.distortions import (
    apply_dimer_distortion,
    distort,
    distort_and_rattle,
    rattle,
)

from relaxgen.structures.bi2te3_defect_generation import (
    file_sha256,
    formula_counts,
    geometry_sha256,
    read_single_structure,
    write_extxyz_float64,
    write_json,
)
DISTORTION_FACTORS = (0.8, 0.6, 1.2)
ISOLATED_DISTORT_AND_RATTLE_FACTORS = (0.8, 0.6, 1.2, 1.4)
RATTLE_SEEDS = (101, 202, 303)
DISTORT_AND_RATTLE_SEEDS = (1101, 1202, 1303, 1404)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def find_defect_manifests(
    input_dir: Path,
    *,
    families: Sequence[str] | None = None,
    config_ids: Sequence[str] | None = None,
) -> list[Path]:
    manifests = sorted(input_dir.glob("defects/*/*/defect.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No defect manifests found under {input_dir / 'defects'}."
        )

    requested_families = set(families or ())
    requested_configs = set(config_ids or ())
    selected = [
        path
        for path in manifests
        if (not requested_families or path.parent.parent.name in requested_families)
        and (not requested_configs or path.parent.name in requested_configs)
    ]
    observed_families = {path.parent.parent.name for path in manifests}
    observed_configs = {path.parent.name for path in manifests}
    if requested_families - observed_families:
        raise ValueError(
            f"Unknown families: {sorted(requested_families - observed_families)}."
        )
    if requested_configs - observed_configs:
        raise ValueError(
            f"Unknown configuration IDs: {sorted(requested_configs - observed_configs)}."
        )
    if not selected:
        raise ValueError("The requested filters selected no defect configurations.")
    return selected


def defect_centers(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover vacancy coordinates or occupied defect indices from stable IDs."""

    typed_actions = manifest.get("typed_actions")
    output_mapping = manifest.get("construction_metadata", {}).get(
        "output_site_mapping"
    )
    if not isinstance(typed_actions, list) or not isinstance(output_mapping, list):
        raise ValueError(
            "The defect manifest lacks typed_actions or output_site_mapping; "
            "regenerate it with generate-defect-catalog."
        )
    output_index_by_site = {
        str(item["bulk_site_id"]): int(item["output_index"])
        for item in output_mapping
    }

    centers: list[dict[str, Any]] = []
    for action in typed_actions:
        operation = str(action["operation"])
        site_ids = [str(value) for value in action["bulk_site_ids"]]
        coordinates = action["canonical_fractional_coordinates"]
        if len(site_ids) != len(coordinates):
            raise ValueError("Defect site IDs and fractional coordinates disagree.")
        for site_id, frac_coords in zip(site_ids, coordinates):
            center = {
                "action_key": str(action["action_key"]),
                "operation": operation,
                "bulk_site_id": site_id,
                "fractional_coordinates": [float(value) for value in frac_coords],
            }
            if operation == "remove":
                center["kind"] = "vacancy"
            elif operation == "substitute":
                if site_id not in output_index_by_site:
                    raise ValueError(f"Substitution site {site_id} is absent from output mapping.")
                center["kind"] = "occupied"
                center["site_index"] = output_index_by_site[site_id]
            else:
                raise ValueError(f"Unsupported defect operation {operation!r}.")
            centers.append(center)

    centers.sort(key=lambda item: (item["action_key"], item["bulk_site_id"]))
    for ordinal, center in enumerate(centers, 1):
        center["center_id"] = f"center_{ordinal:02d}_{center['bulk_site_id']}"
    if not 1 <= len(centers) <= 3:
        raise ValueError(
            f"Expected one to three defect centers, found {len(centers)}."
        )
    return centers


def center_arguments(center: dict[str, Any]) -> dict[str, Any]:
    if center["kind"] == "vacancy":
        return {
            "frac_coords": np.asarray(
                center["fractional_coordinates"], dtype=np.float64
            )
        }
    return {"site_index": int(center["site_index"])}


def rattle_parameters(
    structure: Structure,
    *,
    stdev: float | None,
    d_min: float | None,
) -> tuple[float, float]:
    distances = np.sort(structure.distance_matrix[structure.distance_matrix > 0.8])
    if not distances.size:
        raise ValueError("Could not determine a nearest-neighbor distance.")
    nearest = float(distances[0])
    selected_stdev = float(stdev) if stdev is not None else 0.1 * nearest
    selected_d_min = float(d_min) if d_min is not None else 0.8 * nearest
    if not 0.02 <= selected_stdev <= 0.4:
        selected_stdev = 0.25
    if selected_d_min < 1.0:
        selected_d_min = 2.25
    return selected_stdev, selected_d_min


def as_fixed_cell_atoms(structure: Structure, reference: Atoms) -> Atoms:
    atoms = structure.to_ase_atoms()
    atoms.set_cell(np.asarray(reference.cell.array, dtype=np.float64), scale_atoms=False)
    atoms.set_pbc(np.asarray(reference.pbc, dtype=bool))
    return atoms


def candidate_metrics(candidate: Atoms, reference: Atoms) -> dict[str, float]:
    distances = candidate.get_all_distances(mic=True)
    minimum_distance = float(np.min(distances[np.triu_indices(len(candidate), k=1)]))
    displacements, displacement_lengths = find_mic(
        np.asarray(candidate.positions - reference.positions, dtype=np.float64),
        np.asarray(reference.cell.array, dtype=np.float64),
        pbc=np.asarray(reference.pbc, dtype=bool),
    )
    del displacements
    return {
        "minimum_interatomic_distance_angstrom": minimum_distance,
        "maximum_displacement_from_unperturbed_angstrom": float(
            np.max(displacement_lengths)
        ),
        "rms_displacement_from_unperturbed_angstrom": float(
            np.sqrt(np.mean(np.square(displacement_lengths)))
        ),
    }


def validate_candidate(
    candidate: Atoms,
    reference: Atoms,
    *,
    hard_min_distance: float,
    max_displacement: float,
) -> dict[str, float]:
    if formula_counts(candidate) != formula_counts(reference):
        raise ValueError("A candidate changed the defect composition.")
    if candidate.get_chemical_symbols() != reference.get_chemical_symbols():
        raise ValueError("A candidate changed atom ordering or species.")
    if not np.array_equal(candidate.cell.array, reference.cell.array):
        raise ValueError("A candidate changed the fixed cell.")
    if not np.isfinite(candidate.positions).all():
        raise ValueError("A candidate contains non-finite coordinates.")
    metrics = candidate_metrics(candidate, reference)
    if metrics["minimum_interatomic_distance_angstrom"] < hard_min_distance:
        raise ValueError(
            "A candidate has minimum distance "
            f"{metrics['minimum_interatomic_distance_angstrom']:.6g} Å, below "
            f"the hard limit {hard_min_distance:.6g} Å."
        )
    if metrics["maximum_displacement_from_unperturbed_angstrom"] > max_displacement:
        raise ValueError(
            "A candidate has maximum displacement "
            f"{metrics['maximum_displacement_from_unperturbed_angstrom']:.6g} Å, "
            f"above the limit {max_displacement:.6g} Å."
        )
    return metrics


def distortion_label(factor: float) -> str:
    percentage = int(round(abs(1.0 - factor) * 100))
    direction = "contract" if factor < 1.0 else "expand"
    return f"{direction}_{percentage}pct"


def expected_candidate_count(defect_count: int) -> int:
    return {1: 12, 2: 10, 3: 12}[defect_count]


def generate_for_defect(
    manifest_path: Path,
    output_dir: Path,
    *,
    num_neighbors: int = 6,
    rattle_stdev: float | None = None,
    rattle_d_min: float | None = None,
    hard_min_distance: float = 1.2,
    max_displacement: float = 2.0,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    config_id = str(manifest["defect_config_id"])
    family = str(manifest["defect_family"])
    source_structure = manifest_path.parent / "unperturbed.extxyz"
    reference = read_single_structure(source_structure)
    structure = Structure.from_ase_atoms(reference)
    centers = defect_centers(manifest)
    stdev, d_min = rattle_parameters(
        structure, stdev=rattle_stdev, d_min=rattle_d_min
    )

    candidates: list[tuple[dict[str, Any], Atoms]] = []
    geometry_hashes: set[str] = set()

    def add_candidate(
        candidate_id: str,
        candidate_type: str,
        candidate: Atoms | Structure,
        parameters: dict[str, Any],
        center: dict[str, Any] | None = None,
    ) -> None:
        atoms = (
            candidate.copy()
            if isinstance(candidate, Atoms)
            else as_fixed_cell_atoms(candidate, reference)
        )
        metrics = validate_candidate(
            atoms,
            reference,
            hard_min_distance=hard_min_distance,
            max_displacement=max_displacement,
        )
        geometry_hash = geometry_sha256(atoms)
        if geometry_hash in geometry_hashes:
            raise ValueError(f"Duplicate candidate geometry {candidate_id} for {config_id}.")
        geometry_hashes.add(geometry_hash)
        record = {
            "artifact_type": "unrelaxed_reconstruction_candidate",
            "defect_family": family,
            "defect_config_id": config_id,
            "candidate_id": candidate_id,
            "candidate_type": candidate_type,
            "defect_count": len(centers),
            "center": center,
            "parameters": parameters,
            "metrics": metrics,
            "composition": formula_counts(atoms),
            "natoms": len(atoms),
            "geometry_sha256": geometry_hash,
            "fixed_cell": True,
            "relaxed": False,
            "dft_labeled": False,
        }
        candidates.append((record, atoms))

    add_candidate("unperturbed", "unperturbed", reference, {})

    rattle_seed_count = 2 if len(centers) == 3 else 3
    for seed in RATTLE_SEEDS[:rattle_seed_count]:
        rattled = rattle(
            structure,
            stdev=stdev,
            d_min=d_min,
            seed=seed,
            max_disp=max_displacement,
        )
        add_candidate(
            f"rattle_seed_{seed}",
            "global_rattle",
            rattled,
            {"seed": seed, "stdev_angstrom": stdev, "d_min_angstrom": d_min},
        )

    for center in centers:
        kwargs = center_arguments(center)
        for factor in DISTORTION_FACTORS:
            label = distortion_label(factor)
            result = distort(
                structure,
                num_nearest_neighbours=num_neighbors,
                distortion_factor=factor,
                **kwargs,
            )
            add_candidate(
                f"{center['center_id']}_{label}",
                "single_center_bond_distortion",
                result["distorted_structure"],
                {
                    "distortion_factor": factor,
                    "num_nearest_neighbours": num_neighbors,
                    "distorted_atoms": [
                        [int(index), str(symbol)]
                        for index, symbol in result["distorted_atoms"]
                    ],
                },
                center,
            )

    if len(centers) == 1:
        center = centers[0]
        kwargs = center_arguments(center)
        for factor, seed in zip(
            ISOLATED_DISTORT_AND_RATTLE_FACTORS, DISTORT_AND_RATTLE_SEEDS
        ):
            label = distortion_label(factor)
            result = distort_and_rattle(
                structure,
                distortion_factor=factor,
                num_nearest_neighbours=num_neighbors,
                stdev=stdev,
                d_min=d_min,
                seed=seed,
                max_disp=max_displacement,
                **kwargs,
            )
            add_candidate(
                f"{center['center_id']}_{label}_rattle_seed_{seed}",
                "isolated_distortion_and_rattle",
                result["distorted_structure"],
                {
                    "distortion_factor": factor,
                    "num_nearest_neighbours": num_neighbors,
                    "seed": seed,
                    "stdev_angstrom": stdev,
                    "d_min_angstrom": d_min,
                    "distorted_atoms": [
                        [int(index), str(symbol)]
                        for index, symbol in result["distorted_atoms"]
                    ],
                },
                center,
            )
        dimer = apply_dimer_distortion(structure, **kwargs)
        add_candidate(
            f"{center['center_id']}_dimer",
            "isolated_dimer",
            dimer["distorted_structure"],
            {
                "distorted_atoms": [
                    [int(index), str(symbol)]
                    for index, symbol in dimer["distorted_atoms"]
                ]
            },
            center,
        )

    expected_count = expected_candidate_count(len(centers))
    if len(candidates) != expected_count:
        raise RuntimeError(
            f"{config_id} produced {len(candidates)} candidates; expected {expected_count}."
        )

    config_output = output_dir / family / config_id
    if config_output.exists() and any(config_output.iterdir()):
        raise FileExistsError(f"Candidate output already exists: {config_output}")
    config_output.mkdir(parents=True, exist_ok=True)
    written_records: list[dict[str, Any]] = []
    for record, atoms in candidates:
        structure_path = config_output / f"{record['candidate_id']}.extxyz"
        write_extxyz_float64(structure_path, atoms)
        full_record = dict(record)
        full_record.update(
            {
                "source_defect_manifest": manifest_path.as_posix(),
                "source_defect_manifest_sha256": file_sha256(manifest_path),
                "source_unperturbed_structure": source_structure.as_posix(),
                "source_unperturbed_geometry_sha256": geometry_sha256(reference),
                "candidate_structure": structure_path.as_posix(),
            }
        )
        written_records.append(full_record)

    candidate_catalog = config_output / "candidates.jsonl"
    with candidate_catalog.open("w", encoding="utf-8") as handle:
        for record in written_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "artifact_type": "defect_reconstruction_candidate_set",
        "defect_family": family,
        "defect_config_id": config_id,
        "source_defect_manifest": manifest_path.as_posix(),
        "defect_centers": centers,
        "candidate_count": len(written_records),
        "candidate_type_counts": dict(
            sorted(Counter(record["candidate_type"] for record in written_records).items())
        ),
        "recipe": {
            "num_nearest_neighbours": num_neighbors,
            "single_center_distortion_factors": list(DISTORTION_FACTORS),
            "isolated_distort_and_rattle_factors": list(
                ISOLATED_DISTORT_AND_RATTLE_FACTORS
            ),
            "global_rattle_seeds": list(RATTLE_SEEDS[:rattle_seed_count]),
            "rattle_stdev_angstrom": stdev,
            "rattle_d_min_angstrom": d_min,
            "hard_minimum_distance_angstrom": hard_min_distance,
            "maximum_displacement_angstrom": max_displacement,
            "sequential_multi_center_distortions": False,
        },
        "candidate_catalog": candidate_catalog.as_posix(),
    }
    write_json(config_output / "candidate_summary.json", summary)
    return summary


def generate_candidate_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    families: Sequence[str] | None = None,
    config_ids: Sequence[str] | None = None,
    num_neighbors: int = 6,
    rattle_stdev: float | None = None,
    rattle_d_min: float | None = None,
    hard_min_distance: float = 1.2,
    max_displacement: float = 2.0,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use a new directory."
        )
    if num_neighbors < 1:
        raise ValueError("num_neighbors must be positive.")
    if hard_min_distance <= 0 or max_displacement <= 0:
        raise ValueError("Distance and displacement limits must be positive.")

    manifests = find_defect_manifests(
        input_dir, families=families, config_ids=config_ids
    )
    summaries = [
        generate_for_defect(
            manifest_path,
            output_dir,
            num_neighbors=num_neighbors,
            rattle_stdev=rattle_stdev,
            rattle_d_min=rattle_d_min,
            hard_min_distance=hard_min_distance,
            max_displacement=max_displacement,
        )
        for manifest_path in manifests
    ]
    run_summary = {
        "artifact_type": "bi2te3_reconstruction_candidate_dataset",
        "generation_status": "complete_unrelaxed_candidates",
        "source_defect_catalog": input_dir.as_posix(),
        "selected_families": sorted(
            {str(summary["defect_family"]) for summary in summaries}
        ),
        "defect_configuration_count": len(summaries),
        "candidate_count": sum(int(summary["candidate_count"]) for summary in summaries),
        "configuration_summaries": summaries,
        "scope": {
            "shakenbreak_geometry_primitives": True,
            "sequential_multi_center_distortions": False,
            "coordinated_multi_center_distortions": False,
            "relaxations_included": False,
            "dft_labels_included": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "candidate_generation_summary.json", run_summary)
    return run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 10-15 unrelaxed ShakeNBreak reconstruction candidates "
            "for each cataloged Bi2Te3 defect configuration."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--config-ids", nargs="+")
    parser.add_argument("--num-neighbors", type=int, default=6)
    parser.add_argument("--rattle-stdev", type=float)
    parser.add_argument("--rattle-d-min", type=float)
    parser.add_argument("--hard-min-distance", type=float, default=1.2)
    parser.add_argument("--max-displacement", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_candidate_dataset(
        args.input_dir,
        args.output_dir,
        families=args.families,
        config_ids=args.config_ids,
        num_neighbors=args.num_neighbors,
        rattle_stdev=args.rattle_stdev,
        rattle_d_min=args.rattle_d_min,
        hard_min_distance=args.hard_min_distance,
        max_displacement=args.max_displacement,
    )
    print("Michael, reconstruction-candidate generation completed successfully.")
    print(f"defect_configuration_count={summary['defect_configuration_count']}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"selected_families={','.join(summary['selected_families'])}")
    print(f"summary={(args.output_dir.expanduser().resolve() / 'candidate_generation_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
