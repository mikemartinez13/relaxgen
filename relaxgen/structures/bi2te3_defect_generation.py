#!/usr/bin/env python3
"""Auditable pristine-to-unrelaxed-defect generation for Bi2Te3.

The workflow is intentionally staged:

1. Reconstruct the inferred 40-atom pristine parent from the original ASE DB.
2. Prepare (but do not execute) a fixed-cell native QE/BFGS relaxation.
3. Validate a user-produced QE output and freeze a canonical parent.
4. Enumerate and construct symmetry-inequivalent neutral vacancy/antisite
   structures from a controlled defect-family catalog.

Crystallographic data are kept as float64 throughout.  Extended XYZ files are
written with 17 significant digits so float64 coordinates round-trip exactly.
No surrogate relaxation or ShakeNBreak distortion is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.geometry import find_mic
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.units import Bohr, Rydberg
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import linear_sum_assignment

DEFAULT_SOURCE_DB = Path(
    "/scratch/gqs7zr/ai4mat/equiformer_v3/data/all_bi2te3_defects.db"
)
DEFAULT_ANCHOR_STRUCTURE_ID = "struc_185"
EXPECTED_FORMULA_COUNTS = {"Bi": 16, "Te": 24}
EXPECTED_ORBIT_SIGNATURE = (("Bi", 16), ("Te", 8), ("Te", 16))
EXPECTED_SPACEGROUP_NUMBER = 166
DEFAULT_SYMPRECS_ANGSTROM = (1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1)
DEFAULT_DEFECT_FAMILY_CATALOG = Path(__file__).with_name(
    "bi2te3_defect_family_catalog.json"
)
QE_PSEUDOPOTENTIALS = {
    "Bi": "Bi.pbesol-dn-rrkjus.UPF",
    "Te": "Te.pbesol-n-rrkjus_psl.1.0.0.v2.UPF",
}


def precise_qe_input_data(
    *, ecutwfc_ry: float = 60.0, ecutrho_ry: float = 480.0
) -> dict[str, Any]:
    """Return the repository's precise PBEsol QE electronic settings."""

    return {
        "control": {
            "calculation": "scf",
            "restart_mode": "from_scratch",
            "tprnfor": True,
        },
        "system": {
            "ecutwfc": float(ecutwfc_ry),
            "ecutrho": float(ecutrho_ry),
            "occupations": "smearing",
            "smearing": "mv",
            "degauss": 0.001,
        },
        "electrons": {
            "conv_thr": 1.0e-9,
            "electron_maxstep": 200,
            "diagonalization": "david",
            "mixing_beta": 0.3,
        },
    }


@dataclass(frozen=True)
class SiteOrbit:
    label: str
    species: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class DefectAction:
    """One typed operation applied to an unordered set of pristine sites."""

    operation: str
    source_species: str
    count: int
    target_species: str | None = None

    def __post_init__(self) -> None:
        if self.operation not in {"remove", "substitute"}:
            raise ValueError(
                f"Unsupported defect operation {self.operation!r}; "
                "only remove and substitute are allowed in this stage."
            )
        if self.source_species not in EXPECTED_FORMULA_COUNTS:
            raise ValueError(f"Unsupported source species {self.source_species!r}.")
        if int(self.count) < 1:
            raise ValueError("A defect action count must be at least one.")
        if self.operation == "remove" and self.target_species is not None:
            raise ValueError("A remove action cannot define target_species.")
        if self.operation == "substitute":
            if self.target_species not in EXPECTED_FORMULA_COUNTS:
                raise ValueError(
                    f"Unsupported substitution target {self.target_species!r}."
                )
            if self.target_species == self.source_species:
                raise ValueError("A substitution must change the chemical species.")

    @property
    def key(self) -> str:
        if self.operation == "remove":
            return f"remove:{self.source_species}"
        return f"substitute:{self.source_species}->{self.target_species}"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation": self.operation,
            "source_species": self.source_species,
            "count": int(self.count),
        }
        if self.target_species is not None:
            payload["target_species"] = self.target_species
        return payload


@dataclass(frozen=True)
class DefectFamily:
    """A bounded neutral defect family from the reviewed JSON catalog."""

    name: str
    actions: tuple[DefectAction, ...]
    description: str = ""
    charge: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.name):
            raise ValueError(
                f"Invalid defect-family name {self.name!r}; use letters, digits, "
                "and underscores only."
            )
        if not self.actions:
            raise ValueError(f"Defect family {self.name!r} has no actions.")
        keys = [action.key for action in self.actions]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"Defect family {self.name!r} repeats a typed action; combine it "
                "into one action with a larger count."
            )
        if int(self.charge) != 0:
            raise ValueError(
                f"Defect family {self.name!r} has charge {self.charge}; this "
                "generator intentionally supports neutral structures only."
            )

    @property
    def defect_count(self) -> int:
        return sum(int(action.count) for action in self.actions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "charge": int(self.charge),
            "enabled": bool(self.enabled),
            "defect_count": self.defect_count,
            "actions": [action.as_dict() for action in self.actions],
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any, dtype: Any) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def geometry_sha256(atoms: Atoms) -> str:
    digest = hashlib.sha256()
    for array, dtype in (
        (atoms.numbers, np.int64),
        (atoms.positions, np.float64),
        (atoms.cell.array, np.float64),
        (atoms.pbc, np.bool_),
    ):
        contiguous = np.ascontiguousarray(np.asarray(array, dtype=dtype))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def formula_counts(atoms: Atoms) -> dict[str, int]:
    return dict(sorted(Counter(atoms.get_chemical_symbols()).items()))


def require_pristine_composition(atoms: Atoms) -> None:
    counts = formula_counts(atoms)
    if counts != EXPECTED_FORMULA_COUNTS:
        raise ValueError(
            f"Expected pristine composition {EXPECTED_FORMULA_COUNTS}, got {counts}."
        )
    if len(atoms) != 40:
        raise ValueError(f"Expected 40 pristine atoms, got {len(atoms)}.")


def write_extxyz_float64(path: Path, atoms: Atoms) -> None:
    """Write a minimal extended XYZ with float64 round-trip precision."""

    path.parent.mkdir(parents=True, exist_ok=True)
    cell_values = " ".join(f"{value:.17g}" for value in atoms.cell.array.reshape(-1))
    pbc_values = " ".join("T" if value else "F" for value in atoms.pbc)
    lines = [
        str(len(atoms)),
        (
            f'Lattice="{cell_values}" '
            'Properties=species:S:1:pos:R:3 '
            f'pbc="{pbc_values}"'
        ),
    ]
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
        xyz = " ".join(f"{float(value):.17g}" for value in position)
        lines.append(f"{symbol} {xyz}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    roundtrip = ase_read(path, index=0, format="extxyz")
    if not np.array_equal(
        np.asarray(roundtrip.positions, dtype=np.float64),
        np.asarray(atoms.positions, dtype=np.float64),
    ):
        raise RuntimeError(f"Float64 positions did not round-trip exactly through {path}.")
    if not np.array_equal(
        np.asarray(roundtrip.cell.array, dtype=np.float64),
        np.asarray(atoms.cell.array, dtype=np.float64),
    ):
        raise RuntimeError(f"Float64 cell did not round-trip exactly through {path}.")


def read_single_structure(path: Path) -> Atoms:
    frames = ase_read(path, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) != 1:
        raise ValueError(f"Expected exactly one structure in {path}, found {len(frames)}.")
    atoms = frames[0]
    atoms.set_pbc((True, True, True))
    return atoms


def rewrite_qe_geometry_float64(path: Path, atoms: Atoms) -> None:
    """Replace ASE's rounded QE geometry fields with 17-digit float text."""

    lines = path.read_text(encoding="utf-8").splitlines()
    cell_header = next(
        (index for index, line in enumerate(lines) if line.startswith("CELL_PARAMETERS")),
        None,
    )
    position_header = next(
        (index for index, line in enumerate(lines) if line.startswith("ATOMIC_POSITIONS")),
        None,
    )
    if cell_header is None or position_header is None:
        raise RuntimeError(f"ASE QE input lacks an explicit cell or positions block: {path}")

    lines[cell_header + 1 : cell_header + 4] = [
        " ".join(f"{float(value):.17g}" for value in vector)
        for vector in np.asarray(atoms.cell.array, dtype=np.float64)
    ]
    lines[position_header + 1 : position_header + 1 + len(atoms)] = [
        f"{symbol} "
        + " ".join(f"{float(value):.17g}" for value in position)
        for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def atoms_to_pymatgen(atoms: Atoms) -> Structure:
    return Structure(
        lattice=np.asarray(atoms.cell.array, dtype=np.float64),
        species=atoms.get_chemical_symbols(),
        coords=np.asarray(atoms.get_scaled_positions(wrap=True), dtype=np.float64),
        coords_are_cartesian=False,
        to_unit_cell=True,
    )


def _site_orbits(atoms: Atoms, *, symprec: float) -> tuple[int, list[SiteOrbit]]:
    structure = atoms_to_pymatgen(atoms)
    analyzer = SpacegroupAnalyzer(structure, symprec=float(symprec), angle_tolerance=5)
    number = int(analyzer.get_space_group_number())
    symmetrized = analyzer.get_symmetrized_structure()

    raw: list[tuple[str, tuple[int, ...]]] = []
    for indices in symmetrized.equivalent_indices:
        ordered = tuple(sorted(int(index) for index in indices))
        species = str(structure[ordered[0]].specie)
        if any(str(structure[index].specie) != species for index in ordered):
            raise ValueError("A symmetry-equivalent site group mixes species.")
        raw.append((species, ordered))

    signature = tuple(sorted((species, len(indices)) for species, indices in raw))
    if signature != EXPECTED_ORBIT_SIGNATURE:
        raise ValueError(
            f"Unexpected pristine site-orbit signature at symprec={symprec}: "
            f"{signature}; expected {EXPECTED_ORBIT_SIGNATURE}."
        )

    orbits: list[SiteOrbit] = []
    for species, indices in raw:
        if species == "Bi" and len(indices) == 16:
            label = "Bi"
        elif species == "Te" and len(indices) == 16:
            label = "TeA"
        elif species == "Te" and len(indices) == 8:
            label = "TeB"
        else:  # guarded by the signature check above
            raise AssertionError((species, len(indices)))
        orbits.append(SiteOrbit(label=label, species=species, indices=indices))
    orbits.sort(key=lambda orbit: orbit.label)
    return number, orbits


def build_site_records(
    identity_atoms: Atoms,
    coordinate_atoms: Atoms | None = None,
    *,
    symprec: float = 1.0e-3,
) -> list[dict[str, Any]]:
    """Assign stable site IDs from identity_atoms and coordinates from another cell."""

    coordinate_atoms = coordinate_atoms if coordinate_atoms is not None else identity_atoms
    if len(identity_atoms) != len(coordinate_atoms):
        raise ValueError("Identity and coordinate structures have different atom counts.")
    if identity_atoms.get_chemical_symbols() != coordinate_atoms.get_chemical_symbols():
        raise ValueError("Identity and coordinate structures have different ordered species.")

    number, orbits = _site_orbits(identity_atoms, symprec=symprec)
    if number != EXPECTED_SPACEGROUP_NUMBER:
        raise ValueError(
            f"Expected space group {EXPECTED_SPACEGROUP_NUMBER}, got {number}."
        )

    by_index: dict[int, tuple[str, str]] = {}
    for orbit in orbits:
        for ordinal, index in enumerate(orbit.indices):
            by_index[index] = (orbit.label, f"{orbit.label}_{ordinal:03d}")

    identity_fractional = identity_atoms.get_scaled_positions(wrap=True)
    canonical_fractional = coordinate_atoms.get_scaled_positions(wrap=True)
    records: list[dict[str, Any]] = []
    for index in range(len(identity_atoms)):
        orbit_label, site_id = by_index[index]
        records.append(
            {
                "bulk_index": index,
                "bulk_site_id": site_id,
                "species": identity_atoms[index].symbol,
                "site_orbit": orbit_label,
                "identity_fractional_coordinates": identity_fractional[index].tolist(),
                "identity_cartesian_coordinates_angstrom": identity_atoms.positions[
                    index
                ].tolist(),
                "canonical_fractional_coordinates": canonical_fractional[index].tolist(),
                "canonical_cartesian_coordinates_angstrom": coordinate_atoms.positions[
                    index
                ].tolist(),
            }
        )
    return records


def _mic_distance_matrix(left: Atoms, right: Atoms) -> np.ndarray:
    if not np.allclose(left.cell.array, right.cell.array, rtol=0.0, atol=1.0e-6):
        raise ValueError("Cannot compute fixed-cell mapping for different cells.")
    displacements = (
        np.asarray(right.positions, dtype=np.float64)[None, :, :]
        - np.asarray(left.positions, dtype=np.float64)[:, None, :]
    )
    _, distances = find_mic(
        displacements.reshape(-1, 3),
        np.asarray(left.cell.array, dtype=np.float64),
        pbc=True,
    )
    return np.asarray(distances, dtype=np.float64).reshape(len(left), len(right))


def reconstruct_inferred_pristine(
    source_db: Path,
    *,
    anchor_structure_id: str = DEFAULT_ANCHOR_STRUCTURE_ID,
    cell_tolerance_angstrom: float = 1.0e-7,
    site_tolerance_angstrom: float = 1.0e-3,
) -> tuple[Atoms, dict[str, Any]]:
    source_db = source_db.expanduser().resolve()
    if not source_db.is_file():
        raise FileNotFoundError(source_db)

    database = connect(source_db)
    first_rows: dict[str, Any] = {}
    for row in database.select():
        structure_id = str(row.structure_id)
        previous = first_rows.get(structure_id)
        if previous is None or int(row.frame_idx) < int(previous.frame_idx):
            first_rows[structure_id] = row
    if anchor_structure_id not in first_rows:
        raise ValueError(f"Anchor {anchor_structure_id!r} is absent from {source_db}.")

    anchor_row = first_rows[anchor_structure_id]
    anchor = anchor_row.toatoms()
    if len(anchor) != 40:
        raise ValueError(
            f"Anchor {anchor_structure_id} must contain all 40 sites, got {len(anchor)}."
        )
    anchor.set_pbc((True, True, True))

    cohort: list[tuple[str, Any, Atoms]] = []
    for structure_id, row in sorted(first_rows.items()):
        atoms = row.toatoms()
        if np.allclose(
            atoms.cell.array,
            anchor.cell.array,
            rtol=0.0,
            atol=float(cell_tolerance_angstrom),
        ):
            atoms.set_pbc((True, True, True))
            cohort.append((structure_id, row, atoms))
    if not cohort:
        raise ValueError("No structures share the anchor cell.")

    votes: list[Counter[str]] = [Counter() for _ in range(len(anchor))]
    max_site_residual = 0.0
    cohort_rows: list[dict[str, Any]] = []
    for structure_id, row, atoms in cohort:
        distances = _mic_distance_matrix(atoms, anchor)
        observed_indices, anchor_indices = linear_sum_assignment(distances)
        residuals = distances[observed_indices, anchor_indices]
        if residuals.size and float(residuals.max()) > float(site_tolerance_angstrom):
            raise ValueError(
                f"{structure_id} is not an exact-site derivative of {anchor_structure_id}: "
                f"max site residual {float(residuals.max()):.9g} Å exceeds "
                f"{site_tolerance_angstrom:.9g} Å."
            )
        max_site_residual = max(
            max_site_residual, float(residuals.max()) if residuals.size else 0.0
        )
        symbols = atoms.get_chemical_symbols()
        for observed_index, anchor_index in zip(observed_indices, anchor_indices):
            votes[int(anchor_index)][symbols[int(observed_index)]] += 1
        cohort_rows.append(
            {
                "structure_id": structure_id,
                "source_db_row_id": int(row.id),
                "source_file": str(row.source_file),
                "source_relpath": str(row.data.get("source_relpath", "")),
                "formula": str(row.formula),
                "natoms": len(atoms),
                "max_site_residual_angstrom": (
                    float(residuals.max()) if residuals.size else 0.0
                ),
            }
        )

    inferred_symbols: list[str] = []
    vote_records: list[dict[str, Any]] = []
    for index, counter in enumerate(votes):
        if not counter:
            raise ValueError(f"No species votes were collected for parent site {index}.")
        ordered_votes = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered_votes) > 1 and ordered_votes[0][1] == ordered_votes[1][1]:
            raise ValueError(f"Tied species vote at parent site {index}: {counter}.")
        inferred_symbols.append(ordered_votes[0][0])
        vote_records.append(
            {
                "bulk_index": index,
                "selected_species": ordered_votes[0][0],
                "species_votes": dict(sorted(counter.items())),
            }
        )

    inferred = Atoms(
        symbols=inferred_symbols,
        positions=np.asarray(anchor.positions, dtype=np.float64).copy(),
        cell=np.asarray(anchor.cell.array, dtype=np.float64).copy(),
        pbc=(True, True, True),
    )
    require_pristine_composition(inferred)
    spacegroup_number, orbits = _site_orbits(inferred, symprec=1.0e-3)
    if spacegroup_number != EXPECTED_SPACEGROUP_NUMBER:
        raise ValueError(
            f"Reconstructed parent has space group {spacegroup_number}, expected 166."
        )

    manifest = {
        "artifact_type": "inferred_pristine_bi16te24",
        "numeric_contract": {
            "geometry_dtype": "float64",
            "extxyz_float_significant_digits": 17,
        },
        "source_database": source_db.as_posix(),
        "source_database_sha256": file_sha256(source_db),
        "source_structure_count": len(first_rows),
        "anchor": {
            "structure_id": anchor_structure_id,
            "source_db_row_id": int(anchor_row.id),
            "source_file": str(anchor_row.source_file),
            "source_relpath": str(anchor_row.data.get("source_relpath", "")),
        },
        "cohort_size": len(cohort),
        "cohort": cohort_rows,
        "cell_tolerance_angstrom": float(cell_tolerance_angstrom),
        "site_tolerance_angstrom": float(site_tolerance_angstrom),
        "maximum_cohort_site_residual_angstrom": max_site_residual,
        "formula_counts": formula_counts(inferred),
        "natoms": len(inferred),
        "cell_angstrom": inferred.cell.array.tolist(),
        "cell_sha256": array_sha256(inferred.cell.array, np.float64),
        "ordered_atomic_numbers_sha256": array_sha256(inferred.numbers, np.int64),
        "geometry_sha256": geometry_sha256(inferred),
        "spacegroup_number": spacegroup_number,
        "spacegroup_symbol": "R-3m",
        "site_orbits": [
            {
                "label": orbit.label,
                "species": orbit.species,
                "multiplicity": len(orbit.indices),
                "bulk_indices": list(orbit.indices),
            }
            for orbit in orbits
        ],
        "species_vote_audit": vote_records,
    }
    return inferred, manifest


def export_inferred_pristine(
    source_db: Path,
    output_dir: Path,
    *,
    anchor_structure_id: str = DEFAULT_ANCHOR_STRUCTURE_ID,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    extxyz_path = output_dir / "pristine_bi16te24_inferred.extxyz"
    cif_path = output_dir / "pristine_bi16te24_inferred.cif"
    manifest_path = output_dir / "inferred_manifest.json"
    for path in (extxyz_path, cif_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")

    atoms, manifest = reconstruct_inferred_pristine(
        source_db, anchor_structure_id=anchor_structure_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_extxyz_float64(extxyz_path, atoms)
    ase_write(cif_path, atoms, format="cif")
    manifest["artifacts"] = {
        "extxyz": extxyz_path.as_posix(),
        "extxyz_sha256": file_sha256(extxyz_path),
        "cif": cif_path.as_posix(),
        "cif_sha256": file_sha256(cif_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def prepare_qe_relaxation(
    inferred_path: Path,
    output_dir: Path,
    *,
    pseudo_dir: Path,
    qe_outdir: Path,
    kpts: tuple[int, int, int] = (6, 6, 6),
    ecutwfc_ry: float = 60.0,
    ecutrho_ry: float = 480.0,
    force_max_ev_per_angstrom: float = 0.01,
    max_steps: int = 100,
    overwrite: bool = False,
) -> dict[str, Any]:
    inferred_path = inferred_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    pseudo_dir = pseudo_dir.expanduser().resolve()
    qe_outdir = qe_outdir.expanduser().resolve()
    if not inferred_path.is_file():
        raise FileNotFoundError(inferred_path)
    for filename in QE_PSEUDOPOTENTIALS.values():
        candidate = pseudo_dir / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing QE pseudopotential: {candidate}")

    pwi_path = output_dir / "pristine_relax.pwi"
    manifest_path = output_dir / "qe_prepare_manifest.json"
    for path in (pwi_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")

    atoms = read_single_structure(inferred_path)
    require_pristine_composition(atoms)
    output_dir.mkdir(parents=True, exist_ok=True)
    qe_outdir.mkdir(parents=True, exist_ok=True)

    input_data = precise_qe_input_data(
        ecutwfc_ry=float(ecutwfc_ry),
        ecutrho_ry=float(ecutrho_ry),
    )
    input_data["control"].update(
        calculation="relax",
        restart_mode="from_scratch",
        prefix="bi2te3_pristine",
        outdir=qe_outdir.as_posix(),
        pseudo_dir=pseudo_dir.as_posix(),
        tprnfor=True,
        tstress=True,
        nstep=int(max_steps),
        forc_conv_thr=float(force_max_ev_per_angstrom) / (Rydberg / Bohr),
    )
    input_data["ions"] = {"ion_dynamics": "bfgs"}
    ase_write(
        pwi_path,
        atoms,
        format="espresso-in",
        input_data=input_data,
        pseudopotentials=QE_PSEUDOPOTENTIALS,
        kpts=tuple(int(value) for value in kpts),
    )
    rewrite_qe_geometry_float64(pwi_path, atoms)

    manifest = {
        "artifact_type": "pristine_bi16te24_qe_relaxation_input",
        "execution_status": "prepared_not_executed",
        "inferred_structure": inferred_path.as_posix(),
        "inferred_structure_file_sha256": file_sha256(inferred_path),
        "inferred_geometry_sha256": geometry_sha256(atoms),
        "qe_input": pwi_path.as_posix(),
        "qe_input_sha256": file_sha256(pwi_path),
        "qe_outdir": qe_outdir.as_posix(),
        "protocol": {
            "calculation": "relax",
            "ion_dynamics": "bfgs",
            "fixed_cell": True,
            "calculation_mode": "precise",
            "pseudopotentials": dict(QE_PSEUDOPOTENTIALS),
            "pseudo_dir": pseudo_dir.as_posix(),
            "kpts": list(kpts),
            "ecutwfc_ry": float(ecutwfc_ry),
            "ecutrho_ry": float(ecutrho_ry),
            "force_max_ev_per_angstrom": float(force_max_ev_per_angstrom),
            "forc_conv_thr_ry_per_bohr": float(
                force_max_ev_per_angstrom / (Rydberg / Bohr)
            ),
            "max_steps": int(max_steps),
            "geometry_dtype": "float64",
            "qe_geometry_float_significant_digits": 17,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def _species_constrained_mapping(
    reference: Atoms, candidate: Atoms
) -> tuple[np.ndarray, np.ndarray]:
    if len(reference) != len(candidate):
        raise ValueError(
            f"Atom count changed from {len(reference)} to {len(candidate)} during QE."
        )
    if formula_counts(reference) != formula_counts(candidate):
        raise ValueError(
            "Composition changed during QE: "
            f"{formula_counts(reference)} -> {formula_counts(candidate)}."
        )

    distances = _mic_distance_matrix(reference, candidate)
    reference_symbols = np.asarray(reference.get_chemical_symbols(), dtype=object)
    candidate_symbols = np.asarray(candidate.get_chemical_symbols(), dtype=object)
    mismatch = reference_symbols[:, None] != candidate_symbols[None, :]
    cost = distances.copy()
    cost[mismatch] = 1.0e12
    reference_indices, candidate_indices = linear_sum_assignment(cost)
    if not np.array_equal(reference_indices, np.arange(len(reference), dtype=np.int64)):
        raise RuntimeError("Hungarian mapping did not return every reference site.")
    selected = cost[reference_indices, candidate_indices]
    if np.any(selected >= 1.0e11):
        raise ValueError("No complete species-preserving mapping exists for QE output.")
    return np.asarray(candidate_indices, dtype=np.int64), np.asarray(selected, dtype=np.float64)


def validate_relaxed_atoms(
    inferred: Atoms,
    relaxed_raw: Atoms,
    *,
    cell_tolerance_angstrom: float = 1.0e-6,
    max_rmsd_angstrom: float = 0.10,
    max_displacement_angstrom: float = 0.25,
    symprecs_angstrom: Sequence[float] = DEFAULT_SYMPRECS_ANGSTROM,
) -> tuple[Atoms, dict[str, Any]]:
    require_pristine_composition(inferred)
    require_pristine_composition(relaxed_raw)
    inferred_cell = np.asarray(inferred.cell.array, dtype=np.float64)
    raw_cell = np.asarray(relaxed_raw.cell.array, dtype=np.float64)
    cell_delta = raw_cell - inferred_cell
    max_cell_delta = float(np.max(np.abs(cell_delta)))
    if max_cell_delta > float(cell_tolerance_angstrom):
        raise ValueError(
            f"Fixed-cell QE output changed a cell component by {max_cell_delta:.9g} Å; "
            f"tolerance is {cell_tolerance_angstrom:.9g} Å."
        )

    candidate_indices, displacements = _species_constrained_mapping(inferred, relaxed_raw)
    rmsd = float(np.sqrt(np.mean(displacements * displacements)))
    max_displacement = float(displacements.max()) if displacements.size else 0.0
    if rmsd > float(max_rmsd_angstrom):
        raise ValueError(
            f"QE-relaxed pristine RMSD {rmsd:.9g} Å exceeds {max_rmsd_angstrom:.9g} Å."
        )
    if max_displacement > float(max_displacement_angstrom):
        raise ValueError(
            "QE-relaxed pristine maximum mapped displacement "
            f"{max_displacement:.9g} Å exceeds {max_displacement_angstrom:.9g} Å."
        )

    canonical = relaxed_raw[candidate_indices].copy()
    # The QE output may serialize an identical fixed cell with slightly different
    # decimal rounding.  Preserve relaxed Cartesian coordinates but freeze the
    # canonical cell to the exact float64 reference cell after tolerance checks.
    canonical.set_cell(inferred_cell, scale_atoms=False)
    canonical.set_pbc(inferred.pbc)
    if canonical.get_chemical_symbols() != inferred.get_chemical_symbols():
        raise RuntimeError("Canonical reordering did not restore inferred site ordering.")

    symmetry_trials: list[dict[str, Any]] = []
    accepted_symprec: float | None = None
    for symprec in symprecs_angstrom:
        try:
            number, orbits = _site_orbits(canonical, symprec=float(symprec))
            signature = tuple(
                sorted((orbit.species, len(orbit.indices)) for orbit in orbits)
            )
            passed = number == EXPECTED_SPACEGROUP_NUMBER
            trial = {
                "symprec_angstrom": float(symprec),
                "spacegroup_number": number,
                "spacegroup_symbol": "R-3m" if number == 166 else None,
                "site_orbit_signature": [list(value) for value in signature],
                "passed": passed,
            }
            if passed and accepted_symprec is None:
                accepted_symprec = float(symprec)
        except ValueError as exc:
            trial = {
                "symprec_angstrom": float(symprec),
                "passed": False,
                "error": str(exc),
            }
        symmetry_trials.append(trial)
    if accepted_symprec is None:
        raise ValueError(
            "QE-relaxed pristine did not recover R-3m with the expected "
            f"site-orbit signature at symprecs {list(symprecs_angstrom)}."
        )

    identity_order_preserved = np.array_equal(
        candidate_indices, np.arange(len(inferred), dtype=np.int64)
    )
    metrics = {
        "cell_component_max_abs_delta_angstrom": max_cell_delta,
        "cell_frobenius_delta_angstrom": float(np.linalg.norm(cell_delta)),
        "cell_volume_relative_delta": float(
            abs(relaxed_raw.get_volume() - inferred.get_volume()) / inferred.get_volume()
        ),
        "mapped_rmsd_angstrom": rmsd,
        "mapped_max_displacement_angstrom": max_displacement,
        "mapped_mean_displacement_angstrom": float(displacements.mean()),
        "qe_output_order_preserved": bool(identity_order_preserved),
        "reference_to_qe_output_indices": candidate_indices.tolist(),
        "accepted_symprec_angstrom": accepted_symprec,
        "symmetry_trials": symmetry_trials,
    }
    return canonical, metrics


def _last_qe_frame_with_forces(qe_output: Path) -> tuple[Atoms, float, np.ndarray, int]:
    frames = ase_read(qe_output, index=":", format="espresso-out")
    if not isinstance(frames, list):
        frames = [frames]
    for index in range(len(frames) - 1, -1, -1):
        frame = frames[index]
        try:
            energy = float(frame.get_potential_energy())
            forces = np.asarray(frame.get_forces(), dtype=np.float64)
        except Exception:
            continue
        if forces.shape == (len(frame), 3) and np.isfinite(forces).all():
            return frame, energy, forces, index
    raise ValueError(f"No QE output frame with readable energy and forces: {qe_output}")


def validate_qe_relaxation(
    inferred_path: Path,
    qe_output: Path,
    output_dir: Path,
    *,
    force_max_ev_per_angstrom: float = 0.01,
    cell_tolerance_angstrom: float = 1.0e-6,
    max_rmsd_angstrom: float = 0.10,
    max_displacement_angstrom: float = 0.25,
    overwrite: bool = False,
) -> dict[str, Any]:
    inferred_path = inferred_path.expanduser().resolve()
    qe_output = qe_output.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not qe_output.is_file():
        raise FileNotFoundError(qe_output)

    canonical_path = output_dir / "pristine_bi16te24_qe_relaxed.extxyz"
    raw_path = output_dir / "pristine_bi16te24_qe_relaxed_raw.extxyz"
    manifest_path = output_dir / "qe_validation_manifest.json"
    site_manifest_path = output_dir / "bulk_sites.json"
    for path in (canonical_path, raw_path, manifest_path, site_manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")

    output_text = qe_output.read_text(encoding="utf-8", errors="ignore")
    job_done = "JOB DONE" in output_text
    lowered = output_text.lower()
    ionic_converged = (
        "bfgs converged in" in lowered
        or "end of bfgs geometry optimization" in lowered
    )
    if not job_done:
        raise ValueError(f"QE output does not contain JOB DONE: {qe_output}")
    if not ionic_converged:
        raise ValueError(f"QE output does not report BFGS convergence: {qe_output}")

    inferred = read_single_structure(inferred_path)
    relaxed_raw, energy_ev, forces, frame_index = _last_qe_frame_with_forces(qe_output)
    force_norms = np.linalg.norm(forces, axis=1)
    final_max_force = float(force_norms.max()) if force_norms.size else 0.0
    if final_max_force > float(force_max_ev_per_angstrom):
        raise ValueError(
            f"QE final maximum force {final_max_force:.9g} eV/Å exceeds "
            f"{force_max_ev_per_angstrom:.9g} eV/Å."
        )

    canonical, metrics = validate_relaxed_atoms(
        inferred,
        relaxed_raw,
        cell_tolerance_angstrom=cell_tolerance_angstrom,
        max_rmsd_angstrom=max_rmsd_angstrom,
        max_displacement_angstrom=max_displacement_angstrom,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_extxyz_float64(raw_path, relaxed_raw)
    write_extxyz_float64(canonical_path, canonical)

    site_records = build_site_records(
        inferred,
        canonical,
        symprec=1.0e-3,
    )
    site_manifest = {
        "artifact_type": "canonical_bi16te24_bulk_sites",
        "parent_validation_status": "validated_qe_relaxed",
        "site_identity_source": "inferred_pristine",
        "coordinate_source": "validated_qe_relaxed_pristine",
        "inferred_structure": inferred_path.as_posix(),
        "inferred_geometry_sha256": geometry_sha256(inferred),
        "canonical_structure": canonical_path.as_posix(),
        "canonical_geometry_sha256": geometry_sha256(canonical),
        "canonical_cell_sha256": array_sha256(canonical.cell.array, np.float64),
        "sites": site_records,
    }
    write_json(site_manifest_path, site_manifest)

    manifest = {
        "artifact_type": "validated_pristine_bi16te24_qe_relaxation",
        "validation_status": "passed",
        "qe_output": qe_output.as_posix(),
        "qe_output_sha256": file_sha256(qe_output),
        "qe_job_done": job_done,
        "qe_ionic_converged": ionic_converged,
        "qe_final_frame_index": int(frame_index),
        "final_energy_ev": energy_ev,
        "final_max_force_ev_per_angstrom": final_max_force,
        "final_mean_force_ev_per_angstrom": (
            float(force_norms.mean()) if force_norms.size else 0.0
        ),
        "force_max_threshold_ev_per_angstrom": float(force_max_ev_per_angstrom),
        "inferred_structure": inferred_path.as_posix(),
        "inferred_geometry_sha256": geometry_sha256(inferred),
        "raw_relaxed_structure": raw_path.as_posix(),
        "canonical_structure": canonical_path.as_posix(),
        "canonical_geometry_sha256": geometry_sha256(canonical),
        "canonical_cell_frozen_to_inferred_float64_values": True,
        "structural_equivalence_metrics": metrics,
        "bulk_site_manifest": site_manifest_path.as_posix(),
    }
    write_json(manifest_path, manifest)
    return manifest


def _operation_permutations(
    identity_atoms: Atoms,
    *,
    symprec: float,
    site_match_tolerance_angstrom: float = 1.0e-3,
) -> list[tuple[int, ...]]:
    structure = atoms_to_pymatgen(identity_atoms)
    analyzer = SpacegroupAnalyzer(structure, symprec=float(symprec), angle_tolerance=5)
    if int(analyzer.get_space_group_number()) != EXPECTED_SPACEGROUP_NUMBER:
        raise ValueError("Symmetry reference is not R-3m (No. 166).")
    operations = analyzer.get_symmetry_operations(cartesian=False)
    coordinates = np.asarray(structure.frac_coords, dtype=np.float64)
    cell = np.asarray(identity_atoms.cell.array, dtype=np.float64)
    permutations: set[tuple[int, ...]] = set()
    for operation in operations:
        transformed = np.mod(
            np.asarray([operation.operate(position) for position in coordinates]), 1.0
        )
        permutation: list[int] = []
        for position in transformed:
            fractional_delta = np.mod(coordinates - position + 0.5, 1.0) - 0.5
            distances = np.linalg.norm(fractional_delta @ cell, axis=1)
            target = int(np.argmin(distances))
            if float(distances[target]) > float(site_match_tolerance_angstrom):
                raise ValueError(
                    "A symmetry operation did not map onto the reference sites: "
                    f"nearest residual {float(distances[target]):.9g} Å."
                )
            permutation.append(target)
        if len(set(permutation)) != len(identity_atoms):
            raise ValueError("A symmetry operation produced a non-bijective site mapping.")
        permutations.add(tuple(permutation))
    return sorted(permutations)


def load_defect_family_catalog(
    path: Path = DEFAULT_DEFECT_FAMILY_CATALOG,
    *,
    selected_families: Sequence[str] | None = None,
) -> tuple[list[DefectFamily], dict[str, Any]]:
    """Load and validate the controlled neutral vacancy/antisite catalog."""

    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "bi2te3_defect_family_catalog":
        raise ValueError(f"Unexpected defect catalog type in {path}.")
    maximum_defect_count = int(payload.get("maximum_defect_count", 3))
    if maximum_defect_count < 1:
        raise ValueError("maximum_defect_count must be positive.")

    raw_families = payload.get("families")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError(f"Defect catalog {path} has no family list.")
    families: list[DefectFamily] = []
    seen_names: set[str] = set()
    for raw_family in raw_families:
        if not isinstance(raw_family, dict):
            raise ValueError("Every catalog family must be a JSON object.")
        raw_actions = raw_family.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError(f"Family {raw_family.get('name')!r} has no actions.")
        actions = tuple(
            DefectAction(
                operation=str(raw_action["operation"]),
                source_species=str(raw_action["source_species"]),
                count=int(raw_action.get("count", 1)),
                target_species=(
                    str(raw_action["target_species"])
                    if raw_action.get("target_species") is not None
                    else None
                ),
            )
            for raw_action in raw_actions
        )
        family = DefectFamily(
            name=str(raw_family["name"]),
            description=str(raw_family.get("description", "")),
            actions=actions,
            charge=int(raw_family.get("charge", 0)),
            enabled=bool(raw_family.get("enabled", True)),
        )
        if family.name in seen_names:
            raise ValueError(f"Duplicate defect family {family.name!r} in {path}.")
        if family.defect_count > maximum_defect_count:
            raise ValueError(
                f"Family {family.name!r} has {family.defect_count} actions, exceeding "
                f"the reviewed catalog limit of {maximum_defect_count}."
            )
        seen_names.add(family.name)
        families.append(family)

    requested = set(selected_families or ())
    unknown = requested - seen_names
    if unknown:
        raise ValueError(f"Unknown requested defect families: {sorted(unknown)}.")
    if requested:
        selected = [family for family in families if family.name in requested]
    else:
        selected = [family for family in families if family.enabled]
    if not selected:
        raise ValueError("No defect families were selected for generation.")
    return selected, payload


def _typed_configuration_key(
    configuration: tuple[tuple[str, tuple[int, ...]], ...],
    record_by_index: dict[int, dict[str, Any]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (
            action_key,
            tuple(
                sorted(
                    str(record_by_index[index]["bulk_site_id"])
                    for index in indices
                )
            ),
        )
        for action_key, indices in configuration
    )


def enumerate_defect_family(
    identity_atoms: Atoms,
    canonical_atoms: Atoms,
    site_records: Sequence[dict[str, Any]],
    family: DefectFamily,
    *,
    symprec: float = 1.0e-3,
    permutations: Sequence[tuple[int, ...]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate one typed defect family under the ideal-parent symmetry group."""

    if len(site_records) != len(identity_atoms):
        raise ValueError("Site manifest length does not match the parent structure.")
    if len(identity_atoms) != len(canonical_atoms):
        raise ValueError("Identity and canonical parent atom counts differ.")
    if identity_atoms.get_chemical_symbols() != canonical_atoms.get_chemical_symbols():
        raise ValueError("Identity and canonical ordered species differ.")

    symmetry_permutations = list(
        permutations
        if permutations is not None
        else _operation_permutations(identity_atoms, symprec=symprec)
    )
    if not symmetry_permutations:
        raise ValueError("No symmetry permutations were supplied or discovered.")

    record_by_index = {int(record["bulk_index"]): record for record in site_records}
    if set(record_by_index) != set(range(len(identity_atoms))):
        raise ValueError("Site manifest bulk indices are incomplete or duplicated.")
    for index, symbol in enumerate(identity_atoms.get_chemical_symbols()):
        if str(record_by_index[index]["species"]) != symbol:
            raise ValueError(f"Site manifest species mismatch at bulk index {index}.")

    actions = tuple(sorted(family.actions, key=lambda action: action.key))
    action_by_key = {action.key: action for action in actions}
    action_choices: list[list[tuple[int, ...]]] = []
    eligible_site_counts: dict[str, int] = {}
    for action in actions:
        eligible = tuple(
            index
            for index in range(len(identity_atoms))
            if str(record_by_index[index]["species"]) == action.source_species
        )
        if action.count > len(eligible):
            raise ValueError(
                f"Family {family.name!r} requests {action.count} {action.source_species} "
                f"sites but only {len(eligible)} exist."
            )
        eligible_site_counts[action.key] = len(eligible)
        action_choices.append(list(combinations(eligible, action.count)))

    raw_configurations: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
    for selections in product(*action_choices):
        flat_indices = [index for selection in selections for index in selection]
        if len(set(flat_indices)) != len(flat_indices):
            continue
        raw_configurations.add(
            tuple(
                (action.key, tuple(sorted(int(index) for index in selection)))
                for action, selection in zip(actions, selections)
            )
        )
    if not raw_configurations:
        raise ValueError(f"Family {family.name!r} produced no raw configurations.")

    def transform(
        configuration: tuple[tuple[str, tuple[int, ...]], ...],
        permutation: tuple[int, ...],
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(
            (
                action_key,
                tuple(sorted(int(permutation[index]) for index in indices)),
            )
            for action_key, indices in configuration
        )

    canonical_fractional = np.asarray(
        canonical_atoms.get_scaled_positions(wrap=True), dtype=np.float64
    )
    unseen = set(raw_configurations)
    orbit_records: list[dict[str, Any]] = []
    while unseen:
        seed = min(unseen)
        orbit = {
            transform(seed, tuple(permutation))
            for permutation in symmetry_permutations
        }
        if not orbit <= raw_configurations:
            raise ValueError(
                f"A symmetry orbit for {family.name} mapped outside its typed site pools."
            )
        unseen -= orbit
        representative = min(
            orbit,
            key=lambda configuration: _typed_configuration_key(
                configuration, record_by_index
            ),
        )

        typed_actions: list[dict[str, Any]] = []
        selected_sites: list[dict[str, Any]] = []
        for action_key, indices in representative:
            action = action_by_key[action_key]
            ordered_indices = sorted(
                indices,
                key=lambda index: str(record_by_index[index]["bulk_site_id"]),
            )
            action_record = {
                "action_key": action.key,
                **action.as_dict(),
                "bulk_indices": [int(index) for index in ordered_indices],
                "bulk_site_ids": [
                    str(record_by_index[index]["bulk_site_id"])
                    for index in ordered_indices
                ],
                "site_orbits": [
                    str(record_by_index[index]["site_orbit"])
                    for index in ordered_indices
                ],
                "canonical_fractional_coordinates": [
                    canonical_fractional[index].tolist() for index in ordered_indices
                ],
            }
            typed_actions.append(action_record)
            selected_sites.extend(
                {
                    "action_key": action.key,
                    "bulk_index": int(index),
                    "bulk_site_id": str(record_by_index[index]["bulk_site_id"]),
                    "site_orbit": str(record_by_index[index]["site_orbit"]),
                }
                for index in ordered_indices
            )

        selected_sites.sort(key=lambda item: (item["action_key"], item["bulk_site_id"]))
        pair_distances: list[dict[str, Any]] = []
        for left, right in combinations(selected_sites, 2):
            pair_distances.append(
                {
                    "left_action_key": left["action_key"],
                    "left_bulk_site_id": left["bulk_site_id"],
                    "right_action_key": right["action_key"],
                    "right_bulk_site_id": right["bulk_site_id"],
                    "distance_angstrom": float(
                        canonical_atoms.get_distance(
                            int(left["bulk_index"]),
                            int(right["bulk_index"]),
                            mic=True,
                        )
                    ),
                }
            )
        pair_distances.sort(
            key=lambda item: (
                float(item["distance_angstrom"]),
                item["left_action_key"],
                item["left_bulk_site_id"],
                item["right_action_key"],
                item["right_bulk_site_id"],
            )
        )

        raw_members = [
            [
                {
                    "action_key": action_key,
                    "bulk_site_ids": list(site_ids),
                }
                for action_key, site_ids in _typed_configuration_key(
                    member, record_by_index
                )
            ]
            for member in sorted(
                orbit,
                key=lambda configuration: _typed_configuration_key(
                    configuration, record_by_index
                ),
            )
        ]
        orbit_records.append(
            {
                "defect_family": family.name,
                "description": family.description,
                "charge": int(family.charge),
                "defect_count": family.defect_count,
                "typed_actions": typed_actions,
                "selected_bulk_site_ids": [
                    item["bulk_site_id"] for item in selected_sites
                ],
                "selected_bulk_indices": [
                    int(item["bulk_index"]) for item in selected_sites
                ],
                "site_orbit_signature": [
                    {
                        "action_key": action_record["action_key"],
                        "site_orbits": sorted(action_record["site_orbits"]),
                    }
                    for action_record in typed_actions
                ],
                "symmetry_orbit_multiplicity": len(orbit),
                "raw_member_typed_site_ids": raw_members,
                "canonical_mic_site_pair_distances_angstrom": pair_distances,
                "sorted_mic_pair_distances_angstrom": sorted(
                    float(item["distance_angstrom"]) for item in pair_distances
                ),
            }
        )

    orbit_records.sort(
        key=lambda record: (
            tuple(record["sorted_mic_pair_distances_angstrom"]),
            tuple(
                (
                    action["action_key"],
                    tuple(action["bulk_site_ids"]),
                )
                for action in record["typed_actions"]
            ),
        )
    )
    for ordinal, record in enumerate(orbit_records, 1):
        record["defect_config_id"] = f"{family.name}_{ordinal:03d}"

    category_counts = Counter(
        "|".join(
            f"{entry['action_key']}[{','.join(entry['site_orbits'])}]"
            for entry in record["site_orbit_signature"]
        )
        for record in orbit_records
    )
    summary = {
        "defect_family": family.name,
        "description": family.description,
        "charge": int(family.charge),
        "defect_count": family.defect_count,
        "typed_action_specification": [action.as_dict() for action in actions],
        "symmetry_operation_count": len(symmetry_permutations),
        "eligible_site_counts": eligible_site_counts,
        "raw_configuration_count": len(raw_configurations),
        "inequivalent_configuration_count": len(orbit_records),
        "symmetry_orbit_multiplicity_sum": sum(
            int(record["symmetry_orbit_multiplicity"]) for record in orbit_records
        ),
        "category_counts": dict(sorted(category_counts.items())),
    }
    if (
        summary["symmetry_orbit_multiplicity_sum"]
        != summary["raw_configuration_count"]
    ):
        raise RuntimeError(
            f"Symmetry orbit multiplicities do not partition {family.name}."
        )
    return orbit_records, summary


def enumerate_2vte_pairs(
    identity_atoms: Atoms,
    canonical_atoms: Atoms,
    site_records: Sequence[dict[str, Any]],
    *,
    symprec: float = 1.0e-3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family = DefectFamily(
        name="2VTe",
        description="Two Te vacancies.",
        actions=(DefectAction("remove", "Te", 2),),
    )
    generic_records, generic_summary = enumerate_defect_family(
        identity_atoms,
        canonical_atoms,
        site_records,
        family,
        symprec=symprec,
    )
    orbit_records: list[dict[str, Any]] = []
    for generic_record in generic_records:
        action = generic_record["typed_actions"][0]
        pair_distances = generic_record[
            "canonical_mic_site_pair_distances_angstrom"
        ]
        member_ids = [
            member[0]["bulk_site_ids"]
            for member in generic_record["raw_member_typed_site_ids"]
        ]
        orbit_records.append(
            {
                "representative_bulk_indices": action["bulk_indices"],
                "removed_bulk_site_ids": action["bulk_site_ids"],
                "site_orbit_pair": sorted(action["site_orbits"]),
                "symmetry_orbit_multiplicity": generic_record[
                    "symmetry_orbit_multiplicity"
                ],
                "raw_member_bulk_site_id_pairs": member_ids,
                "canonical_mic_pair_distance_angstrom": float(
                    pair_distances[0]["distance_angstrom"]
                ),
                "vacancy_fractional_coordinates": action[
                    "canonical_fractional_coordinates"
                ],
                "defect_config_id": generic_record["defect_config_id"],
                "defect_family": "2VTe",
                "defect_type": "vacancy_pair",
                "charge": 0,
            }
        )

    category_counts = Counter(
        "-".join(record["site_orbit_pair"]) for record in orbit_records
    )
    summary = {
        "symmetry_operation_count": generic_summary["symmetry_operation_count"],
        "te_site_count": generic_summary["eligible_site_counts"]["remove:Te"],
        "raw_pair_count": generic_summary["raw_configuration_count"],
        "inequivalent_pair_count": generic_summary[
            "inequivalent_configuration_count"
        ],
        "symmetry_orbit_multiplicity_sum": generic_summary[
            "symmetry_orbit_multiplicity_sum"
        ],
        "category_counts": dict(sorted(category_counts.items())),
    }
    return orbit_records, summary


def _load_site_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "canonical_bi16te24_bulk_sites":
        raise ValueError(f"Unexpected site manifest type in {path}.")
    if not isinstance(payload.get("sites"), list):
        raise ValueError(f"Site manifest {path} has no site list.")
    return payload


def _expected_composition_for_family(
    parent: Atoms, family: DefectFamily
) -> dict[str, int]:
    counts = Counter(parent.get_chemical_symbols())
    for action in family.actions:
        if counts[action.source_species] < action.count:
            raise ValueError(
                f"Family {family.name!r} removes/replaces too many "
                f"{action.source_species} sites."
            )
        counts[action.source_species] -= action.count
        if action.operation == "substitute":
            assert action.target_species is not None
            counts[action.target_species] += action.count
    return dict(sorted((symbol, count) for symbol, count in counts.items() if count))


def construct_defect_from_typed_record(
    canonical: Atoms,
    site_records: Sequence[dict[str, Any]],
    family: DefectFamily,
    record: dict[str, Any],
) -> tuple[Atoms, dict[str, Any]]:
    """Construct a defect afresh by stable ID, never by mutating array indices."""

    if len(site_records) != len(canonical):
        raise ValueError("Site manifest length does not match the canonical parent.")
    records_by_id = {
        str(site_record["bulk_site_id"]): site_record
        for site_record in site_records
    }
    if len(records_by_id) != len(site_records):
        raise ValueError("Bulk site IDs are duplicated.")
    action_spec_by_key = {action.key: action for action in family.actions}

    action_by_site_id: dict[str, DefectAction] = {}
    selected_site_details: list[dict[str, Any]] = []
    for action_record in record["typed_actions"]:
        action_key = str(action_record["action_key"])
        if action_key not in action_spec_by_key:
            raise ValueError(
                f"Record {record['defect_config_id']} contains unknown action {action_key}."
            )
        action = action_spec_by_key[action_key]
        site_ids = [str(value) for value in action_record["bulk_site_ids"]]
        if len(site_ids) != action.count:
            raise ValueError(
                f"Record {record['defect_config_id']} has {len(site_ids)} sites for "
                f"{action.key}, expected {action.count}."
            )
        for site_id in site_ids:
            if site_id not in records_by_id:
                raise ValueError(f"Unknown stable bulk site ID {site_id!r}.")
            if site_id in action_by_site_id:
                raise ValueError(f"Multiple defect actions target stable site {site_id}.")
            site_record = records_by_id[site_id]
            index = int(site_record["bulk_index"])
            if canonical[index].symbol != action.source_species:
                raise ValueError(
                    f"Action {action.key} expects {action.source_species} at {site_id}, "
                    f"found {canonical[index].symbol}."
                )
            action_by_site_id[site_id] = action
            selected_site_details.append(
                {
                    "action_key": action.key,
                    "bulk_site_id": site_id,
                    "bulk_index": index,
                    "source_species": action.source_species,
                    "target_species": action.target_species,
                    "site_record": site_record,
                }
            )

    output_symbols: list[str] = []
    output_positions: list[np.ndarray] = []
    output_site_mapping: list[dict[str, Any]] = []
    retained_parent_indices: list[int] = []
    ordered_site_records = sorted(site_records, key=lambda item: int(item["bulk_index"]))
    for site_record in ordered_site_records:
        index = int(site_record["bulk_index"])
        site_id = str(site_record["bulk_site_id"])
        action = action_by_site_id.get(site_id)
        if action is not None and action.operation == "remove":
            continue
        output_symbol = (
            str(action.target_species)
            if action is not None and action.operation == "substitute"
            else canonical[index].symbol
        )
        output_index = len(output_symbols)
        output_symbols.append(output_symbol)
        output_positions.append(
            np.asarray(canonical.positions[index], dtype=np.float64).copy()
        )
        retained_parent_indices.append(index)
        output_site_mapping.append(
            {
                "output_index": output_index,
                "bulk_index": index,
                "bulk_site_id": site_id,
                "source_species": canonical[index].symbol,
                "output_species": output_symbol,
                "action_key": action.key if action is not None else None,
            }
        )

    defect = Atoms(
        symbols=output_symbols,
        positions=np.asarray(output_positions, dtype=np.float64),
        cell=np.asarray(canonical.cell.array, dtype=np.float64).copy(),
        pbc=np.asarray(canonical.pbc, dtype=bool).copy(),
    )
    expected_composition = _expected_composition_for_family(canonical, family)
    if formula_counts(defect) != expected_composition:
        raise RuntimeError(
            f"{record['defect_config_id']} has composition {formula_counts(defect)}, "
            f"expected {expected_composition}."
        )
    expected_natoms = len(canonical) - sum(
        action.count for action in family.actions if action.operation == "remove"
    )
    if len(defect) != expected_natoms:
        raise RuntimeError(
            f"{record['defect_config_id']} has {len(defect)} atoms, expected "
            f"{expected_natoms}."
        )
    if not np.array_equal(defect.cell.array, canonical.cell.array):
        raise RuntimeError(f"{record['defect_config_id']} changed the fixed cell.")
    if not np.array_equal(
        defect.positions,
        np.asarray(canonical.positions, dtype=np.float64)[retained_parent_indices],
    ):
        raise RuntimeError(f"{record['defect_config_id']} moved a retained atom.")

    metadata = {
        "selected_sites": sorted(
            selected_site_details,
            key=lambda item: (item["action_key"], item["bulk_site_id"]),
        ),
        "retained_parent_indices": retained_parent_indices,
        "retained_bulk_site_ids": [
            str(ordered_site_records[index]["bulk_site_id"])
            for index in retained_parent_indices
        ],
        "output_site_mapping": output_site_mapping,
        "expected_composition": expected_composition,
        "expected_natoms": expected_natoms,
    }
    return defect, metadata


def generate_defect_catalog(
    canonical_path: Path,
    site_manifest_path: Path,
    output_dir: Path,
    *,
    catalog_path: Path = DEFAULT_DEFECT_FAMILY_CATALOG,
    selected_families: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate an unrelaxed, symmetry-reduced vacancy/antisite catalog."""

    canonical_path = canonical_path.expanduser().resolve()
    site_manifest_path = site_manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    catalog_path = catalog_path.expanduser().resolve()
    canonical = read_single_structure(canonical_path)
    require_pristine_composition(canonical)
    site_manifest = _load_site_manifest(site_manifest_path)
    if geometry_sha256(canonical) != str(site_manifest["canonical_geometry_sha256"]):
        raise ValueError("Canonical geometry does not match the bulk-site manifest hash.")

    inferred_path = Path(str(site_manifest["inferred_structure"])).expanduser().resolve()
    identity = read_single_structure(inferred_path)
    if geometry_sha256(identity) != str(site_manifest["inferred_geometry_sha256"]):
        raise ValueError("Inferred symmetry reference does not match the manifest hash.")
    if identity.get_chemical_symbols() != canonical.get_chemical_symbols():
        raise ValueError("Inferred and canonical parent site ordering/species differ.")

    families, catalog_payload = load_defect_family_catalog(
        catalog_path, selected_families=selected_families
    )
    permutations = _operation_permutations(identity, symprec=1.0e-3)
    combined_catalog_path = output_dir / "catalog" / "defect_configurations.jsonl"
    summary_path = output_dir / "catalog" / "defect_generation_summary.json"
    prospective_paths: list[Path] = [combined_catalog_path, summary_path]
    for family in families:
        prospective_paths.extend(
            [
                output_dir / "catalog" / f"{family.name}_configurations.jsonl",
                output_dir / "catalog" / f"{family.name}_enumeration_summary.json",
                output_dir / "defects" / family.name,
            ]
        )
    existing = [path for path in prospective_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing defect artifacts: {existing[:5]}; "
            "pass --overwrite to replace this catalog selection."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "catalog").mkdir(parents=True, exist_ok=True)
    all_written_records: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []
    formula_distribution: Counter[str] = Counter()
    site_records = site_manifest["sites"]
    parent_geometry_hash = geometry_sha256(canonical)
    parent_cell_hash = array_sha256(canonical.cell.array, np.float64)

    for family in families:
        records, enumeration_summary = enumerate_defect_family(
            identity,
            canonical,
            site_records,
            family,
            symprec=1.0e-3,
            permutations=permutations,
        )
        family_catalog_path = (
            output_dir / "catalog" / f"{family.name}_configurations.jsonl"
        )
        family_summary_path = (
            output_dir / "catalog" / f"{family.name}_enumeration_summary.json"
        )
        family_written_records: list[dict[str, Any]] = []
        family_formula_distribution: Counter[str] = Counter()

        for record in records:
            defect, construction_metadata = construct_defect_from_typed_record(
                canonical, site_records, family, record
            )
            config_id = str(record["defect_config_id"])
            defect_dir = output_dir / "defects" / family.name / config_id
            defect_path = defect_dir / "unperturbed.extxyz"
            defect_manifest_path = defect_dir / "defect.json"
            defect_dir.mkdir(parents=True, exist_ok=True)
            if not overwrite and (defect_path.exists() or defect_manifest_path.exists()):
                raise FileExistsError(f"Refusing to overwrite {defect_dir}.")
            write_extxyz_float64(defect_path, defect)

            full_record = dict(record)
            full_record.update(
                {
                    "artifact_type": "bi2te3_unrelaxed_defect_configuration",
                    "parent_structure": canonical_path.as_posix(),
                    "parent_geometry_sha256": parent_geometry_hash,
                    "parent_cell_sha256": parent_cell_hash,
                    "parent_validation_status": site_manifest.get(
                        "parent_validation_status", "unspecified"
                    ),
                    "symmetry_identity_structure": inferred_path.as_posix(),
                    "symmetry_identity_geometry_sha256": geometry_sha256(identity),
                    "composition": formula_counts(defect),
                    "formula": defect.get_chemical_formula(mode="hill"),
                    "natoms": len(defect),
                    "construction_metadata": construction_metadata,
                    "unperturbed_structure": defect_path.as_posix(),
                    "unperturbed_geometry_sha256": geometry_sha256(defect),
                    "numeric_contract": {
                        "geometry_dtype": "float64",
                        "extxyz_float_significant_digits": 17,
                    },
                    "construction_contract": {
                        "constructed_from_stable_bulk_site_ids": True,
                        "fixed_cell": True,
                        "all_retained_positions_unchanged": True,
                        "parent_order_preserved_after_removals": True,
                        "substitutions_keep_parent_coordinates": True,
                        "distorted": False,
                        "relaxed": False,
                        "shakenbreak_applied": False,
                    },
                }
            )
            write_json(defect_manifest_path, full_record)
            family_written_records.append(full_record)
            all_written_records.append(full_record)
            family_formula_distribution[full_record["formula"]] += 1
            formula_distribution[full_record["formula"]] += 1

        with family_catalog_path.open("w", encoding="utf-8") as handle:
            for full_record in family_written_records:
                handle.write(
                    json.dumps(full_record, sort_keys=True, default=_json_default) + "\n"
                )
        family_summary = {
            "artifact_type": "bi2te3_defect_family_enumeration",
            "family": family.as_dict(),
            "enumeration": enumeration_summary,
            "computed_not_assumed": True,
            "written_structure_count": len(family_written_records),
            "formula_distribution": dict(sorted(family_formula_distribution.items())),
            "catalog": family_catalog_path.as_posix(),
        }
        write_json(family_summary_path, family_summary)
        family_summaries.append(family_summary)

    with combined_catalog_path.open("w", encoding="utf-8") as handle:
        for full_record in all_written_records:
            handle.write(
                json.dumps(full_record, sort_keys=True, default=_json_default) + "\n"
            )

    summary_payload = {
        "artifact_type": "bi2te3_unrelaxed_defect_catalog",
        "generation_status": "complete_unrelaxed_no_distortions",
        "canonical_parent": canonical_path.as_posix(),
        "canonical_parent_geometry_sha256": parent_geometry_hash,
        "symmetry_identity_parent": inferred_path.as_posix(),
        "symmetry_identity_geometry_sha256": geometry_sha256(identity),
        "parent_validation_status": site_manifest.get(
            "parent_validation_status", "unspecified"
        ),
        "site_manifest": site_manifest_path.as_posix(),
        "defect_family_catalog": catalog_path.as_posix(),
        "defect_family_catalog_sha256": file_sha256(catalog_path),
        "catalog_maximum_defect_count": int(
            catalog_payload.get("maximum_defect_count", 3)
        ),
        "selected_families": [family.name for family in families],
        "family_count": len(families),
        "family_summaries": family_summaries,
        "total_raw_configuration_count": sum(
            int(item["enumeration"]["raw_configuration_count"])
            for item in family_summaries
        ),
        "total_inequivalent_configuration_count": len(all_written_records),
        "written_structure_count": len(all_written_records),
        "formula_distribution": dict(sorted(formula_distribution.items())),
        "combined_catalog": combined_catalog_path.as_posix(),
        "scope": {
            "operations": ["remove", "substitute"],
            "charge_states": [0],
            "interstitials_included": False,
            "distortions_included": False,
            "relaxations_included": False,
        },
    }
    write_json(summary_path, summary_payload)
    return summary_payload


def generate_defect_catalog_preview(
    parent_path: Path,
    output_dir: Path,
    *,
    catalog_path: Path = DEFAULT_DEFECT_FAMILY_CATALOG,
    selected_families: Sequence[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate the catalog from an inferred parent with explicit preview provenance."""

    parent_path = parent_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    parent = read_single_structure(parent_path)
    require_pristine_composition(parent)
    spacegroup_number, _ = _site_orbits(parent, symprec=1.0e-3)
    if spacegroup_number != EXPECTED_SPACEGROUP_NUMBER:
        raise ValueError(
            f"Preview parent has space group {spacegroup_number}, expected 166."
        )

    preview_parent_dir = output_dir / "preview_parent"
    snapshot_path = preview_parent_dir / "pristine_bi16te24_preview_parent.extxyz"
    site_manifest_path = preview_parent_dir / "bulk_sites_preview.json"
    if not overwrite and (snapshot_path.exists() or site_manifest_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite preview parent under {preview_parent_dir}; "
            "pass --overwrite."
        )
    preview_parent_dir.mkdir(parents=True, exist_ok=True)
    write_extxyz_float64(snapshot_path, parent)
    site_manifest = {
        "artifact_type": "canonical_bi16te24_bulk_sites",
        "parent_validation_status": "unvalidated_inferred_preview",
        "site_identity_source": "inferred_pristine_preview",
        "coordinate_source": "inferred_pristine_preview",
        "original_parent_structure": parent_path.as_posix(),
        "inferred_structure": snapshot_path.as_posix(),
        "inferred_geometry_sha256": geometry_sha256(parent),
        "canonical_structure": snapshot_path.as_posix(),
        "canonical_geometry_sha256": geometry_sha256(parent),
        "canonical_cell_sha256": array_sha256(parent.cell.array, np.float64),
        "sites": build_site_records(parent, parent, symprec=1.0e-3),
    }
    write_json(site_manifest_path, site_manifest)
    return generate_defect_catalog(
        snapshot_path,
        site_manifest_path,
        output_dir,
        catalog_path=catalog_path,
        selected_families=selected_families,
        overwrite=overwrite,
    )


def generate_2vte_structures(
    canonical_path: Path,
    site_manifest_path: Path,
    output_dir: Path,
    *,
    expected_inequivalent: int | None = 14,
    overwrite: bool = False,
) -> dict[str, Any]:
    canonical_path = canonical_path.expanduser().resolve()
    site_manifest_path = site_manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    canonical = read_single_structure(canonical_path)
    require_pristine_composition(canonical)
    site_manifest = _load_site_manifest(site_manifest_path)
    if geometry_sha256(canonical) != str(site_manifest["canonical_geometry_sha256"]):
        raise ValueError("Canonical geometry does not match the bulk-site manifest hash.")

    inferred_path = Path(str(site_manifest["inferred_structure"])).expanduser().resolve()
    identity = read_single_structure(inferred_path)
    if geometry_sha256(identity) != str(site_manifest["inferred_geometry_sha256"]):
        raise ValueError("Inferred symmetry reference does not match the manifest hash.")

    records, summary = enumerate_2vte_pairs(
        identity,
        canonical,
        site_manifest["sites"],
        symprec=1.0e-3,
    )
    expected_raw = math.comb(24, 2)
    if summary["raw_pair_count"] != expected_raw:
        raise RuntimeError(
            f"Expected C(24,2)={expected_raw} raw pairs, got {summary['raw_pair_count']}."
        )
    if expected_inequivalent is not None and summary["inequivalent_pair_count"] != int(
        expected_inequivalent
    ):
        raise RuntimeError(
            "Algorithmically enumerated "
            f"{summary['inequivalent_pair_count']} inequivalent pairs; pilot assertion "
            f"expected {expected_inequivalent}."
        )

    catalog_path = output_dir / "catalog" / "2VTe_configurations.jsonl"
    summary_path = output_dir / "catalog" / "2VTe_enumeration_summary.json"
    output_exists = (
        catalog_path.exists()
        or summary_path.exists()
        or (output_dir / "defects").exists()
    )
    if output_exists and not overwrite:
        raise FileExistsError(
            f"Output already exists under {output_dir}; pass --overwrite to replace files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and catalog_path.exists():
        catalog_path.unlink()

    parent_positions = np.asarray(canonical.positions, dtype=np.float64)
    parent_cell = np.asarray(canonical.cell.array, dtype=np.float64)
    parent_symbols = canonical.get_chemical_symbols()
    site_records = {int(record["bulk_index"]): record for record in site_manifest["sites"]}
    written_records: list[dict[str, Any]] = []
    for record in records:
        removed_indices = sorted(
            (int(value) for value in record["representative_bulk_indices"]),
            reverse=True,
        )
        keep_indices = [
            index for index in range(len(canonical)) if index not in set(removed_indices)
        ]
        defect = canonical.copy()
        for index in removed_indices:
            del defect[index]
        if formula_counts(defect) != {"Bi": 16, "Te": 22} or len(defect) != 38:
            raise RuntimeError(
                f"{record['defect_config_id']} has invalid composition {formula_counts(defect)}."
            )
        if not np.array_equal(defect.cell.array, parent_cell):
            raise RuntimeError(f"{record['defect_config_id']} changed the fixed cell.")
        if not np.array_equal(defect.positions, parent_positions[keep_indices]):
            raise RuntimeError(f"{record['defect_config_id']} moved a retained atom.")
        if defect.get_chemical_symbols() != [parent_symbols[index] for index in keep_indices]:
            raise RuntimeError(f"{record['defect_config_id']} changed retained species/order.")

        defect_dir = output_dir / "defects" / str(record["defect_config_id"])
        defect_path = defect_dir / "unperturbed.extxyz"
        defect_manifest_path = defect_dir / "defect.json"
        defect_dir.mkdir(parents=True, exist_ok=True)
        if not overwrite and (defect_path.exists() or defect_manifest_path.exists()):
            raise FileExistsError(f"Refusing to overwrite {defect_dir}.")
        write_extxyz_float64(defect_path, defect)

        full_record = dict(record)
        full_record.update(
            {
                "parent_structure": canonical_path.as_posix(),
                "parent_geometry_sha256": geometry_sha256(canonical),
                "parent_cell_sha256": array_sha256(canonical.cell.array, np.float64),
                "parent_validation_status": site_manifest.get(
                    "parent_validation_status", "unspecified"
                ),
                "composition": formula_counts(defect),
                "natoms": len(defect),
                "removed_bulk_indices_descending": removed_indices,
                "removed_sites": [
                    site_records[index] for index in record["representative_bulk_indices"]
                ],
                "retained_bulk_site_ids": [
                    str(site_records[index]["bulk_site_id"]) for index in keep_indices
                ],
                "unperturbed_structure": defect_path.as_posix(),
                "unperturbed_geometry_sha256": geometry_sha256(defect),
                "construction_contract": {
                    "fixed_cell": True,
                    "retained_positions_unchanged": True,
                    "retained_order_preserved": True,
                    "distorted": False,
                    "relaxed": False,
                },
            }
        )
        write_json(defect_manifest_path, full_record)
        written_records.append(full_record)

    with catalog_path.open("w", encoding="utf-8") as handle:
        for record in written_records:
            handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")

    summary_payload = {
        "artifact_type": "bi2te3_2vte_unrelaxed_defect_cohort",
        "generation_status": "complete_unrelaxed_no_distortions",
        "canonical_parent": canonical_path.as_posix(),
        "canonical_parent_geometry_sha256": geometry_sha256(canonical),
        "parent_validation_status": site_manifest.get(
            "parent_validation_status", "unspecified"
        ),
        "site_manifest": site_manifest_path.as_posix(),
        "enumeration": summary,
        "expected_inequivalent_pilot_assertion": expected_inequivalent,
        "written_structure_count": len(written_records),
        "catalog": catalog_path.as_posix(),
    }
    write_json(summary_path, summary_payload)
    return summary_payload


def generate_2vte_preview(
    parent_path: Path,
    output_dir: Path,
    *,
    expected_inequivalent: int | None = 14,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate a persistent 2VTe preview directly from an inferred parent."""

    parent_path = parent_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    parent = read_single_structure(parent_path)
    require_pristine_composition(parent)
    spacegroup_number, _ = _site_orbits(parent, symprec=1.0e-3)
    if spacegroup_number != EXPECTED_SPACEGROUP_NUMBER:
        raise ValueError(
            f"Preview parent has space group {spacegroup_number}, expected 166."
        )

    preview_parent_dir = output_dir / "preview_parent"
    snapshot_path = preview_parent_dir / "pristine_bi16te24_preview_parent.extxyz"
    site_manifest_path = preview_parent_dir / "bulk_sites_preview.json"
    generated_paths = (
        snapshot_path,
        site_manifest_path,
        output_dir / "catalog" / "2VTe_configurations.jsonl",
        output_dir / "catalog" / "2VTe_enumeration_summary.json",
        output_dir / "defects",
    )
    if not overwrite:
        existing = [path for path in generated_paths if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite preview artifacts: {existing}; "
                "pass --overwrite to replace files."
            )

    preview_parent_dir.mkdir(parents=True, exist_ok=True)
    write_extxyz_float64(snapshot_path, parent)
    site_manifest = {
        "artifact_type": "canonical_bi16te24_bulk_sites",
        "parent_validation_status": "unvalidated_inferred_preview",
        "site_identity_source": "inferred_pristine_preview",
        "coordinate_source": "inferred_pristine_preview",
        "original_parent_structure": parent_path.as_posix(),
        "inferred_structure": snapshot_path.as_posix(),
        "inferred_geometry_sha256": geometry_sha256(parent),
        "canonical_structure": snapshot_path.as_posix(),
        "canonical_geometry_sha256": geometry_sha256(parent),
        "canonical_cell_sha256": array_sha256(parent.cell.array, np.float64),
        "sites": build_site_records(parent, parent, symprec=1.0e-3),
    }
    write_json(site_manifest_path, site_manifest)
    return generate_2vte_structures(
        snapshot_path,
        site_manifest_path,
        output_dir,
        expected_inequivalent=expected_inequivalent,
        overwrite=overwrite,
    )


def _parse_kpts(value: str) -> tuple[int, int, int]:
    tokens = value.replace(",", " ").split()
    if len(tokens) == 1:
        result = (int(tokens[0]),) * 3
    elif len(tokens) == 3:
        result = tuple(int(token) for token in tokens)
    else:
        raise argparse.ArgumentTypeError("k-points must be one integer or three integers")
    if any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("k-points must be positive")
    return result  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export-inferred", help="Recover and export the inferred pristine Bi16Te24."
    )
    export_parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument(
        "--anchor-structure-id", default=DEFAULT_ANCHOR_STRUCTURE_ID
    )
    export_parser.add_argument("--overwrite", action="store_true")

    prepare_parser = subparsers.add_parser(
        "prepare-qe", help="Write a precise fixed-cell QE/BFGS relaxation input."
    )
    prepare_parser.add_argument("--inferred", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--pseudo-dir", type=Path, required=True)
    prepare_parser.add_argument("--qe-outdir", type=Path, required=True)
    prepare_parser.add_argument("--kpts", type=_parse_kpts, default=(6, 6, 6))
    prepare_parser.add_argument("--ecutwfc", type=float, default=60.0)
    prepare_parser.add_argument("--ecutrho", type=float, default=480.0)
    prepare_parser.add_argument("--force-max", type=float, default=0.01)
    prepare_parser.add_argument("--max-steps", type=int, default=100)
    prepare_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate-qe", help="Validate QE output and freeze the canonical parent."
    )
    validate_parser.add_argument("--inferred", type=Path, required=True)
    validate_parser.add_argument("--qe-output", type=Path, required=True)
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.add_argument("--force-max", type=float, default=0.01)
    validate_parser.add_argument("--cell-tolerance", type=float, default=1.0e-6)
    validate_parser.add_argument("--max-rmsd", type=float, default=0.10)
    validate_parser.add_argument("--max-displacement", type=float, default=0.25)
    validate_parser.add_argument("--overwrite", action="store_true")

    generate_parser = subparsers.add_parser(
        "generate-2vte",
        help="Enumerate and construct unrelaxed symmetry-inequivalent 2VTe defects.",
    )
    generate_parser.add_argument("--canonical", type=Path, required=True)
    generate_parser.add_argument("--site-manifest", type=Path, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--expected-inequivalent", type=int, default=14)
    generate_parser.add_argument("--overwrite", action="store_true")

    preview_parser = subparsers.add_parser(
        "generate-2vte-preview",
        help=(
            "Construct the 14 unrelaxed 2VTe structures directly from an "
            "inferred parent for generator testing."
        ),
    )
    preview_parser.add_argument("--parent", type=Path, required=True)
    preview_parser.add_argument("--output-dir", type=Path, required=True)
    preview_parser.add_argument("--expected-inequivalent", type=int, default=14)
    preview_parser.add_argument("--overwrite", action="store_true")

    catalog_parser = subparsers.add_parser(
        "generate-defect-catalog",
        help=(
            "Generate the reviewed neutral vacancy/antisite catalog from the "
            "validated canonical parent."
        ),
    )
    catalog_parser.add_argument("--canonical", type=Path, required=True)
    catalog_parser.add_argument("--site-manifest", type=Path, required=True)
    catalog_parser.add_argument("--output-dir", type=Path, required=True)
    catalog_parser.add_argument(
        "--catalog", type=Path, default=DEFAULT_DEFECT_FAMILY_CATALOG
    )
    catalog_parser.add_argument(
        "--families",
        nargs="+",
        help="Optional family names; omit to generate every enabled catalog family.",
    )
    catalog_parser.add_argument("--overwrite", action="store_true")

    catalog_preview_parser = subparsers.add_parser(
        "generate-defect-catalog-preview",
        help=(
            "Test the same catalog generator from an inferred parent, with all "
            "outputs explicitly marked unvalidated preview."
        ),
    )
    catalog_preview_parser.add_argument("--parent", type=Path, required=True)
    catalog_preview_parser.add_argument("--output-dir", type=Path, required=True)
    catalog_preview_parser.add_argument(
        "--catalog", type=Path, default=DEFAULT_DEFECT_FAMILY_CATALOG
    )
    catalog_preview_parser.add_argument(
        "--families",
        nargs="+",
        help="Optional family names; omit to generate every enabled catalog family.",
    )
    catalog_preview_parser.add_argument("--overwrite", action="store_true")
    return parser


def print_cli_summary(command: str, args: argparse.Namespace, result: dict[str, Any]) -> None:
    """Print a compact status; complete machine-readable details live in manifests."""

    print(f"Michael, {command} completed successfully.")
    if command == "export-inferred":
        print(f"output_dir={Path(args.output_dir).expanduser().resolve()}")
        print(f"cohort_size={result['cohort_size']}")
        print(f"formula_counts={result['formula_counts']}")
        print(f"spacegroup_number={result['spacegroup_number']}")
        print(f"geometry_sha256={result['geometry_sha256']}")
    elif command == "prepare-qe":
        print(f"execution_status={result['execution_status']}")
        print(f"qe_input={result['qe_input']}")
        print(f"qe_outdir={result['qe_outdir']}")
        print(
            "manifest="
            f"{Path(args.output_dir).expanduser().resolve() / 'qe_prepare_manifest.json'}"
        )
    elif command == "validate-qe":
        metrics = result["structural_equivalence_metrics"]
        print(f"validation_status={result['validation_status']}")
        print(f"final_max_force_ev_per_angstrom={result['final_max_force_ev_per_angstrom']}")
        print(f"mapped_rmsd_angstrom={metrics['mapped_rmsd_angstrom']}")
        print(
            "mapped_max_displacement_angstrom="
            f"{metrics['mapped_max_displacement_angstrom']}"
        )
        print(f"accepted_symprec_angstrom={metrics['accepted_symprec_angstrom']}")
        print(f"canonical_structure={result['canonical_structure']}")
    elif command in {"generate-2vte", "generate-2vte-preview"}:
        enumeration = result["enumeration"]
        print(f"raw_pair_count={enumeration['raw_pair_count']}")
        print(f"inequivalent_pair_count={enumeration['inequivalent_pair_count']}")
        print(
            "symmetry_orbit_multiplicity_sum="
            f"{enumeration['symmetry_orbit_multiplicity_sum']}"
        )
        print(f"category_counts={enumeration['category_counts']}")
        print(f"written_structure_count={result['written_structure_count']}")
        print(f"catalog={result['catalog']}")
    elif command in {
        "generate-defect-catalog",
        "generate-defect-catalog-preview",
    }:
        print(f"parent_validation_status={result['parent_validation_status']}")
        print(f"family_count={result['family_count']}")
        print(f"selected_families={','.join(result['selected_families'])}")
        print(
            "total_raw_configuration_count="
            f"{result['total_raw_configuration_count']}"
        )
        print(
            "total_inequivalent_configuration_count="
            f"{result['total_inequivalent_configuration_count']}"
        )
        print(f"formula_distribution={result['formula_distribution']}")
        print(f"combined_catalog={result['combined_catalog']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export-inferred":
        result = export_inferred_pristine(
            args.source_db,
            args.output_dir,
            anchor_structure_id=args.anchor_structure_id,
            overwrite=args.overwrite,
        )
    elif args.command == "prepare-qe":
        result = prepare_qe_relaxation(
            args.inferred,
            args.output_dir,
            pseudo_dir=args.pseudo_dir,
            qe_outdir=args.qe_outdir,
            kpts=args.kpts,
            ecutwfc_ry=args.ecutwfc,
            ecutrho_ry=args.ecutrho,
            force_max_ev_per_angstrom=args.force_max,
            max_steps=args.max_steps,
            overwrite=args.overwrite,
        )
    elif args.command == "validate-qe":
        result = validate_qe_relaxation(
            args.inferred,
            args.qe_output,
            args.output_dir,
            force_max_ev_per_angstrom=args.force_max,
            cell_tolerance_angstrom=args.cell_tolerance,
            max_rmsd_angstrom=args.max_rmsd,
            max_displacement_angstrom=args.max_displacement,
            overwrite=args.overwrite,
        )
    elif args.command == "generate-2vte":
        result = generate_2vte_structures(
            args.canonical,
            args.site_manifest,
            args.output_dir,
            expected_inequivalent=args.expected_inequivalent,
            overwrite=args.overwrite,
        )
    elif args.command == "generate-2vte-preview":
        result = generate_2vte_preview(
            args.parent,
            args.output_dir,
            expected_inequivalent=args.expected_inequivalent,
            overwrite=args.overwrite,
        )
    elif args.command == "generate-defect-catalog":
        result = generate_defect_catalog(
            args.canonical,
            args.site_manifest,
            args.output_dir,
            catalog_path=args.catalog,
            selected_families=args.families,
            overwrite=args.overwrite,
        )
    elif args.command == "generate-defect-catalog-preview":
        result = generate_defect_catalog_preview(
            args.parent,
            args.output_dir,
            catalog_path=args.catalog,
            selected_families=args.families,
            overwrite=args.overwrite,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print_cli_summary(args.command, args, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
