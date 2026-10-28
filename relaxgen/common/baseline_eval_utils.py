import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read

from relaxgen.common.mace_compat import (
    ensure_legacy_mace_codegen_compat,
    patch_mace_calculator_runtime_dtype,
    repair_legacy_e3nn_modules,
)
from relaxgen.common.qe_scratch import temporary_qe_outdir

if TYPE_CHECKING:
    from ase.calculators.espresso import Espresso
    from mace.calculators import MACECalculator


# MACE checkpoints still live in the ConstrDiff_Material checkout.
CONSTR_DIFF_ROOT = Path(
    os.environ.get("CONSTR_DIFF_ROOT", "/home/gqs7zr/ai4mat/ConstrDiff_Material")
)
DEFAULT_MACE_MODEL_PATH = (
    CONSTR_DIFF_ROOT
    / "model_checkpoints"
    / "mace"
    / "bi2te3_defect_finetune_run_42_epoch-98_stagetwo.model"
)
DEFAULT_PSEUDO_DIR = Path(__file__).resolve().parents[2] / "pseudos"
DEFAULT_QE_SIF = Path("/home/gqs7zr/ai4mat/ckpts/qe-7.3.1.sif")
DEFAULT_DFT_COMMAND = f"apptainer exec --nv {DEFAULT_QE_SIF} mpirun -np 2 pw.x -ndiag 1"
DEFAULT_CELL = np.array(
    [
        [8.5755, 0.0000, 18.9311],
        [4.0878, 7.5385, 18.9311],
        [0.0000, 0.0000, 20.7828],
    ],
    dtype=np.float64,
)
DEFAULT_PBC = np.array([True, True, True], dtype=bool)
ATOM_TYPE_TO_SYMBOL = {
    0: "Bi",
    1: "Te",
}
SPECIES_TO_ATOM_TYPE = {
    symbol: atom_type for atom_type, symbol in ATOM_TYPE_TO_SYMBOL.items()
}
PSEUDOPOTENTIALS = {
    "Bi": "Bi.pbesol-dn-rrkjus.UPF",
    "Te": "Te.pbesol-n-rrkjus_psl.1.0.0.v2.UPF",
}


@dataclass
class LoadedSample:
    positions: np.ndarray
    atom_type: np.ndarray
    mask: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    source_path: Path
    payload: dict[str, Any] | list[Any]


def _to_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _pick_first_structure(data: Any) -> tuple[Any, Any, Any, Any, Any]:
    if isinstance(data, dict):
        pos = data["pos"]
        atom_type = data["atom_type"]
        mask = data.get("mask", None)
        cell = data.get("cell", data.get("lattice_cell", None))
        pbc = data.get("pbc", None)

        pos_np = _to_numpy(pos, np.float32)
        atom_type_np = _to_numpy(atom_type, np.int64)
        mask_np = _to_numpy(mask, np.float32)
        cell_np = _to_numpy(cell, np.float32)
        pbc_np = _to_numpy(pbc, bool)

        if pos_np is None or atom_type_np is None:
            raise ValueError("Sample is missing required pos/atom_type fields.")

        if pos_np.ndim == 3:
            pos_np = pos_np[0]
        if atom_type_np.ndim == 2:
            atom_type_np = atom_type_np[0]
        if mask_np is not None and mask_np.ndim >= 2:
            mask_np = mask_np[0]
        if cell_np is not None and cell_np.ndim == 3:
            cell_np = cell_np[0]
        if pbc_np is not None and pbc_np.ndim == 2:
            pbc_np = pbc_np[0]
        return pos_np, atom_type_np, mask_np, cell_np, pbc_np

    if isinstance(data, list):
        if not data:
            raise ValueError("Sample list is empty.")
        first = data[0]
        if not isinstance(first, dict):
            raise ValueError("Expected a dict entry inside sample list.")
        return _pick_first_structure(first)

    raise ValueError(f"Unsupported sample payload type: {type(data)}")


def load_pt_sample(pt_path: str | Path) -> LoadedSample:
    sample_path = Path(pt_path).expanduser().resolve()
    data = torch.load(sample_path, map_location="cpu", weights_only=False)
    pos, atom_type, mask, cell, pbc = _pick_first_structure(data)

    if mask is None:
        mask = np.ones(atom_type.shape, dtype=np.float32)
    else:
        mask = mask.astype(np.float32, copy=False)

    if cell is None:
        cell = DEFAULT_CELL.copy()
    else:
        cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)

    if pbc is None:
        pbc = DEFAULT_PBC.copy()
    else:
        pbc = np.asarray(pbc, dtype=bool).reshape(3)

    positions = np.asarray(pos, dtype=np.float64)
    atom_type = np.asarray(atom_type, dtype=np.int64)

    if positions.shape[0] != atom_type.shape[0]:
        raise ValueError(
            f"Position/atom_type length mismatch in {sample_path}: "
            f"{positions.shape[0]} vs {atom_type.shape[0]}"
        )

    if mask.shape[0] != atom_type.shape[0]:
        raise ValueError(
            f"Mask/atom_type length mismatch in {sample_path}: "
            f"{mask.shape[0]} vs {atom_type.shape[0]}"
        )

    return LoadedSample(
        positions=positions,
        atom_type=atom_type,
        mask=mask,
        cell=cell,
        pbc=pbc,
        source_path=sample_path,
        payload=data,
    )


def load_extxyz_sample(extxyz_path: str | Path, frame_index: int = -1) -> LoadedSample:
    sample_path = Path(extxyz_path).expanduser().resolve()
    frames = ase_read(str(sample_path), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]

    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError(f"No frames found in extxyz file: {sample_path}")

    resolved_index = frame_index if frame_index >= 0 else n_frames + frame_index
    if resolved_index < 0 or resolved_index >= n_frames:
        raise IndexError(
            f"Requested frame_index={frame_index} resolves to {resolved_index}, "
            f"which is out of range for {sample_path} with {n_frames} frame(s)."
        )

    atoms = frames[resolved_index]
    symbols = atoms.get_chemical_symbols()
    if "atom_type" in atoms.arrays:
        atom_type = np.asarray(atoms.arrays["atom_type"], dtype=np.int64)
    else:
        atom_type = np.asarray([SPECIES_TO_ATOM_TYPE[symbol] for symbol in symbols], dtype=np.int64)

    payload = {
        "source_format": "extxyz",
        "frame_index": int(resolved_index),
        "n_frames": int(n_frames),
        "frame_info": dict(atoms.info),
    }

    return LoadedSample(
        positions=np.asarray(atoms.get_positions(), dtype=np.float64),
        atom_type=atom_type,
        mask=np.ones((len(symbols),), dtype=np.float32),
        cell=np.asarray(atoms.cell.array, dtype=np.float64).reshape(3, 3),
        pbc=np.asarray(atoms.get_pbc(), dtype=bool).reshape(3),
        source_path=sample_path,
        payload=payload,
    )


def loaded_sample_to_atoms(sample: LoadedSample) -> Atoms:
    valid_mask = (sample.mask > 0) & (sample.atom_type != 2)
    valid_atom_types = sample.atom_type[valid_mask]
    valid_positions = sample.positions[valid_mask]

    unknown_types = sorted(set(int(v) for v in valid_atom_types.tolist()) - set(ATOM_TYPE_TO_SYMBOL))
    if unknown_types:
        raise ValueError(
            f"Unsupported atom types in {sample.source_path}: {unknown_types}. "
            f"Supported types are {sorted(ATOM_TYPE_TO_SYMBOL)}."
        )

    symbols = [ATOM_TYPE_TO_SYMBOL[int(v)] for v in valid_atom_types.tolist()]
    return Atoms(
        symbols=symbols,
        positions=valid_positions,
        cell=sample.cell,
        pbc=sample.pbc,
    )


def summarize_force_array(forces: np.ndarray) -> dict[str, float]:
    forces = np.asarray(forces, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[1] != 3:
        raise ValueError(f"forces must have shape (N, 3), got {forces.shape}.")
    if not np.isfinite(forces).all():
        raise ValueError("forces contain non-finite values.")
    force_norms = np.linalg.norm(forces, axis=1)
    return {
        "force_l2_norm": float(np.linalg.norm(forces)),
        "force_rms_atom_norm": (
            float(np.sqrt(np.mean(force_norms * force_norms)))
            if force_norms.size
            else 0.0
        ),
        "force_max_atom_norm": float(force_norms.max()) if force_norms.size else 0.0,
        "force_mean_atom_norm": float(force_norms.mean()) if force_norms.size else 0.0,
    }


def build_mace_calculator(model_path: str | Path, device: str | None = None) -> "MACECalculator":
    from mace.calculators import MACECalculator

    ensure_legacy_mace_codegen_compat()
    requested_device = str(device or "auto").strip()
    use_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if requested_device.lower() == "auto" else requested_device
    default_dtype = torch.get_default_dtype()
    try:
        calc = MACECalculator(
            model_paths=str(Path(model_path).expanduser().resolve()),
            device=use_device,
        )
    finally:
        torch.set_default_dtype(default_dtype)
    patch_mace_calculator_runtime_dtype(calc)
    for model in getattr(calc, "models", []):
        repair_legacy_e3nn_modules(model)
        if hasattr(model, "double"):
            model.double()
        if hasattr(model, "eval"):
            model.eval()
    return calc


def evaluate_mace_singlepoint(atoms: Atoms, mace_calc: "MACECalculator") -> dict[str, Any]:
    atoms_eval = atoms.copy()
    atoms_eval.calc = mace_calc
    energy = float(atoms_eval.get_potential_energy())
    forces = np.asarray(atoms_eval.get_forces(), dtype=np.float64)
    result = {
        "energy_ev": energy,
        "forces": forces,
        "status": "ok",
        "error": "",
    }
    result.update(summarize_force_array(forces))
    return result


def resolve_qe_command(command: str | None = None) -> str:
    if command:
        return command
    if "ASE_ESPRESSO_COMMAND" in os.environ:
        return os.environ["ASE_ESPRESSO_COMMAND"]

    pw_path = shutil.which("pw.x")
    if pw_path is None:
        search_dirs = [os.path.dirname(sys.executable)]
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            search_dirs.append(os.path.join(conda_prefix, "bin"))
        for prefix_bin in search_dirs:
            candidate = os.path.join(prefix_bin, "pw.x")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                pw_path = candidate
                break

    if pw_path is None:
        raise RuntimeError("pw.x not found and no --dft_command provided")

    mpirun_path = shutil.which("mpirun")
    if mpirun_path:
        return f"{mpirun_path} -np 2 {pw_path}"
    return pw_path


def build_scf_input_data(
    calculation_mode: str,
    *,
    ecutwfc_override: float | None = None,
    ecutrho_override: float | None = None,
) -> dict[str, Any]:
    mode = calculation_mode.lower()
    if mode == "ultrafast":
        ecutwfc, ecutrho, degauss, conv_thr, electron_maxstep, mixing_beta = (
            25,
            200,
            0.03,
            1.0e-4,
            30,
            0.7,
        )
    elif mode == "coarse":
        ecutwfc, ecutrho, degauss, conv_thr, electron_maxstep, mixing_beta = (
            35,
            280,
            0.01,
            1.0e-6,
            50,
            0.5,
        )
    elif mode == "precise":
        ecutwfc, ecutrho, degauss, conv_thr, electron_maxstep, mixing_beta = (
            60,
            480,
            0.001,
            1.0e-9,
            200,
            0.3,
        )
    else:
        raise ValueError(f"Invalid DFT calculation mode: {calculation_mode}")

    if ecutwfc_override is not None:
        ecutwfc = float(ecutwfc_override)
    if ecutrho_override is not None:
        ecutrho = float(ecutrho_override)

    return {
        "control": {
            "calculation": "scf",
            "restart_mode": "from_scratch",
            "tprnfor": True,
        },
        "system": {
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "occupations": "smearing",
            "smearing": "mv",
            "degauss": degauss,
        },
        "electrons": {
            "conv_thr": conv_thr,
            "electron_maxstep": electron_maxstep,
            "diagonalization": "david",
            "mixing_beta": mixing_beta,
        },
    }


def build_qe_calculator(
    *,
    command: str,
    pseudo_dir: str | Path,
    kpts: tuple[int, int, int],
    directory: Path,
    qe_outdir: Path,
    calculation_mode: str,
    ecutwfc: float | None = None,
    ecutrho: float | None = None,
) -> "Espresso":
    from ase.calculators.espresso import Espresso, EspressoProfile

    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    qe_outdir = Path(qe_outdir).expanduser().resolve()
    qe_outdir.mkdir(parents=True, exist_ok=True)

    input_data = build_scf_input_data(
        calculation_mode,
        ecutwfc_override=ecutwfc,
        ecutrho_override=ecutrho,
    )
    input_data["control"]["outdir"] = str(qe_outdir)

    profile = EspressoProfile(
        command=command,
        pseudo_dir=str(Path(pseudo_dir).expanduser().resolve()),
    )
    return Espresso(
        pseudopotentials=PSEUDOPOTENTIALS,
        kpts=kpts,
        input_data=input_data,
        profile=profile,
        directory=str(directory),
    )


def evaluate_dft_singlepoint(
    atoms: Atoms,
    *,
    command: str,
    pseudo_dir: str | Path,
    kpts: tuple[int, int, int],
    directory: Path,
    calculation_mode: str,
    ecutwfc: float | None = None,
    ecutrho: float | None = None,
    qe_tmp_base_dir: str | Path | None = None,
) -> dict[str, Any]:
    with temporary_qe_outdir(
        qe_tmp_base_dir,
        prefix="singlepoint_",
    ) as qe_outdir:
        calc = build_qe_calculator(
            command=command,
            pseudo_dir=pseudo_dir,
            kpts=kpts,
            directory=directory,
            qe_outdir=qe_outdir,
            calculation_mode=calculation_mode,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
        )
        atoms_eval = atoms.copy()
        atoms_eval.calc = calc
        energy = float(atoms_eval.get_potential_energy())
        forces = np.asarray(atoms_eval.get_forces(), dtype=np.float64)
    result = {
        "energy_ev": energy,
        "forces": forces,
        "status": "ok",
        "error": "",
    }
    result.update(summarize_force_array(forces))
    return result
