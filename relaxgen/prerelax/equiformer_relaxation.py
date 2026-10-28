from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.optimize import BFGS, FIRE, LBFGS


MODULE_ROOT = Path(__file__).resolve().parent
# The EquiformerV3 code still lives in the ConstrDiff_Material checkout.
CONSTR_DIFF_ROOT = Path(
    os.environ.get("CONSTR_DIFF_ROOT", "/home/gqs7zr/ai4mat/ConstrDiff_Material")
)
PROJECT_ROOT = CONSTR_DIFF_ROOT
VENDORED_FAIRCHEM_V1_ROOT = CONSTR_DIFF_ROOT / "fairchemv1"


def _candidate_equiformer_roots() -> list[Path]:
    env_candidates = [
        os.environ.get("EQUIFORMER_V3_ROOT"),
        os.environ.get("EQV3_ROOT"),
    ]
    candidates: list[Path] = []

    for value in env_candidates:
        if value:
            candidates.append(Path(value).expanduser().resolve())

    candidates.extend(
        [
            PROJECT_ROOT / "models" / "equiformer_v3",
            PROJECT_ROOT.parent / "equiformer_v3",
            Path.cwd() / "equiformer_v3",
            Path.cwd().parent / "equiformer_v3",
        ]
    )
    return candidates


from fairchem.core import OCPCalculator


OptimizerName = Literal["FIRE", "LBFGS", "BFGS"]
CellFilterName = Literal["frechet", "unit"]

OPTIMIZER_CLS = {
    "FIRE": FIRE,
    "LBFGS": LBFGS,
    "BFGS": BFGS,
}

CELL_FILTER_CLS = {
    "frechet": FrechetCellFilter,
    "unit": UnitCellFilter,
}


@dataclass
class EquiformerRelaxationResult:
    relaxed_atoms: Atoms
    checkpoint_path: Path
    config_yml: Path | None
    input_path: Path | None
    output_path: Path | None
    trajectory_path: Path | None
    log_path: Path | None
    optimizer: str
    force_max: float
    max_steps: int
    steps_taken: int
    reached_max_steps: bool
    converged: bool
    relax_cell: bool
    cell_filter: str | None
    fixed_symmetry: bool
    wall_time_s: float
    final_energy_ev: float
    final_forces_ev_per_ang: np.ndarray
    final_stress_ev_per_ang3: np.ndarray | None
    final_max_force_ev_per_ang: float
    final_mean_force_ev_per_ang: float

    def to_summary(self) -> dict[str, Any]:
        summary = {
            "checkpoint_path": str(self.checkpoint_path),
            "config_yml": str(self.config_yml) if self.config_yml is not None else None,
            "input_path": str(self.input_path) if self.input_path is not None else None,
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "trajectory_path": (
                str(self.trajectory_path) if self.trajectory_path is not None else None
            ),
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "optimizer": self.optimizer,
            "force_max": float(self.force_max),
            "max_steps": int(self.max_steps),
            "steps_taken": int(self.steps_taken),
            "reached_max_steps": bool(self.reached_max_steps),
            "converged": bool(self.converged),
            "relax_cell": bool(self.relax_cell),
            "cell_filter": self.cell_filter,
            "fixed_symmetry": bool(self.fixed_symmetry),
            "wall_time_s": float(self.wall_time_s),
            "final_energy_ev": float(self.final_energy_ev),
            "final_max_force_ev_per_ang": float(self.final_max_force_ev_per_ang),
            "final_mean_force_ev_per_ang": float(self.final_mean_force_ev_per_ang),
            "final_formula": self.relaxed_atoms.get_chemical_formula(),
        }
        if self.final_stress_ev_per_ang3 is not None:
            summary["final_stress_ev_per_ang3"] = self.final_stress_ev_per_ang3.tolist()
        return summary


def _resolve_cpu_flag(device: str) -> bool:
    requested = device.strip().lower()
    if requested == "auto":
        return not torch.cuda.is_available()
    if requested == "cpu":
        return True
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return True
    return False


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return bool(default)
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _load_atoms(input_path: str | Path, frame_index: int = 0) -> Atoms:
    resolved = Path(input_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input structure does not exist: {resolved}")
    atoms = ase_read(str(resolved), index=frame_index)
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Expected a single ASE Atoms object from {resolved}, got {type(atoms)}")
    return atoms


def _unwrap_atoms(relax_target: Any) -> Atoms:
    return getattr(relax_target, "atoms", relax_target)


def _optimizer_kwargs(optimizer: OptimizerName, optimizer_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    kwargs = dict(optimizer_kwargs or {})
    if optimizer in {"LBFGS", "BFGS"} and "maxstep" not in kwargs:
        kwargs["maxstep"] = 0.2
    return kwargs


def _summarize_force_array(forces: np.ndarray) -> dict[str, float]:
    forces = np.asarray(forces, dtype=np.float64)
    if forces.ndim != 2 or forces.shape[1] != 3 or not np.isfinite(forces).all():
        raise ValueError(f"forces must be finite with shape (N, 3), got {forces.shape}.")
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


def _attach_final_results(atoms: Atoms, include_stress: bool) -> tuple[Atoms, float, np.ndarray, np.ndarray | None]:
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    stress = None
    results: dict[str, Any] = {
        "energy": energy,
        "free_energy": energy,
        "forces": forces,
    }
    if include_stress:
        stress = np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
        results["stress"] = stress

    atoms_with_results = atoms.copy()
    atoms_with_results.constraints = None
    atoms_with_results.calc = SinglePointCalculator(atoms_with_results, **results)
    return atoms_with_results, energy, forces, stress


class EquiformerRelaxer:
    """ASE-based structure relaxer backed by an Equiformer/OCP checkpoint.

    This is intentionally Equiformer-only. It does not depend on the diffusion or
    flow-matching models in `constr_diffusion`; it simply wraps an Equiformer
    checkpoint inside `fairchem.core.OCPCalculator` and uses an ASE optimizer to
    relax one structure.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        config_yml: str | Path | None = None,
        device: str = "auto",
        seed: int = 0,
        only_output: list[str] | None = None,
        disable_amp: bool = True,
        quiet_model_load: bool | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

        self.config_yml = (
            Path(config_yml).expanduser().resolve() if config_yml is not None else None
        )
        if self.config_yml is not None and not self.config_yml.exists():
            raise FileNotFoundError(f"Config does not exist: {self.config_yml}")

        self.device = device
        self.seed = int(seed)
        self.only_output = only_output
        self.disable_amp = disable_amp
        self.quiet_model_load = (
            _env_flag("CONSTRDIFF_QUIET_MODEL_LOAD", default=False)
            if quiet_model_load is None
            else bool(quiet_model_load)
        )

        default_dtype = torch.get_default_dtype()
        try:
            # OCP/Equiformer constructs some runtime buffers with bare
            # `torch.tensor(...)` calls, so we force the expected float32 default
            # during calculator setup to avoid inheriting a float64 default from
            # other model loaders such as MACE.
            torch.set_default_dtype(torch.float32)
            with contextlib.ExitStack() as quiet_stack:
                if self.quiet_model_load:
                    devnull = quiet_stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
                    quiet_stack.enter_context(contextlib.redirect_stdout(devnull))
                self.calculator = OCPCalculator(
                    config_yml=str(self.config_yml) if self.config_yml is not None else None,
                    checkpoint_path=self.checkpoint_path,
                    cpu=_resolve_cpu_flag(device),
                    seed=self.seed,
                    only_output=self.only_output,
                    disable_amp=self.disable_amp,
                )
        finally:
            torch.set_default_dtype(default_dtype)
        self.implemented_properties = set(self.calculator.implemented_properties)
        if self.only_output is not None:
            # OCPCalculator may advertise its class-wide property set even when
            # only_output deliberately disables one of those properties.
            self.implemented_properties.intersection_update(self.only_output)

    def relax_atoms(
        self,
        atoms: Atoms,
        *,
        optimizer: OptimizerName = "FIRE",
        force_max: float = 0.001,
        max_steps: int = 300,
        relax_cell: bool = False,
        cell_filter: CellFilterName = "frechet",
        fix_symmetry: bool = False,
        optimizer_kwargs: dict[str, Any] | None = None,
        trajectory_path: str | Path | None = None,
        log_path: str | Path | None = None,
        output_path: str | Path | None = None,
        summary_path: str | Path | None = None,
        input_path: str | Path | None = None,
    ) -> EquiformerRelaxationResult:
        if optimizer not in OPTIMIZER_CLS:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Choose from {sorted(OPTIMIZER_CLS)}")

        if "energy" not in self.implemented_properties or "forces" not in self.implemented_properties:
            raise RuntimeError(
                "The checkpoint does not expose both energy and forces, which are required for ASE relaxation. "
                f"Implemented properties: {sorted(self.implemented_properties)}"
            )

        if relax_cell and "stress" not in self.implemented_properties:
            raise RuntimeError(
                "Cell relaxation requires stress predictions, but this checkpoint does not expose `stress`."
            )

        atoms_to_relax = atoms.copy()
        atoms_to_relax.calc = self.calculator

        if fix_symmetry:
            atoms_to_relax.set_constraint(FixSymmetry(atoms_to_relax))

        relax_target: Any = atoms_to_relax
        if relax_cell:
            relax_target = CELL_FILTER_CLS[cell_filter](atoms_to_relax)

        optimizer_kwargs = _optimizer_kwargs(optimizer, optimizer_kwargs)
        traj_arg = None
        if trajectory_path is not None:
            traj_arg = str(Path(trajectory_path).expanduser().resolve())

        logfile_arg = None
        if log_path is not None:
            logfile_arg = str(Path(log_path).expanduser().resolve())
            Path(logfile_arg).parent.mkdir(parents=True, exist_ok=True)

        if output_path is not None:
            Path(output_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        if summary_path is not None:
            Path(summary_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        if traj_arg is not None:
            Path(traj_arg).parent.mkdir(parents=True, exist_ok=True)

        optimizer_cls = OPTIMIZER_CLS[optimizer]
        optimizer_instance = optimizer_cls(
            relax_target,
            trajectory=traj_arg,
            logfile=logfile_arg,
            **optimizer_kwargs,
        )

        start_time = time.perf_counter()
        optimizer_instance.run(fmax=float(force_max), steps=int(max_steps))
        wall_time_s = time.perf_counter() - start_time

        relaxed_atoms = _unwrap_atoms(relax_target)
        relaxed_atoms_with_results, energy, forces, stress = _attach_final_results(
            relaxed_atoms,
            include_stress=relax_cell,
        )

        force_norms = np.linalg.norm(forces, axis=1) if len(forces) else np.zeros((0,), dtype=np.float64)
        max_force = float(force_norms.max()) if force_norms.size else 0.0
        mean_force = float(force_norms.mean()) if force_norms.size else 0.0
        steps_taken = int(getattr(optimizer_instance, "nsteps", max_steps))
        reached_max_steps = steps_taken >= int(max_steps)
        converged = max_force <= float(force_max)

        relaxed_atoms_with_results.info.update(
            {
                "equiformer_checkpoint": str(self.checkpoint_path),
                "equiformer_optimizer": optimizer,
                "equiformer_force_max": float(force_max),
                "equiformer_max_steps": int(max_steps),
                "equiformer_steps_taken": steps_taken,
                "equiformer_relax_cell": bool(relax_cell),
                "equiformer_fixed_symmetry": bool(fix_symmetry),
                "equiformer_final_energy_ev": energy,
                "equiformer_final_max_force_ev_per_ang": max_force,
                "equiformer_final_mean_force_ev_per_ang": mean_force,
            }
        )

        result = EquiformerRelaxationResult(
            relaxed_atoms=relaxed_atoms_with_results,
            checkpoint_path=self.checkpoint_path,
            config_yml=self.config_yml,
            input_path=Path(input_path).expanduser().resolve() if input_path is not None else None,
            output_path=Path(output_path).expanduser().resolve() if output_path is not None else None,
            trajectory_path=Path(traj_arg) if traj_arg is not None else None,
            log_path=Path(logfile_arg) if logfile_arg is not None else None,
            optimizer=optimizer,
            force_max=float(force_max),
            max_steps=int(max_steps),
            steps_taken=steps_taken,
            reached_max_steps=reached_max_steps,
            converged=converged,
            relax_cell=bool(relax_cell),
            cell_filter=cell_filter if relax_cell else None,
            fixed_symmetry=bool(fix_symmetry),
            wall_time_s=wall_time_s,
            final_energy_ev=energy,
            final_forces_ev_per_ang=forces,
            final_stress_ev_per_ang3=stress,
            final_max_force_ev_per_ang=max_force,
            final_mean_force_ev_per_ang=mean_force,
        )

        if output_path is not None:
            ase_write(str(result.output_path), result.relaxed_atoms, format=None)
        if summary_path is not None:
            Path(summary_path).write_text(json.dumps(result.to_summary(), indent=2), encoding="utf-8")

        return result

    def evaluate_singlepoint(self, atoms: Atoms) -> dict[str, Any]:
        if "energy" not in self.implemented_properties or "forces" not in self.implemented_properties:
            raise RuntimeError(
                "The checkpoint does not expose both energy and forces, which are required for single-point evaluation. "
                f"Implemented properties: {sorted(self.implemented_properties)}"
            )

        atoms_eval = atoms.copy()
        atoms_eval.calc = self.calculator
        energy = float(atoms_eval.get_potential_energy())
        forces = np.asarray(atoms_eval.get_forces(), dtype=np.float64)
        result: dict[str, Any] = {
            "energy_ev": energy,
            "forces": forces,
            "status": "ok",
            "error": "",
        }
        if "stress" in self.implemented_properties:
            try:
                result["stress"] = np.asarray(
                    atoms_eval.get_stress(voigt=False), dtype=np.float64
                )
            except PropertyNotImplementedError:
                # Some FairChem calculators advertise stress globally although
                # a particular checkpoint/config does not produce it. Stress is
                # optional for a position-only single-point evaluation.
                self.implemented_properties.discard("stress")
        result.update(_summarize_force_array(forces))
        return result

    def relax_file(
        self,
        input_path: str | Path,
        *,
        frame_index: int = 0,
        **kwargs: Any,
    ) -> EquiformerRelaxationResult:
        atoms = _load_atoms(input_path, frame_index=frame_index)
        return self.relax_atoms(atoms, input_path=input_path, **kwargs)


def relax_structure_file(
    input_path: str | Path,
    checkpoint_path: str | Path,
    *,
    config_yml: str | Path | None = None,
    frame_index: int = 0,
    device: str = "auto",
    seed: int = 0,
    optimizer: OptimizerName = "FIRE",
    force_max: float = 0.001,
    max_steps: int = 300,
    relax_cell: bool = False,
    cell_filter: CellFilterName = "frechet",
    fix_symmetry: bool = False,
    optimizer_kwargs: dict[str, Any] | None = None,
    trajectory_path: str | Path | None = None,
    log_path: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> EquiformerRelaxationResult:
    relaxer = EquiformerRelaxer(
        checkpoint_path=checkpoint_path,
        config_yml=config_yml,
        device=device,
        seed=seed,
    )
    return relaxer.relax_file(
        input_path,
        frame_index=frame_index,
        optimizer=optimizer,
        force_max=force_max,
        max_steps=max_steps,
        relax_cell=relax_cell,
        cell_filter=cell_filter,
        fix_symmetry=fix_symmetry,
        optimizer_kwargs=optimizer_kwargs,
        trajectory_path=trajectory_path,
        log_path=log_path,
        output_path=output_path,
        summary_path=summary_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relax one ASE-readable structure with an Equiformer/OCP checkpoint and "
            "write the final relaxed frame."
        )
    )
    parser.add_argument("--input_path", required=True, help="Input structure file, e.g. a multi-frame .extxyz trajectory.")
    parser.add_argument("--checkpoint_path", required=True, help="Equiformer checkpoint to load with fairchem OCPCalculator.")
    parser.add_argument("--config_yml", default=None, help="Optional config yaml. Usually the checkpoint config is enough.")
    parser.add_argument("--frame_index", type=int, default=0, help="Frame index to read from the input file. Use 0 for trajectory frame 0.")
    parser.add_argument("--output_path", default=None, help="Optional output structure path, e.g. relaxed.extxyz.")
    parser.add_argument("--summary_path", default=None, help="Optional JSON summary path.")
    parser.add_argument("--trajectory_path", default=None, help="Optional ASE trajectory path, typically ending in .traj.")
    parser.add_argument("--log_path", default=None, help="Optional ASE optimizer log path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA-style request. Use CUDA_VISIBLE_DEVICES to pick a GPU.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed passed into OCPCalculator.")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZER_CLS), default="FIRE")
    parser.add_argument("--force_max", type=float, default=0.001, help="ASE stopping threshold on the Equiformer-predicted maximum atomic force in eV/Ang.")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum number of relaxation steps.")
    parser.add_argument("--relax_cell", action=argparse.BooleanOptionalAction, default=False, help="Relax the cell as well as atomic positions. Requires a stress-predicting checkpoint.")
    parser.add_argument("--cell_filter", choices=sorted(CELL_FILTER_CLS), default="frechet", help="Cell filter to use when --relax_cell is enabled.")
    parser.add_argument("--fix_symmetry", action=argparse.BooleanOptionalAction, default=False, help="Apply ASE FixSymmetry before relaxation.")
    parser.add_argument(
        "--optimizer_kwargs_json",
        default=None,
        help="Optional JSON dictionary of extra ASE optimizer keyword arguments.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    optimizer_kwargs = json.loads(args.optimizer_kwargs_json) if args.optimizer_kwargs_json else None
    output_path = args.output_path
    summary_path = args.summary_path
    if output_path is None:
        input_path = Path(args.input_path).expanduser().resolve()
        output_path = str(input_path.with_name(f"{input_path.stem}_equiformer_relaxed.extxyz"))
    if summary_path is None:
        summary_path = str(Path(output_path).with_suffix(".json"))
    result = relax_structure_file(
        input_path=args.input_path,
        checkpoint_path=args.checkpoint_path,
        config_yml=args.config_yml,
        frame_index=args.frame_index,
        device=args.device,
        seed=args.seed,
        optimizer=args.optimizer,
        force_max=args.force_max,
        max_steps=args.max_steps,
        relax_cell=args.relax_cell,
        cell_filter=args.cell_filter,
        fix_symmetry=args.fix_symmetry,
        optimizer_kwargs=optimizer_kwargs,
        trajectory_path=args.trajectory_path,
        log_path=args.log_path,
        output_path=output_path,
        summary_path=summary_path,
    )
    print(json.dumps(result.to_summary(), indent=2))


if __name__ == "__main__":
    main()
