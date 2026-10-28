from __future__ import annotations

import io
import pickle
from functools import lru_cache
from typing import Any

import torch


def _load_legacy_codegen_module(buffer: bytes) -> torch.nn.Module:
    stream = io.BytesIO(buffer)
    if buffer.startswith(b"PK"):
        try:
            return torch.jit.load(stream)
        except Exception:
            stream.seek(0)
    return torch.load(stream, map_location="cpu", weights_only=False)


def _restore_codegen_module(value: Any) -> torch.nn.Module:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
        buffer_type, buffer = value
        if not isinstance(buffer, bytes):
            raise TypeError(
                f"Unsupported e3nn codegen buffer type: {type(buffer)}"
            )
        if buffer_type == "fx":
            return pickle.loads(buffer)
        if buffer_type == "torchscript":
            return torch.jit.load(io.BytesIO(buffer))
        raise NotImplementedError(
            f"Unsupported e3nn codegen buffer type tag: {buffer_type}"
        )

    # Older MACE/e3nn checkpoints stored the compiled FX module as a raw
    # latin1-encoded torch archive rather than a tagged (buffer_type, buffer)
    # tuple. Newer e3nn expects the tagged tuple, so we normalize here.
    if isinstance(value, str):
        value = value.encode("latin1")
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return _load_legacy_codegen_module(value)

    raise TypeError(
        f"Unsupported legacy e3nn codegen payload type: {type(value)}"
    )


def ensure_legacy_mace_codegen_compat() -> None:
    """
    Patch e3nn's CodeGenMixin loader so old MACE checkpoints can be restored
    under newer e3nn versions.

    MACE checkpoints trained against older e3nn releases can serialize
    ``__codegen__`` entries as raw bytes instead of the modern
    ``(buffer_type, buffer)`` tuple expected by e3nn>=0.5. This shim accepts
    both formats and is safe to call multiple times.
    """

    try:
        from e3nn.util.codegen._mixin import CodeGenMixin
    except ImportError:
        return

    current_setstate = CodeGenMixin.__setstate__
    if getattr(current_setstate, "_constrdiff_legacy_mace_compat", False):
        return

    def compat_setstate(self, d) -> None:
        d = d.copy()
        codegen_state = d.pop("__codegen__", None)

        if hasattr(super(CodeGenMixin, self), "__setstate__"):
            super(CodeGenMixin, self).__setstate__(d)
        else:
            self.__dict__.update(d)

        if codegen_state is not None:
            restored_names: list[str] = []
            for fname, value in codegen_state.items():
                module = _restore_codegen_module(value)
                setattr(self, fname, module)
                restored_names.append(fname)
            self.__codegen__ = restored_names

    compat_setstate._constrdiff_legacy_mace_compat = True
    compat_setstate._constrdiff_legacy_mace_original = current_setstate
    CodeGenMixin.__setstate__ = compat_setstate


@lru_cache(maxsize=4)
def _resolve_spherical_harmonics_callable() -> Any:
    from e3nn import get_optimization_defaults
    from e3nn.o3._spherical_harmonics import _spherical_harmonics

    defaults = get_optimization_defaults()
    jit_mode = defaults.get("jit_mode")
    if jit_mode == "script":
        return torch.jit.script(_spherical_harmonics)
    if jit_mode == "inductor":
        return torch.compile(_spherical_harmonics, fullgraph=True)
    if jit_mode is not None:
        return _spherical_harmonics
    if defaults.get("jit_script_fx", False):
        return torch.jit.script(_spherical_harmonics)
    return _spherical_harmonics


def _repair_spherical_harmonics_module(module: Any) -> None:
    if hasattr(module, "sph_func"):
        return

    ls_list = getattr(module, "_ls_list", None)
    if ls_list is None:
        irreps_out = getattr(module, "irreps_out", None)
        if irreps_out is not None:
            ls_list = [ir.l for mul, ir in irreps_out for _ in range(mul)]
            module._ls_list = ls_list

    if not hasattr(module, "_lmax") and ls_list:
        module._lmax = max(ls_list)
    if not hasattr(module, "_is_range_lmax") and ls_list:
        module._is_range_lmax = ls_list == list(range(max(ls_list) + 1))
    if not hasattr(module, "_prof_str") and ls_list is not None:
        module._prof_str = f"spherical_harmonics({ls_list})"

    module.sph_func = _resolve_spherical_harmonics_callable()


def _repair_activation_module(module: Any) -> None:
    if hasattr(module, "paths"):
        return

    irreps_in = getattr(module, "irreps_in", None)
    acts = getattr(module, "acts", None)
    if irreps_in is None or acts is None:
        return

    module.paths = [
        (mul, (ir.l, ir.p), act)
        for (mul, ir), act in zip(irreps_in, acts)
    ]


def patch_mace_calculator_runtime_dtype(calc: Any) -> None:
    """
    Keep MACE batch construction and forward passes on the calculator's native
    dtype without leaking a process-wide default dtype change into other models.

    MACE's ASE calculator uses ``torch.get_default_dtype()`` during
    ``calculate()`` to build one-hot node attributes, positions, cells, and
    several temporary tensors. If we restore the global default dtype after
    calculator construction, those runtime tensors can fall back to float32
    while the loaded model remains float64, causing matmul failures inside
    ``AtomicEnergiesBlock``.
    """

    if getattr(calc, "_constrdiff_runtime_dtype_patched", False):
        return

    try:
        from mace.calculators.mace import get_model_dtype
        from mace.tools.torch_tools import default_dtype as mace_default_dtype
    except ImportError:
        return

    if not getattr(calc, "models", None):
        return

    runtime_dtype = get_model_dtype(calc.models[0])
    original_calculate = calc.calculate

    def compat_calculate(*args: Any, **kwargs: Any) -> Any:
        with mace_default_dtype(runtime_dtype):
            return original_calculate(*args, **kwargs)

    calc.calculate = compat_calculate
    calc._constrdiff_runtime_dtype_patched = True
    calc._constrdiff_runtime_dtype = runtime_dtype


def repair_legacy_e3nn_modules(module: torch.nn.Module) -> None:
    """
    Rehydrate runtime-only e3nn fields that older MACE checkpoints can miss
    after unpickling under newer e3nn releases.

    In particular, some modern e3nn modules rebuild runtime-only helpers inside
    ``__init__`` and their ``forward()`` methods always expect them to exist.
    Older pickled checkpoints can restore the persistent module state without
    recreating these helpers, so we rebuild them here after model load.
    """

    try:
        from e3nn.nn._activation import Activation
        from e3nn.o3._spherical_harmonics import SphericalHarmonics
    except ImportError:
        return

    for child in module.modules():
        if isinstance(child, SphericalHarmonics):
            _repair_spherical_harmonics_module(child)
        elif isinstance(child, Activation):
            _repair_activation_module(child)
