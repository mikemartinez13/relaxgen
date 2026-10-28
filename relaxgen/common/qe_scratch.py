"""Short-lived, collision-free Quantum ESPRESSO state directories."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def qe_scratch_base(base_dir: str | Path | None = None) -> Path | None:
    """Resolve an explicit/job-local QE scratch parent, or system temp."""

    value = base_dir or os.environ.get("QE_TMP_BASE_DIR") or os.environ.get(
        "ESPRESSO_TMPDIR"
    )
    if value is None:
        return None
    resolved = Path(value).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"QE scratch parent is not a directory: {resolved}")
    return resolved


@contextmanager
def temporary_qe_outdir(
    base_dir: str | Path | None = None,
    *,
    prefix: str = "qe_",
) -> Iterator[Path]:
    """Yield a unique QE ``control.outdir`` and remove it on every exit path."""

    base = qe_scratch_base(base_dir)
    with tempfile.TemporaryDirectory(
        prefix=prefix,
        dir=str(base) if base is not None else None,
    ) as directory:
        yield Path(directory)
