"""Bi2Te3 defect relaxation-trajectory generation."""

import os
from pathlib import Path

# Fine-tuned uMLIP checkpoints (git-ignored; copied between clusters with rsync).
CKPT_DIR = Path(
    os.environ.get("RELAXGEN_CKPT_DIR", Path(__file__).resolve().parents[1] / "ft_ckpts")
)
