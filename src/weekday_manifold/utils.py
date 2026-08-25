"""Shared utilities — reproducibility helpers."""

from __future__ import annotations

import datetime
import random
import re

import numpy as np


def model_short_name(model_name: str) -> str:
    """Filesystem-safe short model id."""
    short = str(model_name).split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", short).strip("-") or "model"


def run_slug(run_name: str, model_name: str, with_date: bool = True) -> str:
    """Tag a run name with the model (and ISO date) so outputs are not clobbered."""
    parts = [run_name, model_short_name(model_name)]
    if with_date:
        parts.append(datetime.date.today().isoformat())
    return "_".join(parts)


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    # imported here, not at module scope: seeding is the ONLY thing in this module that
    # needs torch, and a top-level import made every pure-numpy consumer pay ~20 s for it.
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
