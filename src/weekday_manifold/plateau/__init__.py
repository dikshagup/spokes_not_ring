"""Layer resolution and model loading -- the seam shared with the capture phase."""

from __future__ import annotations

from weekday_manifold.plateau.config import (
    LAYER_PRESETS,
    PlateauConfig,
    hook_name_for_layer,
    resolve_layer_index,
)

__all__ = [
    "PlateauConfig",
    "LAYER_PRESETS",
    "resolve_layer_index",
    "hook_name_for_layer",
]
