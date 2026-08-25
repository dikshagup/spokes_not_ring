"""Configuration for the layer sweep, and the layer-resolution helpers the capture phase reuses."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Named presets, expressed as FRACTION OF DEPTH (block_output_index / n_layers).
#
# ``paper_early`` ~= 1/12: in GPT2-small the papers patch "after the first
# layer" (blocks.0.hook_resid_post == blocks.1.hook_resid_pre), i.e. the output
# of the 1st of 12 blocks, a depth fraction of ~0.083. We default to this for a
# faithful reproduction; the user can switch presets in config.
LAYER_PRESETS: Dict[str, float] = {
    "paper_early": 1.0 / 12.0,  # ~0.083 -> block 4 of 48 (faithful to the papers)
    "early_mid": 0.25,          # -> block 12 of 48
    "mid": 0.5,                 # -> block 24 of 48
}


def resolve_layer_index(
    n_layers: int,
    layer_frac: Optional[float] = None,
    layer_preset: Optional[str] = None,
    layer_index: Optional[int] = None,
) -> int:
    """Resolve the residual-stream block index to patch."""
    if layer_index is not None:
        idx = int(layer_index)
    else:
        if layer_frac is None:
            if layer_preset is None:
                raise ValueError(
                    "Provide one of layer_index, layer_frac, or layer_preset."
                )
            if layer_preset not in LAYER_PRESETS:
                raise ValueError(
                    f"Unknown layer_preset {layer_preset!r}; "
                    f"choose from {sorted(LAYER_PRESETS)}."
                )
            layer_frac = LAYER_PRESETS[layer_preset]
        idx = int(round(layer_frac * n_layers))
    # Clamp into range. resid_post is defined for every block, so the last
    # valid index is n_layers - 1.
    return max(0, min(n_layers - 1, idx))


def hook_name_for_layer(layer_index: int) -> str:
    """The TransformerLens hook we capture/patch. See module docstring."""
    return f"blocks.{layer_index}.hook_resid_post"


@dataclass
class PlateauConfig:
    """Self-describing configuration for one plateau sweep."""

    # --- Model / processing -------------------------------------------------
    model_name: str = "meta-llama/Llama-3.1-8B"
    dtype: str = "float32"          # float32 needed for Phase-2 Jacobians
    device: Optional[str] = None    # None -> auto (cuda > mps > cpu)

    # TransformerLens weight-processing flags. Pinned to TL defaults; L2 lives
    # in the resulting (centered-logit) space. KL is invariant to these.
    fold_ln: bool = True
    center_writing_weights: bool = True
    center_unembed: bool = True

    # --- Where to patch -----------------------------------------------------
    # Layer selection. Precedence: layer_index > layer_frac > layer_preset.
    # Default: layer_index=0 -> patch after the FIRST block
    # (blocks.0.hook_resid_post), the papers' LITERAL "after the first layer".
    # To select by fraction-of-depth instead, set layer_index=None and use
    # layer_preset (e.g. "paper_early" -> block 4) or layer_frac.
    layer_preset: Optional[str] = "paper_early"  # used only when layer_index is None
    layer_frac: Optional[float] = None           # used only when layer_index is None
    layer_index: Optional[int] = 0               # default; resolve() clamps to range

    # Where the OUTPUT DISTANCE is measured ("recording" point), independent of
    # the patch layer above:
    #   record_layer = -1    -> the LAST block's residual, resolved to
    #                           blocks.{n_layers-1}.hook_resid_post. THE DEFAULT,
    #                           matching the papers (2409.17113 reads the last-layer
    #                           resid; Shinkle reads at L+N). Negative values index
    #                           from the end and are resolved in resolve().
    #   record_layer = R>=0  -> residual at blocks.{R}.hook_resid_post (d_model
    #                           space), last position. Must be >= layer_index.
    #   record_layer = None  -> final-token output LOGITS (vocab space). The ONLY
    #                           setting where KL is defined (KL needs a
    #                           distribution); otherwise KL is NaN and L2 /
    #                           relative / shinkle / speed live in activation space.
    record_layer: Optional[int] = -1

    # Token position to patch & the BOS handling. ``token_position`` is either
    # the string "last" (use each prompt's final token) or an int interpreted
    # as a 0-based index into the *content* tokens (BOS excluded). With a fixed
    # content index, prompt-1 and prompt-2 must tokenize to equal length so the
    # index refers to the same logical token; this is asserted at run time.
    token_position: Union[str, int] = "last"
    prepend_bos: bool = True

    # --- Prompts ------------------------------------------------------------
    # PLACEHOLDER equal-length-ish factual pair. The user supplies real prompts
    # before the RunPod sweep. With token_position="last" the lengths need not
    # match (we patch prompt-2's last-token activation into prompt-1's last
    # position); for a fixed content index they must match.
    prompt_1: str = "The house was big"
    prompt_2: str = "The house was in"

    # --- Interpolation sweep ------------------------------------------------
    # a(t) = a1 + t * (a2 - a1). Default range includes both endpoints (t=0,
    # t=1) AND extrapolation beyond them. t=0 and t=1 are always force-included
    # in the grid (see interpolate.build_t_grid) — t=0 is the reference and t=1
    # is the d(1) normaliser.
    t_min: float = -0.
    t_max: float = 1.
    t_steps: int = 81  # uniform grid before 0/1 are force-merged in

    # Interpolation path between a1 and a2:
    #   "linear" -> a1 + t*(a2-a1)
    #   "slerp"  -> spherical direction + linearly-interpolated norm (Shinkle).
    # Does NOT affect the captured endpoints, so it's excluded from cache_key().
    interp_mode: str = "linear"

    # --- Bookkeeping --------------------------------------------------------
    seed: int = 0
    cache_dir: str = "data/activations"
    notes: str = ""

    # ------------------------------------------------------------------ utils
    def resolve(self, n_layers: int) -> "PlateauConfig":
        """Resolve ``layer_index`` and ``record_layer`` against model depth."""
        idx = resolve_layer_index(
            n_layers=n_layers,
            layer_frac=self.layer_frac,
            layer_preset=self.layer_preset,
            layer_index=self.layer_index,
        )
        rec = self.record_layer
        if rec is not None and rec < 0:
            rec = max(0, min(n_layers - 1, n_layers + rec))
        return dataclasses.replace(self, layer_index=idx, record_layer=rec)

    @property
    def hook_name(self) -> str:
        if self.layer_index is None:
            raise ValueError(
                "layer_index is unresolved; call config.resolve(n_layers) first."
            )
        return hook_name_for_layer(self.layer_index)

    @property
    def record_hook_name(self):
        """Hook to measure distance at, or None for the output logits."""
        if self.record_layer is None:
            return None
        if self.record_layer < 0:
            raise ValueError(
                "record_layer is negative/unresolved; call resolve(n_layers) first."
            )
        return hook_name_for_layer(self.record_layer)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        """Dump the run config next to outputs so results are self-describing."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlateauConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "PlateauConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def cache_key(self) -> str:
        """Stable key for the on-disk activation cache."""
        if self.layer_index is None:
            raise ValueError("Resolve layer_index before computing a cache key.")
        import hashlib

        payload = {
            "model_name": self.model_name,
            "dtype": self.dtype,
            "fold_ln": self.fold_ln,
            "center_writing_weights": self.center_writing_weights,
            "center_unembed": self.center_unembed,
            "layer_index": self.layer_index,
            "token_position": self.token_position,
            "prepend_bos": self.prepend_bos,
            "prompt_1": self.prompt_1,
            "prompt_2": self.prompt_2,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        digest = hashlib.sha256(blob).hexdigest()[:16]
        return f"{self.model_name.replace('/', '_')}_L{self.layer_index}_{digest}"
