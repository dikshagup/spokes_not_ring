"""Self-describing configuration for the activation-manifold phase.

Mirrors ``weekday_manifold.plateau.config.PlateauConfig`` (dumped as JSON next to every
output so runs are reproducible) and REUSES its layer-resolution helpers so the
two phases agree on what ``blocks.{L}.hook_resid_post`` means. The only model-
processing knobs are pinned to the same Phase-1 choices (fp32, TL default
processing flags) because the manifold lives in the residual stream and must be
captured in the same fixed space.

Capture layer differs from Phase 1's DEFAULT: the manifold result wants a LATE
layer. The shipped config pins ``layer_index = 28`` of Llama-3.1-8B's 32 blocks
(0.875 of depth) directly, which is what every experiment here uses; the
fractional presets below are only for sweeping other depths. A small layer sweep
is supported by the runner so the clearest ring can be picked.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Reuse Phase-1 layer resolution so both phases name the same hook.
from weekday_manifold.plateau.config import hook_name_for_layer, resolve_layer_index

# Late-layer presets as FRACTION OF DEPTH, so they transfer across model sizes.
# Resolved against Llama-3.1-8B's 32 blocks, which is the only model this repo uses.
# NOTE none of these is the paper-matched layer: that is block 28 = 0.875 of depth,
# and the shipped config pins it as layer_index rather than going through a preset.
LATE_LAYER_PRESETS: Dict[str, float] = {
    "late_70": 0.70,   # -> block 22 of 32
    "late_78": 0.78,   # -> block 25 of 32
    "late_85": 0.85,   # -> block 27 of 32
}

# Default small sweep so the user can pick the layer with the clearest ring.
DEFAULT_LAYER_SWEEP: List[int] = [30, 34, 37, 41, 44]


@dataclass
class ManifoldConfig:
    """One activation-manifold run, fully described for JSON dump + reproduction."""

    # --- Model / processing (pinned to Phase 1) -----------------------------
    model_name: str = "meta-llama/Llama-3.1-8B"
    dtype: str = "float32"
    device: Optional[str] = None        # None -> auto (cuda > mps > cpu)
    fold_ln: bool = True
    center_writing_weights: bool = True
    center_unembed: bool = True
    prepend_bos: bool = True

    # --- Capture layer (precedence: layer_index > layer_frac > layer_preset) -
    layer_preset: Optional[str] = "late_78"  # used only when layer_index is None
    layer_frac: Optional[float] = None       # used only when layer_index is None
    layer_index: Optional[int] = None        # explicit override; resolve() fills it

    # --- Task / prompts -----------------------------------------------------
    # Formulation: arith | interrogative | seq | relational | mention | trailing.
    # Default = "trailing": full-sentence prompts with the day fronted ("On
    # Tuesday, I went to see a movie.") captured at the sentence-final "."
    # token. Chosen to (a) remove cross-day confusion (no other weekday appears
    # in the prompt) and (b) probe the sentence-summary position that LMs tend
    # to gather clause meaning on. Representation-only, like ``mention`` — the
    # day is GIVEN, not predicted, so competence scoring does not apply.
    formulation: str = "trailing"
    offset_style: str = "digit"          # arith only: digit ("3") | word ("three")
    # seq only: which consecutive-run lengths and surface separators to build.
    # Default = lengths 2-6, both separators = 5x7x2 = 70 prompts (~10/day) so the
    # per-day centroids average over enough samples to be robust to noise. The PCA
    # below is auto-capped to min(n_pca_dims, n_prompts-1) (pca.cap_components).
    seq_run_lengths: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    seq_separators: List[str] = field(default_factory=lambda: ["plain", "comma"])
    # INERT ON THIS BRANCH -- nothing reads it. Kept because all four configs in
    # configs/ set it and dropping the field would make them fail to load, but do not
    # reason from its value: three of them say "input", which is not even one of the two
    # values documented below, and changing any of them changes no output.
    #
    # It is also not the same knob as alpha_ladder_sites.py's --group-by, which takes
    # answer|input|auto and IS read. Same name, different vocabulary, different consumer.
    # If grouping is ever wired back into the capture path, reconcile the two before
    # trusting either.
    #
    # As designed: group prompts by the CORRECT day (the stem's intended answer) or by the
    # day the model actually ELICITS. "correct" is the default; "elicited" lets
    # the ring form around what the model represents even when it answers wrong.
    group_by: str = "correct"            # correct | elicited

    # --- Manifold fit -------------------------------------------------------
    # PCA target dims. Paper uses 64 — we match that. cap_components() clamps to
    # min(n_pca_dims, n_prompts-1, d_model) so a small prompt set is still valid
    # (with the 70-prompt trailing default that cap is 64; the ring only needs
    # <=6 dims since 7 centroids span at most 6, so 64 is ample). Requesting
    # MORE components than n_prompts-1 is not meaningful — see the note in
    # pca.cap_components — so raise n_pca_dims only after growing the prompt set.
    n_pca_dims: Optional[int] = 64
    day_scoring: str = "fullstring"      # fullstring (teacher-forced) | firsttoken
    plot_3d_dims: int = 3                # PCA-projection dims for the figure

    # --- Bookkeeping --------------------------------------------------------
    seed: int = 0
    cache_dir: str = "data/activations"
    out_dir: str = "experiments/results/manifold"
    notes: str = ""

    # ------------------------------------------------------------------ utils
    def resolve(self, n_layers: int) -> "ManifoldConfig":
        """Fill in the concrete ``layer_index`` against the model depth."""
        frac = self.layer_frac
        if frac is None and self.layer_index is None and self.layer_preset is not None:
            if self.layer_preset not in LATE_LAYER_PRESETS:
                raise ValueError(
                    f"Unknown layer_preset {self.layer_preset!r}; "
                    f"choose from {sorted(LATE_LAYER_PRESETS)}."
                )
            frac = LATE_LAYER_PRESETS[self.layer_preset]
        idx = resolve_layer_index(
            n_layers=n_layers,
            layer_frac=frac,
            layer_preset=None,
            layer_index=self.layer_index,
        )
        return dataclasses.replace(self, layer_index=idx)

    @property
    def hook_name(self) -> str:
        if self.layer_index is None:
            raise ValueError("layer_index unresolved; call config.resolve(n_layers).")
        return hook_name_for_layer(self.layer_index)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManifoldConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "ManifoldConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
