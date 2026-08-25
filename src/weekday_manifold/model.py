"""Model loading helpers built on TransformerLens."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def get_device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_name: str = "meta-llama/Llama-3.1-8B",
    device: str | None = None,
    **kwargs,
) -> HookedTransformer:
    """Load a HookedTransformer for interpretability work."""
    device = device or get_device()
    model = HookedTransformer.from_pretrained(model_name, device=device, **kwargs)
    model.eval()
    return model
