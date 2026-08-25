"""Seeding: the same seed must give the same draws across numpy and torch."""
import numpy as np
import torch

from weekday_manifold.utils import set_seed


def test_set_seed_is_reproducible():
    set_seed(42)
    a_np, a_torch = np.random.rand(3), torch.rand(3)
    set_seed(42)
    b_np, b_torch = np.random.rand(3), torch.rand(3)

    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_torch, b_torch)
