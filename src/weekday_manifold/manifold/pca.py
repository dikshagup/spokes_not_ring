"""PCA via NumPy SVD, with the component count capped to what the prompt set can support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def cap_components(n_components: Optional[int], n_samples: int, n_features: int) -> int:
    """Largest valid component count: ``min(n_components, n_samples-1, n_features)``."""
    max_valid = max(1, min(n_samples - 1, n_features))
    if n_components is None:
        return int(max_valid)
    return int(max(1, min(n_components, n_samples - 1, n_features)))


@dataclass
class PCA:
    """A fitted PCA: mean + orthonormal principal axes (rows of ``components``)."""

    mean: np.ndarray                 # [d_model]
    components: np.ndarray           # [k, d_model], orthonormal rows
    explained_variance: np.ndarray   # [k], per-PC variance (eigenvalues)
    explained_variance_ratio: np.ndarray  # [k], fraction of total variance

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project ``X`` ([N, d_model] or [d_model]) into the k-dim PCA space."""
        X = np.asarray(X, dtype=float)
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct ``d_model``-space points from PCA coordinates ``Z``."""
        Z = np.asarray(Z, dtype=float)
        return Z @ self.components + self.mean


def fit_pca(X: np.ndarray, n_components: Optional[int] = None) -> PCA:
    """Fit PCA on ALL rows of ``X`` and keep the top ``n_components`` axes."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"fit_pca expects a 2D [n, d] array, got shape {X.shape}.")
    n_samples, n_features = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean
    # Thin SVD: Xc = U @ diag(S) @ Vt; rows of Vt are the principal axes.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = cap_components(n_components, n_samples, n_features)
    components = Vt[:k]
    # Sample variance along each PC (ddof=1 to match an unbiased estimator;
    # falls back to ddof=0 when there is a single sample).
    denom = max(1, n_samples - 1)
    var_all = (S ** 2) / denom
    total = var_all.sum()
    ratio_all = var_all / total if total > 0 else np.zeros_like(var_all)
    return PCA(
        mean=mean,
        components=components,
        explained_variance=var_all[:k],
        explained_variance_ratio=ratio_all[:k],
    )


def fit_discriminative_pca(
    X: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
    n_components: Optional[int] = None,
) -> PCA:
    """PCA on the CLASS CENTROIDS — the day-discriminative (ring) subspace."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"fit_discriminative_pca expects [n, d], got {X.shape}.")
    mean = X.mean(axis=0)
    cents = group_centroids(X - mean, np.asarray(labels), n_groups)  # [G, d] centered
    _, S, Vt = np.linalg.svd(cents, full_matrices=False)
    # centroids span at most n_groups-1 dims; also bound by n_features.
    k = cap_components(n_components, n_samples=n_groups, n_features=X.shape[1])
    components = Vt[:k]
    denom = max(1, n_groups - 1)
    var_all = (S ** 2) / denom
    total = var_all.sum()
    ratio_all = var_all / total if total > 0 else np.zeros_like(var_all)
    return PCA(
        mean=mean,
        components=components,
        explained_variance=var_all[:k],
        explained_variance_ratio=ratio_all[:k],
    )


def group_centroids(Z: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
    """Mean of ``Z`` rows within each integer label group, in label order."""
    Z = np.asarray(Z, dtype=float)
    labels = np.asarray(labels)
    centroids = np.empty((n_groups, Z.shape[1]), dtype=float)
    for g in range(n_groups):
        rows = Z[labels == g]
        if rows.shape[0] == 0:
            raise ValueError(f"group {g} has no members; every day needs >=1 prompt.")
        centroids[g] = rows.mean(axis=0)
    return centroids
