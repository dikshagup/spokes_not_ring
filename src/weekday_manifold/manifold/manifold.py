"""The activation manifold: PCA, per-day centroids, and the periodic spline through them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from weekday_manifold.manifold.geometry import (
    check_cyclic_order,
    circularity_stats,
    variance_profile,
)
from weekday_manifold.manifold.pca import PCA, fit_discriminative_pca, fit_pca, group_centroids
from weekday_manifold.manifold.spline import PeriodicSpline


@dataclass
class Manifold:
    """A fitted closed curve through the per-day centroids in 64D PCA space."""

    pca: PCA
    centroids: np.ndarray              # [K, k] in true label order (64D space)
    spline: PeriodicSpline
    day_order: List[int]               # spline traversal order (indices into labels)
    labels: List[str]                  # class names aligned to centroid rows

    # ---- coordinate changes (seams) ---------------------------------------
    def to_pca(self, activation: np.ndarray) -> np.ndarray:
        """d_model residual vector(s) -> 64D PCA coordinates."""
        return self.pca.transform(activation)

    def to_activation(self, z64: np.ndarray) -> np.ndarray:
        """64D PCA coordinates -> d_model residual vector(s) (PCA inverse)."""
        return self.pca.inverse_transform(z64)

    # ---- curve maps --------------------------------------------------------
    def forward(self, u) -> np.ndarray:
        """Intrinsic u∈[0,1) -> point on the manifold in 64D PCA space."""
        return self.spline.forward(u)

    def inverse(self, z64) -> np.ndarray:
        """64D PCA point(s) -> nearest intrinsic coordinate u on the curve."""
        return self.spline.inverse(z64)

    def forward_full(self, u) -> np.ndarray:
        """Intrinsic u -> full d_model residual vector (steering seam)."""
        return self.to_activation(self.forward(u))

    def tangent(self, u) -> np.ndarray:
        """Curve tangent dvec/du in 64D (metric/Jacobian seam)."""
        return self.spline.derivative(u)

    @property
    def knot_coords(self) -> np.ndarray:
        """Intrinsic coordinate of each centroid (its spline knot)."""
        return self.spline.knots

    @property
    def n_pca_dims(self) -> int:
        return self.pca.n_components


def fit_manifold(
    activations: np.ndarray,
    labels: Sequence[int],
    n_labels: Optional[int] = None,
    label_names: Optional[Sequence[str]] = None,
    n_pca_dims: Optional[int] = None,
    day_order: Optional[Sequence[int]] = None,
    subspace: str = "variance",
) -> Manifold:
    """Fit ℳ_h: PCA(all activations)->64D, per-class centroids, periodic spline."""
    activations = np.asarray(activations, dtype=float)
    labels = np.asarray(labels)
    if n_labels is None:
        n_labels = int(labels.max()) + 1
    if label_names is None:
        label_names = [str(i) for i in range(n_labels)]
    order = list(range(n_labels)) if day_order is None else list(day_order)

    if subspace == "discriminative":
        pca = fit_discriminative_pca(activations, labels, n_labels, n_pca_dims)
    elif subspace == "variance":
        pca = fit_pca(activations, n_components=n_pca_dims)
    else:
        raise ValueError(f"subspace must be 'variance' or 'discriminative', got {subspace!r}.")
    Z = pca.transform(activations)                      # [N, k]
    centroids = group_centroids(Z, labels, n_labels)    # [K, k] true label order
    # Thread the spline in the requested traversal order, then store centroids in
    # true label order so row i is always class i.
    spline = PeriodicSpline(centroids[order])
    return Manifold(
        pca=pca,
        centroids=centroids,
        spline=spline,
        day_order=order,
        labels=list(label_names),
    )


def manifold_report(manifold: Manifold) -> dict:
    """All 64D geometry results for a fitted manifold (the activation-only stats)."""
    order = check_cyclic_order(manifold.centroids)
    profile = variance_profile(manifold.centroids)
    circ = circularity_stats(manifold.centroids, order.recovered)
    return {
        "cyclic_order_recovery": order.to_dict(),
        "recovered_day_names": [manifold.labels[i] for i in order.recovered],
        "variance_profile": profile,
        "circularity": circ,
        "n_pca_dims": manifold.n_pca_dims,
    }
