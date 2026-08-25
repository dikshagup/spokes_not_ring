"""The behaviour manifold: next-token distributions restricted to the seven days, in Hellinger space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from weekday_manifold.manifold.pca import group_centroids
from weekday_manifold.manifold.spline import PeriodicSpline


# --------------------------------------------------------------- pure helpers
def restrict_to_concept(
    vocab_probs: np.ndarray,
    concept_ids: Sequence[int],
) -> np.ndarray:
    """Restrict a vocab distribution to ``concept_ids`` + an "other" class."""
    vocab_probs = np.asarray(vocab_probs, dtype=float)
    ids = list(concept_ids)
    concept = vocab_probs[..., ids]                       # [..., K]
    other = 1.0 - concept.sum(axis=-1, keepdims=True)     # [..., 1]
    # Numerical guard: tiny negative "other" from float error -> clamp to 0.
    other = np.clip(other, 0.0, 1.0)
    return np.concatenate([concept, other], axis=-1)


def hellinger_embed(p: np.ndarray) -> np.ndarray:
    """Map a distribution (or centroid) into Hellinger space: ``p ↦ √p``."""
    return np.sqrt(np.clip(np.asarray(p, dtype=float), 0.0, None))


def hellinger_decode(y: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hellinger_embed`: ``y ↦ y²`` then renormalize to a dist."""
    y = np.asarray(y, dtype=float)
    p = y ** 2
    total = p.sum(axis=-1, keepdims=True)
    return np.divide(p, total, out=np.zeros_like(p), where=total > 0)


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two distributions ``p``, ``q``."""
    return float(np.linalg.norm(hellinger_embed(p) - hellinger_embed(q)) / np.sqrt(2.0))


# ------------------------------------------------------------- the manifold
@dataclass
class BehaviorManifold:
    """A closed curve through the per-day output-distribution centroids (ℳ_y)."""

    centroids_dist: np.ndarray         # [K, K+1] simplex, true day order
    centroids_hell: np.ndarray         # [K, K+1] √p embedding, true day order
    spline: PeriodicSpline             # periodic curve through centroids_hell[order]
    day_order: List[int]               # spline traversal order (indices into DAYS)
    labels: List[str]                  # day names aligned to centroid rows

    # ---- curve maps (Hellinger space) -------------------------------------
    def forward(self, u) -> np.ndarray:
        """Intrinsic u∈[0,1) -> point on ℳ_y in Hellinger (√p) space."""
        return self.spline.forward(u)

    def forward_dist(self, u) -> np.ndarray:
        """Intrinsic u -> the DECODED output distribution at that point."""
        return hellinger_decode(self.forward(u))

    def inverse(self, y) -> np.ndarray:
        """Hellinger point(s) -> nearest intrinsic coordinate u on the curve."""
        return self.spline.inverse(y)

    def tangent(self, u) -> np.ndarray:
        """Curve tangent dvec/du in Hellinger space (metric seam)."""
        return self.spline.derivative(u)

    def geodesic_matrix(self, n_samples: int = 2000) -> np.ndarray:
        """Day-indexed geodesic (along-curve Hellinger) distances ``[K, K]``."""
        Dk = self.spline.geodesic_matrix(n_samples=n_samples)
        pos = [self.day_order.index(a) for a in range(len(self.day_order))]
        return Dk[np.ix_(pos, pos)]

    @property
    def knot_coords(self) -> np.ndarray:
        return self.spline.knots


def fit_behavior_manifold(
    distributions: np.ndarray,
    labels: Sequence[int],
    n_labels: Optional[int] = None,
    label_names: Optional[Sequence[str]] = None,
    day_order: Optional[Sequence[int]] = None,
) -> BehaviorManifold:
    """Fit ℳ_y: per-class distribution centroids -> √p embed -> periodic spline."""
    distributions = np.asarray(distributions, dtype=float)
    labels = np.asarray(labels)
    if n_labels is None:
        n_labels = distributions.shape[-1] - 1
    if label_names is None:
        label_names = [str(i) for i in range(n_labels)]
    order = list(range(n_labels)) if day_order is None else list(day_order)

    centroids_dist = group_centroids(distributions, labels, n_labels)  # [K, K+1]
    centroids_hell = hellinger_embed(centroids_dist)
    spline = PeriodicSpline(centroids_hell[order])
    return BehaviorManifold(
        centroids_dist=centroids_dist,
        centroids_hell=centroids_hell,
        spline=spline,
        day_order=order,
        labels=list(label_names),
    )
