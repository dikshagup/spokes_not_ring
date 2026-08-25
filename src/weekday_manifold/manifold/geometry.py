"""Geometry of the manifold, computed in full dimensions rather than in a projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import List, Sequence

import numpy as np


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix ``[K, K]`` for ``points`` ``[K, D]``."""
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def minimum_spanning_tree_edges(points: np.ndarray) -> List[tuple]:
    """Edges of the minimum spanning tree over ``points`` (Euclidean, in 64D)."""
    from scipy.sparse.csgraph import minimum_spanning_tree

    points = np.asarray(points, dtype=float)
    dist = pairwise_distances(points)
    mst = minimum_spanning_tree(dist).toarray()
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(mst))]


def geodesic_distance_matrix(
    centroids: np.ndarray,
    day_order: Sequence[int] | None = None,
    n_samples: int = 2000,
) -> np.ndarray:
    """GEODESIC (along-spline) distance matrix between day centroids, day-indexed."""
    from weekday_manifold.manifold.spline import PeriodicSpline

    centroids = np.asarray(centroids, dtype=float)
    K = centroids.shape[0]
    order = list(range(K)) if day_order is None else list(day_order)
    spline = PeriodicSpline(centroids[order])
    Dk = spline.geodesic_matrix(n_samples=n_samples)  # indexed by knot position
    # Knot position k holds day order[k]; remap to a day-indexed matrix.
    pos = [order.index(a) for a in range(K)]
    return Dk[np.ix_(pos, pos)]


def cycle_length(order: Sequence[int], dist: np.ndarray) -> float:
    """Total length of the CLOSED tour visiting ``order`` then wrapping to start."""
    K = len(order)
    return float(sum(dist[order[i], order[(i + 1) % K]] for i in range(K)))


def recover_cyclic_order(points: np.ndarray) -> List[int]:
    """Minimum-perimeter Hamiltonian cycle through ``points`` (geometry only)."""
    points = np.asarray(points, dtype=float)
    K = points.shape[0]
    if K < 3:
        return list(range(K))
    dist = pairwise_distances(points)
    best_order: List[int] = list(range(K))
    best_len = np.inf
    for perm in permutations(range(1, K)):
        order = [0, *perm]
        length = cycle_length(order, dist)
        if length < best_len:
            best_len = length
            best_order = order
    return best_order


def canonical_cycles(order: Sequence[int]) -> List[tuple]:
    """All rotations of ``order`` and of its reflection, as tuples."""
    seq = list(order)
    K = len(seq)
    rots = [tuple(seq[i:] + seq[:i]) for i in range(K)]
    rev = list(reversed(seq))
    rots += [tuple(rev[i:] + rev[:i]) for i in range(K)]
    return rots


def cyclic_equivalent(a: Sequence[int], b: Sequence[int]) -> bool:
    """True iff cycles ``a`` and ``b`` match up to rotation and/or reflection."""
    if sorted(a) != sorted(b):
        return False
    return tuple(b) in set(canonical_cycles(a))


@dataclass
class OrderRecovery:
    """Result of recovering the centroids' cyclic order from geometry alone."""

    recovered: List[int]          # geometry-recovered visiting order (indices)
    true_order: List[int]         # the reference cycle (e.g. Mon..Sun = 0..6)
    passed: bool                  # recovered == true up to rotation/reflection
    recovered_length: float       # tour length of the recovered cycle
    true_length: float            # tour length following the true order

    def to_dict(self) -> dict:
        return {
            "recovered": list(self.recovered),
            "true_order": list(self.true_order),
            "passed": bool(self.passed),
            "recovered_length": float(self.recovered_length),
            "true_length": float(self.true_length),
        }


def check_cyclic_order(
    centroids: np.ndarray,
    true_order: Sequence[int] | None = None,
) -> OrderRecovery:
    """Recover the cyclic order from geometry and compare to ``true_order``."""
    centroids = np.asarray(centroids, dtype=float)
    K = centroids.shape[0]
    ref = list(range(K)) if true_order is None else list(true_order)
    dist = pairwise_distances(centroids)
    recovered = recover_cyclic_order(centroids)
    return OrderRecovery(
        recovered=recovered,
        true_order=ref,
        passed=cyclic_equivalent(recovered, ref),
        recovered_length=cycle_length(recovered, dist),
        true_length=cycle_length(ref, dist),
    )


def variance_profile(centroids: np.ndarray) -> dict:
    """PCA variance-explained profile of the centroids' spread (in 64D)."""
    centroids = np.asarray(centroids, dtype=float)
    Xc = centroids - centroids.mean(axis=0)
    S = np.linalg.svd(Xc, compute_uv=False)
    var = S ** 2
    total = var.sum()
    ratio = var / total if total > 0 else np.zeros_like(var)
    cumulative = np.cumsum(ratio)

    def pcs_for(frac: float) -> int:
        idx = np.searchsorted(cumulative, frac)
        return int(min(idx + 1, len(cumulative)))

    return {
        "explained_variance_ratio": ratio.tolist(),
        "cumulative_ratio": cumulative.tolist(),
        "n_pcs_90pct": pcs_for(0.90),
        "n_pcs_95pct": pcs_for(0.95),
        "n_pcs_99pct": pcs_for(0.99),
    }


def circularity_stats(centroids: np.ndarray, order: Sequence[int]) -> dict:
    """Spread of consecutive-centroid distances around ``order`` (circularity)."""
    centroids = np.asarray(centroids, dtype=float)
    K = len(order)
    steps = np.array(
        [
            np.linalg.norm(centroids[order[i]] - centroids[order[(i + 1) % K]])
            for i in range(K)
        ]
    )
    mean = float(steps.mean())
    std = float(steps.std())
    return {
        "neighbor_distances": steps.tolist(),
        "mean": mean,
        "std": std,
        "min": float(steps.min()),
        "max": float(steps.max()),
        "cv": float(std / mean) if mean > 0 else float("nan"),
    }
