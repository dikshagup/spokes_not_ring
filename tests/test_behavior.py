"""Behavior manifold ℳ_y — Hellinger embedding + fit recovers the cycle. No model."""

import numpy as np

from weekday_manifold.manifold.behavior import (
    BehaviorManifold,
    fit_behavior_manifold,
    hellinger_decode,
    hellinger_distance,
    hellinger_embed,
    restrict_to_concept,
)
from weekday_manifold.manifold.geometry import cyclic_equivalent, recover_cyclic_order


# ------------------------------------------------------------- restriction
def test_restrict_sums_to_one_and_other_is_remainder():
    vocab = np.array([0.1, 0.2, 0.05, 0.3, 0.35])  # sums to 1
    r = restrict_to_concept(vocab, [0, 3])          # keep ids 0 and 3
    assert r.shape == (3,)                           # 2 concepts + other
    assert np.isclose(r[0], 0.1) and np.isclose(r[1], 0.3)
    assert np.isclose(r[2], 1.0 - 0.1 - 0.3)         # the rest
    assert np.isclose(r.sum(), 1.0)
    assert (r >= 0).all()


def test_restrict_batched():
    vocab = np.array([[0.5, 0.5, 0.0], [0.1, 0.1, 0.8]])
    r = restrict_to_concept(vocab, [0])
    assert r.shape == (2, 2)
    assert np.allclose(r.sum(axis=-1), 1.0)
    assert np.allclose(r[:, 0], [0.5, 0.1])


def test_restrict_clamps_tiny_negative_other():
    # Float error can push Σconcept just over 1; "other" must clamp to >= 0.
    vocab = np.array([0.6, 0.4 + 1e-9])
    r = restrict_to_concept(vocab, [0, 1])
    assert r[-1] >= 0.0


# --------------------------------------------------------- Hellinger space
def test_hellinger_embed_decode_round_trip():
    p = np.array([0.2, 0.5, 0.3])
    y = hellinger_embed(p)
    assert np.allclose(y, np.sqrt(p))
    assert np.allclose(hellinger_decode(y), p)       # square back, already normalized


def test_hellinger_decode_renormalizes_off_sphere_point():
    # A point not on the unit sphere still decodes to a valid distribution.
    y = np.array([0.3, 0.4, 0.5])                     # ‖y‖ != 1
    p = hellinger_decode(y)
    assert np.isclose(p.sum(), 1.0)
    assert (p >= 0).all()


def test_hellinger_distance_matches_formula():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    # Two disjoint point masses -> maximal Hellinger distance of 1.
    assert np.isclose(hellinger_distance(p, q), 1.0)
    assert np.isclose(hellinger_distance(p, p), 0.0)


# ------------------------------------------------------------------- fit
def _peaked_distributions(n_per_day=6, noise=0.01, seed=0):
    """Per-prompt distributions peaked on the true day (7 days + 'other')."""
    rng = np.random.default_rng(seed)
    days = np.repeat(np.arange(7), n_per_day)
    dists = np.full((len(days), 8), noise)
    dists[np.arange(len(days)), days] = 1.0 - noise * 7
    dists += rng.normal(0, noise / 4, dists.shape)
    dists = np.clip(dists, 1e-6, None)
    dists /= dists.sum(axis=1, keepdims=True)
    return dists, days


def test_fit_behavior_manifold_shapes_and_valid_centroids():
    dists, days = _peaked_distributions()
    my = fit_behavior_manifold(dists, days)
    assert isinstance(my, BehaviorManifold)
    assert my.centroids_dist.shape == (7, 8)
    assert np.allclose(my.centroids_dist.sum(axis=1), 1.0)   # still distributions
    assert np.allclose(my.centroids_hell, np.sqrt(my.centroids_dist))
    assert my.day_order == list(range(7))


def _cyclic_bump_distributions(n_per_day=6, seed=0):
    """Distributions that form a smooth RING over the 7 day-classes (+ 'other').

    One-hot corners are all equidistant (a simplex, not a ring), so geometry
    can't recover their order; a circular bump makes adjacent days closer than
    opposite days, giving a genuine ring in √p space.
    """
    rng = np.random.default_rng(seed)
    bump = np.array([0.55, 0.18, 0.04, 0.005, 0.005, 0.04, 0.18])
    days = np.repeat(np.arange(7), n_per_day)
    rows = []
    for d in days:
        row = np.append(np.roll(bump, d), 0.02)
        row = np.clip(row + rng.normal(0, 0.002, row.shape), 1e-6, None)
        rows.append(row / row.sum())
    return np.array(rows), days


def test_fit_behavior_manifold_recovers_cycle_from_geometry():
    # A cyclic-bump ℳ_y: its 7 centroids form a ring in √p space, so the cyclic
    # order recovered with no labels is 0..6 up to rotation/reflection.
    dists, days = _cyclic_bump_distributions()
    my = fit_behavior_manifold(dists, days)
    recovered = recover_cyclic_order(my.centroids_hell)
    assert cyclic_equivalent(recovered, list(range(7)))


def test_behavior_geodesic_matrix_is_day_indexed_and_symmetric():
    dists, days = _peaked_distributions()
    my = fit_behavior_manifold(dists, days)
    D = my.geodesic_matrix(n_samples=500)
    assert D.shape == (7, 7)
    assert np.allclose(np.diag(D), 0.0, atol=1e-9)
    assert np.allclose(D, D.T, atol=1e-6)


def test_forward_dist_decodes_to_valid_distribution():
    dists, days = _peaked_distributions()
    my = fit_behavior_manifold(dists, days)
    for u in np.linspace(0, 1, 11, endpoint=False):
        p = my.forward_dist(u)
        assert np.isclose(p.sum(), 1.0)
        assert (p >= -1e-9).all()
