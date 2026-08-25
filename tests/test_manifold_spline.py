"""Periodic spline: closes (value + derivative), and s∘s⁻¹ ≈ identity on knots."""

import numpy as np

from weekday_manifold.manifold.spline import PeriodicSpline, default_knots


def _ring(n=7, dim=5, seed=0):
    """n points on a circle embedded in `dim`-D (first two dims), in order."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.zeros((n, dim))
    pts[:, 0] = np.cos(theta)
    pts[:, 1] = np.sin(theta)
    # A fixed random rotation so the ring isn't axis-aligned (still planar).
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return pts @ Q


def test_default_knots():
    assert np.allclose(default_knots(7), np.arange(7) / 7.0)


def test_spline_closes_value_and_derivative():
    s = PeriodicSpline(_ring())
    # Value at the seam: u=0 and u=period coincide.
    v0 = s._spline(0.0)
    v1 = s._spline(s.period)
    assert np.allclose(v0, v1, atol=1e-9)
    # First derivative matches across the seam (periodic boundary condition).
    d0 = s._spline(0.0, 1)
    d1 = s._spline(s.period, 1)
    assert np.allclose(d0, d1, atol=1e-6)


def test_forward_hits_knot_points():
    pts = _ring()
    s = PeriodicSpline(pts)
    for i, u in enumerate(s.knots):
        assert np.allclose(s.forward(u), pts[i], atol=1e-9)


def test_inverse_left_inverse_on_centroids():
    # s⁻¹(centroid_i) ≈ knot_i, and s(s⁻¹(c)) ≈ c.
    pts = _ring(n=7, dim=8)
    s = PeriodicSpline(pts)
    for i, c in enumerate(pts):
        u = s.inverse(c)
        assert abs((u - s.knots[i] + 0.5) % 1.0 - 0.5) < 1e-3
        assert np.allclose(s.forward(u), c, atol=1e-3)


def test_inverse_batched():
    pts = _ring(n=7, dim=6)
    s = PeriodicSpline(pts)
    us = s.inverse(pts)  # batched
    assert us.shape == (7,)
    recon = s.forward(us)
    assert np.allclose(recon, pts, atol=1e-3)


def test_forward_wraps_periodically():
    s = PeriodicSpline(_ring())
    assert np.allclose(s.forward(0.0), s.forward(1.0), atol=1e-9)
    assert np.allclose(s.forward(0.3), s.forward(1.3), atol=1e-9)
