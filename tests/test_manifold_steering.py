"""Steering/interpolation displacements: endpoints, ring-visiting, short-arc wrap,
per-template demean, and the six-condition intervention identities. Pure (no model)."""

import numpy as np
import pytest

from weekday_manifold.manifold.steering import (
    day_knot,
    day_residual,
    fit_steer_spline,
    intervention_delta,
    linear_displacement,
    manifold_displacement,
    manifold_velocity,
    short_arc,
    steer_vectors,
    template_demeaned_centroids,
)

N_DAYS = 7


def _ring(n=N_DAYS, dim=5, seed=0):
    """n points on a circle embedded (rotated) in `dim`-D, in order — a synthetic ring."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.zeros((n, dim))
    pts[:, 0] = np.cos(theta)
    pts[:, 1] = np.sin(theta)
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return pts @ Q


# ------------------------------------------------------------------ short arc / knots
def test_day_knot():
    assert day_knot(0) == 0.0
    assert day_knot(3) == pytest.approx(3 / 7)


def test_short_arc_takes_geodesic():
    # +3 hop is +3/7 forward; Fri(4)->Mon(0) wraps forward +3/7, not -4/7.
    assert short_arc(0.0, 3 / 7) == pytest.approx(3 / 7)
    assert short_arc(4 / 7, 0.0) == pytest.approx(3 / 7)
    # Mon(0)->Fri(4): short way is 3 back, i.e. -3/7 (not +4/7).
    assert short_arc(0.0, 4 / 7) == pytest.approx(-3 / 7)
    for u0 in np.linspace(0, 1, 13):
        for u1 in np.linspace(0, 1, 13):
            assert abs(short_arc(u0, u1)) <= 0.5 + 1e-9


# ------------------------------------------------------------------ template demean
def test_template_demean_removes_offset_preserves_difference():
    pts = _ring()
    rng = np.random.default_rng(1)
    offsets = rng.normal(size=(3, pts.shape[1]))       # one nuisance offset per template
    rows, days, temps = [], [], []
    for tau in range(3):
        for d in range(N_DAYS):
            rows.append(pts[d] + offsets[tau])
            days.append(d)
            temps.append(tau)
    acts = np.array(rows)
    c = template_demeaned_centroids(acts, np.array(days), np.array(temps), N_DAYS)
    # Per-template demean removes each template's across-day mean -> centroids are
    # exactly the ring centered at its grand mean; template offsets are gone.
    np.testing.assert_allclose(c, pts - pts.mean(axis=0), atol=1e-9)
    # And the day-difference (the steering direction) is offset-invariant.
    for s in range(N_DAYS):
        for t in range(N_DAYS):
            np.testing.assert_allclose(c[t] - c[s], pts[t] - pts[s], atol=1e-9)


# ------------------------------------------------------------------ displacements
@pytest.mark.parametrize("s,t", [(0, 3), (4, 0), (2, 5), (6, 2)])
def test_displacement_endpoints(s, t):
    pts = _ring()
    spline = fit_steer_spline(pts)
    step = pts[t] - pts[s]
    for disp in (linear_displacement(pts, s, t, 0.0),
                 manifold_displacement(spline, s, t, 0.0)):
        np.testing.assert_allclose(disp, 0.0, atol=1e-9)
    np.testing.assert_allclose(linear_displacement(pts, s, t, 1.0), step, atol=1e-9)
    np.testing.assert_allclose(manifold_displacement(spline, s, t, 1.0), step, atol=1e-9)


def test_manifold_visits_intermediate_centroids():
    pts = _ring()
    spline = fit_steer_spline(pts)
    # Mon(0)->Thu(3): lands on Tue(1) at 1/3, Wed(2) at 2/3.
    np.testing.assert_allclose(manifold_displacement(spline, 0, 3, 1 / 3), pts[1] - pts[0], atol=1e-9)
    np.testing.assert_allclose(manifold_displacement(spline, 0, 3, 2 / 3), pts[2] - pts[0], atol=1e-9)
    # Fri(4)->Mon(0) wraps: Sat(5) at 1/3, Sun(6) at 2/3, Mon(0) at 1.
    np.testing.assert_allclose(manifold_displacement(spline, 4, 0, 1 / 3), pts[5] - pts[4], atol=1e-9)
    np.testing.assert_allclose(manifold_displacement(spline, 4, 0, 2 / 3), pts[6] - pts[4], atol=1e-9)
    np.testing.assert_allclose(manifold_displacement(spline, 4, 0, 1.0), pts[0] - pts[4], atol=1e-9)


def test_arclength_constant_speed():
    pts = _ring()
    spline = fit_steer_spline(pts)
    alphas = np.linspace(0.0, 1.0, 21)
    disp = manifold_displacement(spline, 0, 3, alphas, param="arclength")     # [21, d]
    # endpoints preserved
    np.testing.assert_allclose(disp[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(disp[-1], pts[3] - pts[0], atol=2e-3)
    # per-step norm is (near-)constant under arclength, and more uniform than under u-param
    steps_al = np.linalg.norm(np.diff(disp, axis=0), axis=1)
    steps_u = np.linalg.norm(np.diff(manifold_displacement(spline, 0, 3, alphas, param="u"), axis=0), axis=1)
    cv = lambda x: x.std() / x.mean()
    assert cv(steps_al) < 0.02, cv(steps_al)
    assert cv(steps_al) < cv(steps_u)


@pytest.mark.parametrize("s,t", [(0, 3), (4, 0)])
def test_manifold_velocity_matches_finite_difference(s, t):
    # v(alpha) = d f_man/d alpha must match a central finite-difference of the displacement.
    pts = _ring()
    spline = fit_steer_spline(pts)
    alphas = np.linspace(0.05, 0.95, 19)          # stay off the 0/1 grid edges for central diff
    h = 1e-4
    for param in ("u", "arclength"):
        v = manifold_velocity(spline, s, t, alphas, param=param)               # [A, d]
        fwd = manifold_displacement(spline, s, t, alphas + h, param=param)
        bwd = manifold_displacement(spline, s, t, alphas - h, param=param)
        fd = (fwd - bwd) / (2 * h)
        np.testing.assert_allclose(v, fd, atol=2e-3, rtol=0)


def test_manifold_velocity_constant_norm_arclength():
    # Under arclength timing the velocity NORM is ~constant (the flat input stride);
    # under u-param it varies more. Also a scalar alpha returns a [d] vector.
    pts = _ring()
    spline = fit_steer_spline(pts)
    alphas = np.linspace(0.0, 1.0, 41)
    n_al = np.linalg.norm(manifold_velocity(spline, 0, 3, alphas, param="arclength"), axis=1)
    n_u = np.linalg.norm(manifold_velocity(spline, 0, 3, alphas, param="u"), axis=1)
    cv = lambda x: x.std() / x.mean()
    assert cv(n_al) < 0.03, cv(n_al)
    assert cv(n_al) < cv(n_u)
    assert manifold_velocity(spline, 0, 3, 0.5, param="arclength").shape == (pts.shape[1],)


def test_linear_and_manifold_diverge_mid_path():
    pts = _ring()
    spline = fit_steer_spline(pts)
    lin = linear_displacement(pts, 0, 3, 0.5)
    man = manifold_displacement(spline, 0, 3, 0.5)
    assert not np.allclose(lin, man, atol=1e-3)


def test_steer_vectors_grid():
    pts = _ring()
    spline = fit_steer_spline(pts)
    alphas = np.linspace(0.0, 1.3, 10)
    for mode in ("linear", "manifold"):
        V = steer_vectors(pts, spline, 0, 3, alphas, mode=mode)
        assert V.shape == (10, pts.shape[1])
        np.testing.assert_allclose(V[0], 0.0, atol=1e-9)          # alpha=0 -> no displacement
    # grid rows match the scalar displacement per alpha
    V = steer_vectors(pts, spline, 0, 3, alphas, mode="manifold")
    for i, a in enumerate(alphas):
        np.testing.assert_allclose(V[i], manifold_displacement(spline, 0, 3, a), atol=1e-9)
    with pytest.raises(ValueError):
        steer_vectors(pts, spline, 0, 3, alphas, mode="nope")


# ------------------------------------------------------------------ interventions
def test_day_residual():
    rng = np.random.default_rng(2)
    A, mu, c_s = rng.normal(size=5), rng.normal(size=5), rng.normal(size=5)
    np.testing.assert_allclose(day_residual(A, mu, c_s), (A - mu) - c_s, atol=1e-12)


def test_intervention_identities():
    pts = _ring()
    spline = fit_steer_spline(pts)
    s, t = 0, 3
    rng = np.random.default_rng(3)
    r = rng.normal(size=pts.shape[1])
    mu = rng.normal(size=pts.shape[1])
    A = mu + pts[s] + r                       # a synthetic "real prompt" activation
    f0 = manifold_displacement(spline, s, t, 0.0)
    fh = manifold_displacement(spline, s, t, 0.5)
    f1 = manifold_displacement(spline, s, t, 1.0)     # == pts[t]-pts[s]

    # delta identities
    np.testing.assert_allclose(intervention_delta(f0, 0.0, r, "steer"), 0.0, atol=1e-9)
    np.testing.assert_allclose(intervention_delta(f1, 1.0, r, "interp"), f1 - r, atol=1e-12)
    np.testing.assert_allclose(intervention_delta(f1, 1.0, r, "no_resid"), f1 - r, atol=1e-12)
    np.testing.assert_allclose(intervention_delta(fh, 0.5, r, "interp"), fh - 0.5 * r, atol=1e-12)

    # trajectory endpoints x = A + delta
    np.testing.assert_allclose(A + intervention_delta(f1, 1.0, r, "steer"), mu + pts[t] + r, atol=1e-9)
    np.testing.assert_allclose(A + intervention_delta(f1, 1.0, r, "interp"), mu + pts[t], atol=1e-9)
    np.testing.assert_allclose(A + intervention_delta(f0, 0.0, r, "no_resid"), mu + pts[s], atol=1e-9)
    np.testing.assert_allclose(A + intervention_delta(f1, 1.0, r, "no_resid"), mu + pts[t], atol=1e-9)

    # linear interpolation is exactly (1-a)A + a(mu + c_t)
    fl_h = linear_displacement(pts, s, t, 0.5)
    x = A + intervention_delta(fl_h, 0.5, r, "interp")
    np.testing.assert_allclose(x, 0.5 * A + 0.5 * (mu + pts[t]), atol=1e-9)

    with pytest.raises(ValueError):
        intervention_delta(f1, 1.0, r, "bogus")
