"""Steering displacements along the day manifold. Pure numpy: no model, no torch."""

from __future__ import annotations

import numpy as np

from weekday_manifold.manifold.pca import group_centroids
from weekday_manifold.manifold.spline import PeriodicSpline


# --------------------------------------------------------------------- centroids
def template_means(acts: np.ndarray, template_labels: np.ndarray) -> dict:
    """Per-template mean activation ``mu_tau`` (the across-day nuisance offset)."""
    acts = np.asarray(acts, dtype=float)
    labels = np.asarray(template_labels)
    return {t: acts[labels == t].mean(axis=0) for t in np.unique(labels)}


def template_demeaned_centroids(
    acts: np.ndarray,
    day_labels: np.ndarray,
    template_labels: np.ndarray,
    n_days: int = 7,
) -> np.ndarray:
    """Per-day centroids after removing each template's across-day mean."""
    acts = np.asarray(acts, dtype=float)
    labels = np.asarray(template_labels)
    demeaned = acts.copy()
    for t in np.unique(labels):
        m = labels == t
        demeaned[m] = demeaned[m] - acts[m].mean(axis=0)
    return group_centroids(demeaned, np.asarray(day_labels), n_days)


def fit_steer_spline(centroids: np.ndarray) -> PeriodicSpline:
    """Periodic cubic spline threaded through the full-dim centroids (Mon..Sun order)."""
    return PeriodicSpline(np.asarray(centroids, dtype=float))


# ------------------------------------------------------------------- geometry helpers
def day_knot(day: int, n_days: int = 7) -> float:
    """Intrinsic coordinate ``u`` of a day centroid (canonical Mon..Sun order)."""
    return float(day) / float(n_days)


def short_arc(u0: float, u1: float, period: float = 1.0) -> float:
    """Signed shorter-arc step from ``u0`` to ``u1`` on a period-``period`` loop."""
    half = period / 2.0
    return (float(u1) - float(u0) + half) % period - half


# ------------------------------------------------------------------- displacements
def linear_displacement(centroids: np.ndarray, s: int, t: int, alpha) -> np.ndarray:
    """``f_lin(alpha) = alpha * (c_t - c_s)`` — the straight chord displacement."""
    c = np.asarray(centroids, dtype=float)
    step = c[t] - c[s]
    a = np.asarray(alpha, dtype=float)
    return np.multiply.outer(a, step) if a.ndim else a * step


def _arclength_u(spline: PeriodicSpline, u_s: float, du: float, alphas: np.ndarray, n: int = 6000) -> np.ndarray:
    """Intrinsic coords whose ARC LENGTH from ``u_s`` is proportional to ``alpha``."""
    a = np.asarray(alphas, dtype=float)
    ts = np.linspace(min(a.min(), 0.0) - 0.02, max(a.max(), 1.0) + 0.02, n)
    us = u_s + ts * du
    pts = spline.forward(us)                                   # [n, d]
    s_cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    s0 = float(np.interp(0.0, ts, s_cum))                     # arc length at alpha=0 (u_s)
    s1 = float(np.interp(1.0, ts, s_cum))                     # arc length at alpha=1 (u_t)
    t_of = np.interp(s0 + a * (s1 - s0), s_cum, ts)           # invert arc length -> t
    return u_s + t_of * du


def manifold_displacement(
    spline: PeriodicSpline,
    s: int,
    t: int,
    alpha,
    n_days: int = 7,
    wrap: str = "short",
    param: str = "u",
) -> np.ndarray:
    """``f_man(alpha) = spline(u(alpha)) - spline(u_s)`` — the ring-arc displacement."""
    u_s = day_knot(s, n_days)
    u_t = day_knot(t, n_days)
    du = short_arc(u_s, u_t, spline.period) if wrap == "short" else (u_t - u_s)
    a = np.asarray(alpha, dtype=float)
    aa = np.atleast_1d(a)
    if param == "u":
        u = u_s + aa * du
    elif param == "arclength":
        u = _arclength_u(spline, u_s, du, aa)
    else:
        raise ValueError(f"param must be 'u' or 'arclength', got {param!r}.")
    disp = spline.forward(u) - spline.forward(u_s)            # [A, d]
    return disp[0] if a.ndim == 0 else disp


def manifold_velocity(
    spline: PeriodicSpline,
    s: int,
    t: int,
    alphas,
    n_days: int = 7,
    wrap: str = "short",
    param: str = "arclength",
) -> np.ndarray:
    """``v(alpha) = d f_man/d alpha = spline.derivative(u(alpha)) * du/dalpha`` — the steering
    velocity."""
    u_s = day_knot(s, n_days)
    u_t = day_knot(t, n_days)
    du = short_arc(u_s, u_t, spline.period) if wrap == "short" else (u_t - u_s)
    a = np.asarray(alphas, dtype=float)
    aa = np.atleast_1d(a)
    tng = spline.derivative                                  # exact per-u tangent d(point)/du
    if param == "u":
        u = u_s + aa * du
        vel = tng(u) * du                                   # [A, d]
    elif param == "arclength":
        u = _arclength_u(spline, u_s, du, aa)
        # total arc length L of the arc u_s -> u_s+du (unsigned path length)
        us_dense = u_s + np.linspace(0.0, 1.0, 6000) * du
        L = float(np.sum(np.linalg.norm(np.diff(spline.forward(us_dense), axis=0), axis=1)))
        d_u = tng(u)                                         # [A, d]
        unit = d_u / np.linalg.norm(d_u, axis=1, keepdims=True)
        vel = L * np.sign(du) * unit                        # ||vel|| == L
    else:
        raise ValueError(f"param must be 'u' or 'arclength', got {param!r}.")
    return vel[0] if a.ndim == 0 else vel


def steer_vectors(
    centroids: np.ndarray,
    spline: PeriodicSpline,
    s: int,
    t: int,
    alphas: np.ndarray,
    mode: str = "linear",
    n_days: int = 7,
    wrap: str = "short",
    param: str = "u",
) -> np.ndarray:
    """Displacement grid ``[len(alphas), d]`` for ``mode in {"linear", "manifold"}``."""
    a = np.asarray(alphas, dtype=float)
    if mode == "linear":
        return linear_displacement(centroids, s, t, a)
    if mode == "manifold":
        return manifold_displacement(spline, s, t, a, n_days=n_days, wrap=wrap, param=param)
    raise ValueError(f"mode must be 'linear' or 'manifold', got {mode!r}.")


# ------------------------------------------------------------------- interventions
def day_residual(A: np.ndarray, template_mean: np.ndarray, centroid_s: np.ndarray) -> np.ndarray:
    """Day-specific residual ``r = (A - mu_tau) - c_s`` (template nuisance removed)."""
    return np.asarray(A, float) - np.asarray(template_mean, float) - np.asarray(centroid_s, float)


def intervention_delta(f_alpha: np.ndarray, alpha: float, r: np.ndarray, intervention: str) -> np.ndarray:
    """The additive delta injected over the live residual ``A``, per intervention."""
    f = np.asarray(f_alpha, dtype=float)
    r = np.asarray(r, dtype=float)
    if intervention == "steer":
        return f
    if intervention == "interp":
        return f - float(alpha) * r
    if intervention == "no_resid":
        return f - r
    raise ValueError(f"intervention must be 'steer'|'interp'|'no_resid', got {intervention!r}.")


# ------------------------------------------------------------------- ring diagnostics
def ring_report(tag, C, acts, day_labels, template_ids, spline):
    """Read-only diagnostics for a fitted day-ring. Changes nothing that is measured."""
    means = template_means(acts, template_ids)
    dem = acts - np.stack([means[t] for t in template_ids])
    rad = np.linalg.norm(C - C.mean(0), axis=1)
    scatter = float(np.linalg.norm(dem - C[day_labels], axis=1).mean())
    u_hat = np.asarray(spline.inverse(dem))
    pred = np.rint(u_hat * len(C)).astype(int) % len(C)
    acc = float((pred == day_labels).mean())
    off = float(np.mean(np.linalg.norm(dem - spline.forward(u_hat), axis=1)) / rad.mean())
    print(f"[errc] RING {tag}: radius mean={rad.mean():.3f} min={rad.min():.3f} "
          f"max={rad.max():.3f} | off-ring scatter={scatter:.3f} "
          f"(radius/scatter={rad.mean() / scatter:.3f}) | fit-prompt day recovery="
          f"{acc:.3f} (chance {1 / len(C):.3f}) | mean off-ring distance="
          f"{off:.2f} ring radii", flush=True)
    return dict(radius=rad, scatter=scatter, recovery=acc, offring=off)
