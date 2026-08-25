"""Drawing module: where a point lands on the weekday spline. No main.

spline_op builds both figure 3's foot points and figure 7's ring -- the same spline through
the same knots.
"""
from __future__ import annotations
import os, sys

import numpy as np

# No stub-module shim here, unlike the research repo: weekday_manifold/__init__.py
# resolves load_model lazily (PEP 562), so importing the spline does not pull in torch.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, _SRC)
from weekday_manifold.manifold.spline import PeriodicSpline  # noqa: E402


def spline_op(n):
    """[n+1, 7] operator: curve samples = W @ knots. Row n repeats row 0 (the seam)."""
    return np.stack([PeriodicSpline(np.eye(7)[:, [j]]).sample(n)[1][:, 0] for j in range(7)], 1)


def foot_points(z, curve, s_grid, L, sep):
    """Nearest curve sample for each row of z. Returns (s, d_perp, ambiguity ratio)."""
    G = len(curve)
    cn = (curve ** 2).sum(1)
    s_out = np.empty(len(z)); d_out = np.empty(len(z)); amb = np.empty(len(z))
    for a in range(0, len(z), 2048):
        Z = z[a:a + 2048]
        d2 = (Z ** 2).sum(1)[:, None] - 2.0 * (Z @ curve.T) + cn[None, :]
        i = np.argmin(d2, 1)
        s_out[a:a + 2048] = s_grid[i]
        d_out[a:a + 2048] = np.sqrt(np.maximum(d2[np.arange(len(Z)), i], 0.0))
        # second minimum, excluding everything within `sep` arc length of the best
        gap = np.abs(s_grid[None, :] - s_grid[i][:, None])
        m = np.minimum(gap, L - gap) <= sep
        d2b = np.where(m, np.inf, d2)
        amb[a:a + 2048] = np.sqrt(np.maximum(d2b.min(1), 0.0)) / np.maximum(d_out[a:a + 2048], 1e-12)
        del d2, d2b, gap, m
    return s_out, d_out, amb


def chord_feet(z, K):
    """Nearest point on the seven straight chords Mon->Tue->...->Sun->Mon."""
    best = np.full(len(z), np.inf); frac = np.zeros(len(z))
    for k in range(7):
        p, q = K[k], K[(k + 1) % 7]
        e = q - p; ee = float(e @ e)
        t = np.clip(((z - p) @ e) / ee, 0.0, 1.0)
        d = np.linalg.norm(z - (p + t[:, None] * e[None, :]), axis=1)
        w = d < best
        best[w] = d[w]; frac[w] = t[w]
    return best, frac


def arc_frac(s, s_knot, L):
    """Fractional position within the arc each foot point falls in, in [0, 1)."""
    sk = np.r_[s_knot, s_knot[0] + L]
    j = np.searchsorted(sk, s, side="right") - 1
    j = np.clip(j, 0, 6)
    return (s - sk[j]) / (sk[j + 1] - sk[j]), j
