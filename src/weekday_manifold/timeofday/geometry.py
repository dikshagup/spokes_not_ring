"""The ring frame, the in-plane / off-plane decomposition, and the time-of-day metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

N_DAYS = 7


# ------------------------------------------------------------------- frame
@dataclass
class RingFrame:
    """The weekday reference frame at one layer, fitted from neutral prompts only."""

    mu: np.ndarray
    centroids: np.ndarray
    plane: np.ndarray
    span: np.ndarray
    evr: np.ndarray
    theta: np.ndarray
    radius: float
    orient: int

    @property
    def evr_plane(self) -> float:
        """Fraction of weekday-centroid variance captured by the 2-D plane."""
        return float(self.evr[:2].sum())

    @property
    def evr_beyond(self) -> float:
        """Weekday-centroid variance OUTSIDE the plane (Engels/Prieto both flag this)."""
        return float(self.evr[2:].sum())

    def in_plane(self, X: np.ndarray) -> np.ndarray:
        """Coordinates of rows of ``X`` in the plane, relative to ``mu``. ``[N, 2]``"""
        return (np.atleast_2d(X) - self.mu) @ self.plane

    def angle_of(self, X: np.ndarray) -> np.ndarray:
        """Ring angle (radians) of each row of ``X``, oriented so day index increases."""
        c = self.in_plane(X)
        return np.arctan2(self.orient * c[:, 1], c[:, 0])

    def heptagon_check(self) -> Dict[str, object]:
        """Do the 7 neutral centroids sit in weekday order around the ring?"""
        ang = np.arctan2(self.orient * self.in_plane(self.centroids)[:, 1],
                         self.in_plane(self.centroids)[:, 0])
        order = list(np.argsort(ang))
        rot = [(order.index(d) - order.index(0)) % N_DAYS for d in range(N_DAYS)]
        in_order = rot == list(range(N_DAYS))
        gaps = np.diff(np.sort(ang))
        return {
            "in_cyclic_order": bool(in_order),
            "angles_deg": np.degrees(ang).tolist(),
            "order": [int(o) for o in order],
            "min_gap_deg": float(np.degrees(gaps.min())) if gaps.size else float("nan"),
            "mean_gap_deg": float(np.degrees(gaps.mean())) if gaps.size else float("nan"),
        }


def _orthonormal(basis: np.ndarray) -> np.ndarray:
    """Column-orthonormalise (QR); drops numerically-zero columns."""
    q, r = np.linalg.qr(basis)
    keep = np.abs(np.diag(r)) > 1e-8 * max(1.0, float(np.abs(r).max()))
    return q[:, keep]


def fit_ring_frame(acts: np.ndarray, days: np.ndarray) -> RingFrame:
    """Fit the weekday plane from NEUTRAL prompts only (plan section 3)."""
    acts = np.asarray(acts, dtype=np.float64)
    days = np.asarray(days)
    cents = np.stack([acts[days == d].mean(axis=0) for d in range(N_DAYS)])
    mu = cents.mean(axis=0)
    C = cents - mu                                        # [7, d], rank <= 6
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    var = S ** 2
    evr = var / var.sum() if var.sum() > 0 else var
    plane = Vt[:2].T                                      # [d, 2]
    rank = int((S > 1e-8 * S.max()).sum()) if S.size else 0
    span = Vt[:rank].T                                    # [d, r<=6]
    coords = C @ plane                                    # [7, 2]
    raw_ang = np.arctan2(coords[:, 1], coords[:, 0])
    # Orientation: does the day index advance counter-clockwise? Use the mean
    # wrapped step between consecutive days; sign of that decides the convention
    # so "positive angular shift" always means "toward the NEXT day".
    steps = np.angle(np.exp(1j * (np.roll(raw_ang, -1) - raw_ang)))
    orient = 1 if steps.mean() >= 0 else -1
    theta = np.arctan2(orient * coords[:, 1], coords[:, 0])
    radius = float(np.linalg.norm(C, axis=1).mean())
    return RingFrame(mu=mu, centroids=cents, plane=plane, span=span, evr=evr,
                     theta=theta, radius=radius, orient=orient)


# ----------------------------------------------------------- decomposition
def decompose(acts: np.ndarray, days: np.ndarray, frame: RingFrame,
              basis: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Split each row's time displacement into in-plane and orthogonal parts."""
    P = frame.plane if basis is None else basis
    acts = np.asarray(acts, dtype=np.float64)
    days = np.asarray(days)
    delta = acts - frame.centroids[days]
    delta_par = (delta @ P) @ P.T
    delta_perp = delta - delta_par
    n_d = np.linalg.norm(delta, axis=1)
    n_par = np.linalg.norm(delta_par, axis=1)
    n_perp = np.linalg.norm(delta_perp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(n_d > 0, (n_perp ** 2) / (n_d ** 2), np.nan)
    return {"delta": delta, "delta_par": delta_par, "delta_perp": delta_perp,
            "norm": n_d, "norm_par": n_par, "norm_perp": n_perp, "rho": rho}


def cell_table(acts: np.ndarray, days: np.ndarray, mods: np.ndarray,
               extra: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Cell centroids ``a_{d,t}`` -- the unit plan section 4 defines every metric on."""
    acts = np.asarray(acts, dtype=np.float64)
    mod_order: List[str] = []
    for m in mods:
        if m not in mod_order:
            mod_order.append(m)
    cent, cday, cmod, cnt, idx = [], [], [], [], []
    for d in range(N_DAYS):
        for m in mod_order:
            sel = (days == d) & (mods == m)
            if not sel.any():
                continue
            cent.append(acts[sel].mean(axis=0))
            cday.append(d)
            cmod.append(m)
            cnt.append(int(sel.sum()))
            idx.append(int(np.flatnonzero(sel)[0]))
    out = {"cent": np.stack(cent), "day": np.asarray(cday),
           "modifier": np.asarray(cmod, dtype=object),
           "counts": np.asarray(cnt)}
    for name, arr in (extra or {}).items():
        out[name] = np.asarray(arr)[np.asarray(idx)]
    return out


def cell_means(acts: np.ndarray, days: np.ndarray, mods: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse per-carrier rows to (day, modifier) cell centroids."""
    acts = np.asarray(acts, dtype=np.float64)
    mod_order: List[str] = []
    for m in mods:
        if m not in mod_order:
            mod_order.append(m)
    cent, cday, cmod, cnt = [], [], [], []
    for d in range(N_DAYS):
        for m in mod_order:
            sel = (days == d) & (mods == m)
            if not sel.any():
                continue
            cent.append(acts[sel].mean(axis=0))
            cday.append(d)
            cmod.append(m)
            cnt.append(int(sel.sum()))
    return (np.stack(cent), np.asarray(cday), np.asarray(cmod, dtype=object),
            np.asarray(cnt))


# --------------------------------------------- metric 1 / 4: energy fraction
def orthogonal_energy(acts: np.ndarray, days: np.ndarray, frame: RingFrame,
                      basis: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Metric 1 (per-cell mean) and Metric 4 (pooled variance partition)."""
    dec = decompose(acts, days, frame, basis=basis)
    ss_total = float((dec["norm"] ** 2).sum())
    ss_perp = float((dec["norm_perp"] ** 2).sum())
    ss_par = float((dec["norm_par"] ** 2).sum())
    return {
        "rho_mean": float(np.nanmean(dec["rho"])),
        "rho_median": float(np.nanmedian(dec["rho"])),
        "rho_pooled": ss_perp / ss_total if ss_total > 0 else float("nan"),
        "ss_total": ss_total, "ss_par": ss_par, "ss_perp": ss_perp,
        "norm_mean": float(dec["norm"].mean()),
        "norm_par_mean": float(dec["norm_par"].mean()),
        "norm_perp_mean": float(dec["norm_perp"].mean()),
        "norm_par_over_R": float(dec["norm_par"].mean() / frame.radius),
        "norm_perp_over_R": float(dec["norm_perp"].mean() / frame.radius),
        "radius": frame.radius,
        "n": int(len(dec["rho"])),
    }


# ------------------------------------------- metric 2: in-plane angular shift
def wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(a)))


def inplane_shift(acts: np.ndarray, days: np.ndarray, frame: RingFrame
                  ) -> Dict[str, np.ndarray]:
    """Metric 2: signed in-plane angular shift from the day's neutral angle."""
    ang = frame.angle_of(acts)
    dtheta = wrap(ang - frame.theta[np.asarray(days)])
    coords = frame.in_plane(acts)
    radial = np.linalg.norm(coords, axis=1)
    cent_r = np.linalg.norm(frame.in_plane(frame.centroids), axis=1)
    gaps = wrap(np.diff(np.sort(frame.theta)))
    return {
        "dtheta": dtheta,
        "dtheta_deg": np.degrees(dtheta),
        "radial": radial,
        "radial_rel": radial / cent_r[np.asarray(days)],
        "half_step_deg": float(np.degrees(np.abs(gaps).mean()) / 2.0) if gaps.size else float("nan"),
    }


# ------------------------------------------- metric 3: shared orthogonal axis
def shared_axis(delta_perp: np.ndarray) -> Dict[str, object]:
    """Metric 3a: PCA on the orthogonal displacements -> is there ONE shared axis?"""
    X = np.asarray(delta_perp, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    evr = var / var.sum() if var.sum() > 0 else var
    return {"v_time": Vt[0], "evr": evr, "top1_frac": float(evr[0]) if evr.size else float("nan"),
            "top3_frac": float(evr[:3].sum()) if evr.size else float("nan")}


def axis_ordering(delta: np.ndarray, t: np.ndarray, days: np.ndarray,
                  v: np.ndarray) -> Dict[str, object]:
    """Metric 3b: is the projection onto ``v`` monotone in ``t``, for every day?"""
    proj = np.asarray(delta, dtype=np.float64) @ np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(t)
    levels = sorted(set(t[finite].tolist()))
    by_level = {float(l): float(proj[finite & (t == l)].mean()) for l in levels}
    means = np.array([by_level[l] for l in levels])
    monotone_inc = bool(np.all(np.diff(means) > 0))
    monotone_dec = bool(np.all(np.diff(means) < 0))
    per_day = {}
    for d in range(N_DAYS):
        sel = finite & (days == d)
        if not sel.any():
            continue
        m = np.array([proj[sel & (t == l)].mean() if (sel & (t == l)).any() else np.nan
                      for l in levels])
        ok = np.isfinite(m)
        per_day[d] = {
            "means": m.tolist(),
            "monotone": bool(np.all(np.diff(m[ok]) > 0) or np.all(np.diff(m[ok]) < 0)),
            "slope_sign": int(np.sign(np.polyfit(np.array(levels)[ok], m[ok], 1)[0])) if ok.sum() > 1 else 0,
        }
    signs = [v_["slope_sign"] for v_ in per_day.values()]
    return {
        "levels": [float(l) for l in levels],
        "mean_by_level": by_level,
        "monotone": monotone_inc or monotone_dec,
        "direction": "increasing" if monotone_inc else ("decreasing" if monotone_dec else "none"),
        "rank_corr": float(_spearman(proj[finite], t[finite])),
        "per_day": per_day,
        "n_days_consistent_sign": int(max(signs.count(1), signs.count(-1))) if signs else 0,
        "n_days": len(per_day),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / den) if den > 0 else float("nan")


def cross_day_consistency(cent: np.ndarray, cday: np.ndarray, cmod: np.ndarray,
                          early: str, late: str) -> Dict[str, object]:
    """Metric 3c: pairwise cosine of the per-day time direction ``u_d``."""
    u = {}
    for d in range(N_DAYS):
        a = cent[(cday == d) & (cmod == late)]
        b = cent[(cday == d) & (cmod == early)]
        if len(a) and len(b):
            u[d] = a[0] - b[0]
    ds = sorted(u)
    M = np.full((len(ds), len(ds)), np.nan)
    for i, di in enumerate(ds):
        for j, dj in enumerate(ds):
            ni, nj = np.linalg.norm(u[di]), np.linalg.norm(u[dj])
            if ni > 0 and nj > 0:
                M[i, j] = float(u[di] @ u[dj] / (ni * nj))
    off = M[~np.eye(len(ds), dtype=bool)]
    return {"days": ds, "cosine": M, "mean_offdiag": float(np.nanmean(off)) if off.size else float("nan"),
            "min_offdiag": float(np.nanmin(off)) if off.size else float("nan"),
            "u": {int(k): v for k, v in u.items()}}


def principal_angle(v: np.ndarray, basis: np.ndarray) -> float:
    """Metric 3d: angle (degrees) between a vector and a subspace. 90 = orthogonal."""
    v = np.asarray(v, dtype=np.float64)
    nv = np.linalg.norm(v)
    if nv == 0:
        return float("nan")
    proj = np.linalg.norm(basis.T @ v)
    return float(np.degrees(np.arccos(np.clip(proj / nv, 0.0, 1.0))))


# ------------------------------------------------- metric 5: the time probe
def gram(X: np.ndarray) -> np.ndarray:
    """Centered Gram matrix ``Xc @ Xc.T`` -- the only view of ``X`` the probe needs."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0)
    return Xc @ Xc.T


def gram_after_removing(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Gram of ``X`` with ``basis`` projected out, without forming the big matrix."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0)
    C = Xc @ np.asarray(basis, dtype=np.float64)      # [N, k]
    return Xc @ Xc.T - C @ C.T


def _select_alpha_loocv(s: np.ndarray, U: np.ndarray, yc: np.ndarray,
                        alphas: Sequence[float]) -> float:
    """Pick ridge ``alpha`` by closed-form leave-one-ROW-out on the training set."""
    Uty = U.T @ yc
    best_a, best_err = float(alphas[0]), np.inf
    for a in alphas:
        f = s / (s + a)
        fit = U @ (f * Uty)
        h = np.einsum("ij,j,ij->i", U, f, U)
        denom = np.clip(1.0 - h, 1e-9, None)
        err = float((((yc - fit) / denom) ** 2).mean())
        if err < best_err:
            best_err, best_a = err, float(a)
    return best_a


def time_probe_lodo(acts: Optional[np.ndarray], t: np.ndarray, days: np.ndarray,
                    alphas: Sequence[float] = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6),
                    G: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Metric 5: leave-one-DAY-out ridge probe from activation -> ordinal ``t``."""
    y = np.asarray(t, dtype=np.float64)
    days = np.asarray(days)
    keep = np.isfinite(y)
    if G is None:
        G = gram(np.asarray(acts, dtype=np.float64)[keep])
    else:
        G = np.asarray(G, dtype=np.float64)
        if G.shape[0] == len(keep):
            G = G[np.ix_(keep, keep)]
    y, days = y[keep], days[keep]
    uniq = sorted(set(days.tolist()))

    preds = np.full(len(y), np.nan)
    chosen: List[float] = []
    for held in uniq:
        te = days == held
        tr = ~te
        Ktr = G[np.ix_(tr, tr)]
        ym = float(y[tr].mean())
        yc = y[tr] - ym
        s, U = np.linalg.eigh(Ktr)
        s = np.clip(s, 0.0, None)
        a = _select_alpha_loocv(s, U, yc, alphas)
        dual = U @ ((U.T @ yc) / (s + a))
        preds[te] = G[np.ix_(te, tr)] @ dual + ym
        chosen.append(a)

    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "r2_lodo": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rank_corr": float(_spearman(preds, y)),
        "alphas_chosen": [float(a) for a in chosen],
        "preds": preds, "y": y, "days": days,
    }


def probe_direction(acts: np.ndarray, t: np.ndarray, alpha: float = 1e3) -> np.ndarray:
    """The probe's weight vector ``w_time`` (direction only -- never used for R^2)."""
    X = np.asarray(acts, dtype=np.float64)
    y = np.asarray(t, dtype=np.float64)
    keep = np.isfinite(y)
    X, y = X[keep], y[keep]
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    n = len(yc)
    K = Xc @ Xc.T + alpha * np.eye(n)
    return Xc.T @ np.linalg.solve(K, yc)


def remove_subspace(acts: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project a subspace OUT of the activations (plan section 4, Metric 5)."""
    X = np.asarray(acts, dtype=np.float64)
    return X - (X @ basis) @ basis.T


# ------------------------------------------------------ metric 6: helix fit
def helix_fit(cent: np.ndarray, cday: np.ndarray, ct: np.ndarray,
              frame: RingFrame, v_time: np.ndarray) -> Dict[str, object]:
    """Metric 6: does angle-around-the-ring + height-along-``v_time`` beat either alone?"""
    A = np.asarray(cent, dtype=np.float64) - frame.mu
    P, v = frame.plane, np.asarray(v_time, dtype=np.float64)
    v = v / np.linalg.norm(v)
    v_off = v - P @ (P.T @ v)                      # component of v outside the plane
    if np.linalg.norm(v_off) > 1e-12:
        v_off = v_off / np.linalg.norm(v_off)
    B = _orthonormal(np.column_stack([P, v_off]))  # helix basis (plane + axis)

    def r2(basis):
        rec = (A @ basis) @ basis.T
        ss = float((A ** 2).sum())
        return float((rec ** 2).sum() / ss) if ss > 0 else float("nan")

    h = A @ v_off                                   # height per cell
    ang = np.arctan2(frame.orient * (A @ P)[:, 1], (A @ P)[:, 0])
    ct = np.asarray(ct, dtype=float)
    cday = np.asarray(cday)

    def frac_explained_by(values, groups, circular=False):
        vals = np.asarray(values, dtype=float)
        ok = np.isfinite(vals) & np.isfinite(np.asarray(groups, dtype=float)) \
            if not circular else np.isfinite(vals)
        vals, grp = vals[ok], np.asarray(groups)[ok]
        if circular:
            # variance of a circular quantity via the resultant length
            def circ_var(a):
                return 1.0 - float(np.abs(np.exp(1j * a).mean()))
            tot = circ_var(vals)
            within = np.mean([circ_var(vals[grp == g]) for g in sorted(set(grp.tolist()))])
            return float(1.0 - within / tot) if tot > 0 else float("nan")
        tot = float(((vals - vals.mean()) ** 2).sum())
        within = 0.0
        for g in sorted(set(grp.tolist())):
            s = vals[grp == g]
            within += float(((s - s.mean()) ** 2).sum())
        return float(1.0 - within / tot) if tot > 0 else float("nan")

    finite_t = np.isfinite(ct)
    levels = sorted(set(ct[finite_t].tolist()))
    h_by_t = {float(l): float(h[finite_t & (ct == l)].mean()) for l in levels}
    hs = np.array([h_by_t[l] for l in levels])
    return {
        "r2_ring": r2(P),
        "r2_axis": r2(v_off.reshape(-1, 1)),
        "r2_helix": r2(B),
        "height_by_t": h_by_t,
        "height_monotone": bool(np.all(np.diff(hs) > 0) or np.all(np.diff(hs) < 0)),
        "height_explained_by_time": frac_explained_by(h[finite_t], ct[finite_t]),
        "height_explained_by_day": frac_explained_by(h[finite_t], cday[finite_t]),
        "angle_explained_by_day": frac_explained_by(ang, cday, circular=True),
        "angle_explained_by_time": frac_explained_by(ang[finite_t], ct[finite_t], circular=True),
        "pitch_over_R": float((hs.max() - hs.min()) / frame.radius) if hs.size else float("nan"),
    }


# ------------------------------------------------------ section 6: controls
def fit_contaminated_frame(acts_all: np.ndarray, days_all: np.ndarray) -> RingFrame:
    """The WRONG frame, on purpose: Engels-style PCA over all prompts including times."""
    X = np.asarray(acts_all, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    plane = Vt[:2].T
    base = fit_ring_frame(acts_all, days_all)
    coords = (base.centroids - base.mu) @ plane
    raw = np.arctan2(coords[:, 1], coords[:, 0])
    steps = np.angle(np.exp(1j * (np.roll(raw, -1) - raw)))
    orient = 1 if steps.mean() >= 0 else -1
    var = S ** 2
    return RingFrame(mu=base.mu, centroids=base.centroids, plane=plane,
                     span=base.span, evr=var / var.sum() if var.sum() > 0 else var,
                     theta=np.arctan2(orient * coords[:, 1], coords[:, 0]),
                     radius=base.radius, orient=orient)


def contaminated_frame_comparison(acts_neutral: np.ndarray, days_neutral: np.ndarray,
                                  acts_all: np.ndarray, days_all: np.ndarray,
                                  ) -> Dict[str, object]:
    """Plan section 3: the neutral frame vs the Engels-style time-contaminated frame."""
    clean = fit_ring_frame(acts_neutral, days_neutral)
    dirty = fit_contaminated_frame(acts_all, days_all)
    return {
        "clean": orthogonal_energy(acts_all, days_all, clean),
        "contaminated": orthogonal_energy(acts_all, days_all, dirty),
        "plane_principal_angles_deg": _subspace_angles_deg(clean.plane, dirty.plane),
        "frames": {"clean": clean, "contaminated": dirty},
    }


def _subspace_angles_deg(A: np.ndarray, B: np.ndarray) -> List[float]:
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return [float(np.degrees(np.arccos(np.clip(x, 0.0, 1.0)))) for x in s]


def bootstrap_over_carriers(acts: np.ndarray, days: np.ndarray, carriers: np.ndarray,
                            statistic, n_boot: int = 500, seed: int = 0,
                            ) -> Dict[str, object]:
    """Bootstrap any statistic by RESAMPLING CARRIERS (plan section 6)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(carriers)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(carriers == c) for c in pick])
        try:
            vals.append(float(statistic(acts[idx], days[idx])))
        except Exception:
            vals.append(np.nan)
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v)
    return {"mean": float(v[ok].mean()) if ok.any() else float("nan"),
            "lo": float(np.percentile(v[ok], 2.5)) if ok.any() else float("nan"),
            "hi": float(np.percentile(v[ok], 97.5)) if ok.any() else float("nan"),
            "n_boot": int(ok.sum()), "values": v}


def permutation_test_ordering(delta: np.ndarray, t: np.ndarray, days: np.ndarray,
                              n_perm: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Plan section 6: shuffle time labels WITHIN day and re-derive the axis+ordering."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    days = np.asarray(days)
    keep = np.isfinite(t)
    D, tt, dd = np.asarray(delta, float)[keep], t[keep], days[keep]

    def stat(labels):
        ax = shared_axis(D)["v_time"]
        return abs(_spearman(D @ ax, labels))

    obs = stat(tt)
    null = np.empty(n_perm)
    for i in range(n_perm):
        lab = tt.copy()
        for d in set(dd.tolist()):
            sel = dd == d
            lab[sel] = rng.permutation(lab[sel])
        null[i] = stat(lab)
    return {"observed": float(obs),
            "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)),
            "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95))}


def whiten(acts: np.ndarray, ref: np.ndarray, var_keep: float = 0.99
           ) -> Tuple[np.ndarray, Dict[str, object]]:
    """Whiten by the neutral-weekday covariance (plan section 6, secondary check)."""
    R = np.asarray(ref, dtype=np.float64)
    mu = R.mean(axis=0)
    Rc = R - mu
    n_ref = len(Rc)
    cov = (Rc.T @ Rc) / max(1, n_ref - 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    pos = w > 1e-10 * float(w[0]) if w.size and w[0] > 0 else np.zeros(len(w), bool)
    w, V = w[pos], V[:, pos]
    csum = np.cumsum(w) / w.sum() if w.size else np.array([])
    k = int(np.searchsorted(csum, var_keep) + 1) if csum.size else 0
    k = max(1, min(k, len(w), max(1, n_ref - 1)))
    Wk = V[:, :k] * (w[:k] ** -0.5)                  # [d, k]
    info = {"n_ref": int(n_ref), "d_model": int(R.shape[1]), "rank_used": k,
            "var_explained": float(csum[k - 1]) if csum.size else float("nan"),
            "reliable": bool(k < n_ref)}
    return (np.asarray(acts, dtype=np.float64) - mu) @ Wk, info
