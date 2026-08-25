"""Figure 6, panels A-D: where the weekday answer lives, and when.

Writes figures/fig_combined_arith_ad.{pdf,png}; panels E-H are a separate plate from
figure_jac_discs.py. See repro_fig6_combined_arith.sh.
"""
from __future__ import annotations

import argparse
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from weekday_manifold.manifold.behavior import hellinger_distance  # noqa: E402
from weekday_manifold.manifold.days import DAYS, N_DAYS  # noqa: E402
from weekday_manifold.manifold.steering import fit_steer_spline  # noqa: E402

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
# One family per steered site, as sequential ramps: Blues for the weekday token, Reds for the
# answer slot. Depth is the ramp position, so a curve's shade says WHICH LAYER and its hue
# says WHICH SITE -- the two variables the figure is about.
IN_C, ANS_C, OUT_C, GF = "#2b5d9e", "#b3322c", "#1f7a4d", "#7d54c4"
# One hue per weekday, so the ring's ORDER is visible and comparable across depths. The
# Okabe-Ito set in SPECTRAL order -- pink, vermillion, orange, yellow, green, sky, blue --
# so consecutive weekdays are consecutive hues and the ring reads as a progression. Ordering
# them this way is what fixed the worst adjacent pair: Wed/Thu were dE 7.6 under deuteranopia,
# below the target of 8; worst adjacent is now 9.6. Validated with a Python port of the
# dataviz validator, not assumed.
#
# CAVEAT, and it is not fixable by picking different values. IN_C/ANS_C/OUT_C/GF encode the
# SITE, and panel_stack draws them in the same panel as the day-coloured knots. Every one of
# them now sits within dE 15 of some day colour (GF vs Sun is 2.7). Searching for four
# replacements that clear all seven days AND each other tops out at worst-pair dE 12.0, with
# all four near-black: eleven mutually distinguishable categorical hues do not exist. The real
# fix is for panel A to stop using hue for two different variables -- sites by shape or label,
# colour reserved for days -- which is a redesign, not a palette change.
DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9",
         "#0072B2"]
BLUES, REDS = plt.get_cmap("Blues"), plt.get_cmap("Reds")
WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


def shade(cmap, i, n):
    """Position i of n on a sequential ramp, kept clear of the invisible pale end."""
    return cmap(0.32 + 0.62 * (i / max(n - 1, 1)))


def load(paths):
    """Merge the sweeps along the layer axis, ordered by depth."""
    zs = [np.load(p, allow_pickle=True) for p in paths]
    layers = np.concatenate([z["layers"] for z in zs])
    order = np.argsort(layers)
    out = {"layers": layers[order], "us": zs[0]["us"], "sites": [str(s) for s in zs[0]["sites"]],
           "readout_layer": int(zs[0]["readout_layer"]),
           "prompt_texts": [str(t) for t in zs[0]["prompt_texts"]],
           "prompts": np.asarray(zs[0]["prompts"]),
           "dist_unpatched": np.asarray(zs[0]["dist_unpatched"]),
           "model": str(zs[0]["model"])}
    for k in ("dists", "hell", "speed_fulld", "uout", "stride_in"):
        out[k] = np.concatenate([z[k] for z in zs], axis=1)[:, order]
    # `ring_knot_alpha` is deliberately NOT loaded. It records where the days would fall
    # under a global arc-length reparameterisation, which the sweep does not use -- it samples
    # per arc, putting day k at u = k/7 exactly (see `knot_index`). Reading it as a grid
    # position is the mistake this figure made for several revisions, and leaving it in the
    # loaded dict is an invitation to make it again.
    return out


def knot_index(us, k):
    """Grid index of day `k` in the sweep."""
    return int(np.argmin(np.abs(np.asarray(us) - k / N_DAYS)))


def clean_accuracy(d, off, max_offset=N_DAYS):
    """Unpatched accuracy ON EXACTLY THE PROMPTS THE LADDER SWEEPS."""
    d0, sel, off = d["dist_unpatched"], d["prompts"], np.asarray(off)
    ok = d0[sel][:, :N_DAYS].argmax(1) == (sources(d) + off) % N_DAYS
    return float(ok[off <= max_offset].mean())


def sources(d):
    """The day each prompt actually mentions -- the knot at which a steer is the identity."""
    return np.array([DAYS.index(re.search(r"after (\w+)", t).group(1))
                     for t in d["prompt_texts"]])


def offsets(d):
    return np.array([WORD[re.search(r"is (\w+) days", t).group(1)] for t in d["prompt_texts"]])


def accuracy(d, off, src, max_offset=1):
    """Fraction of the steers, over the kept prompts, that yield the correct answer."""
    D, us = d["dists"], d["us"]
    acc = np.zeros((2, len(d["layers"])))
    for si in range(2):
        for li in range(len(d["layers"])):
            hit = tot = 0
            for pi, o in enumerate(off):
                if o > max_offset:                            # prompt-level restriction
                    continue
                for k in range(N_DAYS):
                    if k == src[pi]:                      # never the no-op knot
                        continue
                    i = knot_index(us, k)
                    hit += int(np.argmax(D[si, li, pi, i, :N_DAYS]) == (k + o) % N_DAYS)
                    tot += 1
            acc[si, li] = hit / tot
    return acc


def hell_speed(dists, stride):
    """Behavior distance moved per unit of ACTIVATION distance -- a gain, not a rate."""
    y = np.sqrt(np.clip(dists, 0, None))
    return np.linalg.norm(np.diff(y, axis=-2), axis=-1) / np.sqrt(2.0) / stride


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
        ax.spines[sp].set_color(PALE)
    ax.tick_params(labelsize=8.3, length=2.5, width=0.6, colors=MID)
    ax.grid(True, lw=0.4, color="#eceef2", zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- A: the real rings, in 3-D
def pca3(C):
    """Orthonormal 3-D frame for a set of centroids, plus the projector into it."""
    ctr = C.mean(0)
    _, _, Vt = np.linalg.svd(C - ctr, full_matrices=False)
    basis = Vt[:3]
    return lambda X: (np.atleast_2d(X) - ctr) @ basis.T


def align(Y, X):
    """Rotation (reflections allowed) carrying Y onto X, by Procrustes on the day pairing."""
    U, _, Vt = np.linalg.svd(Y.T @ X)
    return U @ Vt


def camera(P, tilt=np.deg2rad(28.0)):
    """One orthographic camera for every ring, so the stack is directly comparable."""
    ct, st = np.cos(tilt), np.sin(tilt)
    return np.stack([P[:, 0], P[:, 1] * ct + P[:, 2] * st], 1)


def behavior_ring(d):
    """The behavior manifold: the seven answer-day output distributions, in Hellinger space."""
    d0, sel = d["dist_unpatched"], d["prompts"]
    z = (sources(d) + offsets(d)) % N_DAYS
    P = d0[sel]
    C = np.stack([P[z == k].mean(0) for k in range(N_DAYS)])
    Y, E = np.sqrt(np.clip(C, 0, None)), np.sqrt(np.clip(P, 0, None))
    pred = np.argmin(((E[:, None, :] - Y[None, :, :]) ** 2).sum(-1), axis=1)
    return Y, float((pred == z).mean())


def panel_stack(ax, rings, beh, gf_layer):
    """The measured rings at every depth, plus the behavior manifold they end in."""
    L = [int(v) for v in rings["layers"]]
    keys = [str(k) for k in rings["keys"]]
    which = [str(w) for w in rings["which"]]
    C_all = rings["centroids"]
    idx = {(int(k.split("@")[0][1:]), w): i for i, (k, w) in enumerate(zip(keys, which))}
    rnorm = np.asarray(rings["resid_norm"])

    proj, ref = {}, {}
    for w in ("input", "answer"):
        for layer in L:
            C = C_all[idx[(layer, w)]].astype(np.float64)
            f = pca3(C)
            K = f(C)
            R = align(K, ref[w]) if w in ref else np.eye(3)
            K = K @ R
            ref[w] = K
            S = f(np.asarray(fit_steer_spline(C).forward(np.linspace(0, 1, 241)))) @ R
            g = 1.0 / rnorm[idx[(layer, w)]]
            proj[(layer, w)] = (camera(K) * g, camera(S) * g)

    span = max(np.abs(v[1]).max() for v in proj.values())
    # rings sized to fill most of each row's pitch: they are drawn at TRUE relative
    # scale (each divided by its site's residual norm), so this is one global factor
    step, half = 1.02, 0.42 / span
    n = len(L)
    XI, XA = 0.30, 2.15
    Y0 = 1.30
    ytop = Y0 + (n - 1) * step

    def yof(li):                 # L2 at the top, L31 at the foot, behavior below them all
        return Y0 + (n - 1 - li) * step

    ax.set_aspect("equal")
    ax.set_xlim(-0.62, 4.60)
    ax.set_ylim(-2.05, ytop + 1.15)
    ax.axis("off")

    for x, c, lab in ((XI, IN_C, "input-day ring\n(weekday token)"),
                      (XA, ANS_C, "answer-day ring\n(answer slot)")):
        # the column heading already names the site; "steered here" under both of them said
        # nothing that distinguished one column from the other
        ax.text(x, ytop + 0.30, lab, ha="center", va="bottom", fontsize=10.3, color=c,
                linespacing=1.35)

    for li, layer in enumerate(L):
        y0 = yof(li)
        ax.text(-0.60, y0, f"L{layer}", ha="left", va="center", fontsize=10.5, color=MID)
        for x0, w, c in ((XI, "input", IN_C), (XA, "answer", ANS_C)):
            K, S = proj[(layer, w)]
            ax.plot(x0 + S[:, 0] * half, y0 + S[:, 1] * half, color=c, lw=1.0, alpha=0.55,
                    zorder=3, solid_capstyle="round")
            ax.scatter(x0 + K[:, 0] * half, y0 + K[:, 1] * half, s=9.0, c=DAY_C, zorder=4,
                       linewidths=0)

    Yb, rec_b = beh
    fb = pca3(Yb)
    Kb = camera(fb(Yb))
    Sb = camera(fb(np.asarray(fit_steer_spline(Yb).forward(np.linspace(0, 1, 241)))))
    gb = 0.33 / np.abs(Sb).max()        # its own scale: the Hellinger sphere, not the stream
    xb = 0.5 * (XI + XA)
    ax.plot(xb + Sb[:, 0] * gb, Sb[:, 1] * gb, color=OUT_C, lw=1.0, alpha=0.55, zorder=3)
    ax.scatter(xb + Kb[:, 0] * gb, Kb[:, 1] * gb, s=9.0, c=DAY_C, zorder=4, linewidths=0)
    ax.text(-0.60, 0.0, "output", ha="left", va="center", fontsize=10.5, color=MID)
    ax.text(xb, -0.44, "behavior manifold", ha="center", va="top", fontsize=10.5, color=OUT_C)
    for x in (XI, XA):
        ax.annotate("", xy=(xb + 0.30 * np.sign(x - xb), 0.24),
                    xytext=(x, yof(n - 1) - 0.26),
                    arrowprops=dict(arrowstyle="-|>", color=PALE, lw=0.8))

    # The two Goodfire call-outs ("Goodfire steer here" / "and measure here") are dropped: the
    # panel's own labels already say which ring is which and where the behaviour manifold sits,
    # and the call-outs were the only use of GF, whose hue was dE 2.7 from the Sunday knot
    # drawn in the same panel.

    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.6, color=DAY_C[k],
                              label=DAYS[k][:3]) for k in range(N_DAYS)],
              loc="upper center", bbox_to_anchor=(0.33, 0.055), ncol=4, fontsize=9.7,
              frameon=False, handletextpad=0.15, columnspacing=0.75, labelspacing=0.28)


# --------------------------------------------------------------------------- B: control
def panel_acc(ax, d, acc, clean, n_prompts, max_offset, kept=()):
    """Steering accuracy against patch layer, one curve per site."""
    L = d["layers"]
    # With the no-op knot excluded a site the model ignores scores 0, and uniform guessing
    # over the seven days scores 1/7 -- so this band really is chance.
    ax.axhspan(0, 1.0 / N_DAYS, color="#f1f2f5", zorder=0)
    # The no-op knot is excluded, so every counted steer asks for a day the model would not
    # say on its own: ignoring the patch scores 0, and 1/7 is what GUESSING would score.
    ax.axhline(clean, color=PALE, lw=1.0, zorder=1)
    # at the LEFT edge both the line and the weekday curve sit at 1.0, so the label read as
    # if it named the blue curve. At the right edge nothing is near it.
    ax.text(L[-1], clean + 0.015, "unsteered", fontsize=7.6, color=MID, va="bottom",
            ha="right")
    for si, (nm, c) in enumerate((("weekday token", IN_C), ("answer slot", ANS_C))):
        ax.plot(L, acc[si], "o-", color=c, lw=1.9, ms=4.2, label=nm, zorder=3)
    ax.set_xlabel("Steered layer", fontsize=9.0, color=INK)
    # "<= 1" reads as if something sits below it; the smallest offset in this set IS 1, so
    # the bound and the value coincide. Name the offsets actually kept.
    ktxt = (f"= {kept[0]}" if len(kept) == 1 else
            f"in {{{', '.join(str(v) for v in kept)}}}" if kept else f"≤ {max_offset}")
    ax.set_ylabel("Steers giving the correct answer\n"
                  f"(prompts with offset {ktxt})", fontsize=9.0, color=INK)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8.2, frameon=False, loc="center left")
    tidy(ax)


# ------------------------------------------------------- C: Goodfire's scatter, one stage on
def pair_distances(d, si, li, pi):
    """Cumulative on-manifold distance between every pair of days, at three stages."""
    us = d["us"]
    step_act = d["stride_in"][si, li]                                   # [U-1]
    sp = d["speed_fulld"][si, li, pi]
    step_res = sp * step_act if not np.all(np.isnan(sp)) else None
    dd = np.sqrt(np.clip(d["dists"][si, li, pi], 0, None))
    step_beh = np.linalg.norm(np.diff(dd, axis=0), axis=1) / np.sqrt(2.0)

    def cum(s):
        return None if s is None else np.concatenate([[0.0], np.cumsum(s)])

    c_act, c_res, c_beh = cum(step_act), cum(step_res), cum(step_beh)
    ki = [knot_index(us, k) for k in range(N_DAYS)]
    out = []
    for a in range(N_DAYS):
        for b in range(a + 1, N_DAYS):
            ia, ib = sorted((ki[a], ki[b]))
            fwd = c_act[ib] - c_act[ia]
            take_fwd = fwd <= c_act[-1] - fwd            # the geodesic, chosen ONCE

            def arc(c):
                if c is None:
                    return np.nan
                v = c[ib] - c[ia]
                return v if take_fwd else c[-1] - v
            gap = min(abs(a - b), N_DAYS - abs(a - b))   # cyclic day gap: 1, 2 or 3
            out.append((arc(c_act), arc(c_res), arc(c_beh), gap))
    return np.array(out)


def isometry_fit(x, y):
    """Slope through the origin and the correlation, i.e. how well y = k*x holds."""
    k = float(np.sum(x * y) / np.sum(x * x))
    r = float(np.corrcoef(x, y)[0, 1])
    return k, r


def panel_goodfire(ax, d, off, layer, si):
    """Their Fig. 2a, reproduced: is the steer a SCALED ISOMETRY into behavior space?"""
    li = list(d["layers"]).index(layer)
    P = np.concatenate([pair_distances(d, si, li, pi) for pi in range(len(off))])
    x, y = P[:, 0], P[:, 2]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    k, r = isometry_fit(x, y)
    for g, m in ((1, "o"), (2, "s"), (3, "^")):
        sel = P[ok, 3] == g
        ax.scatter(x[sel], y[sel], s=11, marker=m, color=ANS_C if si else IN_C, alpha=0.45,
                   linewidths=0, zorder=3, label=f"{g} day{'s' if g > 1 else ''} apart")
    xs = np.linspace(0, x.max() * 1.04, 2)
    ax.plot(xs, k * xs, color=INK, lw=1.1, zorder=4)
    ax.set_xlim(0, x.max() * 1.06)
    ax.set_ylim(0, y.max() * 1.10)
    ax.set_xlabel(r"on-manifold distances in activation space  $d_{\mathcal{M}_h}$",
                  fontsize=8.8, color=INK)
    ax.set_ylabel("On-manifold distances in\n"
                  r"behavior space  $d_{\mathcal{M}_y}$", fontsize=8.8, color=INK)
    ax.text(0.035, 0.955, f"{'answer slot' if si else 'weekday token'}, L{layer}\n"
            f"$d_{{\\mathcal{{M}}_y}} = {k:.3g}\\,d_{{\\mathcal{{M}}_h}}$   $r = {r:.3f}$",
            transform=ax.transAxes, fontsize=7.9, color=INK, va="top", linespacing=1.6)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right", handletextpad=0.2,
              labelspacing=0.25)
    tidy(ax)


def panel_cascade(ax, d, off):
    """The same measurement one stage earlier, averaged so the shape is readable."""
    L = list(d["layers"])
    site_of = {lay: (0 if lay <= 18 else 1) for lay in L}
    blues = [lay for lay in L if site_of[lay] == 0]
    reds = [lay for lay in L if site_of[lay] == 1]
    RO = d["readout_layer"]
    for layer in L:
        li, si = L.index(layer), site_of[layer]
        fam = blues if si == 0 else reds
        c = shade(BLUES if si == 0 else REDS, fam.index(layer), len(fam))
        P = np.concatenate([pair_distances(d, si, li, pi) for pi in range(len(off))])
        if np.all(np.isnan(P[:, 1])):
            continue                # at or past the readout there is no L28 residual to move
        mx, my, ey = [], [], []
        for g in (1, 2, 3):
            q = P[P[:, 3] == g]
            mx.append(q[:, 0].mean())
            my.append(q[:, 1].mean())
            ey.append(q[:, 1].std())
        ax.errorbar(mx, my, yerr=ey, color=c, lw=1.5, marker="o", ms=4, capsize=2,
                    elinewidth=0.9, zorder=3)
    ax.set_xlabel(r"on-manifold distances in activation space  $d_{\mathcal{M}_h}$",
                  fontsize=8.8, color=INK)
    ax.set_ylabel(f"distance moved in the L{RO}\nresidual stream (4096-d)", fontsize=8.8,
                  color=INK)
    hs = [Line2D([], [], color=shade(BLUES, i, len(blues)), lw=1.6, marker="o", ms=3.5,
                 label=f"L{lay}") for i, lay in enumerate(blues)]
    hs += [Line2D([], [], color=shade(REDS, i, len(reds)), lw=1.6, marker="o", ms=3.5,
                  label=f"L{lay}") for i, lay in enumerate(reds) if lay < RO]
    ax.legend(handles=hs, fontsize=7.1, frameon=False, ncol=2, loc="upper left",
              handletextpad=0.4, columnspacing=0.8, labelspacing=0.25,
              title="steer: weekday | answer", title_fontsize=8.7)
    ax.text(0.985, 0.04, f"three points per layer: mean over prompts and day\npairs at cyclic "
            f"gap 1, 2, 3.  L≥{RO} is absent — the\nresidual is read before those blocks run",
            transform=ax.transAxes, fontsize=7.0, color=MID, ha="right", va="bottom",
            linespacing=1.55)
    tidy(ax)


def in_control(d, acc, min_acc=0.5):
    """Layers whose steer actually controls the answer, each with the site that does it."""
    L = list(d["layers"])
    pick = {lay: int(np.argmax(acc[:, i])) for i, lay in enumerate(L)}
    keep = [lay for i, lay in enumerate(L) if acc[pick[lay], i] >= min_acc]
    return keep, pick


# ---------------------------------------------------------------------------- D: the ladder
def panel_ladder(axes, d, conds=((2, 0), (28, 1))):
    """Behavior distance against activation distance, one layer per axis."""
    L = list(d["layers"])
    for ai, (ax, (layer, si)) in enumerate(zip(axes, conds)):
        li = L.index(layer)
        c = IN_C if si == 0 else ANS_C
        x = np.concatenate([[0.0], np.cumsum(d["stride_in"][si, li])])
        x = x / x[-1]
        dd = np.sqrt(np.clip(d["dists"][si, li], 0, None))
        Y = np.concatenate([np.zeros((dd.shape[0], 1)),
                            np.cumsum(np.linalg.norm(np.diff(dd, axis=1), axis=2)
                                      / np.sqrt(2.0), axis=1)], axis=1)
        for y in Y:
            ax.plot(x, y / y[-1], color=c, lw=0.7, alpha=0.16, zorder=3)
        M = Y.mean(0) / Y.mean(0)[-1]
        ax.plot(x, M, color=c, lw=2.2, zorder=4)
        ki = [knot_index(d["us"], k) for k in range(N_DAYS)]
        kx = [x[i] for i in ki]
        for v in kx:
            ax.axvline(v, color=PALE, ls=":", lw=0.8, zorder=0)
        # the knots on the curve itself, day-coloured as in panels A and E
        ax.scatter(kx, M[ki], s=17, c=DAY_C, zorder=6, edgecolors="white", linewidths=0.6)
        # TWO levels of tick. The axis quantity is a fraction of the ring's total activation
        # distance, so the fractions are the major ticks and are actually printed -- calling
        # the axis a fraction while labelling it only with day names left the reader nothing
        # to check it against. The day knots go on the minors underneath, thinned to every
        # other day because the L2 ring's uneven spacing ran "TueWedThu" together.
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        # Days label their own dotted lines, all at the TOP of the axes rather than tracking
        # the curve: a second row of tick labels collided with the fractions, and following
        # the curve put them among the knot dots. Mon is dropped -- it is the same knot as
        # Sun (the ring's wrap point) and sat clipped against the left spine.
        for kk, vx in enumerate(kx):
            if kk == 0:
                continue
            ax.text(vx, 0.985, DAYS[kk][:3], rotation=90, fontsize=6.0, color=MID,
                    ha="right", va="top", zorder=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.text(0.965, 0.045,
                f"steered at {'answer slot' if si else 'weekday token'}, L{layer}",
                transform=ax.transAxes, fontsize=8.0, color=c, va="bottom", ha="right")
        if ai == 0:                      # side by side: one y label serves both
            ax.set_ylabel("Behavior distance\n(fraction)", fontsize=8.4, color=INK)
        tidy(ax)
        ax.grid(False)
    # ONE y range for both panels. The whole claim of this panel is that the early steers
    # swing about ten times and the late ones about two, and separate autoscaled axes hide
    # exactly that by stretching the flat family to fill its box.
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo * 0.22, hi)          # log axis: room under the curves for the legend
    for ax in axes:                      # both are now bottom axes
        # ONE line. Two lines under both sub-panels cost about as much height as C's plot area
        # and printed straight through D's title.
        ax.set_xlabel("Activation distance (fraction of ring)", fontsize=8.0,
                      color=INK)


# ----------------------------------------------------------------------------- E: the relay
def panel_relay(ax, d, acc, min_acc=0.5):
    """Behavior gain around the ring, all controlling layers on one axis."""
    us, L = d["us"], list(d["layers"])
    keep, pick = in_control(d, acc, min_acc)
    dropped = [l for l in L if l not in keep]
    blues = [lay for lay in keep if pick[lay] == 0]
    reds = [lay for lay in keep if pick[lay] == 1]
    for lay in keep:
        li, si = L.index(lay), pick[lay]
        fam = blues if si == 0 else reds
        c = shade(BLUES if si == 0 else REDS, fam.index(lay), len(fam))
        y = hell_speed(d["dists"][si, li], d["stride_in"][si, li]).mean(0)
        xa = np.concatenate([[0.0], np.cumsum(d["stride_in"][si, li])])
        xa = xa / xa[-1]
        xm = 0.5 * (xa[1:] + xa[:-1])                    # step midpoints, in arc length
        ax.plot(xm, y / y.mean(), color=c, lw=1.5, zorder=3)
        if lay == keep[0]:
            rule_x = [xa[knot_index(us, k)] for k in range(N_DAYS)]
    for v in rule_x:
        ax.axvline(v, color=PALE, ls=":", lw=0.7, zorder=0)
    ax.axhline(1.0, color=PALE, lw=0.9, zorder=1)
    # ticks follow the rules, which are now at the days' TRUE arc positions rather than at
    # k/7; leaving them at k/7 would have named each dotted line after the wrong day
    ax.set_xticks(rule_x)
    ax.set_xticklabels([nm[:3] for nm in DAYS], fontsize=7.2, color=MID)
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Steered coordinate", fontsize=8.4, color=INK)
    ax.set_ylabel("Behavior gain  " r"$d_{\mathcal{M}_y}/d_{\mathcal{M}_h}$" "\n"
                  "relative to this curve's ring average", fontsize=8.0, color=INK)
    hs = [Line2D([], [], color=shade(BLUES, i, len(blues)), lw=1.6, label=f"L{lay}")
          for i, lay in enumerate(blues)]
    hs += [Line2D([], [], color=shade(REDS, i, len(reds)), lw=1.6, label=f"L{lay}")
           for i, lay in enumerate(reds)]
    ax.legend(handles=hs, fontsize=7.1, frameon=False, ncol=2, loc="lower left",
              handletextpad=0.5, columnspacing=0.8, labelspacing=0.25,
              title=f"steering accuracy > {min_acc:.2f}\nweekday   |   answer",
              title_fontsize=7.1)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * 0.30, hi)          # log axis: room under the curves for the legend
    # Which layers were dropped is still printed to stdout and is implicit in the legend, which
    # lists exactly the layers drawn; the in-panel call-out was restating the legend.
    if dropped:
        print(f"[panel D] excluded (steering accuracy <= {min_acc:.2f}): "
              + ", ".join(f"L{l}" for l in dropped))
    tidy(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rings", default="figures/rings.npz",
                    help="ring centroids, from `alpha_ladder_sites.py --dump-rings`")
    ap.add_argument("--npz", nargs="+", default=["figures/alpha_ladder_sites.npz"],
                    help="one sweep covering every layer and every (day x offset) prompt. "
                         "Still accepts several files -- they are merged along the layer "
                         "axis -- but a single run is preferable: the accuracy curve and the "
                         "unpatched reference then describe the same 49 prompts by "
                         "construction rather than by the caller lining them up.")
    ap.add_argument("--out", default="figures/fig_combined_arith.pdf")
    ap.add_argument("--ladder-layer", type=int, default=2,
                    help="which steer layer panel D draws the accumulation for")
    ap.add_argument("--ladder-site", choices=["weekday", "answer"], default="weekday",
                    help="which site panel D steers; at layers past ~18 only 'answer' has "
                         "any control, so a weekday ladder there is a flat line by default")
    ap.add_argument("--relay-min-acc", type=float, default=0.5,
                    help="panel D keeps a layer only if its better site reaches this "
                         "steering accuracy in panel B")
    ap.add_argument("--max-offset", type=int, default=1,
                    help="panel B keeps only prompts whose arithmetic offset is at most this")
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()

    d = load(args.npz)
    rings = np.load(args.rings, allow_pickle=True)
    off = offsets(d)
    clean = clean_accuracy(d, off, args.max_offset)
    acc = accuracy(d, off, sources(d), args.max_offset)
    print("layers:", d["layers"].tolist())
    print(f"swept {len(off)} prompts, offsets {sorted(set(off.tolist()))}; "
          f"unpatched on exactly those: {clean:.3f}")
    nkept = int((off <= args.max_offset).sum())
    print(f"panel B: {nkept} prompts with offset <= {args.max_offset}, "
          f"all {N_DAYS - 1} non-identity knots each")
    for si, nm in enumerate(("weekday", "answer")):
        print(f"  {nm:8s} accuracy " + " ".join(f"{v:.2f}" for v in acc[si]))

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(9.4, 10.7))
    # A keeps the left column; the other three stack down the right. C's two conditions go
    # SIDE BY SIDE rather than stacked, which leaves D a full-width row of its own instead of
    # a narrow column it had to be stretched to fill.
    gs = GridSpec(3, 2, figure=fig, hspace=0.20, wspace=0.06, width_ratios=[1.10, 1.30],
                  height_ratios=[0.44, 0.42, 0.54],
                  left=0.035, right=0.988, top=0.952, bottom=0.075)
    a = fig.add_subplot(gs[:, 0])
    # B is a narrow two-line plot; taking the whole cell height turned it portrait, so it
    # gets the top of its cell and leaves the rest empty.
    b = fig.add_subplot(gs[0, 1].subgridspec(20, 1)[0:17])
    csub = gs[1, 1].subgridspec(20, 1)[0:15].subgridspec(1, 2, wspace=0.28)
    dax = [fig.add_subplot(csub[0]), fig.add_subplot(csub[1])]
    eax = fig.add_subplot(gs[2, 1].subgridspec(20, 1)[0:16])

    panel_stack(a, rings, behavior_ring(d), gf_layer=d["readout_layer"])
    panel_acc(b, d, acc, clean, nkept, args.max_offset,
              kept=sorted(set(off[off <= args.max_offset].tolist())))
    panel_ladder(dax, d)
    panel_relay(eax, d, acc, min_acc=args.relay_min_acc)

    # lettering stays contiguous after the removal, so D becomes C and E becomes D
    # Headers start at the Y-LABEL's left edge, not at the axes box. The gutter holding the
    # tick labels and the rotated ylabel is as wide as those happen to render, so a fixed
    # offset from the axes lands somewhere different in every panel.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, letter, title in ((a, "A", "Weekday representations in the model"),
                              (b, "B", "Layerwise accuracy of steering"),
                              (dax[0], "C", "Steering snaps to weekdays"),
                              (eax, "D", "Path speed is lowest at the weekdays")):
        bb = ax.get_position()
        x = bb.x0 - 0.042
        lab = ax.yaxis.label
        if lab.get_text():
            x = (lab.get_window_extent(rend).transformed(fig.transFigure.inverted())).x0
        t = fig.text(x, bb.y1 + 0.016, letter, fontsize=13.4, fontweight="bold",
                     color=INK, ha="left", va="bottom")
        fig.canvas.draw()
        w_ = (t.get_window_extent(rend).transformed(fig.transFigure.inverted())).width
        fig.text(x + w_ + 0.006, bb.y1 + 0.018, title, fontsize=10.5, color=INK, ha="left",
                 va="bottom")
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    print("[figure] wrote", args.out)
    if args.also_png:
        fig.savefig(args.out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
        print("[figure] wrote", args.out.replace(".pdf", ".png"))


if __name__ == "__main__":
    main()
