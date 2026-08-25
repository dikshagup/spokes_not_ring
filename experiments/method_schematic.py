"""Drawing helpers for the method schematic: the (u, r) chart and the intervention.

Lifted verbatim from ``figure_main.py`` on master, whose panel A this was. Only that
file's unrelated main() is gone; the function bodies are unchanged, so what they draw is
what was published.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from weekday_manifold.manifold.days import DAYS, N_DAYS  # noqa: E402

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
# The intervention's hue. Violet rather than indigo because the day ramp ends on blue
# (Sun #0072B2, Sat #56B4E9) and the steering ray points straight at that knot.
STEER = "#33207a"
DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2"]


def blank(ax):
    """Strip an axes to a bare 0-1 drawing surface."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def box(ax, x, y, w, h, label, fc="white", ec=MID, fs=7.0, lw=0.9, tc=INK):
    """A rounded labelled box in axes coordinates."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=tc,
            zorder=3, linespacing=1.45)


def arrow(ax, p0, p1, color=MID, lw=0.9):
    """A straight arrow in axes coordinates."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=color,
                                 lw=lw, shrinkA=1, shrinkB=1, zorder=4))


def span(ax, t):
    """Axes-fraction bbox actually occupied by an already-placed Text."""
    ax.figure.canvas.draw()
    bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    return bb.transformed(ax.transAxes.inverted())


def panel_headers(fig, lettered, dy=0.075, fs=12.0):
    """Letter and title on one baseline at each panel's top-left, in figure coordinates."""
    line_h = 9.6 * 1.45 / (fig.get_figheight() * 72.0)
    for letter, title, cell in lettered:
        fig.text(cell.x0, cell.y1 + dy + title.count("\n") * line_h, letter, fontsize=fs,
                 fontweight="bold", color=INK, ha="left", va="bottom")
        fig.text(cell.x0 + 0.024, cell.y1 + dy + 0.002, title, fontsize=9.6, color=INK,
                 ha="left", va="bottom", linespacing=1.45)


def panel_traces(ax, spline, mu, Vt, r_show=(0.4, 1.0, 1.5)):
    """The (u, r) coordinate grid drawn on the loop's real projected shape."""
    blank(ax)
    # Equal aspect, or the axes box stretches the projection and part of the irregularity
    # on show is the layout rather than the geometry.
    ax.set_aspect("equal")

    ud = np.linspace(0, 1, 400, endpoint=False)
    ring = (spline.forward(ud) - mu) @ Vt[:2].T
    knots = (spline.forward(np.arange(N_DAYS) / N_DAYS) - mu) @ Vt[:2].T
    sc = 0.395 / (max(r_show) * np.abs(ring).max())
    cx, cy = 0.50, 0.660

    for r in r_show:                                        # constant r: u sweeps
        P = ring * r * sc + [cx, cy]
        on = abs(r - 1.0) < 1e-9
        ax.plot(*np.vstack([P, P[:1]]).T, color=MID if on else PALE,
                lw=1.9 if on else 0.9, ls="-" if on else (0, (3, 2)),
                zorder=3 if on else 2)
        q = ring[330] * r * sc
        lab = ax.text(*(q + q / max(np.linalg.norm(q), 1e-12) * 0.030 + [cx, cy]),
                      f"r={r:g}", fontsize=6.6,
                      color=MID if on else PALE, ha="center", va="center", zorder=5)
        lab.set_bbox(dict(fc="white", ec="none", pad=0.6))

    for k in range(N_DAYS):                                 # constant u: r sweeps
        v = knots[k] * max(r_show) * sc
        ax.plot([cx, cx + v[0]], [cy, cy + v[1]], color=PALE, lw=0.6, ls=(0, (2, 2)),
                zorder=1)
    ax.scatter(*(knots * sc + [cx, cy]).T, s=22, color=DAY_C, ec="white", lw=0.7, zorder=6)
    for k in range(N_DAYS):
        # placed radially OUTWARD from the knot: a fixed pixel offset puts whichever label
        # points inward on top of the ring line.
        d = knots[k] / max(np.linalg.norm(knots[k]), 1e-12)
        t = ax.text(*(knots[k] * sc + d * 0.085 + [cx, cy]), DAYS[k][:3], fontsize=5.8,
                    color=INK, ha="center", va="center", zorder=7)
        t.set_bbox(dict(fc="white", ec="none", pad=0.5))

    seg = ring[238:292] * sc + [cx, cy]                     # highlight one constant-r sweep
    ax.plot(*seg.T, color=STEER, lw=1.8, zorder=5)
    ax.annotate("", xy=seg[-1], xytext=seg[-6],
                arrowprops=dict(arrowstyle="-|>", color=STEER, lw=1.8, mutation_scale=8,
                                shrinkA=0, shrinkB=0), zorder=5)
    a26 = seg[26] - [cx, cy]
    lab_xy = seg[26] + a26 / max(np.linalg.norm(a26), 1e-12) * 0.115
    t = ax.text(*lab_xy, "r fixed,\nu sweeps", fontsize=6.8, color=STEER, linespacing=1.4,
                ha="center", va="center", zorder=7)
    t.set_bbox(dict(fc="white", ec="none", pad=0.5))

    kx = knots[6] * max(r_show) * sc                        # highlight one constant-u ray
    ax.plot([cx, cx + kx[0]], [cy, cy + kx[1]], color=STEER, lw=1.2, zorder=4)
    ax.annotate("u fixed,\nr sweeps", (cx + kx[0], cy + kx[1]),
                textcoords="offset points", xytext=(2, 2), fontsize=6.8, color=STEER,
                linespacing=1.4, zorder=6)
    # mu wears STEER, not INK: it is where the r-sweep starts.
    ax.plot([cx], [cy], marker="o", ms=4.0, color=STEER, zorder=6)
    ax.annotate("μ", (cx, cy), textcoords="offset points", xytext=(0, -13), fontsize=8.2,
                color=INK, ha="center", zorder=6)

    ylo = min(float((ring * max(r_show) * sc)[:, 1].min()) + cy,   # outer ring, and
              float(lab_xy[1]) - 0.055)                             # the sweep label below it
    ax.text(0.5, ylo - 0.045, "$p(u,r)=\\mu+r\\left(\\sigma(u)-\\mu\\right)$",
            ha="center", va="top", fontsize=8.0, color=INK)


def panel_pipeline(ax, day_pos, ro_pos, tokens, patch_layer, read_layer,
                   measures="$\\|J\\|_F$ · $\\|J\\hat t\\|$ · $\\|J\\hat r\\|$"):
    """The intervention, drawn as the forward pass it is."""
    blank(ax)

    n = len(tokens)
    gap = 0.004
    # narrower and shallower than the strip needs to be: the boxes locate two sites, they
    # are not content, so they should not out-weigh the two equations beneath them
    w = min(0.055, (0.80 - (n - 1) * gap) / n)
    x0 = 0.5 - (n * w + (n - 1) * gap) / 2
    for i, t in enumerate(tokens):
        x = x0 + i * (w + gap)
        hot, rd = i == day_pos, i == ro_pos
        box(ax, x, 0.855, w, 0.062, t.replace(" ", "·"),
            fc="#eceaf6" if hot else "white",
            ec=STEER if (hot or rd) else PALE, fs=4.6,
            lw=1.3 if (hot or rd) else 0.7)
    hot_x = x0 + day_pos * (w + gap) + w / 2
    ro_x = x0 + ro_pos * (w + gap) + w / 2
    ax.text(hot_x, 0.968, f"steered\nat layer {patch_layer}", ha="center", va="center",
            fontsize=6.6, color=STEER, linespacing=1.3)
    ax.text(ro_x, 0.968, f"read\nat layer {read_layer}", ha="center", va="center",
            fontsize=6.6, color=STEER, linespacing=1.3)

    arrow(ax, (hot_x, 0.855), (hot_x, 0.818), color=STEER, lw=1.2)

    # Drawn in pieces at known x, so the bracket below can be positioned exactly.
    ex, ey = 0.150, 0.735
    ax.text(ex, ey, f"At layer {patch_layer}:", ha="left", va="center", fontsize=8.0,
            color=INK)
    lhs = ax.text(ex + 0.205, ey, "$x\\;\\leftarrow\\;A_{\\mathrm{prompt}}\\;+$",
                  ha="left", va="center", fontsize=8.2, color=STEER)
    gl = span(ax, lhs).x1 + 0.018
    grp = ax.text(gl, ey, "$\\left(p(u,r)-C_s\\right)$", ha="left", va="center", fontsize=8.2,
                  color=STEER)
    bbg = span(ax, grp)
    gl, gr_ = bbg.x0, bbg.x1
    # Hung off the MEASURED bottom of the group: the parenthesised expression is the tallest
    # thing on the line and its descenders run into a bracket pinned at a fixed y.
    bt = bbg.y0 - 0.012
    gc = (gl + gr_) / 2

    # C_s is the SOURCE -- the centroid of the prompt's own day -- not the target.
    ax.plot([gl, gl, gr_, gr_], [bt, bt - 0.010, bt - 0.010, bt], color=STEER, lw=0.8,
            solid_joinstyle="miter")
    arrow(ax, (gc, bt - 0.010), (gc, bt - 0.034), color=STEER, lw=0.8)
    ax.text(gc, bt - 0.042,
            "the displacement: out from $C_s$, the centroid\n"
            "of the prompt's own day, to the target $p(u,r)$",
            ha="center", va="top", fontsize=6.6, color=STEER, linespacing=1.5)

    jy = 0.390
    ax.text(ex, jy, f"Layers {patch_layer} → {read_layer}:", ha="left",
            va="center", fontsize=8.0, color=INK)
    ax.text(ex + 0.205, jy,
            "$J=\\partial\\,\\mathrm{res}^{(t_{\\mathrm{read}})}_{L%d}\\,/\\,"
            "\\partial\\,x^{(t_{\\mathrm{day}})}_{L%d}$"
            "$\\quad\\Rightarrow\\quad$"
            % (read_layer, patch_layer) + measures,
            ha="left", va="center", fontsize=8.2, color=INK, zorder=3)
