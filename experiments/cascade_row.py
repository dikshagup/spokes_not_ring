"""Drawing module for figure 1's panels A and B. No main.

Lifted verbatim from figure_cascade_pub.py / figure_cascade_mention.py on master, whose own
plates are not part of this branch.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401,E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: F401,E402


INK, MID, PALE, STEER = "#16181d", "#5b6270", "#aeb4c0", "#6A4C93"
# Every panel's x axis is the same quantity: distance travelled along the ring from the
# reference day, as a fraction of the loop. That is exactly Goodfire's on-manifold
# activation distance, so it carries their symbol and one wording everywhere.


def xlab(day=None, symbol=False):
    """The x axis every panel shares: WHERE on the ring the steer points."""
    tail = "  " r"$= d_{\mathcal{M}_h}$" if symbol else ""
    return f"Steered position on the ring\n(arc length, fraction of loop){tail}"


GRID = "#e8eaef"
# One left rail for the panel letter and its title, so they do not sit on different margins.
LETTER_X = -0.24
TITLE_X = -0.135      # far enough right of the letter to clear it at 11.5pt
TITLE_Y = 1.13        # above the day names, which sit at 1.012
DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# How each checkpoint should be NAMED in a figure. The npz carries the HF repo id, which is
# what identifies the weights but not what a reader should see -- "gpt2-xl" and
# "Mistral-7B-v0.1" are paths, not names.
NICE = {"meta-llama/Llama-3.1-8B": "Llama-3.1-8B",
        "mistralai/Mistral-7B-v0.1": "Mistral-7B",
        "gpt2-xl": "GPT-2 XL"}


def nice_name(model, override=None):
    if override:
        return override
    return NICE.get(model, NICE.get(model.split("/")[-1], model.split("/")[-1]))


def tidy(ax, grid_axis=None):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color(PALE)
    ax.tick_params(labelsize=8.3, length=2.5, width=0.7, colors=MID, pad=2)
    if grid_axis:
        ax.grid(True, axis=grid_axis, lw=0.5, color=GRID, zorder=0)
        ax.set_axisbelow(True)


def day_marks(ax, xs, day_ids, y_frac=None):
    """Dotted verticals at the day knots, named horizontally ABOVE the axes."""
    for j in range(0, 7):
        if j:
            ax.axvline(xs[j], color=PALE, ls=(0, (1, 2)), lw=0.7, zorder=1)
        ax.text(xs[j], 1.012, DAYS[day_ids[j]], transform=ax.get_xaxis_transform(),
                rotation=0, fontsize=7.0, color=MID, ha="center", va="bottom", zorder=7)


def knot_order(kidx, ref, n):
    """Where each day knot sits after a curve is rolled to start at `ref`, and which day."""
    kidx = np.asarray(kidx, int)
    order = [(int(ref) + j) % 7 for j in range(7)]
    pos = np.array([(kidx[d] - kidx[int(ref)]) % n for d in order])
    return pos, order


def by_day(arr, days, kidx, closed=True, roll=True):
    """Roll each prompt to its own day, then average WITHIN each starting day."""
    a = arr[:, :-1] if closed else arr
    out = []
    for k in range(7):
        rows = [(np.roll(a[i], -int(kidx[k]), axis=0) if roll else a[i])
                for i in range(len(days)) if days[i] == k]
        out.append(np.mean(rows, axis=0) if rows else np.full_like(a[0], np.nan))
    return np.stack(out)


def arc_x(stride, day, kidx):
    """Cumulative arc length from THIS day's knot, as a fraction of the loop."""
    rolled = np.roll(np.asarray(stride), -int(kidx[int(day)]))
    cum = np.concatenate([[0.0], np.cumsum(rolled)])
    return cum / cum[-1]


def draw_row(fig, axA, axC, axB, npz, position="readout", fix_in=2, compare=28,
             readout_depth=None, ref_in=0, depth_cmap="viridis", depth_every=2,
             normalise=False):
    """One model's three panels, drawn into axes the caller owns."""
    z = np.load(npz, allow_pickle=True)
    layers = [int(v) for v in z["layers"]]
    us = np.asarray(z["us"])
    q = 0 if position == "weekday" else 1
    d_layer = np.asarray(z["d_layer"], np.float32)[..., q]
    step_layer = np.asarray(z["step_layer"], np.float32)[..., q]
    scale = np.asarray(z["resid_scale"], np.float64)[:, q]
    nL = d_layer.shape[-1]
    RO = int(readout_depth if readout_depth is not None else z["readout_layer"])
    assert 0 <= RO < nL, f"readout depth {RO} outside the {nL} recorded depths"
    days = np.asarray(z["prompt_days"], int)
    n_step = len(us) - 1
    KID = np.asarray(z["knot_index"], int)
    ref, fix = int(ref_in), int(fix_in)
    fi = layers.index(fix)
    model = str(z["model"]) if "model" in z.files else "?"
    tok = "weekday token" if position == "weekday" else "full stop"

    # ---- A: the emergence figure's "steering snaps to weekdays" -----------------------
    # Distance travelled in BEHAVIOUR space against distance travelled in ACTIVATION space,
    # both as a fraction of the whole loop, one thin line per prompt. A steer that moved the
    # output in proportion to how far it moved the activation would lie on the diagonal;
    # the staircase is the output holding still near a weekday and then jumping between.
    # Behaviour distance is Hellinger on the restricted simplex (seven weekdays plus an
    # "other" class): sqrt the distributions, Euclidean steps, divide by sqrt 2.
    kid = KID[fi]
    cum_in = arc_x(np.asarray(z["stride_in"])[0, fi], ref, kid)
    ki, kday = knot_order(kid, ref, n_step)

    def behaviour_curve(layer_idx, kidx):
        """Cumulative Hellinger along the walk, per prompt, as a fraction of its own total."""
        roll = -int(kidx[ref])
        dd = np.sqrt(np.clip(np.asarray(z["dists"])[0, layer_idx], 0, None))
        out = []
        for pi in np.where(days == ref)[0]:
            stepb = np.linalg.norm(np.diff(dd[pi], axis=0), axis=1) / np.sqrt(2.0)
            cb = np.concatenate([[0.0], np.cumsum(np.roll(stepb, roll))])
            if cb[-1] > 0:
                out.append(cb / cb[-1])
        return np.array(out)

    axA.plot([0, 1], [0, 1], color=MID, ls=(0, (3, 3)), lw=0.9, zorder=2)
    shown_A = []
    for L, col, dark in ((fix, "#2b5d9e", True), (int(compare), "#b3322c", False)):
        if L not in layers:
            continue
        lj = layers.index(L)
        kj = KID[lj]
        cur = behaviour_curve(lj, kj)
        if not len(cur):
            continue
        xj = arc_x(np.asarray(z["stride_in"])[0, lj], ref, kj)
        kij, kdj = knot_order(kj, ref, n_step)
        m = cur.mean(0)
        lo, hi = np.percentile(cur, [10, 90], axis=0)
        axA.fill_between(xj[:len(m)], lo, hi, color=col, alpha=0.16, lw=0, zorder=3)
        axA.plot(xj[:len(m)], m, color=col, lw=1.9, zorder=5, label=f"steered at L{L}")
        if dark:
            axA.scatter(xj[kij], m[kij], s=20, c=[DAY_C[d] for d in kdj], zorder=6,
                        edgecolors="white", linewidths=0.7)
        shown_A.append(L)
    axA.set_xlim(0, 1); axA.set_ylim(0, 1.02)
    axA.set_xlabel(xlab(DAYS[ref]), fontsize=8.9, color=INK, linespacing=1.4)
    axA.set_ylabel("Behaviour-space distance travelled\n(fraction of total)",
                   fontsize=8.9, color=INK, linespacing=1.4)
    axA.legend(fontsize=7.4, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0),
               handlelength=1.4, labelspacing=0.25)
    axA.text(TITLE_X, TITLE_Y, "Non-uniform downstream behaviour", transform=axA.transAxes,
             fontsize=9.5, color=INK, ha="left", va="baseline")
    tidy(axA)
    day_marks(axA, cum_in[ki], kday)

    # ---- B and C: one steer depth, the readout's motion round the ring ------------------
    steps = by_day(step_layer[0, fi], days, kid, closed=False)[ref]     # [U-1, nL]
    st = np.roll(np.asarray(z["stride_in"], float)[0, fi], -int(kid[ref]))

    # A steer written at block L cannot reach the READOUT POSITION until block L+1's
    # attention runs, so at depths <= L that position is bit-identical to unpatched and
    # every step is exactly zero. Normalising by the total would divide by zero and leave
    # the panel silently blank, so say so instead.
    per = np.stack([np.concatenate([[0.0], np.cumsum(
        np.roll(step_layer[0, fi, pi, :, RO], -int(kid[ref])))])
        for pi in np.where(days == ref)[0]])
    per = per / per[:, -1:][..., None].squeeze(-1)
    cum = np.concatenate([[0.0], np.cumsum(steps[:, RO])])
    assert cum[-1] > 0, (
        f"nothing moves at depth {RO} for a steer at L{fix}: a write at block {fix} reaches "
        f"the readout position only from block {fix + 1} on, so depths <= {fix} are "
        f"untouched by construction. Pass --readout-depth greater than {fix}.")
    cum /= cum[-1]
    axB.plot([0, 1], [0, 1], color=MID, ls=(0, (3, 3)), lw=0.9, zorder=2,
             label="Equal distance per unit steer")
    blo, bhi = np.percentile(per, [10, 90], axis=0)
    axB.fill_between(cum_in[:len(cum)], blo, bhi, color=STEER, alpha=0.18, lw=0, zorder=3)
    axB.plot(cum_in[:len(cum)], cum, color=STEER, lw=2.0, zorder=4, label="Measured")
    axB.scatter(cum_in[ki], cum[ki], s=20, c=[DAY_C[d] for d in kday], zorder=6,
                edgecolors="white", linewidths=0.7)
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.set_xlabel(xlab(DAYS[ref]), fontsize=8.9,
                   color=INK, linespacing=1.4)
    axB.set_ylabel("Cumulative \u2016\u0394 readout resid\u2016\n"
                   f"at L{RO} (fraction of total)", fontsize=8.0, color=INK,
                   linespacing=1.4)
    axB.legend(fontsize=7.0, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.0),
               handlelength=1.4, labelspacing=0.25)
    axB.text(TITLE_X, TITLE_Y, "Non-uniform late-layer residual", transform=axB.transAxes,
             fontsize=9.5, color=INK, ha="left", va="baseline")
    tidy(axB)
    day_marks(axB, cum_in[ki], kday)

    # every other depth: consecutive depths correlate above 0.976 from L20 up, so drawing
    # all of them stacks near-identical lines and hides the ones that do differ.
    deep = list(range(fix + 1, nL, depth_every))
    norm = plt.Normalize(vmin=deep[0], vmax=deep[-1])
    cm = plt.get_cmap(depth_cmap)
    spd = steps / np.maximum(st, 1e-12)[:, None]
    if normalise:
        spd = spd / scale[None, :]
    for r in deep:
        axC.plot(cum_in[:len(spd)], spd[:, r], lw=1.0, zorder=4,
                 color=cm(0.10 + 0.72 * norm(r)))
    axC.set_xlim(0, 1)
    axC.set_ylim(top=float(np.nanmax(spd[:, deep])) * 1.45)
    axC.set_xlabel(xlab(DAYS[ref]), fontsize=8.9,
                   color=INK, linespacing=1.4)
    axC.set_ylabel("\u2016\u0394 readout resid\u2016 / \u2016\u0394 steer\u2016\n"
                   + ("(\u00f7 stream norm)" if normalise else "(raw)"),
                   fontsize=8.9, color=INK, linespacing=1.4)
    axC.text(TITLE_X, TITLE_Y, "Speed peaks between the weekdays", transform=axC.transAxes,
             fontsize=9.5, color=INK, ha="left", va="baseline")
    tidy(axC)
    day_marks(axC, cum_in[ki], kday, y_frac=0.998)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=LinearSegmentedColormap.from_list(
        "deep", cm(0.10 + 0.72 * np.linspace(0, 1, 256))))
    cax = axC.inset_axes([0.31, 0.790, 0.36, 0.026])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Readout depth", fontsize=6.8, color=MID, labelpad=2)
    cb.set_ticks([deep[0], deep[-1]])
    cb.ax.set_xticklabels([f"L{deep[0]}", f"L{deep[-1]}"], fontsize=6.8, color=MID)
    cb.ax.tick_params(length=1.5, width=0.5, colors=MID, pad=1)
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(PALE)

    return dict(model=model, RO=RO, fix=fix, tok=tok, n_prompts=int(d_layer.shape[2]))
