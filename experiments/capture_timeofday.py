#!/usr/bin/env python
"""Capture the clock-family prompts at the weekday token.

Writes experiments/results/timeofday_L28.npz. See repro_fig2_timeofday_with_steering.sh.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.interpolate import CubicSpline

plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
EARLY, LATE, NEUT = "#3b5f9e", "#c8642a", "#5b6270"
DAY_C = ["#7D241D", "#8A4B00", "#7F7700", "#4CA162", "#01B9CF", "#9ABEFF", "#E7CAFF"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# 1..23, NOON INCLUDED. Noon was originally dropped alongside midnight as "ambiguous",
# but the two are not alike: "twelve pm" is read as midday by essentially everyone,
# whereas "twelve am" genuinely straddles the day before and the day after. Dropping
# noon also cost the trajectory its midpoint and inflated the wrap statistic -- with
# noon missing, 11am->1pm spans two hours and measured 3.47x the median one-hour step,
# so the 4.83x quoted for the 11pm->1am wrap was being compared against the wrong
# baseline. Restoring noon makes 11->12->13 two ordinary one-hour steps.
CLOCK_HOURS = tuple(range(1, 24))
# Midnight IS captured -- it belongs in the npz for anyone who wants to test it -- but it
# is deliberately NOT plotted, because "twelve am on Monday" is ambiguous about whether
# it means the start or the end of Monday, and no projection can disentangle that. It
# gets its own family so it never reaches either panel; see the [M] line at the end.
MIDNIGHT_HOUR = 0
# EVERY modifier below is exactly 2 Llama tokens with a leading space, so the weekday
# token sits at ONE absolute position for all three families and any movement of it is
# semantic rather than positional. Bare "morning"/"rainy" are 1 token and would put the
# weekday a position earlier -- the confound the repo's length-matched placebo bank
# (src/weekday_manifold/timeofday/prompts.py) exists to avoid. Verified, then asserted at capture.
WORDS = [("early morning", 7.0), ("late morning", 10.0), ("mid afternoon", 14.0),
         ("early evening", 18.0), ("late evening", 21.0), ("late night", 23.0)]
# Weather / mood, no time-of-day content. "very foggy"/"very stormy" are 3 tokens and
# are therefore excluded rather than quietly length-mismatched.
#
# TWO placebo sub-families, because the first family shares its first token. Ten
# "very X" phrases have "very" in common, which could by itself produce a shared
# displacement and make the placebo band look like a real effect. The second family
# varies BOTH tokens -- every intensifier and every adjective appears once -- so if
# the two families agree, the shared token is not driving anything.
PLACEBOS = ["very rainy", "very quiet", "very sunny", "very windy", "very crowded",
            "very noisy", "very muddy", "very chilly", "very humid", "very cloudy"]
PLACEBOS_VAR = ["quite rainy", "rather quiet", "fairly sunny", "pretty windy",
                "somewhat crowded", "oddly noisy", "unusually muddy", "mildly chilly",
                "slightly humid", "terribly cloudy", "strangely damp", "notably dusty"]


def clock_word(h):
    """24-hour index -> spoken 12-hour phrase, matching Diksha's clock_word()."""
    suffix = "am" if h < 12 else "pm"
    h12 = h % 12
    names = ["twelve", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven"]
    return f"{names[h12]} {suffix}"


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
        ax.spines[sp].set_color(PALE)
    ax.tick_params(labelsize=8.3, length=2.5, width=0.6, colors=MID)
    ax.grid(True, lw=0.4, color="#eceef2", zorder=0)
    ax.set_axisbelow(True)


def build_prompts():
    """(text, day, family, label, hour) for every modifier x day."""
    mods = [(clock_word(h), "clock", f"{h:02d}:00", float(h)) for h in CLOCK_HOURS]
    mods += [(clock_word(MIDNIGHT_HOUR), "clock_midnight",
              f"{MIDNIGHT_HOUR:02d}:00", float(MIDNIGHT_HOUR))]
    mods += [(w, "word", w, hr) for w, hr in WORDS]
    mods += [(p, "placebo_very", p, np.nan) for p in PLACEBOS]
    mods += [(p, "placebo_varied", p, np.nan) for p in PLACEBOS_VAR]
    rows = []
    for phrase, fam, label, hour in mods:
        for d, day in enumerate(DAYS):
            rows.append((f"It was {phrase} on {day}.", d, fam, label, hour))
    return rows


def capture(layer, device, chunk):
    """One forward pass per chunk; read resid_post at the weekday token."""
    import torch
    _SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    sys.path.insert(0, _SRC)
    from weekday_manifold.model import load_model

    rows = build_prompts()
    texts = [r[0] for r in rows]
    # bfloat16 on GPU: float32 weights are 32 GB and OOM a 24 GB card. Activations
    # are cast to float32 on the way out, so the geometry is computed in fp32.
    dtype = "bfloat16" if device == "cuda" else "float32"
    model = load_model("meta-llama/Llama-3.1-8B", device=device,
                       fold_ln=True, center_writing_weights=True,
                       center_unembed=True, dtype=dtype)
    toks = [model.to_tokens(t, prepend_bos=True) for t in texts]
    # The weekday token must sit at ONE position for the site to be shared. Locate it
    # by id and assert, never assume -- multi-token time phrases shift everything after.
    day_id = [model.to_tokens(" " + DAYS[r[1]], prepend_bos=False)[0, 0].item()
              for r in rows]
    pos = [int((toks[i][0] == day_id[i]).nonzero()[0, 0]) for i in range(len(rows))]
    lens = {int(t.shape[1]) for t in toks}
    if len(set(pos)) != 1:
        # Expected on a first draft: report the split rather than silently averaging
        # over two different sites.
        bad = {}
        for i, p in enumerate(pos):
            bad.setdefault(p, []).append(rows[i][3])
        raise SystemExit("weekday token is not at one position; groups = "
                         + "; ".join(f"pos {p}: {sorted(set(v))[:6]}..."
                                     for p, v in sorted(bad.items())))
    print(f"[cap] {len(rows)} prompts, weekday token at position {pos[0]}, "
          f"token lengths {sorted(lens)}, layer {layer}", flush=True)
    P = pos[0]
    out = []
    with torch.no_grad():
        for c0 in range(0, len(toks), chunk):
            batch = torch.cat(toks[c0:c0 + chunk], 0)
            store = {}
            model.run_with_hooks(
                batch, return_type=None,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post",
                            lambda a, hook: store.setdefault(
                                "x", a.detach().float().cpu()))])
            out.append(store["x"][:, P, :].numpy())
    A = np.concatenate(out, 0).astype(np.float64)
    return A, rows, P, sorted(lens)[0]


def plane_of(C, k=2):
    _, _, Vt = np.linalg.svd(C - C.mean(0), full_matrices=False)
    return Vt[:k]                                       # [k, d]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--npz", default="experiments/results/timeofday_L28.npz")
    ap.add_argument("--out", default="figures/timeofday_draft.pdf")
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--elev", type=float, default=24.0)
    ap.add_argument("--azim", type=float, default=-64.0)
    args = ap.parse_args()

    want = sorted({r[3] for r in build_prompts()})
    cached = None
    if os.path.exists(args.npz):
        # Guard against a stale cache: if the modifier set has changed since this npz
        # was written, re-capture rather than silently plotting the old stimuli.
        #
        # The read is guarded because "the file exists" does not mean "the file is
        # whole". An interrupted np.savez -- the disk filling mid-write is the way this
        # actually happens -- leaves a TRUNCATED zip behind, and the reuse check then
        # died on `KeyError: 'lab is not a file in the archive'` instead of noticing it
        # had nothing usable. A cache that cannot be read is a cache miss, not an error:
        # anything we cannot parse we simply recapture over.
        try:
            z = np.load(args.npz, allow_pickle=True)
            labs = sorted(set(z["lab"].astype(str)))
        except Exception as e:                                # truncated / corrupt / stale
            print(f"[cap] {args.npz} is unreadable ({type(e).__name__}: {e}) -- re-capturing")
        else:
            if labs == want:
                cached = z
            else:
                print(f"[cap] {args.npz} has a different modifier set -- re-capturing")
    if cached is not None:
        z = cached
        A = z["A"].astype(np.float64)
        day, fam, lab, hour = z["day"], z["fam"].astype(str), z["lab"].astype(str), z["hour"]
        print(f"[cap] reusing {args.npz}  ({len(A)} prompts)")
    else:
        A, rows, P, ntok = capture(args.layer, args.device, args.chunk)
        day = np.array([r[1] for r in rows])
        fam = np.array([r[2] for r in rows])
        lab = np.array([r[3] for r in rows])
        hour = np.array([r[4] for r in rows], dtype=float)
        os.makedirs(os.path.dirname(args.npz) or ".", exist_ok=True)
        np.savez(args.npz, A=A.astype(np.float32), day=day, fam=fam, lab=lab,
                 hour=hour, layer=args.layer, pos=P, n_tokens=ntok,
                 texts=np.array([r[0] for r in rows]))
        print(f"[cap] wrote {args.npz}")

    # ---- geometry: weekday ring from the day centroids, marginalising modifier ----
    mu = A.mean(0)
    Cday = np.stack([A[day == d].mean(0) for d in range(7)])
    plane = plane_of(Cday)                              # [2, d]
    xy = (A - mu) @ plane.T
    xyC = (Cday - mu) @ plane.T
    thC = np.degrees(np.arctan2(xyC[:, 1], xyC[:, 0]))
    th = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))

    def wrap(a):
        return (a + 180.0) % 360.0 - 180.0

    dth = wrap(th - thC[day])                           # signed in-plane shift, degrees
    # the angular step to the adjacent day centroid, from the data
    # The seven adjacent-centroid gaps. Their MEAN is forced to 360/7 = 51.43 for any
    # seven points on a closed loop, so it says nothing about this model; the spread is
    # the only informative part. Report the whole range and band it.
    order = np.argsort(wrap(thC - thC[0]) % 360.0)
    steps = np.abs(wrap(np.diff(np.append(thC[order], thC[order][0]))))
    STEP_MIN, STEP_MAX = float(steps.min()), float(steps.max())

    FAMS = ["clock", "word", "placebo_very", "placebo_varied"]
    labels = []
    for f in FAMS:
        labels += list(dict.fromkeys(lab[fam == f]))
    stat = {}
    for m in labels:
        v = dth[lab == m]
        per_day = np.array([dth[(lab == m) & (day == d)].mean() for d in range(7)])
        stat[m] = (float(per_day.mean()), float(per_day.std(ddof=1) / np.sqrt(7)),
                   str(fam[lab == m][0]), float(hour[lab == m][0]))

    fig = plt.figure(figsize=(15.5, 9.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.25], wspace=0.16,
                  left=0.088, right=0.985, top=0.895, bottom=0.075)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], projection="3d")

    # ================= A: in-plane angular shift, one row per modifier ==========
    # One colour rule for both panels: the clock rows take the same cyclic map panel B
    # uses for its dots, so a row here and a dot there at the same hour are the same
    # colour. The coarse word modifiers keep the flat am/pm split -- they have no exact
    # hour to look up -- and the placebos stay grey.
    cyc = plt.get_cmap("twilight")
    nrm = Normalize(vmin=0, vmax=24)

    def mod_colour(m):
        _, _, f_, hr_ = stat[m]
        if f_.startswith("placebo"):
            return PALE
        if f_ == "clock":
            return cyc(nrm(hr_))
        return EARLY if hr_ < 13 else LATE

    tidy(axA)
    pos_y = np.arange(len(labels))[::-1]
    # The adjacent-day band: nearest gap to furthest, so the ring's irregularity is
    # visible rather than collapsed into one number.
    # Which side is which is measured, not assumed: the seven centroids run Mon->Sun in
    # increasing angle and all seven steps are positive, so +shift heads to the next
    # weekday. Both band edges share one style -- they are the same kind of boundary,
    # the nearest and furthest of the seven adjacent-day steps.
    for s, side in ((-1, "previous day centroid"), (+1, "next day centroid")):
        lo, hi = sorted((s * STEP_MIN, s * STEP_MAX))
        axA.axvspan(lo, hi, color=INK, alpha=0.09, lw=0, zorder=1)
        for edge in (STEP_MIN, STEP_MAX):
            axA.axvline(s * edge, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=2)
        axA.text(s * (STEP_MIN + STEP_MAX) / 2.0, (len(labels) - 1) / 2.0, side,
                 rotation=90, ha="center", va="center", fontsize=7.2, color=INK,
                 style="italic", zorder=5)
    # The band edges are the nearest (STEP_MIN) and furthest (STEP_MAX) of the seven
    # adjacent-day steps; both numbers are printed in the [A] block at the end.
    for p, m in zip(pos_y, labels):
        mu_, se_, f_, hr_ = stat[m]
        c = mod_colour(m)
        mk = "s" if f_ == "placebo_varied" else "o"
        # twilight is near-white at both midnight ends, so a white marker edge would
        # erase the 11pm/1am rows. Outline those in ink instead of white.
        edge = MID if f_ == "clock" else "white"
        axA.errorbar(mu_, p, xerr=se_, fmt=mk, ms=4.0, color=c, ecolor=c,
                     elinewidth=1.1, capsize=2.0, zorder=4,
                     markeredgecolor=edge, markeredgewidth=0.5)
    axA.axvline(0, color=PALE, lw=1.0, zorder=3)
    axA.set_yticks(pos_y)
    # Every row is labelled with the literal string that filled the {t} slot, nothing
    # else. `lab` keys the clock rows as 01:00..23:00 to sort them, but that form never
    # reached the model, so it does not appear on the axis.
    ticklabels = [clock_word(int(m[:2])) if stat[m][2] == "clock" else m
                  for m in labels]
    axA.set_yticklabels(ticklabels, fontsize=6.4)
    for t, m in zip(axA.get_yticklabels(), labels):
        t.set_color(mod_colour(m))
    axA.set_xlim(-STEP_MAX * 1.12, STEP_MAX * 1.12)
    axA.set_ylim(-1.2, len(labels) + 2.2)
    axA.set_xlabel("in-plane angular shift from the day's own centroid (degrees)",
                   fontsize=8.6, color=INK)
    axA.text(0.015, -0.055, "clock rows: coloured by time of day, the same map panel B "
             "uses  ·  word modifiers: blue am, orange pm  ·  grey ○: “very X” placebo  "
             "·  grey □: placebo with both tokens varied",
             transform=axA.transAxes, fontsize=6.6, color=PALE, ha="left",
             style="italic")

    # ================= B: weekday plane + the hour direction ===================
    Q = plane.T                                          # [d, 2], orthonormal rows
    cl = fam == "clock"
    # axis 3 = PC1 of the HOUR plane, orthogonalised against the weekday plane. The hour
    # plane is PCA on the 22 hour CENTROIDS (each averaged over the seven days), so the
    # axis describes the hour trajectory the days share, not any one day's excursion.
    #
    # NOT Diksha's basis_for(), which takes the direction in this plane that most departs
    # from the weekday plane. That choice is near-degenerate here: the two planes are
    # almost orthogonal (principal angles 87.2 deg, 88.1 deg), so every direction in the
    # hour plane departs by nearly the same amount and the pick came down to a 0.1 %
    # difference in singular value -- an essentially arbitrary rotation within the plane,
    # landing 41.9 deg off PC1. The hour PCA itself is well conditioned (PC1/PC2 = 1.43),
    # so PC1 is a stable choice and shows more of the clock structure (sd 6.87 vs 6.04).
    #
    # CAVEAT, true either way: the hour plane is 2-D and only one of its directions is
    # drawn. The clock direction orthogonal to this one carries sd 5.81 of the hour
    # centroids -- nearly as much -- and is not visible in this figure.
    hrs = np.array(sorted(set(hour[cl].tolist())))
    Chour = np.stack([A[cl & (hour == h)].mean(0) for h in hrs])
    _, _, Vh = np.linalg.svd(Chour - Chour.mean(0), full_matrices=False)
    e3 = Vh[0] - Q @ (Q.T @ Vh[0])                       # hour PC1, weekday plane removed
    e3 /= np.linalg.norm(e3)
    # A PC's sign is arbitrary, so fix it rather than inherit it: point e3 so that early
    # hours sit HIGH, matching the colourbar, which reads 1am at the top down to 11pm.
    if np.corrcoef(hrs, (Chour - Chour.mean(0)) @ e3)[0, 1] > 0:
        e3 = -e3
    B3 = np.column_stack([Q[:, 0], Q[:, 1], e3])
    X = (A - mu) @ B3
    # The ring is built from the CLOCK prompts only -- the same prompts drawn as dots.
    # The all-modifier centroid `Cday` (still panel A's reference) would put the ring at
    # a height set by the clock/placebo mix in the modifier bank rather than by the
    # model, leaving the dots hanging below a ring they are not the average of.
    Cday_cl = np.stack([A[cl & (day == d)].mean(0) for d in range(7)])
    KD = (Cday_cl - mu) @ B3

    # weekday ring: closed, through the day centroids in weekday order
    ring_t = np.arange(8, dtype=float)
    ring = CubicSpline(ring_t, np.vstack([KD, KD[:1]]), bc_type="periodic", axis=0)(
        np.linspace(0, 7, 400))
    axB.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=INK, lw=0.9, zorder=2)
    for d in range(7):
        rows_d = np.array([np.where(cl & (day == d) & (hour == h))[0][0] for h in hrs])
        Pd = X[rows_d]
        S = CubicSpline(hrs, Pd, bc_type="natural", axis=0)(
            np.linspace(hrs[0], hrs[-1], 240))
        axB.plot(S[:, 0], S[:, 1], S[:, 2], color=DAY_C[d], lw=0.9, alpha=0.45, zorder=3)
        axB.scatter(Pd[:, 0], Pd[:, 1], Pd[:, 2], s=13,
                    c=[cyc(nrm(h)) for h in hrs], depthshade=False,
                    edgecolors="white", linewidths=0.3, zorder=4)
        axB.scatter([KD[d, 0]], [KD[d, 1]], [KD[d, 2]], s=45, color=DAY_C[d],
                    depthshade=False, edgecolors=INK, linewidths=0.6, zorder=6)
        axB.text(KD[d, 0] * 1.28, KD[d, 1] * 1.28, KD[d, 2], DAYS[d][:3],
                 fontsize=8.5, color=INK, ha="center", va="center",
                 fontweight="bold", zorder=7)
    # Frame what is actually drawn -- the clock dots and their ring. Using all of X put
    # the box round the unplotted placebos and pushed the clock cloud into a corner.
    drawn = np.vstack([X[cl], KD])
    ctr = (drawn.max(0) + drawn.min(0)) / 2.0
    s = (drawn.max(0) - drawn.min(0)).max() / 2.0 * 1.06      # cube: keep it isotropic
    for setter, c in ((axB.set_xlim, ctr[0]), (axB.set_ylim, ctr[1]),
                      (axB.set_zlim, ctr[2])):
        setter(c - s, c + s)
    axB.set_box_aspect((1, 1, 1))
    axB.set_xticks([]); axB.set_yticks([]); axB.set_zticks([])
    for pane in (axB.xaxis, axB.yaxis, axB.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor(PALE)
        pane.line.set_color(PALE)
    # Matplotlib draws only the three back panes, so the cube is left with a vertical
    # edge missing at the near-left corner. Draw all four verticals; the three that
    # already exist are overdrawn in the same colour and weight.
    xl, yl, zl = axB.get_xlim(), axB.get_ylim(), axB.get_zlim()
    for xc in xl:
        for yc in yl:
            axB.plot([xc, xc], [yc, yc], zl, color=PALE, lw=0.8, zorder=0)
    axB.set_xlabel("weekday plane 1", fontsize=8.0, color=MID, labelpad=-13)
    axB.set_ylabel("weekday plane 2", fontsize=8.0, color=MID, labelpad=-13)
    axB.set_zlabel("clock time plane 1", fontsize=8.0, color=MID, labelpad=-14)
    axB.view_init(elev=args.elev, azim=args.azim)
    # The bar spans only the hours actually sampled, 1am..11pm. The colour mapping is
    # still the full 0..24 cycle -- so a dot's colour is unchanged -- but the bar is
    # clipped to the sampled range, because ending it at "12am" claimed a midnight
    # reading that does not exist: 0 and 12 are dropped from CLOCK_HOURS as ambiguous.
    cb = fig.colorbar(ScalarMappable(norm=nrm, cmap=cyc), ax=axB, fraction=0.022,
                     pad=0.02, shrink=0.52,
                     boundaries=np.linspace(hrs[0], hrs[-1], 256))
    cb.set_ticks([1, 6, 12, 18, 23])
    cb.set_ticklabels(["1am", "6am", "12pm", "6pm", "11pm"])
    cb.ax.invert_yaxis()            # morning at the top, evening at the bottom
    cb.ax.tick_params(labelsize=8.0, length=2.5, width=0.6, colors=MID)
    cb.set_label("clock time", fontsize=8.6, color=INK, labelpad=3)

    fig.suptitle("Time of day moves a weekday activation off the ring, not along it",
                 fontsize=13.0, color=INK, y=0.965)
    fig.text(0.5, 0.925, f"Llama-3.1-8B  ·  “It was {{t}} on {{day}}.” read at the weekday "
             f"token  ·  layer {int(args.layer)}  ·  {len(A)} prompts, no averaging",
             fontsize=9.0, color=MID, ha="center")
    fig.text(0.088, 0.014, "FIRST DRAFT — unrefined on purpose", fontsize=6.6,
             color=PALE, ha="left", style="italic")

    fig.canvas.draw()
    for letter, title, cell in (("A", "in-plane shift, per modifier", gs[0, 0]),
                                ("B", "the weekday plane and the hour direction", gs[0, 1])):
        c = cell.get_position(fig)
        fig.text(c.x0 - 0.055, c.y1 + 0.020, letter, fontsize=10.5, fontweight="bold",
                 color=INK, ha="left", va="bottom")
        fig.text(c.x0 - 0.034, c.y1 + 0.022, title, fontsize=8.3, color=INK,
                 ha="left", va="bottom")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"[saved] {args.out}")
    if args.also_png:
        p = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"[saved] {p}")

    # ---- the numbers, so the draft can be judged rather than admired -----------
    def rng(ms):
        v = np.array([stat[m][0] for m in ms])
        return f"{v.min():+.2f}° to {v.max():+.2f}° (|max| {np.abs(v).max():.2f}°)"
    print(f"\n[A] adjacent-day gaps: nearest {STEP_MIN:.2f}°, furthest {STEP_MAX:.2f}° "
          f"(mean is 360/7 = {360/7:.2f}° by construction)")
    for f in FAMS:
        ms = [m for m in labels if stat[m][2] == f]
        print(f"[A] {f:15s} (n={len(ms):2d})  {rng(ms)}")
    # in-plane vs off-plane displacement, raw and in ring radii (panel C's quantity)
    Rr = float(np.linalg.norm(Cday - Cday.mean(0), axis=1).mean())
    for f in FAMS:
        m = fam == f
        D = A[m] - Cday[day[m]]
        par_v = np.linalg.norm(D @ plane.T, axis=1)
        per_v = np.sqrt(np.maximum((D ** 2).sum(1) - par_v ** 2, 0.0))
        par, per = par_v.mean(), per_v.mean()
        print(f"[C] {f:15s} in-plane {par:7.3f} ({par/Rr:.3f} R)   "
              f"perp {per:7.3f} ({per/Rr:.3f} R)")
    print(f"[C] ring radius R = {Rr:.3f}")

    # ---- [D] energy inside vs outside the FULL 6-D weekday subspace ----------------
    # The seven day centroids span 6 dimensions once centred, so the weekday code is 6-D,
    # not the 2-D ring plane panel B draws. The question "does a modifier move the
    # activation within the weekday code or out of it" is properly asked against all 6.
    #
    # The chance baseline is MEASURED, not assumed, following capture_corpus_windows.py:
    # a nominal k/d share (6/4096 = 0.15%) presumes isotropy, and Llama's residual
    # stream is strongly anisotropic. So the same displacement is also projected into
    # N_NULL random 6-D subspaces and enrichment is the ratio, in which the anisotropy
    # cancels because numerator and denominator see the same vector.
    K_WD, N_NULL = 6, 200
    Wsub = plane_of(Cday, k=K_WD).T                       # [d, 6]
    rng = np.random.default_rng(0)
    nulls = [np.linalg.qr(rng.normal(size=(A.shape[1], K_WD)))[0] for _ in range(N_NULL)]
    print(f"\n[D] displacement from the day's own centroid, split by the {K_WD}-D weekday "
          f"subspace (null = {N_NULL} random {K_WD}-D subspaces)")
    for f in FAMS + ["__time__", "__placebo__"]:
        if f == "__time__":
            m, name = np.isin(fam, ["clock", "word"]), "ALL time-of-day"
        elif f == "__placebo__":
            m, name = np.isin(fam, ["placebo_very", "placebo_varied"]), "ALL placebo"
        else:
            m, name = fam == f, f
        D = A[m] - Cday[day[m]]
        tot = (D ** 2).sum(1)
        inside = ((D @ Wsub) ** 2).sum(1)
        null = np.stack([((D @ B) ** 2).sum(1) for B in nulls])
        print(f"[D] {name:16s} n={int(m.sum()):3d}  inside {100*inside.sum()/tot.sum():5.2f}%"
              f"  outside {100*(1-inside.sum()/tot.sum()):5.2f}%"
              f"  random {100*null.mean()/tot.mean():5.3f}%"
              f"  enrichment {np.mean(inside/null.mean(0)):5.1f}x")

    # ---- the clock trajectory's steps, and where the unplotted midnight falls -------
    # With noon restored every consecutive pair is one hour apart, so the wrap can be
    # compared against a like-for-like baseline instead of a two-hour gap.
    Chour_all = np.stack([A[cl & (hour == h)].mean(0) for h in hrs])
    steps_h = np.linalg.norm(np.diff(Chour_all, axis=0), axis=1)
    med = float(np.median(steps_h))
    wrap_h = float(np.linalg.norm(Chour_all[-1] - Chour_all[0]))
    print(f"\n[H] one-hour steps 1am..11pm: median {med:.3f}, "
          f"range {steps_h.min():.3f}-{steps_h.max():.3f}  (n={len(steps_h)})")
    print(f"[H] wrap 11pm -> 1am (2 h, midnight omitted): {wrap_h:.3f} = "
          f"{wrap_h / med:.2f}x median  -- vs 2.00x if the clock closed uniformly")

    # Midnight: captured, never plotted. Report where it sits so the exclusion can be
    # judged rather than taken on trust -- if the model resolved "twelve am" cleanly it
    # would land near one of the two ends, not between them.
    mid = fam == "clock_midnight"
    if mid.any():
        Cmid = A[mid].mean(0)
        d_to = {int(h): float(np.linalg.norm(Cmid - Chour_all[i]))
                for i, h in enumerate(hrs)}
        near = sorted(d_to, key=d_to.get)[:3]
        print(f"\n[M] midnight (“{clock_word(MIDNIGHT_HOUR)}”, n={int(mid.sum())}) "
              f"captured, NOT plotted")
        print(f"[M]   distance to 11pm {d_to[23]:.3f} ({d_to[23]/med:.2f}x median step), "
              f"to 1am {d_to[1]:.3f} ({d_to[1]/med:.2f}x)")
        print(f"[M]   nearest hours: " + ", ".join(f"{h:02d}:00 ({d_to[h]:.2f})"
                                                   for h in near))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
