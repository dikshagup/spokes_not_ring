"""Figure 1: the cascade panels and the Jacobian discs on one plate.

Two independent routes to one claim -- a finite-difference walk and an autodiff Jacobian --
so the script asserts that both halves come from the same model, prompt family, readout
position and steer layer rather than trusting the file names.

Writes figures/fig_combined_llama.{pdf,png}. See repro_fig1_combined_llama.sh.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from cascade_row import draw_row
from polar_disc import disc, edges

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
MEASURES = [("fro", "Jacobian norm", r"$\|J\|_F$", "all", "estimated (Hutchinson)"),
            ("gain_t", "Jacobian along ring tangent", r"$\|J\hat{t}\|$", "tangent", ""),
            ("gain_r", "Jacobian along ring radius", r"$\|J\hat{r}\|$", "radial", ""),
            ("gain_off", "Jacobian along a random direction",
             r"$\|J\hat{v}_{\perp}\|$", "off", "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade", default="figures/ladder_llama_best.npz")
    ap.add_argument("--field", default="figures/llama_polar70_stop_fp16.npz")
    ap.add_argument("--out", default="figures/fig_combined_llama.pdf")
    ap.add_argument("--fix-steer", type=int, default=2)
    ap.add_argument("--compare-steer", type=int, default=28)
    ap.add_argument("--readout-depth", type=int, default=None)
    ap.add_argument("--scale-band", default="0.75,1.3")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--disc-fontscale", type=float, default=1.30,
                    help="type scale for the disc labels and colourbars, chosen so the "
                         "bottom row's type matches the top row's rather than being set "
                         "independently. The discs are drawn small to buy the space; they "
                         "read fine well below the area they were taking.")
    ap.add_argument("--layout", default="tight", choices=("tight", "full"),
                    help="tight: three panels in one row -- downstream behaviour, readout "
                         "speed, and the Jacobian norm. full: all seven, cascade over discs. "
                         "The tight cut keeps one panel from each of the three things this "
                         "measures (output, residual, derivative) and drops the ones that "
                         "restate them: the cumulative panel is the speed panel integrated, "
                         "and the three remaining discs are the norm resolved by direction.")
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--background", default="transparent",
                    choices=("transparent", "white"))
    args = ap.parse_args()

    zc = np.load(args.cascade, allow_pickle=True)
    zf = np.load(args.field, allow_pickle=True)
    meta = json.loads(str(zf["meta"]))
    # The plate's whole claim is that the two halves measure the same thing, so the things
    # that would make that false are checked rather than assumed from the file names.
    assert str(zc["model"]) == meta["model"], (
        f"different models: cascade {zc['model']}, field {meta['model']}")
    assert str(zc["formulation"]) == meta["formulation"], (
        f"different prompt families: {zc['formulation']} vs {meta['formulation']}")
    assert int(zc["ro_pos"]) == meta["ro_pos"], "different readout positions"
    assert meta["patch_layer"] == int(args.fix_steer), (
        f"the field steers at L{meta['patch_layer']}, the cascade row at L{args.fix_steer}")

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    tight = args.layout == "tight"
    if tight:
        fig = plt.figure(figsize=(9.8, 3.1))
        gs_top = GridSpec(1, 3, figure=fig, left=0.098, right=0.940, top=0.820,
                          bottom=0.215, wspace=0.40,
                          width_ratios=[1.0, 1.0, 0.74])
        gs_bot = gs_top
    else:
        fig = plt.figure(figsize=(12.2, 7.6))
        gs_top = GridSpec(1, 3, figure=fig, left=0.070, right=0.992, top=0.940,
                          bottom=0.630, wspace=0.36)
        gs_bot = GridSpec(1, 4, figure=fig, left=0.008, right=0.972, top=0.415,
                          bottom=0.045, wspace=0.02)

    # ---- A B C: the cascade -----------------------------------------------------------
    if tight:
        aA = fig.add_subplot(gs_top[0, 0])
        aC = fig.add_subplot(gs_top[0, 1])
        aB = fig.add_subplot(gs_top[0, 2])       # drawn, then discarded below
    else:
        aA, aC, aB = (fig.add_subplot(gs_top[0, i]) for i in range(3))
    nfo = draw_row(fig, aA, aC, aB, args.cascade, "readout", args.fix_steer,
                   args.compare_steer, args.readout_depth, 0)
    # the cascade row is drawn at its standalone sizes; on this plate it sits beside discs
    # whose labels have been scaled up, so it is brought up to match rather than left small
    for ax in (aA, aC, aB):
        ax.xaxis.label.set_fontsize(10.8)
        ax.yaxis.label.set_fontsize(10.8)
        ax.tick_params(labelsize=10.0)
        for t in ax.texts:
            fsz = t.get_fontsize()
            if abs(fsz - 9.5) < 0.01:          # the panel title
                t.set_fontsize(10.0)
            elif fsz < 9.0:                     # day names, in-panel notes
                t.set_fontsize(fsz * 1.30)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                t.set_fontsize(9.2)
    if tight:
        aA.set_ylabel("Behaviour distance\n(fraction)", fontsize=10.0, color=INK,
                      linespacing=1.4)
        aC.set_ylabel("Readout speed\n\u2016\u0394resid\u2016 / \u2016\u0394steer\u2016",
                      fontsize=10.0, color=INK, linespacing=1.4)
        for ax in (aA, aC):
            ax.set_xlabel("Steered position on the ring\n(arc length, fraction of loop)",
                          fontsize=10.0, color=INK, linespacing=1.4)

    from cascade_row import LETTER_X, TITLE_Y
    keep_axes = (aA, aC) if tight else (aA, aC, aB)
    if tight:
        # the cumulative panel is the speed panel integrated; the tight cut shows the rate
        fig.delaxes(aB)
    for j, ax in enumerate(keep_axes):
        ax.text(LETTER_X, TITLE_Y, "ABCDEFG"[j], transform=ax.transAxes, fontsize=12.0,
                fontweight="bold", color=INK, ha="left", va="baseline")
    ap_ = aA.get_position()
    title_fig_y = ap_.y0 + TITLE_Y * ap_.height        # the row's shared baseline

    # ---- D E F G: the Jacobian field at r = 1 ------------------------------------------
    us, rs = np.asarray(zf["us"]), np.asarray(zf["rs"])
    th_e = edges(np.asarray(us) * 2 * np.pi)
    r_e = edges(rs, lo=0.0)
    rmid = rs
    band = [float(v) for v in args.scale_band.split(",")]
    keep = (rmid >= band[0]) & (rmid <= band[1])
    if not keep.any():
        keep = np.ones_like(rmid, dtype=bool)
    shown = [m for m in MEASURES if m[0] in zf.files]
    if tight:
        shown = [m for m in shown if m[0] == "fro"]     # the norm; E-G resolve it by direction
    for i, (key, title, sym, dirn, lab) in enumerate(shown):
        a = zf[key].mean(0)
        bb = gs_bot[0, 2 if tight else i].get_position(fig)
        k = 0.98 if tight else 0.66
        w, h = bb.width * k, bb.height * 0.84 * k
        ax = fig.add_axes([bb.x0 + (bb.width - w) / 2,
                           bb.y0 + (bb.height * 0.84 - h), w, h], projection="polar")
        v = a.T[keep]
        disc(ax, th_e, r_e, a.T, args.cmap,
             float(np.percentile(v, 1.0)), float(np.percentile(v, 99.0)), lab, dirn=dirn,
             fs=args.disc_fontscale, ring_label=True)
        # Letter and title on one baseline, as the cascade row above does it; the symbol on
        # its OWN line beneath, left-aligned to the title. It used to be centred over the
        # disc, which put it under the title's tail rather than clear of it -- and a symbol
        # centred on the disc while the title starts at the rail have no fixed relationship,
        # so whether they collided depended on how long the title happened to be.
        letter = "C" if tight else "DEFG"[i]
        ty = title_fig_y if tight else bb.y1 + 0.036
        fig.text(bb.x0 + 0.002, ty, letter, fontsize=12.0,
                 fontweight="bold", color=INK, ha="left", va="baseline")
        if tight:
            # one panel, so the symbol can share the title's line; it is the last thing on
            # it and nothing follows to collide with
            fig.text(bb.x0 + 0.030, ty, title, fontsize=10.4, color=INK,
                     ha="left", va="baseline")
            fig.text(bb.x0 + 0.030, ty, "                     " + sym,
                     fontsize=11.5, color=INK, ha="left", va="baseline")
        else:
            fig.text(bb.x0 + 0.028, bb.y1 + 0.036, title, fontsize=10.4, color=INK,
                     ha="left", va="baseline")
            ab = ax.get_position()
            fig.text(ab.x0 + ab.width / 2, bb.y1 + 0.006, sym, fontsize=12.0, color=INK,
                     ha="center", va="baseline")

    tp = args.background == "transparent"
    fig.savefig(args.out, dpi=300, transparent=tp)
    print("[figure] wrote", args.out)
    if args.also_png:
        fig.savefig(args.out.replace(".pdf", ".png"), dpi=300, transparent=tp)
        print("[figure] wrote", args.out.replace(".pdf", ".png"))


if __name__ == "__main__":
    main()
