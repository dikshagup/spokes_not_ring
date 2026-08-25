"""Figure 4: the methods schematic -- the (u, r) chart and the intervention.

Reads the same polar-field npz figure 1's panel C is drawn from; nothing in the plate is
data, only the geometry of the loop and the token layout of the prompt.

Run:
  PYTHONPATH=src:experiments python experiments/figure_method_steer.py \
      --npz figures/llama_polar70_stop_fp16.npz --out figures/fig_method_steer.pdf --also-png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from weekday_manifold.manifold.days import DAYS  # noqa: E402
from method_schematic import panel_headers, panel_pipeline, panel_traces  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default="figures/llama_polar70_stop_fp16.npz",
                    help="polar field; read for the centroids, mu and the two positions")
    ap.add_argument("--context", default="figures/readout_context.json",
                    help="token strip, matched to this run by its two positions")
    ap.add_argument("--out", default="figures/fig_method_steer.pdf")
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--background", default="transparent",
                    choices=("transparent", "white"))
    args = ap.parse_args()

    from weekday_manifold.manifold.steering import fit_steer_spline

    z = np.load(args.npz, allow_pickle=True)
    C, mu = z["centroids"], z["mu"]
    meta = json.loads(str(z["meta"]))
    sp = fit_steer_spline(C)
    _, _, Vt = np.linalg.svd(C - mu, full_matrices=False)

    # The token strip comes from readout_context.json, which the model wrote: a hand-written
    # list drifts and then mislabels the patch site. Matched to this run by its two
    # positions, so the plate cannot show one prompt's tokens over another's measurement.
    day_pos, ro_pos = meta["day_pos"], meta["ro_pos"]
    toks = None
    if os.path.exists(args.context):
        for _, c in json.load(open(args.context)).items():
            if c["patch_pos"] == day_pos and c["read_pos"] == ro_pos:
                toks = ["<bos>"] + [t.replace("\n", "\\n") for t in c["tokens"][1:]]
                break
    assert toks is not None, (
        f"no entry in {args.context} for patch_pos={day_pos}, read_pos={ro_pos}; "
        "rebuild it with experiments/readout_context.py")
    assert toks[day_pos].strip() in DAYS, (
        "patch site is %r, not a weekday -- token list is out of step" % toks[day_pos])
    assert 0 <= ro_pos < len(toks), "readout position is outside the token strip"
    # Attention is causal, so nothing after the readout can reach it.
    toks = toks[:ro_pos + 1]

    measures = "$\\|J\\|_F$ · $\\|J\\hat t\\|$ · $\\|J\\hat r\\|$"
    if "gain_off" in z.files:
        measures += " · $\\|J\\hat v_{\\perp}\\|$"

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    n_disc = 4 if "gain_off" in z.files else 3
    fig = plt.figure(figsize=(2.10 * n_disc, 2.85))
    gs = GridSpec(1, n_disc, figure=fig, hspace=0.20, wspace=0.04,
                  left=0.006, right=0.994, top=0.865, bottom=0.04)
    ba = gs[0, 0].get_position(fig)
    ax_a = fig.add_axes([ba.x0 + ba.width * 0.16, ba.y0, ba.width, ba.height])
    bb = gs[0, 1:].get_position(fig)
    w = bb.width * 0.80
    ax_b = fig.add_axes([bb.x0 + (bb.width - w) / 2, bb.y0, w, bb.height])

    panel_traces(ax_a, sp, mu, Vt)
    panel_pipeline(ax_b, day_pos, ro_pos, toks,
                   meta["patch_layer"], meta["readout_layer"], measures)
    panel_headers(fig, [("A", "Measuring sensitivity to perturbations along and across "
                             "the ring", gs[0, :].get_position(fig))], dy=0.014)

    tp = args.background == "transparent"
    fig.savefig(args.out, bbox_inches="tight", dpi=300, transparent=tp)
    print(f"[figure] wrote {args.out}", flush=True)
    if args.also_png:
        png = args.out.rsplit(".", 1)[0] + ".png"
        fig.savefig(png, bbox_inches="tight", dpi=300, transparent=tp)
        print(f"[figure] wrote {png}", flush=True)


if __name__ == "__main__":
    main()
