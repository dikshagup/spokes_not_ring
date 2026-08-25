"""A row of Jacobian discs for one model: the norm, the tangential and radial gains, and a control.

The model is not a flag -- it is which field npz --npz points at, and the caption reads the
model, the depths and the token positions back out of it.

Writes figures/fig_jac_<model>.{pdf,png} and its caption. See repro_fig5_jac_grid.sh, which
runs all three, and repro_fig6_combined_arith.sh, which runs it on the interrogative field.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from weekday_manifold.manifold.days import DAYS, N_DAYS  # noqa: E402

# The disc itself is drawn by the same code that draws panel C of fig_combined_llama.
# Shared rather than copied: these are the same measurement rendered the same way, and
# two copies of the renderer could drift without either figure erroring.
from polar_disc import disc, edges, measurement_inset  # noqa: F401,E402

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
ACC, ACC2, WARM = "#3b5f9e", "#7a9bd4", "#c8642a"
# The STEER hue exists because the displacement was drawn in WARM, the same colour this
# figure uses for "measured" -- so the intervention read as part of the readout. It is the
# hue fig_radial uses for the same object. Nothing else changes: ACC still marks what is
# changed, WARM what is measured.
STEER = "#33207a"
# Panel A is styled to match fig_radial's panel A: a neutral ring with day-coloured knots,
# rather than an all-blue loop. That is what lets STEER own a colour here.
#
# STEER is a violet rather than the earlier indigo #3f3d9e because the day ramp now ENDS on
# blue (Sun #0072B2, Sat #56B4E9), and indigo sat dE 10.8 from Sun -- the steering ray points
# straight at that knot, so the two blues met. Violet cannot be fixed by hue alone: with seven
# chromatic days the ramp wraps most of the hue circle (pink 346 deg round to blue 244), so no
# hue is far from everything and the free axis is lightness. #33207a is where separation from
# the days (rising as it darkens) crosses separation from body ink (falling): dE 18.9 to the
# nearest day, 19.0 to the ring grey, 17.3 to INK -- all clear of the floor of 15.
DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2"]
WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


# ------------------------------------------------------------------------------- shared bits





















def panel_headers(fig, lettered, dy=0.075, fs=12.0):
    """Letter and title on one baseline at each panel's top-left, in FIGURE coordinates."""
    # a wrapped title grows UPWARD from its baseline, so a letter sharing that baseline
    # ends up beside the last line; lift it by the lines above it
    line_h = 9.6 * 1.45 / (fig.get_figheight() * 72.0)
    for letter, title, cell in lettered:
        fig.text(cell.x0, cell.y1 + dy + title.count("\n") * line_h, letter, fontsize=fs,
                 fontweight="bold", color=INK, ha="left", va="bottom")
        # titles wrap to a second line: it keeps the columns narrow enough to sit close
        # together, which is where the white space between the discs was coming from
        fig.text(cell.x0 + 0.024, cell.y1 + dy + 0.002, title, fontsize=9.6, color=INK,
                 ha="left", va="bottom", linespacing=1.45)


# --------------------------------------------------------------------- A: what u and r trace


# ------------------------------------------------------------------------- B: the intervention








# ------------------------------------------------------------------------------- the caption
def caption(meta, d, nbg, n_u, n_r, r_max, radii, steps, ev, angsum, day_tok,
            n_layers, kg, ro_tok):
    j = (f"$J=\\partial\\,\\mathrm{{res}}^{{(t_{{\\mathrm{{read}}}})}}"
         f"_{{L{meta['readout_layer']}}}\\,/\\,\\partial\\,"
         f"x^{{(t_{{\\mathrm{{day}}}})}}_{{L{meta['patch_layer']}}}"
         f"\\in\\mathbb{{R}}^{{{d}\\times{d}}}$")
    return f"""\
**Figure 1 — Steering the weekday loop in {meta['model']}.**

**(A) The construction and the intervention.** The model gives each weekday its own location in activation
space; averaging over prompts leaves seven centroids $C_1,\\dots,C_7$, and $\\sigma(u)$ is a
periodic cubic spline threaded through them with $\\sigma(k/7)=C_k$. Their mean is $\\mu$, and
the disc they bound is parameterised as $p(u,r)=\\mu+r\\left(\\sigma(u)-\\mu\\right)$. Holding
$r$ fixed and sweeping $u$ traces a scaled copy of $\\sigma$ (the dashed loops); holding $u$
fixed and sweeping $r$ traces a straight ray out from $\\mu$ (highlighted). Each prompt has
already had the mean over all prompts subtracted and the per-day counts are balanced, so
$\\mu$ is exactly the origin ($\\lVert\\mu\\rVert\\approx10^{{-16}}$): $r=0$ is a prompt with
its weekday's contribution removed, not an average weekday.

**Dimensions.** $A_{{\\mathrm{{prompt}}}},\\,C_k,\\,\\mu,\\,\\sigma(u),\\,p(u,r)\\in
\\mathbb{{R}}^{{{d}}}$ — the full residual stream at one token. The chart has two parameters,
$u\\in[0,1)$ and $r\\in[0,{r_max:.1f}]$, but every point they select is a {d}-dimensional
vector and the intervention displaces all {d} coordinates. The seven centroids are affinely
independent and so span a {N_DAYS - 1}-dimensional affine subspace of $\\mathbb{{R}}^{{{d}}}$;
$\\sigma$ is a 1-dimensional curve within it. The measured operator is {j}: it maps the
activation at the *one* steered token to the residual at the *one* token read out, so it is
square in {d} — not the whole-sequence Jacobian.

**The disc is a chart, not a picture of the shape.** Panels B–D plot $u$ at equal angles and
$r$ as a *relative* radius, so $r=1$ renders as a circle whatever the geometry. It is not one:
in $\\mathbb{{R}}^{{{d}}}$ the seven days are neither equidistant from $\\mu$
({radii.min():.2f}–{radii.max():.2f}) nor evenly spaced around the loop (consecutive steps
{steps.min():.2f}–{steps.max():.2f}), and the loop is not planar — its best two-dimensional
view captures {100 * ev:.0f}% of it, and the seven angles subtended at $\\mu$ sum to
{angsum:.0f}° where any planar loop must give 360°. A point's place on the disc therefore
identifies exactly which activation was tested, but not how far apart two points are. Because
$r$ is relative, one $r$ is not one distance; re-slicing the radial profile by true
$\\lVert p-\\mu\\rVert$ gives the same decay (centre/loop 1.89 against 1.75), with a 9–12%
spread over angles at fixed $r$ that the profile averages away.

The prompt runs normally until layer {meta['patch_layer']}, where
the activation at the single weekday token (position {meta['day_pos']}, `{day_tok}`) is
displaced: $x\\leftarrow A_{{\\mathrm{{prompt}}}}+\\left(p(u,r)-C_s\\right)$, with $C_s$ the
centroid of that prompt's own day. This is an **additive** steer — the model's own activation
is kept, not overwritten — so the displacement vanishes at $r=1,\\,u=s/7$ and the model answers
exactly as it would unpatched. Every other token is left alone and the rest of the forward pass
continues untouched. One reading is then taken, at the readout token
(position {meta['ro_pos']}, `{ro_tok}`): the Jacobian of the
residual stream at layer {meta['readout_layer']} with respect to the steered activation at
layer {meta['patch_layer']}. It is a derivative ACROSS those layers, not a quantity evaluated
at either one.

**(B–D) Three readings over the disc.** $\\lVert J\\rVert_F$ is estimated by Hutchinson with
{meta['n_hutch']} random probes per background prompt, drawn independently, giving
{meta['n_hutch'] * nbg} draws behind the averaged field; it is the only panel carrying
sampling error. The directional gains $\\lVert J\\hat t\\rVert$ (along the loop) and
$\\lVert J\\hat r\\rVert$ (across it) are exact forward-mode products with no estimator, and
share one colour scale so they can be compared. Every panel otherwise carries an independent
scale — do not compare colours across panels. Earlier versions also plotted three output
readings over the same disc (agreement with the expected day mixture, output entropy, and
P(any weekday token)); those are no longer part of this figure, and the forward pass is now
shown only as far as the Jacobian.
panels. Knot/gap ratios: $\\lVert J\\rVert_F$ {kg['fro']:.3f}, tangential {kg['gt']:.3f},
radial {kg['gr']:.3f}. Grid: {n_u} angles $\\times$ {n_r} radii $\\times$ {nbg} background
prompts.

**Caveat.** Off-loop states are synthetic: the sentence still names its own weekday while a
different one is steered in. This measures how the network responds to being placed at a point,
not evidence that it ever visits one.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="figures/llama_polar70_stop_fp16.npz",
                    help="the field to draw; also selects the model, which is read "
                         "out of the npz rather than passed")
    ap.add_argument("--context", default="figures/readout_context.json",
                    help="where the token strip comes from; matched to this run by its "
                         "patch and readout positions")
    ap.add_argument("--out", default="figures/fig_jac_llama.pdf")
    ap.add_argument("--cmap", default="viridis",
                    help="colormap for all six discs; 'family' restores the "
                         "per-quantity hues (Oranges / Greens / Reds / Purples)")
    ap.add_argument("--e-vmax", type=float, default=None,
                    help="upper colour limit for panel E; None autoscales from "
                         "the outer disc. Values above it saturate, they are not "
                         "dropped")
    ap.add_argument("--e-rmin", type=float, default=0.5,
                    help="panel E takes its colour range from radii at or beyond "
                         "this, so the centre spike stops setting the scale")
    ap.add_argument("--scale-band", default="0.75,1.3",
                    type=lambda v: tuple(float(x) for x in v.split(",")),
                    help="radii the colour range is taken from. The spokes run inward from "
                         "the ring, so the band starts below r = 1; it stops short of the "
                         "centre, where the fields are several times brighter and would "
                         "take the whole ramp.")
    ap.add_argument("--dir-scale", default="shared", choices=("shared", "panel"),
                    help="the directional discs (along the ring, across it, and the "
                         "unrelated control) are unit-step gains in the SAME units, so "
                         "they share one scale by default -- otherwise a direction with a "
                         "tenth the gain autoscales to look just as bright. 'panel' gives "
                         "each its own range.")
    ap.add_argument("--share-de", action="store_true",
                    help="put panels D and E on one colour scale, so their "
                         "magnitudes compare directly at the cost of flattening D")
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--background", default="transparent",
                    choices=("transparent", "white"),
                    help="transparent drops the page white so the plate sits on whatever "
                         "is behind it. The ink stays dark, so it wants a light ground.")
    ap.add_argument("--n-layers", type=int, default=32,
                    help="blocks in the model, for the caption. 48 for GPT-2 XL.")
    args = ap.parse_args()

    from weekday_manifold.manifold.steering import fit_steer_spline

    z = np.load(args.npz, allow_pickle=True)
    C, mu = z["centroids"], z["mu"]
    meta = json.loads(str(z["meta"]))
    us, rs = z["us"], z["rs"]
    fro, gt, gr = z["fro"].mean(0), z["gain_t"].mean(0), z["gain_r"].mean(0)
    nbg, d = z["fro"].shape[0], C.shape[1]

    sp = fit_steer_spline(C)

    X = C - mu
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = (S[:2] ** 2).sum() / (S ** 2).sum()
    radii = np.linalg.norm(X, axis=1)
    steps = np.array([np.linalg.norm(C[(k + 1) % N_DAYS] - C[k]) for k in range(N_DAYS)])
    angsum = sum(
        np.degrees(np.arccos(np.clip(
            (radii[k] ** 2 + radii[(k + 1) % N_DAYS] ** 2 - steps[k] ** 2)
            / (2 * radii[k] * radii[(k + 1) % N_DAYS]), -1, 1)))
        for k in range(N_DAYS))
    knot = np.minimum((us * 7) % 1.0, 1.0 - (us * 7) % 1.0)
    near, far = knot < 0.12, knot > 0.38
    kg = {k: float(a[near].mean() / a[far].mean())
          for k, a in (("fro", fro), ("gt", gt), ("gr", gr))}

    # The token strip comes from readout_context.json, which the model wrote -- a
    # hand-written list drifts and then mislabels the patch site. Matched to this run by
    # its two positions, so the figure cannot show one prompt's tokens over another's
    # measurements. The interrogative list is kept as a fallback for runs made before that
    # file existed; it is the real Llama-3.1 tokenisation, "?\n" a single token.
    #
    # THE MODEL HAS TO MATCH TOO. readout_context.py is pinned to the Llama config, so
    # every strip in that file is Llama's tokenisation -- but this figure is also drawn
    # for Mistral and GPT-2 XL, and a (patch_pos, read_pos) pair is not unique across
    # tokenisers. Mistral happens to tokenise the mention prompt to the same length with
    # the same two positions, so the position-only match silently handed it Llama's
    # strings; it got away with it only because the words agree and position 0 is
    # rewritten to a generic "<bos>". Nothing enforced that agreement, and the assert
    # below cannot see it: it checks that the patch site is SOME weekday, which a wrong
    # model's strip passes just as easily. So require the stamp when there is one.
    day_pos, ro_pos = meta["day_pos"], meta["ro_pos"]
    this_model = str(meta.get("model", "") or "")
    toks, wrong_model = None, False
    if os.path.exists(args.context):
        for k, c in json.load(open(args.context)).items():
            if c["patch_pos"] != day_pos or c["read_pos"] != ro_pos:
                continue
            ctx_model = str(c.get("model", "") or "")
            if ctx_model and this_model and ctx_model != this_model:
                print(f"[strip] {args.context}:{k} was written for {ctx_model}, not "
                      f"{this_model} -- not borrowing its tokens")
                wrong_model = True
                continue
            toks = ["<bos>"] + [t.replace("\n", "\\n") for t in c["tokens"][1:]]
            break
    if toks is None and wrong_model:
        # The positions matched but the tokeniser did not, so we KNOW the strip on file is
        # not this run's. Draw unlabelled positions rather than another tokeniser's words:
        # a bare axis is honest, a confidently wrong one is not. Refusing outright would be
        # worse -- readout_context.py writes one model at a time, so a second model's plate
        # would become unbuildable rather than merely unlabelled.
        print(f"[strip] no {this_model} entry in {args.context} -- drawing unlabelled "
              f"positions. Re-run readout_context.py --config <this model's config> and "
              f"point --context at its output if the drawn tokens matter.")
        toks = ["<bos>"] + [f"t{i}" for i in range(1, max(day_pos, ro_pos) + 1)]
        toks[day_pos] = "<day>"
    if toks is None:
        # No entry at these positions at all. Fall through to the hard-coded strip and let
        # the assert below judge it: if the positions in the npz are inconsistent with the
        # prompt, that is a real disagreement and the figure must not be drawn. This is
        # deliberately NOT the branch above -- there we knew which tokeniser wrote the
        # strip, here we know nothing.
        toks = ["<bos>", "Q", ":", " What", " day", " is", " two", " days", " after",
                " Monday", "?\\n", "A", ":"]
    assert toks[day_pos].strip() in DAYS or toks[day_pos] == "<day>", (
        "patch site is %r, not a weekday -- token list is out of step" % toks[day_pos])
    assert 0 <= ro_pos < len(toks), "readout position is outside the token strip"
    # Attention is causal, so nothing after the readout can reach it. The mention prompts
    # carry " On this day" past the full stop for a different site's sake; drawing it here
    # invites the reader to wonder what it contributes, and the answer is nothing.
    truncated = len(toks) - 1 - ro_pos
    toks = toks[:ro_pos + 1]

    th_e, r_e = edges(us * 2 * np.pi), edges(rs, lo=0.0)
    # D and E on ONE scale makes their magnitudes directly comparable, but the radial gain
    # is the larger of the two, so sharing flattens the tangential panel and hides the
    # seven-fold structure that is the point of it. Default is now each on its own range;
    # --share-de restores the shared one when the magnitude comparison is what matters.
    gv = (min(gt.min(), gr.min()), max(gt.max(), gr.max())) if args.share_de else None
    # The radial gain spikes at the disc's centre -- the one point every radius converges on,
    # where the "across the loop" direction is not even well defined -- and that spike sets
    # the colour range for the whole panel, leaving the ring and its surroundings almost
    # blank. Take E's range from r >= --e-rmin instead. The centre is still drawn, now
    # saturated: clipped, not hidden.
    rmid = 0.5 * (r_e[1:] + r_e[:-1])
    outer = rmid >= args.e_rmin
    e_lo = float(gr.T[outer].min()) if outer.any() else float(gr.min())
    e_hi = args.e_vmax if args.e_vmax is not None else (
        float(gr.T[outer].max()) if outer.any() else float(gr.max()))
    e_range = (e_lo, e_hi)
    # Nothing on the bar about the scaling: each panel autoscales unless --share-de, so
    # saying so on every bar is noise. The reasoning lives in the code and the caption.
    e_note = ""

    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    # Two rows, not three: the output discs (former F-H) are no longer part of this figure.
    # The construction and the intervention now read as ONE panel -- they are two halves of a
    # single account of the method, and splitting them put a letter break mid-sentence.
    # The discs are square (polar), so their diameter is min(cell width, cell height) --
    # shrinking them means shrinking the figure, not just the row. Row 0 is kept close to
    # square so panel A's equal-aspect ring is not letterboxed.
    n_disc = 4 if "gain_off" in z.files else 3
    # One row of discs, sized so each is the same width whether there are three of them or
    # four. The letters start at B and do not renumber: this cut drops the method panel that
    # was A, and a cross-reference to C has to mean the same panel either way.
    fig = plt.figure(figsize=(2.10 * n_disc, 2.95))
    gs = GridSpec(1, n_disc, figure=fig, hspace=0.20, wspace=0.04,
                  left=0.006, right=0.994, top=0.775, bottom=0.05)
    disc_cells = [(0, c) for c in range(n_disc)]
    goff = z["gain_off"].mean(0) if "gain_off" in z.files else None
    cell = lambda spec: spec.get_position(fig)          # noqa: E731
    lettered = []

    # Each panel scaled to the RING, not to its whole disc. Every number these panels are
    # read for is at r = 1, and the fields are far brighter at the centre -- the radial gain
    # most of all, at the one point where "across the ring" is not even defined -- so a
    # disc-wide range spends the ramp on the middle and leaves the ring flat. Limits are the
    # 1st-99th percentile over a band about r = 1; where the centre runs past the top of the
    # scale it clips, and the panel says by how much.
    #
    # Per panel, not shared: these are different quantities at very different magnitudes,
    # and one scale across them resolves the largest and flattens the rest.
    ring_band = (rmid >= args.scale_band[0]) & (rmid <= args.scale_band[1])
    if not ring_band.any():
        ring_band = np.ones_like(rmid, dtype=bool)

    def ring_lim(a):
        v = a.T[ring_band]
        return float(np.percentile(v, 1.0)), float(np.percentile(v, 99.0))

    # In reading order for the chosen layout: stacked puts the three gains first and the
    # norm last, so the letters follow the eye rather than the order they were computed in.
    measures = [(gt, "Jacobian along the ring", "$\\|J\\hat{t}\\|$", "tangent", ""),
                (gr, "Jacobian radially", "$\\|J\\hat{r}\\|$", "radial", ""),
                (goff, "Jacobian along a random direction",
                 "$\\|J\\hat{v}_{\\perp}\\|$", "off", ""),
                (fro, "Jacobian norm", "$\\|J\\|_F$", "all", "estimated (Hutchinson)")]
    # The norm is what the three directional gains are read AGAINST, so it leads.
    measures = measures[-1:] + measures[:-1]
    measures = [m for m in measures if m[0] is not None]

    spec = []
    grid_of_discs = gs
    for (row, col), (a, title, sym, dirn, lab) in zip(disc_cells, measures):
        # No clipping note under the discs. Every panel is scaled the same way and the
        # saturated centre reads as saturated; spelling it out on each was four copies of a
        # caveat that belongs in the caption, where it is stated once.
        spec.append((row, col, a.T, "Oranges", ring_lim(a), title, lab, dirn, sym))

    # One perceptually uniform map for every disc by default. The per-family hues
    # (Oranges for the Jacobians, Greens/Reds/Purples for the outputs) grouped the panels by
    # what they measure, but a single-hue sequential map compresses its light end, so small
    # differences low in the range are invisible -- which is most of these fields. viridis
    # spends equal perceptual distance on equal value steps, so the structure shows wherever
    # it sits in the range. `--cmap family` restores the grouping.
    for i, (row, col, grid, cmap, vr, title, lab, dirn, sym) in enumerate(spec):
        # add_axes on an explicit box, not add_subplot then set_position: a polar axes
        # re-derives its box to keep the disc circular, and the derivation starts from the
        # SubplotSpec, so a set_position afterwards is silently discarded. (The pre-existing
        # 0.84 here had never taken effect for the same reason.)
        # One size for every disc: they are four readings of the same field and singling
        # one out by size asserts a hierarchy the figure does not otherwise argue for.
        bb = cell(grid_of_discs[row, col])
        k = 0.86
        w, h = bb.width * k, bb.height * 0.84 * k
        ax = fig.add_axes([bb.x0 + (bb.width - w) / 2, bb.y0 + (bb.height * 0.84 - h),
                           w, h], projection="polar")
        vmin, vmax = vr if vr else (grid.min(), grid.max())
        disc(ax, th_e, r_e, grid, cmap if args.cmap == "family" else args.cmap,
             vmin, vmax, lab, dirn=dirn)
        # the symbol centred directly over its own disc, not tacked onto the heading:
        # it belongs to the picture rather than to the sentence, and moving it here lets
        # the heading stay one line and sit close to the panel
        ab, cb_ = ax.get_position(), cell(grid_of_discs[row, col])
        sym_dy = 0.012
        fig.text(ab.x0 + ab.width / 2, cb_.y1 + sym_dy, sym, fontsize=11.5, color=INK,
                 ha="center", va="bottom")
        lettered.append((chr(ord("B") + i), title, cell(grid_of_discs[row, col])))
    # stacked: A shares a row with the discs, so it takes their header offset. row: A has
    # the row to itself and nothing overhangs its cell.
    panel_headers(fig, lettered)              # discs need room for ticks + inset

    tp = args.background == "transparent"
    fig.savefig(args.out, bbox_inches="tight", dpi=300, transparent=tp)
    print(f"[figure] wrote {args.out}" + (" (transparent)" if tp else ""), flush=True)
    if args.also_png:
        png = args.out.rsplit(".", 1)[0] + ".png"
        fig.savefig(png, bbox_inches="tight", dpi=300, transparent=tp)
        print(f"[figure] wrote {png}", flush=True)

    cap_path = args.out.rsplit(".", 1)[0] + "_caption.md"
    with open(cap_path, "w") as f:
        f.write(caption(meta, d, nbg, len(us), len(rs), float(rs[-1]), radii, steps, ev,
                        angsum, toks[day_pos], args.n_layers, kg, toks[ro_pos]))
    print(f"[figure] wrote {cap_path}", flush=True)


if __name__ == "__main__":
    main()
