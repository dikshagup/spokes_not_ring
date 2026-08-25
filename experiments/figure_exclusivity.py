"""Figure 7: the weekday subspace is privileged but not exclusive.

Four panels off figure 3's capture chain, with no captures of its own.

Writes figures/exclusivity_figure.{pdf,png}. See repro_fig7_exclusivity.sh.
"""
from __future__ import annotations
import argparse, os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

# The research repo needed a stub-module shim here to import the spline without executing
# its package __init__, which pulled in torch purely to re-export load_model. This repo's
# weekday_manifold/__init__.py resolves load_model lazily (PEP 562) for exactly that reason,
# so a plain import is already torch-free and the shim is gone.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, _SRC)
from weekday_manifold.manifold.spline import PeriodicSpline  # noqa: F401,E402
# The same operator arc_occupancy_main's foot points are built on. Shared rather
# than redefined: these two figures draw the same spline through the same knots,
# and two copies of it could drift apart without either figure erroring.
from arc_geometry import spline_op  # noqa: E402

# Large intermediates default under the user cache rather than a fixed pod path, so this
# runs without root and off the working tree. repro_fig3_arc_occupancy.sh passes these
# explicitly (CORPUS= / RAWDIR=), so the published figure does not depend on the default.
_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                      "weekday-manifold")
CORPUS_DIR = os.path.join(_CACHE, "corpus_v2")
RAW_DIR = os.path.join(_CACHE, "corpus_v2_raw")

plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
# A, C and D all show a ring, and they are NOT the same ring: A draws the pooled day
# centroids of real corpus windows, C and D draw the mean of the de-meaned swap families.
# The two differ by 1.25x in this plane and by 38% in the full space, so one encoding per
# object -- BLACK solid is the corpus ring and appears only in A, RED solid is the swap
# mean and appears only in C and D. Individual contexts in C are dashed greys and never
# solid, so no dashed curve can be read as either reference.
C_SWAP = "#c02a2a"
DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2"]
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
FS_LAB, FS_TICK, FS_LEG, FS_TITLE, FS_LET, FS_DAY = 8.5, 8.0, 7.2, 9.0, 11.0, 7.6
FAMILIES = [("CMS timestamp", r"\b(last updated|updated on|posted on|last edited|published|posted|in sale since)\b"),
            ("opening hours", r"\b(open|opening|hours|closed)\b"),
            ("news attribution", r"\b(said|says|told|announced|reported|testified|stated)\b"),
            ("plain date ref", r"\bon$")]


def tidy(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6); ax.spines[sp].set_color(PALE)
    ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6, colors=MID)
    ax.set_axisbelow(True)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(RAW_DIR, "raw_L28_n16.npz"))
    ap.add_argument("--swap", default=os.path.join(RAW_DIR, "swap_L28_n16.npz"))
    ap.add_argument("--list", default=os.path.join(CORPUS_DIR, "capture_list_n256.npz"))
    ap.add_argument("--out", default="figures/exclusivity_figure.pdf")
    ap.add_argument("--plane", choices=("centroid", "rawpca"), default="centroid",
                    help="centroid: top-2 PCs of the seven day centroids, and the rank-6 "
                         "centroid span for panel B (the published figure). rawpca: fully "
                         "unsupervised -- PCA of the raw weekday windows with no day "
                         "labels, components ranked by held-out between-day SS, top 2 for "
                         "the plane and top 6 for panel B's subspace.")
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    W = spline_op(600)

    # ---------- one geometry, fitted once ----------------------------------
    R = np.load(args.raw, allow_pickle=True)
    A16 = R["acts"]
    cls = np.asarray(R["cls"]).astype(str); day = np.asarray(R["day"]).astype(str)
    isfit = np.asarray(R["is_fit"]).astype(bool)
    fitm = (cls == "positive") & isfit
    Xf = A16[fitm].astype(np.float64)
    mu = Xf.mean(0)
    cents = np.stack([Xf[day[fitm] == d].mean(0) for d in DAYS])
    basis6 = np.linalg.svd(cents - mu, full_matrices=False)[2][:6]
    mean_c = cents.mean(0)
    sv, Vt = np.linalg.svd(cents - mean_c, full_matrices=False)[1:]
    plane, pc_var = Vt[:2], (sv ** 2 / (sv ** 2).sum())[:2]
    ring_R = np.linalg.norm(cents - mean_c, axis=1).mean()
    axis_note = "top-2 PCs of the seven day centroids (supervised)"

    if args.plane == "rawpca":
        # The unsupervised alternative: PCA on the raw weekday windows themselves, no day
        # labels, so the plane cannot have been shaped by the thing it is used to display.
        # Components are ranked by between-day sum of squares on the HELD-OUT positives --
        # same statistic and same held-out scoring as experiments/figure_corpus_rawpca.py,
        # so the two agree on which components carry weekday.
        Vu = np.linalg.svd(Xf - mu, full_matrices=False)[2][:30]
        Xh = A16[(cls == "positive") & ~isfit].astype(np.float64)
        dh = day[(cls == "positive") & ~isfit]
        zh = (Xh - mu) @ Vu.T; tot_h = float(((Xh - mu) ** 2).sum()); del Xh
        d_ss = np.array([sum(((zh[dh == u, k].mean() - zh[:, k].mean()) ** 2) * (dh == u).sum()
                             for u in DAYS) for k in range(30)])
        pick = np.argsort(-d_ss)[:6]
        # Panel B has to move with the plane, or the figure would be half-supervised:
        # its 6-D subspace becomes the six unsupervised components carrying the most
        # weekday, by the same held-out between-day SS that ranks the plane's two.
        basis6 = Vu[pick]
        pick = pick[:2]
        plane = Vu[pick]
        pc_var = (sv ** 2 / (sv ** 2).sum())[:2] * 0 + np.nan   # not a centroid-variance plane
        axis_note = (f"unsupervised PCs {pick[0]+1} and {pick[1]+1} of the raw weekday windows "
                     f"(between-day SS {100*d_ss[pick[0]]/tot_h:.2f}% and "
                     f"{100*d_ss[pick[1]]/tot_h:.2f}% of held-out total variance)")
    del Xf

    N = len(A16)
    xy = np.empty((N, 2)); frac = np.empty(N)
    for s in range(0, N, 20000):                      # chunked: 117k x 4096 in f64 is 3.8 GB
        C = A16[s:s + 20000].astype(np.float64)
        xy[s:s + 20000] = (C - mean_c) @ plane.T
        d = C - mu
        frac[s:s + 20000] = ((d @ basis6.T) ** 2).sum(1) / np.maximum((d ** 2).sum(1), 1e-30)
    ps, mneg = (cls == "positive") & ~isfit, cls == "matched_negative"

    # ---------- panels C/D: the swap set ------------------------------------
    S = np.load(args.swap, allow_pickle=True)
    o = np.lexsort((S["swap"], S["fam"]))
    sel = (S["fam_cls"] == "positive") & ~S["fam_is_fit"]
    Xs = S["acts"][o][np.repeat(sel, 7)[o]].reshape(sel.sum(), 7, 4096).astype(np.float64)
    fam_mean = Xs.mean(1); D = Xs - fam_mean[:, None, :]; del Xs
    T = D.mean(0); offs = (fam_mean - mean_c) @ plane.T
    shift = np.linalg.norm(offs, axis=1) / ring_R
    KT = T @ plane.T

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    IDS = np.load(args.list, allow_pickle=True)["ids"]
    rows = S["fam_row"][sel]; fday = S["fam_day"][sel]
    ctxl = [tok.decode(IDS[r][-16:-1]).lower() for r in rows]
    picks = []
    for _, pat in FAMILIES:
        hit = np.array([bool(re.search(pat, c)) for c in ctxl])
        if hit.any():
            cand = np.where(hit)[0]; picks.append(int(cand[np.argmax(shift[cand])]))

    def prompt(i):
        r = rows[i]
        return tok.decode(IDS[r][-16:-1]) + tok.decode([IDS[r][-1]]).replace(str(fday[i]), "{weekday}")

    # ---------- layout ------------------------------------------------------
    fig = plt.figure(figsize=(7.4, 7.45))
    gs = fig.add_gridspec(2, 2, left=0.100, right=0.985, top=0.966, bottom=0.060,
                          wspace=0.28, hspace=0.34)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for a in (axA, axC, axD): a.set_anchor("W")

    # ===== A: the corpus in the ring plane ==================================
    tidy(axA); axA.set_aspect("equal")
    sn = rng.choice(np.where(mneg)[0], 4000, replace=False)
    axA.plot(xy[sn, 0], xy[sn, 1], ".", ms=1.0, color=PALE, alpha=0.30, zorder=2)
    # ALL 3,294 held-out weekday mentions, not a subsample: they are the same windows
    # panels C and D are built from, so the three panels show one population throughout.
    # The grey control cloud stays subsampled (4,000 of 68,000) purely for legibility --
    # which means the relative density of the two clouds carries no information.
    sp = np.where(ps)[0]
    for k in range(7):
        kk = sp[day[sp] == DAYS[k]]
        axA.plot(xy[kk, 0], xy[kk, 1], "o", ms=1.9, color=DAY_C[k], alpha=0.75,
                 markeredgecolor="none", zorder=3)
    cxy = W @ ((cents - mean_c) @ plane.T)
    axA.plot(cxy[:, 0], cxy[:, 1], "-", color="white", lw=2.6, zorder=4)
    axA.plot(cxy[:, 0], cxy[:, 1], "-", color=INK, lw=1.0, zorder=5)
    ck = (cents - mean_c) @ plane.T
    for k in range(7):
        axA.plot(ck[k, 0], ck[k, 1], "o", ms=5.5, color=DAY_C[k], markeredgecolor=INK,
                 markeredgewidth=0.7, zorder=6)
        axA.annotate(DAYS[k][:3], (ck[k, 0] * 1.45, ck[k, 1] * 1.45), fontsize=FS_DAY,
                     color=INK, ha="center", va="center", zorder=7, fontweight="bold")
    axA.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                               color=PALE, alpha=0.95,
                               label="Other capitalised words")],
               frameon=False, fontsize=FS_LEG, loc="upper right", handletextpad=0.4,
               borderaxespad=0.1)

    # ===== B: share of energy inside the subspace ===========================
    tidy(axB)
    # Class labels say what the window ENDS IN, which is what the selection pass keyed
    # on: near_miss is 91% month names and 9% yesterday/tomorrow, matched_negative is any
    # other capitalised word (The, This, You, And -- matched to the positives on
    # capitalisation, the obvious lexical confound), floor is an arbitrary final token.
    # Wrapped to <=11 characters a line: each category gets ~0.7in of panel, so a longer
    # line runs into its neighbour.
    CL = [("Weekday\nmentions", ps),
          ("Months,\nyesterday,\ntomorrow", cls == "near_miss"),
          ("Other\ncapitalised\nwords", mneg),
          ("Random\ntokens", cls == "floor")]
    for i, (nm, m) in enumerate(CL):
        v = frac[m]
        pv = axB.violinplot([v], positions=[i], widths=0.74, showextrema=False, showmedians=False)
        for b in pv["bodies"]:
            b.set_facecolor(MID); b.set_alpha(0.30); b.set_edgecolor(MID); b.set_linewidth(0.7)
        sub = rng.choice(v, min(350, len(v)), replace=False)
        axB.plot(i + rng.normal(0, 0.055, len(sub)), sub, ".", ms=1.3, color=MID, alpha=0.20, zorder=2)
        md = np.median(v)
        axB.plot([i - 0.30, i + 0.30], [md] * 2, color=MID, lw=2.0, zorder=4)
        # Sitting the label ON the median put it against the violin edge. Lifted just
        # above the bar instead, still anchored to it so it cannot drift onto a neighbour.
        axB.text(i + 0.30, md + 0.012, f"{100*md:.2f}%", fontsize=FS_LEG, color=MID,
                 zorder=6, va="bottom", ha="left",
                 bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8))
    axB.set_xticks(range(4)); axB.set_xticklabels([n for n, _ in CL], fontsize=FS_TICK - 1.2)
    axB.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    shown = np.zeros(N, bool)
    for _, m in CL: shown |= m
    axB.set_ylim(0, 1.03 * frac[shown].max()); axB.set_xlim(-0.6, 3.9)
    axB.set_ylabel("Share of energy inside the 6-D weekday subspace", fontsize=FS_LAB, color=INK)
    print("panel B medians: " + ", ".join(
        f"{nm.replace(chr(10),' ')} {100*np.median(frac[m]):.2f}%" for nm, m in CL))

    # ===== C: one ring per context ==========================================
    tidy(axC); axC.set_aspect("equal")
    LS, GREY = [(0, (7, 2)), "--", "-.", ":"], ["#16181d", "#4a5160", "#79808f", "#a6acb8"]
    for j, i in enumerate(picks):
        K = D[i] @ plane.T + offs[i]; c = W @ K
        axC.plot(c[:, 0], c[:, 1], linestyle=LS[j], color=GREY[j], lw=1.2, zorder=4,
                 label=f"Template {j+1}")
        for k in range(7):
            axC.plot(K[k, 0], K[k, 1], "o", ms=4.4, color=DAY_C[k],
                     markeredgecolor=GREY[j], markeredgewidth=0.6, zorder=5)
    tc = W @ KT
    axC.plot(tc[:, 0], tc[:, 1], "-", color=C_SWAP, lw=1.9, alpha=0.9, zorder=3,
             label="Mean")
    axC.legend(frameon=False, fontsize=FS_LEG, loc="upper right", handletextpad=0.5,
               borderaxespad=0.1, labelspacing=0.28)

    # ===== D: the whole de-meaned population ================================
    tidy(axD); axD.set_aspect("equal")
    take = rng.choice(len(D), min(2200, len(D)), replace=False)
    P = D[take] @ plane.T
    for k in range(7):
        axD.plot(P[:, k, 0], P[:, k, 1], "o", ms=1.6, color=DAY_C[k], alpha=0.5,
                 markeredgecolor="none", zorder=3)
    axD.plot(tc[:, 0], tc[:, 1], "-", color="white", lw=2.6, zorder=5)
    axD.plot(tc[:, 0], tc[:, 1], "-", color=C_SWAP, lw=1.6, zorder=6)
    for k in range(7):
        axD.plot(KT[k, 0], KT[k, 1], "o", ms=5.5, color=DAY_C[k], markeredgecolor=INK,
                 markeredgewidth=0.7, zorder=7)
        axD.annotate(DAYS[k][:3], (KT[k, 0] * 1.42, KT[k, 1] * 1.42), fontsize=FS_DAY,
                     color=INK, ha="center", va="center", zorder=8, fontweight="bold")

    # C and D share limits: same plane, so a shared scale keeps C's displacement
    # comparable to D's spread, and equal-aspect boxes of unequal limits open a gap.
    ALL = np.concatenate([np.r_[tc, KT * 1.5], P.reshape(-1, 2)] +
                         [W @ (D[i] @ plane.T + offs[i]) for i in picks])
    lo, hi = ALL.min(0), ALL.max(0); pad = 0.04 * (hi - lo).max()
    # Extended up and to the right by 20% to clear C's legend, anchored at the lower
    # left, and kept SQUARE: with set_aspect("equal") an unequal span would reshape the
    # axes box and break the row alignment with A and B.
    span = 1.04 * ((hi - lo).max() + 2 * pad)
    for a in (axC, axD):
        a.set_xlim(lo[0] - pad, lo[0] - pad + span)
        a.set_ylim(lo[1] - pad, lo[1] - pad + span)

    xl, yl = ("PC1", "PC2") if args.plane == "centroid" else ("unsup. PC8", "unsup. PC5")
    for a in (axA, axC, axD):
        a.set_xlabel(xl, fontsize=FS_LAB, color=INK); a.set_ylabel(yl, fontsize=FS_LAB, color=INK)

    fig.canvas.draw()
    HEAD = [("A", "FineWeb prompts in the ring plane", axA),
            ("B", "Energy inside the subspace vs outside", axB),
            ("C", "Weekday ring is translated by prompt template", axC),
            ("D", "De-meaned prompt projections", axD)]
    for letter, title, a in HEAD:
        p = a.get_position()
        fig.text(p.x0 - 0.075, p.y1 + 0.014, letter, fontsize=FS_LET, fontweight="bold",
                 color=INK, ha="left", va="bottom")
        fig.text(p.x0 - 0.048, p.y1 + 0.016, title, fontsize=FS_TITLE, color=INK,
                 ha="left", va="bottom")

    # No overall title and no prompt list on the figure: both are caption material.
    # Printed instead, along with the per-axis variance shares the axis labels no longer
    # carry, so everything the caption needs comes off one run.
    print(f"\nfor the caption: plane = {axis_note}")
    if args.plane == "centroid":
        print(f"  PC1 = {100*pc_var[0]:.1f}% of centroid variance, PC2 = {100*pc_var[1]:.1f}%")
    print("panel C templates:")
    for j, i in enumerate(picks):
        print(f"  {j+1}. {prompt(i)!r}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[saved] {args.out}")
    if args.also_png:
        p = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(p, dpi=300, bbox_inches="tight"); print(f"[saved] {p}")
    return 0


if __name__ == "__main__":
    rc = main(); sys.stdout.flush(); os._exit(rc)
