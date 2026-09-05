#!/usr/bin/env python
"""Figure 3, and the appendix plate off the same refit.

--panels abcd is the main-text cut: a 2x2 that pairs where weekday windows fall (A, B)
with what the 7-way weekday swap does to that (C, D). C and D were the appendix plate's
B and C; the subspace-energy violin the old --panels abc carried as its third panel is
dropped, and with it the appendix plate -- see --appendix, which is now off by default.

--panels ab is the previously published two-panel cut, and --panels abc the three-panel
row whose md5 is pinned in repro_fig3_arc_occupancy.sh -- kept because that hash is the
isolation check on this stage.

Writes figures/arc_occupancy_main_abcd. The appendix plate is opt-in (--appendix) and
--verify turns it on, since its md5 is pinned. See repro_fig3_arc_occupancy.sh.
"""
from __future__ import annotations
import argparse, os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

# _HERE on the path so the foot-point machinery is IMPORTED from the single-figure
# script rather than copied: the two files must not be able to drift apart on what
# "nearest point on the spline" means.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _SRC); sys.path.insert(0, _HERE)
from weekday_manifold.manifold.spline import PeriodicSpline  # noqa: E402
from arc_geometry import foot_points, chord_feet, arc_frac  # noqa: E402

# Large intermediates default under the user cache rather than a fixed pod path, so this
# runs without root and off the working tree. repro_fig3_arc_occupancy.sh passes these
# explicitly (CORPUS= / RAWDIR=), so the published figure does not depend on the default.
_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                      "weekday-manifold")
CORPUS_DIR = os.path.join(_CACHE, "corpus_v2")
RAW_DIR = os.path.join(_CACHE, "corpus_v2_raw")

plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
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


def spline_op(n=600):
    return np.stack([PeriodicSpline(np.eye(7)[:, [j]]).sample(n)[1][:, 0] for j in range(7)], 1)


def square(ax, pts, pad_frac=0.04, grow=1.04, dy=0.0):
    lo, hi = pts.min(0), pts.max(0)
    pad = pad_frac * (hi - lo).max()
    span = grow * ((hi - lo).max() + 2 * pad)
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2 - dy * span, mid[1] + span / 2 - dy * span)


def letter_titles(fig, head, dx_let=0.072, dx_tit=0.046, dy=0.020, fs_title=FS_TITLE,
                  fs_let=FS_LET):
    """Letter + title above each panel, keyed off the axes box AFTER equal-aspect shrink."""
    fig.canvas.draw()
    for letter, title, a in head:
        p = a.get_position()
        fig.text(p.x0 - dx_let, p.y1 + dy - 0.002, letter, fontsize=fs_let,
                 fontweight="bold", color=INK, ha="left", va="bottom")
        fig.text(p.x0 - dx_tit, p.y1 + dy, title, fontsize=fs_title, color=INK,
                 ha="left", va="bottom")


def fit_titles(fig, head, gap=0.009, right=0.999, ratio=1.20, fs_max=13.0, dy=0.020,
               dx=None):
    """Letter + title above each panel, at the largest size that fits ALL of them."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W = fig.get_figwidth() * fig.dpi
    REF = 10.0

    def w_ref(txt, weight="normal"):
        t = fig.text(0, 0, txt, fontsize=REF, fontweight=weight)
        w = t.get_window_extent(r).width / W
        t.remove()
        return w

    anchors = [a.get_tightbbox(r).x0 / W + (dx[i] if dx else 0.0)
               for i, (_, _, a) in enumerate(head)]
    # Nudging one anchor right does not only move that panel: it widens the run of the
    # panel to its LEFT, so the common size can go up rather than down.
    bounds = [anchors[i + 1] - gap for i in range(len(head) - 1)] + [right]
    wt = [w_ref(t) for _, t, _ in head]
    wl = max(w_ref(l, "bold") for l, _, _ in head)
    room = [(bounds[i] - anchors[i] - gap) * REF / (wl * ratio + wt[i])
            for i in range(len(head))]
    fs = min([fs_max] + room)
    print("  title fit: " + ", ".join(f"{h[0]} <= {v:.1f}pt" for h, v in zip(head, room))
           + f"  -> {fs:.1f}pt")
    y = max(a.get_position().y1 for _, _, a in head) + dy
    for i, (letter, title, _) in enumerate(head):
        fig.text(anchors[i], y - 0.002, letter, fontsize=fs * ratio, fontweight="bold",
                 color=INK, ha="left", va="bottom")
        fig.text(anchors[i] + wl * fs * ratio / REF + gap, y, title, fontsize=fs,
                 color=INK, ha="left", va="bottom")
    return fs


def vcenter(fig, move, ref):
    """Shift `move` axes so their inked height sits centred on the band `ref` spans."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    H = fig.get_figheight() * fig.dpi
    bb = [a.get_tightbbox(r) for a in ref]
    lo, hi = min(b.y0 for b in bb), max(b.y1 for b in bb)
    for a in move:
        b = a.get_tightbbox(r)
        dy = (0.5 * (hi + lo) - 0.5 * (b.y1 + b.y0)) / H
        p = a.get_position()
        a.set_position([p.x0, p.y0 + dy, p.width, p.height], which="both")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(RAW_DIR, "raw_L28_n16.npz"))
    ap.add_argument("--swap", default=os.path.join(RAW_DIR, "swap_L28_n16.npz"))
    ap.add_argument("--list", default=os.path.join(CORPUS_DIR, "capture_list_n256.npz"))
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--panels", default="abcd", choices=("ab", "abc", "abcd"),
                    help="main plate: 'abcd' (default) is the 2x2 -- the ring plane, the "
                         "ring itself, and the two swap panels promoted from the "
                         "appendix -- and writes figures/arc_occupancy_main_abcd. "
                         "'ab' is the previously published cut (figures/"
                         "arc_occupancy_main_ab) -- the ring plane and the ring itself. "
                         "'abc' adds the subspace-energy violin as panel C and writes "
                         "figures/arc_occupancy_main; that panel is the appendix plate's "
                         "panel B, so the two would state it twice. 'abc' is what the "
                         "pinned md5 covers, so --verify still renders it.")
    ap.add_argument("--pick-context", action="append", metavar="TEXT",
                    help="pin a panel-C template by a substring of its decoded context, "
                         "repeatable, matched case-insensitively against the same 15 "
                         "tokens the FAMILIES regexes see. Checked before those regexes; "
                         "whatever is left of the four slots they fill as before. Use it "
                         "to reproduce a published set of exemplars, which the "
                         "one-per-family rule cannot.")
    ap.add_argument("--appendix", action="store_true",
                    help="also render figures/arc_occupancy_appendix. Off by default: "
                         "its swap panels are now the main plate's C and D, and rendering "
                         "both would state one measurement twice. --verify turns it back "
                         "on, since the pinned md5 below covers it.")
    ap.add_argument("--grid", type=int, default=7000)
    ap.add_argument("--bins", type=int, default=112)
    ap.add_argument("--gate-q", type=float, default=0.25)
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    W = spline_op()

    # ---------- one geometry, fitted once: all six panels share this fit ----------
    R = np.load(args.raw, allow_pickle=True)
    A16 = R["acts"]
    cls = np.asarray(R["cls"]).astype(str); day = np.asarray(R["day"]).astype(str)
    isfit = np.asarray(R["is_fit"]).astype(bool)
    fitm = (cls == "positive") & isfit
    Xf = A16[fitm].astype(np.float64)
    mu = Xf.mean(0)
    cents = np.stack([Xf[day[fitm] == d].mean(0) for d in DAYS]); del Xf
    basis6 = np.linalg.svd(cents - mu, full_matrices=False)[2][:6]
    mean_c = cents.mean(0)
    sv, Vt = np.linalg.svd(cents - mean_c, full_matrices=False)[1:]
    plane, pc_var = Vt[:2], (sv ** 2 / (sv ** 2).sum())[:2]
    hull6 = Vt[:6]                       # the centroids' affine hull: where the spline lives
    ring_R = np.linalg.norm(cents - mean_c, axis=1).mean()

    N = len(A16)
    xy = np.empty((N, 2)); frac = np.empty(N)
    for s in range(0, N, 20000):
        C = A16[s:s + 20000].astype(np.float64)
        xy[s:s + 20000] = (C - mean_c) @ plane.T
        d = C - mu
        frac[s:s + 20000] = ((d @ basis6.T) ** 2).sum(1) / np.maximum((d ** 2).sum(1), 1e-30)
    ps, mneg = (cls == "positive") & ~isfit, cls == "matched_negative"

    # ---------- foot points on the spline ----------------------------------
    K6 = (cents - mean_c) @ hull6.T
    Wd = spline_op(args.grid)
    curve_full = Wd @ K6
    s_full = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve_full, axis=0), axis=1))]
    L = float(s_full[-1])
    curve, s_grid = curve_full[:-1], s_full[:-1]
    s_knot = s_full[np.arange(7) * (args.grid // 7)]
    foot = {}
    for name, m in (("positives", ps), ("negatives", mneg)):
        idx = np.where(m)[0]
        z = np.empty((len(idx), 6))
        for a in range(0, len(idx), 20000):
            z[a:a + 20000] = (A16[idx[a:a + 20000]].astype(np.float64) - mean_c) @ hull6.T
        sp_, dp_, amb_ = foot_points(z, curve, s_grid, L, sep=L / 14.0)
        cd_, cf_ = chord_feet(z, K6)
        foot[name] = dict(z=z, s=sp_, d=dp_, amb=amb_, chord=cd_, chordf=cf_,
                          rad=np.linalg.norm(z, axis=1))
    gate = float(np.quantile(foot["positives"]["rad"], args.gate_q))
    pday = np.array([DAYS.index(d) for d in day[ps]])

    edges = np.linspace(0, L, args.bins + 1); wid = L / args.bins
    hp_day = np.stack([np.histogram(foot["positives"]["s"][pday == k], bins=edges)[0]
                       for k in range(7)]).astype(float)
    hp_day /= (hp_day.sum() * wid)
    hn = np.histogram(foot["negatives"]["s"], bins=edges)[0].astype(float)
    hn /= (hn.sum() * wid)
    cum = np.vstack([np.zeros(args.bins), np.cumsum(hp_day, 0)])
    ctr = 0.5 * (edges[:-1] + edges[1:])

    # ---------- the swap set (appendix B and C) ----------------------------
    S = np.load(args.swap, allow_pickle=True)
    o = np.lexsort((S["swap"], S["fam"]))
    sel = (S["fam_cls"] == "positive") & ~S["fam_is_fit"]
    Xs = S["acts"][o][np.repeat(sel, 7)[o]].reshape(sel.sum(), 7, 4096).astype(np.float64)
    fam_mean = Xs.mean(1); Dm = Xs - fam_mean[:, None, :]; del Xs
    T = Dm.mean(0); offs = (fam_mean - mean_c) @ plane.T
    shift = np.linalg.norm(offs, axis=1) / ring_R
    KT = T @ plane.T; tc = W @ KT

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    IDS = np.load(args.list, allow_pickle=True)["ids"]
    rows = S["fam_row"][sel]; fday = S["fam_day"][sel]
    ctxl = [tok.decode(IDS[r][-16:-1]).lower() for r in rows]
    # --pick-context PINS a template by its text, and is checked before the family
    # regexes. The regexes take at most ONE window per family and so cannot return two
    # from the same family -- which the published set contained ("...is open 6 days a
    # week," and "...The winery is open" are both `opening hours`), so that set is not
    # reachable through them at all. Pinning by substring sidesteps the rule rather
    # than reweighting it. A pin that matches nothing is REPORTED, not silently
    # dropped: a corpus that no longer holds the window is a fact worth seeing, and a
    # family fallback would paper over it.
    picks, pinned = [], []
    for want in (args.pick_context or []):
        w = want.strip().lower()
        hit = np.array([w in c for c in ctxl])
        if hit.any():
            cand = np.where(hit)[0]
            i = int(cand[np.argmax(shift[cand])])
            if i not in pinned:
                picks.append(i); pinned.append(i)
                print(f"[pick] pinned  {want!r}")
            else:
                print(f"[pick] dup     {want!r} -- already pinned by an earlier string")
        else:
            print(f"[pick] MISSING {want!r} -- not in this corpus")
    n_pinned = len(picks)
    for _, pat in FAMILIES:
        if len(picks) >= 4:
            break
        hit = np.array([bool(re.search(pat, c)) for c in ctxl])
        hit[pinned] = False                      # never draw one window twice
        if hit.any():
            cand = np.where(hit)[0]; picks.append(int(cand[np.argmax(shift[cand])]))
    if args.pick_context:
        print(f"[pick] {n_pinned} pinned, {len(picks) - n_pinned} from families")

    def prompt(i):
        r = rows[i]
        return tok.decode(IDS[r][-16:-1]) + tok.decode([IDS[r][-1]]).replace(str(fday[i]), "{weekday}")

    sn = rng.choice(np.where(mneg)[0], 4000, replace=False)
    take = rng.choice(len(Dm), min(2200, len(Dm)), replace=False)
    P = Dm[take] @ plane.T
    ck = (cents - mean_c) @ plane.T
    cxy = W @ ck

    os.makedirs(args.outdir, exist_ok=True)
    xl, yl = "PC1", "PC2"

    def save(fig, stem):
        p = os.path.join(args.outdir, stem + ".pdf")
        fig.savefig(p, bbox_inches="tight", pad_inches=0.015)
        w, h = fig.get_size_inches()
        print(f"[saved] {p}  (authored {w:.2f} x {h:.2f} in)")
        if args.also_png:
            q = os.path.join(args.outdir, stem + ".png")
            fig.savefig(q, dpi=300, bbox_inches="tight", pad_inches=0.015)
            print(f"[saved] {q}")

    # ---------- reusable panel painters ------------------------------------
    def panel_ring(ax, legend_loc="upper right", fs_day=FS_DAY, legend=True,
                   fs_lab=FS_LAB, ms_pos=1.3, alpha_pos=0.50, ms_neg=0.9,
                   alpha_neg=0.45, fs_leg=FS_LEG, ms_leg=4.5):
        """FineWeb windows in the ring plane, controls in grey."""
        tidy(ax); ax.set_aspect("equal")
        ax.plot(xy[sn, 0], xy[sn, 1], ".", ms=ms_neg, color=PALE, alpha=alpha_neg,
                zorder=2)
        spi = np.where(ps)[0]
        for k in range(7):
            kk = spi[day[spi] == DAYS[k]]
            ax.plot(xy[kk, 0], xy[kk, 1], "o", ms=ms_pos, color=DAY_C[k], alpha=alpha_pos,
                    markeredgecolor="none", zorder=3)
        ax.plot(cxy[:, 0], cxy[:, 1], "-", color="white", lw=2.6, zorder=4)
        ax.plot(cxy[:, 0], cxy[:, 1], "-", color=INK, lw=1.0, zorder=5)
        for k in range(7):
            ax.plot(ck[k, 0], ck[k, 1], "o", ms=5.5, color=DAY_C[k], markeredgecolor=INK,
                    markeredgewidth=0.7, zorder=6)
            ax.annotate(DAYS[k][:3], (ck[k, 0] * 1.45, ck[k, 1] * 1.45), fontsize=fs_day,
                        color=INK, ha="center", va="center", zorder=7, fontweight="bold")
        if legend:
            ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=ms_leg,
                                      color=PALE, alpha=0.95, label="Other capitalised words")],
                      frameon=False, fontsize=fs_leg, loc=legend_loc, handletextpad=0.4,
                      borderaxespad=0.1)
        ax.set_xlabel(xl, fontsize=fs_lab, color=INK)
        ax.set_ylabel(yl, fontsize=fs_lab, color=INK)

    def panel_wrapped(ax, label_cloud, fs_day=FS_DAY, fs_txt=FS_LEG, day_r=1.72,
                      note_at=(0.995, 0.80), day_pad=1.06, controls=True):
        """The foot-point histogram, drawn along the spline in the ring plane."""
        tidy(ax); ax.set_aspect("equal")
        for sp_ in ("left", "bottom"): ax.spines[sp_].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        M = hull6 @ plane.T
        c2 = curve_full @ M; k2 = K6 @ M
        rr = float(np.linalg.norm(k2, axis=1).mean())
        gi = np.clip(np.searchsorted(s_full, ctr), 0, len(s_full) - 1)
        tan = np.gradient(c2, axis=0)
        nrm_f = np.stack([tan[:, 1], -tan[:, 0]], 1)
        nrm_f /= np.maximum(np.linalg.norm(nrm_f, axis=1, keepdims=True), 1e-12)
        nrm_f *= np.sign((nrm_f * c2).sum(1))[:, None]
        nrm, ft = nrm_f[gi], c2[gi]
        H = 0.52 * rr / cum[-1].max()
        if controls:
            # the controls' own position in this plane. Indexed through the SAME 4,000-row
            # subsample panel A draws, so the two panels show the identical grey dots.
            zn = foot["negatives"]["z"]
            pos_in_neg = np.searchsorted(np.where(mneg)[0], sn)
            ax.plot(*(zn[pos_in_neg] @ M).T, ".", ms=1.0, color=PALE, alpha=0.30, zorder=2)
            # the stretch of curve the control class collapses onto, drawn just inside so
            # the spline itself stays legible
            c2_in = c2 - 0.075 * rr * nrm_f
            q = np.argsort(-hn)
            keep = q[:max(1, int(np.searchsorted(np.cumsum(hn[q]) / hn.sum(), 0.80)) + 1)]
            band = np.zeros(args.bins, bool); band[keep] = True
            for a0 in np.flatnonzero(band):
                g0 = np.searchsorted(s_full, edges[a0]); g1 = np.searchsorted(s_full, edges[a0 + 1])
                ax.plot(c2_in[g0:g1 + 1, 0], c2_in[g0:g1 + 1, 1], "-", color=MID, lw=2.6,
                        solid_capstyle="butt", alpha=0.9, zorder=8)
        for k in range(7):
            inner = ft + (H * cum[k])[:, None] * nrm
            outer = ft + (H * cum[k + 1])[:, None] * nrm
            ax.add_patch(Polygon(np.vstack([inner, outer[::-1]]), closed=True,
                                 facecolor=DAY_C[k], edgecolor="none", zorder=5))
        ax.plot(c2[:, 0], c2[:, 1], "-", color="white", lw=2.4, zorder=6)
        ax.plot(c2[:, 0], c2[:, 1], "-", color=INK, lw=0.9, zorder=7)
        for k in range(7):
            ax.plot(k2[k, 0], k2[k, 1], "o", ms=5.0, color=DAY_C[k], markeredgecolor=INK,
                    markeredgewidth=0.7, zorder=9)
            ax.annotate(DAYS[k][:3], (k2[k, 0] * day_r, k2[k, 1] * day_r), fontsize=fs_day,
                        color=INK, ha="center", va="center", zorder=10, fontweight="bold")
        if controls and label_cloud:
            # Centred on the ring's own width AT THE LABEL'S HEIGHT, not on the cloud
            # centroid and not on the origin. The ring is not a circle: the cloud sits
            # right of centre, so centring on it pushed the label into the Saturday
            # band, and centring on the origin pushed it onto the curve at Wednesday.
            cen = (zn @ M).mean(0)
            y_lab = cen[1] - 0.30 * rr
            near = np.abs(c2[:, 1] - (y_lab - 0.10 * rr)) < 0.14 * rr
            x_mid = 0.5 * (c2[near, 0].min() + c2[near, 0].max()) if near.any() else 0.0
            # Broken after the hyphen, not after "word": the ring's interior at this
            # height is 0.95in and "Capitalised-word" is 0.94in of it.
            ax.text(x_mid, y_lab, "Capitalised-\nword controls", fontsize=fs_txt,
                    color=MID, ha="center", va="top", zorder=11)
        if controls:
            smid = c2_in[np.searchsorted(s_full, float(np.median(ctr[band])))]
            ax.annotate("80% of controls\nland here", xy=(smid[0], smid[1]),
                        xytext=note_at, textcoords="axes fraction", fontsize=fs_txt,
                        color=MID, ha="right", va="top", zorder=11,
                        arrowprops=dict(arrowstyle="-", color=MID, lw=0.6, shrinkA=1.0, shrinkB=2.0))
        square(ax, np.vstack([ft + (H * cum[-1])[:, None] * nrm, k2 * day_r * day_pad, c2]),
               grow=1.03, dy=0.02)

    def panel_violin(ax, widths=0.74, bar=0.30, jit=0.055, fs_tick=FS_TICK - 1.2,
                     fs_num=FS_LEG, fs_lab=FS_LAB, labels=None, stagger=False,
                     num_above=False, nums=True,
                     ylab="Share of energy inside the\n6-D weekday subspace"):
        tidy(ax)
        CL = list(zip(labels or ["Weekday\nmentions", "Months,\nyesterday,\ntomorrow",
                                 "Other\ncapitalised\nwords", "Random\ntokens"],
                      [ps, cls == "near_miss", mneg, cls == "floor"]))
        for i, (nm, m) in enumerate(CL):
            v = frac[m]
            pv = ax.violinplot([v], positions=[i], widths=widths, showextrema=False,
                               showmedians=False)
            for b in pv["bodies"]:
                b.set_facecolor(MID); b.set_alpha(0.30); b.set_edgecolor(MID); b.set_linewidth(0.7)
            sub = rng.choice(v, min(350, len(v)), replace=False)
            ax.plot(i + rng.normal(0, jit, len(sub)), sub, ".", ms=1.3, color=MID,
                    alpha=0.20, zorder=2)
            md = np.median(v)
            ax.plot([i - bar, i + bar], [md] * 2, color=MID, lw=2.0, zorder=4)
            # Right of the bar, as in the source panel, unless the panel is too narrow
            # for that: the number then runs into the NEXT violin. Centred above the bar
            # it has the whole category slot to itself instead.
            if nums:
                ax.text(i if num_above else i + bar, md + 0.012, f"{100*md:.2f}%",
                        fontsize=fs_num, color=MID, zorder=6, va="bottom",
                        ha="center" if num_above else "left",
                        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                                  alpha=0.8))
        # STAGGERED into two rows when asked. Each label then has two category slots to
        # itself rather than one, which is worth about 1.6x in point size -- the whole
        # reason this panel can be read at all at 1.8in wide. The offset has to clear the
        # FULL height of the upper row, not one line, or the upper labels' last line and
        # the lower labels' first line land on top of each other.
        names = [n for n, _ in CL]
        if stagger:
            top = max(names[i].count("\n") + 1 for i in range(0, len(names), 2))
            names = [n if i % 2 == 0 else "\n" * top + n for i, n in enumerate(names)]
        ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=fs_tick)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        shown = np.zeros(N, bool)
        for _, m in CL: shown |= m
        ax.set_ylim(0, 1.03 * frac[shown].max()); ax.set_xlim(-0.6, 3.9)
        ax.set_ylabel(ylab, fontsize=fs_lab, color=INK)

    # The swap panels are painters like the two above, not inline code, because the 2x2
    # main cut and the appendix plate both draw them. Same reason polar_disc.py is a
    # module: two copies of one renderer can drift apart without either figure erroring.
    _LS = [(0, (7, 2)), "--", "-.", ":"]
    _GREY = ["#16181d", "#4a5160", "#79808f", "#a6acb8"]

    def panel_swap_rings(ax, fs_leg=FS_LEG, ms=4.4, lw=1.2, legend_loc="upper right"):
        """One ring per prompt template, from the 7-way weekday swap: context moves the
        ring bodily. The red curve is the mean over families."""
        tidy(ax); ax.set_aspect("equal")
        for j, i in enumerate(picks):
            Kp = Dm[i] @ plane.T + offs[i]; c = W @ Kp
            ax.plot(c[:, 0], c[:, 1], linestyle=_LS[j], color=_GREY[j], lw=lw, zorder=4,
                    label=f"Template {j+1}")
            for k in range(7):
                ax.plot(Kp[k, 0], Kp[k, 1], "o", ms=ms, color=DAY_C[k],
                        markeredgecolor=_GREY[j], markeredgewidth=0.6, zorder=5)
        ax.plot(tc[:, 0], tc[:, 1], "-", color=C_SWAP, lw=1.9, alpha=0.9, zorder=3,
                label="Mean")
        ax.legend(frameon=False, fontsize=fs_leg, loc=legend_loc, handletextpad=0.5,
                  borderaxespad=0.1, labelspacing=0.28)

    def panel_demeaned(ax, fs_day=FS_DAY, ms_pt=1.6, ms_knot=5.5, day_r=1.42):
        """The same projections with each family's own mean removed. The seven clumps
        come back on top of each other, so the context translated the ring without
        rotating it."""
        tidy(ax); ax.set_aspect("equal")
        for k in range(7):
            ax.plot(P[:, k, 0], P[:, k, 1], "o", ms=ms_pt, color=DAY_C[k], alpha=0.5,
                    markeredgecolor="none", zorder=3)
        ax.plot(tc[:, 0], tc[:, 1], "-", color="white", lw=2.6, zorder=5)
        ax.plot(tc[:, 0], tc[:, 1], "-", color=C_SWAP, lw=1.6, zorder=6)
        for k in range(7):
            ax.plot(KT[k, 0], KT[k, 1], "o", ms=ms_knot, color=DAY_C[k],
                    markeredgecolor=INK, markeredgewidth=0.7, zorder=7)
            ax.annotate(DAYS[k][:3], (KT[k, 0] * day_r, KT[k, 1] * day_r), fontsize=fs_day,
                        color=INK, ha="center", va="center", zorder=8, fontweight="bold")

    def swap_frame():
        """The frame the two swap panels share. The exemplars in panel_swap_rings are
        picked for maximal displacement and so set it, which zooms panel_demeaned out --
        that relative scale is the comparison, and giving each panel its own limits
        would destroy it."""
        return np.concatenate([np.r_[tc, KT * 1.5], P.reshape(-1, 2)] +
                              [W @ (Dm[i] @ plane.T + offs[i]) for i in picks])

    # ======================================================================
    # arc_occupancy_main -- ONE ROW at A4 text width
    # ======================================================================
    # Three panels across 6.9in is the constraint every other choice here follows from.
    # C is the one that can give: nothing in it is read off the horizontal, so it takes
    # 0.88x the square panels' width and hands the difference to A and B, which are
    # showing structure. What C pays for that is its tick type, down to 5.8pt, since
    # four wrapped class labels over 1.8in leave 0.44in each and "capitalised" is the
    # longest line. B is weekday foot points ALONE: the control cloud and the arc it
    # collapses onto were competing with the seven clumps that are the panel's whole
    # claim, and both facts are carried elsewhere -- the cloud by A, the arc by the
    # appendix strip. With B no longer naming the grey dots, A takes the legend back.
    # 3.10in tall rather than 2.80: the extra 0.30in is the staggered label block in C.
    # Width is the constraint from the brief, height is free, and A and B stay exactly
    # the size they were -- they are limited by their column width, not by the row.
    # Two panels or three. A and B keep the width they have in the three-panel row --
    # they are limited by their column, not by the row -- so dropping C narrows the
    # figure rather than growing them, and the type sizes below carry over unchanged.
    # ---------- the 2x2 cut ------------------------------------------------
    # Four square panels, so a 1x4 row at text width would put each at 1.6in -- narrower
    # than the three-panel row's squares, and A and B are exactly the panels that stop
    # being readable when they shrink. 2x2 gives each 3.0in instead, and it pairs the
    # panels by what they claim: the top row is where weekday windows fall, the bottom
    # row is what the 7-way swap does to that. C and D keep the shared frame they had as
    # the appendix's B and C -- see swap_frame().
    if args.panels == "abcd":
        fig = plt.figure(figsize=(6.9, 7.30))
        # hspace has to clear the top row's x label AND the bottom row's letter+title,
        # which is most of a line each: at 0.20 the "C" title lands on top of A's "PC1".
        gs = fig.add_gridspec(2, 2, left=0.075, right=0.988, top=0.945, bottom=0.055,
                              wspace=0.16, hspace=0.34)
        axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        # The type sizes are the three-panel row's, not the two-panel row's: at 3.0in
        # these panels are wider than either, so nothing needs to shrink to fit.
        panel_ring(axA, legend=True, legend_loc="upper right", fs_day=8.2, fs_lab=10.0,
                   fs_leg=9.0, ms_leg=5.5)
        square(axA, np.vstack([xy[ps], ck * 1.6]))
        panel_wrapped(axB, label_cloud=False, controls=False, fs_day=8.2, fs_txt=8.5,
                      day_r=1.62, day_pad=1.16)
        panel_swap_rings(axC, fs_leg=8.0)
        panel_demeaned(axD, fs_day=8.2)
        ALL = swap_frame()
        for a in (axC, axD):
            square(a, ALL, grow=1.20)
            a.set_xlabel(xl, fontsize=10.0, color=INK)
            a.set_ylabel(yl, fontsize=10.0, color=INK)
        # 12.0pt, above the 10.0pt axis labels. The 2x2's panels are 3.0in wide against
        # the one-row cut's 2.03in, so the titles have room the row never had -- the
        # longest of them, D's, sets the ceiling at about 2.8in of the 3.0.
        letter_titles(fig, [("A", "FineWeb prompts in the ring plane", axA),
                            ("B", "Where they land on the ring", axB),
                            ("C", "Prompt context translates the ring", axC),
                            ("D", "Mean-subtracted prompt projections", axD)],
                      fs_title=12.0, fs_let=13.0)
        save(fig, "arc_occupancy_main_abcd")
        print("main: 2x2 cut, four panels")
        plt.close(fig)

    # The one-row cuts, kept whole. --verify renders 'abc' for its pinned md5, so
    # this path has to stay reachable; under --panels abcd it is simply not one of
    # the outputs and the 2x2 above is the main-text plate.
    if args.panels != "abcd":
        abc = args.panels == "abc"
        fig = plt.figure(figsize=(7.4 if abc else 4.55, 3.10))
        if abc:
            gs = fig.add_gridspec(1, 4, left=0.060, right=0.995, top=0.895, bottom=0.230,
                                  wspace=0.19, width_ratios=[1.0, 1.0, 0.10, 0.88])
            axC = fig.add_subplot(gs[0, 3])      # column 2 is an empty spacer: C carries a
            #                                      two-line y label and two gaps' worth of
            #                                      slack is what keeps it off B
        else:
            gs = fig.add_gridspec(1, 2, left=0.098, right=0.995, top=0.895, bottom=0.230,
                                  wspace=0.19, width_ratios=[1.0, 1.0])
            axC = None
        axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        # 9.0pt legend, above the 8.0pt ticks: it is the only key in the figure and the one
        # thing that says what the grey is, so it should not be the smallest type on the page.
        # 10.0, matching the axis labels, is the ceiling -- the string is then wider than the
        # 2.03in panel. The swatch grows with the type so it stays a dot rather than a speck.
        panel_ring(axA, legend=True, legend_loc="upper right", fs_day=8.2, fs_lab=10.0,
                   fs_leg=9.0, ms_leg=5.5)
        square(axA, np.vstack([xy[ps], ck * 1.6]))
        panel_wrapped(axB, label_cloud=False, controls=False, fs_day=8.2, fs_txt=8.5,
                      day_r=1.62, day_pad=1.16)   # room for the bigger day labels in the frame
        # Class labels wrapped to <=8 characters a line, not the 11 the source panel uses:
        # 1.8in over four categories is 0.45in each, and "capitalised" needs more than that
        # at any size worth setting. The full class definitions are caption material.
        # Only the EVEN labels drive the stagger offset, so the odd ones are free to run to
        # three lines -- which buys back the full "yesterday, tomorrow" the single-row
        # version had to abbreviate.
        # No printed medians: they are reported in the text, and the four bars are far
        # enough apart on this axis to be read off it. Printed to stdout instead.
        if axC is not None:
            panel_violin(axC, widths=0.62, bar=0.27, jit=0.045, fs_tick=8.5, nums=False,
                         stagger=True,
                         labels=["Weekday\nmentions", "Months,\nyesterday,\ntomorrow",
                                 "Capit. word\ncontrols", "Random\ntokens"],
                         # 10pt, matching A's axis labels: one axis-label size per figure
                         ylab="Share inside the\n6-D weekday subspace", fs_lab=10.0)
        # Titles are cut to fit the column they sit over: at 8.5pt the 2x2's originals run
        # 1.9in and 2.3in, which is the whole panel. The full versions are caption material.
        head = [("A", "Windows in the ring plane", axA),
                ("B", "Where they land on the ring", axB)]
        dx = [0.0, -0.020]
        if axC is not None:
            head.append(("C", "Energy in the subspace", axC))
            dx.append(0.034)
        vcenter(fig, move=(axA, axB), ref=tuple(a for a in (axA, axB, axC) if a is not None))
        fs = fit_titles(fig, head, dx=dx, fs_max=10.5)
        print(f"main: titles set at {fs:.1f}pt")
        save(fig, "arc_occupancy_main_ab" if axC is None else "arc_occupancy_main")

    if args.appendix:
        # ======================================================================
        # arc_occupancy_appendix
        # ======================================================================
        # TWO gridspecs, not one: gridspec hspace is uniform, and the gap the strip needs
        # below it (its tick labels, its axis label, then B and C's letters and titles) is
        # several times the gap the histogram needs above it.
        fig = plt.figure(figsize=(7.4, 7.2))
        gs_t = fig.add_gridspec(2, 1, left=0.088, right=0.985, top=0.955, bottom=0.600,
                                hspace=0.10, height_ratios=[1.0, 0.30])
        gs_b = fig.add_gridspec(1, 2, left=0.088, right=0.985, top=0.485, bottom=0.068,
                                wspace=0.22)
        axA = fig.add_subplot(gs_t[0, 0]); axN = fig.add_subplot(gs_t[1, 0], sharex=axA)
        axB, axC = fig.add_subplot(gs_b[0, 0]), fig.add_subplot(gs_b[0, 1])

        tidy(axA)
        for k in range(7):
            axA.bar(ctr, hp_day[k], bottom=cum[k], width=wid, color=DAY_C[k], linewidth=0, zorder=3)
        for k in range(7):
            axA.axvline(s_knot[k], color=MID, lw=0.5, ls=(0, (2, 2)), zorder=1)
        axA.set_xlim(0, L); axA.set_ylim(0, 1.24 * cum[-1].max())
        axA.tick_params(labelbottom=False)
        axA.set_ylabel("Density of foot points", fontsize=FS_LAB, color=INK)
        axA.text(0.5, 0.995, "Weekday mentions, coloured by the day the window ends in",
                 transform=axA.transAxes, ha="center", va="top", fontsize=FS_LEG, color=INK,
                 bbox=dict(boxstyle="square,pad=0.20", fc="white", ec="none"))

        tidy(axN)
        for k in range(7):
            axN.axvline(s_knot[k], color=MID, lw=0.5, ls=(0, (2, 2)), zorder=1)
        axN.fill_between(ctr, 0, hn, step="mid", color=PALE, zorder=2)
        axN.step(np.r_[edges[0], ctr, edges[-1]], np.r_[hn[0], hn, hn[-1]], where="mid",
                 color=MID, lw=0.8, zorder=3)
        axN.set_ylim(0, 1.30 * hn.max()); axN.set_yticks([])
        axN.spines["left"].set_visible(False)
        # Monday is labelled at BOTH ends: the axis is a closed loop cut open at the Monday
        # knot, so Monday's single mode is split across the two edges rather than bimodal.
        axN.set_xticks(np.r_[s_knot, L])
        axN.set_xticklabels([d[:3] for d in DAYS] + ["Mon"], fontsize=FS_DAY)
        axN.set_xlabel("Arc length along the spline, 6-D", fontsize=FS_LAB, color=INK)
        axN.text(0.015, 0.92, f"Other capitalised words — own scale, peaks at "
                              f"{hn.max()/cum[-1].max():.0f}x the tallest bar above",
                 transform=axN.transAxes, ha="left", va="top", fontsize=FS_LEG, color=MID,
                 bbox=dict(boxstyle="square,pad=0.20", fc="white", ec="none"))

        panel_swap_rings(axB)
        panel_demeaned(axC)
        # B and C sit next to each other again, as they did in the 2x2, so they share a
        # frame: C's spread stays readable against B's displacement, and two equal-aspect
        # boxes of unequal limits would open a gap between them. B's exemplars are chosen
        # for maximal displacement and therefore set the frame, which zooms C out.
        ALL = swap_frame()
        for a in (axB, axC):
            square(a, ALL, grow=1.20)
            a.set_xlabel(xl, fontsize=FS_LAB, color=INK); a.set_ylabel(yl, fontsize=FS_LAB, color=INK)

        letter_titles(fig, [("A", "Foot points along the spline", axA),
                            ("B", "Weekday ring is translated by prompt template", axB),
                            ("C", "De-meaned prompt projections", axC)])
        save(fig, "arc_occupancy_appendix")

    # ---------- everything the captions need, off one run -------------------
    print(f"\nplane: PC1 = {100*pc_var[0]:.1f}% of centroid variance, PC2 = {100*pc_var[1]:.1f}%")
    print(f"spline length {L:.2f} in 6-D; knot spacing "
          + " ".join(f"{d:.2f}" for d in np.diff(np.r_[s_knot, L])))
    print(f"\n{'population':<24}{'n':>7}{'arc-int':>9}{'arc-int':>9}{'med|s-knot|':>13}"
          f"    unif: 1/3, 1/3, {L/28:.2f}")
    print(f"{'':<24}{'':>7}{'spline':>9}{'chords':>9}")
    for name in ("positives", "negatives"):
        r = foot[name]
        for tag, m in (("all", np.ones(len(r["s"]), bool)), ("gated", r["rad"] >= gate)):
            p_, _j = arc_frac(r["s"][m], s_knot, L)
            dk = np.abs(r["s"][m][:, None] - s_knot[None, :]); dk = np.minimum(dk, L - dk).min(1)
            print(f"{name + ', ' + tag:<24}{int(m.sum()):>7}"
                  f"{float(((p_ > 1/3) & (p_ < 2/3)).mean()):>9.3f}"
                  f"{float(((r['chordf'][m] > 1/3) & (r['chordf'][m] < 2/3)).mean()):>9.3f}"
                  f"{float(np.median(dk)):>13.3f}")
    nb = np.argmin(np.minimum(np.abs(foot["positives"]["s"][:, None] - s_knot[None, :]),
                              L - np.abs(foot["positives"]["s"][:, None] - s_knot[None, :])), 1)
    print(f"positives at their own day's knot: {float((nb == pday).mean()):.3f}")
    ja = arc_frac(foot["negatives"]["s"], s_knot, L)[1]
    big = np.bincount(ja, minlength=7).argmax()
    print(f"control collapse: {float((ja == big).mean()):.3f} in the "
          f"{DAYS[big][:3]}->{DAYS[(big+1)%7][:3]} arc alone; strip peaks at "
          f"{hn.max()/cum[-1].max():.1f}x the tallest weekday bar")
    print("main panel C medians, for the text: " + ", ".join(
        f"{nm} {100*np.median(frac[m]):.2f}%" for nm, m in
        (("weekday", ps), ("near_miss", cls == "near_miss"), ("matched_neg", mneg),
         ("floor", cls == "floor"))))
    print("panel C templates:")
    for j, i in enumerate(picks):
        print(f"  {j+1}. {prompt(i)!r}")
    return 0


if __name__ == "__main__":
    rc = main(); sys.stdout.flush(); os._exit(rc)
