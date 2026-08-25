#!/usr/bin/env python
"""Figure 2: time of day is carried orthogonally to the weekday ring.

Writes figures/timeofday_with_steering.{pdf,png}. The export arguments are not the
script's own defaults and the figure does not error without them, so
repro_fig2_timeofday_with_steering.sh is the authority on them.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import CubicSpline

plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})

INK, MID, PALE = "#16181d", "#5b6270", "#aeb4c0"
INK_D, MID_D, PALE_D = "#f2f4f8", "#aeb4c0", "#6b7280"
EARLY, LATE = "#3b5f9e", "#c8642a"
# Panel C's third family: the modifier's own in-plane PUSH, which is a translation
# inside the ring plane rather than a rotation of it, so it cannot share the ring
# colour. 4.9:1 on white, and far enough from both EARLY and LATE to read as a
# separate thing rather than a shade of either.
INPLANE = "#2f7f7a"
# Panel C in greyscale: three values, all above the 3:1 floor for a graphic mark on white,
# and each separated from the next by enough luminance to be told apart at a distance.
#   #16181d  17.9:1   the ring rotations -- the subject of the panel
#   #4a5160   7.4:1   the off-plane ladder
#   #858b99   3.4:1   the in-plane arm
# Adjacent pairs sit at 2.4:1 and 2.2:1 against each other. PALE (#aeb4c0, 2.1:1 on white)
# and PLACEBO_W (#747a88, 4.1:1) were both tried and rejected: PALE dies on a projector,
# and PLACEBO_W is only 1.5:1 from MID, which is not a difference anyone reads as a
# difference.
MONO_RING, MONO_LAD, MONO_IN = "#16181d", "#4a5160", "#858b99"
# Panel C's matched pairs, in ASCENDING ||Delta|| -- 13.1, 17.4, 48.0 -- so the x axis is
# ordered by the one quantity the two families are held equal on, rather than by the order
# the arms happen to appear in the run. Each entry is (in-plane arm, its matched off-plane
# arm, tick label). The label names the PART of the vector the edit uses: all three come
# from the in-plane component of mean(late) - mean(early), but a rotation takes only its
# ANGLE and keeps the live radius, while the step applies the whole thing. Calling both
# "the early-to-late shift" made them look like one edit measured twice.
# Labels are minimal, and minimal HERE means narrow rather than short: three of them share
# panel C's width, so the widest LINE is the constraint and vertical space is free. Wrapping
# a 28-character phrase onto two 16-character lines buys 38% of the column back; saying the
# same thing in one shorter line does not. The edit sizes are in the caption -- they are the
# quantity held equal WITHIN a pair, so they annotate the design, not the group.
#
# The quotation marks are gone from late-early. They marked the two words as the literal
# modifiers inserted into the prompts, which the caption says anyway, and at 12 characters
# against every other line's 10 that one line was capping the size of all three labels.
# Two glyphs were costing 20% of the type size.
#
# Initial capital on the leading word of every label and legend row, matching what
# --cap-labels does to the axis labels. Tick labels are not axis labels, so --cap-labels
# never reached these and the panel was mixing two conventions.
PAIR_ARMS = [
    ("clamp_ring_own", "clamp_off_m_ring_own", "Late−early\nrotation"),
    ("clamp_in_a1", "clamp_off_m_in_a1", "Late−early\nstep"),
    # Three lines, not two. The group labels are capped by the widest LINE against one
    # group's share of the column, so "rotation halfway" -- 16 characters against the
    # 12 of "late-early" -- was holding all three labels a third smaller than the
    # column could carry. Height costs nothing here.
    ("clamp_ring_half", "clamp_off_m_ring_half", "Rotation\nhalfway to\nnext day"),
]
# Two families, two values, per the panel's one comparison. Black goes to the IN-PLANE
# edits, which are what the panel title is about -- steering the ring is the manipulation
# under test -- and the off-plane edits are the reference it is measured against, so they
# take the lighter value. 17.9:1 and 3.4:1 on white, 2.4:1 against each other. The third
# grey the ladder used is free now that the ladder is gone.
PAIR_IN, PAIR_OFF = "#16181d", "#858b99"
# The placebo family on a white slide: a step lighter than MID so the null family reads as
# the quiet one, but nowhere near PALE, which is 1.9:1 on white and dies on a projector.
# 4.3:1 -- comfortably over the 3:1 floor for a graphic mark.
PLACEBO_W = "#747a88"
# The weekday_manifold palette, verbatim: Okabe-Ito in spectral order, Mon..Sun. See the
# longer note in figure_corpus_exclusivity.py. Only figures/carrier2_daytime_talk_white_rawpca
# has been re-rendered under it -- the white, dark, light and slide exports on disk still
# carry the previous palette. See regen_carrier2_daytime_talk.sh before re-running anything.
DAY_C = ["#CC79A7", "#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------------------------------------------------------- the kept modifiers
# Each list keeps the family's MOST NEGATIVE and MOST POSITIVE row, so the plotted
# spread matches the full-figure spread and the trim cannot quietly help the result.
# Values in the comments are from the L28 capture, in degrees of in-plane shift.

# A 4-hourly ladder over the whole cycle. Holds the clock family's true maximum
# (eleven pm, +4.03) and, at seven am (-3.55), a row within 0.09 of its true minimum
# (nine am, -3.64) -- inside the standard error, and worth it for an even ladder.
KEEP_CLOCK = (3, 7, 11, 15, 19, 23)
# All six. They are the family a listener can hold in their head, they mirror the clock
# trend independently, and late night (+6.34) is the largest shift anywhere in the figure.
KEEP_WORDS = ("early morning", "late morning", "mid afternoon",
              "early evening", "late evening", "late night")
# THREE placebos, drawn as one family. The paper figure splits them into "very X" and a
# both-tokens-varied set, which existed to prove the shared "very" was not itself
# producing the displacement -- a control that mattered when the WORD modifiers were
# "very early"/"very late" and shared that token. They no longer do, so the split is
# answering a question this figure no longer asks, and two grey families cost a marker
# shape and a legend to say nothing.
#
# The three still span the full combined placebo range, -2.06 to +1.91, so the null band
# is no narrower than the 22-placebo version: notably dusty is the most negative of all
# 22 and very rainy the most positive. They also happen to come from both original
# sub-families (2 "very", 1 varied), so the intensifier is still not held constant.
KEEP_PLACEBO = ("very rainy",      # +1.91, most positive of all 22 placebos
                "very sunny",      # -0.03, essentially zero
                "notably dusty")   # -2.06, most negative of all 22 placebos


def auto_placebos(dth, lab, fam, day):
    """KEEP_PLACEBO's stated rule, applied to whatever placebo band the npz holds."""
    keys = list(dict.fromkeys(lab[np.char.startswith(fam.astype(str), "placebo")]))
    means = {}
    for k in keys:
        per_day = np.array([dth[(lab == k) & (day == d)].mean() for d in range(7)])
        means[k] = float(per_day.mean())
    hi = max(keys, key=lambda k: means[k])
    lo = min(keys, key=lambda k: means[k])
    mid = min((k for k in keys if k not in (hi, lo)), key=lambda k: abs(means[k]))
    return (hi, mid, lo)

WORD_HOURS = {"early morning": 7.0, "late morning": 10.0, "mid afternoon": 14.0,
              "early evening": 18.0, "late evening": 21.0, "late night": 23.0}


def clock_word(h):
    """24-hour index -> spoken 12-hour phrase. Same mapping as the paper figure."""
    names = ["twelve", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven"]
    return f"{names[h % 12]} {'am' if h < 12 else 'pm'}"


def plane_of(C, k=2):
    """Top-k right singular vectors of the centred centroids: the weekday plane."""
    _, _, V = np.linalg.svd(C - C.mean(0), full_matrices=False)
    return V[:k]


def _between_ss(x, g):
    """One-way between-group sum of squares of scores x under grouping g."""
    m = x.mean()
    return float(sum(((x[g == u].mean() - m) ** 2) * (g == u).sum() for u in np.unique(g)))


def ring_handedness(plane, Cday, mu):
    """Does calendar-forward run WITH increasing in-plane angle, or against it?"""
    xyC = (Cday - mu) @ plane.T
    th = np.degrees(np.arctan2(xyC[:, 1], xyC[:, 0]))
    step = (th[(np.arange(7) + 1) % 7] - th + 180.0) % 360.0 - 180.0
    if (step > 0).all():
        return +1
    if (step < 0).all():
        return -1
    return 0


def orient_plane(plane, Cday, mu, mode):
    """Make calendar-forward mean increasing angle, or refuse to draw a mislabelled panel."""
    if mode == "off":
        return plane
    hand = ring_handedness(plane, Cday, mu)
    if hand == 0:
        print("[ring] WARNING: the 7 day centroids are NOT in cyclic order in this "
              "plane. 'previous day' and 'next day' are not defined here and panel A's "
              "bands mean nothing. Pick another layer or another basis.")
        return plane
    if hand == +1:
        print("[ring] calendar-forward is anticlockwise: negative shift = previous day, "
              "as panel A's bands assume.")
        return plane
    if mode == "flip":
        print("[ring] calendar-forward was CLOCKWISE in this plane; flipping the second "
              "plane axis so negative shift = previous day. This mirrors panel B too -- "
              "it is the same basis, drawn from the other side.")
        return np.stack([plane[0], -plane[1]])
    raise SystemExit(
        "ABORT: calendar-forward runs CLOCKWISE in this plane, so a NEGATIVE in-plane\n"
        "shift points at the NEXT day -- the opposite of what panel A's bands say. The\n"
        "plate would be a mirror image with both band labels wrong.\n\n"
        "  --orient-ring flip   negate one plane axis and draw it the right way round\n"
        "  --orient-ring off    draw it anyway (only if you are relabelling by hand)\n\n"
        "This is a property of an arbitrary SVD sign, not of the model: the same 364\n"
        "prompts come out +1 on Llama-3.1-8B L28 and -1 on Mistral-7B-v0.1 L30.")


def raw_pca_axes(A, day, hour, cl, fit=None, n_scan=30):
    """UNSUPERVISED basis: PCA the prompt set, then only LABEL the components."""
    fit = np.ones(len(A), bool) if fit is None else fit
    mu = A[fit].mean(0)
    _, S, V = np.linalg.svd(A[fit] - mu, full_matrices=False)
    n_scan = min(n_scan, V.shape[0])
    Z = (A - mu) @ V[:n_scan].T
    varf = S[:n_scan] ** 2 / float((S ** 2).sum())

    d_ss = np.array([_between_ss(Z[:, k], day) for k in range(n_scan)])
    h_ss = np.array([_between_ss(Z[cl, k], hour[cl]) for k in range(n_scan)])
    d_tot = ((Z - Z.mean(0)) ** 2).sum(0)
    h_tot = ((Z[cl] - Z[cl].mean(0)) ** 2).sum(0)
    d_eta, h_eta = d_ss / d_tot, h_ss / h_tot
    # Absolute scores are reported as a share of the total variance of the rows each was
    # computed over, so the two columns are comparable to the var% column beside them.
    d_abs = d_ss / ((A - mu) ** 2).sum()
    h_abs = h_ss / ((A[cl] - A[cl].mean(0)) ** 2).sum()

    p = np.argsort(-d_abs)[:2]
    e3i = int(next(k for k in np.argsort(-h_abs) if k not in set(p.tolist())))

    table = dict(varf=varf, d_abs=d_abs, h_abs=h_abs, d_eta=d_eta, h_eta=h_eta,
                 plane_idx=tuple(int(i) for i in p), e3_idx=e3i, n_fit=int(fit.sum()))
    return V[list(p)], V[e3i], table, mu


def matched_pairs(rows_csv, boot_n=2000, seed=0):
    """Panel C's edit-size-matched pairs, from steer_clamp_matched_pairs.py's rows.csv."""
    import csv

    by, cells = {}, {}
    for r in csv.DictReader(open(rows_csv)):
        by.setdefault(r["arm"], {})[(r["carrier"], r["day"])] = r
        cells[(r["carrier"], r["day"])] = None
    missing = [a for p in PAIR_ARMS for a in p[:2] if a not in by]
    if missing:
        raise SystemExit(f"{rows_csv} has no {missing} -- panel C's matched pairs need a "
                         f"steer_clamp_matched_pairs.py run, not a gated one. Use --steer "
                         f"for the dose-ladder panel.")
    keys = sorted(cells)
    rng = np.random.default_rng(seed)

    def col(a, k):
        return np.array([float(by[a][q][k]) for q in keys])

    def agg(v, carr):
        # bootstrap the 12 carrier means, not the 84 cells: a carrier is the independent unit
        cm = np.array([v[carr == c].mean() for c in np.unique(carr)])
        b = np.array([rng.choice(cm, len(cm), True).mean() for _ in range(boot_n)])
        return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    carr = np.array([q[0] for q in keys])
    out = []
    for in_arm, off_arm, label in PAIR_ARMS:
        e = {}
        for side, a in (("in", in_arm), ("off", off_arm)):
            v, d = col(a, "d_cm"), col(a, "displacement")
            m, lo, hi = agg(v, carr)
            e[side] = dict(v=v, d=d, m=m, lo=lo, hi=hi)
        e["label"] = label
        e["dmean"] = float(0.5 * (e["in"]["d"].mean() + e["off"]["d"].mean()))
        # the worst per-cell mismatch between the two edits of a pair, for the caption
        e["mismatch"] = float(np.abs(e["off"]["d"] / e["in"]["d"] - 1).max())
        e["n_up"] = int((e["off"]["v"] > e["in"]["v"]).sum())
        out.append(e)
    od = col("clamp_ring_own", "own_deg")
    return dict(pairs=out, n=len(keys), n_carriers=int(len(np.unique(carr))),
                own_deg=float(np.mean(od)))


def steering_curve(rows_csv, boot_n=2000, seed=0):
    """The clamp dose-response, for panel C. Reads the gated clamp run's rows.csv only."""
    import csv

    by = {}
    for r in csv.DictReader(open(rows_csv)):
        by.setdefault(r["arm"], []).append(r)
    rng = np.random.default_rng(seed)

    def agg(a):
        car = {}
        for r in by[a]:
            car.setdefault(r["carrier"], []).append(float(r["d_cm"]))
        v = np.array([np.mean(x) for x in car.values()])
        b = np.array([rng.choice(v, len(v), True).mean() for _ in range(boot_n)])
        # prompt arms have no hook and therefore no displacement column; nan, not 0
        dd = [float(r["displacement"]) for r in by[a] if r["displacement"]]
        return dict(m=float(v.mean()), lo=float(np.percentile(b, 2.5)),
                    hi=float(np.percentile(b, 97.5)),
                    d=float(np.mean(dd)) if dd else float("nan"))

    lad = sorted([a for a in by if a.startswith("clamp_off_a")] + ["clamp_off_full"],
                 key=lambda a: agg(a)["d"])
    L = [agg(a) for a in lad]
    # Three kinds, because they are three different operations and only one of them is a
    # rotation. "inplane" holds the modifier's own in-plane displacement at 1x -- the 2x and
    # 3x arms are in the run and behave identically, and 1x is the one that needs no
    # explaining: it is exactly the push the word itself gives, kept inside the plane.
    # Positive rotations only. The -25.71 deg arm is in the run and is what the
    # antisymmetry test turns on, but this panel is a matched-size comparison and a second
    # point at the same x carrying the opposite sign is a different argument. The identity
    # hook is dropped for the same reason: its edit size is 4.86 rather than 0 -- bf16
    # round-off against a separately captured reference, compounding over 27 layers -- and
    # explaining that costs more than the null is worth here.
    # clamp_ring_own, NOT clamp_ring_mod. The two are the same edit at two angles: _mod
    # applies a fixed 8.07 deg everywhere, a grand average inherited from earlier work and
    # not measured in this run, while _own holds every cell at exactly the rotation ITS
    # modifier induces -- mean +6.38 deg, per-cell range +2.11 to +12.29. Panel C labels
    # its points with the angle applied, so the arm plotted has to be the one whose angle
    # the label states. They agree anyway: 13.13 vs 16.09 in edit size, -0.001 vs +0.003 h.
    IN = [("clamp_ring_own", "ring"), ("clamp_ring_half", "ring"),
          ("clamp_in_a1", "inplane")]
    A = {a: dict(agg(a), kind=k) for a, k in IN}
    od = [float(r["own_deg"]) for r in by["clamp_ring_own"] if r.get("own_deg")]
    A["clamp_ring_own"]["deg"] = float(np.mean(od)) if od else float("nan")
    # the crop is set by the ring rotations, the family the panel is about
    xmax = max(A[a]["d"] for a, k in IN if k == "ring")
    lx = np.array([r["d"] for r in L])
    ly = np.array([r["m"] for r in L])
    o = np.argsort(lx)
    # the off-plane effect at an arbitrary edit size, interpolated on the ladder
    read = lambda x: float(np.interp(x, lx[o], ly[o]))
    return dict(lx=lx[o], ly=ly[o],
                llo=np.array([r["lo"] for r in L])[o],
                lhi=np.array([r["hi"] for r in L])[o],
                arms=A, xmax=float(xmax), read=read,
                anchor=agg("prompt_late"), n_cells=len(by["clean"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # experiments/capture_timeofday.py writes this. The research repo called the same
    # file carrier2_daytime_L28.npz and this default used to point at that name, which
    # does not exist here -- the two are byte-identical (md5 6680dd31ceeca5671041dbe0b1f91e19),
    # so this is a rename, not a different capture.
    ap.add_argument("--npz", default="experiments/results/timeofday_L28.npz")
    ap.add_argument("--out", default="figures/carrier2_daytime_talk.pdf")
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--basis", choices=("supervised", "raw-pca"), default="supervised",
                    help="how panel B's three axes are chosen. 'supervised' (default, and "
                         "what every existing export used) builds the plane from the SVD "
                         "of the seven day centroids and the third axis from the SVD of "
                         "the hour centroids -- the axes are constructed out of the labels. "
                         "'raw-pca' runs ONE unlabelled SVD over all 364 prompts, then uses "
                         "the labels only to pick which components to draw. NOTE this also "
                         "changes panel A, which measures its angles in the same plane; the "
                         "two panels are never allowed to use different planes")
    ap.add_argument("--pca-rows", choices=("all", "time", "clock"), default="time",
                    help="which prompts the --basis raw-pca SVD is FITTED on. Everything is "
                         "projected in either way, so panel A still plots all 22 placebos. "
                         "'time' (default) fits on the clock and word prompts and leaves the "
                         "placebos out, so they stay an out-of-sample null; 'clock' is the "
                         "same idea one step stricter. 'all' includes the placebos, which "
                         "makes the time-vs-placebo contrast the single biggest axis in the "
                         "data and tilts the weekday ring off its own plane -- see raw_pca_axes")
    ap.add_argument("--pc-scan", type=int, default=30,
                    help="how many leading components --basis raw-pca scores and prints")
    ap.add_argument("--orient-ring", choices=("check", "flip", "off"), default="check",
                    help="what to do when calendar-forward runs CLOCKWISE in the fitted "
                         "plane, which makes a negative in-plane shift point at the NEXT "
                         "day and inverts both of panel A's band labels. 'check' "
                         "(default) leaves the plane alone -- so every existing export is "
                         "byte-identical -- and aborts rather than draw the mirrored "
                         "plate; 'flip' negates one plane axis so the labels are right; "
                         "'off' is the pre-2026-08-20 behaviour, which never looked.")
    ap.add_argument("--keep-placebo", nargs="*", default=None, metavar="PHRASE",
                    help="which placebo rows panel A draws, overriding KEEP_PLACEBO. "
                         "Needed for a capture whose placebo band is not the Llama "
                         "bank's -- Mistral's tokenizer splits 'very rainy' into three "
                         "pieces, so experiments/carrier2_mistral_prompts.py substitutes "
                         "a length-matched band and the hard-coded phrases are absent. "
                         "The single word 'auto' applies KEEP_PLACEBO's own stated rule "
                         "to whatever band the npz holds: most positive shift, nearest "
                         "zero, most negative. That rule is what makes the trim unable to "
                         "flatter the result, so 'auto' is the honest port of it; hand-"
                         "picking three phrases is not, unless you check the range you "
                         "kept against the full band the way the methods file does.")
    ap.add_argument("--theme",
                    choices=("light", "dark", "both", "slide", "slide-white"),
                    default="light",
                    help="'slide' draws every piece of chrome -- text, spines, ticks, "
                         "grid, 3-D axis lines, colourbar -- in white, and drops the "
                         "backing boxes behind the weekday labels. It is for dropping "
                         "the transparent export onto a mid-toned slide, where the grey "
                         "furniture of the other themes goes illegible and a backing box "
                         "shows up as a filled rectangle. Preview it with --opaque, which "
                         "paints a stand-in slide colour (see --preview-bg). "
                         "'slide-white' is that same slide treatment with the ink polarity "
                         "flipped, for a WHITE slide: every structural choice of 'slide' is "
                         "kept -- one colour for all chrome, no backing boxes, one marker "
                         "shape, the word family all one colour -- but drawn dark on white. "
                         "It is NOT the 'light' theme, which predates those changes.")
    ap.add_argument("--opaque", action="store_true",
                    help="fill the background instead of leaving it transparent")
    ap.add_argument("--preview-bg", default="#9fc2e8",
                    help="the stand-in slide colour --theme slide paints under --opaque. "
                         "Proofing only -- it is never written into a transparent export")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier on every type size at once")
    ap.add_argument("--line-color", default=None,
                    help="colour of the hour trajectories in panel B. WARNING: the "
                         "default white is invisible against a white slide -- it is "
                         "meant for the dark theme or a mid/dark slide behind a "
                         "transparent export. Unset means white, except under "
                         "--theme slide-white, which uses MID (#5b6270): the light figure's "
                         "#c9ced8 is only 1.6:1 on white and disappears when projected")
    ap.add_argument("--steer", metavar="ROWS_CSV", default=None,
                    help="add panel C, the clamp dose-response, from a gated clamp run's "
                         "rows.csv. Off by default: without it this script produces exactly "
                         "the two-panel exports pinned in regen_carrier2_daytime_talk.sh")
    ap.add_argument("--steer-pairs", metavar="ROWS_CSV", default=None,
                    help="add panel C as edit-size-matched PAIRS -- one box per edit, the "
                         "in-plane edit beside an off-plane edit of the same ||Delta|| -- "
                         "from a steer_clamp_matched_pairs.py rows.csv. Mutually exclusive "
                         "with --steer, which draws the older dose-ladder panel from a "
                         "gated run and is what the fullstop plate still uses")
    ap.add_argument("--steer-width", type=float, default=1.05,
                    help="panel C's width relative to panel A")
    ap.add_argument("--pairs-width", type=float, default=1.75,
                    help="panel C's width relative to panel A under --steer-pairs. Wider "
                         "than --steer-width: six boxes and three three-line tick labels "
                         "need more room than one curve and three points did. At the "
                         "ladder's 1.05 the group labels overlap each other outright")
    ap.add_argument("--pairs-figsize", default="25.5x7.6",
                    help="figsize used when --steer-pairs is given and --figsize is left at "
                         "its default. Wider than --steer-figsize as well as re-weighted: "
                         "buying panel C's extra width out of the ladder canvas alone would "
                         "take it from the forest and the cube, which need what they have")
    ap.add_argument("--figsize", default="14.5x7.6")
    ap.add_argument("--steer-figsize", default="22.4x7.6",
                    help="figsize used when --steer is given and --figsize is left at its "
                         "default, so the three-panel plate is not squeezed into the "
                         "two-panel canvas")
    ap.add_argument("--c-mono", action="store_true",
                    help="draw panel C in greyscale, telling the three arms apart by marker "
                         "shape instead of hue. Colour there was decorative: the legend and "
                         "the point labels already name every arm. Applies to the --steer "
                         "ladder only; --steer-pairs has two families and is grey against "
                         "black either way")
    ap.add_argument("--pad-inches", type=float, default=None,
                    help="white left around the tight crop. The default scales with "
                         "--scale to keep mplot3d's axis labels on the canvas; short, "
                         "tucked-in PC labels do not need that much, and at --scale 2.15 "
                         "the default is 1.25 in a side. Check the export after lowering "
                         "it -- an mplot3d label clipped here cannot be recovered")
    ap.add_argument("--cb-shrink", type=float, default=0.56,
                    help="height of the 3-D panel's colourbar as a fraction of that "
                         "panel's slot. The slot is much taller than the cube inside it, "
                         "so the default reads shorter beside the cube than the number "
                         "suggests")
    ap.add_argument("--text-boost", type=float, default=1.0,
                    help="multiply the SMALL type -- tick labels, the forest's row labels "
                         "and its previous/next-day band labels, panel C's legend, and the "
                         "two axis labels -- without touching marker sizes or line widths. "
                         "--scale moves type and ink together, so pushing it far enough to "
                         "fix small labels turns the forest's markers into blobs that "
                         "overlap each other; this raises the type on its own")
    ap.add_argument("--align-titles", action="store_true",
                    help="put every panel letter and title on one baseline instead of "
                         "hanging each off its own panel's top edge. The tops are not "
                         "level: mplot3d squares the 3-D panel's box, which drops it below "
                         "the 2-D panels. Opt-in because the pinned exports carry the "
                         "per-panel placement")
    ap.add_argument("--top", type=float, default=None,
                    help="override the axes' top edge in figure coordinates")
    ap.add_argument("--bottom", type=float, default=None,
                    help="override the axes' bottom edge in figure coordinates. The default "
                         "scales with --scale to clear mplot3d's z label; short PC labels "
                         "do not need that much and the difference is panel height")
    ap.add_argument("--a-row-size", type=float, default=1.0,
                    help="multiply the forest's row labels alone, on top of --scale and "
                         "--text-boost. They are the longest strings on the plate and the "
                         "first thing a larger --scale collides")
    ap.add_argument("--a-tight-rows", action="store_true",
                    help="cut the forest's top and bottom margin from ~1.2 and 1.0 rows to "
                         "0.6 each, so the rows use the panel height instead of the "
                         "margin. Buys ~8% more pitch per row, which is ~8% more type "
                         "before the row labels start meeting each other")
    ap.add_argument("--cube-text", type=float, default=1.0,
                    help="multiply the 3-D panel's own type -- day labels, PC axis labels "
                         "and the colourbar -- by this factor on top of --scale. The cube "
                         "is drawn inside an mplot3d box much wider than itself, so it has "
                         "white to spare that the other panels do not, and matching its "
                         "type to theirs leaves it looking undersized")
    ap.add_argument("--check-overlap", action="store_true",
                    help="after drawing, measure the placed titles, panel letters and axis "
                         "labels and report any pair whose drawn boxes intersect. Use when "
                         "changing --scale or --wspace: the fitted sizes cannot see an "
                         "artist they do not measure against, and panel B's x label is "
                         "wider than its own axes")
    ap.add_argument("--wspace", type=float, default=None,
                    help="override the gap between panel columns, as a fraction of the "
                         "average column width. The three-panel default is 0.56, sized so "
                         "two long centred titles could not meet in the middle; short "
                         "titles do not need it, and the gap is the largest single block "
                         "of unused canvas on the plate. Lowering it widens every panel "
                         "but pulls the panel CENTRES together, which is what the title "
                         "fit measures against -- so it trades title size for panel size")
    ap.add_argument("--fit-titles", action="store_true",
                    help="with --centre-titles, scale the panel titles to the largest size "
                         "that still clears the neighbouring panel letter, GROWING them "
                         "into spare room as well as shrinking them out of a shortfall. "
                         "Also cuts the letter reserve from 0.021 to 0.008 of the canvas "
                         "width. Off by default because every pinned export was measured "
                         "under the shrink-only rule and would change size under this one")
    ap.add_argument("--b-balance", action="store_true",
                    help="after the titles are placed, measure the white gap between "
                         "title A and letter B and between title B and letter C, and slide "
                         "panel B so the two are equal. Applied on top of --b-shift, and "
                         "exact -- it reads the drawn text rather than estimating widths")
    ap.add_argument("--a-xlabel", default=None,
                    help="override the modifier panel's x label. The default names the "
                         "quantity in full; when the panel title already says what is "
                         "plotted, a shorter one stops it running off the plate")
    ap.add_argument("--centre-titles", action="store_true",
                    help="centre each panel title on its own axes box and place the panel "
                         "letter just left of the drawn title, instead of left-aligning "
                         "both off a per-panel anchor. Needed once a wide panel leads the "
                         "plate: its title is then wider than its column and runs into the "
                         "next panel's letter")
    ap.add_argument("--swap-ab", action="store_true",
                    help="put the 3-D panel first and the modifier forest second. --title-a "
                         "and --title-b stay POSITIONAL -- title-a is whatever is drawn "
                         "first -- and the panel letters follow the new order")
    ap.add_argument("--no-gridlines", action="store_true",
                    help="drop the background grid from panels A and C. The rows and the "
                         "points are read against the zero line and their own labels, not "
                         "off a grid, and at slide weight the faint rules are noise")
    ap.add_argument("--b-panel-shift", type=float, default=0.0,
                    help="move panel B's cube and colourbar this far LEFT, leaving its "
                         "letter and title where they are. Use it to open the gap between "
                         "the panels without dragging the headings out of alignment; "
                         "negative values move the panel right")
    ap.add_argument("--b-shift", type=float, default=0.0,
                    help="move panel B this far LEFT in figure coordinates, before the "
                         "colourbar is attached so it follows. Opens the B-to-C gap "
                         "without restyling anything")
    ap.add_argument("--b-label-pad", type=float, default=None,
                    help="place each weekday label a CONSTANT distance beyond its own "
                         "centroid, measured from the ring centre, instead of scaling its "
                         "radius by a fixed factor. The ring is not circular -- radii run "
                         "5.7 to 8.8 -- so the multiplicative default throws Sun far out "
                         "and leaves Wed sitting on the z axis")
    ap.add_argument("--b-label-pad-day", type=float, nargs=7, default=None,
                    metavar=("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"),
                    help="per-day override of --b-label-pad, Mon first. Even a constant "
                         "offset cannot satisfy every label at once: the days sit at "
                         "different radii AND at different angles to the drawn axes, so "
                         "the one that reads as crowded is not the one furthest out")
    ap.add_argument("--b-rotate-labels", action="store_true",
                    help="rotate panel B's three axis labels to lie along their own axes "
                         "as drawn, instead of sitting horizontally across them")
    ap.add_argument("--b-axis-labels", choices=("full", "pc"), default="full",
                    help="'pc' shortens panel B's axis labels to PC1/PC2/PC3, dropping "
                         "the 'raw' prefix and the (weekday)/(clock time) annotations")
    ap.add_argument("--b-labelpad", type=float, default=None,
                    help="override panel B's three axis labelpads. The z label keeps its "
                         "2 pt of extra tuck")
    ap.add_argument("--b-zoom", type=float, default=1.35,
                    help="how much of its axes panel B's cube fills. Above ~1.5 the day "
                         "labels start to clip at the panel edge")
    ap.add_argument("--elev", type=float, default=24.0)
    ap.add_argument("--azim", type=float, default=-64.0)
    ap.add_argument("--panel-labels", action=argparse.BooleanOptionalAction, default=True,
                    help="draw the A/B panel letters. On by default because the write-up "
                         "cites panels by letter; --no-panel-labels gives the original "
                         "talk cut, which dropped them")
    ap.add_argument("--title",
                    default="Time of day moves a weekday activation off the ring, "
                            "not along it",
                    help="the plate-wide title. Pass an empty string to drop it, which is "
                         "what the per-panel titles are for")
    ap.add_argument("--cap-labels", action="store_true",
                    help="capitalise the first letter of every axis label and the "
                         "colourbar label. Opt-in so the pinned exports keep their "
                         "lower-case labels")
    ap.add_argument("--subtitle", action=argparse.BooleanOptionalAction, default=True,
                    help="the model/prompt/layer line under the title. --no-subtitle "
                         "removes it; the same information is in the caption file")
    # Per-panel titles. Off unless given, so the pinned two-panel exports are unaffected.
    # A plate-wide title states ONE finding; three panels making three different claims
    # need three, and a viewer reading a panel should not have to infer which part of a
    # single sentence it belongs to.
    for L in "abc":
        ap.add_argument(f"--title-{L}", default=None,
                        help=f"title drawn above panel {L.upper()}")
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        raise SystemExit(f"{args.npz} not found. This script never captures -- run "
                         f"figure_carrier2_daytime.py first to write it.")
    z = np.load(args.npz, allow_pickle=True)
    A = z["A"].astype(np.float64)
    day, fam = z["day"], z["fam"].astype(str)
    lab, hour = z["lab"].astype(str), z["hour"]
    layer = int(z["layer"]) if "layer" in z else 28
    # The capture site is named in the caption, so it has to come from the npz rather
    # than being hardcoded: carrier2_daytime_stop_L28.npz holds the sentence-final full
    # stop, and a figure of it captioned "weekday token" would be a mislabel no reader
    # could catch. The paper cache has no `site` key, so it keeps the original wording
    # and its export stays byte-identical.
    site = str(z["site"]) if "site" in z else "weekday token"

    # ---- geometry -------------------------------------------------------------------
    # Under --basis supervised this is identical to the paper figure. Under raw-pca the
    # plane and the third axis both come from one unlabelled SVD instead; e3_raw is None
    # in the supervised case and the hour axis is derived in panel B as before.
    mu = A.mean(0)
    cl = fam == "clock"
    e3_raw, pcs = None, None
    if args.basis == "supervised":
        Cday = np.stack([A[day == d].mean(0) for d in range(7)])
        plane = plane_of(Cday)
        ax_lab = ("weekday plane 1", "weekday plane 2", "clock time plane 1")
    else:
        fit = {"all": np.ones(len(A), bool),
               "time": np.isin(fam, ["clock", "clock_midnight", "word"]),
               "clock": cl}[args.pca_rows]
        plane, e3_raw, pcs, mu = raw_pca_axes(A, day, hour, cl, fit=fit,
                                              n_scan=args.pc_scan)
        i, j = pcs["plane_idx"]
        ax_lab = (f"raw PC{i + 1}  (weekday)", f"raw PC{j + 1}  (weekday)",
                  f"raw PC{pcs['e3_idx'] + 1}  (clock time)")
        # Raw first: the full scored table, before any of it is reduced to three picks.
        print(f"\n[raw-pca] unlabelled SVD fitted on {pcs['n_fit']} of {A.shape[0]} prompts "
              f"(--pca-rows {args.pca_rows}); all {A.shape[0]} projected in; components "
              f"scored afterwards on absolute between-group variance")
        print("  PC   var%   day-var%  day-eta2   hour-var%  hour-eta2")
        for k in range(len(pcs["varf"])):
            mark = ("  <- plane" if k in pcs["plane_idx"] else
                    "  <- time" if k == pcs["e3_idx"] else "")
            print(f"  {k + 1:3d} {100 * pcs['varf'][k]:6.2f} {100 * pcs['d_abs'][k]:9.2f}"
                  f" {pcs['d_eta'][k]:9.3f} {100 * pcs['h_abs'][k]:11.2f}"
                  f" {pcs['h_eta'][k]:10.3f}{mark}")
        print(f"  (day-var%/day-eta2 over all rows; hour columns over the "
              f"{int(cl.sum())} clock rows panel B plots)")
    # Day centroids over ALL prompts, including any the SVD was not fitted on -- panel A's
    # reference angle for a placebo has to be the same day centroid every other row uses.
    Cday = np.stack([A[day == d].mean(0) for d in range(7)])
    plane = orient_plane(plane, Cday, mu, args.orient_ring)
    xy, xyC = (A - mu) @ plane.T, (Cday - mu) @ plane.T
    thC = np.degrees(np.arctan2(xyC[:, 1], xyC[:, 0]))
    th = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))

    def wrap(a):
        return (a + 180.0) % 360.0 - 180.0

    dth = wrap(th - thC[day])
    order = np.argsort(wrap(thC - thC[0]) % 360.0)
    steps = np.abs(wrap(np.diff(np.append(thC[order], thC[order][0]))))
    STEP_MIN, STEP_MAX = float(steps.min()), float(steps.max())

    # ---- the kept rows, in plot order (top to bottom) -----------------------------
    rows = []                                   # (key, display label, family, hour)
    for h in KEEP_CLOCK:
        rows.append((f"{h:02d}:00", clock_word(h), "clock", float(h)))
    for w in KEEP_WORDS:
        rows.append((w, w, "word", WORD_HOURS[w]))
    # One "placebo" family regardless of which sub-family the npz filed it under: the
    # lookup below is by phrase, so merging the label here changes only colour and grouping.
    keep_placebo = KEEP_PLACEBO
    if args.keep_placebo:
        if len(args.keep_placebo) == 1 and args.keep_placebo[0] == "auto":
            keep_placebo = auto_placebos(dth, lab, fam, day)
            print(f"[placebo] auto: {list(keep_placebo)}")
        else:
            keep_placebo = tuple(args.keep_placebo)
    for p in keep_placebo:
        rows.append((p, p, "placebo", np.nan))

    missing = [k for k, _, _, _ in rows if not (lab == k).any()]
    if missing:
        raise SystemExit(f"not in {args.npz}: {missing}")

    stat = {}
    for key, _, _, _ in rows:
        per_day = np.array([dth[(lab == key) & (day == d)].mean() for d in range(7)])
        stat[key] = (float(per_day.mean()), float(per_day.std(ddof=1) / np.sqrt(7)))
    widest = max(abs(stat[k][0]) for k, _, _, _ in rows)

    cyc = plt.get_cmap("twilight")
    nrm = Normalize(vmin=0, vmax=24)

    def mod_colour(f_, hr_, pale, slide=False, s_word="#ffffff", s_placebo="#000000"):
        # The clock family keeps twilight in every theme: that colour is the hour, it is
        # shared with panel B, and it is the only place colour carries information here.
        if f_ == "clock":
            return cyc(nrm(hr_))
        # On a slide, the other two families drop to plain white and black. The am/blue
        # pm/orange split duplicates what the row label already says, and against a
        # coloured background four hues plus a colourmap is more than the panel can hold.
        # Two families, one neutral axis. On a mid-toned slide that is white vs black;
        # on a white slide the same distinction has to run dark vs pale instead, and pale
        # for the placebos is what the paper figure already uses for them.
        if slide:
            return s_placebo if f_.startswith("placebo") else s_word
        if f_.startswith("placebo"):
            return pale
        return EARLY if hr_ < 13 else LATE

    if args.steer and args.steer_pairs:
        raise SystemExit("--steer and --steer-pairs both draw panel C; pick one")
    STEER = steering_curve(args.steer) if args.steer else None
    MPAIRS = matched_pairs(args.steer_pairs) if args.steer_pairs else None
    HAS_C = STEER is not None or MPAIRS is not None
    if HAS_C and args.figsize == ap.get_default("figsize"):
        args.figsize = args.pairs_figsize if MPAIRS is not None else args.steer_figsize
    w_in, h_in = (float(v) for v in args.figsize.lower().split("x"))
    s = args.scale
    themes = ("light", "dark") if args.theme == "both" else (args.theme,)

    for theme in themes:
        dark = theme == "dark"
        slide = theme in ("slide", "slide-white")
        # White slide: same slide treatment, opposite ink polarity. Kept as one branch so
        # every later `if slide` decision -- no backing box, no marker outline, label
        # offset 1.72, single chrome colour -- applies to both and the two cannot drift.
        white = theme == "slide-white"
        # MID, not PALE, for both the placebo markers and the hour trajectories. The paper
        # figure's PALE reads at 1.9:1 on white -- fine on a monitor, gone on a projector.
        # MID is 6.0:1 and still clearly the quiet family. The one pale thing kept pale is
        # panel A's adjacent-day shading, which is ink at alpha 0.10 and is meant to sit
        # behind the rows; and the grid, which should recede rather than compete.
        s_word, s_placebo = (INK, PLACEBO_W) if white else ("#ffffff", "#000000")
        line_c = args.line_color or (MID if white else "white")
        if slide:
            # One colour for every piece of chrome. ink/mid/pale exist to grade text and
            # rules against a KNOWN background; over an unknown slide there is no such
            # grading to do, and the mid greys were the part that failed. A white slide IS
            # a known background, but the grading is dropped there too so the two slide
            # variants stay the same figure.
            ink = mid = pale = INK if white else "#ffffff"
            bg = "white" if white else args.preview_bg
        else:
            ink, mid, pale = (INK_D, MID_D, PALE_D) if dark else (INK, MID, PALE)
            bg = "#11131a" if dark else "white"
        # What a marker outline is drawn in. Elsewhere it is the background colour, which
        # only works when the background is known; over a slide that colour is a visible
        # ring, so outlines go except on the clock markers, whose twilight colour runs
        # near-white at both midnight ends and would otherwise disappear. That one stays a
        # neutral grey: it is a hairline around a marker, not text.
        edge_none = "none" if slide else bg
        edge_clock = "#5b6270" if slide else mid

        fig = plt.figure(figsize=(w_in, h_in))
        # Bottom margin grows with --scale. Panel B's "weekday plane 1" is a 3-D axis label,
        # which mplot3d places below the cube in FIGURE space -- at larger type it runs off
        # the canvas entirely, and once an artist is off-canvas no bbox_inches="tight" or
        # pad_inches can bring it back (checked: it is clipped on the untrimmed canvas too).
        # The s == 1 value is left exactly as it was so the existing exports are unchanged.
        # Panel C is opt-in and appended, so the two-panel geometry below is untouched
        # when --steer is absent and the pinned exports stay byte-identical.
        WR = [1.0, 1.55] + ([args.pairs_width if MPAIRS is not None else args.steer_width]
                            if HAS_C else [])
        if args.swap_ab:
            WR[0], WR[1] = WR[1], WR[0]
        gs = GridSpec(1, len(WR), figure=fig, width_ratios=WR,
                      # swapped, the cube's slot is the wide one and its centred title
                      # fills it, so the columns need a real gap or title A meets letter B
                      wspace=(args.wspace if args.wspace is not None else
                              (0.045 if not HAS_C else 0.36)
                              + (0.20 if args.swap_ab else 0.0)),
                      # swapped, the forest is on the RIGHT and its x label is wider than
                      # its axes, so it overhangs the canvas and is clipped at draw time --
                      # bbox_inches="tight" cannot recover an artist already off-figure.
                      # The left margin shrinks to match: the colourbar that now sits there
                      # needs less room than the forest's y tick labels did.
                      # keyed on HAS_C, not on STEER: --steer-pairs is a three-panel
                      # plate too and was picking up the two-panel left margin, which
                      # costs it 4% of the canvas at the end where the colourbar sits.
                      left=(0.115 if not HAS_C else 0.078) - (0.07 if args.swap_ab
                                                              else 0.0),
                      # The 0.095 reserve exists because swapping puts the FOREST on the
                      # right, and its x label is wider than its axes. On the pairs plate
                      # the rightmost panel is C, whose group labels are centred under
                      # their own boxes and overhang by far less, so the reserve is most
                      # of an inch of canvas bought for a panel that is not there.
                      right=0.985 - (0.020 if MPAIRS is not None else
                                     (0.095 if args.swap_ab else 0.0)),
                      # The bottom margin grows with --scale because mplot3d places the
                      # cube's z label BELOW the cube in figure space, and at large type it
                      # walks off the canvas where no tight bbox can recover it. That
                      # allowance is sized for the long default labels: with
                      # --b-axis-labels pc the label is "PC2" and --b-labelpad tucks it in,
                      # so at --scale 1.95 the formula reserves a quarter of the plate
                      # height for three glyphs. --top/--bottom take it back, and the
                      # overlap check is what says whether taking it back was safe.
                      top=(args.top if args.top is not None
                           else 0.855 - 0.055 * (s - 1)),
                      bottom=(args.bottom if args.bottom is not None
                              else 0.095 + 0.16 * (s - 1)))
        # axA is always the modifier forest and axB always the 3-D cube, whichever column
        # each ends up in -- every downstream reference is to the CONTENT, not the slot.
        c_forest, c_cube = (1, 0) if args.swap_ab else (0, 1)
        axA = fig.add_subplot(gs[0, c_forest])
        axB = fig.add_subplot(gs[0, c_cube], projection="3d")
        axC = fig.add_subplot(gs[0, 2]) if HAS_C else None
        for ax in (axA, axB) + ((axC,) if axC is not None else ()):
            ax.patch.set_alpha(0.0)

        # ================= A: in-plane shift, 18 rows ==========================
        for sp in ("top", "right"):
            axA.spines[sp].set_visible(False)

        for sp in ("left", "bottom"):
            axA.spines[sp].set_linewidth(0.8)
            axA.spines[sp].set_color(pale)
        axA.tick_params(labelsize=11.0 * s * args.text_boost, length=3.5, width=0.8,
                        colors=mid)
        # A white grid at 0.22 is the right weight over a mid-toned slide; on white it is
        # nothing at all, so the white variant borrows the light theme's grid.
        # grid(False, **kwargs) does NOT turn the grid off -- matplotlib warns "line
        # properties are supplied, the grid will be enabled" and draws it anyway. The two
        # cases have to be separate calls.
        if args.no_gridlines:
            axA.grid(False)
        else:
            axA.grid(True, lw=0.5, alpha=0.22 if (slide and not white) else 0.35, zorder=0,
                     color="#ffffff" if (slide and not white)
                     else ("#8a8f99" if dark else "#eceef2"))
        axA.set_axisbelow(True)

        pos_y = np.arange(len(rows))[::-1]
        # The adjacent-day band: what a shift would have to reach to move the activation
        # onto the NEXT day. Everything plotted falls far short of it -- that is the point.
        # Initial capitals, matching the axis labels under --cap-labels and panel C's
        # group labels. These are annotations, not axis labels, so --cap-labels never
        # reaches them. Gated on the pairs plate for the same reason as the margins above:
        # this is shared panel-A code, and the two-panel and ladder exports were pinned
        # under the lower-case form. Their md5 check caught this the first time.
        BANDS = ((-1, "Previous day"), (+1, "Next day")) if MPAIRS is not None \
            else ((-1, "previous day"), (+1, "next day"))
        for sgn, side in BANDS:
            lo, hi = sorted((sgn * STEP_MIN, sgn * STEP_MAX))
            axA.axvspan(lo, hi, color=ink, alpha=0.10, lw=0, zorder=1)
            for edge in (STEP_MIN, STEP_MAX):
                axA.axvline(sgn * edge, color=ink, lw=1.0, ls=(0, (3, 2)), zorder=2)
            axA.text(sgn * (STEP_MIN + STEP_MAX) / 2.0, (len(rows) - 1) / 2.0, side,
                     rotation=90, ha="center", va="center",
                     fontsize=11.5 * s * args.text_boost,
                     color=ink, style="italic", zorder=5)

        axA.axvline(0, color=pale, lw=1.2, zorder=3)
        # One marker shape throughout. The paper figure reserves a square for the
        # both-tokens-varied placebos; with that split gone there is nothing for a second
        # shape to distinguish.
        for p, (key, disp, f_, hr_) in zip(pos_y, rows):
            mu_, se_ = stat[key]
            c = mod_colour(f_, hr_, pale, slide, s_word, s_placebo)
            axA.errorbar(mu_, p, xerr=se_, fmt="o",
                         ms=7.5 * s, color=c, ecolor=c, elinewidth=1.8, capsize=3.0,
                         zorder=4, markeredgewidth=0.9,
                         markeredgecolor=edge_clock if f_ == "clock" else edge_none)

        axA.set_yticks(pos_y)
        # The row labels are the longest strings on the plate and there are sixteen of
        # them in a fixed height, so they are what --scale runs into first. --a-row-size
        # trims them alone, which is the cheapest way to buy the panel room without
        # pulling every other size down with it.
        axA.set_yticklabels([d for _, d, _, _ in rows],
                            fontsize=12.5 * s * args.text_boost * args.a_row_size)
        for t, (_, _, f_, hr_) in zip(axA.get_yticklabels(), rows):
            # Slide theme: every row label white. Colour-coding them repeated what the
            # marker beside each already shows, and the dark blue of the am words was the
            # worst-contrasting text on the plate.
            t.set_color(ink if (slide or f_ == "clock")
                        else mod_colour(f_, hr_, pale))
        # Hairlines between the four families: with the row count down to 18 the groups
        # are readable, and a rule is quieter than repeating a family name on every row.
        edges, seen = [], None
        for i, (_, _, f_, _) in enumerate(rows):
            if seen is not None and f_ != seen:
                edges.append(len(rows) - i - 0.5)
            seen = f_
        for e in edges:
            axA.axhline(e, color=pale, lw=0.7, alpha=0.55, zorder=2)

        axA.set_xlim(-STEP_MAX * 1.12, STEP_MAX * 1.12)
        # -1.0 / +0.2 leaves 1.0 of a row below the bottom row and 1.2 above the top one:
        # 13.6% of the panel's height spent on margin the rows could be using. --a-tight-rows
        # cuts both to 0.6, which is half the row pitch and still clears the markers and
        # their error bars. Opt-in: every pinned export was measured under the loose limits.
        axA.set_ylim(*((-0.6, len(rows) - 0.4) if args.a_tight_rows
                       else (-1.0, len(rows) + 0.2)))
        axA.set_xlabel(args.a_xlabel or
                       "in-plane angular shift from the day's own centroid  (degrees)",
                       fontsize=13.0 * s * args.text_boost, color=ink, labelpad=6)

        # ================= B: weekday plane + hour direction ===================
        Q = plane.T
        hrs = np.array(sorted(set(hour[cl].tolist())))
        Chour = np.stack([A[cl & (hour == h)].mean(0) for h in hrs])
        if e3_raw is None:
            _, _, Vh = np.linalg.svd(Chour - Chour.mean(0), full_matrices=False)
            e3 = Vh[0] - Q @ (Q.T @ Vh[0])
            e3 /= np.linalg.norm(e3)
        else:
            # Already orthogonal to the plane -- all three are components of the same
            # SVD -- so projecting it out would only be numerical noise. Left unprojected
            # so the axis drawn is exactly the component the table scored.
            e3 = e3_raw
        # Fix the sign so early hours sit high, matching the colourbar's 1am-at-top.
        if np.corrcoef(hrs, (Chour - Chour.mean(0)) @ e3)[0, 1] > 0:
            e3 = -e3
        B3 = np.column_stack([Q[:, 0], Q[:, 1], e3])
        X = (A - mu) @ B3
        Cday_cl = np.stack([A[cl & (day == d)].mean(0) for d in range(7)])
        KD = (Cday_cl - mu) @ B3

        RING_C = KD[:, :2].mean(0)
        ring = CubicSpline(np.arange(8, dtype=float), np.vstack([KD, KD[:1]]),
                           bc_type="periodic", axis=0)(np.linspace(0, 7, 400))
        axB.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=ink, lw=1.8, zorder=2)
        CUBE_LABELS = []   # the cube's weekday text artists, for --check-overlap
        for d in range(7):
            rows_d = np.array([np.where(cl & (day == d) & (hour == h))[0][0]
                               for h in hrs])
            Pd = X[rows_d]
            S = CubicSpline(hrs, Pd, bc_type="natural", axis=0)(
                np.linspace(hrs[0], hrs[-1], 240))
            # One neutral colour for all seven, not the day colour. The dots already carry
            # a colour dimension (twilight = hour); a second one (day) on the curve made
            # the panel argue with itself about what colour meant.
            axB.plot(S[:, 0], S[:, 1], S[:, 2], color=line_c, lw=1.7,
                     alpha=0.75, zorder=3)
            axB.scatter(Pd[:, 0], Pd[:, 1], Pd[:, 2], s=26, c=[cyc(nrm(h)) for h in hrs],
                        depthshade=False, edgecolors=edge_none, linewidths=0.5, zorder=4)
            axB.scatter([KD[d, 0]], [KD[d, 1]], [KD[d, 2]], s=110, color=DAY_C[d],
                        depthshade=False, edgecolors=ink, linewidths=1.0, zorder=6)
            # Well outside the ring, not the paper figure's 1.28. At slide type size the
            # word is wider than the marker, and the hour trajectories rise vertically out
            # of each centroid -- so a label close in sits on its own dot AND on its own
            # curve. Pushing it radially clears both, which matters most once the backing
            # box is gone (slide theme) and there is nothing to hide an overlap.
            off = 1.72 if slide else 1.55
            if args.b_label_pad is None:
                lx_, ly_ = KD[d, 0] * off, KD[d, 1] * off
            else:
                # constant clearance beyond the centroid, from the RING centre. Scaling the
                # radius instead makes the offset proportional to a radius that is not
                # constant around this ring, which is what pushed Sun (r = 8.6) half again
                # as far out as Wed (r = 6.0).
                pad = (args.b_label_pad if args.b_label_pad_day is None
                       else args.b_label_pad_day[d])
                v = KD[d, :2] - RING_C
                lx_, ly_ = RING_C + v * (1.0 + pad / max(np.hypot(*v), 1e-9))
            t = axB.text(lx_, ly_, KD[d, 2], DAYS[d][:3],
                         fontsize=13.0 * s * args.cube_text, color=ink,
                         ha="center", va="center",
                         fontweight="bold", zorder=7)
            # Backed only when the background is known. The box is there because the day
            # label sits over the hour trajectories -- but it is painted in the background
            # colour, so over a slide it stops being invisible backing and becomes a
            # filled rectangle in the wrong colour. Transparent exports go without it.
            if not slide:
                t.set_bbox(dict(boxstyle="round,pad=0.16", fc=bg, ec="none", alpha=0.78))
            CUBE_LABELS.append(t)

        drawn = np.vstack([X[cl], KD])
        ctr = (drawn.max(0) + drawn.min(0)) / 2.0
        half = (drawn.max(0) - drawn.min(0)).max() / 2.0 * 1.06
        for setter, c in ((axB.set_xlim, ctr[0]), (axB.set_ylim, ctr[1]),
                          (axB.set_zlim, ctr[2])):
            setter(c - half, c + half)
        # zoom, not just a wider cell: a 3-D axes reserves room for the corners the cube
        # would sweep through at any view angle, so most of that margin is unreachable at
        # this fixed elev/azim and reads as dead space. zoom scales the drawn box inside
        # the same axes and takes it back.
        axB.set_box_aspect((1, 1, 1), zoom=args.b_zoom)
        axB.set_xticks([]); axB.set_yticks([]); axB.set_zticks([])
        # Panes off rather than filled: a filled pane is an opaque rectangle, which would
        # punch a solid block through an otherwise transparent export.
        for axis in (axB.xaxis, axB.yaxis, axB.zaxis):
            axis.pane.set_visible(False)
            axis.line.set_color(pale)
        # The paper figure draws four vertical box edges, to complete a cube whose panes
        # are filled white. With the panes off they complete nothing -- they read as four
        # stray lines floating past the data -- so they are not drawn here. The three
        # axis lines and the labelled directions carry the orientation on their own.
        if args.b_axis_labels == "pc" and args.basis != "supervised":
            i, j = pcs["plane_idx"]
            ax_lab = (f"PC{i + 1}", f"PC{j + 1}", f"PC{pcs['e3_idx'] + 1}")
        lp = -6.0 if args.b_labelpad is None else args.b_labelpad
        axB.set_xlabel(ax_lab[0], fontsize=12.0 * s * args.cube_text, color=mid, labelpad=lp)
        axB.set_ylabel(ax_lab[1], fontsize=12.0 * s * args.cube_text, color=mid, labelpad=lp)
        axB.set_zlabel(ax_lab[2], fontsize=12.0 * s * args.cube_text, color=mid,
                       labelpad=lp - 2)
        axB.view_init(elev=args.elev, azim=args.azim)

        # Lay each label along its own axis. The angle is the on-screen direction of that
        # axis under the current projection, measured by projecting the axis' two end
        # points -- not guessed from elev/azim, which do not map to a screen angle in any
        # simple way once the perspective transform is applied. This is only a valid
        # rotation because apply_aspect has squared panel B's box (5.37 x 5.37 inches
        # here), so the projected unit square maps to pixels isotropically; on a
        # non-square box the same angle would shear and the labels would sit off their
        # axes by a few degrees.
        if args.b_rotate_labels:
            M = axB.get_proj()
            lims = (axB.get_xlim(), axB.get_ylim(), axB.get_zlim())
            ctr = [float(np.mean(l)) for l in lims]
            for i, axis in enumerate((axB.xaxis, axB.yaxis, axB.zaxis)):
                p0, p1 = list(ctr), list(ctr)
                p0[i], p1[i] = lims[i]
                x0, y0, _ = proj3d.proj_transform(*p0, M)
                x1, y1, _ = proj3d.proj_transform(*p1, M)
                a = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
                # keep the text upright: a label is a line, not an arrow
                a = a - 180 if a > 90 else (a + 180 if a <= -90 else a)
                axis.set_rotate_label(False)
                axis.label.set_rotation(a)

        # pad well clear of the cube: the colourbar steals its space from axB's right edge,
        # which is exactly where the z-axis label sits, and at the enlarged zoom the two
        # were printing on top of each other.
        # On the LEFT when the cube leads the plate. The colourbar is placed against its
        # axes' slot, not against the drawn cube, and apply_aspect leaves that slot much
        # wider than the cube -- so a right-hand bar in column 0 lands on top of whatever
        # is in column 1 rather than beside the cube.
        # shrink is --cb-shrink, defaulting to the 0.56 every pinned export carries. The
        # bar is sized off axB's SLOT, and apply_aspect leaves that slot far taller than
        # the cube drawn inside it, so 0.56 of the slot is a good deal less than 0.56 of
        # the cube -- the bar reads short beside it, and shorter still once the tick
        # labels grow.
        cb = fig.colorbar(ScalarMappable(norm=nrm, cmap=cyc), ax=axB, fraction=0.026,
                          pad=0.11, shrink=args.cb_shrink,
                          location="left" if args.swap_ab else "right",
                          boundaries=np.linspace(hrs[0], hrs[-1], 256))
        cb.set_ticks([1, 6, 12, 18, 23])
        cb.set_ticklabels(["1am", "6am", "12pm", "6pm", "11pm"])
        cb.ax.invert_yaxis()
        cb.ax.tick_params(labelsize=11.5 * s * args.cube_text, length=3.0, width=0.8,
                          colors=mid)
        cb.outline.set_edgecolor(pale)
        cb.set_label("clock time", fontsize=12.5 * s * args.cube_text, color=ink,
                     labelpad=5)

        # Panel B moves only AFTER the colourbar exists, and both axes move together.
        # Shifting it earlier does nothing at any value: the colourbar sizes itself from
        # axB's subplotspec slot, and Axes3D.apply_aspect re-derives the drawn box from the
        # original position during every draw, so an earlier set_position is overwritten
        # twice over. which="both" moves the original too, which is what apply_aspect then
        # squares up inside.
        if args.b_shift or args.b_panel_shift:
            # --b-shift moves the panel AND its heading; --b-panel-shift moves only the
            # panel. The heading is placed later, off axB's position, so the difference is
            # made there: panel B's anchor adds b_panel_shift straight back.
            for a_ in (axB, cb.ax):
                q = a_.get_position()
                a_.set_position([q.x0 - args.b_shift - args.b_panel_shift, q.y0,
                                 q.width, q.height], which="both")

        # ================= C: steer the ring, and the clock does not follow =========
        # Panel A measures how far a word modifier rotates the ring; panel C asks whether
        # that rotation is what the model READS. Both families of edit are clamped onto the
        # weekday token over the gated band and read as a shift in the predicted hour. The
        # x-axis is trimmed to the largest ring edit, so what is shown is exactly the regime
        # panel A lives in -- the off-plane ladder runs 8x further right than this crop.
        if axC is not None:
            for sp in ("top", "right"):
                axC.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                axC.spines[sp].set_linewidth(0.8)
                axC.spines[sp].set_color(pale)
            axC.tick_params(labelsize=11.0 * s * args.text_boost, length=3.5, width=0.8,
                            colors=mid)
            if args.no_gridlines:                       # see the note on panel A's grid
                axC.grid(False)
            else:
                axC.grid(True, lw=0.5, alpha=0.22 if (slide and not white) else 0.35,
                         zorder=0, color="#ffffff" if (slide and not white)
                         else ("#8a8f99" if dark else "#eceef2"))
            axC.set_axisbelow(True)

            if MPAIRS is not None:
                # ---- the matched pairs, as paired distributions -------------------------
                # One box per EDIT and both members of a pair adjacent, because the claim is
                # a within-pair comparison: at the same ||Delta||, does leaving the weekday
                # plane move the clock more than staying in it. The dose ladder this
                # replaces could only answer that by interpolation.
                #
                # Points under the boxes, all 84 of them. The box alone would show a spread
                # and hide that it is 12 carriers x 7 days -- and at these sizes the spread
                # IS the interesting part: the in-plane boxes sit at the readout's bf16
                # noise floor (a no-op `identity` clamp scores sd 0.059 against in-plane
                # 0.072-0.199), so a reader has to be able to see how little separates them
                # from nothing. The mean and its bootstrap CI go on top because that, not
                # the box, is the inferential claim, and the CI is over the 12 carriers.
                W, GAP = 0.30, 0.185
                labels = []
                for gi, e in enumerate(MPAIRS["pairs"]):
                    for side, col in (("in", PAIR_IN), ("off", PAIR_OFF)):
                        x = gi + (-GAP if side == "in" else GAP)
                        v = e[side]["v"]
                        # jitter is deterministic: a figure that redraws differently on
                        # every run cannot be checksummed, and this plate is pinned
                        # NOT hash(): PYTHONHASHSEED randomises string hashing per
                        # process, so that seed would differ between two runs of the same
                        # command. The index does not.
                        jr = np.random.default_rng(2 * gi + (side == "off"))
                        axC.plot(x + jr.uniform(-W * 0.36, W * 0.36, v.size), v, "o",
                                 ms=2.9 * s, mfc=col, mec="none", alpha=0.38, zorder=3)
                        bp = axC.boxplot([v], positions=[x], widths=W, whis=1.5,
                                         showfliers=False, showcaps=False,
                                         patch_artist=True, zorder=4)
                        for pt in bp["boxes"]:
                            pt.set(facecolor="none", edgecolor=col, lw=1.5 * s)
                        for pt in bp["whiskers"]:
                            pt.set(color=col, lw=1.2 * s)
                        for pt in bp["medians"]:
                            pt.set(color=col, lw=2.2 * s)
                        axC.plot([x] * 2, [e[side]["lo"], e[side]["hi"]], color=col,
                                 lw=2.0 * s, solid_capstyle="butt", zorder=6)
                        axC.plot(x, e[side]["m"], "D", ms=4.6 * s, color=col,
                                 mec=edge_none if slide else bg, mew=1.0 * s, zorder=7)
                    labels.append(e["label"])
                axC.axhline(0, color=mid, lw=0.9, zorder=1)
                axC.set_xticks(range(len(MPAIRS["pairs"])))
                # The group labels are FITTED, not set. They are the tightest type on the
                # plate -- three of them, each up to 26 characters, sharing panel C's width
                # -- so they, and not the panels, are what caps --scale. Measuring the
                # widest line against one group's share of the axes lets the rest of the
                # plate scale without this silently overlapping: at the ladder panel's
                # width they overlap outright, and an overlap here is not visible in the
                # log, only in the export.
                r_ = fig.canvas.get_renderer()
                iv_ = fig.transFigure.inverted()
                def _lw(txt, fs):
                    t = fig.text(0.5, 0.5, txt, fontsize=fs)
                    bb = t.get_window_extent(r_)
                    t.remove()
                    return float(iv_.transform(bb.p1)[0] - iv_.transform(bb.p0)[0])
                # 12.0, not the 9.5 the three-line labels needed. The labels are one line
                # narrower since the edit sizes moved to the caption, so the column can
                # carry more type; the fit below caps it at whatever actually clears.
                # Deliberately larger than anything that will survive: the group labels
                # are the panel's x axis and the fit below caps them at whatever the column
                # actually clears, so setting the base high makes the COLUMN the only thing
                # deciding, rather than a constant that has to be re-tuned every time the
                # labels or the width change.
                FS_GRP = 20.0 * s
                grp_w = axC.get_position().width / len(MPAIRS["pairs"])
                # White left between one group label and the next. The labels are CENTRED
                # on their groups, so this is the whole separation between two of them --
                # at 0.006 the first two read as one run, "late-early late-early", because
                # they nearly touch. It costs about 9% of the type size and buys the panel
                # three labels instead of one long one.
                GRP_GAP = 0.014
                widest = max(_lw(ln, FS_GRP) for lab in labels for ln in lab.split("\n"))
                if widest > grp_w - GRP_GAP:
                    print(f"[talk] panel C group labels {FS_GRP:.1f}pt -> "
                          f"{FS_GRP * (grp_w - GRP_GAP) / widest:.1f}pt "
                          f"to clear a {grp_w:.4f} column")
                    FS_GRP *= (grp_w - GRP_GAP) / widest
                axC.set_xticklabels(labels, fontsize=FS_GRP, color=ink, linespacing=1.6)
                axC.tick_params(axis="x", length=0, pad=7)
                axC.set_xlim(-0.5, len(MPAIRS["pairs"]) - 0.5)
                lo = min(e[k]["v"].min() for e in MPAIRS["pairs"] for k in ("in", "off"))
                hi = max(e[k]["v"].max() for e in MPAIRS["pairs"] for k in ("in", "off"))
                # Headroom for the two legend rows, which sit over the LEFTMOST pair, where
                # nothing reaches above ~0.45 -- so the pad only has to clear the legend
                # itself, not the tallest box. It is a fraction of the drawn range, and the
                # range is set by the half-day pair at the right, so a generous fraction
                # buys a band of empty canvas the width of the whole panel.
                axC.set_ylim(lo - 0.05 * (hi - lo), hi + 0.20 * (hi - lo))
                # mew scales with --scale, like the box edges it stands for. It was a flat
                # 1.5 against the boxes' 1.5 * s, so at --scale 2.15 the key was drawn at
                # under half the weight of the thing it is the key TO, and read as a
                # different, lighter mark rather than the same one.
                hs = [plt.Line2D([], [], color=c, marker="s", mfc="none", mew=1.5 * s,
                                 ms=8.5 * s, lw=0) for c in (PAIR_IN, PAIR_OFF)]
                axC.legend(hs, ["In-plane edit",
                                "Off-plane edit, matched ‖Δ‖"],
                           # 13.0, matching the axis-label tier rather than the tick tier.
                           # The legend names the panel's two families; it is the key to
                           # the whole comparison, not an annotation on it.
                           frameon=False, fontsize=13.0 * s * args.text_boost,
                           loc="upper left",
                           # handletextpad is in FONT-SIZE units, so it grew with the
                           # legend type: at 0.6 and 32 pt the gap between a square and
                           # its label was wider than the square. handlelength matters
                           # more here and is easy to miss -- it reserves a 2.0-unit box
                           # for the handle and CENTRES the marker in it, so for a
                           # marker-only entry most of the visible gap is empty handle
                           # box, not pad. Trimming the pad alone cannot close it.
                           handlelength=0.9, handletextpad=0.2,
                           borderpad=0.1, labelcolor=ink)
                # "shift in predicted hour", not "shift in the hour the model predicts".
                # The y label is ROTATED, so its length is measured against the axes HEIGHT,
                # and the long form is taller than panel C's axes above --scale 1.4 -- it
                # runs past both ends and collides with panel B's title. Nothing is lost:
                # the quantity is the same one and the caption defines it.
                axC.set_ylabel("shift in predicted hour  (h)",
                               fontsize=13.5 * s * args.text_boost, color=ink, labelpad=6)
            else:
                XMAX = STEER["xmax"] * 1.06
                lx, ly = STEER["lx"], STEER["ly"]
                keep = lx <= XMAX * 1.45          # one rung past the crop, so the line exits
                # Panel C's three arms, by hue or by grey value. The marker stays a circle in
                # both: shape was tried as the mono encoding and reads as a second, unrelated
                # dimension, where a grey ramp reads as one scale of emphasis.
                C_LAD = MONO_LAD if args.c_mono else LATE
                C_RING = MONO_RING if args.c_mono else EARLY
                C_IN = MONO_IN if args.c_mono else INPLANE
                axC.fill_between(lx[keep], STEER["llo"][keep], STEER["lhi"][keep],
                                 color=C_LAD, alpha=0.20 if not args.c_mono else 0.14,
                                 lw=0, zorder=2)
                lad_line, = axC.plot(lx[keep], ly[keep], "-o", color=C_LAD, lw=2.6 * s,
                                     ms=6.4 * s, mec=edge_none if slide else bg,
                                     mew=1.2 * s, zorder=4)
                axC.axhline(0, color=mid, lw=0.9, zorder=1)

                # below the axis for this one: at x = 13 the ladder is already at 0.13, so a
                # label placed above lands on the orange curve
                # One legend row per EDIT, each with its own mark. Grey value carries the
                # family -- black rotates the ring, mid grey is the in-plane step, the ladder is
                # the line -- and within the black family the marker SHAPE separates the two
                # rotations. A fourth grey value could not: the lightest that still clears 3:1
                # on white is already spent on the in-plane step.
                #
                # The two black marks and the grey one come from the SAME vector, the in-plane
                # component of mean(late) - mean(early), used two ways. A rotation takes only
                # its ANGLE and keeps the live radius; the in-plane step applies the whole
                # thing, radius and all. Naming both "the early-to-late shift" made them look
                # like one edit measured twice, so each row now names the part it uses.
                ARM = {"clamp_ring_own": (C_RING, "o",
                                          "+{deg:.2f}\u00b0 rotation \u2014 the "
                                          "\u201clate\u2212early\u201d angle"),
                       "clamp_ring_half": (C_RING, "s",
                                           "+25.71\u00b0 rotation \u2014 halfway to the "
                                           "next day"),
                       "clamp_in_a1": (C_IN, "o",
                                       "in-plane \u201clate\u2212early\u201d step "
                                       "\u2014 angle and radius")}
                # NB: no loop variable here may be named A, lab, fam, day or hour -- those are
                # the capture arrays of panels A and B, still live in this scope, and shadowing
                # one fails 200 lines later in the summary print rather than here.
                hs, ls = [], []
                for arm_name in ("clamp_ring_own", "clamp_ring_half", "clamp_in_a1"):
                    arm = STEER["arms"][arm_name]
                    col, mk, leg = ARM[arm_name]
                    axC.plot([arm["d"]] * 2, [arm["lo"], arm["hi"]], color=col, lw=2.0 * s,
                             zorder=5, solid_capstyle="butt")
                    axC.plot(arm["d"], arm["m"], mk, ms=8.0 * s, color=col,
                             mec=edge_none if slide else bg, mew=1.2 * s, zorder=6)
                    hs.append(plt.Line2D([], [], color=col, marker=mk, ms=8.0 * s, lw=0))
                    ls.append(leg.format(**arm))
                # The ladder last: it is the one arm that leaves the weekday plane, so it reads
                # as the comparison the other three are held against, not as the subject.
                hs.append(lad_line)
                ls.append("off-plane \u201clate\u2212early\u201d direction")
                axC.legend(hs, ls, frameon=False, fontsize=10.5 * s, loc="upper left",
                           handletextpad=0.6, borderpad=0.1, labelcolor=ink)

                axC.set_xlim(0, XMAX)
                # four legend rows need real headroom, or they crowd the ladder
                axC.set_ylim(-0.10, STEER["read"](XMAX) * 1.72)
                axC.set_xlabel("size of the edit,  ‖Δ‖", fontsize=13.5 * s, color=ink,
                               labelpad=6)
                axC.set_ylabel("shift in the hour the model predicts  (h)",
                               fontsize=13.5 * s, color=ink, labelpad=6)

        # Panel letters. Dropped in the original talk cut -- with two panels and a title
        # that states the finding, "A" and "B" label nothing a viewer needs, and on a slide
        # they are two more glyphs competing with the data. They are back because prose
        # cites the panels ("see Fig. 4A"), and a citation to a letter the plate does not
        # carry is worse than a redundant letter. --no-panel-labels restores the talk cut.
        #
        # Placed in FIGURE coordinates off each axes' bbox, not in axes coordinates: panel
        # A's y tick labels ("mid afternoon") run far to the left of its axes box, and panel
        # B is an mplot3d axes whose box is much wider than the cube drawn inside it, so an
        # axes-relative offset that clears one lands wrongly on the other.
        # Letters and per-panel titles share one x, so a title always starts just after
        # its own letter. Both are placed in FIGURE coordinates off each axes' bbox rather
        # than in axes coordinates: panel A's y tick labels ("mid afternoon") run far to
        # the left of its axes box, and panel B is an mplot3d axes whose box is much wider
        # than the cube drawn inside it, so an axes-relative offset that clears one lands
        # wrongly on the other.
        # Sentence case, applied once at the end rather than at each set_xlabel call, so
        # there is exactly one place the convention lives and panel C cannot drift from
        # panels A and B. Re-setting the text keeps the size and colour already on the
        # Text object; only the string changes.
        if args.cap_labels:
            cap = lambda t: (t[:1].upper() + t[1:]) if t else t
            for ax in [axA, axB, cb.ax] + ([axC] if axC is not None else []):
                ax.set_xlabel(cap(ax.get_xlabel()))
                ax.set_ylabel(cap(ax.get_ylabel()))
                if hasattr(ax, "get_zlabel"):
                    ax.set_zlabel(cap(ax.get_zlabel()))

        # Letter bold, title in the regular weight beside it, left-aligned off one shared
        # anchor per panel -- the convention combined_2x2 already uses, so the two plates
        # read as one set. The anchors are per-panel because the boxes are not comparable:
        # panel A's y tick labels ("mid afternoon") run far to the left of its axes box,
        # and panel B is an mplot3d axes whose box is much wider than the cube inside it.
        # B's anchor is near its box edge rather than near its cube, which is where the
        # two-panel cut put it -- with a title attached, the old offset drops the pair into
        # the middle of the cube.
        # B's anchor moves ONLY when B has a title. Without one the letter stays at the
        # two-panel offset, which is what the pinned exports carry.
        # B's anchor moves left only on the THREE-panel plate, where its box is wide and a
        # letter near the cube would land inside its own title. On the two-panel plate the
        # boxes are adjacent, and -0.020 puts the letter inside panel A's box: the gap
        # between title A and letter B collapses to 54 px against the 3-panel plate's 450.
        # Titles are positional: --title-a labels whatever is drawn first. The letter
        # OFFSETS are not -- they belong to the content. The forest needs -0.085 because
        # its y tick labels ("mid afternoon") run far left of its axes box; the cube needs
        # a positive offset because its mplot3d box is much wider than the cube inside it.
        cube_title = args.title_a if args.swap_ab else args.title_b
        b_dx = (-0.020 if (cube_title and axC is not None) else 0.055) + args.b_panel_shift
        order = ([(axB, b_dx), (axA, -0.085)] if args.swap_ab
                 else [(axA, -0.085), (axB, b_dx)])
        if axC is not None:
            order.append((axC, -0.058))
        titles = [args.title_a, args.title_b] + ([args.title_c] if axC is not None else [])
        spots = [(ax, dx, "ABC"[k], titles[k]) for k, (ax, dx) in enumerate(order)]
        if axC is not None:
            spots.append((axC, -0.058, "C", args.title_c))
        HEAD = {}
        # --centre-titles measures the drawn title and hangs the letter off it, rather than
        # placing both from a fixed anchor. Fixed anchors work while the narrow panel leads;
        # once the 3-D panel does, its title is wider than its own column and lands on the
        # next panel's letter, and no single offset fixes that for both orders.
        rend = fig.canvas.get_renderer() if args.centre_titles else None
        inv = fig.transFigure.inverted()
        FS_TTL = 15.5 * s
        FS_LET = 19.0 * s
        # TWO different offsets, which the single 0.021 constant used to conflate:
        #   LET_OFF   how far LEFT of its own title a panel letter is placed. The letter is
        #             drawn ha="left", so the white between letter and title is
        #             LET_OFF - (letter width) -- set LET_OFF below the letter width and
        #             the letter runs into its own title, which is what happened first.
        #   PAIR_GAP  the white left between one title's RIGHT edge and the next panel's
        #             letter. This is the gap a reader sees between two panels' headings.
        # Under the default the two are 0.021*s and letw + 0.012, which is what every
        # pinned export carries. --fit-titles cuts both and spends the canvas on type.
        LET_OFF = 0.021 * s
        if rend is not None:
            # Centring is per-box, so two long titles on adjacent boxes can still meet in
            # the middle -- and on this plate they do, filling 86% of the width between
            # them. Rather than pick a size that happens to fit these strings, measure them
            # and shrink only as far as the tightest pair requires. At the weekday-token
            # titles this lands near 1.0 and nothing visibly changes.
            def _w(txt, fs):
                t = fig.text(0.5, 0.5, txt, fontsize=fs)
                bb = t.get_window_extent(rend)
                t.remove()
                return float(inv.transform(bb.p1)[0] - inv.transform(bb.p0)[0])
            letw = _w("M", 19.0 * s) if args.panel_labels else 0.0
            LET_OFF = (letw + 0.005 * s) if args.fit_titles else 0.021 * s
            PAIR_GAP = (0.014 if args.fit_titles else letw + 0.012)
            cent = [a.get_position().x0 + a.get_position().width / 2 for a, *_ in spots]
            shrink = 1.0 if not args.fit_titles else 99.0
            for k in range(len(spots) - 1):
                ti, tj = spots[k][3], spots[k + 1][3]
                if not (ti and tj):
                    continue
                # need is (w_i + w_j)/2 because each title is CENTRED on its own box, so
                # only half of each reaches into the space between the two centres.
                #
                # Under --fit-titles the LETTER scales with the title, so letw is not a
                # constant the fit can subtract off once -- it grows with the answer. The
                # constraint is d - (letw0 + 0.005s) - PAIR_GAP >= need0*k with both letw
                # and need proportional to k, which solves in closed form rather than
                # needing a fixed-point loop:
                #     k <= (d - 0.005s - PAIR_GAP) / (need0 + letw0)
                if args.fit_titles:
                    d_ = cent[k + 1] - cent[k]
                    need0 = (_w(ti, FS_TTL) + _w(tj, FS_TTL)) / 2.0
                    room = d_ - 0.005 * s - PAIR_GAP
                    need = need0 + letw
                    # room > 0 skips the degenerate self-pair: `spots` lists panel C twice,
                    # so the last pair has d = 0 and a negative room that would drive the
                    # whole fit negative. The duplicate predates this code and is left
                    # alone because the pinned fullstop plates were exported with it.
                    if need > 0 and room > 0:
                        shrink = min(shrink, room / need)
                    continue
                room = cent[k + 1] - cent[k] - LET_OFF - PAIR_GAP
                need = (_w(ti, FS_TTL) + _w(tj, FS_TTL)) / 2.0
                # --fit-titles takes the min over pairs in BOTH directions, so the type
                # grows into spare room instead of only shrinking out of a shortfall.
                if room > 0 and (args.fit_titles or need > room):
                    shrink = min(shrink, room / need)
            if shrink != 1.0 and shrink < 99.0:
                print(f"[talk] centred titles at {FS_TTL:.1f}pt "
                      f"{'shrink' if shrink < 1 else 'grow'} to {FS_TTL * shrink:.1f}pt")
                FS_TTL *= shrink
                if args.fit_titles:
                    # The letter keeps its 19.0/15.5 ratio to the title, so it stays the
                    # larger of the two -- a heading whose letter is smaller than its own
                    # words reads as a footnote marker rather than a label.
                    FS_LET *= shrink
                    letw *= shrink
                    LET_OFF = letw + 0.005 * s
                    print(f"[talk] panel letters follow to {FS_LET:.1f}pt")
        # One shared baseline for every heading, rather than each hanging off its own
        # panel's top edge. The three tops are NOT level: apply_aspect squares the cube's
        # box inside a slot much wider than it is tall, which shortens the box and drops
        # its y1 below the two 2-D panels', so panel A's heading sat visibly low. Align to
        # the highest of them, so no heading moves DOWN and none can collide with a panel.
        y_top = max(a_.get_position().y1 for a_, *_ in spots) if args.align_titles else None
        for ax, dx, letter, ptitle in spots:
            p = ax.get_position()
            x, y = p.x0 + dx, (y_top if y_top is not None else p.y1) + 0.012
            if ptitle and args.centre_titles:
                t = fig.text(p.x0 + p.width / 2, y + 0.002, ptitle,
                             fontsize=FS_TTL, color=ink, ha="center", va="bottom")
                HEAD["ttl" + letter] = t
                x = float(inv.transform(t.get_window_extent(rend).p0)[0]) - LET_OFF
            elif ptitle:
                HEAD["ttl" + letter] = fig.text(
                    x + (0.026 if args.panel_labels else 0.0), y + 0.002, ptitle,
                    fontsize=15.5 * s, color=ink, ha="left", va="bottom")
            if args.panel_labels:
                HEAD["let" + letter] = fig.text(
                    x, y, letter, fontsize=FS_LET, color=ink, fontweight="bold",
                    ha="left", va="bottom")

        # Equalise the two white gaps by MEASURING them. Estimating the widths in figure
        # coordinates does not work: matplotlib's text extents carry padding the drawn
        # glyphs do not, and the answer was out by a factor of three when checked against
        # the exported pixels. Reading the placed artists back is exact, and the correction
        # is a rigid slide of panel B plus its own letter and title, so nothing inside the
        # panel changes and panels A and C do not move at all.
        if args.b_balance and {"ttlA", "letB", "ttlB", "letC"} <= set(HEAD):
            rend = fig.canvas.get_renderer()
            inv = fig.transFigure.inverted()
            edge = lambda k, right: float(
                inv.transform(HEAD[k].get_window_extent(rend).p1 if right
                              else HEAD[k].get_window_extent(rend).p0)[0])
            gap1 = edge("letB", False) - edge("ttlA", True)
            gap2 = edge("letC", False) - edge("ttlB", True)
            d = (gap1 - gap2) / 2.0
            print(f"[talk] title gaps A->B {gap1:.4f}, B->C {gap2:.4f}; "
                  f"sliding panel B {'left' if d > 0 else 'right'} by {abs(d):.4f}")
            for a_ in (axB, cb.ax):
                q = a_.get_position()
                a_.set_position([q.x0 - d, q.y0, q.width, q.height], which="both")
            for k in ("letB", "ttlB"):
                HEAD[k].set_x(HEAD[k].get_position()[0] - d)

        if args.title.strip():
            fig.suptitle(args.title, fontsize=21.0 * s, color=ink, y=0.975,
                         fontweight="semibold")
        if args.subtitle:
            fig.text(0.5, 0.900, f"Llama-3.1-8B  ·  “It was {{t}} on {{day}}.” read at the "
                     f"{site}  ·  layer {layer}", fontsize=12.0 * s, color=mid,
                     ha="center")

        # A drawn-extent collision check. This plate is FITTED rather than hand-placed --
        # the titles, the panel letters and panel C's group labels all size themselves off
        # measured room -- so the failure mode is no longer "a number was tuned at one
        # --scale and is wrong at another", it is "two artists the fit does not know about
        # each other met". The two that actually do are panel B's x label, which is wider
        # than its own axes, and panel C's y label. Report rather than raise: an overlap of
        # a few pixels between two long labels is sometimes the right call, and only a look
        # at the export can decide.
        if args.check_overlap:
            # Draw first. Axis labels are POSITIONED at draw time -- before one, every
            # ax.xaxis.label reports its default extent at the origin, which collides with
            # everything at the left edge and reports pairs that are at opposite ends of
            # the plate. The panel titles are fig.text and are placed on creation, which is
            # why only the axis labels looked wrong.
            fig.canvas.draw()
            rend_ = fig.canvas.get_renderer()
            # The cube's weekday labels are axB.text artists placed by hand at the
            # --b-label-pad-day offsets, which were tuned at one type size. They are what
            # actually runs into panel B's row labels when the cube's type grows, and the
            # check called a plate clean without them while "Thu" sat on "early morning".
            named = [(f"cube {t_.get_text()}", t_) for t_ in CUBE_LABELS]
            named += [("title " + k[3:], v) for k, v in HEAD.items() if k.startswith("ttl")]
            named += [("letter " + k[3:], v) for k, v in HEAD.items() if k.startswith("let")]
            # Letters follow DRAW order, so under --swap-ab panel "A" is the cube (axB)
            # and panel "B" is the forest (axA). Naming these off the axes variables
            # instead reported "title A x C y label" for two artists at opposite ends.
            _axes = ([(axB, "A"), (axA, "B")] if args.swap_ab else [(axA, "A"), (axB, "B")])
            for ax_, nm in (_axes + ([(axC, "C")] if axC is not None else [])):
                # zaxis included: on the cube that is PC1, and leaving it out let it sit
                # on panel B's row labels with the check reporting clean.
                for which in ("xaxis", "yaxis", "zaxis"):
                    if not hasattr(ax_, which):
                        continue
                    lb = getattr(ax_, which).label
                    if lb.get_text():
                        named.append((f"{nm} {which[0]} label", lb))
                # Tick labels too. Leaving them out made the check pass at a --scale where
                # panel B's x label and panel C's group labels visibly met: both sit on the
                # bottom margin, neither is an axis label, and the sweep called it clean.
                # mplot3d ticks are skipped -- their extents are not comparable to the 2-D
                # ones and the cube's day labels are annotations placed by hand anyway.
                if getattr(ax_, "name", "") == "3d":
                    continue
                for which in ("xaxis", "yaxis"):
                    axis_ = getattr(ax_, which)
                    v0, v1 = sorted(axis_.get_view_interval())
                    # get_ticklabels() returns ticks OUTSIDE the view limits too, and
                    # matplotlib never draws those. Their extents are stale, and left in
                    # they reported panel B colliding on a "100" tick that is not on the
                    # plate -- the forest only runs to +-67 -- and panel C on a "2.0" above
                    # its own ylim. Every remaining hit is then something a reader can see.
                    for loc_, t_ in zip(axis_.get_ticklocs(), axis_.get_ticklabels()):
                        if t_.get_text() and v0 - 1e-9 <= loc_ <= v1 + 1e-9:
                            named.append((f"{nm} {which[0]}tick {t_.get_text()[:14]!r}", t_))
            # Text3D.get_window_extent does NOT return display pixels -- it returns the
            # artist's DATA coordinates, which on this cube run to +-40000. Compared against
            # a 2-D artist's pixel box nothing can ever intersect, so the cube's weekday
            # labels were in the check and silently unable to fail it while "Thu" sat on
            # "early morning". Project the 3-D point by hand and build the box from a
            # measurement of the same string at the same size.
            def _box(nm_, ar_):
                if nm_.startswith("cube "):
                    x3, y3, z3 = ar_.get_position_3d()
                    x2, y2, _ = proj3d.proj_transform(x3, y3, z3, axB.get_proj())
                    px, py = axB.transData.transform((x2, y2))
                    probe = fig.text(0.5, 0.5, ar_.get_text(),
                                     fontsize=ar_.get_fontsize(),
                                     fontweight=ar_.get_fontweight())
                    pb = probe.get_window_extent(rend_)
                    probe.remove()
                    w_, h_ = pb.width, pb.height
                    return Bbox.from_bounds(px - w_ / 2, py - h_ / 2, w_, h_)
                return ar_.get_window_extent(rend_)

            hits = []
            for i in range(len(named)):
                for j in range(i + 1, len(named)):
                    bi = _box(*named[i])
                    bj = _box(*named[j])
                    if bi.overlaps(bj):
                        ov = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                        hits.append(f"{named[i][0]} x {named[j][0]} ({ov:.0f} px)")
            print(f"[overlap] {len(hits)} collision(s) among {len(named)} placed labels"
                  + ("".join("\n           " + h for h in hits) if hits else ""))

        out = args.out
        if args.theme == "both":
            stem, ext = out.rsplit(".", 1)
            out = f"{stem}_{theme}.{ext}"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        # bbox_inches="tight" computes the crop from artist extents, and mplot3d's axis
        # labels are not in them -- so panel B's "weekday plane 1" hangs below the crop line
        # that panel A's xlabel sets. Pad the crop by the amount the type grew. 0.1 at s == 1
        # is matplotlib's own savefig.pad_inches default, so existing exports are byte-identical.
        # The default pad grows with --scale because mplot3d's axis labels are not in the
        # tight bbox, so panel B's z label hangs below the crop line that panel A's xlabel
        # sets, and it grows worse as the type does. At --scale 2.15 that is 1.25 in of
        # white on EVERY side. --pad-inches overrides it; the labels it was protecting are
        # short PC names tucked in by --b-labelpad here, so most of it is buying nothing.
        save_kw = dict(bbox_inches="tight", dpi=300,
                       pad_inches=(args.pad_inches if args.pad_inches is not None
                                   else 0.1 + (s - 1)))
        save_kw.update({"facecolor": bg} if args.opaque else {"transparent": True})
        fig.savefig(out, **save_kw)
        print(f"[talk] wrote {out}")
        if args.also_png:
            p = os.path.splitext(out)[0] + ".png"
            fig.savefig(p, **save_kw)
            print(f"[talk] wrote {p}")
        plt.close(fig)

    print(f"\n[A] adjacent-day step: nearest {STEP_MIN:.2f}°, furthest {STEP_MAX:.2f}°")
    print(f"[A] widest kept shift: {widest:.2f}°  "
          f"({widest / STEP_MIN * 100:.1f}% of the nearest day step)")
    # "placebo" here is the merged family, so it is checked against BOTH of the npz's
    # placebo sub-families at once -- otherwise the kept rows would be compared to only
    # half the pool they were drawn from and the range check would prove nothing.
    SRC = {"clock": ("clock",), "word": ("word",),
           "placebo": ("placebo_very", "placebo_varied")}
    for f_, src in SRC.items():
        v = np.array([stat[k][0] for k, _, ff, _ in rows if ff == f_])
        pool = dict.fromkeys(lab[np.isin(fam, src)])
        full = np.array([np.array([dth[(lab == m) & (day == d)].mean()
                                   for d in range(7)]).mean() for m in pool])
        print(f"[A] {f_:9s} kept n={len(v):2d} {v.min():+.2f}..{v.max():+.2f}°   "
              f"| all n={len(full):2d} {full.min():+.2f}..{full.max():+.2f}°")


if __name__ == "__main__":
    main()
