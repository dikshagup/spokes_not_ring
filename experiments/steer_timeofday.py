#!/usr/bin/env python
"""Steering run for figure 2: edit the ring plane at each layer and read the predicted hour.

Fits a weekday plane per layer from neutral prompts, decomposes the late-early modifier
displacement into in-plane and off-plane parts, and clamps each in turn.

Writes experiments/results/steer_timeofday/<model>/. See
repro_fig2_timeofday_with_steering.sh.
"""
from __future__ import annotations

import argparse, json, math, os, sys
import numpy as np
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, _SRC)

from weekday_manifold.model import load_model
from weekday_manifold.timeofday.prompts import CARRIERS, validate_carriers
from weekday_manifold.timeofday.geometry import fit_ring_frame
from weekday_manifold.manifold.spline import PeriodicSpline

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
N_DAYS = 7
DAY_STEP_DEG = 360.0 / N_DAYS          # 51.43
TAIL = "at around"

# hour -> the three surface forms we sum over. Measured preference: 7pm > 7 a.m. > 7 am;
# the repo's spelled-out clock_word ("seven pm") is worst of nine and is NOT used here.
def hour_forms(h):
    hh = 12 if h % 12 == 0 else h % 12
    ap = "am" if h < 12 else "pm"
    dotted = "a.m." if h < 12 else "p.m."
    return [f" {hh}{ap}", f" {hh} {ap}", f" {hh} {dotted}"]


# --------------------------------------------------------------------- prompts
def prompt(carrier, mod, day, tail=TAIL):
    m = f" {mod}" if mod else ""
    return f"{carrier}{m} on {day} {tail}"


# ------------------------------------------------------------------- scoring
class Scorer:
    """Teacher-forced hour scoring with an optional residual-stream edit."""

    def __init__(self, model, layer, prepend_bos=True):
        self.m = model
        self.layer = layer
        self.hook = f"blocks.{layer}.hook_resid_post"
        self.bos = prepend_bos
        self.dev = model.cfg.device
        self.dtype = model.W_E.dtype
        self._cand_cache = {}

    def cand_ids(self):
        key = "cands"
        if key not in self._cand_cache:
            forms, owner = [], []
            for h in range(24):
                for f in hour_forms(h):
                    forms.append(self.m.to_tokens(f, prepend_bos=False)[0].tolist())
                    owner.append(h)
            self._cand_cache[key] = (forms, np.array(owner))
        return self._cand_cache[key]

    def weekday_pos(self, text, day):
        toks = self.m.to_tokens(text, prepend_bos=self.bos)[0]
        did = self.m.to_tokens(" " + day, prepend_bos=False)[0]
        # last sub-token of the day word (multi-token weekdays on Llama)
        n = len(did)
        for i in range(len(toks) - n, -1, -1):
            if torch.equal(toks[i:i + n], did):
                return i + n - 1
        raise SystemExit(f"weekday {day!r} not found in {text!r}")

    @torch.no_grad()
    def score(self, text, day, vec=None, allpos=False):
        """-> (logp_hour[24], pos, prefix_logits_last)."""
        pre = self.m.to_tokens(text, prepend_bos=self.bos)[0].tolist()
        P = self.weekday_pos(text, day)
        forms, owner = self.cand_ids()
        seqs = [pre + f for f in forms]
        mx = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), mx), dtype=torch.long, device=self.dev)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s, device=self.dev)
        if vec is None:
            logits = self.m(ids, return_type="logits")
        else:
            band = vec if isinstance(vec, dict) else {self.layer: vec}
            def mk(vv):
                v = torch.as_tensor(np.asarray(vv), device=self.dev, dtype=self.dtype)
                def fn(resid, hook):
                    if allpos:
                        resid[:, P:, :] = resid[:, P:, :] + v
                    else:
                        resid[:, P, :] = resid[:, P, :] + v
                    return resid
                return fn
            logits = self.m.run_with_hooks(
                ids, return_type="logits",
                fwd_hooks=[(f"blocks.{L}.hook_resid_post", mk(vv))
                           for L, vv in sorted(band.items())])
        lg = torch.log_softmax(logits.float(), -1)
        n_pre = len(pre)
        out = np.full(len(seqs), -np.inf)
        for j, s in enumerate(seqs):
            idx = torch.arange(n_pre, len(s), device=self.dev)
            out[j] = float(lg[j, idx - 1, torch.tensor(s[n_pre:], device=self.dev)].sum())
        # logsumexp over the surface forms of each hour
        per_hour = np.array([_lse(out[owner == h]) for h in range(24)])
        # per-form view [n_forms, 24] -- candidates are built hour-major, form-minor
        self.last_per_form = out.reshape(24, -1).T
        # next-token distribution at the readout slot, for entropy / KL (row 0, clean prefix end)
        tail_lp = lg[0, n_pre - 1].float().cpu().numpy()
        return per_hour, P, tail_lp


def _lse(a):
    m = np.max(a)
    return float(m + np.log(np.exp(a - m).sum()))


# ------------------------------------------------------------------- summaries
def summarise(logp_hour):
    """circular mean (h), log-odds am/pm, concentration, from 24 hour log-probs."""
    p = np.exp(logp_hour - _lse(logp_hour))
    ang = 2 * np.pi * np.arange(24) / 24.0
    R = complex((p * np.cos(ang)).sum(), (p * np.sin(ang)).sum())
    cm = (math.atan2(R.imag, R.real) % (2 * np.pi)) * 24 / (2 * np.pi)
    am = p[:12].sum()
    return dict(circ_mean=cm, conc=abs(R),
                logodds=float(np.log(max(am, 1e-12)) - np.log(max(1 - am, 1e-12))),
                p=p)


def circ_diff(a, b):
    """signed circular difference a-b in hours, wrapped to (-12, 12]."""
    return (a - b + 12.0) % 24.0 - 12.0


# ------------------------------------------------------------------- geometry
class Geom:
    """Ring frame + spline + modifier directions at one layer, from NEUTRAL prompts only."""

    def __init__(self, A_neutral, days, A_early, A_late, A_plac, seed=0):
        self.F = fit_ring_frame(A_neutral, days)
        self.P = self.F.plane                                   # [d, 2]
        self.mu = self.F.mu
        self.C = self.F.centroids                               # [7, d]
        self.R = self.F.radius
        self.orient = self.F.orient
        self.spline = PeriodicSpline(self.C)                    # knots at d/7
        # per-day modifier directions (position-matched: both arms are 1-word modifiers)
        self.d_time = np.stack([A_late[days == d].mean(0) - A_early[days == d].mean(0)
                                for d in range(N_DAYS)])
        self.d_plac = np.stack([A_plac[days == d].mean(0) - A_neutral[days == d].mean(0)
                                for d in range(N_DAYS)])
        # norm-match the placebo direction to the time direction, per day
        for d in range(N_DAYS):
            n = np.linalg.norm(self.d_plac[d])
            if n > 0:
                self.d_plac[d] *= np.linalg.norm(self.d_time[d]) / n
        rng = np.random.default_rng(seed)
        # Two random unit directions per day, fixed by seed:
        #   null_off  -- orthogonal to the FULL weekday span: "is the weekday code
        #                involved at all, or would any edit of this size do?"
        #   null_span -- INSIDE the weekday span: the harder control, "is it this
        #                particular arc, or would any move within the weekday code do?"
        self.null_off, self.null_span = [], []
        for d in range(N_DAYS):
            v = rng.normal(size=self.mu.shape[0])
            v -= self.F.span @ (self.F.span.T @ v)
            self.null_off.append(v / np.linalg.norm(v))
            w = self.F.span @ rng.normal(size=self.F.span.shape[1])
            self.null_span.append(w / np.linalg.norm(w))

    # ---- in-plane helpers
    def coords(self, h):
        return (h - self.mu) @ self.P                            # [2]

    def rot_delta(self, h, theta_deg):
        """rigid rotation of the in-plane component about mu; orthogonal complement untouched."""
        t = np.radians(theta_deg) * self.orient
        c = self.coords(h)
        Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        return (Rm @ c - c) @ self.P.T

    def spline_delta(self, day, delta_days):
        u0 = day / N_DAYS
        return self.spline.forward(u0 + delta_days / N_DAYS) - self.spline.forward(u0)

    def chord_delta(self, day, delta_days):
        return delta_days * (self.C[(day + 1) % N_DAYS] - self.C[day])

    def radial_unit(self, h):
        """unit vector along the in-plane radial direction at h (outward)."""
        c = self.coords(h)
        n = np.linalg.norm(c)
        if n == 0:
            return np.zeros_like(self.mu)
        v = (c / n) @ self.P.T
        return v / np.linalg.norm(v)

    def in_plane(self, v):
        return (v @ self.P) @ self.P.T

    def day_readout(self, h):
        """nearest centroid + ring angle (degrees, oriented so + = next day)."""
        d = int(np.argmin(((self.C - h) ** 2).sum(1)))
        c = self.coords(h)
        return d, float(np.degrees(np.arctan2(self.orient * c[1], c[0])))


# ---------------------------------------------------------------------- main
def capture_layers(layers, capture_to, bands, n_layers):
    """Which residual-stream layers the capture pass hooks, clamped to the model."""
    wanted = (set(layers) | set(range(min(layers), capture_to + 1))
              | set(range(min(layers), max(layers) + max(bands))))
    return sorted(L for L in wanted if L < n_layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(32)))
    ap.add_argument("--headline-layer", type=int, default=-1,
                help="-1 = choose by measured ceiling specificity")
    ap.add_argument("--n-carriers", type=int, default=12)
    ap.add_argument("--max-conc", type=float, default=0.55,
                    help="drop carriers whose neutral hour distribution is more concentrated than this (no headroom)")
    ap.add_argument("--thetas", type=float, nargs="+",
                    default=[-51.43, -25.71, -15, -8, -4, -2, 2, 4, 8, 15, 25.71, 51.43])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--capture-to", type=int, default=28,
                    help="capture (and build geometry) up to this layer, even if pass 1 sweeps a narrower range -- bands may reach deeper and it saves a re-capture for any follow-up")
    ap.add_argument("--bands", type=int, nargs="+", default=[1, 3, 5, 9],
                    help="hold the steer across this many consecutive layers")
    ap.add_argument("--out", default="experiments/results/steer_timeofday")
    ap.add_argument("--smoke", action="store_true", help="2 carriers, 1 day, 1 layer")
    args = ap.parse_args()

    validate_carriers()
    outdir = os.path.join(args.out, args.model.replace("/", "_"))
    os.makedirs(outdir, exist_ok=True)

    print(f"[steer] loading {args.model} ...", flush=True)
    model = load_model(args.model, device="cuda", fold_ln=True,
                       center_writing_weights=True, center_unembed=True, dtype="bfloat16")
    print("[steer] loaded", flush=True)

    layers = args.layers if not args.smoke else [10, 13]
    cap_layers = capture_layers(layers, args.capture_to, args.bands, model.cfg.n_layers)
    days = list(range(N_DAYS)) if not args.smoke else [2]

    # ---------------------------------------------------------------- E0 screen
    sc28 = Scorer(model, layers[-1])
    print("\n[E0] screening carriers (cross-form agreement + concentration)", flush=True)
    screen = []
    for c in CARRIERS:
        # ONE batched call gives both the pooled readout and the per-form breakdown,
        # so the cross-form screen costs nothing beyond the readout itself.
        lp_all, _, _ = sc28.score(prompt(c, "", "Wednesday"), "Wednesday")
        cms = [summarise(f)["circ_mean"] for f in sc28.last_per_form]
        spread = max(abs(circ_diff(a, b)) for a in cms for b in cms)
        s = summarise(lp_all)
        screen.append(dict(carrier=c, form_spread=float(spread), conc=float(s["conc"]),
                           circ_mean=float(s["circ_mean"])))
        print(f"   {c:<24} form-spread={spread:5.2f}h  conc={s['conc']:.3f}  "
              f"cm={s['circ_mean']:5.2f}h", flush=True)
    # Rank by cross-form stability, but CAP concentration. Stable-across-forms turns out
    # to correlate with confident, so ranking on spread alone selects exactly the
    # time-locked carriers ("The office closes", conc 0.81) that have no headroom to move
    # -- which halved the anchor on the first run. The cap is the fix.
    elig = [r for r in screen if r["conc"] <= args.max_conc]
    elig.sort(key=lambda r: r["form_spread"])
    keep = [r["carrier"] for r in elig[:args.n_carriers]]
    print(f"[E0] concentration cap {args.max_conc}: {len(elig)}/{len(screen)} eligible")
    if args.smoke:
        keep = keep[:2]
    print(f"[E0] kept {len(keep)}/{len(CARRIERS)} carriers (smallest cross-form spread); "
          f"dropped {len(CARRIERS) - len(keep)}", flush=True)
    print(f"[E0] kept: {keep}", flush=True)

    # ------------------------------------------------------- capture activations
    print("\n[cap] capturing weekday-token resid_post for neutral/early/late/placebo",
          flush=True)
    MODS = {"neutral": "", "early": "early", "late": "late", "placebo": "quietly"}
    acts = {L: {k: [] for k in MODS} for L in cap_layers}
    lab_day = []
    for ci, c in enumerate(keep):
        for d in range(N_DAYS):
            for k, mod in MODS.items():
                text = prompt(c, mod, DAYS[d])
                toks = model.to_tokens(text, prepend_bos=True)
                P = sc28.weekday_pos(text, DAYS[d])
                store = {}
                with torch.no_grad():
                    model.run_with_hooks(
                        toks, return_type=None,
                        fwd_hooks=[(f"blocks.{L}.hook_resid_post",
                                    (lambda LL: lambda a, hook: store.__setitem__(
                                        LL, a[0, P].detach().float().cpu().numpy()))(L))
                                   for L in cap_layers])
                for L in cap_layers:
                    acts[L][k].append(store[L])
            if ci == 0:
                pass
        lab_day.extend(list(range(N_DAYS)))
    lab_day = np.array(lab_day)
    A = {L: {k: np.stack(v) for k, v in acts[L].items()} for L in cap_layers}
    print(f"[cap] {len(lab_day)} (carrier,day) cells x {len(MODS)} modifiers "
          f"x {len(layers)} layers", flush=True)

    G = {L: Geom(A[L]["neutral"], lab_day, A[L]["early"], A[L]["late"], A[L]["placebo"])
         for L in cap_layers}
    for L in layers:
        g = G[L]
        dt = g.d_time.mean(0)
        ip = np.linalg.norm(g.in_plane(dt)) / g.R
        op = np.linalg.norm(dt - g.in_plane(dt)) / g.R
        print(f"[geom] L{L}: ring R={g.R:.3f}  evr_plane={g.F.evr_plane:.3f}  "
              f"|d_time| in-plane={ip:.3f}R  off-plane={op:.3f}R  ratio={op/max(ip,1e-9):.1f}x",
          flush=True)

    # sanity: spline at integer step == chord to the next centroid
    g = G[layers[len(layers)//2]]
    err = np.abs(g.spline_delta(0, 1.0) - (g.C[1] - g.C[0])).max()
    print(f"[check] spline(delta=1) vs chord: max abs diff {err:.2e} (should be ~0)", flush=True)

    # ---- ARE rot AND spline THE SAME OPERATOR? measured, not assumed ---------------
    # They are not, and the three reasons are separable:
    #   (a) the ring is not uniform -- the seven adjacent-day angular gaps differ, so a
    #       rigid rotation by 360/7 does NOT land on the next day, whereas spline(1) does;
    #   (b) the spline runs through the centroids in the FULL weekday span, so it leaves
    #       the 2-D plane that `rot` is confined to;
    #   (c) `rot` turns the PROMPT's own in-plane component (radius |c(h)|), the spline
    #       moves along the CENTROID curve (radius R) -- different lever arms.
    gaps = np.degrees(np.diff(np.sort(g.F.theta)))
    gaps = np.append(gaps, 360.0 - gaps.sum())
    r_prompt = np.linalg.norm(A[layers[len(layers)//2]]["neutral"] @ g.P, axis=1).mean()
    s1 = g.spline_delta(0, 1.0)
    off1 = np.linalg.norm(s1 - g.in_plane(s1)) / np.linalg.norm(s1)
    print(f"[rot-vs-spline] adjacent-day gaps: min {gaps.min():.2f} deg, "
          f"max {gaps.max():.2f} deg, mean {gaps.mean():.2f} deg "
          f"(uniform would be {DAY_STEP_DEG:.2f})")
    print(f"[rot-vs-spline] prompt in-plane radius {r_prompt:.3f} vs ring radius "
          f"{g.R:.3f}  ({r_prompt/g.R:.2f}x)")
    print(f"[rot-vs-spline] spline(1) off-plane fraction {off1:.3f} "
          f"(rot is 0.000 by construction)")
    for th in (4.0, 15.0, 51.43):
        rel, nr, ns = [], [], []
        for d in range(N_DAYS):
            hh = A[layers[len(layers)//2]]["neutral"][d]
            a_, b_ = g.rot_delta(hh, th), g.spline_delta(d, th / DAY_STEP_DEG)
            rel.append(np.linalg.norm(a_ - b_) / max(np.linalg.norm(b_), 1e-12))
            nr.append(np.linalg.norm(a_)); ns.append(np.linalg.norm(b_))
        print(f"[rot-vs-spline] theta={th:6.2f} deg: |rot|={np.mean(nr):7.3f}  "
              f"|spline|={np.mean(ns):7.3f}  |rot-spline|/|spline| = {np.mean(rel):.3f}")
    # sanity: +theta moves the ring angle toward the next day
    h0 = A[layers[len(layers)//2]]["neutral"][0]
    d0, a0 = g.day_readout(h0)
    _, a1 = g.day_readout(h0 + g.rot_delta(h0, +10.0))
    print(f"[check] rot(+10 deg) moves ring angle {a0:+.2f} -> {a1:+.2f} "
          f"(delta {circ_deg(a1 - a0):+.2f}, should be about +10)", flush=True)

    # ------------------------------------------------------------- conditions
    def build_conditions(g, h, day):
        """-> list of (name, param, delta_vector)"""
        out = []
        for th in args.thetas:
            sp = g.spline_delta(day, th / DAY_STEP_DEG)
            n = float(np.linalg.norm(sp))
            sgn = float(np.sign(th))
            dr = g.rot_delta(h, th)
            n_rot = float(np.linalg.norm(dr))
            out.append(("spline", th, sp))
            out.append(("rot", th, dr * (n / n_rot) if n_rot > 0 else dr))
            out.append(("radial", th, sgn * n * g.radial_unit(h)))
            out.append(("null_span", th, sgn * n * g.null_span[day]))
            out.append(("null_off", th, sgn * n * g.null_off[day]))
            out.append(("rot_true", th, dr))
            # `chord` (the §8 straight-line operator) dropped per user decision. It
            # survives only as the spline correctness check, where chord(1) == spline(1).
        for al in args.alphas:
            dt = g.d_time[day]
            out.append(("delta_full", al, al * dt))
            out.append(("delta_in", al, al * g.in_plane(dt)))
            out.append(("delta_off", al, al * (dt - g.in_plane(dt))))
            out.append(("delta_placebo", al, al * g.d_plac[day]))
        return out

    # ------------------------------------------------------------------ run
    rows = []
    scorers = {L: Scorer(model, L) for L in cap_layers}

    def emit(L, c, d, name, param, vec, sc, g, s0, tail0, h, allpos=False):
        lp, _, tail = sc.score(text_of(c, d), DAYS[d], vec, allpos=allpos)
        s = summarise(lp)
        dd, ang = g.day_readout(h + (vec[min(vec)] if isinstance(vec, dict) else vec))
        p0 = np.exp(tail0 - _lse(tail0))
        v0 = vec[min(vec)] if isinstance(vec, dict) else vec
        rows.append(dict(layer=L, carrier=c, day=d, cond=name, param=float(param),
                         dnorm=float(np.linalg.norm(v0)),
                         band=len(vec) if isinstance(vec, dict) else 1,
                         d_cm=circ_diff(s["circ_mean"], s0["circ_mean"]),
                         d_logodds=s["logodds"] - s0["logodds"],
                         cm=s["circ_mean"], conc=s["conc"], day_dec=dd, ring_ang=ang,
                         ent=float(-(np.exp(tail) * tail).sum()),
                         kl=float((p0 * (tail0 - tail)).sum())))

    def text_of(c, d):
        return prompt(c, "", DAYS[d])

    def clean(L, c, d):
        sc = scorers[L]
        lp0, _, tail0 = sc.score(text_of(c, d), DAYS[d])
        return summarise(lp0), tail0

    # --- PASS 1: dense layer sweep of the CEILING + a norm-matched random control.
    # The point of steering is to edit where the weekday token is still READ; the earlier
    # run showed that is not true at L25+, where even a 22x-residual-norm edit does
    # nothing. So the steering layer is chosen by measurement, at every layer, not picked.
    print(f"\n[pass1] ceiling sweep over {len(layers)} layers "
          f"(delta_full a=1 + norm-matched random)", flush=True)
    for L in layers:
        g, sc = G[L], scorers[L]
        for ci, c in enumerate(keep):
            for d in days:
                s0, tail0 = clean(L, c, d)
                h = A[L]["neutral"][ci * N_DAYS + d]
                dt = g.d_time[d]
                emit(L, c, d, "delta_full", 1.0, dt, sc, g, s0, tail0, h)
                emit(L, c, d, "sweep_rand", 1.0,
                     np.linalg.norm(dt) * g.null_off[d], sc, g, s0, tail0, h)
        print(f"   L{L} done ({len(rows)} rows)", flush=True)

    # pick the headline layer: largest |ceiling| that also beats its random control
    import collections
    best, best_v = None, -1.0
    for L in layers:
        sel = [r for r in rows if r["layer"] == L]
        ce = np.mean([r["d_cm"] for r in sel if r["cond"] == "delta_full"])
        rd = np.mean([r["d_cm"] for r in sel if r["cond"] == "sweep_rand"])
        print(f"[pass1] L{L:>2}: ceiling {ce:+.3f} h   random {rd:+.3f} h   "
              f"specificity {abs(ce) - abs(rd):+.3f}")
        if abs(ce) - abs(rd) > best_v:
            best_v, best = abs(ce) - abs(rd), L
    HEAD = best if args.headline_layer < 0 else args.headline_layer
    print(f"[pass1] headline layer = L{HEAD} (specificity {best_v:+.3f} h)", flush=True)

    # --- PASS 2: full dose-response at the headline layer.
    g, sc = G[HEAD], scorers[HEAD]
    print(f"\n[pass2] full dose-response at L{HEAD}", flush=True)
    for ci, c in enumerate(keep):
        for d in days:
            s0, tail0 = clean(HEAD, c, d)
            h = A[HEAD]["neutral"][ci * N_DAYS + d]
            for k in ("early", "late"):
                lpa, _, _ = sc.score(prompt(c, MODS[k], DAYS[d]), DAYS[d])
                sa = summarise(lpa)
                rows.append(dict(layer=HEAD, carrier=c, day=d, cond=f"anchor_{k}",
                                 param=0.0, dnorm=0.0,
                                 d_cm=circ_diff(sa["circ_mean"], s0["circ_mean"]),
                                 d_logodds=sa["logodds"] - s0["logodds"],
                                 cm=sa["circ_mean"], conc=sa["conc"], day_dec=d,
                                 ring_ang=0.0, ent=0.0, kl=0.0, band=0))
            for name, param, vec in build_conditions(g, h, d):
                emit(HEAD, c, d, name, param, vec, sc, g, s0, tail0, h)
            # all-positions variants: inject at every position from the weekday onward,
            # which bypasses attention transport entirely. Separates "the code is not
            # read" from "the code is read, but only before the information has moved".
            emit(HEAD, c, d, "spline_allpos", 51.43, g.spline_delta(d, 1.0),
                 sc, g, s0, tail0, h, allpos=True)
            emit(HEAD, c, d, "delta_full_allpos", 1.0, g.d_time[d],
                 sc, g, s0, tail0, h, allpos=True)
            # --- HOLD the steer across a band of layers. Motivation: the ring is
            # error-correcting near the centroids (experiments/jacobian_error_correction.py
            # measures the staircase), so a single-layer edit can simply be snapped back by
            # the next block. Re-imposing it at every layer of a band removes that escape
            # route. Each layer uses ITS OWN geometry, so "hold the steer" means "keep the
            # representation displaced by the same delta along that layer's manifold",
            # not "add a stale L0 vector to a deeper layer".
            for w in args.bands:
                bl = [L for L in cap_layers if HEAD <= L < HEAD + w]
                if len(bl) < w:
                    continue
                for th in (4.0, DAY_STEP_DEG):
                    emit(HEAD, c, d, "spline_band", th,
                         {L: G[L].spline_delta(d, th / DAY_STEP_DEG) for L in bl},
                         sc, g, s0, tail0, h)
                    emit(HEAD, c, d, "null_off_band", th,
                         {L: np.linalg.norm(G[L].spline_delta(d, th / DAY_STEP_DEG))
                            * G[L].null_off[d] for L in bl},
                         sc, g, s0, tail0, h)
                emit(HEAD, c, d, "delta_full_band", 1.0,
                     {L: G[L].d_time[d] for L in bl}, sc, g, s0, tail0, h)
        print(f"   carrier {ci+1}/{len(keep)} done ({len(rows)} rows)", flush=True)

    # -------------------------------------------------------------- persist
    import csv
    csv_path = os.path.join(outdir, "steer_rows.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    meta = dict(model=args.model, layers=layers, cap_layers=cap_layers, headline_layer=int(HEAD),
                carriers_kept=keep, screen=screen, thetas=args.thetas, alphas=args.alphas,
                tail=TAIL, n_rows=len(rows), day_step_deg=DAY_STEP_DEG,
                geom={str(L): dict(radius=float(G[L].R), evr_plane=float(G[L].F.evr_plane),
                                   orient=int(G[L].orient)) for L in layers})
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[saved] {csv_path}  ({len(rows)} rows)")
    print(f"[saved] {os.path.join(outdir, 'meta.json')}")
    return 0



def circ_deg(x):
    return (x + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
