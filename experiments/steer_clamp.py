#!/usr/bin/env python
"""Clamp hooks and frames: assert an absolute ring angle at every layer, rather than add to it.

Used by steer_timeofday.py and steer_clamp_matched_pairs.py; no main.
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weekday_manifold.model import load_model
from weekday_manifold.timeofday.geometry import fit_ring_frame
from steer_timeofday import (Scorer, summarise, prompt, circ_diff, circ_deg,
                             DAYS, N_DAYS, DAY_STEP_DEG)


# --------------------------------------------------------------------- frames
MODS = {"neutral": "", "early": "early", "late": "late", "placebo": "quietly"}


def fit_pos_frames(model, sc, carriers, layers, offsets, verbose=True):
    """Ring frame per (layer, offset), fitted from NEUTRAL prompts only."""
    store = {(m, L, o): [] for m in MODS for L in layers for o in offsets}
    days = []
    for c in carriers:
        for d in range(N_DAYS):
            for m, word in MODS.items():
                text = prompt(c, word, DAYS[d])
                P = sc.weekday_pos(text, DAYS[d])
                toks = model.to_tokens(text, prepend_bos=True)
                n_tok = toks.shape[1]
                grab = {}
                with torch.no_grad():
                    model.run_with_hooks(
                        toks, return_type=None,
                        fwd_hooks=[(f"blocks.{L}.hook_resid_post",
                                    (lambda LL: lambda a, hook: grab.__setitem__(
                                        LL, a[0].detach().float().cpu().numpy()))(L))
                                   for L in layers])
                for L in layers:
                    for o in offsets:
                        if P + o >= n_tok:
                            raise SystemExit(
                                f"offset {o} runs past the end of {text!r} "
                                f"(weekday at {P}, {n_tok} tokens)")
                        store[(m, L, o)].append(grab[L][P + o])
            days.append(d)
    days = np.array(days)
    acts = {k: np.stack(v) for k, v in store.items()}
    frames = {(L, o): fit_ring_frame(acts[("neutral", L, o)], days)
              for L in layers for o in offsets}

    if verbose:
        print("\n[frames] is there a weekday ring at each position? "
              "(evr_plane, and do the 7 days sit in cyclic order?)")
        print(f"    {'L':>3} " + "".join(f"{'off=' + str(o):>22}" for o in offsets))
        for L in layers:
            cells = []
            for o in offsets:
                f = frames[(L, o)]
                hc = f.heptagon_check()
                cells.append(f"{f.evr_plane:.2f} R={f.radius:6.3f} "
                             f"{'cyc' if hc['in_cyclic_order'] else 'NOT':>3}")
            print(f"    {L:>3} " + "".join(f"{c:>22}" for c in cells))
    return frames, acts, days


# ---------------------------------------------------------------------- clamp
def clamp_hook(frame, target_deg, positions, device, dtype):
    """Hook that SETS the in-plane angle at `positions` to `target_deg` (absolute)."""
    mu = torch.as_tensor(frame.mu, device=device, dtype=torch.float32)
    Pl = torch.as_tensor(frame.plane, device=device, dtype=torch.float32)
    o = float(frame.orient)
    tgt = torch.as_tensor(np.radians(np.asarray(target_deg, dtype=float)),
                          device=device, dtype=torch.float32)          # [n_pos]
    pos = torch.as_tensor(positions, device=device, dtype=torch.long)

    def fn(resid, hook):
        x = resid[:, pos, :].float()                                   # [B, n_pos, d]
        c = (x - mu) @ Pl                                              # [B, n_pos, 2]
        r = c.norm(dim=-1)                                             # [B, n_pos]
        c_new = torch.stack([r * torch.cos(tgt), o * r * torch.sin(tgt)], dim=-1)
        resid[:, pos, :] = (x + (c_new - c) @ Pl.T).to(resid.dtype)
        return resid

    return fn


def clamp_dir_hook(u, target_coef, positions, device, dtype, record=None, key=None):
    """SET the component along unit direction `u` to `target_coef`. The linear sibling of
    `clamp_hook`: that one asserts an angle in a 2-plane, this one asserts a scalar coordinate
    along a line. Everything orthogonal to `u` is untouched."""
    uu = torch.as_tensor(np.asarray(u, dtype=float), device=device, dtype=torch.float32)
    uu = uu / uu.norm()
    tgt = torch.as_tensor(float(target_coef), device=device, dtype=torch.float32)
    pos = torch.as_tensor(positions, device=device, dtype=torch.long)

    def fn(resid, hook):
        x = resid[:, pos, :].float()                                   # [B, n_pos, d]
        cur = x @ uu                                                   # [B, n_pos]
        delta = (tgt - cur).unsqueeze(-1) * uu                         # [B, n_pos, d]
        if record is not None:
            record[key] = float(delta[0].norm())
        resid[:, pos, :] = (x + delta).to(resid.dtype)
        return resid

    return fn


def clamp_fwd_hooks(frames, ref_deg, dtheta, layers, offsets, P, device, dtype):
    """fwd_hooks clamping every (layer, offset) to ref_deg[(L, off)] + dtheta."""
    hooks = []
    for L in layers:
        for o in offsets:
            hooks.append((f"blocks.{L}.hook_resid_post",
                          clamp_hook(frames[(L, o)], [ref_deg[(L, o)] + dtheta],
                                     [P + o], device, dtype)))
    return hooks


def clean_ref_angles(model, sc, text, day, frames, layers, offsets):
    """Clean ring angle (deg) of this prompt at every (layer, offset). The clamp target."""
    P = sc.weekday_pos(text, day)
    grab = {}
    with torch.no_grad():
        model.run_with_hooks(
            model.to_tokens(text, prepend_bos=True), return_type=None,
            fwd_hooks=[(f"blocks.{L}.hook_resid_post",
                        (lambda LL: lambda a, hook: grab.__setitem__(
                            LL, a[0].detach().float().cpu().numpy()))(L))
                       for L in layers])
    return {(L, o): float(np.degrees(frames[(L, o)].angle_of(grab[L][P + o])[0]))
            for L in layers for o in offsets}, P


# ----------------------------------------------------------------- verification
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/results/steer_timeofday/meta-llama_Llama-3.1-8B")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--start-layer", type=int, default=2)
    ap.add_argument("--end-layer", type=int, default=28)
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 1, 2],
                    help="token offsets from the weekday token to clamp")
    ap.add_argument("--theta", type=float, default=DAY_STEP_DEG)
    ap.add_argument("--n-carriers", type=int, default=12)
    ap.add_argument("--cells", type=int, default=6)
    ap.add_argument("--verify", action="store_true")
    # THE DOSE THAT MATTERS. A full day step is not a time-of-day steer: it maps the
    # representation to the SAME relative position in the next day's basin, so any
    # coordinate defined relative to the nearest centroid is invariant to it by
    # construction. The informative range is the small in-plane rotation a real modifier
    # actually induces -- measured at L12 as +8.07 deg for a full modifier (alpha=1) and
    # +4.21 deg for half a modifier -- swept out to half a day (25.71 deg), which is the
    # largest rotation that still cannot be dismissed as "you just moved to the next day".
    ap.add_argument("--sweep", action="store_true", help="dose-response over --thetas")
    ap.add_argument("--thetas", type=float, nargs="+",
                    default=[-25.71, -15.0, -8.07, -4.0, -2.0, -1.0, 0.0,
                             1.0, 2.0, 4.0, 8.07, 15.0, 25.71])
    ap.add_argument("--mod-deg", type=float, default=8.07,
                    help="the modifier's own realized in-plane rotation, for the annotation")
    ap.add_argument("--out", default="figures/clamp_theta_sweep.png")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.dir, "meta.json")))
    keep = meta["carriers_kept"][:args.n_carriers]
    LAYERS = list(range(args.start_layer, args.end_layer + 1))
    OFF = args.offsets

    model = load_model(args.model, device="cuda", fold_ln=True,
                       center_writing_weights=True, center_unembed=True, dtype="bfloat16")
    sc = Scorer(model, LAYERS[0])
    dev, dt = model.cfg.device, model.W_E.dtype

    frames, acts, days = fit_pos_frames(model, sc, keep, LAYERS, OFF)

    if args.sweep:
        return sweep(model, sc, frames, keep, LAYERS, OFF, args)

    cells = [(c, d) for c in keep for d in range(N_DAYS)][:args.cells]

    # ---- 1. IDEMPOTENCE: T(T(x)) == T(x), to numerical precision -------------------
    print("\n[verify] idempotence of the clamp (max |T(T(x)) - T(x)| over the batch)")
    c0, d0 = cells[0]
    text0 = prompt(c0, "", DAYS[d0])
    ref0, P0 = clean_ref_angles(model, sc, text0, DAYS[d0], frames, LAYERS, OFF)
    L_t, o_t = LAYERS[0], OFF[0]
    x = torch.as_tensor(acts[("neutral", L_t, o_t)][:8], device=dev, dtype=dt).unsqueeze(1)
    h = clamp_hook(frames[(L_t, o_t)], [ref0[(L_t, o_t)] + args.theta], [0], dev, dt)
    once = h(x.clone(), None)
    twice = h(once.clone(), None)
    print(f"    once vs twice: {float((twice - once).abs().max()):.3e}   "
          f"(clamp is {'IDEMPOTENT' if float((twice-once).abs().max()) < 1e-2 else 'NOT idempotent'})")
    ang_once = np.degrees(frames[(L_t, o_t)].angle_of(once[:, 0].float().cpu().numpy()))
    print(f"    target {ref0[(L_t, o_t)] + args.theta:+8.2f} deg -> achieved "
          f"{ang_once.mean():+8.2f} deg  (spread {ang_once.std():.2e})")

    # ---- 2. DOES IT HOLD? realized angle at every layer, under the clamp ------------
    print(f"\n[verify] does the clamp HOLD the target through the stack? "
          f"dtheta={args.theta:+.2f} deg, offsets {OFF}, L{LAYERS[0]}..L{LAYERS[-1]}")

    @torch.no_grad()
    def trace_clamped(text, day, dtheta, offsets):
        """-> (pre_deg, post_deg, dnorm) per layer at offset 0, under the clamp."""
        ref, P = clean_ref_angles(model, sc, text, day, frames, LAYERS, offsets)
        pre, post, dn = {}, {}, {}

        def mk(L, o):
            base = clamp_hook(frames[(L, o)], [ref[(L, o)] + dtheta], [P + o], dev, dt)
            def fn(resid, hook):
                if o == 0:
                    pre[L] = resid[0, P].detach().float().cpu().numpy().copy()
                before = resid[:, P + o, :].detach().float().clone()
                out = base(resid, hook)
                dn[(L, o)] = float((out[0, P + o].float() - before[0]).norm())
                if o == 0:
                    post[L] = out[0, P].detach().float().cpu().numpy().copy()
                return out
            return fn

        model.run_with_hooks(
            model.to_tokens(text, prepend_bos=True), return_type=None,
            fwd_hooks=[(f"blocks.{L}.hook_resid_post", mk(L, o))
                       for L in LAYERS for o in offsets])
        ang = lambda L, v: float(np.degrees(frames[(L, 0)].angle_of(v)[0]))
        return ({L: ang(L, pre[L]) for L in LAYERS},
                {L: ang(L, post[L]) for L in LAYERS},
                {L: sum(dn[(L, o)] for o in offsets) for L in LAYERS})

    drift, achieved, dnorms, dh = [], [], [], []
    for c, d in cells:
        text = prompt(c, "", DAYS[d])
        ref, P = clean_ref_angles(model, sc, text, DAYS[d], frames, LAYERS, OFF)
        pre, post, dn = trace_clamped(text, DAYS[d], args.theta, OFF)
        # drift: how far the BLOCK moved it away from the target the previous layer set
        # NB the first layer is excluded from every drift summary below: nothing has
        # clamped yet when it fires, so its "drift" is trivially the whole -dtheta.
        drift.append([circ_deg(pre[L] - (ref[(L, 0)] + args.theta)) for L in LAYERS])
        achieved.append([circ_deg(post[L] - (ref[(L, 0)] + args.theta)) for L in LAYERS])
        dnorms.append([dn[L] for L in LAYERS])
        s0 = summarise(sc.score(text, DAYS[d])[0])
        hooks = clamp_fwd_hooks(frames, ref, args.theta, LAYERS, OFF, P, dev, dt)
        s = summarise(score_with_hooks(sc, text, DAYS[d], hooks))
        dh.append(circ_diff(s["circ_mean"], s0["circ_mean"]))
        print(f"   {c!r} {DAYS[d]}: mean per-layer drift "
              f"{np.mean(np.abs(drift[-1][1:])):5.2f} deg, residual error after clamp "
              f"{np.mean(np.abs(achieved[-1])):.2e} deg, dhour {dh[-1]:+.3f} h", flush=True)

    drift, achieved, dnorms = np.array(drift), np.array(achieved), np.array(dnorms)
    print(f"\n[raw] per-layer DRIFT (deg the block moved it off target before the next clamp)")
    print("  L  " + "".join(f"{L:>7}" for L in LAYERS))
    for i, (c, d) in enumerate(cells):
        print(f"{c[:14]:<14}" + "".join(f"{v:>7.1f}" for v in drift[i]))
    print(f"{'MEAN|.|':<14}" + "".join(f"{v:>7.1f}" for v in np.abs(drift).mean(0)))
    print(f"    (L{LAYERS[0]} is not drift: nothing has clamped yet when it fires)")
    print(f"\n[raw] clamp edit norm ||Delta|| per layer (summed over {len(OFF)} offsets)")
    print("  L  " + "".join(f"{L:>7}" for L in LAYERS))
    for i, (c, d) in enumerate(cells):
        print(f"{c[:14]:<14}" + "".join(f"{v:>7.2f}" for v in dnorms[i]))
    print(f"{'MEAN':<14}" + "".join(f"{v:>7.2f}" for v in dnorms.mean(0)))

    print(f"\n[verify] residual error after each clamp: "
          f"{np.abs(achieved).max():.2e} deg (should be ~0 — the clamp is exact)")
    print(f"[verify] the model's own pull-back, L{LAYERS[1]}+: "
          f"{np.abs(drift[:, 1:]).mean():.2f} deg per layer mean, "
          f"{np.abs(drift[:, 1:]).max():.2f} deg max, against a {args.theta:.2f} deg target"
          + (f" -> it recovers {np.abs(drift[:, 1:]).mean()/abs(args.theta):.1%} of the "
             f"clamp per layer" if args.theta else " (identity control: nothing to recover)"))
    print(f"[verify] delta predicted hour under the clamp: "
          f"{np.mean(dh):+.3f} h  (per cell: {', '.join(f'{v:+.2f}' for v in dh)})")
    print(f"[verify] total clamp edit norm across the band: "
          f"{dnorms.sum(1).mean():.1f} (mean over cells)")


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    r = np.random.default_rng(seed)
    m = np.array([r.choice(x, len(x), True).mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sweep(model, sc, frames, keep, LAYERS, OFF, args):
    """Dose-response: clamp the ring to clean+theta for a range of SMALL theta."""
    dev, dt = model.cfg.device, model.W_E.dtype
    TH = list(args.thetas)
    cells = [(c, d) for c in keep for d in range(N_DAYS)]
    print(f"\n[sweep] {len(cells)} cells x {len(TH)} thetas, clamp at L{LAYERS[0]}..L"
          f"{LAYERS[-1]}, offsets {OFF}", flush=True)
    print(f"[sweep] thetas: {', '.join(f'{t:g}' for t in TH)}  "
          f"(modifier's own = {args.mod_deg:g}, half day = {DAY_STEP_DEG/2:.2f})", flush=True)

    rows = []
    for i, (c, d) in enumerate(cells):
        text = prompt(c, "", DAYS[d])
        ref, P = clean_ref_angles(model, sc, text, DAYS[d], frames, LAYERS, OFF)
        lp0 = sc.score(text, DAYS[d])[0]
        s0 = summarise(lp0)
        p0 = s0["p"]
        for th in TH:
            hooks = clamp_fwd_hooks(frames, ref, th, LAYERS, OFF, P, dev, dt)
            s = summarise(score_with_hooks(sc, text, DAYS[d], hooks))
            p = s["p"]
            rows.append(dict(carrier=c, day=d, theta=th,
                             d_cm=circ_diff(s["circ_mean"], s0["circ_mean"]),
                             d_logodds=s["logodds"] - s0["logodds"],
                             conc=s["conc"], conc0=s0["conc"],
                             ent=float(-(p * np.log(p + 1e-30)).sum()),
                             ent0=float(-(p0 * np.log(p0 + 1e-30)).sum()),
                             kl=float((p0 * (np.log(p0 + 1e-30) - np.log(p + 1e-30))).sum())))
        if (i + 1) % 7 == 0:
            print(f"   {i+1}/{len(cells)} cells", flush=True)

    import pandas as pd
    R = pd.DataFrame(rows)
    outdir = os.path.join(args.dir, "clamp")
    os.makedirs(outdir, exist_ok=True)
    R.to_csv(os.path.join(outdir, "clamp_theta_sweep.csv"), index=False)

    # ---- raw first: every cell at every theta ------------------------------------
    piv = R.pivot_table(index=["carrier", "day"], columns="theta", values="d_cm")
    print(f"\n[raw] delta predicted hour per cell (rows) x theta (cols), in hours")
    print("      " + "".join(f"{t:>8g}" for t in piv.columns))
    for (c, d), r in piv.iterrows():
        print(f"{c[:12]:<12}{DAYS[d][:3]:<4}" + "".join(f"{v:>8.3f}" for v in r.values))

    print(f"\n[dose] mean delta predicted hour, 95% CI bootstrapped over "
          f"{R.carrier.nunique()} carriers")
    print(f"    {'theta':>8} {'d_hour':>9} {'95% CI':>20} {'|d| vs theta=0':>16} "
          f"{'dKL':>8} {'dEnt':>8}")
    stat = {t: boot(R[np.isclose(R.theta, t)].groupby("carrier").d_cm.mean().values)
            for t in TH}
    # theta=0 is the identity clamp -- a true no-op on the representation, so whatever it
    # reads is the measurement's own noise floor (bf16, batched forward). Every other
    # theta is quoted against it.
    floor = abs(stat[0.0][0]) if 0.0 in stat else float("nan")
    for t in TH:
        S = R[np.isclose(R.theta, t)]
        m, lo, hi = stat[t]
        print(f"    {t:>8g} {m:>+9.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>20} "
              f"{abs(m) - floor:>+16.3f} "
              f"{S.kl.mean():>8.4f} {S.ent.mean() - S.ent0.mean():>+8.4f}")

    # ---- antisymmetry: the signature of a real coordinate -------------------------
    # A real signed coordinate gives d(-theta) = -d(theta), so d(theta)+d(-theta) = 0 and
    # the ANTISYMMETRIC part (d(theta)-d(-theta))/2 carries the whole effect. Symmetric
    # blurring (an edit that degrades the representation either way) shows up in the sum.
    print(f"\n[antisym] paired within cell: A = (d(+t) - d(-t))/2 is the real-coordinate "
          f"part, S = (d(+t) + d(-t))/2 is symmetric damage")
    print(f"    {'theta':>8} {'A (antisym)':>14} {'95% CI':>20} {'S (sym)':>10}")
    for t in [x for x in TH if x > 0]:
        neg = -t
        cand = [u for u in TH if np.isclose(u, neg)]
        if not cand:
            continue
        P_ = R[np.isclose(R.theta, t)].set_index(["carrier", "day"]).d_cm
        N_ = R[np.isclose(R.theta, cand[0])].set_index(["carrier", "day"]).d_cm
        A = ((P_ - N_.reindex(P_.index)) / 2).reset_index()
        S_ = ((P_ + N_.reindex(P_.index)) / 2).reset_index()
        m, lo, hi = boot(A.groupby("carrier").d_cm.mean().values)
        print(f"    {t:>8g} {m:>+14.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>20} "
              f"{S_.groupby('carrier').d_cm.mean().mean():>+10.3f}")

    # ---- figure -------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    tidy_ax(ax)
    th = np.array(TH)
    mu = np.array([stat[t][0] for t in TH])
    lo = np.array([stat[t][1] for t in TH])
    hi = np.array([stat[t][2] for t in TH])
    ax.fill_between(th, lo, hi, color="#3b5f9e", alpha=0.18, zorder=3, lw=0)
    ax.plot(th, mu, color="#3b5f9e", lw=2.0, marker="o", ms=4.0, markeredgecolor="white",
            markeredgewidth=0.6, zorder=4, label="clamp: ring held at clean + θ, L2–L28, "
                                                 f"offsets {OFF}")
    ax.axhline(0, color="#aeb4c0", lw=1, zorder=1)
    for s_ in (+1, -1):
        ax.axvline(s_ * args.mod_deg, color="#c8642a", lw=1.0, ls=(0, (3, 2)), zorder=2)
    ax.text(args.mod_deg, ax.get_ylim()[1], f"  what a real modifier\n  rotates ({args.mod_deg:g}°)",
            fontsize=7.0, color="#c8642a", va="top")
    ax.set_xlabel("θ — ring angle the clamp holds, relative to clean (degrees).  "
                  f"±{DAY_STEP_DEG/2:.2f}° = halfway to the next day",
                  fontsize=8.4, color="#16181d")
    ax.set_ylabel("Δ predicted hour (clamped − clean)", fontsize=8.4, color="#16181d")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    ax.set_title("Holding the weekday ring at a rotated angle, all layers, all prompt "
                 "positions", fontsize=9.6, color="#16181d", loc="left")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"\n[saved] {args.out}")
    print(f"[saved] {os.path.join(outdir, 'clamp_theta_sweep.csv')}")
    return 0


def tidy_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6); ax.spines[sp].set_color("#aeb4c0")
    ax.tick_params(labelsize=8.0, length=2.5, width=0.6, colors="#5b6270")
    ax.grid(True, lw=0.4, color="#eceef2", zorder=0)
    ax.set_axisbelow(True)


@torch.no_grad()
def score_with_hooks(sc, text, day, fwd_hooks):
    """Scorer.score's readout, but with caller-supplied hooks (the clamp needs its own)."""
    pre = sc.m.to_tokens(text, prepend_bos=sc.bos)[0].tolist()
    forms, owner = sc.cand_ids()
    seqs = [pre + f for f in forms]
    mx = max(len(s) for s in seqs)
    ids = torch.zeros((len(seqs), mx), dtype=torch.long, device=sc.dev)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s, device=sc.dev)
    logits = sc.m.run_with_hooks(ids, return_type="logits", fwd_hooks=fwd_hooks)
    lg = torch.log_softmax(logits.float(), -1)
    n_pre = len(pre)
    out = np.full(len(seqs), -np.inf)
    for j, s in enumerate(seqs):
        idx = torch.arange(n_pre, len(s), device=sc.dev)
        out[j] = float(lg[j, idx - 1, torch.tensor(s[n_pre:], device=sc.dev)].sum())
    from steer_timeofday import _lse
    return np.array([_lse(out[owner == h]) for h in range(24)])


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
