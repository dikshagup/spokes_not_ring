#!/usr/bin/env python
"""Figure 2 panel C: every in-plane edit against an off-plane edit of the same size.

Writes a rows.csv of the 84 carrier x day cells. See repro_fig2_timeofday_with_steering.sh.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weekday_manifold.model import load_model
from steer_timeofday import Scorer, prompt, circ_diff, DAYS, N_DAYS, DAY_STEP_DEG
from steer_clamp import fit_pos_frames, clamp_hook, clamp_dir_hook, MODS, boot
from steer_clamp_dists import clean_state
from time_readout import TimeReadout

# The three in-plane edits panel C pairs, and the matched arm each one gets. Kept as data
# rather than spelled out three times, because the dose solver, the row writer and the
# summary all have to agree on which off-plane arm answers which in-plane arm.
PAIRS = [("clamp_ring_own", "clamp_off_m_ring_own"),
         ("clamp_in_a1", "clamp_off_m_in_a1"),
         ("clamp_ring_half", "clamp_off_m_ring_half")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/results/steer_timeofday/meta-llama_Llama-3.1-8B")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--start-layer", type=int, default=2)
    ap.add_argument("--end-layer", type=int, default=28)
    ap.add_argument("--n-carriers", type=int, default=12)
    ap.add_argument("--n-cells", type=int, default=0,
                    help="stop after this many cells; 0 = all. For timing a short prefix "
                         "before committing to the full run")
    ap.add_argument("--tol", type=float, default=0.002,
                    help="relative displacement match the dose solver accepts")
    ap.add_argument("--max-probes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-against", default=None,
                    help="gated rows.csv to compare the re-run reference arms against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.dir, "meta.json")))
    keep = meta["carriers_kept"][:args.n_carriers]
    LAYERS = list(range(args.start_layer, args.end_layer + 1))
    OFF = [0]                      # offset0 only: the one band that passes the gate at 27/27
    HALF = DAY_STEP_DEG / 2

    model = load_model(args.model, device="cuda", fold_ln=True,
                       center_writing_weights=True, center_unembed=True, dtype="bfloat16")
    sc = Scorer(model, LAYERS[0])
    dev, dtp = model.cfg.device, model.W_E.dtype
    frames, acts, days = fit_pos_frames(model, sc, keep, LAYERS, OFF, verbose=False)
    TR = TimeReadout(model)
    print(f"[readout] extended: {len(TR.forms)} unambiguous strings, "
          f"{len(TR.amb)} ambiguous prefixes held out of the estimate")

    # ---- THE GATE, identical to steer_clamp_gated.py -----------------------------------
    SITES, dropped = [], []
    for L in LAYERS:
        for o in OFF:
            (SITES if frames[(L, o)].heptagon_check()["in_cyclic_order"] else dropped
             ).append((L, o))
    print(f"[gate] offset0: {len(SITES)} sites kept, {len(dropped)} dropped")
    if not SITES:
        raise SystemExit("no site passes the heptagon check")

    # ---- per (site, day) modifier directions, split against THAT frame -------------------
    d_in, d_off = {}, {}
    for (L, o) in SITES:
        f = frames[(L, o)]
        A_e, A_l = acts[("early", L, o)], acts[("late", L, o)]
        for d in range(N_DAYS):
            dt_ = A_l[days == d].mean(0) - A_e[days == d].mean(0)
            ip = (dt_ @ f.plane) @ f.plane.T
            d_in[(L, o, d)], d_off[(L, o, d)] = ip, dt_ - ip

    # ---- hook builders, each wrapped so both edit sizes are recorded --------------------
    def wrap(base, ref_site, rec, L, o, pos):
        def fn(resid, hook):
            before = resid[0, pos, :].detach().float().clone()
            outp = base(resid, hook)
            after = outp[0, pos].float()
            r = torch.as_tensor(ref_site, device=after.device, dtype=torch.float32)
            rec[(L, o)] = (float((after - before).norm()), float((after - r).norm()))
            return outp
        return fn

    def ring_hooks(ref, P, theta, rec, per_cell=None):
        hk = []
        for (L, o) in SITES:
            f = frames[(L, o)]
            a0 = float(np.degrees(f.angle_of(ref[(L, o)])[0]))
            th = per_cell[(L, o)] if per_cell is not None else theta
            base = clamp_hook(f, [a0 + th], [P + o], dev, dtp)
            hk.append((f"blocks.{L}.hook_resid_post", wrap(base, ref[(L, o)], rec, L, o, P + o)))
        return hk

    def dir_hooks(ref, P, vecs, rec, scale=1.0):
        hk = []
        for (L, o) in SITES:
            v = vecs[(L, o)]
            n = np.linalg.norm(v)
            if n == 0:
                continue
            u = v / n
            base = clamp_dir_hook(u, float(ref[(L, o)] @ u) + n * scale, [P + o], dev, dtp)
            hk.append((f"blocks.{L}.hook_resid_post", wrap(base, ref[(L, o)], rec, L, o, P + o)))
        return hk

    ARMS = ["clean", "identity"] + [a for p in PAIRS for a in p]
    cells = [(c, d) for c in keep for d in range(N_DAYS)]
    if args.n_cells:
        cells = cells[:args.n_cells]
    rows, sites_rows, solves = [], [], []
    print(f"[pairs] {len(cells)} cells x {len(ARMS)} arms on {len(SITES)} sites", flush=True)
    t0 = time.time()

    for i, (c, d) in enumerate(cells):
        text = prompt(c, "", DAYS[d])
        pre_ids = model.to_tokens(text, prepend_bos=TR.bos)[0].tolist()
        ref, P = clean_state(model, sc, text, DAYS[d], LAYERS, OFF)
        s0 = TR.score(text)
        INV = {(L, o): d_in[(L, o, d)] for (L, o) in SITES}
        OFV = {(L, o): d_off[(L, o, d)] for (L, o) in SITES}

        # each cell's OWN modifier rotation, per site: angle(h + d_in) - angle(h)
        own = {}
        for (L, o) in SITES:
            f = frames[(L, o)]
            a0 = f.angle_of(ref[(L, o)])[0]
            a1 = f.angle_of(ref[(L, o)] + INV[(L, o)])[0]
            own[(L, o)] = float(np.degrees(np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0))))

        recs = {a: {} for a in ARMS}
        specs = {
            "identity":        ring_hooks(ref, P, 0.0, recs["identity"]),
            "clamp_ring_own":  ring_hooks(ref, P, 0.0, recs["clamp_ring_own"], per_cell=own),
            "clamp_ring_half": ring_hooks(ref, P, HALF, recs["clamp_ring_half"]),
            "clamp_in_a1":     dir_hooks(ref, P, INV, recs["clamp_in_a1"], scale=1.0),
        }

        # ---- the in-plane arms first: their realized ||Delta|| is what the doses target --
        scored = {}
        for arm in ["identity"] + [a for a, _ in PAIRS]:
            scored[arm] = TR.score(text, specs[arm])

        # ---- solve the off-plane dose, per cell, per target ----------------------------
        # The probe re-runs the LAST forward TR.score makes -- the 60 ambiguous prefixes --
        # and nothing else. It must be that forward and not a cheaper batch-1 one: `wrap`
        # records resid[0] on every call and the last call wins, so `displacement` as
        # written to rows.csv is whatever the ambiguous chunk saw, and bf16 kernels are
        # batch-shape dependent. Probing at batch 1 solved the dose to 0.1% and still
        # landed up to 5% off the recorded value, biased high and worst at the smallest
        # edits, where the fixed drift floor is the largest share of ||Delta||. Probing on
        # the same shape makes probe and realized the same measurement by construction.
        # Cost: 60 sequences against the 685 a full scoring puts through, ~9% of one.
        def disp_at(alpha):
            rec = {}
            with torch.no_grad():
                TR._logp(pre_ids, TR._amb_ids,
                         fwd_hooks=dir_hooks(ref, P, OFV, rec, scale=alpha))
            return sum(v[1] for v in rec.values())

        for in_arm, off_arm in PAIRS:
            D_t = sum(v[1] for v in recs[in_arm].values())
            # Two Newton steps on the population slope to get close, then BISECT on a
            # bracket, keeping the best alpha ever probed rather than the last one. Plain
            # secant was tried and stalls: displacement(alpha) is monotone but locally a
            # STAIRCASE, because the state the hook writes is rounded to bf16 and a small
            # enough change in alpha does not change it at all. On a step function the
            # secant denominator collapses, the step lands nowhere near the root, and the
            # iterate wanders -- 6 of 7 probe cells hit the cap at 0.4-1.8% error with a
            # better alpha already visited and thrown away. Bisection cannot wander, and
            # best-tracking means the answer is never worse than the best thing seen.
            pts = []

            def probe(a):
                a = float(np.clip(a, 1e-4, 10.0))
                pts.append((a, disp_at(a)))
                return pts[-1][1]

            # seed and refine off the gated run's global fit, displacement ~= 2.5 + 173.6a
            a = max((D_t - 2.5) / 173.6, 1e-3)
            probe(a)
            probe(pts[-1][0] + (D_t - pts[-1][1]) / 173.6)
            # widen until the root is bracketed; the seed is close, so this rarely fires
            while len(pts) < args.max_probes and all(d > D_t for _, d in pts):
                probe(min(a_ for a_, _ in pts) * 0.6)
            while len(pts) < args.max_probes and all(d < D_t for _, d in pts):
                probe(max(a_ for a_, _ in pts) * 1.6)
            while len(pts) < args.max_probes:
                below = [a_ for a_, d_ in pts if d_ <= D_t]
                above = [a_ for a_, d_ in pts if d_ >= D_t]
                if abs(min(abs(d_ - D_t) for _, d_ in pts)) <= args.tol * D_t:
                    break
                if not below or not above:
                    break
                probe(0.5 * (max(below) + min(above)))
            a1, d1 = min(pts, key=lambda p: abs(p[1] - D_t))
            n_probe = len(pts)
            specs[off_arm] = dir_hooks(ref, P, OFV, recs[off_arm], scale=a1)
            scored[off_arm] = TR.score(text, specs[off_arm])
            D_r = sum(v[1] for v in recs[off_arm].values())
            solves.append(dict(carrier=c, day=d, in_arm=in_arm, off_arm=off_arm,
                               target=D_t, alpha=a1, probe_disp=d1, realized=D_r,
                               rel_err=(D_r - D_t) / D_t, n_probe=n_probe))

        # ---- rows ----------------------------------------------------------------------
        for arm in ARMS:
            s = s0 if arm == "clean" else scored[arm]
            p, p0 = s["p"], s0["p"]
            R_ = recs.get(arm, {})
            rows.append(dict(
                carrier=c, day=d, arm=arm,
                d_cm=circ_diff(s["circ_mean"], s0["circ_mean"]),
                d_logodds=s["logodds"] - s0["logodds"], conc=s["conc"],
                ent=float(-(p * np.log(p + 1e-30)).sum()),
                kl=float((p0 * (np.log(p0 + 1e-30) - np.log(p + 1e-30))).sum()),
                maintenance=sum(v[0] for v in R_.values()) if R_ else float("nan"),
                displacement=sum(v[1] for v in R_.values()) if R_ else float("nan"),
                own_deg=float(np.mean(list(own.values()))) if arm == "clamp_ring_own"
                        else float("nan"),
                alpha=next((r["alpha"] for r in solves
                            if r["off_arm"] == arm and r["carrier"] == c and r["day"] == d),
                           float("nan")),
                captured=s.get("captured", float("nan")),
                ambiguous=s.get("ambiguous", float("nan")),
                cm_am=TR.bounds(s)["am"], cm_pm=TR.bounds(s)["pm"]))
            for (L, o), (mnt, dsp) in R_.items():
                sites_rows.append(dict(carrier=c, day=d, arm=arm, layer=L, offset=o,
                                       maintenance=mnt, displacement=dsp))
        if (i + 1) % 7 == 0:
            el = time.time() - t0
            print(f"   {i+1}/{len(cells)} cells  {el/(i+1):.1f}s/cell  "
                  f"eta {(len(cells)-i-1)*el/(i+1)/60:.1f} min", flush=True)

    import pandas as pd
    outdir = args.out or os.path.join(args.dir, "clamp_gated", "matched_pairs")
    os.makedirs(outdir, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(outdir, "rows.csv"), index=False)
    S = pd.DataFrame(solves)
    S.to_csv(os.path.join(outdir, "dose_solve.csv"), index=False)
    pd.DataFrame(sites_rows).to_csv(os.path.join(outdir, "site_norms.csv"), index=False)

    # ---- did the match land? ------------------------------------------------------------
    print(f"\n[match] realized ||Delta|| against the in-plane arm it answers")
    print(f"    {'pair':<22}{'target':>9}{'realized':>10}{'rel err':>10}{'worst':>9}{'probes':>8}")
    for in_arm, off_arm in PAIRS:
        Q = S[S.in_arm == in_arm]
        print(f"    {in_arm:<22}{Q.target.mean():>9.2f}{Q.realized.mean():>10.2f}"
              f"{Q.rel_err.mean()*100:>9.2f}%{Q.rel_err.abs().max()*100:>8.2f}%"
              f"{Q.n_probe.mean():>8.1f}")

    print(f"\n[raw] delta predicted hour, {len(SITES)} sites, "
          f"{len(cells)} cells, CI over {R.carrier.nunique()} carriers")
    print(f"    {'arm':<22}{'dh':>9}  {'95% CI':<22}{'sd':>8}{'displac':>10}{'KL':>9}")
    for a in ARMS[1:]:
        Q = R[R.arm == a]
        m, lo, hi = boot(Q.groupby("carrier").d_cm.mean().values)
        print(f"    {a:<22}{m:>+9.3f}  [{lo:+.3f}, {hi:+.3f}]     "
              f"{Q.d_cm.std():>8.4f}{Q.displacement.mean():>10.2f}{Q.kl.mean():>9.5f}")

    print(f"\n[paired] per-cell off-plane minus in-plane, at matched ||Delta||")
    for in_arm, off_arm in PAIRS:
        A = R[R.arm == in_arm].set_index(["carrier", "day"]).d_cm
        B = R[R.arm == off_arm].set_index(["carrier", "day"]).d_cm
        diff = (B - A).dropna()
        m, lo, hi = boot(diff.groupby("carrier").mean().values)
        print(f"    {in_arm:<22}{m:>+9.3f}  [{lo:+.3f}, {hi:+.3f}]     "
              f"{int((diff > 0).sum())}/{len(diff)} cells off-plane > in-plane")

    # ---- reproducibility: do the re-run reference arms match the gated run? --------------
    if args.check_against:
        G = pd.read_csv(args.check_against)
        print(f"\n[check] re-run vs {args.check_against}: per-cell d_cm")
        for a in ["identity"] + [x for x, _ in PAIRS]:
            g = G[G.arm == a].set_index(["carrier", "day"]).d_cm
            n = R[R.arm == a].set_index(["carrier", "day"]).d_cm
            j = (n - g).dropna()
            if len(j):
                print(f"    {a:<22}n={len(j):<4} max |diff| {j.abs().max():.5f}  "
                      f"mean {j.mean():+.5f}  (this run sd {n.std():.4f})")

    print(f"\n[saved] {outdir}   ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
