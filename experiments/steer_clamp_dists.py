#!/usr/bin/env python
"""The clean-state reference distributions the clamp runs are read against. No figure of its own."""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weekday_manifold.model import load_model
from steer_timeofday import (Scorer, summarise, prompt, circ_diff, circ_deg,
                             DAYS, N_DAYS, DAY_STEP_DEG)
from steer_clamp import (fit_pos_frames, clamp_hook, clamp_dir_hook, score_with_hooks,
                         MODS)


def clean_state(model, sc, text, day, layers, offsets):
    """Raw clean activation at every (layer, offset) -- the reference every clamp targets."""
    P = sc.weekday_pos(text, day)
    grab = {}
    with torch.no_grad():
        model.run_with_hooks(
            model.to_tokens(text, prepend_bos=True), return_type=None,
            fwd_hooks=[(f"blocks.{L}.hook_resid_post",
                        (lambda LL: lambda a, hook: grab.__setitem__(
                            LL, a[0].detach().float().cpu().numpy()))(L))
                       for L in layers])
    return {(L, o): grab[L][P + o] for L in layers for o in offsets}, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/results/steer_timeofday/meta-llama_Llama-3.1-8B")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--start-layer", type=int, default=2)
    ap.add_argument("--end-layer", type=int, default=28)
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-carriers", type=int, default=12)
    ap.add_argument("--mod-deg", type=float, default=8.07,
                    help="the modifier's own realized in-plane rotation at L12")
    ap.add_argument("--seed", type=int, default=0)
    # The off-plane DOSE-RESPONSE. A norm-matched positive control is geometrically
    # impossible here: a rotation spends ||Delta|| = 2 r sin(theta/2), capped at the ring
    # diameter, while the modifier's off-plane displacement is an order of magnitude
    # larger (observationally, 1.21 ring radii off-plane vs 0.06 in-plane). So instead of
    # matching, MEASURE THE FLOOR -- sweep the off-plane clamp down through the norms the
    # ring clamp can actually reach, and read off how much the readout needs to respond.
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.dir, "meta.json")))
    keep = meta["carriers_kept"][:args.n_carriers]
    LAYERS = list(range(args.start_layer, args.end_layer + 1))
    OFF = args.offsets
    HALF = DAY_STEP_DEG / 2

    model = load_model(args.model, device="cuda", fold_ln=True,
                       center_writing_weights=True, center_unembed=True, dtype="bfloat16")
    sc = Scorer(model, LAYERS[0])
    dev, dt = model.cfg.device, model.W_E.dtype

    frames, acts, days = fit_pos_frames(model, sc, keep, LAYERS, OFF)

    # ---- per (layer, offset, day) modifier directions, split against THAT frame --------
    # d_time is the late-minus-early contrast (position-matched: both arms are one-word
    # modifiers), exactly as steer_timeofday.Geom builds it, but fitted at every offset.
    rng = np.random.default_rng(args.seed)
    d_off, d_plac_off, d_rand = {}, {}, {}
    for L in LAYERS:
        for o in OFF:
            f = frames[(L, o)]
            A_e, A_l = acts[("early", L, o)], acts[("late", L, o)]
            A_n, A_p = acts[("neutral", L, o)], acts[("placebo", L, o)]
            for d in range(N_DAYS):
                dt_ = A_l[days == d].mean(0) - A_e[days == d].mean(0)
                dp_ = A_p[days == d].mean(0) - A_n[days == d].mean(0)
                off = dt_ - (dt_ @ f.plane) @ f.plane.T
                poff = dp_ - (dp_ @ f.plane) @ f.plane.T
                # placebo norm-matched to the time direction, as in Geom.__init__
                n = np.linalg.norm(poff)
                if n > 0:
                    poff = poff * (np.linalg.norm(off) / n)
                v = rng.normal(size=f.mu.shape[0])
                v -= f.span @ (f.span.T @ v)            # orthogonal to the WHOLE weekday span
                v = v / np.linalg.norm(v) * np.linalg.norm(off)
                d_off[(L, o, d)], d_plac_off[(L, o, d)], d_rand[(L, o, d)] = off, poff, v

    cells = [(c, d) for c in keep for d in range(N_DAYS)]
    print(f"\n[dists] {len(cells)} cells, clamp L{LAYERS[0]}..L{LAYERS[-1]}, offsets {OFF}",
          flush=True)

    # ---- PASS 1: what does the ring clamp at the modifier's own angle actually SPEND? --
    # The matched control is built from these numbers, per (cell, layer, offset), so the
    # comparison is exact rather than approximately norm-matched.
    # Measured at BOTH ring angles, because matching to the modifier's own 8.07 deg turns
    # out to be uninformative: an 8 deg in-plane rotation is a tiny edit (||Delta|| = 2 r
    # sin(theta/2)), ~2% of what the modifier's off-plane displacement spends, and at that
    # norm the positive control is silent too -- every curve then coincides for an
    # uninteresting reason, exactly the trap figure_steer_hour_dists.py documents. The
    # informative match is against the LARGEST defensible ring edit, the half-day clamp,
    # which is also the one "you just moved to the next day" cannot explain away.
    print(f"[pass1] measuring the ring clamp's edit norm at "
          f"{args.mod_deg:+.2f} deg and {HALF:+.2f} deg", flush=True)
    ring_norm, ring_norm_half = {}, {}
    for tgt_deg, sink in ((args.mod_deg, ring_norm), (HALF, ring_norm_half)):
        for i, (c, d) in enumerate(cells):
            text = prompt(c, "", DAYS[d])
            ref, P = clean_state(model, sc, text, DAYS[d], LAYERS, OFF)
            rec = {}
            hooks = []
            for L in LAYERS:
                for o in OFF:
                    f = frames[(L, o)]
                    a0 = float(np.degrees(f.angle_of(ref[(L, o)])[0]))
                    base = clamp_hook(f, [a0 + tgt_deg], [P + o], dev, dt)
                    def mk(base, L, o):
                        def fn(resid, hook):
                            before = resid[:, P + o, :].detach().float().clone()
                            out = base(resid, hook)
                            rec[(L, o)] = float((out[0, P + o].float() - before[0]).norm())
                            return out
                        return fn
                    hooks.append((f"blocks.{L}.hook_resid_post", mk(base, L, o)))
            with torch.no_grad():
                model.run_with_hooks(model.to_tokens(text, prepend_bos=True),
                                     return_type=None, fwd_hooks=hooks)
            sink[(c, d)] = dict(rec)
    tot = np.mean([sum(v.values()) for v in ring_norm.values()])
    tot_h = np.mean([sum(v.values()) for v in ring_norm_half.values()])
    off_tot = np.mean([sum(np.linalg.norm(d_off[(L, o, d)]) for L in LAYERS for o in OFF)
                       for _, d in cells])
    print(f"[pass1] ring clamp spends ||Delta|| = {tot:.3f} at {args.mod_deg:+.2f} deg and "
          f"{tot_h:.3f} at {HALF:+.2f} deg, total across the band (mean over cells)",
          flush=True)
    print(f"[pass1] the modifier's off-plane direction spends {off_tot:.1f} -> the two "
          f"matched arms are {tot/off_tot:.3f}x and {tot_h/off_tot:.3f}x of it", flush=True)

    # ---- PASS 2: every arm, full 24-hour distribution ---------------------------------
    ARMS = ["clean", "prompt_early", "prompt_late", "identity",
            "clamp_ring_mod", "clamp_ring_half",
            "clamp_off_matched", "clamp_off_matched_half", "clamp_off_full", "clamp_full",
            "clamp_off_plac", "clamp_off_rand"] + [f"clamp_off_a{a:g}" for a in args.alphas]
    out = {a: [] for a in ARMS}
    rows = []

    def dir_hooks(ref, P, vecs, scale=None, per_norm=None):
        """Clamp the component along each vec to clean + (its norm, or a matched norm)."""
        hk = []
        for L in LAYERS:
            for o in OFF:
                v = vecs[(L, o)]
                n = np.linalg.norm(v)
                if n == 0:
                    continue
                u = v / n
                mag = per_norm[(L, o)] if per_norm is not None else n * (scale or 1.0)
                cur = float(ref[(L, o)] @ u)
                hk.append((f"blocks.{L}.hook_resid_post",
                           clamp_dir_hook(u, cur + mag, [P + o], dev, dt)))
        return hk

    def ring_hooks(ref, P, dtheta):
        hk = []
        for L in LAYERS:
            for o in OFF:
                f = frames[(L, o)]
                a0 = float(np.degrees(f.angle_of(ref[(L, o)])[0]))
                hk.append((f"blocks.{L}.hook_resid_post",
                           clamp_hook(f, [a0 + dtheta], [P + o], dev, dt)))
        return hk

    for i, (c, d) in enumerate(cells):
        text = prompt(c, "", DAYS[d])
        ref, P = clean_state(model, sc, text, DAYS[d], LAYERS, OFF)
        s0 = summarise(sc.score(text, DAYS[d])[0])
        OFFV = {(L, o): d_off[(L, o, d)] for L in LAYERS for o in OFF}
        specs = {
            "identity":          ring_hooks(ref, P, 0.0),
            "clamp_ring_mod":    ring_hooks(ref, P, args.mod_deg),
            "clamp_ring_half":   ring_hooks(ref, P, HALF),
            "clamp_off_matched": dir_hooks(ref, P, OFFV, per_norm=ring_norm[(c, d)]),
            "clamp_off_matched_half": dir_hooks(ref, P, OFFV,
                                                per_norm=ring_norm_half[(c, d)]),
            "clamp_off_full":    dir_hooks(ref, P, OFFV),
            "clamp_full":        ring_hooks(ref, P, args.mod_deg)
                                 + dir_hooks(ref, P, OFFV),
            "clamp_off_plac":    dir_hooks(
                ref, P, {(L, o): d_plac_off[(L, o, d)] for L in LAYERS for o in OFF}),
            "clamp_off_rand":    dir_hooks(
                ref, P, {(L, o): d_rand[(L, o, d)] for L in LAYERS for o in OFF}),
        }
        for a in args.alphas:
            specs[f"clamp_off_a{a:g}"] = dir_hooks(ref, P, OFFV, scale=a)
        rn, rnh = sum(ring_norm[(c, d)].values()), sum(ring_norm_half[(c, d)].values())
        DNORM = {"clamp_ring_mod": rn, "clamp_off_matched": rn,
                 "clamp_ring_half": rnh, "clamp_off_matched_half": rnh,
                 "clamp_off_full": sum(np.linalg.norm(OFFV[(L, o)])
                                       for L in LAYERS for o in OFF)}
        DNORM.update({f"clamp_off_a{a:g}": a * DNORM["clamp_off_full"]
                      for a in args.alphas})
        for arm in ARMS:
            if arm == "clean":
                s = s0
            elif arm in ("prompt_early", "prompt_late"):
                w = MODS["early" if arm.endswith("early") else "late"]
                s = summarise(sc.score(prompt(c, w, DAYS[d]), DAYS[d])[0])
            else:
                s = summarise(score_with_hooks(sc, text, DAYS[d], specs[arm]))
            out[arm].append(s["p"])
            p, p0 = s["p"], s0["p"]
            rows.append(dict(
                carrier=c, day=d, arm=arm,
                d_cm=circ_diff(s["circ_mean"], s0["circ_mean"]),
                d_logodds=s["logodds"] - s0["logodds"], conc=s["conc"],
                ent=float(-(p * np.log(p + 1e-30)).sum()),
                kl=float((p0 * (np.log(p0 + 1e-30) - np.log(p + 1e-30))).sum()),
                dnorm=DNORM.get(arm, float("nan"))))
        if (i + 1) % 7 == 0:
            print(f"   {i+1}/{len(cells)} cells", flush=True)

    import pandas as pd
    outdir = os.path.join(args.dir, "clamp")
    os.makedirs(outdir, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(outdir, "clamp_dists_rows.csv"), index=False)
    P_ = {a: np.stack(v) for a, v in out.items()}
    npz = os.path.join(outdir, "hour_dists_clamp.npz")
    np.savez(npz, days=np.array([d for _, d in cells]),
             carriers=np.array(keep), layers=np.array(LAYERS), offsets=np.array(OFF),
             mod_deg=args.mod_deg,
             ring_norm_total=np.array([sum(ring_norm[k].values()) for k in cells]),
             ring_norm_half_total=np.array([sum(ring_norm_half[k].values()) for k in cells]),
             **{f"p__{a}": v for a, v in P_.items()})

    print(f"\n[raw] delta predicted hour by arm (mean over {len(cells)} cells, "
          f"CI over {len(keep)} carriers)")
    from steer_clamp import boot
    for a in ARMS[1:]:
        S = R[R.arm == a]
        m, lo, hi = boot(S.groupby("carrier").d_cm.mean().values)
        print(f"    {a:<20} {m:>+8.3f} h  [{lo:+.3f}, {hi:+.3f}]   "
              f"KL {S.kl.mean():.4f}  ent {S.ent.mean():.4f}")
    print(f"\n[saved] {npz}")
    print(f"[saved] {os.path.join(outdir, 'clamp_dists_rows.csv')}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
