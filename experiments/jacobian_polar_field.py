"""Jacobian of the readout over the whole day disc, by forward-mode autodiff.

Sweeps the polar chart p(u, r) = mu + r (sigma(u) - mu) and records the Frobenius norm
(Hutchinson), the tangential and radial gains, and a control direction.

Writes a field npz. repro_fig1_combined_llama.sh, repro_fig4_method_steer.sh,
repro_fig5_jac_grid.sh and repro_fig6_combined_arith.sh all read one.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import torch

from weekday_manifold.manifold.config import ManifoldConfig
from weekday_manifold.manifold.days import DAYS, N_DAYS, days_token_ids
from weekday_manifold.manifold.probes import (build_mention_early,
                                              build_time_statements)
from weekday_manifold.manifold.steering import (
    fit_steer_spline, template_demeaned_centroids)
from weekday_manifold.plateau.model import load_plateau_model
from weekday_manifold.utils import set_seed


def polar_point(spline, mu, u, r):
    """p(u, r) = mu + r * (sigma(u) - mu), plus the two natural unit directions."""
    import numpy as np
    P = spline.forward(u)                      # [n, d]
    T = spline.derivative(u)                   # [n, d]
    rad = P - mu[None, :]
    pt = mu[None, :] + r[:, None] * rad
    vt = T                                     # direction of travel round the ring
    vr = rad
    vt = vt / np.maximum(np.linalg.norm(vt, axis=1, keepdims=True), 1e-30)
    vr = vr / np.maximum(np.linalg.norm(vr, axis=1, keepdims=True), 1e-30)
    return pt, vt, vr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llama31_8b_fp16.json")
    ap.add_argument("--formulation", default="clock",
                    help="clock (read) | mention_early (day merely named) | "
                         "interrogative (compute, Goodfire)")
    ap.add_argument("--patch-layer", type=int, default=2)
    ap.add_argument("--readout-layer", type=int, default=28)
    ap.add_argument("--readout-pos", type=int, default=-1)
    ap.add_argument("--patch-pos", type=int, default=None)
    ap.add_argument("--n-u", type=int, default=56, help="samples around the ring")
    ap.add_argument("--n-r", type=int, default=17, help="radial samples")
    ap.add_argument("--r-max", type=float, default=1.6, help="1.0 = the ring itself")
    ap.add_argument("--radii", default=None,
                    help="explicit radii instead of a linspace, e.g. '1.0'. Every number "
                         "the figures quote is read at r = 1, so restricting to that ring "
                         "buys a factor of n-r in probes for the same wall clock -- which "
                         "is how a 100-probe estimate becomes affordable.")
    ap.add_argument("--require-seq-len", type=int, default=0,
                    help="keep only prompts of exactly this tokenised length, "
                         "dropping whole templates so the days stay balanced. "
                         "Pass 12 for Mistral, whose SentencePiece splits the "
                         "mention templates into 12/13/14/15 tokens where Llama's BPE gives all of them 12.")
    ap.add_argument("--n-prompts", type=int, default=7)
    ap.add_argument("--hour", type=int, default=11, help="clock hour for the backgrounds")
    ap.add_argument("--n-hutch", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"figures/polar_field_{args.formulation}.npz"

    cfg = ManifoldConfig.from_json(args.config)
    set_seed(cfg.seed)
    model = load_plateau_model(cfg)
    model.requires_grad_(False)
    dev, dt = model.cfg.device, model.W_E.dtype
    Lp, RO = args.patch_layer, args.readout_layer

    if args.formulation == "clock":
        specs = build_time_statements()
        day_ids = np.array([int(sp.meta["z"]) for sp in specs])
        keep = np.array([sp.meta["clock_hour"] == args.hour for sp in specs])
    elif args.formulation in ("mention_early", "me"):
        # the day named early in a plain sentence, nothing asked of it. 20 templates, so the
        # centroids are template-demeaned; with one template that step is a no-op, here it
        # is what stops a single wording dominating a day's centre.
        specs = build_mention_early("thisday")
        day_ids = np.array([int(sp.meta["z"]) for sp in specs])
        keep = np.ones(len(specs), dtype=bool)
    else:
        from weekday_manifold.manifold.days import build_prompts
        specs = build_prompts(args.formulation)
        day_ids = np.array([int(sp.meta["z"]) for sp in specs])
        keep = np.ones(len(specs), dtype=bool)
    # Same length filter the ladder needs, and for the same reason: the field addresses the
    # steer and readout sites by INDEX, and torch.cat below requires one sequence length.
    # Llama's BPE gives every mention template 12 tokens with the full stop at 8; Mistral's
    # SentencePiece gives 12/13/14/15, so without this the run dies on a shape mismatch.
    # Whole templates are dropped, so the days stay balanced.
    if args.require_seq_len:
        ok = np.array([len(model.to_tokens(sp.text, prepend_bos=cfg.prepend_bos)[0])
                       == args.require_seq_len for sp in specs])
        assert ok.any(), f"no prompt tokenises to {args.require_seq_len} tokens"
        cnt = collections.Counter(int(sp.meta["z"]) for sp, k in zip(specs, ok) if k)
        assert len(set(cnt.values())) == 1, f"length filter unbalanced the days: {dict(cnt)}"
        print(f"[polar] length filter: kept {int(ok.sum())}/{len(specs)} prompts at "
              f"{args.require_seq_len} tokens, {cnt[0]} per day", flush=True)
        # every per-prompt array has to be cut with it, not just `specs`: day_ids is built
        # above (it differs per formulation) and `keep` is the hour mask the clock family
        # uses, read later as keep[i] over the SURVIVING prompts.
        specs = [sp for sp, k in zip(specs, ok) if k]
        day_ids = day_ids[ok]
        keep = np.asarray(keep)[ok]
    texts = [sp.text for sp in specs]
    template_ids = np.array([int(sp.meta.get("content_id", 0)) for sp in specs])
    assert len(day_ids) == len(specs) == len(template_ids) == len(keep), (
        f"per-prompt arrays disagree after filtering: {len(day_ids)}, {len(specs)}, "
        f"{len(template_ids)}, {len(keep)}")

    toks_all = torch.cat([model.to_tokens(t, prepend_bos=cfg.prepend_bos) for t in texts], 0)
    seq_len = toks_all.shape[1]
    _store = {}

    def _rec_all(act, hook):
        _store["x"] = act.detach().float().cpu().numpy()
        return act
    with torch.no_grad():
        model.run_with_hooks(toks_all, return_type=None,
                             fwd_hooks=[(f"blocks.{Lp}.hook_resid_post", _rec_all)])
    A_all = _store["x"]

    if args.patch_pos is not None:
        day_pos = args.patch_pos % seq_len
    else:
        spreads = [float(np.linalg.norm(
            (Cp := template_demeaned_centroids(A_all[:, p, :], day_ids, template_ids, N_DAYS))
            - Cp.mean(0), axis=1).mean()) for p in range(seq_len)]
        day_pos = int(np.argmax(spreads))
        st = model.to_str_tokens(texts[0], prepend_bos=cfg.prepend_bos)
        print("[polar] day-ring radius by position: "
              + ", ".join(f"{p}({st[p]!r}):{s:.2f}" for p, s in enumerate(spreads)), flush=True)
    ro_pos = args.readout_pos % seq_len

    C = template_demeaned_centroids(A_all[:, day_pos, :], day_ids, template_ids, N_DAYS)
    spline = fit_steer_spline(C)
    mu = C.mean(0)
    d = C.shape[1]
    rt_d = np.sqrt(d)
    ring_r = float(np.linalg.norm(C - mu, axis=1).mean())
    print(f"[polar] {args.formulation}: patch L{Lp}@pos{day_pos} -> read L{RO}@pos{ro_pos}; "
          f"ring radius {ring_r:.2f}", flush=True)

    def make_F_batched(tokens):
        pname, rname = f"blocks.{Lp}.hook_resid_post", f"blocks.{RO}.hook_resid_post"

        def F(X):
            store = {}
            B = X.shape[0]

            def patch(resid, hook):
                return torch.cat([resid[:, :day_pos, :], X.view(B, 1, -1),
                                  resid[:, day_pos + 1:, :]], dim=1)

            def rec(act, hook):
                store["x"] = act[:, ro_pos, :]
                return act
            model.run_with_hooks(tokens.expand(B, -1), return_type=None,
                                 fwd_hooks=[(pname, patch), (rname, rec)])
            return store["x"]
        return F

    def make_logits_batched(tokens):
        """Same patch, but returns the softmaxed next-token distribution at ``ro_pos``."""
        pname = f"blocks.{Lp}.hook_resid_post"

        def L(X_np):
            x = torch.as_tensor(np.ascontiguousarray(X_np), device=dev, dtype=dt)
            B = x.shape[0]

            def patch(resid, hook):
                return torch.cat([resid[:, :day_pos, :], x.view(B, 1, -1),
                                  resid[:, day_pos + 1:, :]], dim=1)
            with torch.no_grad():
                lg = model.run_with_hooks(tokens.expand(B, -1), return_type="logits",
                                          fwd_hooks=[(pname, patch)])
            return torch.softmax(lg[:, ro_pos, :].float(), dim=-1).cpu().numpy()
        return L

    def jvp_b(F, X_np, U_np):
        x = torch.as_tensor(np.ascontiguousarray(X_np), device=dev, dtype=dt)
        u = torch.as_tensor(np.ascontiguousarray(U_np), device=dev, dtype=dt)
        _, tang = torch.func.jvp(F, (x,), (u,))
        return tang.detach().float().cpu().numpy()

    us = np.linspace(0.0, 1.0, args.n_u, endpoint=False)
    rs = (np.array([float(v) for v in args.radii.split(",")]) if args.radii
          else np.linspace(0.0, args.r_max, args.n_r))
    UU, RR = np.meshgrid(us, rs, indexing="ij")            # [n_u, n_r]
    flat_u, flat_r = UU.ravel(), RR.ravel()
    P, VT, VR = polar_point(spline, mu, flat_u, flat_r)     # [M, d] each
    M = len(flat_u)

    rng = np.random.default_rng(cfg.seed)

    # Backgrounds, balanced across days: round-robin so 7 gives one per day, 49 gives all.
    bg = [i for i in range(len(specs)) if keep[i]]
    by_day = {k: [i for i in bg if day_ids[i] == k] for k in range(N_DAYS)}
    sel, rank = [], 0
    while len(sel) < args.n_prompts:
        grew = False
        for k in range(N_DAYS):
            if rank < len(by_day[k]) and len(sel) < args.n_prompts:
                sel.append(by_day[k][rank]); grew = True
        if not grew:
            break
        rank += 1
    print(f"[polar] {len(sel)} backgrounds, grid {args.n_u}x{args.n_r} = {M} points, "
          f"r up to {args.r_max}, {args.n_hutch} probes/background "
          f"(independent -> {len(sel) * args.n_hutch} effective for the mean field)", flush=True)

    # First token of each weekday, for the readout distribution.
    day_tok = np.array([days_token_ids(model)[k][0] for k in range(N_DAYS)])

    # A control for the tangential and radial gains: one random UNIT direction orthogonal
    # to the ring's 6-D span. ||J v|| is then on exactly the same footing as ||J t|| and
    # ||J r|| -- a unit step, its response measured -- but along a direction the ring
    # construction knows nothing about, so it says what a gain of this size looks like when
    # the day geometry is not involved. (Not the Frobenius norm over the complement: that
    # sums 4090 directions and is not comparable to a single-direction gain.)
    # One draw per background, so the mean field averages over as many independent
    # directions as there are backgrounds, the same way the Hutchinson probes do.
    SPAN = np.linalg.svd(C - mu, full_matrices=False)[2][:N_DAYS - 1]
    assert np.allclose(SPAN @ SPAN.T, np.eye(SPAN.shape[0]), atol=1e-6), "span not orthonormal"

    FRO = np.zeros((len(sel), M)); GT = np.zeros((len(sel), M)); GR = np.zeros((len(sel), M))
    GO = np.zeros((len(sel), M))                         # ||J v||, v a unit vector off the span
    VOFF = np.zeros((len(sel), d), np.float32)           # the direction used, for the record
    PSQ = np.zeros((len(sel), M, args.n_hutch))          # per-probe ||J u||^2
    PROB = np.zeros((len(sel), M, N_DAYS))               # P(weekday | steered state)
    PMASS = np.zeros((len(sel), M))                      # total mass on the 7 weekdays
    for n, pidx in enumerate(sel):
        tokens = model.to_tokens(texts[pidx], prepend_bos=cfg.prepend_bos)
        Fb = make_F_batched(tokens)
        Lb = make_logits_batched(tokens)
        s = int(day_ids[pidx])
        base_off = A_all[pidx, day_pos] - C[s]
        X = base_off[None, :] + P
        # Independent probes per background: the across-background mean then averages
        # len(sel) independent Hutchinson estimates instead of reusing one draw.
        U = rng.normal(size=(args.n_hutch, d))
        v = rng.normal(size=d)
        v -= SPAN.T @ (SPAN @ v)                         # strip everything the grid can reach
        v /= np.linalg.norm(v)
        assert np.abs(SPAN @ v).max() < 1e-6, "control direction is not orthogonal to the span"
        VOFF[n] = v
        for c0 in range(0, M, args.batch):
            idx = np.arange(c0, min(c0 + args.batch, M))
            Xb = X[idx]
            GT[n, idx] = np.linalg.norm(jvp_b(Fb, Xb, VT[idx]), axis=1)
            GR[n, idx] = np.linalg.norm(jvp_b(Fb, Xb, VR[idx]), axis=1)
            GO[n, idx] = np.linalg.norm(
                jvp_b(Fb, Xb, np.broadcast_to(v.astype(np.float32), Xb.shape)), axis=1)
            for q, u in enumerate(U):
                PSQ[n, idx, q] = np.sum(
                    jvp_b(Fb, Xb, np.broadcast_to(u, Xb.shape)) ** 2, axis=1)
            FRO[n, idx] = np.sqrt(PSQ[n, idx].mean(axis=1))
            pr = Lb(Xb)                                   # [b, vocab] softmaxed
            PROB[n, idx] = pr[:, day_tok]
            PMASS[n, idx] = pr[:, day_tok].sum(axis=1)
        print(f"[polar] {n + 1}/{len(sel)} {DAYS[s]} done", flush=True)

    shape = (args.n_u, len(rs))          # not args.n_r: --radii may override the count
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, us=us, rs=rs, fro=FRO.reshape((len(sel),) + shape),
             gain_t=GT.reshape((len(sel),) + shape), gain_r=GR.reshape((len(sel),) + shape),
             psq=PSQ.reshape((len(sel),) + shape + (args.n_hutch,)),
             gain_off=GO.reshape((len(sel),) + shape),
             span_basis=SPAN, v_off=VOFF,
             prob=PROB.reshape((len(sel),) + shape + (N_DAYS,)),
             pmass=PMASS.reshape((len(sel),) + shape),
             centroids=C, mu=mu, ring_radius=ring_r, days=np.array(DAYS, dtype=object),
             prompts=np.array([texts[i] for i in sel], dtype=object),
             src_day=np.array([int(day_ids[i]) for i in sel]),
             meta=np.array(json.dumps({
                 "formulation": args.formulation, "patch_layer": Lp, "readout_layer": RO,
                 "day_pos": day_pos, "ro_pos": ro_pos, "n_hutch": args.n_hutch,
                 "hour": args.hour, "model": cfg.model_name}), dtype=object))
    print(f"[polar] wrote {out}", flush=True)

    fro = FRO.mean(0).reshape(shape); gt = GT.mean(0).reshape(shape); gr = GR.mean(0).reshape(shape)
    knot = np.array([int(round(k * args.n_u / N_DAYS)) % args.n_u for k in range(N_DAYS)])
    gapi = np.array([int(round((k + 0.5) * args.n_u / N_DAYS)) % args.n_u for k in range(N_DAYS)])
    ir = int(np.argmin(np.abs(rs - 1.0)))
    print(f"\n[polar] at r = 1 (on the ring):   knot/gap  fro {fro[knot, ir].mean() / fro[gapi, ir].mean():.3f}"
          f"   tangential {gt[knot, ir].mean() / gt[gapi, ir].mean():.3f}"
          f"   radial {gr[knot, ir].mean() / gr[gapi, ir].mean():.3f}")
    print(f"[polar] radial profile (mean over u):")
    print(f"{'r':>6} {'||J||_F':>10} {'tangential':>12} {'radial':>10}")
    for j in range(0, len(rs), max(1, len(rs) // 8)):
        print(f"{rs[j]:>6.2f} {fro[:, j].mean():>10.2f} {gt[:, j].mean():>12.2f} {gr[:, j].mean():>10.2f}")


if __name__ == "__main__":
    main()
