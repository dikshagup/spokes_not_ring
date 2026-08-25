"""Walk the weekday ring by finite differences and record how far the readout moves.

Every steer layer in one pass, at one or more steer sites. Two readings per forward pass:
the readout residual's position on the day ring, and the Hellinger distance of the
restricted next-token distribution from the same prompt's unsteered one.

Writes an npz of the sweep. repro_fig1_combined_llama.sh (panels A, B) and
repro_fig6_combined_arith.sh (panels A-D) are the authority on the arguments.
"""
from __future__ import annotations

import argparse
import collections
import os
import time

import numpy as np
import torch

from weekday_manifold.manifold.behavior import hellinger_distance, restrict_to_concept
from weekday_manifold.manifold.config import ManifoldConfig
from weekday_manifold.manifold.days import DAYS, N_DAYS, build_prompts, days_token_ids
from weekday_manifold.manifold.probes import build_mention_early
from weekday_manifold.manifold.steering import (_arclength_u, fit_steer_spline,
                                                ring_report,
                                                template_demeaned_centroids, template_means)
from weekday_manifold.plateau.model import load_plateau_model
from weekday_manifold.utils import set_seed

SITES = ("weekday", "answer", "both")


def global_arclength(spline, n_u, n_days=N_DAYS, dense=4000):
    """Arc-length-uniform round the WHOLE loop, with the seven knots kept on the grid."""
    ug = np.linspace(0.0, 1.0, dense, endpoint=False)
    P = spline.forward(ug)
    seg = np.linalg.norm(np.diff(np.r_[P, P[:1]], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    ki = [int(np.argmin(np.abs(ug - k / n_days))) for k in range(n_days)]
    arc = np.array([(cum[ki[j + 1]] if j < n_days - 1 else cum[dense]) - cum[ki[j]]
                    for j in range(n_days)])
    total = n_u - 1
    exact = arc / arc.sum() * total
    cnt = np.floor(exact).astype(int)
    for j in np.argsort(-(exact - cnt))[:total - int(cnt.sum())]:
        cnt[j] += 1
    assert (cnt > 0).all() and cnt.sum() == total, "arc allocation is degenerate"
    uu, kidx = [], []
    for j in range(n_days):
        kidx.append(len(uu))
        targets = cum[ki[j]] + np.arange(cnt[j]) * arc[j] / cnt[j]
        got = np.interp(targets, cum, np.r_[ug, 1.0])
        # Pin the arc's first sample to the knot exactly. The dense table locates it only
        # to one of its own steps (2.5e-4 here), and this is the alpha = 0 sample: the
        # identity check compares the write against the recorded activation, so an
        # approximate knot is a real error rather than a rounding detail.
        got[0] = j / n_days
        uu.extend(got)
    # close the loop, as the per-arc scheme does: the grid is n_u long with the last
    # sample at u = 1, which is u = 0 come round again
    uu = np.array(uu + [1.0])
    assert len(uu) == n_u, f"grid is {len(uu)} long, expected {n_u}"
    assert np.allclose(uu[kidx], np.arange(n_days) / n_days, atol=1e-6), (
        "a day knot is not sampled exactly; the identity write would be off the grid")
    return uu, np.array(kidx)


def ring_samples(spline, n_u, param):
    """The spline parameters to sweep, one per sample, for one ring."""
    m, r = divmod(n_u - 1, N_DAYS)
    assert r == 0, f"n_u must be 7m+1 so the day knots are sampled exactly, got {n_u}"
    if param == "arclength-global":
        return global_arclength(spline, n_u)
    if param == "u":
        return np.linspace(0.0, 1.0, n_u), np.arange(N_DAYS) * m
    a = np.linspace(0.0, 1.0, m, endpoint=False)
    per_arc = []
    for k in range(N_DAYS):
        arc = _arclength_u(spline, k / N_DAYS, 1.0 / N_DAYS, a)
        # `_arclength_u` inverts a sampled arc-length table, so a=0 comes back as k/7 only to
        # ~1e-8. Pin it: this is the alpha = 0 sample, and the identity check compares the
        # write with the recorded activation, so it must be the knot exactly.
        arc[0] = k / N_DAYS
        per_arc.append(arc)
    return np.concatenate(per_arc + [np.array([1.0])]), np.arange(N_DAYS) * m


def knot_arclength(spline, n=20001):
    """Arc-length fraction round the loop at which each day knot sits."""
    u = np.linspace(0.0, 1.0, n)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(spline.forward(u), axis=0),
                                                        axis=1))])
    return np.interp(np.arange(N_DAYS) / N_DAYS, u, s / s[-1])


def spread_pos(A_layer, day_ids, template_ids):
    """Token position whose day-centroids are most spread out -- the weekday-mention site."""
    spreads = []
    for p in range(A_layer.shape[1]):
        Cp = template_demeaned_centroids(A_layer[:, p, :], day_ids, template_ids, N_DAYS)
        spreads.append(float(np.linalg.norm(Cp - Cp.mean(0), axis=1).mean()))
    return int(np.argmax(spreads))


def is_mention(formulation):
    """The two spellings of the mention family, in one place."""
    return formulation in ("mention_early", "me")


def resolve_family_defaults(args):
    """Fill --readout-group and --sites from the formulation when left unset."""
    if args.readout_group is None:
        args.readout_group = "input" if is_mention(args.formulation) else "answer"
    if args.sites is None:
        args.sites = "weekday" if is_mention(args.formulation) else ",".join(SITES)
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llama31_8b_fp16.json")
    ap.add_argument("--layers", default="2,6,10,14,18,22,26,28,30,31")
    ap.add_argument("--readout-layer", type=int, default=28)
    ap.add_argument("--readout-pos", type=int, default=-1)
    ap.add_argument("--n-prompts", type=int, default=0,
                    help="0 = every prompt, which is every (day x offset) pair. Anything "
                         "else takes one prompt per day, and since the prompt list is "
                         "ordered by offset that silently means the FIRST offset only -- so "
                         "the swept population stops matching the unpatched reference the "
                         "accuracy is compared against.")
    ap.add_argument("--n-u", type=int, default=141)
    ap.add_argument("--angle-every", type=int, default=4,
                    help="record the state's angular position on the ring at every Nth "
                         "depth; the readout depth is always included")
    ap.add_argument("--require-seq-len", type=int, default=0,
                    help="keep only prompts whose tokenised length is exactly this, "
                         "dropping whole templates so the days stay balanced. The sweep "
                         "addresses the steer and readout by INDEX, so every prompt must "
                         "put the same thing at the same position. Llama's BPE gives all 20 "
                         "mention templates 12 tokens with the full stop at 8; Mistral's "
                         "SentencePiece gives 12/13/14/15, so pass 12 there and 15 of the 20 "
                         "templates survive (105 prompts). 0 disables the filter.")
    ap.add_argument("--step-k", default="3,9",
                    help="also record TRUE k-step differences ||res(u+k) - res(u)||, "
                         "comma separated. The one-step difference subtracts two residuals "
                         "that are very close, so its rounding error does not shrink with "
                         "the step while the signal does -- in bfloat16 that floor was 69% "
                         "of the measured step, and it is 6% even in float16. A k-step "
                         "difference adds signal as k and isotropic rounding as sqrt(k), so "
                         "the floor falls as 1/sqrt(k). This CANNOT be recovered afterwards "
                         "from the saved one-step norms: summing k norms sums their noise "
                         "coherently too. Empty string to skip.")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--formulation", default="interrogative")
    # PER-POSITION by default. A single global choice cannot be right for the "both"
    # site: the weekday token encodes the day the question MENTIONS, the answer slot
    # encodes the day the model must SAY. Steering the answer slot along an input-day ring
    # drives a direction the unembedding does not read, which is why it scored at chance.
    ap.add_argument("--group-by", choices=["answer", "input", "auto"], default="auto")
    # The readout sits at the answer slot, so what lands there is the ANSWER day.
    # Left unset these follow the formulation. The mention family has no answer day at
    # all, so "answer" and the answer/both sites are not a different choice there, they
    # are an impossible one -- the asserts below used to reject the DEFAULTS, which made
    # `--formulation me` unrunnable without two extra flags nobody could guess from the
    # error. Resolving by formulation keeps the guard for an explicit wrong choice while
    # letting the defaults be right for both families.
    ap.add_argument("--readout-group", choices=["answer", "input"], default=None,
                    help="default: 'input' for the mention family, 'answer' otherwise")
    ap.add_argument("--patch-pos", type=int, default=None,
                    help="override the weekday-mention position (default: max day spread).")
    ap.add_argument("--sites", default=None,
                    help=f"comma-separated subset of {SITES}; default: 'weekday' for the "
                         f"mention family, all of them otherwise")
    ap.add_argument("--param", choices=["arclength-global", "arclength", "u"],
                    default="arclength-global",
                    help="how the ring is sampled: arc-length uniform per knot-to-knot arc "
                         "(default, constant input stride) or uniform in the spline "
                         "parameter (what the earlier per-layer sweep used).")
    ap.add_argument("--steer-mode", choices=["add", "pca-replace"], default="add",
                    help="'add' displaces the activation by sigma(u) - C_ref, keeping every "
                         "prompt-specific component, so alpha=0 is exactly the unmodified "
                         "run. 'pca-replace' is causalab's: fit PCA_k at the site, OVERWRITE "
                         "all k coordinates with the manifold point, hold the orthogonal "
                         "complement at base (methods/steer/collect.py replace_fn with "
                         "k_t = k_full = ambient_dim = 32). Every prompt then receives the "
                         "same k-D content, and alpha=0 is NOT the identity -- the prompt's "
                         "own deviation inside the subspace is discarded along with the day.")
    ap.add_argument("--pca-k", type=int, default=32,
                    help="feature-space width for pca-replace; causalab use 32")
    ap.add_argument("--out", default="figures/alpha_ladder_sites.npz")
    ap.add_argument("--dump-rings", default=None,
                    help="Save the ring CENTROIDS (and the readout ring) to this npz and "
                         "exit before the sweep. The sweep never needed the centroids "
                         "themselves, only the splines fitted to them, so figures that draw "
                         "the real rings would otherwise have to rebuild this setup -- and a "
                         "second copy of it could drift from the one that was measured.")
    ap.add_argument("--smoke-allow-multi-token-days", action="store_true",
                    help="CPU smoke tests on non-Llama tokenizers ONLY. The first-token "
                         "restriction misattributes a multi-token day's tail to 'other', so "
                         "the output distance is not the paper's metric under this flag; it "
                         "is recorded in the npz and the figure refuses to plot it.")
    args = ap.parse_args()

    resolve_family_defaults(args)
    sites = [s for s in args.sites.split(",") if s.strip()]
    assert all(s in SITES for s in sites), f"sites must be a subset of {SITES}"
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    RO = args.readout_layer
    # Layers at or after the readout are ALLOWED, but only the output reading is valid for
    # them: the layer-RO residual is computed before those blocks run, so it cannot see the
    # steer. They are recorded with NaN residual columns rather than being silently dropped
    # or, worse, contributing a flat line that reads as a measured null.
    late = [L for L in layers if L >= RO]
    if late:
        print(f"[sites] layers {late} are at or after the readout (L{RO}): OUTPUT ONLY -- "
              f"their residual columns are NaN by construction, not by measurement",
              flush=True)

    cfg = ManifoldConfig.from_json(args.config)
    set_seed(cfg.seed)
    model = load_plateau_model(cfg)
    assert all(0 <= L < model.cfg.n_layers for L in layers), (
        f"patch layers must be valid blocks 0..{model.cfg.n_layers - 1}; got {layers}")
    dev, dt = model.cfg.device, model.W_E.dtype

    # The mention family has no answer to speak of -- the day is named and nothing follows
    # from it -- so `answer_day` does not exist on those specs and the answer-side ring and
    # its groupings are meaningless there. Everything answer-shaped is therefore keyed off
    # this flag rather than assumed, and --group-by/--readout-group are pinned to `input`.
    mention = args.formulation in ("mention_early", "me")
    specs = (build_mention_early("thisday") if mention
             else build_prompts(args.formulation))
    # Keep only prompts of one tokenised length, when asked. The sweep addresses the steer
    # and readout sites by INDEX, so every prompt has to put the same thing at the same
    # position -- with Llama's BPE all 20 mention templates come to 12 tokens and the full
    # stop lands at 8 for every one, which is what makes a scalar --readout-pos meaningful.
    # Mistral's SentencePiece splits five of them differently (12, 13, 14, 15 tokens), so a
    # fixed index would read the full stop in some prompts and a content word in others.
    # Dropping whole templates keeps the days balanced, since each template covers all seven.
    if args.require_seq_len:
        keep = [sp for sp in specs
                if len(model.to_tokens(sp.text, prepend_bos=cfg.prepend_bos)[0])
                == args.require_seq_len]
        dropped = sorted({sp.meta["template"] for sp in specs}
                         - {sp.meta["template"] for sp in keep})
        assert keep, (f"no prompt tokenises to {args.require_seq_len} tokens; run the "
                      f"position check to find the right length for this tokenizer")
        by_day_n = collections.Counter(int(sp.meta["z"]) for sp in keep)
        assert len(set(by_day_n.values())) == 1, (
            f"the length filter unbalanced the days: {dict(sorted(by_day_n.items()))}")
        print(f"[sites] length filter: kept {len(keep)}/{len(specs)} prompts at "
              f"{args.require_seq_len} tokens, {by_day_n[0]} per day; dropped "
              f"{len(dropped)} template(s): {dropped}", flush=True)
        specs = keep
    texts = [sp.text for sp in specs]
    tn = sorted({sp.meta["template"] for sp in specs})
    ti = {t: i for i, t in enumerate(tn)}
    template_ids = np.array([ti[sp.meta["template"]] for sp in specs])
    z_in = np.array([int(sp.meta["z"]) for sp in specs])
    if mention:
        assert args.group_by != "answer" and args.readout_group != "answer", (
            "the mention family has no answer day; group by input")
        assert "answer" not in sites and "both" not in sites, (
            "the mention family has no answer slot to steer -- run --sites weekday")
        z_ans = z_in
    else:
        z_ans = np.array([int(sp.answer_day) for sp in specs])
    def labels_at(pos, dpos):
        """Day labels for the ring at `pos`: mentioned day at the weekday token, answer day at
        the answer slot. `--group-by input|answer` forces one for both."""
        if args.group_by == "input":
            return z_in
        if args.group_by == "answer":
            return z_ans
        return z_in if pos == dpos else z_ans

    day_ids = z_in if args.group_by != "answer" else z_ans   # only for the position search

    toks_all = torch.cat([model.to_tokens(t, prepend_bos=cfg.prepend_bos) for t in texts], 0)
    seq_len = toks_all.shape[1]
    ro_pos = args.readout_pos % seq_len

    # The seven weekday answer tokens. A multi-token day would have its tail silently
    # counted as "other" by the first-token restriction, so this is asserted, not assumed.
    dtoks = days_token_ids(model)
    multi = [DAYS[k] for k in range(N_DAYS) if len(dtoks[k]) != 1]
    single_token = not multi
    if multi:
        msg = (f"weekdays are not single tokens for this tokenizer: {multi} -- the "
               f"first-token restriction would misattribute their tails to the 'other' class")
        if not args.smoke_allow_multi_token_days:
            raise AssertionError(msg)
        print(f"[sites] WARNING (smoke mode): {msg}", flush=True)
    concept_ids = [int(dtoks[k][0]) for k in range(N_DAYS)]
    print(f"[sites] weekday first-token ids={concept_ids} single-token={single_token}",
          flush=True)

    # ---------------------------------------------------------------- one capture forward
    # Every patch layer and the readout layer in a single pass; the day-ring at each site is
    # fitted from these, so all sites/layers see the same activations.
    cap_layers = sorted(set(layers) | {RO})
    store = {}

    def mk(L):
        def fn(act, hook):
            store[L] = act.detach().float().cpu().numpy()
            return act
        return fn
    with torch.no_grad():
        model.run_with_hooks(toks_all, return_type=None,
                             fwd_hooks=[(f"blocks.{L}.hook_resid_post", mk(L))
                                        for L in cap_layers])
    A = {L: store[L] for L in cap_layers}                        # [N, seq, d] each
    d = A[RO].shape[2]

    # The UNSTEERED reference distribution q, run independently of the sweep so the identity
    # check at alpha = 0 compares two different code paths rather than one against itself.
    # Run at the SAME batch width as the sweep and with no hooks: bf16 GEMM kernels are
    # chosen by shape, so a reference run at a different batch size disagrees with the sweep
    # by ~2e-2 in Hellinger for reasons that have nothing to do with the patch.
    d0 = []
    for c0 in range(0, toks_all.shape[0], args.chunk):
        ch = toks_all[c0:c0 + args.chunk]
        pad = np.minimum(np.arange(args.chunk), ch.shape[0] - 1)
        with torch.no_grad():
            lg0 = model(ch[pad], return_type="logits")[:, ro_pos, :]
            probs0 = torch.softmax(lg0.float(), dim=-1).cpu().numpy()
        d0.append(restrict_to_concept(probs0, concept_ids)[:ch.shape[0]])
    dist0 = np.concatenate(d0, 0)                                # [N, 8]
    acc0 = float((dist0[:, :N_DAYS].argmax(1) == z_ans).mean())
    print(f"[sites] unpatched: argmax-over-weekdays accuracy={acc0:.3f} (chance "
          f"{1 / N_DAYS:.3f}); mass on the seven days="
          f"{dist0[:, :N_DAYS].sum(1).mean():.3f}", flush=True)

    # ------------------------------------------------------------------------- the rings
    day_pos = {L: (args.patch_pos % seq_len if args.patch_pos is not None
                   else spread_pos(A[L], day_ids, template_ids)) for L in layers}
    print(f"[sites] weekday-mention position by layer: {day_pos}; answer position {ro_pos} "
          f"of {seq_len}", flush=True)

    # the unpatched stream's own scale at the readout position, per depth. A displacement
    # of 5 is not the same intervention at layer 2 and at layer 26, and dividing by this is
    # what makes the depths comparable.
    scale_store = {}

    def _scale_hook(L_):
        def fn(act, hook):
            a = act[:, [day_pos_scale, ro_pos], :].detach().float().cpu().numpy()
            scale_store[L_] = np.linalg.norm(a, axis=2).mean(0)      # [2]
            return act
        return fn
    day_pos_scale = day_pos[layers[0]]
    with torch.no_grad():
        model.run_with_hooks(toks_all, return_type=None, fwd_hooks=[
            (f"blocks.{L_}.hook_resid_post", _scale_hook(L_))
            for L_ in range(model.cfg.n_layers)])

    # A ring at every DEPTH (not just the patch layers) at both positions, so the state's
    # angular coordinate can be read off wherever it is observed. Only the two positions
    # are captured, so this costs one forward and a few hundred MB rather than the 9 GB a
    # full-sequence capture at 32 depths would take.
    ang_depths = list(range(0, model.cfg.n_layers, args.angle_every))
    if RO not in ang_depths:
        ang_depths.append(RO)
    ang_depths = sorted(set(ang_depths))
    cap2 = {}

    def _mk2(L_):
        def fn(act, hook):
            cap2[L_] = act[:, [day_pos[layers[0]], ro_pos], :].detach().float().cpu().numpy()
            return act
        return fn
    with torch.no_grad():
        model.run_with_hooks(toks_all, return_type=None, fwd_hooks=[
            (f"blocks.{L_}.hook_resid_post", _mk2(L_)) for L_ in ang_depths])

    # For each (depth, position): the ring, its 6-D span, and the ring sampled densely IN
    # that span. The nearest point on a ring depends only on the query's component inside
    # the span -- everything orthogonal shifts all candidates equally -- so the search runs
    # in 6 dimensions instead of 4096 and is exact, not an approximation.
    ANG = {}
    for L_ in ang_depths:
        for q_, pos_ in enumerate((day_pos[layers[0]], ro_pos)):
            Xq = cap2[L_][:, q_, :]
            Cq = template_demeaned_centroids(Xq, z_in, template_ids, N_DAYS)
            muq = Cq.mean(0)
            Bq = np.linalg.svd(Cq - muq, full_matrices=False)[2][:N_DAYS - 1]
            spq = fit_steer_spline(Cq)
            ug = np.linspace(0.0, 1.0, 700, endpoint=False)
            ringq = (spq.forward(ug) - muq) @ Bq.T
            ANG[(L_, q_)] = (muq, Bq, ug, ringq, Xq.mean(0))
    print(f"[sites] angular rings at depths {ang_depths} x 2 positions", flush=True)

    rings = {}                       # (layer, pos) -> (centroids, spline, diagnostics)
    for L in layers:
        for pos in sorted({day_pos[L], ro_pos}):
            lab = labels_at(pos, day_pos[L])
            which = "input" if lab is z_in else "answer"
            C = template_demeaned_centroids(A[L][:, pos, :], lab, template_ids, N_DAYS)
            sp = fit_steer_spline(C)
            dg = ring_report(f"patch L{L}@pos{pos} ({which} day)", C, A[L][:, pos, :],
                             lab, template_ids, sp)
            uu, kidx = ring_samples(sp, args.n_u, args.param)
            # causalab's featurizer: PCA_k on the RAW activations at this site (no template
            # demeaning), with the ring fitted through the day centroids IN that space. The
            # spline is the same object, just living in k dimensions instead of d.
            pca = None
            if args.steer_mode == "pca-replace":
                X = A[L][:, pos, :].astype(np.float64)
                mu = X.mean(0)
                _, _, Wt = np.linalg.svd(X - mu, full_matrices=False)
                P = Wt[:args.pca_k]                                   # [k, d], orthonormal
                F = (X - mu) @ P.T
                pca = (mu, P, fit_steer_spline(np.stack(
                    [F[lab == k].mean(0) for k in range(N_DAYS)])))
            st = np.linalg.norm(np.diff(sp.forward(uu), axis=0), axis=1)
            print(f"[sites] ring L{L}@pos{pos}: input stride min={st.min():.4f} "
                  f"max={st.max():.4f} ratio={st.max() / st.min():.2f} | day knots at "
                  f"arc-length fraction {np.round(knot_arclength(sp), 3).tolist()} "
                  f"(k/7 would be {np.round(np.arange(N_DAYS) / N_DAYS, 3).tolist()})",
                  flush=True)
            rings[(L, pos)] = (C, sp, dg, uu, pca, kidx)

    ro_labels = z_ans if args.readout_group == "answer" else z_in
    C_ro = template_demeaned_centroids(A[RO][:, ro_pos, :], ro_labels, template_ids, N_DAYS)
    spline_ro = fit_steer_spline(C_ro)
    tmean_ro = template_means(A[RO][:, ro_pos, :], template_ids)
    dg_ro = ring_report(f"readout L{RO}@pos{ro_pos} ({args.readout_group})", C_ro,
                        A[RO][:, ro_pos, :], ro_labels, template_ids, spline_ro)

    if args.dump_rings:
        os.makedirs(os.path.dirname(args.dump_rings) or ".", exist_ok=True)
        keys = [(L, pos) for L in layers for pos in sorted({day_pos[L], ro_pos})]
        np.savez(args.dump_rings,
                 layers=np.array(layers), readout_layer=RO, ro_pos=ro_pos,
                 keys=np.array([f"L{L}@{p}" for L, p in keys]),
                 which=np.array(["input" if labels_at(p, day_pos[L]) is z_in else "answer"
                                 for L, p in keys]),
                 centroids=np.stack([rings[k][0] for k in keys]).astype(np.float32),
                 recovery=np.array([rings[k][2]["recovery"] for k in keys]),
                 radius=np.stack([rings[k][2]["radius"] for k in keys]),
                 centroids_ro=C_ro.astype(np.float32), recovery_ro=dg_ro["recovery"],
                 radius_ro=dg_ro["radius"], model=cfg.model_name,
                 # The residual stream grows in norm with depth, so a ring radius quoted in
                 # raw units grows too, whether or not the ring itself became any better
                 # defined. Saving the stream's own scale at each site lets a figure divide
                 # it out and show the RELATIVE size of the day code.
                 resid_norm=np.array([float(np.linalg.norm(A[L][:, pos, :], axis=1).mean())
                                      for L, pos in keys]))
        print(f"[sites] wrote rings -> {args.dump_rings}; exiting before the sweep",
              flush=True)
        return

    # ------------------------------------------------------------------------- the sweep
    us = np.linspace(0.0, 1.0, args.n_u)
    if args.n_prompts <= 0:
        prompts = list(range(len(texts)))          # every day x offset pair
    else:
        # Round-robin over the days, so any n is balanced across them. The old form took
        # the first prompt of each day and then capped the list, which meant it could never
        # return more than seven however large n was -- asking for 70 silently gave 7.
        by_day = {k: [i for i in range(len(texts)) if int(z_in[i]) == k]
                  for k in range(N_DAYS)}
        prompts, rank = [], 0
        while len(prompts) < args.n_prompts:
            grew = False
            for k in range(N_DAYS):
                if rank < len(by_day[k]) and len(prompts) < args.n_prompts:
                    prompts.append(by_day[k][rank]); grew = True
            if not grew:
                break
            rank += 1
        assert len(prompts) == min(args.n_prompts, len(texts)), (
            f"asked for {args.n_prompts} prompts, the family has {len(texts)}")
    P = len(prompts)
    # alpha = 0 for a prompt is ITS OWN day knot. Under the global scheme that index
    # depends on the ring -- arcs carry samples in proportion to their length -- so it is
    # read off the ring being patched rather than from a single global grid.
    def identity_index(L, pos, pidx):
        return int(rings[(L, pos)][5][int(z_in[pidx])])

    def sweep(tokens, patch_vals, Lp):
        """Forward with `patch_vals` (position -> [n_u, d]) written in at layer `Lp`."""
        n = next(iter(patch_vals.values())).shape[0]
        res, dists = [], []
        for c0 in range(0, n, args.chunk):
            # PAD every chunk to the same width. A ragged final batch is not just untidy:
            # bf16 GEMM kernel selection is shape-dependent, so the same prompt run in a
            # batch of 45 and a batch of 48 gives logits that differ by ~2e-2 in Hellinger
            # (measured directly: batches 45 and 46 disagree with 47/48/49, which agree
            # bit-for-bit). With 141 samples chunked at 48 that made the alpha=0 identity
            # check fail by that much for exactly the two prompts whose knot fell in the
            # short chunk. Padding removes the inhomogeneity instead of documenting it.
            take = min(args.chunk, n - c0)
            pad = np.minimum(np.arange(args.chunk), take - 1)     # repeat the last row
            vals = {p: torch.tensor(v[c0:c0 + take][pad], device=dev, dtype=dt)
                    for p, v in patch_vals.items()}
            b = args.chunk
            rec = {}

            def patch(resid, hook, vals=vals):
                resid = resid.clone()
                for p, v in vals.items():
                    resid[:, p, :] = v
                return resid

            # Read the residual at EVERY layer, not only at RO. The forward pass computes
            # them all regardless, so keeping them turns one readout depth into a full
            # steer-layer x readout-layer grid for no additional compute. Only the
            # per-layer DISTANCES are kept downstream; the activations themselves are far
            # too large to store and are not needed.
            # both positions: the readout, and the steered weekday token itself. The
            # second is not redundant -- at the steered token the write is visible from
            # its own layer onward, while at the readout nothing appears until the next
            # block's attention carries it across, so the two show different things.
            def read(act, hook, rec=rec):
                li_ = int(hook.name.split(".")[1])
                rec[li_] = act[:, [day_pos[Lp], ro_pos], :].detach().float().cpu().numpy()
                return act
            with torch.no_grad():
                lg = model.run_with_hooks(
                    tokens.expand(b, -1), return_type="logits",
                    fwd_hooks=[(f"blocks.{Lp}.hook_resid_post", patch)]
                    + [(f"blocks.{L_}.hook_resid_post", read)
                       for L_ in range(model.cfg.n_layers)])[:, ro_pos, :]
                pr = torch.softmax(lg.float(), dim=-1).cpu().numpy()
            res.append(np.stack([rec[L_][:take] for L_ in range(model.cfg.n_layers)], 1))
            dists.append(restrict_to_concept(pr, concept_ids)[:take])
        # [n_u, n_layers, 2, d]; position 0 is the weekday token, 1 the readout
        return np.concatenate(res, 0), np.concatenate(dists, 0)

    def angles(res_all):
        """Where the state sits ALONG the ring, per depth and position."""
        out = np.zeros((res_all.shape[0], len(ang_depths), 2), dtype=np.float32)
        for ai, L_ in enumerate(ang_depths):
            for q_ in (0, 1):
                muq, Bq, ug, ringq, _ = ANG[(L_, q_)]
                Z = (res_all[:, L_, q_, :] - muq) @ Bq.T
                out[:, ai, q_] = ug[((Z[:, None, :] - ringq[None]) ** 2).sum(-1).argmin(1)]
        return out

    S, Lc, U = len(sites), len(layers), args.n_u
    uout = np.zeros((S, Lc, P, U))
    hell = np.zeros((S, Lc, P, U))
    dists_all = np.zeros((S, Lc, P, U, N_DAYS + 1), dtype=np.float32)
    offring = np.zeros((S, Lc, P, U))
    # FULL-DIMENSIONAL residual path speed at the readout site: ||res(u_{k+1}) - res(u_k)||
    # over all d_model coordinates, no projection and no restriction. The Hellinger speed
    # derived from `dists` lives in the 8-class output simplex, which is Goodfire's metric
    # but is emphatically not full D; this is the full-D counterpart. Saved as a speed
    # rather than the residuals themselves -- those would be ~340 MB.
    speed = np.zeros((S, Lc, P, U - 1))
    stride = np.zeros((S, Lc, U - 1))       # input step length; same for every prompt
    # STRAIGHT-LINE distance in the readout residual, from the identity write to the steer
    # at u: ||res(u) - res(u_identity)||, full-dimensional. The counterpart of `hell`, which
    # is the same displacement measured in behaviour space -- so the two can be read against
    # each other. Not derivable afterwards from `speed`, which gives step norms and hence
    # only the CUMULATIVE distance along the path; a walk that returns towards where it
    # started has a long path and a short displacement, and the difference between those is
    # the thing worth seeing.
    d_res = np.zeros((S, Lc, P, U), dtype=np.float32)
    # the same displacement at EVERY readout depth: [site, steer layer, prompt, u, layer].
    # float32: at 70 prompts this is 3.2M values, 13 MB -- the earlier float16 was guarding
    # against a size that arithmetic does not support, and it cost 0.06% on every distance.
    nL = model.cfg.n_layers
    d_layer = np.zeros((S, Lc, P, U, nL, 2), dtype=np.float32)
    u_layer = np.zeros((S, Lc, P, U, len(ang_depths), 2), dtype=np.float32)
    step_layer = np.zeros((S, Lc, P, U - 1, nL, 2), dtype=np.float32)
    step_ks = [int(v) for v in args.step_k.split(",") if v.strip()]
    assert all(1 <= k < U for k in step_ks), f"--step-k out of range for U={U}"
    step_k = {k: np.zeros((S, Lc, P, U - k, nL, 2), dtype=np.float32) for k in step_ks}
    # the stream's own scale at each depth, unpatched, so a figure can divide it out: a
    # displacement of 5 means something different at layer 2 and at layer 26
    resid_scale = np.stack([scale_store[L_] for L_ in range(nL)]).astype(np.float32)
    h_ident = np.zeros((S, Lc, P))          # Hellinger(sweep alpha=0, unpatched forward)
    w_ident = np.zeros((S, Lc, P))          # max |written bf16 - recorded bf16| at alpha=0
    t0 = time.time()
    for si, site in enumerate(sites):
        for li, L in enumerate(layers):
            pos_list = ({day_pos[L]} if site == "weekday" else
                        {ro_pos} if site == "answer" else {day_pos[L], ro_pos})
            for pi, pidx in enumerate(prompts):
                tokens = model.to_tokens(texts[pidx], prepend_bos=cfg.prepend_bos)
                s, tmpl = int(z_in[pidx]), int(template_ids[pidx])
                s_ans = int(z_ans[pidx])
                # Each position is referenced to ITS OWN day and rotated on ITS OWN ring.
                #
                # Two things this gets right that a single (s, u) for both did not:
                #  * the reference. The weekday token sits at the MENTIONED day, the answer
                #    slot at the ANSWER day. Subtracting the mentioned-day centroid from the
                #    answer slot would not cancel, so alpha = 0 would not be the identity.
                #  * the phase. Steering the weekday token to day k means the answer should
                #    become k + offset. Putting BOTH positions at the same u therefore
                #    contradicts itself -- it tells the weekday token "day k" and the answer
                #    slot "day k" when the prompt demands k + offset. The two rings must
                #    rotate together with that offset held between them, which is what
                #    `du` below does: the answer ring leads by (s_ans - s)/7.
                du = ((s_ans - s) % N_DAYS) / N_DAYS
                patch_vals = {}
                for pos in sorted(pos_list):
                    C, sp, _, uu, pca, kidx_ring = rings[(L, pos)]
                    is_answer_slot = (pos == ro_pos and pos != day_pos[L])
                    ref = s_ans if (is_answer_slot and args.group_by != "input") else s
                    shift = du if (is_answer_slot and args.group_by != "input") else 0.0
                    if pca is None:
                        patch_vals[pos] = ((A[L][pidx, pos] - C[ref])[None, :]
                                           + sp.forward((uu + shift) % 1.0))
                    else:
                        mu, P, spp = pca
                        a = A[L][pidx, pos].astype(np.float64)
                        # hold the complement of the feature space at base, overwrite the
                        # rest with the manifold point -- replace_fn, k_t = k_full
                        a_perp = (a - mu) - ((a - mu) @ P.T) @ P
                        patch_vals[pos] = (mu + a_perp)[None, :] + \
                            np.asarray(spp.forward((uu + shift) % 1.0)) @ P
                res_all, dd = sweep(tokens, patch_vals, L)      # [U, nL, 2, d]
                res = res_all[:, RO, 1, :]                     # the readout position

                # A steer at or after the readout layer cannot be seen by the readout: the
                # layer-RO residual is computed before those blocks run, so `res` is simply
                # the unpatched value. Recording it would put a flat line in the residual
                # columns that looks like a measured null. The OUTPUT is downstream of every
                # block, so it stays valid -- these layers contribute to the output column
                # only, and the residual columns are NaN so the figure omits them.
                joint_all = np.concatenate([patch_vals[p] for p in sorted(patch_vals)],
                                           axis=1)
                stride[si, li] = np.linalg.norm(np.diff(joint_all, axis=0), axis=1)
                if L >= RO:
                    uout[si, li, pi] = np.nan
                    speed[si, li, pi] = np.nan
                    offring[si, li, pi] = np.nan
                    i0 = identity_index(L, sorted(pos_list)[0], pidx)
                    hell[si, li, pi] = [hellinger_distance(dd[j], dd[i0])
                                        for j in range(U)]
                    dists_all[si, li, pi] = dd
                    d_res[si, li, pi] = np.nan
                    d_layer[si, li, pi] = np.linalg.norm(res_all - res_all[i0][None],
                                                         axis=3)
                    step_layer[si, li, pi] = np.linalg.norm(np.diff(res_all, axis=0),
                                                            axis=3)
                    for k in step_ks:
                        step_k[k][si, li, pi] = np.linalg.norm(
                            res_all[k:] - res_all[:-k], axis=3)
                    u_layer[si, li, pi] = angles(res_all)
                    h_ident[si, li, pi] = np.nan
                    w_ident[si, li, pi] = np.nan
                    continue

                dem = res - tmean_ro[tmpl]
                uo = np.asarray(spline_ro.inverse(dem))
                # The input step length, over every patched position at once: for the
                # "both" site the intervention is one vector in the concatenated space of
                # the two slots, so its norm is the joint one. Dividing by it makes the
                # speed a GAIN -- readout distance moved per unit distance moved at the
                # patch site -- which is what removes the input-speed confound, whatever
                # the sampling leaves behind between arcs.
                speed[si, li, pi] = (np.linalg.norm(np.diff(res, axis=0), axis=1)
                                     / stride[si, li])
                uout[si, li, pi] = uo
                offring[si, li, pi] = (np.linalg.norm(dem - spline_ro.forward(uo), axis=1)
                                       / dg_ro["radius"].mean())
                dists_all[si, li, pi] = dd
                i0 = identity_index(L, sorted(pos_list)[0], pidx)
                hell[si, li, pi] = [hellinger_distance(dd[j], dd[i0]) for j in range(U)]
                d_res[si, li, pi] = np.linalg.norm(res - res[i0], axis=1)
                # where the state sits ALONG the ring, at each depth and position: the
                # steer names a target angle, and this is the angle actually reached
                for ai, L_ in enumerate(ang_depths):
                    for q_ in (0, 1):
                        muq, Bq, ug, ringq, _ = ANG[(L_, q_)]
                        Z = (res_all[:, L_, q_, :] - muq) @ Bq.T          # [U, 6]
                        d2 = ((Z[:, None, :] - ringq[None]) ** 2).sum(-1)
                        u_layer[si, li, pi, :, ai, q_] = ug[d2.argmin(1)]
                d_layer[si, li, pi] = np.linalg.norm(res_all - res_all[i0][None], axis=3)
                step_layer[si, li, pi] = np.linalg.norm(np.diff(res_all, axis=0), axis=3)
                for k in step_ks:
                    step_k[k][si, li, pi] = np.linalg.norm(res_all[k:] - res_all[:-k],
                                                           axis=3)
                u_layer[si, li, pi] = angles(res_all)
                # A steer written after block L cannot reach a DIFFERENT position before
                # block L+1's attention runs, so every depth up to and including L must be
                # bit-identical to unpatched there. Structural, and therefore worth
                # asserting: if it ever fails, the patch is leaking somewhere it should not.
                #
                # It only holds when the write and the readout are at different positions.
                # The "answer" site writes AT the readout position, where the value is its
                # own from depth L onwards with no attention needed to carry it -- so
                # asserting it there says a steer must not change the thing it just set.
                if ro_pos not in pos_list:
                    assert np.abs(d_layer[si, li, pi, :, :L + 1, 1]).max() == 0.0, (
                        f"steer at L{L} moved the readout at or before its own depth")

                # --- the alpha = 0 identity, checked three ways -------------------------
                # (1) the tensor actually written at the prompt's own knot is the recorded
                #     activation. Tolerance is one bf16 ulp and not zero on principle: the
                #     write is A - C_s + sigma(s/7) evaluated in float64, and sigma(s/7)
                #     equals C_s only to ~1e-16, so a single element could in principle land
                #     the other side of a bf16 rounding boundary. In practice it is exact --
                #     `w_identity` in the npz is 0 -- and the tolerance never bites.
                if args.steer_mode != "add":
                    # replacement discards the prompt's own component inside the feature
                    # space, so the write at its own knot is the day PROTOTYPE, not the
                    # recorded activation. Record how far that is instead of asserting it.
                    w_ident[si, li, pi] = max(
                        float(np.abs(v[i0] - A[L][pidx, p]).max())
                        for p, v in patch_vals.items())
                    h_ident[si, li, pi] = hellinger_distance(dd[i0], dist0[pidx])
                    continue
                w = max(float((torch.tensor(v[i0], dtype=dt)
                               - torch.tensor(A[L][pidx, p], dtype=dt)).abs().max())
                        for p, v in patch_vals.items())
                tol = 2.0 ** -7 * max(float(np.abs(A[L][pidx, p]).max()) for p in patch_vals)
                # (2) the output distribution there equals an INDEPENDENTLY run unpatched
                #     forward's. Not asserted at 0: that pass has batch 49, the sweep's has
                #     batch <= chunk, and bf16 kernels need not select the same reduction
                #     order for both. A real identity failure moves this to O(0.5), not 1e-3.
                h = hellinger_distance(dd[i0], dist0[pidx])
                w_ident[si, li, pi], h_ident[si, li, pi] = w, h
                assert w <= tol, (f"{site} L{L} {DAYS[s]}: the alpha=0 write differs from the "
                                  f"recorded activation by {w:g} (> one bf16 ulp {tol:g})")
                assert h < 0.05, (f"{site} L{L} {DAYS[s]}: the alpha=0 output distribution is "
                                  f"{h:g} from an unpatched forward, not 0")
                # (3) the plotted curve is exactly 0 at alpha = 0, by construction.
                assert hell[si, li, pi, i0] == 0.0, "the ladder is not 0 at alpha=0"
            print(f"[sites] {site:8s} L{L:<3d} done  ({time.time() - t0:.0f}s)  "
                  f"identity: write max|diff|={w_ident[si, li].max():g} "
                  f"H(alpha=0, unpatched)={h_ident[si, li].max():.2e} | "
                  f"steered responses {offring[si, li].mean():.1f} ring radii off the readout "
                  f"ring | output distance at the far side of the ring="
                  f"{hell[si, li].max():.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ring_keys = [(L, pos) for L in layers for pos in sorted({day_pos[L], ro_pos})]
    np.savez(
        args.out, us=us, layers=np.array(layers), sites=np.array(sites), uout=uout,
        hell=hell, dists=dists_all, offring=offring, speed_fulld=speed, stride_in=stride,
        d_res=d_res, d_layer=d_layer, step_layer=step_layer, resid_scale=resid_scale,
        step_k_values=np.array(step_ks),
        **{f"step_layer_k{k}": step_k[k] for k in step_ks},
        u_layer=u_layer, ang_depths=np.array(ang_depths),
        knot_index=np.stack([rings[(L, day_pos[L])][5] for L in layers]),
        param=args.param, h_identity=h_ident, w_identity=w_ident,
        prompts=np.array(prompts), prompt_days=np.array([int(z_in[p]) for p in prompts]),
        prompt_texts=np.array([texts[p] for p in prompts]),
        day_pos=np.array([day_pos[L] for L in layers]), ro_pos=ro_pos, seq_len=seq_len,
        readout_layer=RO, d_model=d, n_prompts=P, concept_ids=np.array(concept_ids),
        steer_mode=args.steer_mode, pca_k=args.pca_k,
        single_token_days=single_token, clean_acc=acc0, dist_unpatched=dist0.astype(np.float32),
        formulation=args.formulation, group_by=args.group_by,
        readout_group=args.readout_group, model=cfg.model_name,
        ring_sites=np.array([f"L{L}@{pos}" for L, pos in ring_keys]),
        ring_radius=np.array([rings[k][2]["radius"] for k in ring_keys]),
        ring_scatter=np.array([rings[k][2]["scatter"] for k in ring_keys]),
        ring_recovery=np.array([rings[k][2]["recovery"] for k in ring_keys]),
        ring_u=np.array([rings[k][3] for k in ring_keys]),
        ring_knot_alpha=np.array([knot_arclength(rings[k][1]) for k in ring_keys]),
        ring_radius_ro=dg_ro["radius"], ring_scatter_ro=dg_ro["scatter"],
        ring_recovery_ro=dg_ro["recovery"])
    print(f"[sites] wrote {args.out}  ({time.time() - t0:.0f}s of forward passes)", flush=True)


if __name__ == "__main__":
    main()
