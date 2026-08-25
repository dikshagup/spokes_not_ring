#!/usr/bin/env python
"""Project the corpus windows into the weekday subspace, and fix the fit/score split.

The split is what the two raw passes read rather than recompute, so the plane is fitted on
exactly the windows the published figure fits on.

Writes projections_n16.npz. See repro_fig3_arc_occupancy.sh.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

# Large intermediates default under the user cache rather than a fixed pod path, so this
# runs without root and off the working tree. repro_fig3_arc_occupancy.sh passes these
# explicitly (CORPUS= / RAWDIR=), so the published figure does not depend on the default.
_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                      "weekday-manifold")
CORPUS_DIR = os.path.join(_CACHE, "corpus_v2")
RAW_DIR = os.path.join(_CACHE, "corpus_v2_raw")

SUBSPACES = ("fineweb", "m1", "m3")


def assert_local_package():
    """Fail loudly if `weekday_manifold` resolves to a different worktree than this script."""
    import weekday_manifold.manifold.capture as _c
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    got = os.path.abspath(_c.__file__)
    if not got.startswith(here + os.sep):
        raise SystemExit(
            f"`weekday_manifold` resolves to {got}\n"
            f"but this script lives in {here}.\n"
            f"Re-run with PYTHONPATH={os.path.join(here, 'src')}")
    return got


def fit_fineweb_subspace(model, ids_fit, days_fit, layers, n_dims, prepend_bos,
                         batch_size, dev):
    """PRIMARY subspace: discriminative fit on held-out FineWeb weekday windows."""
    n_layers = len(layers)
    d = model.cfg.d_model
    sums = torch.zeros(7, n_layers, d, dtype=torch.float64, device=dev)
    counts = np.zeros(7, dtype=np.int64)
    grab = {}

    def make_hook(li):
        def hook(act, hook):
            grab[li] = act[:, -1, :].detach().float()
        return hook

    fwd = [(f"blocks.{l}.hook_resid_post", make_hook(li))
           for li, l in enumerate(layers)]
    bos = (torch.tensor([[model.tokenizer.bos_token_id]], device=dev)
           if prepend_bos else None)

    for s in range(0, len(ids_fit), batch_size):
        chunk = torch.tensor(ids_fit[s:s + batch_size].astype(np.int64), device=dev)
        if bos is not None:
            chunk = torch.cat([bos.expand(chunk.shape[0], 1), chunk], dim=1)
        with torch.no_grad():
            model.run_with_hooks(chunk, return_type=None, fwd_hooks=fwd)
        dd = days_fit[s:s + len(chunk)]
        for li in range(n_layers):
            h = grab[li].double()
            for day in np.unique(dd):
                sums[day, li] += h[torch.tensor(dd == day, device=dev)].sum(0)
        for day in np.unique(dd):
            counts[day] += int((dd == day).sum())
        grab.clear()

    if (counts == 0).any():
        raise SystemExit(f"fit split is missing days {np.where(counts == 0)[0]}")
    print(f"[fineweb-fit] per-day counts {counts.tolist()}", flush=True)

    cnt = torch.tensor(counts, dtype=torch.float64, device=dev)[:, None, None]
    day_means = sums / cnt                                   # [7, L, d]
    global_mean = (sums.sum(0) / cnt.sum())                  # [L, d]
    mus, bases = {}, {}
    for li, l in enumerate(layers):
        fit_matrix = (day_means[:, li, :] - global_mean[li]).cpu().numpy()
        _, _, Vt = np.linalg.svd(fit_matrix, full_matrices=False)
        bases[l] = torch.tensor(Vt[:n_dims], dtype=torch.float32, device=dev)
        mus[l] = global_mean[li].float()
    return mus, bases, counts


def build_subspaces(model, cfg, layers, site, n_dims, n_null, seed, use_cache=True):
    """Per-layer (mean, weekday basis, [n_null, k, d] random bases) on the GPU."""
    from weekday_manifold.manifold.capture import capture_manifold_activations_all_layers
    from weekday_manifold.manifold.competence import annotate_model_correct
    from weekday_manifold.manifold.prompt_library import (annotate_token_lengths,
                                                 assign_splits, build_library)
    from weekday_manifold.manifold.subspace import fit_subspace, three_manifold_specs

    specs = build_library()
    annotate_token_lengths(specs, model, prepend_bos=cfg.prepend_bos)
    assign_splits(specs)
    # REQUIRED, and it fails silently if omitted. M1's selector is
    # sel_and(sel_train, sel_offset_core), and sel_offset_core admits a COMPUTE
    # prompt only when meta["model_correct"] is True. build_library initialises
    # that key to None, and only annotate_model_correct sets it -- so without this
    # call every compute prompt is rejected, all seven days are still present via
    # the read prompts, fit_subspace succeeds, and "M.read+compute_train" is
    # quietly fitted on READ PROMPTS ALONE. That is M3 wearing M1's name: no
    # error, no warning, a plausible subspace that is not the registered one.
    print("[competence] scoring the library for model_correct ...", flush=True)
    annotate_model_correct(model, specs, prepend_bos=cfg.prepend_bos)
    n_ok = sum(1 for s in specs if s.meta.get("model_correct") is True)
    n_comp = sum(1 for s in specs if s.meta.get("role") == "compute")
    print(f"[competence] {n_ok}/{n_comp} compute prompts correct", flush=True)
    if n_ok == 0:
        raise SystemExit("no compute prompt is model_correct -- M1 would silently "
                         "degrade to a read-only fit. Refusing to continue.")
    # WHERE THIS CACHE LIVES, because it is not obvious and it is large. On a hit or a
    # miss it reads/writes cfg.cache_dir, which defaults to the RELATIVE path
    # "data/activations" -- and every repro_*.sh does `cd "$(dirname "$0")/.."` first, so
    # it always lands INSIDE THE REPO. The file this run writes is ~642 MB. It is
    # gitignored, but it is not free: on a small container disk it is the single largest
    # thing this pipeline produces, and it is not the output anyone came for.
    #
    # It is also the reason a second run of figure 3 is not a from-scratch run: keyed by
    # (model, flags, sites, prompts), it returns the stored activations instead of
    # recapturing, so a rerun can look like a reproduction while touching no GPU at all.
    # --no-cache exists to make "actually recapture" expressible; it suppresses the write
    # as well as the read, so a from-scratch run also costs nothing on disk here.
    print(f"[library] {len(specs)} prompts; capturing all layers at {site!r} "
          f"({'cached after the first run' if use_cache else 'cache DISABLED'})", flush=True)
    store = capture_manifold_activations_all_layers(model, specs, cfg, sites=(site,),
                                                    use_cache=use_cache)
    metas = [s.meta for s in specs]

    dev = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    # index 0 = M1 (read+compute pooled), index 2 = M3 (read only). Both are
    # SECONDARY here: they are fitted on 1-14 tokens of left context against our
    # 64-256, so they answer "is the constructed weekday code the same subspace"
    # rather than defining the one we score against.
    mus = {"m1": {}, "m3": {}}
    bases = {"m1": {}, "m3": {}}
    for l in layers:
        for name, idx in (("m1", 0), ("m3", 2)):
            spec = three_manifold_specs(l, site, n_dims=n_dims)[idx]
            m = fit_subspace(store, metas, spec)
            mus[name][l] = torch.tensor(m.pca.mean, dtype=torch.float32, device=dev)
            bases[name][l] = torch.tensor(m.pca.components, dtype=torch.float32,
                                          device=dev)
            if l == layers[0]:
                print(f"[subspace] L{l} {name}: basis {bases[name][l].shape} "
                      f"({spec.name})", flush=True)
    return mus, bases, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=CORPUS_DIR)
    ap.add_argument("--n-max", type=int, default=256, help="length the list was built at")
    ap.add_argument("--n", type=int, default=64, help="sweep length: ids[:, -n:]")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--site", default="mention_token")
    ap.add_argument("--n-dims", type=int, default=6)
    ap.add_argument("--n-null", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-frac", type=float, default=0.34,
                    help="fraction of positive DOCUMENTS used to fit the primary "
                         "FineWeb subspace; the rest are scored")
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N windows")
    ap.add_argument("--no-cache", action="store_true",
                    help="recapture the prompt-library activations instead of reading "
                         "cfg.cache_dir (data/activations, ~642 MB, inside the repo), and "
                         "do not write them either. Pass this whenever the point of the "
                         "run is to reproduce from scratch -- without it a rerun silently "
                         "reuses stored activations and never touches the GPU for this "
                         "stage.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"[env] pivotal -> {assert_local_package()}", flush=True)
    from weekday_manifold.manifold.config import ManifoldConfig
    from weekday_manifold.plateau.model import load_plateau_model

    blob = np.load(os.path.join(args.inp, f"capture_list_n{args.n_max}.npz"),
                   allow_pickle=True)
    ids_all = blob["ids"]
    # --limit must SAMPLE, not take a prefix: the list is grouped by class in
    # alphabetical order, so a prefix is entirely "dirty_positive" -- no positives,
    # no seven days, and the fit dies on a missing-day error that says nothing
    # about the plumbing being tested.
    sel = np.arange(len(ids_all))
    if args.limit and args.limit < len(sel):
        sel = np.sort(np.random.default_rng(args.seed).choice(
            len(sel), args.limit, replace=False))
    ids_all = ids_all[sel][:, -args.n:]            # prefix truncation, prereg 2
    meta = {k: np.asarray(blob[k])[sel] for k in blob.files
            if k not in ("ids", "n_max", "model", "seed")}
    cls = meta["cls"]
    print(f"[list] {ids_all.shape} windows at n={args.n}: "
          f"{dict(collections.Counter(cls.tolist()))}", flush=True)

    cfg = ManifoldConfig(model_name=args.model, dtype=args.dtype)
    print(f"[model] loading {args.model} ({args.dtype}) ...", flush=True)
    model = load_plateau_model(cfg)
    cfg = cfg.resolve(model.cfg.n_layers)
    layers = list(range(model.cfg.n_layers))
    dev = next(model.parameters()).device

    # ---- document-disjoint fit/score split of the positives -----------------
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
            "Sunday"]
    day_ix = {d: i for i, d in enumerate(DAYS)}
    docs = meta["doc"]
    day_s = meta["day"]
    pos_mask = cls == "positive"
    # Split by DOCUMENT, not by window, so no document contributes to both the fit
    # and the thing the fit is used to score. Without this the positive scores are
    # circular -- the classic failure the Phase-4 plan flagged as the single most
    # important methodology point.
    rs = np.random.default_rng(args.seed)
    pos_docs = np.unique(docs[pos_mask])
    rs.shuffle(pos_docs)
    fit_docs = set(pos_docs[:int(round(args.fit_frac * len(pos_docs)))].tolist())
    is_fit = pos_mask & np.array([d in fit_docs for d in docs])
    print(f"[split] positives {int(pos_mask.sum())} from {len(pos_docs)} docs -> "
          f"fit {int(is_fit.sum())} / score {int((pos_mask & ~is_fit).sum())} "
          f"(document-disjoint)", flush=True)

    # ---- content-balance check: does day correlate with domain composition? --
    urls = meta["url"]
    dom = np.array([str(u).split("/")[2].lower() if str(u).count("/") > 2 else ""
                    for u in urls])
    print("[content] top domain share within each day's positives "
          "(a day that looks different in CONTENT would leak into its centroid):")
    for d in DAYS:
        m = pos_mask & (day_s == d)
        if not m.any():
            continue
        c = collections.Counter(dom[m]).most_common(3)
        tot = int(m.sum())
        print(f"    {d:<10} n={tot:>5}  top3 = "
              + ", ".join(f"{k or '?'} {100*v/tot:.1f}%" for k, v in c), flush=True)

    fit_ids = ids_all[is_fit]
    fit_days = np.array([day_ix[d] for d in day_s[is_fit]])
    print(f"[fineweb-fit] fitting on {len(fit_ids)} windows at n={args.n} ...",
          flush=True)
    fw_mus, fw_bases, fw_counts = fit_fineweb_subspace(
        model, fit_ids, fit_days, layers, args.n_dims, cfg.prepend_bos,
        args.batch_size, dev)

    lib_mus, lib_bases, lib_nulls = build_subspaces(
        model, cfg, layers, args.site, args.n_dims, args.n_null, args.seed,
        use_cache=not args.no_cache)

    MUS = {"fineweb": fw_mus, "m1": lib_mus["m1"], "m3": lib_mus["m3"]}
    BASES = {"fineweb": fw_bases, "m1": lib_bases["m1"], "m3": lib_bases["m3"]}
    rng = np.random.default_rng(args.seed + 1)
    NULLS = {}
    for name in SUBSPACES:
        NULLS[name] = {}
        for l in layers:
            d = MUS[name][l].shape[0]
            nb = np.stack([np.linalg.qr(rng.normal(size=(d, args.n_dims)))[0].T
                           for _ in range(args.n_null)])
            NULLS[name][l] = torch.tensor(nb, dtype=torch.float32, device=dev)

    # ---- principal angles: is the natural weekday code the constructed one? --
    ang = {}
    for other in ("m1", "m3"):
        ang[other] = {}
        for l in layers:
            sv = np.linalg.svd(
                (BASES["fineweb"][l] @ BASES[other][l].T).cpu().numpy(),
                compute_uv=False)
            ang[other][l] = sv.tolist()
    for other in ("m1", "m3"):
        for l in (layers[len(layers) // 4], layers[len(layers) // 2],
                  layers[3 * len(layers) // 4]):
            sv = np.round(ang[other][l], 3)
            print(f"[angles] L{l:<3} fineweb vs {other}: cos = {sv}  "
                  f"(mean {np.mean(sv):.3f})", flush=True)

    N, L, K = len(ids_all), len(layers), args.n_dims
    S = len(SUBSPACES)
    z = np.zeros((N, L, S, K), dtype=np.float32)
    # Only the null ENERGY is kept, not the null coordinates. Enrichment is
    # ||z||^2 / mean_j ||z_null_j||^2 -- the ||x - mu||^2 denominator cancels
    # between numerator and null because both are computed on the SAME centred
    # vector, which is the whole point of using a random subspace rather than k/d
    # as the baseline. Keeping the 6-vectors instead would cost 2.2 GB to store
    # information no estimator uses.
    null_e = np.zeros((N, L, S, 2), dtype=np.float32)   # (mean, std) over draws
    norm = np.zeros((N, L, S), dtype=np.float32)        # ||x - mu||, mu differs per fit

    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]
    grab = {}

    def make_hook(l):
        def hook(act, hook):        # act [B, P, d]
            grab[l] = act[:, -1, :].detach().float()      # capture site = final token
        return hook

    fwd_hooks = [(n, make_hook(l)) for l, n in zip(layers, hook_names)]
    bos = torch.tensor([[model.tokenizer.bos_token_id]], device=dev) \
        if cfg.prepend_bos else None

    t0 = time.time()
    for s in range(0, N, args.batch_size):
        chunk = torch.tensor(ids_all[s:s + args.batch_size].astype(np.int64),
                             device=dev)
        if bos is not None:
            chunk = torch.cat([bos.expand(chunk.shape[0], 1), chunk], dim=1)
        with torch.no_grad():
            model.run_with_hooks(chunk, return_type=None, fwd_hooks=fwd_hooks)
        for li, l in enumerate(layers):
            h = grab[l]                                   # [B, d]
            B = len(h)
            for si, name in enumerate(SUBSPACES):
                c = h - MUS[name][l]                      # each fit has its own mean
                z[s:s + B, li, si] = (c @ BASES[name][l].T).cpu().numpy()
                norm[s:s + B, li, si] = c.norm(dim=1).cpu().numpy()
                zn = torch.einsum("bd,nkd->bnk", c, NULLS[name][l])
                e = (zn ** 2).sum(-1)                      # [B, n_null]
                null_e[s:s + B, li, si, 0] = e.mean(-1).cpu().numpy()
                null_e[s:s + B, li, si, 1] = e.std(-1).cpu().numpy()
        grab.clear()
        if s and (s // args.batch_size) % 50 == 0:
            el = time.time() - t0
            print(f"[capture] {s}/{N}  {el:.0f}s  eta {el/s*(N-s):.0f}s", flush=True)

    out = args.out or os.path.join(args.inp, f"projections_n{args.n}.npz")
    np.savez_compressed(out, z=z, null_e=null_e, norm=norm, is_fit=is_fit,
                        subspaces=np.array(SUBSPACES), fw_day_counts=fw_counts,
                        angles_m1=np.array([ang["m1"][l] for l in layers]),
                        angles_m3=np.array([ang["m3"][l] for l in layers]),
                        layers=np.array(layers), n=args.n, site=args.site,
                        n_dims=K, model=args.model, prepend_bos=cfg.prepend_bos,
                        **meta)
    print(f"\n[saved] {out}  z={z.shape} null_e={null_e.shape} "
          f"({os.path.getsize(out)/1e6:.0f} MB) in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
