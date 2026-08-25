#!/usr/bin/env python
"""Dump the raw 4096-d activation of every corpus window at one layer.

Writes raw_L28_n16.npz. See repro_fig3_arc_occupancy.sh.
"""

from __future__ import annotations

import argparse
import collections
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


def assert_local_package():
    """Same guard as capture_corpus_windows.py -- see its docstring for why."""
    import weekday_manifold.manifold.capture as _c
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    got = os.path.abspath(_c.__file__)
    if not got.startswith(here + os.sep):
        raise SystemExit(
            f"`weekday_manifold` resolves to {got}\n"
            f"but this script lives in {here}.\n"
            f"Re-run with PYTHONPATH={os.path.join(here, 'src')}")
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=CORPUS_DIR)
    ap.add_argument("--n-max", type=int, default=256, help="length the list was built at")
    ap.add_argument("--n", type=int, default=16, help="sweep length: ids[:, -n:]")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--classes", default="all",
                    help="comma-separated classes to capture, or 'all'. Panel A draws "
                         "positives and matched_negative; the default keeps everything "
                         "so a later question about near_miss/floor needs no re-run.")
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N windows")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"[env] weekday_manifold -> {assert_local_package()}", flush=True)
    from weekday_manifold.manifold.config import ManifoldConfig
    from weekday_manifold.plateau.model import load_plateau_model

    # ---- the window list, exactly as the projections pass read it ------------
    blob = np.load(os.path.join(args.inp, f"capture_list_n{args.n_max}.npz"),
                   allow_pickle=True)
    ids_all = blob["ids"][:, -args.n:]                  # prefix truncation, prereg 2
    cls = np.asarray(blob["cls"]).astype(str)
    day = np.asarray(blob["day"]).astype(str)
    doc = np.asarray(blob["doc"]).astype(str)

    # ---- the split, READ from the cached projections rather than recomputed --
    proj_path = os.path.join(args.inp, f"projections_n{args.n}.npz")
    pb = np.load(proj_path, allow_pickle=True)
    is_fit = np.asarray(pb["is_fit"]).astype(bool)
    if len(is_fit) != len(ids_all):
        raise SystemExit(f"{proj_path} has {len(is_fit)} rows, capture list has "
                         f"{len(ids_all)} -- row order cannot be assumed")
    # Row-for-row identity, not just matching lengths: if these disagree the two
    # files are describing different windows and the fitted plane would be applied
    # to the wrong activations, silently.
    for k, mine in (("cls", cls), ("day", day), ("doc", doc)):
        theirs = np.asarray(pb[k]).astype(str)
        if not np.array_equal(mine, theirs):
            bad = int((mine != theirs).sum())
            raise SystemExit(f"{k} differs between capture list and {proj_path} "
                             f"in {bad} rows -- lists are out of sync")
    print(f"[split] verified against {os.path.basename(proj_path)}: "
          f"positives {int((cls == 'positive').sum())}, "
          f"fit {int(is_fit.sum())}, held out {int(((cls == 'positive') & ~is_fit).sum())}",
          flush=True)

    keep = np.ones(len(ids_all), bool) if args.classes == "all" else \
        np.isin(cls, [c.strip() for c in args.classes.split(",")])
    if args.limit and args.limit < int(keep.sum()):
        # Sample, do not take a prefix: the list is grouped by class, so a prefix
        # is one class only -- the same trap capture_corpus_windows.py documents.
        hit = np.where(keep)[0]
        keep = np.zeros(len(ids_all), bool)
        keep[np.sort(np.random.default_rng(0).choice(hit, args.limit, replace=False))] = True

    sel = np.where(keep)[0]
    ids_all, cls, day, doc, is_fit = (a[sel] for a in (ids_all, cls, day, doc, is_fit))
    N = len(sel)
    print(f"[list] {N} windows at n={args.n}: "
          f"{dict(collections.Counter(cls.tolist()))}", flush=True)

    cfg = ManifoldConfig(model_name=args.model, dtype=args.dtype)
    print(f"[model] loading {args.model} ({args.dtype}) ...", flush=True)
    model = load_plateau_model(cfg)
    cfg = cfg.resolve(model.cfg.n_layers)
    dev = next(model.parameters()).device
    d_model = model.cfg.d_model
    if not 0 <= args.layer < model.cfg.n_layers:
        raise SystemExit(f"layer {args.layer} outside 0..{model.cfg.n_layers - 1}")

    acts = np.zeros((N, d_model), dtype=np.float16)
    grab = {}

    def hook(act, hook):            # act [B, P, d]
        grab["h"] = act[:, -1, :].detach().float()        # capture site = final token

    fwd_hooks = [(f"blocks.{args.layer}.hook_resid_post", hook)]
    bos = torch.tensor([[model.tokenizer.bos_token_id]], device=dev) \
        if cfg.prepend_bos else None

    # Nothing after --layer can affect a resid_post hook AT that layer, so the
    # remaining blocks are dead work. Guarded rather than assumed: stop_at_layer is
    # a transformer_lens forward kwarg and a version without it must not silently
    # capture nothing.
    stop_kw = {"stop_at_layer": args.layer + 1}
    try:
        with torch.no_grad():
            model.run_with_hooks(
                torch.tensor(ids_all[:1].astype(np.int64), device=dev)
                if bos is None else torch.cat(
                    [bos, torch.tensor(ids_all[:1].astype(np.int64), device=dev)], dim=1),
                return_type=None, fwd_hooks=fwd_hooks, **stop_kw)
        if "h" not in grab:
            raise RuntimeError("hook did not fire under stop_at_layer")
        print(f"[speed] stop_at_layer={args.layer + 1} of {model.cfg.n_layers}", flush=True)
    except (TypeError, RuntimeError) as e:
        stop_kw = {}
        print(f"[speed] stop_at_layer unavailable ({type(e).__name__}), running "
              f"all {model.cfg.n_layers} layers", flush=True)
    grab.clear()

    t0 = time.time()
    for s in range(0, N, args.batch_size):
        chunk = torch.tensor(ids_all[s:s + args.batch_size].astype(np.int64), device=dev)
        if bos is not None:
            chunk = torch.cat([bos.expand(chunk.shape[0], 1), chunk], dim=1)
        with torch.no_grad():
            model.run_with_hooks(chunk, return_type=None, fwd_hooks=fwd_hooks, **stop_kw)
        h = grab["h"]
        acts[s:s + len(h)] = h.cpu().numpy().astype(np.float16)
        grab.clear()
        if s and (s // args.batch_size) % 200 == 0:
            el = time.time() - t0
            print(f"[capture] {s}/{N}  {el:.0f}s  eta {el / s * (N - s):.0f}s", flush=True)

    out = args.out or os.path.join(args.inp, f"raw_L{args.layer}_n{args.n}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # Uncompressed: fp16 activations are incompressible noise to DEFLATE (measured
    # ~2% on a sample) and savez_compressed would spend minutes to save nothing.
    np.savez(out, acts=acts, cls=cls, day=day, doc=doc, is_fit=is_fit,
             row=sel, layer=args.layer, n=args.n, site="final_token",
             model=args.model, dtype=args.dtype, prepend_bos=cfg.prepend_bos,
             d_model=d_model, source=os.path.abspath(proj_path))
    print(f"\n[saved] {out}  acts={acts.shape} {acts.dtype} "
          f"({os.path.getsize(out) / 1e6:.0f} MB) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
