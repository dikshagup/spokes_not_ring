#!/usr/bin/env python
"""Capture the weekday-swap set at the same layer and settings as the raw pass.

Writes swap_L28_n16.npz. See repro_fig3_arc_occupancy.sh.
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
    ap.add_argument("--set", dest="sset", default=os.path.join(RAW_DIR, "swap_set_n16.npz"))
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit-families", type=int, default=None,
                    help="smoke test on the first N families (whole families only, "
                         "so the de-meaning downstream still sees complete rings)")
    ap.add_argument("--out", default=os.path.join(RAW_DIR, "swap_L28_n16.npz"))
    args = ap.parse_args()

    print(f"[env] weekday_manifold -> {assert_local_package()}", flush=True)
    from weekday_manifold.manifold.config import ManifoldConfig
    from weekday_manifold.plateau.model import load_plateau_model

    b = np.load(args.sset, allow_pickle=True)
    ids_all = b["ids"]
    fam = b["fam"]
    V = 7
    if args.limit_families:
        keep = fam < args.limit_families
        ids_all, fam = ids_all[keep], fam[keep]
    N = len(ids_all)
    if N % V:
        raise SystemExit(f"{N} rows is not a whole number of {V}-variant families")
    print(f"[set] {N} prompts = {N // V} families x {V}, n={int(b['n'])} tokens; "
          f"classes {dict(collections.Counter(np.asarray(b['fam_cls']).astype(str).tolist()))}",
          flush=True)

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

    stop_kw = {"stop_at_layer": args.layer + 1}
    try:
        with torch.no_grad():
            chunk = torch.tensor(ids_all[:1].astype(np.int64), device=dev)
            if bos is not None:
                chunk = torch.cat([bos, chunk], dim=1)
            model.run_with_hooks(chunk, return_type=None, fwd_hooks=fwd_hooks, **stop_kw)
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

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Uncompressed for the same reason as the original raw pass: fp16 activations
    # are incompressible to DEFLATE and savez_compressed would spend minutes on ~2%.
    # Everything from the set file except `ids` is carried through, so the analysis
    # never has to re-open two files and hope their row orders still agree.
    carry = {k: b[k] for k in b.files if k not in ("ids",)}
    if args.limit_families:
        for k in list(carry):
            if k.startswith("fam_") and len(np.shape(carry[k])) and \
                    np.shape(carry[k])[0] == int(b["fam"].max()) + 1:
                carry[k] = carry[k][:args.limit_families]
        carry["fam"], carry["swap"] = fam, b["swap"][:N]
        carry["is_self"] = b["is_self"][:N]
    np.savez(args.out, acts=acts, layer=args.layer, site="final_token",
             dtype=args.dtype, prepend_bos=cfg.prepend_bos, d_model=d_model,
             source_set=os.path.abspath(args.sset), **carry)
    print(f"\n[saved] {args.out}  acts={acts.shape} {acts.dtype} "
          f"({os.path.getsize(args.out) / 1e6:.0f} MB) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
