"""What the model is about to say at each readout site, unpatched.

Only the token list and the two positions reach a figure, and those are tokeniser
properties. Writes figures/readout_context.json, which is tracked.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from weekday_manifold.manifold.config import ManifoldConfig
from weekday_manifold.manifold.days import DAYS as DAY_WORDS
from weekday_manifold.manifold.days import build_prompts, days_token_ids
from weekday_manifold.manifold.probes import build_mention_early
from weekday_manifold.plateau.model import load_plateau_model

PATCH_LAYER, READ_LAYER = 2, 28

# name, builder, steer pos, read pos, how many backgrounds the polar field ran on.
# That last number is the point of this file agreeing with the field: every quantity
# here is measured on exactly the prompts the field averaged over, so the figure
# carries one n rather than two that a reader has to reconcile.
SETTINGS = [
    ("interrogative", lambda: build_prompts("interrogative"), None, -1, 49),
    ("mention_early_stop", lambda: build_mention_early("thisday"), 2, 8, 70),
    ("mention_early_on", lambda: build_mention_early("thisday"), 2, 9, 70),
    ("mention_early_day", lambda: build_mention_early("thisday"), 2, 11, 70),
]


def balanced(day, n):
    """The polar field's background selection, verbatim: round-robin over the days."""
    by_day = {k: [i for i in range(len(day)) if day[i] == k] for k in range(7)}
    sel, rank = [], 0
    while len(sel) < n:
        grew = False
        for k in range(7):
            if rank < len(by_day[k]) and len(sel) < n:
                sel.append(by_day[k][rank]); grew = True
        if not grew:
            break
        rank += 1
    return sorted(sel)


def day_variance(A, day, grp):
    """How much of what varies at this token is the day."""
    A = np.asarray(A, np.float64)
    C = np.stack([A[day == k].mean(0) for k in range(7)])
    gm = A.mean(0)
    ssb = sum((day == k).sum() * np.sum((C[k] - gm) ** 2) for k in range(7))
    sst = np.sum((A - gm) ** 2)
    hit = tot = 0
    for g in np.unique(grp):
        tr, te = grp != g, grp == g
        if not tr.any() or not te.any():
            continue
        Ct = np.stack([A[tr & (day == k)].mean(0) for k in range(7)])
        d2 = ((A[te][:, None, :] - Ct[None]) ** 2).sum(-1)
        hit += int((d2.argmin(1) == day[te]).sum()); tot += int(te.sum())
    return (float(ssb / sst) if sst > 0 else 0.0), (hit / tot if tot else float("nan"))


def main():
    # THE CONFIG THIS USED TO NAME DOES NOT EXIST ON THIS BRANCH. It was pinned to
    # a config that was pruned when the branch was cut down to
    # the six figure closures -- so the script raised FileNotFoundError on its first line
    # and figures/readout_context.json, which figure 5 reads on every render, was tracked
    # but unrebuildable. Defaulting to the Llama config that IS here restores that.
    #
    # Only `tokens`, `patch_pos` and `read_pos` reach the figure, and those are tokeniser
    # properties: they do not depend on which Llama config wrote them. The statistics in
    # the rest of the file (eta2, decodable, p_weekday) are float16 forward passes and
    # will differ in the last places from the committed copy. That is the fp16 drift
    # described in the README, not a disagreement about the site.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/llama31_8b_fp16.json")
    ap.add_argument("--out", default="figures/readout_context",
                    help="path stem; .json and .npz are written")
    args = ap.parse_args()

    cfg = ManifoldConfig.from_json(args.config)
    model = load_plateau_model(cfg)
    day_tok = np.array([days_token_ids(model)[k][0] for k in range(7)])
    out, cents = {}, {}
    for name, build, ppos, rpos, n_bg in SETTINGS:
        full = build()
        dz_full = np.array([int(sp.meta["z"]) for sp in full])
        sel = balanced(dz_full, n_bg)
        specs = [full[i] for i in sel]
        toks = torch.stack([model.to_tokens(sp.text, prepend_bos=cfg.prepend_bos)[0]
                            for sp in specs], 0)
        with torch.no_grad():
            pr = torch.softmax(model(toks.to(model.cfg.device),
                                     return_type="logits").float(), -1).cpu().numpy()
        lab = model.to_str_tokens(specs[0].text, prepend_bos=cfg.prepend_bos)
        r = rpos % len(lab)
        p = (ppos if ppos is not None
             else int(np.argmax([1 if t.strip() in
                                 ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                                  "Saturday", "Sunday") else 0 for t in lab])))
        mean = pr[:, r, :].mean(0)
        top = np.argsort(mean)[::-1][:6]
        store = {}
        with torch.no_grad():
            model.run_with_hooks(
                toks.to(model.cfg.device), return_type=None,
                fwd_hooks=[
                    (f"blocks.{PATCH_LAYER}.hook_resid_post",
                     lambda a, hook: store.__setitem__("x", a[:, p, :].float().cpu().numpy())
                     or a),
                    # the same question at the token the response is READ from: the steer
                    # site is the weekday token itself, where the day is ~100% of the
                    # variation by construction and cannot separate one row from another
                    (f"blocks.{READ_LAYER}.hook_resid_post",
                     lambda a, hook: store.__setitem__("y", a[:, r, :].float().cpu().numpy())
                     or a)])
        dz = np.array([int(sp.meta["z"]) for sp in specs])
        # the wording with the day removed: the offset for a question, the template for a
        # sentence. Groups the folds so decodability is never scored on a seen phrasing.
        keys = [sp.text.replace(DAY_WORDS[int(sp.meta["z"])], "{}") for sp in specs]
        uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
        grp = np.array([uniq[k] for k in keys])
        eta2, dec = day_variance(store["x"], dz, grp)
        eta2_r, dec_r = day_variance(store["y"], dz, grp)
        C7 = np.stack([store["x"][dz == k].mean(0) for k in range(7)])
        cents[name] = C7
        # the spline the field swept was fitted on the whole family; these centroids come
        # from the backgrounds only. Worth knowing they are the same ring, not assuming it.
        drift = float("nan")
        if len(specs) < len(full):
            with torch.no_grad():
                allt = torch.stack([model.to_tokens(sp.text, prepend_bos=cfg.prepend_bos)[0]
                                    for sp in full], 0)
                fs = {}
                model.run_with_hooks(
                    allt.to(model.cfg.device), return_type=None,
                    fwd_hooks=[(f"blocks.{PATCH_LAYER}.hook_resid_post",
                                lambda a, hook: fs.__setitem__(
                                    "x", a[:, p, :].float().cpu().numpy()) or a)])
            Cf = np.stack([fs["x"][dz_full == k].mean(0) for k in range(7)])
            drift = float(np.max(np.linalg.norm(C7 - Cf, axis=1)
                                 / np.linalg.norm(Cf - Cf.mean(0), axis=1)))
        # how faithful the 3-D panel is: the plotted basis is the top 3 PCs of the seven
        # centroids, so ask what share of the FULL activation variance at this token --
        # every prompt, all 4096 dimensions -- survives projection onto it
        V3 = np.linalg.svd(C7 - C7.mean(0), full_matrices=False)[2][:3]
        Ac = store["x"] - store["x"].mean(0)
        frac3 = float((Ac @ V3.T).var(0).sum() / Ac.var(0).sum())
        out_extra = dict(var3_patch=frac3)
        out[name] = dict(
            # STAMP THE MODEL. This token strip is one tokeniser's output, and
            # figure_jac_discs.py draws it under whichever model's measurements it is
            # plotting. It used to match an entry by (patch_pos, read_pos) alone, so a
            # second model that happened to land on the same two positions would silently
            # borrow Llama's token strings. Recording the model here lets the reader
            # refuse that instead of mislabelling the axis.
            model=cfg.model_name,
            prompt=specs[sel[0]].text, tokens=lab, patch_pos=p, read_pos=r,
            eta2_patch=eta2, decodable_patch=dec, n_groups=len(uniq),
            eta2_read=eta2_r, decodable_read=dec_r, **out_extra,
            p_weekday=float(pr[:, r, day_tok].sum(-1).mean()),
            top_tokens=[[model.to_string([int(t)]), float(mean[t])] for t in top],
            n_prompts=len(specs), n_family=len(full), centroid_drift=drift)
        print(f"[{name}] steer {lab[p]!r} eta2 {eta2:.3f} 3-plane {frac3:.3f} | "
              f"read {lab[r]!r} "
              f"L{READ_LAYER} eta2 {eta2_r:.3f} decodable {dec_r:.3f} | "
              f"n {len(specs)}/{len(full)} drift {drift:.3f} | "
              f"P(weekday) {out[name]['p_weekday']:.4f} "
              f"top {[t[0] for t in out[name]['top_tokens'][:3]]}")
    with open(f"{args.out}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez(f"{args.out}.npz", **{f"C_{k}": v for k, v in cents.items()})
    print(f"[out] wrote {args.out}.json and .npz")


if __name__ == "__main__":
    main()
