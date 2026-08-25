#!/usr/bin/env python
"""Build the weekday-swap control set: seven one-token variants of every weekday window.

Writes swap_set_n16.npz. See repro_fig3_arc_occupancy.sh.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

# Large intermediates default under the user cache rather than a fixed pod path, so this
# runs without root and off the working tree. repro_fig3_arc_occupancy.sh passes these
# explicitly (CORPUS= / RAWDIR=), so the published figure does not depend on the default.
_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
                      "weekday-manifold")
CORPUS_DIR = os.path.join(_CACHE, "corpus_v2")
RAW_DIR = os.path.join(_CACHE, "corpus_v2_raw")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_CLASSES = ("positive", "proper_noun_weekday", "dirty_positive")
# Carried per family so the analysis can slice on contamination without reloading
# the capture list (and without any risk of re-deriving a different row order).
META = ("pos", "url", "pn_side", "title_case", "left_month", "left_coref",
        "left_year", "right_month", "right_year", "would_demote_on_year")


def weekday_token_map(model_name):
    """{token_id: (day, form)} for the 14 single-token weekday spellings."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    fwd, back = {}, {}
    for d in DAYS:
        for form, s in (("spaced", " " + d), ("bare", d)):
            t = tok.encode(s, add_special_tokens=False)
            if len(t) != 1:
                raise SystemExit(f"{s!r} is not a single token under {model_name}: {t}")
            fwd[t[0]] = (d, form)
            back[(d, form)] = t[0]
    return fwd, back


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=CORPUS_DIR)
    ap.add_argument("--n-max", type=int, default=256)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--classes", default=",".join(WEEKDAY_CLASSES))
    ap.add_argument("--out", default=os.path.join(RAW_DIR, "swap_set_n16.npz"))
    args = ap.parse_args()

    blob = np.load(os.path.join(args.inp, f"capture_list_n{args.n_max}.npz"),
                   allow_pickle=True)
    ids_all = blob["ids"][:, -args.n:]
    cls = np.asarray(blob["cls"]).astype(str)
    day = np.asarray(blob["day"]).astype(str)
    doc = np.asarray(blob["doc"]).astype(str)

    # The fit/score split is READ from the cached projections, never recomputed --
    # same discipline as capture_corpus_raw_l28.py, and for the same reason: a
    # reseed there must not silently desynchronise the plane from the activations.
    proj_path = os.path.join(args.inp, f"projections_n{args.n}.npz")
    pb = np.load(proj_path, allow_pickle=True)
    is_fit = np.asarray(pb["is_fit"]).astype(bool)
    if len(is_fit) != len(ids_all):
        raise SystemExit(f"{proj_path} has {len(is_fit)} rows vs {len(ids_all)}")
    for k, mine in (("cls", cls), ("day", day), ("doc", doc)):
        if not np.array_equal(mine, np.asarray(pb[k]).astype(str)):
            raise SystemExit(f"{k} differs from {proj_path} -- lists out of sync")

    want = [c.strip() for c in args.classes.split(",")]
    rows = np.where(np.isin(cls, want))[0]
    print(f"[list] {len(rows)} weekday-class windows: "
          f"{dict(collections.Counter(cls[rows].tolist()))}", flush=True)

    fwd, back = weekday_token_map(args.model)
    last = ids_all[rows, -1]
    unknown = sorted({int(t) for t in last if int(t) not in fwd})
    if unknown:
        # A multi-token or non-weekday final token would mean the premise of the
        # whole experiment (mention == capture site == one token) is false for some
        # rows, and a silent skip would quietly change what "family" means.
        raise SystemExit(f"{len(unknown)} final tokens are not weekday forms: {unknown[:8]}")
    fam_day = np.array([fwd[int(t)][0] for t in last])
    fam_form = np.array([fwd[int(t)][1] for t in last])
    mism = int((fam_day != day[rows]).sum())
    if mism:
        raise SystemExit(f"{mism} windows' final token disagrees with their day label")
    print(f"[forms] {dict(collections.Counter(fam_form.tolist()))}", flush=True)

    F, V = len(rows), 7
    ids = np.repeat(ids_all[rows], V, axis=0)              # [7F, n], family-major
    swap_ix = np.tile(np.arange(V, dtype=np.int8), F)
    # The ONLY edit: the final token, kept in the family's own spelling.
    ids[:, -1] = [back[(DAYS[v], f)] for f, v in
                  zip(np.repeat(fam_form, V), swap_ix)]
    # Cheap invariant: every variant differs from its family's original in at most
    # the final column, and in exactly zero columns when the swap is a no-op.
    orig = np.repeat(ids_all[rows], V, axis=0)
    if (ids[:, :-1] != orig[:, :-1]).any():
        raise SystemExit("context columns were modified -- swap is not a last-token edit")
    same = np.repeat(fam_day, V) == np.array(DAYS)[swap_ix]
    if not (ids[same, -1] == orig[same, -1]).all():
        raise SystemExit("self-swap did not reproduce the original token")
    print(f"[swap] {F} families x {V} variants = {len(ids)} prompts; "
          f"{int(same.sum())} self-swaps (one per family)", flush=True)

    out = {"ids": ids.astype(np.int32),
           "fam": np.repeat(np.arange(F, dtype=np.int32), V),
           "swap": swap_ix,                                  # 0..6 index into DAYS
           "is_self": same,
           "days": np.array(DAYS),
           # ---- per-family (length F), NOT per-row -------------------------
           "fam_row": rows.astype(np.int32),                 # row in capture_list / raw_L28
           "fam_cls": cls[rows], "fam_day": fam_day, "fam_form": fam_form,
           "fam_doc": doc[rows], "fam_is_fit": is_fit[rows],
           "n": args.n, "n_max": args.n_max, "model": args.model,
           "source_list": os.path.abspath(
               os.path.join(args.inp, f"capture_list_n{args.n_max}.npz")),
           "source_proj": os.path.abspath(proj_path)}
    for k in META:
        out["fam_" + k] = np.asarray(blob[k])[rows]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(f"[saved] {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
