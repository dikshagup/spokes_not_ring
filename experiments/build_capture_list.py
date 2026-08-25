#!/usr/bin/env python
"""Turn the deduplicated windows into the exact capture list. Tokeniser only.

Writes capture_list_n256.npz. See repro_fig3_arc_occupancy.sh.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
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

# Prereg 4b targets. None = take everything (the scarce classes).
CAPS = {"positive": None, "dirty_positive": None, "proper_noun_weekday": None,
        "near_miss": 18000, "matched_negative": 68000, "floor": 20000}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=CORPUS_DIR)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--n-max", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    day_ids = {}
    for d in DAYS:
        for f in (d, " " + d):
            e = tok.encode(f, add_special_tokens=False)
            if len(e) == 1:
                day_ids[e[0]] = d

    path = os.path.join(args.inp, f"windows_n{args.n_max}_dedup.jsonl")
    rows = [json.loads(l) for l in open(path)]
    idpath = os.path.join(args.inp, f"windows_n{args.n_max}_dedup_ids.npy")
    if not os.path.exists(idpath):
        raise SystemExit(f"missing {idpath}. Token ids are the authoritative window "
                         f"definition and must come from selection, not from "
                         f"re-tokenising text. Re-run select_corpus_windows.py.")
    all_ids = np.load(idpath)
    if len(all_ids) != len(rows):
        raise SystemExit(f"id/row mismatch: {len(all_ids)} vs {len(rows)}")
    for r, i in zip(rows, all_ids):
        r["_ids"] = i.tolist()
    print(f"[load] {len(rows)} deduplicated windows + ids {all_ids.shape}")

    rng = random.Random(args.seed)
    by_cls = collections.defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)
    chosen = []
    for cls, rs in sorted(by_cls.items()):
        cap = CAPS.get(cls)
        if cap is not None and len(rs) > cap:
            rs = rng.sample(rs, cap)
        chosen.extend(rs)
        print(f"  {cls:<22}{len(by_cls[cls]):>8} -> {len(rs):<8}"
              f"{'(capped)' if cap and len(by_cls[cls]) > cap else ''}")

    ids = np.zeros((len(chosen), args.n_max), dtype=np.int32)
    bad = collections.Counter()
    keep = []
    for k, r in enumerate(chosen):
        enc = r["_ids"]
        if len(enc) != args.n_max:
            bad[f"length {len(enc)} != {args.n_max}"] += 1
            continue
        if r["cls"] in ("positive", "dirty_positive", "proper_noun_weekday"):
            if enc[-1] not in day_ids:
                bad["final token is not a weekday"] += 1
                continue
            if day_ids[enc[-1]] != r["day"]:
                bad["final weekday != stored day label"] += 1
                continue
        ids[len(keep)] = enc
        keep.append(r)
    ids = ids[:len(keep)]

    # NOT checked: decode(ids) == stored text. tok.decode is lossy -- it drops
    # spaces before some punctuation ("batted .212" -> "batted.212", "a ?best" ->
    # "a?best") -- so that comparison fails for ~7% of windows whose encoding is
    # perfectly correct. It measures decoder behaviour, not window integrity, and
    # using it as an invariant would silently discard good data.
    print(f"\n[verify] {len(keep)}/{len(chosen)} windows passed the invariants")
    for k, v in bad.items():
        print(f"    dropped {v:>6}  {k}")

    out = args.out or os.path.join(args.inp, f"capture_list_n{args.n_max}.npz")
    meta_keys = ["cls", "day", "doc", "pos", "url", "dump", "hash", "pn_side",
                 "title_case", "left_month", "left_coref", "left_year",
                 "right_month", "right_year", "would_demote_on_year"]
    cols = {k: np.array([r.get(k) if r.get(k) is not None else "" for r in keep],
                        dtype=object) for k in meta_keys}
    np.savez_compressed(out, ids=ids, n_max=args.n_max,
                        model=args.model, seed=args.seed, **cols)
    counts = collections.Counter(r["cls"] for r in keep)
    print(f"\n[saved] {out}  ids={ids.shape} "
          f"({os.path.getsize(out)/1e6:.0f} MB)")
    for c, n in sorted(counts.items()):
        print(f"    {c:<22}{n:>8}")
    with open(os.path.join(args.inp, f"capture_meta_n{args.n_max}.json"), "w") as f:
        json.dump({"counts": dict(counts), "dropped": dict(bad),
                   "n_max": args.n_max, "model": args.model,
                   "total": len(keep)}, f, indent=2)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
