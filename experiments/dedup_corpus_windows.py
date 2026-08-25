#!/usr/bin/env python
"""MinHash/LSH near-duplicate removal over the selected windows, per dump. NumPy only."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
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

PRIORITY = {"positive": 0, "dirty_positive": 1, "proper_noun_weekday": 2,
            "near_miss": 3, "matched_negative": 4, "floor": 5}
MERSENNE = (1 << 61) - 1


def shingles(text, k=5):
    """Word k-gram shingle hashes as uint64. Falls back to char 12-grams if short."""
    w = text.split()
    if len(w) >= k:
        grams = (" ".join(w[i:i + k]) for i in range(len(w) - k + 1))
    else:
        grams = (text[i:i + 12] for i in range(max(1, len(text) - 11)))
    out = {np.uint64(int.from_bytes(hashlib.blake2b(g.encode(), digest_size=7).digest(),
                                    "little"))
           for g in grams}
    return np.fromiter(out, dtype=np.uint64, count=len(out))


def signatures(texts, n_perm=64, seed=0):
    """[n_texts, n_perm] MinHash signatures."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE, size=n_perm, dtype=np.uint64)
    b = rng.integers(0, MERSENNE, size=n_perm, dtype=np.uint64)
    sig = np.empty((len(texts), n_perm), dtype=np.uint64)
    for i, t in enumerate(texts):
        h = shingles(t)
        if h.size == 0:
            sig[i] = 0
            continue
        # (a*h + b) mod (2^61 - 1), vectorised over permutations x shingles
        v = (np.multiply.outer(a, h) + b[:, None]) % np.uint64(MERSENNE)
        sig[i] = v.min(axis=1)
        if i and i % 25000 == 0:
            print(f"[minhash] {i}/{len(texts)}", flush=True)
    return sig


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[ry] = rx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=CORPUS_DIR)
    ap.add_argument("--n-max", type=int, default=256)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--min-gap", type=int, default=None,
                    help="min capture-position gap within a document; "
                         "default = n_max, i.e. windows share no text at all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = os.path.join(args.inp, f"windows_n{args.n_max}.jsonl")
    raw = [json.loads(l) for l in open(path)]
    print(f"[load] {len(raw)} windows from {path}")

    # Token ids travel WITH the rows through every filter. They are the authoritative
    # window definition: the class and every flag were computed on them, and the text
    # is a display intermediate cut by character offsets. Dropping them here and
    # re-deriving later would undo the reason they are persisted at all.
    idpath = os.path.join(args.inp, f"windows_n{args.n_max}_ids.npy")
    raw_ids = None
    if os.path.exists(idpath):
        raw_ids = np.load(idpath)
        if len(raw_ids) != len(raw):
            raise SystemExit(f"id/row mismatch: {len(raw_ids)} ids vs {len(raw)} rows")
        print(f"[load] token ids {raw_ids.shape} from {idpath}")
    else:
        print(f"[warn] no {idpath}; capture would have to re-tokenise (see docstring)")

    # STAGE 1 -- exact, position-based non-overlap within a document.
    # The pilot showed 100% of MinHash near-duplicates were same-document window
    # overlap rather than web boilerplate: with per_doc_cap=2 at N=256, two capture
    # positions 5 tokens apart share 251 of 256 tokens. That is a deterministic
    # artifact of the selection cap, so it deserves a deterministic fix rather than
    # a probabilistic one -- keeping windows whose capture positions differ by at
    # least N means surviving windows share NO text at all.
    #
    # Same-document windows that do not overlap are kept: they are genuinely
    # different text. Their residual topical correlation is handled where it
    # belongs, in the document-clustered bootstrap, not by discarding data.
    gap = args.min_gap if args.min_gap is not None else args.n_max
    by_doc = collections.defaultdict(list)
    for i, r in enumerate(raw):
        by_doc[r["doc"]].append(i)
    keep_stage1 = []
    for doc, idxs in by_doc.items():
        idxs.sort(key=lambda i: (PRIORITY.get(raw[i]["cls"], 9), raw[i]["pos"]))
        taken = []
        for i in idxs:
            if all(abs(raw[i]["pos"] - raw[j]["pos"]) >= gap for j in taken):
                taken.append(i)
        keep_stage1.extend(taken)
    keep_stage1.sort()
    s1_before = collections.Counter(r["cls"] for r in raw)
    rows = [raw[i] for i in keep_stage1]
    rows_ids = raw_ids[keep_stage1] if raw_ids is not None else None
    s1_after = collections.Counter(r["cls"] for r in rows)
    print(f"[non-overlap] min capture-position gap {gap}: "
          f"{len(raw)} -> {len(rows)} windows")
    for c in sorted(s1_before):
        print(f"    {c:<22}{s1_before[c]:>8} -> {s1_after[c]:<8}"
              f"(-{s1_before[c]-s1_after[c]})")

    sig = signatures([r["text"] for r in rows], n_perm=args.bands * args.rows,
                     seed=args.seed)
    print(f"[minhash] signatures {sig.shape}")

    dsu = DSU(len(rows))
    pairs = 0
    for band in range(args.bands):
        cols = sig[:, band * args.rows:(band + 1) * args.rows]
        buckets = collections.defaultdict(list)
        for i, key in enumerate(map(bytes, cols.astype("<u8").view(np.uint8)
                                    .reshape(len(rows), -1))):
            buckets[key].append(i)
        for members in buckets.values():
            if len(members) > 1:
                first = members[0]
                for m in members[1:]:
                    dsu.union(first, m)
                    pairs += 1
    print(f"[lsh] {pairs} candidate merges across {args.bands} bands")

    # one survivor per cluster, chosen by class priority then by original order
    best = {}
    for i, r in enumerate(rows):
        root = dsu.find(i)
        key = (PRIORITY.get(r["cls"], 9), i)
        if root not in best or key < best[root][0]:
            best[root] = (key, i)
    keep = sorted(v[1] for v in best.values())
    keepset = set(keep)

    before = collections.Counter(r["cls"] for r in rows)   # post non-overlap
    after = collections.Counter(rows[i]["cls"] for i in keep)
    same_doc = sum(1 for i, r in enumerate(rows)
                   if i not in keepset and
                   rows[dsu.find(i)]["doc"] == r["doc"])

    print("\n" + "=" * 72)
    print(f"NEAR-DUPLICATE REMOVAL  (Jaccard ~>= {(1/args.bands)**(1/args.rows):.2f})")
    print("=" * 72)
    print(f"  {'class':<22}{'before':>9}{'after':>9}{'dropped':>9}{'%':>7}")
    for c in ["positive", "dirty_positive", "proper_noun_weekday", "near_miss",
              "matched_negative", "floor"]:
        d = before[c] - after[c]
        print(f"  {c:<22}{before[c]:>9}{after[c]:>9}{d:>9}"
              f"{100*d/before[c] if before[c] else 0:>6.1f}%")
    tot_d = len(rows) - len(keep)
    print(f"  {'TOTAL':<22}{len(rows):>9}{len(keep):>9}{tot_d:>9}"
          f"{100*tot_d/len(rows):>6.1f}%")
    print(f"\n  of the dropped, {same_doc} ({100*same_doc/max(tot_d,1):.1f}%) were "
          f"overlapping windows from the SAME document")

    ndocs = len({rows[i]["doc"] for i in keep if rows[i]["cls"] == "matched_negative"})
    n_neg = after["matched_negative"]
    n_eff = n_neg / max(1.0, n_neg / max(ndocs, 1))
    print(f"\n  matched negatives {n_neg} from {ndocs} docs "
          f"({n_neg/max(ndocs,1):.2f}/doc)")
    print(f"  n_eff >= {n_eff:.0f}  ->  exceedance bound 3/n_eff = "
          f"{3/max(n_eff,1):.2e}")

    out = os.path.join(args.inp, f"windows_n{args.n_max}_dedup.jsonl")
    with open(out, "w") as f:
        for i in keep:
            f.write(json.dumps(rows[i]) + "\n")
    if rows_ids is not None:
        oid = os.path.join(args.inp, f"windows_n{args.n_max}_dedup_ids.npy")
        np.save(oid, rows_ids[keep])
        print(f"[saved] {oid}  {rows_ids[keep].shape}")
    meta = {"before_all": dict(s1_before), "after_nonoverlap": dict(s1_after),
            "min_gap": gap, "before": dict(before), "after": dict(after),
            "bands": args.bands, "rows": args.rows,
            "jaccard_threshold": (1 / args.bands) ** (1 / args.rows),
            "dropped_total": tot_d, "dropped_same_doc": same_doc,
            "matched_negative_docs": ndocs, "n_eff_matched_negative": n_eff,
            "exceedance_bound": 3 / max(n_eff, 1)}
    with open(os.path.join(args.inp, f"dedup_meta_n{args.n_max}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
