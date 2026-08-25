#!/usr/bin/env python
"""Stream FineWeb and select the 16-token windows. CPU only -- tokeniser, no activations.

Writes the window shards. See repro_fig3_arc_occupancy.sh.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
# CAPITALISED FULL NAMES ONLY, everywhere -- capture set and exclusion set alike.
# Lowercase coverage is inconsistent across both days and months (" monday" and
# " december" are single Llama tokens, " tuesday" and " january" are not), and
# abbreviations collide with word-internal pieces ("Monaco" -> ['Mon','aco']).
# Matching them would make coverage uneven across the seven days, which would
# skew the day-decode test -- the experiment's specificity test -- for a small
# gain in recall. The cost is accepted UNDER-exclusion: a lowercase "december" in
# left context is not caught. Flags make that revisitable without re-capturing.
#
# SHIFTING vs CO-REFERENTIAL deixis. Only "yesterday"/"tomorrow" demote a window
# to dirty, because they move the deictic anchor off the named day -- the exact
# mechanism the C1 compute templates exploit ("Today's Monday, so tomorrow's" ->
# Tuesday), so the weekday token may be representing an offset day and the
# day-decode label would be wrong. "today"/"tonight" are co-referential with the
# named day and induce no such shift, so they are FLAGGED but not excluded.
SHIFTING_DEIXIS = ["yesterday", "tomorrow"]
COREF_DEIXIS = ["today", "tonight"]

# Regex temporal forms that do not tokenise predictably (digits split many ways),
# so they are found in the RAW TEXT and mapped back through the offset mapping.
RE_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm|AM|PM|a\.m\.|p\.m\.)\b")
RE_ORDINAL_DATE = re.compile(r"\b(?:[12]?\d|3[01])(?:st|nd|rd|th)\b")
RE_YEAR = re.compile(r"\b(?:1[0-9]{3}|2[0-9]{3})\b")
RE_SENT_END = re.compile(r"[.!?]")


def build_token_sets(tok):
    """Token-id sets, capitalised full names only. Returns a dict of id sets."""
    def ids_for(forms):
        out = set()
        for f in forms:
            enc = tok.encode(f, add_special_tokens=False)
            if len(enc) == 1:
                out.add(enc[0])
        return out

    def both(words):                       # bare and space-prefixed
        return ids_for(list(words) + [" " + w for w in words])

    day_capture = {d: both([d]) for d in DAYS}
    day_all = set().union(*day_capture.values())
    return {
        "day_capture": day_capture,        # capture set, uniform across 7 days
        "day_exclude": day_all,            # same set: no lowercase, no abbrevs
        "month": both(MONTHS),
        "shifting": both(SHIFTING_DEIXIS) | both([w.capitalize()
                                                  for w in SHIFTING_DEIXIS]),
        "coref": both(COREF_DEIXIS) | both([w.capitalize() for w in COREF_DEIXIS]),
    }


def regex_temporal_token_mask(text, offsets):
    """Token indices overlapping each regex temporal form, kept SEPARATE."""
    out = {}
    for name, rx in (("clock", RE_CLOCK), ("ordinal", RE_ORDINAL_DATE),
                     ("year", RE_YEAR)):
        spans = [(m.start(), m.end()) for m in rx.finditer(text)]
        marks = set()
        if spans:
            for i, (a, b) in enumerate(offsets):
                if a == b:
                    continue
                for (s, e) in spans:
                    if a < e and s < b:
                        marks.add(i)
                        break
        out[name] = marks
    return out


def title_case_run(pieces, i, look=5, need=3):
    """True if the capture token sits in a run of capitalised words (a headline)."""
    lo = max(0, i - look)
    caps = sum(1 for p in pieces[lo:i] if is_capitalised_word(p))
    return caps >= need


def is_capitalised_word(piece):
    p = piece.lstrip()
    return bool(p) and p[0].isupper() and p.isalpha() and len(p) >= 3


def proper_noun_trigger(pieces, i):
    """(side, trigger) if the weekday at i looks like part of a proper name."""
    if i + 1 < len(pieces) and is_capitalised_word(pieces[i + 1]):
        return "right", pieces[i + 1].strip()
    if i - 1 >= 0 and is_capitalised_word(pieces[i - 1]):
        prev2 = pieces[i - 2].strip() if i - 2 >= 0 else ""
        if not RE_SENT_END.search(prev2):
            return "left", pieces[i - 1].strip()
    return None, None


def scan_document(doc_idx, rec, tok, sets, n_max, rng, per_doc_cap):
    """Yield window dicts for one document. All classes, all flags."""
    day_capture = sets["day_capture"]
    day_exclude, month_ids = sets["day_exclude"], sets["month"]
    shifting_ids, coref_ids = sets["shifting"], sets["coref"]
    text = rec["text"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) < n_max:
        return
    pieces = tok.convert_ids_to_tokens(ids)
    pieces = [p.replace("Ġ", " ") for p in pieces]
    rx_marks = regex_temporal_token_mask(text, offsets)

    day_of = {}
    for d, idset in day_capture.items():
        for t in idset:
            day_of[t] = d

    def left_flags(i):
        """Flags over the causal context. ``left_coref`` is recorded but does NOT demote a
        window -- see the SHIFTING vs CO-REFERENTIAL note at module top."""
        f = {"left_weekday": False, "left_month": False, "left_shifting": False,
             "left_coref": False, "left_clock": False, "left_ordinal": False,
             "left_year": False, "left_month_dist": None,
             "left_temporal_dist": None}
        for j in range(i - n_max + 1, i):
            t = ids[j]
            if t in day_exclude:
                f["left_weekday"] = True
            if t in month_ids:
                f["left_month"] = True
                f["left_month_dist"] = min(f["left_month_dist"] or 10**9, i - j)
            if t in shifting_ids:
                f["left_shifting"] = True
            if t in coref_ids:
                f["left_coref"] = True
            for kind in ("clock", "ordinal", "year"):
                if j in rx_marks[kind]:
                    f[f"left_{kind}"] = True
            if (t in day_exclude or t in month_ids or t in shifting_ids
                    or j in rx_marks["clock"] or j in rx_marks["ordinal"]):
                f["left_temporal_dist"] = min(f["left_temporal_dist"] or 10**9, i - j)
        return f

    def right_flags(i):
        hi = min(len(ids), i + 9)
        f = {"right_month": False, "right_weekday": False, "right_shifting": False,
             "right_coref": False, "right_clock": False, "right_ordinal": False,
             "right_year": False}
        for j in range(i + 1, hi):
            if ids[j] in month_ids:
                f["right_month"] = True
            if ids[j] in day_exclude:
                f["right_weekday"] = True
            if ids[j] in shifting_ids:
                f["right_shifting"] = True
            if ids[j] in coref_ids:
                f["right_coref"] = True
            for kind in ("clock", "ordinal", "year"):
                if j in rx_marks[kind]:
                    f[f"right_{kind}"] = True
        return f

    def emit(i, cls, extra=None):
        lo = i - n_max + 1
        w = {"doc": doc_idx, "id": rec.get("id"), "url": rec.get("url"),
             "dump": rec.get("dump"), "pos": i, "cls": cls,
             "capture_piece": pieces[i],
             "window_ids": ids[lo:i + 1],
             "text": text[offsets[lo][0]:offsets[i][1]]}
        w.update(left_flags(i))
        w.update(right_flags(i))
        if extra:
            w.update(extra)
        w["hash"] = hashlib.blake2b(w["text"].encode(), digest_size=12).hexdigest()
        return w

    out = []
    # --- weekday-final windows: positive / dirty positive / proper-noun ------
    for i in range(n_max - 1, len(ids)):
        if ids[i] not in day_of:
            continue
        side, trig = proper_noun_trigger(pieces, i)
        lf = left_flags(i)
        tc = title_case_run(pieces, i)
        extra = {"day": day_of[ids[i]], "pn_side": side, "pn_trigger": trig,
                 "title_case": tc}
        # A bare 4-digit YEAR does not demote: it specifies no weekday and induces
        # no deictic shift, so it is "temporal but not day-referring" exactly like
        # the word "year", which the prereg already excludes. Demoting it cost 45%
        # of all windows in the pilot for no clear gain. Recorded either way, and
        # the counterfactual loss is reported so the choice stays auditable.
        demote = (lf["left_weekday"] or lf["left_month"] or lf["left_shifting"]
                  or lf["left_clock"] or lf["left_ordinal"])
        if side is not None:
            cls = "proper_noun_weekday"
        elif demote:
            cls = "dirty_positive"
        else:
            cls = "positive"                # left_coref, left_year allowed here
        extra["would_demote_on_year"] = bool(not demote and lf["left_year"])
        out.append(emit(i, cls, extra))

    # --- near-miss: month / shifting-deixis final, no weekday in window -----
    for i in range(n_max - 1, len(ids)):
        if ids[i] not in month_ids and ids[i] not in shifting_ids:
            continue
        if ids[i] in day_exclude or left_flags(i)["left_weekday"]:
            continue
        out.append(emit(i, "near_miss"))

    # --- matched negative: capitalised word final, no weekday in window -----
    cands = [i for i in range(n_max - 1, len(ids))
             if is_capitalised_word(pieces[i]) and ids[i] not in day_exclude
             and ids[i] not in month_ids and ids[i] not in shifting_ids
             and ids[i] not in coref_ids]
    rng.shuffle(cands)
    for i in cands[:per_doc_cap]:
        if left_flags(i)["left_weekday"]:
            continue
        out.append(emit(i, "matched_negative"))

    # --- floor: arbitrary final token, no weekday in window -----------------
    cands = list(range(n_max - 1, len(ids)))
    rng.shuffle(cands)
    for i in cands[:per_doc_cap]:
        if left_flags(i)["left_weekday"]:
            continue
        out.append(emit(i, "floor"))

    # Cap per document per class so one long page cannot dominate the sample.
    by_cls = collections.defaultdict(list)
    for w in out:
        by_cls[w["cls"]].append(w)
    for cls, ws in by_cls.items():
        rng.shuffle(ws)
        for w in ws[:per_doc_cap]:
            yield w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--max-docs", type=int, default=20000)
    ap.add_argument("--n-max", type=int, default=256,
                    help="class membership is decided at this length (prereg)")
    ap.add_argument("--per-doc-cap", type=int, default=2,
                    help="max windows per class per document")
    ap.add_argument("--target-positives", type=int, default=5000,
                    help="prereg 4b target; scan stops when this AND "
                         "--target-negatives are both met")
    ap.add_argument("--target-negatives", type=int, default=60000,
                    help="matched-negative target; sets the exceedance bound")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/results/corpus_windows")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    sets = build_token_sets(tok)

    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
    rng = random.Random(args.seed)

    counts = collections.Counter()
    flagc = collections.Counter()
    seen_hash = set()
    samples = collections.defaultdict(list)
    days = collections.Counter()
    dup = 0
    kept = []
    kept_ids = []

    # Prereg 4b: stop when BOTH targets are met, or at --max-docs. The rule reads
    # only window counts -- which exist before any activation does -- so it cannot
    # be tuned toward a result. Positives are the binding constraint in practice
    # (~18x rarer than matched negatives), but both are checked explicitly so the
    # code matches the registered rule rather than relying on that ratio holding.
    for doc_idx, rec in enumerate(ds):
        if doc_idx >= args.max_docs:
            break
        if (counts["positive"] >= args.target_positives
                and counts["matched_negative"] >= args.target_negatives):
            break
        for w in scan_document(doc_idx, rec, tok, sets, args.n_max, rng,
                               args.per_doc_cap):
            if w["hash"] in seen_hash:
                dup += 1
                continue
            seen_hash.add(w["hash"])
            counts[w["cls"]] += 1
            if w["cls"] in ("positive", "dirty_positive"):
                days[w["day"]] += 1
            for k in ("left_month", "left_weekday", "left_shifting", "left_coref",
                      "left_clock", "left_ordinal", "left_year", "right_month",
                      "right_weekday", "right_shifting", "right_coref",
                      "right_clock", "right_ordinal", "right_year",
                      "title_case", "would_demote_on_year"):
                if w.get(k):
                    flagc[f"{w['cls']}:{k}"] += 1
            if w.get("pn_side"):
                flagc[f"{w['cls']}:pn_{w['pn_side']}"] += 1
            if len(samples[w["cls"]]) < 12:
                samples[w["cls"]].append(w)
            # Persist the window's TOKEN IDS, aligned row-for-row with the JSONL.
            # Every flag and every class decision above was computed on THESE ids.
            # Re-deriving them at capture time by tokenising the stored text is a
            # different operation -- the text was cut by character offsets, and a
            # substring tokenised in isolation is not guaranteed to reproduce the
            # segmentation that span had inside the full document. Carrying the ids
            # makes the question moot rather than bounding it.
            #
            # Note on a related non-issue, recorded so nobody re-derives it: a
            # decode round-trip is NOT a valid check here. tok.decode drops spaces
            # before some punctuation ("a ?best" -> "a?best", "batted .212" ->
            # "batted.212"), so decode(encode(text)) != text for ~7% of windows
            # even when the encoding is perfectly correct. That measures lossy
            # DEcoding, not divergent encoding.
            kept_ids.append(w["window_ids"])
            kept.append({k: v for k, v in w.items() if k != "window_ids"})
        if doc_idx % 2000 == 0 and doc_idx:
            print(f"[scan] {doc_idx} docs | " +
                  " ".join(f"{c}={counts[c]}" for c in sorted(counts)), flush=True)

    print("\n" + "=" * 78)
    print(f"SELECTION SUMMARY  ({doc_idx + 1} docs scanned, N_max={args.n_max})")
    print("=" * 78)
    for c in ["positive", "dirty_positive", "proper_noun_weekday", "near_miss",
              "matched_negative", "floor"]:
        print(f"  {c:<24}{counts[c]:>8}")
    print(f"  {'exact-dup dropped':<24}{dup:>8}")
    print(f"\n  positive day balance: " +
          ", ".join(f"{d[:3]}={days[d]}" for d in DAYS))
    tot_wd = counts["positive"] + counts["dirty_positive"] + counts["proper_noun_weekday"]
    if tot_wd:
        print(f"  clean-positive yield: {counts['positive']}/{tot_wd} = "
              f"{100*counts['positive']/tot_wd:.1f}% of all weekday-final windows")
    print("\n  flags:")
    for k in sorted(flagc):
        print(f"    {k:<44}{flagc[k]:>7}")

    for c, ws in samples.items():
        print(f"\n--- {c} samples (last 60 chars of window, capture token in <>) ---")
        for w in ws[:8]:
            tail = w["text"][-60:].replace("\n", " ")
            print(f"    ...{tail!r}  <{w['capture_piece'].strip()}>")

    os.makedirs(args.out, exist_ok=True)
    import numpy as np
    idpath = os.path.join(args.out, f"windows_n{args.n_max}_ids.npy")
    np.save(idpath, np.asarray(kept_ids, dtype=np.int32))
    print(f"[saved] {idpath}  {np.asarray(kept_ids).shape}")
    path = os.path.join(args.out, f"windows_n{args.n_max}.jsonl")
    with open(path, "w") as f:
        for w in kept:
            f.write(json.dumps(w) + "\n")
    meta = {"dataset": args.dataset, "config": args.config, "model": args.model,
            "n_max": args.n_max, "docs_scanned": doc_idx + 1, "seed": args.seed,
            "per_doc_cap": args.per_doc_cap, "counts": dict(counts),
            "day_balance": dict(days), "dups_dropped": dup, "flags": dict(flagc)}
    with open(os.path.join(args.out, f"selection_meta_n{args.n_max}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[saved] {path}")
    return 0


if __name__ == "__main__":
    rc = main()
    # The HF `datasets` streaming reader keeps worker threads alive, and the Rust
    # tokenizer releases the GIL from one of them during interpreter finalization:
    # "Fatal Python error: PyGILState_Release: thread state must be current".
    # It fires AFTER every output is written, so the data is complete, but the
    # process aborts with 134 and any caller checking the exit status sees a
    # failure that did not happen. Skip finalization rather than let that stand.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
