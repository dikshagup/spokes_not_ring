"""Prompt-library invariants for the weekday prompt library (mostly pure; a char-level
fake model for token-length annotation, so no real tokenizer is needed)."""

import re

import numpy as np
import torch

from weekday_manifold.manifold.days import DAYS
from weekday_manifold.manifold.prompt_library import (
    annotate_token_lengths,
    assign_splits,
    build_distractors,
    build_library,
)

DAY_RE = re.compile(r"\b(" + "|".join(DAYS) + r")\b")


class CharModel:
    """Char-level 'tokenizer' (prefix-consistent, like BPE for our prefixes)."""

    def to_tokens(self, s, prepend_bos=True):
        ids = [256] if prepend_bos else []
        ids += [ord(c) % 256 for c in s]
        return torch.tensor([ids])


def test_build_library_nonempty_and_meta_complete():
    specs = build_library()
    assert len(specs) > 300
    required = {"role", "inference", "family", "template_id", "register", "z", "k",
                "offset", "k_dir", "content_id", "position_target",
                "n_weekday_mentions", "stated", "sites"}
    for s in specs:
        assert required <= set(s.meta)
        assert 0 <= s.answer_day < 7
        assert s.capture_text == s.text


def test_every_prompt_names_exactly_one_weekday():
    for s in build_library():
        n = len(DAY_RE.findall(s.text))
        assert n == 1 == s.meta["n_weekday_mentions"], s.text
        if s.meta["stated"]:                       # read: the anchor IS the label
            assert re.search(rf"\b{DAYS[s.answer_day]}\b", s.text), s.text
        elif s.answer_day != s.meta["z"]:          # compute: answer is latent (unless same-day)
            assert not re.search(rf"\b{DAYS[s.answer_day]}\b", s.text), s.text


def test_grammar_no_singular_plural_slip():
    # duration rendering must be grammatical across k=1..7 ("one day", not "one days").
    for s in build_library(families=["C3"]):
        assert "one days" not in s.text, s.text
        assert "-day-" not in s.text, s.text


def test_offsets_present():
    ks = {s.meta["k"] for s in build_library(families=["C3"])}
    assert ks == {1, 2, 3, 4, 5, 6, 7}
    # read families have fixed offset 0, no duration k
    assert all(s.meta["k"] is None for s in build_library(roles=["read"]))


def test_every_day_represented_per_family():
    specs = build_library()
    for fam in {s.meta["family"] for s in specs}:
        days = {s.answer_day for s in specs if s.meta["family"] == fam}
        assert days == set(range(7)), fam


def test_sites_present_and_are_prefixes():
    for s in build_library():
        sites = s.meta["sites"]
        assert sites["last_token"] == s.text
        assert ("mention_token" in sites) == s.meta["stated"]
        for cap in sites.values():
            assert s.text.startswith(cap)


def test_assign_splits_holds_out_whole_templates():
    specs = build_library()
    assign_splits(specs, iid_frac=0.2, seed=0)
    tags = {s.meta["split"] for s in specs}
    assert tags == {"train", "test:iid"}
    # every template is wholly train OR wholly test
    by_tid = {}
    for s in specs:
        by_tid.setdefault(s.meta["template_id"], set()).add(s.meta["split"])
    assert all(len(v) == 1 for v in by_tid.values())
    # each family keeps >=1 train template, and all 7 days survive in train
    for fam in {s.meta["family"] for s in specs}:
        fam_specs = [s for s in specs if s.meta["family"] == fam]
        train = [s for s in fam_specs if s.meta["split"] == "train"]
        assert train
        assert {s.answer_day for s in train} == set(range(7)), fam
    # deterministic
    specs2 = build_library()
    assign_splits(specs2, iid_frac=0.2, seed=0)
    a = {s.text: s.meta["split"] for s in specs}
    assert all(a[s.text] == s.meta["split"] for s in specs2)


def test_distractors_months_and_random():
    d = build_distractors()
    assert len(d["months"]) == 12 * 3
    assert all(0 <= s.answer_day < 12 for s in d["months"])
    assert all(s.answer_day == -1 for s in d["random"])


def test_annotate_fills_token_fields():
    specs = build_library(families=["C3", "R1", "R3", "R7"])
    annotate_token_lengths(specs, CharModel())
    for s in specs:
        assert s.meta["n_tokens"] > 0
        assert 0.0 <= s.meta["position_frac"] <= 1.0
        assert s.meta["position_bin"] in ("early", "mid", "late")
        assert s.meta["length_bin"] in ("short", "med", "long")
        assert set(s.meta["site_pos"]) == set(s.meta["sites"])


def test_read_anchor_position_orders_early_mid_late():
    specs = build_library(families=["R1"])
    annotate_token_lengths(specs, CharModel())
    frac = {p: np.mean([s.meta["position_frac"] for s in specs
                        if s.meta["position_target"] == p])
            for p in ("early", "mid", "late")}
    assert frac["early"] < frac["mid"] < frac["late"]
