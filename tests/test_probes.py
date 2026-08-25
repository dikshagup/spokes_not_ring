"""Unit tests for the between-days probe builders (pure; no model)."""

from __future__ import annotations

import pytest

from weekday_manifold.manifold.days import DAYS, N_DAYS
from weekday_manifold.manifold.probes import (
    GOODFIRE_TEMPLATE,
    NO_ANSWER_DAY,
    PROBE_ORDER,
    build_all_probes,
    build_deictic_probes,
    build_fractional_probes,
    build_hours_probes,
    build_integer_controls,
    build_typo_probes,
    knot_distance,
)


# ------------------------------------------------------------------ the frame
def test_integer_word_controls_reproduce_the_goodfire_prompts():
    """C-int-word must be byte-identical to the canonical 49 interrogative prompts.

    This is the substrate contract: if the control arm drifts from the prompts the
    ring is fitted on, the knot baseline is measuring a different stimulus.
    """
    from weekday_manifold.manifold.days import build_prompts

    canonical = {s.text for s in build_prompts("interrogative")}
    ctrl = {s.text for s in build_integer_controls(renderings=("word",))}
    assert ctrl == canonical


def test_fractional_probes_keep_the_template_and_vary_only_the_offset():
    head, tail = GOODFIRE_TEMPLATE.split("{k} days")
    for sp in build_fractional_probes():
        assert sp.text.startswith(head)
        assert sp.text.endswith(tail.format(z=DAYS[sp.meta["z"]]))


def test_hours_probes_swap_the_unit_not_the_frame():
    from weekday_manifold.manifold.probes import _WORD_HOURS

    for sp in build_hours_probes():
        h = sp.meta["hours"]
        h_str = _WORD_HOURS[h] if sp.meta["rendering"] == "word" else str(h)
        assert sp.text.startswith("Q: What day is ")
        assert f"{h_str} hours after {DAYS[sp.meta['z']]}" in sp.text
        assert "days after" not in sp.text


def test_hours_probes_cover_both_renderings_and_agree_on_u():
    """Word is the default arm; the digit arm exists only as the surface control.

    Phase 0 measured digits collapsing this template even for plain integers
    (C-int-digit P(weekday) 0.484 vs word 0.739), so the two arms must differ in text
    while asking for the identical offset.
    """
    specs = build_hours_probes()
    assert {s.meta["rendering"] for s in specs} == {"word", "digit"}
    for z in range(N_DAYS):
        w = next(s for s in specs if s.meta["rendering"] == "word"
                 and s.meta["z"] == z and s.meta["hours"] == 36)
        d = next(s for s in specs if s.meta["rendering"] == "digit"
                 and s.meta["z"] == z and s.meta["hours"] == 36)
        assert "thirty-six hours" in w.text and "36 hours" in d.text
        assert w.meta["u_expected"] == pytest.approx(d.meta["u_expected"])
        # 36h = 1.5 days -> strictly between two knots
        assert knot_distance(w.meta["u_expected"]) == pytest.approx(0.5)


# ----------------------------------------------------------- the u coordinate
@pytest.mark.parametrize(
    "z,k,expected",
    [
        (0, 2.5, 2.5 / 7),    # 2.5 days after Monday -> between Wed (2/7) and Thu (3/7)
        (0, 1.5, 1.5 / 7),    # 36 hours after Monday -> between Tue and Wed
        (5, 3.5, (5 + 3.5) / 7 % 1.0),   # wraps past Sunday
        (6, 6.5, (6 + 6.5) / 7 % 1.0),
    ],
)
def test_u_expected_places_fractional_offsets_between_knots(z, k, expected):
    (sp,) = [s for s in build_fractional_probes(ks=(k,), renderings=("digit",))
             if s.meta["z"] == z]
    assert sp.meta["u_expected"] == pytest.approx(expected)
    # and it genuinely sits strictly between two knots, not on one
    frac = (sp.meta["u_expected"] * N_DAYS) % 1.0
    assert 0.01 < frac < 0.99


def test_integer_controls_land_exactly_on_knots():
    """The whole method rests on this: integer offsets must have u = k/7 exactly."""
    for sp in build_integer_controls():
        assert knot_distance(sp.meta["u_expected"]) == pytest.approx(0.0, abs=1e-9)
        assert sp.answer_day == (sp.meta["z"] + sp.meta["k"]) % N_DAYS


def test_knot_distance_is_wrap_safe():
    """A u just below a knot is ON the knot, not maximally far from it."""
    assert knot_distance(0.0) == pytest.approx(0.0)
    assert knot_distance(1.0 - 1e-15) == pytest.approx(0.0, abs=1e-9)
    assert knot_distance(1.0 / N_DAYS) == pytest.approx(0.0, abs=1e-9)
    assert knot_distance(0.5 / N_DAYS) == pytest.approx(0.5)   # exactly mid-gap
    assert knot_distance(2.5 / N_DAYS) == pytest.approx(0.5)


def test_exact_hour_multiples_are_flagged_as_controls_and_land_on_knots():
    exact = [s for s in build_hours_probes() if s.meta["is_exact_day"]]
    assert exact, "expected 24/48/72-hour rows as the C-int-hours control"
    for sp in exact:
        assert sp.meta["control"] == "C-int-hours"
        assert knot_distance(sp.meta["u_expected"]) == pytest.approx(0.0, abs=1e-9)
        assert sp.answer_day != NO_ANSWER_DAY

    inexact = [s for s in build_hours_probes() if not s.meta["is_exact_day"]]
    for sp in inexact:
        assert sp.answer_day == NO_ANSWER_DAY, "fractional rows must carry no day label"


# ------------------------------------------------------- sites & label hygiene
def test_every_probe_has_a_resolvable_capture_site():
    for sp in build_all_probes():
        sites = sp.meta["sites"]
        assert sites["last_token"] == sp.text
        # every site is a genuine text prefix -> resolve_capture_position can index it
        for name, prefix in sites.items():
            assert sp.text.startswith(prefix), (name, prefix, sp.text)


def test_typo_probes_reuse_the_goodfire_frame_verbatim():
    """P3 must vary ONLY the day spelling, so it stays comparable to C-int and P1/P2.

    Same template, same offset word, same capture site — the single difference from
    ``C-int-word`` at k=1 is that ``{z}`` is misspelled.
    """
    from weekday_manifold.manifold.probes import TYPO_OFFSET_K, _WORD_INTS

    k_str = _WORD_INTS[TYPO_OFFSET_K]
    for sp in build_typo_probes():
        assert sp.text == GOODFIRE_TEMPLATE.format(k=k_str, z=sp.meta["typo"])
        assert sp.meta["role"] == "compute"


def test_typo_probes_get_a_mention_site_on_the_typo_itself():
    """The typo carries no real weekday, so the mention site must end at the typo.

    ``_mention_prefix`` only does a ``str.find``, so passing the misspelling works —
    this is the same freedom ``capture_controls_mention.py`` relies on.
    """
    for sp in build_typo_probes():
        typo = sp.meta["typo"]
        assert sp.meta["sites"]["mention_token"].endswith(typo)
        a, b = sp.meta["blends"]
        assert 0 <= a < N_DAYS and 0 <= b < N_DAYS
        if sp.meta["is_single_typo"]:
            # a near-typo of ONE day has a definite intended answer (that day + k)
            from weekday_manifold.manifold.probes import TYPO_OFFSET_K
            assert a == b
            assert sp.answer_day == (a + TYPO_OFFSET_K) % N_DAYS
            assert knot_distance(sp.meta["u_expected"]) == pytest.approx(0.0, abs=1e-9)
            assert sp.meta["control"] == "C-typo-single"
        else:
            assert a != b, "a two-day blend must name two different days"
            assert sp.meta["u_expected"] is None, "blend intent is ambiguous by design"
            assert sp.answer_day == NO_ANSWER_DAY
            assert sp.meta["control"] is None


def test_typo_family_has_both_blends_and_single_day_controls():
    specs = build_typo_probes()
    singles = [s for s in specs if s.meta["is_single_typo"]]
    blends = [s for s in specs if not s.meta["is_single_typo"]]
    assert len(singles) >= 3, "need within-family controls that should land on a knot"
    assert len(blends) >= 8, "the blends are the probes proper"


def test_typo_probes_contain_no_real_weekday():
    """If a typo still spells a real day, it is not a blend and would confound P3."""
    for sp in build_typo_probes():
        for day in DAYS:
            assert day not in sp.text, f"{sp.meta['typo']!r} contains {day!r}"


def test_deictic_probes_add_a_time_of_day_inside_the_goodfire_frame():
    """P4 = P2's frame with a time-of-day on the anchor ("... after Monday afternoon")."""
    from weekday_manifold.manifold.probes import _WORD_HOURS

    specs = build_deictic_probes()
    assert specs
    for sp in specs:
        tod, z, h = sp.meta["time_of_day"], sp.meta["z"], sp.meta["hours"]
        assert sp.text == (
            GOODFIRE_TEMPLATE.replace("{k} days", f"{_WORD_HOURS[h]} hours")
            .format(z=f"{DAYS[z]} {tod}"))
        if sp.meta["is_exact_day"]:
            # a whole-day hop keeps a definite answer and must sit on a knot
            assert knot_distance(sp.meta["u_expected"]) == pytest.approx(0.0, abs=1e-9)
            assert sp.answer_day == (z + h // 24) % N_DAYS
        else:
            # 12h / 18h from a time-of-day cross midnight — the between-days case
            assert sp.answer_day == NO_ANSWER_DAY
            assert knot_distance(sp.meta["u_expected"]) > 0.01


def test_deictic_and_hours_families_are_directly_comparable():
    """P4 must differ from P2 at the same hour count ONLY by the time-of-day phrase."""
    p2 = {(s.meta["z"], s.meta["hours"]): s for s in build_hours_probes()
          if s.meta["rendering"] == "word"}
    for sp in build_deictic_probes():
        key = (sp.meta["z"], sp.meta["hours"])
        if key not in p2:
            continue
        assert sp.text == p2[key].text.replace(
            f"{DAYS[sp.meta['z']]}?", f"{DAYS[sp.meta['z']]} {sp.meta['time_of_day']}?")
        assert sp.meta["u_expected"] == pytest.approx(p2[key].meta["u_expected"])


def test_probes_bypass_library_assertions_that_would_reject_them():
    """``build_library`` would reject the typo probes; the direct path must not.

    ``prompt_library._make_spec`` asserts exactly one weekday mention. Typo probes have
    zero, which is precisely why they are constructed directly.
    """
    from weekday_manifold.manifold.prompt_library import _weekday_mentions

    for sp in build_typo_probes():
        assert _weekday_mentions(sp.text) == 0


# ------------------------------------------------------------------ assembly
def test_build_all_probes_is_deduplicated_and_ordered():
    specs = build_all_probes()
    texts = [s.text for s in specs]
    assert len(texts) == len(set(texts)), "capture cache keys on text; no duplicates"

    seen_order = []
    for s in specs:
        if s.meta["family"] not in seen_order:
            seen_order.append(s.meta["family"])
    assert seen_order == [f for f in PROBE_ORDER if f in seen_order]


def test_question_families_share_the_goodfire_frame():
    """Every COMPUTE family sits in Goodfire's frame, so their landings are comparable.

    T is deliberately exempt: it is a plain statement with no question and no arithmetic,
    which is the whole point of that family — so it is asserted separately below.
    """
    for sp in build_all_probes():
        if sp.meta["role"] != "compute":
            continue
        assert sp.text.endswith("?\nA:"), sp.text
        assert sp.text.startswith("Q: "), sp.text


def test_time_statements_are_not_questions():
    from weekday_manifold.manifold.probes import build_time_statements

    for sp in build_time_statements():
        assert "?" not in sp.text and "Q:" not in sp.text and "A:" not in sp.text
        assert sp.meta["role"] == "read"
        assert sp.meta["sites"]["mention_token"].endswith(DAYS[sp.meta["z"]])
        # capture site is the sentence-final full stop, identical for every prompt, so
        # neither the weekday token nor the am/pm token can confound it
        assert sp.text.endswith(".")
        assert not sp.text[:-1].rstrip().endswith(("am", "pm"))


def test_no_probe_is_captured_on_a_weekday_token():
    """The capture site is the final token; it must never BE the weekday.

    RESULTS.md §7 measured the weekday mention site as essentially day-token identity
    (decodes at 1.00 from layer 0). An activation read there reflects the embedding, not
    the concept, so any family whose last token is a weekday is measuring the wrong thing.

    T is allowed to have the weekday immediately BEFORE its capture token — the capture
    is the sentence-final full stop. That adjacency is a stated caveat, not a violation:
    inherited day identity is constant across a day's 24 times and so predicts zero time
    sensitivity, meaning it can dilute the measured effect but never manufacture it.
    """
    for sp in build_all_probes():
        last = sp.text.rstrip()
        assert not any(last.endswith(d) for d in DAYS), sp.text
        assert sp.meta["sites"]["last_token"] == sp.text


def test_time_statements_cover_the_clock():
    from weekday_manifold.manifold.probes import build_time_statements

    specs = build_time_statements()
    assert all(s.meta["day_is_last"] is False for s in specs)
    for day_last in (False,):
        arm = [s for s in specs if s.meta["day_is_last"] == day_last]
        assert len(arm) == N_DAYS * 22
        # 22 within-day positions: midnight and noon are excluded because "twelve am"
        # and "twelve pm" reuse one hour word at both ends of the am/pm split
        fr = sorted({round(s.meta["k_days"], 6) for s in arm})
        assert len(fr) == 22
        assert all(s.meta["clock_hour"] not in (0, 12) for s in arm)
        assert min(s.meta["clock_hour"] for s in arm) == 1
        assert max(s.meta["clock_hour"] for s in arm) == 23


def test_hour_sweep_is_dense_and_carries_its_own_knot_controls():
    from weekday_manifold.manifold.probes import build_hour_sweep

    specs = build_hour_sweep()
    assert len(specs) == N_DAYS * 72
    exact = [s for s in specs if s.meta["is_exact_day"]]
    assert len(exact) == N_DAYS * 3           # 24h, 48h, 72h per anchor
    for sp in exact:
        assert knot_distance(sp.meta["u_expected"]) == pytest.approx(0.0, abs=1e-9)
        assert sp.answer_day == (sp.meta["z"] + sp.meta["hours"] // 24) % N_DAYS
    # 1-hour resolution: consecutive requests differ by 1/24 day on the ring
    mon = sorted((s for s in specs if s.meta["z"] == 0), key=lambda s: s.meta["hours"])
    d = (mon[1].meta["u_expected"] - mon[0].meta["u_expected"]) * N_DAYS
    assert d == pytest.approx(1 / 24)
    assert "one hour after" in mon[0].text, "h=1 must take the singular"


def test_family_subset_selection():
    only = build_all_probes(families=("P3", "P4"))
    assert {s.meta["family"] for s in only} == {"P3", "P4"}
    with pytest.raises(ValueError, match="unknown probe families"):
        build_all_probes(families=("P9",))


def test_fractional_probes_cover_both_renderings():
    specs = build_fractional_probes()
    assert {s.meta["rendering"] for s in specs} == {"digit", "word"}
    # the digit/word pair must differ in surface form but agree on u_expected
    for z in range(N_DAYS):
        d = next(s for s in specs if s.meta["rendering"] == "digit"
                 and s.meta["z"] == z and s.meta["k_days"] == 2.5)
        w = next(s for s in specs if s.meta["rendering"] == "word"
                 and s.meta["z"] == z and s.meta["k_days"] == 2.5)
        assert d.text != w.text
        assert d.meta["u_expected"] == pytest.approx(w.meta["u_expected"])
