"""Prompt families for the weekday ring: mention, interrogative, and the between-day probes."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from weekday_manifold.manifold.concept import PromptSpec
from weekday_manifold.manifold.days import DAYS, N_DAYS
from weekday_manifold.manifold.prompt_library import _sites

# The Goodfire template, verbatim (days.INTERROGATIVE_TEMPLATES[0]).
GOODFIRE_TEMPLATE = "Q: What day is {k} days after {z}?\nA:"
# NOTE ON ANSWER POSITION (why every family stays in the Q:/A: frame). The capture site
# is the prompt's final token, and the whole method assumes its NEXT-token prediction is
# the weekday. Prose framings break that assumption: an earlier deictic form ("It is
# {day} afternoon. What day is it 24 hours from now?\nA:") answered " It" for all 7
# anchors — the model starts "It is Tuesday", so the day lands two tokens past the
# capture site and the captured activation would encode "about to say It" rather than a
# day. A relational typo form ("What day comes after {typo}?\nA:") likewise spent its
# mass re-spelling the misspelling (" Tu", " S").
#
# Both were fixed by moving INTO Goodfire's frame rather than by patching the answer
# slot: the canonical "Q: ... ?\nA:" reliably puts a weekday at the very next token
# (C-int-word P(weekday) 0.739, top tokens are weekdays). So no answer-forcing suffix is
# used anywhere, and every family shares one frame, one capture site, one scale.

# P3 reuses GOODFIRE_TEMPLATE UNCHANGED and misspells only the day name — exactly
# parallel to how P1/P2 vary only the offset. Every family therefore asks the same
# question in the same frame at the same capture site, which is what makes their landing
# positions comparable to each other and to the canonical 49.
#
# Two other framings were tried and rejected. "Q: What day comes after {typo}?\nA:"
# changes the frame (relational, no offset slot) and spends mass re-spelling the
# misspelling (top tokens ' Tu', ' S'; P(weekday) 0.310). Appending " It is" to force
# the answer slot made things worse, not better: it invites a meta-linguistic answer and
# the top token becomes ' called' — "It is CALLED Tuurday" (0.254, 17/17 off-concept).
# The offset used here is fixed at one day so the only variable against C-int-word is
# the spelling of {z}.
TYPO_OFFSET_K = 1
# P4: "what is 24 hours after Monday afternoon". Built from GOODFIRE_TEMPLATE with the
# same unit swap P2 uses, and a time-of-day appended to {z} — so the frame stays
# byte-identical to P2/C-int and only the anchor gains a sub-day position.
#
# The earlier standalone prose form ("It is {z} afternoon. What day is it 24 hours from
# now?") is not used: it left the frame, and it put the weekday two tokens past the
# capture site (top token ' It' for all 7 anchors, P(weekday) 0.257), which would have
# made the captured activation encode "about to say It" rather than a day.
TIMES_OF_DAY: List[str] = ("morning", "afternoon", "evening")

# Hour offsets from a time-of-day anchor. 24 is the exact-day hop the family is named
# for; 12 and 18 CROSS MIDNIGHT from an afternoon anchor, which is the naturally
# "between days" case and the reason the family is interesting at all.
DEICTIC_HOURS: List[int] = (12, 18, 24, 36)

HOURS_PER_DAY = 24

# Word forms for the hour counts. Measured Phase-0 fact: DIGIT rendering collapses this
# template on base Llama-3.1-8B — and it does so for plain INTEGER offsets too
# (C-int-digit P(weekday) 0.484 vs C-int-word 0.739; P1-digit 0.298 vs P1-word 0.717,
# with a bare ' ' as the top token 37/49 times). So the digit arm is a surface effect,
# not an arithmetic failure, and word rendering is the default everywhere.
_WORD_HOURS: Dict[int, str] = {
    6: "six", 12: "twelve", 18: "eighteen", 24: "twenty-four", 30: "thirty",
    36: "thirty-six", 42: "forty-two", 48: "forty-eight", 54: "fifty-four",
    60: "sixty", 72: "seventy-two", 84: "eighty-four",
}

# Word forms for the half-integer offsets, matching the paper's word-number style.
_WORD_HALVES: Dict[float, str] = {
    0.5: "half a",
    1.5: "one and a half",
    2.5: "two and a half",
    3.5: "three and a half",
    4.5: "four and a half",
    5.5: "five and a half",
    6.5: "six and a half",
}
_WORD_INTS: Dict[int, str] = {
    1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven",
}

# Typo pairs: a misspelling that sits BETWEEN two real weekdays, and the two days it
# blends. Chosen so the surface form is genuinely ambiguous rather than a clear typo of
# one day — "Tuurday" reads toward both Tuesday and Thursday, whereas "Mondayy" would
# just be Monday. ``blends`` records the intended pair for analysis; a pair (d, d) marks
# a deliberate SINGLE-day near-typo control, which should land on that day's knot.
#
# Phase-0 note: this family's weekday mass is partly spent on the model spelling the
# misspelled word back out (top tokens ' Tu', ' S' — i.e. continuing "Tuurday" rather
# than answering). That depresses P(weekday) without meaning the concept is unengaged,
# so the family is kept larger than the others to survive that dilution, and the
# per-prompt verdicts matter more here than the family mean.
TYPO_BLENDS: List[Dict[str, object]] = [
    # two-day blends — the probes proper
    {"typo": "Tuurday", "blends": (1, 3)},      # Tuesday / Thursday
    {"typo": "Thuesday", "blends": (1, 3)},     # Tuesday / Thursday
    {"typo": "Tuednesday", "blends": (1, 2)},   # Tuesday / Wednesday
    {"typo": "Thursnesday", "blends": (2, 3)},  # Wednesday / Thursday
    {"typo": "Fridnesday", "blends": (2, 4)},   # Wednesday / Friday
    {"typo": "Frisday", "blends": (4, 5)},      # Friday / Saturday
    {"typo": "Satursday", "blends": (5, 6)},    # Saturday / Sunday
    {"typo": "Satunday", "blends": (5, 6)},     # Saturday / Sunday
    {"typo": "Sunsday", "blends": (5, 6)},      # Saturday / Sunday
    {"typo": "Sundnesday", "blends": (2, 6)},   # Wednesday / Sunday
    {"typo": "Monsday", "blends": (0, 6)},      # Monday / Sunday
    {"typo": "Fronday", "blends": (0, 4)},      # Monday / Friday
    {"typo": "Mridayy", "blends": (0, 4)},      # Monday / Friday
    # single-day near-typos — within-family controls; must land ON their own knot
    {"typo": "Wednesay", "blends": (2, 2)},
    {"typo": "Thurday", "blends": (3, 3)},
    {"typo": "Tusday", "blends": (1, 1)},
    {"typo": "Mondey", "blends": (0, 0)},
]

# ``answer_day`` sentinel for probes with no integer ground-truth day. Matches the
# ``build_distractors`` convention. Read ``u`` directly for these — every accuracy
# helper (``day_recovery``, ``nc_decode``) rounds ``u`` to a knot and would destroy the
# measurement.
NO_ANSWER_DAY = -1


def _spec(text: str, family: str, meta: Dict[str, object],
          stated_day: Optional[str], answer_day: int) -> PromptSpec:
    """One probe ``PromptSpec`` with library-compatible ``meta["sites"]``."""
    full = dict(meta)
    full["sites"] = _sites(text, stated_day)
    full["family"] = family
    full["template_id"] = f"{family}:{full.get('content_id', 0)}"
    full["probe"] = True
    return PromptSpec(text=text, answer_day=answer_day, capture_text=text,
                      formulation=family, meta=full)


def _u_expected(z: int, k_days: float) -> float:
    """Stimulus ring coordinate for a ``k_days`` offset from anchor day ``z``."""
    return ((z + k_days) / float(N_DAYS)) % 1.0


def knot_distance(u, n_labels: int = N_DAYS):
    """Distance from ``u`` to the nearest day knot, in DAY units (0 at a day, 0.5 mid-gap)."""
    import numpy as np

    f = np.mod(np.asarray(u, dtype=float) * n_labels, 1.0)
    return np.minimum(f, 1.0 - f)


# --------------------------------------------------------------------- families
def build_fractional_probes(
    ks: Sequence[float] = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5),
    renderings: Sequence[str] = ("digit", "word"),
) -> List[PromptSpec]:
    """P1 — fractional day offsets. Highest arithmetic load; stretch goal."""
    out: List[PromptSpec] = []
    for ri, rendering in enumerate(renderings):
        for z in range(N_DAYS):
            for k in ks:
                if rendering == "digit":
                    k_str = f"{k:g}"
                else:
                    if k not in _WORD_HALVES:
                        raise ValueError(f"no word rendering for k={k}.")
                    k_str = _WORD_HALVES[k]
                if k < 1.0:
                    # "half a days" is ungrammatical; swap the whole "{k} days" slot so
                    # the rest of the frame stays byte-identical (same idiom as P2).
                    text = (GOODFIRE_TEMPLATE
                            .replace("{k} days", f"{k_str} day" if rendering == "word"
                                     else f"{k_str} days")
                            .format(z=DAYS[z]))
                else:
                    text = GOODFIRE_TEMPLATE.format(k=k_str, z=DAYS[z])
                out.append(_spec(
                    text, "P1",
                    {"z": z, "k_days": float(k), "rendering": rendering,
                     "content_id": ri, "u_expected": _u_expected(z, k),
                     "arith_load": "high", "role": "compute"},
                    DAYS[z], NO_ANSWER_DAY))
    return out


def build_hours_probes(
    hours: Sequence[int] = (12, 18, 24, 30, 36, 42, 48, 60, 84),
    renderings: Sequence[str] = ("word", "digit"),
) -> List[PromptSpec]:
    """P2 — unit-mismatched offsets ("thirty-six hours after Monday"). Moderate load."""
    out: List[PromptSpec] = []
    for ri, rendering in enumerate(renderings):
        for z in range(N_DAYS):
            for h in hours:
                k_days = h / float(HOURS_PER_DAY)
                if rendering == "word":
                    if h not in _WORD_HOURS:
                        raise ValueError(f"no word rendering for h={h}.")
                    h_str = _WORD_HOURS[h]
                else:
                    h_str = str(h)
                # Swap the unit inside the frame ("{k} days" -> "<h> hours") rather than
                # filling {k}, so the rest of the template stays byte-identical.
                text = (GOODFIRE_TEMPLATE
                        .replace("{k} days", f"{h_str} hours").format(z=DAYS[z]))
                is_exact = (h % HOURS_PER_DAY) == 0
                out.append(_spec(
                    text, "P2",
                    {"z": z, "hours": int(h), "k_days": k_days,
                     "rendering": rendering, "content_id": ri,
                     "u_expected": _u_expected(z, k_days),
                     "is_exact_day": is_exact,
                     "arith_load": "moderate", "role": "compute",
                     "control": "C-int-hours" if is_exact else None},
                    DAYS[z],
                    (z + int(k_days)) % N_DAYS if is_exact else NO_ANSWER_DAY))
    return out


def build_typo_probes(blends: Optional[Sequence[Dict[str, object]]] = None) -> List[PromptSpec]:
    """P3 — typo-blended day names. ZERO arithmetic load; the most robust family."""
    rows = list(TYPO_BLENDS if blends is None else blends)
    out: List[PromptSpec] = []
    k_str = _WORD_INTS[TYPO_OFFSET_K]
    for ci, row in enumerate(rows):
        typo = str(row["typo"])
        a, b = row["blends"]  # type: ignore[misc]
        single = int(a) == int(b)
        text = GOODFIRE_TEMPLATE.format(k=k_str, z=typo)
        out.append(_spec(
            text, "P3",
            {"typo": typo, "blends": (int(a), int(b)),
             "z": int(a) if single else None, "k_days": float(TYPO_OFFSET_K),
             "content_id": ci, "rendering": "word",
             # a near-typo of ONE day has a definite intended answer (that day + k); a
             # two-day blend does not, and its landing position is the measurement.
             "u_expected": _u_expected(int(a), TYPO_OFFSET_K) if single else None,
             "is_single_typo": single,
             "control": "C-typo-single" if single else None,
             "arith_load": "none", "role": "compute"},
            typo, (int(a) + TYPO_OFFSET_K) % N_DAYS if single else NO_ANSWER_DAY))
    return out


def build_deictic_probes(
    hours: Sequence[int] = DEICTIC_HOURS,
    times: Sequence[str] = TIMES_OF_DAY,
) -> List[PromptSpec]:
    """P4 — "What day is twenty-four hours after Monday afternoon?". Lowest load."""
    out: List[PromptSpec] = []
    for ci, tod in enumerate(times):
        for z in range(N_DAYS):
            for h in hours:
                if h not in _WORD_HOURS:
                    raise ValueError(f"no word rendering for h={h}.")
                k_days = h / float(HOURS_PER_DAY)
                text = (GOODFIRE_TEMPLATE
                        .replace("{k} days", f"{_WORD_HOURS[h]} hours")
                        .format(z=f"{DAYS[z]} {tod}"))
                is_exact = (h % HOURS_PER_DAY) == 0
                out.append(_spec(
                    text, "P4",
                    {"z": z, "hours": int(h), "k_days": k_days, "time_of_day": tod,
                     "rendering": "word", "content_id": ci,
                     "u_expected": _u_expected(z, k_days),
                     "is_exact_day": is_exact,
                     "control": "C-deictic-exact" if is_exact else None,
                     "arith_load": "low", "role": "compute"},
                    DAYS[z],
                    (z + int(k_days)) % N_DAYS if is_exact else NO_ANSWER_DAY))
    return out


# --------------------------------------------------------------------- controls
def build_integer_controls(
    ks: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    renderings: Sequence[str] = ("word", "digit"),
) -> List[PromptSpec]:
    """C-int-word / C-int-digit — the knot baseline and the rendering control."""
    out: List[PromptSpec] = []
    for ri, rendering in enumerate(renderings):
        for z in range(N_DAYS):
            for k in ks:
                k_str = str(k) if rendering == "digit" else _WORD_INTS[k]
                text = GOODFIRE_TEMPLATE.format(k=k_str, z=DAYS[z])
                out.append(_spec(
                    text, "C-int",
                    {"z": z, "k_days": float(k), "k": int(k), "rendering": rendering,
                     "content_id": ri, "u_expected": _u_expected(z, k),
                     "arith_load": "canonical", "role": "compute",
                     "control": f"C-int-{rendering}"},
                    DAYS[z], (z + k) % N_DAYS))
    return out



# ---------------------------------------------------------------- the hour sweep
_UNITS = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy"}


def number_word(n: int) -> str:
    """Spell an integer 1..79 (word rendering is mandatory — see _WORD_HOURS)."""
    if n < 20:
        return _UNITS[n]
    tens, unit = divmod(n, 10)
    if tens not in _TENS:
        raise ValueError(f"no word form for {n}.")
    return _TENS[tens] + (f"-{_UNITS[unit]}" if unit else "")


def build_hour_sweep(hours: Sequence[int] = tuple(range(1, 73)),
                     anchors: Optional[Sequence[int]] = None) -> List[PromptSpec]:
    """H — a DENSE hour sweep: "one hour after Monday", "two hours after Monday", ..."""
    anchors = list(range(N_DAYS)) if anchors is None else list(anchors)
    out: List[PromptSpec] = []
    for z in anchors:
        for h in hours:
            k_days = h / float(HOURS_PER_DAY)
            unit = "hour" if h == 1 else "hours"
            text = (GOODFIRE_TEMPLATE
                    .replace("{k} days", f"{number_word(h)} {unit}").format(z=DAYS[z]))
            exact = (h % HOURS_PER_DAY) == 0
            out.append(_spec(
                text, "H",
                {"z": z, "hours": int(h), "k_days": k_days, "rendering": "word",
                 "content_id": 0, "u_expected": _u_expected(z, k_days),
                 "is_exact_day": exact,
                 "control": "H-exact-day" if exact else None,
                 "arith_load": "moderate", "role": "compute"},
                DAYS[z], (z + h // HOURS_PER_DAY) % N_DAYS if exact else NO_ANSWER_DAY))
    return out


# ------------------------------------------------------- clock-time statements (T)
# CAPTURE AT THE SENTENCE-FINAL FULL STOP. Two capture sites had to be ruled out first:
#
#   "one pm on Monday"  ends on the WEEKDAY token. RESULTS.md §7 measured that site as
#                       essentially day-token identity (decodes at 1.00 from layer 0,
#                       before any computation), so any continuity there would belong to
#                       the embedding, not the concept.
#   "Monday at one pm"  ends on "am"/"pm". That is worse than it looks: hour < 12 iff the
#                       token is "am", so a BINARY token difference is perfectly
#                       confounded with the continuous variable. Measured at L11 this
#                       produced an apparent r=+0.46 that was entirely an am/pm jump of
#                       1.38 days — within-half correlations were NEGATIVE.
#
# "It was one pm on Monday." ends on "." for every one of the 168 prompts, so the surface
# at the capture site is IDENTICAL across the whole sweep and neither confound can enter.
# The sentence-final period is also the repo's established summary site (days.py
# TRAILING_TEMPLATES, "the sentence-summary position that LMs tend to gather clause
# meaning on"). Past tense keeps the sentence natural without a deictic "now".
TIME_TEMPLATES: List[str] = ["It was {t} on {z}."]


def clock_word(h24: int) -> str:
    """24-hour index -> 12-hour clock phrase ("one pm"). 0 = midnight, 12 = noon."""
    suffix = "am" if h24 < 12 else "pm"
    h12 = h24 % 12
    return f"{_UNITS[12 if h12 == 0 else h12]} {suffix}"


# Midnight and noon are EXCLUDED. "twelve am" and "twelve pm" reuse a single hour word at
# the two extremes of the am/pm split, where every other word appears once in each half,
# so the word-to-clock-position mapping is not monotone at those two points. 22 hours x 7
# weekdays = 154 prompts.
CLOCK_HOURS: tuple = tuple(h for h in range(24) if h not in (0, 12))


def build_time_statements(hours: Sequence[int] = CLOCK_HOURS,
                          templates: Optional[Sequence[str]] = None) -> List[PromptSpec]:
    """T — plain statements of a time and a weekday. NO question, NO arithmetic."""
    tmpls = list(TIME_TEMPLATES if templates is None else templates)
    out: List[PromptSpec] = []
    for ci, tmpl in enumerate(tmpls):
        for z in range(N_DAYS):
            for h in hours:
                t = clock_word(h)
                text = tmpl.format(z=DAYS[z], t=t)
                out.append(_spec(
                    text, "T",
                    {"z": z, "clock_hour": int(h), "time": t, "k_days": h / 24.0,
                     "rendering": "word", "content_id": ci, "template": tmpl,
                     "day_is_last": False, "capture": "sentence_final_stop",
                     "u_expected": _u_expected(z, h / 24.0),
                     "is_exact_day": False,
                     "control": None,
                     "arith_load": "none", "role": "read", "gate_exempt": True},
                    DAYS[z], z))
    # The capture token must be the full stop for every prompt — that is what makes the
    # surface at the capture site constant across the sweep.
    #
    # CAVEAT worth stating rather than asserting away: the weekday sits immediately
    # before that stop, so the period's activation may inherit day identity from its
    # neighbour. That cannot manufacture the effect being looked for, though — pure
    # inherited day identity is constant across all 24 times of the same day, so it
    # predicts ZERO time sensitivity. It can only dilute a real effect, not create one.
    for sp in out:
        assert sp.text.endswith("."), f"capture site must be the full stop: {sp.text!r}"
        assert not sp.text[:-1].rstrip().endswith(("am", "pm")), (
            f"am/pm token adjacent to the capture site: {sp.text!r}")
    return out


# M -- plain mentions of a weekday in a sentence that ASKS NOTHING.
#
# Chosen so every prompt tokenises to the SAME length (8 with BOS), the weekday is a single
# token at a fixed index, and the sentence-final "." is always the last token. That makes the
# capture site positionally identical across templates, so pooling their centroids introduces
# no positional jitter -- the thing that would otherwise smear a Jacobian measured at a site
# whose index moves from template to template.
MENTION_STATEMENT_TEMPLATES: List[str] = [
    "I go to market on {z}.",
    "She flies to Berlin on {z}.",
    "The library is closed on {z}.",
    "He calls his mother on {z}.",
    "We paint the fence on {z}.",
    "The train leaves early on {z}.",
]


# ME -- the day named EARLY, then content, then the capture site. Every template is 9
# tokens including BOS, with the weekday at index 2, so the capture position is fixed and
# the day has to survive five intervening content tokens to be visible at all. That is the
# difference from M, where the day is the last word before the stop.
MENTION_EARLY_TEMPLATES: List[str] = [
    # Third person or impersonal throughout: a first-person subject makes the clause about
    # the speaker, and "I"/"we"/"my" carry their own strong positional statistics.
    # Day-neutral content only. Anything a corpus associates with a particular day --
    # markets and fixtures with weekends, starting a job with Monday -- would let the day
    # reach the capture site through the CONTENT rather than through the weekday
    # representation, which is the one thing this family exists to isolate.
    "On {z}, she missed her train.",
    "On {z}, they painted the fence.",
    "On {z}, he sold his bike.",
    "On {z}, the pipes froze solid.",
    "On {z}, rain flooded the road.",
    "On {z}, the baby slept well.",
    "On {z}, he burned the toast.",
    "On {z}, they closed the bridge.",
    "On {z}, the lamp stopped working.",
    "On {z}, the door stuck again.",
    "On {z}, the kettle boiled over.",
    "On {z}, she lost her keys.",
    "On {z}, he missed the bus.",
    "On {z}, they fixed the roof.",
    "On {z}, the letter finally arrived.",
    "On {z}, he dropped his phone.",
    "On {z}, the heater broke down.",
    "On {z}, she cut her finger.",
    "On {z}, the tap kept dripping.",
    "On {z}, they moved the piano.",
]

# What gets appended before the capture token, and what that token is.
MENTION_EARLY_SUFFIX = {"stop": "", "thisday": " On this day"}
MENTION_EARLY_POS_DAY = 2


def build_mention_early(variant: str = "stop",
                        templates: Optional[Sequence[str]] = None) -> List[PromptSpec]:
    """ME -- a weekday named early in a plain sentence, captured after the content."""
    if variant not in MENTION_EARLY_SUFFIX:
        raise ValueError(f"variant must be one of {sorted(MENTION_EARLY_SUFFIX)}")
    suffix = MENTION_EARLY_SUFFIX[variant]
    tmpls = list(MENTION_EARLY_TEMPLATES if templates is None else templates)
    out: List[PromptSpec] = []
    for ci, tmpl in enumerate(tmpls):
        for z in range(N_DAYS):
            out.append(_spec(
                tmpl.format(z=DAYS[z]) + suffix, "ME",
                {"z": z, "rendering": "word", "content_id": ci, "template": tmpl,
                 "day_is_last": False, "capture": f"final_token::{variant}",
                 "variant": variant, "pos_day": MENTION_EARLY_POS_DAY,
                 "u_expected": float(z) / N_DAYS,
                 "is_exact_day": True, "control": None,
                 "arith_load": "none", "role": "mention", "gate_exempt": True},
                DAYS[z], z))
    return out


def build_mention_statements(
        templates: Optional[Sequence[str]] = None) -> List[PromptSpec]:
    """M -- a weekday stated in passing, captured at the sentence-final full stop."""
    tmpls = list(MENTION_STATEMENT_TEMPLATES if templates is None else templates)
    out: List[PromptSpec] = []
    for ci, tmpl in enumerate(tmpls):
        for z in range(N_DAYS):
            text = tmpl.format(z=DAYS[z])
            out.append(_spec(
                text, "M",
                {"z": z, "rendering": "word", "content_id": ci, "template": tmpl,
                 "day_is_last": False, "capture": "sentence_final_stop",
                 "u_expected": float(z) / N_DAYS,
                 "is_exact_day": True, "control": None,
                 "arith_load": "none", "role": "mention", "gate_exempt": True},
                DAYS[z], z))
    for sp in out:
        assert sp.text.endswith("."), f"capture site must be the full stop: {sp.text!r}"
    return out


PROBE_BUILDERS = {
    "H": build_hour_sweep,
    "T": build_time_statements,
    "M": build_mention_statements,
    "ME": build_mention_early,
    "P1": build_fractional_probes,
    "P2": build_hours_probes,
    "P3": build_typo_probes,
    "P4": build_deictic_probes,
    "C-int": build_integer_controls,
}

# Families where nothing is predicted, so the P(weekday) competence gate is inapplicable.
# Empty by design: every family is a question in the same frame, at the same capture
# site, so the same gate applies to all of them and the results stay comparable.
# T is a STATEMENT, not a question: nothing is predicted at the capture site, so a
# next-token P(weekday) criterion is meaningless for it (measured 0.001, which says only
# that "Monday at one pm" is not followed by a weekday — as intended). Its validity check
# is instead that midnight lands on its day's knot, tested in Phase A.
GATE_EXEMPT_FAMILIES: tuple = ("T", "M", "ME")

# Controls first (they gate everything), then the probe families. The original ordering
# was by predicted arithmetic load; Phase 0 refuted that prediction — the HIGH-load
# families (P1 fractional, P2 hours) clear the gate comfortably in word rendering, while
# the zero-arithmetic ones (P3, P4) struggle for reasons of answer POSITION, not
# competence. Order is now controls-then-probes and the load ranking is recorded per
# spec in ``meta["arith_load"]`` rather than baked into the sequence.
# The dense hour sweep is the primary probe: one axis, one framing, 1-hour resolution.
# The four earlier families sampled the offset axis sparsely across four different
# framings, which is what a regression of position-on-requested-offset cannot use.
PROBE_ORDER = ("C-int", "H", "T", "M")
# Retained and runnable via --families, but not in the default set.
LEGACY_FAMILIES = ("P1", "P2", "P3", "P4")


def build_all_probes(families: Optional[Sequence[str]] = None) -> List[PromptSpec]:
    """Every probe + control, in ascending arithmetic-load order."""
    names = list(PROBE_ORDER if families is None else families)
    unknown = set(names) - set(PROBE_BUILDERS)
    if unknown:
        raise ValueError(f"unknown probe families: {sorted(unknown)}; "
                         f"choose from {sorted(PROBE_BUILDERS)}.")
    out: List[PromptSpec] = []
    seen = set()
    for name in names:
        for sp in PROBE_BUILDERS[name]():
            if sp.text in seen:
                continue
            seen.add(sp.text)
            out.append(sp)
    return out
