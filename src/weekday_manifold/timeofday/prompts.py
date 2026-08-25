"""Stimuli for the time-of-day test: clock hours, coarse word times, and placebo modifiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from weekday_manifold.manifold.days import DAYS, N_DAYS, PromptSpec

# --------------------------------------------------------------------- scale


@dataclass(frozen=True)
class Modifier:
    """One modifier phrase inserted before ``on {day}``."""

    phrase: str
    t: Optional[int]
    family: str
    sets: Sequence[str] = ()

    @property
    def n_words(self) -> int:
        return len(self.phrase.split())

    @property
    def name(self) -> str:
        return self.phrase.replace(" ", "_") if self.phrase else "neutral"


# The ordinal scale of plan section 2.1, plus the synonyms it asks to "also run",
# plus length-matched placebos (section 2.3). Order here is the canonical order.
MODIFIERS: List[Modifier] = [
    # --- graded time scale ------------------------------------------------
    Modifier("very early", -2, "time", ("graded", "fig22")),
    Modifier("morning", -1, "time", ("graded", "fig9")),
    Modifier("", 0, "neutral", ("graded", "baseline")),
    Modifier("evening", +1, "time", ("graded", "fig9")),
    Modifier("very late", +2, "time", ("graded", "fig22")),
    # --- synonym robustness (same ordinal levels, different surface form) --
    Modifier("early", -1, "time", ("alt",)),
    Modifier("midday", 0, "time", ("alt",)),
    Modifier("noon", 0, "time", ("alt",)),
    Modifier("late", +1, "time", ("alt",)),
    # --- placebos: 1 word (match morning/evening/early/late/midday/noon) --
    Modifier("quietly", None, "placebo", ("placebo1",)),
    Modifier("secretly", None, "placebo", ("placebo1",)),
    Modifier("supposedly", None, "placebo", ("placebo1",)),
    Modifier("reportedly", None, "placebo", ("placebo1",)),
    # --- placebos: 2 words (match very early / very late) ------------------
    Modifier("very quietly", None, "placebo", ("placebo2",)),
    Modifier("very secretly", None, "placebo", ("placebo2",)),
]

# The 5-point graded scale, in ordinal order, as (t, phrase).
GRADED_LEVELS: List[int] = [-2, -1, 0, +1, +2]


def modifier_bank(sets: Optional[Sequence[str]] = None,
                  families: Optional[Sequence[str]] = None) -> List[Modifier]:
    """Modifiers filtered by analysis set and/or family (order preserved)."""
    out = list(MODIFIERS)
    if sets is not None:
        want = set(sets)
        out = [m for m in out if want & set(m.sets)]
    if families is not None:
        want_f = set(families)
        out = [m for m in out if m.family in want_f]
    return out


# ------------------------------------------------------------------ carriers
# 32 sentence prefixes (plan section 2.2 asks for >=30). Each renders as
# ``<carrier> <modifier> on <Day>`` with the weekday last. Invariants:
#   * no carrier contains a weekday name (would contaminate the day signal),
#   * no carrier contains a time word (would interact with the manipulation),
#   * each is a complete clause so the continuation is natural English.
CARRIERS: List[str] = [
    "It happened",
    "The meeting is",
    "I'll see you",
    "She left",
    "We arrived",
    "The delivery came",
    "They departed",
    "The concert starts",
    "He called",
    "The train leaves",
    "My flight lands",
    "The shop opens",
    "The results came out",
    "We finished",
    "The package arrived",
    "She started work",
    "The class begins",
    "They announced it",
    "The storm hit",
    "I woke up",
    "The game ended",
    "He arrived home",
    "The office closes",
    "We spoke",
    "The letter came",
    "She phoned",
    "The bus departs",
    "It was decided",
    "The market opens",
    "They met",
    "The film premiered",
    "I sent it",
]

_TIME_WORDS = {
    "early", "late", "morning", "evening", "midday", "noon", "night",
    "afternoon", "dawn", "dusk", "hour", "time",
}


def validate_carriers(carriers: Sequence[str] = tuple(CARRIERS)) -> None:
    """Fail loudly if a carrier smuggles in a weekday or a time word."""
    lowered_days = {d.lower() for d in DAYS}
    for c in carriers:
        words = {w.strip(".,'").lower() for w in c.split()}
        bad_day = words & lowered_days
        bad_time = words & _TIME_WORDS
        if bad_day:
            raise ValueError(f"carrier {c!r} contains a weekday: {sorted(bad_day)}")
        if bad_time:
            raise ValueError(f"carrier {c!r} contains a time word: {sorted(bad_time)}")


# ------------------------------------------------------------------- render
def render(carrier: str, modifier: str, day: str) -> str:
    """``<carrier> <modifier> on <Day>``; the modifier is omitted when empty."""
    tail = f"{modifier} on {day}" if modifier else f"on {day}"
    return f"{carrier} {tail}" if carrier else tail


def build_timeofday_prompts(
    modifiers: Optional[Sequence[Modifier]] = None,
    carriers: Optional[Sequence[str]] = None,
    regime: str = "diverse",
) -> List[PromptSpec]:
    """The (day x modifier x carrier) prompt grid."""
    mods = list(modifiers) if modifiers is not None else list(MODIFIERS)
    if regime == "engels_bare":
        carrier_list: List[str] = [""]
    elif regime == "diverse":
        carrier_list = list(carriers) if carriers is not None else list(CARRIERS)
        validate_carriers(carrier_list)
    else:
        raise ValueError(f"unknown regime {regime!r}; use 'diverse' or 'engels_bare'.")

    out: List[PromptSpec] = []
    for ci, carrier in enumerate(carrier_list):
        for m in mods:
            for z in range(N_DAYS):
                text = render(carrier, m.phrase, DAYS[z])
                # capture_text ends AT the weekday -> last day sub-token.
                capture_text = text
                out.append(
                    PromptSpec(
                        text=text,
                        answer_day=z,
                        capture_text=capture_text,
                        formulation="timeofday",
                        meta={
                            "z": z,
                            "day": DAYS[z],
                            "carrier": carrier,
                            "carrier_idx": ci,
                            "modifier": m.phrase,
                            "modifier_name": m.name,
                            "t": m.t,
                            "family": m.family,
                            "sets": list(m.sets),
                            "n_words": m.n_words,
                            "regime": regime,
                            "sites": {"last_token": text, "day_token": capture_text},
                        },
                    )
                )
    return out


# --------------------------------------------------------------- audits
def annotate_token_lengths(specs: Sequence[PromptSpec], model,
                           prepend_bos: bool = True) -> Sequence[PromptSpec]:
    """Record ``meta["n_tokens"]`` and ``meta["day_pos"]`` using the real tokenizer."""
    from weekday_manifold.manifold.capture import resolve_capture_position

    for s in specs:
        s.meta["n_tokens"] = int(model.to_tokens(s.text, prepend_bos=prepend_bos).shape[1])
        s.meta["day_pos"] = int(resolve_capture_position(model, s, prepend_bos))
    return specs


def day_tokenization_audit(model, prepend_bos: bool = False) -> Dict[str, object]:
    """Plan section 1: confirm every weekday's token split, fail loudly if surprising."""
    def tok(s: str) -> List[int]:
        return model.to_tokens(s, prepend_bos=prepend_bos)[0].tolist()

    table = {d: tok(" " + d) for d in DAYS}
    n_multi = sum(1 for ids in table.values() if len(ids) > 1)
    lines = ["weekday tokenization (leading space, no BOS):"]
    for d, ids in table.items():
        lines.append(f"  {d:<9} -> {ids}  ({'single' if len(ids) == 1 else str(len(ids)) + '-token'})")
    lines.append(f"  => {n_multi}/{len(table)} weekdays are multi-token.")
    return {"table": table, "n_multi": n_multi, "report": "\n".join(lines)}


def position_audit(specs: Sequence[PromptSpec]) -> Dict[str, object]:
    """Check that day position is constant within (carrier, n_words) groups."""
    groups: Dict[tuple, List[int]] = {}
    for s in specs:
        if "day_pos" not in s.meta:
            raise ValueError("run annotate_token_lengths() before position_audit().")
        # ``regime`` MUST be part of the key: the bare-Engels regime reuses
        # carrier_idx 0, so omitting it compares "morning on Monday" against
        # "It happened morning on Monday" and reports a spurious spread.
        key = (s.meta["regime"], s.meta["carrier_idx"], s.meta["n_words"], s.meta["z"])
        groups.setdefault(key, []).append(int(s.meta["day_pos"]))
    spreads = {k: (max(v) - min(v)) for k, v in groups.items()}
    bad = {k: v for k, v in spreads.items() if v != 0}
    # Per-regime, because only the DIVERSE regime carries the statistics. In the
    # bare-Engels regime the modifier is sentence-initial and so has no leading
    # space, and GPT-2 BPE splits several of those ("evening", "midday" -> 2
    # tokens; "supposedly" -> 3) where the space-prefixed forms are single
    # tokens. That regime exists only to reproduce Engels' Fig. 9 picture, so a
    # spread there is a documented caveat, not a broken control.
    per_regime: Dict[str, Dict[str, object]] = {}
    for key in groups:
        reg = key[0]
        d = per_regime.setdefault(reg, {"n_groups": 0, "n_with_spread": 0, "max_spread": 0})
        d["n_groups"] += 1
        if spreads[key]:
            d["n_with_spread"] += 1
            d["max_spread"] = max(d["max_spread"], spreads[key])
    for d in per_regime.values():
        d["ok"] = d["n_with_spread"] == 0
    # Name the OFFENDING modifiers rather than just counting bad groups: a
    # modifier whose token count differs from its word count breaks the
    # length-matching, and which one it is decides whether the headline metric is
    # affected at all. (On Llama-3.1 " midday" is one word but two tokens,
    # " mid"+"day"; it is an `alt` modifier outside the graded set, so the graded
    # headline is untouched and only `time_all` sees it.)
    neutral_pos: Dict[int, int] = {}
    for s in specs:
        if s.meta["regime"] == "diverse" and s.meta["family"] == "neutral":
            neutral_pos[int(s.meta["carrier_idx"])] = int(s.meta["day_pos"])
    extra: Dict[str, set] = {}
    for s in specs:
        if s.meta["regime"] != "diverse":
            continue
        base = neutral_pos.get(int(s.meta["carrier_idx"]))
        if base is not None:
            extra.setdefault(str(s.meta["modifier_name"]), set()).add(
                int(s.meta["day_pos"]) - base)
    offenders = {
        m: sorted(v) for m, v in extra.items()
        if len(v) > 1 or (v and next(iter(v)) != _word_count_of(specs, m))
    }

    by_nwords: Dict[int, List[int]] = {}
    for s in specs:
        if s.meta["regime"] == "diverse":
            by_nwords.setdefault(int(s.meta["n_words"]), []).append(int(s.meta["day_pos"]))
    graded = {m.name for m in modifier_bank(sets=["graded"])}
    return {
        "n_groups": len(groups),
        "n_groups_with_spread": len(bad),
        "max_spread": max(spreads.values()) if spreads else 0,
        "day_pos_by_n_words_diverse": {k: sorted(set(v)) for k, v in sorted(by_nwords.items())},
        "per_regime": per_regime,
        "ok": not bad,
        "ok_diverse": per_regime.get("diverse", {}).get("ok", True),
        "offending_modifiers": offenders,
        # the only thing that would invalidate the HEADLINE metric
        "offenders_in_graded_set": sorted(set(offenders) & graded),
    }


def _word_count_of(specs: Sequence[PromptSpec], modifier_name: str) -> int:
    for s in specs:
        if str(s.meta["modifier_name"]) == modifier_name:
            return int(s.meta["n_words"])
    return -1


def prompt_set_summary(specs: Sequence[PromptSpec]) -> Dict[str, object]:
    """Counts + a stable hash of the prompt set (plan section 1: cache by prompt hash)."""
    mods = sorted({s.meta["modifier_name"] for s in specs})
    fams: Dict[str, int] = {}
    for s in specs:
        fams[s.meta["family"]] = fams.get(s.meta["family"], 0) + 1
    payload = json.dumps([s.text for s in specs], sort_keys=False).encode()
    return {
        "n_prompts": len(specs),
        "n_days": len({s.meta["z"] for s in specs}),
        "n_carriers": len({s.meta["carrier_idx"] for s in specs}),
        "n_modifiers": len(mods),
        "modifiers": mods,
        "by_family": fams,
        "regimes": sorted({s.meta["regime"] for s in specs}),
        "prompt_hash": hashlib.sha256(payload).hexdigest()[:16],
    }
