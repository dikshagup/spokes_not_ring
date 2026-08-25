"""The seven weekdays, their tokenisation, and the prompt formulations built on them."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

# PromptSpec lives in ``concept`` now (concept-neutral); re-exported here so
# existing ``from weekday_manifold.manifold.days import PromptSpec`` imports keep working.
from weekday_manifold.manifold.concept import Concept, PromptSpec

# Weekday order is the ground truth for the cyclic task: index 0 = Monday.
DAYS: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
N_DAYS = len(DAYS)

# Offset words for "k days after": digit vs spelled-out (tested in Step 0).
DIGIT_OFFSETS: Dict[int, str] = {k: str(k) for k in range(0, 8)}
WORD_OFFSETS: Dict[int, str] = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
}


def day_index(name: str) -> int:
    """Index of a weekday name (case-insensitive); raises if unknown."""
    key = name.strip().capitalize()
    if key not in DAYS:
        raise ValueError(f"unknown weekday {name!r}; expected one of {DAYS}.")
    return DAYS.index(key)


def add_days(start: int, k: int) -> int:
    """Weekday index ``k`` days after ``start`` (cyclic, mod 7)."""
    return (start + k) % N_DAYS


def offset_word(k: int, style: str) -> str:
    """Render an offset ``k`` as a digit ("3") or spelled-out word ("three")."""
    table = DIGIT_OFFSETS if style == "digit" else WORD_OFFSETS
    if style not in ("digit", "word"):
        raise ValueError(f"offset style must be 'digit' or 'word', got {style!r}.")
    if k not in table:
        raise ValueError(f"no offset rendering for k={k} in style {style!r}.")
    return table[k]


# --------------------------------------------------------------------- tokens
def tokenize_day(
    tokenize_fn: Callable[[str], Sequence[int]],
    day: str,
    leading_space: bool = True,
) -> List[int]:
    """Token-id list for a weekday, with a leading space by default."""
    text = (" " + day) if leading_space else day
    ids = list(tokenize_fn(text))
    if not ids:
        raise ValueError(f"tokenizer returned no tokens for {text!r}.")
    return ids


def day_token_table(
    tokenize_fn: Callable[[str], Sequence[int]],
    leading_space: bool = True,
) -> Dict[str, List[int]]:
    """``{day: [token ids]}`` for all seven weekdays — printed/asserted in Step 0."""
    return {day: tokenize_day(tokenize_fn, day, leading_space) for day in DAYS}


def tokenization_report(table: Dict[str, List[int]]) -> str:
    """Human-readable summary of how each weekday tokenizes (pure)."""
    lines = ["weekday tokenization (leading space, no BOS):"]
    for day, ids in table.items():
        kind = "single-token" if len(ids) == 1 else f"{len(ids)}-token"
        lines.append(f"  {day:<9} -> {ids}  ({kind})")
    n_multi = sum(1 for ids in table.values() if len(ids) > 1)
    lines.append(
        f"  => {n_multi}/{len(table)} weekdays are multi-token; "
        "scorer teacher-forces the FULL day string."
    )
    return "\n".join(lines)


# --- formulation (a): arithmetic cloze -----------------------------------
ARITH_TEMPLATES: List[str] = [
    "The day {k} days after {z} is",
    "{k} days after {z} is",
    "If today is {z}, then {k} days later it is",
    "Starting from {z} and counting forward {k} days, you reach",
    "Counting {k} days ahead from {z}, you land on",
]


def build_arithmetic_prompts(
    offset_style: str = "digit",
    ks: Sequence[int] = (1, 2, 3, 4, 5, 6),
    templates: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """Declarative arithmetic-cloze prompts: "The day {k} days after {z} is"."""
    templates = list(templates) if templates is not None else ARITH_TEMPLATES
    out: List[PromptSpec] = []
    for tmpl in templates:
        for z in range(N_DAYS):
            for k in ks:
                text = tmpl.format(k=offset_word(k, offset_style), z=DAYS[z])
                out.append(
                    PromptSpec(
                        text=text,
                        answer_day=add_days(z, k),
                        capture_text=text,
                        formulation="arith",
                        meta={"z": z, "k": k, "offset_style": offset_style,
                              "template": tmpl},
                    )
                )
    return out


# --- formulation (a'): interrogative arithmetic (PAPER TEMPLATE) -----------
# The papers' own phrasing — "What day is k days after z?" (Manifold Steering /
# "Arithmetic in the Wild", arXiv:2605.05115 / 2605.01148) — as an explicit
# QUESTION ending on the answer slot (capture site = final token). On base
# Llama-3.1-8B this reads ~0.96 next-token accuracy zero-shot vs ~0.35-0.54 for
# the declarative cloze (a), i.e. it measures the model's real competence, not
# the prompt's. Like (a) it is `scored` (day is COMPUTED, not given).
#
# We keep the paper template ALONE by default for faithful replication (7 days x
# 7 offsets = 49 prompts, matching Goodfire's enumerate_all; PCA auto-caps to
# n_prompts-1, ample for a ~1-D ring).
# The `Q:/A:` frame is the minimal scaffold a base model needs to place the
# answer immediately after the stem; it does not change the arithmetic asked.
# Extra paraphrases can be passed via `templates=` when more prompts/day help.
# EXACT Goodfire causalab weekday template (natural_domains_arithmetic config.py):
#   "Q: What day is {number} days after {entity}?\nA:"  with WORD numbers one..seven.
# Newline before "A:" (not a space) and word-form offsets are part of the paper spec.
INTERROGATIVE_TEMPLATES: List[str] = [
    "Q: What day is {k} days after {z}?\nA:",
]


def build_interrogative_prompts(
    offset_style: str = "word",
    ks: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    templates: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """Interrogative-arithmetic prompts: "Q: What day is {k} days after {z}? A:"."""
    templates = list(templates) if templates is not None else INTERROGATIVE_TEMPLATES
    out: List[PromptSpec] = []
    for tmpl in templates:
        for z in range(N_DAYS):
            for k in ks:
                text = tmpl.format(k=offset_word(k, offset_style), z=DAYS[z])
                out.append(
                    PromptSpec(
                        text=text,
                        answer_day=add_days(z, k),
                        capture_text=text,
                        formulation="interrogative",
                        meta={"z": z, "k": k, "offset_style": offset_style,
                              "template": tmpl},
                    )
                )
    return out


# --- formulation (b): sequence completion + single-step relational --------
# Run-lengths (how many consecutive days to list before the answer) and surface
# separators. Defaults give 5 lengths x 7 start-days x 2 separators = 70 prompts
# (~10 per answer-day) — enough for a paper-faithful 64D PCA fit (needs >65
# prompts). The competence script can pass a smaller set for a quick read.
SEQ_RUN_LENGTHS = (2, 3, 4, 5, 6)
SEQ_SEPARATORS = ("plain", "comma")
_SEP_STR = {"plain": " ", "comma": ", "}


def build_sequence_prompts(
    run_lengths: Sequence[int] = SEQ_RUN_LENGTHS,
    separators: Sequence[str] = SEQ_SEPARATORS,
) -> List[PromptSpec]:
    """Sequence-completion cloze: "Monday Tuesday Wednesday" -> next "Thursday"."""
    out: List[PromptSpec] = []
    for sep_name in separators:
        if sep_name not in _SEP_STR:
            raise ValueError(f"unknown separator {sep_name!r}; "
                             f"choose from {sorted(_SEP_STR)}.")
        sep = _SEP_STR[sep_name]
        for L in run_lengths:
            for z in range(N_DAYS):
                seq = [DAYS[add_days(z, i)] for i in range(L)]
                text = sep.join(seq)
                out.append(
                    PromptSpec(
                        text=text,
                        answer_day=add_days(z, L),
                        capture_text=text,
                        formulation="seq",
                        meta={"z": z, "run_length": L, "separator": sep_name},
                    )
                )
    return out


RELATIONAL_TEMPLATES: List[str] = [
    "The day after {z} is",
    "The day that comes right after {z} is",
    "After {z} comes",
]


def build_relational_prompts(
    templates: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """Single-step relational cloze: "The day after {z} is" -> "(z+1)"."""
    templates = list(templates) if templates is not None else RELATIONAL_TEMPLATES
    out: List[PromptSpec] = []
    for tmpl in templates:
        for z in range(N_DAYS):
            text = tmpl.format(z=DAYS[z])
            out.append(
                PromptSpec(
                    text=text,
                    answer_day=add_days(z, 1),
                    capture_text=text,
                    formulation="relational",
                    meta={"z": z, "template": tmpl},
                )
            )
    return out


# --- formulation (c): pure mention (representation-only fallback) ----------
# The weekday is the FINAL token of every template (capture site = the day mention).
#
# FIXED-LENGTH INVARIANT (holds for GPT-2 BPE and Llama-3.1, verified for all 7 days;
# see scratch verify_mention_tokens.py / tests/test_manifold_days.py): every
# rendered prompt is EXACTLY 6 tokens and 6 whitespace words, so the weekday
# token sits at the SAME absolute position for every (template, day). This removes
# positional-encoding drift as a confounder and DENOISES the per-day centroids
# (mirrors the TRAILING_TEMPLATES invariant). All weekdays are single leading-space
# tokens under both tokenizers, so swapping {z} never changes the length; no OTHER weekday name may
# appear in a template. Adding one: keep it 6 words ending in " {z}" and re-verify.
MENTION_TEMPLATES: List[str] = [
    "He left the city on {z}",
    "I saw them again last {z}",
    "She was born on a {z}",
    "They danced all night on {z}",
    "We flew back home last {z}",
    "I will call you next {z}",
    "They met up again last {z}",
    "We had dinner together last {z}",
    "The store was closed last {z}",
    "I woke up early last {z}",
    "She sent the email last {z}",
    "The children returned home last {z}",
    "I finished the book last {z}",
]


assert all(len(t.split()) == 6 and t.endswith(" {z}") for t in MENTION_TEMPLATES), (
    "MENTION_TEMPLATES must stay 6 whitespace words ending in ' {z}' so the day token "
    "sits at a fixed position; the token-count half of the invariant is tokenizer-"
    "specific and is checked in tests/test_manifold_days.py")


def build_mention_prompts(
    templates: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """Pure-mention prompts where the day is GIVEN, captured AT the day token."""
    templates = list(templates) if templates is not None else MENTION_TEMPLATES
    out: List[PromptSpec] = []
    for tmpl in templates:
        for z in range(N_DAYS):
            before, after = tmpl.split("{z}")
            capture_text = before + DAYS[z]            # ends AT the day mention
            text = capture_text + after
            out.append(
                PromptSpec(
                    text=text,
                    answer_day=z,
                    capture_text=capture_text,
                    formulation="mention",
                    meta={"z": z, "template": tmpl},
                )
            )
    return out


# --- formulation (d): trailing — day given, capture at sentence-final period ---
# Sentence templates where the weekday appears ONCE at the front and the sentence
# ends in a lone "." token. Capture site is the final "." (LMs tend to summarise
# each sentence on the concluding punctuation).
#
# HARD INVARIANTS (hold for GPT-2 BPE and Llama-3.1, verified for all 7 days):
#   * Every rendered prompt is EXACTLY 8 tokens long, so the "." token sits
#     at the SAME absolute position for every (template, day) pair. This removes
#     positional-encoding drift as a confounder in the per-day centroids and
#     the ring geometry.
#   * The final decoded token is a standalone ".".
#   * All seven weekdays are single tokens with a leading space, so
#     rendering ``{z}`` never changes the length.
#   * No OTHER weekday name may appear anywhere in the template.
#
# Token layout (8 tokens): [ On | " {day}" | , | X | Y | Z | W | . ]
#                             1      2      3  4  5  6  7  8
#
# Adding a new template: pick a subject + past-tense verb + 2-word object with
# common single-token BPE pieces, then re-run tests/test_manifold_days.py
# (structural check) AND verify token count with the real GPT-2 tokenizer.
TRAILING_TEMPLATES: List[str] = [
    "On {z}, she baked a cake.",
    "On {z}, he read a book.",
    "On {z}, I called my mother.",
    "On {z}, they cleaned the house.",
    "On {z}, they visited the museum.",
    "On {z}, I attended a meeting.",
    "On {z}, we cooked some pasta.",
    "On {z}, she wrote a letter.",
    "On {z}, he bought a newspaper.",
    "On {z}, we planted some flowers.",
]


def build_trailing_prompts(
    templates: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """Full-sentence prompts with the day fronted; capture at the trailing period."""
    templates = list(templates) if templates is not None else TRAILING_TEMPLATES
    out: List[PromptSpec] = []
    for tmpl in templates:
        for z in range(N_DAYS):
            text = tmpl.format(z=DAYS[z])
            out.append(
                PromptSpec(
                    text=text,
                    answer_day=z,
                    capture_text=text,          # capture at final token = "."
                    formulation="trailing",
                    meta={"z": z, "template": tmpl},
                )
            )
    return out


# Registry so config/experiments can select a formulation by name.
FORMULATION_BUILDERS: Dict[str, Callable[..., List[PromptSpec]]] = {
    "arith": build_arithmetic_prompts,
    "interrogative": build_interrogative_prompts,
    "seq": build_sequence_prompts,
    "relational": build_relational_prompts,
    "mention": build_mention_prompts,
    "trailing": build_trailing_prompts,
}


def build_prompts(formulation: str, **kwargs) -> List[PromptSpec]:
    """Build the prompt set for a named formulation (see ``FORMULATION_BUILDERS``)."""
    if formulation not in FORMULATION_BUILDERS:
        raise ValueError(
            f"unknown formulation {formulation!r}; "
            f"choose from {sorted(FORMULATION_BUILDERS)}."
        )
    builder = FORMULATION_BUILDERS[formulation]
    per_formulation = {
        "arith": ("offset_style", "ks", "templates"),
        "interrogative": ("offset_style", "ks", "templates"),
        "seq": ("run_lengths", "separators"),
        "relational": ("templates",),
        "mention": ("templates",),
        "trailing": ("templates",),
    }
    allowed_keys = per_formulation.get(formulation, ())
    allowed = {k: v for k, v in kwargs.items()
               if k in allowed_keys and v is not None}
    return builder(**allowed)


# ------------------------------------------------------------------- concept
def days_token_ids(model) -> Dict[int, List[int]]:
    """``{day_index: [token ids]}`` with a leading space, no BOS (run-time)."""
    def tok(s: str) -> List[int]:
        return model.to_tokens(s, prepend_bos=False)[0].tolist()

    return {i: tokenize_day(tok, DAYS[i], leading_space=True) for i in range(N_DAYS)}


def make_days_concept() -> Concept:
    """The weekdays concept: a 7-label cyclic ring (Mon adjacent to Tue and Sun)."""
    return Concept(
        name="days",
        labels=list(DAYS),
        is_cyclic=True,
        build_prompts=build_prompts,
        token_ids=days_token_ids,
        coordinate=None,   # label order 0..6 IS the ground-truth cyclic order
    )
