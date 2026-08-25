"""A natural weekday prompt library, for pinning down the weekday subspace."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from weekday_manifold.manifold.concept import PromptSpec
from weekday_manifold.manifold.days import DAYS, N_DAYS, add_days, offset_word

POSITION_BIN_EDGES = (0.34, 0.67)          # frac < .34 early, < .67 mid, else late
LENGTH_BIN_EDGES = (10, 22)                # n_tokens <= 10 short, <= 22 med, else long

LIBRARY_AXES: Dict[str, Tuple[str, ...]] = {
    "role": ("compute", "read"),
    "inference": ("deixis", "arithmetic", "stated"),
    "position": ("early", "mid", "late"),
}

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "dozen",
}
_DAY_RE = re.compile(r"\b(" + "|".join(DAYS) + r")\b", re.IGNORECASE)


# ----------------------------------------------------------------- helpers
def _cap(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s


def _weekday_mentions(text: str) -> int:
    return len(_DAY_RE.findall(text))


def _kadj(k: int) -> str:
    """Adjective duration, always singular: 'one-day', 'three-day', 'seven-day'."""
    return f"{offset_word(k, 'word')}-day"


def _kd(k: int) -> str:
    """Noun duration with correct plural: 'one day', 'three days', 'seven days'."""
    w = offset_word(k, "word")
    return f"{w} day" if k == 1 else f"{w} days"


def _summary_prefix(text: str) -> str:
    idxs = [i for i, ch in enumerate(text) if ch in ".?!\n"]
    return text if not idxs else text[: idxs[-1] + 1]


def _mention_prefix(text: str, day_str: str) -> str:
    i = text.find(day_str)
    if i < 0:
        raise ValueError(f"day {day_str!r} not found in {text!r}")
    return text[: i + len(day_str)]


def _sites(text: str, stated_day: Optional[str]) -> Dict[str, str]:
    sites = {"last_token": text, "summary_token": _summary_prefix(text)}
    if stated_day is not None:
        sites["mention_token"] = _mention_prefix(text, stated_day)
    return sites


# -------------------------------------------------------------- template model
@dataclass(frozen=True)
class _Tmpl:
    """One template. ``offset is None`` -> contains a duration (crossed over ``ks``); the
    answer is ``z + k_dir * k``. Otherwise the answer is ``z + offset`` (no k)."""

    text: str
    offset: Optional[int]
    position: str
    k_dir: int = 1


@dataclass(frozen=True)
class _Family:
    fid: str
    role: str
    register: str
    inference: str
    templates: Tuple[_Tmpl, ...]
    stated: bool


# ============================================================ the families ===
def _families() -> List[_Family]:
    fams: List[_Family] = []

    # --- COMPUTE (answer = next token; day latent) --------------------------
    # We CAPTURE broadly (all families/offsets below) so each activation can be
    # attributed to the correct answer vs the model's predicted token vs the context
    # anchor. The PCA/manifold is FIT on the correct subset (condition on
    # model_correct); OFFSET_CORE below marks the 3 forward templates that keep
    # competence past |offset|=1 (C3:0/C3:3/C3:5) — the clean offset-diverse core.

    # C1 deixis — single/two-step "today/tomorrow/yesterday" reasoning (no counting).
    fams.append(_Family("C1", "compute", "prose", "deixis", (
        _Tmpl("today's {z}, so tomorrow's", 1, "early"),
        _Tmpl("yesterday was {z}, so today's", 1, "early"),
        _Tmpl("if today's {z}, the day after tomorrow's", 2, "early"),
        _Tmpl("tomorrow's {z}, which makes today", -1, "early"),
        _Tmpl("today is {z}. tomorrow will be", 1, "early"),
        _Tmpl("today is {z}, so the day before was", -1, "early"),
        _Tmpl("if it's {z} now, then tomorrow it'll be", 1, "early"),
        _Tmpl("assuming today is {z}, yesterday was", -1, "early"),
        _Tmpl("it's {z} today; the day before yesterday was", -2, "early"),
        _Tmpl("it's {z}, so two days from now it'll be", 2, "early"),
        _Tmpl("today being {z}, tomorrow lands on", 1, "early"),
        _Tmpl("it's currently {z}; the previous day was", -1, "early"),
    ), stated=False))

    # C2 naturalistic arithmetic (HARD) — mostly incorrect past k=1; kept so the
    # probe has cases where model_pred_day != answer_day.
    fams.append(_Family("C2", "compute", "prose", "arithmetic", (
        _Tmpl("we leave {z} on a {kadj} trip, so we're back on", None, "early"),
        _Tmpl("I ordered it {z} with {kadj} shipping, so it arrives on", None, "early"),
        _Tmpl("the sale opens {z} and runs {kd}, so it closes on", None, "early"),
        _Tmpl("the show's {z}, and dress rehearsal's {kd} before, on", None, "early", k_dir=-1),
        _Tmpl("rent's due {z}, but I always pay {kd} early, so it goes out on", None, "early", k_dir=-1),
    ), stated=False))

    # C3 direct arithmetic + Q:/A: interrogative, forward & backward, k=1..7.
    # OFFSET_CORE = C3:0, C3:3, C3:5 (forward) stay competent past |offset|=1.
    fams.append(_Family("C3", "compute", "qa", "arithmetic", (
        _Tmpl("Q: What day is {kd} after {z}?\nA:", None, "early"),
        _Tmpl("{kd} after {z} is", None, "early"),
        _Tmpl("the day {kd} after {z} is", None, "early"),
        _Tmpl("counting {kd} forward from {z} lands on", None, "early"),
        _Tmpl("add {kd} to {z} and you get", None, "early"),
        _Tmpl("starting on {z} and moving {kd} ahead brings us to", None, "early"),
        _Tmpl("Q: What day is {kd} before {z}?\nA:", None, "early", k_dir=-1),
        _Tmpl("{kd} before {z} is", None, "early", k_dir=-1),
        _Tmpl("the day {kd} before {z} is", None, "early", k_dir=-1),
        _Tmpl("counting {kd} back from {z} lands on", None, "early", k_dir=-1),
    ), stated=False))

    # C4 relational — single-step neighbours via memorized week order (no counting).
    fams.append(_Family("C4", "compute", "prose", "relational", (
        _Tmpl("the day after {z} is", 1, "early"),
        _Tmpl("the day before {z} is", -1, "early"),
        _Tmpl("after {z} comes", 1, "early"),
        _Tmpl("just before {z} comes", -1, "early"),
        _Tmpl("the day that follows {z} is", 1, "early"),
        _Tmpl("the day right after {z} is", 1, "early"),
        _Tmpl("the day right before {z} is", -1, "early"),
        _Tmpl("the day preceding {z} is", -1, "early"),
        _Tmpl("immediately after {z} we get", 1, "early"),
        _Tmpl("{z} is followed by", 1, "early"),
        _Tmpl("one day on from {z} is", 1, "early"),
        _Tmpl("the day directly before {z} is", -1, "early"),
    ), stated=False))

    # C5 interrogative — single-step Q:/A: cloze (no counting; the model's strong suit).
    fams.append(_Family("C5", "compute", "qa", "relational", (
        _Tmpl("Q: What day comes after {z}?\nA:", 1, "early"),
        _Tmpl("Q: What day comes before {z}?\nA:", -1, "early"),
        _Tmpl("Q: What day follows {z}?\nA:", 1, "early"),
        _Tmpl("Q: {z} is followed by which day?\nA:", 1, "early"),
        _Tmpl("Q: Which day precedes {z}?\nA:", -1, "early"),
        _Tmpl("Q: The day right after {z} is which day?\nA:", 1, "early"),
    ), stated=False))

    # --- READ (day present; exactly one weekday; captured at the last token) --
    # R1 mention — day is the subject; early / mid / late.
    fams.append(_Family("R1", "read", "prose", "stated", (
        _Tmpl("that {z}, the package finally arrived.", 0, "early"),
        _Tmpl("{z} morning, everyone gathered in the hall.", 0, "early"),
        _Tmpl("by {z}, the whole plan had come together.", 0, "early"),
        _Tmpl("we'd agreed the visit would land on {z} without any fuss.", 0, "mid"),
        _Tmpl("everyone assumed the ceremony was set for {z} at noon.", 0, "mid"),
        _Tmpl("we finally met on a {z}", 0, "late"),
        _Tmpl("after all that back-and-forth, we finally met on a {z}", 0, "late"),
        _Tmpl("nobody remembered exactly, but the fair opened on a {z}", 0, "late"),
    ), stated=True))

    # R2 trailing — capture is fixed at the sentence-final ".", but the weekday sits
    # early / mid / late, so this family isolates ANCHOR position at a constant site.
    fams.append(_Family("R2", "read", "prose", "stated", (
        _Tmpl("on {z}, she baked a cake.", 0, "early"),
        _Tmpl("on {z}, the market was crowded.", 0, "early"),
        _Tmpl("on {z}, a storm rolled through.", 0, "early"),
        _Tmpl("she baked a cake on {z} for the school fair.", 0, "mid"),
        _Tmpl("the market was packed on {z}, oddly enough.", 0, "mid"),
        _Tmpl("a storm rolled through on {z} and cut the power.", 0, "mid"),
        _Tmpl("she still talks about the cake she baked that {z}.", 0, "late"),
        _Tmpl("everyone remembers the storm that hit that {z}.", 0, "late"),
        _Tmpl("the whole street lost power that {z}.", 0, "late"),
    ), stated=True))

    # R3 downstream — day stated, clause continues, captured at the end.
    fams.append(_Family("R3", "read", "prose", "stated", (
        _Tmpl("{z} dragged on and the office felt restless.", 0, "early"),
        _Tmpl("{z} dragged on, and by dusk the whole street felt hushed.", 0, "early"),
        _Tmpl("the memo said that on {z} the team would relocate downtown.", 0, "mid"),
        _Tmpl("word got around that by {z} the shop would finally reopen.", 0, "mid"),
        _Tmpl("after much dithering, the crew agreed to gather on {z}", 0, "late"),
        _Tmpl("the shipment finally cleared customs on {z}", 0, "late"),
    ), stated=True))

    # R5 diary / journal.
    fams.append(_Family("R5", "read", "diary", "stated", (
        _Tmpl("{z}: woke up late, missed the bus, lost my keys.", 0, "early"),
        _Tmpl("{z}, and already the whole day feels endless.", 0, "early"),
        _Tmpl("note to self — the interview got moved to {z}, apparently.", 0, "mid"),
        _Tmpl("dear diary, after a long slow week it's finally {z}", 0, "late"),
        _Tmpl("captain's log: we reached the coast at last on a {z}", 0, "late"),
    ), stated=True))

    # R6 list / instruction.
    fams.append(_Family("R6", "read", "instruction", "stated", (
        _Tmpl("{z}: water the plants and email the landlord.", 0, "early"),
        _Tmpl("{z} — pick up the dry cleaning and pay the rent.", 0, "early"),
        _Tmpl("checklist: confirm the venue for {z} and send the invites.", 0, "mid"),
        _Tmpl("reminder: finish the report before {z}", 0, "late"),
        _Tmpl("please make sure the order ships out by {z}", 0, "late"),
    ), stated=True))

    # R7 declarative — recurring facts that happen on days.
    fams.append(_Family("R7", "read", "prose", "stated", (
        _Tmpl("every {z}, the farmers' market fills the square.", 0, "early"),
        _Tmpl("{z} is when the choir rehearses.", 0, "early"),
        _Tmpl("the choir meets on {z} in the church hall.", 0, "mid"),
        _Tmpl("recycling goes out each {z} without fail.", 0, "mid"),
        _Tmpl("the library closes early on {z}", 0, "late"),
        _Tmpl("our team meets every {z}", 0, "late"),
        _Tmpl("fresh bread comes out of the oven on {z}", 0, "late"),
    ), stated=True))

    return fams


# --------------------------------------------------------------- the builder
def build_library(
    ks: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    roles: Sequence[str] = ("compute", "read"),
    families: Optional[Sequence[str]] = None,
) -> List[PromptSpec]:
    """Build the full weekday prompt library as a list of ``PromptSpec``."""
    fams = [f for f in _families()
            if f.role in roles and (families is None or f.fid in families)]
    out: List[PromptSpec] = []
    for fam in fams:
        for tmpl in fam.templates:
            content_id = fam.templates.index(tmpl)
            k_values = ks if tmpl.offset is None else (None,)
            for z in range(N_DAYS):
                for k in k_values:
                    out.append(_make_spec(fam, tmpl, z, k, content_id))
    return out


def _make_spec(fam: _Family, tmpl: _Tmpl, z: int, k: Optional[int],
               content_id: int) -> PromptSpec:
    label = add_days(z, tmpl.k_dir * int(k) if tmpl.offset is None else tmpl.offset)

    kw: Dict[str, str] = {"z": DAYS[z]}
    if "{kadj}" in tmpl.text:
        kw["kadj"] = _kadj(int(k))
    if "{kd}" in tmpl.text:
        kw["kd"] = _kd(int(k))
    if "{k}" in tmpl.text:
        kw["k"] = offset_word(int(k), "word")
    text = _cap(tmpl.text.format(**kw))

    n_mentions = _weekday_mentions(text)
    assert n_mentions == 1, (
        f"prompt must name exactly one weekday, got {n_mentions}: {text!r}")
    # For compute the answer is latent — it must not be written. (Exception: a
    # same-day offset like "this time next week", where the answer equals the
    # written anchor; that's fine.)
    if not fam.stated and label != z:
        assert not re.search(rf"\b{DAYS[label]}\b", text), (
            f"{fam.fid} label day {DAYS[label]!r} appears in a compute prompt: {text!r}")

    stated_day = DAYS[z] if fam.stated else None
    sites = _sites(text, stated_day)

    meta = {
        "role": fam.role,
        "inference": fam.inference,
        "family": fam.fid,
        "template_id": f"{fam.fid}:{content_id}",
        "register": fam.register,
        "z": z,
        "k": (None if k is None else int(k)),
        "offset": (None if tmpl.offset is None else int(tmpl.offset)),
        "k_dir": tmpl.k_dir,
        "content_id": content_id,
        "position_target": tmpl.position,
        "n_weekday_mentions": n_mentions,
        "stated": fam.stated,
        "sites": sites,
        # filled by annotate_token_lengths:
        "n_tokens": None,
        "position_frac": None,
        "position_bin": None,
        "length_bin": None,
        "site_pos": None,
        "site_offset": None,
        "model_correct": None,
        "model_pred_day": None,
        "model_pred_token": None,
        "model_top_is_weekday": None,
        "split": None,
    }
    return PromptSpec(text=text, answer_day=label, capture_text=text,
                      formulation=fam.fid, meta=meta)


# --------------------------------------------------------------- iid split
def assign_splits(specs: Sequence[PromptSpec], iid_frac: float = 0.2,
                  seed: int = 0) -> List[PromptSpec]:
    """Hold out whole TEMPLATES as the test set — a real "unseen phrasing" held-out."""
    import hashlib

    by_family: Dict[str, Dict[str, List[PromptSpec]]] = {}
    for s in specs:
        by_family.setdefault(s.meta["family"], {}).setdefault(
            s.meta["template_id"], []).append(s)

    for fam, templates in by_family.items():
        def rank(tid: str) -> float:
            h = hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest()
            return int(h[:8], 16) / 0xFFFFFFFF
        ordered = sorted(templates, key=rank)
        n_test = int(round(iid_frac * len(ordered)))
        n_test = min(n_test, len(ordered) - 1)      # keep >=1 template in train
        n_test = max(n_test, 1) if len(ordered) >= 2 else 0
        for i, tid in enumerate(ordered):
            split = "test:iid" if i < n_test else "train"
            for sp in templates[tid]:
                sp.meta["split"] = split
    return specs


# --------------------------------------------- off-concept distractor sets
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]

_RANDOM_LINES = [
    "The kettle whistled while the cat watched the rain.",
    "A quiet library smells faintly of old paper and dust.",
    "He tightened the last bolt and wiped his hands clean.",
    "The soup needed more salt but nobody wanted to say so.",
    "Bright kites tugged hard against the steady sea wind.",
    "She sketched the bridge from memory on a napkin.",
    "The old radio hummed a tune from another decade.",
    "Fresh bread cooled on the sill by the open window.",
]


def build_distractors(
    kinds: Sequence[str] = ("months", "random", "weekday_free"),
) -> Dict[str, List[PromptSpec]]:
    """Off-concept prompts for the exclusivity tests (never used to fit)."""
    out: Dict[str, List[PromptSpec]] = {}
    if "months" in kinds:
        specs: List[PromptSpec] = []
        templates = ["We met in {m}", "On {m}, she baked a cake.", "It happened back in {m}"]
        for ti, tmpl in enumerate(templates):
            for mi, month in enumerate(_MONTHS):
                text = _cap(tmpl.format(m=month))
                specs.append(PromptSpec(
                    text=text, answer_day=mi, capture_text=text, formulation="month",
                    meta={"concept": "months", "role": "read", "content_id": ti,
                          "sites": _sites(text, month)}))
        out["months"] = specs
    if "random" in kinds:
        specs = []
        for i, line in enumerate(_RANDOM_LINES):
            specs.append(PromptSpec(
                text=line, answer_day=-1, capture_text=line, formulation="random",
                meta={"concept": "random", "role": "none", "content_id": i,
                      "sites": _sites(line, None)}))
        out["random"] = specs
    if "weekday_free" in kinds:
        specs = []
        # (a) Q:/A: cloze — same interrogative frame as C3/C5, NO weekday; the model
        #     answers with a non-weekday token (number / letter / month). The last
        #     "day is ... after the wedding" keeps the day-arithmetic FRAME with no anchor.
        qa = [
            "Q: What number comes after seven?\nA:",
            "Q: What number comes before nine?\nA:",
            "Q: What letter comes after B?\nA:",
            "Q: What letter comes before M?\nA:",
            "Q: What month comes after March?\nA:",
            "Q: What month comes before August?\nA:",
            "Q: What day is three days after the wedding?\nA:",
            "Q: What comes after the number five?\nA:",
        ]
        for ci, tmpl in enumerate(qa):
            text = _cap(tmpl)
            specs.append(PromptSpec(
                text=text, answer_day=-1, capture_text=text, formulation="wkfree_qa",
                meta={"concept": "weekday_free", "control_kind": "qa", "role": "compute",
                      "register": "qa", "family": "X_qa", "template_id": f"X_qa:{ci}",
                      "k": None, "z": -1, "position_target": "early", "content_id": ci,
                      "stated": False, "sites": _sites(text, None)}))
        # (b) mention / read surface (same shape as R1/R2/R3/R7) but the weekday slot
        #     holds a NON-weekday time/event noun; the mention site sits on that noun.
        mentions = [
            ("that morning, the package finally arrived.", "morning", "early"),
            ("on holiday, she baked a cake.", "holiday", "early"),
            ("the festival dragged on and the office felt restless.", "festival", "early"),
            ("every summer, the farmers' market fills the square.", "summer", "early"),
            ("we finally met on a rainy afternoon", "afternoon", "late"),
            ("the whole street lost power that night.", "night", "late"),
        ]
        for ci, (tmpl, noun, pos) in enumerate(mentions):
            text = _cap(tmpl)
            specs.append(PromptSpec(
                text=text, answer_day=-1, capture_text=text, formulation="wkfree_mention",
                meta={"concept": "weekday_free", "control_kind": "mention", "role": "read",
                      "register": "prose", "family": "X_read", "template_id": f"X_read:{ci}",
                      "k": None, "z": -1, "position_target": pos, "content_id": ci,
                      "stated": True, "sites": _sites(text, noun)}))
        out["weekday_free"] = specs
    return out


# --------------------------------------------------- token-length annotation
def _bin(value, edges, names) -> str:
    for edge, name in zip(edges, names[:-1]):
        if value <= edge:
            return name
    return names[-1]


def annotate_token_lengths(specs: Sequence[PromptSpec], model,
                           prepend_bos: bool = True) -> List[PromptSpec]:
    """Fill token-derived ``meta`` fields from the real tokenizer, and validate."""
    from weekday_manifold.manifold.capture import resolve_capture_position

    def _pos(cap_text: str) -> int:
        probe = PromptSpec(text=spec.text, answer_day=spec.answer_day,
                           capture_text=cap_text, formulation=spec.formulation)
        return resolve_capture_position(model, probe, prepend_bos)

    for spec in specs:
        n_tok_full = model.to_tokens(spec.text, prepend_bos=prepend_bos).shape[1]
        n_tokens = n_tok_full - (1 if prepend_bos else 0)
        site_pos = {name: _pos(cap) for name, cap in spec.meta["sites"].items()}
        mention_pos = site_pos.get("mention_token")
        site_offset = {name: (None if mention_pos is None else pos - mention_pos)
                       for name, pos in site_pos.items()}

        z = spec.meta.get("z")
        if z is not None:
            anchor_pos = (mention_pos if mention_pos is not None
                          else _pos(_mention_prefix(spec.text, DAYS[z])))
        else:
            anchor_pos = site_pos["last_token"]
        frac = anchor_pos / max(1, n_tok_full - 1)

        n_mentions = spec.meta.get("n_weekday_mentions")
        if n_mentions is not None:
            assert n_mentions == 1, f"mention limit violated at annotate for {spec.text!r}"

        spec.meta.update(
            n_tokens=int(n_tokens),
            anchor_pos=int(anchor_pos),
            position_frac=float(frac),
            position_bin=_bin(frac, POSITION_BIN_EDGES, ("early", "mid", "late")),
            length_bin=_bin(n_tokens, LENGTH_BIN_EDGES, ("short", "med", "long")),
            site_pos=site_pos,
            site_offset=site_offset,
        )
    return specs


# --------------------------------------------------------------- iid split
def assign_splits(specs: Sequence[PromptSpec], iid_frac: float = 0.2,
                  seed: int = 0) -> List[PromptSpec]:
    """Hold out whole TEMPLATES as the test set — a real "unseen phrasing" held-out."""
    import hashlib

    by_family: Dict[str, Dict[str, List[PromptSpec]]] = {}
    for s in specs:
        by_family.setdefault(s.meta["family"], {}).setdefault(
            s.meta["template_id"], []).append(s)

    for fam, templates in by_family.items():
        def rank(tid: str) -> float:
            h = hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest()
            return int(h[:8], 16) / 0xFFFFFFFF
        ordered = sorted(templates, key=rank)
        n_test = int(round(iid_frac * len(ordered)))
        n_test = min(n_test, len(ordered) - 1)      # keep >=1 template in train
        n_test = max(n_test, 1) if len(ordered) >= 2 else 0
        for i, tid in enumerate(ordered):
            split = "test:iid" if i < n_test else "train"
            for sp in templates[tid]:
                sp.meta["split"] = split
    return specs


# --------------------------------------------- off-concept distractor sets
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]

_RANDOM_LINES = [
    "The kettle whistled while the cat watched the rain.",
    "A quiet library smells faintly of old paper and dust.",
    "He tightened the last bolt and wiped his hands clean.",
    "The soup needed more salt but nobody wanted to say so.",
    "Bright kites tugged hard against the steady sea wind.",
    "She sketched the bridge from memory on a napkin.",
    "The old radio hummed a tune from another decade.",
    "Fresh bread cooled on the sill by the open window.",
]


def build_distractors(
    kinds: Sequence[str] = ("months", "random", "weekday_free"),
) -> Dict[str, List[PromptSpec]]:
    """Off-concept prompts for the exclusivity tests (never used to fit)."""
    out: Dict[str, List[PromptSpec]] = {}
    if "months" in kinds:
        specs: List[PromptSpec] = []
        templates = ["We met in {m}", "On {m}, she baked a cake.", "It happened back in {m}"]
        for ti, tmpl in enumerate(templates):
            for mi, month in enumerate(_MONTHS):
                text = _cap(tmpl.format(m=month))
                specs.append(PromptSpec(
                    text=text, answer_day=mi, capture_text=text, formulation="month",
                    meta={"concept": "months", "role": "read", "content_id": ti,
                          "sites": _sites(text, month)}))
        out["months"] = specs
    if "random" in kinds:
        specs = []
        for i, line in enumerate(_RANDOM_LINES):
            specs.append(PromptSpec(
                text=line, answer_day=-1, capture_text=line, formulation="random",
                meta={"concept": "random", "role": "none", "content_id": i,
                      "sites": _sites(line, None)}))
        out["random"] = specs
    if "weekday_free" in kinds:
        specs = []
        # (a) Q:/A: cloze — same interrogative frame as C3/C5, NO weekday; the model
        #     answers with a non-weekday token (number / letter / month). The last
        #     "day is ... after the wedding" keeps the day-arithmetic FRAME with no anchor.
        qa = [
            "Q: What number comes after seven?\nA:",
            "Q: What number comes before nine?\nA:",
            "Q: What letter comes after B?\nA:",
            "Q: What letter comes before M?\nA:",
            "Q: What month comes after March?\nA:",
            "Q: What month comes before August?\nA:",
            "Q: What day is three days after the wedding?\nA:",
            "Q: What comes after the number five?\nA:",
        ]
        for ci, tmpl in enumerate(qa):
            text = _cap(tmpl)
            specs.append(PromptSpec(
                text=text, answer_day=-1, capture_text=text, formulation="wkfree_qa",
                meta={"concept": "weekday_free", "control_kind": "qa", "role": "compute",
                      "register": "qa", "family": "X_qa", "template_id": f"X_qa:{ci}",
                      "k": None, "z": -1, "position_target": "early", "content_id": ci,
                      "stated": False, "sites": _sites(text, None)}))
        # (b) mention / read surface (same shape as R1/R2/R3/R7) but the weekday slot
        #     holds a NON-weekday time/event noun; the mention site sits on that noun.
        mentions = [
            ("that morning, the package finally arrived.", "morning", "early"),
            ("on holiday, she baked a cake.", "holiday", "early"),
            ("the festival dragged on and the office felt restless.", "festival", "early"),
            ("every summer, the farmers' market fills the square.", "summer", "early"),
            ("we finally met on a rainy afternoon", "afternoon", "late"),
            ("the whole street lost power that night.", "night", "late"),
        ]
        for ci, (tmpl, noun, pos) in enumerate(mentions):
            text = _cap(tmpl)
            specs.append(PromptSpec(
                text=text, answer_day=-1, capture_text=text, formulation="wkfree_mention",
                meta={"concept": "weekday_free", "control_kind": "mention", "role": "read",
                      "register": "prose", "family": "X_read", "template_id": f"X_read:{ci}",
                      "k": None, "z": -1, "position_target": pos, "content_id": ci,
                      "stated": True, "sites": _sites(text, noun)}))
        out["weekday_free"] = specs
    return out
