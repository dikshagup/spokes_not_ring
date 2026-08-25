"""Weekday tokenization handling + prompt builders (no model — fake tokenizer)."""

import numpy as np

from weekday_manifold.manifold.days import (
    DAYS,
    MENTION_TEMPLATES,
    N_DAYS,
    TRAILING_TEMPLATES,
    add_days,
    build_arithmetic_prompts,
    build_mention_prompts,
    build_relational_prompts,
    build_sequence_prompts,
    build_trailing_prompts,
    day_index,
    day_token_table,
    offset_word,
    tokenization_report,
    tokenize_day,
)


class FakeGPT2Tokenizer:
    """Mimics GPT-2 BPE: leading-space words map to ids; some days are multi-token.

    The exact ids are arbitrary; what matters is that (a) the leading space is
    honored and (b) some weekdays return >1 token, exercising the multi-token
    path that the real GPT-2 tokenizer also hits.
    """

    # Pretend Wednesday/Saturday split into two pieces (as longer words can).
    _multi = {" Wednesday": [37, 1014], " Saturday": [33, 4690]}
    _single = {
        " Monday": [800], " Tuesday": [801], " Thursday": [802],
        " Friday": [803], " Sunday": [804],
    }

    def __call__(self, s):
        if s in self._multi:
            return self._multi[s]
        if s in self._single:
            return self._single[s]
        # Fallback: one id per whitespace-split word (good enough for prompts).
        return [hash(w) % 50000 for w in s.split()]


def test_day_index_and_add_days():
    assert day_index("Monday") == 0
    assert day_index("sunday") == 6
    assert add_days(5, 3) == 1  # Sat + 3 = Tue (wraps)
    assert add_days(0, 7) == 0  # full cycle


def test_offset_word():
    assert offset_word(3, "digit") == "3"
    assert offset_word(3, "word") == "three"


def test_tokenize_day_leading_space_and_multitoken():
    tok = FakeGPT2Tokenizer()
    # Single-token day.
    assert tokenize_day(tok, "Monday") == [800]
    # Multi-token day handled (full list returned, not truncated).
    assert tokenize_day(tok, "Wednesday") == [37, 1014]
    # Without leading space the lookup differs (footgun the function guards).
    assert tokenize_day(tok, "Monday", leading_space=False) != [800]


def test_day_token_table_covers_all_and_reports():
    tok = FakeGPT2Tokenizer()
    table = day_token_table(tok)
    assert set(table) == set(DAYS)
    assert all(len(ids) >= 1 for ids in table.values())
    report = tokenization_report(table)
    assert "multi-token" in report
    # Two of our fake days are multi-token.
    assert "2/7" in report


def test_arithmetic_prompts_answer_and_capture():
    specs = build_arithmetic_prompts(offset_style="digit", ks=(1, 2, 3))
    assert len(specs) == len(specs)  # builds
    for s in specs:
        z = s.meta["z"]
        k = s.meta["k"]
        assert s.answer_day == add_days(z, k)
        # Cloze: capture site is the whole stem (ends where the day goes).
        assert s.capture_text == s.text
        assert s.scored
    # Every day appears as an answer at least once.
    answers = {s.answer_day for s in specs}
    assert answers == set(range(N_DAYS))


def test_arithmetic_digit_vs_word_text_differs():
    d = build_arithmetic_prompts(offset_style="digit", ks=(3,))[0]
    w = build_arithmetic_prompts(offset_style="word", ks=(3,))[0]
    assert "3" in d.text and "three" in w.text


def test_interrogative_prompts_answer_capture_and_paper_template():
    from weekday_manifold.manifold.days import build_interrogative_prompts
    specs = build_interrogative_prompts(offset_style="digit", ks=(1, 2, 3))
    for s in specs:
        z, k = s.meta["z"], s.meta["k"]
        assert s.answer_day == add_days(z, k)
        assert s.capture_text == s.text   # cloze-style: capture at final token
        assert s.scored                   # day is computed, not given
        assert s.formulation == "interrogative"
    # Every day appears as an answer at least once.
    assert {s.answer_day for s in specs} == set(range(N_DAYS))
    # Default is the paper's single interrogative template (question phrasing).
    txt = build_interrogative_prompts(ks=(3,))[0].text
    assert txt.startswith("Q:") and "days after" in txt and txt.rstrip().endswith("A:")


def test_sequence_prompts_consecutive():
    specs = build_sequence_prompts(run_lengths=(3,))
    s = specs[0]  # starts at Monday
    assert s.text.startswith("Monday Tuesday Wednesday")
    assert s.answer_day == add_days(0, 3)  # Thursday
    assert s.capture_text == s.text


def test_relational_prompts_single_step():
    specs = build_relational_prompts()
    for s in specs:
        assert s.answer_day == add_days(s.meta["z"], 1)


def test_mention_prompts_capture_at_day_token():
    specs = build_mention_prompts()
    assert len(specs) == len(MENTION_TEMPLATES) * N_DAYS
    for s in specs:
        z = s.meta["z"]
        assert s.answer_day == z
        # capture_text ends at the day mention (a prefix of the full text).
        assert s.capture_text.endswith(DAYS[z])
        assert s.text.startswith(s.capture_text)
        assert not s.scored  # mention is representation-only
        # The weekday is the FINAL token/word, and no OTHER weekday name leaks in.
        assert s.text.endswith(DAYS[z])
        assert s.text.count(DAYS[z]) == 1
        for other in (d for i, d in enumerate(DAYS) if i != z):
            assert other not in s.text, (
                f"mention template leaks another weekday ({other!r}): {s.text!r}"
            )


def test_mention_templates_uniform_length():
    """Mention templates must render to the same length for every day.

    Fixed length keeps the (final) weekday token at the SAME absolute position
    across templates, removing positional drift and denoising the per-day
    centroids. Word count is a coarse proxy for GPT-2 token count; real-tokenizer
    verification (6 tokens per prompt, all 7 days) was done when picking the
    templates — see the MENTION_TEMPLATES module comment.
    """
    counts = [len(t.split()) for t in MENTION_TEMPLATES]
    assert len(set(counts)) == 1, (
        f"mention templates have inconsistent word counts: {counts}. "
        "This breaks the equal-token-length invariant (positional confounder)."
    )
    target = counts[0]
    for tmpl in MENTION_TEMPLATES:
        assert tmpl.rstrip().endswith("{z}"), f"mention template must end in the day: {tmpl!r}"
        for day in DAYS:
            rendered = tmpl.format(z=day)
            assert len(rendered.split()) == target


def test_trailing_prompts_capture_at_trailing_period():
    specs = build_trailing_prompts()
    # 10 default templates x 7 days.
    assert len(specs) == len(TRAILING_TEMPLATES) * N_DAYS
    for s in specs:
        z = s.meta["z"]
        # Day is given (answer_day == z), scored is False (representation-only).
        assert s.answer_day == z
        assert not s.scored
        # Capture at final token — capture_text is the whole prompt, ending ".".
        assert s.capture_text == s.text
        assert s.text.endswith(".")
        # The named day appears exactly once; NO OTHER weekday name may appear
        # anywhere in the prompt (the whole point of this formulation).
        assert s.text.count(DAYS[z]) == 1
        other_days = [d for i, d in enumerate(DAYS) if i != z]
        for other in other_days:
            assert other not in s.text, (
                f"trailing template leaks another weekday ({other!r}): {s.text!r}"
            )
    # Every day appears as answer at least once.
    assert {s.answer_day for s in specs} == set(range(N_DAYS))


def test_trailing_templates_uniform_length():
    """Every trailing template must render to the same length for every day.

    We assert two things:
      (a) All templates have the same word count (whitespace split) — a coarse
          proxy for GPT-2 token count, catching structural drift like an extra
          adjective sneaking in. Real-tokenizer verification (8 tokens per
          prompt, all 7 days) was done externally when the templates were
          picked; see the TRAILING_TEMPLATES module comment.
      (b) Substituting any day into any template preserves that word count —
          would fail if a template got a multi-word day interpolated in.
    """
    counts = [len(t.split()) for t in TRAILING_TEMPLATES]
    assert len(set(counts)) == 1, (
        f"trailing templates have inconsistent word counts: {counts}. "
        "This breaks the equal-token-length invariant (positional-encoding "
        "confounder). Fix the outlier template(s)."
    )
    target = counts[0]
    for tmpl in TRAILING_TEMPLATES:
        for day in DAYS:
            rendered = tmpl.format(z=day)
            assert len(rendered.split()) == target, (
                f"rendered prompt {rendered!r} has {len(rendered.split())} words, "
                f"expected {target}."
            )
