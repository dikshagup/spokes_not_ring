"""Pure competence helpers: full-string aggregation, argmax, verdict."""

import numpy as np

from weekday_manifold.manifold.competence import (
    predict_day_from_scores,
    teacher_forced_logprob,
    verdict_for_accuracy,
)


def test_teacher_forced_logprob_sums_tokens():
    # Joint logprob of a multi-token day = sum of per-token conditional logprobs.
    assert teacher_forced_logprob([-1.0, -2.0, -0.5]) == -3.5
    assert teacher_forced_logprob([-0.7]) == -0.7  # single-token day


def test_predict_day_argmax():
    scores = [-5.0, -1.0, -3.0, -9.0, -2.0, -8.0, -7.0]
    assert predict_day_from_scores(scores) == 1  # highest (least negative)


def test_verdict_bands():
    assert "FAIL" in verdict_for_accuracy(0.10)
    assert "WEAK" in verdict_for_accuracy(0.30)
    assert "PASS" in verdict_for_accuracy(0.60)
    assert "strong" in verdict_for_accuracy(0.95)
