"""Task-competence gate: does the model elicit the right weekday?"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from weekday_manifold.manifold.days import DAYS, N_DAYS, PromptSpec, tokenize_day


# --------------------------------------------------------------- pure helpers
def predict_day_from_scores(day_scores: Sequence[float]) -> int:
    """Argmax weekday index over the seven per-day scores (higher = preferred)."""
    return int(np.argmax(np.asarray(day_scores, dtype=float)))


def teacher_forced_logprob(
    token_logprobs: Sequence[float],
) -> float:
    """Joint log-prob of a day string = sum of its per-token conditional logprobs."""
    return float(np.sum(np.asarray(token_logprobs, dtype=float)))


@dataclass
class PromptResult:
    text: str
    answer_day: int           # intended/correct day
    predicted_day: int        # full-string argmax
    firsttoken_day: int       # first-token argmax (cross-check)
    correct: bool             # predicted_day == answer_day (per chosen rule)
    answer_logprob: float     # joint logprob of the CORRECT day
    predicted_logprob: float  # joint logprob of the PREDICTED day
    top_token_id: int = -1        # raw argmax next token over the FULL vocab
    top_token_logprob: float = 0.0


@dataclass
class FormulationReport:
    name: str
    n_prompts: int
    overall_accuracy: float
    per_day_accuracy: Dict[str, float]   # day name -> accuracy when that's the answer
    examples: List[Dict[str, object]]
    verdict: str
    results: List[PromptResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_prompts": self.n_prompts,
            "overall_accuracy": self.overall_accuracy,
            "per_day_accuracy": self.per_day_accuracy,
            "examples": self.examples,
            "verdict": self.verdict,
        }


def verdict_for_accuracy(acc: float) -> str:
    """Heuristic label for an overall accuracy (chance for 7 days ≈ 0.143)."""
    if acc >= 0.80:
        return "PASS (strong) — use for Step 1"
    if acc >= 0.50:
        return "PASS (usable) — ring should form; consider for Step 1"
    if acc >= 0.25:
        return "WEAK — above chance but noisy; prefer a stronger formulation"
    return "FAIL (≈chance) — model can't do this; fall back / use mention"


# ------------------------------------------------------------- model-driven
def _day_token_ids(model, prepend_bos_unused: bool = False) -> Dict[int, List[int]]:
    """``{day_index: [token ids]}`` with a leading space, no BOS (run-time)."""
    def tok(s: str) -> List[int]:
        return model.to_tokens(s, prepend_bos=False)[0].tolist()

    return {i: tokenize_day(tok, DAYS[i], leading_space=True) for i in range(N_DAYS)}


def score_prompt(
    model,
    spec: PromptSpec,
    day_ids: Dict[int, List[int]],
    prepend_bos: bool = True,
    scoring: str = "fullstring",
) -> PromptResult:
    """Score one cloze prompt: which weekday does the model elicit next?"""
    base = model.to_tokens(spec.text, prepend_bos=prepend_bos)  # [1, P]
    P = base.shape[1]
    device = base.device

    # One forward on the bare stem: its final-position logprobs cover every day's
    # first token (the next-token slot) — all we need for single-token days.
    with torch.no_grad():
        base_logits = model(base, return_type="logits")[0, -1]  # [vocab]
    base_logprobs = torch.log_softmax(base_logits, dim=-1)
    first_tok_score = {
        i: float(base_logprobs[ids[0]]) for i, ids in day_ids.items()
    }
    firsttoken_day = predict_day_from_scores(
        [first_tok_score[i] for i in range(N_DAYS)]
    )

    joint: Dict[int, float] = {}
    for i, ids in day_ids.items():
        if len(ids) == 1:
            # Single-token day: joint logprob == first-token logprob (no forward).
            joint[i] = first_tok_score[i]
            continue
        # Multi-token day: teacher-force the continuation (one extra forward).
        cont = torch.tensor([ids], device=device)
        seq = torch.cat([base, cont], dim=1)                    # [1, P+m]
        with torch.no_grad():
            logits = model(seq, return_type="logits")[0]        # [P+m, vocab]
        token_lps: List[float] = [first_tok_score[i]]
        for j in range(1, len(ids)):
            lp = torch.log_softmax(logits[P - 1 + j], dim=-1)[ids[j]]
            token_lps.append(float(lp))
        joint[i] = teacher_forced_logprob(token_lps)

    fullstring_day = predict_day_from_scores([joint[i] for i in range(N_DAYS)])
    predicted_day = fullstring_day if scoring == "fullstring" else firsttoken_day
    top_id = int(torch.argmax(base_logits))
    return PromptResult(
        text=spec.text,
        answer_day=spec.answer_day,
        predicted_day=fullstring_day,
        firsttoken_day=firsttoken_day,
        correct=(predicted_day == spec.answer_day),
        answer_logprob=joint[spec.answer_day],
        predicted_logprob=joint[predicted_day],
        top_token_id=top_id,
        top_token_logprob=float(base_logprobs[top_id]),
    )


def evaluate_formulation(
    model,
    name: str,
    specs: Sequence[PromptSpec],
    prepend_bos: bool = True,
    scoring: str = "fullstring",
    n_examples: int = 8,
    day_ids: Optional[Dict[int, List[int]]] = None,
) -> FormulationReport:
    """Score every prompt in a formulation and summarize accuracy + examples."""
    if day_ids is None:
        day_ids = _day_token_ids(model)
    results = [
        score_prompt(model, s, day_ids, prepend_bos=prepend_bos, scoring=scoring)
        for s in specs
    ]
    n = len(results)
    overall = float(np.mean([r.correct for r in results])) if n else 0.0

    per_day: Dict[str, float] = {}
    for d in range(N_DAYS):
        subset = [r for r in results if r.answer_day == d]
        per_day[DAYS[d]] = (
            float(np.mean([r.correct for r in subset])) if subset else float("nan")
        )

    examples = [
        {
            "prompt": r.text,
            "answer": DAYS[r.answer_day],
            "predicted": DAYS[r.predicted_day],
            "firsttoken": DAYS[r.firsttoken_day],
            "correct": r.correct,
        }
        for r in results[:n_examples]
    ]
    return FormulationReport(
        name=name,
        n_prompts=n,
        overall_accuracy=overall,
        per_day_accuracy=per_day,
        examples=examples,
        verdict=verdict_for_accuracy(overall),
        results=results,
    )


def annotate_model_correct(
    model,
    specs: Sequence[PromptSpec],
    prepend_bos: bool = True,
    scoring: str = "fullstring",
) -> List[PromptSpec]:
    """Record the model's next-token prediction on EVERY prompt (compute and read)."""
    day_ids = _day_token_ids(model)
    day_first = {ids[0] for ids in day_ids.values()}
    tok = getattr(model, "tokenizer", None)
    for s in specs:
        r = score_prompt(model, s, day_ids, prepend_bos=prepend_bos, scoring=scoring)
        s.meta["model_pred_day"] = int(r.predicted_day)
        s.meta["model_pred_firsttoken_day"] = int(r.firsttoken_day)
        s.meta["model_pred_token_id"] = int(r.top_token_id)
        s.meta["model_pred_token"] = tok.decode([r.top_token_id]) if tok is not None else None
        s.meta["model_top_is_weekday"] = bool(r.top_token_id in day_first)
        if s.meta.get("role") == "compute":
            s.meta["model_correct"] = bool(r.correct)
    return specs


def format_formulation_report(report: FormulationReport) -> str:
    """Pretty console block for one formulation (pure)."""
    lines = [
        f"=== formulation: {report.name} ===",
        f"  prompts: {report.n_prompts}   overall next-token accuracy: "
        f"{report.overall_accuracy:.3f}",
        "  per-day accuracy (when that day is the answer):",
    ]
    for day in DAYS:
        acc = report.per_day_accuracy.get(day, float("nan"))
        lines.append(f"    {day:<9} {acc:.3f}")
    lines.append("  examples (prompt -> predicted [answer]):")
    for ex in report.examples:
        mark = "✓" if ex["correct"] else "✗"
        lines.append(
            f"    {mark} {ex['prompt']!r} -> {ex['predicted']} "
            f"[{ex['answer']}] (1st-tok {ex['firsttoken']})"
        )
    lines.append(f"  VERDICT: {report.verdict}")
    return "\n".join(lines)
