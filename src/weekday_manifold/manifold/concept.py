"""Concept abstraction -- the seam that keeps the manifold stack concept-agnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# -------------------------------------------------------------------- prompts
@dataclass
class PromptSpec:
    """One prompt, its group label, and where to capture."""

    text: str
    answer_day: int
    capture_text: str
    formulation: str
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> int:
        """Concept-class index this prompt belongs to (alias for ``answer_day``)."""
        return self.answer_day

    @property
    def scored(self) -> bool:
        """Whether next-token accuracy is meaningful."""
        return self.formulation not in ("mention", "trailing", "read_mention")


# ------------------------------------------------------------------- concept
@dataclass(frozen=True)
class Concept:
    """A concept the manifold stack can fit, independent of the geometry."""

    name: str
    labels: Sequence[str]
    is_cyclic: bool
    build_prompts: Callable[..., List[PromptSpec]]
    token_ids: Callable[[Any], Dict[int, List[int]]]
    coordinate: Optional[np.ndarray] = None

    @property
    def n_labels(self) -> int:
        return len(self.labels)


# ------------------------------------------------------------------ registry
def get_concept(name: str) -> Concept:
    """Resolve a concept by name (lazy import to avoid a module cycle)."""
    if name == "days":
        from weekday_manifold.manifold.days import make_days_concept
        return make_days_concept()
    raise ValueError(f"unknown concept {name!r}; choose from ['days'].")
