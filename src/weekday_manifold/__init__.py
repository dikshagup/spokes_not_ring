"""weekday-manifold: the polar Jacobian field over Llama-3.1-8B's weekday loop.

A standalone extract of one experiment. Llama-3.1-8B places the seven weekdays on a
closed loop in residual-stream space; ``experiments/jacobian_polar_field.py`` fills the
whole disc that loop bounds and measures the input-output Jacobian at every point in it
three ways (Frobenius norm, tangential gain, radial gain), plus the model's own
next-token distribution over the seven weekdays.

Layout mirrors the research repo this was cut from, so the cross-references in the
docstrings still resolve:

  * ``weekday_manifold.manifold`` — the concept, its prompts, and the geometry
    (centroids, periodic spline, steering displacements) that define the loop.
  * ``weekday_manifold.plateau`` — the shared layer-resolution + model-loading seam.

Everything except ``model.py`` / ``plateau/model.py`` is pure numpy/scipy and is
unit-tested without a GPU.
"""

from weekday_manifold.utils import set_seed

__all__ = ["load_model", "set_seed"]


def __getattr__(name):
    """Resolve ``load_model`` only when it is actually asked for (PEP 562)."""
    if name == "load_model":
        from weekday_manifold.model import load_model as _f
        return _f
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
