"""Subspace definitions for the weekday manifold, built so two fits stay comparable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from weekday_manifold.manifold.manifold import Manifold
from weekday_manifold.manifold.pca import PCA, cap_components, group_centroids
from weekday_manifold.manifold.spline import PeriodicSpline

Selector = Callable[[dict], bool]


@dataclass(frozen=True)
class SubspaceSpec:
    """A complete recipe for one candidate weekday subspace."""

    layer: int
    capture_site: str = "last_token"
    fit_selector: Optional[Selector] = None
    method: str = "variance"
    demean: Optional[Sequence[str]] = None
    n_dims: int = 64
    name: str = ""

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name, "layer": self.layer, "capture_site": self.capture_site,
            "method": self.method, "demean": list(self.demean) if self.demean else None,
            "n_dims": self.n_dims,
        }


# ---------------------------------------------------------------- selectors
def sel_all(_: dict) -> bool:
    return True


def sel_family(*fids: str) -> Selector:
    fset = set(fids)
    return lambda m: m.get("family") in fset


def sel_role(role: str) -> Selector:
    return lambda m: m.get("role") == role


def sel_correct(m: dict) -> bool:
    """Compute prompts the model got right (read prompts pass through)."""
    return m.get("role") != "compute" or m.get("model_correct") is True


# The forward templates that keep competence past |offset|=1 (competence screen).
# A single family sweeping k=1..7 forward makes the anchor sit 1..7 steps behind the
# answer, so each answer day co-occurs with many anchors -> dissolves the adjacency
# confound. This is the clean offset-diverse core for the computed-day manifold.
#
# FLAG (revisit before the final fit): C3:5 is the weakest of the three
# ("starting on {z} and moving {k} ahead ...", competence ~0.73 vs ~0.92 / ~0.90 for
# C3:0 / C3:3). We may LIMIT the core to ("C3:0", "C3:3") — both >= 0.90 at every
# offset used — to avoid feeding lower-competence (noisier) anchors into the PCA.
# To switch: drop "C3:5" from the tuple below (offset_core / M.compute_train follow).
OFFSET_CORE_TEMPLATES = ("C3:0", "C3:3", "C3:5")


def sel_offset_core(m: dict) -> bool:
    """Offset-diverse compute core (correct only) + all read prompts."""
    if m.get("role") != "compute":
        return True
    return m.get("template_id") in OFFSET_CORE_TEMPLATES and m.get("model_correct") is True


def sel_and(*sels: Selector) -> Selector:
    return lambda m: all(s(m) for s in sels)


def sel_train(m: dict) -> bool:
    """Train split only (whole-template holdout keeps test phrasings unseen)."""
    return m.get("split") == "train"


# The three flagship manifolds, each fit on the TRAIN split with per-template
# demeaning. Compute prompts are conditioned on model_correct (their latent day is
# only trustworthy when the model actually computed it); read prompts carry the
# stated weekday. Compared to ask whether the computed-day and read-day manifolds
# coincide / span a shared weekday subspace.
def three_manifold_specs(layer: int, site: str, n_dims: int = 6) -> List["SubspaceSpec"]:
    """Fit the COMPUTE side on the offset-diverse core only (C3:0/C3:3/C3:5, correct), so the
    manifold isn't poisoned by the |offset|=1 adjacency of the single-step templates. Other
    compute templates are captured but projected IN, not fit."""
    d = min(n_dims, 6)
    # compute part = offset core (3 forward templates) AND model_correct
    sel_compute_core = sel_and(sel_role("compute"), sel_offset_core)
    return [
        SubspaceSpec(layer, site, sel_and(sel_train, sel_offset_core), "discriminative",
                     TEMPLATE_NUISANCE_KEYS, d, name=f"M.read+compute_train[{site}]"),
        SubspaceSpec(layer, site, sel_and(sel_train, sel_compute_core), "discriminative",
                     TEMPLATE_NUISANCE_KEYS, d, name=f"M.compute_train[{site}]"),
        SubspaceSpec(layer, site, sel_and(sel_train, sel_role("read")), "discriminative",
                     TEMPLATE_NUISANCE_KEYS, d, name=f"M.read_train[{site}]"),
    ]


# ------------------------------------------------------------- preprocessing
# The flagship nuisance grouping: the exact prompt TEMPLATE at a fixed arithmetic
# offset. Within such a group only the weekday (z / answer day) varies, so subtracting
# the group's MEAN ACTIVATION removes the template baseline and leaves weekday-related
# variance — the "remove each template's mean" recipe. (template_id already encodes
# family + position, since position variants are distinct template strings.)
TEMPLATE_NUISANCE_KEYS = ("template_id", "k")


def condition_demean(X: np.ndarray, metas: Sequence[dict], keys: Sequence[str],
                     ref_X: Optional[np.ndarray] = None,
                     ref_metas: Optional[Sequence[dict]] = None) -> np.ndarray:
    """Subtract each condition group's MEAN activation; groups keyed by ``meta[keys]``."""
    X = np.asarray(X, dtype=float)
    src_X = X if ref_X is None else np.asarray(ref_X, dtype=float)
    src_metas = metas if ref_metas is None else ref_metas
    means: Dict[tuple, np.ndarray] = {}
    groups: Dict[tuple, List[int]] = {}
    for i, m in enumerate(src_metas):
        groups.setdefault(tuple(m.get(k) for k in keys), []).append(i)
    for g, idx in groups.items():
        means[g] = src_X[idx].mean(axis=0)
    out = X.copy()
    for i, m in enumerate(metas):
        g = tuple(m.get(k) for k in keys)
        out[i] -= means.get(g, X[i])     # fallback: self (no reference group)
    return out


def _pca_from_matrix(fit_matrix: np.ndarray, global_mean: np.ndarray,
                     n_dims: Optional[int]) -> PCA:
    """PCA whose axes are the top right-singular vectors of ``fit_matrix`` but whose mean is
    ``global_mean`` (so ``transform`` still centers real activations)."""
    _, S, Vt = np.linalg.svd(fit_matrix, full_matrices=False)
    n_rows = fit_matrix.shape[0]
    k = cap_components(n_dims, n_samples=n_rows + 1, n_features=fit_matrix.shape[1])
    k = min(k, Vt.shape[0])
    denom = max(1, n_rows - 1)
    var_all = (S ** 2) / denom
    total = var_all.sum()
    ratio = var_all / total if total > 0 else np.zeros_like(var_all)
    return PCA(mean=global_mean, components=Vt[:k],
               explained_variance=var_all[:k], explained_variance_ratio=ratio[:k])


def fit_subspace(store: dict, metas: Sequence[dict], spec: SubspaceSpec,
                 n_labels: int = 7, label_names: Optional[Sequence[str]] = None) -> Manifold:
    """Fit the :class:`Manifold` for one :class:`SubspaceSpec`."""
    X_all = np.asarray(store["acts"][spec.capture_site][spec.layer], dtype=float)
    labels_all = np.asarray(store["labels"])
    if spec.fit_selector is None:
        mask = np.ones(len(labels_all), dtype=bool)
    else:
        mask = np.array([bool(spec.fit_selector(m)) for m in metas])
    X, labels = X_all[mask], labels_all[mask]
    metas_f = [m for m, keep in zip(metas, mask) if keep]
    if X.shape[0] < n_labels:
        raise ValueError(f"subspace {spec.name!r}: only {X.shape[0]} rows selected.")
    present = set(int(v) for v in labels)
    missing = set(range(n_labels)) - present
    if missing:
        raise ValueError(f"subspace {spec.name!r}: days {sorted(missing)} have no rows.")

    global_mean = X.mean(axis=0)
    # Per-template demean from the SELECTED (train) rows ONLY — no held-out data
    # enters the fit. With whole-template holdout each train template keeps its full
    # 7-day grid here, so its mean subtracts exactly and no template variance leaks.
    Xd = condition_demean(X, metas_f, spec.demean) if spec.demean else (X - global_mean)

    if spec.method == "variance":
        fit_matrix = Xd
    elif spec.method == "discriminative":
        # Between-day centroids of the (demeaned) data — the ring subspace.
        fit_matrix = group_centroids(Xd, labels, n_labels)
    else:
        raise ValueError(f"method must be 'variance' or 'discriminative', got {spec.method!r}.")

    pca = _pca_from_matrix(fit_matrix, global_mean, spec.n_dims)
    Z = pca.transform(X)
    centroids = group_centroids(Z, labels, n_labels)
    order = list(range(n_labels))
    names = list(label_names) if label_names is not None else [str(i) for i in range(n_labels)]
    return Manifold(pca=pca, centroids=centroids,
                    spline=PeriodicSpline(centroids[order]),
                    day_order=order, labels=names)


# ---------------------------------------------------------- the default grid
def default_spec_grid(layer: int, sites: Sequence[str] = ("last_token", "summary_token"),
                      n_dims: int = 64) -> List[SubspaceSpec]:
    """The flagship comparison at one layer (crossed with capture sites)."""
    nuisance = TEMPLATE_NUISANCE_KEYS
    grid: List[SubspaceSpec] = []
    for site in sites:
        grid += [
            SubspaceSpec(layer, site, sel_all, "variance", None, n_dims,
                         name=f"A.pooled_var[{site}]"),
            SubspaceSpec(layer, site, sel_all, "discriminative", nuisance, min(n_dims, 6),
                         name=f"B.nuisance_removed[{site}]"),
            SubspaceSpec(layer, site, sel_family("C1"), "variance", None, n_dims,
                         name=f"C.single_C1[{site}]"),
            SubspaceSpec(layer, site, sel_role("compute"), "discriminative", nuisance,
                         min(n_dims, 6), name=f"compute_only[{site}]"),
            SubspaceSpec(layer, site, sel_role("read"), "discriminative", nuisance,
                         min(n_dims, 6), name=f"read_only[{site}]"),
            SubspaceSpec(layer, site, sel_correct, "discriminative", nuisance,
                         min(n_dims, 6), name=f"model_correct[{site}]"),
            SubspaceSpec(layer, site, sel_offset_core, "discriminative", nuisance,
                         min(n_dims, 6), name=f"offset_core[{site}]"),
        ]
        # the three flagship train-fit manifolds (read+compute / compute / read)
        grid += three_manifold_specs(layer, site, n_dims)
    return grid
