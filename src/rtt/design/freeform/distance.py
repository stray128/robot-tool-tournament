"""Normalised distance between v2 recipes, for the diversity rule and same-cell reporting.

A recipe is a point in the 9 scored dimensions (:data:`~rtt.design.freeform.schema.SCORED_PARAMS`),
taken after canonicalisation (inactive parameters at their fixed values, quantised). Each
continuous parameter contributes ``|a - b| / (range width)`` and each categorical 0 (same) or 1
(different); the distance is the mean of the 9 terms, so it lies in ``[0, 1]``. Product-only
options, ``name`` and ``rationale`` are ignored.

The diversity threshold ``t`` is not set here: it is fixed by the written rule (at most 1% of
10k random valid Sobol pairs fall within ``t``) and passed in by the search.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from rtt.design.freeform.schema import (
    CATEGORICAL_OPTIONS,
    CONTINUOUS_PARAMS,
    PARAM_BOUNDS,
    SCORED_PARAMS,
    FingerRecipe,
    quantized_values,
)


def recipe_cell(recipe: FingerRecipe) -> str:
    """The recipe's cell, ``"<pad>/<tip>"`` (9 cells)."""
    return f"{recipe.pad}/{recipe.tip}"


def recipe_vector(recipe: FingerRecipe) -> np.ndarray:
    """The canonical point: categorical indices then continuous values scaled to [0, 1]."""
    v = quantized_values(recipe)
    out = [float(CATEGORICAL_OPTIONS[k].index(v[k])) for k in ("pad", "tip")]
    for k in CONTINUOUS_PARAMS:
        lo, hi = PARAM_BOUNDS[k]
        out.append((float(v[k]) - lo) / (hi - lo))
    return np.array(out)


def parameter_distances(a: FingerRecipe, b: FingerRecipe) -> dict[str, float]:
    """Per-parameter normalised distances, each in [0, 1]."""
    va, vb = recipe_vector(a), recipe_vector(b)
    out = {}
    for i, k in enumerate(SCORED_PARAMS):
        out[k] = float(va[i] != vb[i]) if i < 2 else min(1.0, abs(float(va[i] - vb[i])))
    return out


def recipe_distance(a: FingerRecipe, b: FingerRecipe) -> float:
    """Mean of the 9 per-parameter distances, in [0, 1]."""
    d = parameter_distances(a, b)
    return sum(d.values()) / len(d)


def vectors_distance(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """Vectorised :func:`recipe_distance` on :func:`recipe_vector` rows (broadcasts)."""
    va, vb = np.asarray(va), np.asarray(vb)
    cat = (va[..., :2] != vb[..., :2]).astype(float).sum(axis=-1)
    cont = np.minimum(1.0, np.abs(va[..., 2:] - vb[..., 2:])).sum(axis=-1)
    return (cat + cont) / len(SCORED_PARAMS)


def min_distance(recipe: FingerRecipe, others: Iterable[FingerRecipe]) -> float:
    """Distance to the nearest of ``others`` (``inf`` if there are none)."""
    return min((recipe_distance(recipe, o) for o in others), default=math.inf)


def pairwise_distances(recipes: Sequence[FingerRecipe]) -> np.ndarray:
    """Symmetric ``(n, n)`` distance matrix with a zero diagonal."""
    vs = np.stack([recipe_vector(r) for r in recipes]) if recipes else np.zeros((0, len(SCORED_PARAMS)))
    return vectors_distance(vs[:, None, :], vs[None, :, :])


def too_similar(recipe: FingerRecipe, others: Iterable[FingerRecipe], threshold: float) -> bool:
    """True when ``recipe`` lies within ``threshold`` of any earlier design (the diversity rule)."""
    return min_distance(recipe, others) < threshold
