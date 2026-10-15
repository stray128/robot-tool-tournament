"""Normalised distance between finger designs, for diversity checks.

Each continuous parameter contributes ``|a - b| / (range width)`` and each
categorical parameter contributes 0 (same) or 1 (different). The design distance
is the mean of the 10 per-parameter terms, so it lies in ``[0, 1]``: 0 for
geometrically identical designs, 1 only when every parameter sits at opposite
ends of its range and both categoricals differ. ``name`` and ``rationale`` are
ignored.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from rtt.design.schema import CATEGORICAL_OPTIONS, GEOMETRY_PARAMS, PARAM_BOUNDS, FingerDesign


def parameter_distances(a: FingerDesign, b: FingerDesign) -> dict[str, float]:
    """Per-parameter normalised distances, each in [0, 1]."""
    out: dict[str, float] = {}
    for key in GEOMETRY_PARAMS:
        va, vb = getattr(a, key), getattr(b, key)
        if key in CATEGORICAL_OPTIONS:
            out[key] = 0.0 if va == vb else 1.0
        else:
            lo, hi = PARAM_BOUNDS[key]
            out[key] = min(1.0, abs(float(va) - float(vb)) / (hi - lo))
    return out


def design_distance(a: FingerDesign, b: FingerDesign) -> float:
    """Mean of the per-parameter normalised distances, in [0, 1]."""
    terms = parameter_distances(a, b)
    return sum(terms.values()) / len(terms)


def min_distance(design: FingerDesign, others: Iterable[FingerDesign]) -> float:
    """Distance from ``design`` to its nearest neighbour in ``others`` (``inf`` if empty)."""
    return min((design_distance(design, other) for other in others), default=math.inf)


def pairwise_distances(designs: Sequence[FingerDesign]) -> np.ndarray:
    """Symmetric (n, n) matrix of design distances with a zero diagonal."""
    n = len(designs)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = design_distance(designs[i], designs[j])
    return out
