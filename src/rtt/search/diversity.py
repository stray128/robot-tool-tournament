"""Diversity check for proposed designs.

A proposal is *too similar* when its normalised design distance
(:func:`rtt.design.distance.design_distance`, the mean over the 10 parameters of
``|difference| / range`` for numbers and 0/1 for categories) to any design
already simulated in the run, or to a proposal already accepted this round, is
below :data:`MIN_DISTANCE` (0.08). One changed categorical alone (0.1) clears
the bar; a single numeric parameter must move by at least 80 % of its range.

The check uses only this run's designs, so it never carries information across
runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from rtt.design.distance import design_distance, pairwise_distances
from rtt.design.schema import FingerDesign

__all__ = ["MIN_DISTANCE", "Nearest", "diversity_stats", "nearest", "passes"]

MIN_DISTANCE: Final = 0.08


@dataclass(frozen=True)
class Nearest:
    """Closest reference design: its position in the reference list, label and distance."""

    index: int | None
    label: str | None
    distance: float


def nearest(
    design: FingerDesign, references: Sequence[FingerDesign], labels: Sequence[str] | None = None
) -> Nearest:
    """Nearest neighbour of ``design`` among ``references`` (distance ``inf`` when empty)."""
    best_i: int | None = None
    best_d = float("inf")
    for i, other in enumerate(references):
        d = design_distance(design, other)
        if d < best_d:
            best_i, best_d = i, d
    label = labels[best_i] if labels is not None and best_i is not None else None
    return Nearest(best_i, label, best_d)


def passes(
    design: FingerDesign, references: Sequence[FingerDesign], min_distance: float = MIN_DISTANCE
) -> bool:
    """True when ``design`` is at least ``min_distance`` from every reference."""
    return nearest(design, references).distance >= min_distance


def diversity_stats(designs: Sequence[FingerDesign]) -> dict[str, float]:
    """Mean and minimum pairwise distance of a design set (reported per run)."""
    n = len(designs)
    if n < 2:
        return {"mean_pairwise": 0.0, "min_pairwise": 0.0}
    dist = pairwise_distances(designs)
    upper = dist[np.triu_indices(n, k=1)]
    return {"mean_pairwise": float(upper.mean()), "min_pairwise": float(upper.min())}
