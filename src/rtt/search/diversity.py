"""Diversity check for proposed designs, and the "points" view of design space shown to the proposer.

**The check.** A proposal is *too similar* when its normalised design distance
(:func:`rtt.design.distance.design_distance`, the mean over the 10 parameters of
``|difference| / range`` for numbers and 0/1 for categories) to any design
already simulated in the run, or to a proposal already accepted this round, is
below :data:`MIN_DISTANCE` (0.08). One changed categorical alone (0.1) clears
the bar; a single numeric parameter must move by at least 80 % of its range.

**Why 0.08.** For random designs (the shared round 0 plus 32 Sobol draws, 30 run
seeds, 960 draws) the distance from each new draw to its nearest earlier design
in the run has 1st / 5th / 50th percentiles of 0.105 / 0.124 / 0.212, and only
0.4 % of draws fall below 0.08. The bar therefore never binds on random search
and only rejects deliberate near-copies (one- or two-number tweaks of a tested
design), which is its purpose. It is kept at 0.08; see ``docs/PROPOSER.md``.

**Points.** Prompts express the same metric in integer *points*: each numeric
parameter's range spans 0-100 points, and a different ``inner_face`` or
``tip_profile`` counts 100 points, so the distance in points is
``1000 x design_distance`` (:func:`distance_points`) and 0.08 is 80 points. A
design's *cell* is its ``(inner_face, tip_profile)`` pair (:func:`cell_of`):
designs in different cells are at least 100 points apart, so only same-cell
designs can ever be too similar, and a design in a cell no tested design uses
always passes. :func:`explore_check` adds the stricter novelty rule for the
``explore`` slot: an untried cell, or at least ``min_cell_distance`` from every
tested design in its cell.

The check uses only this run's designs, so it never carries information across
runs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from rtt.design.distance import design_distance, pairwise_distances
from rtt.design.schema import CATEGORICAL_OPTIONS, GEOMETRY_PARAMS, PARAM_BOUNDS, FingerDesign

__all__ = [
    "CELL_PARAMS",
    "EXPLORE_MIN_DISTANCE",
    "MIN_DISTANCE",
    "NUMERIC_PARAMS",
    "POINTS_PER_RANGE",
    "Nearest",
    "all_cells",
    "cell_name",
    "cell_of",
    "distance_points",
    "diversity_stats",
    "explore_check",
    "nearest",
    "neighbours",
    "passes",
    "points_of",
    "sequential_nn_distances",
    "to_points",
]

MIN_DISTANCE: Final = 0.08
#: Within-cell distance an ``explore`` design needs when its cell is already tried.
EXPLORE_MIN_DISTANCE: Final = 0.2
#: Points spanned by one numeric parameter's range (and charged for one categorical change).
POINTS_PER_RANGE: Final = 100
#: The two categorical parameters; their pair is a design's cell.
CELL_PARAMS: Final[tuple[str, str]] = ("inner_face", "tip_profile")
#: The 8 numeric parameters in canonical order.
NUMERIC_PARAMS: Final[tuple[str, ...]] = tuple(p for p in GEOMETRY_PARAMS if p not in CATEGORICAL_OPTIONS)

Cell = tuple[str, str]


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


def neighbours(
    design: FingerDesign,
    references: Sequence[FingerDesign],
    labels: Sequence[str] | None = None,
    k: int = 3,
    *,
    max_distance: float = math.inf,
) -> list[Nearest]:
    """The ``k`` nearest references (closest first, ties in reference order) within ``max_distance``."""
    scored = sorted((design_distance(design, other), i) for i, other in enumerate(references))
    out: list[Nearest] = []
    for d, i in scored[: max(0, k)]:
        if d > max_distance:
            break
        out.append(Nearest(i, labels[i] if labels is not None else None, d))
    return out


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


# --------------------------------------------------------------------------- points and cells


def cell_of(design: FingerDesign) -> Cell:
    """The design's ``(inner_face, tip_profile)`` pair."""
    return (design.inner_face, design.tip_profile)


def cell_name(cell: Cell) -> str:
    """Prompt name of a cell, e.g. ``"concave_arc + hook"``."""
    return f"{cell[0]} + {cell[1]}"


def all_cells() -> list[Cell]:
    """Every ``(inner_face, tip_profile)`` pair, in schema order."""
    return [(f, t) for f in CATEGORICAL_OPTIONS["inner_face"] for t in CATEGORICAL_OPTIONS["tip_profile"]]


def to_points(name: str, value: float) -> float:
    """``value`` of numeric parameter ``name`` on its 0-100 point scale (unrounded)."""
    lo, hi = PARAM_BOUNDS[name]
    return POINTS_PER_RANGE * (float(value) - lo) / (hi - lo)


def points_of(design: FingerDesign) -> dict[str, int]:
    """The 8 numeric parameters on the 0-100 point scale, rounded to integers (prompt form)."""
    return {name: round(to_points(name, getattr(design, name))) for name in NUMERIC_PARAMS}


def distance_points(a: FingerDesign, b: FingerDesign) -> int:
    """Design distance in points: ``1000 x design_distance`` rounded (sum of point differences)."""
    return round(design_distance(a, b) * POINTS_PER_RANGE * len(GEOMETRY_PARAMS))


def explore_check(
    design: FingerDesign,
    references: Sequence[FingerDesign],
    labels: Sequence[str] | None = None,
    min_cell_distance: float = EXPLORE_MIN_DISTANCE,
) -> tuple[bool, Nearest | None]:
    """Novelty rule for an ``explore`` design.

    Passes when no reference shares the design's cell, or when the nearest
    same-cell reference is at least ``min_cell_distance`` away. Returns
    ``(passes, nearest same-cell reference or None)``.
    """
    cell = cell_of(design)
    same = [i for i, ref in enumerate(references) if cell_of(ref) == cell]
    if not same:
        return True, None
    near = nearest(design, [references[i] for i in same])
    assert near.index is not None
    index = same[near.index]
    found = Nearest(index, labels[index] if labels is not None else None, near.distance)
    return near.distance >= min_cell_distance, found


def sequential_nn_distances(designs: Sequence[FingerDesign], start: int = 1) -> list[float]:
    """For each design from position ``start`` on, its distance to the nearest earlier design.

    This is the quantity the diversity check thresholds, so its distribution over
    random runs shows whether :data:`MIN_DISTANCE` is scaled sensibly.
    """
    return [nearest(designs[i], designs[:i]).distance for i in range(max(1, start), len(designs))]
