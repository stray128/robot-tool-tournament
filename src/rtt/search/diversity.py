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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from rtt.design.distance import design_distance, pairwise_distances
from rtt.design.schema import CATEGORICAL_OPTIONS, GEOMETRY_PARAMS, PARAM_BOUNDS, FingerDesign
from rtt.search.space import V1

if TYPE_CHECKING:
    from rtt.search.space import DesignSpace

__all__ = [
    "BASELINE_CELL",
    "CELL_PARAMS",
    "EXPLORE_MIN_DISTANCE",
    "EXPLORE_MIN_DISTANCE_V2",
    "MIN_DISTANCE",
    "MIN_DISTANCE_V2",
    "NUMERIC_PARAMS",
    "POINTS_PER_RANGE",
    "SCREEN_REASONS",
    "THRESHOLD_RULE_SEED",
    "UNREACHABLE_POINTS",
    "Nearest",
    "Screened",
    "ThresholdRule",
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
    "screen",
    "sequential_nn_distances",
    "threshold_rule",
    "to_points",
]

MIN_DISTANCE: Final = 0.08
#: v2 diversity bar t from :func:`threshold_rule` (1 % of 10k random valid Sobol pairs fall within
#: t; 73 points). [derived] with the schema validators only: recompute when track G's geometry
#: validators land and before PREREG (``tests/test_search_dedup.py`` pins the rule's output).
MIN_DISTANCE_V2: Final = 0.081
#: v2 explore bar: the median distance between random valid recipes in the same cell (153 points).
EXPLORE_MIN_DISTANCE_V2: Final = 0.17
#: Within-cell distance an ``explore`` design needs when its cell is already tried.
EXPLORE_MIN_DISTANCE: Final = 0.2
#: Points spanned by one numeric parameter's range (and charged for one categorical change).
POINTS_PER_RANGE: Final = 100
#: The two categorical parameters; their pair is a design's cell.
CELL_PARAMS: Final[tuple[str, str]] = ("inner_face", "tip_profile")
#: The 8 numeric parameters in canonical order.
NUMERIC_PARAMS: Final[tuple[str, ...]] = tuple(p for p in GEOMETRY_PARAMS if p not in CATEGORICAL_OPTIONS)

Cell = tuple[str, str]
#: Pseudo-cell of a v2 baseline that has no recipe point (the stock blade).
BASELINE_CELL: Final[Cell] = ("baseline", "baseline")
#: Distance in points reported for a v2 baseline without a recipe point (it is never too close).
UNREACHABLE_POINTS: Final = 10**6


@dataclass(frozen=True)
class Nearest:
    """Closest reference design: its position in the reference list, label and distance."""

    index: int | None
    label: str | None
    distance: float


def _metric(space: DesignSpace | None) -> Callable[[Any, Any], float]:
    """The design distance of ``space`` (v1's :func:`design_distance` when ``space`` is None or v1)."""
    if space is None or space.name == V1:
        return design_distance
    return space.distance


def nearest(
    design: Any,
    references: Sequence[Any],
    labels: Sequence[str] | None = None,
    *,
    space: DesignSpace | None = None,
) -> Nearest:
    """Nearest neighbour of ``design`` among ``references`` (distance ``inf`` when empty)."""
    dist = _metric(space)
    best_i: int | None = None
    best_d = float("inf")
    for i, other in enumerate(references):
        d = dist(design, other)
        if d < best_d:
            best_i, best_d = i, d
    label = labels[best_i] if labels is not None and best_i is not None else None
    return Nearest(best_i, label, best_d)


def neighbours(
    design: Any,
    references: Sequence[Any],
    labels: Sequence[str] | None = None,
    k: int = 3,
    *,
    max_distance: float = math.inf,
    space: DesignSpace | None = None,
) -> list[Nearest]:
    """The ``k`` nearest references (closest first, ties in reference order) within ``max_distance``."""
    dist = _metric(space)
    scored = sorted((dist(design, other), i) for i, other in enumerate(references))
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


def diversity_stats(designs: Sequence[Any], *, space: DesignSpace | None = None) -> dict[str, float]:
    """Mean and minimum pairwise distance of a design set (reported per run).

    In v2, baselines without a recipe point (the stock blade) are left out.
    """
    if space is not None and space.name != V1:
        designs = [d for d in designs if space.cell(d) is not None]
    n = len(designs)
    if n < 2:
        return {"mean_pairwise": 0.0, "min_pairwise": 0.0}
    if space is None or space.name == V1:
        dist = pairwise_distances(designs)
    else:
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist[i, j] = dist[j, i] = space.distance(designs[i], designs[j])
    upper = dist[np.triu_indices(n, k=1)]
    return {"mean_pairwise": float(upper.mean()), "min_pairwise": float(upper.min())}


# --------------------------------------------------------------------------- points and cells


def cell_of(design: Any, space: DesignSpace | None = None) -> Cell:
    """The design's cell: ``(inner_face, tip_profile)`` in v1, ``(pad, tip)`` in v2.

    A v2 baseline without a recipe point (the stock blade) is in the pseudo-cell
    :data:`BASELINE_CELL`.
    """
    if space is None or space.name == V1:
        return (design.inner_face, design.tip_profile)
    return space.cell(design) or BASELINE_CELL


def cell_name(cell: Cell) -> str:
    """Prompt name of a cell, e.g. ``"concave_arc + hook"``."""
    return f"{cell[0]} + {cell[1]}"


def all_cells(space: DesignSpace | None = None) -> list[Cell]:
    """Every cell, in schema order (v1: ``(inner_face, tip_profile)``; v2: ``(pad, tip)``)."""
    if space is not None and space.name != V1:
        return space.cells()
    return [(f, t) for f in CATEGORICAL_OPTIONS["inner_face"] for t in CATEGORICAL_OPTIONS["tip_profile"]]


def to_points(name: str, value: float) -> float:
    """``value`` of numeric parameter ``name`` on its 0-100 point scale (unrounded)."""
    lo, hi = PARAM_BOUNDS[name]
    return POINTS_PER_RANGE * (float(value) - lo) / (hi - lo)


def points_of(design: FingerDesign) -> dict[str, int]:
    """The 8 numeric parameters on the 0-100 point scale, rounded to integers (prompt form)."""
    return {name: round(to_points(name, getattr(design, name))) for name in NUMERIC_PARAMS}


def distance_points(a: Any, b: Any, space: DesignSpace | None = None) -> int:
    """Design distance in points: ``100 x n_terms x distance`` rounded (sum of point differences).

    v1 has 10 terms (``1000 x design_distance``), v2 has 9. A v2 baseline without a recipe
    point is infinitely far from everything; it is reported as a large sentinel.
    """
    if space is None or space.name == V1:
        return round(design_distance(a, b) * POINTS_PER_RANGE * len(GEOMETRY_PARAMS))
    d = space.distance(a, b)
    return UNREACHABLE_POINTS if math.isinf(d) else round(d * POINTS_PER_RANGE * space.n_terms)


def explore_check(
    design: Any,
    references: Sequence[Any],
    labels: Sequence[str] | None = None,
    min_cell_distance: float = EXPLORE_MIN_DISTANCE,
    *,
    space: DesignSpace | None = None,
) -> tuple[bool, Nearest | None]:
    """Novelty rule for an ``explore`` design.

    Passes when no reference shares the design's cell, or when the nearest
    same-cell reference is at least ``min_cell_distance`` away. Returns
    ``(passes, nearest same-cell reference or None)``.
    """
    cell = cell_of(design, space)
    same = [i for i, ref in enumerate(references) if cell_of(ref, space) == cell]
    if not same:
        return True, None
    near = nearest(design, [references[i] for i in same], space=space)
    assert near.index is not None
    index = same[near.index]
    found = Nearest(index, labels[index] if labels is not None else None, near.distance)
    return near.distance >= min_cell_distance, found


def sequential_nn_distances(
    designs: Sequence[Any], start: int = 1, *, space: DesignSpace | None = None
) -> list[float]:
    """For each design from position ``start`` on, its distance to the nearest earlier design.

    This is the quantity the diversity check thresholds, so its distribution over
    random runs shows whether :data:`MIN_DISTANCE` is scaled sensibly.
    """
    return [
        nearest(designs[i], designs[:i], space=space).distance for i in range(max(1, start), len(designs))
    ]


# --------------------------------------------------------------------------- v2: duplicates and the threshold rule

#: Rejection reasons of the shared v2 screen, cheapest first (spec section 13, step 6).
SCREEN_REASONS: Final[tuple[str, ...]] = ("duplicate_design", "duplicate_geometry", "too_similar")
#: Seed of the scrambled Sobol sample the v2 threshold rule is computed on (never a gate seed).
THRESHOLD_RULE_SEED: Final = 20261016


@dataclass(frozen=True)
class Screened:
    """Why a design failed the shared screen: reason, message and the reference it collided with."""

    reason: str
    message: str
    near: Nearest


def screen(
    design: Any,
    references: Sequence[Any],
    labels: Sequence[str] | None,
    min_distance: float | None,
    *,
    space: DesignSpace,
    context: Any = None,
) -> Screened | None:
    """The v2 duplicate and diversity screen every arm applies (``None`` when ``design`` passes).

    In order: ``duplicate_design`` (same canonical recipe as a reference, i.e. the same
    ``design_id``), ``duplicate_geometry`` (same ``geometry_id`` on the training parts, when the
    space can compute it), then ``too_similar`` (closer than ``min_distance``; skipped when
    ``min_distance`` is None).
    """

    def label(i: int) -> str | None:
        return labels[i] if labels is not None else None

    ident = space.design_id(design)
    for i, ref in enumerate(references):
        if space.design_id(ref) == ident:
            return Screened(
                "duplicate_design",
                f"identical to {label(i) or 'a tested design'} after canonicalisation (inactive parameters are "
                "ignored and values are quantised)",
                Nearest(i, label(i), 0.0),
            )
    geometry = space.geometry_id(design, context)
    if geometry is not None:
        for i, ref in enumerate(references):
            if space.geometry_id(ref, context) == geometry:
                return Screened(
                    "duplicate_geometry",
                    f"builds exactly the same jaws as {label(i) or 'a tested design'} on every training part",
                    Nearest(i, label(i), space.distance(design, ref)),
                )
    if min_distance is not None:
        near = nearest(design, references, labels, space=space)
        if near.distance < min_distance:
            return Screened(
                "too_similar",
                f"distance {near.distance:.3f} to {near.label} (needs >= {min_distance:g})",
                near,
            )
    return None


@dataclass(frozen=True)
class ThresholdRule:
    """Result of the pre-registered v2 diversity-threshold rule (spec section 12, D6).

    ``min_distance`` (t) is the largest value with at most ``quantile`` of the random valid
    pairs strictly closer than t, rounded down to 0.001. ``explore_min_distance`` is the median
    distance between random valid designs that share a cell: an ``explore`` in a tried cell
    must sit at least as far from every design there as a typical random design does.
    """

    min_distance: float
    explore_min_distance: float
    quantile: float
    n_pairs: int
    pool: int
    draws: int
    seed: int
    pair_quantiles: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "min_distance": self.min_distance,
            "explore_min_distance": self.explore_min_distance,
            "quantile": self.quantile,
            "n_pairs": self.n_pairs,
            "pool": self.pool,
            "draws": self.draws,
            "seed": self.seed,
            "pair_quantiles": dict(self.pair_quantiles),
        }


def threshold_rule(
    space: DesignSpace,
    context: Any = None,
    *,
    n_pairs: int = 10_000,
    quantile: float = 0.01,
    pool: int = 2_000,
    seed: int = THRESHOLD_RULE_SEED,
) -> ThresholdRule:
    """Apply the v2 threshold rule: at most 1 % of 10k random valid Sobol pairs fall within t.

    ``pool`` valid designs are drawn from the space's scrambled Sobol sampler (seed ``seed``,
    the same validators as every arm, under ``context``) and ``n_pairs`` distinct pairs are
    drawn uniformly from them with a seeded generator. The pool keeps the rule cheap when the
    geometry validators are slow; with ``pool >= 2 * n_pairs`` the pairs can be disjoint.
    """
    sampler = space.sample_sobol(seed, context)
    designs = sampler.sample(pool)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, pool, size=n_pairs)
    j = (i + rng.integers(1, pool, size=n_pairs)) % pool  # j != i
    dist = np.array([space.distance(designs[a], designs[b]) for a, b in zip(i, j, strict=True)])
    order = np.sort(dist)
    k = math.floor(quantile * n_pairs)  # at most k pairs may lie strictly below t
    t = float(order[min(k, n_pairs - 1)])
    t = math.floor(t * 1000.0 + 1e-9) / 1000.0
    same = [d for d, a, b in zip(dist, i, j, strict=True) if space.cell(designs[a]) == space.cell(designs[b])]
    explore = math.floor(float(np.median(same)) * 100.0 + 1e-9) / 100.0 if same else 0.0
    qs = {f"p{q:g}": float(np.quantile(dist, q / 100.0)) for q in (0.5, 1, 5, 50)}
    return ThresholdRule(
        min_distance=t,
        explore_min_distance=explore,
        quantile=quantile,
        n_pairs=n_pairs,
        pool=pool,
        draws=int(sampler.draws),
        seed=seed,
        pair_quantiles=qs,
    )
