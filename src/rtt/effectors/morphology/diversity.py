"""Canonical recipes, the design distance in points, and the near-duplicate screen.

**Canonical form.** Parameters a recipe does not use (``spread_deg`` of an even layout or of a
two-digit hand, ``proximal_frac`` of a one-phalanx finger, ``pad_thickness_mm`` without a pad,
``hook_height_mm`` without a hook) are set to their defaults, and every number is rounded to
its grid (0.5 mm, 1 deg, 0.5 N, 0.01 for fractions; 1/200 of the range for other units), so
two recipes that build the same effector have the same ``recipe_id``.

**Distance in points.** Within one design space each numeric parameter's range spans 100
points and a different choice or boolean counts 100 points; the distance is the sum. Recipes of
different spaces are never near-duplicates (their distance is infinite).

**Screen** (:func:`screen`, the diversity rule, applied to both arms): a recipe is refused as
``duplicate`` when its canonical form equals a reference recipe's, and as ``too_similar`` when
it is closer than :data:`MIN_POINTS` (100 points: one changed choice, or numbers moved by a
whole range in total) to any reference: the designs already tested on this part and the ones
already accepted in this round.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from rtt.effectors.morphology.spaces import SpaceKey, design_spaces
from rtt.effectors.types import EffectorFamily, MorphologyRecipe, ParamSpec

__all__ = [
    "MIN_POINTS",
    "Screened",
    "canonical_params",
    "canonical_recipe",
    "distance_points",
    "grid_of",
    "inactive_params",
    "nearest",
    "quantise",
    "screen",
]

#: The diversity bar (points): a new recipe must be at least this far from every reference.
MIN_POINTS: Final = 100
_GRID: Final[Mapping[str, float]] = {"mm": 0.5, "deg": 1.0, "N": 0.5, "kPa": 0.5}


def grid_of(p: ParamSpec) -> float:
    """Rounding step of a float parameter."""
    if p.unit in _GRID:
        return _GRID[p.unit]
    assert p.low is not None and p.high is not None
    if p.unit == "" and p.high - p.low <= 2.0:
        return 0.01
    return (p.high - p.low) / 200.0


def quantise(p: ParamSpec, value: float) -> float | int:
    """``value`` rounded to ``p``'s grid and kept inside its bounds."""
    assert p.low is not None and p.high is not None
    if p.kind == "int":
        return int(min(max(round(value), p.low), p.high))
    step = grid_of(p)
    q = round(round(value / step) * step, 6)
    q = min(max(q, p.low), p.high)
    return int(q) if float(q).is_integer() else q


def inactive_params(key: SpaceKey, params: Mapping[str, object]) -> tuple[str, ...]:
    """Parameters of ``params`` that do not change the effector (our generators only)."""
    fam, kind = key
    if fam is not EffectorFamily.DESIGNED:
        return ()
    out = []
    n = params.get("n_fingers", params.get("n_digits"))
    if params.get("layout") == "even" or (isinstance(n, int | float) and n <= 2):
        out.append("spread_deg")
    if kind == "fingered_hand":
        if params.get("n_phalanges") == 1:
            out.append("proximal_frac")
        if params.get("pad") == "none":
            out.append("pad_thickness_mm")
    hook = params.get("hook_depth_mm")
    if kind == "hook_gripper" and isinstance(hook, int | float) and float(hook) == 0.0:
        out.append("hook_height_mm")
    return tuple(out)


def canonical_params(
    key: SpaceKey, space: Sequence[ParamSpec], params: Mapping[str, object]
) -> dict[str, object]:
    """``params`` with numbers on their grid and inactive parameters at their defaults.

    Unknown parameters and invalid values are left untouched (the validators report them).
    """
    out = dict(params)
    by_name = {p.name: p for p in space}
    for name, value in params.items():
        p = by_name.get(name)
        if p is None or p.kind not in ("float", "int") or p.problems(value):
            continue
        out[name] = quantise(p, float(value))  # type: ignore[arg-type]
    for name in inactive_params(key, out):
        p = by_name.get(name)
        if p is not None and p.default is not None and name in out:
            out[name] = p.default
    return out


def canonical_recipe(recipe: MorphologyRecipe) -> MorphologyRecipe:
    """The canonical form of ``recipe`` (same hypothesis, origin and parents)."""
    key = (recipe.family, recipe.kind)
    space = design_spaces().get(key)
    if space is None:
        return recipe
    params = canonical_params(key, space, recipe.params)
    if params == dict(recipe.params):
        return recipe
    return MorphologyRecipe(
        recipe.family,
        recipe.kind,
        params,  # type: ignore[arg-type]
        hypothesis=recipe.hypothesis,
        origin=recipe.origin,
        parents=recipe.parents,
    )


def distance_points(a: MorphologyRecipe, b: MorphologyRecipe) -> float:
    """Design distance in points (module docstring); ``inf`` across design spaces."""
    if (a.family, a.kind) != (b.family, b.kind):
        return math.inf
    space = design_spaces().get((a.family, a.kind))
    if space is None:
        return math.inf
    total = 0.0
    for p in space:
        va, vb = a.params.get(p.name), b.params.get(p.name)
        if p.kind in ("float", "int"):
            assert p.low is not None and p.high is not None
            span = p.high - p.low
            if span > 0 and isinstance(va, int | float) and isinstance(vb, int | float):
                total += 100.0 * abs(float(va) - float(vb)) / span
        elif va != vb:
            total += 100.0
    return total


@dataclass(frozen=True)
class Screened:
    """Why a recipe failed the screen, and the reference it is closest to."""

    reason: str  # "duplicate" | "too_similar"
    message: str
    index: int
    points: float


def nearest(recipe: MorphologyRecipe, references: Sequence[MorphologyRecipe]) -> tuple[int | None, float]:
    """Index of and distance to the nearest reference (``(None, inf)`` when none)."""
    best_i, best = None, math.inf
    for i, ref in enumerate(references):
        d = distance_points(recipe, ref)
        if d < best:
            best_i, best = i, d
    return best_i, best


def screen(
    recipe: MorphologyRecipe,
    references: Sequence[MorphologyRecipe],
    *,
    labels: Sequence[str] | None = None,
    min_points: float = MIN_POINTS,
) -> Screened | None:
    """``None`` when ``recipe`` passes the duplicate and diversity rule (module docstring)."""
    rid = recipe.recipe_id
    for i, ref in enumerate(references):
        if ref.recipe_id == rid:
            name = labels[i] if labels else f"reference {i + 1}"
            return Screened("duplicate", f"duplicate: the same design as {name} once canonicalised", i, 0.0)
    i, d = nearest(recipe, references)
    if i is not None and d < min_points:
        name = labels[i] if labels else f"reference {i + 1}"
        return Screened(
            "too_similar",
            f"too_similar: {d:.0f} points from {name} (needs at least {min_points:g}: change a choice, or move "
            "the numbers by a whole range in total)",
            i,
            d,
        )
    return None
