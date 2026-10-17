"""The equal-budget random control: recipes drawn uniformly from the same spaces, same validators.

:func:`random_recipes` draws recipes for one part until ``n`` pass
:func:`~rtt.effectors.morphology.validators.validate_recipe` (for several parts: on at least
two thirds of them, the rule the Nemotron arm follows) and the same duplicate and
diversity screen the Nemotron arm faces (:func:`~rtt.effectors.morphology.diversity.screen`),
then returns them with ``origin = "random"``. Each draw picks a design space uniformly, then
every parameter uniformly within its bounds (integers and choices uniformly over their
values), rounded to the parameter's grid (:func:`~rtt.effectors.morphology.diversity.quantise`)
so recipes stay short and readable. The result is a pure function of the part, ``n``,
``seed`` and the spaces (PCG64 seeded from ``seed`` and the part's geometry digest).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from rtt.effectors.morphology.diversity import MIN_POINTS, canonical_recipe, quantise, screen
from rtt.effectors.morphology.spaces import SpaceKey, select_spaces
from rtt.effectors.types import MorphologyRecipe, ParamSpec

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec
    from rtt.sim.task import TaskV2

__all__ = ["RandomDrawStats", "draw_params", "random_recipes", "random_recipes_with_stats"]

#: Draws allowed per requested recipe before giving up (a part no space fits stays short).
MAX_DRAWS_PER_RECIPE = 400


class RandomDrawStats:
    """How many draws the control needed and why the rest were refused (by validator code)."""

    def __init__(self) -> None:
        self.draws = 0
        self.refused: dict[str, int] = {}

    def refuse(self, code: str) -> None:
        self.refused[code] = self.refused.get(code, 0) + 1

    def to_dict(self) -> dict[str, object]:
        return {"draws": self.draws, "refused": dict(sorted(self.refused.items()))}


def draw_params(space: Sequence[ParamSpec], rng: np.random.Generator) -> dict[str, float | int | str | bool]:
    """One uniform draw from ``space`` (rounded to each parameter's grid)."""
    out: dict[str, float | int | str | bool] = {}
    for p in space:
        if p.kind == "bool":
            out[p.name] = bool(rng.integers(0, 2))
        elif p.kind == "choice":
            out[p.name] = p.choices[int(rng.integers(0, len(p.choices)))]
        elif p.kind == "int":
            assert p.low is not None and p.high is not None
            out[p.name] = int(rng.integers(int(p.low), int(p.high) + 1))
        else:
            assert p.low is not None and p.high is not None
            out[p.name] = quantise(p, float(rng.uniform(p.low, p.high)))
    return out


def _rng(parts: Sequence[PartSpec], seed: int) -> np.random.Generator:
    key = ",".join(p.geometry_sha256 for p in parts)
    digest = hashlib.sha256(f"{key}:{int(seed)}:random-control".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def random_recipes_with_stats(
    part: PartSpec | Sequence[PartSpec],
    n: int,
    seed: int,
    *,
    kinds: Sequence[str | SpaceKey] | None = None,
    task: TaskV2 | None = None,
    history: Sequence[MorphologyRecipe] = (),
    min_points: int = MIN_POINTS,
    min_valid_parts: int | None = None,
) -> tuple[list[MorphologyRecipe], RandomDrawStats]:
    """:func:`random_recipes` plus the draw statistics."""
    from rtt.effectors.morphology.proposer import _as_parts, _default_min_valid, recipe_problems
    from rtt.sim.task import default_task_v2

    if n < 0:
        raise ValueError("n must be >= 0")
    parts = _as_parts(part)
    min_valid = min_valid_parts or _default_min_valid(len(parts))
    task = task or default_task_v2()
    spaces = select_spaces(kinds)
    keys = list(spaces)
    rng = _rng(parts, seed)
    stats = RandomDrawStats()
    accepted: list[MorphologyRecipe] = []
    while len(accepted) < n and stats.draws < MAX_DRAWS_PER_RECIPE * max(n, 1):
        stats.draws += 1
        key = keys[int(rng.integers(0, len(keys)))]
        recipe = canonical_recipe(
            MorphologyRecipe(
                key[0],
                key[1],
                draw_params(spaces[key], rng),
                hypothesis="random draw (equal-budget control)",
                origin="random",
            )
        )
        problems = recipe_problems(recipe, parts, task, min_valid)
        if problems:
            code = problems[0] if len(parts) == 1 else problems[1].split(": ", 1)[1]
            stats.refuse(code.split(":", 1)[0])
            continue
        verdict = screen(recipe, [*history, *accepted], min_points=min_points)
        if verdict is not None:
            stats.refuse(verdict.reason)
            continue
        accepted.append(recipe)
    return accepted, stats


def random_recipes(
    part: PartSpec | Sequence[PartSpec],
    n: int,
    seed: int,
    *,
    kinds: Sequence[str | SpaceKey] | None = None,
    task: TaskV2 | None = None,
    min_valid_parts: int | None = None,
) -> list[MorphologyRecipe]:
    """``n`` valid random recipes for ``part`` (fewer only if the draw cap is reached).

    ``part`` may be several parts: a recipe then has to be buildable on ``min_valid_parts`` of
    them (two thirds by default), the rule the Nemotron arm's multi-part designs follow.
    """
    return random_recipes_with_stats(part, n, seed, kinds=kinds, task=task, min_valid_parts=min_valid_parts)[
        0
    ]
