"""Arm A (random search) and the shared round 0.

:class:`DesignSampler` draws the 8 continuous parameters from a scrambled Sobol
sequence and the 2 categorical parameters uniformly, then rejection-samples until
:func:`rtt.design.validate.validate` accepts the design. ``inner_face_depth_mm`` is
inactive for a flat face (the validator requires it to be 0), so it is set to 0
there instead of being rejected; otherwise flat faces would almost never survive
rejection and the sampler would not cover the design space evenly.

Sobol points are drawn in blocks of 64 (a power of two keeps the sequence's balance
properties) and consumed in order, so a sampler with a given seed always yields the
same designs.

:func:`round0_designs` builds the shared round 0: the manual entrants followed by a
Sobol fill from a stream seeded only by the run seed, so for a given run seed it is
byte-identical in every arm.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

import numpy as np
from scipy.stats import qmc

from rtt.design.schema import CATEGORICAL_OPTIONS, PARAM_BOUNDS, FingerDesign, GripperTask
from rtt.design.validate import validate
from rtt.search.base import Proposal, RunContext, SearchStrategy, derive_seed

__all__ = ["CONTINUOUS_PARAMS", "DesignSampler", "RandomSearch", "SamplerExhaustedError", "round0_designs"]

CONTINUOUS_PARAMS: Final[tuple[str, ...]] = tuple(PARAM_BOUNDS)
_SOBOL_BLOCK: Final = 64


class SamplerExhaustedError(RuntimeError):
    """No acceptable design was found within the draw limit."""


class DesignSampler:
    """Seeded, rejection-sampled source of valid random designs.

    Args:
        seed: Seed for both the Sobol scrambling and the categorical draws.
        gripper_task: Validator constants.
        max_draws: Draws allowed per requested design before giving up.
    """

    def __init__(self, seed: int, gripper_task: GripperTask, *, max_draws: int = 20_000) -> None:
        self.seed = int(seed)
        self.gripper_task = gripper_task
        self.max_draws = max_draws
        self._sobol = qmc.Sobol(d=len(CONTINUOUS_PARAMS), scramble=True, rng=np.random.default_rng(self.seed))
        self._cats = np.random.default_rng(derive_seed(self.seed, "categoricals"))
        self._block = np.empty((0, len(CONTINUOUS_PARAMS)))
        self._pos = 0
        self.draws = 0
        self.rejected = 0

    def _next_unit(self) -> np.ndarray:
        if self._pos >= len(self._block):
            self._block = self._sobol.random(_SOBOL_BLOCK)
            self._pos = 0
        point = self._block[self._pos]
        self._pos += 1
        return point

    def draw(self) -> FingerDesign:
        """One raw draw (not yet validated)."""
        u = self._next_unit()
        params: dict[str, float | str] = {}
        for value, name in zip(u, CONTINUOUS_PARAMS, strict=True):
            lo, hi = PARAM_BOUNDS[name]
            params[name] = float(lo + value * (hi - lo))
        for name, options in CATEGORICAL_OPTIONS.items():
            params[name] = options[int(self._cats.integers(len(options)))]
        if params["inner_face"] == "flat":
            params["inner_face_depth_mm"] = 0.0
        self.draws += 1
        return FingerDesign.model_validate(params)

    def sample(self, n: int, accept: Callable[[FingerDesign], bool] | None = None) -> list[FingerDesign]:
        """``n`` designs that pass the validator (and ``accept``, when given), in draw order."""
        out: list[FingerDesign] = []
        for _ in range(n):
            for _attempt in range(self.max_draws):
                design = self.draw()
                if validate(design, self.gripper_task).ok and (accept is None or accept(design)):
                    out.append(design)
                    break
                self.rejected += 1
            else:
                raise SamplerExhaustedError(
                    f"no acceptable design in {self.max_draws} draws (seed {self.seed})"
                )
        return out


def round0_designs(
    entrants: Sequence[FingerDesign], run_seed: int, n: int, gripper_task: GripperTask
) -> list[Proposal]:
    """The shared round 0 for ``run_seed``: entrants first, then a Sobol fill up to ``n``.

    Depends only on its arguments, so every arm gets byte-identical round-0
    designs for the same run seed.
    """
    if len(entrants) > n:
        raise ValueError(f"{len(entrants)} entrants do not fit in a round 0 of {n}")
    proposals: list[Proposal] = []
    for design in entrants:
        result = validate(design, gripper_task)
        if not result.ok:
            raise ValueError(f"manual entrant {design.name!r} is invalid under this task: {result.summary()}")
        proposals.append(Proposal(design=design, origin="entrant", hypothesis=design.rationale))
    sampler = DesignSampler(derive_seed(run_seed, "round0"), gripper_task)
    for design in sampler.sample(n - len(entrants)):
        proposals.append(Proposal(design=design, origin="sobol"))
    return proposals


class RandomSearch(SearchStrategy):
    """Arm A: every later round is a fresh batch from the run's own Sobol stream."""

    arm = "random"

    def __init__(self, ctx: RunContext) -> None:
        super().__init__(ctx)
        self.sampler = DesignSampler(derive_seed(ctx.run_seed, "random"), ctx.gripper_task)

    def ask(self, round_index: int, n: int) -> list[Proposal]:
        before = self.sampler.rejected
        designs = self.sampler.sample(n)
        self.ctx.budget.sampler_rejections += self.sampler.rejected - before
        return [Proposal(design=d, origin="random") for d in designs]
