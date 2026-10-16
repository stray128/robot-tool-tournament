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

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np
from scipy.stats import qmc

from rtt.design.schema import CATEGORICAL_OPTIONS, PARAM_BOUNDS, FingerDesign, GripperTask
from rtt.design.validate import validate
from rtt.search.base import Proposal, RunContext, SearchStrategy, derive_seed
from rtt.search.space import V1

__all__ = [
    "CONTINUOUS_PARAMS",
    "DesignSampler",
    "RandomSearch",
    "SamplerExhaustedError",
    "SpaceSampler",
    "round0_designs",
    "screen_accept",
]

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


class SpaceSampler:
    """Arm A for any :class:`~rtt.search.space.DesignSpace` (used for v2): Sobol, canonicalise, validate.

    The space's numeric parameters come from a scrambled Sobol sequence (blocks of 64,
    consumed in order) and its two categoricals uniformly from a second seeded stream,
    exactly as :class:`DesignSampler` does for v1. Each draw is canonicalised and checked
    by the space's validators (the same code every arm uses) and then by ``accept``;
    rejections are counted by reason, so the valid rate can be logged.

    ``accept`` may return ``True`` / ``False`` or a rejection reason (``str``: rejected,
    counted under that reason; ``None``: accepted).
    """

    def __init__(self, space: Any, seed: int, context: Any, *, max_draws: int = 20_000) -> None:
        self.space = space
        self.seed = int(seed)
        self.gripper_task = context  # same attribute name as DesignSampler: the validation context
        self.max_draws = max_draws
        dims = len(space.numeric_params)
        self._sobol = qmc.Sobol(d=dims, scramble=True, rng=np.random.default_rng(self.seed))
        self._cats = np.random.default_rng(derive_seed(self.seed, "categoricals"))
        self._options = [tuple(space.categorical_options[name]) for name in space.cell_params]
        self._block = np.empty((0, dims))
        self._pos = 0
        self.draws = 0
        self.rejected = 0
        self.accepted = 0
        self.reasons: Counter[str] = Counter()

    def _next_unit(self) -> np.ndarray:
        if self._pos >= len(self._block):
            self._block = self._sobol.random(_SOBOL_BLOCK)
            self._pos = 0
        point = self._block[self._pos]
        self._pos += 1
        return point

    def draw_raw(self) -> dict[str, Any]:
        """One raw draw (before canonicalisation and validation)."""
        u = self._next_unit()
        cats = [options[int(self._cats.integers(len(options)))] for options in self._options]
        self.draws += 1
        return self.space.from_unit(u, *cats)

    def draw(self) -> Any | None:
        """One draw, canonicalised and validated: the design, or ``None`` (counted as a rejection)."""
        check = self.space.validate(self.draw_raw(), self.gripper_task)
        if not check.ok:
            self.reasons[check.stage or "invalid"] += 1
            return None
        return check.design

    def sample(self, n: int, accept: Callable[[Any], bool | str | None] | None = None) -> list[Any]:
        """``n`` designs that pass the validators (and ``accept``, when given), in draw order."""
        out: list[Any] = []
        for _ in range(n):
            for _attempt in range(self.max_draws):
                design = self.draw()
                verdict: bool | str | None = True
                if design is not None and accept is not None:
                    verdict = accept(design)
                if design is not None and (verdict is True or verdict is None):
                    out.append(design)
                    self.accepted += 1
                    break
                if design is not None:
                    self.reasons[verdict if isinstance(verdict, str) else "rejected"] += 1
                self.rejected += 1
            else:
                raise SamplerExhaustedError(
                    f"no acceptable design in {self.max_draws} draws (seed {self.seed})"
                )
        return out

    @property
    def valid_rate(self) -> float:
        """Share of draws that passed the validators (before the duplicate and diversity screen)."""
        invalid = sum(v for k, v in self.reasons.items() if k not in _SCREEN_KEYS)
        return (self.draws - invalid) / self.draws if self.draws else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "draws": self.draws,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "valid_rate": round(self.valid_rate, 6),
            "rejections_by_reason": dict(sorted(self.reasons.items())),
        }


_SCREEN_KEYS: Final = frozenset({"duplicate_design", "duplicate_geometry", "too_similar", "rejected"})


def screen_accept(
    space: Any,
    context: Any,
    references: Callable[[], Sequence[Any]],
    min_distance: float | None,
) -> Callable[[Any], str | None]:
    """An ``accept`` for :class:`SpaceSampler`: the shared v2 duplicate and diversity screen.

    ``references`` is called on every draw, so designs accepted earlier in the same batch can
    be included by the caller.
    """
    from rtt.search.diversity import screen

    def accept(design: Any) -> str | None:
        found = screen(design, references(), None, min_distance, space=space, context=context)
        return None if found is None else found.reason

    return accept


def round0_designs(
    entrants: Sequence[Any],
    run_seed: int,
    n: int,
    gripper_task: Any,
    *,
    space: Any = None,
    min_distance: float | None = None,
) -> list[Proposal]:
    """The shared round 0 for ``run_seed``: entrants first, then a Sobol fill up to ``n``.

    Depends only on its arguments, so every arm gets byte-identical round-0
    designs for the same run seed. With a v2 ``space`` the fill goes through the shared
    duplicate and diversity screen (``min_distance``) against the entrants and the fill so far.
    """
    if space is not None and space.name != V1:
        return _round0_space(entrants, run_seed, n, gripper_task, space, min_distance)
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


def _round0_space(
    entrants: Sequence[Any], run_seed: int, n: int, context: Any, space: Any, min_distance: float | None
) -> list[Proposal]:
    if len(entrants) > n:
        raise ValueError(f"{len(entrants)} entrants do not fit in a round 0 of {n}")
    proposals: list[Proposal] = []
    for entrant in entrants:
        check = space.validate(entrant, context)
        if not check.ok:
            name = space.label(entrant) or space.design_id(entrant)[:12]
            raise ValueError(f"entrant {name!r} is invalid under this task: {check.summary()}")
        rationale = getattr(space, "rationale", lambda d: None)(entrant)
        proposals.append(Proposal(design=entrant, origin="entrant", hypothesis=rationale))
    sampler = space.sample_sobol(derive_seed(run_seed, "round0"), context)
    chosen: list[Any] = [p.design for p in proposals]
    accept = screen_accept(space, context, lambda: chosen, min_distance)
    for _ in range(n - len(entrants)):
        design = sampler.sample(1, accept)[0]
        chosen.append(design)
        proposals.append(Proposal(design=design, origin="sobol"))
    return proposals


class RandomSearch(SearchStrategy):
    """Arm A: every later round is a fresh batch from the run's own Sobol stream.

    In v1 every valid draw is used. In v2 every arm's designs go through the same
    duplicate and diversity screen (spec section 13, step 6), so arm A rejection-samples
    against its own run's designs at ``ctx.shared_min_distance`` too.
    """

    arm = "random"

    def __init__(self, ctx: RunContext) -> None:
        super().__init__(ctx)
        self.sampler = ctx.space.sample_sobol(derive_seed(ctx.run_seed, "random"), ctx.gripper_task)

    def ask(self, round_index: int, n: int) -> list[Proposal]:
        before = self.sampler.rejected
        if self.ctx.space.name == V1:
            designs = self.sampler.sample(n)
        else:
            designs = []
            refs = [item.design for item in self.ctx.history]
            accept = screen_accept(
                self.ctx.space, self.ctx.gripper_task, lambda: [*refs, *designs], self.ctx.shared_min_distance
            )
            for _ in range(n):
                designs.append(self.sampler.sample(1, accept)[0])
        self.ctx.budget.sampler_rejections += self.sampler.rejected - before
        return [Proposal(design=d, origin="random") for d in designs]
