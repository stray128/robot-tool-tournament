"""The reference family: the pre-registered parallel jaw, run through the frozen v2 path unchanged.

This is the delegated path of :mod:`rtt.effectors.registry`. An episode builds the jaw pair with
the frozen generator (:func:`rtt.design.freeform.baselines.build_entrant` for a named entrant,
:func:`rtt.design.freeform.build_pair` for a recipe), refuses a pair the scene would refuse
(:func:`rtt.sim.fingers_v2.check_pair`), runs :func:`rtt.sim.episode.run_episode_v2` and maps
its :class:`~rtt.sim.episode.EpisodeResultV2` onto the shared
:class:`~rtt.effectors.types.EffectorEpisodeResult`. The numbers are therefore exactly the
experiment's (registered grasp = ``success``, lift-and-hold = ``held``, seat = ``engaged``),
with the analytic registration reference (the jaw's centre line).

Specs: ``params = {"entrant": <name in configs/entrants_v2.yaml>}`` or
``params = {"recipe": <FingerRecipe fields>}`` (for example the experiment's frozen top design).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from rtt.effectors.registry import EffectorFamilyImpl, register_family
from rtt.effectors.runner import invalid_build_result, perturbation_name
from rtt.effectors.types import (
    EffectorEpisodeResult,
    EffectorFamily,
    EffectorSpec,
    RegistrationReference,
)

if TYPE_CHECKING:
    from rtt.design.freeform.types import FingerPair
    from rtt.parts.types import PartSpec
    from rtt.sim.episode import EpisodeResultV2
    from rtt.sim.objects import Placement
    from rtt.sim.perturb import Perturbation
    from rtt.sim.task import TaskV2

__all__ = ["DISPLAY_NAMES", "ParallelJawReference", "jaw_spec"]

#: On-screen names of the v2 entrants (no brands).
DISPLAY_NAMES: dict[str, str] = {
    "stock": "Parallel jaw, stock blade",
    "flat": "Parallel jaw, flat pad",
    "imprint_configurator": "Parallel jaw, configurator imprint",
    "imprint_swept": "Parallel jaw, swept imprint",
}
PRESET_ENTRANTS: tuple[str, ...] = ("stock", "flat", "imprint_configurator")


def jaw_spec(
    entrant: str | None = None, *, recipe: Any = None, display_name: str | None = None
) -> EffectorSpec:
    """The spec of a named v2 entrant, or of a :class:`~rtt.design.freeform.schema.FingerRecipe`."""
    if (entrant is None) == (recipe is None):
        raise ValueError("give exactly one of entrant or recipe")
    if entrant is not None:
        name = display_name or DISPLAY_NAMES.get(entrant, f"Parallel jaw, {entrant.replace('_', ' ')}")
        return EffectorSpec(
            family=EffectorFamily.PARALLEL_JAW,
            model=entrant.replace("_", "-"),
            display_name=name,
            primitive="jaw",
            params={"entrant": entrant},
            origin="reference",
        )
    fields = recipe.model_dump(mode="json") if hasattr(recipe, "model_dump") else dict(recipe)
    return EffectorSpec(
        family=EffectorFamily.PARALLEL_JAW,
        model="recipe",
        display_name=display_name or "Parallel jaw, designed fingers",
        primitive="jaw",
        params={"recipe": fields},
        origin="reference",
    )


@lru_cache(maxsize=1)
def _entrants() -> dict[str, Any]:
    from rtt.design.freeform.baselines import load_entrants_v2

    return {b.name: b for b in load_entrants_v2()}


class ParallelJawReference(EffectorFamilyImpl):
    """The parallel-jaw reference (fidelity ``core``): delegates to the frozen v2 episode."""

    family: ClassVar[EffectorFamily] = EffectorFamily.PARALLEL_JAW
    reference_mode: ClassVar[Literal["analytic", "nominal_episode"]] = "analytic"

    def presets(self) -> tuple[EffectorSpec, ...]:
        return tuple(jaw_spec(name) for name in PRESET_ENTRANTS)

    def pair_for(self, spec: EffectorSpec, part: PartSpec, task: TaskV2) -> FingerPair:
        """Build and check the jaw pair (``ValueError``, e.g. ``InvalidDesign``, if it cannot be)."""
        from rtt.design.freeform import build_pair
        from rtt.design.freeform.baselines import build_entrant
        from rtt.design.freeform.schema import FingerRecipe
        from rtt.sim.fingers_v2 import check_pair

        self.check_spec(spec)
        params = spec.params
        if "entrant" in params:
            name = str(params["entrant"])
            if name not in _entrants():
                raise ValueError(f"unknown v2 entrant {name!r}")
            pair = build_entrant(_entrants()[name], part, task)
        elif "recipe" in params:
            recipe = FingerRecipe.model_validate(_plain(params["recipe"]))
            pair = build_pair(recipe, part, task)
        else:
            raise ValueError("a parallel-jaw spec needs params.entrant or params.recipe")
        check_pair(pair)
        return pair

    def run_episode(
        self,
        spec: EffectorSpec,
        part: PartSpec,
        placement: Placement,
        seed: int,
        *,
        task: TaskV2 | None = None,
        perturbation: Perturbation | None = None,
        record: bool = False,
        reference: RegistrationReference | None = None,
        condition_id: str = "",
        scene: Any = None,
    ) -> EffectorEpisodeResult:
        """The frozen v2 episode, mapped (``reference`` and ``scene`` are not used: analytic)."""
        from rtt.sim.episode import run_episode_v2
        from rtt.sim.task import default_task_v2

        task = task or default_task_v2()
        try:
            pair = self.pair_for(spec, part, task)
        except ValueError as exc:
            return invalid_build_result(
                spec,
                part.name,
                seed,
                str(exc) or type(exc).__name__,
                condition_id=condition_id,
                placement=placement,
                perturbation=perturbation,
            )
        result = run_episode_v2(
            pair, part, placement, seed, record=record, task=task, perturbation=perturbation
        )
        return from_v2(
            result, spec, placement=placement, perturbation=perturbation, condition_id=condition_id
        )


def _plain(value: Any) -> Any:
    if hasattr(value, "items"):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def from_v2(
    result: EpisodeResultV2,
    spec: EffectorSpec,
    *,
    placement: Placement | None = None,
    perturbation: Perturbation | None = None,
    condition_id: str = "",
) -> EffectorEpisodeResult:
    """Map a frozen v2 episode result onto the shared result (nothing is recomputed)."""
    import dataclasses

    reg = result.registration
    m = result.metrics
    seated = reg is not None and reg.seated
    pos = None
    if reg is not None and reg.measured:
        # offset from the jaw centre line (the analytic reference), metres; the v2 record keeps
        # only the yaw residual's magnitude, so no orientation is reported
        pos = (reg.dx_mm / 1000.0, reg.dy_mm / 1000.0, 0.0)
    return EffectorEpisodeResult(
        spec=spec,
        part_id=result.part_id,
        seed=result.seed,
        success=result.success,
        held=result.lift_success,
        engaged=seated,
        failure_mode=result.failure_mode,
        registration_flags=tuple(reg.sub_flags) if reg is not None else (),
        flags=tuple(f.split("@", 1)[0] for f in result.flags),
        max_lift=result.max_lift,
        hold_time=result.hold_time,
        bad_qacc=result.bad_qacc,
        penetration_mm=1000.0 * m.get("obj_finger_pen_max", math.nan),
        traj_sha=result.traj_sha,
        n_steps=result.n_steps,
        wall_time_s=result.wall_time_s,
        metrics=dict(m),
        reference="analytic",
        reg_pos=pos,
        reg_quat_wxyz=None,
        condition_id=condition_id,
        placement={k: float(v) for k, v in dataclasses.asdict(placement).items()}
        if placement is not None
        else {},
        perturbation=perturbation_name(perturbation),
        effector_mass_kg=m.get("finger_mass", math.nan),
        replay=result.replay,
        qpos=result.qpos,
        overlays=result.overlays,
    )


register_family(ParallelJawReference())
