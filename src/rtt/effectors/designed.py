"""The ``designed`` family: our own physics-validated effector morphologies (product mode).

A designed effector is a :class:`~rtt.effectors.types.MorphologyRecipe` of one of our
generators (:data:`rtt.effectors.morphology.spaces.DESIGNED_KINDS`: ``fingered_hand``,
``hook_gripper``, ``segmented_tentacle``), proposed by Nemotron or drawn at random by the
equal-budget control (:mod:`rtt.effectors.morphology`). The recipe is the whole design: the
morphology, the grasp synergy, the grip height and the effector's yaw relative to the part.

* **Build** (:meth:`DesignedFamily.build`): the recipe must pass every validator for this part
  (:func:`rtt.effectors.morphology.validators.validate_recipe`: bounds, printability,
  self-intersection, mass, actuation budget, opening, reach, closing sweep); otherwise
  :class:`~rtt.effectors.types.InvalidEffector` carries the reasons and the episode scores
  ``invalid_build``. Then :func:`rtt.effectors.morphology.generator.add_hand` writes the palm,
  the digits, pads, hooks, tendons and servos under the wrist.
* **Primitive** (:class:`DesignedPrimitive`): the shared synergy (controls move from open to
  closed on the close ramp) plus one closed-loop rule for a per-joint pinch: the last phalanx
  is driven to minus the angles above it, so it stays upright and its pad meets the part flat.
  It reads joint positions only and acts only through the returned controls.
* **Criterion**: the shared :class:`~rtt.effectors.primitives.ContactCriterion` (at least two
  effector bodies on the part with forces at least 90 deg apart).
* **Registration**: ``nominal_episode`` (repeatability against the design's own nominal
  certificate episode), as for every family but the jaws.

Actuation limits (docs/EFFECTORS.md 2.4): at most 50 N total closing force at the nominal
contact lever and 1.5 N m per joint; every limit is printed on the card
(:meth:`~rtt.effectors.morphology.layout.HandLayout.actuation_card`).

:meth:`DesignedFamily.presets` is empty on purpose: the tournament's designed group holds
Nemotron proposals and its random group the control's draws; manual reference designs for
tests and demos are :func:`reference_recipes`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import mujoco
import numpy as np

from rtt.effectors.morphology import generator as gen
from rtt.effectors.morphology.layout import HandLayout, layout_of
from rtt.effectors.morphology.spaces import DESIGNED_KINDS, space_for
from rtt.effectors.primitives import ContactCriterion, part_height
from rtt.effectors.registry import EffectorFamilyImpl, register_family
from rtt.effectors.types import (
    BuildContext,
    EffectorBuild,
    EffectorFamily,
    EffectorSpec,
    InvalidEffector,
    MorphologyRecipe,
    ParamSpec,
    PrimitivePlan,
)

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec
    from rtt.sim.controller import Phase
    from rtt.sim.objects import Placement
    from rtt.sim.task import TaskV2

__all__ = [
    "DesignedFamily",
    "DesignedPrimitive",
    "display_name_of",
    "reference_recipes",
    "spec_from_recipe",
]

_NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
_ORDINAL_JOINTS = {1: "1 joint", 2: "2 joints", 3: "3 joints"}


def display_name_of(recipe: MorphologyRecipe) -> str:
    """A brand-free on-screen name (at most 60 characters) with a short id for uniqueness."""
    p = recipe.params
    short = recipe.recipe_id[:4]
    if recipe.kind == "fingered_hand":
        n, ph = int(p["n_fingers"]), int(p["n_phalanges"])
        base = f"Designed {_NUMBERS[n]}-finger hand, {_ORDINAL_JOINTS[ph]} per finger"
    elif recipe.kind == "hook_gripper":
        n = int(p["n_fingers"])
        what = "sliding" if p["motion"] == "slide" else "swinging"
        hook = "hook" if float(p["hook_depth_mm"]) > 0 else "pad"
        base = f"Designed {_NUMBERS[n]}-finger {what} {hook} gripper"
    else:
        n = int(p["n_digits"])
        base = f"Designed {_NUMBERS[n]}-digit segmented tentacle"
    return f"{base} {short}"


def spec_from_recipe(recipe: MorphologyRecipe) -> EffectorSpec:
    """The :class:`EffectorSpec` of a designed recipe (its origin decides the tournament group)."""
    if recipe.family is not EffectorFamily.DESIGNED:
        raise ValueError(f"not a designed recipe: family {recipe.family.value!r}")
    if recipe.kind not in DESIGNED_KINDS:
        raise ValueError(f"unknown designed kind {recipe.kind!r}; known: {DESIGNED_KINDS}")
    p = recipe.params
    if recipe.kind == "fingered_hand":
        primitive = str(p.get("grasp", "pinch"))
    elif recipe.kind == "hook_gripper":
        primitive = "pinch"
    else:
        primitive = "power"
    origin = recipe.origin if recipe.origin in ("nemotron", "random") else "manual"
    return EffectorSpec(
        family=EffectorFamily.DESIGNED,
        model=f"{recipe.kind.replace('_', '-')}-{recipe.recipe_id[:12]}",
        display_name=display_name_of(recipe),
        primitive=primitive,
        params={},
        recipe=recipe,
        origin=origin,  # type: ignore[arg-type]
    )


def reference_recipes() -> tuple[MorphologyRecipe, ...]:
    """Hand-written designs, one or two per generator (tests, demos, determinism gates)."""
    return (
        MorphologyRecipe(
            EffectorFamily.DESIGNED,
            "fingered_hand",
            {
                "n_fingers": 3,
                "n_phalanges": 2,
                "layout": "even",
                "spread_deg": 60.0,
                "palm_radius_mm": 42.0,
                "palm_thickness_mm": 8.0,
                "finger_length_mm": 80.0,
                "proximal_frac": 0.55,
                "finger_width_mm": 16.0,
                "finger_thickness_mm": 8.0,
                "pad": "flat",
                "pad_thickness_mm": 3.0,
                "actuation": "per_joint",
                "grasp": "pinch",
                "material": "pla",
                "grip_force_n": 36.0,
                "grip_height_frac": 0.35,
                "yaw_deg": 0.0,
            },
            hypothesis="Three upright-tip fingers centre the part from three sides.",
        ),
        MorphologyRecipe(
            EffectorFamily.DESIGNED,
            "fingered_hand",
            {
                "n_fingers": 4,
                "n_phalanges": 3,
                "layout": "opposed",
                "spread_deg": 70.0,
                "palm_radius_mm": 45.0,
                "palm_thickness_mm": 8.0,
                "finger_length_mm": 90.0,
                "proximal_frac": 0.45,
                "finger_width_mm": 14.0,
                "finger_thickness_mm": 8.0,
                "pad": "round",
                "pad_thickness_mm": 3.0,
                "actuation": "tendon",
                "grasp": "power",
                "material": "pla",
                "grip_force_n": 40.0,
                "grip_height_frac": 0.2,
                "yaw_deg": 0.0,
            },
            hypothesis="An underactuated thumb-and-three hand wraps the part.",
        ),
        MorphologyRecipe(
            EffectorFamily.DESIGNED,
            "hook_gripper",
            {
                "n_fingers": 3,
                "layout": "even",
                "spread_deg": 60.0,
                "motion": "slide",
                "palm_radius_mm": 40.0,
                "palm_thickness_mm": 8.0,
                "finger_length_mm": 60.0,
                "finger_width_mm": 14.0,
                "finger_thickness_mm": 8.0,
                "pad_thickness_mm": 2.0,
                "hook_depth_mm": 2.0,
                "hook_height_mm": 4.0,
                "material": "pla",
                "grip_force_n": 45.0,
                "grip_height_frac": 0.15,
                "yaw_deg": 0.0,
            },
            hypothesis="A three-jaw chuck with small hooks centres round and hex parts.",
        ),
        MorphologyRecipe(
            EffectorFamily.DESIGNED,
            "segmented_tentacle",
            {
                "n_digits": 3,
                "n_segments": 5,
                "layout": "even",
                "spread_deg": 60.0,
                "palm_radius_mm": 40.0,
                "palm_thickness_mm": 8.0,
                "digit_length_mm": 80.0,
                "root_radius_mm": 6.0,
                "taper": 0.6,
                "curl_bias": 0.3,
                "material": "pla",
                "grip_force_n": 30.0,
                "grip_height_frac": 0.2,
                "yaw_deg": 0.0,
            },
            hypothesis="Three curling digits wrap the part from below its middle.",
        ),
    )


@dataclass
class DesignedPrimitive:
    """The designed family's grasp primitive (module docstring).

    ``upright``: per actuator index, the (digit, joint) pairs whose measured angles it cancels.
    Joint addresses are resolved from the model on first use.
    """

    name: str
    ctrl_open: tuple[float, ...]
    ctrl_closed: tuple[float, ...]
    grip_height_frac: float
    yaw_offset: float
    upright: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] = ()
    prefix: str = "eff_"
    _addr: dict[int, tuple[np.ndarray, float, float]] | None = field(default=None, repr=False)

    def plan(self, part: PartSpec, placement: Placement, task: TaskV2, build: EffectorBuild) -> PrimitivePlan:
        if len(self.ctrl_open) != build.n_actuators:
            raise ValueError(f"{self.name}: {len(self.ctrl_open)} controls for {build.n_actuators} actuators")
        tip = max(task.controller.min_tip_clearance, self.grip_height_frac * part_height(part))
        return PrimitivePlan(tip_height=tip, yaw_offset=self.yaw_offset, note=f"designed {self.name}")

    def _resolve(self, model: mujoco.MjModel) -> dict[int, tuple[np.ndarray, float, float]]:
        out = {}
        for a, pairs in self.upright:
            adr = np.array(
                [model.jnt_qposadr[model.joint(gen.joint_name(self.prefix, d, j)).id] for d, j in pairs],
                dtype=np.int64,
            )
            act = model.actuator(4 + a)
            out[a] = (adr, float(act.ctrlrange[0]), float(act.ctrlrange[1]))
        return out

    def command(
        self, t: float, phase: Phase, closure: float, model: mujoco.MjModel, data: mujoco.MjData, scene: Any
    ) -> np.ndarray:
        a = np.asarray(self.ctrl_open, dtype=np.float64)
        u = a + closure * (np.asarray(self.ctrl_closed, dtype=np.float64) - a)
        if self.upright:
            if self._addr is None:
                self._addr = self._resolve(model)
            for idx, (adr, lo, hi) in self._addr.items():
                u[idx] = min(max(-float(data.qpos[adr].sum()), lo), hi)
        return u


class DesignedFamily(EffectorFamilyImpl):
    """Our own fingered, hooked and segmented morphologies (module docstring)."""

    family: ClassVar[EffectorFamily] = EffectorFamily.DESIGNED
    reference_mode: ClassVar[Literal["analytic", "nominal_episode"]] = "nominal_episode"

    def presets(self) -> tuple[EffectorSpec, ...]:
        return ()

    def recipe_kinds(self) -> tuple[str, ...]:
        return DESIGNED_KINDS

    def param_space(self, kind: str) -> tuple[ParamSpec, ...]:
        if kind not in DESIGNED_KINDS:
            return ()
        return space_for(EffectorFamily.DESIGNED, kind)

    def spec_from_recipe(self, recipe: MorphologyRecipe) -> EffectorSpec:
        return spec_from_recipe(recipe)

    def _layout(self, spec: EffectorSpec) -> HandLayout:
        if spec.recipe is None:
            raise InvalidEffector("a designed effector needs its MorphologyRecipe")
        return layout_of(spec.recipe)

    def build(self, ctx: BuildContext) -> EffectorBuild:
        from rtt.effectors.morphology.validators import validate_recipe

        spec = ctx.effector
        if spec.recipe is None:
            raise InvalidEffector("a designed effector needs its MorphologyRecipe")
        problems = validate_recipe(spec.recipe, ctx.part, ctx.task)
        if problems:
            raise InvalidEffector("; ".join(problems))
        layout = layout_of(spec.recipe)
        names = gen.add_hand(
            ctx.spec, ctx.mount, layout, prefix=ctx.prefix, z0=ctx.mount_z, wrist=ctx.mount.name
        )
        tip_z = ctx.mount_z - layout.reach
        notes = [
            (
                f"{spec.recipe.kind.replace('_', ' ')}: {layout.n_digits} digits, {layout.n_joints} joints, "
                f"{layout.n_actuators} servos ({layout.actuation.replace('_', ' ')})"
            ),
            f"material {layout.material}; rigid links, scripted {spec.primitive} synergy (no tactile feedback)",
        ]
        return EffectorBuild(
            spec=spec,
            bodies=names.bodies,
            actuators=names.actuators,
            ctrl_open=layout.ctrl_open(),
            reach=layout.reach,
            tcp=(0.0, 0.0, tip_z + 0.5 * layout.finger_drop),
            actuation=layout.actuation_card(),
            notes=tuple(notes),
            max_opening=layout.max_opening,
        )

    def primitive(self, spec: EffectorSpec, build: EffectorBuild) -> DesignedPrimitive:
        layout = self._layout(spec)
        return DesignedPrimitive(
            name=spec.primitive,
            ctrl_open=layout.ctrl_open(),
            ctrl_closed=layout.ctrl_closed(),
            grip_height_frac=layout.grip_height_frac,
            yaw_offset=layout.yaw,
            upright=layout.upright_rules(),
        )

    def criterion(self, spec: EffectorSpec, build: EffectorBuild) -> ContactCriterion:
        return ContactCriterion()


register_family(DesignedFamily())
