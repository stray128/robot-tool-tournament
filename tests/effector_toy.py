"""A toy native effector family for the contract tests: two box pads on independent slides.

It exercises the native path end to end (scene build, primitive, criterion, runner,
certificate, replay) without any vendored model. Tests register it with
:func:`rtt.effectors.registry.override_family` (under the catalog-gripper family id), so it never
leaks into other tests or into worker processes.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import mujoco

from rtt.effectors import (
    BuildContext,
    EffectorBuild,
    EffectorFamily,
    EffectorFamilyImpl,
    EffectorSpec,
    InvalidEffector,
)
from rtt.effectors.primitives import ContactCriterion, SynergyPrimitive

STROKE = 0.04  # m per pad
OPEN = -0.002
CLOSED = STROKE + 0.01  # overdrive so the force saturates
OPENING = 0.074  # m between the pads when open
TOY_MASS = 0.052  # kg: base 30 x 120 x 10 mm + two 20 x 8 x 50 mm pads at 1000 kg/m^3


def toy_spec(force_n: float = 25.0) -> EffectorSpec:
    return EffectorSpec(
        family=EffectorFamily.CATALOG_GRIPPER,
        model="toy-two-pad",
        display_name="Two-pad test gripper",
        primitive="pinch",
        params={"force_n": force_n},
        origin="manual",
    )


class ToyTwoPad(EffectorFamilyImpl):
    family: ClassVar[EffectorFamily] = EffectorFamily.CATALOG_GRIPPER
    reference_mode: ClassVar[Literal["analytic", "nominal_episode"]] = "nominal_episode"

    def presets(self) -> tuple[EffectorSpec, ...]:
        return (toy_spec(),)

    def build(self, ctx: BuildContext) -> EffectorBuild:
        p = ctx.prefix
        force = float(ctx.effector.params["force_n"])
        if ctx.part.size[1] > OPENING - 0.004:
            raise InvalidEffector(f"the part is wider than the {OPENING * 1000:.0f} mm opening")
        base = ctx.mount.add_body(name=f"{p}base", pos=[0.0, 0.0, ctx.mount_z])
        base.add_geom(
            name=f"{p}base", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.06, 0.005], pos=[0, 0, -0.005]
        )
        actuators = []
        for side, sign in (("left", 1.0), ("right", -1.0)):
            body = base.add_body(name=f"{p}{side}", pos=[0.0, sign * 0.045, -0.01])
            body.add_joint(
                name=f"{p}{side}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=[0.0, -sign, 0.0],
                range=[0.0, STROKE],
                armature=0.02,
                damping=2.0,
            )
            body.add_geom(
                name=f"{p}{side}_pad",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.01, 0.004, 0.025],
                pos=[0.0, -sign * 0.004, -0.025],
            )
            act = ctx.spec.add_actuator(
                name=f"{p}{side}", target=f"{p}{side}", trntype=mujoco.mjtTrn.mjTRN_JOINT
            )
            act.set_to_position(kp=2000.0, kv=60.0)
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = [OPEN, CLOSED]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-force, force]
            actuators.append(f"{p}{side}")
        return EffectorBuild(
            spec=ctx.effector,
            bodies=(f"{p}base", f"{p}left", f"{p}right"),
            actuators=tuple(actuators),
            ctrl_open=(OPEN, OPEN),
            reach=0.06,
            tcp=(0.0, 0.0, ctx.mount_z - 0.04),
            actuation={"grip_force_n_per_pad": force},
            max_opening=OPENING,
            notes=("test fixture",),
        )

    def primitive(self, spec: EffectorSpec, build: EffectorBuild) -> SynergyPrimitive:
        return SynergyPrimitive("pinch", (OPEN, OPEN), (CLOSED, CLOSED), grip_height_frac=0.5)

    def criterion(self, spec: EffectorSpec, build: EffectorBuild) -> ContactCriterion:
        return ContactCriterion()
