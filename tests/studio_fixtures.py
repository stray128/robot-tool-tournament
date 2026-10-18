"""Fixtures for the Studio contract tests: effectors, a dummy task and test-only template stand-ins.

The stand-ins (:class:`ToyKnob`, :class:`ToyLid`, :class:`ToySocket`) are the simplest models of a
latch, a lid and a socket, so every S1 reference controller can run once. They are registered
with :func:`rtt.studio.templates.override_template_builder` inside a test only; the real
articulated templates are track S2's.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import mujoco

from rtt.effectors.registry import get_family
from rtt.effectors.types import EffectorSpec
from rtt.studio.templates import (
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
)
from rtt.studio.types import (
    ObjectInstance,
    ObjectPose,
    Step,
    SuccessCriterion,
    TaskSpec,
    TemplateKind,
)


def preset(family: str, model: str, primitive: str) -> EffectorSpec:
    return next(s for s in get_family(family).presets() if s.model == model and s.primitive == primitive)


def gripper() -> EffectorSpec:
    return preset("catalog_gripper", "adaptive-2f", "pinch")


def cup() -> EffectorSpec:
    return preset("suction", "cup-16mm", "seal")


def two_press_task() -> TaskSpec:
    """Two dummy push steps on two parts (and a tote that nothing touches)."""
    objects = (
        ObjectInstance.of(
            "carton",
            "part",
            "Carton",
            ObjectPose(100.0, 0.0, 0.0),
            size_x_mm=30,
            size_y_mm=40,
            size_z_mm=60,
            mass_g=150,
        ),
        ObjectInstance.of(
            "flat",
            "part",
            "Flat item",
            ObjectPose(-100.0, 50.0, 0.0),
            size_x_mm=80,
            size_y_mm=60,
            size_z_mm=15,
            mass_g=60,
        ),
        ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0.0, -200.0, 0.0)),
    )
    upright = [SuccessCriterion("tilt_deg", at_most=30.0)]
    steps = (
        Step.of("press_carton", "push", "carton", "part", "Press the carton down", upright),
        Step.of("press_flat", "push", "flat", "part", "Press the flat item down", upright),
    )
    return TaskSpec("Two presses", "Two dummy steps for the contract tests.", objects, steps, source="test")


def _yaw_quat(deg: float) -> list[float]:
    a = math.radians(deg)
    return [math.cos(0.5 * a), 0.0, 0.0, math.sin(0.5 * a)]


def _pos(ctx: TemplateContext, z: float = 0.0) -> list[float]:
    return [ctx.instance.pose.x_mm / 1000.0, ctx.instance.pose.y_mm / 1000.0, ctx.table_top_z + z]


class ToyKnob(TemplateBuilder):
    """A latch stand-in: a cylinder head on a vertical hinge with friction, on a fixed base."""

    kind: ClassVar[TemplateKind] = TemplateKind.QUARTER_TURN_LATCH

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        p, prm = ctx.prefix, ctx.instance.params
        r, h = 0.5e-3 * float(prm["head_diameter_mm"]), 1e-3 * float(prm["head_height_mm"])
        base = ctx.spec.worldbody.add_body(
            name=f"{p}base", pos=_pos(ctx, 0.01), quat=_yaw_quat(ctx.instance.pose.yaw_deg)
        )
        base.add_geom(
            name=f"{p}plate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.04, 0.04, 0.01],
            pos=[0, 0, -0.005],
            priority=1,
        )
        head = base.add_body(name=f"{p}head", pos=[0.0, 0.0, 0.0])
        head.add_joint(
            name=f"{p}hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            range=[-math.radians(float(prm["turn_deg"])), math.radians(float(prm["turn_deg"]))],
            frictionloss=float(prm["torque_nm"]),
            damping=0.01,
        )
        head.add_geom(
            name=f"{p}knob",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[r, 0.5 * h, 0],
            pos=[0, 0, 0.5 * h],
            mass=0.05,
            priority=1,
        )
        head.add_site(name=f"{p}site", pos=[0, 0, h], size=[0.003, 0, 0])
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}base", f"{p}head"),
            geoms=(f"{p}plate", f"{p}knob"),
            joints={"hinge": f"{p}hinge"},
            sites={"head": f"{p}site"},
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        return {"angle_deg": math.degrees(float(data.qpos[obj.joint_qpos["hinge"]]))}


class ToyLid(TemplateBuilder):
    """A lid stand-in: a panel on a hinge along its back edge, with a lip site on the front edge."""

    kind: ClassVar[TemplateKind] = TemplateKind.HINGED_LID

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        p, prm = ctx.prefix, ctx.instance.params
        w, d, t = (1e-3 * float(prm[k]) for k in ("width_mm", "depth_mm", "thickness_mm"))
        frame = ctx.spec.worldbody.add_body(
            name=f"{p}frame", pos=_pos(ctx, 0.08), quat=_yaw_quat(ctx.instance.pose.yaw_deg)
        )
        frame.add_geom(
            name=f"{p}post",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * w, 0.01, 0.04],
            pos=[0, -0.5 * d - 0.01, -0.04],
            priority=1,
        )
        lid = frame.add_body(name=f"{p}lid", pos=[0.0, -0.5 * d, 0.0])
        lid.add_joint(
            name=f"{p}hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[0, 1.9], damping=0.05
        )
        lid.add_geom(
            name=f"{p}panel",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * w, 0.5 * d, 0.5 * t],
            pos=[0, 0.5 * d, 0.5 * t],
            mass=0.2,
            priority=1,
        )
        lid.add_site(name=f"{p}lip", pos=[0.0, d, t], size=[0.003, 0, 0])
        lid.add_site(name=f"{p}top", pos=[0.0, 0.5 * d, t], size=[0.003, 0, 0])
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}frame", f"{p}lid"),
            geoms=(f"{p}post", f"{p}panel"),
            joints={"hinge": f"{p}hinge"},
            sites={"handle": f"{p}lip", "lid": f"{p}top"},
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        return {"angle_deg": math.degrees(float(data.qpos[obj.joint_qpos["hinge"]]))}


class ToySocket(TemplateBuilder):
    """A port-and-plug stand-in: a fixed block (socket mouth site on top) and a free plug beside it."""

    kind: ClassVar[TemplateKind] = TemplateKind.PORT_PLUG

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        p, prm = ctx.prefix, ctx.instance.params
        r, length = 0.5e-3 * float(prm["plug_diameter_mm"]), 1e-3 * float(prm["plug_length_mm"])
        block = ctx.spec.worldbody.add_body(name=f"{p}block", pos=_pos(ctx, 0.02))
        block.add_geom(name=f"{p}block", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.02], priority=1)
        block.add_site(name=f"{p}mouth", pos=[0, 0, 0.02], size=[0.003, 0, 0])
        plug = ctx.spec.worldbody.add_body(
            name=f"{p}plug", pos=[_pos(ctx)[0] + 0.08, _pos(ctx)[1], 0.5 * length]
        )
        plug.add_freejoint(name=f"{p}free")
        plug.add_geom(
            name=f"{p}plug",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[r, 0.5 * length, 0],
            mass=0.02,
            priority=1,
        )
        plug.add_site(name=f"{p}grip", pos=[0, 0, 0.5 * length], size=[0.003, 0, 0])
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}block", f"{p}plug"),
            geoms=(f"{p}block", f"{p}plug"),
            joints={"free": f"{p}free"},
            sites={"plug": f"{p}grip", "socket": f"{p}mouth"},
            info={"plug_length_m": length},
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        mouth = data.site_xpos[obj.site_ids["socket"]][2]
        bottom = data.xpos[obj.body_ids[1]][2] - 0.5 * obj.handles.info["plug_length_m"]
        return {"depth_mm": max(0.0, 1000.0 * float(mouth - bottom))}


def toy_task(kind: str) -> tuple[TaskSpec, Any]:
    """A one-step task for a stand-in, and its builder."""
    if kind == "twist":
        obj = ObjectInstance.of("latch", "quarter_turn_latch", "Latch", ObjectPose(0.0, 0.0, 0.0))
        step = Step.of(
            "turn", "twist", "latch", "head", "Turn the latch", [SuccessCriterion("torque_nm", at_most=100.0)]
        )
        builder: TemplateBuilder = ToyKnob()
    elif kind == "hook_pull":
        obj = ObjectInstance.of(
            "lid", "hinged_lid", "Lid", ObjectPose(0.0, 0.0, 0.0), width_mm=150, depth_mm=100
        )
        step = Step.of(
            "open",
            "hook_pull",
            "lid",
            "handle",
            "Pull the lid open",
            [SuccessCriterion("force_n", at_most=1000.0)],
        )
        builder = ToyLid()
    else:
        obj = ObjectInstance.of("port", "port_plug", "Port", ObjectPose(0.0, 0.0, 0.0))
        step = Step.of(
            "plug_in",
            "insert",
            "port",
            "plug",
            "Insert the plug",
            [SuccessCriterion("force_n", at_most=1000.0)],
        )
        builder = ToySocket()
    return TaskSpec(f"Toy {kind}", "", (obj,), (step,), source="test"), builder
