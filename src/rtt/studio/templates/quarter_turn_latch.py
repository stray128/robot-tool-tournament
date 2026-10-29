"""The ``quarter_turn_latch`` template: a slot or wing head on a vertical revolute joint (track S2).

Fidelity: **approximation** (the template's caveat: a detent torque on a revolute joint, no cam
or spring geometry).

* **Panel** (static): a square plate on the table, ``max(1.6 x head, 40 mm)`` across and
  :data:`PANEL_T` thick; the head turns on its top face.
* **Head** (one body on a hinge about the vertical): **opening is counter-clockwise seen from
  above** (the common quarter-turn convention, lefty-loosey); the joint range is
  ``[0, turn_deg]``, so the stop at ``turn_deg`` is the open position and 0 is locked. A twist
  step that turns clockwise drives the head into its locked stop.

  * ``wing``: a round boss (half the head diameter, 30 % of its height) with a fin across the
    full ``head_diameter_mm`` and ``head_height_mm``, 20 % of the diameter thick (at least
    4 mm) [assumed]: a gripper closes across the fin.
  * ``slot``: a round head with a slot across its top (width 30 % of the diameter, at least
    2 mm; depth half the head height) [assumed]: it needs a blade; fingers that grip the round
    head hold it by friction only.

* **Torque to turn** (``torque_nm``): without a detent it is the joint's dry friction (MuJoCo
  friction loss) over the whole turn. With ``detent``, a quarter of it is the running friction
  and the rest is a detent at the locked position: the template's own motor actuator applies a
  torque toward locked of ``0.75 x torque_nm`` over the first :data:`DETENT_DEG` of the turn
  (and pulls a head released inside that band back to locked). Breakaway = ``torque_nm`` either
  way [assumed split].

Sites: ``head`` (the top centre of the head) and the auxiliary ``mark`` (on the head's local
``+x`` at its rim and top: the fin's end or the slot's end, so a controller sees the head's
rotation). ``info``: the head code (0 slot, 1 wing), sizes, ``turn_rad`` and the panel top.

State: ``angle_deg``, the head's turn from locked (counter-clockwise positive), within its stops.
"""

from __future__ import annotations

import math
from typing import ClassVar, Final

import mujoco
import numpy as np

from rtt.sim.scene import OBJECT_RGBA
from rtt.studio.templates import (
    FRAME_RGBA,
    FRICTION_SOLIMP,
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
    quat_z,
    register_template_builder,
)
from rtt.studio.types import TemplateKind

__all__ = ["DETENT_DEG", "PANEL_T", "QuarterTurnLatchBuilder"]

PANEL_T: Final = 0.008  # m [assumed]
DETENT_DEG: Final = 8.0  # deg [assumed]
RUNNING_FRACTION: Final = 0.25  # of torque_nm when there is a detent [assumed]
FIN_FRACTION: Final = 0.20
SLOT_FRACTION: Final = 0.30
DENSITY: Final = 2000.0  # kg/m^3, a filled polymer or zinc-alloy head [assumed]


def _segment_prism(r: float, y0: float, z0: float, z1: float, n: int = 10) -> list[float]:
    """Vertices of the convex prism under a circle of radius ``r`` cut by the chord ``y = y0``
    (the side ``y >= y0``), from ``z0`` to ``z1``."""
    a0 = math.asin(max(-1.0, min(1.0, y0 / r)))
    angles = np.linspace(a0, math.pi - a0, n)
    verts: list[float] = []
    for z in (z0, z1):
        for a in angles:
            verts += [r * math.cos(a), r * math.sin(a), z]
    return verts


class QuarterTurnLatchBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.QUARTER_TURN_LATCH

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        dia, hh = float(prm["head_diameter_mm"]) / 1000.0, float(prm["head_height_mm"]) / 1000.0
        r = 0.5 * dia
        torque = float(prm["torque_nm"])
        detent = bool(prm["detent"])
        wing = prm["head"] == "wing"
        yaw = math.radians(pose.yaw_deg)
        side = max(1.6 * dia, 0.040)
        panel = ctx.spec.worldbody.add_body(
            name=f"{p}panel",
            pos=[pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z],
            quat=quat_z(yaw),
        )
        panel.add_geom(
            name=f"{p}plate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * side, 0.5 * side, 0.5 * PANEL_T],
            pos=[0.0, 0.0, 0.5 * PANEL_T],
            priority=1,
            rgba=list(FRAME_RGBA),
        )
        geoms = [f"{p}plate"]
        head = panel.add_body(name=f"{p}head", pos=[0.0, 0.0, PANEL_T + 0.0005])
        head.add_joint(
            name=f"{p}hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0.0, 0.0, 1.0],
            range=[0.0, math.radians(float(prm["turn_deg"]))],
            frictionloss=torque * (RUNNING_FRACTION if detent else 1.0),
            damping=0.002,
            armature=0.0002,
            solref_friction=list(ctx.task.physics.limit_solref),
            solimp_friction=list(FRICTION_SOLIMP),
        )

        def add(name: str, **kw: object) -> None:
            head.add_geom(name=f"{p}{name}", density=DENSITY, priority=1, rgba=list(OBJECT_RGBA), **kw)
            geoms.append(f"{p}{name}")

        if wing:
            fin_t = max(0.004, FIN_FRACTION * dia)
            boss_h = 0.3 * hh
            add(
                "boss",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.25 * dia, 0.5 * boss_h, 0.0],
                pos=[0, 0, 0.5 * boss_h],
            )
            add("fin", type=mujoco.mjtGeom.mjGEOM_BOX, size=[r, 0.5 * fin_t, 0.5 * hh], pos=[0, 0, 0.5 * hh])
            slot_w = 0.0
        else:
            slot_w = max(0.002, SLOT_FRACTION * dia)
            slot_d = 0.5 * hh
            add(
                "core",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[r, 0.5 * (hh - slot_d), 0.0],
                pos=[0, 0, 0.5 * (hh - slot_d)],
            )
            for k, sign in enumerate((1.0, -1.0)):
                mesh = f"{p}cheek{k}"
                verts = np.array(_segment_prism(r, 0.5 * slot_w, hh - slot_d, hh)).reshape(-1, 3)
                verts[:, 1] *= sign
                ctx.spec.add_mesh(name=mesh, uservert=verts.ravel().tolist())
                add(f"cheek{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh)
            fin_t = 0.0
        head.add_site(name=f"{p}head", pos=[0.0, 0.0, hh], size=[0.003, 0.0, 0.0])
        head.add_site(name=f"{p}mark", pos=[r, 0.0, hh], size=[0.003, 0.0, 0.0])
        act = ctx.spec.add_actuator(name=f"{p}detent", target=f"{p}hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-torque - 0.01, torque + 0.01]
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}panel", f"{p}head"),
            geoms=tuple(geoms),
            joints={"hinge": f"{p}hinge"},
            sites={"head": f"{p}head", "mark": f"{p}mark"},
            actuators=(f"{p}detent",),
            info={
                "head": float(wing),
                "diameter_m": dia,
                "height_m": hh,
                "fin_m": fin_t,
                "slot_m": slot_w,
                "turn_rad": math.radians(float(prm["turn_deg"])),
                "detent_nm": torque * (1.0 - RUNNING_FRACTION) if detent else 0.0,
                "torque_nm": torque,
                "base_z": float(ctx.table_top_z + PANEL_T + 0.0005),
                "panel_m": side,
            },
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        angle = float(data.qpos[obj.joint_qpos["hinge"]])  # within the joint's own stops
        return {"angle_deg": math.degrees(min(max(angle, 0.0), obj.handles.info["turn_rad"]))}

    def passive(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        detent = obj.handles.info["detent_nm"]
        if detent <= 0.0:
            return np.zeros(1)
        deg = math.degrees(float(data.qpos[obj.joint_qpos["hinge"]]))
        return np.array([-detent if deg < DETENT_DEG else 0.0])


register_template_builder(QuarterTurnLatchBuilder())
