"""The ``handwheel_valve`` template: a handwheel on a vertical stem (track S2).

Fidelity: **approximation** (the template's caveat: a resistive torque curve on a revolute
joint; no packing or seat dynamics).

* **Valve body** (static): a box on the table, a bonnet and a stem up to the wheel, whose
  mid-plane is :data:`WHEEL_Z` above the table [assumed: the template has no height param].
* **Wheel** (one body on a hinge about the vertical): **opening is counter-clockwise seen from
  above** (lefty-loosey); the joint range is ``[0, travel_deg]``, so it starts closed against its
  seat stop and opens to the backseat stop. Geometry: a hub (radius ``max(10 mm, 12 % of the
  rim radius)``, as thick as the rim section), a rim of :data:`RIM_SEGMENTS` capsules of
  diameter ``rim_section_mm`` on the circle of diameter ``rim_diameter_mm`` (outer), and
  ``spokes`` capsules (40 % of the section thick) from the hub to the rim; ``spokes = 0`` is a
  solid disc instead. Density :data:`DENSITY`.
* **Resistive torque curve** (``torque_nm`` = the breakaway): ``c(angle) x torque_nm`` with
  ``c`` = 1.0 at closed, falling linearly to 0.6 over the first 20 % of the travel (running),
  rising to 0.8 over the last 10 % (backseat) [assumed shape]. Half of the torque is the joint's
  dry friction loss (it holds the wheel at rest); the rest of the curve is the template's own
  motor actuator, a velocity-saturated friction (:data:`OMEGA_S`).

Sites: ``wheel`` (the top centre of the hub) and the auxiliary ``spoke0`` (on spoke 0 at the rim
centre line, in the wheel's mid-plane: a controller sees the wheel's rotation). ``info``: the rim
radius and section, hub radius and height, spoke count and radius, ``wheel_z``, ``travel_rad``.

State: ``angle_deg``, how far it is open (counter-clockwise positive).
"""

from __future__ import annotations

import math
from typing import ClassVar, Final

import mujoco
import numpy as np

from rtt.sim.scene import OBJECT_RGBA
from rtt.studio.templates import (
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
    register_template_builder,
)
from rtt.studio.types import TemplateKind

__all__ = ["OMEGA_S", "WHEEL_Z", "HandwheelValveBuilder", "torque_curve"]

WHEEL_Z: Final = 0.110  # m, wheel mid-plane above the table [assumed]
RIM_SEGMENTS: Final = 24
DENSITY: Final = 2700.0  # kg/m^3, a cast aluminium wheel [assumed]
OMEGA_S: Final = 0.1  # rad/s [assumed]
STATIC_FRACTION: Final = 0.5
FRAME_RGBA: Final = (0.55, 0.58, 0.62, 1.0)
FRICTION_SOLIMP: Final = (0.999, 0.9999, 0.0001, 0.5, 2.0)  # a near-hard dry friction (no creep)


def torque_curve(angle: float, travel: float) -> float:
    """The resistive torque as a fraction of ``torque_nm`` at ``angle`` of ``travel`` (rad)."""
    u = angle / travel if travel > 0 else 0.0
    if u >= 0.9:
        return 0.6 + 0.2 * min(1.0, (u - 0.9) / 0.1)
    return 1.0 - 0.4 * min(1.0, max(0.0, u) / 0.2)


def _quat_z(angle: float) -> list[float]:
    return [math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)]


class HandwheelValveBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.HANDWHEEL_VALVE

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        big_r = 0.5 * float(prm["rim_diameter_mm"]) / 1000.0
        sec = float(prm["rim_section_mm"]) / 1000.0
        n_spokes = int(prm["spokes"])
        torque = float(prm["torque_nm"])
        travel = math.radians(float(prm["travel_deg"]))
        rc = big_r - 0.5 * sec  # rim centre line
        r_hub = max(0.010, 0.12 * big_r)
        hub_h = max(sec, 0.010)
        yaw = math.radians(pose.yaw_deg)
        body_h = 0.050
        frame = ctx.spec.worldbody.add_body(
            name=f"{p}body",
            pos=[pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z],
            quat=_quat_z(yaw),
        )
        geoms: list[str] = []

        def static(name: str, **kw: object) -> None:
            frame.add_geom(name=f"{p}{name}", priority=1, rgba=list(FRAME_RGBA), **kw)
            geoms.append(f"{p}{name}")

        static(
            "valve",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.035, 0.025, 0.5 * body_h],
            pos=[0, 0, 0.5 * body_h],
        )
        stem_top = WHEEL_Z - 0.5 * hub_h - 0.001
        static(
            "bonnet",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.006, 0.5 * (stem_top - body_h), 0.0],
            pos=[0, 0, 0.5 * (stem_top + body_h)],
        )
        wheel = frame.add_body(name=f"{p}wheel", pos=[0.0, 0.0, WHEEL_Z])
        wheel.add_joint(
            name=f"{p}hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0.0, 0.0, 1.0],
            range=[0.0, travel],
            frictionloss=STATIC_FRACTION * torque,
            damping=0.005,
            armature=0.0005,
            solref_friction=list(ctx.task.physics.limit_solref),
            solimp_friction=list(FRICTION_SOLIMP),
        )

        def moving(name: str, **kw: object) -> None:
            wheel.add_geom(name=f"{p}{name}", density=DENSITY, priority=1, rgba=list(OBJECT_RGBA), **kw)
            geoms.append(f"{p}{name}")

        moving("hub", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r_hub, 0.5 * hub_h, 0.0], pos=[0, 0, 0])
        if n_spokes == 0:
            moving("disc", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[big_r, 0.5 * sec, 0.0], pos=[0, 0, 0])
            spoke_r = 0.0
        else:
            for k in range(RIM_SEGMENTS):
                a0, a1 = (2.0 * math.pi * (k + j) / RIM_SEGMENTS for j in (0, 1))
                moving(
                    f"rim{k:02d}",
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    fromto=[
                        rc * math.cos(a0),
                        rc * math.sin(a0),
                        0.0,
                        rc * math.cos(a1),
                        rc * math.sin(a1),
                        0.0,
                    ],
                    size=[0.5 * sec, 0.0, 0.0],
                )
            spoke_r = max(0.0025, 0.2 * sec)
            for k in range(n_spokes):
                a = 2.0 * math.pi * k / n_spokes
                c, s = math.cos(a), math.sin(a)
                moving(
                    f"spoke{k}",
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    fromto=[0.9 * r_hub * c, 0.9 * r_hub * s, 0.0, rc * c, rc * s, 0.0],
                    size=[spoke_r, 0.0, 0.0],
                )
        wheel.add_site(name=f"{p}wheel", pos=[0.0, 0.0, 0.5 * hub_h], size=[0.003, 0.0, 0.0])
        wheel.add_site(name=f"{p}spoke0", pos=[rc, 0.0, 0.0], size=[0.003, 0.0, 0.0])
        act = ctx.spec.add_actuator(name=f"{p}curve", target=f"{p}hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-torque - 0.01, torque + 0.01]
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}body", f"{p}wheel"),
            geoms=tuple(geoms),
            joints={"hinge": f"{p}hinge"},
            sites={"wheel": f"{p}wheel", "spoke0": f"{p}spoke0"},
            actuators=(f"{p}curve",),
            info={
                "rim_m": big_r,
                "section_m": sec,
                "rim_centre_m": rc,
                "hub_m": r_hub,
                "hub_h_m": hub_h,
                "spokes": float(n_spokes),
                "spoke_m": spoke_r,
                "wheel_z": float(ctx.table_top_z + WHEEL_Z),
                "travel_rad": travel,
                "torque_nm": torque,
            },
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        return {"angle_deg": math.degrees(float(data.qpos[obj.joint_qpos["hinge"]]))}

    def passive(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        info = obj.handles.info
        angle = float(data.qpos[obj.joint_qpos["hinge"]])
        omega = float(data.qvel[obj.joint_dof["hinge"]])
        extra = (torque_curve(angle, info["travel_rad"]) - STATIC_FRACTION) * info["torque_nm"]
        return np.array([-max(0.0, extra) * math.tanh(omega / OMEGA_S)])


register_template_builder(HandwheelValveBuilder())
