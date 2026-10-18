"""The ``hinged_lid`` template: a lid or a door on a revolute hinge (track S2, fidelity rigid).

Instance frame: origin at the pose on the table top, x along the hinge (lid) or across the door,
rotated by ``yaw_deg``. The **front** (where a tool engages the free edge) faces local ``-y``.

* **Lid** (``orientation = "lid"``): a horizontal panel ``width_mm`` (x) by ``depth_mm`` (y) by
  ``thickness_mm``. Closed, its underside is :data:`LID_HINGE_HEIGHT` above the table (the
  default bin's height, so a lid with default params covers a default bin placed at the same
  pose) [assumed: the template has no hinge-height param]. The hinge runs along the back edge
  (local ``y = +depth/2``) at mid-thickness; opening lifts the front edge. A static back wall
  under the hinge carries it. Gravity acts on the panel, so a lid stays open only past its
  vertical (``max_angle_deg > 90``) or against its stop.
* **Door** (``orientation = "door"``): a vertical panel ``depth_mm`` wide (local x) and
  ``width_mm`` tall (along its vertical hinge), its bottom edge :data:`DOOR_BOTTOM` above the
  table, front face at local ``-y``. The hinge is the left edge (local ``x = -depth/2``); opening
  swings the free edge toward ``-y``. A static jamb post behind the hinge line carries it.

Joint: hinge range ``[0, max_angle_deg]`` (0 is closed, the stop), damping ``damping_nms``,
positive angle = opening. ``open_torque_nm`` is a catch near closed (a magnetic or spring
detent): a torque toward closed of that size over the first :data:`CATCH_DEG` of opening,
fading to zero at twice that angle, applied by the template's own motor actuator
(:meth:`HingedLidBuilder.passive`). It resists opening and snaps a closing panel shut.

Feature at the free edge (``feature``):

* ``lip``: the free edge itself. The ``handle`` site is at the middle of the free edge, at
  mid-thickness (a door's: 40 mm below its top): where a tool pinches or hooks it. For a door, a
  static stop post behind the panel (60 % of its height) sits ``feature_mm`` in from the free
  edge, so the free edge overhangs the frame by ``feature_mm``; a lid's lip overhangs whatever
  it covers.
* ``handle``: a bar handle on the outer face near the free edge, standing ``feature_mm`` off
  the face (the clearance under the bar). Lid: along x, centred 20 mm in from the front edge.
  Door: vertical, 25 mm in from the free edge, on the upper part of the door so a top-down tool
  reaches it. Bar diameter :data:`BAR_DIAMETER`, length :data:`BAR_LENGTH` (at most 60 % of the
  edge) [assumed].

Sites: ``handle`` (above), ``lid`` (the centre of the outer face: where a push closes it) and the
auxiliary ``hinge`` site (a point on the hinge axis, static). ``info`` carries the hinge axis in
the world (``axis_x/y/z``, positive = opening), the closed ``lid`` site and outer-face normal
relative to the hinge site (``lid0_*``, ``out0_*``: a controller turns them by the angle it
observes), the geometry and the codes (``orientation``: 0 lid, 1 door; ``feature``: 0 lip,
1 handle).

State: ``angle_deg``, the hinge angle (0 closed, positive open), within the hinge's stops (the
solver's soft limits let a panel resting on a stop lean a few thousandths of a degree past it).
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

__all__ = ["BAR_DIAMETER", "CATCH_DEG", "DOOR_BOTTOM", "LID_HINGE_HEIGHT", "HingedLidBuilder"]

LID_HINGE_HEIGHT: Final = 0.120  # m, closed lid underside above the table [assumed]
DOOR_BOTTOM: Final = 0.010  # m, door bottom edge above the table [assumed]
PANEL_DENSITY: Final = 700.0  # kg/m^3, a light panel (plywood, filled polymer) [assumed]
BAR_DIAMETER: Final = 0.010  # m [assumed]
BAR_LENGTH: Final = 0.090  # m, at most 60 % of the edge [assumed]
CATCH_DEG: Final = 6.0  # deg, the catch acts over this much opening, fading out by twice it
FRAME_RGBA: Final = (0.55, 0.58, 0.62, 1.0)


def _quat_z(angle: float) -> list[float]:
    return [math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)]


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class HingedLidBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.HINGED_LID

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        w, d, t = (float(prm[k]) / 1000.0 for k in ("width_mm", "depth_mm", "thickness_mm"))
        f = float(prm["feature_mm"]) / 1000.0
        door = prm["orientation"] == "door"
        handle_bar = prm["feature"] == "handle"
        yaw = math.radians(pose.yaw_deg)
        rot = _rot_z(yaw)
        origin = np.array([pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z])
        frame = ctx.spec.worldbody.add_body(name=f"{p}frame", pos=origin.tolist(), quat=_quat_z(yaw))
        geoms: list[str] = []
        sites: dict[str, str] = {}
        r_bar = 0.5 * BAR_DIAMETER

        def static_box(name: str, lo: list[float], hi: list[float]) -> None:
            c = [0.5 * (a + b) for a, b in zip(lo, hi, strict=True)]
            h = [0.5 * (b - a) for a, b in zip(lo, hi, strict=True)]
            frame.add_geom(
                name=f"{p}{name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=h,
                pos=c,
                priority=1,
                rgba=list(FRAME_RGBA),
            )
            geoms.append(f"{p}{name}")

        if not door:
            # hinge axis: along -x through the back edge at mid-thickness (positive angle lifts the front)
            hinge_local = np.array([0.0, 0.5 * d, LID_HINGE_HEIGHT + 0.5 * t])
            axis_local = np.array([-1.0, 0.0, 0.0])
            gap = t
            static_box(
                "back", [-0.5 * w, 0.5 * d + gap, 0.0], [0.5 * w, 0.5 * d + gap + 0.008, LID_HINGE_HEIGHT]
            )
            body = frame.add_body(name=f"{p}panel", pos=hinge_local.tolist())
            # panel frame = hinge frame: the panel spans local y in [-d, 0], z in [-t/2, t/2]
            panel_center = [0.0, -0.5 * d, 0.0]
            outer_n = np.array([0.0, 0.0, 1.0])  # the lid's top face normal (closed)
            if handle_bar:
                length = min(BAR_LENGTH, 0.6 * w)
                yb, zb = -d + 0.020, 0.5 * t + f + r_bar
                handle_pos = [0.0, yb, zb]
                bar_from = [-0.5 * length, yb, zb]
                bar_to = [0.5 * length, yb, zb]
                posts = [
                    ([-0.5 * length, yb, 0.5 * t + 0.5 * (f + r_bar)], [0.004, 0.004, 0.5 * (f + r_bar)]),
                    ([0.5 * length, yb, 0.5 * t + 0.5 * (f + r_bar)], [0.004, 0.004, 0.5 * (f + r_bar)]),
                ]
            else:
                handle_pos = [0.0, -d, 0.0]  # the front edge, mid-thickness
        else:
            hinge_local = np.array([-0.5 * d, 0.0, DOOR_BOTTOM + 0.5 * w])
            axis_local = np.array([0.0, 0.0, -1.0])  # free edge swings toward -y
            hx = -0.5 * d  # the jamb stands behind the hinge line, clear of the swinging panel
            static_box(
                "jamb", [hx - 0.015, 0.5 * t + 0.001, 0.0], [hx + 0.005, 0.5 * t + 0.021, DOOR_BOTTOM + w]
            )
            body = frame.add_body(name=f"{p}panel", pos=hinge_local.tolist())
            # panel frame = hinge frame: the panel spans local x in [0, d], y in [-t/2, t/2], z in [-w/2, w/2]
            panel_center = [0.5 * d, 0.0, 0.0]
            outer_n = np.array([0.0, -1.0, 0.0])
            if handle_bar:
                length = min(BAR_LENGTH * 0.67, 0.6 * w)  # 60 mm on a default door
                xb, yb = d - 0.025, -0.5 * t - f - r_bar
                zc = 0.5 * w - 0.015 - 0.5 * length
                handle_pos = [xb, yb, zc]
                bar_from = [xb, yb, zc - 0.5 * length]
                bar_to = [xb, yb, zc + 0.5 * length]
                posts = [
                    (
                        [xb, -0.5 * t - 0.5 * (f + r_bar), zc - 0.5 * length],
                        [0.004, 0.5 * (f + r_bar), 0.004],
                    ),
                    (
                        [xb, -0.5 * t - 0.5 * (f + r_bar), zc + 0.5 * length],
                        [0.004, 0.5 * (f + r_bar), 0.004],
                    ),
                ]
            else:
                handle_pos = [d, 0.0, 0.5 * w - 0.040]  # the free edge, mid-thickness, 40 mm below the top
                stop_x = 0.5 * d - f  # instance frame: the stop post's free-side face
                static_box(
                    "stop",
                    [stop_x - 0.012, 0.5 * t + 0.0005, 0.0],
                    [stop_x, 0.5 * t + 0.020, DOOR_BOTTOM + 0.6 * w],
                )
        body.add_joint(
            name=f"{p}hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=axis_local.tolist(),
            range=[0.0, math.radians(float(prm["max_angle_deg"]))],
            damping=float(prm["damping_nms"]),
            armature=0.0005,
        )
        size = [0.5 * w, 0.5 * d, 0.5 * t] if not door else [0.5 * d, 0.5 * t, 0.5 * w]
        body.add_geom(
            name=f"{p}panel",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=panel_center,
            density=PANEL_DENSITY,
            priority=1,
            rgba=list(OBJECT_RGBA),
        )
        geoms.append(f"{p}panel")
        if handle_bar:
            body.add_geom(
                name=f"{p}bar",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[*bar_from, *bar_to],
                size=[r_bar, 0.0, 0.0],
                density=PANEL_DENSITY,
                priority=1,
                rgba=list(OBJECT_RGBA),
            )
            geoms.append(f"{p}bar")
            for k, (pos, half) in enumerate(posts):
                body.add_geom(
                    name=f"{p}post{k}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=half,
                    pos=pos,
                    density=PANEL_DENSITY,
                    priority=1,
                    rgba=list(OBJECT_RGBA),
                )
                geoms.append(f"{p}post{k}")
        lid_site = np.asarray(panel_center) + 0.5 * t * outer_n
        body.add_site(name=f"{p}handle", pos=list(handle_pos), size=[0.004, 0.0, 0.0])
        body.add_site(name=f"{p}lid", pos=lid_site.tolist(), size=[0.004, 0.0, 0.0])
        frame.add_site(name=f"{p}hinge", pos=hinge_local.tolist(), size=[0.004, 0.0, 0.0])
        sites.update(handle=f"{p}handle", lid=f"{p}lid", hinge=f"{p}hinge")
        act = ctx.spec.add_actuator(name=f"{p}catch", target=f"{p}hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        catch = float(prm["open_torque_nm"])
        act.ctrlrange = [-catch - 0.01, catch + 0.01]
        axis_world = rot @ axis_local
        lid0 = rot @ lid_site  # the panel frame is the hinge frame, unrotated when closed
        out0 = rot @ outer_n
        hp = np.asarray(handle_pos, dtype=np.float64)
        radius = float(np.linalg.norm(hp[1:] if not door else hp[:2]))  # handle site to the hinge axis
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}frame", f"{p}panel"),
            geoms=tuple(geoms),
            joints={"hinge": f"{p}hinge"},
            sites=sites,
            actuators=(f"{p}catch",),
            info={
                "orientation": float(door),
                "feature": float(handle_bar),
                "axis_x": float(axis_world[0]),
                "axis_y": float(axis_world[1]),
                "axis_z": float(axis_world[2]),
                "width_m": w,
                "depth_m": d,
                "thickness_m": t,
                "feature_m": f,
                "grip_m": BAR_DIAMETER if handle_bar else t,  # what a gripper closes across
                "bar_radius_m": r_bar,
                "handle_radius_m": radius,
                "max_angle_rad": math.radians(float(prm["max_angle_deg"])),
                "catch_nm": catch,
                "hinge_z": float(origin[2] + hinge_local[2]),
                # closed: the lid site and the outer normal relative to the hinge site (world)
                "lid0_x": float(lid0[0]),
                "lid0_y": float(lid0[1]),
                "lid0_z": float(lid0[2]),
                "out0_x": float(out0[0]),
                "out0_y": float(out0[1]),
                "out0_z": float(out0[2]),
                "top_z": float(origin[2] + (LID_HINGE_HEIGHT + t if not door else DOOR_BOTTOM + w)),
            },
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        angle = float(data.qpos[obj.joint_qpos["hinge"]])  # within the joint's own stops (the
        # solver's soft limits let it lean a thousandth of a degree past them)
        return {"angle_deg": math.degrees(min(max(angle, 0.0), obj.handles.info["max_angle_rad"]))}

    def passive(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        catch = obj.handles.info["catch_nm"]
        if catch <= 0.0:
            return np.zeros(1)
        deg = math.degrees(float(data.qpos[obj.joint_qpos["hinge"]]))
        weight = 1.0 if deg <= CATCH_DEG else max(0.0, 2.0 - deg / CATCH_DEG)
        return np.array([-catch * weight])


register_template_builder(HingedLidBuilder())
