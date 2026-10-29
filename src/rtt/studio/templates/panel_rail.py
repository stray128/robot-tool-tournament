"""The ``panel_rail`` template: a panel on two mounting rails, with a threaded clamp (Studio v2).

The solar-install job: a vacuum frame lifts the panel and sets it back on its rails against their
end stops; a nut runner starts the clamp on its thread (``insert``) and torques it (``twist``).

Fidelity: **approximation** (the template's caveat). The clamp is a threaded member (the
``threaded_joint`` model, :func:`~rtt.studio.templates.threaded_joint.add_thread`): a 13 mm
across-flats hex (the M8 nut's size, ISO 4032) **modelled 15 mm tall so the simulated socket can
seat; a standard M8 nut is 6.8 mm tall** (the nut runner engages members at least
:data:`~rtt.effectors.process_tool.MIN_MEMBER_MM` = 12 mm tall). It turns on an M8 coarse thread
(1.25 mm pitch, ISO 261), three turns from the thread start to seated [assumed], its friction
ramp rising from 0.3 N m [assumed] to ``clamp_torque_nm`` (default 6.0 N m [assumed], below the
nut runner's 6.7 N m rating). The clamp does not press the panel (it stands on a post beside the
panel's edge): seating it is the clamping step.

Instance frame: origin at the pose on the table top, turned by ``yaw_deg``; the panel's length
along local x, its width along local y. **Reach**: the two places a tool works, the panel's top
centre (a frame's lift) and the clamp, lie on local y either side of the origin, each
:func:`reach_of` (``(width / 2 + CLAMP_OUT) / 2``, 162.5 mm by default) from it, so both stay inside
the palm's travel (+-350 mm, :data:`rtt.sim.scene.PALM_XY_LIMIT`) for every pose within
+-(345 mm - reach) on x and y (+-182 mm by default) at any yaw.

* **Rails** (static): two rails along local y at ``x = +-length / 4`` [assumed], :data:`RAIL_W`
  wide and :data:`RAIL_H` tall [assumed], each with an end stop (:data:`STOP_T` along y,
  :data:`STOP_H` above the rail [assumed]) at the panel's ``+y`` edge, ``start_gap_mm`` away from
  it at the start.
* **Panel** (a free body, the template's last body: the one a pick lifts): a
  ``length_mm`` x ``width_mm`` x ``thickness_mm`` box of ``mass_g`` resting on the rails, its
  centre at local ``(0, +reach)``.
* **Clamp** (at the middle of the panel's ``-y`` edge, :data:`CLAMP_OUT` beyond it, local
  ``(0, -reach)``): a short bracket rail (:data:`RAIL_H` tall, ending 5 mm short of the panel) and
  a post (:data:`POST_W` square) rising :data:`POST_ABOVE` above the panel's top [assumed], so a
  socket over the clamp stays above the panel, with the threaded stub and the hex on it.
* **Clamp latch** (a light, gravity-compensated slide with no spring and :data:`LATCH_FRICTION`
  [assumed]): a tendon pushes it along as the clamp goes down its thread, and it stays, so it
  records the deepest the clamp has been driven (a tightened clamp that springs back a few
  hundredths of a millimetre off its seat stop when the socket lets go is still tightened).

Sites: ``panel`` (the panel's top centre: where a pick seals or grips), ``member`` (the clamp's
top centre), and the auxiliaries ``seat`` (the stub top) and ``flat`` (a hex flat's centre).
``info``: the panel's sizes and mass, its rest height, the stops' face, the clamp's numbers under
``clamp_`` keys (:func:`rtt.studio.templates.thread_info`), and the pick stand-in: the picks plan
a part from ``diameter_m`` and ``length_m`` (a round slab of :data:`PICK_STANDIN_M` as thick as the
panel: only its height places the tool, and the frozen parts library caps a part at 0.5 m).

State: ``gap_mm`` (the larger of the two gaps between the panel's ``+y`` edge and the stops, at
the rails), ``clamp_torque_nm`` (the torque the clamp has been tightened to: its thread's friction
ramp at the deepest point it reached, the latch's; 0 on the lead-in, the start torque at the
thread start rising to ``clamp_torque_nm`` on the seat stop, since the clamp goes no deeper than a
tightening torque over the ramp takes it), and the clamp's present ``depth_mm``, ``turns`` and
``angle_deg`` (as a threaded joint's, which the twist and insert primitives read).
"""

from __future__ import annotations

import math
from typing import ClassVar, Final

import mujoco
import numpy as np

from rtt.studio.templates import (
    CLAMP,
    FRAME_RGBA,
    FRICTION_SOLIMP,
    PICK_STANDIN_M,
    ResolvedHandles,
    TemplateBuilder,
    TemplateContext,
    TemplateHandles,
    quat_z,
    register_template_builder,
    thread_info,
)
from rtt.studio.templates.threaded_joint import (
    add_thread,
    ramp_torque,
    thread_passive,
    thread_state,
)
from rtt.studio.types import TemplateKind

__all__ = ["CLAMP_AF", "CLAMP_HEIGHT", "PanelRailBuilder", "reach_of"]

RAIL_W: Final = 0.040  # m [assumed]
RAIL_H: Final = 0.040  # m [assumed]
RAIL_X_FRAC: Final = 0.25  # rails at +-length x this [assumed]
RAIL_PAST: Final = 0.010  # m of rail beyond the stops [assumed]
STOP_T: Final = 0.010  # m along y [assumed]
STOP_H: Final = 0.020  # m above the rail [assumed]
CLAMP_OUT: Final = 0.025  # m from the panel's edge to the clamp's axis [assumed]
POST_W: Final = 0.020  # m [assumed]
POST_ABOVE: Final = 0.005  # m: the post's top above the panel's top [assumed]
CLAMP_AF: Final = 0.013  # m across flats: the M8 nut's size (ISO 4032)
CLAMP_HEIGHT: Final = 0.015  # m: modelled tall so the socket seats (an M8 nut is 6.8 mm) [assumed]
CLAMP_PITCH: Final = 0.00125  # m: M8 coarse (ISO 261)
CLAMP_TURNS: Final = 3.0  # turns from the thread start to seated [assumed]
CLAMP_CHAMFER_MM: Final = 1.0  # [assumed]
CLAMP_START_NM: Final = 0.3  # N m at the thread start [assumed]
LATCH_FRICTION: Final = 0.02  # N on the clamp latch [assumed]
LATCH_DAMPING: Final = 20.0  # N s/m: 0.01 N on the clamp at its 0.5 mm/s tightening feed [assumed]
PANEL_RGBA: Final = (0.12, 0.18, 0.32, 1.0)


def reach_of(width_m: float) -> float:
    """How far the panel's top centre and the clamp each lie from the pose (m, on local y)."""
    return 0.5 * (0.5 * width_m + CLAMP_OUT)


class PanelRailBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.PANEL_RAIL

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        length, width, thick = (float(prm[k]) / 1000.0 for k in ("length_mm", "width_mm", "thickness_mm"))
        mass = float(prm["mass_g"]) / 1000.0
        gap0 = float(prm["start_gap_mm"]) / 1000.0
        yaw = math.radians(pose.yaw_deg)
        origin = np.array([pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z])
        frame = ctx.spec.worldbody.add_body(name=f"{p}frame", pos=origin.tolist(), quat=quat_z(yaw))
        geoms: list[str] = []

        def box(body: mujoco.MjsBody, name: str, lo: list[float], hi: list[float]) -> None:
            body.add_geom(
                name=f"{p}{name}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.5 * (b - a) for a, b in zip(lo, hi, strict=True)],
                pos=[0.5 * (a + b) for a, b in zip(lo, hi, strict=True)],
                priority=1,
                rgba=list(FRAME_RGBA),
            )
            geoms.append(f"{p}{name}")

        reach = reach_of(width)
        rail_x = RAIL_X_FRAC * length
        edge_y = reach - 0.5 * width  # the panel's -y edge
        stop_y = reach + 0.5 * width + gap0  # the stops' faces
        clamp_y = -reach
        y1 = stop_y + STOP_T + RAIL_PAST
        hw = 0.5 * RAIL_W
        for k, x in enumerate((-rail_x, rail_x)):
            box(frame, f"rail{k}", [x - hw, edge_y - RAIL_PAST, 0.0], [x + hw, y1, RAIL_H])
            box(frame, f"stop{k}", [x - hw, stop_y, RAIL_H], [x + hw, stop_y + STOP_T, RAIL_H + STOP_H])
        # the clamp's bracket: a short rail under the clamp, ending short of the panel
        box(frame, "bracket", [-hw, clamp_y - 0.5 * POST_W - RAIL_PAST, 0.0], [hw, edge_y - 0.005, RAIL_H])

        # the clamp: a post on its bracket, the threaded member on its stub
        post_h = thick + POST_ABOVE
        seat = frame.add_body(name=f"{p}{CLAMP}seat", pos=[0.0, clamp_y, RAIL_H])
        box(seat, f"{CLAMP}post", [-0.5 * POST_W, -0.5 * POST_W, 0.0], [0.5 * POST_W, 0.5 * POST_W, post_h])
        thread = add_thread(
            ctx,
            seat,
            f"{p}{CLAMP}",
            member="nut",
            diameter=CLAMP_AF / math.cos(math.pi / 6.0),  # the hexagon's corner-to-corner size
            length=CLAMP_HEIGHT,
            pitch=CLAMP_PITCH,
            engage_turns=CLAMP_TURNS,
            chamfer_mm=CLAMP_CHAMFER_MM,
            start_nm=CLAMP_START_NM,
            final_nm=float(prm["clamp_torque_nm"]),
            base_h=post_h,
            origin_z=ctx.table_top_z + RAIL_H,
            geoms=geoms,
        )
        # the latch: it records the deepest the clamp has gone down its thread
        travel = thread.info["lead_m"] + thread.info["seat_m"]
        latch = seat.add_body(name=f"{p}{CLAMP}latch", pos=[0.0, 0.0, 0.5 * post_h])
        latch.gravcomp = 1.0
        latch.add_joint(
            name=f"{p}{CLAMP}latch",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, -1.0],
            range=[0.0, travel],
            frictionloss=LATCH_FRICTION,
            damping=LATCH_DAMPING,  # overdamped: pushed along, it stops with the clamp
            armature=0.0005,
            solref_friction=list(ctx.task.physics.limit_solref),
            solimp_friction=list(FRICTION_SOLIMP),
        )
        latch.add_geom(
            name=f"{p}{CLAMP}latch",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.001, 0.0, 0.0],
            mass=0.001,
            contype=0,
            conaffinity=0,
            group=3,
            priority=1,
            rgba=[0.0, 0.0, 0.0, 0.0],
        )
        geoms.append(f"{p}{CLAMP}latch")
        tendon = ctx.spec.add_tendon(name=f"{p}{CLAMP}latch")
        tendon.wrap_joint(f"{p}{CLAMP}latch", 1.0)
        tendon.wrap_joint(thread.joints["slide"], -1.0)
        tendon.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        tendon.range = [0.0, 1.0]  # latch >= the clamp's depth
        tendon.solref_limit = list(ctx.task.physics.limit_solref)
        tendon.solimp_limit = list(FRICTION_SOLIMP)

        # the panel, a free body resting on the rails (the last body: the one a pick lifts)
        rest_z = ctx.table_top_z + RAIL_H + 0.5 * thick
        centre = origin[:2] + reach * np.array([-math.sin(yaw), math.cos(yaw)])  # local (0, +reach)
        panel = ctx.spec.worldbody.add_body(
            name=f"{p}panel", pos=[float(centre[0]), float(centre[1]), rest_z], quat=quat_z(yaw)
        )
        panel.add_freejoint(name=f"{p}free")
        panel.add_geom(
            name=f"{p}panel",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * length, 0.5 * width, 0.5 * thick],
            mass=mass,
            priority=1,
            rgba=list(PANEL_RGBA),
        )
        geoms.append(f"{p}panel")
        panel.add_site(name=f"{p}panel", pos=[0.0, 0.0, 0.5 * thick], size=[0.004, 0.0, 0.0])
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}frame", f"{p}{CLAMP}seat", thread.member_body, f"{p}{CLAMP}latch", f"{p}panel"),
            geoms=tuple(geoms),
            joints={**thread.joints, "latch": f"{p}{CLAMP}latch", "free": f"{p}free"},
            sites={"panel": f"{p}panel", **thread.sites},
            actuators=(thread.actuator,),
            info={
                # the pick stand-in (module docstring): a round slab as thick as the panel
                "diameter_m": PICK_STANDIN_M,
                "length_m": thick,
                "panel_x_m": length,
                "panel_y_m": width,
                "panel_t_m": thick,
                "mass_kg": mass,
                "rest_z": rest_z,
                "rail_x_m": rail_x,
                "stop_y_m": stop_y,
                "reach_m": reach,  # the panel's centre and the clamp, each this far from the pose
                **{f"{CLAMP}{k}": v for k, v in thread.info.items()},
            },
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        clamp = thread_info(obj)
        phi = float(data.qpos[obj.joint_qpos["hinge"]])
        thread = thread_state(clamp, phi, float(data.qpos[obj.joint_qpos["slide"]]))
        seated = clamp["lead_m"] + clamp["seat_m"]
        deepest = min(max(float(data.qpos[obj.joint_qpos["latch"]]), 0.0), seated)  # within its stop
        torque = clamp["final_nm"] if deepest >= seated else ramp_torque(clamp, deepest)
        return {"gap_mm": self._gap_mm(obj, data), "clamp_torque_nm": torque, **thread}

    @staticmethod
    def _gap_mm(obj: ResolvedHandles, data: mujoco.MjData) -> float:
        """The larger gap between the panel's +y edge and the stops, at the two rails (mm, >= 0)."""
        info = obj.handles.info
        frame, panel = obj.body_ids[0], obj.body_ids[-1]
        r_f = data.xmat[frame].reshape(3, 3)
        r_p = data.xmat[panel].reshape(3, 3)
        hx, hy = 0.5 * info["panel_x_m"], 0.5 * info["panel_y_m"]
        corners = [
            r_f.T @ (data.xpos[panel] + r_p @ np.array([x, hy, 0.0]) - data.xpos[frame]) for x in (-hx, hx)
        ]
        (x1, y1), (x2, y2) = corners[0][:2], corners[1][:2]
        gaps = []
        for x in (-info["rail_x_m"], info["rail_x_m"]):
            edge = y1 + (x - x1) * (y2 - y1) / (x2 - x1) if abs(x2 - x1) > 1e-9 else max(y1, y2)
            gaps.append(info["stop_y_m"] - float(edge))
        return 1000.0 * max(0.0, max(gaps))

    def passive(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        return np.array([thread_passive(obj, data, thread_info(obj))])


register_template_builder(PanelRailBuilder())
