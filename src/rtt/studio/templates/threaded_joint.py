"""The ``threaded_joint`` template: a cap, coupling or nut on a fixed threaded stub (track S2).

Fidelity: **approximation** (the template's caveat: a helical joint with a torque ramp; no thread
flanks, cross-threading or galling).

* **Seat** (static): a base plate on the table (``max(1.6 x diameter, 40 mm)`` across,
  :data:`BASE_H` tall) and a vertical stub that fits the member's bore with
  :data:`BORE_CLEARANCE`, its top cut by a ``chamfer_mm`` lead-in.
* **Member** (one body on two joints on the stub's axis): a slide along the axis (``depth``,
  downward positive) and a hinge about it (``angle``, **tightening positive: clockwise seen from
  above**, a right-hand thread). It starts resting on the stub top, its bore on the lead-in.

  * **Lead-in**: the first ``chamfer_mm + 3 mm`` of travel (:func:`lead_of`) is free of
    the thread; the slide's friction loss (1.5 x the member's weight + 0.2 N) holds the member
    there, so an insertion pushes it to the thread start.
  * **Thread** (the helical coupling): a fixed tendon ``depth - pitch / (2 pi) x angle`` limited
    above at ``lead``: past the thread start the member cannot go deeper than the helix allows,
    so pushing does not advance it and tightening (with an axial feed) does, ``pitch_mm`` per
    turn. It is one-sided (the load flank): unscrewing lifts the member, lifting it off the flank
    is free [approximation].
  * **Seated** after ``engage_turns`` turns from the thread start: the slide's end stop.
  * **Friction torque ramp**: engaged, the torque that resists tightening rises linearly from
    ``start_torque_nm`` at the thread start to ``final_torque_nm`` when seated, and stays at
    ``final_torque_nm`` beyond (so a tightening step stops on torque), applied by the
    template's own motor actuator as a velocity-saturated friction (:data:`OMEGA_S`); a small
    dry friction loss holds it at rest. On the lead-in only that dry friction acts.

  Shapes (outer diameter ``diameter_mm``, length ``length_mm``, wall
  ``max(1.5 mm, 15 % of the diameter)`` [assumed]): ``coupling`` a 12-sided tube, ``cap`` the
  same tube closed on top, ``nut`` a hexagon (corner-to-corner ``diameter_mm``) with a bore. The
  member's contacts with the stub are excluded (the joints keep it on the axis).

Sites: ``member`` (the member's top centre), ``seat`` (the stub top, static) and the auxiliary
``flat`` (on the member's side at local ``+x``, mid-height: the centre of a nut flat). ``info``:
the lead, seated depth, pitch, sizes and the member code (0 cap, 1 coupling, 2 nut).

State: ``turns`` and ``angle_deg`` (tightening from the start), ``depth_mm`` (slide travel from
the stub top: the thread starts at the lead, seated at lead plus ``engage_turns x pitch_mm``).
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

__all__ = ["BASE_H", "BORE_CLEARANCE", "OMEGA_S", "ThreadedJointBuilder", "lead_of"]

BASE_H: Final = 0.020  # m [assumed]
BORE_CLEARANCE: Final = 0.0005  # m, radial, member bore to stub [assumed]
SEAT_GAP: Final = 0.001  # m, member bottom above the base when seated
DENSITY: Final = 2000.0  # kg/m^3 [assumed]
OMEGA_S: Final = 0.3  # rad/s: the ramp's friction saturates above this speed [assumed]
HOLD_NM: Final = 0.02  # N m of dry friction always on the hinge [assumed]
MEMBER_CODES: Final = {"cap": 0.0, "coupling": 1.0, "nut": 2.0}
FRAME_RGBA: Final = (0.55, 0.58, 0.62, 1.0)
FRICTION_SOLIMP: Final = (0.999, 0.9999, 0.0001, 0.5, 2.0)  # a near-hard dry friction (no creep)


def lead_of(chamfer_mm: float) -> float:
    """The free lead-in above the thread start (m): the chamfer plus 3 mm [assumed]."""
    return chamfer_mm / 1000.0 + 0.003


def _quat_z(angle: float) -> list[float]:
    return [math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)]


def _ring(n: int, r_in: float, r_out: float, z0: float, z1: float, twist: float = 0.0) -> list[np.ndarray]:
    """``n`` convex pieces of an ``n``-sided tube: inner and outer apothems ``r_in`` and ``r_out``."""
    half = math.pi / n
    pieces = []
    for k in range(n):
        a = twist + 2.0 * math.pi * k / n
        pts = []
        for r in (r_in, r_out):
            rc = r / math.cos(half)  # the polygon's corner radius for apothem r
            for b in (a - half, a + half):
                pts.append((rc * math.cos(b), rc * math.sin(b)))
        pieces.append(np.array([[x, y, z] for z in (z0, z1) for x, y in pts]))
    return pieces


class ThreadedJointBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.THREADED_JOINT

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        d, length = float(prm["diameter_mm"]) / 1000.0, float(prm["length_mm"]) / 1000.0
        pitch = float(prm["pitch_mm"]) / 1000.0
        seat_travel = float(prm["engage_turns"]) * pitch
        lead = lead_of(float(prm["chamfer_mm"]))
        member = str(prm["member"])
        r_out = 0.5 * d
        wall = max(0.0015, 0.15 * d)
        r_in = r_out - wall
        r_stub = r_in - BORE_CLEARANCE
        chamfer = min(float(prm["chamfer_mm"]) / 1000.0, 0.8 * r_stub)
        yaw = math.radians(pose.yaw_deg)
        z_seated = BASE_H + SEAT_GAP
        z_start = z_seated + lead + seat_travel  # member bottom at the start = the stub top
        side = max(1.6 * d, 0.040)
        seat = ctx.spec.worldbody.add_body(
            name=f"{p}seat",
            pos=[pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z],
            quat=_quat_z(yaw),
        )
        geoms: list[str] = []

        def static(name: str, **kw: object) -> None:
            seat.add_geom(name=f"{p}{name}", priority=1, rgba=list(FRAME_RGBA), **kw)
            geoms.append(f"{p}{name}")

        static(
            "base",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * side, 0.5 * side, 0.5 * BASE_H],
            pos=[0, 0, 0.5 * BASE_H],
        )
        z_cyl = z_start - chamfer
        static(
            "stub",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[r_stub, 0.5 * (z_cyl - BASE_H), 0.0],
            pos=[0, 0, 0.5 * (z_cyl + BASE_H)],
        )
        if chamfer > 1e-5:
            ring = [(math.cos(a), math.sin(a)) for a in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)]
            verts = [[r_stub * c, r_stub * s, z_cyl] for c, s in ring]
            verts += [[(r_stub - chamfer) * c, (r_stub - chamfer) * s, z_start] for c, s in ring]
            ctx.spec.add_mesh(name=f"{p}lead", uservert=np.ravel(verts).tolist())
            static("lead", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{p}lead")

        body = seat.add_body(name=f"{p}member", pos=[0.0, 0.0, z_start])
        slide = body.add_joint(
            name=f"{p}slide",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, -1.0],
            range=[0.0, lead + seat_travel],
            armature=0.002,
            damping=0.5,
            solref_friction=list(ctx.task.physics.limit_solref),
            solimp_friction=list(FRICTION_SOLIMP),
            solimp_limit=list(FRICTION_SOLIMP),  # a near-hard seat
        )
        body.add_joint(
            name=f"{p}hinge",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0.0, 0.0, -1.0],
            limited=mujoco.mjtLimited.mjLIMITED_FALSE,
            armature=0.0005,
            damping=0.001,
            frictionloss=min(HOLD_NM, float(prm["start_torque_nm"])),
            solref_friction=list(ctx.task.physics.limit_solref),
            solimp_friction=list(FRICTION_SOLIMP),
        )
        mass = 0.0
        pieces: list[np.ndarray]
        if member == "nut":
            pieces = []
            for k in range(6):
                a = 2.0 * math.pi * k / 6.0
                corner = r_out  # the hexagon's corner radius; flats at apothem r_out cos 30
                inner = r_in / math.cos(math.pi / 6.0)
                pts = [(inner * math.cos(b), inner * math.sin(b)) for b in (a - math.pi / 6, a + math.pi / 6)]
                pts += [
                    (corner * math.cos(b), corner * math.sin(b)) for b in (a - math.pi / 6, a + math.pi / 6)
                ]
                pieces.append(np.array([[x, y, z] for z in (0.0, length) for x, y in pts]))
            mass = (
                DENSITY * length * (1.5 * math.sqrt(3.0) * (r_out**2 - (r_in / math.cos(math.pi / 6)) ** 2))
            )
        else:
            pieces = _ring(12, r_in, r_out, 0.0, length)
            mass = DENSITY * length * 12 * math.tan(math.pi / 12) * (r_out**2 - r_in**2)
        for k, verts in enumerate(pieces):
            mesh = f"{p}piece{k:02d}"
            ctx.spec.add_mesh(name=mesh, uservert=verts.ravel().tolist())
            body.add_geom(
                name=f"{p}piece{k:02d}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh,
                density=DENSITY,
                priority=1,
                rgba=list(OBJECT_RGBA),
            )
            geoms.append(f"{p}piece{k:02d}")
        if member == "cap":
            top = max(0.002, 0.15 * length)
            body.add_geom(
                name=f"{p}top",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[r_out, 0.5 * top, 0.0],
                pos=[0.0, 0.0, length - 0.5 * top],
                density=DENSITY,
                priority=1,
                rgba=list(OBJECT_RGBA),
            )
            geoms.append(f"{p}top")
            mass += DENSITY * math.pi * r_out**2 * top
        # the slide's friction holds the member on the lead-in until it is pushed
        slide.frictionloss = 1.5 * mass * 9.81 + 0.2
        body.add_site(name=f"{p}member", pos=[0.0, 0.0, length], size=[0.003, 0.0, 0.0])
        flat_r = r_out * (math.cos(math.pi / 6.0) if member == "nut" else 1.0)
        body.add_site(name=f"{p}flat", pos=[flat_r, 0.0, 0.5 * length], size=[0.003, 0.0, 0.0])
        seat.add_site(name=f"{p}seat", pos=[0.0, 0.0, z_start], size=[0.003, 0.0, 0.0])

        tendon = ctx.spec.add_tendon(name=f"{p}flank")
        tendon.wrap_joint(f"{p}slide", 1.0)
        tendon.wrap_joint(f"{p}hinge", -pitch / (2.0 * math.pi))
        tendon.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        tendon.range = [-1.0, lead]
        tendon.solref_limit = list(ctx.task.physics.limit_solref)
        tendon.solimp_limit = list(FRICTION_SOLIMP)  # a near-hard flank
        ctx.spec.add_exclude(bodyname1=f"{p}seat", bodyname2=f"{p}member")
        final = float(prm["final_torque_nm"])
        act = ctx.spec.add_actuator(name=f"{p}ramp", target=f"{p}hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-final - 0.1, final + 0.1]
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}seat", f"{p}member"),
            geoms=tuple(geoms),
            joints={"slide": f"{p}slide", "hinge": f"{p}hinge"},
            sites={"member": f"{p}member", "seat": f"{p}seat", "flat": f"{p}flat"},
            actuators=(f"{p}ramp",),
            info={
                "member": MEMBER_CODES[member],
                "diameter_m": d,
                "length_m": length,
                "pitch_m": pitch,
                "lead_m": lead,
                "seat_m": seat_travel,
                "bore_m": r_in,
                "stub_m": r_stub,
                "flat_m": flat_r,
                "start_nm": float(prm["start_torque_nm"]),
                "final_nm": final,
                "hold_nm": min(HOLD_NM, float(prm["start_torque_nm"])),
                "z_start": float(ctx.table_top_z + z_start),
                "base_z": float(ctx.table_top_z + BASE_H),
                "mass_kg": mass,
            },
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        phi = float(data.qpos[obj.joint_qpos["hinge"]])
        return {
            "turns": phi / (2.0 * math.pi),
            "angle_deg": math.degrees(phi),
            "depth_mm": 1000.0 * float(data.qpos[obj.joint_qpos["slide"]]),
        }

    def passive(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        info = obj.handles.info
        depth = float(data.qpos[obj.joint_qpos["slide"]])
        if depth < info["lead_m"] - 0.0003:  # on the lead-in: only the dry friction loss
            return np.zeros(1)
        s = min(1.0, max(0.0, (depth - info["lead_m"]) / max(info["seat_m"], 1e-6)))
        torque = info["start_nm"] + (info["final_nm"] - info["start_nm"]) * s - info["hold_nm"]
        omega = float(data.qvel[obj.joint_dof["hinge"]])
        return np.array([-max(0.0, torque) * math.tanh(omega / OMEGA_S)])


register_template_builder(ThreadedJointBuilder())
