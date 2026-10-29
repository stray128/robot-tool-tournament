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
  * **Friction torque ramp**: engaged, the torque that resists turning rises linearly from
    ``start_torque_nm`` at the thread start to ``final_torque_nm`` when seated, and stays at
    ``final_torque_nm`` beyond (so a tightening step stops on torque). It is a Coulomb friction
    that holds the member until the applied torque exceeds it and then lets it slide at that
    torque (S2c; it was an explicit velocity-saturated motor that flipped sign every tick and let
    1 N m turn a 4 N m ramp 380 deg in 2 s): the template's own actuator is a stiff torsion
    spring (:data:`FRICTION_KP`, damped) to an anchor angle, and :func:`friction_anchor` moves
    the anchor each control tick only as far as keeps the spring's torque within the ramp. Below
    the ramp the member stays put (the spring's twist is under a degree per N m); above it, it
    turns. A small dry friction loss (:data:`HOLD_NM`) acts always; on the lead-in only that.

  Shapes (outer diameter ``diameter_mm``, length ``length_mm``, wall
  ``max(1.5 mm, 15 % of the diameter)`` [assumed]): ``coupling`` a 12-sided tube, ``cap`` the
  same tube closed on top, ``nut`` a hexagon (corner-to-corner ``diameter_mm``) with a bore. The
  member's contacts with the stub are excluded (the joints keep it on the axis).

Sites: ``member`` (the member's top centre), ``seat`` (the stub top, static) and the auxiliary
``flat`` (on the member's side at local ``+x``, mid-height: the centre of a nut flat). ``info``:
the lead, seated depth, pitch, sizes and the member code (0 cap, 1 coupling, 2 nut).

State: ``turns`` and ``angle_deg`` (tightening from the start), ``depth_mm`` (slide travel from
the stub top: the thread starts at the lead, seated at lead plus ``engage_turns x pitch_mm``),
read within the joints' own limits (the seat stop and the helical flank).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
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

__all__ = [
    "BASE_H",
    "BORE_CLEARANCE",
    "FRICTION_KP",
    "Thread",
    "ThreadedJointBuilder",
    "add_friction_anchor",
    "add_thread",
    "friction_anchor",
    "lead_of",
    "ramp_torque",
    "thread_passive",
    "thread_state",
]

BASE_H: Final = 0.020  # m [assumed]
BORE_CLEARANCE: Final = 0.0005  # m, radial, member bore to stub [assumed]
SEAT_GAP: Final = 0.001  # m, member bottom above the base when seated
DENSITY: Final = 2000.0  # kg/m^3 [assumed]
HOLD_NM: Final = 0.02  # N m of dry friction always on the hinge [assumed]
FRICTION_KP: Final = 100.0  # N m/rad: the friction anchor's torsion spring [assumed]
FRICTION_KV: Final = 0.05  # N m s/rad: its damping (0.05 N m at a 57 deg/s turn) [assumed]
MEMBER_CODES: Final = {"cap": 0.0, "coupling": 1.0, "nut": 2.0}


def add_friction_anchor(spec: mujoco.MjSpec, name: str, joint: str, limit_nm: float) -> None:
    """A position actuator on ``joint`` whose control is the friction anchor: torque
    ``FRICTION_KP x (anchor - angle) - FRICTION_KV x velocity`` (the velocity term implicit in the
    integrator), at most ``limit_nm`` plus a margin."""
    act = spec.add_actuator(name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm[0] = FRICTION_KP
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.biasprm[0] = 0.0
    act.biasprm[1] = -FRICTION_KP
    act.biasprm[2] = -FRICTION_KV
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-limit_nm - 0.5, limit_nm + 0.5]


def friction_anchor(
    obj: ResolvedHandles, data: mujoco.MjData, joint: str, limit_nm: float, period_s: float
) -> float:
    """The friction anchor for this control tick: where it was (the actuator's last control; at
    the segment's first tick, the joint's own angle), moved only as far as keeps the spring's
    torque within ``limit_nm`` at the middle of the coming tick (a stick-slip Coulomb friction)."""
    q = float(data.qpos[obj.joint_qpos[joint]])
    w = float(data.qvel[obj.joint_dof[joint]])
    act = int(obj.actuator_ids[0])
    anchor = q if data.time <= 0.0 else float(data.ctrl[act])
    mid = q + 0.5 * w * period_s
    reach = max(0.0, limit_nm) / FRICTION_KP
    return min(max(anchor, mid - reach), mid + reach)


def lead_of(chamfer_mm: float) -> float:
    """The free lead-in above the thread start (m): the chamfer plus 3 mm [assumed]."""
    return chamfer_mm / 1000.0 + 0.003


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


@dataclass(frozen=True)
class Thread:
    """A threaded member authored by :func:`add_thread`: its element names and its numbers."""

    joints: Mapping[str, str]  # "slide", "hinge"
    sites: Mapping[str, str]  # "member", "seat", "flat"
    member_body: str
    actuator: str
    info: dict[str, float]


def add_thread(
    ctx: TemplateContext,
    seat: mujoco.MjsBody,
    q: str,
    *,
    member: str,
    diameter: float,
    length: float,
    pitch: float,
    engage_turns: float,
    chamfer_mm: float,
    start_nm: float,
    final_nm: float,
    base_h: float,
    origin_z: float,
    geoms: list[str],
) -> Thread:
    """Author a threaded member on ``seat`` (module docstring): the stub standing on ``base_h``
    (m above the seat body's origin, which is ``origin_z`` in the world), the member on its slide and
    hinge, the helical flank and the friction ramp's actuator. Every element is named ``q...``;
    geom names are appended to ``geoms``. The ``threaded_joint`` template calls it on its base
    plate, ``panel_rail`` for its clamp on a post."""
    seat_travel = engage_turns * pitch
    lead = lead_of(chamfer_mm)
    r_out = 0.5 * diameter
    wall = max(0.0015, 0.15 * diameter)
    r_in = r_out - wall
    r_stub = r_in - BORE_CLEARANCE
    chamfer = min(chamfer_mm / 1000.0, 0.8 * r_stub)
    z_seated = base_h + SEAT_GAP
    z_start = z_seated + lead + seat_travel  # member bottom at the start = the stub top

    def static(name: str, **kw: object) -> None:
        seat.add_geom(name=f"{q}{name}", priority=1, rgba=list(FRAME_RGBA), **kw)
        geoms.append(f"{q}{name}")

    z_cyl = z_start - chamfer
    static(
        "stub",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[r_stub, 0.5 * (z_cyl - base_h), 0.0],
        pos=[0, 0, 0.5 * (z_cyl + base_h)],
    )
    if chamfer > 1e-5:
        ring = [(math.cos(a), math.sin(a)) for a in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)]
        verts = [[r_stub * c, r_stub * s, z_cyl] for c, s in ring]
        verts += [[(r_stub - chamfer) * c, (r_stub - chamfer) * s, z_start] for c, s in ring]
        ctx.spec.add_mesh(name=f"{q}lead", uservert=np.ravel(verts).tolist())
        static("lead", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{q}lead")

    body = seat.add_body(name=f"{q}member", pos=[0.0, 0.0, z_start])
    slide = body.add_joint(
        name=f"{q}slide",
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
        name=f"{q}hinge",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 0.0, -1.0],
        limited=mujoco.mjtLimited.mjLIMITED_FALSE,
        armature=0.0005,
        damping=0.001,
        frictionloss=min(HOLD_NM, start_nm),
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
            pts += [(corner * math.cos(b), corner * math.sin(b)) for b in (a - math.pi / 6, a + math.pi / 6)]
            pieces.append(np.array([[x, y, z] for z in (0.0, length) for x, y in pts]))
        mass = DENSITY * length * (1.5 * math.sqrt(3.0) * (r_out**2 - (r_in / math.cos(math.pi / 6)) ** 2))
    else:
        pieces = _ring(12, r_in, r_out, 0.0, length)
        mass = DENSITY * length * 12 * math.tan(math.pi / 12) * (r_out**2 - r_in**2)
    for k, verts in enumerate(pieces):
        mesh = f"{q}piece{k:02d}"
        ctx.spec.add_mesh(name=mesh, uservert=verts.ravel().tolist())
        body.add_geom(
            name=f"{q}piece{k:02d}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh,
            density=DENSITY,
            priority=1,
            rgba=list(OBJECT_RGBA),
        )
        geoms.append(f"{q}piece{k:02d}")
    if member == "cap":
        top = max(0.002, 0.15 * length)
        body.add_geom(
            name=f"{q}top",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[r_out, 0.5 * top, 0.0],
            pos=[0.0, 0.0, length - 0.5 * top],
            density=DENSITY,
            priority=1,
            rgba=list(OBJECT_RGBA),
        )
        geoms.append(f"{q}top")
        mass += DENSITY * math.pi * r_out**2 * top
    # the slide's friction holds the member on the lead-in until it is pushed
    slide.frictionloss = 1.5 * mass * 9.81 + 0.2
    body.add_site(name=f"{q}member", pos=[0.0, 0.0, length], size=[0.003, 0.0, 0.0])
    flat_r = r_out * (math.cos(math.pi / 6.0) if member == "nut" else 1.0)
    body.add_site(name=f"{q}flat", pos=[flat_r, 0.0, 0.5 * length], size=[0.003, 0.0, 0.0])
    seat.add_site(name=f"{q}seat", pos=[0.0, 0.0, z_start], size=[0.003, 0.0, 0.0])

    tendon = ctx.spec.add_tendon(name=f"{q}flank")
    tendon.wrap_joint(f"{q}slide", 1.0)
    tendon.wrap_joint(f"{q}hinge", -pitch / (2.0 * math.pi))
    tendon.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    tendon.range = [-1.0, lead]
    tendon.solref_limit = list(ctx.task.physics.limit_solref)
    tendon.solimp_limit = list(FRICTION_SOLIMP)  # a near-hard flank
    ctx.spec.add_exclude(bodyname1=seat.name, bodyname2=f"{q}member")
    add_friction_anchor(ctx.spec, f"{q}ramp", f"{q}hinge", final_nm)
    return Thread(
        joints={"slide": f"{q}slide", "hinge": f"{q}hinge"},
        sites={"member": f"{q}member", "seat": f"{q}seat", "flat": f"{q}flat"},
        member_body=f"{q}member",
        actuator=f"{q}ramp",
        info={
            "member": MEMBER_CODES[member],
            "diameter_m": diameter,
            "length_m": length,
            "pitch_m": pitch,
            "lead_m": lead,
            "seat_m": seat_travel,
            "bore_m": r_in,
            "stub_m": r_stub,
            "flat_m": flat_r,
            "start_nm": start_nm,
            "final_nm": final_nm,
            "hold_nm": min(HOLD_NM, start_nm),
            "z_start": float(origin_z + z_start),
            "base_z": float(origin_z + base_h),
            "mass_kg": mass,
            "period_s": float(ctx.task.controller.control_period),
        },
    )


def thread_state(info: Mapping[str, float], phi: float, slide: float) -> dict[str, float]:
    """``turns``, ``angle_deg`` and ``depth_mm`` of a threaded member (its info, hinge angle and
    slide travel), read within the joints' own limits: the seat stop and the helical flank (whose
    soft limit lets a member pushed onto the thread start sit a fraction of a micrometre past it)."""
    flank = info["lead_m"] + info["pitch_m"] * phi / (2.0 * math.pi)
    depth = min(max(slide, 0.0), info["lead_m"] + info["seat_m"], flank)
    return {
        "turns": phi / (2.0 * math.pi),
        "angle_deg": math.degrees(phi),
        "depth_mm": 1000.0 * max(depth, 0.0),
    }


def ramp_torque(info: Mapping[str, float], depth: float) -> float:
    """The friction ramp's torque at slide travel ``depth`` (m): 0 on the lead-in, ``start_nm`` at the
    thread start rising linearly to ``final_nm`` when seated (and beyond)."""
    if depth < info["lead_m"] - 0.0003:
        return 0.0
    s = min(1.0, max(0.0, (depth - info["lead_m"]) / max(info["seat_m"], 1e-6)))
    return info["start_nm"] + (info["final_nm"] - info["start_nm"]) * s


def thread_passive(obj: ResolvedHandles, data: mujoco.MjData, info: Mapping[str, float]) -> float:
    """The friction anchor of a threaded member's ramp for this control tick (module docstring)."""
    depth = float(data.qpos[obj.joint_qpos["slide"]])
    torque = 0.0  # on the lead-in: only the dry friction loss
    if depth >= info["lead_m"] - 0.0003:
        torque = ramp_torque(info, depth) - info["hold_nm"]
    return friction_anchor(obj, data, "hinge", torque, info["period_s"])


class ThreadedJointBuilder(TemplateBuilder):
    kind: ClassVar[TemplateKind] = TemplateKind.THREADED_JOINT

    def add(self, ctx: TemplateContext) -> TemplateHandles:
        prm, p, pose = ctx.instance.params, ctx.prefix, ctx.instance.pose
        d = float(prm["diameter_mm"]) / 1000.0
        yaw = math.radians(pose.yaw_deg)
        side = max(1.6 * d, 0.040)
        seat = ctx.spec.worldbody.add_body(
            name=f"{p}seat",
            pos=[pose.x_mm / 1000.0, pose.y_mm / 1000.0, ctx.table_top_z],
            quat=quat_z(yaw),
        )
        geoms: list[str] = []
        seat.add_geom(
            name=f"{p}base",
            priority=1,
            rgba=list(FRAME_RGBA),
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.5 * side, 0.5 * side, 0.5 * BASE_H],
            pos=[0, 0, 0.5 * BASE_H],
        )
        geoms.append(f"{p}base")
        thread = add_thread(
            ctx,
            seat,
            p,
            member=str(prm["member"]),
            diameter=d,
            length=float(prm["length_mm"]) / 1000.0,
            pitch=float(prm["pitch_mm"]) / 1000.0,
            engage_turns=float(prm["engage_turns"]),
            chamfer_mm=float(prm["chamfer_mm"]),
            start_nm=float(prm["start_torque_nm"]),
            final_nm=float(prm["final_torque_nm"]),
            base_h=BASE_H,
            origin_z=ctx.table_top_z,
            geoms=geoms,
        )
        return TemplateHandles(
            instance=ctx.instance.name,
            kind=self.kind,
            bodies=(f"{p}seat", thread.member_body),
            geoms=tuple(geoms),
            joints=thread.joints,
            sites=thread.sites,
            actuators=(thread.actuator,),
            info=thread.info,
        )

    def state(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        phi = float(data.qpos[obj.joint_qpos["hinge"]])
        return thread_state(obj.handles.info, phi, float(data.qpos[obj.joint_qpos["slide"]]))

    def passive(self, obj: ResolvedHandles, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        return np.array([thread_passive(obj, data, obj.handles.info)])


register_template_builder(ThreadedJointBuilder())
