"""Registration: is the part seated and centred in the jaws before the lift (task v2, spec D1)?

Under lift-only scoring a jammed imprint still passes: it lifts the part wherever it landed.
Task v2 therefore scores a *registered grasp*. At the registration moment, ``settle_time``
(0.25 s) after the grip ramp ends and before the lift, the part's pose is taken in the palm
frame (x across the jaw, y along the closing axis, z up; :func:`part_pose_in_palm`) and:

* **seated**: ``seat_short_mm = (q_seat - q) * 1000 <= seat_short_max_mm`` (1.0 mm), where
  ``q`` is the mean jaw closure and ``q_seat`` the closure at which both pads first touch the
  centred part (``FingerPair.q_seat``). A jaw that stalls on a rim stops short of its seat
  (``jam_on_close``);
* **centred**: ``|dx|`` and ``|dy|`` <= 0.75 mm, the yaw residual (taken modulo the part's
  ``yaw_symmetry``) <= 2 deg and tilt <= 3 deg (``off_centre``). A part checks only the axes it
  declares in ``register_axes``; the seat is always checked;
* **upright after the hold**: tilt <= 5 deg when the hold is confirmed (``tilt_after_hold``).

The centred reference is the nominal grasp pose: the part's object-frame origin (its
bounding-box centre) on the palm's centre line with the nominal yaw, which is where seat-aligned
jaws hold a part that arrives with no placement error.

:func:`probe_first_contact` finds a jaw's first-contact closure in the compiled scored scene
with MuJoCo's own distance query, which gives ``q_seat`` for v1 box fingers and checks the
designed ``q_seat`` and clearance of a v2 pair (gate GV1).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from rtt.sim import objects as objs
from rtt.sim.scene import SceneIds
from rtt.sim.task import RegistrationConfig

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec
    from rtt.sim.controller import GraspPlan

AXES: tuple[str, ...] = ("x", "y", "yaw", "tilt")
SUB_JAM = "jam_on_close"
SUB_OFF_CENTRE = "off_centre"
SUB_TILT_HOLD = "tilt_after_hold"
SUB_NOT_MEASURED = "not_measured"  # the episode ended before the registration moment
SUB_FLAGS: tuple[str, ...] = (SUB_JAM, SUB_OFF_CENTRE, SUB_TILT_HOLD, SUB_NOT_MEASURED)


# --------------------------------------------------------------------------- symmetry


@dataclass(frozen=True)
class RegistrationSpec:
    """Which registration axes apply to a part, and its yaw symmetry (see ``PartSpec``)."""

    yaw_symmetry: int = 1  # n-fold about z; 0 = symmetric under every yaw
    register_axes: tuple[str, ...] = AXES

    def __post_init__(self) -> None:
        if self.yaw_symmetry < 0:
            raise ValueError(f"yaw_symmetry must be >= 0, got {self.yaw_symmetry}")
        if not set(self.register_axes) <= set(AXES):
            raise ValueError(f"register_axes must be a subset of {AXES}, got {self.register_axes}")

    def yaw_residual_deg(self, yaw_deg: float) -> float:
        """Absolute yaw error (deg) once the symmetry is taken out (0 for ``yaw_symmetry == 0``)."""
        if self.yaw_symmetry == 0:
            return 0.0
        period = 360.0 / self.yaw_symmetry
        return abs((yaw_deg + period / 2.0) % period - period / 2.0)

    def checks(self, axis: str) -> bool:
        if axis == "yaw" and self.yaw_symmetry == 0:
            return False
        return axis in self.register_axes


def registration_spec(obj: objs.AnyObject | PartSpec) -> RegistrationSpec:
    """The registration spec of a part (``PartSpec``) or of a v1 object.

    v1 objects: a box is 2-fold symmetric about z (4-fold when square in plan), a standing
    cylinder symmetric under every yaw, a lying rod 2-fold; a mesh object has no symmetry.
    """
    sym = getattr(obj, "yaw_symmetry", None)
    if sym is not None:
        return RegistrationSpec(int(sym), tuple(obj.register_axes))  # type: ignore[union-attr]
    kind = getattr(obj, "kind", "mesh")
    if kind == "box":
        sx, sy, _ = obj.size
        return RegistrationSpec(4 if math.isclose(sx, sy, rel_tol=1e-9) else 2)
    if kind == "cylinder":
        return RegistrationSpec(0, ("x", "y", "tilt"))
    if kind == "rod":
        return RegistrationSpec(2)
    return RegistrationSpec(1)


# --------------------------------------------------------------------------- pose in the palm frame


@dataclass(frozen=True)
class PalmPose:
    """The part's object-frame pose relative to the palm, metres and radians.

    ``dx``, ``dy``, ``dz``: its origin in the palm frame; ``yaw``: rotation about the palm z
    axis, wrapped to [-pi, pi); ``tilt``: angle between the part's z axis and the palm's.
    """

    dx: float
    dy: float
    dz: float
    yaw: float
    tilt: float


def part_pose_in_palm(data: mujoco.MjData, ids: SceneIds) -> PalmPose:
    """The object's pose in the palm frame (uses ``xpos``/``xmat``: call after a step or forward)."""
    r_palm = data.xmat[ids.palm_body].reshape(3, 3)
    rel = r_palm.T @ (data.xpos[ids.object_body] - data.xpos[ids.palm_body])
    rot = r_palm.T @ data.xmat[ids.object_body].reshape(3, 3)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    tilt = math.acos(max(-1.0, min(1.0, float(rot[2, 2]))))
    return PalmPose(float(rel[0]), float(rel[1]), float(rel[2]), math.remainder(yaw, 2 * math.pi), tilt)


def mean_closure(data: mujoco.MjData, ids: SceneIds) -> float:
    """Mean jaw closure ``q`` (m; 0 = open): the grip tendon's length."""
    return float(np.mean(data.qpos[ids.finger_qpos]))


# --------------------------------------------------------------------------- the rule


@dataclass(frozen=True)
class Registration:
    """The registration verdict of one episode (lengths in mm, angles in degrees).

    ``measured`` is False when the episode ended before the registration moment; it then
    counts as not registered with the ``not_measured`` sub-flag and NaN values.
    """

    measured: bool
    q_seat_mm: float
    q_mm: float
    seat_short_mm: float
    dx_mm: float
    dy_mm: float
    dyaw_deg: float  # residual modulo the yaw symmetry
    tilt_deg: float
    hold_tilt_deg: float  # at hold confirmation; NaN if the hold was never confirmed
    seated: bool
    centred: bool
    upright_after_hold: bool
    sub_flags: tuple[str, ...]

    @property
    def registered(self) -> bool:
        """Seated and centred at the registration moment, and upright after the hold."""
        return self.measured and self.seated and self.centred and self.upright_after_hold

    def as_metrics(self) -> dict[str, float]:
        return {
            "registered": float(self.registered),
            "q_seat_mm": self.q_seat_mm,
            "reg_q_mm": self.q_mm,
            "seat_short_mm": self.seat_short_mm,
            "reg_dx_mm": self.dx_mm,
            "reg_dy_mm": self.dy_mm,
            "reg_dyaw_deg": self.dyaw_deg,
            "reg_tilt_deg": self.tilt_deg,
            "reg_hold_tilt_deg": self.hold_tilt_deg,
        }


def not_measured(q_seat: float) -> Registration:
    nan = math.nan
    return Registration(
        measured=False,
        q_seat_mm=q_seat * 1000.0,
        q_mm=nan,
        seat_short_mm=nan,
        dx_mm=nan,
        dy_mm=nan,
        dyaw_deg=nan,
        tilt_deg=nan,
        hold_tilt_deg=nan,
        seated=False,
        centred=False,
        upright_after_hold=False,
        sub_flags=(SUB_NOT_MEASURED,),
    )


def assess(
    pose: PalmPose,
    q: float,
    q_seat: float,
    spec: RegistrationSpec,
    cfg: RegistrationConfig,
    hold_tilt: float = math.nan,
) -> Registration:
    """Apply the registration rule to a measured pose (``pose`` at the registration moment,
    ``q`` the mean closure then, ``hold_tilt`` the tilt in rad at hold confirmation or NaN)."""
    seat_short_mm = (q_seat - q) * 1000.0
    dx_mm, dy_mm = pose.dx * 1000.0, pose.dy * 1000.0
    dyaw = spec.yaw_residual_deg(math.degrees(pose.yaw))
    tilt = math.degrees(pose.tilt)
    seated = seat_short_mm <= cfg.seat_short_max_mm
    ok = {
        "x": abs(dx_mm) <= cfg.dx_max_mm,
        "y": abs(dy_mm) <= cfg.dy_max_mm,
        "yaw": dyaw <= cfg.yaw_max_deg,
        "tilt": tilt <= cfg.tilt_max_deg,
    }
    centred = all(ok[a] for a in AXES if spec.checks(a))
    hold_tilt_deg = math.degrees(hold_tilt) if math.isfinite(hold_tilt) else math.nan
    upright = not (spec.checks("tilt") and hold_tilt_deg > cfg.hold_tilt_max_deg)
    sub = []
    if not seated:
        sub.append(SUB_JAM)
    if not centred:
        sub.append(SUB_OFF_CENTRE)
    if not upright:
        sub.append(SUB_TILT_HOLD)
    return Registration(
        measured=True,
        q_seat_mm=q_seat * 1000.0,
        q_mm=q * 1000.0,
        seat_short_mm=seat_short_mm,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        dyaw_deg=dyaw,
        tilt_deg=tilt,
        hold_tilt_deg=hold_tilt_deg,
        seated=seated,
        centred=centred,
        upright_after_hold=upright,
        sub_flags=tuple(sub),
    )


# --------------------------------------------------------------------------- static probes


@dataclass(frozen=True)
class ProbeScene:
    """A compiled scored scene posed for static distance queries (nothing is simulated)."""

    model: mujoco.MjModel
    ids: SceneIds
    data: mujoco.MjData
    jaw_geoms: tuple[np.ndarray, np.ndarray]  # geom ids of the left and right finger bodies

    @staticmethod
    def pose(
        model: mujoco.MjModel,
        ids: SceneIds,
        plan: GraspPlan,
        obj: objs.AnyObject,
        placement: objs.Placement,
        table_top_z: float,
    ) -> ProbeScene:
        """Palm at the grasp pose of ``plan``, the object at ``placement``'s true pose, jaws open.

        Distances are queried on a copy of ``model`` with MuJoCo's native collision (GJK/EPA)
        enabled whatever the task sets: with it disabled, ``mj_geomDistance`` on convex mesh
        pairs is not a true distance (it reported first contact 5-10 mm early in tests).
        """
        model = copy.deepcopy(model)
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        data = mujoco.MjData(model)
        data.qpos[ids.palm_qpos] = [plan.x, plan.y, plan.grasp_z, plan.yaw]
        pos, quat = objs.rest_pose(obj, placement.actual(), table_top_z)
        data.qpos[ids.object_qpos : ids.object_qpos + 3] = pos
        data.qpos[ids.object_qpos + 3 : ids.object_qpos + 7] = quat
        data.qpos[ids.finger_qpos] = 0.0
        mujoco.mj_kinematics(model, data)
        left = np.flatnonzero(model.geom_bodyid == ids.finger_left_body)
        right = np.flatnonzero(model.geom_bodyid == ids.finger_right_body)
        return ProbeScene(model, ids, data, (left, right))

    def distance(self, side: int, q: float, distmax: float = 0.05) -> float:
        """Smallest signed distance (m) between jaw ``side`` (0 left, 1 right) at closure ``q``
        and the object; the other jaw stays fully open."""
        d = self.data
        d.qpos[self.ids.finger_qpos] = 0.0
        d.qpos[self.ids.finger_qpos[side]] = q
        mujoco.mj_kinematics(self.model, d)
        best = distmax
        for g1 in self.jaw_geoms[side]:
            for g2 in self.ids.object_geoms:
                best = min(best, mujoco.mj_geomDistance(self.model, d, int(g1), int(g2), distmax, None))
        return float(best)

    def first_contact(self, side: int, tol: float = 1e-8) -> float:
        """Closure (m) at which jaw ``side`` first touches the object, by bisection on the
        signed distance; NaN if it never does within the stroke, 0 if it touches when open."""
        lo, hi = 0.0, float(self.model.jnt_range[self.model.body_jntadr[self.jaw_body(side)], 1])
        if self.distance(side, lo) <= 0.0:
            return 0.0
        if self.distance(side, hi) > 0.0:
            return math.nan
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.distance(side, mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def jaw_body(self, side: int) -> int:
        return self.ids.finger_left_body if side == 0 else self.ids.finger_right_body


def probe_first_contact(
    model: mujoco.MjModel,
    ids: SceneIds,
    plan: GraspPlan,
    obj: objs.AnyObject,
    table_top_z: float = 0.0,
    placement: objs.Placement | None = None,
) -> tuple[float, float]:
    """``(q_left, q_right)``: each jaw's first-contact closure on the part at its centred pose
    (or at ``placement``'s true pose), with the palm at the grasp pose."""
    nominal = objs.Placement(plan.x, plan.y, plan.yaw)
    scene = ProbeScene.pose(model, ids, plan, obj, placement or nominal, table_top_z)
    return scene.first_contact(0), scene.first_contact(1)


def seat_clearance(
    model: mujoco.MjModel,
    ids: SceneIds,
    plan: GraspPlan,
    obj: objs.AnyObject,
    q_seat: float,
    gaps: Sequence[float],
    table_top_z: float = 0.0,
) -> dict[str, list[float]]:
    """MuJoCo's jaw-to-part distance with both jaws backed off from the seat (gate GV1).

    For each ``g`` in ``gaps`` the jaw is put at closure ``q_seat - g`` and the smallest signed
    distance to the centred part is measured, per jaw. A pad that matches its design shows
    ``g`` exactly: the imprint's designed clearance is the gap at ``q_seat - clearance``.
    Returns ``{"left": [...], "right": [...], "first_contact": [q_left, q_right]}`` in metres.
    """
    nominal = objs.Placement(plan.x, plan.y, plan.yaw)
    scene = ProbeScene.pose(model, ids, plan, obj, nominal, table_top_z)
    out: dict[str, list[float]] = {"left": [], "right": []}
    for side, key in ((0, "left"), (1, "right")):
        for g in gaps:
            out[key].append(scene.distance(side, q_seat - g))
    out["first_contact"] = [scene.first_contact(0), scene.first_contact(1)]
    return out


def lateral_clearance(
    model: mujoco.MjModel,
    ids: SceneIds,
    plan: GraspPlan,
    obj: objs.AnyObject,
    q_seat: float,
    reach: float = 0.006,
    backoff: float = 1e-6,
    tol: float = 1e-6,
    table_top_z: float = 0.0,
) -> tuple[float, float]:
    """How far (m) the part can slide across the jaw (palm +x, then -x) at the seat before
    either jaw touches it (gate GV1 diagnostic).

    Both jaws stand ``backoff`` short of ``q_seat``, so the faces they seat on do not touch,
    and the part is moved along palm x by bisection on MuJoCo's signed distance. For a
    straight-walled imprint on a part whose sides run along the closing axis (a block) this is
    the designed side clearance; on curved sides it is larger. ``inf`` if nothing touches
    within ``reach``.
    """
    scene = ProbeScene.pose(model, ids, plan, obj, objs.Placement(plan.x, plan.y, plan.yaw), table_top_z)
    d = scene.data
    d.qpos[ids.finger_qpos] = max(0.0, q_seat - backoff)
    home = d.qpos[ids.object_qpos : ids.object_qpos + 3].copy()
    across = np.array([math.cos(plan.yaw), math.sin(plan.yaw), 0.0])  # palm x in the world
    jaw_geoms = np.concatenate(scene.jaw_geoms)

    def touching(offset: float) -> bool:
        d.qpos[ids.object_qpos : ids.object_qpos + 3] = home + offset * across
        mujoco.mj_kinematics(scene.model, d)
        return any(
            mujoco.mj_geomDistance(scene.model, d, int(g1), int(g2), 0.01, None) <= 0.0
            for g1 in jaw_geoms
            for g2 in ids.object_geoms
        )

    out = []
    for sign in (1.0, -1.0):
        if touching(0.0):
            out.append(0.0)
            continue
        if not touching(sign * reach):
            out.append(math.inf)
            continue
        lo, hi = 0.0, reach
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if touching(sign * mid) else (mid, hi)
        out.append(hi)
    return out[0], out[1]
