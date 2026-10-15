"""Open-loop grasp controller: a time-based state machine with smooth targets.

Phases (start times are fixed per task, identical for every entrant)::

    SETTLE   palm waits at the approach pose, fingers open, object settles
    DESCEND  palm descends (minimum-jerk) to the grasp height
    CLOSE    grip target ramps at constant speed to fully closed + overdrive;
             once the fingers touch the object the servo force ramps up and
             saturates at the grip force limit F; a short squeeze dwell follows
    LIFT     palm rises by ``lift_height`` (minimum-jerk)
    HOLD     palm holds still until the episode ends

The controller targets the *nominal* placement: the placement's pose error and
the seed's jitter are unknown to it, as perception error would be.

Grasp height (controller setting ``k``): the finger tips, i.e. the lowest point
of the finger geometry, go to a fraction ``k`` of the *reachable band* of tip
heights. The band runs from the lowest tip height that keeps the palm underside
(the finger roots) ``palm_clearance`` above the object's top,
``max(0, h + palm_clearance - reach)`` for a finger reaching ``reach`` below its
root, up to the object's top ``h``. A finger longer than ``h + palm_clearance``
therefore grasps at ``k * h``; a shorter one grasps higher up the object instead of
driving the palm into it. Tips never go below ``min_tip_clearance``. The approach
pose is ``approach_offset`` higher, and never lower than ``approach_clearance``
above the object's top.

During HOLD an optional :class:`~rtt.sim.perturb.Shake` moves the palm target
sinusoidally around the hold pose (robustness tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from rtt.design.types import FingerGeometry
from rtt.sim import objects as objs
from rtt.sim.perturb import Shake, axis_vector
from rtt.sim.scene import PALM_XY_LIMIT, finger_tip_depth
from rtt.sim.task import TaskConfig


class Phase(IntEnum):
    SETTLE = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    HOLD = 4


def min_jerk(s: float) -> float:
    """Minimum-jerk interpolation profile on [0, 1] (zero velocity and acceleration at both ends)."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


@dataclass(frozen=True)
class GraspPlan:
    """All targets and phase boundaries of one episode's controller."""

    x: float
    y: float
    yaw: float
    approach_z: float  # palm origin heights
    grasp_z: float
    lift_z: float
    tip_height: float  # finger-tip height above the table at the grasp pose
    reach_floor: float  # lowest palm-clear tip height (bottom of the reachable band)
    palm_gap: float  # palm underside to object top at the grasp pose (m)
    t_descend: float
    t_close: float
    t_lift: float
    t_hold: float
    close_time: float
    open_target: float
    close_target: float

    @property
    def palm_start(self) -> np.ndarray:
        """Initial palm joint values (x, y, z, yaw)."""
        return np.array([self.x, self.y, self.approach_z, self.yaw])


def reach_floor(reach: float, obj_height: float, task: TaskConfig) -> float:
    """Lowest tip height that keeps the palm ``palm_clearance`` above an object's top."""
    return max(0.0, obj_height + task.controller.palm_clearance - reach)


def grasp_fraction(setting: int, task: TaskConfig) -> float:
    fracs = task.controller.grasp_height_fracs
    if not 0 <= setting < len(fracs):
        raise ValueError(f"controller setting must be in [0, {len(fracs)}), got {setting}")
    return fracs[setting]


def plan_grasp(
    finger: FingerGeometry,
    obj: objs.AnyObject,
    placement: objs.Placement,
    setting: int,
    task: TaskConfig,
) -> GraspPlan:
    """Compute the controller targets for one finger/object/placement/setting.

    Raises ``ValueError`` for a placement outside the palm's x/y travel. The yaw
    target is wrapped to [-pi, pi] (the same closing axis), so any placement yaw
    stays inside the palm's yaw range.
    """
    if max(abs(placement.x), abs(placement.y)) > PALM_XY_LIMIT:
        raise ValueError(
            f"placement ({placement.x}, {placement.y}) is outside the palm's +-{PALM_XY_LIMIT} m reach"
        )
    c, g = task.controller, task.gripper
    table = task.physics.table_top_z
    h = objs.height(obj)
    reach = finger_tip_depth(finger)
    root_below_palm = g.palm_half_size[2] + reach  # palm origin to finger tip

    floor = reach_floor(reach, h, task)
    band_tip = floor + grasp_fraction(setting, task) * (h - floor)
    tip = max(c.min_tip_clearance, band_tip + c.grasp_z_offset)
    approach_tip = max(tip + c.approach_offset, h + c.approach_clearance)
    grasp_z = table + tip + root_below_palm
    approach_z = table + approach_tip + root_below_palm

    t_descend = c.settle_time
    t_close = t_descend + c.descend_time
    t_lift = t_close + c.close_time + c.squeeze_time
    t_hold = t_lift + c.lift_time
    return GraspPlan(
        x=placement.x,
        y=placement.y,
        yaw=math.remainder(placement.yaw, 2.0 * math.pi),
        approach_z=approach_z,
        grasp_z=grasp_z,
        lift_z=grasp_z + c.lift_height,
        tip_height=tip,
        reach_floor=floor,
        palm_gap=tip + reach - h,
        t_descend=t_descend,
        t_close=t_close,
        t_lift=t_lift,
        t_hold=t_hold,
        close_time=c.close_time,
        open_target=c.open_target,
        close_target=g.stroke + g.grip_overdrive,
    )


class GraspController:
    """Maps simulation time to actuator targets ``[x, y, z, yaw, grip]``."""

    def __init__(self, plan: GraspPlan, shake: Shake | None = None) -> None:
        self.plan = plan
        self.shake = shake
        self._ctrl = np.array([plan.x, plan.y, plan.approach_z, plan.yaw, plan.open_target])
        self._shake_dir = axis_vector(shake.axis, plan.yaw) if shake is not None else np.zeros(3)

    @property
    def shake_start(self) -> float:
        """Time the shake starts (inf without a shake)."""
        return self.plan.t_hold + self.shake.delay if self.shake is not None else math.inf

    def phase(self, t: float) -> Phase:
        p = self.plan
        if t < p.t_descend:
            return Phase.SETTLE
        if t < p.t_close:
            return Phase.DESCEND
        if t < p.t_lift:
            return Phase.CLOSE
        if t < p.t_hold:
            return Phase.LIFT
        return Phase.HOLD

    def palm_z(self, t: float) -> float:
        p = self.plan
        if t < p.t_close:
            s = min_jerk((t - p.t_descend) / (p.t_close - p.t_descend))
            return p.approach_z + s * (p.grasp_z - p.approach_z)
        s = min_jerk((t - p.t_lift) / (p.t_hold - p.t_lift))
        return p.grasp_z + s * (p.lift_z - p.grasp_z)

    def grip(self, t: float) -> float:
        p = self.plan
        s = min(max((t - p.t_close) / p.close_time, 0.0), 1.0)
        return p.open_target + s * (p.close_target - p.open_target)

    def command(self, t: float) -> np.ndarray:
        """Actuator targets at time ``t`` (the returned array is reused between calls)."""
        self._ctrl[2] = self.palm_z(t)
        self._ctrl[4] = self.grip(t)
        if self.shake is not None:
            offset = self.shake.displacement(t - self.shake_start) * self._shake_dir
            self._ctrl[0] = self.plan.x + offset[0]
            self._ctrl[1] = self.plan.y + offset[1]
            self._ctrl[2] += offset[2]
        return self._ctrl
