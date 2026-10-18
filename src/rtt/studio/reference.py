"""Trivial reference controllers and one reference predicate for the new primitives (S1 only).

They exist so the contract runs end to end before the real controllers land (track S2): each
compiles, plans a short min-jerk wrist path from the step's handle site and keeps the tool open.

* ``push``: down onto the handle (the top of a part), 3 mm past contact, back up.
* ``twist``: down onto the handle, then the wrist roll turns by the step's ``angle_deg`` target
  (90 deg if it has none) at the step's speed, capped at the wrist's roll speed limit.
* ``insert``: over the template's seat site (``socket`` or ``seat``), then down by the step's
  ``depth_mm`` target (10 mm if none) at the step's speed.
* ``hook_pull``: beside the handle, then 100 mm (or the ``travel_mm`` target) horizontally away
  from the object's body.

:class:`ReferencePredicate` records the template state, the peak tool-object contact force, the
peak roll torque and the deepest penetration, then applies
:func:`rtt.studio.plan.criteria_verdict`. Results from this set say ``controllers =
"reference"`` and are never evidence for a recommendation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import mujoco
import numpy as np

from rtt.sim.controller import min_jerk
from rtt.studio.plan import (
    PrimitiveCommand,
    StepContext,
    StepPlan,
    StepVerdict,
    contact_summary,
    criteria_verdict,
    register_primitive,
)
from rtt.studio.types import PrimitiveKind, TemplateKind

__all__ = ["ReferenceController", "ReferencePredicate", "waypoint_targets"]

CLEARANCE = 0.04  # m above the handle before engaging
_SEAT = {"plug": "socket", "member": "seat"}


def _target(ctx: StepContext, quantity: str, default: float) -> float:
    for c in ctx.step.success:
        if c.quantity == quantity:
            return c.at_least if c.at_least is not None else (c.at_most if c.at_most is not None else default)
    return default


def waypoint_targets(times: np.ndarray, poses: np.ndarray, t: float) -> np.ndarray:
    """Min-jerk interpolation through wrist poses ``poses[i]`` reached at ``times[i]``."""
    if t <= times[0]:
        return poses[0].copy()
    if t >= times[-1]:
        return poses[-1].copy()
    i = int(np.searchsorted(times, t, side="right")) - 1
    s = min_jerk((t - times[i]) / (times[i + 1] - times[i]))
    return poses[i] + s * (poses[i + 1] - poses[i])


@dataclass(frozen=True)
class ReferenceController:
    """A trivial waypoint controller for one new primitive (module docstring)."""

    primitive: PrimitiveKind
    name: str = "reference"

    SPEED_MM_S: ClassVar[float] = 80.0  # free-space moves

    def _wrist_at(self, ctx: StepContext, tcp_point: np.ndarray, yaw: float, roll: float) -> np.ndarray:
        palm = tcp_point - ctx.tcp_offset(yaw)
        return np.array([palm[0], palm[1], palm[2], yaw, roll])

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        step, start = ctx.step, np.asarray(ctx.wrist_start, dtype=np.float64)
        yaw, roll = float(start[3]), float(start[4])
        handle = ctx.observe_site(data, step.handle)
        if ctx.obj.kind is TemplateKind.PART:
            handle = handle + np.array([0.0, 0.0, 0.5 * ctx.obj.handles.info.get("height_m", 0.0)])
        above = handle + np.array([0.0, 0.0, CLEARANCE])
        poses = [start, self._wrist_at(ctx, above, yaw, roll)]
        durations = [self._move_s(poses[0], poses[1])]
        kind = self.primitive
        if kind is PrimitiveKind.PUSH:
            press = handle - np.array([0.0, 0.0, 0.003])
            poses += [self._wrist_at(ctx, handle, yaw, roll), self._wrist_at(ctx, press, yaw, roll)]
            poses += [poses[-1], poses[1]]
            durations += [0.8, 0.4, 0.3, 0.6]
        elif kind is PrimitiveKind.TWIST:
            angle = math.radians(_target(ctx, "angle_deg", 90.0))
            sign = -1.0 if step.params["direction"] == "cw" else 1.0
            speed = math.radians(min(float(step.params["speed_deg_s"]), ctx.scene.wrist.roll_speed_max_deg_s))
            engage = self._wrist_at(ctx, handle, yaw, roll)
            turned = engage.copy()
            turned[4] = roll + sign * abs(angle)
            # min-jerk peaks at 1.875x the mean speed: stretch the turn so the peak stays under the cap
            poses += [engage, turned, turned, self._wrist_at(ctx, above, yaw, float(turned[4]))]
            durations += [0.8, max(0.2, 1.875 * abs(angle) / speed), 0.2, 0.6]
        elif kind is PrimitiveKind.INSERT:
            seat = ctx.observe_site(data, _SEAT.get(step.handle, step.handle))
            standoff = float(step.params["standoff_mm"]) / 1000.0
            depth = _target(ctx, "depth_mm", 10.0) / 1000.0
            speed = float(step.params["speed_mm_s"]) / 1000.0
            over = self._wrist_at(ctx, seat + np.array([0.0, 0.0, standoff]), yaw, roll)
            down = self._wrist_at(ctx, seat - np.array([0.0, 0.0, depth]), yaw, roll)
            poses += [over, down, down]
            durations += [self._move_s(poses[-2], over), 1.875 * (standoff + depth) / speed, 0.3]
        else:  # hook pull
            root = data.xpos[ctx.obj.body_ids[0]] if ctx.obj.body_ids else handle
            away = handle - root
            away[2] = 0.0
            norm = float(np.linalg.norm(away))
            away = away / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
            dist = _target(ctx, "travel_mm", 100.0) / 1000.0
            speed = float(step.params["speed_mm_s"]) / 1000.0
            beside = handle + 0.015 * away
            poses += [
                self._wrist_at(ctx, beside, yaw, roll),
                self._wrist_at(ctx, beside + dist * away, yaw, roll),
            ]
            poses += [poses[-1].copy() + np.array([0.0, 0.0, CLEARANCE, 0.0, 0.0])]
            durations += [0.8, 1.875 * dist / speed, 0.6]
        times = np.concatenate([[0.0], np.cumsum(durations)])
        return StepPlan(
            duration_s=float(times[-1]),
            values={"times": times.tolist(), "poses": [list(map(float, p)) for p in poses]},
            note=f"S1 reference {kind.value}",
        )

    def _move_s(self, a: np.ndarray, b: np.ndarray) -> float:
        return max(0.4, 1.875 * float(np.linalg.norm(b[:3] - a[:3])) / (self.SPEED_MM_S / 1000.0))

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        times = np.asarray(plan.values["times"], dtype=np.float64)
        poses = np.asarray(plan.values["poses"], dtype=np.float64)
        return PrimitiveCommand(waypoint_targets(times, poses, t), np.asarray(ctx.head.build.ctrl_open))


@dataclass(frozen=True)
class ReferencePredicate:
    """Records state, peak force, peak roll torque and penetration; then the shared rule."""

    primitive: PrimitiveKind
    name: str = "reference"

    def observe(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None:
        acc.update(ctx.object_state(model, data))
        force, depth, _ = contact_summary(model, data, ctx.scene.ids.active_geoms, ctx.obj.geom_ids)
        acc["force_n"] = max(acc.get("force_n", 0.0), force)
        acc["penetration_mm"] = max(acc.get("penetration_mm", 0.0), 1000.0 * depth)
        if ctx.step.info.uses_roll:
            torque = abs(float(data.actuator_force[ctx.scene.ids.wrist_actuators[4]]))
            acc["torque_nm"] = max(acc.get("torque_nm", 0.0), torque)

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict:
        return criteria_verdict(ctx.step, acc)


for _kind in (PrimitiveKind.HOOK_PULL, PrimitiveKind.TWIST, PrimitiveKind.INSERT, PrimitiveKind.PUSH):
    register_primitive(ReferenceController(_kind), ReferencePredicate(_kind), controllers="reference")
