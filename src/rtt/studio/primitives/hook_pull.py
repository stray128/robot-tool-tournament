"""The ``hook_pull`` primitive: engage a handle or lip and pull it along the object's arc or line.

Objects: ``hinged_lid`` (lid or door) and ``drawer``; handle ``handle`` (the template's feature).

**Controller** (:class:`HookPullController`):

1. **Engagement plan.** The engagement point is the observed ``handle`` site. The closing axis
   of the tool (its y) is set across the feature: across a horizontal bar or a lip (along the
   object's y), along the face for a door's vertical bar, across the knob's neck. The heading
   is set with the wrist yaw (the roll stays).
2. **Approach** (``approach``): ``top`` descends onto the feature from above; ``front`` descends
   30 mm out in front of it (away from the hinge or the carcass) and moves in horizontally.
   The tool is open (fingers open, cup vacuum off, cable slack).
3. **Engage**: the head's own v1 primitive closes on the shared ramp (1.0 s, 0.3 s squeeze):
   fingers grip the bar or pinch the lip, a cup seals only if the face under it is flat, a soft
   arm curls.
4. **Pull**: the engagement point is carried along the object's motion, a rotation about the
   hinge axis (through the observed ``hinge`` site, axis from the template) or a translation
   along the drawer's slide, from the present state to the goal, at the step's ``speed_mm_s``
   (min-jerk, so the peak is 1.875 x the mean). A door's heading turns with the door. Goal: the
   criterion on ``angle_deg`` or ``travel_mm`` plus a margin (8 deg, 15 mm). A lid that has to
   stay open is pulled past vertical (at least 95 deg) when its stop allows, because gravity
   closes it below vertical.
5. **Release and retract**: open over 0.5 s, rise 60 mm, settle 0.3 s.

**Predicate** (:class:`HookPullPredicate`): ``angle_deg`` or ``travel_mm`` (the template state
when the step ends, after the release), ``slip_mm`` (the largest displacement of the handle relative to the
tool's TCP, in the tool's frame, from the end of the engagement to the release: the tool frame
turns with a door and not with a lid, whose handle turns in the grip), ``force_n`` (peak net force of the tool on the object), ``penetration_mm``; then the
shared rule. A step that fails its criteria after slipping more than 10 mm is ``slip``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import mujoco
import numpy as np

from rtt.sim.controller import Phase, min_jerk
from rtt.studio.plan import (
    PrimitiveCommand,
    StepContext,
    StepPlan,
    StepVerdict,
    criteria_verdict,
    register_primitive,
)
from rtt.studio.primitives._common import (
    BUDGET_S,
    Guarded,
    Path,
    Tool,
    contacts,
    criterion_bounds,
    heading_for_axis,
    observed_state,
    panel_angle,
    panel_normal,
    rot_z,
    rotate,
    safe_palm_z,
    schedule,
    tool_geoms,
    tool_of,
    tool_point_world,
    transit,
    wrist_target,
)
from rtt.studio.types import PrimitiveKind, TemplateKind

__all__ = ["HookPullController", "HookPullPredicate"]

ANGLE_MARGIN = math.radians(8.0)
TRAVEL_MARGIN = 0.015
LID_STAY_OPEN = math.radians(100.0)
RELEASE_S = 0.5
FRONT_STANDOFF = 0.030
RETRACT = 0.060
RETRACT_OUT = 0.040  # m away from the face the feature stands on, before rising
SLIP_FAIL_MM = 10.0


def _object_yaw(ctx: StepContext) -> float:
    return math.radians(ctx.task_spec.object(ctx.step.object).pose.yaw_deg)


@dataclass(frozen=True)
class _Motion:
    """The object's motion: a hinge (axis through ``point``) or a slide (along ``axis``)."""

    hinged: bool
    point: np.ndarray
    axis: np.ndarray  # unit; positive = opening
    current: float  # rad or m, as observed
    goal: float

    def carry(self, p: np.ndarray, s: float) -> np.ndarray:
        """Where a point ``p`` attached to the object goes when the motion progresses by ``s``."""
        if self.hinged:
            return self.point + rotate(p - self.point, self.axis, s)
        return p + s * self.axis


def _motion(ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> _Motion:
    obj, info = ctx.obj, ctx.obj.handles.info
    state = observed_state(ctx, data)
    if obj.kind is TemplateKind.HINGED_LID:
        axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
        current = math.radians(state["angle_deg"])
        lo, hi = criterion_bounds(ctx, "angle_deg")
        top = info["max_angle_rad"] - math.radians(2.0)
        if lo is not None:
            goal = math.radians(lo) + ANGLE_MARGIN
            if info["orientation"] == 0.0 and info["max_angle_rad"] > LID_STAY_OPEN + math.radians(2.0):
                goal = max(goal, LID_STAY_OPEN)  # a lid stays open only past vertical
            goal = min(goal, top)
        elif hi is not None:
            goal = max(0.0, math.radians(hi) - ANGLE_MARGIN)
        else:
            goal = min(top, math.radians(90.0))
        return _Motion(True, ctx.observe_site(data, "hinge"), axis, current, goal)
    axis = np.array([info["pull_x"], info["pull_y"], 0.0])
    current = state["travel_mm"] / 1000.0
    lo, hi = criterion_bounds(ctx, "travel_mm")
    if lo is not None:
        goal = min(info["travel_m"] - 0.003, lo / 1000.0 + TRAVEL_MARGIN)
    elif hi is not None:
        goal = max(0.0, hi / 1000.0 - TRAVEL_MARGIN)
    else:
        goal = 0.8 * info["travel_m"]
    return _Motion(False, ctx.observe_site(data, "front"), axis, current, goal)


def _face_out(ctx: StepContext, data: mujoco.MjData) -> np.ndarray:
    """Unit direction away from the face the feature stands on, from the observed panel angle."""
    obj, info = ctx.obj, ctx.obj.handles.info
    if obj.kind is TemplateKind.DRAWER:
        return (
            np.array([0.0, 0.0, 1.0])
            if info["handle"] == 2.0
            else np.array([info["pull_x"], info["pull_y"], 0.0])
        )
    if info["feature"] == 1.0:  # a bar on the outer face
        return panel_normal(ctx, panel_angle(ctx, data))
    axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
    radial = ctx.observe_site(data, "handle") - ctx.observe_site(
        data, "hinge"
    )  # an edge: outward in its plane
    radial = radial - axis * float(axis @ radial)
    return radial / max(1e-9, float(np.linalg.norm(radial)))


def _closing_axis(ctx: StepContext, angle: float) -> np.ndarray:
    """Across the feature (module docstring), world frame, horizontal; a door's turns with it."""
    info = ctx.obj.handles.info
    yaw = _object_yaw(ctx)
    ex, ey = rot_z(yaw)[:, 0], rot_z(yaw)[:, 1]
    if ctx.obj.kind is TemplateKind.DRAWER:
        return ex if info["handle"] == 1.0 else ey  # knob: across the neck
    if info["orientation"] == 1.0:
        axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
        return rotate(ex if info["feature"] == 1.0 else ey, axis, angle)  # a vertical bar: along the face
    return ey


def _outward(ctx: StepContext, data: mujoco.MjData, handle: np.ndarray, motion: _Motion) -> np.ndarray:
    """Horizontal unit direction from the object toward the tool's side of the feature."""
    if not motion.hinged:
        return motion.axis.copy()
    if ctx.obj.handles.info["orientation"] == 1.0:  # door: its front face side
        return panel_normal(ctx, motion.current)
    radial = handle - motion.point
    radial = radial - motion.axis * float(motion.axis @ radial)
    radial[2] = 0.0
    n = float(np.linalg.norm(radial))
    return radial / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class HookPullController:
    primitive: PrimitiveKind = PrimitiveKind.HOOK_PULL
    name: str = "studio_hook_pull"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        step = ctx.step
        tool = tool_of(ctx, ctx.obj.body_ids[-1])
        sch = schedule(ctx)
        sch.tool = tool
        motion = _motion(ctx, model, data)
        handle = ctx.observe_site(data, "handle")
        start = np.asarray(ctx.wrist_start, dtype=np.float64)
        roll = float(start[4])
        heading = heading_for_axis(
            _closing_axis(ctx, motion.current), _yaw_offset(ctx, tool), float(start[3] + roll)
        )
        yaw = heading - roll
        pre = tool.preshape(ctx.obj.handles.info["grip_m"])
        engage = wrist_target(handle, tool.tcp, yaw, roll)
        out = _outward(ctx, data, handle, motion)
        path = Path.start(start)
        lift = np.array([0.0, 0.0, 0.08, 0.0, 0.0])
        safe = safe_palm_z(ctx, model, data, tool)
        if step.params["approach"] == "front":
            stand = engage.copy()
            stand[:2] += FRONT_STANDOFF * out[:2]
            transit(path, stand + lift, max(stand[2] + lift[2], safe))
            path.to(stand, 0.8)
        else:
            stand = engage + lift
            transit(path, stand, max(stand[2], safe))
        guard = Guarded(path.last, engage, path.end, 1.0 + BUDGET_S["engage_slack"])
        t_close = guard.end
        t_pull = t_close + 1.0 + 0.3
        delta = motion.goal - motion.current
        radius = float(np.linalg.norm(np.cross(motion.axis, handle - motion.point))) if motion.hinged else 1.0
        length = abs(delta) * (radius if motion.hinged else 1.0)
        pull_t = max(0.5, 1.875 * length / (float(step.params["speed_mm_s"]) / 1000.0))
        t_release = t_pull + pull_t
        t_end = t_release + RELEASE_S + 0.4 + 0.6 + 0.3 + BUDGET_S["hook_slack"]
        sch.engaged = t_pull
        sch.released = t_release
        sch.values.update(t_close=t_close, t_release=t_release, pre=pre)
        run = {
            "path": path,
            "guard": guard,
            "motion": motion,
            "delta": delta,
            "t_pull": t_pull,
            "pull_t": pull_t,
            "yaw": yaw,
            "roll": roll,
            "engaged": None,  # the commanded TCP point when the pull starts
            "last": None,
        }
        return StepPlan(
            duration_s=min(t_end, 30.0),
            values={"run": run, "schedule": sch},
            note=f"hook pull: {'arc' if motion.hinged else 'line'} by {delta:.3f}, {step.params['approach']} approach",
        )

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        sch = plan.values["schedule"]
        run = plan.values["run"]
        tool = sch.tool
        v = sch.values
        guard: Guarded = run["guard"]
        if t < guard.t0:
            wrist = run["path"].at(t)
        elif t < run["t_pull"]:
            pressing = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids).normal
            wrist = guard.at(min(t, guard.end), pressing) if t <= guard.end else guard.at(guard.end, 1e9)
        elif t < v["t_release"]:
            if run["engaged"] is None:
                run["engaged"] = guard.at(guard.end, 1e9)
            base = run["engaged"]
            motion: _Motion = run["motion"]
            s = run["delta"] * min_jerk((t - run["t_pull"]) / run["pull_t"])
            point = base[:3] + rot_z(run["yaw"] + run["roll"]) @ tool.tcp
            p = motion.carry(point, s)
            dyaw = s * float(motion.axis[2]) if motion.hinged else 0.0
            wrist = wrist_target(p, tool.tcp, run["yaw"] + dyaw, run["roll"])
            run["last"] = wrist
        else:
            last = run["last"] if run["last"] is not None else guard.at(guard.end, 1e9)
            if run.get("out") is None:
                run["out"] = _face_out(ctx, data)
            back = min_jerk(min(1.0, max(0.0, (t - v["t_release"] - RELEASE_S) / 0.4)))
            up = min_jerk(min(1.0, max(0.0, (t - v["t_release"] - RELEASE_S - 0.4) / 0.6)))
            shift = RETRACT_OUT * back * run["out"] + np.array([0.0, 0.0, RETRACT * up])
            wrist = last + np.concatenate([shift, [0.0, 0.0]])
        pre = v["pre"]
        closure = pre + (1.0 - pre) * min(max((t - v["t_close"]) / 1.0, 0.0), 1.0)
        if t >= v["t_release"]:  # open back to the pre-shape only, so no finger shoves the panel
            closure = pre + (1.0 - pre) * max(0.0, 1.0 - (t - v["t_release"]) / RELEASE_S)
        phase = Phase.HOLD if t >= v["t_close"] else Phase.DESCEND
        return PrimitiveCommand(wrist, tool.controls(t, closure, model, data, phase))


def _yaw_offset(ctx: StepContext, tool: Tool) -> float:
    """The head's v1 yaw offset (a hand turned to close along its own axis); 0 for cups and arms."""
    if tool.mechanism != "fingers":
        return 0.0
    from rtt.sim.objects import Placement
    from rtt.studio.primitives._pick import _slug

    return float(
        tool.primitive.plan(
            _slug("studio_feature", 0.010, 0.020), Placement(), ctx.task, ctx.head.build
        ).yaw_offset
    )


@dataclass(frozen=True)
class HookPullPredicate:
    primitive: PrimitiveKind = PrimitiveKind.HOOK_PULL
    name: str = "studio_hook_pull"

    def observe(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None:
        sch = schedule(ctx)
        acc.update(ctx.object_state(model, data))
        con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
        acc["force_n"] = max(acc.get("force_n", 0.0), float(np.linalg.norm(con.net)))
        acc["penetration_mm"] = max(acc.get("penetration_mm", 0.0), 1000.0 * con.depth)
        acc.setdefault("slip_mm", 0.0)
        if sch.active(t) and sch.tool is not None:
            site = ctx.obj.site_ids["handle"]
            q = data.qpos
            heading = float(q[ctx.scene.ids.palm_qpos[3]] + q[ctx.scene.ids.roll_qpos])
            rel = rot_z(heading).T @ (data.site_xpos[site] - tool_point_world(ctx.scene, data, sch.tool.tcp))
            ref = sch.values.get("rel0")
            if ref is None:
                sch.values["rel0"] = rel.tolist()  # type: ignore[assignment]
            else:
                slip = 1000.0 * float(np.linalg.norm(rel - np.asarray(ref)))
                acc["slip_mm"] = max(acc["slip_mm"], slip)

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict:
        out = criteria_verdict(ctx.step, acc)
        if (
            not out.success
            and out.failure_mode == "criterion_not_met"
            and acc.get("slip_mm", 0.0) > SLIP_FAIL_MM
        ):
            return StepVerdict(False, "slip", dict(acc))
        return out


register_primitive(HookPullController(), HookPullPredicate())
