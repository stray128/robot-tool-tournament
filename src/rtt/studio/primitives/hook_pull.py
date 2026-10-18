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
   closes it below vertical. A ``force_n`` at_least is a goal on top of the motion's: the pull
   goes on to it (:func:`_pull_on`) before it lets go.
5. **Release and retract**: open over 0.5 s, rise 60 mm, settle 0.3 s.

**A lid's lip** (a horizontal panel with nothing to grip) is hooked instead: the tool's sideways
hook from its calibration (``Footprint.hook``: the geoms near the tip that reach furthest beyond
the rest of the tool, a soft arm's horizontal arm) is turned toward the hinge, lowered outside
the free edge (stood off :data:`HOOK_STANDOFF`, plus whatever of the tool reaches further than
the hook), moved in under it, guarded, by the hook's depth less :data:`HOOK_MARGIN` (at most the
lip less 2 mm and :data:`HOOK_MAX`), raised onto the underside, and carried along the arc with the
fingers open. A tool without a hook reaches nothing under the lip and fails honestly. Every
guarded move stops on contact with any object in the scene, not only the step's.

**Predicate** (:class:`HookPullPredicate`): ``angle_deg`` or ``travel_mm`` (the template state
when the step ends, after the release), ``slip_mm`` (the largest displacement of the handle
relative to the tool, in the tool's frame (the panel's for a hooked lip), from the end of the
engagement to the release, counted while the tool touches the object or the object lags more
than 1 deg or 2 mm behind where the tool is taking it: a lid that falls open ahead of the hook
past vertical has not slipped), ``force_n`` (peak net force of the tool on the object),
``penetration_mm``; then the shared rule. A step that fails its criteria after slipping more
than 10 mm is ``slip``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
    FORCE_HOLD_S,
    Guarded,
    Path,
    Tool,
    contacts,
    criterion_bounds,
    force_allowed,
    force_goal,
    heading_for_axis,
    observed_state,
    panel_angle,
    panel_normal,
    press_reach,
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
HOOK_STANDOFF = 0.010  # m in front of a lid's free edge before the hook goes under it
HOOK_GAP = 0.002  # m under the lid's underside while the hook goes in
HOOK_PRELOAD = 0.001  # m the hook then presses up into the underside
HOOK_MARGIN = 0.003  # m of the hook kept short of its full reach
HOOK_MAX = 0.040  # m: the deepest a hook goes under a lip
SLIP_FAIL_MM = 10.0
PULL_N = 30.0  # N, the pull limit while pulling on to a force goal [assumed]
SLIP_TOUCH_N = 0.3  # N of tool-object contact that counts as touching the feature


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


def _pull_on(ctx: StepContext, motion: _Motion, delta: float, radius: float, speed: float) -> dict[str, Any]:
    """Pulling on to a ``force_n`` at_least, a goal on top of the motion's: once the pull has
    reached its goal, it goes on along the motion (toward the object's end, and into it by
    :func:`~._common.press_reach` of the pull limit, unless the step bounds the motion on both
    sides) while the measured force is under the pull limit, until the force has been over the
    goal for :data:`~._common.FORCE_HOLD_S`, within the equal budget ``force_press`` plus that
    travel."""
    goal = force_goal(ctx)
    if goal is None:
        return {"force_goal": None, "ext_t": 0.0}
    info = ctx.obj.handles.info
    q = "angle_deg" if motion.hinged else "travel_mm"
    lo, hi = criterion_bounds(ctx, q)
    scale = radius if motion.hinged else 1.0
    sense = 1.0 if delta >= 0.0 else -1.0
    end = (info["max_angle_rad"] if motion.hinged else info["travel_m"]) if sense > 0 else 0.0
    limit = force_allowed(ctx, PULL_N)
    cap = (
        0.0
        if (lo is not None and hi is not None)
        else max(0.0, sense * (end - motion.goal)) + press_reach(ctx, limit) / scale
    )
    rate = speed / scale
    return {
        "force_goal": goal,
        "force_held": 0.0,
        "pull_limit": limit,
        "ext": 0.0,
        "ext_cap": cap,
        "ext_rate": rate,
        "ext_sense": sense,
        "ext_t": 1.3 * cap / rate + BUDGET_S["force_press"],
    }


def _pull_on_step(
    t: float, ctx: StepContext, run: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData, t_goal: float
) -> bool:
    """One control tick of :func:`_pull_on`; ``True`` once the force goal is met after ``t_goal``
    (the end of the pull to the motion's goal)."""
    if run["force_goal"] is None:
        return False
    force = float(np.linalg.norm(contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids).net))
    dt = ctx.task.controller.control_period
    if force >= run["force_goal"]:
        run["force_held"] += dt
    if t < t_goal:
        return False
    if run["force_held"] >= FORCE_HOLD_S:
        return True
    if force < run["pull_limit"]:
        run["ext"] = min(run["ext_cap"], run["ext"] + run["ext_rate"] * dt)
    return False


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
        info = ctx.obj.handles.info
        if motion.hinged and info["orientation"] == 0.0 and info["feature"] == 0.0:
            return self._plan_hook(ctx, model, data, tool, motion, handle, start)
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
        speed = float(step.params["speed_mm_s"]) / 1000.0
        pull_t = max(0.5, 1.875 * length / speed)
        on = _pull_on(ctx, motion, delta, radius, speed)
        t_release = t_pull + pull_t + on["ext_t"]
        t_end = t_release + RELEASE_S + 0.4 + 0.6 + 0.3 + BUDGET_S["hook_slack"]
        sch.engaged = t_pull
        sch.released = t_release
        sch.values.update(t_close=t_close, t_release=t_release, pre=pre, sense=math.copysign(1.0, delta))
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
            **on,
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
        if run.get("mode") == "hook":
            return self._command_hook(t, ctx, run, model, data)
        guard: Guarded = run["guard"]
        if t < guard.t0:
            wrist = run["path"].at(t)
        elif t < run["t_pull"]:
            pressing = contacts(model, data, tool_geoms(ctx.scene), ctx.scene.ids.object_geoms).normal
            wrist = guard.at(min(t, guard.end), pressing) if t <= guard.end else guard.at(guard.end, 1e9)
        elif t < v["t_release"]:
            if run["engaged"] is None:
                run["engaged"] = guard.at(guard.end, 1e9)
            base = run["engaged"]
            motion: _Motion = run["motion"]
            s = run["delta"] * min_jerk((t - run["t_pull"]) / run["pull_t"])
            if run["force_goal"] is not None:
                s += run["ext_sense"] * run["ext"]
                if _pull_on_step(t, ctx, run, model, data, run["t_pull"] + run["pull_t"]):
                    v["t_release"] = t  # every goal met: let go now
                    sch.released = t
            point = base[:3] + rot_z(run["yaw"] + run["roll"]) @ tool.tcp
            p = motion.carry(point, s)
            v["progress"] = motion.current + s  # where the tool is taking the object
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

    # ------------------------------------------------------------------ hooking under a lid's lip

    def _plan_hook(
        self,
        ctx: StepContext,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tool: Tool,
        motion: _Motion,
        handle: np.ndarray,
        start: np.ndarray,
    ) -> StepPlan:
        """A lid's lip has nothing to grip: the tool's sideways hook (``Footprint.hook``, open) goes
        under the free edge from the front and lifts it along the arc (module docstring)."""
        step, info = ctx.step, ctx.obj.handles.info
        sch = schedule(ctx)
        roll = float(start[4])
        fp = tool.footprint(0.0)
        d, depth, far, top = fp.hook()
        # the approach keeps the whole tool outside the edge: whatever reaches further toward the
        # lid than the hook's end (all of a tool without a hook) stands off by that much more
        standoff = HOOK_STANDOFF + max(0.0, fp.front(d) - far)
        out = _outward(ctx, data, handle, motion)
        u_in = -out  # from the free edge toward the hinge, horizontal
        heading = math.atan2(float(u_in[1]), float(u_in[0])) - math.atan2(float(d[1]), float(d[0]))
        heading += 2.0 * math.pi * round((float(start[3]) + roll - heading) / (2.0 * math.pi))
        yaw = heading - roll
        n_out = panel_normal(ctx, motion.current)
        underside = handle - 0.5 * info["thickness_m"] * n_out  # the free edge's underside
        insert = float(np.clip(depth - HOOK_MARGIN, 0.0, min(info["feature_m"] - 0.002, HOOK_MAX)))
        tip = d * far + np.array([0.0, 0.0, top])  # the hook's far end, on its top (wrist frame)
        down = np.array([0.0, 0.0, -HOOK_GAP])
        engaged = wrist_target(underside + insert * u_in + down, tip, yaw, roll)
        pre = wrist_target(underside - standoff * u_in + down, tip, yaw, roll)
        path = Path.start(start)
        above = pre + np.array([0.0, 0.0, 0.08, 0.0, 0.0])
        transit(path, above, max(above[2], safe_palm_z(ctx, model, data, tool)))
        path.to(pre, 0.8)
        guard = Guarded(path.last, engaged, path.end, 1.0 + BUDGET_S["engage_slack"])
        t_seat = guard.end
        t_pull = t_seat + 0.3  # rise onto the underside: the gap plus 1 mm of preload
        contact = tip - 0.5 * insert * d  # the hook point under the lid that lifts it (wrist frame)
        delta = motion.goal - motion.current
        radius = float(np.linalg.norm(np.cross(motion.axis, handle - motion.point)))
        speed = float(step.params["speed_mm_s"]) / 1000.0
        pull_t = max(0.5, 1.875 * abs(delta) * radius / speed)
        on = _pull_on(ctx, motion, delta, radius, speed)
        t_release = t_pull + pull_t + on["ext_t"]
        t_end = t_release + 0.6 + 0.6 + 0.3 + BUDGET_S["hook_slack"]
        sch.engaged = t_pull
        sch.released = t_release
        sch.values.update(t_close=t_seat, t_release=t_release, pre=0.0, sense=math.copysign(1.0, delta))
        sch.values["slip_point"] = contact.tolist()  # type: ignore[assignment]
        run = {
            "mode": "hook",
            "path": path,
            "guard": guard,
            "motion": motion,
            "delta": delta,
            "t_seat": t_seat,
            "t_pull": t_pull,
            "pull_t": pull_t,
            "t_release": t_release,
            "yaw": yaw,
            "roll": roll,
            "contact": contact,
            "u_in": u_in,
            "insert": insert,
            "standoff": standoff,
            "seated": None,
            "last": None,
            **on,
        }
        return StepPlan(
            duration_s=min(t_end, 30.0),
            values={"run": run, "schedule": sch},
            note=f"hook pull: under the lip by {1000 * insert:.0f} mm (hook {1000 * depth:.0f} mm), arc by {delta:.3f}",
        )

    def _command_hook(
        self, t: float, ctx: StepContext, run: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        tool: Tool = schedule(ctx).tool
        guard: Guarded = run["guard"]
        lift = np.array([0.0, 0.0, HOOK_GAP + HOOK_PRELOAD, 0.0, 0.0])
        if t < guard.t0:
            wrist = run["path"].at(t)
        elif t < run["t_seat"]:
            pressing = contacts(model, data, tool_geoms(ctx.scene), ctx.scene.ids.object_geoms).normal
            wrist = guard.at(t, pressing)
        elif t < run["t_pull"]:
            wrist = guard.at(guard.end, 1e9) + lift * min_jerk((t - run["t_seat"]) / 0.3)
        elif t < run["t_release"]:
            if run["seated"] is None:
                run["seated"] = guard.at(guard.end, 1e9) + lift
            base = run["seated"]
            motion: _Motion = run["motion"]
            s = run["delta"] * min_jerk((t - run["t_pull"]) / run["pull_t"])
            if run["force_goal"] is not None:
                s += run["ext_sense"] * run["ext"]
                if _pull_on_step(t, ctx, run, model, data, run["t_pull"] + run["pull_t"]):
                    run["t_release"] = t  # every goal met: let go now
                    schedule(ctx).released = t
                    schedule(ctx).values["t_release"] = t
            point = base[:3] + rot_z(run["yaw"] + run["roll"]) @ run["contact"]
            schedule(ctx).values["progress"] = motion.current + s  # where the tool is taking the lid
            wrist = wrist_target(motion.carry(point, s), run["contact"], run["yaw"], run["roll"])
            run["last"] = wrist
        else:  # draw the hook back out from under the lid, then rise
            last = run["last"] if run["last"] is not None else guard.at(guard.end, 1e9)
            back = min_jerk(min(1.0, max(0.0, (t - run["t_release"]) / 0.6)))
            up = min_jerk(min(1.0, max(0.0, (t - run["t_release"] - 0.6) / 0.6)))
            shift = -(run["insert"] + run["standoff"]) * back * run["u_in"] + np.array(
                [0.0, 0.0, RETRACT * up]
            )
            wrist = last + np.concatenate([shift, [0.0, 0.0]])
        return PrimitiveCommand(wrist, tool.controls(t, 0.0, model, data, Phase.SETTLE))


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
            point = sch.values.get("slip_point")
            if point is not None:  # a hook: its lifting point in the panel's frame
                rel = data.site_xmat[site].reshape(3, 3).T @ (
                    tool_point_world(ctx.scene, data, np.asarray(point)) - data.site_xpos[site]
                )
            else:
                q = data.qpos
                heading = float(q[ctx.scene.ids.palm_qpos[3]] + q[ctx.scene.ids.roll_qpos])
                rel = rot_z(heading).T @ (
                    data.site_xpos[site] - tool_point_world(ctx.scene, data, sch.tool.tcp)
                )
            ref = sch.values.get("rel0")
            if ref is None:
                sch.values["rel0"] = rel.tolist()  # type: ignore[assignment]
            elif con.normal > SLIP_TOUCH_N or self._lags(ctx, acc, sch.values):
                # slip is the tool moving along the feature while it touches it, or losing it while
                # the object falls behind; an object that runs ahead of the tool (a lid falling open
                # past vertical) has not slipped from it
                slip = 1000.0 * float(np.linalg.norm(rel - np.asarray(ref)))
                acc["slip_mm"] = max(acc["slip_mm"], slip)

    @staticmethod
    def _lags(ctx: StepContext, acc: Mapping[str, float], v: Mapping[str, Any]) -> bool:
        progress = v.get("progress")
        if progress is None:
            return True
        if ctx.obj.kind is TemplateKind.HINGED_LID:
            now, tol = math.radians(acc["angle_deg"]), math.radians(1.0)
        else:
            now, tol = acc["travel_mm"] / 1000.0, 0.002
        return v.get("sense", 1.0) * (progress - now) > tol

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
