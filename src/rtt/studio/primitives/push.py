"""The ``push`` primitive: press or close an object with a force limit until it reaches its target.

Objects and handles: ``hinged_lid`` (``lid``: close, or open further, a lid or a door),
``drawer`` (``drawer``: close it), ``part`` (``part``: press it down) and ``port_plug``
(``plug``: press a plug that sits in its socket home).

**Controller** (:class:`PushController`):

1. **Tool.** Fingers close into a fist (closure 1); a cup keeps its vacuum off and a soft arm its
   cable slack. The tool pushes with its tip region: the contact point is the tool's lowest point,
   moved out to the side of its calibrated footprint (closure 1) that faces the push.
2. **Where it pushes** (from observed sites and the template geometry; the object's angle or
   travel is the observed one):

   * a panel: on its outer face 15 mm in from the free edge (the most leverage), along the arc
     about the hinge to the ``angle_deg`` goal (3 deg past it). A bar handle stands off the face
     there: a lid with one is pushed on the bar's outer side, just on the hinge side of it
     (:data:`BAR_CLEAR`), so the tool, which reaches from its tip toward the free edge, meets the
     whole bar; a door with one at the middle of its outer face (the ``lid`` site), clear of the
     bar that runs up it. A lid open past vertical that has to close is pushed only to 80 deg;
     its weight closes it, and the tool then presses the closed lid home from above at the middle
     of its outer face (a second, downward push);
   * a drawer: on the front of its handle (the part that stands out), along the slide to the
     ``travel_mm`` goal (3 mm past it);
   * a part: straight down on its top until the press force; a plug: straight down on its top to
     the ``depth_mm`` goal (0.5 mm past it; without one, seated).

3. **Approach** (S2e): a transit to above the standoff point (:data:`APPROACH` off it), then a
   guarded move in (:data:`APPROACH_S`; it stops advancing while the tool presses on any object
   harder than the guard force or the push limit, and the push then starts where it stopped).
   The standoff (:data:`STANDOFF`) is the whole tool's: whatever reaches further toward the face
   than the contact point (a wrist flange, a long finger, :func:`_overhang`) stands off it too.
   The push meets the face, and its object its stop (the last :data:`END_MM`), at
   :data:`TOUCH_SPEED` at most, so an impact stays under the limit.
4. **Force-limited progress** (force feedback through the wrist only): the push advances at the
   step's ``speed_mm_s`` while the tool's net force on the object is under the limit (80 % of
   ``max_force_n``, else :data:`PUSH_N`; a ``force_n`` at_most limits it the same way), and holds
   while it is over, within the equal budget :data:`~._common.BUDGET_S` ``push_slack``. A press
   stops 0.5 s after reaching :data:`PRESS_N` (or 60 % of the limit).
5. **Every goal** (``docs/STUDIO_PRIMITIVES.md`` section 10): a push to a criterion on the
   object's state ends when the observed state meets it (:data:`MET_DEG`, :data:`MET_MM`,
   :data:`MET_DEPTH_MM` inside its bounds), not when the command arrives (the wrist servo lags
   under the object's resistance), going on up to :func:`~._common.press_reach` of the limit
   past its path. A ``force_n`` at_least is a goal of the last push on top of its travel: it
   presses on (within that reach, while the observed state still meets its criteria) until the
   force is reached and held 0.5 s, within the equal budget ``force_press`` plus that travel. A
   push whose observed state meets its criteria while it is held at its limit (a panel or a
   drawer shut against its stop) is done.
6. **Retract**: the wrist target comes back to where the wrist is (the push stops), hold 0.3 s,
   back off 30 mm along the face normal, rise 60 mm.

**Predicate** (:class:`PushPredicate`): the template state at the end (``angle_deg``,
``travel_mm``, ``depth_mm``, ``lift_mm``/``tilt_deg``), ``force_n`` (the peak net force of the
tool on the object), ``penetration_mm``; then the shared rule.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from rtt.effectors.scene import FLANGE_RADIUS
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
    GUARD_FORCE_N,
    Guarded,
    Path,
    Tool,
    contacts,
    criterion_bounds,
    force_allowed,
    force_goal,
    observed_state,
    panel_normal,
    press_reach,
    rot_z,
    rotate,
    safe_palm_z,
    schedule,
    tool_geoms,
    tool_of,
    transit,
    wrist_target,
)
from rtt.studio.types import PrimitiveKind, TemplateKind

__all__ = ["PushController", "PushPredicate"]

PUSH_N = 30.0  # N, the default push limit [assumed]
PRESS_N = 10.0  # N, a press on a part or a closed lid [assumed]
ANGLE_PAST = math.radians(3.0)
TRAVEL_PAST = 0.003
EDGE_IN = 0.015  # m in from a panel's free edge
BAR_CLEAR = 0.005  # m: a lid with a bar handle is pushed this far on the hinge side of the bar
GRAVITY_HANDOFF = math.radians(80.0)
STANDOFF = 0.010  # m off the face before a push starts
APPROACH = 0.030
APPROACH_S = 0.8  # s of the guarded move into a push's standoff point
TOUCH_SPEED = 0.015  # m/s: a push meets the face no faster than this [assumed]
TOUCH_MM = 0.005  # m past the face it keeps to that speed (aim errors) [assumed]
END_MM = 0.020  # m: the last of a push's travel, where its object meets its stop, at that speed [assumed]
MET_DEG = 1.0  # the observed state inside a criterion's bound by this much meets it [assumed]
MET_MM = 1.0
MET_DEPTH_MM = 0.2
BACK_OFF = 0.030
RETRACT = 0.060
_DOWN = np.array([0.0, 0.0, -1.0])


@dataclass
class Segment:
    """One push: the push point ``point(u)`` and the face normal ``normal(u)`` for ``u`` in [0, 1],
    ``length`` m of travel, and an optional press force to reach and hold. ``reach``: the travel
    is a goal (a press on a part only has room to press into); ``met``: whether the observed
    object state meets the step's criteria on it (``None``: the travel alone is the goal)."""

    point: Callable[[float], np.ndarray]
    normal: Callable[[float], np.ndarray]
    length: float
    press: float | None = None
    reach: bool = True
    met: Callable[[mujoco.MjData], bool] | None = None
    pressing_on: bool = False  # a force goal asks for more than the travel or the press alone
    state: dict[str, Any] = field(default_factory=dict)


def _state_met(ctx: StepContext, quantity: str, margin: float) -> Callable[[mujoco.MjData], bool] | None:
    """The step's criteria on the object's ``quantity`` as a test of the observed state, inside
    each bound by ``margin`` (at most a quarter of a window); ``None`` without a criterion."""
    lo, hi = criterion_bounds(ctx, quantity)
    if lo is None and hi is None:
        return None
    m = margin if (lo is None or hi is None) else min(margin, 0.25 * (hi - lo))

    def met(data: mujoco.MjData) -> bool:
        v = observed_state(ctx, data)[quantity]
        return (lo is None or v >= lo + m) and (hi is None or v <= hi - m)

    return met


def _segments(ctx: StepContext, data: mujoco.MjData) -> list[Segment]:
    """The pushes; a ``force_n`` at_least is a goal of the last one on top of its travel goal
    (module docstring): it never stops at the first goal it meets."""
    segs = _pushes(ctx, data)
    goal = force_goal(ctx)
    last = segs[-1]
    if goal is not None and (last.reach or goal > (last.press or 0.0)):
        last.press = max(last.press or 0.0, goal)
        last.pressing_on = True
    return segs


def _pushes(ctx: StepContext, data: mujoco.MjData) -> list[Segment]:
    obj, info, step = ctx.obj, ctx.obj.handles.info, ctx.step
    state = observed_state(ctx, data)
    press = min(PRESS_N, 0.6 * step.constraints.max_force_n) if step.constraints.max_force_n else PRESS_N
    if obj.kind is TemplateKind.HINGED_LID:
        axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
        hinge = ctx.observe_site(data, "hinge")
        lid0 = np.array([info["lid0_x"], info["lid0_y"], info["lid0_z"]])
        out0 = np.array([info["out0_x"], info["out0_y"], info["out0_z"]])
        mid = lid0 - 0.5 * info["thickness_m"] * out0  # panel centre on its mid-plane
        unit = mid / float(np.linalg.norm(mid))
        edge0 = 2.0 * mid - EDGE_IN * unit + 0.5 * info["thickness_m"] * out0  # outer face near the edge
        theta = math.radians(state["angle_deg"])
        if info["feature"] == 1.0 and info["orientation"] == 0.0:
            # a lid's bar handle stands off the face near the free edge, over the point above: the
            # push is planned on the bar's outer side (the clearance under it plus its diameter off
            # the face), just on the hinge side of the bar, so the move in stops clear of the bar
            # and the tool, which reaches from its tip toward the free edge, meets the whole bar
            r_bar = info["bar_radius_m"]
            handle0 = rotate(ctx.observe_site(data, "handle") - hinge, axis, -theta)
            bar_in = float((2.0 * mid - handle0) @ unit)  # the bar's middle, in from the free edge
            proud = info["feature_m"] + 2.0 * r_bar + 0.5 * info["thickness_m"]
            edge0 = 2.0 * mid - (bar_in + r_bar + BAR_CLEAR) * unit + proud * out0
        elif info["feature"] == 1.0:
            # a door's bar runs up the face near the free edge and across the tool's way down to
            # it: a door with a bar is pushed at the middle of its outer face (the template's
            # ``lid`` site), clear of the bar
            edge0 = lid0.copy()
        radius = float(np.linalg.norm(np.cross(axis, edge0)))
        lo, hi = criterion_bounds(ctx, "angle_deg")
        if hi is not None:
            goal = max(-ANGLE_PAST, math.radians(hi) - ANGLE_PAST)
        elif lo is not None:
            goal = min(info["max_angle_rad"], math.radians(lo) + ANGLE_PAST)
        else:
            goal = 0.0

        def arc(a: float, b: float, met: Callable[[mujoco.MjData], bool] | None) -> Segment:
            return Segment(
                point=lambda u: hinge + rotate(edge0, axis, a + u * (b - a)),
                normal=lambda u: panel_normal(ctx, a + u * (b - a)),
                length=abs(b - a) * radius,
                met=met,
            )

        lid = info["orientation"] == 0.0
        if lid and theta > math.radians(92.0) and goal < GRAVITY_HANDOFF:
            # the closed lid is pressed home at the centre of its outer face (the template's
            # ``lid`` site): a press near the free edge lands on a bar handle standing off it
            closed = hinge + rotate(lid0, axis, max(0.0, goal))
            return [
                arc(theta, GRAVITY_HANDOFF, None),
                Segment(
                    point=lambda u: closed + u * 0.010 * _DOWN,
                    normal=lambda u: out0,
                    length=0.010,
                    press=press,
                    reach=False,
                ),
            ]
        return [arc(theta, goal, _state_met(ctx, "angle_deg", MET_DEG))]
    if obj.kind is TemplateKind.DRAWER:
        pull = np.array([info["pull_x"], info["pull_y"], 0.0])
        p0 = ctx.observe_site(data, "handle") + info["push_out_m"] * pull
        travel = state["travel_mm"] / 1000.0
        lo, hi = criterion_bounds(ctx, "travel_mm")
        goal = lo / 1000.0 + TRAVEL_PAST if (lo is not None and hi is None) else -TRAVEL_PAST
        if hi is not None:
            goal = max(-TRAVEL_PAST, hi / 1000.0 - TRAVEL_PAST)
        span = goal - travel
        return [
            Segment(
                point=lambda u: p0 + u * span * pull,
                normal=lambda u: pull,
                length=abs(span),
                met=_state_met(ctx, "travel_mm", MET_MM),
            )
        ]
    up = np.array([0.0, 0.0, 1.0])
    if obj.kind is TemplateKind.PORT_PLUG:
        top = ctx.observe_site(data, "plug")
        lo, _ = criterion_bounds(ctx, "depth_mm")
        goal = (lo / 1000.0 + 0.0005) if lo is not None else info["depth_m"]
        span = max(0.0, goal - state["depth_mm"] / 1000.0)
        return [
            Segment(
                point=lambda u: top + u * span * _DOWN,
                normal=lambda u: up,
                length=span,
                met=_state_met(ctx, "depth_mm", MET_DEPTH_MM),
            )
        ]
    top = ctx.observe_site(data, "part") + np.array([0.0, 0.0, 0.5 * info.get("height_m", 0.0)])
    return [
        Segment(
            point=lambda u: top + u * 0.005 * _DOWN,
            normal=lambda u: up,
            length=0.005,
            press=press,
            reach=False,
        )
    ]


def _contact_point(tool: Tool, normal: np.ndarray, heading: float) -> np.ndarray:
    """The tool point (wrist frame, roll 0) that meets a face of outward ``normal``: the tip,
    moved out to the footprint's side that faces the face (by how horizontal the normal is)."""
    fp = tool.footprint(1.0 if tool.mechanism == "fingers" else 0.0)
    n_h = np.array([normal[0], normal[1], 0.0])
    tip = tool.tip.copy()
    size = float(np.linalg.norm(n_h))
    if size < 1e-6:
        return tip
    d_wrist = rot_z(heading).T @ (-n_h / size)  # the tool side that faces the face points into it
    side = tip + fp.half_extent(d_wrist) * d_wrist * min(1.0, size / 0.7)
    side[2] = tip[2] + 0.003 * min(1.0, size / 0.7)
    return side


def _overhang(tool: Tool, normal: np.ndarray, heading: float) -> float:
    """How much further than its contact point (:func:`_contact_point`) the whole tool, the wrist
    flange included, reaches toward a face of outward ``normal`` (m along the normal; 0 for a face
    pressed from above): what meets a tall face first."""
    n_h = np.array([normal[0], normal[1], 0.0])
    size = float(np.linalg.norm(n_h))
    if size < 1e-6:
        return 0.0
    d_wrist = rot_z(heading).T @ (-n_h / size)
    fp = tool.footprint(1.0 if tool.mechanism == "fingers" else 0.0)
    side = float(_contact_point(tool, normal, heading)[:2] @ d_wrist[:2])
    return max(0.0, max(fp.front(d_wrist), FLANGE_RADIUS) - side) * size


@dataclass(frozen=True)
class PushController:
    primitive: PrimitiveKind = PrimitiveKind.PUSH
    name: str = "studio_push"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        step = ctx.step
        tool = tool_of(ctx, ctx.obj.body_ids[-1])
        sch = schedule(ctx)
        sch.tool = tool
        segments = _segments(ctx, data)
        start = np.asarray(ctx.wrist_start, dtype=np.float64)
        yaw, roll = float(start[3]), float(start[4])
        speed = float(step.params["speed_mm_s"]) / 1000.0
        limit = force_allowed(ctx, PUSH_N)
        over = press_reach(ctx, limit)  # how far past the goal a push may go on (module docstring)
        safe = safe_palm_z(ctx, model, data, tool)
        touch = min(speed, TOUCH_SPEED)
        # the standoff and the first TOUCH_MM of the face, and the last END_MM of the travel, at the
        # touch speed (the same for every tool)
        touch_t = 1.3 * (STANDOFF + TOUCH_MM + END_MM) * (1.0 / touch - 1.0 / speed)
        t = 0.0
        here = start.copy()
        for seg in segments:  # each push: a transit to above its standoff point, a guarded move in, the push
            n0 = seg.normal(0.0)
            # the standoff is the whole tool's: whatever of it reaches further toward the face than
            # its contact point (a wrist flange, a long finger) stands off STANDOFF too
            ov = _overhang(tool, n0, yaw + roll)
            pre = wrist_target(
                seg.point(0.0) + (STANDOFF + ov) * n0, _contact_point(tool, n0, yaw + roll), yaw, roll
            )
            path = Path.start(here)
            above = pre + np.concatenate([APPROACH * n0, [0.0, 0.0]])
            above[2] = max(above[2], pre[2] + APPROACH)
            transit(path, above, max(above[2], safe))
            guard = Guarded(above.copy(), pre, t + path.end, APPROACH_S, force=min(GUARD_FORCE_N, limit))
            seg.state.update(path=path, guard=guard, t0=t, t_push=t + path.end + APPROACH_S, ov=ov)
            t = t + path.end + APPROACH_S + touch_t
            t += 1.3 * (seg.length + STANDOFF + ov) / speed + 0.5 * BUDGET_S["push_slack"]
            if seg.pressing_on:  # pressing on to the force goal
                t += 1.3 * over / speed + BUDGET_S["force_press"]
            seg.state["t_end"] = t
            here = pre.copy()
            here[2] = max(here[2], safe)
        duration = min(30.0, t + 0.3 + 0.5 + 0.6 + 0.3)
        sch.engaged = segments[0].state["t_push"]
        run: dict[str, Any] = {
            "segments": segments,
            "index": 0,
            "speed": speed,
            "touch": touch,
            "limit": limit,
            "over": over,
            "yaw": yaw,
            "roll": roll,
            "done_at": None,
            "last": start.copy(),
            "n_end": np.array([0.0, 0.0, 1.0]),
            "duration": duration,
        }
        return StepPlan(
            duration_s=duration,
            values={"run": run, "schedule": sch},
            note=f"push: {len(segments)} segment(s), {1000 * sum(s.length for s in segments):.0f} mm",
        )

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        sch = plan.values["schedule"]
        run = plan.values["run"]
        tool: Tool = sch.tool
        if run["done_at"] is None:
            wrist = self._segment(t, ctx, run, model, data)
        else:
            back = min_jerk(min(1.0, max(0.0, (t - run["done_at"] - 0.3) / 0.5)))
            up = min_jerk(min(1.0, max(0.0, (t - run["done_at"] - 0.8) / 0.6)))
            shift = BACK_OFF * back * run["n_end"] + np.array([0.0, 0.0, RETRACT * up])
            wrist = run["last"] + np.concatenate([shift, [0.0, 0.0]])
        closure = 1.0 if tool.mechanism == "fingers" else 0.0
        return PrimitiveCommand(wrist, tool.controls(t, closure, model, data, Phase.SETTLE))

    def _segment(
        self, t: float, ctx: StepContext, run: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
    ) -> np.ndarray:
        segs: list[Segment] = run["segments"]
        seg = segs[run["index"]]
        st = seg.state
        if t < st["t_push"]:  # the transit to above this segment's standoff point, then the guarded move in
            start = st.get("from")
            if start is None:
                start = run["last"].copy()
                st["from"] = start
                path: Path = st["path"]
                path.poses[0] = start
            if t - st["t0"] < st["path"].end:
                return st["path"].at(t - st["t0"])
            # it stops on anything in the way (any object, over the guard force or the step's limit)
            pressing = contacts(model, data, tool_geoms(ctx.scene), ctx.scene.ids.object_geoms).normal
            return st["guard"].at(t, pressing)
        if "shift" not in st:
            # a guarded move stopped short: the push starts where it stopped, its standoff longer
            # by how far short along the push and its line moved by the rest
            guard: Guarded = st["guard"]
            delta = guard.at(t, math.inf)[:3] - guard.b[:3] if guard.blocked else np.zeros(3)
            n0 = seg.normal(0.0)
            st["extra"] = max(0.0, float(delta @ n0))
            st["shift"] = delta - st["extra"] * n0
        tool: Tool = schedule(ctx).tool
        dt = ctx.task.controller.control_period
        force = float(np.linalg.norm(contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids).net))
        stand = STANDOFF + st["ov"] + st["extra"]
        total = stand + seg.length  # the contact point travels the standoff, then the push
        # a push to a criterion goes on (up to ``over``) until the observed state meets it, and a
        # press until its force: the servo lags under the object's resistance
        cap = total + (run["over"] if (seg.press is not None or seg.met is not None) else 0.0)
        ok = seg.met is None or seg.met(data)
        travelled = st.get("s", 0.0)
        arrived = travelled >= total - 1e-9 and (ok or travelled >= cap - 1e-9)
        travel = seg.reach and not arrived
        stop_at = run["limit"] if (seg.press is None or travel) else min(run["limit"], seg.press)
        # it meets the face, and its object meets its stop, at the touch speed, so an impact stays
        # under the limit
        near = travelled < stand - st["ov"] + TOUCH_MM or travelled > total - END_MM
        speed = run["touch"] if near else run["speed"]
        if force < stop_at and (travel or ok):  # pressing on keeps the state inside its criteria
            st["s"] = min(cap, travelled + speed * dt)
        travelled = st.get("s", 0.0)
        arrived = travelled >= total - 1e-9 and (ok or travelled >= cap - 1e-9)
        # the observed state meets its criteria and the push is held at its limit (a panel or a
        # drawer shut against its stop): nothing is left to push for
        arrived = arrived or (seg.met is not None and ok and force >= run["limit"])
        off = max(0.0, stand - travelled) - max(0.0, travelled - total)
        u = 1.0 if seg.length <= 1e-9 else min(1.0, max(0.0, travelled - stand) / seg.length)
        n = seg.normal(u)
        heading = run["yaw"] + run["roll"]
        wrist = wrist_target(
            seg.point(u) + off * n + st["shift"], _contact_point(tool, n, heading), run["yaw"], run["roll"]
        )
        if seg.press is not None and force >= seg.press and st.get("pressed_t") is None:
            st["pressed_t"] = t
        pressed = st.get("pressed_t") is not None and t - st["pressed_t"] >= 0.5
        # every goal of the push: its travel (when that is a goal) and its press force
        done = (arrived or not seg.reach) and (seg.press is None or pressed)
        run["last"] = wrist.copy()
        if done or t >= st["t_end"]:
            if run["index"] + 1 < len(segs):  # start the next push's transit now
                run["index"] += 1
                nxt = segs[run["index"]].state
                budget = nxt["t_end"] - nxt["t_push"]
                nxt["t0"] = t
                nxt["t_push"] = t + nxt["path"].end + APPROACH_S
                nxt["t_end"] = nxt["t_push"] + budget
                nxt["from"] = None
                lifted = wrist.copy()
                lifted[2] += APPROACH
                run["last"] = lifted
            else:
                run["done_at"] = t
                run["n_end"] = np.asarray(n, dtype=np.float64)
                schedule(ctx).released = t
                # the push stops: the wrist target comes back to where the wrist is, so the hold
                # and the retract do not go on pressing with the servo's lag behind the command
                for i in range(3):
                    run["last"][i] = float(data.qpos[ctx.scene.ids.palm_qpos[i]])
        return wrist


@dataclass(frozen=True)
class PushPredicate:
    primitive: PrimitiveKind = PrimitiveKind.PUSH
    name: str = "studio_push"

    def observe(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None:
        acc.update(ctx.object_state(model, data))
        con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
        acc["force_n"] = max(acc.get("force_n", 0.0), float(np.linalg.norm(con.net)))
        acc["penetration_mm"] = max(acc.get("penetration_mm", 0.0), 1000.0 * con.depth)

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict:
        return criteria_verdict(ctx.step, acc)


register_primitive(PushController(), PushPredicate())
