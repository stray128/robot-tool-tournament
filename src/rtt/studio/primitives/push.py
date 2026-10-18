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

   * a panel: on its outer face 15 mm in from the free edge (the most leverage, and clear of
     a bar handle), along the arc about the hinge to the ``angle_deg`` goal (3 deg past it). A
     lid open past vertical that has to close is pushed only to 80 deg; its weight closes it,
     and the tool then presses the closed lid home from above (a second, downward push);
   * a drawer: on the front of its handle (the part that stands out), along the slide to the
     ``travel_mm`` goal (3 mm past it);
   * a part: straight down on its top until the press force; a plug: straight down on its top to
     the ``depth_mm`` goal (0.5 mm past it; without one, seated).

3. **Force-limited progress** (force feedback through the wrist only): the push advances at the
   step's ``speed_mm_s`` while the tool's net force on the object is under the limit (80 % of
   ``max_force_n``, else :data:`PUSH_N`), and holds while it is over, within the equal budget
   :data:`~._common.BUDGET_S` ``push_slack``. A press stops 0.5 s after reaching
   :data:`PRESS_N` (or 60 % of the limit).
4. **Retract**: hold 0.3 s, back off 30 mm along the face normal, rise 60 mm.

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
    Path,
    Tool,
    contacts,
    criterion_bounds,
    force_limit,
    observed_state,
    panel_normal,
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
GRAVITY_HANDOFF = math.radians(80.0)
STANDOFF = 0.010  # m off the face before a push starts
APPROACH = 0.030
BACK_OFF = 0.030
RETRACT = 0.060
_DOWN = np.array([0.0, 0.0, -1.0])


@dataclass
class Segment:
    """One push: the push point ``point(u)`` and the face normal ``normal(u)`` for ``u`` in [0, 1],
    ``length`` m of travel, and an optional press force to stop at."""

    point: Callable[[float], np.ndarray]
    normal: Callable[[float], np.ndarray]
    length: float
    press: float | None = None
    state: dict[str, Any] = field(default_factory=dict)


def _segments(ctx: StepContext, data: mujoco.MjData) -> list[Segment]:
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
        radius = float(np.linalg.norm(np.cross(axis, edge0)))
        theta = math.radians(state["angle_deg"])
        lo, hi = criterion_bounds(ctx, "angle_deg")
        if hi is not None:
            goal = max(-ANGLE_PAST, math.radians(hi) - ANGLE_PAST)
        elif lo is not None:
            goal = min(info["max_angle_rad"], math.radians(lo) + ANGLE_PAST)
        else:
            goal = 0.0

        def arc(a: float, b: float) -> Segment:
            return Segment(
                point=lambda u: hinge + rotate(edge0, axis, a + u * (b - a)),
                normal=lambda u: panel_normal(ctx, a + u * (b - a)),
                length=abs(b - a) * radius,
            )

        lid = info["orientation"] == 0.0
        if lid and theta > math.radians(92.0) and goal < GRAVITY_HANDOFF:
            closed = hinge + rotate(edge0, axis, max(0.0, goal))
            return [
                arc(theta, GRAVITY_HANDOFF),
                Segment(
                    point=lambda u: closed + u * 0.010 * _DOWN,
                    normal=lambda u: out0,
                    length=0.010,
                    press=press,
                ),
            ]
        return [arc(theta, goal)]
    if obj.kind is TemplateKind.DRAWER:
        pull = np.array([info["pull_x"], info["pull_y"], 0.0])
        p0 = ctx.observe_site(data, "handle") + info["push_out_m"] * pull
        travel = state["travel_mm"] / 1000.0
        lo, hi = criterion_bounds(ctx, "travel_mm")
        goal = lo / 1000.0 + TRAVEL_PAST if (lo is not None and hi is None) else -TRAVEL_PAST
        if hi is not None:
            goal = max(-TRAVEL_PAST, hi / 1000.0 - TRAVEL_PAST)
        span = goal - travel
        return [Segment(point=lambda u: p0 + u * span * pull, normal=lambda u: pull, length=abs(span))]
    up = np.array([0.0, 0.0, 1.0])
    if obj.kind is TemplateKind.PORT_PLUG:
        top = ctx.observe_site(data, "plug")
        lo, _ = criterion_bounds(ctx, "depth_mm")
        goal = (lo / 1000.0 + 0.0005) if lo is not None else info["depth_m"]
        span = max(0.0, goal - state["depth_mm"] / 1000.0)
        return [Segment(point=lambda u: top + u * span * _DOWN, normal=lambda u: up, length=span)]
    top = ctx.observe_site(data, "part") + np.array([0.0, 0.0, 0.5 * info.get("height_m", 0.0)])
    return [Segment(point=lambda u: top + u * 0.005 * _DOWN, normal=lambda u: up, length=0.005, press=press)]


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
        safe = safe_palm_z(ctx, model, data, tool)
        t = 0.0
        here = start.copy()
        for seg in segments:  # each push: a transit to its standoff point, then the force-limited push
            n0 = seg.normal(0.0)
            pre = wrist_target(
                seg.point(0.0) + STANDOFF * n0, _contact_point(tool, n0, yaw + roll), yaw, roll
            )
            path = Path.start(here)
            above = pre + np.concatenate([APPROACH * n0, [0.0, 0.0]])
            above[2] = max(above[2], pre[2] + APPROACH)
            transit(path, above, max(above[2], safe))
            path.to(pre, 0.8)
            seg.state.update(path=path, t0=t, t_push=t + path.end)
            t = t + path.end + 1.3 * (seg.length + STANDOFF) / speed + 0.5 * BUDGET_S["push_slack"]
            seg.state["t_end"] = t
            here = pre.copy()
            here[2] = max(here[2], safe)
        duration = min(30.0, t + 0.3 + 0.5 + 0.6 + 0.3)
        sch.engaged = segments[0].state["t_push"]
        run: dict[str, Any] = {
            "segments": segments,
            "index": 0,
            "speed": speed,
            "limit": force_limit(ctx, PUSH_N),
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
        if t < st["t_push"]:  # the transit to this segment's standoff point
            start = st.get("from")
            if start is None:
                start = run["last"].copy()
                st["from"] = start
                path: Path = st["path"]
                path.poses[0] = start
            return st["path"].at(t - st["t0"])
        tool: Tool = schedule(ctx).tool
        dt = ctx.task.controller.control_period
        force = float(np.linalg.norm(contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids).net))
        total = STANDOFF + seg.length  # the contact point travels the standoff, then the push
        stop_at = run["limit"] if seg.press is None else min(run["limit"], seg.press)
        if force < stop_at:
            st["s"] = min(total, st.get("s", 0.0) + run["speed"] * dt)
        travelled = st.get("s", 0.0)
        off = max(0.0, STANDOFF - travelled)
        u = 1.0 if seg.length <= 1e-9 else min(1.0, max(0.0, travelled - STANDOFF) / seg.length)
        n = seg.normal(u)
        heading = run["yaw"] + run["roll"]
        wrist = wrist_target(
            seg.point(u) + off * n, _contact_point(tool, n, heading), run["yaw"], run["roll"]
        )
        if seg.press is not None and force >= seg.press and st.get("pressed_t") is None:
            st["pressed_t"] = t
        pressed = st.get("pressed_t") is not None and t - st["pressed_t"] >= 0.5
        arrived = seg.press is None and travelled >= total - 1e-9
        run["last"] = wrist.copy()
        if arrived or pressed or t >= st["t_end"]:
            if run["index"] + 1 < len(segs):  # start the next push's transit now
                run["index"] += 1
                nxt = segs[run["index"]].state
                budget = nxt["t_end"] - nxt["t_push"]
                nxt["t0"] = t
                nxt["t_push"] = t + nxt["path"].end
                nxt["t_end"] = nxt["t_push"] + budget
                nxt["from"] = None
                lifted = wrist.copy()
                lifted[2] += APPROACH
                run["last"] = lifted
            else:
                run["done_at"] = t
                run["n_end"] = np.asarray(n, dtype=np.float64)
                schedule(ctx).released = t
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
