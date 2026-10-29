"""The ``twist`` primitive: grip or engage the object and turn the wrist roll under a resistive torque.

Objects: ``quarter_turn_latch`` (``head``), ``threaded_joint`` (``member``), ``handwheel_valve``
(``wheel``), and ``panel_rail`` (``member``: its clamp, a threaded member read through
:func:`~rtt.studio.templates.thread_info` and turned like a threaded joint's). Working senses (the
templates): a latch and a valve open counter-clockwise seen from above, a thread tightens
clockwise; ``direction`` is the step's, so a step that turns against the object's sense meets its
stop and stalls.

**Out of reach** (:func:`~._common.reach_problem`, Studio v2): a target whose engagement pose puts
the palm outside its travel (+-350 mm, :data:`rtt.sim.scene.PALM_XY_LIMIT`, less 5 mm) is not
attempted: the step holds still briefly and fails ``out_of_reach``, reporting where the palm would
have to go (``reach_x_mm``, ``reach_y_mm``) and its limit (``palm_limit_mm``), not a slip or a jam.

**Controller** (:class:`TwistController`):

1. **Engagement plan**, from the observed axis site (``head``, ``member``, ``wheel``) and the
   observed rotation mark (``mark``, ``flat``, ``spoke0``), so the object's present angle is
   known:

   * fingers on a latch: the closing axis across the wing's fin, or across the slot (gripping
     the round head: friction only, a slot needs a blade), fingertips 2 mm above the panel;
   * fingers on a threaded member: the closing axis along a flat's normal (a nut's flats), tips
     3 mm above the member's bottom;
   * fingers on a spoked handwheel: **spoke engagement**: the open fingers go down into the
     gaps between the spokes, at the heading (1 deg search over the calibrated open footprint)
     that keeps every tip clear of the hub, the rim and the spokes and puts a finger closest
     behind a spoke in the turn direction; a solid disc is gripped across its rim when the
     fingers open wide enough, else pressed on its hub;
   * a cup on the top face (sealing only where the face is flat and whole), a soft arm curled
     around the head.

   The fingers are pre-shaped to the feature (6 mm clear each side) and lowered with a guarded
   move (it stops on contact above 5 N) to where their lowest point while closing
   (``Tool.closing_tip``: a linkage finger's tip swings down as it closes) is at the tip height.
2. **Engage**: the head's own v1 primitive closes on the shared ramp (1.0 s plus 0.3 s);
   spoke engagement keeps the fingers at their pre-shape.
3. **Turn** (closed loop, force feedback through the wrist only): the roll target advances in
   the step's direction at ``speed_deg_s`` (capped at the wrist's roll speed limit, ramped over
   0.2 s). It stops advancing while the roll torque is at the limit (90 % of the step's
   ``max_torque_nm``, else 95 % of the wrist's) or, under a step ``max_force_n``, while the
   tool's contact force on the object is over :data:`FORCE_HOLD` of it; while it presses down
   harder than that the wrist eases up (:data:`Z_RELIEF_SPEED`, at most :data:`Z_RELIEF_MAX`).
   **Goals**: every at_least criterion of the step is a goal, and the turn runs until all of
   them are met: the rotation goal (the largest of the ``angle_deg``, ``turns`` and ``depth_mm``
   at_least criteria, plus 3 deg) and a ``torque_nm`` at_least (the roll torque over it plus
   :data:`~._common.FORCE_MARGIN` for :data:`~._common.FORCE_HOLD_S`). Without either, the goal
   is the object's whole travel (to seated for a thread); with only a torque goal the turn may
   run as far as the object goes. It never stops at the first goal reached. Limits end it
   earlier: an ``angle_deg`` or ``turns`` at_most in the working sense (3 deg before it); a
   ``torque_nm`` at_most holds the roll torque under 95 % of it; a stall (0.3 s at the torque
   limit without progress); slipping (the tool 90 deg past the object: past the rotation goal,
   or with a torque goal, ahead of the object); the budget. A thread is fed: the wrist first
   takes the member back down to where it was before the fingers closed (closing fingers can
   draw it up its lead-in), then descends with it, one pitch per turn, plus 1.5 mm of push.
   Budget: the nominal turn time x 1.3 plus :data:`~._common.BUDGET_S` ``twist_slack``, the same
   rule for every candidate.
4. **Release**: the roll target backs off to the measured roll (at the speed cap); fingers in a
   wheel's spoke gaps then turn back by their free play, to the middle of the gaps, so none lifts
   a spoke on the way out; the wrist holds where it is and lifts :data:`RELEASE_LIFT` while the
   head opens over 0.4 s, then rises to 50 mm (``hold_after`` keeps the grip).

**Predicate** (:class:`TwistPredicate`): the template state at the end (``angle_deg``, and
``turns`` and ``depth_mm`` for a thread), ``torque_nm`` (the peak roll torque),
``slip_deg`` (the largest difference between the tool's and the object's rotation, counted
from the moment the object first moves 0.5 deg until the turn stops, so the wind-up before the
object breaks away and the back-off after are not counted; when the object never moves, the
tool's whole rotation), ``force_n`` (peak net tool force), ``penetration_mm``. A step that
fails its criteria after slipping more than 15 deg is ``slip``.
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
    FORCE_MARGIN,
    PRESHAPE_CLOSURES,
    Guarded,
    Path,
    Tool,
    angle_about,
    contacts,
    criterion_bounds,
    heading_for_axis,
    hold_out_of_reach,
    idle_command,
    observed_state,
    offset,
    out_of_reach_verdict,
    reach_problem,
    rot_z,
    safe_palm_z,
    schedule,
    start_pose,
    tool_geoms,
    tool_of,
    torque_limit,
    transit,
    unwrap,
    wrist_target,
)
from rtt.studio.templates import member_body, thread_info
from rtt.studio.types import THREAD_KINDS, PrimitiveKind, TemplateKind

__all__ = [
    "TwistController",
    "TwistPredicate",
    "axis_site",
    "spoke_heading",
]

AXIS_SITE = {
    TemplateKind.QUARTER_TURN_LATCH: ("head", "mark", 1.0),
    TemplateKind.THREADED_JOINT: ("member", "flat", -1.0),
    TemplateKind.HANDWHEEL_VALVE: ("wheel", "spoke0", 1.0),
}


def axis_site(kind: TemplateKind) -> tuple[str, str, float]:
    """The axis site, the rotation mark and the working sense of a twistable kind (a panel's clamp
    is a threaded member: its sites and sense are a threaded joint's)."""
    return AXIS_SITE[TemplateKind.THREADED_JOINT if kind in THREAD_KINDS else kind]


GOAL_MARGIN = math.radians(3.0)
STALL_S = 0.3
RUNAWAY = math.radians(90.0)
FEED_PUSH = 0.0015  # m below the feed: about 6 N on the z servo
RELEASE_S = 0.4
RETRACT = 0.050
SLIP_FAIL_DEG = 15.0
SPOKE_CLEAR = 0.002  # m
FORCE_HOLD = 0.9  # of a step's max_force_n: the contact force the turn holds at
Z_RELIEF_SPEED = 0.010  # m/s: the wrist eases up while it presses down over the limit
Z_RELIEF_MAX = 0.004  # m [assumed]
RELEASE_LIFT = 0.003  # m: the wrist lifts this much while the fingers open


def _observed_angle(ctx: StepContext, data: mujoco.MjData) -> float:
    axis, mark, _ = axis_site(ctx.obj.kind)
    return angle_about(ctx.observe_site(data, axis), ctx.observe_site(data, mark))


def spoke_heading(
    tool: Tool, centre_angle: float, info: Mapping[str, float], sign: float
) -> tuple[float, float, float] | None:
    """The pre-shape closure and heading (rad) that put the fingers into the spoke gaps of a
    handwheel whose spoke 0 is at ``centre_angle``, and the free turn before a finger meets a
    spoke; the most open closure that works (the longest lever); ``None`` when no calibrated
    closure and heading clear the hub, the rim and every spoke (module docstring)."""
    for closure in PRESHAPE_CLOSURES:
        found = _spoke_heading(tool, closure, centre_angle, info, sign)
        if found is not None:
            return (closure, *found)
    return None


def _spoke_heading(
    tool: Tool, closure: float, centre_angle: float, info: Mapping[str, float], sign: float
) -> tuple[float, float] | None:
    fp = tool.footprint(closure)
    if not len(fp.boxes):
        return None
    pts = []
    for x0, x1, y0, y1 in fp.boxes:
        for x in (x0, 0.5 * (x0 + x1), x1):
            for y in (y0, 0.5 * (y0 + y1), y1):
                pts.append((x, y))
    p = np.array(pts) - tool.tcp[:2]  # about the TCP, which sits on the roll axis
    r_in = info["hub_m"] + SPOKE_CLEAR
    r_out = info["rim_centre_m"] - 0.5 * info["section_m"] - SPOKE_CLEAR
    radius = np.hypot(p[:, 0], p[:, 1])
    if np.any(radius < r_in) or np.any(radius > r_out):
        return None
    n = int(info["spokes"])
    half = np.arcsin(np.clip((info["spoke_m"] + SPOKE_CLEAR) / radius, 0.0, 1.0))
    base = np.arctan2(p[:, 1], p[:, 0])
    best: tuple[float, float] | None = None
    for k in range(360):
        psi = math.radians(k)
        ang = base + psi
        free = math.inf
        ok = True
        for j in range(n):
            spoke = centre_angle + 2.0 * math.pi * j / n
            gap = np.mod(sign * (spoke - ang), 2.0 * math.pi)  # turn until this spoke's centre line
            rev = np.mod(-sign * (spoke - ang), 2.0 * math.pi)
            if np.any(gap < half) or np.any(rev < half):
                ok = False
                break
            free = min(free, float(np.min(gap - half)))
        if ok and free >= math.radians(2.0) and (best is None or free < best[1]):
            best = (psi, free)
    return best


@dataclass(frozen=True)
class _Goals:
    """A turn's goals: ``rotation`` (rad, or ``None``) and ``torque`` (N m, or ``None``) are
    at-least goals that must all be met; ``cap`` (rad) and ``torque_max`` (N m) are the step's
    upper bounds; ``horizon`` (rad) is how far the turn may run (its budget and runaway)."""

    rotation: float | None
    horizon: float
    cap: float | None
    torque: float | None
    torque_max: float | None


@dataclass(frozen=True)
class TwistController:
    primitive: PrimitiveKind = PrimitiveKind.TWIST
    name: str = "studio_twist"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        step, obj = ctx.step, ctx.obj
        info = thread_info(obj)
        tool = tool_of(ctx, member_body(obj))
        sch = schedule(ctx)
        sch.tool = tool
        axis_name, _, w = axis_site(obj.kind)
        axis_pt = ctx.observe_site(data, axis_name)
        mark_angle = _observed_angle(ctx, data)
        sign = 1.0 if step.params["direction"] == "ccw" else -1.0
        start = start_pose(ctx)
        roll = float(start[4])
        current = float(start[3] + roll)
        mode, heading, pre, tip_z, free = self._engagement(
            ctx, tool, axis_pt, mark_angle, sign, current, info
        )
        yaw = heading - roll
        point = np.array([axis_pt[0], axis_pt[1], tip_z])
        # fingers: the planned tip height holds for the lowest point they reach while closing
        # (in the spoke gaps they stay at the pre-shape)
        if mode == "grip":
            low = tool.closing_tip(pre)
        elif mode == "spokes":
            low = np.array([tool.tip[0], tool.tip[1], min(float(tool.tip[2]), tool.footprint(pre).tip_z)])
        else:
            low = tool.tip
        engage = wrist_target(point, low, yaw, roll)
        far = reach_problem(engage)
        if far is not None:
            return hold_out_of_reach(ctx, start, far, "the engagement")
        path = Path.start(start, ctx.scene.wrist.pitch_speed_dps)
        above = engage + offset(z=0.06)
        transit(path, above, max(above[2], safe_palm_z(ctx, model, data, tool)))
        guard = Guarded(path.last, engage, path.end, 1.0 + BUDGET_S["engage_slack"])
        t_close = guard.end
        t_turn = t_close + 1.3
        goals = self._goal(ctx, data, sign, w, info)
        omega = math.radians(
            min(float(step.params["speed_deg_s"]), 0.98 * ctx.scene.wrist.roll_speed_max_deg_s)
        )
        turn_budget = 1.3 * (goals.horizon + RUNAWAY * 0.25) / omega + 0.2 + BUDGET_S["twist_slack"]
        t_end_turn = t_turn + turn_budget
        tail = 0.0 if step.hold_after else RELEASE_S + 0.6 + 0.3
        duration = min(30.0, t_end_turn + 0.5 + tail)
        sch.engaged = t_turn
        sch.values.update(t_close=t_close, pre=pre, sign=sign, w=w)
        run: dict[str, Any] = {
            "path": path,
            "guard": guard,
            "mode": mode,
            "t_turn": t_turn,
            "t_end_turn": min(t_end_turn, duration - 0.5 - tail),
            "omega": omega,
            "goals": goals,
            "torque_held": 0.0,
            # an upper bound on the torque criterion limits the turn too (it holds below it)
            "torque_limit": min(torque_limit(ctx), 0.95 * goals.torque_max)
            if goals.torque_max is not None
            else torque_limit(ctx),
            "roll_cmd": roll,
            "obs0": None,
            "obs_prev": mark_angle,
            "progress": 0.0,
            "stall": 0.0,
            "done_at": None,
            "base": None,
            "feed": obj.kind in THREAD_KINDS,
            "depth0": observed_state(ctx, data)["depth_mm"] / 1000.0 if obj.kind in THREAD_KINDS else 0.0,
            "pitch": info.get("pitch_m", 0.0),
            "hold": step.hold_after,
            # spoke engagement: after the turn the fingers turn back by their free play, to the
            # middle of the gaps, before they rise (so none lifts a spoke on the way out)
            "unwind": free if mode == "spokes" else 0.0,
            # a step force limit holds against the measured contact force (module docstring)
            "force_cap": FORCE_HOLD * step.constraints.max_force_n
            if step.constraints.max_force_n
            else math.inf,
            "z_relief": 0.0,
            "z_hold": None,
        }
        return StepPlan(
            duration_s=duration,
            values={"run": run, "schedule": sch},
            note=(
                f"twist {step.params['direction']}: "
                + (
                    f"{math.degrees(goals.rotation):.0f} deg"
                    if goals.rotation is not None
                    else "no rotation goal"
                )
                + (f" and {goals.torque:g} N m" if goals.torque is not None else "")
                + f", {mode} engagement"
            ),
        )

    # ------------------------------------------------------------------ planning helpers

    def _engagement(
        self,
        ctx: StepContext,
        tool: Tool,
        axis_pt: np.ndarray,
        mark_angle: float,
        sign: float,
        current: float,
        info: Mapping[str, float],
    ) -> tuple[str, float, float, float, float]:
        """(mode, heading, pre-shape closure, tip height, free turn before a finger meets a
        spoke) of the engagement."""
        kind = ctx.obj.kind
        u = np.array([math.cos(mark_angle), math.sin(mark_angle), 0.0])
        across = rot_z(0.5 * math.pi) @ u
        if tool.mechanism == "suction":  # the lip pressed 4 mm into the top face (the bellows takes it)
            return "seal", current, 0.0, float(axis_pt[2]) - 0.004, 0.0
        if kind is TemplateKind.QUARTER_TURN_LATCH:
            width = info["fin_m"] if info["head"] == 1.0 else info["diameter_m"]
            tip = info["base_z"] + 0.002
            head = heading_for_axis(across, 0.0, current)
            return "grip", head, tool.preshape(width), tip, 0.0
        if kind in THREAD_KINDS:
            width = 2.0 * info["flat_m"]
            tip = float(axis_pt[2]) - info["length_m"] + 0.003
            head = heading_for_axis(u, 0.0, current)
            return "grip", head, tool.preshape(width), tip, 0.0
        # handwheel
        tip = info["wheel_z"] - info["spoke_m"] - 0.004
        if info["spokes"] > 0 and tool.mechanism == "fingers":
            found = spoke_heading(tool, mark_angle, info, sign)
            if found is not None:
                closure, psi, free = found
                psi = psi + 2.0 * math.pi * round((current - psi) / (2.0 * math.pi))
                return "spokes", psi, closure, tip, free
        if info["spokes"] == 0 and tool.mechanism == "fingers":
            ap = tool.footprint(0.0).aperture()
            if ap is not None and ap >= 2.0 * info["rim_m"] + 0.004:  # the fingers open past the disc
                pre = tool.preshape(2.0 * info["rim_m"], 0.002)
                return "grip", current, pre, info["wheel_z"] - 0.5 * info["section_m"] + 0.001, 0.0
        return "press", current, 0.0, float(axis_pt[2]) - 0.003, 0.0  # the guarded move stops on contact

    def _goal(
        self, ctx: StepContext, data: mujoco.MjData, sign: float, w: float, info: Mapping[str, float]
    ) -> _Goals:
        """The turn's goals and limits from the step's criteria (module docstring)."""
        state = observed_state(ctx, data)
        theta = math.radians(state["angle_deg"])
        along = sign * w > 0  # the commanded direction is the object's working sense
        kind = ctx.obj.kind
        goal: float | None = None
        cap: float | None = None
        for q, scale in (("angle_deg", math.radians(1.0)), ("turns", 2.0 * math.pi)):
            lo, hi = criterion_bounds(ctx, q)
            now = theta if q == "angle_deg" else state.get("turns", 0.0) * 2.0 * math.pi
            if along and lo is not None:
                goal = max(goal or 0.0, lo * scale - now)
            if along and hi is not None:  # an upper bound in the working sense limits the turn
                cap = min(cap if cap is not None else math.inf, hi * scale - now)
            if not along and hi is not None:
                goal = max(goal or 0.0, now - hi * scale)
        lo, _ = criterion_bounds(ctx, "depth_mm")
        if kind in THREAD_KINDS and lo is not None and along:
            need = (lo / 1000.0 - max(info["lead_m"], state["depth_mm"] / 1000.0)) / info["pitch_m"]
            goal = max(goal or 0.0, 2.0 * math.pi * need)
        if kind is TemplateKind.QUARTER_TURN_LATCH:
            travel = info["turn_rad"] if along else theta
        elif kind is TemplateKind.HANDWHEEL_VALVE:
            travel = info["travel_rad"] if along else theta
        else:
            left = info["lead_m"] + info["seat_m"] - state["depth_mm"] / 1000.0
            travel = 2.0 * math.pi * (left / info["pitch_m"] + 0.25)
        t_lo, t_hi = criterion_bounds(ctx, "torque_nm")
        if goal is not None:
            goal = max(math.radians(5.0), goal + GOAL_MARGIN)
        elif t_lo is None:  # nothing asked: the object's whole travel (to seated for a thread)
            goal = max(math.radians(5.0), travel + GOAL_MARGIN)
        # how far the turn may run: its rotation goal; with a torque goal, as far as the object goes
        horizon = goal if t_lo is None else max(goal or 0.0, travel + GOAL_MARGIN)
        if cap is not None:
            cap = max(0.0, cap - GOAL_MARGIN)
        return _Goals(rotation=goal, horizon=float(horizon or 0.0), cap=cap, torque=t_lo, torque_max=t_hi)

    # ------------------------------------------------------------------ the control loop

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        sch = plan.values["schedule"]
        run = plan.values["run"]
        tool: Tool = sch.tool
        if "idle" in run:  # out of reach: hold still, the tool as it was
            return idle_command(t, ctx, run, model, data)
        v = sch.values
        guard: Guarded = run["guard"]
        pre = v["pre"]
        mode = run["mode"]
        if t < guard.t0:
            wrist = run["path"].at(t)
        elif t < run["t_turn"]:
            pressing = contacts(model, data, tool_geoms(ctx.scene), ctx.scene.ids.object_geoms).normal
            wrist = guard.at(min(t, guard.end), pressing) if t <= guard.end else guard.at(guard.end, 1e9)
        else:
            wrist = self._turn(t, ctx, run, model, data, guard, v)
        if t > guard.end:  # engaged: pressing down harder than the step's limit eases the wrist up
            if (
                -float(contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids).net[2])
                > run["force_cap"]
            ):
                dt = ctx.task.controller.control_period
                run["z_relief"] = min(Z_RELIEF_MAX, run["z_relief"] + Z_RELIEF_SPEED * dt)
            wrist = wrist + offset(z=run["z_relief"])
        closure = pre
        if mode != "spokes":
            closure = pre + (1.0 - pre) * min(max((t - v["t_close"]) / 1.0, 0.0), 1.0)
        release = run.get("release_at")
        if release is not None and not run["hold"] and t >= release:
            if run["z_hold"] is None:  # stop pressing: hold the wrist where it is
                run["z_hold"] = float(data.qpos[ctx.scene.ids.palm_qpos[2]])
            wrist = wrist.copy()
            wrist[2] = run["z_hold"]
            closure = closure * max(0.0, 1.0 - (t - release) / RELEASE_S) if mode != "spokes" else 0.0
            # lift clear of the panel while the fingers open (an opening finger can swing down),
            # then retract
            lift = RELEASE_LIFT * min_jerk(min(1.0, (t - release) / RELEASE_S))
            up = min_jerk(min(1.0, max(0.0, (t - release - RELEASE_S) / 0.6)))
            wrist = wrist + offset(z=lift + (RETRACT - RELEASE_LIFT) * up)
        phase = Phase.HOLD if t >= v["t_close"] else Phase.DESCEND
        return PrimitiveCommand(wrist, tool.controls(t, closure, model, data, phase))

    def _turn(
        self,
        t: float,
        ctx: StepContext,
        run: dict[str, Any],
        model: mujoco.MjModel,
        data: mujoco.MjData,
        guard: Guarded,
        v: Mapping[str, float],
    ) -> np.ndarray:
        if run["base"] is None:
            run["base"] = guard.at(guard.end, 1e9).copy()
            if run["feed"]:
                # fingers that closed on a threaded member may have drawn it up its lead-in: the
                # wrist takes it back down to where it was (on its thread) before it turns
                lifted = run["depth0"] - observed_state(ctx, data)["depth_mm"] / 1000.0
                run["base"][2] -= max(0.0, lifted)
            sch = schedule(ctx)
            sch.engaged = t
        base = run["base"]
        dt = ctx.task.controller.control_period
        sign = v["sign"]
        roll_meas = float(data.qpos[ctx.scene.ids.roll_qpos])
        torque = abs(float(data.actuator_force[ctx.scene.ids.wrist_actuators[4]]))
        obs = unwrap(run["obs_prev"], _observed_angle(ctx, data))
        run["obs_prev"] = obs
        if run["obs0"] is None:
            run["obs0"] = obs
        progress = sign * (obs - run["obs0"])  # object rotation in the commanded direction
        cap = math.radians(ctx.scene.wrist.roll_speed_max_deg_s) * dt * 0.98
        con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
        if run["done_at"] is None:
            ramp = min(1.0, (t - run["t_turn"]) / 0.2)
            step = run["omega"] * dt * ramp
            # the roll holds while the torque or the contact force is at its limit
            limited = torque >= run["torque_limit"] or float(np.linalg.norm(con.net)) > run["force_cap"]
            if limited and progress <= run["progress"] + math.radians(0.05):
                run["stall"] += dt
            else:
                run["stall"] = 0.0
            run["progress"] = max(run["progress"], progress)
            turned = sign * (run["roll_cmd"] - float(base[4]))
            g: _Goals = run["goals"]
            if g.torque is not None and torque >= g.torque * (1.0 + FORCE_MARGIN):
                run["torque_held"] += dt  # over the torque goal, held long enough to be measured
            # every at-least goal must be met: the rotation and the torque
            met = (g.rotation is None or progress >= g.rotation) and (
                g.torque is None or run["torque_held"] >= FORCE_HOLD_S
            )
            at_cap = g.cap is not None and progress >= g.cap
            # a torque goal turns as far as the object goes, so slipping is told by the tool
            # running 90 deg ahead of the object
            slipping = g.torque is not None and turned - progress >= RUNAWAY
            if (
                met
                or at_cap
                or slipping
                or run["stall"] >= STALL_S
                or turned >= g.horizon + RUNAWAY
                or t >= run["t_end_turn"]
            ):
                run["done_at"] = t
                schedule(ctx).values["turn_end"] = t
            elif not limited:
                run["roll_cmd"] += sign * step
        else:  # back the target off toward the measured roll (and unwind the free play), then release
            gap = roll_meas - run["roll_cmd"]
            run["roll_cmd"] += max(-cap, min(cap, gap))
            if abs(gap) <= cap and run["unwind"] > 0.0:
                back = min(cap - abs(gap), run["unwind"])  # within the roll speed cap, with the step above
                run["roll_cmd"] -= sign * back
                run["unwind"] -= back
                gap = 0.0
            if run.get("release_at") is None and (
                (abs(gap) <= cap and run["unwind"] <= 0.0) or t >= run["done_at"] + 0.4
            ):
                run["release_at"] = t + 0.05
                sch = schedule(ctx)
                sch.values["released_at"] = run["release_at"]
                sch.released = run["release_at"]
        wrist = base.copy()
        wrist[4] = run["roll_cmd"]
        if run["feed"]:
            turns = max(0.0, run["progress"]) / (2.0 * math.pi)
            wrist[2] = base[2] - run["pitch"] * turns - FEED_PUSH
        return wrist


@dataclass(frozen=True)
class TwistPredicate:
    primitive: PrimitiveKind = PrimitiveKind.TWIST
    name: str = "studio_twist"

    def observe(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None:
        sch = schedule(ctx)
        v = sch.values
        acc.update(ctx.object_state(model, data))
        con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
        acc["force_n"] = max(acc.get("force_n", 0.0), float(np.linalg.norm(con.net)))
        acc["penetration_mm"] = max(acc.get("penetration_mm", 0.0), 1000.0 * con.depth)
        torque = abs(float(data.actuator_force[ctx.scene.ids.wrist_actuators[4]]))
        acc["torque_nm"] = max(acc.get("torque_nm", 0.0), torque)
        acc.setdefault("slip_deg", 0.0)
        engaged = sch.engaged
        if math.isnan(engaged) or t < engaged:
            return
        q = data.qpos
        heading = float(q[ctx.scene.ids.palm_qpos[3]] + q[ctx.scene.ids.roll_qpos])
        obj_z = math.radians(acc["angle_deg"]) / v["w"]  # the object's rotation about +z
        if "h0" not in v:
            v["h0"], v["o0"], v["moving"] = heading, obj_z, 0.0
        if not v["moving"] and abs(obj_z - v["o0"]) >= math.radians(0.5):
            v["moving"] = 1.0
            v["h1"], v["o1"] = heading, obj_z
        turn_end = sch.values.get("turn_end", math.nan)
        if not math.isnan(turn_end) and t > turn_end:
            return  # the back-off and the release are not slip
        if v["moving"]:
            slip = abs((heading - v["h1"]) - (obj_z - v["o1"]))
            acc["slip_deg"] = max(acc["slip_deg"], math.degrees(slip))
        else:
            v["unmoved"] = max(v.get("unmoved", 0.0), math.degrees(abs(heading - v["h0"])))

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict:
        far = out_of_reach_verdict(ctx, acc)
        if far is not None:
            return far
        v = schedule(ctx).values
        acc = dict(acc)
        if not v.get("moving"):
            acc["slip_deg"] = max(acc.get("slip_deg", 0.0), v.get("unmoved", 0.0))
        out = criteria_verdict(ctx.step, acc)
        if (
            not out.success
            and out.failure_mode == "criterion_not_met"
            and acc.get("slip_deg", 0.0) > SLIP_FAIL_DEG
        ):
            return StepVerdict(False, "slip", dict(acc))
        return out


register_primitive(TwistController(), TwistPredicate())
