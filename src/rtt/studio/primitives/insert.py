"""The ``insert`` primitive: align the held member over its seat and push it in with compliance.

Objects: ``port_plug`` (``plug``), ``threaded_joint`` (``member``) and ``panel_rail`` (``member``: its
clamp, a threaded member read through :func:`~rtt.studio.templates.thread_info`). A step whose
planned moves or seat would put the palm outside its travel fails ``out_of_reach``, as every
primitive does (:func:`~._common.reach_problem`).

**Controller** (:class:`InsertController`):

1. **Hold it.** If the tool does not already hold the member (a previous ``grasp`` step with
   ``hold_after``), it picks it: a plug off the table with the head's own v1 primitive and the v1
   phases (as :mod:`._pick`, lifted to a carry height); a threaded member, which lives on its
   stub, is gripped in place across its flats (as :mod:`.twist`), or pressed on its top by a cup
   or a soft arm. Fingers grip a plug
   low (:func:`grip_tip`): :data:`FINGER_ROOM` above the socket depth (so they clear the socket
   when it is home), but no higher than :data:`TIP_SAFE` of its tip-over height ``r / mu``
   (closing fingers that touch a standing pin above that tip it over instead of sliding it), and
   between the pads wherever they are mounted (``Footprint.grip_centre``: a head whose pads sit
   off its TCP grips the plug at the pads' middle).
2. **Align** (a plug): over the socket, with the plug's tip ``standoff_mm`` above the mouth. The
   plug's place in the tool is estimated when the move starts (the observed tip relative to the
   TCP, from proprioception) and once more when it has settled 0.3 s over the mouth (the
   observed tip against the observed mouth, at most 15 mm of correction), so a plug that tilted
   or slid in the grasp is re-aimed before the push.
3. **Push** (closed loop, force feedback through the wrist only): the wrist's z target descends
   at ``speed_mm_s`` but never more than ``F / k_z`` below the TCP (``F``: 80 % of the step's
   ``max_force_n``, else :data:`PUSH_N`; ``k_z`` the z servo's stiffness); under a step force
   limit it also rises at that speed while the measured push is over :data:`FORCE_HOLD` of the
   limit (a tool's weight or a stiff cup adds to the servo). Lateral compliance: the x/y target yields to the
   lateral force on the tool (:data:`ADMITTANCE` m per N s, at most 3 mm). Search: a plug that
   stalls on the socket's top face spirals (0.4 mm per turn, up to 3 mm) while pressing, within
   the equal budget :data:`~._common.BUDGET_S` ``insert_search``. A plug is pushed home (onto
   the socket floor) when the fingertips reach (they stop :data:`START_CLEAR` over the socket);
   a grip too low for that only starts it, :data:`START_DEPTH` past the chamfer, where the bore
   holds it upright. A threaded member goes to the ``depth_mm`` criterion plus 0.5 mm (without
   one, the thread start). The push stops there, or when the depth stalls for :data:`STALL_S`.
   **Press home** (a plug started or stalled at least :data:`RESEAT_MIN` deep, within the equal
   budget :data:`~._common.BUDGET_S` ``insert_reseat`` plus the press from :data:`RESEAT_MIN`
   to the goal at ``speed_mm_s``, the same for every tool): let go (0.3 s), rise until the tip is :data:`RESEAT_CLEAR` over the plug's top,
   close into a fist and press the plug home from its top with the same force limit: a plug held
   cocked in the grasp straightens once released. A ``force_n`` at_least is a goal on top of the
   depth: at the goal depth the push (or the press home) goes on pressing until the measured net
   force is over it for :data:`~._common.FORCE_HOLD_S`, within the equal budget ``force_press``.
4. **Release**: the z target comes back to the wrist (the push stops) before the fingers open
   over 0.4 s, then rise 50 mm (``hold_after`` keeps the grip and skips the recovery).

**Predicate** (:class:`InsertPredicate`): ``depth_mm`` (the template state at the end),
``lateral_force_n`` (the peak force across the insertion axis the tool applied, during the
push), ``force_n`` (the peak net tool force), ``jam`` (1 when the member ends more than 0.5 mm
short of the depth the step asks for: its ``depth_mm`` criterion, else the full depth), ``penetration_mm``. A step that fails its criteria with ``jam = 1`` is
``jam``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rtt.sim.controller import Phase, min_jerk
from rtt.sim.objects import Placement
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
    contacts,
    criterion_bounds,
    force_allowed,
    force_goal,
    heading_for_axis,
    hold_out_of_reach,
    idle_command,
    move_time,
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
    tool_point_world,
    transit,
    wrist_frame,
    wrist_target,
)
from rtt.studio.primitives._pick import pick_geometry
from rtt.studio.templates import member_body, thread_info
from rtt.studio.types import PrimitiveKind, TemplateKind

__all__ = ["InsertController", "InsertPredicate", "goal_depth"]

PUSH_N = 30.0  # N, the default push limit [assumed]
ADMITTANCE = 0.002  # m per N s of lateral force [assumed]
MAX_SHIFT = 0.003  # m
STALL_S = 0.6
STALL_MM = 0.1
SPIRAL_PITCH = 0.0004  # m per turn
SPIRAL_PERIOD = 0.5  # s per turn
CARRY = 0.020  # m: the plug's tip this far above the standoff point while carried
FINGER_ROOM = 0.005  # m: fingertips grip this far above the socket depth (and below the plug top)
TIP_SAFE = 0.8  # of the tip-over height r / mu: a grip contact below it slides a plug, not tips it
PAD_CONTACT = 0.003  # m: a finger's first contact sits this far above its tip [assumed]
START_CLEAR = 0.003  # m: a gripped plug goes in until the fingertips are this far over the socket
START_DEPTH = 0.001  # m past the chamfer: how far the grip starts a plug it cannot take home
RELEASE_S = 0.4
RETRACT = 0.050
FORCE_HOLD = 0.9  # of a step's max_force_n: the measured push the z servo backs off above
PULL_TOL = 0.0003  # m: a pressed plug that comes up this far with the rising fist is being held
RESEAT_MIN = 0.001  # m: a plug at least this deep in the bore stands upright for a press home
RESEAT_CLEAR = 0.015  # m over the plug's top while the fingers close into a fist


def grip_start(reach: float, goal: float, info: Mapping[str, float]) -> float:
    """How deep a finger grip takes a plug before it lets go: all the way when its fingertips
    reach (``reach`` >= ``goal``), else only just past the chamfer, where the bore holds the plug
    upright for the press home (the plan budgets the press from the same depth)."""
    if reach >= goal:
        return goal
    return min(reach, info["chamfer_m"] + START_DEPTH)


def grip_tip(info: Mapping[str, float], height: float, mu: float, floor: float) -> float:
    """Where fingers grip a standing plug (their tips, m above its bottom).

    High enough that they clear the socket at full depth (:data:`FINGER_ROOM` above it), but
    never so high that the first finger to touch tips the plug over instead of sliding it: a
    push at height ``z`` on a plug of radius ``r`` standing on the table tips it once ``z``
    exceeds ``r / mu`` (the friction at its foot against its weight's lever), so the contact
    (:data:`PAD_CONTACT` above the tips) stays below :data:`TIP_SAFE` of that. A short or slim
    plug is therefore gripped low and only started in its socket by the grip; it is pressed home
    from its top (``press_home``).
    """
    full = info["depth_m"] + FINGER_ROOM
    tip_safe = TIP_SAFE * (0.5 * info["plug_d_m"]) / mu - PAD_CONTACT
    return max(floor, min(full, tip_safe, height - FINGER_ROOM))


def push_depth(ctx: StepContext) -> float:
    """How deep the controller pushes (m): a plug all the way home (seated on the socket floor,
    a connector is not left part way); a threaded member to its goal depth (it cannot pass the
    thread start without turning)."""
    if ctx.obj.kind is TemplateKind.PORT_PLUG:
        return ctx.obj.handles.info["depth_m"] - 0.0002
    return goal_depth(ctx)


def goal_depth(ctx: StepContext) -> float:
    """The depth the step asks for (m): the ``depth_mm`` criterion plus 0.5 mm, else the full
    depth (seated, or for a thread its start)."""
    info = thread_info(ctx.obj)
    full = info["depth_m"] - 0.0003 if ctx.obj.kind is TemplateKind.PORT_PLUG else info["lead_m"]
    lo, _ = criterion_bounds(ctx, "depth_mm")
    if lo is not None:
        cap = info["depth_m"] if ctx.obj.kind is TemplateKind.PORT_PLUG else info["lead_m"] + info["seat_m"]
        return min(cap, lo / 1000.0 + 0.0005)
    return full


def _servo_z(z: float, palm_z: float, push: float, run: Mapping[str, Any], dt: float) -> float:
    """The z target of a force-limited push: it descends at the insertion speed but never more
    than ``F / k_z`` below the wrist (so the servo pushes at most ``F``), and, under a step force
    limit, rises at that speed while the measured push on the object is over :data:`FORCE_HOLD` of
    the limit (the tool's own weight or a stiff cup adds to the servo's force)."""
    if push > run["force_cap"]:
        return z + run["speed"] * dt
    return max(z - run["speed"] * dt, palm_z - run["push_n"] / run["kz"])


def _pressing(run: dict[str, Any], t: float, force: float, dt: float) -> bool:
    """At the goal depth: whether the push goes on pressing to the step's force goal (the measured
    net force over it for :data:`~._common.FORCE_HOLD_S`), within the equal budget
    :data:`~._common.BUDGET_S` ``force_press`` from when the depth was reached. A step without a
    force goal is done at the depth."""
    goal = run["force_goal"]
    if goal is None:
        return False
    if run["press_from"] is None:
        run["press_from"] = t
    if force >= goal:
        run["force_held"] += dt
    return run["force_held"] < FORCE_HOLD_S and t - run["press_from"] < BUDGET_S["force_press"]


def _depth_observed(ctx: StepContext, data: mujoco.MjData, run: Mapping[str, Any]) -> float:
    if ctx.obj.kind is TemplateKind.PORT_PLUG:
        return float(ctx.observe_site(data, "socket")[2] - ctx.observe_site(data, "plug_tip")[2])
    return run["depth0"] + float(run["member_z0"] - ctx.observe_site(data, "member")[2])


def _held(ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
    if ctx.obj.kind is TemplateKind.PORT_PLUG:
        lifted = float(ctx.observe_site(data, "plug_tip")[2]) > ctx.task.physics.table_top_z + 0.003
        return con.n > 0 and lifted
    return len(con.bodies) >= 2


@dataclass(frozen=True)
class InsertController:
    primitive: PrimitiveKind = PrimitiveKind.INSERT
    name: str = "studio_insert"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        step, obj, task = ctx.step, ctx.obj, ctx.task
        info = thread_info(obj)
        tool = tool_of(ctx, member_body(obj))
        sch = schedule(ctx)
        sch.tool = tool
        start = start_pose(ctx)
        roll = float(start[4])
        held = _held(ctx, model, data)
        path = Path.start(start, ctx.scene.wrist.pitch_speed_dps)
        port = obj.kind is TemplateKind.PORT_PLUG
        yaw = float(start[3])
        guard = None
        t_close = math.nan
        if not held and port:
            part, site, base_z, h = pick_geometry(ctx, data)
            inst = ctx.task_spec.object(step.object)
            yaw_obj = math.radians(inst.pose.yaw_deg)
            pplan = tool.primitive.plan(
                part, Placement(float(site[0]), float(site[1]), yaw_obj), task, ctx.head.build
            )
            heading = yaw_obj + pplan.yaw_offset
            heading += 2.0 * math.pi * round((start[3] + roll - heading) / (2.0 * math.pi))
            yaw = heading - roll
            c = task.controller
            tip = max(c.min_tip_clearance, pplan.tip_height)
            grip = tool.tip
            if tool.mechanism == "fingers":
                tip = grip_tip(info, h, float(task.physics.friction), c.min_tip_clearance)
                grip = tool.closing_tip(0.0)  # the tips' lowest point while they close
                # the plug goes between the pads, wherever they are mounted
                grip = grip + np.array([tool.footprint(0.0).grip_centre(0.5 * info["plug_d_m"]), 0.0, 0.0])
            approach = wrist_target(
                np.array([site[0], site[1], base_z + max(tip + c.approach_offset, h + c.approach_clearance)]),
                grip,
                yaw,
                roll,
            )
            grasp = wrist_target(np.array([site[0], site[1], base_z + tip]), grip, yaw, roll)
            transit(path, approach, max(approach[2], safe_palm_z(ctx, model, data, tool)))
            path.hold(c.settle_time)
            t_close = path.to(grasp, c.descend_time)
            path.hold(c.close_time + c.squeeze_time)
            carry_z = info["socket_top_z"] + float(step.params["standoff_mm"]) / 1000.0 + CARRY
            lifted = grasp.copy()
            lifted[2] = max(
                grasp[2], carry_z + tip - float(tool.tip[2]), safe_palm_z(ctx, model, data, tool, tip + 0.005)
            )
            path.to(lifted, max(1.0, move_time(grasp, lifted)))
        elif not held:  # a threaded member on its stub: grip it in place
            axis_pt = ctx.observe_site(data, "member")
            flat = ctx.observe_site(data, "flat")
            u = np.array([flat[0] - axis_pt[0], flat[1] - axis_pt[1], 0.0])
            heading = heading_for_axis(u / max(1e-9, float(np.linalg.norm(u))), 0.0, float(start[3] + roll))
            yaw = heading - roll
            if tool.mechanism == "fingers":  # across the flats, tips 3 mm above the member's bottom
                tip_z = float(axis_pt[2]) - info["length_m"] + 0.003
            else:  # a cup or a soft arm presses on the member's top
                tip_z = float(axis_pt[2]) - 0.004
            engage = wrist_target(np.array([axis_pt[0], axis_pt[1], tip_z]), tool.tip, yaw, roll)
            above = engage + offset(z=0.06)
            transit(path, above, max(above[2], safe_palm_z(ctx, model, data, tool)))
            guard = Guarded(path.last, engage, path.end, 1.0 + BUDGET_S["engage_slack"])
            t_close = guard.end
            path.to(engage, guard.duration)
            path.hold(1.3)
        # every planned move, and the seat the member goes into, within the palm's travel
        mouth = wrist_target(ctx.observe_site(data, "socket" if port else "member"), tool.tip, yaw, roll)
        far = reach_problem(*path.poses, mouth)
        if far is not None:
            return hold_out_of_reach(ctx, start, far, "a planned move or the seat")
        pre = tool.preshape(2.0 * info["flat_m"]) if (not held and not port) else 0.0
        t_align = path.end
        align_t = 1.2 if port else 0.0
        t_push = t_align + align_t + (0.3 if port else 0.0)
        goal = goal_depth(ctx)
        target = max(goal, push_depth(ctx))
        depth_now = observed_state(ctx, data)["depth_mm"] / 1000.0
        speed = float(step.params["speed_mm_s"]) / 1000.0
        travel = target - depth_now + (float(step.params["standoff_mm"]) / 1000.0 if port else 0.0)
        push_t = 1.3 * max(travel, 0.001) / speed + BUDGET_S["insert_slack"]
        press_t = 0.0
        if port:
            # jam recovery presses the plug home from wherever the grip left it (a grip that cannot
            # take it home starts it just past the chamfer; a slipping grip stalls anywhere): every
            # tool gets the budget of the longest press, from the shallowest plug it recovers
            press_t = 1.3 * max(0.0, target - RESEAT_MIN) / speed
            push_t += BUDGET_S["insert_search"] + BUDGET_S["insert_reseat"] + press_t
        f_goal = force_goal(ctx)
        if f_goal is not None:  # pressing on at the goal depth to a force_n at_least
            push_t += BUDGET_S["force_press"]
        t_end_push = t_push + push_t
        tail = 0.0 if step.hold_after else RELEASE_S + 0.6 + 0.2
        duration = min(30.0, t_end_push + 0.2 + tail)
        sch.engaged = t_push
        sch.values.update(t_close=t_close, pre=pre, held=float(held), goal=goal)
        member0 = ctx.observe_site(data, "member") if not port else np.zeros(3)
        run: dict[str, Any] = {
            "path": path,
            "guard": guard,
            "t_align": t_align,
            "align_t": align_t,
            "t_push": t_push,
            "t_end_push": min(t_end_push, duration - tail - 0.1),
            "yaw": yaw,
            "roll": roll,
            "goal": target,
            "press_t": press_t,
            "speed": speed,
            "push_n": force_allowed(ctx, PUSH_N),
            "force_goal": f_goal,
            "force_held": 0.0,
            "press_from": None,
            # a step's force limit is also held against the measured push (the servo cap alone
            # lets a tool's weight or a stiff cup add to it); without one, the cap is the rule
            "force_cap": FORCE_HOLD * step.constraints.max_force_n
            if step.constraints.max_force_n
            else math.inf,
            "kz": task.gripper.z_kp,
            "port": port,
            "standoff": float(step.params["standoff_mm"]) / 1000.0,
            "depth0": depth_now,
            "member_z0": float(member0[2]),
            "align": None,
            "z_cmd": None,
            "shift": np.zeros(2),
            "best": -math.inf,
            "best_t": 0.0,
            "search_t": 0.0,
            "done_at": None,
            "hold": step.hold_after,
            "last": None,
        }
        return StepPlan(
            duration_s=duration,
            values={"run": run, "schedule": sch},
            note=f"insert to {1000 * goal:.1f} mm ({'held' if held else 'picked first'})",
        )

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        sch = plan.values["schedule"]
        run = plan.values["run"]
        tool = sch.tool
        if "idle" in run:  # out of reach: hold still, the tool as it was
            return idle_command(t, ctx, run, model, data)
        v = sch.values
        guard: Guarded | None = run["guard"]
        if guard is not None and guard.t0 <= t < guard.end + 1e-9:
            pressing = contacts(model, data, tool_geoms(ctx.scene), ctx.scene.ids.object_geoms).normal
            wrist = guard.at(t, pressing)
        elif t < run["t_align"]:
            wrist = run["path"].at(t) if guard is None or t < guard.t0 else guard.at(guard.end, 1e9)
        elif t < run["t_push"]:
            wrist = self._align(t, ctx, run, data)
        else:
            wrist = self._push(t, ctx, run, model, data)
        held = v["held"] > 0.0
        t_close = v["t_close"]
        if held:
            closure = 1.0
        else:
            pre = v["pre"]
            closure = pre + (1.0 - pre) * min(max((t - t_close) / 1.0, 0.0), 1.0)
        if "closure_override" in run:
            closure = run["closure_override"]
        release = run.get("release_at")
        if release is not None and not run["hold"] and t >= release:
            if "reseat_at" not in run:
                closure *= max(0.0, 1.0 - (t - release) / RELEASE_S)
                rise = RETRACT * min_jerk(min(1.0, max(0.0, (t - release - RELEASE_S) / 0.6)))
            else:
                rise = self._fist_rise(t, ctx, run, data, release)
                if run.get("let_go_at") is not None:
                    closure *= max(0.0, 1.0 - (t - run["let_go_at"]) / RELEASE_S)
            wrist = wrist + offset(z=rise)
        phase = Phase.HOLD if held or t >= t_close else Phase.DESCEND
        return PrimitiveCommand(wrist, tool.controls(t, closure, model, data, phase))

    @staticmethod
    def _fist_rise(
        t: float, ctx: StepContext, run: dict[str, Any], data: mujoco.MjData, release: float
    ) -> float:
        """How far a fist that pressed the plug home has risen: it rises closed; if the plug comes
        up with it (fingers that closed around the plug's top), it holds and lets go first."""
        depth = _depth_observed(ctx, data, run)
        if run.get("seated") is None:
            run["seated"] = depth
        if run.get("let_go_at") is None and depth < run["seated"] - PULL_TOL:
            run["let_go_at"] = t
            run["rise_held"] = run.get("rise", 0.0)
        if run.get("let_go_at") is None:
            rise = RETRACT * min_jerk(min(1.0, max(0.0, (t - release - RELEASE_S) / 0.6)))
        else:
            held = run["rise_held"]
            rise = held + (RETRACT - held) * min_jerk(
                min(1.0, max(0.0, (t - run["let_go_at"] - RELEASE_S) / 0.6))
            )
        run["rise"] = rise
        return rise

    def _align(self, t: float, ctx: StepContext, run: dict[str, Any], data: mujoco.MjData) -> np.ndarray:
        sch = schedule(ctx)
        tool = sch.tool
        if run["align"] is None:
            start = run["path"].last if run["guard"] is None else run["guard"].at(run["guard"].end, 1e9)
            tcp = tool_point_world(ctx.scene, data, tool.tcp)
            tip_in_tool = wrist_frame(ctx.scene, data).T @ (ctx.observe_site(data, "plug_tip") - tcp)
            socket = ctx.observe_site(data, "socket")
            over = (
                socket + np.array([0.0, 0.0, run["standoff"]]) - rot_z(run["yaw"] + run["roll"]) @ tip_in_tool
            )
            run["align"] = (start, wrist_target(over, tool.tcp, run["yaw"], run["roll"]))
        a, b = run["align"]
        if run["align_t"] <= 0.0:
            return b.copy()
        s = min_jerk(min(1.0, (t - run["t_align"]) / run["align_t"]))
        return a + s * (b - a)

    def _push(
        self, t: float, ctx: StepContext, run: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
    ) -> np.ndarray:
        sch = schedule(ctx)
        tool = sch.tool
        if run["align"] is not None:
            base = run["align"][1]
        elif run["guard"] is not None:
            base = run["guard"].at(run["guard"].end, 1e9)
        else:
            base = run["path"].last
        dt = ctx.task.controller.control_period
        if run["z_cmd"] is None:
            run["z_cmd"] = float(base[2])
            if run["port"]:  # re-aim once the carried plug has settled: its observed tip over the mouth
                gap = ctx.observe_site(data, "socket") - ctx.observe_site(data, "plug_tip")
                run["realign"] = np.clip(gap[:2], -0.015, 0.015)
                # the grip takes the plug in until the tool's tip is START_CLEAR over the mouth
                tip_z = float(tool_point_world(ctx.scene, data, tool.tip)[2])
                reach = tip_z - float(ctx.observe_site(data, "plug_tip")[2]) - START_CLEAR
                if tool.mechanism == "fingers":
                    run["grip_goal"] = grip_start(reach, run["goal"], ctx.obj.handles.info)
                else:
                    run["grip_goal"] = run["goal"]
        base = base.copy()
        base[:2] += run.get("realign", np.zeros(2))
        if "reseat_at" in run and run["done_at"] is None:
            return self._reseat(t, ctx, run, model, data)
        depth = _depth_observed(ctx, data, run)
        if run["done_at"] is None:
            con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
            palm_z = float(data.qpos[ctx.scene.ids.palm_qpos[2]])
            run["z_cmd"] = _servo_z(run["z_cmd"], palm_z, -float(con.net[2]), run, dt)
            run["shift"] = np.clip(run["shift"] - ADMITTANCE * dt * con.net[:2], -MAX_SHIFT, MAX_SHIFT)
            if depth > run["best"] + STALL_MM / 1000.0:
                run["best"], run["best_t"] = depth, t
            stalled = t - run["best_t"] >= STALL_S
            at_mouth = run["port"] and depth < 0.0005
            reached = depth >= run.get("grip_goal", run["goal"])
            if (
                reached
                and depth >= run["goal"] - 0.0005
                and _pressing(run, t, float(np.linalg.norm(con.net)), dt)
            ):
                pass  # at the goal depth, pressing on to the force goal
            elif stalled and at_mouth and run["search_t"] < BUDGET_S["insert_search"]:
                run["search_t"] += dt  # spiral at the mouth while pressing
                phi = 2.0 * math.pi * run["search_t"] / SPIRAL_PERIOD
                r = min(MAX_SHIFT, SPIRAL_PITCH * phi / (2.0 * math.pi))
                run["spiral"] = np.array([r * math.cos(phi), r * math.sin(phi)])
            elif reached or stalled or t >= run["t_end_push"]:
                short = depth < run["goal"] - 0.0005
                started = depth >= run.get("grip_goal", run["goal"])  # as far as the grip can take it
                if (
                    short
                    and (stalled or started)
                    and run["port"]
                    and not run["hold"]
                    and depth > RESEAT_MIN
                    and "reseat_at" not in run
                ):
                    run["reseat_at"] = t  # let go, then press the plug home from its top
                    run["reseat_from"] = np.array(
                        [
                            *base[:2] + run["shift"] + run.get("spiral", np.zeros(2)),
                            run["z_cmd"],
                            base[3],
                            base[4],
                            base[5],
                        ]
                    )
                    # under a step force limit the push eases off while the fingers let go (an
                    # opening finger on a pushed plug presses it down past the limit)
                    run["reseat_palm"] = palm_z if math.isfinite(run["force_cap"]) else run["z_cmd"]
                    tip_z = float(tool_point_world(ctx.scene, data, tool.tip)[2])
                    run["reseat_rise"] = float(ctx.observe_site(data, "plug")[2]) + RESEAT_CLEAR - tip_z
                else:
                    run["done_at"] = t
                    run["release_at"] = t + 0.1
                    sch.released = run["release_at"]
                    sch.values["push_end"] = t
        if "reseat_at" in run and run["done_at"] is None:
            return self._reseat(t, ctx, run, model, data)
        wrist = base.copy()
        wrist[:2] += run["shift"] + run.get("spiral", np.zeros(2))
        wrist[2] = run["z_cmd"]
        if run["done_at"] is not None and run["last"] is None:
            run["last"] = wrist.copy()
            run["last"][2] = float(data.qpos[ctx.scene.ids.palm_qpos[2]])  # stop pushing, then let go
        if run["last"] is not None:
            wrist = run["last"].copy()
        return wrist

    def _reseat(
        self, t: float, ctx: StepContext, run: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
    ) -> np.ndarray:
        """Jam recovery (the fixed budget ``insert_reseat``): open 0.3 s, rise until the tip is
        :data:`RESEAT_CLEAR` over the plug's top, close into a fist 0.3 s, then press down with the
        push limit."""
        sch = schedule(ctx)
        dt = ctx.task.controller.control_period
        u = t - run["reseat_at"]
        start = run["reseat_from"]
        wrist = start.copy()
        if u < 0.3:
            run["closure_override"] = max(0.0, 1.0 - u / 0.3)
            wrist[2] = start[2] + (run["reseat_palm"] - start[2]) * min(1.0, u / 0.1)
            return wrist
        top = start[2] + run["reseat_rise"]  # where the fist closes, over the plug's top
        wrist[2] = run["reseat_palm"] + (top - run["reseat_palm"]) * min_jerk(min(1.0, (u - 0.3) / 0.5))
        if u < 1.1:
            run["closure_override"] = 0.0 if u < 0.8 else min(1.0, (u - 0.8) / 0.3)
            run["press_z"] = float(wrist[2])
            return wrist
        run["closure_override"] = 1.0
        palm_z = float(data.qpos[ctx.scene.ids.palm_qpos[2]])
        push = -float(contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids).net[2])
        run["press_z"] = _servo_z(run["press_z"], palm_z, push, run, dt)
        wrist[2] = run["press_z"]
        depth = _depth_observed(ctx, data, run)
        net = float(np.linalg.norm(contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids).net))
        home = depth >= run["goal"] and not _pressing(run, t, net, dt)
        if home or u >= BUDGET_S["insert_reseat"] + run["press_t"] + (
            BUDGET_S["force_press"] if run["force_goal"] is not None else 0.0
        ):
            run["done_at"] = t
            run["release_at"] = t + 0.1
            run["last"] = wrist.copy()
            run["last"][2] = palm_z  # stop pressing, then rise
            sch.released = run["release_at"]
            sch.values["push_end"] = t
        return wrist


@dataclass(frozen=True)
class InsertPredicate:
    primitive: PrimitiveKind = PrimitiveKind.INSERT
    name: str = "studio_insert"

    def observe(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None:
        sch = schedule(ctx)
        acc.update(ctx.object_state(model, data))
        con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
        acc["force_n"] = max(acc.get("force_n", 0.0), float(np.linalg.norm(con.net)))
        acc["penetration_mm"] = max(acc.get("penetration_mm", 0.0), 1000.0 * con.depth)
        acc.setdefault("lateral_force_n", 0.0)
        end = sch.values.get("push_end", math.nan)
        if not math.isnan(sch.engaged) and t >= sch.engaged and (math.isnan(end) or t <= end):
            acc["lateral_force_n"] = max(acc["lateral_force_n"], con.lateral)

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict:
        far = out_of_reach_verdict(ctx, acc)
        if far is not None:
            return far
        q = dict(acc)
        goal = schedule(ctx).values.get("goal", goal_depth(ctx))
        q["jam"] = float(q.get("depth_mm", 0.0) < 1000.0 * goal - 1.0)  # goal carries the 0.5 mm margin
        out = criteria_verdict(ctx.step, q)
        if not out.success and out.failure_mode == "criterion_not_met" and q["jam"] > 0.0:
            return StepVerdict(False, "jam", q)
        return out


register_primitive(InsertController(), InsertPredicate())
