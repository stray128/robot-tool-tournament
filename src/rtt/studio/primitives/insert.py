"""The ``insert`` primitive: align the held member over its seat and push it in with compliance.

Objects: ``port_plug`` (``plug``) and ``threaded_joint`` (``member``).

**Controller** (:class:`InsertController`):

1. **Hold it.** If the tool does not already hold the member (a previous ``grasp`` step with
   ``hold_after``), it picks it: a plug off the table with the head's own v1 primitive and the v1
   phases (as :mod:`._pick`, lifted to a carry height), fingertips at least
   :data:`FINGER_ROOM` above the socket depth so they clear the socket; a threaded member, which
   lives on its stub, is gripped in place across its flats (as :mod:`.twist`), or pressed on its
   top by a cup or a soft arm.
2. **Align** (a plug): over the socket, with the plug's tip ``standoff_mm`` above the mouth. The
   plug's place in the tool is estimated when the move starts (the observed tip relative to the
   TCP, from proprioception) and once more when it has settled 0.3 s over the mouth (the
   observed tip against the observed mouth, at most 15 mm of correction), so a plug that tilted
   or slid in the grasp is re-aimed before the push.
3. **Push** (closed loop, force feedback through the wrist only): the wrist's z target descends
   at ``speed_mm_s`` but never more than ``F / k_z`` below the TCP (``F``: 80 % of the step's
   ``max_force_n``, else :data:`PUSH_N`; ``k_z`` the z servo's stiffness), so the push force stays
   under ``F``. Lateral compliance: the x/y target yields to the lateral force on the tool
   (:data:`ADMITTANCE` m per N s, at most 3 mm). Search: a plug that stalls on the socket's top
   face spirals (0.4 mm per turn, up to 3 mm) while pressing, within the equal budget
   :data:`~._common.BUDGET_S` ``insert_search``. A plug is pushed home (onto the socket floor);
   a threaded member to the ``depth_mm`` criterion plus 0.5 mm (without one, the thread start).
   It stops there, or when the depth stalls for :data:`STALL_S` (a jam).
   **Jam recovery** (a plug stalled part way in, within the equal budget
   :data:`~._common.BUDGET_S` ``insert_reseat``): let go (0.3 s), rise until the tip is
   :data:`RESEAT_CLEAR` over the plug's top, close into a fist and press the plug home from its top
   with the same force limit: a plug held cocked in the grasp straightens once released.
4. **Release**: open over 0.4 s, rise 50 mm (``hold_after`` keeps the grip and skips the
   recovery).

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
    Guarded,
    Path,
    contacts,
    criterion_bounds,
    force_limit,
    heading_for_axis,
    move_time,
    observed_state,
    rot_z,
    safe_palm_z,
    schedule,
    tool_geoms,
    tool_of,
    tool_point_world,
    transit,
    wrist_target,
)
from rtt.studio.primitives._pick import pick_geometry
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
RELEASE_S = 0.4
RETRACT = 0.050
RESEAT_CLEAR = 0.015  # m over the plug's top while the fingers close into a fist


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
    info = ctx.obj.handles.info
    full = info["depth_m"] - 0.0003 if ctx.obj.kind is TemplateKind.PORT_PLUG else info["lead_m"]
    lo, _ = criterion_bounds(ctx, "depth_mm")
    if lo is not None:
        cap = info["depth_m"] if ctx.obj.kind is TemplateKind.PORT_PLUG else info["lead_m"] + info["seat_m"]
        return min(cap, lo / 1000.0 + 0.0005)
    return full


def _depth_observed(ctx: StepContext, data: mujoco.MjData, run: Mapping[str, Any]) -> float:
    if ctx.obj.kind is TemplateKind.PORT_PLUG:
        return float(ctx.observe_site(data, "socket")[2] - ctx.observe_site(data, "plug_tip")[2])
    return run["depth0"] + float(run["member_z0"] - ctx.observe_site(data, "member")[2])


def _held(ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
    if ctx.obj.kind is TemplateKind.PORT_PLUG:
        lifted = float(ctx.obj.site_pos(data, "plug_tip")[2]) > ctx.task.physics.table_top_z + 0.003
        return con.n > 0 and lifted
    return len(con.bodies) >= 2


@dataclass(frozen=True)
class InsertController:
    primitive: PrimitiveKind = PrimitiveKind.INSERT
    name: str = "studio_insert"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        step, obj, task = ctx.step, ctx.obj, ctx.task
        info = obj.handles.info
        tool = tool_of(ctx, obj.body_ids[-1])
        sch = schedule(ctx)
        sch.tool = tool
        start = np.asarray(ctx.wrist_start, dtype=np.float64)
        roll = float(start[4])
        held = _held(ctx, model, data)
        path = Path.start(start)
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
            # grip above the insertion depth so the fingers clear the socket, but still on the plug
            room = info["depth_m"] + FINGER_ROOM
            tip = max(c.min_tip_clearance, pplan.tip_height)
            if tool.mechanism == "fingers":
                tip = max(tip, min(room, h - FINGER_ROOM))
            approach = wrist_target(
                np.array([site[0], site[1], base_z + max(tip + c.approach_offset, h + c.approach_clearance)]),
                tool.tip,
                yaw,
                roll,
            )
            grasp = wrist_target(np.array([site[0], site[1], base_z + tip]), tool.tip, yaw, roll)
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
            above = engage + np.array([0.0, 0.0, 0.06, 0.0, 0.0])
            transit(path, above, max(above[2], safe_palm_z(ctx, model, data, tool)))
            guard = Guarded(path.last, engage, path.end, 1.0 + BUDGET_S["engage_slack"])
            t_close = guard.end
            path.to(engage, guard.duration)
            path.hold(1.3)
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
        if port:
            push_t += BUDGET_S["insert_search"] + BUDGET_S["insert_reseat"]
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
            "speed": speed,
            "push_n": force_limit(ctx, PUSH_N),
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
        v = sch.values
        guard: Guarded | None = run["guard"]
        if guard is not None and guard.t0 <= t < guard.end + 1e-9:
            pressing = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids).normal
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
            if "reseat_at" not in run:  # a fist that pressed the plug home just rises
                closure *= max(0.0, 1.0 - (t - release) / RELEASE_S)
            up = min_jerk(min(1.0, max(0.0, (t - release - RELEASE_S) / 0.6)))
            wrist = wrist + np.array([0.0, 0.0, RETRACT * up, 0.0, 0.0])
        phase = Phase.HOLD if held or t >= t_close else Phase.DESCEND
        return PrimitiveCommand(wrist, tool.controls(t, closure, model, data, phase))

    def _align(self, t: float, ctx: StepContext, run: dict[str, Any], data: mujoco.MjData) -> np.ndarray:
        sch = schedule(ctx)
        tool = sch.tool
        if run["align"] is None:
            start = run["path"].last if run["guard"] is None else run["guard"].at(run["guard"].end, 1e9)
            q = data.qpos
            heading = float(q[ctx.scene.ids.palm_qpos[3]] + q[ctx.scene.ids.roll_qpos])
            tcp = tool_point_world(ctx.scene, data, tool.tcp)
            tip_in_tool = rot_z(heading).T @ (ctx.observe_site(data, "plug_tip") - tcp)
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
        base = base.copy()
        base[:2] += run.get("realign", np.zeros(2))
        if "reseat_at" in run and run["done_at"] is None:
            return self._reseat(t, ctx, run, data)
        depth = _depth_observed(ctx, data, run)
        if run["done_at"] is None:
            con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
            palm_z = float(data.qpos[ctx.scene.ids.palm_qpos[2]])
            lead = run["push_n"] / run["kz"]  # the z servo pushes at most push_n
            run["z_cmd"] = max(run["z_cmd"] - run["speed"] * dt, palm_z - lead)
            run["shift"] = np.clip(run["shift"] - ADMITTANCE * dt * con.net[:2], -MAX_SHIFT, MAX_SHIFT)
            if depth > run["best"] + STALL_MM / 1000.0:
                run["best"], run["best_t"] = depth, t
            stalled = t - run["best_t"] >= STALL_S
            at_mouth = run["port"] and depth < 0.0005
            if stalled and at_mouth and run["search_t"] < BUDGET_S["insert_search"]:
                run["search_t"] += dt  # spiral at the mouth while pressing
                phi = 2.0 * math.pi * run["search_t"] / SPIRAL_PERIOD
                r = min(MAX_SHIFT, SPIRAL_PITCH * phi / (2.0 * math.pi))
                run["spiral"] = np.array([r * math.cos(phi), r * math.sin(phi)])
            elif depth >= run["goal"] or stalled or t >= run["t_end_push"]:
                short = depth < run["goal"] - 0.0005
                if (
                    short
                    and stalled
                    and run["port"]
                    and not run["hold"]
                    and depth > 0.001
                    and "reseat_at" not in run
                ):
                    run["reseat_at"] = t  # let go, then press the plug home from its top
                    run["reseat_from"] = np.array(
                        [
                            *base[:2] + run["shift"] + run.get("spiral", np.zeros(2)),
                            run["z_cmd"],
                            base[3],
                            base[4],
                        ]
                    )
                    tip_z = float(tool_point_world(ctx.scene, data, tool.tip)[2])
                    run["reseat_rise"] = float(ctx.observe_site(data, "plug")[2]) + RESEAT_CLEAR - tip_z
                else:
                    run["done_at"] = t
                    run["release_at"] = t + 0.1
                    sch.released = run["release_at"]
                    sch.values["push_end"] = t
        if "reseat_at" in run and run["done_at"] is None:
            return self._reseat(t, ctx, run, data)
        wrist = base.copy()
        wrist[:2] += run["shift"] + run.get("spiral", np.zeros(2))
        wrist[2] = run["z_cmd"]
        if run["done_at"] is not None and run["last"] is None:
            run["last"] = wrist.copy()
        if run["last"] is not None:
            wrist = run["last"].copy()
        return wrist

    def _reseat(self, t: float, ctx: StepContext, run: dict[str, Any], data: mujoco.MjData) -> np.ndarray:
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
            return wrist
        wrist[2] = start[2] + run["reseat_rise"] * min_jerk(min(1.0, (u - 0.3) / 0.5))
        if u < 1.1:
            run["closure_override"] = 0.0 if u < 0.8 else min(1.0, (u - 0.8) / 0.3)
            run["press_z"] = float(wrist[2])
            return wrist
        run["closure_override"] = 1.0
        palm_z = float(data.qpos[ctx.scene.ids.palm_qpos[2]])
        run["press_z"] = max(run["press_z"] - run["speed"] * dt, palm_z - run["push_n"] / run["kz"])
        wrist[2] = run["press_z"]
        depth = _depth_observed(ctx, data, run)
        if depth >= run["goal"] or u >= BUDGET_S["insert_reseat"]:
            run["done_at"] = t
            run["release_at"] = t + 0.1
            run["last"] = wrist.copy()
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
        q = dict(acc)
        goal = schedule(ctx).values.get("goal", goal_depth(ctx))
        q["jam"] = float(q.get("depth_mm", 0.0) < 1000.0 * goal - 1.0)  # goal carries the 0.5 mm margin
        out = criteria_verdict(ctx.step, q)
        if not out.success and out.failure_mode == "criterion_not_met" and q["jam"] > 0.0:
            return StepVerdict(False, "jam", q)
        return out


register_primitive(InsertController(), InsertPredicate())
