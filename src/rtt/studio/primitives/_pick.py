"""The three picks (``grasp``, ``suction``, ``wrap``): the v1 family primitives in the Studio.

**Controller** (:class:`PickController`): the wrist follows the v1 phases with the v1 arithmetic
(:mod:`rtt.sim.controller`): after a transit (rise to a safe height, travel and turn, descend),
SETTLE 0.3 s at the approach height, DESCEND 1.0 s to the grasp height, CLOSE 1.0 s ramp plus
0.3 s squeeze, LIFT by the step's ``lift_mm`` in 1.5 s, HOLD for ``hold_s``. The head's own v1
primitive plans the tip height (:meth:`GraspPrimitive.plan`) and drives its actuators from the
shared closure ramp (:meth:`GraspPrimitive.command` through a
:class:`~rtt.studio.plan.V1SceneView`): fingers close, a cup's vacuum switches on only while
sealed, a soft arm pulls its cable. Then:

* with a ``target`` bin: carry the part over the bin (its bottom 20 mm above the bin's rim),
  lower it until its bottom is 12 mm above the bin floor, open over 0.5 s, wait 0.4 s, retract;
* without a target and without ``hold_after``: put it back where it was, open, retract;
* with ``hold_after``: the step ends holding it (the next step uses the same tool).

The pick's mechanism is the tool's: a step's primitive names the intended mechanism and sets the
fidelity tier, and physics decides whether this tool achieves it.

**Predicate** (:class:`PickPredicate`), on the true state at every check tick:
``lift_mm`` (the peak rise of the object above where it started the step), ``hold_s`` (the
longest time it stayed lifted at least half the planned lift, at least 10 mm, while the tool
touched it), ``in_target`` (1 when it ends inside the target bin's interior and the tool no
longer touches it), ``tilt_deg`` (at the end, from its attitude at the start), ``force_n`` (the
peak sum of the tool's normal forces on it), ``penetration_mm`` (the deepest tool-object
penetration), and ``engaged`` (the v1 criterion at the v1 registration moment, 0.25 s after the
close ramp; informative). Then :func:`~rtt.studio.plan.criteria_verdict`; a part that was lifted
and lost before the step's release is ``dropped``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from rtt.sim.controller import Phase
from rtt.sim.objects import Placement
from rtt.studio.plan import (
    PrimitiveCommand,
    StepContext,
    StepPlan,
    StepVerdict,
    criteria_verdict,
)
from rtt.studio.primitives._common import (
    Path,
    contacts,
    move_time,
    safe_palm_z,
    schedule,
    tool_geoms,
    tool_of,
    transit,
    wrist_target,
)
from rtt.studio.types import PrimitiveKind, TemplateKind

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec

__all__ = ["PickController", "PickPredicate", "pick_body", "pick_geometry"]

RELEASE_S = 0.5
RELEASE_WAIT_S = 0.4
RETRACT = 0.08  # m
BIN_RIM_CLEARANCE = 0.020
BIN_DROP = 0.012


def pick_body(ctx: StepContext) -> int:
    """The body a pick moves: the part, the plug or the threaded member."""
    obj = ctx.obj
    return int(obj.body_ids[0] if obj.kind is TemplateKind.PART else obj.body_ids[-1])


def _slug(name: str, diameter: float, length: float) -> PartSpec:
    from rtt.parts.library import Material, build_part

    return build_part(
        name,
        "slug",
        {"diameter": 1000.0 * diameter, "length": 1000.0 * length},
        Material("studio", 1200.0, (0.2, 0.5, 0.9)),
    )


def pick_geometry(ctx: StepContext, data: mujoco.MjData) -> tuple[PartSpec, np.ndarray, float, float]:
    """The part spec the v1 primitive plans with, the aim point (x, y of the observed handle),
    the object's bottom height and its height (m)."""
    from rtt.studio.templates.part import part_spec_of

    obj, step = ctx.obj, ctx.step
    site = ctx.observe_site(data, step.handle)
    inst = ctx.task_spec.object(step.object)
    info = obj.handles.info
    if obj.kind is TemplateKind.PART:
        part = part_spec_of(inst)
        h = float(part.size[2])
        return part, site, float(site[2]) - 0.5 * h, h
    if obj.kind is TemplateKind.PORT_PLUG:
        d, h = info["plug_d_m"], info["plug_len_m"]
    else:  # threaded member: its top site
        d, h = info["diameter_m"], info["length_m"]
    return _slug(f"studio_{inst.name}", d, h), site, float(site[2]) - h, h


@dataclass(frozen=True)
class PickController:
    """One of the three picks (module docstring)."""

    primitive: PrimitiveKind
    name: str = "studio_pick"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        task, step = ctx.task, ctx.step
        tool = tool_of(ctx, pick_body(ctx))
        sch = schedule(ctx)
        sch.tool = tool
        part, site, base_z, h = pick_geometry(ctx, data)
        inst = ctx.task_spec.object(step.object)
        yaw_obj = math.radians(inst.pose.yaw_deg)
        pplan = tool.primitive.plan(
            part, Placement(float(site[0]), float(site[1]), yaw_obj), task, ctx.head.build
        )
        start = np.asarray(ctx.wrist_start, dtype=np.float64)
        roll = float(start[4])
        heading = yaw_obj + pplan.yaw_offset
        heading = heading + 2.0 * math.pi * round((start[3] + roll - heading) / (2.0 * math.pi))
        yaw = heading - roll
        c = task.controller
        tip = max(c.min_tip_clearance, pplan.tip_height + c.grasp_z_offset)
        approach_tip = max(tip + c.approach_offset, h + c.approach_clearance)
        xy = site[:2]
        grasp = wrist_target(np.array([xy[0], xy[1], base_z + tip]), tool.tip, yaw, roll)
        approach = wrist_target(np.array([xy[0], xy[1], base_z + approach_tip]), tool.tip, yaw, roll)
        path = Path.start(start)
        transit(path, approach, max(approach[2], safe_palm_z(ctx, model, data, tool)))
        path.hold(c.settle_time)
        t_close = path.to(grasp, c.descend_time)
        t_lift = path.hold(c.close_time + c.squeeze_time)
        lifted = grasp + np.array([0.0, 0.0, float(step.params["lift_mm"]) / 1000.0, 0.0, 0.0])
        t_hold = path.to(lifted, c.lift_time)
        t_hold_end = path.hold(float(step.params["hold_s"]))
        sch.reg = t_close + c.close_time + task.registration.settle_time
        sch.hold = (t_hold, t_hold_end)
        sch.engaged = t_lift
        t_release = math.nan
        carried = tip + 0.005  # the object hangs this far below the tool tip
        if ctx.target is not None:
            tinfo = ctx.target.handles.info
            floor = ctx.observe_site(data, "bin", ctx.target.name)
            wall = tinfo["floor_z"] - task.physics.table_top_z  # the bin's floor thickness
            rim = float(floor[2]) - wall + tinfo["height_m"]
            over_z = rim + BIN_RIM_CLEARANCE + carried - float(tool.tip[2])
            safe = max(over_z, safe_palm_z(ctx, model, data, tool, carried), lifted[2])
            above = wrist_target(np.array([floor[0], floor[1], 0.0]), tool.tip, yaw, roll)
            above[2] = safe
            transit(path, above, safe)
            down = above.copy()
            down[2] = float(floor[2]) + BIN_DROP + tip - float(tool.tip[2])
            t_release = path.to(down, max(0.6, move_time(above, down)))
        elif not step.hold_after:
            down = grasp + np.array([0.0, 0.0, 0.002, 0.0, 0.0])
            t_release = path.to(down, c.lift_time)
        if not math.isnan(t_release):
            path.hold(RELEASE_S + RELEASE_WAIT_S)
            up = path.last + np.array([0.0, 0.0, RETRACT, 0.0, 0.0])
            path.to(up, 0.6)
            sch.released = t_release
        sch.values.update(t_close=t_close, t_lift=t_lift, t_release=t_release, close_time=c.close_time)
        sch.values["lift_threshold_mm"] = max(10.0, 0.5 * float(step.params["lift_mm"]))
        return StepPlan(
            duration_s=path.end + 0.1,
            values={"path": path, "schedule": sch},
            note=f"{self.primitive.value} with the head's v1 {ctx.head.spec.primitive} primitive",
        )

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        sch = plan.values["schedule"]
        v = sch.values
        t_close, t_rel = v["t_close"], v["t_release"]
        closure = min(max((t - t_close) / v["close_time"], 0.0), 1.0)
        if not math.isnan(t_rel) and t >= t_rel:
            closure = max(0.0, 1.0 - (t - t_rel) / RELEASE_S)
        if t < t_close:
            phase = Phase.DESCEND
        elif t < v["t_lift"]:
            phase = Phase.CLOSE
        else:
            phase = Phase.HOLD
        u = sch.tool.controls(t, closure, model, data, phase)
        return PrimitiveCommand(plan.values["path"].at(t), u)


def _tilt_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r0[:, 2] @ r1[:, 2])))))


@dataclass(frozen=True)
class PickPredicate:
    """The picks' success test (module docstring)."""

    primitive: PrimitiveKind
    name: str = "studio_pick"

    def observe(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None:
        sch = schedule(ctx)
        v = sch.values
        body = pick_body(ctx)
        z = float(data.xpos[body][2])
        rot = data.xmat[body].reshape(3, 3).copy()
        if "z0" not in v:
            v["z0"] = z
            sch.values["r0"] = rot.ravel().tolist()  # type: ignore[assignment]
        r0 = np.asarray(v["r0"], dtype=np.float64).reshape(3, 3)
        lift = 1000.0 * (z - v["z0"])
        tool = sch.tool
        con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
        state = ctx.object_state(model, data)
        acc.update({k: x for k, x in state.items() if k not in ("lift_mm", "tilt_deg")})
        v["lift_peak"] = max(v.get("lift_peak", -math.inf), lift)
        acc["lift_mm"] = v["lift_peak"]
        acc["force_n"] = max(acc.get("force_n", 0.0), con.normal)
        acc["penetration_mm"] = max(acc.get("penetration_mm", 0.0), 1000.0 * con.depth)
        thr = v["lift_threshold_mm"]
        dt = ctx.task.score.check_period
        if lift >= thr and con.n > 0:
            v["run"] = v.get("run", 0.0) + dt
            acc["hold_s"] = max(acc.get("hold_s", 0.0), v["run"])
        else:
            v["run"] = 0.0
            acc.setdefault("hold_s", 0.0)
        if "engaged" not in acc and t >= sch.reg and tool is not None:
            acc["engaged"] = float(tool.criterion.engaged(model, data, tool.view).ok)
        lifted = acc["lift_mm"] >= thr
        before_release = math.isnan(sch.released) or t < sch.released
        if lifted and before_release and lift < 0.5 * thr and con.n == 0:
            acc["dropped"] = 1.0
        acc["tilt_deg"] = _tilt_deg(r0, rot)
        acc["in_target"] = self._in_target(ctx, data, body, con.n)

    def _in_target(self, ctx: StepContext, data: mujoco.MjData, body: int, touching: int) -> float:
        tgt = ctx.target
        if tgt is None or touching:
            return 0.0
        info = tgt.handles.info
        b = tgt.body_ids[0]
        rel = data.xmat[b].reshape(3, 3).T @ (data.xpos[body] - data.xpos[b])
        z_floor = float(data.site_xpos[tgt.site_ids["bin"]][2])
        inside = abs(rel[0]) <= info["inner_hx"] and abs(rel[1]) <= info["inner_hy"]
        height_ok = z_floor - 0.005 <= float(data.xpos[body][2]) <= z_floor + info["height_m"]
        return float(inside and height_ok)

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict:
        q = {k: float(v) for k, v in acc.items() if k != "dropped"}
        q.setdefault("hold_s", 0.0)
        q.setdefault("in_target", 0.0)
        out = criteria_verdict(ctx.step, q)
        if not out.success and out.failure_mode == "criterion_not_met" and acc.get("dropped"):
            return StepVerdict(False, "dropped", {**q, "dropped": 1.0})
        return out
