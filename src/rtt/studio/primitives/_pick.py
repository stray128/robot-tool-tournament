"""The three picks (``grasp``, ``suction``, ``wrap``): the v1 family primitives in the Studio.

**Controller** (:class:`PickController`): the wrist follows the v1 phases with the v1 arithmetic
(:mod:`rtt.sim.controller`): after a transit (rise to a safe height, travel and turn, descend),
SETTLE 0.3 s at the approach height, DESCEND 1.0 s to the grasp height, CLOSE 1.0 s ramp plus
0.3 s squeeze, LIFT by the step's ``lift_mm`` in 1.5 s, HOLD for ``hold_s``. The head's own v1
primitive plans the tip height (:meth:`GraspPrimitive.plan`) and drives its actuators from the
shared closure ramp (:meth:`GraspPrimitive.command` through a
:class:`~rtt.studio.plan.V1SceneView`): fingers close, a cup's vacuum switches on only while
sealed, a soft arm pulls its cable. Then:

* with a ``target`` bin: carry the part over the bin (its bottom 20 mm above the bin's rim) and
  **set it down** on the bin floor, or on whatever is already in the bin under it (seen as the
  transit height is seen, :func:`~._common.top_within`); open over 0.5 s, wait 0.4 s, retract;
* without a target and without ``hold_after``: set it back down where it was, open, retract;
* with ``hold_after``: the step ends holding it (the next step uses the same tool).

**Setting down** (:class:`SetDown`): the part is lowered (peak :data:`LOWER_SPEED`) until its
bottom is :data:`SET_DOWN_FROM` above the support as planned, then slowly
(:data:`SET_DOWN_SPEED`) on toward :data:`SET_DOWN_PAST` below it, both guarded on the wrist's
force sensor (:func:`~._common.wrist_load`): the descent stops advancing once the load the tool
carries has dropped by :data:`SET_DOWN_N` from what it carried when the descent began, that is
once the support has taken the part (a part that hangs lower than planned lands early and stops
there). Only then does the tool open: the part is let go resting on its support, never above it
by more than :data:`~._common.RELEASE_HEIGHT_MAX_MM` (a cup's lip, which presses the part away
from the cup while the vacuum holds it, reads as a load like any other).

**Out of reach** (Studio v2): a pick whose grasp, or whose set-down in its target bin, would put
the palm outside its travel (:func:`~._common.reach_problem`, +-350 mm less 5 mm) is not
attempted: it holds still 0.2 s and fails ``out_of_reach``, with where the palm would have to go
and a plain reason.

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

from rtt.sim.controller import Phase, min_jerk
from rtt.sim.objects import Placement
from rtt.studio.plan import (
    PrimitiveCommand,
    StepContext,
    StepPlan,
    StepVerdict,
    criteria_verdict,
)
from rtt.studio.primitives._common import (
    RELEASE_HEIGHT_MAX_MM,
    TRANSIT_SPEED,
    Guarded,
    Path,
    contacts,
    hold_out_of_reach,
    idle_command,
    move_time,
    offset,
    out_of_reach_verdict,
    reach_problem,
    safe_palm_z,
    schedule,
    start_pose,
    tool_geoms,
    tool_of,
    top_within,
    touching,
    transit,
    wrist_load,
    wrist_target,
)
from rtt.studio.types import PrimitiveKind, TemplateKind

if TYPE_CHECKING:
    from rtt.parts.types import PartSpec

__all__ = ["PickController", "PickPredicate", "SetDown", "pick_body", "pick_geometry"]

RELEASE_S = 0.5
RELEASE_WAIT_S = 0.4
RETRACT = 0.08  # m
BIN_RIM_CLEARANCE = 0.020
LOWER_SPEED = TRANSIT_SPEED  # m/s peak while a carried part is lowered toward its support
SET_DOWN_FROM = 0.015  # m above the support (as planned) where the slow descent starts [assumed]
SET_DOWN_PAST = 0.005  # m below it the slow descent aims, so it always meets it [assumed]
SET_DOWN_SPEED = 0.016  # m/s mean of the slow descent (peak 1.875 x) [assumed]
SET_DOWN_N = 3.0  # N: the support has the part once the wrist's load has changed by this [assumed]


@dataclass
class SetDown:
    """Lowering a carried part onto its support (module docstring): ``lower`` to above it, then
    ``slow`` on toward below it (built where ``lower`` stopped), both guarded on the change of the
    wrist's vertical load from ``baseline`` (read when the descent begins)."""

    lower: Guarded
    past: np.ndarray  # the slow descent's goal (six wrist targets)
    slow_s: float
    slow: Guarded | None = None
    baseline: float = math.nan

    @property
    def t0(self) -> float:
        return self.lower.t0

    @property
    def end(self) -> float:
        return self.lower.end + self.slow_s

    def finished(self, t: float) -> bool:
        """Whether the descent is over at ``t`` (its whole planned time: a guard that held it back
        for a moment lets it go on once the load is off again, so a heavy part is not let go early)."""
        return t >= self.end

    def at(self, t: float, load_z: float) -> np.ndarray:
        """The wrist targets at ``t`` with the tool now applying ``load_z`` (N, up) to the world."""
        if math.isnan(self.baseline):
            self.baseline = load_z
        pressing = self.baseline - load_z  # the support's share of the part
        if t <= self.lower.end:
            return self.lower.at(t, pressing)
        if self.slow is None:
            here = self.lower.at(self.lower.end, math.inf)
            self.slow = Guarded(here, self.past, self.lower.end, self.slow_s, force=SET_DOWN_N)
        return self.slow.at(min(t, self.end), pressing)

    def final(self) -> np.ndarray:
        """Where the descent stopped (or will stop): the wrist targets the release holds."""
        if self.slow is not None:
            return self.slow.at(self.slow.end, math.inf)
        return self.lower.at(self.lower.end, math.inf)


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
        start = start_pose(ctx)
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
        far = reach_problem(grasp)  # the approach, the lift and a put-back share its x and y
        if far is not None:
            return hold_out_of_reach(ctx, start, far, "the pick")
        path = Path.start(start, ctx.scene.wrist.pitch_speed_dps)
        transit(path, approach, max(approach[2], safe_palm_z(ctx, model, data, tool)))
        path.hold(c.settle_time)
        t_close = path.to(grasp, c.descend_time)
        t_lift = path.hold(c.close_time + c.squeeze_time)
        lifted = grasp + offset(z=float(step.params["lift_mm"]) / 1000.0)
        t_hold = path.to(lifted, c.lift_time)
        t_hold_end = path.hold(float(step.params["hold_s"]))
        sch.reg = t_close + c.close_time + task.registration.settle_time
        sch.hold = (t_hold, t_hold_end)
        sch.engaged = t_lift
        t_release = math.nan
        carried = tip + 0.005  # the object hangs this far below the tool tip
        support, support_on = math.nan, ""  # where the part is set down (its bottom, m), and on what
        if ctx.target is not None:
            tinfo = ctx.target.handles.info
            floor = ctx.observe_site(data, "bin", ctx.target.name)
            wall = tinfo["floor_z"] - task.physics.table_top_z  # the bin's floor thickness
            rim = float(floor[2]) - wall + tinfo["height_m"]
            over_z = rim + BIN_RIM_CLEARANCE + carried - float(tool.tip[2])
            safe = max(over_z, safe_palm_z(ctx, model, data, tool, carried), lifted[2])
            above = wrist_target(np.array([floor[0], floor[1], 0.0]), tool.tip, yaw, roll)
            above[2] = safe
            far = reach_problem(above)  # the set-down and the retract share its x and y
            if far is not None:
                label = ctx.task_spec.object(ctx.target.name).label.lower()
                return hold_out_of_reach(ctx, start, far, f"setting the part down in the {label}")
            transit(path, above, safe)
            # the bin floor, or the top of whatever already lies in the bin under the part
            reach = 0.5 * math.hypot(*part.size[:2])
            support, support_on = float(floor[2]), "the floor of the bin"
            for name, o in ctx.scene.objects.items():
                if name in (ctx.obj.name, ctx.target.name):
                    continue
                top = top_within(model, data, o.geom_ids, floor[:2], reach, rim)
                if top > support:
                    support, support_on = (
                        top,
                        f"the top of the {ctx.task_spec.object(name).label.lower()} in the bin",
                    )
        elif not step.hold_after:
            support, support_on = base_z, "where it was picked from"  # back where it was
        if not math.isnan(support):
            start_down = path.last
            near = start_down.copy()
            near[2] = support + SET_DOWN_FROM + tip - float(tool.tip[2])
            near[2] = min(near[2], start_down[2])
            past = near.copy()
            past[2] = support - SET_DOWN_PAST + tip - float(tool.tip[2])
            lower_t = max(0.4, move_time(start_down, near, LOWER_SPEED))
            slow_t = (SET_DOWN_FROM + SET_DOWN_PAST) / SET_DOWN_SPEED
            sch.values["guard"] = SetDown(  # type: ignore[assignment]
                Guarded(start_down.copy(), near.copy(), path.end, lower_t, force=SET_DOWN_N), past, slow_t
            )
            path.to(near, lower_t)
            t_release = path.to(past, slow_t)  # at the latest: it lets go once the support has it
            path.hold(RELEASE_S + RELEASE_WAIT_S)
            up = path.last + offset(z=RETRACT)
            path.to(up, 0.6)
            sch.released = t_release
            # the part's bottom is this far below its observed handle site (it is seen there when
            # the set-down ends, wherever the grasp let it slide)
            sch.values.update(support=support, site_above_bottom=float(site[2]) - base_z)
            sch.values["support_on"] = support_on  # type: ignore[assignment]
        # the release moment is decided when the part is down (command); nan until then
        sch.values.update(t_close=t_close, t_lift=t_lift, t_release=math.nan, close_time=c.close_time)
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
        run = plan.values.get("run")
        if run is not None and "idle" in run:  # out of reach: hold still, the tool as it was
            return idle_command(t, ctx, run, model, data)
        v = sch.values
        guard = v.get("guard")
        if (
            isinstance(guard, SetDown)
            and guard.finished(t)
            and math.isnan(v["t_release"])
            and "held_high_mm" not in v
        ):  # the set-down is over: decide before this tick's controls (the release, as planned, at its end)
            self._decide_release(guard.end, ctx, model, data)
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
        if isinstance(guard, SetDown) and t >= guard.t0:
            # setting the part down stops when its support takes it (the wrist's load drops); then
            # it is let go only if it rests at most RELEASE_HEIGHT_MAX_MM above where it was to be
            # set down (the tool on a bin's rim holds it high: letting go would drop it)
            if not guard.finished(t):
                return PrimitiveCommand(guard.at(t, float(wrist_load(ctx.scene, model, data).net[2])), u)
            wrist = guard.final()
            if not math.isnan(v["t_release"]):
                up = min_jerk(min(1.0, max(0.0, (t - v["t_release"] - RELEASE_S - RELEASE_WAIT_S) / 0.6)))
                wrist = wrist + offset(z=RETRACT * up)
            return PrimitiveCommand(wrist, u)
        return PrimitiveCommand(plan.values["path"].at(t), u)

    @staticmethod
    def _decide_release(t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """At the end of the set-down: let go at ``t`` (its end), or not at all when the part (seen at
        its handle site, as the plan saw it) is more than :data:`~._common.RELEASE_HEIGHT_MAX_MM`
        above its support as planned; the predicate then fails the step and says why."""
        sch = schedule(ctx)
        v = sch.values
        bottom = float(ctx.observe_site(data, ctx.step.handle)[2]) - v["site_above_bottom"]
        high = 1000.0 * (bottom - v["support"])
        if high <= RELEASE_HEIGHT_MAX_MM:
            v["t_release"] = t
            sch.released = t
            return
        v["held_high_mm"] = high
        sch.released = math.nan


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
        if v.get("out_of_reach"):  # not attempted (out_of_reach_verdict): the object as it lies
            con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
            acc.update(ctx.object_state(model, data))
            acc["force_n"] = max(acc.get("force_n", 0.0), con.normal)
            acc["penetration_mm"] = max(acc.get("penetration_mm", 0.0), 1000.0 * con.depth)
            return
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
        if "held_high_mm" in v:  # not let go: how high, and what holds it up (the verdict's caveat)
            acc["held_high_mm"] = float(v["held_high_mm"])
            tgt = ctx.target
            if tgt is not None and "held_on" not in acc:
                if touching(model, data, ctx.obj.geom_ids, tgt.geom_ids):
                    acc["held_on"] = 1.0
                elif touching(model, data, tool_geoms(ctx.scene), tgt.geom_ids):
                    acc["held_on"] = 2.0

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
        far = out_of_reach_verdict(ctx, acc)
        if far is not None:
            return far
        q = {k: float(v) for k, v in acc.items() if k not in ("dropped", "held_on")}
        q.setdefault("hold_s", 0.0)
        q.setdefault("in_target", 0.0)
        out = criteria_verdict(ctx.step, q)
        if not out.success and out.failure_mode == "criterion_not_met" and acc.get("dropped"):
            return StepVerdict(False, "dropped", {**q, "dropped": 1.0})
        high = acc.get("held_high_mm")
        if high is not None and out.failure_mode not in ("physics_invalid", "physics_unreliable"):
            # the part could not be set down: the step did not do its job, whatever it measured
            # the support as planned (read only here: a refused release always has a schedule)
            where = str(schedule(ctx).values.get("support_on", "where it was to be set down"))
            held_on = {1.0: " (it rests on the bin's walls)", 2.0: " (the tool rests on the bin's walls)"}
            note = (
                f"The part stopped {high:.0f} mm above {where}{held_on.get(acc.get('held_on', 0.0), '')}; "
                "it was not let go, since releasing it would drop it"
            )
            failure = out.failure_mode or "criterion_not_met"
            return StepVerdict(False, failure, out.quantities, (*out.caveats, note))
        return out
