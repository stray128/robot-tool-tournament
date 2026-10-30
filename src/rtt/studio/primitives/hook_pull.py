"""The ``hook_pull`` primitive: engage a handle or lip and pull it along the object's arc or line.

Objects: ``hinged_lid`` (lid or door) and ``drawer``; handle ``handle`` (the template's feature).

**Controller** (:class:`HookPullController`):

1. **Engagement plan.** The engagement point is the observed ``handle`` site. The closing axis
   of the tool (its y) is set across the feature: across a horizontal bar or a lip (along the
   object's y), across the knob's neck (a door's vertical bar is gripped from the front only,
   below). The heading is set with the wrist yaw (the roll stays).
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
   closes it below vertical; one with an at_most alone is taken all the way shut (let go above
   its stop, a lid falls onto it). A ``force_n`` at_least is a goal on top of the motion's: the
   pull goes on to it (:func:`_pull_on`) before it lets go.
5. **Release and retract**: open over 0.5 s, draw back 40 mm off the face (less where the palm
   meets its travel), rise 60 mm, settle 0.3 s.

**A lid's lip** (a horizontal panel with nothing to grip) is pinched or hooked.

* **Pinch** (mode ``"pinch"``, wrist pitch): a two-finger gripper (:func:`two_finger`: a parallel
  or part-shaped jaw, a catalog gripper with two fingers, a designed hand with two) whose open
  fingers clear the lip by :data:`PINCH_CLEARANCE` a side and close tighter than it
  (:func:`lip_pinch_fits`) tilts its wrist:
  the yaw sets the pitch axis along the hinge, the pitch (:data:`PINCH_PITCH`, 90 deg) points the
  tool along the lid toward the hinge with its closing axis across the lid's thickness, the roll
  goes to the nearest half turn. Pre-shaped as closed as still clears the lip, it stands
  :data:`PINCH_STANDOFF` out from the free edge at the lip's height and moves in from the front,
  guarded, until the finger tips are the lip less :data:`PINCH_TIP_MARGIN` (at most
  :data:`PINCH_TIP_MAX`) under the overhang. It pinches the lip and carries it along the arc with
  the pitch at 90 deg less the lid's angle (no faster than the pitch cap allows, so a fast pull
  on a small lid slows down; a start within :data:`PITCH_CLAMP` past the range is clamped to
  it), so the grip turns with the lid; a pull on to a ``force_n`` goal goes on along the arc with
  the pitch held at the goal's, so the pitch never leaves the arc from the start to the goal (the
  arc the search's swept-volume check covers). While pinched, the
  wrist yields along the tool axis to the force the tool puts on the lid there
  (:data:`PINCH_COMPLY`, at most :data:`PINCH_COMPLY_MAX`; force feedback through the wrist
  command). At the goal it lets go, draws back off the lip along the tool axis
  (:data:`PINCH_RETRACT` past the free edge), rises and turns the pitch back to straight down, all
  within the pitch and roll speed caps. A carry that closes a lid (an at_most goal, or a push
  carrying a lid shut, :mod:`.push`) advances only while the tool meets nothing but the lid over
  :data:`~._common.GUARD_FORCE_N` and pushes the lid under the step's limit, lets go once it is
  shut or held back for :data:`CARRY_BLOCKED_S`, and rises clear of everything before it turns
  back (the tool then lies along the lid). Any pinch is taken only when the palm stays within its
  travel, less :data:`PINCH_REACH_MARGIN`, along its whole path (:func:`pinch_reach`: open past
  vertical or shut, the palm sits the tool's length beyond the free edge) and its plan fits a step;
  otherwise the hook, and the plan's note says why. A goal the pitch cannot follow within +-90 deg, or a
  tilted approach that would put the palm past its travel (:func:`~._common.in_reach`: the palm
  sits the tool's length out from the lip), falls back to the hook.
* **Hook** (mode ``"hook"``, any other tool, a hand with more fingers, a hook gripper and a
  tentacle included): the tool's sideways hook from its calibration
  (``Footprint.hook``: the geoms near the tip that reach furthest beyond the rest of the tool, a
  soft arm's horizontal arm) is turned toward the hinge, lowered outside the free edge (stood off
  :data:`HOOK_STANDOFF`, plus whatever of the tool reaches further than the hook), moved in under
  it, guarded, by the hook's depth less :data:`HOOK_MARGIN` (at most the lip less 2 mm and
  :data:`HOOK_MAX`), raised onto the underside, and carried along the arc with the fingers open. A
  tool without a hook reaches nothing under the lip and fails honestly.

**A door's bar** (``orientation`` door, ``feature`` handle: a vertical bar standing off the outer
face near the free edge, :func:`door_bar`) is gripped from the front only (mode ``"bar"``, wrist
pitch), by a two-finger gripper whose open fingers clear the bar by :data:`PINCH_CLEARANCE` a side
and close tighter than it (:func:`bar_fits`, the lip's rule on the bar). The pitch (:data:`PINCH_PITCH`, 90 deg) lays the
tool horizontal, the yaw points it at the door along the face normal (:func:`bar_yaw`, as
:func:`pinch_yaw` with the hinge vertical) and a quarter-turn roll lays its closing axis horizontal
along the face, across the bar (:func:`bar_roll`), at the bar's mid-height. The wrist rises to the
transit height and turns there (within the pitch and roll caps), descends with the tips
:data:`BAR_STANDOFF` out from the bar's front, pre-shapes as closed as still clears the bar and
moves in along the tool axis, guarded, until the tips are past the bar's front by its depth off the
face less :data:`PINCH_TIP_MARGIN` (:func:`bar_depth`: they stop that far short of the face; how
far the tips reach beyond the TCP as the fingers close is :func:`bar_drop`). It closes (1.0 s plus
0.3 s) and carries the bar along the door's arc to the pull's goal with the heading turning with
the door, so the tool stays normal to the face, no faster than the wrist's turn cap allows, and
yields along the tool axis as the lip pinch does. At the goal it opens to the pre-shape, draws back
along the tool axis until the tips are :data:`BAR_RETRACT` clear of the bar, and rises 60 mm while
the pitch turns back to straight down. The palm sits the tool's length out from the bar along the
face normal, so the front grip is made only when the grip, the whole pull (a pull on to a force
goal included) and the draw back keep it within its travel less :data:`PINCH_REACH_MARGIN`
(:func:`bar_reach`) and its plan fits a step. Nothing else engages a door's bar: a tool that does
not fit it fails the step ``criterion_not_met`` in 0.2 s with the reason ("Cannot grip the door's
bar from the front: ..."), and a front grip the palm cannot follow fails it ``out_of_reach`` ("the
front grip's pull would take the palm to ..."); neither step is attempted. The Task Engineer's
layout fixer moves such a door along x into the front grip's reach
(:func:`rtt.studio.engineer.layout.fix_door_bars`).

Every guarded move stops on contact with any object in the scene, not only the step's.

**Out of reach** (Studio v2): before it commits, a grip checks its engagement, its whole pull (every
2 deg of an arc) and a pull on to a ``force_n`` at_least goal against the palm's travel
(:func:`~._common.reach_problem`, +-350 mm less 5 mm), and a hook the same from under the lip
(less :data:`PINCH_REACH_MARGIN`, the pinch's, on the same lip). A door bar's front grip checks
its draw back too (:func:`bar_reach`). The retreat after the pull (the
grip's :data:`RETRACT_OUT` off the face, the hook's draw back out from under the lip) is never a
reason to refuse a step: it stops at the palm's travel less the same margin
(:func:`_retreat_within`), then the tool rises. The free move to a standoff is not checked: a palm
held at its travel's end short of it still meets the feature from its side. A tool without a hook lifts nothing,
so its pull on to a force goal runs along the lid's whole arc and past its stop: the hook of the
240 mm lid at y = 150 mm commanded the palm to 368.7 mm for 865 ticks. Such a step holds still
0.2 s and fails ``out_of_reach`` with where the palm would have to go and a plain reason.

**Predicate** (:class:`HookPullPredicate`): ``angle_deg`` or ``travel_mm`` (the template state
when the step ends, after the release), ``slip_mm`` (the largest displacement of the handle
relative to the tool, in the tool's frame (the panel's for a pinched or hooked lip or a bar gripped
from the front), from the end of the
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

from rtt.effectors.types import EffectorFamily, EffectorSpec
from rtt.sim.controller import Phase, min_jerk
from rtt.sim.scene import PALM_XY_LIMIT
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
    GUARD_FORCE_N,
    PRESHAPE_CLEARANCE,
    REACH_MARGIN,
    TURN_MARGIN,
    Guarded,
    Path,
    Tool,
    closing_low,
    contacts,
    criterion_bounds,
    force_allowed,
    force_goal,
    heading_for_axis,
    hold_out_of_reach,
    idle_command,
    idle_run,
    in_reach,
    move_time,
    observed_state,
    offset,
    out_of_reach_verdict,
    panel_angle,
    panel_normal,
    preshape_closure,
    press_reach,
    reach_problem,
    rot_z,
    rotate,
    safe_palm_z,
    schedule,
    scheduled,
    start_pose,
    tool_geoms,
    tool_of,
    tool_point_world,
    transit,
    turn_time,
    wrist_frame,
    wrist_load,
    wrist_target,
)
from rtt.studio.scene import wrist_rotation
from rtt.studio.types import MAX_STEP_S, SLIDE_KINDS, PrimitiveKind, TemplateKind

__all__ = [
    "BAR_RETRACT",
    "BAR_STANDOFF",
    "PINCH_REACH_MARGIN",
    "HookPullController",
    "HookPullPredicate",
    "Motion",
    "bar_depth",
    "bar_drop",
    "bar_far",
    "bar_fits",
    "bar_misfit",
    "bar_pinch_fits",
    "bar_points",
    "bar_reach",
    "bar_roll",
    "bar_yaw",
    "command_pinch",
    "door_bar",
    "lip_pinch_fits",
    "panel_goal",
    "pinch_depth",
    "pinch_fits",
    "pinch_pitches",
    "pinch_reach",
    "pinch_yaw",
    "plan_pinch",
    "pull_on_extent",
    "two_finger",
]

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
# a lid's lip pinched from the front with the wrist tilted (module docstring)
PINCH_PITCH = math.radians(90.0)  # the tool horizontal, its closing axis along the panel normal
PINCH_CLEARANCE = PRESHAPE_CLEARANCE  # m each side of the lip when the fingers are pre-shaped
PINCH_TIP_MARGIN = 0.004  # m: the finger tips stop this far short of the lip's full depth [assumed]
PINCH_TIP_MAX = 0.025  # m: the deepest the finger tips go in under a lip [assumed]
PINCH_STANDOFF = 0.020  # m: the tips stand this far out from the free edge before going in [assumed]
PINCH_RETRACT = 0.020  # m: drawn back off the lip past the free edge before rising [assumed]
#: m/(N s): while pinched, the wrist yields along the tool axis (in the lid's plane, toward the
#: hinge) to the force the tool puts on the lid there; a time constant of about 0.1 s against the
#: 2000 N/m xy servo [assumed]
PINCH_COMPLY = 0.005
PINCH_COMPLY_MAX = 0.010  # m: the most it yields [assumed]
PITCH_CLAMP = math.radians(2.0)  # a pinch's start this far past the pitch range is clamped to it
CARRY_BLOCKED_S = 0.3  # s: a lid carried shut that is held back this long is let go there [assumed]
PINCH_AIM_ERROR = 0.002  # m: how far off its observed sites a lid may be (the hidden placement error)
#: m kept inside the palm's travel on a pinch's path: the wrist's yield along the tool axis and the
#: aim error, so neither takes it past its travel
PINCH_REACH_MARGIN = PINCH_COMPLY_MAX + PINCH_AIM_ERROR
# a door's vertical bar gripped from the front with the wrist tilted (module docstring)
BAR_STANDOFF = 0.025  # m: the tips stand this far out from the bar's front before going in [assumed]
BAR_RETRACT = 0.030  # m: drawn back along the tool axis until the tips are this clear of the bar [assumed]


def _object_yaw(ctx: StepContext) -> float:
    return math.radians(ctx.task_spec.object(ctx.step.object).pose.yaw_deg)


@dataclass(frozen=True)
class Motion:
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


def _motion(ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> Motion:
    obj, info = ctx.obj, ctx.obj.handles.info
    state = observed_state(ctx, data)
    if obj.kind is TemplateKind.HINGED_LID:
        axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
        current = math.radians(state["angle_deg"])
        goal = panel_goal(info, *criterion_bounds(ctx, "angle_deg"))
        return Motion(True, ctx.observe_site(data, "hinge"), axis, current, goal)
    axis = np.array([info["pull_x"], info["pull_y"], 0.0])
    current = state["travel_mm"] / 1000.0
    lo, hi = criterion_bounds(ctx, "travel_mm")
    if lo is not None:
        goal = min(info["travel_m"] - 0.003, lo / 1000.0 + TRAVEL_MARGIN)
    elif hi is not None:
        goal = max(0.0, hi / 1000.0 - TRAVEL_MARGIN)
    else:
        goal = 0.8 * info["travel_m"]
    return Motion(False, ctx.observe_site(data, "front"), axis, current, goal)


def panel_goal(info: Mapping[str, float], lo: float | None, hi: float | None) -> float:
    """The angle (rad) a pull takes a lid or a door to, from the step's ``angle_deg`` bounds (deg):
    an at_least plus :data:`ANGLE_MARGIN` (a lid past vertical, :data:`LID_STAY_OPEN`, when its stop
    allows), an at_most less it (a lid all the way shut: let go above its stop, it would fall onto
    it), else 90 deg; within 2 deg of the stop."""
    top = info["max_angle_rad"] - math.radians(2.0)
    if lo is not None:
        goal = math.radians(lo) + ANGLE_MARGIN
        if info["orientation"] == 0.0 and info["max_angle_rad"] > LID_STAY_OPEN + math.radians(2.0):
            goal = max(goal, LID_STAY_OPEN)  # a lid stays open only past vertical
        return min(goal, top)
    if hi is not None:
        return 0.0 if info["orientation"] == 0.0 else max(0.0, math.radians(hi) - ANGLE_MARGIN)
    return min(top, math.radians(90.0))


def _pull_on(
    ctx: StepContext, motion: Motion, delta: float, radius: float, speed: float, max_rate: float = math.inf
) -> dict[str, Any]:
    """Pulling on to a ``force_n`` at_least, a goal on top of the motion's: once the pull has
    reached its goal, it goes on along the motion (toward the object's end, and into it by
    :func:`~._common.press_reach` of the pull limit, unless the step bounds the motion on both
    sides) while the measured force is under the pull limit, until the force has been over the
    goal for :data:`~._common.FORCE_HOLD_S`, within the equal budget ``force_press`` plus that
    travel. ``max_rate``: the most the motion may advance per second (a pinch: the pitch cap)."""
    goal = force_goal(ctx)
    if goal is None:
        return {"force_goal": None, "ext_t": 0.0}
    scale = radius if motion.hinged else 1.0
    sense = 1.0 if delta >= 0.0 else -1.0
    limit = force_allowed(ctx, PULL_N)
    cap = pull_on_extent(ctx, ctx.obj.handles.info, motion.hinged, motion.goal, delta, radius)
    rate = min(speed / scale, max_rate)
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


def pull_on_extent(
    view: Any, info: Mapping[str, float], hinged: bool, goal: float, delta: float, radius: float
) -> float:
    """How far past its goal a pull on to a ``force_n`` at_least may go (rad on a hinge, m on a
    slide; 0 without one, :func:`_pull_on`): to the object's end and into it by the pull limit
    over the servo's stiffness. ``view``: the step's context, or anything with its ``step`` and
    ``task`` (the search, which has no context)."""
    if force_goal(view) is None:
        return 0.0
    lo, hi = criterion_bounds(view, "angle_deg" if hinged else "travel_mm")
    if lo is not None and hi is not None:
        return 0.0
    sense = 1.0 if delta >= 0.0 else -1.0
    end = (info["max_angle_rad"] if hinged else info["travel_m"]) if sense > 0 else 0.0
    scale = radius if hinged else 1.0
    return max(0.0, sense * (end - goal)) + press_reach(view, force_allowed(view, PULL_N)) / scale


def pinch_reach(
    point: np.ndarray,
    t_dir: np.ndarray,
    depth: float,
    tip_drop: float,
    tool_point: np.ndarray,
    hinge: np.ndarray,
    axis: np.ndarray,
    yaw: float,
    roll: float,
    sense: float,
    pitch0: float,
    pitch_bounds: tuple[float, float],
    arc: float,
) -> str | None:
    """Why a lid pinch's palm would leave its travel (:func:`~._common.in_reach`), or ``None``.
    Checked: the standoff and the engaged pose at ``point`` (the free edge or a bar, world, at the
    lid's angle now), and with ``path`` every 2 deg of the carry by ``arc`` (rad about the hinge
    ``axis``, signed; a pull on past the goal included) with the pitch following the lid within
    ``pitch_bounds``, and the draw-back off the lid at the end, each within the palm's travel less
    :data:`PINCH_REACH_MARGIN`. ``tool_point``: the TCP in the wrist frame (pitch and roll 0),
    ``t_dir``: the tool axis at ``pitch0``. The controller and the search's tilted poses both call
    it, so they agree on which carries are taken (a lid opened past vertical, or carried shut,
    puts the palm the tool's length beyond the free edge)."""
    r0 = wrist_rotation(yaw, roll, pitch0)
    tcp0 = point + (depth - tip_drop) * t_dir
    palms = [point - (PINCH_STANDOFF + tip_drop) * t_dir - r0 @ tool_point, tcp0 - r0 @ tool_point]
    n = max(2, math.ceil(abs(math.degrees(arc)) / 2.0) + 1)
    r = r0
    for s in np.linspace(0.0, arc, n):
        pitch = min(pitch_bounds[1], max(pitch_bounds[0], pitch0 + sense * float(s)))
        r = wrist_rotation(yaw, roll, pitch)
        palms.append(hinge + rotate(tcp0 - hinge, axis, float(s)) - r @ tool_point)
    palms.append(palms[-1] + (depth + PINCH_RETRACT) * (r @ np.array([0.0, 0.0, 1.0])))
    if in_reach(*palms, margin=PINCH_REACH_MARGIN):
        return None
    worst = max(max(abs(float(p[0])), abs(float(p[1]))) for p in palms)
    return (
        f"the carry would take the palm to {1000 * worst:.0f} mm from the table centre, past its "
        f"{1000 * PALM_XY_LIMIT:.0f} mm travel less {1000 * PINCH_REACH_MARGIN:.0f} mm"
    )


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
    if obj.kind in SLIDE_KINDS:  # a drawer, or a sliding door (its fin handle is a lip)
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
    if ctx.obj.kind in SLIDE_KINDS:
        return ex if info["handle"] == 1.0 else ey  # knob: across the neck
    if info["orientation"] == 1.0:
        axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
        return rotate(ex if info["feature"] == 1.0 else ey, axis, angle)  # a vertical bar: along the face
    return ey


def _outward(ctx: StepContext, data: mujoco.MjData, handle: np.ndarray, motion: Motion) -> np.ndarray:
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


def _carried(motion: Motion, point: np.ndarray, s0: float, s1: float) -> list[tuple[float, np.ndarray]]:
    """Where ``point`` (world, attached to the object) goes as the motion progresses from ``s0`` to
    ``s1``: every 2 deg on a hinge (as :func:`pinch_reach` samples a carry), both ends on a slide."""
    n = max(2, math.ceil(abs(math.degrees(s1 - s0)) / 2.0) + 1) if motion.hinged else 2
    return [(float(s), motion.carry(point, float(s))) for s in np.linspace(s0, s1, n)]


def _out_of_reach(
    ctx: StepContext, start: np.ndarray, moves: tuple[tuple[str, list[np.ndarray]], ...], margin: float
) -> StepPlan | None:
    """The out-of-reach plan (:func:`~._common.hold_out_of_reach`) of the first of ``moves`` (what
    it is, in words, and its palm poses) that would take the palm outside its travel less
    ``margin``; ``None`` when the palm reaches them all."""
    for what, poses in moves:
        far = reach_problem(*poses, margin=margin)
        if far is not None:
            return hold_out_of_reach(ctx, start, far, what)
    return None


def _retreat_within(pose: np.ndarray, direction: np.ndarray, full: float, margin: float) -> float:
    """How far (m, at most ``full``) the palm draws back from ``pose`` along ``direction`` (world;
    its x and y count) and stays within its travel less ``margin``: ``full`` when it stays inside
    all the way, else where it meets that bound (0 when it is there already). A retreat is never a
    reason to refuse a step (module docstring)."""
    lim = PALM_XY_LIMIT - margin
    room = full
    for i in (0, 1):
        u, p = float(direction[i]), float(pose[i])
        if u > 1e-12:
            room = min(room, (lim - p) / u)
        elif u < -1e-12:
            room = min(room, (-lim - p) / u)
    return max(0.0, room)


@dataclass(frozen=True)
class HookPullController:
    primitive: PrimitiveKind = PrimitiveKind.HOOK_PULL
    name: str = "studio_hook_pull"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        tool = tool_of(ctx, ctx.obj.body_ids[-1])
        sch = schedule(ctx)
        sch.tool = tool
        motion = _motion(ctx, model, data)
        handle = ctx.observe_site(data, "handle")
        start = start_pose(ctx)
        info = ctx.obj.handles.info
        if motion.hinged and info["orientation"] == 0.0 and info["feature"] == 0.0:
            why: list[str] = []
            if pinch_fits(ctx, tool):
                plan = self._plan_pinch(ctx, model, data, tool, motion, handle, start, why=why)
                if plan is not None:
                    return plan
            hook = self._plan_hook(ctx, model, data, tool, motion, handle, start)
            if not why:
                return hook
            return StepPlan(hook.duration_s, hook.values, f"{hook.note}; not pinched: {why[0]}")
        if motion.hinged and door_bar(ctx):  # gripped from the front, or not at all
            misfit = bar_misfit(ctx, tool)
            if misfit is not None:
                return _hold_no_front_grip(ctx, start, misfit)
            why = []
            far: list[tuple[str, np.ndarray]] = []
            plan = self._plan_bar(ctx, model, data, tool, motion, handle, start, why, far)
            if plan is not None:
                return plan
            if far:
                what, pose = far[0]
                hit = reach_problem(pose, margin=PINCH_REACH_MARGIN)
                assert hit is not None
                return hold_out_of_reach(ctx, start, hit, f"the front grip's {what}")
            return _hold_no_front_grip(ctx, start, why[0])
        return self._plan_grip(ctx, model, data, tool, motion, handle, start)

    def _plan_grip(
        self,
        ctx: StepContext,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tool: Tool,
        motion: Motion,
        handle: np.ndarray,
        start: np.ndarray,
    ) -> StepPlan:
        """The tool straight down, closed across the feature (module docstring, steps 1 to 5)."""
        step = ctx.step
        sch = schedule(ctx)
        roll = float(start[4])
        heading = heading_for_axis(
            _closing_axis(ctx, motion.current), _yaw_offset(ctx, tool), float(start[3] + roll)
        )
        yaw = heading - roll
        pre = tool.preshape(ctx.obj.handles.info["grip_m"])
        engage = wrist_target(handle, tool.tcp, yaw, roll)
        out = _outward(ctx, data, handle, motion)
        path = Path.start(start, ctx.scene.wrist.pitch_speed_dps)
        lift = offset(z=0.08)
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
        # the grip, the pull and the pull on within the palm's travel (the free move to the standoff is
        # not checked: a palm held at its travel's end short of it still meets the handle from its
        # side); the retreat off the handle stops at the travel instead (_retreat_within)
        ext = on["ext_sense"] * on["ext_cap"] if on["force_goal"] is not None else 0.0
        point = engage[:3] + rot_z(yaw + roll) @ tool.tcp  # the TCP the pull carries

        def at(s: float, p: np.ndarray) -> np.ndarray:
            return wrist_target(
                p, tool.tcp, yaw + (s * float(motion.axis[2]) if motion.hinged else 0.0), roll
            )

        pulled = [at(u, p) for u, p in _carried(motion, point, 0.0, delta)]
        pulled_on = [at(u, p) for u, p in _carried(motion, point, delta, delta + ext)] if ext else []
        out_end = _face_out(ctx, data)
        if motion.hinged:
            out_end = rotate(out_end, motion.axis, delta + ext)
        moves = (
            ("the grip", [engage]),
            ("the pull", pulled),
            ("the pull on to the force goal", pulled_on),
        )
        refused = _out_of_reach(ctx, start, moves, REACH_MARGIN)
        if refused is not None:
            return refused
        retreat = _retreat_within((pulled_on or pulled)[-1], out_end, RETRACT_OUT, REACH_MARGIN)
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
            "retreat": retreat,  # m off the face after the release (RETRACT_OUT, or less at the travel)
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
        if "idle" in run:  # out of reach: hold still, the tool as it was
            return idle_command(t, ctx, run, model, data)
        v = sch.values
        if run.get("mode") == "hook":
            return self._command_hook(t, ctx, run, model, data)
        if run.get("mode") == "pinch":
            return self._command_pinch(t, ctx, run, model, data)
        if run.get("mode") == "bar":
            return self._command_bar(t, ctx, run, model, data)
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
            motion: Motion = run["motion"]
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
            shift = run["retreat"] * back * run["out"] + np.array([0.0, 0.0, RETRACT * up])
            wrist = last + offset(*shift)
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
        motion: Motion,
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
        path = Path.start(start, ctx.scene.wrist.pitch_speed_dps)
        above = pre + offset(z=0.08)
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
        # the hook keeps the pinch's margin under the lid, the pull on to a force goal included (a hook
        # that lifts nothing is carried along the lid's whole arc and on past its stop); the free move
        # to its standoff is not checked, and drawing it back out stops at the travel instead
        ext = on["ext_sense"] * on["ext_cap"] if on["force_goal"] is not None else 0.0
        point = engaged[:3] + rot_z(yaw + roll) @ contact  # the hook point the lift carries
        lifted = [wrist_target(p, contact, yaw, roll) for _, p in _carried(motion, point, 0.0, delta)]
        lifted_on = (
            [wrist_target(p, contact, yaw, roll) for _, p in _carried(motion, point, delta, delta + ext)]
            if ext
            else []
        )
        moves = (
            ("the hook", [engaged]),
            ("the hook's lift", lifted),
            ("the hook's pull on to the force goal", lifted_on),
        )
        refused = _out_of_reach(ctx, start, moves, PINCH_REACH_MARGIN)
        if refused is not None:
            return refused
        draw_back = _retreat_within((lifted_on or lifted)[-1], -u_in, insert + standoff, PINCH_REACH_MARGIN)
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
            "draw_back": draw_back,  # m back out from under the lid (insert + standoff, or less at the travel)
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
        lift = offset(z=HOOK_GAP + HOOK_PRELOAD)
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
            motion: Motion = run["motion"]
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
            shift = -run["draw_back"] * back * run["u_in"] + np.array([0.0, 0.0, RETRACT * up])
            wrist = last + offset(*shift)
        return PrimitiveCommand(wrist, tool.controls(t, 0.0, model, data, Phase.SETTLE))

    # ------------------------------------------------------------------ pinching a lid's lip, tilted

    def _plan_pinch(
        self,
        ctx: StepContext,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tool: Tool,
        motion: Motion,
        handle: np.ndarray,
        start: np.ndarray,
        *,
        u_in: np.ndarray | None = None,
        depth: float | None = None,
        speed: float | None = None,
        limit: float | None = None,
        grip: float | None = None,
        why: list[str] | None = None,
    ) -> StepPlan | None:
        """A two-finger tool tilts and pinches the lip from the front (module docstring); ``None``
        when the pitch cannot follow the lid to its goal within its range, the palm would leave its
        travel anywhere on the carry (:func:`pinch_reach`), or the plan would not fit a step
        (:data:`~rtt.studio.types.MAX_STEP_S`); the hook then, and the reason appended to ``why``.
        A push that closes a lid uses it too (:mod:`.push`): ``u_in`` (horizontal, from the closed
        lid's free edge toward its hinge), ``depth`` (how far the tips go in), ``speed`` (of the
        carry, m/s), ``limit`` (the force on the lid that stops a closing carry) and ``grip``
        (what the fingers close on, m: the lid's thickness, or a bar) are then its own."""
        step, info = ctx.step, ctx.obj.handles.info
        sch = schedule(ctx)
        wrist = ctx.scene.wrist
        why = [] if why is None else why
        yaw_now, roll_now, pitch_now = float(start[3]), float(start[4]), float(start[5])
        if u_in is None:
            u_in = -_outward(ctx, data, handle, motion)  # horizontal, from the free edge toward the hinge
        yaw, sense = pinch_yaw(u_in, motion.axis, yaw_now)
        roll = math.pi * round(roll_now / math.pi)  # the closing axis on the panel normal
        lim = math.radians(wrist.pitch_range_deg)
        pitch0, pitch_end = pinch_pitches(sense, motion.current, motion.goal, lim)
        if not (abs(pitch0) <= lim and abs(pitch_end) <= lim):
            why.append("the wrist's pitch cannot follow the lid over that arc")
            return None
        t_dir = wrist_rotation(yaw, roll, pitch0) @ np.array([0.0, 0.0, -1.0])  # the tool axis
        d_tip = pinch_depth(info["feature_m"]) if depth is None else float(depth)
        tcp = tool.tcp
        engaged = wrist_target(handle + (d_tip - tool.tip_drop) * t_dir, tcp, yaw, roll, pitch0)
        pre = wrist_target(handle - (PINCH_STANDOFF + tool.tip_drop) * t_dir, tcp, yaw, roll, pitch0)
        delta = motion.goal - motion.current
        radius = float(np.linalg.norm(np.cross(motion.axis, handle - motion.point)))
        if speed is None:
            speed = float(step.params["speed_mm_s"]) / 1000.0
        on = _pull_on(ctx, motion, delta, radius, speed, math.radians(wrist.pitch_speed_dps) / TURN_MARGIN)
        ext = on["ext_sense"] * on["ext_cap"] if on["force_goal"] is not None else 0.0
        # the pitch stays within the arc from the start to the goal, the arc the search checks: a
        # pull on to a force goal goes on along the lid's arc with the pitch held at the goal's
        lo_p, hi_p = sorted((max(-lim, min(lim, pitch0)), max(-lim, min(lim, pitch_end))))
        out = pinch_reach(
            handle, t_dir, d_tip, tool.tip_drop, tcp, motion.point, motion.axis, yaw, roll, sense, pitch0,
            (lo_p, hi_p), delta + ext,
        )  # fmt: skip
        if out is not None:
            why.append(out)  # the palm would sit the tool's length out from the lid, past its travel
            return None
        path = Path.start(start, ctx.scene.wrist.pitch_speed_dps)
        safe = safe_palm_z(ctx, model, data, tool)
        if start[2] < safe - 1e-4:
            up = start.copy()
            up[2] = safe
            path.to(up)
        above = pre.copy()
        above[2] = max(safe, pre[2])
        t_turn = max(
            turn_time(pitch0 - pitch_now, wrist.pitch_speed_dps),
            turn_time(roll - roll_now, wrist.roll_speed_max_deg_s),
        )
        path.to(above, max(0.4 + 0.3 * abs(yaw - yaw_now), t_turn, move_time(path.last, above)))
        path.to(pre, max(0.8, move_time(path.last, pre)))
        guard = Guarded(path.last, engaged, path.end, 1.0 + BUDGET_S["engage_slack"])
        pre_close = tool.preshape(info["thickness_m"] if grip is None else float(grip), PINCH_CLEARANCE)
        t_close = guard.end
        t_pull = t_close + 1.0 + 0.3
        # the pitch turns with the lid: the carry is no faster than the pitch cap allows
        pull_t = max(0.5, 1.875 * abs(delta) * radius / speed, turn_time(delta, wrist.pitch_speed_dps))
        t_release = t_pull + pull_t + on["ext_t"]
        t_back = max(
            0.6,
            turn_time(max(abs(lo_p), abs(hi_p)) if delta < 0.0 else abs(pitch_end), wrist.pitch_speed_dps),
        )
        closing = delta < 0.0
        rise = RETRACT
        rise_t = 0.0
        if closing:
            # a lid carried shut leaves the tool lying along it: it rises until the tool hanging
            # straight down clears every object before it turns back (its tip swings down by the
            # tool's length as it turns); the lid is lower once shut, so this bounds the rise
            goal_tcp = motion.carry(handle + (d_tip - tool.tip_drop) * t_dir, delta)
            palm_goal = wrist_target(goal_tcp, tcp, yaw, roll, pitch_end)
            rise = max(RETRACT, safe - float(palm_goal[2]))
            rise_t = max(0.6, move_time(np.zeros(3), np.array([0.0, 0.0, rise])))
        t_end = t_release + RELEASE_S + 0.5 + rise_t + t_back + 0.3 + BUDGET_S["hook_slack"]
        if t_end > MAX_STEP_S:
            why.append(f"the carry would take {t_end:.0f} s, past the {MAX_STEP_S:g} s a step may last")
            return None
        sch.engaged = t_pull
        sch.released = t_release
        sch.values.update(
            t_close=t_close, t_release=t_release, pre=pre_close, sense=math.copysign(1.0, delta)
        )
        sch.values["slip_point"] = tcp.tolist()  # type: ignore[assignment]
        run = {
            "mode": "pinch",
            "path": path,
            "guard": guard,
            "motion": motion,
            "delta": delta,
            "t_close": t_close,
            "t_pull": t_pull,
            "pull_t": pull_t,
            "t_release": t_release,
            "t_back": t_back,
            "yaw": yaw,
            "roll": roll,
            "pitch0": pitch0,
            "pitch_lim": lim,
            "pitch_lo": lo_p,
            "pitch_hi": hi_p,
            "pitch_dps": wrist.pitch_speed_dps,
            "sense": sense,
            "d_tip": d_tip,
            "engaged": None,
            "last": None,
            "ax_off": 0.0,
            "closing": closing,
            "carry_u": 0.0,
            "carry_t": math.nan,
            "blocked_s": 0.0,
            "limit": force_allowed(ctx, PULL_N) if limit is None else float(limit),
            "rise": rise,
            "rise_t": rise_t,
            **on,
        }
        return StepPlan(
            duration_s=t_end,
            values={"run": run, "schedule": sch},
            note=(
                f"hook pull: tilt {math.degrees(pitch0):.0f} deg and pinch the lip "
                f"{1000 * d_tip:.0f} mm deep, arc by {delta:.3f}"
            ),
        )

    def _command_pinch(
        self, t: float, ctx: StepContext, run: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        sch = schedule(ctx)
        tool: Tool = sch.tool
        v = sch.values
        guard: Guarded = run["guard"]
        if t < guard.t0:
            wrist = run["path"].at(t)
        elif t < run["t_pull"]:
            pressing = contacts(model, data, tool_geoms(ctx.scene), ctx.scene.ids.object_geoms).normal
            wrist = guard.at(min(t, guard.end), pressing) if t <= guard.end else guard.at(guard.end, 1e9)
        elif t < run["t_release"]:
            if run["engaged"] is None:
                run["engaged"] = guard.at(guard.end, 1e9)
            base = run["engaged"]
            motion: Motion = run["motion"]
            s = run["delta"] * min_jerk((t - run["t_pull"]) / run["pull_t"])
            if run["force_goal"] is not None:
                s += run["ext_sense"] * run["ext"]
                if _pull_on_step(t, ctx, run, model, data, run["t_pull"] + run["pull_t"]):
                    run["t_release"] = t  # every goal met: let go now
                    sch.released = t
                    v["t_release"] = t
            if run["closing"]:
                s = run["delta"] * min_jerk(self._closing_progress(t, ctx, run, model, data))
            point = base[:3] + wrist_rotation(run["yaw"], run["roll"], run["pitch0"]) @ tool.tcp
            v["progress"] = motion.current + s  # where the tool is taking the lid
            # the grip turns with the lid, within the arc from the start to the goal
            pitch = min(run["pitch_hi"], max(run["pitch_lo"], run["pitch0"] + run["sense"] * s))
            wrist = wrist_target(motion.carry(point, s), tool.tcp, run["yaw"], run["roll"], pitch)
            run["last"] = wrist
        else:  # open, draw back off the lip along the tool axis, then rise and turn back straight down
            last = run["last"] if run["last"] is not None else guard.at(guard.end, 1e9)
            if run.get("t_back_now") is None:  # the turn back from where the pitch is, within the cap
                run["t_back_now"] = max(0.6, turn_time(float(last[5]), run["pitch_dps"]))
            t_r = run["t_release"]
            back_dir = wrist_rotation(run["yaw"], run["roll"], float(last[5])) @ np.array([0.0, 0.0, 1.0])
            back = min_jerk(min(1.0, max(0.0, (t - t_r - RELEASE_S) / 0.5)))
            if run["closing"]:  # rise clear first, then turn back (the tip swings down as it turns)
                if run.get("rise_now") is None:
                    safe = safe_palm_z(ctx, model, data, tool)
                    run["rise_now"] = min(run["rise"], max(RETRACT, safe - float(last[2])))
                rise = run["rise_now"] * min_jerk(
                    min(1.0, max(0.0, (t - t_r - RELEASE_S - 0.5) / run["rise_t"]))
                )
                turn = min_jerk(
                    min(1.0, max(0.0, (t - t_r - RELEASE_S - 0.5 - run["rise_t"]) / run["t_back_now"]))
                )
            else:
                turn = min_jerk(min(1.0, max(0.0, (t - t_r - RELEASE_S - 0.5) / run["t_back_now"])))
                rise = RETRACT * turn
            wrist = last + offset(
                *((run["d_tip"] + PINCH_RETRACT) * back * back_dir + np.array([0.0, 0.0, rise]))
            )
            wrist[5] = float(last[5]) * (1.0 - turn)
        closure = v["pre"] + (1.0 - v["pre"]) * min(max((t - run["t_close"]) / 1.0, 0.0), 1.0)
        if t >= run["t_release"]:  # open back to the pre-shape only
            closure = v["pre"] + (1.0 - v["pre"]) * max(0.0, 1.0 - (t - run["t_release"]) / RELEASE_S)
        if run["t_close"] <= t < run["t_release"]:  # yield along the tool axis (force feedback)
            axis = wrist_rotation(run["yaw"], run["roll"], float(wrist[5])) @ np.array([0.0, 0.0, -1.0])
            push = float(contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids).net @ axis)
            dt = ctx.task.controller.control_period
            run["ax_off"] = float(
                np.clip(run["ax_off"] - PINCH_COMPLY * push * dt, -PINCH_COMPLY_MAX, PINCH_COMPLY_MAX)
            )
        if run["ax_off"] and t >= run["t_close"]:
            axis = wrist_rotation(run["yaw"], run["roll"], float(wrist[5])) @ np.array([0.0, 0.0, -1.0])
            wrist = wrist + offset(*(run["ax_off"] * axis))
        phase = Phase.HOLD if t >= run["t_close"] else Phase.DESCEND
        return PrimitiveCommand(wrist, tool.controls(t, closure, model, data, phase))

    @staticmethod
    def _closing_progress(
        t: float, ctx: StepContext, run: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
    ) -> float:
        """How far along its arc (0 to 1) a lid being carried shut is now: the carry advances on its
        clock while the tool meets nothing but the lid harder than :data:`~._common.GUARD_FORCE_N`
        and pushes the lid with less than the step's limit (force feedback through the wrist). A
        carry held back for :data:`CARRY_BLOCKED_S` (the lid on its stop, a finger on a rim) lets
        go there."""
        dt = t - run["carry_t"] if not math.isnan(run["carry_t"]) else 0.0
        run["carry_t"] = t
        everything = wrist_load(ctx.scene, model, data)
        lid = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
        others = max(0.0, everything.normal - lid.normal)
        if others <= GUARD_FORCE_N and float(np.linalg.norm(lid.net)) <= run["limit"]:
            run["carry_u"] = min(1.0, run["carry_u"] + dt / run["pull_t"])
            run["blocked_s"] = 0.0
        else:
            run["blocked_s"] += dt
        if run["carry_u"] >= 1.0 or run["blocked_s"] >= CARRY_BLOCKED_S:
            sch = schedule(ctx)
            run["t_release"] = t  # the lid is shut (or as far as it goes): let go now
            sch.released = t
            sch.values["t_release"] = t
        return float(run["carry_u"])

    # ------------------------------------------------------------------ a door's bar, from the front

    def _plan_bar(
        self,
        ctx: StepContext,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tool: Tool,
        motion: Motion,
        handle: np.ndarray,
        start: np.ndarray,
        why: list[str],
        far: list[tuple[str, np.ndarray]] | None = None,
    ) -> StepPlan | None:
        """A two-finger tool tilts and grips a door's vertical bar from the front (module docstring);
        ``None`` when the palm would leave its travel on the grip, the pull or the draw back
        (:func:`bar_far`: the move and the palm's furthest pose are appended to ``far``), or the plan
        would not fit a step (:data:`~rtt.studio.types.MAX_STEP_S`); the reason is appended to ``why``
        and the step is not attempted."""
        step, info = ctx.step, ctx.obj.handles.info
        sch = schedule(ctx)
        wrist = ctx.scene.wrist
        yaw_now, roll_now, pitch_now = float(start[3]), float(start[4]), float(start[5])
        n_out = panel_normal(ctx, motion.current)  # the door's outer face now (horizontal)
        yaw, roll = bar_yaw(n_out, yaw_now), bar_roll(roll_now)
        grip = float(info["grip_m"])
        pre_close = tool.preshape(grip, PINCH_CLEARANCE)
        drop = bar_drop(tool.footprint, float(tool.tcp[2]), tool.tip_drop, grip)
        d_tip = bar_depth(info)
        tcp = tool.tcp
        stand_tcp, engaged_tcp = bar_points(handle, n_out, float(info["bar_radius_m"]), d_tip, drop)
        stand = wrist_target(stand_tcp, tcp, yaw, roll, PINCH_PITCH)
        engaged = wrist_target(engaged_tcp, tcp, yaw, roll, PINCH_PITCH)
        delta = motion.goal - motion.current
        radius = float(np.linalg.norm(np.cross(motion.axis, handle - motion.point)))
        speed = float(step.params["speed_mm_s"]) / 1000.0
        # the heading turns with the door: the pull is no faster than the wrist's turn caps allow
        on = _pull_on(ctx, motion, delta, radius, speed, math.radians(wrist.pitch_speed_dps) / TURN_MARGIN)
        ext = on["ext_sense"] * on["ext_cap"] if on["force_goal"] is not None else 0.0
        out = bar_far(
            stand_tcp, engaged_tcp, n_out, d_tip, tcp, motion.point, motion.axis, yaw, roll, delta + ext
        )
        if out is not None:  # the palm sits the tool's length out from the bar, past its travel
            why.append(_reach_words(*out))
            if far is not None:
                far.append(out)
            return None
        path = Path.start(start, wrist.pitch_speed_dps)
        safe = safe_palm_z(ctx, model, data, tool)
        if start[2] < safe - 1e-4:
            up = start.copy()
            up[2] = safe
            path.to(up)
        above = stand.copy()
        above[2] = max(safe, stand[2])
        t_turn = max(
            turn_time(PINCH_PITCH - pitch_now, wrist.pitch_speed_dps),
            turn_time(roll - roll_now, wrist.roll_speed_max_deg_s),
        )
        path.to(above, max(0.4 + 0.3 * abs(yaw - yaw_now), t_turn, move_time(path.last, above)))
        path.to(stand, max(0.8, move_time(path.last, stand)))
        guard = Guarded(path.last, engaged, path.end, 1.0 + BUDGET_S["engage_slack"])
        t_close = guard.end
        t_pull = t_close + 1.0 + 0.3
        pull_t = max(0.5, 1.875 * abs(delta) * radius / speed, turn_time(delta, wrist.pitch_speed_dps))
        t_release = t_pull + pull_t + on["ext_t"]
        t_back = max(0.6, turn_time(PINCH_PITCH, wrist.pitch_speed_dps))
        t_end = t_release + RELEASE_S + 0.5 + t_back + 0.3 + BUDGET_S["hook_slack"]
        if t_end > MAX_STEP_S:
            why.append(f"the pull would take {t_end:.0f} s, past the {MAX_STEP_S:g} s a step may last")
            return None
        sch.engaged = t_pull
        sch.released = t_release
        sch.values.update(
            t_close=t_close, t_release=t_release, pre=pre_close, sense=math.copysign(1.0, delta)
        )
        sch.values["slip_point"] = tcp.tolist()  # type: ignore[assignment]
        run = {
            "mode": "bar",
            "path": path,
            "guard": guard,
            "motion": motion,
            "delta": delta,
            "t_close": t_close,
            "t_pull": t_pull,
            "pull_t": pull_t,
            "t_release": t_release,
            "yaw": yaw,
            "roll": roll,
            "pitch_dps": wrist.pitch_speed_dps,
            "stand": stand,
            "d_tip": d_tip,
            "engaged": None,
            "back": d_tip + BAR_RETRACT,  # m drawn back after the release (set once the grip is made)
            "last": None,
            "ax_off": 0.0,
            **on,
        }
        return StepPlan(
            duration_s=t_end,
            values={"run": run, "schedule": sch},
            note=(
                f"hook pull: tilt 90 deg and grip the bar from the front, the tips "
                f"{1000 * d_tip:.0f} mm past it, arc by {delta:.3f}"
            ),
        )

    def _command_bar(
        self, t: float, ctx: StepContext, run: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        sch = schedule(ctx)
        tool: Tool = sch.tool
        v = sch.values
        guard: Guarded = run["guard"]
        if t < guard.t0:
            wrist = run["path"].at(t)
        elif t < run["t_pull"]:
            pressing = contacts(model, data, tool_geoms(ctx.scene), ctx.scene.ids.object_geoms).normal
            wrist = guard.at(min(t, guard.end), pressing) if t <= guard.end else guard.at(guard.end, 1e9)
        elif t < run["t_release"]:
            if run["engaged"] is None:
                run["engaged"] = guard.at(guard.end, 1e9)
                # the tips went in past the bar's front by what the guarded move covered less the
                # standoff: drawn back by that and BAR_RETRACT, they end BAR_RETRACT clear of it
                went = float(np.linalg.norm(run["engaged"][:3] - run["stand"][:3]))
                run["back"] = max(0.0, went - BAR_STANDOFF) + BAR_RETRACT
            base = run["engaged"]
            motion: Motion = run["motion"]
            s = run["delta"] * min_jerk((t - run["t_pull"]) / run["pull_t"])
            if run["force_goal"] is not None:
                s += run["ext_sense"] * run["ext"]
                if _pull_on_step(t, ctx, run, model, data, run["t_pull"] + run["pull_t"]):
                    run["t_release"] = t  # every goal met: let go now
                    sch.released = t
                    v["t_release"] = t
            point = base[:3] + wrist_rotation(run["yaw"], run["roll"], PINCH_PITCH) @ tool.tcp
            v["progress"] = motion.current + s  # where the tool is taking the door
            heading = run["yaw"] + s * float(motion.axis[2])  # the tool turns with the door
            wrist = wrist_target(motion.carry(point, s), tool.tcp, heading, run["roll"], PINCH_PITCH)
            run["last"] = wrist
        else:  # open, draw back along the tool axis clear of the bar, rise as the pitch turns back
            last = run["last"] if run["last"] is not None else guard.at(guard.end, 1e9)
            if run.get("t_back_now") is None:  # the turn back from where the pitch is, within the cap
                run["t_back_now"] = max(0.6, turn_time(float(last[5]), run["pitch_dps"]))
            t_r = run["t_release"]
            back_dir = -_tool_axis(last)
            back = min_jerk(min(1.0, max(0.0, (t - t_r - RELEASE_S) / 0.5)))
            turn = min_jerk(min(1.0, max(0.0, (t - t_r - RELEASE_S - 0.5) / run["t_back_now"])))
            wrist = last + offset(*(run["back"] * back * back_dir + np.array([0.0, 0.0, RETRACT * turn])))
            wrist[5] = float(last[5]) * (1.0 - turn)
        closure = v["pre"] + (1.0 - v["pre"]) * min(max((t - run["t_close"]) / 1.0, 0.0), 1.0)
        if t >= run["t_release"]:  # open back to the pre-shape only
            closure = v["pre"] + (1.0 - v["pre"]) * max(0.0, 1.0 - (t - run["t_release"]) / RELEASE_S)
        if run["t_close"] <= t < run["t_release"]:  # yield along the tool axis (force feedback)
            push = float(
                contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids).net @ _tool_axis(wrist)
            )
            dt = ctx.task.controller.control_period
            run["ax_off"] = float(
                np.clip(run["ax_off"] - PINCH_COMPLY * push * dt, -PINCH_COMPLY_MAX, PINCH_COMPLY_MAX)
            )
        if run["ax_off"] and t >= run["t_close"]:
            wrist = wrist + offset(*(run["ax_off"] * _tool_axis(wrist)))
        phase = Phase.HOLD if t >= run["t_close"] else Phase.DESCEND
        return PrimitiveCommand(wrist, tool.controls(t, closure, model, data, phase))


def _tool_axis(wrist: np.ndarray) -> np.ndarray:
    """The tool axis (world, from the wrist toward the tip) of the six wrist targets ``wrist``."""
    return wrist_rotation(float(wrist[3]), float(wrist[4]), float(wrist[5])) @ np.array([0.0, 0.0, -1.0])


def plan_pinch(
    ctx: StepContext,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tool: Tool,
    motion: Motion,
    point: np.ndarray,
    start: np.ndarray,
    **kw: Any,
) -> StepPlan | None:
    """The lid pinch of this module (its docstring) as a plan of its own: a lid's ``motion`` from
    its angle now to its goal, pinched at ``point`` (world); :meth:`HookPullController._plan_pinch`
    has the keywords. A push carries a lid shut with it (:mod:`.push`); its plan's ``run`` is
    driven by :func:`command_pinch`."""
    return HookPullController()._plan_pinch(ctx, model, data, tool, motion, point, start, **kw)


def command_pinch(
    t: float, ctx: StepContext, run: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData
) -> PrimitiveCommand:
    """One control tick of a :func:`plan_pinch` plan (``plan.values["run"]``)."""
    return HookPullController()._command_pinch(t, ctx, run, model, data)


def lip_pinch_fits(open_aperture: float | None, shut_aperture: float | None, thickness: float) -> bool:
    """Whether a two-finger tool pinches a lid's lip ``thickness`` m thick: its open fingers clear
    the lip by :data:`PINCH_CLEARANCE` a side and they close tighter than it (apertures in m along
    the closing axis, ``None`` for a tool without a pair of opposed tips)."""
    return (
        open_aperture is not None
        and shut_aperture is not None
        and open_aperture >= thickness + 2.0 * PINCH_CLEARANCE
        and shut_aperture < thickness
    )


def pinch_yaw(u_in: np.ndarray, hinge_axis: np.ndarray, yaw_now: float) -> tuple[float, float]:
    """The wrist yaw of a lip pinch, the equivalent nearest ``yaw_now``: it sets the pitch axis (the
    palm's x) along the hinge, so at :data:`PINCH_PITCH` the tool points along the horizontal
    ``u_in`` (from the free edge toward the hinge); and the sense (+-1) of the pitch per radian of
    the lid's opening about ``hinge_axis``."""
    yaw = math.atan2(-float(u_in[0]), float(u_in[1]))  # Rz(yaw) e_y = u_in
    yaw += 2.0 * math.pi * round((yaw_now - yaw) / (2.0 * math.pi))
    sense = math.copysign(1.0, float(np.asarray(hinge_axis) @ rot_z(yaw)[:, 0]))
    return yaw, sense


def pinch_depth(lip: float) -> float:
    """How far (m) the finger tips go in under a lip ``lip`` m deep: the lip less
    :data:`PINCH_TIP_MARGIN`, within 4 mm and :data:`PINCH_TIP_MAX`."""
    return float(np.clip(lip - PINCH_TIP_MARGIN, 0.004, PINCH_TIP_MAX))


def pinch_pitches(sense: float, current: float, goal: float, lim: float = math.inf) -> tuple[float, float]:
    """The pitch of a lip pinch when the lid is at ``current`` and at ``goal`` (rad): 90 deg less
    the lid's angle (with the sense of :func:`pinch_yaw`). A start within :data:`PITCH_CLAMP` past
    the pitch range ``lim`` (a lid observed a hair below closed) is clamped to it."""
    pitch0 = PINCH_PITCH + sense * current
    if lim < abs(pitch0) <= lim + PITCH_CLAMP:
        pitch0 = math.copysign(lim, pitch0)
    return pitch0, PINCH_PITCH + sense * goal


def two_finger(spec: EffectorSpec) -> bool:
    """Whether a tool is a two-finger gripper, the tools that pinch a lid's lip: a parallel or a
    part-shaped jaw; a catalog gripper with two fingers (parallel, angular or adaptive: its
    ``n_fingers`` when it has one, else a ``gripper-2f-`` or ``gripper-angular-jaw`` model); a
    designed fingered hand with two fingers. Hands with more fingers, hook grippers, single-point
    tools, tentacles, soft arms and cups are not (they hook the lip)."""
    if spec.family is EffectorFamily.PARALLEL_JAW:
        return True
    if spec.family is EffectorFamily.CATALOG_GRIPPER:
        n = spec.params.get("n_fingers")
        if n is not None:
            return int(n) == 2
        return spec.model.startswith(("gripper-2f-", "gripper-angular-jaw"))
    if spec.family is EffectorFamily.DESIGNED and spec.recipe is not None:
        return spec.recipe.kind == "fingered_hand" and spec.recipe.params.get("n_fingers") == 2
    return False


def pinch_fits(ctx: StepContext, tool: Tool) -> bool:
    """:func:`lip_pinch_fits` for the step's tool (a :func:`two_finger` gripper only) and lid."""
    if not two_finger(ctx.head.spec):
        return False
    return lip_pinch_fits(
        tool.footprint(0.0).aperture(),
        tool.footprint(1.0).aperture(),
        float(ctx.obj.handles.info["thickness_m"]),
    )


# ---------------------------------------------------------------------- a door's bar, from the front


def bar_pinch_fits(open_aperture: float | None, shut_aperture: float | None, bar: float) -> bool:
    """Whether a two-finger tool grips a door's bar ``bar`` m across from the front: as a lid's lip
    (:func:`lip_pinch_fits`), its open fingers clear the bar by :data:`PINCH_CLEARANCE` a side and
    they close tighter than it (apertures in m along the closing axis, ``None`` for a tool without
    a pair of opposed tips)."""
    return lip_pinch_fits(open_aperture, shut_aperture, bar)


def door_bar(ctx: StepContext) -> bool:
    """Whether the step's object is a door with a bar handle (``orientation`` door, ``feature``
    handle): its bar is gripped from the front or not at all (module docstring)."""
    info = ctx.obj.handles.info
    return ctx.obj.kind is TemplateKind.HINGED_LID and info["orientation"] == 1.0 and info["feature"] == 1.0


def bar_misfit(ctx: StepContext, tool: Tool) -> str | None:
    """Why the step's tool cannot grip its door's bar from the front, in plain words, or ``None`` when
    it is a :func:`two_finger` gripper whose calibrated apertures fit the bar (:func:`bar_pinch_fits`)."""
    bar = float(ctx.obj.handles.info["grip_m"])
    if not two_finger(ctx.head.spec):
        return "only a two-finger gripper closes across a door's bar, and this tool is not one"
    open_ap, shut_ap = tool.footprint(0.0).aperture(), tool.footprint(1.0).aperture()
    if bar_pinch_fits(open_ap, shut_ap, bar):
        return None
    if open_ap is None or shut_ap is None:
        return "its calibration finds no pair of fingers either side of its axis to close across the bar"
    if open_ap < bar + 2.0 * PINCH_CLEARANCE:
        return (
            f"its fingers open {1000 * open_ap:.0f} mm, under the {1000 * bar:.0f} mm bar plus "
            f"{1000 * PINCH_CLEARANCE:.0f} mm a side"
        )
    return f"its fingers close to {1000 * shut_ap:.0f} mm, not under the {1000 * bar:.0f} mm bar"


def bar_fits(ctx: StepContext, tool: Tool) -> bool:
    """Whether the step grips a door's vertical bar from the front: its object is a door with a bar
    handle (:func:`door_bar`) and its tool fits the bar (:func:`bar_misfit` is ``None``)."""
    return door_bar(ctx) and bar_misfit(ctx, tool) is None


def _hold_no_front_grip(ctx: StepContext, start: np.ndarray, why: str) -> StepPlan:
    """The plan of a door bar the tool cannot grip from the front (``why``): hold still 0.2 s, the
    tool as it was (:func:`~._common.idle_run`); its predicate fails it with the reason."""
    sch = schedule(ctx)
    sch.engaged = sch.released = sch.reg = math.nan  # nothing is attempted
    sch.hold = (math.nan, math.nan)
    sch.values["no_front_grip"] = 1.0
    sch.values["front_grip_why"] = why  # type: ignore[assignment]
    note = f"not attempted: cannot grip the door's bar from the front: {why}"
    return StepPlan(duration_s=0.2, values={"run": idle_run(ctx, start), "schedule": sch}, note=note)


def _no_front_grip_verdict(ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict | None:
    """``criterion_not_met`` with a plain reason for a step planned by :func:`_hold_no_front_grip`."""
    sch = scheduled(ctx)
    if sch is None or not sch.values.get("no_front_grip"):
        return None
    reason = f"Cannot grip the door's bar from the front: {sch.values['front_grip_why']}; the step was not attempted"
    return StepVerdict(False, "criterion_not_met", dict(acc), (reason,))


def bar_yaw(n_out: np.ndarray, yaw_now: float) -> float:
    """The wrist yaw of a door bar's front grip, the equivalent nearest ``yaw_now``: at
    :data:`PINCH_PITCH` the tool points along ``-n_out``, at the door's outer face (``n_out``, its
    horizontal normal). It is :func:`pinch_yaw`'s with the hinge vertical."""
    return pinch_yaw(-np.asarray(n_out, dtype=np.float64), np.array([0.0, 0.0, 1.0]), yaw_now)[0]


def bar_roll(roll_now: float) -> float:
    """The wrist roll of a door bar's front grip, the equivalent nearest ``roll_now``: a quarter
    turn, so with the tool horizontal (:data:`PINCH_PITCH`) its closing axis (the tool's y) lies
    horizontal along the face, across the vertical bar."""
    return 0.5 * math.pi + math.pi * round((roll_now - 0.5 * math.pi) / math.pi)


def bar_depth(info: Mapping[str, float]) -> float:
    """How far (m) a door bar's front grip takes the finger tips past the bar's front: the bar's
    depth off the face (its clearance ``feature_m`` and its diameter) less :data:`PINCH_TIP_MARGIN`,
    so the tips stop that far short of the door's face."""
    return float(info["feature_m"]) + 2.0 * float(info["bar_radius_m"]) - PINCH_TIP_MARGIN


def bar_drop(footprint_at: Any, tcp_z: float, tip_drop: float, grip: float) -> float:
    """How far (m) a gripper's finger tips reach beyond its TCP (wrist z ``tcp_z``) along its axis as
    it closes on a bar ``grip`` m across from its pre-shape: the further of its axis tip
    (``tip_drop``) and the lowest point its calibrated footprints (``footprint_at(closure)``) reach
    from the pre-shape to closed (a linkage finger's tip swings out as it closes,
    :meth:`~._common.Tool.closing_tip`). The controller and the search's tilted poses both use it."""
    pre = preshape_closure(footprint_at, grip, PINCH_CLEARANCE)
    return tcp_z - min(tcp_z - tip_drop, closing_low(footprint_at, pre))


def bar_points(
    bar: np.ndarray, n_out: np.ndarray, radius: float, depth: float, drop: float
) -> tuple[np.ndarray, np.ndarray]:
    """The TCP (world) of a door bar's front grip standing off and engaged: the tool along
    ``-n_out`` through the bar's centre ``bar`` (world, at its mid-height), its tips ``drop`` m beyond
    the TCP; standing off, the tips are :data:`BAR_STANDOFF` out from the bar's front (``radius`` out
    from its centre), engaged ``depth`` past it (:func:`bar_depth`)."""
    n = np.asarray(n_out, dtype=np.float64)
    front = np.asarray(bar, dtype=np.float64) + radius * n
    return front + (BAR_STANDOFF + drop) * n, front - (depth - drop) * n


def bar_reach(
    stand: np.ndarray,
    engaged: np.ndarray,
    n_out: np.ndarray,
    depth: float,
    tool_point: np.ndarray,
    hinge: np.ndarray,
    axis: np.ndarray,
    yaw: float,
    roll: float,
    arc: float,
) -> str | None:
    """Why a door bar's front grip would take the palm out of its travel (:func:`~._common.in_reach`,
    less :data:`PINCH_REACH_MARGIN`), or ``None``. Checked: the grip (the TCP standing off at
    ``stand`` and engaged at ``engaged``, :func:`bar_points`, the tool along ``-n_out`` at ``yaw``,
    ``roll`` and :data:`PINCH_PITCH`), the pull every 2 deg of ``arc`` (rad about the hinge ``axis``,
    signed; a pull on included) with the heading turning with the door, and the draw back at its end
    (``depth`` plus :data:`BAR_RETRACT` along the face normal). ``tool_point``: the TCP in the wrist
    frame (pitch and roll 0). The palm sits the tool's length out from the bar along the face normal,
    so a door that swings its face toward a table edge takes it there. The controller and the search's
    tilted poses both call it, so they agree on which pulls are made from the front."""
    out = bar_far(stand, engaged, n_out, depth, tool_point, hinge, axis, yaw, roll, arc)
    return None if out is None else _reach_words(*out)


def bar_far(
    stand: np.ndarray,
    engaged: np.ndarray,
    n_out: np.ndarray,
    depth: float,
    tool_point: np.ndarray,
    hinge: np.ndarray,
    axis: np.ndarray,
    yaw: float,
    roll: float,
    arc: float,
) -> tuple[str, np.ndarray] | None:
    """:func:`bar_reach`'s check: the first move (``grip``, ``pull`` or ``draw back``) that would take
    the palm out of its travel, and the palm's furthest pose on it; ``None`` when it reaches them all."""
    up = np.asarray(axis, dtype=np.float64)

    def palm(tcp: np.ndarray, heading: float) -> np.ndarray:
        return tcp - wrist_rotation(heading, roll, PINCH_PITCH) @ tool_point

    n = max(2, math.ceil(abs(math.degrees(arc)) / 2.0) + 1)
    pulled = [
        palm(hinge + rotate(engaged - hinge, up, float(s)), yaw + float(s) * float(up[2]))
        for s in np.linspace(0.0, arc, n)
    ]
    back = pulled[-1] + (depth + BAR_RETRACT) * rotate(n_out, up, arc)
    moves = (("grip", [palm(stand, yaw), palm(engaged, yaw)]), ("pull", pulled), ("draw back", [back]))
    for what, poses in moves:
        if not in_reach(*poses, margin=PINCH_REACH_MARGIN):
            return what, max(poses, key=lambda p: max(abs(float(p[0])), abs(float(p[1]))))
    return None


def _reach_words(what: str, pose: np.ndarray) -> str:
    return (
        f"the {what} would take the palm to x {1000 * float(pose[0]):.0f} mm, y "
        f"{1000 * float(pose[1]):.0f} mm, past its {1000 * PALM_XY_LIMIT:.0f} mm travel less "
        f"{1000 * PINCH_REACH_MARGIN:.0f} mm"
    )


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
                rel = wrist_frame(ctx.scene, data).T @ (
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
        far = out_of_reach_verdict(ctx, acc)
        if far is not None:
            return far
        unfit = _no_front_grip_verdict(ctx, acc)
        if unfit is not None:
            return unfit
        out = criteria_verdict(ctx.step, acc)
        if (
            not out.success
            and out.failure_mode == "criterion_not_met"
            and acc.get("slip_mm", 0.0) > SLIP_FAIL_MM
        ):
            return StepVerdict(False, "slip", dict(acc))
        return out


register_primitive(HookPullController(), HookPullPredicate())
