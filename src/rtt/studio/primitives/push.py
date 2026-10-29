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
     bar that runs up it. A lid open past vertical that has to close is **carried shut** (below)
     when the tool can hold it; otherwise it is pushed only to 80 deg, its weight closes it (it
     falls onto its stop: a push from outside cannot hold it back), and the tool then presses the
     closed lid home from above at the middle of its outer face (a second, downward push);
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
   past its path (``over``), and while the state is still unmet **on past that** by
   :data:`PUSH_ON_MAX` at most (Studio v2): the tool's own give takes travel too. The gripper's fist
   (closure 1 drives both tendons of the adaptive gripper to their limit, tips pressed together)
   has no lateral stiffness: after a pick that let go fully open it swings on contact with a door
   (a finger to its stop) and took 43 mm of the push, so the push used to let go 40 mm open. The
   push on goes on while the force is under the limit, the tool presses on nothing else harder than
   :data:`~._common.GUARD_FORCE_N` and the palm stays within its travel, within the segment's
   budget; at the step's speed while the observed object is still more than :data:`END_MM` from
   the segment's goal, then at the touch speed (it meets its stop at that speed). The fist still flexes under contact; only the travel it takes is made up.
   A ``force_n`` at_least is a goal of the last push on top of its travel: it
   presses on (within that reach, while the observed state still meets its criteria) until the
   force is reached and held 0.5 s, within the equal budget ``force_press`` plus that travel. A
   push whose observed state meets its criteria while it is held at its limit (a panel or a
   drawer shut against its stop) is done.
6. **Retract**: the wrist target comes back to where the wrist is (the push stops), hold 0.3 s,
   back off 30 mm along the face normal, rise 60 mm.

**Not tried** (Studio v2): a push whose tool, on the face where the push starts or where it ends,
would put the palm outside its travel (:func:`~._common.reach_problem`, +-350 mm less 5 mm) holds
still 0.2 s and fails ``out_of_reach`` with where the palm would have to go and a plain reason. The
free moves to its standoff are not checked: a palm held at the end of its travel short of the
standoff still meets the face from its side (a lid pushed over from past vertical: the transit
above the cup's standoff on the 110 deg S2e lid is 371 mm out, the standoff 343 mm, the face
318 mm). A button whose press cap (the press
force within the step's limit) is under its actuation force is not pressed either: it fails
``criterion_not_met`` at once with a plain reason ("The step's force limit allows 24 N; this button
needs at least 25 N to actuate, so it was not pressed"), instead of pressing for 8 s in vain. A cap
at or over the actuation force is tried, and the physics decides whether it latches: pressed from
the top, a desk button latches below the spring force at its latch point, and a press goes 0.3 to
0.5 N over its cap. A step that is not tried keeps its tool as it was (closed on a part the
previous step holds, :func:`~._common.idle_command`).

**Carrying a lid shut** (a lid open past vertical, a goal below 80 deg, no ``force_n`` at_least):
pushed over from outside, such a lid falls onto its stop (the frozen hero tote lid met it at about
340 deg/s). A two-finger gripper whose fingers fit the lid's free edge
(:func:`~.hook_pull.two_finger`, :func:`~.hook_pull.lip_pinch_fits`) instead pinches the free edge
from the front as :mod:`.hook_pull` pinches a lip (the wrist tilted, the tool pointing along the
lid at its hinge; the tips go in as far as a lip allows, or stop short of a bar handle,
:func:`edge_pinch`), carries the lid down its arc to its stop (:data:`CARRY_SPEED` at least),
guarded (it stops on anything but the lid and on the step's force limit) and lets go only there,
so the lid is never dropped from more than where it was held back. A tilted approach the palm
cannot reach (:func:`~._common.in_reach`), or any other tool, takes the push above.

**From the front** (``approach="front"`` in the step's params; default ``"top"``): a push on a
face that stands upright (its outward normal within :data:`FRONT_MAX_TILT` of horizontal: a
drawer's front, a door, a button on a wall) tilts the wrist so the tool points straight at the
face (pitch 90 deg, the pitch axis along the face) and pushes with the tool's tip on its axis, the
lowest point of its calibrated footprint; the yaw keeps the tool square to a face that turns (a
door). The transit turns the pitch at its height, the guarded move goes in horizontally along the
normal, and the retract backs off, rises until the tool hanging straight down would clear every
object (at least :data:`RETRACT`), then turns the pitch back within the pitch speed cap (the tip
swings down by the tool's length as it turns). A face the tilted palm cannot reach (it sits the
tool's length out, :func:`~._common.in_reach`) is pushed from the top. Faces pushed from above (a
lid, a part, a plug) ignore it. ``approach`` is an optional param of the push schema row,
omitted at its default so existing task ids are unchanged; the controller reads
``step.params.get("approach", "top")``.

**Predicate** (:class:`PushPredicate`): the template state at the end (``angle_deg``,
``travel_mm``, ``depth_mm``, ``lift_mm``/``tilt_deg``), ``force_n`` (the peak net force of the
tool on the object), ``penetration_mm``; then the shared rule. On a lid, sampled every control
tick (2 ms, every physics step: a falling lid crosses its last degrees in a few milliseconds),
``lid_release_deg`` (the angle a lid that fell shut fell from: where it last closed, faster than
:data:`LID_FREE_DPS`, with nothing of the tool on it; 0 when it never fell) and ``lid_stop_dps``
(the fastest it closed within :data:`LID_STOP_WINDOW_DEG` of its stop). A lid that met its stop
faster than :data:`LID_STOP_DPS_MAX` is not a failure (a lid with a catch snaps shut), but the
step says so in a plain caveat.
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
    PRESHAPE_CLEARANCE,
    Contact,
    Guarded,
    Path,
    Tool,
    box_corners,
    contacts,
    criterion_bounds,
    force_allowed,
    force_goal,
    geom_boxes,
    hold_out_of_reach,
    idle_command,
    idle_run,
    in_reach,
    move_time,
    observed_state,
    offset,
    out_of_reach_verdict,
    panel_normal,
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
    touching,
    transit,
    turn_time,
    wrist_load,
    wrist_target,
)
from rtt.studio.scene import wrist_rotation
from rtt.studio.types import SLIDE_KINDS, PrimitiveKind, Step, TemplateKind

__all__ = [
    "LID_STOP_DPS_MAX",
    "PUSH_ON_MAX",
    "PushController",
    "PushPredicate",
    "closed_inward",
    "edge_pinch",
    "lid_fall_caveat",
    "pinch_closes",
]

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
FRONT_PITCH = math.radians(90.0)  # a push from the front: the tool horizontal, at the face
FRONT_MAX_TILT = math.radians(30.0)  # a face whose normal is this close to horizontal stands upright
CARRY_SPEED = 0.060  # m/s: a lid carried shut moves along its arc at least this fast [assumed]
LID_FREE_DPS = 5.0  # deg/s: a lid closing faster than this with nothing of the tool on it is falling
LID_SHUT_DEG = 1.0  # deg: a falling lid this close to its stop has fallen shut
LID_STOP_WINDOW_DEG = 5.0  # deg: a lid's stop speed is the fastest it closes this close to its stop
LID_STOP_DPS_MAX = 30.0  # deg/s: a lid meeting its stop faster than this gets a caveat [assumed]
PAST_VERTICAL = math.radians(92.0)  # a lid open this far rests on its stop: closing it needs it carried
#: m a push to an observed-state criterion goes on past ``over`` while that state is unmet (the
#: tool's own give: the adaptive gripper's fist swung on contact and took 43 mm of a door's push
#: after a pick; this covers that with 40 % to spare, still short of the fist's 110 mm length)
#: [assumed]
PUSH_ON_MAX = 0.060
_DOWN = np.array([0.0, 0.0, -1.0])


def _panel_goal(info: Mapping[str, float], lo: float | None, hi: float | None) -> float:
    """The angle (rad) a push takes a panel to from the step's ``angle_deg`` bounds (deg)."""
    if hi is not None:
        return max(-ANGLE_PAST, math.radians(hi) - ANGLE_PAST)
    if lo is not None:
        return min(info["max_angle_rad"], math.radians(lo) + ANGLE_PAST)
    return 0.0


def pinch_closes(step: Step, info: Mapping[str, float]) -> bool:
    """Whether a push on this panel carries it shut when it starts open past vertical (module
    docstring): a lid (not a door) that can rest past vertical, a goal below the gravity hand-off
    and no ``force_n`` at_least (a press needs the push from above)."""
    angle = [c for c in step.success if c.quantity == "angle_deg"]
    lo = max((c.at_least for c in angle if c.at_least is not None), default=None)
    hi = min((c.at_most for c in angle if c.at_most is not None), default=None)
    press = any(c.quantity == "force_n" and c.at_least is not None for c in step.success)
    return (
        info["orientation"] == 0.0
        and info["max_angle_rad"] > PAST_VERTICAL
        and _panel_goal(info, lo, hi) < GRAVITY_HANDOFF
        and not press
    )


def _mid(info: Mapping[str, float]) -> np.ndarray:
    """The closed panel's centre on its mid-plane, from the hinge site."""
    lid0 = np.array([info["lid0_x"], info["lid0_y"], info["lid0_z"]])
    out0 = np.array([info["out0_x"], info["out0_y"], info["out0_z"]])
    return lid0 - 0.5 * info["thickness_m"] * out0


def closed_inward(info: Mapping[str, float]) -> np.ndarray:
    """Horizontal unit direction from a closed lid's free edge toward its hinge (the pinch's yaw)."""
    m = _mid(info)
    m[2] = 0.0
    return -m / float(np.linalg.norm(m))


def edge_pinch(
    info: Mapping[str, float], hinge: np.ndarray, handle: np.ndarray, angle: float
) -> tuple[np.ndarray, float]:
    """Where a lid open by ``angle`` (rad) is pinched to carry it shut, from its ``hinge`` and
    ``handle`` sites (world): the middle of its free edge at mid-thickness, and how far the finger
    tips go in from it (m): a lip's depth less the pinch's margin (:func:`~.hook_pull.pinch_depth`),
    or short of a bar handle standing off the face near the edge by that margin."""
    from rtt.studio.primitives.hook_pull import PINCH_TIP_MARGIN, PINCH_TIP_MAX, pinch_depth

    axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
    edge0 = 2.0 * _mid(info)  # the free edge's middle, closed, from the hinge
    edge = hinge + rotate(edge0, axis, angle)
    if info["feature"] == 0.0:
        return edge, pinch_depth(info["feature_m"])
    unit = edge0 / float(np.linalg.norm(edge0))
    bar_in = float(
        (edge0 - rotate(handle - hinge, axis, -angle)) @ unit
    )  # the bar's middle, in from the edge
    depth = float(np.clip(bar_in - info["bar_radius_m"] - PINCH_TIP_MARGIN, 0.004, PINCH_TIP_MAX))
    return edge, depth


LID_CLEARANCE = 0.002  # m kept between a carrying finger and whatever lies around the lid [assumed]
RELEASE_MARGIN = math.radians(
    2.0
)  # a carry that cannot reach the stop lets go this far above where it would touch


@dataclass(frozen=True)
class LidHold:
    """What a two-finger gripper holds a lid by to carry it shut (:func:`lid_hold`)."""

    kind: str  # "edge" (the free edge, pinched across the panel) or "bar" (a bar handle, pinched across it)
    point: np.ndarray  # world, at the lid's angle now: the free edge's middle, or the bar's centre
    depth: float  # m the finger tips go in past it along the tool axis
    grip: float  # m the fingers close on: the panel's thickness or the bar's diameter
    release: float  # rad: where the carry lets go (0: on its stop)


def lid_hold(
    info: Mapping[str, float],
    hinge: np.ndarray,
    handle: np.ndarray,
    angle: float,
    yaw: float,
    roll: float,
    sense: float,
    tip_drop: float,
    tool_point: np.ndarray,
    corners_at: Callable[[float], np.ndarray],
    obstacles: np.ndarray,
) -> LidHold | None:
    """How a two-finger gripper holds a lid open by ``angle`` (rad) to carry it shut, gripping what
    clears whatever lies under the lid (``obstacles``: ``(k, 6)`` world boxes, the box it covers
    among them); ``None`` when nothing can be carried any way down.

    In order: the free edge, its depth sized from what the edge overhangs (the deepest that carries
    the lid to its stop without a finger meeting an obstacle, :data:`LID_CLEARANCE`); a bar handle,
    when the fingers fit between the bar and the face (the inner finger rides above the face, so a
    lid flush with its box closes all the way); else the free edge carried down to just above
    where its finger would meet the box (:data:`RELEASE_MARGIN`), and let go there (the lid then
    falls the rest of the way: the predicate says so). ``corners_at(grip)``: the tool's ``(m, 8, 3)``
    geom corners pre-shaped to clear ``grip`` (the frame ``tool_point``, its TCP, is given in), and
    ``handle``: the handle site now. The controller and the search both use it."""
    from rtt.studio.primitives.hook_pull import PINCH_PITCH, PINCH_TIP_MARGIN, pinch_pitches

    axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
    lim = math.radians(90.0)

    def touch(point: np.ndarray, depth: float, grip: float) -> float:
        """The highest angle (rad) of the carry where the tool meets an obstacle (0: none)."""
        corners = corners_at(grip)
        pitch0 = pinch_pitches(sense, angle, 0.0, lim)[0]
        r0 = wrist_rotation(yaw, roll, pitch0)
        tcp0 = point + (depth - tip_drop) * (r0 @ _DOWN)
        lo_o, hi_o = obstacles[:, :3] - LID_CLEARANCE, obstacles[:, 3:] + LID_CLEARANCE
        n = max(2, math.ceil(math.degrees(angle)) + 1)
        for theta in np.linspace(angle, 0.0, n):
            pitch = max(-lim, min(lim, PINCH_PITCH + sense * float(theta)))
            r = wrist_rotation(yaw, roll, pitch)
            palm = hinge + rotate(tcp0 - hinge, axis, float(theta) - angle) - r @ tool_point
            world = palm + corners @ r.T  # (m, 8, 3)
            lo, hi = world.min(axis=1), world.max(axis=1)
            if not len(obstacles):
                return 0.0
            hit = np.all((lo[:, None, :] < hi_o[None]) & (hi[:, None, :] > lo_o[None]), axis=2)
            if np.any(hit):
                return float(theta)
        return 0.0

    edge, deepest = edge_pinch(info, hinge, handle, angle)
    t = float(info["thickness_m"])
    best: tuple[float, float] | None = None  # (touch angle, depth) of the edge pinch that goes lowest
    depth = deepest
    while depth >= 0.004 - 1e-9:
        a = touch(edge, depth, t)
        if a <= 0.0:
            return LidHold("edge", edge, depth, t, 0.0)
        if best is None or a < best[0] - 1e-9:
            best = (a, depth)
        depth -= 0.002
    if info["feature"] == 1.0 and _bar_fits(
        info, hinge, handle, axis, angle, sense, yaw, roll, tip_drop, tool_point, corners_at
    ):
        bar_depth = float(info["bar_radius_m"]) + PINCH_TIP_MARGIN
        grip = 2.0 * float(info["bar_radius_m"])
        if touch(handle, bar_depth, grip) <= 0.0:
            return LidHold("bar", handle.copy(), bar_depth, grip, 0.0)
    if best is None or best[0] + RELEASE_MARGIN >= angle:
        return None
    return LidHold("edge", edge, best[1], t, best[0] + RELEASE_MARGIN)


def _bar_fits(
    info: Mapping[str, float],
    hinge: np.ndarray,
    bar: np.ndarray,
    axis: np.ndarray,
    angle: float,
    sense: float,
    yaw: float,
    roll: float,
    tip_drop: float,
    tool_point: np.ndarray,
    corners_at: Callable[[float], np.ndarray],
) -> bool:
    """Whether the pre-shaped fingers pinch a lid's bar handle (its centre ``bar``, world, now)
    without the inner one meeting the panel (in the panel's frame, where a pinch carry does not
    move the tool)."""
    from rtt.studio.primitives.hook_pull import PINCH_TIP_MARGIN, pinch_pitches

    r_bar = float(info["bar_radius_m"])
    pitch0 = pinch_pitches(sense, angle, 0.0, math.radians(90.0))[0]
    r = wrist_rotation(yaw, roll, pitch0)
    m = rotate(_mid(info), axis, angle)  # the panel centre (mid-plane), from the hinge
    length = 2.0 * float(np.linalg.norm(m))
    u = m / float(np.linalg.norm(m))  # in the panel, from the hinge toward the free edge
    n = panel_normal_at(info, angle)  # the outer face's normal
    tcp = bar + (r_bar + PINCH_TIP_MARGIN - tip_drop) * (r @ _DOWN)
    palm = tcp - r @ tool_point
    world = palm + corners_at(2.0 * r_bar) @ r.T - hinge
    along, out = world @ u, world @ n
    across = world @ axis
    half_w = 0.5 * float(info["width_m"])
    lo_out, hi_out = out.min(axis=1), out.max(axis=1)
    lo_u, hi_u = along.min(axis=1), along.max(axis=1)
    lo_x, hi_x = across.min(axis=1), across.max(axis=1)
    slab = (lo_out < 0.5 * info["thickness_m"] + LID_CLEARANCE) & (hi_out > -0.5 * info["thickness_m"])
    over = (lo_u < length) & (hi_u > 0.0) & (lo_x < half_w) & (hi_x > -half_w)
    return not bool(np.any(slab & over))


def panel_normal_at(info: Mapping[str, float], angle: float) -> np.ndarray:
    """A lid's outer-face normal when open by ``angle`` (rad), from its template ``info``."""
    axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
    return rotate(np.array([info["out0_x"], info["out0_y"], info["out0_z"]]), axis, angle)


def lid_obstacles(scene: Any, lid: str, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """``(k, 6)`` world boxes of what the lid ``lid`` carried shut may meet: every other object's
    geoms (the box the lid covers among them) and the table, seen as the transit height is seen."""
    geoms = [o.geom_ids for name, o in scene.objects.items() if name != lid]
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if table >= 0:
        geoms.append(np.array([table]))
    if not geoms:
        return np.zeros((0, 6))
    return geom_boxes(model, data, np.concatenate(geoms))


@dataclass
class Segment:
    """One push: the push point ``point(u)`` and the face normal ``normal(u)`` for ``u`` in [0, 1],
    ``length`` m of travel, and an optional press force to reach and hold. ``reach``: the travel
    is a goal (a press on a part only has room to press into); ``met``: whether the observed
    object state meets the step's criteria on it (``None``: the travel alone is the goal);
    ``left``: how far (m, along the push) the observed object still is from the segment's goal,
    for a push to a criterion."""

    point: Callable[[float], np.ndarray]
    normal: Callable[[float], np.ndarray]
    length: float
    press: float | None = None
    reach: bool = True
    met: Callable[[mujoco.MjData], bool] | None = None
    pressing_on: bool = False  # a force goal asks for more than the travel or the press alone
    state: dict[str, Any] = field(default_factory=dict)
    left: Callable[[mujoco.MjData], float] | None = None


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
        goal = _panel_goal(info, *criterion_bounds(ctx, "angle_deg"))

        def arc(a: float, b: float, met: Callable[[mujoco.MjData], bool] | None) -> Segment:
            return Segment(
                point=lambda u: hinge + rotate(edge0, axis, a + u * (b - a)),
                normal=lambda u: panel_normal(ctx, a + u * (b - a)),
                length=abs(b - a) * radius,
                met=met,
                left=lambda d: abs(math.radians(observed_state(ctx, d)["angle_deg"]) - b) * radius,
            )

        lid = info["orientation"] == 0.0
        if lid and theta > PAST_VERTICAL and goal < GRAVITY_HANDOFF:
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
    if obj.kind in SLIDE_KINDS:  # a drawer, or a door sliding sideways
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
                left=lambda d: abs(observed_state(ctx, d)["travel_mm"] / 1000.0 - goal),
            )
        ]
    if obj.kind is TemplateKind.SPRING_BUTTON:
        # a button: its face pressed in along its axis, its travel and 2 mm more, until the press
        # force holds (on its stop, or short of it when the force runs out first); the press is at
        # least 1.5 x the button's actuation force (past its bottom-out at 1.3 x), within the limit
        n = np.array([info["normal_x"], info["normal_y"], info["normal_z"]])
        face = ctx.observe_site(data, "button")
        depth = info["travel_m"] + 0.002
        firm = min(max(press, 1.5 * info["force_n"]), force_allowed(ctx, PUSH_N))
        return [
            Segment(
                point=lambda u: face - u * depth * n,
                normal=lambda u: n,
                length=depth,
                press=firm,
                reach=False,
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
                left=lambda d: max(0.0, goal - observed_state(ctx, d)["depth_mm"] / 1000.0),
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


def upright(normal: np.ndarray) -> bool:
    """Whether a face of outward ``normal`` stands upright enough to be pushed from the front."""
    return abs(float(normal[2])) <= math.sin(FRONT_MAX_TILT)


def front_yaw(normal: np.ndarray, yaw_now: float) -> float:
    """The wrist yaw that points the tool, tilted :data:`FRONT_PITCH`, into a face of outward
    ``normal`` (``Rz(yaw) e_y`` along the normal's horizontal part, reversed), nearest ``yaw_now``."""
    yaw = math.atan2(float(normal[0]), -float(normal[1]))
    return yaw + 2.0 * math.pi * round((yaw_now - yaw) / (2.0 * math.pi))


def _front_point(tool: Tool) -> np.ndarray:
    """The tool point a push from the front meets the face with: on the tool's axis, at the lowest
    point of its calibrated footprint at the push's closure (wrist frame, pitch and roll 0)."""
    fp = tool.footprint(1.0 if tool.mechanism == "fingers" else 0.0)
    return np.array([tool.tcp[0], tool.tcp[1], min(float(tool.tip[2]), fp.tip_z)])


def _face_pose(
    seg: Segment, u: float, tool: Tool, front: bool, front_tip: np.ndarray, yaw: float, roll: float
) -> np.ndarray:
    """The wrist targets with the push's contact point on the face at ``u`` of its path (0: where
    it starts, 1: its goal, before ``over``), the tool tilted at the face (``front``) or straight
    down."""
    n = seg.normal(u)
    if front:
        return wrist_target(seg.point(u), front_tip, front_yaw(n, yaw), roll, FRONT_PITCH)
    return wrist_target(seg.point(u), _contact_point(tool, n, yaw + roll), yaw, roll)


def _may_push_on(
    ctx: StepContext,
    run: Mapping[str, Any],
    seg: Segment,
    con: Contact,
    step_m: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> bool:
    """Whether a push past its ``over`` (its observed state unmet) goes on by ``step_m`` this tick:
    the tool presses on nothing but its object harder than :data:`~._common.GUARD_FORCE_N` (``con``:
    its contacts with the object now), and the palm stays within its travel (the wrist moves along
    the push's last normal, from where it was last commanded)."""
    if wrist_load(ctx.scene, model, data).normal - con.normal > GUARD_FORCE_N:
        return False
    return in_reach(run["last"] + offset(*(-step_m * seg.normal(1.0))))


def _too_stiff(ctx: StepContext, cap: float) -> dict[str, float] | None:
    """A button the press cannot actuate: its press cap ``cap`` (N: the press force within the
    step's limit) under the button's actuation force, and that force; else ``None``. A cap at or
    over the actuation force is tried: whether it latches is the physics' (from the top a desk
    button latches below the spring force at its latch point, and a press goes 0.3 to 0.5 N over
    its cap)."""
    need = float(ctx.obj.handles.info["force_n"])
    if cap >= need:
        return None
    return {"press_cap_n": cap, "actuation_force_n": need}


def _hold_too_stiff(ctx: StepContext, start: np.ndarray, stiff: Mapping[str, float]) -> StepPlan:
    """The plan of a press the limit cannot actuate: hold still 0.2 s, the tool as it was; its
    predicate fails it with the reason (:func:`_too_stiff_verdict`)."""
    sch = schedule(ctx)
    sch.values.update(stiff)
    sch.values["too_stiff"] = 1.0
    note = (
        f"not pressed: the limit allows {_newtons(stiff['press_cap_n'])} N, the button actuates at "
        f"{_newtons(stiff['actuation_force_n'])} N"
    )
    return StepPlan(duration_s=0.2, values={"run": idle_run(ctx, start), "schedule": sch}, note=note)


def _newtons(f: float) -> str:
    """A force in N for a plain reason: whole newtons, else one decimal (24, 14.4)."""
    return f"{f:.1f}".removesuffix(".0")


def _too_stiff_verdict(ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict | None:
    """``criterion_not_met`` with a plain reason for a press planned by :func:`_hold_too_stiff`."""
    sch = scheduled(ctx)
    if sch is None or not sch.values.get("too_stiff"):
        return None
    v = sch.values
    cap, need = float(v["press_cap_n"]), float(v["actuation_force_n"])
    limited = ctx.step.constraints.max_force_n is not None or criterion_bounds(ctx, "force_n")[1] is not None
    who = "The step's force limit allows" if limited else "Without a force limit on the step, the push allows"
    reason = (
        f"{who} {_newtons(cap)} N; this button needs at least {_newtons(need)} N to actuate, so it "
        "was not pressed"
    )
    q = {**acc, "press_cap_n": cap, "actuation_force_n": need}
    return StepVerdict(False, "criterion_not_met", q, (reason,))


@dataclass(frozen=True)
class PushController:
    primitive: PrimitiveKind = PrimitiveKind.PUSH
    name: str = "studio_push"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        step = ctx.step
        tool = tool_of(ctx, ctx.obj.body_ids[-1])
        sch = schedule(ctx)
        sch.tool = tool
        start = start_pose(ctx)
        why: list[str] = []
        carried = self._plan_carry(ctx, model, data, tool, start, why)
        if carried is not None:
            return carried
        segments = _segments(ctx, data)
        yaw, roll = float(start[3]), float(start[4])
        front = step.params.get("approach", "top") == "front"
        front_tip = _front_point(tool)
        speed = float(step.params["speed_mm_s"]) / 1000.0
        limit = force_allowed(ctx, PUSH_N)
        if ctx.obj.kind is TemplateKind.SPRING_BUTTON:  # a press the limit cannot actuate is not tried
            stiff = _too_stiff(ctx, min(limit, float(segments[-1].press or 0.0)))
            if stiff is not None:
                return _hold_too_stiff(ctx, start, stiff)
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
            tilted = None
            if front and upright(n0):  # the tool tilted at the face, its tip on its axis meets it
                y0 = front_yaw(n0, yaw)
                tilted = wrist_target(seg.point(0.0) + STANDOFF * n0, front_tip, y0, roll, FRONT_PITCH)
                if not in_reach(tilted, tilted + offset(*(APPROACH * n0))):
                    tilted = None  # the palm would sit the tool's length out, past its travel
            if tilted is not None:
                seg.state["front"] = True
                pre = tilted
                ov = 0.0  # nothing of the tool reaches further toward the face than its tip
                above = pre + offset(*(APPROACH * n0))
            else:
                # the standoff is the whole tool's: whatever of it reaches further toward the face
                # than its contact point (a wrist flange, a long finger) stands off STANDOFF too
                ov = _overhang(tool, n0, yaw + roll)
                pre = wrist_target(
                    seg.point(0.0) + (STANDOFF + ov) * n0, _contact_point(tool, n0, yaw + roll), yaw, roll
                )
                above = pre + offset(*(APPROACH * n0))
                above[2] = max(above[2], pre[2] + APPROACH)
            # the tool on the face where the push starts and where it ends within the palm's travel,
            # or it is not tried (the free moves to its standoff may meet the travel's end: a tool held
            # there short of its standoff still meets the face from its side, a lid pushed over from
            # past vertical; a face the tool cannot reach, it never gets behind)
            front_here = tilted is not None
            far = reach_problem(
                *(_face_pose(seg, u, tool, front_here, front_tip, yaw, roll) for u in (0.0, 1.0))
            )
            if far is not None:
                return hold_out_of_reach(ctx, start, far, "the push")
            path = Path.start(here, ctx.scene.wrist.pitch_speed_dps)
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
        # the retract: a tool tilted at the face rises clear of everything before it turns back
        # straight down (its tip swings down by the tool's length as it turns)
        rise, rise_t, turn_t = RETRACT, 0.6, 0.0
        if segments[-1].state.get("front"):
            rise = max(RETRACT, safe - float(segments[-1].state["guard"].b[2]))
            rise_t = max(0.6, move_time(np.zeros(3), np.array([0.0, 0.0, rise])))
            turn_t = turn_time(FRONT_PITCH, ctx.scene.wrist.pitch_speed_dps)
        duration = min(30.0, t + 0.3 + 0.5 + rise_t + turn_t + 0.3)
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
            "rise": rise,
            "rise_t": rise_t,
            "turn_t": turn_t,
            "front_tip": front_tip,
        }
        return StepPlan(
            duration_s=duration,
            values={"run": run, "schedule": sch},
            note=f"push: {len(segments)} segment(s), {1000 * sum(s.length for s in segments):.0f} mm"
            + (f"; the lid is pushed over, not carried shut: {why[0]}" if why else ""),
        )

    def _plan_carry(
        self,
        ctx: StepContext,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tool: Tool,
        start: np.ndarray,
        why: list[str],
    ) -> StepPlan | None:
        """Carrying a lid open past vertical shut (module docstring); ``None`` when this push does
        not, with the reason appended to ``why`` when a carry was considered (the tool cannot hold
        the lid, nothing of it clears what lies under it, the palm would leave its travel, the carry
        would not fit a step)."""
        from rtt.studio.primitives.hook_pull import Motion, pinch_fits, pinch_yaw, plan_pinch

        obj, info = ctx.obj, ctx.obj.handles.info
        if obj.kind is not TemplateKind.HINGED_LID or not pinch_closes(ctx.step, info):
            return None
        angle = math.radians(observed_state(ctx, data)["angle_deg"])
        if angle <= PAST_VERTICAL:
            return None
        if not pinch_fits(ctx, tool):
            why.append("the tool cannot hold it (a two-finger gripper whose fingers fit the lid's edge can)")
            return None
        hinge = ctx.observe_site(data, "hinge")
        handle = ctx.observe_site(data, "handle")
        axis = np.array([info["axis_x"], info["axis_y"], info["axis_z"]])
        yaw, sense = pinch_yaw(closed_inward(info), axis, float(start[3]))
        roll = math.pi * round(float(start[4]) / math.pi)
        hold = lid_hold(
            info,
            hinge,
            handle,
            angle,
            yaw,
            roll,
            sense,
            tool.tip_drop,
            tool.tcp,
            lambda grip: box_corners(tool.footprint(tool.preshape(grip, PRESHAPE_CLEARANCE)).solids),
            lid_obstacles(ctx.scene, ctx.obj.name, model, data),
        )
        if hold is None:
            why.append("no hold of it clears what lies under the lid")
            return None
        speed = max(float(ctx.step.params["speed_mm_s"]) / 1000.0, CARRY_SPEED)
        plan = plan_pinch(
            ctx,
            model,
            data,
            tool,
            Motion(True, hinge, axis, angle, hold.release),
            hold.point,
            start,
            u_in=closed_inward(info),
            depth=hold.depth,
            speed=speed,
            limit=force_allowed(ctx, PUSH_N),
            grip=hold.grip,
            why=why,
        )
        if plan is None:
            return None
        what = "the free edge" if hold.kind == "edge" else "the bar handle"
        where = (
            "to its stop"
            if hold.release <= 0.0
            else f"down to {math.degrees(hold.release):.0f} deg, where a finger would meet what lies under it, "
            "and let it fall the rest"
        )
        return StepPlan(
            duration_s=plan.duration_s,
            values=plan.values,
            note=f"push: pinch {what} {1000 * hold.depth:.0f} mm deep and carry the lid {where}",
        )

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        sch = plan.values["schedule"]
        run = plan.values["run"]
        if (
            "idle" in run
        ):  # not tried (out of reach, or a press the limit cannot actuate): hold still, the tool as it was
            return idle_command(t, ctx, run, model, data)
        if run.get("mode") == "pinch":  # a lid carried shut
            from rtt.studio.primitives.hook_pull import command_pinch

            return command_pinch(t, ctx, run, model, data)
        tool: Tool = sch.tool
        if run["done_at"] is None:
            wrist = self._segment(t, ctx, run, model, data)
        else:
            d = t - run["done_at"]
            back = min_jerk(min(1.0, max(0.0, (d - 0.3) / 0.5)))
            up = min_jerk(min(1.0, max(0.0, (d - 0.8) / run["rise_t"])))
            shift = BACK_OFF * back * run["n_end"] + np.array([0.0, 0.0, run["rise"] * up])
            wrist = run["last"] + offset(*shift)
            if run["turn_t"]:  # a tool tilted at the face turns back straight down once clear
                turn = min_jerk(min(1.0, max(0.0, (d - 0.8 - run["rise_t"]) / run["turn_t"])))
                wrist[5] = float(run["last"][5]) * (1.0 - turn)
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
        con = contacts(model, data, tool_geoms(ctx.scene), ctx.obj.geom_ids)
        force = float(np.linalg.norm(con.net))
        stand = STANDOFF + st["ov"] + st["extra"]
        total = stand + seg.length  # the contact point travels the standoff, then the push
        # a push to a criterion goes on (up to ``over``) until the observed state meets it, and a
        # press until its force: the servo lags under the object's resistance
        cap = total + (run["over"] if (seg.press is not None or seg.met is not None) else 0.0)
        # past ``over`` a push whose observed state is still unmet goes on, by PUSH_ON_MAX at most
        # (the tool's own give: a fist that swings on contact), under the limit and in reach
        end = cap + PUSH_ON_MAX if seg.met is not None else cap
        ok = seg.met is None or seg.met(data)
        travelled = st.get("s", 0.0)
        arrived = travelled >= total - 1e-9 and (ok or travelled >= end - 1e-9)
        travel = seg.reach and not arrived
        stop_at = run["limit"] if (seg.press is None or travel) else min(run["limit"], seg.press)
        # it meets the face, and its object meets its stop, at the touch speed, so an impact stays
        # under the limit
        near = travelled < stand - st["ov"] + TOUCH_MM or travelled > total - END_MM
        if near and travelled >= cap and seg.left is not None and seg.left(data) > END_MM:
            near = False  # pushing on, its object (observed) still short of where it meets its stop
        speed = run["touch"] if near else run["speed"]
        if force < stop_at and (travel or ok):  # pressing on keeps the state inside its criteria
            if travelled < cap:
                st["s"] = min(cap, travelled + speed * dt)
            elif not ok and _may_push_on(ctx, run, seg, con, speed * dt, model, data):
                st["s"] = min(end, travelled + speed * dt)
        travelled = st.get("s", 0.0)
        arrived = travelled >= total - 1e-9 and (ok or travelled >= end - 1e-9)
        # the observed state meets its criteria and the push is held at its limit (a panel or a
        # drawer shut against its stop): nothing is left to push for
        arrived = arrived or (seg.met is not None and ok and force >= run["limit"])
        off = max(0.0, stand - travelled) - max(0.0, travelled - total)
        u = 1.0 if seg.length <= 1e-9 else min(1.0, max(0.0, travelled - stand) / seg.length)
        n = seg.normal(u)
        if st.get("front"):  # square to the face as it turns (a door), the tip on the tool's axis
            y = front_yaw(n, st.get("yaw", run["yaw"]))
            st["yaw"] = y
            wrist = wrist_target(
                seg.point(u) + off * n + st["shift"], run["front_tip"], y, run["roll"], FRONT_PITCH
            )
        else:
            heading = run["yaw"] + run["roll"]
            wrist = wrist_target(
                seg.point(u) + off * n + st["shift"],
                _contact_point(tool, n, heading),
                run["yaw"],
                run["roll"],
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


def _lid(ctx: StepContext) -> bool:
    return ctx.obj.kind is TemplateKind.HINGED_LID and ctx.obj.handles.info["orientation"] == 0.0


def lid_fall_caveat(release_deg: float, stop_dps: float) -> str:
    """The plain caveat of a lid that met its stop faster than :data:`LID_STOP_DPS_MAX`."""
    start = f"{release_deg:.0f}" if release_deg >= 10.0 else f"{release_deg:.1f}"
    return (
        f"The lid fell shut from {start} deg and met its stop at about {10 * round(stop_dps / 10):.0f} "
        "deg/s: the tool let go of it above its stop"
    )


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
        if _lid(ctx):
            acc.setdefault("lid_release_deg", 0.0)
            acc.setdefault("lid_stop_dps", 0.0)

    def observe_tick(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None:
        """Every control tick on a lid: ``lid_release_deg`` and ``lid_stop_dps`` (module docstring)."""
        if not _lid(ctx):
            return
        obj = ctx.obj
        angle = max(0.0, math.degrees(float(data.qpos[obj.joint_qpos["hinge"]])))
        closing = -math.degrees(float(data.qvel[obj.joint_dof["hinge"]]))
        v = schedule(ctx).values
        if angle <= LID_STOP_WINDOW_DEG:
            acc["lid_stop_dps"] = max(acc.get("lid_stop_dps", 0.0), closing)
        if closing > LID_FREE_DPS and not touching(model, data, tool_geoms(ctx.scene), obj.geom_ids):
            v.setdefault("lid_free_from", angle)  # where this unheld fall began
        else:
            v.pop("lid_free_from", None)
        start = v.get("lid_free_from")
        if start is not None and angle <= LID_SHUT_DEG:
            acc["lid_release_deg"] = max(acc.get("lid_release_deg", 0.0), float(start))

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict:
        far = out_of_reach_verdict(ctx, acc)
        if far is not None:
            return far
        stiff = _too_stiff_verdict(ctx, acc)
        if stiff is not None:
            return stiff
        out = criteria_verdict(ctx.step, acc)
        stop = float(acc.get("lid_stop_dps", 0.0))
        if stop > LID_STOP_DPS_MAX:
            note = lid_fall_caveat(float(acc.get("lid_release_deg", 0.0)), stop)
            return StepVerdict(out.success, out.failure_mode, out.quantities, (*out.caveats, note))
        return out


register_primitive(PushController(), PushPredicate())
