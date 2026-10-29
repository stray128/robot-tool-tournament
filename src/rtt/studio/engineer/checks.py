"""Checks on top of :class:`~rtt.studio.types.TaskSpec`: is the plan runnable, not only well formed?

:class:`TaskSpec` checks structure and cross-references (names, kinds, handles, measured
quantities, order, holding). :func:`check_plan` adds what a run in the Studio's simulator needs:
the rules the prompt states (:func:`rtt.studio.engineer.catalog.render_rules`), on the
simulator's own conventions (track S2, ``docs/STUDIO_PRIMITIVES.md``):

* **layout** (:mod:`rtt.studio.engineer.layout`): footprints and swings 20 mm apart, a lid at
  its bin's position and the plug of a port excepted, so a picked part starts outside its bin;
* **windows**: a criterion with both bounds is at least :data:`WINDOW_MIN` wide;
* every step has a **goal** criterion (a state quantity of its object, ``lift_mm``, ``hold_s`` or
  ``in_target`` for a pick, or ``torque_nm`` at_least for a twist), not only limits;
* ``in_target`` needs a target, and a step that places its object releases it;
* **directions**: a twist opens a latch or a valve counter-clockwise and tightens a thread
  clockwise; the other way only turns back an object an earlier twist turned the working way; a
  twist stops at a ``torque_nm`` at_least, so that criterion only tightens a thread until snug;
* **threaded members** live on their stub: no pick acts on one, a twist on one comes after an
  insert of it, and an insert pushes it no deeper than the thread start (``chamfer_mm`` + 3 mm)
  and lets go (``hold_after`` false: the twist grips the member again);
* **reachable** criteria (a lid inside its hinge stop, a drawer inside its travel, a latch
  inside its turn, a valve inside its travel, a thread inside its engagement, a plug inside its
  socket, a pick inside its own lift and hold);
* **torque** a twist needs (the object's, a thread's on its ramp, a ``torque_nm`` at_least) within
  what the twist allows itself (90 % of ``max_torque_nm``, else 95 % of the wrist roll's) and any
  ``torque_nm`` at_most; **force**: a hook pull on a drawer within ``max_force_n``, a push on a
  drawer or an insert of a plug within 80 % of ``max_force_n`` (else 30 N);
* **refs**: uploads named exist and an uploaded part is used; library refs name training parts;
* a **refusal** lists what is not simulated with at least one nearest primitive;
* Studio v2's **job templates**: a panel's clamp is a threaded member (no pick, insert then twist
  cw, its fixed thread's reach and torque; clamp_torque_nm, not the wrist's torque_nm, says how
  tight it got) and a pick lifts the panel or a tile and sets it back (no target); a wall button
  is pushed from the front and a desk one from the top, judged by ``pressed`` (``travel_mm`` reads
  0 once it springs back), with a limit that lets the push press 1.5 x its actuation force; a press
  on a tile asks for the force that sinks it to its level (the bed's yield and hardening, the
  tile's weight, its spring-back) without sinking it past, within the push's limit; a sliding
  door's hook pull leaves ``slip_mm`` 40 mm, its forces are a drawer's, and a pick of something
  inside the machine (or into a fixture bin there) comes after a hook pull that opens the door far
  enough to uncover it, with no push shutting it in between. Their layout and reach rules live in
  :mod:`rtt.studio.engineer.layout`.

Problems carry a path (``objects[2] (tote): ...``, ``steps[1] (open_lid): ...``) so a repair
prompt and the intake page can point at the entry. A lid over a bin (the same position) needs the
bin to be as tall as the lid's fixed hinge height (120 mm), and a pick into that bin needs the lid
opened past vertical (``angle_deg`` at_least 95) by an earlier hook pull that no push undoes before
the pick. Warnings never refuse a plan: objects
nothing touches, a lid that falls shut (its stop at or below vertical).

:func:`nearest_primitive` maps free text (an unsupported action) to the closest primitive by
keyword; it backs a refusal whose nearest primitive the model left out.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Sequence
from functools import cache
from typing import Final

from rtt.studio.engineer.catalog import (
    BUTTON_PRESS_FACTOR,
    CLAMP_CHAMFER_MM,
    CLAMP_PITCH_MM,
    CLAMP_START_NM,
    CLAMP_TURNS,
    CLAMP_WRIST_SHARE,
    DEFAULT_PUSH_N,
    FORCE_GOAL_MARGIN,
    FORCE_SHARE,
    HOLD_LIFT_MIN_MM,
    HOLD_LIFT_SHARE,
    INSERT_PAST_MM,
    LID_HINGE_MM,
    LID_OPEN_DEG,
    NO_PICK,
    NO_PICK_HANDLES,
    PICK_FORCE_MIN_N,
    PICK_GIVE_MM,
    PICKS,
    PRESS_N,
    SET_BACK,
    SLIDE_PULL_FACTOR,
    SLIDE_PUSH_MAX_MM_S,
    SLIDE_SLIP_MIN_MM,
    THREAD_LEAD_EXTRA_MM,
    TILE_OVER_MM,
    TILE_REBOUND_MM,
    TORQUE_SHARE,
    TWIST_SENSE,
    goal_quantities,
    lift_give_mm,
    num,
    wrist_twist_limit,
)
from rtt.studio.engineer.layout import (
    CLEARANCE_MM,
    FINGER_ROOM_MM,
    MACHINE_OVERLAP_MM,
    inside_machine,
    layout_problems,
    regions,
)
from rtt.studio.types import (
    QUANTITIES,
    THREAD_KINDS,
    ObjectInstance,
    PrimitiveKind,
    Step,
    SuccessCriterion,
    TaskSpec,
    TemplateKind,
)

__all__ = [
    "DEPTH_TOLERANCE_MM",
    "RULES",
    "RULES_REVISION",
    "WINDOW_MIN",
    "WINDOW_MIN_V2",
    "check_parts",
    "check_plan",
    "claim_problems",
    "claimed_inside",
    "library_part_names",
    "nearest_primitive",
    "plan_problems",
]

#: Slack on a threaded member's seated depth before a criterion is unreachable (also in
#: ``web_catalog()["sim"]``).
DEPTH_TOLERANCE_MM: Final = 0.5

#: The narrowest window a criterion with both bounds may leave, per quantity (its unit) [assumed]:
#: measurements never land exactly on a value. The flags ``in_target`` and ``jam`` are exempt.
WINDOW_MIN: Final[dict[str, float]] = {
    "depth_mm": 1.0,
    "travel_mm": 1.0,
    "lift_mm": 1.0,
    "slip_mm": 1.0,
    "penetration_mm": 1.0,
    "angle_deg": 5.0,
    "tilt_deg": 5.0,
    "slip_deg": 5.0,
    "turns": 0.25,
    "torque_nm": 0.2,
    "force_n": 1.0,
    "lateral_force_n": 1.0,
    "hold_s": 0.2,
}

#: The rule sets: ``v1``, frozen with the v1 prompt (its recording replays with the very verdicts it
#: recorded), and ``v2``, Studio v2's (module docstring), the default.
RULES: Final[tuple[str, ...]] = ("v1", "v2")
#: Each rule set's revision. v1 is frozen. Rule set v2 was revised in place as Studio v2's heroes were
#: recorded: r1, the first v2 rules; r2, a pick's lift leaves the tool's give under its load, a hold
#: counts above the pick's threshold, a call cut off at max_tokens is never judged, and what the plan
#: puts inside a machine is placed and kept apart there; r3, a hook pull's force limit clears a sliding
#: door's pull and a part picked inside a machine keeps a gripper's finger room; r4, a claim binds only
#: what a clause names with the machine, and the repairs of prompt v2.3 name the field to change. A
#: recorded manifest names the revision it was verified with (``rules_revision``), and the test suite
#: re-judges every current recorded row: an edit that changes a recorded verdict or problem fails.
RULES_REVISION: Final[dict[str, str]] = {"v1": "v1", "v2": "v2-r4"}
#: Windows of the Studio v2 quantities (rule set v2) [assumed].
WINDOW_MIN_V2: Final[dict[str, float]] = {"clamp_torque_nm": 0.2, "gap_mm": 1.0, "height_err_mm": 0.5}
#: Quantities that only bound how a step went: a step never asks one to reach a value (rule set v2).
_LIMIT_ONLY: Final = ("penetration_mm", "slip_mm", "slip_deg", "jam", "lateral_force_n")
#: A set-back pick ends this close to where the object started (mm) [assumed].
SET_BACK_SLACK_MM: Final = 0.5

_TWIST_NAMES: Final = {
    TemplateKind.QUARTER_TURN_LATCH: "latch",
    TemplateKind.HANDWHEEL_VALVE: "valve",
    TemplateKind.THREADED_JOINT: "threaded member",
    TemplateKind.PANEL_RAIL: "panel clamp",
}

#: A panel clamp's thread, as a threaded joint's params (``rtt.studio.templates.panel_rail``).
_CLAMP_THREAD: Final = {
    "chamfer_mm": CLAMP_CHAMFER_MM,
    "engage_turns": CLAMP_TURNS,
    "pitch_mm": CLAMP_PITCH_MM,
    "start_torque_nm": CLAMP_START_NM,
}


def _thread(kind: TemplateKind, params: dict) -> dict:
    """A threaded member's numbers: a threaded joint's own params, a panel clamp's fixed thread."""
    if kind is TemplateKind.PANEL_RAIL:
        return {**_CLAMP_THREAD, "final_torque_nm": float(params["clamp_torque_nm"])}
    return params


def _where_step(i: int, s: Step) -> str:
    return f"steps[{i}] ({s.id})"


def _demand(c: SuccessCriterion) -> float | None:
    """How far a criterion asks the quantity to go from zero (a positive magnitude), or ``None``."""
    out = None
    if c.at_least is not None and c.at_least > 0:
        out = c.at_least
    if c.at_most is not None and c.at_most < 0:
        out = max(out or 0.0, -c.at_most)
    return out


def _lead(params: dict) -> float:
    """A thread's start below the stub top (mm): the chamfer plus the free lead-in."""
    return float(params["chamfer_mm"]) + THREAD_LEAD_EXTRA_MM


def _ceiling(
    kind: TemplateKind, params: dict, quantity: str, primitive: PrimitiveKind
) -> tuple[float, str] | None:
    """The largest value ``quantity`` can reach on an object of ``kind`` and what limits it."""
    p = params
    if kind is TemplateKind.HINGED_LID and quantity == "angle_deg":
        return float(p["max_angle_deg"]), "its hinge stop max_angle_deg"
    if kind is TemplateKind.DRAWER and quantity == "travel_mm":
        return float(p["travel_mm"]), "its travel_mm"
    if kind is TemplateKind.QUARTER_TURN_LATCH and quantity == "angle_deg":
        return float(p["turn_deg"]), "its turn_deg"
    if kind is TemplateKind.HANDWHEEL_VALVE and quantity == "angle_deg":
        return float(p["travel_deg"]), "its travel_deg"
    if kind is TemplateKind.THREADED_JOINT and quantity == "turns":
        return float(p["engage_turns"]), "its engage_turns"
    if kind is TemplateKind.THREADED_JOINT and quantity == "depth_mm":
        if primitive is PrimitiveKind.INSERT:
            return _lead(p), "the thread start, chamfer_mm + 3 mm; a twist runs it further"
        seated = _lead(p) + float(p["engage_turns"]) * float(p["pitch_mm"])
        return seated + DEPTH_TOLERANCE_MM, "seated, chamfer_mm + 3 mm + engage_turns x pitch_mm"
    if kind is TemplateKind.THREADED_JOINT and quantity == "angle_deg":
        return 360.0 * float(p["engage_turns"]), "its engage_turns x 360 deg"
    if kind is TemplateKind.PORT_PLUG and quantity == "depth_mm":
        return float(p["depth_mm"]), "the socket's depth_mm"
    if kind is TemplateKind.PANEL_RAIL:
        c = _thread(kind, p)
        if quantity == "turns":
            return CLAMP_TURNS, f"the clamp's {num(CLAMP_TURNS)} turns to seated"
        if quantity == "depth_mm":
            if primitive is PrimitiveKind.INSERT:
                return _lead(c), f"the clamp's thread start, {num(_lead(c))} mm; a twist runs it further"
            seated = _lead(c) + CLAMP_TURNS * CLAMP_PITCH_MM
            return seated + DEPTH_TOLERANCE_MM, f"the clamp seated, {num(seated)} mm"
        if quantity == "angle_deg":
            return 360.0 * CLAMP_TURNS, f"the clamp's {num(CLAMP_TURNS)} turns x 360 deg"
        if quantity == "clamp_torque_nm":
            return float(p["clamp_torque_nm"]), "its clamp_torque_nm, the torque that seats it"
        if quantity == "torque_nm" and primitive is PrimitiveKind.TWIST:
            return CLAMP_WRIST_SHARE * float(p["clamp_torque_nm"]), (
                "a seated clamp keeps turning at its clamp_torque_nm, so the wrist's torque stays just under it; "
                "judge the tightening by clamp_torque_nm at_least"
            )
    if kind in (TemplateKind.SLIDING_DOOR, TemplateKind.SPRING_BUTTON) and quantity == "travel_mm":
        return float(p["travel_mm"]), "its travel_mm"
    return None


def _torque_needed(kind: TemplateKind, params: dict, step: Step) -> tuple[float, str] | None:
    """The torque a twist step must deliver on ``kind`` (where it stops), and why."""
    if step.primitive is not PrimitiveKind.TWIST:
        return None
    if kind is TemplateKind.QUARTER_TURN_LATCH:
        return float(params["torque_nm"]), "the latch's torque_nm"
    if kind is TemplateKind.HANDWHEEL_VALVE:
        return float(params["torque_nm"]), "the valve's torque_nm"
    if kind in THREAD_KINDS:
        params = _thread(kind, params)
        engage = float(params["engage_turns"])
        seat = engage * float(params["pitch_mm"])
        share = 0.0
        for c in step.success:
            d = _demand(c)
            if d is None:
                continue
            if c.quantity == "turns":
                share = max(share, d / engage)
            elif c.quantity == "angle_deg":
                share = max(share, d / (360.0 * engage))
            elif c.quantity == "depth_mm":
                share = max(share, (d - _lead(params)) / seat)
        share = min(1.0, max(0.0, share))
        start, final = float(params["start_torque_nm"]), float(params["final_torque_nm"])
        torque = start + (final - start) * share
        if kind is TemplateKind.PANEL_RAIL:
            asked = [c.at_least for c in step.success if c.quantity == "clamp_torque_nm" and c.at_least]
            if asked and max(asked) > torque:
                return max(asked), "its clamp_torque_nm at_least criterion"
            return torque, (
                f"the clamp's torque there, {num(CLAMP_START_NM)} N m at its thread start rising to clamp_torque_nm"
            )
        return torque, "the thread's torque there, start_torque_nm rising to final_torque_nm"
    return None


def _force_needed(kind: TemplateKind, params: dict, step: Step) -> tuple[float, str, bool] | None:
    """(the force a step must apply, why, whether the controller keeps to a share of the limit)."""
    if kind is TemplateKind.DRAWER and step.primitive is PrimitiveKind.HOOK_PULL:
        return float(params["pull_force_n"]), "the drawer's pull_force_n", False
    if kind is TemplateKind.DRAWER and step.primitive is PrimitiveKind.PUSH:
        return float(params["pull_force_n"]), "the drawer's pull_force_n", True
    if kind is TemplateKind.PORT_PLUG and step.primitive is PrimitiveKind.INSERT:
        return float(params["insert_force_n"]), "the plug's insert_force_n", True
    if kind is TemplateKind.SLIDING_DOOR and step.primitive is PrimitiveKind.HOOK_PULL:
        return float(params["pull_force_n"]), "the door's pull_force_n", False
    if kind is TemplateKind.SLIDING_DOOR and step.primitive is PrimitiveKind.PUSH:
        return float(params["pull_force_n"]), "the door's pull_force_n", True
    return None


def _cents_up(value: float) -> float:
    """``value`` rounded up to the next hundredth (a limit that still allows what it must)."""
    return math.ceil(round(value * 100.0, 6)) / 100.0


def _push_allowed(step: Step) -> float:
    """The most a push presses with (N): 80 % of ``max_force_n``, else 30 N, and 80 % of a
    ``force_n`` at_most (``rtt.studio.primitives._common.force_allowed``)."""
    cap = step.constraints.max_force_n
    out = FORCE_SHARE * cap if cap is not None else DEFAULT_PUSH_N
    for c in step.success:
        if c.quantity == "force_n" and c.at_most is not None:
            out = min(out, FORCE_SHARE * c.at_most)
    return out


def _allowed_text(step: Step) -> str:
    cap = step.constraints.max_force_n
    caps = [c.at_most for c in step.success if c.quantity == "force_n" and c.at_most is not None]
    if caps and (cap is None or min(caps) < cap):
        return f"{num(_push_allowed(step))} N ({num(100 * FORCE_SHARE)} % of its force_n at_most {num(min(caps))})"
    if cap is not None:
        return f"{num(_push_allowed(step))} N ({num(100 * FORCE_SHARE)} % of max_force_n {num(cap)})"
    return f"{num(DEFAULT_PUSH_N)} N (no max_force_n)"


def _bound(step: Step, quantity: str, which: str) -> float | None:
    vals = [
        getattr(c, which) for c in step.success if c.quantity == quantity and getattr(c, which) is not None
    ]
    if not vals:
        return None
    return max(vals) if which == "at_least" else min(vals)


def _criterion_path(where: str, s: Step, quantity: str, at_least: float) -> str:
    """``steps[i].success[j]``: the criterion of ``s`` on ``quantity`` whose at_least is ``at_least``."""
    j = next(k for k, c in enumerate(s.success) if c.quantity == quantity and c.at_least == at_least)
    return f"{where.split(' ', 1)[0]}.success[{j}]"


def _v2_step_problems(where: str, obj: ObjectInstance, s: Step, at: str, paths: bool = True) -> list[str]:
    """Rule set v2's rules on one step: limits are never goals, criteria reachable from the object's
    start, and the job templates' rules (a panel clamp, a button, a tile, a door)."""
    out: list[str] = []
    kind, p = obj.kind, dict(obj.params)
    if s.primitive in PICKS:
        cap = s.constraints.max_force_n
        if cap is not None and cap < PICK_FORCE_MIN_N - 1e-9:
            out.append(
                f"{where}: max_force_n {num(cap)} on a pick stops most tools: they grip or seal a part with 30 to 92 N "
                "of their own, whatever it weighs; leave it null unless the use case limits the squeeze, then at "
                f"least {num(PICK_FORCE_MIN_N)}"
            )
        lift = _bound(s, "lift_mm", "at_least")
        own = float(s.params.get("lift_mm", 0.0))
        mass = float(p.get("mass_g", 0.0) or 0.0)
        give = lift_give_mm(mass)
        heavy = give > PICK_GIVE_MM + 1e-9
        if lift is not None and lift > own - give + 1e-9 and lift <= own + 1e-9:
            if not heavy and not paths:  # the words recorded repairs answered (prompts before v2.3)
                out.append(
                    f"{where}: lift_mm at_least {num(lift)} leaves under {num(PICK_GIVE_MM)} mm of the step's lift_mm "
                    f"({num(own)}) for the tool's give (a vacuum frame gives about 8.5 mm before the part rises); "
                    f"raise lift_mm to at least {num(lift + PICK_GIVE_MM)} or lower the criterion"
                )
            elif not heavy:
                path = where.split(" ", 1)[0]
                out.append(
                    f"{where}: its criterion lift_mm at_least {num(lift)} leaves under {num(PICK_GIVE_MM)} mm of "
                    f"{path}.params.lift_mm ({num(own)}, how far the wrist rises) for the tool's give (a vacuum frame "
                    f"gives about 8.5 mm before the part rises); raise {path}.params.lift_mm to at least "
                    f"{num(lift + PICK_GIVE_MM)} and keep the criterion at {num(lift)}, or lower the criterion"
                )
            else:
                path = where.split(" ", 1)[0]
                out.append(
                    f"{where}: its criterion lift_mm at_least {num(lift)} leaves under {num(give)} mm of "
                    f"{path}.params.lift_mm ({num(own)}, how far the wrist rises) for the tool's give under "
                    f"{obj.name}'s {num(mass / 1000.0)} kg (a vacuum frame gives 3 mm plus 2.5 mm per kg before the "
                    f"part rises, 23 mm under 8 kg, and 1.5 mm is kept spare); raise {path}.params.lift_mm to at least "
                    f"{num(lift + give)} and keep the criterion at {num(lift)} (raising both leaves the same shortfall)"
                )
        else:
            hold = _bound(s, "hold_s", "at_least")
            need = max(HOLD_LIFT_MIN_MM + give, give / (1.0 - HOLD_LIFT_SHARE))
            if hold is not None and hold > 0 and own < need - 1e-9:
                up = max(HOLD_LIFT_MIN_MM, HOLD_LIFT_SHARE * own)
                path = where.split(" ", 1)[0]
                out.append(
                    f"{where}: hold_s at_least {num(hold)} counts only while the part is up at least {num(up)} mm "
                    f"({num(HOLD_LIFT_MIN_MM)} mm or {num(100 * HOLD_LIFT_SHARE)} % of {path}.params.lift_mm, whichever "
                    f"is more), and the tool's give takes up to {num(give)} mm of {path}.params.lift_mm ({num(own)}) "
                    f"before {obj.name} rises; raise {path}.params.lift_mm to at least {num(need)}"
                )
    for c in s.success:
        if c.quantity in _LIMIT_ONLY and c.at_least is not None and c.at_least > 0:
            out.append(
                f"{where}: {c.quantity} at_least {num(c.at_least)} asks the step to reach a limit; {c.quantity} only "
                "bounds how a step went (physics validity, slip or jam), so give it at_most or leave it out"
            )
    if kind is TemplateKind.PANEL_RAIL:
        start = float(p["start_gap_mm"])
        for j, c in enumerate(s.success):
            if c.quantity != "gap_mm":
                continue
            gap = f"{at}.params.start_gap_mm" if paths else "start_gap_mm"
            bound = f"{where.split(' ', 1)[0]}.success[{j}]"
            if c.at_most is not None and c.at_most < start - SET_BACK_SLACK_MM - 1e-9:
                out.append(
                    f"{where}: gap_mm at_most {num(c.at_most)} cannot be met: the panel starts start_gap_mm "
                    f"({num(start)} mm) from the stops and a pick sets it back where it was; set {gap} to "
                    f"the gap the use case allows, at most {num(c.at_most)}, or raise the bound"
                    + (f" ({bound}.at_most)" if paths else "")
                )
            if c.at_least is not None and c.at_least > start + SET_BACK_SLACK_MM + 1e-9:
                out.append(
                    f"{where}: gap_mm at_least {num(c.at_least)} cannot be met: the panel stays start_gap_mm "
                    f"({num(start)} mm) from the stops"
                    + (f"; lower {bound}.at_least to at most {num(start)} or raise {gap}" if paths else "")
                )
    if kind is TemplateKind.TILE_BED and s.primitive in PICKS:
        proud = float(p["proud_mm"])
        for c in s.success:
            if (
                c.quantity == "height_err_mm"
                and c.at_most is not None
                and c.at_most < proud - SET_BACK_SLACK_MM
            ):
                out.append(
                    f"{where}: height_err_mm at_most {num(c.at_most)} cannot be met by a pick: it lays the tile back "
                    f"{num(proud)} mm proud (proud_mm); a push presses it to its level"
                )
    if s.primitive in PICKS and kind in SET_BACK and s.target is not None:
        what = "panel on its rails" if kind is TemplateKind.PANEL_RAIL else "tile on its bed"
        out.append(
            f"{where}: a pick on a {what} lifts it and sets it back where it was; target null (bringing it "
            "from elsewhere is not modelled)"
        )
    if s.primitive in PICKS and kind in SET_BACK:
        lifted = any(c.quantity in ("lift_mm", "hold_s") and (c.at_least or 0.0) > 0.0 for c in s.success)
        if not lifted:
            what = "panel" if kind is TemplateKind.PANEL_RAIL else "tile"
            out.append(
                f"{where}: a pick that sets the {what} back must say it lifted it: add lift_mm at_least (more than "
                "0, within the step's lift_mm) or hold_s at_least; without one the step passes when nothing lifts it"
            )
    if s.primitive in PICKS and kind is TemplateKind.PANEL_RAIL and s.handle in NO_PICK_HANDLES:
        out.append(
            f"{where}: the panel's clamp lives on its post and cannot be picked up; a pick lifts the panel "
            "(handle panel), insert and twist work the clamp"
        )
    if kind is TemplateKind.SPRING_BUTTON and s.primitive is PrimitiveKind.PUSH:
        approach = s.params.get("approach", "top")
        if p["mount"] == "wall" and approach != "front":
            out.append(
                f"{where}: a wall button faces sideways; push it with approach front (the tool tilted to point "
                "at its face)"
            )
        if p["mount"] == "desk" and approach == "front":
            out.append(f"{where}: a desk button faces up; push it from the top (approach top)")
        travel = _bound(s, "travel_mm", "at_least")
        if travel is not None and travel > 0:
            out.append(
                f"{where}: a button springs back when the tool lets go, so its travel_mm reads 0 when the step "
                "ends; judge the press by pressed at_least 1"
            )
        need = BUTTON_PRESS_FACTOR * float(p["actuation_force_n"])
        if _push_allowed(s) < need - 1e-9:
            out.append(
                f"{where}: a press on a button pushes with at least {num(BUTTON_PRESS_FACTOR)} x its "
                f"actuation_force_n ({num(need)} N, past its bottom-out), but this push keeps under "
                f"{_allowed_text(s)}; set {where.split(' ', 1)[0] + '.constraints.' if paths else ''}max_force_n "
                f"to at least {num(_cents_up(need / FORCE_SHARE))} N"
            )
    if kind is TemplateKind.TILE_BED and s.primitive is PrimitiveKind.PUSH:
        weight = float(p["mass_g"]) / 1000.0 * 9.81
        yield_n, hard, proud = float(p["bed_yield_n"]), float(p["bed_hardening_n_mm"]), float(p["proud_mm"])
        level = _bound(s, "height_err_mm", "at_most")
        press = _bound(s, "force_n", "at_least")
        least = None
        if level is not None and proud - level + TILE_REBOUND_MM > 0:
            sink = proud - level + TILE_REBOUND_MM
            least = max(0.0, yield_n + hard * sink - weight)
        if press is None:
            if least is not None:
                out.append(
                    f"{where}: a push on a tile presses with {num(PRESS_N)} N unless it asks for a force; add "
                    f"force_n at_least {num(math.ceil(least))} (the bed yields at bed_yield_n and hardens by "
                    "bed_hardening_n_mm per mm it sinks; the tile's weight helps)"
                )
        else:
            goal = press * (1.0 + FORCE_GOAL_MARGIN)
            if least is not None and press < least - 1e-9:
                out.append(
                    f"{where}: force_n at_least {num(press)} leaves the tile proud: sinking it to within "
                    f"{num(level)} mm of its level (it stands {num(proud)} mm proud and springs back up to "
                    f"{num(TILE_REBOUND_MM)} mm) takes {num(least)} N on this bed (bed_yield_n + "
                    "bed_hardening_n_mm x the sink, less the tile's weight)"
                )
            if level is not None and level < TILE_OVER_MM:
                most = yield_n + hard * (proud + level) - weight
                if goal > most + 1e-9:
                    out.append(
                        f"{where}: force_n at_least {num(press)} (pressed to {num(goal)} N) sinks the tile more than "
                        f"{num(level)} mm below its level on this bed (at most {num(most)} N keeps it within)"
                    )
            if goal > _push_allowed(s) + 1e-9:
                out.append(
                    f"{where}: the push presses to {num(goal)} N (force_n at_least {num(press)} and "
                    f"{num(100 * FORCE_GOAL_MARGIN)} %) but keeps under {_allowed_text(s)}; raise max_force_n "
                    f"to at least {num(_cents_up(goal / FORCE_SHARE))} N"
                )
    if kind is TemplateKind.SLIDING_DOOR and s.primitive is PrimitiveKind.PUSH:
        speed = float(s.params["speed_mm_s"])
        if speed > SLIDE_PUSH_MAX_MM_S + 1e-9:
            out.append(
                f"{where}: a push shutting a sliding door reaches its closed stop in time only at speed_mm_s "
                f"{num(SLIDE_PUSH_MAX_MM_S)} or less (faster, it runs out of time a few mm open); set "
                f"{where.split(' ', 1)[0] + '.params.' if paths else ''}speed_mm_s to at most {num(SLIDE_PUSH_MAX_MM_S)}"
            )
    if kind is TemplateKind.SLIDING_DOOR and s.primitive is PrimitiveKind.HOOK_PULL:
        path = where.split(" ", 1)[0]
        need = SLIDE_PULL_FACTOR * float(p["pull_force_n"])
        drag = (
            f"a gripper drags a sliding door at up to {num(SLIDE_PULL_FACTOR)} x its pull_force_n "
            f"({num(need)} N for this one; 23 to 26 N measured on a 10 N door)"
        )
        cap = s.constraints.max_force_n
        if cap is not None and cap < need - 1e-9:
            out.append(
                f"{where}: {path}.constraints.max_force_n {num(cap)} N stops the pull: {drag}; set "
                f"{path}.constraints.max_force_n to at least {num(need)}, or null"
            )
        for k, c in enumerate(s.success):
            if c.quantity == "force_n" and c.at_most is not None and c.at_most < need - 1e-9:
                out.append(
                    f"{where}: {path}.success[{k}] force_n at_most {num(c.at_most)} stops the pull: {drag}; set "
                    f"{path}.success[{k}].at_most to at least {num(need)}, or leave the bound out"
                )
        slip = _bound(s, "slip_mm", "at_most")
        if slip is not None and slip < SLIDE_SLIP_MIN_MM - 1e-9:
            out.append(
                f"{where}: a pinch on a sliding door's fin gives about 33 mm before a 10 N door moves; bound "
                f"slip_mm at {num(SLIDE_SLIP_MIN_MM)} mm or more, or leave it out"
            )
    return out


@cache
def library_part_names() -> tuple[str, ...]:
    """Parts a ``library:<name>`` ref may name: the training parts of the parts library."""
    from rtt.parts.library import TRAINING_FAMILIES, load_parts_library

    lib = load_parts_library()
    return tuple(n for n in lib.names() if lib.entry(n).family in TRAINING_FAMILIES)


def _ref_problems(
    objects: Sequence[ObjectInstance],
    uploads: Collection[str],
    required: Collection[str],
    check_library: bool,
) -> list[str]:
    out: list[str] = []
    used: set[str] = set()
    for i, o in enumerate(objects):
        if o.ref is None:
            continue
        scheme, _, key = o.ref.partition(":")
        where = f"objects[{i}] ({o.name})"
        if scheme == "upload":
            used.add(key)
            if key not in uploads:
                known = ", ".join(f"upload:{u}" for u in uploads) or "none"
                out.append(f"{where}: {o.ref} was not uploaded (uploads: {known}); use a box or cylinder")
        elif scheme == "scan":
            out.append(f"{where}: scan refs cannot be resolved here; use a box or cylinder (ref null)")
        elif scheme == "library" and check_library:
            names = library_part_names()
            if key not in names:
                out.append(
                    f"{where}: no library part {key!r} (known: {', '.join(names)}); use a box or cylinder"
                )
    for key in required:
        if key not in used:
            out.append(
                f"objects: the uploaded part upload:{key} is not used; model it as a part with shape mesh "
                f"and ref upload:{key}"
            )
    return out


def _step_problems(
    objects: Sequence[ObjectInstance], steps: Sequence[Step], rules: str = "v2", paths: bool = True
) -> list[str]:
    out: list[str] = []
    v2 = rules == "v2"
    windows = {**WINDOW_MIN, **WINDOW_MIN_V2} if v2 else WINDOW_MIN
    wrist = wrist_twist_limit()
    by_name = {o.name: o for o in objects}
    for i, s in enumerate(steps):
        where = _where_step(i, s)
        obj = by_name.get(s.object)
        if obj is None or obj.kind not in s.info.objects:
            continue  # the contract's parser reports it
        kind, params = obj.kind, dict(obj.params)
        twist = s.primitive is PrimitiveKind.TWIST
        goals = goal_quantities(obj.template)
        snug = twist and any(c.quantity == "torque_nm" and c.at_least is not None for c in s.success)
        if not snug and not any(c.quantity in goals for c in s.success):
            usable = sorted(goals & (set(obj.template.quantities) | set(s.info.quantities)))
            if twist:
                usable.append("torque_nm at_least")
            out.append(
                f"{where}: add a criterion on what the step achieves ({', '.join(usable)}); force, torque, "
                "slip and penetration limits alone do not say when it succeeded"
            )
        for c in s.success:
            least, lo, hi = windows.get(c.quantity), c.at_least, c.at_most
            if least is None or lo is None or hi is None or hi - lo >= least - 1e-9:
                continue
            out.append(
                f"{where}: the {c.quantity} window {num(lo)} to {num(hi)} is narrower than {num(least)} "
                f"{QUANTITIES[c.quantity].unit}; a measurement never lands exactly, so widen it or keep one bound"
            )
        if any(c.quantity == "in_target" for c in s.success) and s.target is None:
            out.append(f"{where}: in_target needs a target bin")
        if s.target is not None and s.hold_after:
            out.append(f"{where}: placing {s.object!r} in {s.target!r} releases it; set hold_after false")
        if s.primitive in PICKS and kind in NO_PICK:
            out.append(
                f"{where}: a threaded member lives on its stub and cannot be picked up or moved; leave the "
                "pick out (insert and twist grip the member in place)"
            )
        if twist and kind in TWIST_SENSE:
            sense = TWIST_SENSE[kind]
            if s.params["direction"] != sense:
                back = any(
                    p.primitive is PrimitiveKind.TWIST
                    and p.object == s.object
                    and p.params["direction"] == sense
                    for p in steps[:i]
                )
                if not back:
                    name = _TWIST_NAMES[kind]
                    if sense == "ccw":
                        out.append(
                            f"{where}: a {name} starts closed at its stop and opens counter-clockwise; twist "
                            "it ccw (cw meets the stop)"
                        )
                    else:
                        out.append(
                            f"{where}: a {name} tightens clockwise from its start; twist it cw (ccw meets "
                            "the thread start)"
                        )
        # a panel's clamp, and in rule set v2 a threaded joint's member too (the frozen v1 rule set keeps
        # the v1 recording's verdicts: its e2 nut asks for depth_mm at_least 0)
        if v2 and s.primitive is PrimitiveKind.INSERT and kind in THREAD_KINDS:
            start = _lead(_thread(kind, params))
            low = _bound(s, "depth_mm", "at_least")
            if low is not None and low < start - INSERT_PAST_MM - 1e-9:
                out.append(
                    f"{where}: an insert stops {num(INSERT_PAST_MM)} mm past its depth_mm at_least, so at_least "
                    f"{num(low)} leaves the member {num(low + INSERT_PAST_MM)} mm down, short of the thread start "
                    f"({num(start)} mm): the thread never catches; "
                    + (
                        f"set {_criterion_path(where, s, 'depth_mm', low)}.at_least to {num(start - INSERT_PAST_MM)} "
                        if paths
                        else f"ask depth_mm at_least {num(start - INSERT_PAST_MM)} "
                    )
                    + "or leave at_least out (the insert then goes to the thread start)"
                )
        if s.primitive is PrimitiveKind.INSERT and kind in THREAD_KINDS and s.hold_after:
            out.append(
                f"{where}: end the insert of a threaded member with hold_after false; the twist grips it "
                "again in place"
            )
        stops_early = any(c.quantity == "torque_nm" and c.at_least is not None for c in s.success)
        if twist and stops_early and kind in (TemplateKind.QUARTER_TURN_LATCH, TemplateKind.HANDWHEEL_VALVE):
            out.append(
                f"{where}: a twist stops as soon as the roll torque reaches a torque_nm at_least, and a "
                f"{_TWIST_NAMES[kind]} resists most before it turns; drop torque_nm at_least here (it "
                "tightens a thread until snug)"
            )
        if twist and kind in THREAD_KINDS:
            inserted = any(p.primitive is PrimitiveKind.INSERT and p.object == s.object for p in steps[:i])
            if not inserted:
                out.append(
                    f"{where}: a threaded member starts on its lead-in, free of the thread; insert "
                    f"{s.object!r} to the thread start before this twist"
                )
        if v2:
            at = f"objects[{list(objects).index(obj)}]"  # obj is one of objects (by_name)
            out += _v2_step_problems(where, obj, s, at, paths)
            following = steps[i + 1] if i + 1 < len(steps) else None
            if (
                s.hold_after
                and following is not None
                and following.object == s.object
                and following.handle != s.handle
            ):
                out.append(
                    f"{where}: it ends holding {s.object!r} by its {s.handle}, but the next step works its "
                    f"{following.handle}; a held object goes on by the same handle, so let go (hold_after false)"
                )
        for c in s.success:
            d = _demand(c)
            if d is None:
                continue
            if c.quantity in ("lift_mm", "hold_s") and c.quantity in s.params:
                own = float(s.params[c.quantity])
                if d > own:
                    out.append(
                        f"{where}: {c.quantity} at_least {num(d)} is more than the step's own {c.quantity} "
                        f"param ({num(own)}); raise the param or lower the criterion"
                    )
                continue
            ceiling = _ceiling(kind, params, c.quantity, s.primitive)
            if ceiling is not None and d > ceiling[0] + 1e-9:
                out.append(
                    f"{where}: {c.quantity} {num(d)} cannot be reached; {s.object!r} stops at {num(ceiling[0])} "
                    f"({ceiling[1]})"
                )
        need = _torque_needed(kind, params, s)
        if need is not None:
            torque, why = need
            for c in s.success:
                if c.quantity == "torque_nm" and c.at_least is not None and c.at_least > torque:
                    torque, why = c.at_least, "its torque_nm at_least criterion"
            cap = s.constraints.max_torque_nm
            allowed = min(wrist, TORQUE_SHARE * cap) if cap is not None else wrist
            if torque > allowed + 1e-9:
                if cap is not None and TORQUE_SHARE * cap < wrist:
                    out.append(
                        f"{where}: max_torque_nm {num(cap)} lets the twist turn up to {num(TORQUE_SHARE * cap)} "
                        f"N m ({num(100 * TORQUE_SHARE)} %), below the {num(torque)} N m it takes ({why})"
                    )
                else:
                    out.append(
                        f"{where}: it takes {num(torque)} N m ({why}) but a twist turns at most {num(wrist)} N m "
                        "(95 % of the wrist roll's 15 N m); lower the torque or refuse the step"
                    )
            object_torque = need[0]
            for c in s.success:
                if c.quantity == "torque_nm" and c.at_most is not None and c.at_most < object_torque:
                    out.append(
                        f"{where}: torque_nm at_most {num(c.at_most)} is below the {num(object_torque)} N m it "
                        f"takes ({need[1]})"
                    )
        need_f = _force_needed(kind, params, s)
        if need_f is not None:
            force, why, shared = need_f
            cap_f = s.constraints.max_force_n
            if not shared:
                if cap_f is not None and cap_f < force:
                    out.append(
                        f"{where}: max_force_n {num(cap_f)} is below the {num(force)} N it takes ({why})"
                    )
            elif cap_f is not None and FORCE_SHARE * cap_f < force:
                out.append(
                    f"{where}: max_force_n {num(cap_f)} lets the push reach {num(FORCE_SHARE * cap_f)} N "
                    f"({num(100 * FORCE_SHARE)} %), below the {num(force)} N it takes ({why})"
                )
            elif cap_f is None and DEFAULT_PUSH_N < force:
                out.append(
                    f"{where}: without a max_force_n the push stops at {num(DEFAULT_PUSH_N)} N, below the "
                    f"{num(force)} N it takes ({why}); set max_force_n"
                )
    return out


def _layout_warnings(objects: Sequence[ObjectInstance], steps: Sequence[Step]) -> list[str]:
    """Lids the simulator cannot keep open (warnings)."""
    out: list[str] = []
    opened = {s.object for s in steps if s.primitive is PrimitiveKind.HOOK_PULL}
    for i, o in enumerate(objects):
        if o.kind is not TemplateKind.HINGED_LID or o.params["orientation"] != "lid":
            continue
        where = f"objects[{i}] ({o.name})"
        stop = float(o.params["max_angle_deg"])
        if o.name in opened and stop <= 90.0:
            out.append(
                f"{where}: a lid stays open only past vertical; with max_angle_deg {num(stop)} it falls shut "
                "when the tool lets go"
            )
    return out


def _lid_problems(objects: Sequence[ObjectInstance]) -> list[str]:
    """A lid over a bin (same position) whose height is not the lid's fixed hinge height, and a lip lid
    that does not overhang its bin by ``feature_mm`` on every side (CR-S2b-1: S2b's hook slides under
    the free edge)."""
    out: list[str] = []
    for i, o in enumerate(objects):
        if o.kind is not TemplateKind.HINGED_LID or o.params["orientation"] != "lid":
            continue
        for b in objects:
            if b.kind is not TemplateKind.BIN:
                continue
            same = abs(b.pose.x_mm - o.pose.x_mm) < 1.0 and abs(b.pose.y_mm - o.pose.y_mm) < 1.0
            if not same:
                continue
            height = float(b.params["height_mm"])
            if abs(height - LID_HINGE_MM) > 1.0:
                out.append(
                    f"objects[{i}] ({o.name}): the lid template's hinge height is fixed at {num(LID_HINGE_MM)} mm, "
                    f"so the bin {b.name!r} it covers must be {num(LID_HINGE_MM)} mm tall (it is {num(height)} mm); "
                    "if the use case states another height, put that lid on that bin in unsupported (not "
                    "modelled) instead of changing it"
                )
            if o.params["feature"] == "lip":
                # the bin's plan extent in the lid's frame (length along x, width along y, turned by the
                # yaw between them)
                rel = math.radians(b.pose.yaw_deg - o.pose.yaw_deg)
                c, s = abs(math.cos(rel)), abs(math.sin(rel))
                length, width = float(b.params["length_mm"]), float(b.params["width_mm"])
                ex, ey = length * c + width * s, length * s + width * c
                lip = float(o.params["feature_mm"])
                need_x, need_y = ex + 2.0 * lip, ey + 2.0 * lip
                w, d = float(o.params["width_mm"]), float(o.params["depth_mm"])
                if w < need_x - 1e-9 or d < need_y - 1e-9:
                    out.append(
                        f"objects[{i}] ({o.name}): a lip lid is opened by a hook under its free edge, so it "
                        f"overhangs its bin by feature_mm ({num(lip)} mm) on every side: at least {num(need_x)} x "
                        f"{num(need_y)} mm over the bin {b.name!r} ({num(ex)} x {num(ey)} mm), it is {num(w)} x "
                        f"{num(d)} mm; make it larger or give it a handle"
                    )
    return out


def _lid_open_problems(objects: Sequence[ObjectInstance], steps: Sequence[Step]) -> list[str]:
    """A pick into a bin under a lid needs an earlier hook pull of that lid past vertical
    (``angle_deg`` at_least :data:`LID_OPEN_DEG`), and no push closing it in between: a lid opened
    less falls shut when the tool lets go, and the pick then meets a closed lid."""
    out: list[str] = []
    lids: dict[str, list[ObjectInstance]] = {}
    for b in objects:
        if b.kind is TemplateKind.BIN:
            lids[b.name] = [
                o
                for o in objects
                if o.kind is TemplateKind.HINGED_LID
                and o.params["orientation"] == "lid"
                and abs(b.pose.x_mm - o.pose.x_mm) < 1.0
                and abs(b.pose.y_mm - o.pose.y_mm) < 1.0
            ]
    seen: set[tuple[int, str]] = set()  # one problem per hook pull and lid, at its first pick
    need = num(LID_OPEN_DEG)
    for i, s in enumerate(steps):
        if s.primitive not in PICKS or s.target is None:
            continue
        for lid in lids.get(s.target, []):
            pulls = [
                j
                for j in range(i)
                if steps[j].object == lid.name and steps[j].primitive is PrimitiveKind.HOOK_PULL
            ]
            if not pulls:
                out.append(
                    f"{_where_step(i, s)}: it picks into {s.target!r}, which the lid {lid.name!r} covers, and no "
                    f"earlier step opens that lid; hook_pull {lid.name!r} first with angle_deg at_least {need} "
                    "(past vertical: a lid opened less falls shut)"
                )
                continue
            k = pulls[-1]
            pull = steps[k]
            pushes = [
                j
                for j in range(k + 1, i)
                if steps[j].object == lid.name and steps[j].primitive is PrimitiveKind.PUSH
            ]
            if pushes:
                out.append(
                    f"{_where_step(i, s)}: {steps[pushes[0]].id!r} pushes the lid {lid.name!r} shut after "
                    f"{pull.id!r} opened it, so this pick into {s.target!r} meets a closed lid; hook_pull it open "
                    f"again (angle_deg at_least {need}) after that push"
                )
                continue
            opened = [
                c.at_least for c in pull.success if c.quantity == "angle_deg" and c.at_least is not None
            ]
            if opened and max(opened) >= LID_OPEN_DEG - 1e-9:
                continue
            if (k, lid.name) in seen:
                continue
            seen.add((k, lid.name))
            asks = f"it asks for {num(max(opened))}" if opened else "it asks for no angle_deg at_least"
            out.append(
                f"{_where_step(k, pull)}: {s.id!r} picks into {s.target!r} under this lid later, so open it past "
                f"vertical: angle_deg at_least {need} ({asks}); a lid opened less falls shut when the tool lets go"
            )
    return out


def _door_clears(machine: ObjectInstance, other: ObjectInstance, room: float = CLEARANCE_MM) -> float:
    """How far a machine's sliding door must open (``travel_mm``) to uncover ``other`` inside it with
    ``room`` to spare (:data:`~rtt.studio.engineer.layout.CLEARANCE_MM`, or the finger room around a
    part a step picks there): its leaf's edge, ``width_mm / 2`` + the overlap at the start, slides
    toward local -y past the far side of ``other``'s footprint."""
    a = math.radians(machine.pose.yaw_deg)
    c, sn = math.cos(a), math.sin(a)
    low = min(
        -sn * (x - machine.pose.x_mm) + c * (y - machine.pose.y_mm)
        for _, poly in regions(other)
        for x, y in poly
    )
    return 0.5 * float(machine.params["width_mm"]) + MACHINE_OVERLAP_MM - low + room


def _machine_open_problems(objects: Sequence[ObjectInstance], steps: Sequence[Step]) -> list[str]:
    """A pick of something inside a machine, or into a bin inside it, needs an earlier hook pull that
    opens the machine's sliding door far enough to uncover it (:func:`_door_clears`), and no push
    shutting the door in between: a pick from above meets the closed door's roof leaf."""
    out: list[str] = []
    by_name = {o.name: o for o in objects}
    machines = [o for o in objects if o.kind is TemplateKind.SLIDING_DOOR]
    seen: set[tuple[int, str]] = set()
    for i, s in enumerate(steps):
        if s.primitive not in PICKS:
            continue
        for name in (s.object, s.target):
            thing = by_name.get(name) if name is not None else None
            if thing is None:
                continue
            for m in machines:
                if not inside_machine(m, thing):
                    continue
                picked = name == s.object  # a part picked there leaves a gripper's fingers room past it
                need = _door_clears(m, thing, FINGER_ROOM_MM if picked else CLEARANCE_MM)
                if need > float(m.params["travel_mm"]) + 1e-9:
                    out.append(
                        f"{_where_step(i, s)}: {thing.name!r} stands inside {m.name!r}, whose door must open "
                        f"{num(need)} mm to uncover it (its travel_mm is {num(float(m.params['travel_mm']))}); give "
                        "the door more travel or stand it nearer the opening's +y side"
                    )
                    continue
                pulls = [
                    j
                    for j in range(i)
                    if steps[j].object == m.name and steps[j].primitive is PrimitiveKind.HOOK_PULL
                ]
                if not pulls:
                    out.append(
                        f"{_where_step(i, s)}: {thing.name!r} stands inside {m.name!r} under its closed door; "
                        f"hook_pull {m.name!r} open first with travel_mm at_least {num(math.ceil(need))}"
                    )
                    continue
                k = pulls[-1]
                pull = steps[k]
                shut = [
                    j
                    for j in range(k + 1, i)
                    if steps[j].object == m.name and steps[j].primitive is PrimitiveKind.PUSH
                ]
                if shut:
                    out.append(
                        f"{_where_step(i, s)}: {steps[shut[0]].id!r} slides the door of {m.name!r} shut after "
                        f"{pull.id!r} opened it, so this pick meets a closed door; open it again after that push"
                    )
                    continue
                opened = _bound(pull, "travel_mm", "at_least")
                if (opened is None or opened < need - 1e-9) and (k, thing.name) not in seen:
                    seen.add((k, thing.name))
                    asks = (
                        f"it asks for {num(opened)}"
                        if opened is not None
                        else "it asks for no travel_mm at_least"
                    )
                    at = next(
                        (
                            n
                            for n, c in enumerate(pull.success)
                            if c.quantity == "travel_mm" and c.at_least is not None
                        ),
                        None,
                    )
                    field = f"steps[{k}].success[{at}].at_least" if at is not None else f"steps[{k}].success"
                    room = (
                        f" and {num(FINGER_ROOM_MM)} mm past it for a gripper's open fingers"
                        if picked
                        else ""
                    )
                    out.append(
                        f"{_where_step(k, pull)}: {s.id!r} picks {'into' if name == s.target else 'out of'} "
                        f"{m.name!r} later, so open its door far enough to uncover {thing.name!r}{room}: travel_mm "
                        f"at_least {num(math.ceil(need))} ({asks}; set {field})"
                    )
    return out


#: Words that say something stands inside a machine (an assumption or a label) [assumed].
_INSIDE = re.compile(r"\b(inside|in the machine|interior|under the (loading )?opening)\b", re.IGNORECASE)
#: ... unless the same words put it outside (a line that says both is not a claim to check).
_OUTSIDE = re.compile(r"\b(outside|beside|next to|clear of|not inside|out of)\b", re.IGNORECASE)


def _mentions(text: str, o: ObjectInstance) -> bool:
    low = text.lower()
    return o.name.lower() in low or o.name.replace("_", " ").lower() in low or o.label.lower() in low


#: Words that name the machine itself (with its own name or label): a claim binds only what a
#: clause names together with the machine ("the tray is inside the machine"), not what it puts
#: inside something else ("the finished part is inside the out bin").
_MACHINE = re.compile(r"\b(machine|enclosure)\b|\binside the door\b", re.IGNORECASE)
#: Clauses of a line: claims are read clause by clause.
_CLAUSE = re.compile(r";|\.\s")


def _claims(spec: TaskSpec) -> list[tuple[str, ObjectInstance, ObjectInstance]]:
    """Every ``(where, thing, machine)`` an assumption or a label of the plan puts inside a machine
    (the first machine): a clause that says inside, names the machine (:data:`_MACHINE` or its name)
    and names the thing; a clause that also puts something outside is not a claim."""
    machines = [o for o in spec.objects if o.kind is TemplateKind.SLIDING_DOOR]
    if not machines:
        return []
    things = [o for o in spec.objects if o.kind in (TemplateKind.BIN, TemplateKind.PART)]
    lines = [(f"assumptions[{k}]", a) for k, a in enumerate(spec.assumptions)]
    lines += [(f"objects[{spec.objects.index(o)}] ({o.name}).label", o.label) for o in things]
    out = []
    for where, line in lines:
        for text in _CLAUSE.split(line):
            if not _INSIDE.search(text) or _OUTSIDE.search(text):
                continue
            if not (_MACHINE.search(text) or any(_mentions(text, m) for m in machines)):
                continue
            for o in things:
                if (_mentions(text, o) or where.endswith(f"({o.name}).label")) and (
                    where,
                    o,
                    machines[0],
                ) not in out:
                    out.append((where, o, machines[0]))
    return out


def claimed_inside(spec: TaskSpec) -> dict[str, str]:
    """What the plan's words put inside a machine (thing's name to the machine's), wherever it stands
    (rule set v2): the layout fixer places it there (:func:`rtt.studio.engineer.layout.fix_layout`)."""
    out: dict[str, str] = {}
    for _, o, m in _claims(spec):
        out.setdefault(o.name, m.name)
    return out


def claim_problems(spec: TaskSpec) -> list[str]:
    """Layout claims the plan makes in words that its poses contradict (rule set v2): an assumption
    or a label that puts a bin or a part inside a machine that it does not stand inside."""
    out: list[str] = []
    machines = [o for o in spec.objects if o.kind is TemplateKind.SLIDING_DOOR]
    for where, o, m in _claims(spec):
        if not any(inside_machine(x, o) for x in machines):
            out.append(
                f"{where}: it puts {o.name!r} inside {m.name!r}, but at ({num(o.pose.x_mm)}, {num(o.pose.y_mm)}) its "
                f"footprint is not inside the machine's interior (depth_mm - 40 x width_mm - 20 mm about "
                f"{m.name!r}'s pose, at most height_mm - 20 mm tall); move it there or correct the words"
            )
    return out


def check_parts(
    objects: Sequence[ObjectInstance],
    steps: Sequence[Step],
    *,
    uploads: Collection[str] = (),
    required_uploads: Collection[str] = (),
    check_library: bool = True,
    rules: str = "v2",
    paths: bool = True,
) -> list[str]:
    """The plan checks on whatever objects and steps parsed (also when the task spec as a whole
    did not), so one repair prompt can list every problem at once, under rule set ``rules``."""
    if rules not in RULES:
        raise ValueError(f"unknown rule set {rules!r} (known: {', '.join(RULES)})")
    out = (
        _ref_problems(objects, uploads, required_uploads, check_library)
        + _lid_problems(objects)
        + layout_problems(objects, steps, rules=rules)
        + _step_problems(objects, steps, rules, paths)
        + _lid_open_problems(objects, steps)
    )
    if rules == "v2":
        out += _machine_open_problems(objects, steps)
    return out


def check_plan(
    spec: TaskSpec,
    *,
    uploads: Collection[str] = (),
    required_uploads: Collection[str] = (),
    check_library: bool = True,
    rules: str = "v2",
    paths: bool = True,
) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` of a constructed task spec (module docstring), under rule set
    ``rules`` (v1: frozen with the v1 prompt; v2: the default).

    ``uploads``: upload ids that exist (``upload:<id>`` refs may name them);
    ``required_uploads``: ids the use case came with, which the plan must use.
    """
    problems: list[str] = []
    warnings: list[str] = []
    if spec.refusal is not None:
        if not spec.unsupported:
            problems.append(
                "unsupported: a refusal lists what is not simulated, why, and the nearest primitive"
            )
        elif all(u.nearest_primitive is None for u in spec.unsupported):
            problems.append(
                "unsupported: name the nearest primitive for at least one item (what the simulator could "
                "do instead)"
            )
        return problems, warnings
    problems += check_parts(
        spec.objects,
        spec.steps,
        uploads=uploads,
        required_uploads=required_uploads,
        check_library=check_library,
        rules=rules,
        paths=paths,
    )
    if rules == "v2":
        problems += claim_problems(spec)
    touched = {s.object for s in spec.steps} | {s.target for s in spec.steps if s.target}
    for i, o in enumerate(spec.objects):
        if o.name not in touched:
            warnings.append(f"objects[{i}] ({o.name}): no step acts on it or places into it")
    warnings += _layout_warnings(spec.objects, spec.steps)
    return problems, warnings


def plan_problems(spec: TaskSpec, **kwargs: object) -> list[str]:
    """The problems of :func:`check_plan` (no warnings)."""
    return check_plan(spec, **kwargs)[0]  # type: ignore[arg-type]


# --------------------------------------------------------------------------- nearest primitive

#: Keyword -> primitive, first match in the text wins (deformables map to the pick that would move them).
_NEAREST: Final[tuple[tuple[re.Pattern[str], PrimitiveKind], ...]] = tuple(
    (re.compile(rf"(?<![a-z]){pattern}", re.IGNORECASE), kind)
    for pattern, kind in (
        (r"(unscrew|screw|tighten|loosen|twist|turn|rotat|knob|valve|wheel|cap\b|nut\b|latch)", "twist"),
        (r"(insert|plug|socket|peg|seat|slot in|thread (it )?on|connector)", "insert"),
        (r"(open|pull|drawer|hatch|door|flap)", "hook_pull"),
        (r"(close|shut|press|push|slide|tuck|button)", "push"),
        (r"(sheet|film|flat|panel|glass|card|label|sticker|envelope|smooth)", "suction"),
        (r"(wrap|curl|hug|round|ball|fruit|bottle)", "wrap"),
        (
            (
                r"(pick|lift|take|move|place|put|carry|hold|stack|sort|hang|fold|pour|scoop|cloth|shirt|towel|"
                r"fabric|bag|cable|rope|string|food|liquid|water|powder)"
            ),
            "grasp",
        ),
    )
)


def nearest_primitive(text: str) -> PrimitiveKind | None:
    """The primitive whose keyword occurs first in ``text`` (``None`` when none does)."""
    best: tuple[int, PrimitiveKind] | None = None
    for pattern, kind in _NEAREST:
        m = pattern.search(text or "")
        if m is not None and (best is None or m.start() < best[0]):
            best = (m.start(), PrimitiveKind(kind))
    return best[1] if best else None
