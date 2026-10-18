"""Checks on top of :class:`~rtt.studio.types.TaskSpec`: is the plan runnable, not only well formed?

:class:`TaskSpec` checks structure and cross-references (names, kinds, handles, measured
quantities, order, holding). :func:`check_plan` adds what a run in the Studio's simulator needs:
the rules the prompt states (:func:`rtt.studio.engineer.catalog.render_rules`), on the
simulator's own conventions (track S2, ``docs/STUDIO_PRIMITIVES.md``):

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
* a **refusal** lists what is not simulated with at least one nearest primitive.

Problems carry a path (``objects[2] (tote): ...``, ``steps[1] (open_lid): ...``) so a repair
prompt and the intake page can point at the entry. A lid over a bin (the same position) needs the
bin to be as tall as the lid's fixed hinge height (120 mm). Warnings never refuse a plan: objects
nothing touches, a lid that falls shut (its stop at or below vertical).

:func:`nearest_primitive` maps free text (an unsupported action) to the closest primitive by
keyword; it backs a refusal whose nearest primitive the model left out.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from functools import cache
from typing import Final

from rtt.studio.engineer.catalog import (
    DEFAULT_PUSH_N,
    FORCE_SHARE,
    LID_HINGE_MM,
    NO_PICK,
    PICKS,
    THREAD_LEAD_EXTRA_MM,
    TORQUE_SHARE,
    TWIST_SENSE,
    goal_quantities,
    num,
    wrist_twist_limit,
)
from rtt.studio.types import (
    ObjectInstance,
    PrimitiveKind,
    Step,
    SuccessCriterion,
    TaskSpec,
    TemplateKind,
)

__all__ = [
    "DEPTH_TOLERANCE_MM",
    "check_parts",
    "check_plan",
    "library_part_names",
    "nearest_primitive",
    "plan_problems",
]

#: Slack on a threaded member's seated depth before a criterion is unreachable (also in
#: ``web_catalog()["sim"]``).
DEPTH_TOLERANCE_MM: Final = 0.5

_TWIST_NAMES: Final = {
    TemplateKind.QUARTER_TURN_LATCH: "latch",
    TemplateKind.HANDWHEEL_VALVE: "valve",
    TemplateKind.THREADED_JOINT: "threaded member",
}


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
    return None


def _torque_needed(kind: TemplateKind, params: dict, step: Step) -> tuple[float, str] | None:
    """The torque a twist step must deliver on ``kind`` (where it stops), and why."""
    if step.primitive is not PrimitiveKind.TWIST:
        return None
    if kind is TemplateKind.QUARTER_TURN_LATCH:
        return float(params["torque_nm"]), "the latch's torque_nm"
    if kind is TemplateKind.HANDWHEEL_VALVE:
        return float(params["torque_nm"]), "the valve's torque_nm"
    if kind is TemplateKind.THREADED_JOINT:
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
        return start + (
            final - start
        ) * share, "the thread's torque there, start_torque_nm rising to final_torque_nm"
    return None


def _force_needed(kind: TemplateKind, params: dict, step: Step) -> tuple[float, str, bool] | None:
    """(the force a step must apply, why, whether the controller keeps to a share of the limit)."""
    if kind is TemplateKind.DRAWER and step.primitive is PrimitiveKind.HOOK_PULL:
        return float(params["pull_force_n"]), "the drawer's pull_force_n", False
    if kind is TemplateKind.DRAWER and step.primitive is PrimitiveKind.PUSH:
        return float(params["pull_force_n"]), "the drawer's pull_force_n", True
    if kind is TemplateKind.PORT_PLUG and step.primitive is PrimitiveKind.INSERT:
        return float(params["insert_force_n"]), "the plug's insert_force_n", True
    return None


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


def _step_problems(objects: Sequence[ObjectInstance], steps: Sequence[Step]) -> list[str]:
    out: list[str] = []
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
        if s.primitive is PrimitiveKind.INSERT and kind is TemplateKind.THREADED_JOINT and s.hold_after:
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
        if twist and kind is TemplateKind.THREADED_JOINT:
            inserted = any(p.primitive is PrimitiveKind.INSERT and p.object == s.object for p in steps[:i])
            if not inserted:
                out.append(
                    f"{where}: a threaded member starts on its lead-in, free of the thread; insert "
                    f"{s.object!r} to the thread start before this twist"
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
    """A lid over a bin (same position) whose height is not the lid's fixed hinge height."""
    out: list[str] = []
    for i, o in enumerate(objects):
        if o.kind is not TemplateKind.HINGED_LID or o.params["orientation"] != "lid":
            continue
        for b in objects:
            if b.kind is not TemplateKind.BIN:
                continue
            same = abs(b.pose.x_mm - o.pose.x_mm) < 1.0 and abs(b.pose.y_mm - o.pose.y_mm) < 1.0
            height = float(b.params["height_mm"])
            if same and abs(height - LID_HINGE_MM) > 1.0:
                out.append(
                    f"objects[{i}] ({o.name}): the lid template's hinge height is fixed at {num(LID_HINGE_MM)} mm, "
                    f"so the bin {b.name!r} it covers must be {num(LID_HINGE_MM)} mm tall (it is {num(height)} mm); "
                    "if the use case states another height, put that lid on that bin in unsupported (not "
                    "modelled) instead of changing it"
                )
    return out


def check_parts(
    objects: Sequence[ObjectInstance],
    steps: Sequence[Step],
    *,
    uploads: Collection[str] = (),
    required_uploads: Collection[str] = (),
    check_library: bool = True,
) -> list[str]:
    """The plan checks on whatever objects and steps parsed (also when the task spec as a whole
    did not), so one repair prompt can list every problem at once."""
    return (
        _ref_problems(objects, uploads, required_uploads, check_library)
        + _lid_problems(objects)
        + _step_problems(objects, steps)
    )


def check_plan(
    spec: TaskSpec,
    *,
    uploads: Collection[str] = (),
    required_uploads: Collection[str] = (),
    check_library: bool = True,
) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` of a constructed task spec (module docstring).

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
    )
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
