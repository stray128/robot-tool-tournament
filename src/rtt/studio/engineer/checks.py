"""Checks on top of :class:`~rtt.studio.types.TaskSpec`: is the plan runnable, not only well formed?

:class:`TaskSpec` checks structure and cross-references (names, kinds, handles, measured
quantities, order, holding). :func:`check_plan` adds what a simulator run needs (the rules the
prompt states, :func:`rtt.studio.engineer.catalog.render_rules`):

* every step has a **goal** criterion (a state quantity of its object, or ``lift_mm``,
  ``hold_s``, ``in_target``), not only force or torque limits;
* ``in_target`` needs a target, and a step that places its object releases it;
* an **insert** follows a step that holds the same object;
* **reachable** criteria (a lid inside its hinge stop, a drawer inside its travel, a latch
  inside its turn, a valve inside its travel, a thread inside its engagement, a plug inside its
  socket, a pick inside its own lift and hold);
* **torque** a twist needs within the wrist roll's limit, the step's ``max_torque_nm`` and any
  ``torque_nm`` bound; **force** a drawer or plug needs within the step's ``max_force_n``;
* **refs**: uploads named exist and an uploaded part is used; library refs name training parts;
* a **refusal** lists what is not simulated with at least one nearest primitive.

Problems carry a path (``objects[2] (tote): ...``, ``steps[1] (open_lid): ...``) so a repair
prompt and the intake page can point at the entry. Warnings (objects nothing touches) never
refuse a plan.

:func:`nearest_primitive` maps free text (an unsupported action) to the closest primitive by
keyword; it backs a refusal whose nearest primitive the model left out.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from functools import cache
from typing import Final

from rtt.studio.engineer.catalog import goal_quantities, num
from rtt.studio.types import (
    STUDIO_WRIST,
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

#: Slack on a threaded member's seated depth (engage_turns x pitch) before a criterion is unreachable.
DEPTH_TOLERANCE_MM: Final = 0.5
#: Share of engage_turns from which a twist counts as seating the thread (needs final_torque_nm).
_SEATING_FRACTION: Final = 0.9


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


def _ceiling(kind: TemplateKind, params: dict, quantity: str) -> tuple[float, str] | None:
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
        seated = float(p["engage_turns"]) * float(p["pitch_mm"])
        return seated + DEPTH_TOLERANCE_MM, "its seated depth engage_turns x pitch_mm"
    if kind is TemplateKind.THREADED_JOINT and quantity == "angle_deg":
        return 360.0 * float(p["engage_turns"]), "its engage_turns x 360 deg"
    if kind is TemplateKind.PORT_PLUG and quantity == "depth_mm":
        return float(p["depth_mm"]), "the socket's depth_mm"
    return None


def _torque_needed(kind: TemplateKind, params: dict, step: Step) -> tuple[float, str] | None:
    """The torque a twist step must deliver on ``kind``, and why."""
    if step.primitive is not PrimitiveKind.TWIST:
        return None
    if kind is TemplateKind.QUARTER_TURN_LATCH:
        return float(params["torque_nm"]), "the latch's torque_nm"
    if kind is TemplateKind.HANDWHEEL_VALVE:
        return float(params["torque_nm"]), "the valve's torque_nm"
    if kind is TemplateKind.THREADED_JOINT:
        engage = float(params["engage_turns"])
        seated = engage * float(params["pitch_mm"])
        seats = False
        for c in step.success:
            d = _demand(c)
            if d is None:
                continue
            if c.quantity == "turns" and d >= _SEATING_FRACTION * engage:
                seats = True
            if c.quantity == "angle_deg" and d >= _SEATING_FRACTION * 360.0 * engage:
                seats = True
            if c.quantity == "depth_mm" and d >= _SEATING_FRACTION * seated:
                seats = True
        if seats:
            return float(params["final_torque_nm"]), "the thread's final_torque_nm (seating it)"
        return float(params["start_torque_nm"]), "the thread's start_torque_nm"
    return None


def _force_needed(kind: TemplateKind, params: dict, step: Step) -> tuple[float, str] | None:
    if kind is TemplateKind.DRAWER and step.primitive in (PrimitiveKind.HOOK_PULL, PrimitiveKind.PUSH):
        return float(params["pull_force_n"]), "the drawer's pull_force_n"
    if kind is TemplateKind.PORT_PLUG and step.primitive is PrimitiveKind.INSERT:
        return float(params["insert_force_n"]), "the plug's insert_force_n"
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
    torque_limit = STUDIO_WRIST.roll_torque_nm
    by_name = {o.name: o for o in objects}
    for i, s in enumerate(steps):
        where = _where_step(i, s)
        obj = by_name.get(s.object)
        if obj is None or obj.kind not in s.info.objects:
            continue  # the contract's parser reports it
        kind, params = obj.kind, dict(obj.params)
        goals = goal_quantities(obj.template)
        if not any(c.quantity in goals for c in s.success):
            usable = sorted(goals & (set(obj.template.quantities) | set(s.info.quantities)))
            out.append(
                f"{where}: add a criterion on what the step achieves ({', '.join(usable)}); force, torque, "
                "slip and penetration limits alone do not say when it succeeded"
            )
        if any(c.quantity == "in_target" for c in s.success) and s.target is None:
            out.append(f"{where}: in_target needs a target bin")
        if s.target is not None and s.hold_after:
            out.append(f"{where}: placing {s.object!r} in {s.target!r} releases it; set hold_after false")
        if s.primitive is PrimitiveKind.INSERT:
            prev = steps[i - 1] if i > 0 else None
            if prev is None or prev.object != s.object or not prev.hold_after:
                out.append(
                    f"{where}: insert pushes a member the tool holds; put a grasp of {s.object!r} with "
                    "hold_after true right before it"
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
            ceiling = _ceiling(kind, params, c.quantity)
            if ceiling is not None and d > ceiling[0] + 1e-9:
                out.append(
                    f"{where}: {c.quantity} {num(d)} cannot be reached; {s.object!r} stops at {num(ceiling[0])} "
                    f"({ceiling[1]})"
                )
        need = _torque_needed(kind, params, s)
        if need is not None:
            torque, why = need
            if torque > torque_limit:
                out.append(
                    f"{where}: it takes {num(torque)} N m ({why}) but the wrist roll gives at most "
                    f"{num(torque_limit)} N m; lower the torque or refuse the step"
                )
            cap = s.constraints.max_torque_nm
            if cap is not None and cap < torque:
                out.append(
                    f"{where}: max_torque_nm {num(cap)} is below the {num(torque)} N m it takes ({why})"
                )
            for c in s.success:
                if c.quantity == "torque_nm" and c.at_most is not None and c.at_most < torque:
                    out.append(
                        f"{where}: torque_nm at_most {num(c.at_most)} is below the {num(torque)} N m it takes "
                        f"({why})"
                    )
        need_f = _force_needed(kind, params, s)
        if need_f is not None:
            force, why = need_f
            cap_f = s.constraints.max_force_n
            if cap_f is not None and cap_f < force:
                out.append(f"{where}: max_force_n {num(cap_f)} is below the {num(force)} N it takes ({why})")
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
    return _ref_problems(objects, uploads, required_uploads, check_library) + _step_problems(objects, steps)


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
