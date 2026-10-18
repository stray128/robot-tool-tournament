"""What the Task Engineer is told about the Studio, and what the intake page is given.

Everything here is rendered from the frozen contract (:mod:`rtt.studio.types`): the object
templates with their bounded params, handles and state quantities; the primitives with what
they act on, their "how" params and what they observe; the quantities; the rules the checks
enforce; the action-to-primitive mapping; and the wrist and table limits. The prompt and the
intake page (:func:`web_catalog`) therefore cannot drift from the code that validates a plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from rtt.effectors.types import ParamSpec
from rtt.studio.types import (
    CONTRACT_VERSION,
    MAX_OBJECTS,
    MAX_STEP_S,
    MAX_STEPS,
    PRIMITIVES,
    PRODUCT_MODE_LABEL,
    QUANTITIES,
    STUDIO_WRIST,
    ObjectPose,
    ObjectTemplate,
    PrimitiveInfo,
    PrimitiveKind,
    StepConstraints,
    TemplateKind,
    get_template,
    templates,
)

__all__ = [
    "GOAL_EXTRA",
    "HANDLE_TEXT",
    "LIMIT_QUANTITIES",
    "MAPPING_ROWS",
    "SEPARATION_MM",
    "goal_quantities",
    "handles_for",
    "num",
    "render_limits",
    "render_mapping",
    "render_primitives",
    "render_quantities",
    "render_rules",
    "render_templates",
    "web_catalog",
]

#: What each handle is, in words (the contract names handles only).
HANDLE_TEXT: Final[Mapping[str, str]] = {
    "lid": "the panel itself (push it closed)",
    "handle": "the lip or handle at the free edge or front (hook it to pull open)",
    "drawer": "the drawer front (push it shut)",
    "head": "the latch head, a slot or a wing (twist it)",
    "member": "the cap, coupling or nut (grasp, insert or twist it)",
    "seat": "the fixed threaded seat (no primitive acts on it)",
    "wheel": "the handwheel rim (twist it)",
    "plug": "the free plug or connector (grasp, insert or push it)",
    "socket": "the fixed socket (no primitive acts on it; the plug goes in)",
    "bin": "the open box (a destination for picks)",
    "part": "the part itself",
}

#: Quantities that only bound how a step went (limits), never whether it achieved anything.
LIMIT_QUANTITIES: Final[frozenset[str]] = frozenset(
    {"force_n", "lateral_force_n", "torque_nm", "penetration_mm", "slip_deg", "slip_mm", "jam"}
)
#: Pick quantities that say a step achieved something, besides the object's state quantities.
GOAL_EXTRA: Final[frozenset[str]] = frozenset({"lift_mm", "hold_s", "in_target"})
#: Footprint gap the prompt asks for between separate objects (a layout hint, not a check).
SEPARATION_MM: Final = 60.0

#: Template notes the contract leaves to the reader (how the Engineer should use a kind).
_TEMPLATE_NOTES: Final[Mapping[TemplateKind, str]] = {
    TemplateKind.HINGED_LID: "A tote lid, a hatch or a cabinet door (orientation door).",
    TemplateKind.DRAWER: "A drawer in a cabinet or shelf unit; it starts closed.",
    TemplateKind.QUARTER_TURN_LATCH: "A panel fastener or cam lock; it starts locked.",
    TemplateKind.THREADED_JOINT: (
        "A pipe coupling, cap or nut and the stub, rod or neck it threads onto (the seat)."
    ),
    TemplateKind.HANDWHEEL_VALVE: "A valve or any round handwheel; angle_deg counts from where it starts.",
    TemplateKind.PORT_PLUG: "A connector and its port, a peg and its hole, a vial and its holder.",
    TemplateKind.BIN: "A tote, tray, basket or box that picks drop into (fixed true unless it moves).",
    TemplateKind.PART: (
        "A rigid item: shape box or cylinder (cylinder: size_x_mm is the diameter, size_y_mm is ignored) "
        "with ref null; shape mesh only for an uploaded part, with ref upload:<id>."
    ),
}


def num(value: float, digits: int = 2) -> str:
    """``value`` with at most ``digits`` decimals and no trailing zeros (``100`` stays ``100``)."""
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _range(p: ParamSpec) -> str:
    if p.kind == "choice":
        return ", ".join(p.choices)
    if p.kind == "bool":
        return "true, false"
    assert p.low is not None and p.high is not None
    unit = f" {p.unit}" if p.unit and p.unit != "count" else ""
    return f"{num(p.low)} to {num(p.high)}{unit}"


def _default(p: ParamSpec) -> str:
    if p.kind == "bool":
        return "true" if p.default else "false"
    if p.kind == "choice":
        return str(p.default)
    return num(float(p.default))  # type: ignore[arg-type]


def _param_table(params: tuple[ParamSpec, ...]) -> str:
    rows = ["| parameter | range | default | what it is |", "|---|---|---|---|"]
    rows += [f"| {p.name} | {_range(p)} | {_default(p)} | {p.description} |" for p in params]
    return "\n".join(rows)


def handles_for(info: PrimitiveInfo, kind: TemplateKind) -> list[str]:
    """The handles ``info`` may address on an object of ``kind`` (template order)."""
    return [h for h in get_template(kind).handles if h in info.handles]


def goal_quantities(template: ObjectTemplate) -> set[str]:
    """Quantities that say a step on ``template`` achieved something."""
    return set(template.quantities) | set(GOAL_EXTRA)


def render_templates() -> str:
    blocks = []
    for t in templates():
        handles = "; ".join(f"{h}: {HANDLE_TEXT.get(h, h)}" for h in t.handles)
        state = ", ".join(t.quantities) if t.quantities else "none"
        tier = t.fidelity.value + (f" ({t.caveat})" if t.caveat else "")
        lines = [
            f"## {t.kind.value}: {t.display_name}",
            "",
            f"{t.description} {_TEMPLATE_NOTES.get(t.kind, '')}".rstrip(),
            f"Handles: {handles}. State quantities: {state}. Simulated as: {t.model}; fidelity {tier}.",
            "",
            _param_table(t.params),
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_primitives() -> str:
    blocks = []
    for kind in PrimitiveKind:
        info = PRIMITIVES[kind]
        acts = "; ".join(f"{k.value} via {' or '.join(handles_for(info, k))}" for k in info.objects)
        target = "a bin, or null" if info.targets else "always null"
        hold = (
            "yes (hold_after true keeps it for the next step)" if info.may_hold else "no (hold_after false)"
        )
        tier = info.fidelity.value + (f" ({info.caveat})" if info.caveat else "")
        lines = [
            f"## {kind.value}",
            "",
            info.description,
            f"Acts on: {acts}. Target: {target}. May end holding its object: {hold}. "
            f"Observes: {', '.join(info.quantities)}."
            + (" Drives the continuous wrist roll." if info.uses_roll else "")
            + f" Fidelity {tier}.",
            "",
            _param_table(info.params),
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_quantities() -> str:
    rows = ["| quantity | unit | range | meaning |", "|---|---|---|---|"]
    for q in QUANTITIES.values():
        rows.append(f"| {q.name} | {q.unit} | {num(q.low)} to {num(q.high)} | {q.description} |")
    return "\n".join(rows)


def render_rules() -> str:
    b = StepConstraints.BOUNDS
    torque = num(STUDIO_WRIST.roll_torque_nm)
    lim = num(ObjectPose.LIMIT_MM)
    return "\n".join(
        [
            (
                "- Names: object names and step ids are unique snake_case ids (a lower-case letter, then "
                "letters, digits or underscores; at most 32 characters)."
            ),
            f"- Size: 1 to {MAX_OBJECTS} objects and 1 to {MAX_STEPS} steps (a refused use case has no steps).",
            (
                "- Objects: kind is one template; label is a short generic name (at most 60 characters); "
                f"params hold every parameter of the kind inside its range; pose x_mm and y_mm within "
                f"-{lim} to {lim}, yaw_deg within -180 to 180; ref is null except for an uploaded mesh part."
            ),
            (
                "- Steps: the object exists and is a kind the primitive acts on; the handle is one listed "
                "for that kind and primitive; target is null or the name of a bin (not the object itself), "
                "and only grasp, suction and wrap take one; every success quantity is measured by the step "
                "(a state quantity of its object or one its primitive observes); after lists only earlier "
                "step ids; params hold every parameter of the primitive inside its range; the description "
                "is one plain sentence (at most 200 characters)."
            ),
            (
                "- Goal: every step has at least one criterion on what it achieves: a state quantity of its "
                "object, or lift_mm, hold_s or in_target. Force, torque, slip, jam and penetration limits "
                "alone do not say when a step succeeded."
            ),
            (
                "- in_target needs a target bin. A step that places its object in a target releases it "
                "(hold_after false)."
            ),
            (
                "- Holding: hold_after true (grasp, suction, wrap, twist and insert only) means the step ends "
                "still holding its object; the next step must act on that same object, and the last step "
                "must release (hold_after false)."
            ),
            (
                "- Insert pushes a member the tool already holds: the step right before an insert acts on the "
                "same object with hold_after true (a grasp)."
            ),
            (
                "- Reachable criteria: a lid's or door's angle within its max_angle_deg; a drawer's travel "
                "within its travel_mm; a latch's angle within its turn_deg; a valve's angle within its "
                "travel_deg; a threaded member's turns within its engage_turns and its depth within "
                "engage_turns x pitch_mm; a plug's depth within the socket's depth_mm; a pick's lift_mm and "
                "hold_s within the step's own lift_mm and hold_s params."
            ),
            (
                "- Torque: a twist delivers the object's torque (a latch's or valve's torque_nm; a threaded "
                "member's final_torque_nm to seat it, its start_torque_nm only to start it). That torque must "
                f"not exceed {torque} N m (the wrist roll's limit), the step's max_torque_nm or a torque_nm "
                "at_most criterion."
            ),
            (
                "- Force: a drawer's pull_force_n and a plug's insert_force_n must not exceed the step's "
                "max_force_n."
            ),
            (
                f"- Constraints: max_force_n {num(b['max_force_n'][0])} to {num(b['max_force_n'][1])} N, "
                f"max_torque_nm {num(b['max_torque_nm'][0])} to {num(b['max_torque_nm'][1])} N m, precision_mm "
                f"{num(b['precision_mm'][0])} to {num(b['precision_mm'][1])} mm, precision_deg "
                f"{num(b['precision_deg'][0])} to {num(b['precision_deg'][1])} deg, each or null. Criterion "
                "bounds stay inside the quantity's range; at_least is not above at_most."
            ),
            (
                "- Text: title 3 to 80 characters, description at most 600, at most 10 assumptions and at "
                "most 6 clarifications and 6 unsupported items, each line at most 200 characters (what at "
                "most 120); no brand, product or company names anywhere."
            ),
            (
                "- Refusal: refusal (at most 300 characters) says why the use case cannot be simulated; then "
                "steps is empty and unsupported lists what is not simulated, why, and the nearest primitive "
                "(at least one item names one)."
            ),
            (
                "- Uploads: an uploaded part is used as exactly the part object it describes (shape mesh, "
                "ref upload:<id>); refs name only uploads listed in the use case."
            ),
        ]
    )


#: (what the user says, the primitive or what to do).
MAPPING_ROWS: Final[tuple[tuple[str, str], ...]] = (
    (
        "pick up, take, lift, move, place, put in, drop in",
        (
            "grasp (fingers on a rigid part), suction (a flat or smooth top face), wrap (a round or odd "
            "shape); target the bin it goes into"
        ),
    ),
    ("open a lid, hatch or door; pull a drawer or a handle", "hook_pull on its handle"),
    ("close a lid, door or drawer; press; slide shut", "push on the lid or drawer"),
    (
        "turn a latch, knob, valve, cap, nut or coupling; screw, unscrew, tighten, loosen",
        "twist on its head, wheel or member (direction cw or ccw)",
    ),
    (
        "plug in, insert, seat, start a thread on a stub",
        "grasp with hold_after true, then insert (then twist for a thread)",
    ),
    (
        (
            "cloth, bags, film, foam, cables, rope, food, liquids, powders; cutting, welding, painting, "
            "gluing, sanding; people or animals"
        ),
        (
            "not simulated: an unsupported item with the nearest primitive, or a refusal when it is the "
            "use case itself"
        ),
    ),
)


def render_mapping() -> str:
    rows = ["| the user says | use |", "|---|---|"]
    rows += [f"| {a} | {b} |" for a, b in MAPPING_ROWS]
    return "\n".join(rows)


def render_limits() -> str:
    w = STUDIO_WRIST
    lim = num(ObjectPose.LIMIT_MM)
    return "\n".join(
        [
            (
                "- The wrist hangs over one table: it moves in x, y and z, turns about the vertical axis and "
                f"rolls the tool continuously (at most {num(w.roll_torque_nm)} N m and "
                f"{num(w.roll_speed_max_deg_s)} deg/s). It does not tilt a held part."
            ),
            (
                f"- Every object stands on the table, within -{lim} to {lim} mm of its centre in x and y. Give "
                "separate objects their own spots so their footprints do not overlap (at least "
                f"{num(SEPARATION_MM)} mm between edges); parts of one assembly (a lid on its tote, a door on "
                "its cabinet) may share a position."
            ),
            f"- A step lasts at most {num(MAX_STEP_S)} s, so choose speeds that finish the motion in time.",
        ]
    )


# --------------------------------------------------------------------------- web


def _param_json(p: ParamSpec) -> dict[str, Any]:
    return {
        "name": p.name,
        "kind": p.kind,
        "description": p.description,
        "unit": p.unit,
        "low": p.low,
        "high": p.high,
        "choices": list(p.choices),
        "default": p.default,
    }


def _library_parts() -> tuple[str, ...]:
    from rtt.studio.engineer.checks import library_part_names

    return library_part_names()


def web_catalog() -> dict[str, Any]:
    """The catalog the intake page validates and edits plans with (``catalog.json``).

    Brand tokens are not shipped to the browser; the API and the Python parser check them. The
    page's validator mirrors :class:`~rtt.studio.types.TaskSpec` and
    :func:`rtt.studio.engineer.checks.check_plan` on these numbers; a test regenerates this
    file and the parity cases and compares them with the committed copies.
    """
    return {
        "format": "rtt-studio-catalog",
        "version": 1,
        "contract": CONTRACT_VERSION,
        "label": PRODUCT_MODE_LABEL,
        "templates": [
            {
                "kind": t.kind.value,
                "display_name": t.display_name,
                "description": t.description,
                "note": _TEMPLATE_NOTES.get(t.kind, ""),
                "model": t.model,
                "fidelity": t.fidelity.value,
                "caveat": t.caveat,
                "params": [_param_json(p) for p in t.params],
                "handles": list(t.handles),
                "quantities": list(t.quantities),
                "needs_ref": t.needs_ref,
            }
            for t in templates()
        ],
        "primitives": [
            {
                "kind": k.value,
                "description": PRIMITIVES[k].description,
                "fidelity": PRIMITIVES[k].fidelity.value,
                "caveat": PRIMITIVES[k].caveat,
                "objects": [o.value for o in PRIMITIVES[k].objects],
                "handles": list(PRIMITIVES[k].handles),
                "targets": [o.value for o in PRIMITIVES[k].targets],
                "params": [_param_json(p) for p in PRIMITIVES[k].params],
                "quantities": list(PRIMITIVES[k].quantities),
                "uses_roll": PRIMITIVES[k].uses_roll,
                "may_hold": PRIMITIVES[k].may_hold,
            }
            for k in PrimitiveKind
        ],
        "quantities": [
            {"name": q.name, "unit": q.unit, "low": q.low, "high": q.high, "description": q.description}
            for q in QUANTITIES.values()
        ],
        "handles": dict(HANDLE_TEXT),
        "library_parts": list(_library_parts()),
        "limit_quantities": sorted(LIMIT_QUANTITIES),
        "goal_extra": sorted(GOAL_EXTRA),
        "limits": {
            "max_objects": MAX_OBJECTS,
            "max_steps": MAX_STEPS,
            "pose_mm": ObjectPose.LIMIT_MM,
            "roll_torque_nm": STUDIO_WRIST.roll_torque_nm,
            "constraints": {k: list(v) for k, v in StepConstraints.BOUNDS.items()},
            "text": {"title": 80, "description": 600, "label": 60, "line": 200, "note": 200, "what": 120},
            "max_assumptions": 10,
            "max_clarifications": 6,
            "max_unsupported": 6,
            "refusal": 300,
        },
    }
