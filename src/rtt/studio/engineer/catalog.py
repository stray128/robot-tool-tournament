"""What the Task Engineer is told about the Studio, and what the intake page is given.

Everything here is rendered from the frozen contract (:mod:`rtt.studio.types`): the object
templates with their bounded params, handles and state quantities; the primitives with what
they act on, their "how" params and what they observe; the quantities; the rules the checks
enforce; the action-to-primitive mapping; and the wrist and table limits. The prompt and the
intake page (:func:`web_catalog`) therefore cannot drift from the code that validates a plan.

The simulator's own conventions (track S2, ``docs/STUDIO_PRIMITIVES.md``) are stated next to the
contract: which way each twistable object works (:data:`TWIST_SENSE`), that a threaded member is
captive on its stub (no pick acts on it, :data:`NO_PICK`), the thread's lead-in
(:data:`THREAD_LEAD_EXTRA_MM`), the force and torque shares a controller allows itself
(:data:`FORCE_SHARE`, :data:`DEFAULT_PUSH_N`, :data:`TORQUE_SHARE`, :data:`WRIST_TORQUE_SHARE`) and
the fixed heights of a lid and a handwheel. ``tests/test_studio_engineer.py`` checks these numbers
against the template and primitive modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from rtt.effectors.types import ParamSpec
from rtt.studio.engineer.layout import CLEARANCE_MM
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
    "DEFAULT_PUSH_N",
    "FORCE_SHARE",
    "GOAL_EXTRA",
    "HANDLE_TEXT",
    "LID_HINGE_MM",
    "LIMIT_QUANTITIES",
    "MAPPING_ROWS",
    "NO_PICK",
    "PICKS",
    "THREAD_LEAD_EXTRA_MM",
    "TORQUE_SHARE",
    "TWIST_SENSE",
    "WRIST_TORQUE_SHARE",
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


def num_static(value: float) -> str:
    """A number for the prompt text: at most two decimals, no trailing zeros."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


#: What each handle is, in words (the contract names handles only).
HANDLE_TEXT: Final[Mapping[str, str]] = {
    "lid": "the panel itself (push it closed)",
    "handle": "the lip or handle at the free edge or front (hook it to pull open)",
    "drawer": "the drawer front (push it shut)",
    "head": "the latch head, a slot or a wing (twist it)",
    "member": "the cap, coupling or nut, captive on its stub (insert or twist it)",
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
#: The pick primitives (they lift the object they act on).
PICKS: Final[frozenset[PrimitiveKind]] = frozenset(
    {PrimitiveKind.GRASP, PrimitiveKind.SUCTION, PrimitiveKind.WRAP}
)

# --------------------------------------------------------------------------- the simulator's conventions (S2)

#: The direction that moves each twistable object away from where it starts (seen from above):
#: latches and valves start closed at a stop and open counter-clockwise; a thread tightens clockwise.
TWIST_SENSE: Final[Mapping[TemplateKind, str]] = {
    TemplateKind.QUARTER_TURN_LATCH: "ccw",
    TemplateKind.HANDWHEEL_VALVE: "ccw",
    TemplateKind.THREADED_JOINT: "cw",
}
#: Templates no pick (grasp, suction, wrap) acts on: a threaded member lives on its stub.
NO_PICK: Final[frozenset[TemplateKind]] = frozenset({TemplateKind.THREADED_JOINT})
#: The lead-in of a thread beyond its chamfer: the thread starts ``chamfer_mm`` + this far down.
THREAD_LEAD_EXTRA_MM: Final = 3.0
#: A push or an insert keeps its force under this share of ``max_force_n`` ...
FORCE_SHARE: Final = 0.8
#: ... or under this many newtons without one.
DEFAULT_PUSH_N: Final = 30.0
#: A twist keeps its roll torque under this share of ``max_torque_nm`` ...
TORQUE_SHARE: Final = 0.9
#: ... and under this share of the wrist roll's limit.
WRIST_TORQUE_SHARE: Final = 0.95
#: A lid's closed underside above the table (the simulator has no hinge-height parameter).
LID_HINGE_MM: Final = 120.0
#: A handwheel's mid-plane above the table.
WHEEL_HEIGHT_MM: Final = 110.0


def wrist_twist_limit() -> float:
    """The most a twist turns without a ``max_torque_nm`` (N m): 95 % of the wrist roll's limit."""
    return WRIST_TORQUE_SHARE * STUDIO_WRIST.roll_torque_nm


#: Template notes the contract leaves to the reader: how the simulator models each kind.
_TEMPLATE_NOTES: Final[Mapping[TemplateKind, str]] = {
    TemplateKind.HINGED_LID: (
        "A tote lid, a hatch, a cover or a cabinet door. orientation lid: a horizontal panel hinged "
        f"along its back edge, its closed underside {num_static(LID_HINGE_MM)} mm above the table (there is "
        f"no hinge-height parameter; {num_static(LID_HINGE_MM)} mm is a default bin's height, so a lid at a "
        "bin's position covers a bin of that height); it stays open only past vertical. orientation "
        "door: a vertical panel depth_mm wide and width_mm tall, 10 mm above the table, hinged on its "
        "left edge, opening toward its front. It starts closed; angle_deg is 0 closed, positive open."
    ),
    TemplateKind.DRAWER: (
        "A drawer in a cabinet or shelf unit. It starts closed; travel_mm is 0 closed, positive open."
    ),
    TemplateKind.QUARTER_TURN_LATCH: (
        "A panel fastener or cam lock on its own plate. It starts locked at its stop and opens "
        "counter-clockwise seen from above (twist ccw); angle_deg counts counter-clockwise from locked. "
        "Locking a latch that starts locked is not modelled."
    ),
    TemplateKind.THREADED_JOINT: (
        "A pipe coupling, cap or nut (the member) and the fixed stub, rod or neck it threads onto (the "
        "seat), in one object. The member starts resting loose on top of the stub, captive on the "
        "stub's axis: it cannot be picked up, carried or placed, so no pick acts on it. insert grips it "
        f"in place and pushes it down the lead-in to the thread start (chamfer_mm + "
        f"{num_static(THREAD_LEAD_EXTRA_MM)} mm down); twist clockwise seen from above runs it down the "
        "thread, engage_turns turns to seated, while the resisting torque rises from start_torque_nm to "
        "final_torque_nm. depth_mm counts down from the stub top; turns and angle_deg count clockwise "
        "from the start. Picking a loose member and starting it on a stub is not modelled."
    ),
    TemplateKind.HANDWHEEL_VALVE: (
        f"A valve or any round handwheel on a vertical stem, the wheel {num_static(WHEEL_HEIGHT_MM)} mm above "
        "the table. It starts closed at its stop and opens counter-clockwise seen from above (twist "
        "ccw); angle_deg counts counter-clockwise from closed. Closing a valve that starts closed is "
        "not modelled."
    ),
    TemplateKind.PORT_PLUG: (
        "A connector and its port, a peg and its hole, a vial and its holder: one object holds both, the "
        "plug standing free on the table 40 mm beside its socket (do not add the plug as a separate "
        "part). depth_mm is the plug tip's depth below the socket mouth, seated at the socket's depth_mm."
    ),
    TemplateKind.BIN: "A tote, tray, basket or box that picks drop into (fixed true unless it moves).",
    TemplateKind.PART: (
        "A rigid item: shape box or cylinder (cylinder: size_x_mm is the diameter, size_y_mm is ignored) "
        "with ref null; shape mesh only for an uploaded part, with ref upload:<id>."
    ),
}

#: How the simulator runs each primitive, where the contract's one line is not enough.
_PRIMITIVE_NOTES: Final[Mapping[PrimitiveKind, str]] = {
    PrimitiveKind.GRASP: "Not on a threaded member (it lives on its stub).",
    PrimitiveKind.HOOK_PULL: (
        "It pulls past the criterion (8 deg or 15 mm more) and takes a lid that must stay open past "
        "vertical when its stop allows."
    ),
    PrimitiveKind.TWIST: (
        "direction ccw opens a latch or a valve, cw tightens a thread; the other way meets the object's "
        "stop, so it only turns back an object an earlier twist turned. It stops at its criterion, at a "
        "torque_nm at_least criterion's torque (tighten until snug), or when the torque limit stalls "
        f"it: {num_static(100 * TORQUE_SHARE)} % of the step's max_torque_nm, else "
        f"{num_static(100 * WRIST_TORQUE_SHARE)} % of the wrist roll's {num_static(STUDIO_WRIST.roll_torque_nm)} N m."
    ),
    PrimitiveKind.INSERT: (
        "On a port_plug it picks the plug itself when the tool does not already hold it (a grasp with "
        "hold_after before it is optional) and pushes it home to the socket's depth_mm. On a "
        "threaded_joint it grips the member in place and pushes it to the thread start (depth_mm at most "
        f"chamfer_mm + {num_static(THREAD_LEAD_EXTRA_MM)} mm); a twist then runs it down. The push force stays "
        f"under {num_static(100 * FORCE_SHARE)} % of max_force_n, else {num_static(DEFAULT_PUSH_N)} N."
    ),
    PrimitiveKind.PUSH: (
        f"The push force stays under {num_static(100 * FORCE_SHARE)} % of max_force_n, else "
        f"{num_static(DEFAULT_PUSH_N)} N. A lid pushed shut from past vertical is let fall by its own weight "
        "from 80 deg, then pressed home."
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
        kinds = [k for k in info.objects if not (k in NO_PICK and kind in PICKS)]
        acts = "; ".join(f"{k.value} via {' or '.join(handles_for(info, k))}" for k in kinds)
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
            + f" Fidelity {tier}."
            + (f" {_PRIMITIVE_NOTES[kind]}" if kind in _PRIMITIVE_NOTES else ""),
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
                "object, or lift_mm, hold_s or in_target (a pick), or torque_nm at_least (a twist that tightens "
                "until snug). Force, torque at_most, slip, jam and penetration limits alone do not say when a "
                "step succeeded."
            ),
            (
                "- Windows: a criterion with both at_least and at_most leaves room for measurement: at least 1 mm "
                "(depth_mm, travel_mm, lift_mm, slip_mm, penetration_mm), 5 deg (angle_deg, tilt_deg, slip_deg), "
                "0.25 turns, 0.2 N m (torque_nm), 1 N (force_n, lateral_force_n) or 0.2 s (hold_s) between them; "
                "the flags in_target and jam are exempt. Use one bound when the other adds nothing."
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
                "- Directions: a twist on a quarter_turn_latch or a handwheel_valve is ccw (it opens them from "
                "their closed stop); a twist on a threaded_joint is cw (it tightens it). The other direction "
                "only turns back an object an earlier twist step turned the working way."
            ),
            (
                "- Threaded members: no pick (grasp, suction, wrap) acts on one, it lives on its stub; a twist "
                "on one comes after an insert of it (insert it to the thread start with hold_after false, then "
                "twist cw); an insert "
                f"pushes it no deeper than the thread start, chamfer_mm + {num(THREAD_LEAD_EXTRA_MM)} mm."
            ),
            (
                f"- Lids on bins: a lid's hinge height is fixed ({num(LID_HINGE_MM)} mm, its closed underside), so "
                f"a bin a lid covers (at the lid's position) is {num(LID_HINGE_MM)} mm tall; when the use case "
                "states another bin height, put that lid on that bin in unsupported (not modelled) instead of "
                "changing the height."
            ),
            (
                "- Reachable criteria: a lid's or door's angle within its max_angle_deg; a drawer's travel "
                "within its travel_mm; a latch's angle within its turn_deg; a valve's angle within its "
                "travel_deg; a threaded member's turns within its engage_turns, its angle within "
                f"engage_turns x 360 deg and its depth within chamfer_mm + {num(THREAD_LEAD_EXTRA_MM)} mm + "
                "engage_turns x pitch_mm (seated); a plug's depth within the socket's depth_mm; a pick's "
                "lift_mm and hold_s within the step's own lift_mm and hold_s params."
            ),
            (
                "- Torque: a twist needs the object's torque (a latch's or valve's torque_nm; a thread's torque "
                "where the step stops, rising from start_torque_nm at the thread start to final_torque_nm when "
                "seated) and any torque_nm at_least it asks for. That must fit what the twist allows itself, "
                f"{num(100 * TORQUE_SHARE)} % of the step's max_torque_nm, else {num(100 * WRIST_TORQUE_SHARE)} % "
                f"of the wrist roll's {torque} N m ({num(wrist_twist_limit())} N m), and a torque_nm at_most "
                "criterion."
            ),
            (
                "- Force: a hook_pull on a drawer needs its pull_force_n within the step's max_force_n; a push "
                "on a drawer (its pull_force_n) and an insert of a plug (its insert_force_n) push with at most "
                f"{num(100 * FORCE_SHARE)} % of max_force_n, else {num(DEFAULT_PUSH_N)} N."
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
            "shape); target the bin it goes into; never a threaded member"
        ),
    ),
    ("open a lid, hatch or door; pull a drawer or a handle", "hook_pull on its handle"),
    ("close a lid, door or drawer; press; slide shut", "push on the lid or drawer"),
    ("open or unlock a latch, open a valve, turn a handwheel open", "twist ccw on its head or wheel"),
    (
        "tighten, run down or screw on a cap, nut or coupling that sits on its thread",
        "insert it to the thread start, then twist cw",
    ),
    (
        "plug in, insert, seat a connector, peg or vial",
        "insert (it picks the plug itself; a grasp with hold_after true first is optional)",
    ),
    (
        (
            "close a valve or lock a latch that starts closed; pick a loose cap, nut or coupling and start "
            "it on a thread"
        ),
        (
            "not modelled: an unsupported item (nearest primitive twist or insert); plan the rest, a "
            "member resting on its stub where the use case allows it"
        ),
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
            f"- Every object stands on the table, within -{lim} to {lim} mm of its centre in x and y.",
            (
                "- Plan-view footprints (mm, centred on the pose, x and y before yaw_deg turns them): a bin "
                "length_mm x width_mm; a part size_x_mm x size_y_mm (a cylinder size_x_mm across); a latch's "
                "plate and a threaded joint's base max(1.6 x head_diameter_mm or diameter_mm, 40) square; a "
                "valve max(rim_diameter_mm, 70) x max(rim_diameter_mm, 50); a port its socket (about "
                "plug_diameter_mm + 2 x max(6, plug_diameter_mm / 2) + 2 x chamfer_mm across) with its plug "
                "standing 40 mm beyond it along +x; a drawer width_mm x depth_mm plus its front and handle toward "
                "-y, and it slides out toward -y by travel_mm; a lid width_mm x depth_mm with its hinge and back "
                "wall on the +y edge, swinging past vertical up to depth_mm x -cos(max_angle_deg) beyond it; a "
                "door a panel depth_mm long along x, hinged at x = -depth_mm / 2, swinging toward -y through the "
                "circular sector of radius depth_mm from closed to max_angle_deg."
            ),
            (
                f"- Layout: keep every footprint and every swing (a lid's, a door's, a drawer's travel) at least "
                f"{num(CLEARANCE_MM)} mm clear of every other object's. Only a lid at its bin's position (the lid "
                "covers the bin) may overlap. A part a step picks into a bin starts outside that bin and outside "
                "every lid and door; a latch or anything else on a cabinet stands beside the door, clear of its "
                "swing, not on it."
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


def _window_min() -> dict[str, float]:
    from rtt.studio.engineer.checks import WINDOW_MIN

    return dict(WINDOW_MIN)


def _layout_constants() -> dict[str, Any]:
    from rtt.studio.engineer import layout as lay

    return {
        "clearance_mm": lay.CLEARANCE_MM,
        "door_jamb_mm": list(lay.DOOR_JAMB_MM),
        "bar_diameter_mm": lay.BAR_DIAMETER_MM,
        "drawer_front_mm": lay.DRAWER_FRONT_MM,
        "knob_head_mm": lay.KNOB_HEAD_MM,
        "valve_body_mm": list(lay.VALVE_BODY_MM),
        "plate_min_mm": lay.LATCH_PLATE_MIN_MM,
        "port_wall_mm": lay.PORT_WALL_MM,
        "port_sides": lay.PORT_SIDES,
        "plug_gap_mm": lay.PLUG_GAP_MM,
        "lid_back_wall_mm": lay.LID_BACK_WALL_MM,
        "arc_step_deg": lay.ARC_STEP_DEG,
    }


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
        "sim": {
            "twist_sense": {k.value: v for k, v in TWIST_SENSE.items()},
            "no_pick": sorted(k.value for k in NO_PICK),
            "picks": sorted(k.value for k in PICKS),
            "thread_lead_extra_mm": THREAD_LEAD_EXTRA_MM,
            "force_share": FORCE_SHARE,
            "default_push_n": DEFAULT_PUSH_N,
            "torque_share": TORQUE_SHARE,
            "wrist_torque_share": WRIST_TORQUE_SHARE,
            "lid_hinge_mm": LID_HINGE_MM,
            "depth_tolerance_mm": 0.5,
            "window_min": dict(_window_min()),
            "layout": _layout_constants(),
        },
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
