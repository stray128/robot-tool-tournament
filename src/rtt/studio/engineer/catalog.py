"""What the Task Engineer is told about the Studio, and what the intake page is given.

Everything here is rendered from the frozen contract (:mod:`rtt.studio.types`): the object
templates with their bounded params, handles and state quantities; the primitives with what
they act on, their "how" params and what they observe; the quantities; the rules the checks
enforce; the action-to-primitive mapping; and the wrist and table limits. The prompt and the
intake page (:func:`web_catalog`) therefore cannot drift from the code that validates a plan.
They render the planner view of the contract (:func:`~rtt.studio.types.planner_templates` and its
siblings): a template, quantity or optional step param the Studio runs before the Engineer is
taught it stays out of the recorded prompt, tool and catalog until then.

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
from rtt.studio.engineer.layout import (
    CLEARANCE_MM,
    FINGER_ROOM_MM,
    FRONT_REACH_MM,
    INSIDE_MARGIN_MM,
    MACHINE_OVERLAP_MM,
    MACHINE_WALL_MM,
    PALM_LIMIT_MM,
    PANEL_CLAMP_OUT_MM,
    WORK_LIMIT_MM,
    panel_reach_mm,
)
from rtt.studio.types import (
    CONTRACT_VERSION,
    MAX_OBJECTS,
    MAX_STEP_S,
    MAX_STEPS,
    PRIMITIVES,
    PRODUCT_MODE_LABEL,
    STUDIO_WRIST,
    ObjectPose,
    ObjectTemplate,
    PrimitiveInfo,
    PrimitiveKind,
    StepConstraints,
    TemplateKind,
    get_template,
    planner_handles,
    planner_kinds,
    planner_params,
    planner_quantities,
    planner_templates,
)

__all__ = [
    "DEFAULT_PUSH_N",
    "FORCE_SHARE",
    "GOAL_EXTRA",
    "HANDLE_TEXT",
    "LID_HINGE_MM",
    "LID_OPEN_DEG",
    "LIMIT_QUANTITIES",
    "MAPPING_ROWS",
    "NO_PICK",
    "NO_PICK_HANDLES",
    "PICKS",
    "SET_BACK",
    "THREAD_LEAD_EXTRA_MM",
    "TOOL_ROWS",
    "TORQUE_SHARE",
    "TWIST_SENSE",
    "WRIST_TORQUE_SHARE",
    "goal_quantities",
    "handles_for",
    "lift_give_mm",
    "num",
    "render_limits",
    "render_mapping",
    "render_primitives",
    "render_quantities",
    "render_rules",
    "render_templates",
    "render_tools",
    "web_catalog",
]


def num_static(value: float) -> str:
    """A number for the prompt text: at most two decimals, no trailing zeros."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


#: What each handle is, in words (the contract names handles only).
HANDLE_TEXT: Final[Mapping[str, str]] = {
    "lid": "the panel itself (push it closed)",
    "handle": "the lip or handle at the free edge or front, or a sliding door's fin (hook it to pull open)",
    "drawer": "the drawer front (push it shut)",
    "head": "the latch head, a slot or a wing (twist it)",
    "member": "the cap, coupling, nut or clamp, captive on its stub or post (insert or twist it)",
    "seat": "the fixed threaded seat (no primitive acts on it)",
    "wheel": "the handwheel rim (twist it)",
    "plug": "the free plug or connector (grasp, insert or push it)",
    "socket": "the fixed socket (no primitive acts on it; the plug goes in)",
    "bin": "the open box (a destination for picks)",
    "part": "the part itself, or a tile on its bed (pick it, or press it)",
    "panel": "the panel on its rails (a pick lifts it and sets it back)",
    "button": "the button's face (push it in)",
    "door": "the sliding door (push it shut)",
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
    TemplateKind.PANEL_RAIL: "cw",  # its clamp is a threaded member
}
#: Templates no pick (grasp, suction, wrap) acts on: a threaded member lives on its stub.
NO_PICK: Final[frozenset[TemplateKind]] = frozenset({TemplateKind.THREADED_JOINT})
#: Handles no pick acts on: a threaded member (a panel's clamp) lives on its stub or post.
NO_PICK_HANDLES: Final[frozenset[str]] = frozenset({"member"})
#: Templates a pick lifts and sets back where they were (no target): a panel on its rails, a tile
#: on its bed.
SET_BACK: Final[frozenset[TemplateKind]] = frozenset({TemplateKind.PANEL_RAIL, TemplateKind.TILE_BED})
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
#: A pick into a bin under a lid needs that lid opened at least this far first (past vertical: a lid
#: below it falls shut when the tool lets go).
LID_OPEN_DEG: Final = 95.0
#: A handwheel's mid-plane above the table.
WHEEL_HEIGHT_MM: Final = 110.0

# Studio v2's job templates (``rtt.studio.templates`` and the primitives; the tests compare them)
#: A panel's clamp, the threaded member on its post: turns from its thread start to seated, its
#: thread's pitch and chamfer, and the torque at its thread start.
CLAMP_TURNS: Final = 3.0
CLAMP_PITCH_MM: Final = 1.25
CLAMP_CHAMFER_MM: Final = 1.0
CLAMP_START_NM: Final = 0.3
#: A seated clamp keeps turning at its clamp_torque_nm, so the wrist's own torque reaches only
#: this share of it [measured: 5.96 N m on a 6 N m clamp]; clamp_torque_nm says how tight it got.
CLAMP_WRIST_SHARE: Final = 0.95
#: A press on a button pushes at least this many times its actuation force (past its bottom-out).
BUTTON_PRESS_FACTOR: Final = 1.5
#: A push presses a part or a tile with this force (N) unless it asks for a force_n at_least ...
PRESS_N: Final = 10.0
#: ... and a force_n at_least is pressed to this much over it.
FORCE_GOAL_MARGIN: Final = 0.03
#: A tile pressed into its bed springs back up to this much when the press ends (mm) [measured:
#: 0.06 to 0.31 mm], so a press sinks it this much past the level it must keep.
TILE_REBOUND_MM: Final = 0.3
#: A tile cannot be pressed more than this below its level: the substrate stops it (mm).
TILE_OVER_MM: Final = 3.0
#: An insert pushes a member this much past its depth_mm at_least (``insert.goal_depth``) (mm) ...
INSERT_PAST_MM: Final = 0.5
#: A push shutting a sliding door reaches its closed stop within its time budget only this slowly
#: (mm/s) [measured: at 10 N friction after a pick; 60 to 100 mm/s left it 6 to 9 mm open].
SLIDE_PUSH_MAX_MM_S: Final = 40.0
#: The Studio's tools grip or seal a part with 31.8 to 92.2 N of their own [measured:
#: a 25 mm cup on a 500 g box, the 2 x 2 vacuum frame on a 2.2 kg tile, a 110 mm servo gripper on a
#: carton], so a max_force_n on a pick, when the use case gives one, is at least this (N).
PICK_FORCE_MIN_N: Final = 100.0
#: A pick's lift criterion leaves this much of its lift_mm param for the tool's give (mm) [measured: the
#: 2 x 2 vacuum frame gave 8.5 mm before a tile rose, at lift_mm 10 and 40] ...
PICK_GIVE_MM: Final = 10.0
#: ... and, for a heavier part, the give under its weight plus a margin: a vacuum frame gives
#: LIFT_GIVE_MM + LIFT_GIVE_MM_PER_KG per kg before the part rises [measured: both
#: vacuum frames, 2.2 to 10 kg, at lift_mm 40 and 60, within 0.2 mm (8.5 mm at 2.2 kg, 23 mm at 8 kg,
#: 27.9 mm at 10 kg); assumed for the other tools], and the margin is the 1.5 mm the 10 mm rule keeps
#: over its 8.5 mm [assumed] (:func:`lift_give_mm`).
LIFT_GIVE_MM: Final = 3.0
LIFT_GIVE_MM_PER_KG: Final = 2.5
LIFT_GIVE_MARGIN_MM: Final = 1.5
#: A pick's hold_s counts only while its part is up at least the larger of this (mm) and this share of
#: the step's lift_mm (``rtt.studio.primitives._pick``: ``lift_threshold_mm``).
HOLD_LIFT_MIN_MM: Final = 10.0
HOLD_LIFT_SHARE: Final = 0.5
#: A pinch on a sliding door's fin gives before the door moves [measured: 31.6 to 33.3 mm at 10 N],
#: so a slip_mm at_most on its hook pull leaves at least this much (mm).
SLIDE_SLIP_MIN_MM: Final = 40.0
#: A hook pull drags a sliding door at up to this many times its pull_force_n, so its force limit
#: (max_force_n, a force_n at_most) is at least that [measured: the 85 mm adaptive
#: gripper pulled a 10 N door at 23 to 26 N; the factor keeps about 4 N spare, assumed].
SLIDE_PULL_FACTOR: Final = 3.0


def lift_give_mm(mass_g: float) -> float:
    """How much of a pick's lift_mm the tool's give takes under a part of ``mass_g`` (mm), with the
    margin: :data:`PICK_GIVE_MM`, or :data:`LIFT_GIVE_MM` + :data:`LIFT_GIVE_MM_PER_KG` per kg +
    :data:`LIFT_GIVE_MARGIN_MM` when that is more."""
    return max(
        PICK_GIVE_MM, LIFT_GIVE_MM + LIFT_GIVE_MM_PER_KG * float(mass_g) / 1000.0 + LIFT_GIVE_MARGIN_MM
    )


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
    TemplateKind.PANEL_RAIL: (
        "A solar module or any large flat panel on two mounting rails along local y, with a clamp (a "
        f"13 mm hex member on a post) at the middle of its -y long edge, {num_static(PANEL_CLAMP_OUT_MM)} mm out. The panel "
        "starts resting on the rails, start_gap_mm short of their end stops at its +y edge; a pick (handle "
        "panel, target null) lifts it and sets it back on the rails, which stands for setting it down there "
        "(bringing it from a crate or stand is not modelled: list that in unsupported, nearest primitive "
        "suction), and gap_mm at_most says it sits against the stops. The clamp is a threaded member: "
        f"insert it to its thread start ({num_static(CLAMP_CHAMFER_MM + THREAD_LEAD_EXTRA_MM)} mm down), then "
        f"twist cw; it seats after {num_static(CLAMP_TURNS)} turns of {num_static(CLAMP_PITCH_MM)} mm pitch, its "
        f"torque rising from {num_static(CLAMP_START_NM)} N m to clamp_torque_nm. clamp_torque_nm (the state) is "
        "how tight it got: judge the tightening by clamp_torque_nm at_least (at most the param), not by the "
        "wrist's torque_nm, which stays just under it. The panel's centre and the clamp lie (width_mm / 2 + "
        f"{num_static(PANEL_CLAMP_OUT_MM)}) / 2 mm either side of the pose along local y."
    ),
    TemplateKind.TILE_BED: (
        "A floor or wall tile on a fresh mortar or adhesive bed. The tile starts laid on the bed, proud_mm above "
        "its finished level. A pick (handle part, target null) lifts it and lays it back on the bed (laying it "
        "from a stack is not modelled: unsupported, nearest primitive suction). A push (handle part) presses "
        "it: the bed gives at bed_yield_n plus bed_hardening_n_mm for every mm it sinks (the tile's weight "
        f"helps), keeps its set and lets the tile spring back up to {num_static(TILE_REBOUND_MM)} mm; it stops a "
        f"tile {num_static(TILE_OVER_MM)} mm below its level. Judge the press by height_err_mm at_most (how far "
        "the top ends from its level, either way) and tilt_deg at_most."
    ),
    TemplateKind.SPRING_BUTTON: (
        "A call button, a push button or a switch. mount wall: a plate standing on the table, its face through "
        "the pose and facing local -y (yaw_deg turns it), the button's centre height_mm up; push it with "
        "approach front. mount desk: a console height_mm tall, the button on top facing up; push it with "
        "approach top. A press past 70 % of its travel latches pressed to 1; the button springs back when the "
        "tool lets go, so travel_mm reads 0 at the end: judge it by pressed at_least 1."
    ),
    TemplateKind.SLIDING_DOOR: (
        "A machine with a guard door that slides sideways, loaded from above (a machine tool, a test chamber, "
        "an enclosure): the pose is the centre of its loading opening (width_mm along local y, the machine "
        "depth_mm along local x, its front facing local -x). The door starts closed over the opening and "
        "slides toward local -y to open (travel_mm 0 closed, positive open): hook_pull its handle (a fin on "
        "top, approach top) to open it, push it (handle door) to shut it. A fixture bin or a part inside the machine "
        "stands on the machine bed under the opening, inside its interior (see the layout rules)."
    ),
}

#: How the simulator runs each primitive, where the contract's one line is not enough.
_PRIMITIVE_NOTES: Final[Mapping[PrimitiveKind, str]] = {
    PrimitiveKind.GRASP: (
        "Not on a threaded member or a panel's clamp (they live on their stub or post). On a panel or a "
        "tile it lifts it and sets it back where it was (target null)."
    ),
    PrimitiveKind.SUCTION: "On a panel or a tile it lifts it and sets it back where it was (target null).",
    PrimitiveKind.HOOK_PULL: (
        "It pulls past the criterion (8 deg or 15 mm more) and takes a lid that must stay open past "
        "vertical when its stop allows. On a sliding door it pinches the fin and pulls it along the slide; "
        "the pinch gives about 33 mm before a 10 N door moves."
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
        f"chamfer_mm + {num_static(THREAD_LEAD_EXTRA_MM)} mm); a twist then runs it down. A panel's clamp "
        f"starts the same way ({num_static(CLAMP_CHAMFER_MM + THREAD_LEAD_EXTRA_MM)} mm). The push force stays "
        f"under {num_static(100 * FORCE_SHARE)} % of max_force_n, else {num_static(DEFAULT_PUSH_N)} N."
    ),
    PrimitiveKind.PUSH: (
        f"The push force stays under {num_static(100 * FORCE_SHARE)} % of max_force_n (and of a force_n at_most), "
        f"else {num_static(DEFAULT_PUSH_N)} N. A lid pushed shut from past vertical is let fall by its own weight "
        "from 80 deg, then pressed home. approach top pushes straight down; approach front tilts the wrist so "
        "the tool points straight at an upright face (a wall button, a drawer front, a door) and pushes along "
        f"it. A part or a tile is pressed with {num_static(PRESS_N)} N, or to a force_n at_least plus "
        f"{num_static(100 * FORCE_GOAL_MARGIN)} %; a button with at least {num_static(BUTTON_PRESS_FACTOR)} x its "
        "actuation_force_n."
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
    for t in planner_templates():
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
        kinds = [k for k in planner_kinds(info) if not (k in NO_PICK and kind in PICKS)]

        def via(k: TemplateKind, info: PrimitiveInfo = info, kind: PrimitiveKind = kind) -> str:
            hs = [h for h in handles_for(info, k) if not (kind in PICKS and h in NO_PICK_HANDLES)]
            return " or ".join(hs)

        acts = "; ".join(f"{k.value} via {via(k)}" for k in kinds)
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
            _param_table(planner_params(info)),
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_quantities() -> str:
    rows = ["| quantity | unit | range | meaning |", "|---|---|---|---|"]
    for q in planner_quantities():
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
                "0.25 turns, 0.2 N m (torque_nm, clamp_torque_nm), 1 N (force_n, lateral_force_n), 0.2 s (hold_s), "
                "1 mm (gap_mm) or 0.5 mm (height_err_mm) between them; the flags in_target, jam and pressed are "
                "exempt. Use one bound when the other adds nothing."
            ),
            (
                "- in_target needs a target bin. A step that places its object in a target releases it "
                "(hold_after false)."
            ),
            (
                "- Holding: hold_after true (grasp, suction, wrap, twist and insert only) means the step ends "
                "still holding its object; the next step must act on that same object by the same handle, and "
                "the last step must release (hold_after false)."
            ),
            (
                "- Picks: a tool grips or seals its part with 30 to 92 N of its own, whatever the part weighs, so "
                "leave max_force_n null on a pick unless the use case limits the squeeze (then at least "
                f"{num(PICK_FORCE_MIN_N)} N). A pick's lift_mm (its params.lift_mm, not a criterion) is how far the wrist rises; the tool gives under the "
                f"part's weight before the part itself rises, about {num(LIFT_GIVE_MM)} mm plus "
                f"{num(LIFT_GIVE_MM_PER_KG)} mm per kg (a vacuum frame: 8.5 mm at 2.2 kg, 28 mm at 10 kg). So a "
                f"lift_mm at_least leaves that give and {num(LIFT_GIVE_MARGIN_MM)} mm more of the step's lift_mm, and at "
                f"least {num(PICK_GIVE_MM)} mm: a 4 kg part with lift_mm at_least 10 needs lift_mm "
                f"{num(10 + lift_give_mm(4000))} or more. A hold_s at_least counts only while the part is up at least "
                f"{num(HOLD_LIFT_MIN_MM)} mm or {num(100 * HOLD_LIFT_SHARE)} % of the step's lift_mm, whichever is "
                "more, so its lift_mm leaves that plus the give."
            ),
            (
                "- Directions: a twist on a quarter_turn_latch or a handwheel_valve is ccw (it opens them from "
                "their closed stop); a twist on a threaded_joint or a panel_rail's clamp is cw (it tightens it). "
                "The other direction only turns back an object an earlier twist step turned the working way."
            ),
            (
                "- Threaded members (a threaded_joint's member, a panel_rail's clamp): no pick (grasp, suction, "
                "wrap) acts on one, it lives on its stub or post; a twist on one comes after an insert of it "
                "(insert it to the thread start with hold_after false, then twist cw). The insert takes it to the "
                f"thread start, chamfer_mm + {num(THREAD_LEAD_EXTRA_MM)} mm down "
                f"({num(CLAMP_CHAMFER_MM + THREAD_LEAD_EXTRA_MM)} mm for a clamp), and no deeper: it stops "
                f"{num(INSERT_PAST_MM)} mm past its depth_mm at_least, so ask depth_mm at_least the thread start less "
                f"{num(INSERT_PAST_MM)} mm, or no at_least (a shallower one leaves the thread uncaught)."
            ),
            (
                "- Panels and tiles: a pick on a panel_rail (handle panel) or a tile_bed (handle part) has target "
                "null: it lifts the panel or tile and sets it back on its rails or bed, and says it lifted it "
                "(lift_mm or hold_s at_least above 0). A tightened clamp is judged by clamp_torque_nm at_least, "
                f"at most the panel's clamp_torque_nm; a torque_nm at_least above {num(100 * CLAMP_WRIST_SHARE)} % of "
                "it is never reached."
            ),
            (
                "- Buttons: a push on a spring_button has approach front on a wall button and top on a desk one, "
                "and is judged by pressed at_least 1 (never travel_mm at_least: it springs back). It presses with "
                f"at least {num(BUTTON_PRESS_FACTOR)} x the button's actuation_force_n, so that must fit what the push "
                f"allows itself ({num(100 * FORCE_SHARE)} % of max_force_n, else {num(DEFAULT_PUSH_N)} N): max_force_n "
                f"at least {num(BUTTON_PRESS_FACTOR / FORCE_SHARE)} x actuation_force_n. The way in front of a wall "
                f"button's face, its plate's width wide and {num(FRONT_REACH_MM)} mm out, stays clear: no door (open "
                "or closed), bin or part stands in it."
            ),
            (
                "- Reachable from the start: every criterion must be reachable from where its object starts. "
                "penetration_mm, slip_mm, slip_deg, jam and lateral_force_n only bound how a step went: give them "
                "at_most, never at_least. A pick sets a panel or a tile back where it was, so a panel's gap_mm "
                "stays at its start_gap_mm (set start_gap_mm to the gap the use case allows) and a tile's "
                "height_err_mm at its proud_mm until a push presses it."
            ),
            (
                "- Claims: an assumption or a label that says where something stands (inside the machine, under "
                "its opening) must match its pose; code checks that what one puts inside a machine stands inside "
                "it."
            ),
            (
                "- Tiles: a push that presses a tile to height_err_mm at_most h asks for force_n at_least F, where F "
                "sinks it past that level: at least bed_yield_n + bed_hardening_n_mm x (proud_mm - h + "
                f"{num(TILE_REBOUND_MM)}) - the tile's weight (mass_g x 9.81 / 1000 N), and at most bed_yield_n + "
                "bed_hardening_n_mm x (proud_mm + h) - its weight; the push presses to F + "
                f"{num(100 * FORCE_GOAL_MARGIN)} %, which must fit {num(100 * FORCE_SHARE)} % of max_force_n."
            ),
            (
                "- Machines (sliding_door): a pick out of the machine or into a bin inside it comes after a "
                "hook_pull of its door with travel_mm at_least enough to uncover that object, width_mm / 2 + "
                f"{num(MACHINE_OVERLAP_MM)} + d + {num(CLEARANCE_MM)} mm where d is how far the object's footprint "
                "reaches from the pose toward local -y (the way the door slides), within the door's travel_mm, and "
                "no push shuts the door in between. A "
                f"hook_pull on the door bounds slip_mm at {num(SLIDE_SLIP_MIN_MM)} mm or more, if at all; its "
                f"max_force_n (and any force_n at_most) is at least {num(SLIDE_PULL_FACTOR)} x pull_force_n (a gripper "
                "drags a 10 N door at 23 to 26 N), and a push shutting it pushes with at most "
                f"{num(100 * FORCE_SHARE)} % of max_force_n (else {num(DEFAULT_PUSH_N)} N), at speed_mm_s "
                f"{num(SLIDE_PUSH_MAX_MM_S)} or less (faster, it runs out of time short of the stop). A part a step "
                f"picks inside a machine leaves a gripper's open fingers {num(FINGER_ROOM_MM)} mm around it: from "
                "every other bin or part inside, from the walls and the opening's +y edge (within depth_mm / 2 - "
                f"{num(MACHINE_WALL_MM + FINGER_ROOM_MM)} mm of the pose along local x, at most width_mm / 2 - "
                f"{num(FINGER_ROOM_MM)} mm toward local +y), and from the open door's edge (its hook_pull's travel_mm "
                f"at_least uses {num(FINGER_ROOM_MM)} in place of {num(CLEARANCE_MM)} above)."
            ),
            (
                f"- Lids on bins: a lid's hinge height is fixed ({num(LID_HINGE_MM)} mm, its closed underside), so "
                f"a bin a lid covers (at the lid's position) is {num(LID_HINGE_MM)} mm tall; when the use case "
                "states another bin height, put that lid on that bin in unsupported (not modelled) instead of "
                "changing the height. A lid with feature lip is opened by a hook slid under its free edge, so it "
                "overhangs its bin by feature_mm on every side (width_mm at least the bin's length_mm + 2 x "
                "feature_mm, depth_mm at least the bin's width_mm + 2 x feature_mm); a lid with feature handle "
                "may match its bin. A pick (grasp, suction, wrap) into a bin under a lid comes after a hook_pull "
                f"of that lid with angle_deg at_least {num(LID_OPEN_DEG)} (past vertical: a lid opened less falls "
                "shut when the tool lets go), and no push closes the lid between that hook_pull and the pick."
            ),
            (
                "- Reachable criteria: a lid's or door's angle within its max_angle_deg; a drawer's travel "
                "within its travel_mm; a latch's angle within its turn_deg; a valve's angle within its "
                "travel_deg; a threaded member's turns within its engage_turns, its angle within "
                f"engage_turns x 360 deg and its depth within chamfer_mm + {num(THREAD_LEAD_EXTRA_MM)} mm + "
                "engage_turns x pitch_mm (seated); a plug's depth within the socket's depth_mm; a pick's "
                "lift_mm and hold_s within the step's own lift_mm and hold_s params; a sliding door's travel "
                f"within its travel_mm; a panel clamp's turns within {num(CLAMP_TURNS)} and its depth within "
                f"{num(CLAMP_CHAMFER_MM + THREAD_LEAD_EXTRA_MM + CLAMP_TURNS * CLAMP_PITCH_MM)} mm (seated)."
            ),
            (
                "- Torque: a twist needs the object's torque (a latch's or valve's torque_nm; a thread's torque "
                "where the step stops, rising from start_torque_nm at the thread start to final_torque_nm when "
                "seated, a clamp's to clamp_torque_nm) and any torque_nm or clamp_torque_nm at_least it asks for. That must fit what the twist allows itself, "
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
    (
        "open a lid, hatch or door; pull a drawer or a handle; slide a machine's door open",
        "hook_pull on its handle",
    ),
    ("close a lid, door or drawer; slide shut", "push on the lid, drawer or door"),
    (
        "press a button, call a lift, hit a switch",
        "push on the button (approach front for a wall button, top for a desk one), pressed at_least 1",
    ),
    (
        "lay a tile, set a panel on its rails or mounts",
        "suction or grasp on the tile or panel (target null: it is lifted and set back on its bed or rails)",
    ),
    ("tamp, press or bed a tile level", "push on the tile with force_n at_least and height_err_mm at_most"),
    (
        "tighten a clamp, a mounting nut or a bolt on a panel's rail",
        "insert the panel's clamp to its thread start, then twist cw with clamp_torque_nm at_least",
    ),
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
                "- The wrist hangs over one table: it moves in x, y and z, turns about the vertical axis, pitches "
                f"the tool up to {num(w.pitch_range_deg)} deg from straight down (at most {num(w.pitch_torque_nm)} N m "
                f"and {num(w.pitch_speed_dps)} deg/s) and rolls the tool continuously (at most "
                f"{num(w.roll_torque_nm)} N m and {num(w.roll_speed_max_deg_s)} deg/s). Picks, twists and inserts "
                "work with the tool straight down and carry a part level; a push with approach front and a pinch "
                "of a lid's lip tilt it."
            ),
            f"- Every object stands on the table, within -{lim} to {lim} mm of its centre in x and y.",
            (
                f"- Reach: the palm travels -{num(PALM_LIMIT_MM)} to {num(PALM_LIMIT_MM)} mm in x and y; keep every "
                f"place a tool works within -{num(WORK_LIMIT_MM)} to {num(WORK_LIMIT_MM)} mm. A panel_rail's pose "
                f"stays within -({num(WORK_LIMIT_MM)} - (width_mm / 2 + {num(PANEL_CLAMP_OUT_MM)}) / 2) to +(that) in x "
                f"and y ({num(WORK_LIMIT_MM - panel_reach_mm(600))} mm for a 600 mm wide panel), so its centre and its "
                "clamp are both reachable at any yaw. A wall "
                "button's face is pressed by a tool tilted to point at it, which puts the wrist up to "
                f"{num(FRONT_REACH_MM)} mm in front of the face: stand the button so its face looks toward the table's "
                f"centre and the point {num(FRONT_REACH_MM)} mm in front of it is within reach. Every handle, lip or "
                "free edge a step must reach stays within reach along its whole motion: a lid's free edge closed "
                "and at its stop, a door's free edge and handle along its swing to max_angle_deg, a drawer's handle "
                "out to its travel_mm, a sliding door's handle at both ends of its travel_mm, a port's plug."
            ),
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
                "circular sector of radius depth_mm from closed to max_angle_deg; a panel_rail length_mm along x "
                "and, along y, from (width_mm / 2 + 25) / 2 + 20 mm on the -y side to (width_mm / 2 + 25) / 2 + "
                "width_mm / 2 + start_gap_mm + 20 mm on the +y side; a tile_bed tile_mm + 40 mm square; a wall "
                "button its plate, max(160, diameter_mm + 80) along x and 20 mm behind its face (local +y), a desk "
                "button its console max(120, diameter_mm + 40) square; a sliding_door depth_mm + 8 mm along x (its "
                "front toward -x) and width_mm + 40 mm + travel_mm along y, from width_mm / 2 + 20 on the +y side "
                "(the open door parks toward -y)."
            ),
            (
                f"- Layout: keep every footprint and every swing (a lid's, a door's, a drawer's travel) at least "
                f"{num(CLEARANCE_MM)} mm clear of every other object's. Only a lid at its bin's position (the lid "
                "covers the bin) and what stands inside a machine may overlap: a bin (a fixture) or a part inside "
                f"a sliding_door's interior, its footprint within depth_mm - {num(2 * (MACHINE_WALL_MM + INSIDE_MARGIN_MM))} "
                f"mm along local x and width_mm - {num(2 * INSIDE_MARGIN_MM)} mm along local y about the pose, and "
                f"at most height_mm - {num(MACHINE_WALL_MM + INSIDE_MARGIN_MM)} mm tall (things inside keep "
                f"{num(CLEARANCE_MM)} mm from each other). A part a step picks into a bin starts outside that bin "
                "and outside every lid and door; a latch or anything else on a cabinet stands beside the door, "
                "clear of its swing, not on it."
            ),
            f"- A step lasts at most {num(MAX_STEP_S)} s, so choose speeds that finish the motion in time.",
        ]
    )


#: The kinds of tool the physics search tries (``rtt.studio.search.default_candidates``), in words.
#: The model is told what exists, never which one wins: the search measures that.
TOOL_ROWS: Final[tuple[str, ...]] = (
    "two- and three-finger grippers (parallel and adaptive, with stock, padded or part-shaped jaws)",
    "dexterous hands (four and five fingers) and soft arms",
    (
        "suction cups (10 to 25 mm) and vacuum area frames (2 x 2 cups over 300 mm, 2 x 3 cups over 600 mm) "
        "for large flat faces"
    ),
    "a nut runner: a 13 mm hex socket turned by the wrist roll through a slipping clutch (insert and twist only)",
    "a tamper pad: a flat 200 mm pad on a spring stage that slips at its force (push only)",
    "designed tools from the effector library",
)


def render_tools() -> str:
    lines = [
        (
            "You plan the task, not the tool. The physics search tries every step with every tool that can do "
            "its primitive and recommends a setup from what the simulation measures, so never name, choose or "
            "assume a tool in the plan, and do not shape the plan around one. It tries:"
        ),
        "",
    ]
    lines += [f"- {row}" for row in TOOL_ROWS]
    return "\n".join(lines)


# --------------------------------------------------------------------------- web


def _param_json(p: ParamSpec, *, optional: bool = False) -> dict[str, Any]:
    """One param row; ``optional``: a step may leave it out (it is then its default, and a step at
    its default drops it, so the task id is the same)."""
    return {
        "name": p.name,
        "kind": p.kind,
        "description": p.description,
        "unit": p.unit,
        "low": p.low,
        "high": p.high,
        "choices": list(p.choices),
        "default": p.default,
        "optional": optional,
    }


def _step_params_json(info: PrimitiveInfo) -> list[dict[str, Any]]:
    optional = {p.name for p in info.optional}
    return [_param_json(p, optional=p.name in optional) for p in planner_params(info)]


def _library_parts() -> tuple[str, ...]:
    from rtt.studio.engineer.checks import library_part_names

    return library_part_names()


def _window_min() -> dict[str, float]:
    from rtt.studio.engineer.checks import WINDOW_MIN, WINDOW_MIN_V2

    return {**WINDOW_MIN, **WINDOW_MIN_V2}  # the page validates with rule set v2


def _v2_rules() -> dict[str, Any]:
    from rtt.studio.engineer import checks

    return {"limit_only": list(checks._LIMIT_ONLY), "set_back_slack_mm": checks.SET_BACK_SLACK_MM}


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
        "panel_rail_past_mm": lay.PANEL_RAIL_PAST_MM,
        "panel_stop_mm": lay.PANEL_STOP_MM,
        "panel_post_mm": lay.PANEL_POST_MM,
        "panel_clamp_out_mm": lay.PANEL_CLAMP_OUT_MM,
        "tile_border_mm": lay.TILE_BORDER_MM,
        "button_plate_mm": list(lay.BUTTON_PLATE_MM),
        "button_console_mm": list(lay.BUTTON_CONSOLE_MM),
        "machine_wall_mm": lay.MACHINE_WALL_MM,
        "machine_leaf_mm": lay.MACHINE_LEAF_MM,
        "machine_gap_mm": lay.MACHINE_GAP_MM,
        "machine_overlap_mm": lay.MACHINE_OVERLAP_MM,
        "inside_margin_mm": lay.INSIDE_MARGIN_MM,
        "finger_room_mm": lay.FINGER_ROOM_MM,
        "work_limit_mm": lay.WORK_LIMIT_MM,
        "front_reach_mm": lay.FRONT_REACH_MM,
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
            for t in planner_templates()
        ],
        "primitives": [
            {
                "kind": k.value,
                "description": PRIMITIVES[k].description,
                "fidelity": PRIMITIVES[k].fidelity.value,
                "caveat": PRIMITIVES[k].caveat,
                "objects": [o.value for o in planner_kinds(PRIMITIVES[k])],
                "handles": list(planner_handles(PRIMITIVES[k])),
                "targets": [o.value for o in PRIMITIVES[k].targets],
                "params": _step_params_json(PRIMITIVES[k]),
                "quantities": list(PRIMITIVES[k].quantities),
                "uses_roll": PRIMITIVES[k].uses_roll,
                "may_hold": PRIMITIVES[k].may_hold,
            }
            for k in PrimitiveKind
        ],
        "quantities": [
            {"name": q.name, "unit": q.unit, "low": q.low, "high": q.high, "description": q.description}
            for q in planner_quantities()
        ],
        "handles": dict(HANDLE_TEXT),
        "library_parts": list(_library_parts()),
        "sim": {
            "twist_sense": {k.value: v for k, v in TWIST_SENSE.items()},
            "no_pick": sorted(k.value for k in NO_PICK),
            "no_pick_handles": sorted(NO_PICK_HANDLES),
            "set_back": sorted(k.value for k in SET_BACK),
            "picks": sorted(k.value for k in PICKS),
            "thread_lead_extra_mm": THREAD_LEAD_EXTRA_MM,
            "force_share": FORCE_SHARE,
            "default_push_n": DEFAULT_PUSH_N,
            "torque_share": TORQUE_SHARE,
            "wrist_torque_share": WRIST_TORQUE_SHARE,
            "lid_hinge_mm": LID_HINGE_MM,
            "lid_open_deg": LID_OPEN_DEG,
            "clamp": {
                "turns": CLAMP_TURNS,
                "pitch_mm": CLAMP_PITCH_MM,
                "chamfer_mm": CLAMP_CHAMFER_MM,
                "start_nm": CLAMP_START_NM,
                "wrist_share": CLAMP_WRIST_SHARE,
            },
            "button_press_factor": BUTTON_PRESS_FACTOR,
            "press_n": PRESS_N,
            "force_goal_margin": FORCE_GOAL_MARGIN,
            "insert_past_mm": INSERT_PAST_MM,
            **_v2_rules(),
            "tile_rebound_mm": TILE_REBOUND_MM,
            "tile_over_mm": TILE_OVER_MM,
            "slide_slip_min_mm": SLIDE_SLIP_MIN_MM,
            "slide_push_max_mm_s": SLIDE_PUSH_MAX_MM_S,
            "pick_force_min_n": PICK_FORCE_MIN_N,
            "pick_give_mm": PICK_GIVE_MM,
            "slide_pull_factor": SLIDE_PULL_FACTOR,
            "lift_give_mm": LIFT_GIVE_MM,
            "lift_give_mm_per_kg": LIFT_GIVE_MM_PER_KG,
            "lift_give_margin_mm": LIFT_GIVE_MARGIN_MM,
            "hold_lift_min_mm": HOLD_LIFT_MIN_MM,
            "hold_lift_share": HOLD_LIFT_SHARE,
            "depth_tolerance_mm": 0.5,
            "window_min": dict(_window_min()),
            "layout": _layout_constants(),
        },
        "tools": list(TOOL_ROWS),
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
