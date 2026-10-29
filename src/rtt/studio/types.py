"""Frozen data contract of Effector Lab Studio: ``rtt-effectors/2`` (product mode, not pre-registered).

The per-part tournament (``rtt-effectors/1``, :mod:`rtt.effectors`) asks which effector should
hold one part. The Studio asks a wider question: which **setup** (one effector, a two-tool head,
or a tool-change sequence) completes a **use case** made of several interactions with real
objects. This module is that contract. It imports numpy and the v1 contract only, so every
builder track can use it cheaply (MuJoCo is imported by :mod:`rtt.studio.scene`).

* :class:`TaskSpec`: one use case: title, description, the scene's objects
  (:class:`ObjectInstance`, each an :class:`ObjectTemplate` kind with bounded params), the
  ordered :class:`Step` list, assumptions, clarifying questions and what is not simulated.
* :class:`Step`: one interaction: a :class:`PrimitiveKind` acting on an object (and handle),
  bounded primitive params, :class:`StepConstraints`, :class:`SuccessCriterion` rows and
  order dependencies.
* :class:`ObjectTemplate` and its registry (:func:`get_template`): the parametric object
  kinds (hinged lid, drawer, latch, threaded joint, handwheel valve, port and plug, bin, part;
  Studio v2 adds a panel on mounting rails, a tile on a mortar bed, a push button and a sliding
  machine door) with their param spaces, handles, state quantities and fidelity tier. Physics
  builders for each kind live in :mod:`rtt.studio.templates`.
* **What the Task Engineer is offered** (:func:`planner_templates`, :func:`planner_kinds`,
  :func:`planner_handles`, :func:`planner_quantities`, :func:`planner_params`): its prompt, tool
  schema and web catalog are recorded and hashed, so a template, quantity or optional step param
  added later runs in the Studio at once but reaches the Engineer only when it is taught it
  (``planner=False``, or an optional param left out of :attr:`PrimitiveInfo.taught`, until then).
  Studio v2 taught it the four job templates, their quantities and push's ``approach``.
* :class:`Setup`: what gets recommended: ``single``, ``multi_head`` (two effectors on one
  bracket) or ``tool_change`` (an effector per step, a changeover time and a mass swap), and
  :meth:`Setup.segments`, the episodes a full task runs as.
* :class:`StepOutcome`, :class:`TaskResult`, :class:`SetupResult`: scored results with per-step
  outcomes, full-task completion, cycle time including changes, complexity and certificates.
* :class:`BuildPackManifest`: the manifest of the files that build the recommended setup.
* :func:`task_plan_json_schema` and :func:`submit_task_plan_tool`: the strict, bounded JSON
  schema of the Task Engineer's forced tool call; :func:`task_spec_from_tool_args` parses it
  and reports every problem at once (:class:`TaskSpecError`).

Units: params, poses and criteria are in the units their names or :class:`ParamSpec` rows
say (mm, deg, N, N m, s), because that is what the Task Engineer reads and writes. Physics code
converts to SI. Every result carries :data:`PRODUCT_MODE_LABEL`.
"""

from __future__ import annotations

import enum
import hashlib
import math
import numbers
import re
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal

from rtt.effectors.types import (
    ALLOWED_LICENSES,
    PRODUCT_MODE_LABEL,
    EffectorSpec,
    Fidelity,
    FrozenDict,
    ParamSpec,
    _canon,
    _thaw,
    canonical_json,
    find_brand_tokens,
    params_json_schema,
)

__all__ = [
    "ALLOWED_LICENSES",
    "CONTRACT_VERSION",
    "CRITERION_TOLERANCE",
    "MAX_OBJECTS",
    "MAX_STEPS",
    "MAX_STEP_S",
    "MAX_TOOLS",
    "PRESENT_S",
    "PRIMITIVES",
    "PRODUCT_MODE_LABEL",
    "QUANTITIES",
    "SLIDE_KINDS",
    "STEP_FAILURE_MODES",
    "STUDIO_WRIST",
    "TASK_PLAN_TOOL",
    "THREAD_KINDS",
    "TOOL_CHANGE_S",
    "V1_CONTRACT",
    "BomLine",
    "BracketSpec",
    "BuildPackFile",
    "BuildPackLink",
    "BuildPackManifest",
    "CertificateRef",
    "ChangeKind",
    "Clarification",
    "ComplexityMetrics",
    "FidelityTier",
    "HeadMount",
    "ObjectInstance",
    "ObjectPose",
    "ObjectTemplate",
    "PrimitiveInfo",
    "PrimitiveKind",
    "Quantity",
    "Segment",
    "Setup",
    "SetupKind",
    "SetupResult",
    "Step",
    "StepConstraints",
    "StepOutcome",
    "SuccessCriterion",
    "TaskResult",
    "TaskSpec",
    "TaskSpecError",
    "TemplateKind",
    "ToolChanger",
    "UnsupportedItem",
    "WristLimits",
    "buildpack_manifest_json_schema",
    "get_template",
    "planner_handles",
    "planner_kinds",
    "planner_params",
    "planner_quantities",
    "planner_templates",
    "register_template",
    "submit_task_plan_tool",
    "task_plan_json_schema",
    "task_spec_from_tool_args",
    "templates",
    "tier_of_effector",
    "worst_tier",
]

CONTRACT_VERSION: Final = "rtt-effectors/2"
#: The per-part contract this one extends; its code and records are unchanged.
V1_CONTRACT: Final = "rtt-effectors/1"
#: The Task Engineer's forced tool.
TASK_PLAN_TOOL: Final = "submit_task_plan"
MAX_STEPS: Final = 12
MAX_OBJECTS: Final = 8
#: A tool-change setup holds at most this many effectors (a multi-tool head holds exactly 2).
MAX_TOOLS: Final = 4
#: Longest scripted step a controller may plan (s).
MAX_STEP_S: Final = 30.0
#: Changeover time of an automatic tool changer (s) [assumed: sources 5–10 s; one changer measured
#: 7.33 s], the same for every candidate.
TOOL_CHANGE_S: Final = 7.5
#: Time for a multi-tool head to present its other tool (wrist re-orientation) (s) [assumed].
PRESENT_S: Final = 1.5

_SLUG = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REF = re.compile(r"^(library|upload|scan):[a-z0-9][a-z0-9_.-]{0,63}$")
_SPDX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
_TEXT_MAX = {"title": 80, "description": 600, "label": 60, "line": 200, "note": 200}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _slug(what: str, value: object) -> str:
    if not isinstance(value, str) or not _SLUG.match(value):
        raise ValueError(f"{what} must be a short snake_case id ({_SLUG.pattern}), got {value!r}")
    return value


def _text(what: str, value: object, max_len: int, *, required: bool = True) -> str:
    """A display string: bounded, brand-free (on screen), non-empty when ``required``."""
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    if required and not value.strip():
        raise ValueError(f"{what} must not be empty")
    if len(value) > max_len:
        raise ValueError(f"{what} is longer than {max_len} characters")
    hits = find_brand_tokens(value)
    if hits:
        raise ValueError(f"{what} names a brand ({', '.join(hits)}); use generic words")
    return value


def _number(what: str, value: object, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    v = float(value)
    if (low is not None and v < low) or (high is not None and v > high):
        raise ValueError(f"{what} must be in [{low:g}, {high:g}], got {v:g}")
    return v


def _optional(what: str, value: object, low: float, high: float) -> float | None:
    return None if value is None else _number(what, value, low, high)


# --------------------------------------------------------------------------- enums and tiers


class PrimitiveKind(enum.StrEnum):
    """Primitive skills: scripted controllers, the same wrist limits for every candidate."""

    GRASP = "grasp"
    SUCTION = "suction"
    WRAP = "wrap"
    HOOK_PULL = "hook_pull"
    TWIST = "twist"
    INSERT = "insert"
    PUSH = "push"


class TemplateKind(enum.StrEnum):
    """Parametric object and scene templates (MuJoCo builders in :mod:`rtt.studio.templates`)."""

    HINGED_LID = "hinged_lid"
    DRAWER = "drawer"
    QUARTER_TURN_LATCH = "quarter_turn_latch"
    THREADED_JOINT = "threaded_joint"
    HANDWHEEL_VALVE = "handwheel_valve"
    PORT_PLUG = "port_plug"
    BIN = "bin"
    PART = "part"
    # Studio v2: the job templates (solar install, tile setting, call button, machine tending)
    PANEL_RAIL = "panel_rail"
    TILE_BED = "tile_bed"
    SPRING_BUTTON = "spring_button"
    SLIDING_DOOR = "sliding_door"


#: Kinds the primitives drive along a straight slide (a drawer pulled out, a door slid sideways):
#: the same pull, push and observed travel (``pull_x``/``pull_y``, ``travel_m``, ``handle``...).
SLIDE_KINDS: Final[frozenset[TemplateKind]] = frozenset({TemplateKind.DRAWER, TemplateKind.SLIDING_DOOR})
#: Kinds with a threaded member the twist and insert primitives turn and start (a threaded joint,
#: the clamp of a panel on its rails): see :func:`rtt.studio.templates.thread_info`.
THREAD_KINDS: Final[frozenset[TemplateKind]] = frozenset(
    {TemplateKind.THREADED_JOINT, TemplateKind.PANEL_RAIL}
)


class SetupKind(enum.StrEnum):
    SINGLE = "single"  # one effector does every step
    MULTI_HEAD = "multi_head"  # two effectors on one bracket; the wrist presents the active one
    TOOL_CHANGE = "tool_change"  # an effector per step, swapped by a tool changer


class ChangeKind(enum.StrEnum):
    """What happens before a :class:`Segment`."""

    NONE = "none"  # the first segment
    PRESENT = "present"  # a multi-tool head presents its other tool (:data:`PRESENT_S`)
    TOOL_CHANGE = "tool_change"  # the tool changer swaps effectors (``ToolChanger.changeover_s``)


class FidelityTier(enum.StrEnum):
    """How much to trust a step's physics, printed next to every step result.

    ``rigid``: a rigid multibody model with a scripted controller. ``approximation``: a
    labelled approximation (suction seal, soft arm, threads, latches, valves): a screen, not a
    promise. ``not_simulated``: no physics model yet (deformables, or a primitive whose fidelity
    gate failed); such a step is never scored.
    """

    RIGID = "rigid"
    APPROXIMATION = "approximation"
    NOT_SIMULATED = "not_simulated"


_TIER_ORDER = {FidelityTier.RIGID: 0, FidelityTier.APPROXIMATION: 1, FidelityTier.NOT_SIMULATED: 2}


def worst_tier(*tiers: FidelityTier | str) -> FidelityTier:
    """The least trustworthy of ``tiers`` (``rigid`` when none is given)."""
    out = FidelityTier.RIGID
    for t in tiers:
        t = FidelityTier(t)
        if _TIER_ORDER[t] > _TIER_ORDER[out]:
            out = t
    return out


def tier_of_effector(fidelity: Fidelity | str) -> FidelityTier:
    """The Studio tier of a v1 effector family's fidelity (``core`` and ``rigid`` are rigid)."""
    return FidelityTier.APPROXIMATION if Fidelity(fidelity) is Fidelity.APPROXIMATION else FidelityTier.RIGID


# --------------------------------------------------------------------------- quantities


@dataclass(frozen=True)
class Quantity:
    """A measured quantity a success criterion can bound (``unit`` as written, e.g. ``deg``)."""

    name: str
    unit: str
    low: float
    high: float
    description: str
    planner: bool = True  # offered to the Task Engineer (module docstring)


def _q(
    name: str, unit: str, low: float, high: float, description: str, *, planner: bool = True
) -> tuple[str, Quantity]:
    return name, Quantity(name, unit, low, high, description, planner)


#: Every quantity a :class:`SuccessCriterion` may name. Template state quantities are read
#: from the object (:attr:`ObjectTemplate.quantities`); the others are observed by the step's
#: success predicate (:attr:`PrimitiveInfo.quantities`).
QUANTITIES: Final[Mapping[str, Quantity]] = MappingProxyType(
    dict(
        [
            _q(
                "angle_deg",
                "deg",
                -1440.0,
                1440.0,
                "joint angle of the object from its start (lid, latch, valve)",
            ),
            _q(
                "travel_mm",
                "mm",
                -1000.0,
                1000.0,
                "slide travel of the object from its start (drawer, sliding door, button)",
            ),
            _q("depth_mm", "mm", 0.0, 200.0, "insertion depth of a plug or threaded member in its seat"),
            _q("turns", "turn", -20.0, 20.0, "turns of a threaded member from its start"),
            _q("lift_mm", "mm", -50.0, 500.0, "rise of a part above where it started"),
            _q("hold_s", "s", 0.0, 30.0, "time the part stayed lifted"),
            _q("in_target", "flag", 0.0, 1.0, "1 when the part rests inside the destination"),
            _q("tilt_deg", "deg", 0.0, 180.0, "tilt of a part from its starting attitude"),
            _q("slip_deg", "deg", 0.0, 1440.0, "largest rotation slip between the tool and the object"),
            _q("slip_mm", "mm", 0.0, 500.0, "largest slip of the tool along the object's feature"),
            _q("force_n", "N", 0.0, 1000.0, "peak force the tool applied to the object"),
            _q("lateral_force_n", "N", 0.0, 1000.0, "peak force across the insertion axis"),
            _q("torque_nm", "N m", 0.0, 100.0, "peak torque the wrist applied about the tool axis"),
            _q("jam", "flag", 0.0, 1.0, "1 when an insertion stalled short of its depth"),
            # Since studio v2 the runner measures it over every contact pair but the designed ones,
            # not only tool and object (docs/STUDIO.md 3.3); the wording says so since the v2
            # re-record (it is part of the Task Engineer's tool schema, whose hash keys the recording).
            _q(
                "penetration_mm",
                "mm",
                0.0,
                100.0,
                "deepest penetration between any two bodies, designed contacts excepted (physics validity)",
            ),
            # Studio v2 job templates
            _q("gap_mm", "mm", 0.0, 1000.0, "gap between a panel's edge and its rail stops"),
            _q(
                "clamp_torque_nm",
                "N m",
                0.0,
                100.0,
                "the highest torque a clamp reached as it was driven home (its rated torque once seated)",
            ),
            _q(
                "height_err_mm",
                "mm",
                0.0,
                1000.0,
                "how far a tile's top stands from its finished level",
            ),
            _q(
                "pressed",
                "flag",
                0.0,
                1.0,
                "1 once a button has been pressed past its actuation point",
            ),
        ]
    )
)

#: Per quantity, the slack (in the quantity's unit) a success bound is met within: a value meets
#: an ``at_least`` or an ``at_most`` when it misses it by no more than this. A bound that the
#: physics itself sets is never hit to the last digit: the published H2 coupling pass read
#: 3.9999936 mm against an "at most 4.0 mm" (the thread starts at 4.0 mm), and the same run after
#: an unrelated physics change reads 4.0000012 mm. An exact comparison turns that rounding into a
#: failure. The slack is small against the margins a task asks for, so it does not hide a real
#: miss: a part 3 mm short of its depth still fails. Quantities without a row
#: (``hold_s``, flags, slips, tilt, penetration) have no slack.
CRITERION_TOLERANCE: Final[Mapping[str, float]] = MappingProxyType(
    {
        "depth_mm": 0.1,
        "travel_mm": 0.5,
        "lift_mm": 0.5,
        "angle_deg": 0.5,
        "turns": 0.01,
        "torque_nm": 0.02,
        "force_n": 0.2,
    }
)

# --------------------------------------------------------------------------- primitives


@dataclass(frozen=True)
class PrimitiveInfo:
    """What a primitive is, what it may act on and what its predicate observes.

    * ``objects``: template kinds ``Step.object`` may be; ``handles``: the handles it may
      address on them; ``targets``: template kinds ``Step.target`` may be (empty: no target).
    * ``params``: the bounded "how" of the step (speeds, lift, direction); the "what" is in the
      step's success criteria.
    * ``quantities``: what the primitive's predicate observes besides the template state.
    * ``uses_roll``: drives the continuous wrist roll; ``may_hold``: may end holding its object.
    * ``status``: ``existing`` (a v1 grasp primitive, adapted by S2) or ``new`` (S2 builds it).
    * ``optional``: params a step may leave out (each has a default). A step omits one set to its
      default, so a step written before the param existed hashes as it did (its task id is
      unchanged). ``taught`` names the ones offered to the Task Engineer (module docstring): its
      schema lists them with the required params (:func:`planner_params`), and a step it writes
      with one at its default still drops it.
    """

    kind: PrimitiveKind
    description: str
    status: Literal["existing", "new"]
    fidelity: FidelityTier
    caveat: str
    objects: tuple[TemplateKind, ...]
    handles: tuple[str, ...]
    targets: tuple[TemplateKind, ...]
    params: tuple[ParamSpec, ...]
    quantities: tuple[str, ...]
    uses_roll: bool = False
    may_hold: bool = False
    optional: tuple[ParamSpec, ...] = ()
    taught: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = [q for q in self.quantities if q not in QUANTITIES]
        if unknown:
            raise ValueError(f"{self.kind}: unknown quantities {unknown}")
        params_json_schema(self.params + self.optional)  # unique names
        if any(p.default is None for p in self.optional):
            raise ValueError(f"{self.kind}: an optional param needs its default")
        if not set(self.taught) <= {p.name for p in self.optional}:
            raise ValueError(f"{self.kind}: taught names only optional params")


def _speed(name: str, unit: str, low: float, high: float, default: float, what: str) -> ParamSpec:
    return ParamSpec(name, "float", what, unit, low, high, default=default)


_LIFT = (
    ParamSpec(
        "lift_mm", "float", "How high to lift the part before moving it", "mm", 0.0, 300.0, default=100.0
    ),
    ParamSpec("hold_s", "float", "How long to hold the part lifted", "s", 0.0, 10.0, default=2.0),
)
_PICK_QUANTITIES = ("lift_mm", "hold_s", "in_target", "tilt_deg", "force_n", "penetration_mm")

PRIMITIVES: Final[Mapping[PrimitiveKind, PrimitiveInfo]] = MappingProxyType(
    {
        PrimitiveKind.GRASP: PrimitiveInfo(
            PrimitiveKind.GRASP,
            "Approach, close the fingers, lift and hold; place in the target if one is given.",
            "existing",
            FidelityTier.RIGID,
            "",
            (
                TemplateKind.PART,
                TemplateKind.PORT_PLUG,
                TemplateKind.THREADED_JOINT,
                TemplateKind.PANEL_RAIL,
                TemplateKind.TILE_BED,
            ),
            ("part", "plug", "member", "panel"),
            (TemplateKind.BIN,),
            _LIFT,
            _PICK_QUANTITIES,
            may_hold=True,
        ),
        PrimitiveKind.SUCTION: PrimitiveInfo(
            PrimitiveKind.SUCTION,
            "Approach, seal a suction cup on a flat face, lift and hold; place if a target is given.",
            "existing",
            FidelityTier.APPROXIMATION,
            "Suction is an approximation: the seal is a geometric test gating a capped adhesion force; "
            "no air flow, leak rate or porosity.",
            (TemplateKind.PART, TemplateKind.PANEL_RAIL, TemplateKind.TILE_BED),
            ("part", "panel"),
            (TemplateKind.BIN,),
            _LIFT,
            _PICK_QUANTITIES,
            may_hold=True,
        ),
        PrimitiveKind.WRAP: PrimitiveInfo(
            PrimitiveKind.WRAP,
            "Curl a soft arm around the part (at least 180 deg), lift and hold; place if a target is given.",
            "existing",
            FidelityTier.APPROXIMATION,
            "The soft arm is an approximation: rigid segments on spring joints curled by a cable.",
            (TemplateKind.PART,),
            ("part",),
            (TemplateKind.BIN,),
            _LIFT,
            _PICK_QUANTITIES,
            may_hold=True,
        ),
        PrimitiveKind.HOOK_PULL: PrimitiveInfo(
            PrimitiveKind.HOOK_PULL,
            "Engage a handle or lip and pull it along the object's arc or line.",
            "new",
            FidelityTier.RIGID,
            "",
            (TemplateKind.HINGED_LID, TemplateKind.DRAWER, TemplateKind.SLIDING_DOOR),
            ("handle",),
            (),
            (
                _speed("speed_mm_s", "mm/s", 5.0, 200.0, 50.0, "Pull speed at the handle"),
                ParamSpec(
                    "approach",
                    "choice",
                    "Where the tool comes from to engage the handle",
                    choices=("top", "front"),
                    default="top",
                ),
            ),
            ("slip_mm", "force_n", "penetration_mm"),
        ),
        PrimitiveKind.TWIST: PrimitiveInfo(
            PrimitiveKind.TWIST,
            "Grip or engage the object and turn the wrist roll under a resistive torque.",
            "new",
            FidelityTier.RIGID,
            "",
            (
                TemplateKind.QUARTER_TURN_LATCH,
                TemplateKind.THREADED_JOINT,
                TemplateKind.HANDWHEEL_VALVE,
                TemplateKind.PANEL_RAIL,
            ),
            ("head", "member", "wheel"),
            (),
            (
                ParamSpec(
                    "direction",
                    "choice",
                    "Turn direction seen from above",
                    choices=("cw", "ccw"),
                    default="cw",
                ),
                _speed("speed_deg_s", "deg/s", 5.0, 180.0, 60.0, "Wrist roll speed"),
            ),
            ("slip_deg", "torque_nm", "force_n", "penetration_mm"),
            uses_roll=True,
            may_hold=True,
        ),
        PrimitiveKind.INSERT: PrimitiveInfo(
            PrimitiveKind.INSERT,
            "Align the held member over its seat and push it in along the seat axis with compliance.",
            "new",
            FidelityTier.RIGID,
            "",
            (TemplateKind.PORT_PLUG, TemplateKind.THREADED_JOINT, TemplateKind.PANEL_RAIL),
            ("plug", "member"),
            (),
            (
                _speed("speed_mm_s", "mm/s", 1.0, 50.0, 10.0, "Insertion speed along the seat axis"),
                ParamSpec(
                    "standoff_mm",
                    "float",
                    "Gap above the seat before the push starts",
                    "mm",
                    2.0,
                    50.0,
                    default=10.0,
                ),
            ),
            ("lateral_force_n", "force_n", "jam", "penetration_mm"),
            may_hold=True,
        ),
        PrimitiveKind.PUSH: PrimitiveInfo(
            PrimitiveKind.PUSH,
            "Press or close the object with a force limit until it reaches its target.",
            "new",
            FidelityTier.RIGID,
            "",
            (
                TemplateKind.HINGED_LID,
                TemplateKind.DRAWER,
                TemplateKind.PART,
                TemplateKind.PORT_PLUG,
                TemplateKind.SLIDING_DOOR,
                TemplateKind.TILE_BED,
                TemplateKind.SPRING_BUTTON,
            ),
            ("lid", "drawer", "part", "plug", "door", "button"),
            (),
            (_speed("speed_mm_s", "mm/s", 1.0, 100.0, 20.0, "Push speed"),),
            ("force_n", "penetration_mm"),
            optional=(
                ParamSpec(
                    "approach",
                    "choice",
                    "Where the tool comes from: top (straight down) or front (tilted square to an "
                    "upright face)",
                    choices=("top", "front"),
                    default="top",
                ),
            ),
            taught=("approach",),
        ),
    }
)

# --------------------------------------------------------------------------- object templates


@dataclass(frozen=True)
class ObjectTemplate:
    """A parametric object kind (the base of every template; registry: :func:`get_template`).

    * ``params``: the bounded :class:`ParamSpec` rows (every one has a default).
    * ``handles``: named features a step can address (``"handle"``, ``"member"``, ``"seat"``...).
    * ``quantities``: state quantities its builder reports (:data:`QUANTITIES`).
    * ``fidelity`` and ``caveat``: printed next to every result that touches it.
    * ``needs_ref``: instances name their geometry with :attr:`ObjectInstance.ref` (parts).
    * ``model``: one line on how it is simulated.
    * ``planner``: offered to the Task Engineer (module docstring).
    * ``check``: rules across params (plain sentences, empty when they hold), checked once every
      param is within its bounds.
    """

    kind: TemplateKind
    display_name: str
    description: str
    model: str
    params: tuple[ParamSpec, ...]
    handles: tuple[str, ...]
    quantities: tuple[str, ...]
    fidelity: FidelityTier
    caveat: str = ""
    articulated: bool = True
    needs_ref: bool = False
    planner: bool = True
    check: Callable[[Mapping[str, Any]], list[str]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TemplateKind(self.kind))
        _text("template display_name", self.display_name, _TEXT_MAX["label"])
        params_json_schema(self.params)
        missing = [p.name for p in self.params if p.default is None]
        if missing:
            raise ValueError(f"{self.kind}: every template param needs a default, missing {missing}")
        if not self.handles or any(not _SLUG.match(h) for h in self.handles):
            raise ValueError(f"{self.kind}: handles must be snake_case ids")
        unknown = [q for q in self.quantities if q not in QUANTITIES]
        if unknown:
            raise ValueError(f"{self.kind}: unknown quantities {unknown}")
        if self.fidelity is FidelityTier.APPROXIMATION and not self.caveat:
            raise ValueError(f"{self.kind}: an approximation needs its caveat")

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def problems(self, params: Mapping[str, Any]) -> list[str]:
        """Unknown, missing or out-of-bounds params, then the template's own rules (``check``)."""
        by_name = {p.name: p for p in self.params}
        out = [f"unknown param {k!r}" for k in params if k not in by_name]
        out += [f"missing param {k!r}" for k in by_name if k not in params]
        for k, v in params.items():
            if k in by_name:
                out += by_name[k].problems(v)
        if not out and self.check is not None:
            out += self.check(params)
        return out


def _f(name: str, unit: str, low: float, high: float, default: float, what: str) -> ParamSpec:
    return ParamSpec(name, "float", what, unit, low, high, default=default)


def _c(name: str, choices: tuple[str, ...], default: str, what: str) -> ParamSpec:
    return ParamSpec(name, "choice", what, choices=choices, default=default)


#: A tile's plausible density (kg/m^3): a light composite to a dense stone [assumed].
TILE_DENSITY_KG_M3: Final = (800.0, 4000.0)


def _tile_bed_check(p: Mapping[str, Any]) -> list[str]:
    """A tile the bed carries unpressed (its weight under the bed's yield) and of a plausible mass."""
    out = []
    weight = float(p["mass_g"]) / 1000.0 * 9.81
    if weight >= float(p["bed_yield_n"]):
        out.append(
            f"the tile weighs {weight:.1f} N, at or over the bed's {float(p['bed_yield_n']):g} N yield: "
            "it would sink into the bed by itself; lower mass_g or raise bed_yield_n"
        )
    volume = (float(p["tile_mm"]) / 1000.0) ** 2 * float(p["thickness_mm"]) / 1000.0
    lo, hi = TILE_DENSITY_KG_M3
    density = float(p["mass_g"]) / 1000.0 / volume
    if not lo <= density <= hi:
        out.append(
            f"a {float(p['tile_mm']):g} mm tile {float(p['thickness_mm']):g} mm thick of "
            f"{float(p['mass_g']):g} g is {density:.0f} kg/m^3; a tile is {lo:g} to {hi:g} kg/m^3"
        )
    return out


_BUILTIN_TEMPLATES: tuple[ObjectTemplate, ...] = (
    ObjectTemplate(
        TemplateKind.HINGED_LID,
        "Hinged lid or door",
        "A panel on a revolute hinge: a lid opens upward about its back edge, a door swings about a side edge.",
        "revolute joint with damping, a hinge stop and an optional opening torque (spring or detent)",
        (
            _c("orientation", ("lid", "door"), "lid", "Lid (opens upward) or door (swings sideways)"),
            _f("width_mm", "mm", 60.0, 600.0, 300.0, "Panel width along the hinge"),
            _f("depth_mm", "mm", 40.0, 600.0, 200.0, "Panel depth from the hinge to the free edge"),
            _f("thickness_mm", "mm", 2.0, 30.0, 6.0, "Panel thickness"),
            _f("max_angle_deg", "deg", 60.0, 180.0, 110.0, "Hinge stop: the widest opening"),
            _f("open_torque_nm", "N m", 0.0, 5.0, 0.3, "Torque that resists opening at the hinge"),
            _f("damping_nms", "N m s", 0.0, 2.0, 0.05, "Hinge damping"),
            _c("feature", ("lip", "handle"), "lip", "What a hook can engage at the free edge"),
            _f("feature_mm", "mm", 5.0, 50.0, 15.0, "Lip overhang or handle clearance"),
        ),
        ("lid", "handle"),
        ("angle_deg",),
        FidelityTier.RIGID,
    ),
    ObjectTemplate(
        TemplateKind.DRAWER,
        "Drawer",
        "A box on a prismatic slide with friction, opened by a bar, lip or knob handle.",
        "prismatic joint with friction loss and end stops",
        (
            _f("width_mm", "mm", 80.0, 600.0, 300.0, "Drawer front width"),
            _f("height_mm", "mm", 40.0, 300.0, 100.0, "Drawer front height"),
            _f("depth_mm", "mm", 80.0, 600.0, 350.0, "Drawer depth"),
            _f("travel_mm", "mm", 20.0, 500.0, 250.0, "Slide travel to the open stop"),
            _f("pull_force_n", "N", 0.5, 50.0, 5.0, "Friction force that resists sliding"),
            _c("handle", ("bar", "lip", "knob"), "bar", "Handle shape"),
            _f("handle_mm", "mm", 10.0, 60.0, 25.0, "Handle clearance from the front"),
        ),
        ("drawer", "handle"),
        ("travel_mm",),
        FidelityTier.RIGID,
    ),
    ObjectTemplate(
        TemplateKind.QUARTER_TURN_LATCH,
        "Quarter-turn latch",
        "A slot or wing head on a revolute joint with a detent and a stop at the end of its turn.",
        "revolute joint with a torque to open, a detent and a stop",
        (
            _c("head", ("slot", "wing"), "wing", "Head shape: a slot for a blade or a wing to grip"),
            _f("head_diameter_mm", "mm", 10.0, 60.0, 25.0, "Head diameter"),
            _f("head_height_mm", "mm", 3.0, 40.0, 12.0, "Head height above its panel"),
            _f("turn_deg", "deg", 45.0, 180.0, 90.0, "Turn from locked to open (the stop)"),
            _f("torque_nm", "N m", 0.05, 5.0, 0.8, "Torque needed to turn it"),
            ParamSpec(
                "detent", "bool", "A detent holds it locked until the torque is exceeded", default=True
            ),
        ),
        ("head",),
        ("angle_deg",),
        FidelityTier.APPROXIMATION,
        "Latches are an approximation: a detent torque on a revolute joint; no cam or spring geometry.",
    ),
    ObjectTemplate(
        TemplateKind.THREADED_JOINT,
        "Threaded joint",
        "A cap, coupling or nut that starts on a fixed threaded seat and tightens to a final torque.",
        "revolute joint coupled to a prismatic joint (helical equality) with a friction torque ramp",
        (
            _c("member", ("cap", "coupling", "nut"), "coupling", "The moving threaded part"),
            _f("diameter_mm", "mm", 6.0, 120.0, 27.0, "Outer diameter of the moving part"),
            _f("length_mm", "mm", 5.0, 120.0, 40.0, "Length of the moving part"),
            _f("pitch_mm", "mm", 0.5, 5.0, 1.8, "Thread pitch"),
            _f("engage_turns", "turn", 0.25, 10.0, 3.0, "Turns from the thread start to seated"),
            _f("start_torque_nm", "N m", 0.0, 2.0, 0.1, "Running torque at the thread start"),
            _f("final_torque_nm", "N m", 0.1, 20.0, 5.0, "Torque when seated"),
            _f("chamfer_mm", "mm", 0.0, 5.0, 1.0, "Lead-in chamfer on the seat"),
        ),
        ("member", "seat"),
        ("turns", "depth_mm", "angle_deg"),
        FidelityTier.APPROXIMATION,
        "Threads are an approximation: a helical joint with a torque ramp; no thread flanks, "
        "cross-threading or galling. A screen, not a promise.",
    ),
    ObjectTemplate(
        TemplateKind.HANDWHEEL_VALVE,
        "Handwheel valve",
        "A handwheel on a vertical stem with a resistive torque over its travel.",
        "revolute joint with a resistive torque curve and travel stops",
        (
            _f("rim_diameter_mm", "mm", 40.0, 400.0, 120.0, "Rim diameter"),
            _f("rim_section_mm", "mm", 6.0, 30.0, 12.0, "Rim cross-section diameter"),
            ParamSpec("spokes", "int", "Number of spokes (0: a solid disc)", "count", 0, 6, default=3),
            _f("torque_nm", "N m", 0.1, 20.0, 3.0, "Torque needed to turn it"),
            _f("travel_deg", "deg", 30.0, 1440.0, 90.0, "Travel from closed to open"),
        ),
        ("wheel",),
        ("angle_deg",),
        FidelityTier.APPROXIMATION,
        "Valves are an approximation: a resistive torque curve on a revolute joint; no packing or "
        "seat dynamics.",
    ),
    ObjectTemplate(
        TemplateKind.PORT_PLUG,
        "Port and plug",
        "A fixed socket with a chamfer and a clearance, and a plug that starts free on the table.",
        "rigid socket and plug with contact, an insertion friction and an optional latch click",
        (
            _f("plug_diameter_mm", "mm", 3.0, 60.0, 12.0, "Plug diameter"),
            _f("plug_length_mm", "mm", 10.0, 120.0, 40.0, "Plug length"),
            _f("clearance_mm", "mm", 0.02, 2.0, 0.2, "Diametral clearance in the socket"),
            _f("chamfer_mm", "mm", 0.0, 5.0, 1.0, "Socket lead-in chamfer"),
            _f("depth_mm", "mm", 3.0, 60.0, 15.0, "Socket depth"),
            _f("insert_force_n", "N", 0.0, 50.0, 5.0, "Friction force that resists insertion"),
            ParamSpec("latch", "bool", "A latch clicks when the plug is seated", default=False),
        ),
        ("plug", "socket"),
        ("depth_mm",),
        FidelityTier.RIGID,
    ),
    ObjectTemplate(
        TemplateKind.BIN,
        "Bin or tote",
        "An open box: a fixed destination or a free body.",
        "static or free rigid box walls",
        (
            _f("length_mm", "mm", 60.0, 800.0, 300.0, "Outer length (x)"),
            _f("width_mm", "mm", 60.0, 600.0, 200.0, "Outer width (y)"),
            _f("height_mm", "mm", 20.0, 400.0, 120.0, "Outer height"),
            _f("wall_mm", "mm", 1.0, 20.0, 4.0, "Wall thickness"),
            ParamSpec("fixed", "bool", "Fixed to the table (otherwise a free body)", default=True),
        ),
        ("bin",),
        (),
        FidelityTier.RIGID,
        articulated=False,
    ),
    ObjectTemplate(
        TemplateKind.PART,
        "Rigid part",
        "A rigid part: a library or uploaded mesh (ref), or a box or cylinder of the given size.",
        "free rigid body (exact convex pieces, or CoACD hulls for scans and uploads)",
        (
            _c("shape", ("mesh", "box", "cylinder"), "box", "Mesh from ref, or a procedural box or cylinder"),
            _f("size_x_mm", "mm", 5.0, 400.0, 40.0, "Size along x (box), or diameter (cylinder)"),
            _f("size_y_mm", "mm", 5.0, 400.0, 40.0, "Size along y (box; ignored for a cylinder)"),
            _f("size_z_mm", "mm", 2.0, 400.0, 40.0, "Height"),
            _f("mass_g", "g", 1.0, 5000.0, 100.0, "Mass (box and cylinder; a mesh uses its material)"),
        ),
        ("part",),
        ("lift_mm", "tilt_deg"),
        FidelityTier.RIGID,
        articulated=False,
        needs_ref=True,
    ),
    # ----------------------------------------------------------------- Studio v2: the job templates
    ObjectTemplate(
        TemplateKind.PANEL_RAIL,
        "Panel on mounting rails",
        "A large flat panel (a solar module) resting on two mounting rails against their end stops, "
        "with a threaded clamp on a post beside its edge.",
        "free rigid slab on two static rails with end stops; the clamp is a threaded member on a post "
        "(revolute and prismatic joints coupled by a helical tendon, with a friction torque ramp)",
        (
            _f("length_mm", "mm", 300.0, 1200.0, 1000.0, "Panel length (along x)"),
            _f("width_mm", "mm", 200.0, 800.0, 600.0, "Panel width (along y, the way the rails run)"),
            _f("thickness_mm", "mm", 10.0, 60.0, 35.0, "Panel thickness"),
            _f("mass_g", "g", 500.0, 25000.0, 8000.0, "Panel mass"),
            _f(
                "start_gap_mm",
                "mm",
                0.0,
                30.0,
                2.0,
                "Gap between the panel's edge and the rail stops at the start",
            ),
            _f(
                "clamp_torque_nm",
                "N m",
                0.5,
                10.0,
                6.0,
                "Torque that seats the clamp (its thread's final torque)",
            ),
        ),
        ("panel", "member"),
        ("gap_mm", "clamp_torque_nm", "depth_mm", "turns", "angle_deg"),
        FidelityTier.APPROXIMATION,
        "The clamp is an approximation: a 13 mm hex modelled 15 mm tall so the simulated socket can seat "
        "(a standard M8 nut is 6.8 mm tall), on a helical joint with a torque ramp; it does not press the "
        "panel. The panel is a rigid slab.",
    ),
    ObjectTemplate(
        TemplateKind.TILE_BED,
        "Tile on a mortar bed",
        "A square floor tile laid on a fresh mortar bed, standing proud of its finished level until it "
        "is pressed in.",
        "free rigid tile on a 3 x 3 grid of plastic bed cells (vertical slides that yield, harden and "
        "keep their set)",
        (
            _f("tile_mm", "mm", 100.0, 600.0, 300.0, "Tile side (square)"),
            _f("thickness_mm", "mm", 5.0, 30.0, 10.0, "Tile thickness"),
            _f("mass_g", "g", 100.0, 20000.0, 2160.0, "Tile mass"),
            _f("proud_mm", "mm", 0.5, 6.0, 2.0, "How far the laid tile stands above its finished level"),
            _f("bed_yield_n", "N", 5.0, 200.0, 30.0, "Total force at which the fresh bed starts to give"),
            _f(
                "bed_hardening_n_mm", "N/mm", 1.0, 100.0, 15.0, "Extra force per mm the bed needs as it gives"
            ),
        ),
        ("part",),
        ("height_err_mm", "tilt_deg"),
        FidelityTier.APPROXIMATION,
        "The mortar bed is an approximation: nine plastic cells that yield and harden linearly and keep "
        "their set; no flow, suction or curing. A screen, not a promise.",
        check=_tile_bed_check,
    ),
    ObjectTemplate(
        TemplateKind.SPRING_BUTTON,
        "Push button",
        "A round spring-return push button in a bezel, on a wall or on a desk console; a press past its "
        "actuation point latches the call.",
        "prismatic joint with a preloaded return spring and an end stop, and a latch that records a "
        "press past the actuation point",
        (
            _c(
                "mount",
                ("wall", "desk"),
                "wall",
                "On a wall (it faces sideways) or a desk console (it faces up)",
            ),
            _f("diameter_mm", "mm", 10.0, 80.0, 40.0, "Button diameter"),
            _f("travel_mm", "mm", 1.0, 10.0, 3.0, "Button travel to its stop"),
            _f("actuation_force_n", "N", 0.5, 30.0, 4.0, "Force that presses it to its actuation point"),
            _f("proud_mm", "mm", 0.0, 10.0, 1.0, "How far the button stands out of its bezel at rest"),
            _f(
                "height_mm",
                "mm",
                30.0,
                300.0,
                120.0,
                "Wall: the button's centre above the table; desk: the console's height",
            ),
        ),
        ("button",),
        ("travel_mm", "pressed"),
        FidelityTier.RIGID,
    ),
    ObjectTemplate(
        TemplateKind.SLIDING_DOOR,
        "Sliding machine door",
        "A machine enclosure whose door slides sideways: a front leaf and a roof leaf over the loading "
        "opening, moved by a fin handle on the roof.",
        "prismatic joint with friction loss and end stops (a drawer's slide, turned sideways)",
        (
            _f("width_mm", "mm", 120.0, 400.0, 250.0, "Loading opening width, along the slide"),
            _f("depth_mm", "mm", 150.0, 500.0, 300.0, "Machine depth, front to back"),
            _f("height_mm", "mm", 120.0, 400.0, 250.0, "Machine height"),
            _f("travel_mm", "mm", 50.0, 450.0, 260.0, "Door travel to the open stop"),
            _f("pull_force_n", "N", 0.5, 60.0, 10.0, "Friction force that resists sliding"),
            _f("handle_mm", "mm", 15.0, 60.0, 40.0, "Height of the fin handle above the roof"),
        ),
        ("door", "handle"),
        ("travel_mm",),
        FidelityTier.RIGID,
    ),
)

_TEMPLATES: dict[TemplateKind, ObjectTemplate] = {t.kind: t for t in _BUILTIN_TEMPLATES}


def get_template(kind: TemplateKind | str) -> ObjectTemplate:
    return _TEMPLATES[TemplateKind(kind)]


def templates() -> tuple[ObjectTemplate, ...]:
    """Every registered template, in :class:`TemplateKind` order."""
    return tuple(_TEMPLATES[k] for k in TemplateKind if k in _TEMPLATES)


def planner_templates() -> tuple[ObjectTemplate, ...]:
    """The templates offered to the Task Engineer (module docstring), in :class:`TemplateKind` order."""
    return tuple(t for t in templates() if t.planner)


def planner_kinds(info: PrimitiveInfo) -> tuple[TemplateKind, ...]:
    """The object kinds of ``info`` the Task Engineer is offered."""
    return tuple(k for k in info.objects if get_template(k).planner)


def planner_handles(info: PrimitiveInfo) -> tuple[str, ...]:
    """The handles of ``info`` that some kind the Task Engineer is offered has."""
    kinds = planner_kinds(info)
    return tuple(h for h in info.handles if any(h in get_template(k).handles for k in kinds))


def planner_params(info: PrimitiveInfo) -> tuple[ParamSpec, ...]:
    """The step params of ``info`` the Task Engineer is offered: every required one, then the
    optional ones it has been taught (:attr:`PrimitiveInfo.taught`)."""
    return info.params + tuple(p for p in info.optional if p.name in info.taught)


def planner_quantities() -> tuple[Quantity, ...]:
    """The quantities offered to the Task Engineer, in :data:`QUANTITIES` order."""
    return tuple(q for q in QUANTITIES.values() if q.planner)


def register_template(template: ObjectTemplate, *, replace: bool = False) -> ObjectTemplate:
    """Register (or, with ``replace``, redefine) a template descriptor (a contract change)."""
    if not isinstance(template, ObjectTemplate):
        raise TypeError("register_template takes an ObjectTemplate")
    if template.kind in _TEMPLATES and not replace:
        raise ValueError(f"template {template.kind.value!r} is already registered")
    _TEMPLATES[template.kind] = template
    return template


# --------------------------------------------------------------------------- the task spec


@dataclass(frozen=True)
class ObjectPose:
    """Where an object sits on the table: mm from the table centre, yaw in deg."""

    x_mm: float = 0.0
    y_mm: float = 0.0
    yaw_deg: float = 0.0

    LIMIT_MM: ClassVar[float] = 300.0  # inside the wrist's reach (+-350 mm)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_mm", _number("pose.x_mm", self.x_mm, -self.LIMIT_MM, self.LIMIT_MM))
        object.__setattr__(self, "y_mm", _number("pose.y_mm", self.y_mm, -self.LIMIT_MM, self.LIMIT_MM))
        object.__setattr__(self, "yaw_deg", _number("pose.yaw_deg", self.yaw_deg, -180.0, 180.0))

    def to_dict(self) -> dict[str, float]:
        return {"x_mm": self.x_mm, "y_mm": self.y_mm, "yaw_deg": self.yaw_deg}


@dataclass(frozen=True, eq=False)
class ObjectInstance:
    """One object in the scene: a template kind, its params, a pose and (parts) a geometry ref.

    * ``name``: unique in the task (``"tote_lid"``); every MuJoCo element it adds is named
      ``obj_<name>_...``.
    * ``label``: what the screen calls it (brand-free).
    * ``ref``: ``library:<part>``, ``upload:<id>`` or ``scan:<id>`` for a part whose
      ``shape`` is ``mesh``; ``None`` otherwise.
    """

    name: str
    kind: TemplateKind
    label: str
    params: Mapping[str, Any]
    pose: ObjectPose = field(default_factory=ObjectPose)
    ref: str | None = None

    def __post_init__(self) -> None:
        _slug("object name", self.name)
        object.__setattr__(self, "kind", TemplateKind(self.kind))
        _text(f"object {self.name} label", self.label, _TEXT_MAX["label"])
        params = _canon(self.params, "params")
        if not isinstance(params, dict):
            raise TypeError("object params must be a mapping")
        problems = get_template(self.kind).problems(params)
        if problems:
            raise ValueError(f"object {self.name} ({self.kind.value}): " + "; ".join(problems))
        object.__setattr__(self, "params", FrozenDict(params))
        if not isinstance(self.pose, ObjectPose):
            raise TypeError("pose must be an ObjectPose")
        if self.ref is not None and (not isinstance(self.ref, str) or not _REF.match(self.ref)):
            raise ValueError(f"object {self.name}: ref must match {_REF.pattern}, got {self.ref!r}")
        needs = self.kind is TemplateKind.PART and self.params.get("shape") == "mesh"
        if needs and self.ref is None:
            raise ValueError(f"object {self.name}: a mesh part needs its ref")
        if not needs and self.ref is not None:
            raise ValueError(f"object {self.name}: only a mesh part has a ref")

    @classmethod
    def of(
        cls,
        name: str,
        kind: TemplateKind | str,
        label: str,
        pose: ObjectPose | None = None,
        ref: str | None = None,
        **params: Any,
    ) -> ObjectInstance:
        """An instance with the template's defaults for every param not given."""
        full = {**get_template(kind).defaults(), **params}
        return cls(name, TemplateKind(kind), label, full, pose or ObjectPose(), ref)

    @property
    def template(self) -> ObjectTemplate:
        return get_template(self.kind)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "params": _thaw(self.params),
            "pose": self.pose.to_dict(),
            "ref": self.ref,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_dict(), "label": self.label}


@dataclass(frozen=True)
class StepConstraints:
    """Task limits a step must respect (``None``: no limit). The wrist's own limits are separate
    (:data:`STUDIO_WRIST`) and the same for every candidate."""

    max_force_n: float | None = None
    max_torque_nm: float | None = None
    precision_mm: float | None = None
    precision_deg: float | None = None

    BOUNDS: ClassVar[Mapping[str, tuple[float, float]]] = MappingProxyType(
        {
            "max_force_n": (0.1, 500.0),
            "max_torque_nm": (0.01, 50.0),
            "precision_mm": (0.05, 50.0),
            "precision_deg": (0.1, 45.0),
        }
    )

    def __post_init__(self) -> None:
        for name, (lo, hi) in self.BOUNDS.items():
            object.__setattr__(self, name, _optional(f"constraints.{name}", getattr(self, name), lo, hi))

    def to_dict(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in self.BOUNDS}


@dataclass(frozen=True)
class SuccessCriterion:
    """``quantity`` must be at least ``at_least`` and/or at most ``at_most`` when the step ends
    (peaks for peak quantities such as ``force_n``)."""

    quantity: str
    at_least: float | None = None
    at_most: float | None = None

    def __post_init__(self) -> None:
        q = QUANTITIES.get(self.quantity)
        if q is None:
            raise ValueError(f"unknown quantity {self.quantity!r}; known: {sorted(QUANTITIES)}")
        lo = _optional(f"{self.quantity}.at_least", self.at_least, q.low, q.high)
        hi = _optional(f"{self.quantity}.at_most", self.at_most, q.low, q.high)
        if lo is None and hi is None:
            raise ValueError(f"criterion on {self.quantity} needs at_least or at_most")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"criterion on {self.quantity}: at_least {lo:g} > at_most {hi:g}")
        object.__setattr__(self, "at_least", lo)
        object.__setattr__(self, "at_most", hi)

    @property
    def tolerance(self) -> float:
        """The slack this criterion's bounds are met within (:data:`CRITERION_TOLERANCE`)."""
        return CRITERION_TOLERANCE.get(self.quantity, 0.0)

    def passes(self, value: float | None) -> bool:
        """Whether ``value`` meets the bounds, each within :attr:`tolerance`."""
        if value is None or not math.isfinite(value):
            return False
        tol = self.tolerance
        if self.at_least is not None and value < self.at_least - tol:
            return False
        return not (self.at_most is not None and value > self.at_most + tol)

    def to_dict(self) -> dict[str, Any]:
        return {"quantity": self.quantity, "at_least": self.at_least, "at_most": self.at_most}


@dataclass(frozen=True, eq=False)
class Step:
    """One interaction of the use case (see the module docstring).

    * ``object``/``handle``: the object instance and which of its handles the tool addresses;
      ``target``: the destination bin of a pick, else ``None``.
    * ``params``: one value per :attr:`PrimitiveInfo.params` row, and any of its
      :attr:`PrimitiveInfo.optional` rows (omitted when set to their default, so the step hashes
      as one written without them).
    * ``after``: ids of steps that must finish first (all earlier in the list).
    * ``hold_after``: the step ends still holding its object, so the next step must use the
      same tool (no tool change or head switch while holding).
    """

    id: str
    primitive: PrimitiveKind
    object: str
    handle: str
    description: str
    params: Mapping[str, Any]
    success: tuple[SuccessCriterion, ...]
    constraints: StepConstraints = field(default_factory=StepConstraints)
    target: str | None = None
    after: tuple[str, ...] = ()
    hold_after: bool = False

    def __post_init__(self) -> None:
        _slug("step id", self.id)
        object.__setattr__(self, "primitive", PrimitiveKind(self.primitive))
        _slug(f"step {self.id} object", self.object)
        _slug(f"step {self.id} handle", self.handle)
        if self.target is not None:
            _slug(f"step {self.id} target", self.target)
        _text(f"step {self.id} description", self.description, _TEXT_MAX["note"])
        info = PRIMITIVES[self.primitive]
        params = _canon(self.params, "params")
        if not isinstance(params, dict):
            raise TypeError("step params must be a mapping")
        by_name = {p.name: p for p in info.params}
        optional = {p.name: p for p in info.optional}
        problems = [f"unknown param {k!r}" for k in params if k not in by_name and k not in optional]
        problems += [f"missing param {k!r}" for k in by_name if k not in params]
        for k, v in params.items():
            row = by_name.get(k) or optional.get(k)
            if row is not None:
                problems += row.problems(v)
        if problems:
            raise ValueError(f"step {self.id} ({self.primitive.value}): " + "; ".join(problems))
        params = {k: v for k, v in params.items() if not (k in optional and v == optional[k].default)}
        object.__setattr__(self, "params", FrozenDict(params))
        success = tuple(self.success)
        if not 1 <= len(success) <= 4 or not all(isinstance(c, SuccessCriterion) for c in success):
            raise ValueError(f"step {self.id}: 1 to 4 SuccessCriterion rows")
        object.__setattr__(self, "success", success)
        if not isinstance(self.constraints, StepConstraints):
            raise TypeError("constraints must be StepConstraints")
        after = tuple(self.after)
        for a in after:
            _slug(f"step {self.id} after", a)
        if len(set(after)) != len(after) or self.id in after:
            raise ValueError(f"step {self.id}: after must list distinct other steps")
        object.__setattr__(self, "after", after)
        if not isinstance(self.hold_after, bool):
            raise TypeError("hold_after must be a boolean")
        if self.hold_after and not info.may_hold:
            raise ValueError(f"step {self.id}: a {self.primitive.value} step cannot end holding its object")

    @classmethod
    def of(
        cls,
        id: str,
        primitive: PrimitiveKind | str,
        object: str,
        handle: str,
        description: str,
        success: Sequence[SuccessCriterion],
        **kwargs: Any,
    ) -> Step:
        """A step with the primitive's default params, overridden by ``kwargs['params']``."""
        prim = PrimitiveKind(primitive)
        params = {p.name: p.default for p in PRIMITIVES[prim].params}
        params.update(kwargs.pop("params", {}))
        return cls(id, prim, object, handle, description, params, tuple(success), **kwargs)

    @property
    def info(self) -> PrimitiveInfo:
        return PRIMITIVES[self.primitive]

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "primitive": self.primitive.value,
            "object": self.object,
            "handle": self.handle,
            "target": self.target,
            "params": _thaw(self.params),
            "constraints": self.constraints.to_dict(),
            "success": [c.to_dict() for c in self.success],
            "after": list(self.after),
            "hold_after": self.hold_after,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_dict(), "description": self.description}


@dataclass(frozen=True)
class Clarification:
    """A question for the user and the answer assumed until they answer it."""

    question: str
    assumed_answer: str

    def __post_init__(self) -> None:
        _text("clarification question", self.question, _TEXT_MAX["line"])
        _text("clarification assumed_answer", self.assumed_answer, _TEXT_MAX["line"])


@dataclass(frozen=True)
class UnsupportedItem:
    """Something in the use case that is not simulated (deformables, for example), why, and the
    nearest primitive offered instead (``None`` when there is none)."""

    what: str
    reason: str
    nearest_primitive: PrimitiveKind | None = None

    def __post_init__(self) -> None:
        _text("unsupported what", self.what, 120)
        _text("unsupported reason", self.reason, _TEXT_MAX["line"])
        if self.nearest_primitive is not None:
            object.__setattr__(self, "nearest_primitive", PrimitiveKind(self.nearest_primitive))


class TaskSpecError(ValueError):
    """A task spec (or a Task Engineer tool call) is invalid; ``problems`` lists every reason."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("invalid task spec: " + "; ".join(self.problems))


TaskSource = Literal["nemotron", "manual", "recorded", "test"]


@dataclass(frozen=True, eq=False)
class TaskSpec:
    """One use case: objects, ordered steps and what the Task Engineer assumed or asked.

    Construction checks every cross-reference (:meth:`problems`) and raises
    :class:`TaskSpecError` listing them all. ``refusal`` is set (and ``steps`` empty) when the
    whole use case cannot be simulated. :attr:`task_id` hashes the objects and steps only (not
    the free text), so an edited title does not change the task.
    """

    title: str
    description: str
    objects: tuple[ObjectInstance, ...]
    steps: tuple[Step, ...]
    assumptions: tuple[str, ...] = ()
    clarifications: tuple[Clarification, ...] = ()
    unsupported: tuple[UnsupportedItem, ...] = ()
    refusal: str | None = None
    source: TaskSource = "manual"

    def __post_init__(self) -> None:
        for name in ("objects", "steps", "assumptions", "clarifications", "unsupported"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        problems = self.problems()
        if problems:
            raise TaskSpecError(problems)

    def problems(self) -> list[str]:
        out: list[str] = []
        try:
            _text("title", self.title, _TEXT_MAX["title"])
            _text("description", self.description, _TEXT_MAX["description"], required=False)
            if self.refusal is not None:
                _text("refusal", self.refusal, 300)
        except (TypeError, ValueError) as exc:
            out.append(str(exc))
        if self.source not in ("nemotron", "manual", "recorded", "test"):
            out.append(f"source must be nemotron, manual, recorded or test, got {self.source!r}")
        if len(self.assumptions) > 10:
            out.append("at most 10 assumptions")
        for i, a in enumerate(self.assumptions):
            try:
                _text(f"assumptions[{i}]", a, _TEXT_MAX["line"])
            except (TypeError, ValueError) as exc:
                out.append(str(exc))
        if len(self.clarifications) > 6 or len(self.unsupported) > 6:
            out.append("at most 6 clarifications and 6 unsupported items")
        if not all(isinstance(o, ObjectInstance) for o in self.objects):
            return [*out, "objects must be ObjectInstance values"]
        if not all(isinstance(s, Step) for s in self.steps):
            return [*out, "steps must be Step values"]
        if self.refusal is not None:
            if self.steps:
                out.append("a refused task has no steps")
            return out
        if not self.steps:
            out.append("a task needs at least one step (or a refusal)")
        if not 1 <= len(self.objects) <= MAX_OBJECTS:
            out.append(f"a task has 1 to {MAX_OBJECTS} objects")
        if len(self.steps) > MAX_STEPS:
            out.append(f"a task has at most {MAX_STEPS} steps")
        objects = {o.name: o for o in self.objects}
        if len(objects) != len(self.objects):
            out.append("object names must be unique")
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            out.append("step ids must be unique")
        seen: set[str] = set()
        held: str | None = None
        for i, s in enumerate(self.steps):
            where = f"step {s.id}"
            info = s.info
            obj = objects.get(s.object)
            if obj is None:
                out.append(f"{where}: unknown object {s.object!r}")
            else:
                if obj.kind not in info.objects:
                    out.append(
                        f"{where}: {s.primitive.value} cannot act on a {obj.kind.value} "
                        f"(it acts on {', '.join(k.value for k in info.objects)})"
                    )
                if s.handle not in obj.template.handles or s.handle not in info.handles:
                    allowed = [h for h in obj.template.handles if h in info.handles]
                    out.append(f"{where}: handle {s.handle!r} is not one of {allowed}")
                measurable = set(obj.template.quantities) | set(info.quantities)
                for c in s.success:
                    if c.quantity not in measurable:
                        out.append(f"{where}: {c.quantity!r} is not measured here (use {sorted(measurable)})")
            if s.target is not None:
                tgt = objects.get(s.target)
                if tgt is None:
                    out.append(f"{where}: unknown target {s.target!r}")
                elif tgt.kind not in info.targets:
                    out.append(
                        f"{where}: target {s.target!r} must be one of {[k.value for k in info.targets]}"
                    )
                elif s.target == s.object:
                    out.append(f"{where}: the target is the object itself")
            for a in s.after:
                if a not in seen:
                    out.append(f"{where}: depends on {a!r}, which is not an earlier step")
            if held is not None and s.object != held:
                out.append(f"{where}: acts on {s.object!r} while {held!r} is still held by the previous step")
            held = s.object if s.hold_after else None
            seen.add(s.id)
            if i == len(self.steps) - 1 and s.hold_after:
                out.append(f"{where}: the last step must release its object")
        return out

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def object(self, name: str) -> ObjectInstance:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    @property
    def step_ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.steps)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "contract": CONTRACT_VERSION,
            "objects": [o.canonical_dict() for o in self.objects],
            "steps": [s.canonical_dict() for s in self.steps],
        }

    @property
    def task_id(self) -> str:
        """sha256 (hex) of :meth:`canonical_dict`."""
        return _sha256(canonical_json(self.canonical_dict()))

    def fidelity(self, step_id: str) -> FidelityTier:
        """The step's tier from its primitive and the templates it touches (the effector's tier
        is added by the runner)."""
        s = self.step(step_id)
        tiers = [s.info.fidelity, self.object(s.object).template.fidelity]
        if s.target is not None:
            tiers.append(self.object(s.target).template.fidelity)
        return worst_tier(*tiers)

    def caveats(self, step_id: str) -> tuple[str, ...]:
        s = self.step(step_id)
        names = [s.object] + ([s.target] if s.target else [])
        texts = [s.info.caveat] + [self.object(n).template.caveat for n in names]
        return tuple(dict.fromkeys(t for t in texts if t))

    def to_tool_args(self) -> dict[str, Any]:
        """The Task Engineer tool-call form (:func:`task_plan_json_schema`); round-trips through
        :func:`task_spec_from_tool_args`. A step lists every param the schema does, a taught
        optional one at its default included (the step itself drops it, so its id is the same)."""
        args = self._args()
        for d, s in zip(args["steps"], self.steps, strict=True):
            d["params"] = {**d["params"]}
            for p in planner_params(s.info):
                d["params"].setdefault(p.name, p.default)
        return args

    def _args(self) -> dict[str, Any]:
        """The plan's fields as :meth:`to_dict` records them (each step's own params only)."""
        return {
            "title": self.title,
            "description": self.description,
            "objects": [o.to_dict() for o in self.objects],
            "steps": [s.to_dict() for s in self.steps],
            "assumptions": list(self.assumptions),
            "clarifications": [
                {"question": c.question, "assumed_answer": c.assumed_answer} for c in self.clarifications
            ],
            "unsupported": [
                {
                    "what": u.what,
                    "reason": u.reason,
                    "nearest_primitive": u.nearest_primitive.value if u.nearest_primitive else None,
                }
                for u in self.unsupported
            ],
            "refusal": self.refusal,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": CONTRACT_VERSION,
            "label": PRODUCT_MODE_LABEL,
            "task_id": self.task_id,
            "source": self.source,
            **self._args(),
            "fidelity": {s.id: self.fidelity(s.id).value for s in self.steps},
        }


# --------------------------------------------------------------------------- JSON schema (Task Engineer)


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {**schema, "type": [schema["type"], "null"]}


def _string(
    max_len: int, description: str, *, min_len: int = 0, pattern: str | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "string", "maxLength": max_len, "description": description}
    if min_len:
        out["minLength"] = min_len
    if pattern:
        out["pattern"] = pattern
    return out


def _object(props: dict[str, Any], description: str = "") -> dict[str, Any]:
    out = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    if description:
        out["description"] = description
    return out


def _object_instance_schema(t: ObjectTemplate) -> dict[str, Any]:
    ref = (
        _nullable(
            _string(
                70,
                "library:<part>, upload:<id> or scan:<id> when shape is mesh, else null",
                pattern=_REF.pattern,
            )
        )
        if t.needs_ref
        else {"type": "null", "description": "always null for this kind"}
    )
    lim = ObjectPose.LIMIT_MM
    return _object(
        {
            "name": _string(32, "Unique snake_case id of this object", pattern=_SLUG.pattern),
            "kind": {"type": "string", "const": t.kind.value, "description": t.description},
            "label": _string(_TEXT_MAX["label"], "Short generic on-screen name (no brand names)", min_len=1),
            "params": params_json_schema(t.params),
            "pose": _object(
                {
                    "x_mm": {"type": "number", "minimum": -lim, "maximum": lim},
                    "y_mm": {"type": "number", "minimum": -lim, "maximum": lim},
                    "yaw_deg": {"type": "number", "minimum": -180.0, "maximum": 180.0},
                },
                "Pose on the table, mm from its centre",
            ),
            "ref": ref,
        },
        f"{t.display_name}: {t.model}",
    )


def _step_schema(info: PrimitiveInfo) -> dict[str, Any]:
    """One primitive's step branch, as offered to the Task Engineer (:func:`planner_kinds`)."""
    kinds = planner_kinds(info)
    quantities = sorted(set(info.quantities) | {q for t in kinds for q in get_template(t).quantities})
    handles = sorted(set(planner_handles(info)))
    bounds = StepConstraints.BOUNDS
    constraints = _object(
        {
            name: {"type": ["number", "null"], "minimum": lo, "maximum": hi}
            for name, (lo, hi) in bounds.items()
        },
        "Task limits for this step (null: no limit)",
    )
    criterion = _object(
        {
            "quantity": {"type": "string", "enum": quantities},
            "at_least": {"type": ["number", "null"]},
            "at_most": {"type": ["number", "null"]},
        }
    )
    slug = {"type": "string", "pattern": _SLUG.pattern, "maxLength": 32}
    return _object(
        {
            "id": {**slug, "description": "Unique snake_case id of this step"},
            "primitive": {"type": "string", "const": info.kind.value, "description": info.description},
            "object": {**slug, "description": "Name of the object acted on"},
            "handle": {"type": "string", "enum": handles, "description": "Which feature of the object"},
            "target": (
                {
                    "type": ["string", "null"],
                    "pattern": _SLUG.pattern,
                    "maxLength": 32,
                    "description": "Destination object (a bin) or null",
                }
                if info.targets
                else {"type": "null"}
            ),
            "description": _string(_TEXT_MAX["note"], "One plain sentence for the step list", min_len=1),
            "params": params_json_schema(planner_params(info)),
            "constraints": constraints,
            "success": {"type": "array", "items": criterion, "minItems": 1, "maxItems": 4},
            "after": {"type": "array", "items": slug, "maxItems": MAX_STEPS},
            "hold_after": (
                {"type": "boolean", "description": "True if the step ends still holding the object"}
                if info.may_hold
                else {"type": "boolean", "const": False}
            ),
        },
        f"{info.kind.value}: {info.description}",
    )


def task_plan_json_schema() -> dict[str, Any]:
    """The strict, bounded schema of the Task Engineer's tool arguments.

    Every object is closed (``additionalProperties: false``) with every property required;
    every number and string is bounded; objects and steps are one ``anyOf`` branch per template
    kind and per primitive. Cross-references (step -> object, handle, quantities, order) are
    checked by :class:`TaskSpec` after parsing. It offers what the Engineer has been taught
    (:func:`planner_templates`; module docstring).
    """
    line = _string(_TEXT_MAX["line"], "", min_len=1)
    return _object(
        {
            "title": _string(_TEXT_MAX["title"], "Short use-case title", min_len=3),
            "description": _string(_TEXT_MAX["description"], "The use case in two or three sentences"),
            "objects": {
                "type": "array",
                "items": {"anyOf": [_object_instance_schema(t) for t in planner_templates()]},
                "maxItems": MAX_OBJECTS,
            },
            "steps": {
                "type": "array",
                "items": {"anyOf": [_step_schema(PRIMITIVES[k]) for k in PrimitiveKind]},
                "maxItems": MAX_STEPS,
            },
            "assumptions": {"type": "array", "items": line, "maxItems": 10},
            "clarifications": {
                "type": "array",
                "items": _object({"question": line, "assumed_answer": line}),
                "maxItems": 6,
            },
            "unsupported": {
                "type": "array",
                "items": _object(
                    {
                        "what": _string(120, "What is not simulated", min_len=1),
                        "reason": line,
                        "nearest_primitive": {
                            "type": ["string", "null"],
                            "enum": [*[k.value for k in PrimitiveKind], None],
                        },
                    }
                ),
                "maxItems": 6,
            },
            "refusal": {
                "type": ["string", "null"],
                "maxLength": 300,
                "description": "Why the whole use case cannot be simulated (then steps is empty), else null",
            },
        }
    )


def submit_task_plan_tool() -> dict[str, Any]:
    """The forced tool call (OpenAI-compatible ``tools`` entry) for the Task Engineer."""
    return {
        "type": "function",
        "function": {
            "name": TASK_PLAN_TOOL,
            "description": (
                "Submit the use case as objects and ordered steps (primitive x object x params x "
                "constraints x success criteria), with assumptions, clarifying questions and anything "
                "that is not simulated."
            ),
            "parameters": task_plan_json_schema(),
        },
    }


def _get(d: Mapping[str, Any], key: str, where: str, problems: list[str], default: Any = None) -> Any:
    if key not in d:
        problems.append(f"{where}: missing {key!r}")
        return default
    return d[key]


def task_spec_from_tool_args(args: Mapping[str, Any], *, source: TaskSource = "nemotron") -> TaskSpec:
    """Parse the tool arguments into a :class:`TaskSpec`, collecting every problem (with its
    path) before raising one :class:`TaskSpecError`, so a repair prompt can list them all."""
    problems: list[str] = []
    if not isinstance(args, Mapping):
        raise TaskSpecError(["the arguments must be a JSON object"])
    objects: list[ObjectInstance] = []
    for i, o in enumerate(_get(args, "objects", "$", problems, []) or []):
        where = f"objects[{i}]"
        try:
            pose = o.get("pose") or {}
            objects.append(
                ObjectInstance(
                    name=o["name"],
                    kind=o["kind"],
                    label=o.get("label", o.get("name", "")),
                    params=o["params"],
                    pose=ObjectPose(**pose),
                    ref=o.get("ref"),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            problems.append(f"{where}: {exc}")
    steps: list[Step] = []
    for i, s in enumerate(_get(args, "steps", "$", problems, []) or []):
        where = f"steps[{i}]"
        try:
            steps.append(
                Step(
                    id=s["id"],
                    primitive=s["primitive"],
                    object=s["object"],
                    handle=s["handle"],
                    description=s["description"],
                    params=s["params"],
                    success=tuple(SuccessCriterion(**c) for c in s["success"]),
                    constraints=StepConstraints(**(s.get("constraints") or {})),
                    target=s.get("target"),
                    after=tuple(s.get("after") or ()),
                    hold_after=bool(s.get("hold_after", False)),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            problems.append(f"{where}: {exc}")
    clar: list[Clarification] = []
    for i, c in enumerate(args.get("clarifications") or []):
        try:
            clar.append(Clarification(c["question"], c["assumed_answer"]))
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"clarifications[{i}]: {exc}")
    unsup: list[UnsupportedItem] = []
    for i, u in enumerate(args.get("unsupported") or []):
        try:
            unsup.append(UnsupportedItem(u["what"], u["reason"], u.get("nearest_primitive")))
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"unsupported[{i}]: {exc}")
    title = _get(args, "title", "$", problems, "")
    if problems:
        raise TaskSpecError(problems)
    return TaskSpec(
        title=title,
        description=args.get("description", ""),
        objects=tuple(objects),
        steps=tuple(steps),
        assumptions=tuple(args.get("assumptions") or ()),
        clarifications=tuple(clar),
        unsupported=tuple(unsup),
        refusal=args.get("refusal"),
        source=source,
    )


# --------------------------------------------------------------------------- the wrist (fairness)


@dataclass(frozen=True)
class WristLimits:
    """The Studio wrist's pitch and continuous roll, the same for every candidate [assumed values].

    The four core wrist servos (x, y, z, yaw) are the frozen ``task_v2`` ones, unchanged. The
    **pitch** is a limited hinge about the palm's x axis through the wrist centre, after the yaw
    and before the roll: at pitch 0 the tool points straight down, at +-``pitch_range_deg`` (90)
    it is horizontal. A stiff position servo drives it (``pitch_kp``, so a held pitch barely
    deflects under contact), its torque saturating at ``pitch_torque_nm`` (the core yaw servo's
    15 N m). The **roll** is a continuous hinge about the tool axis (vertical when a tool is
    presented at pitch 0) between the pitch stage and the flange, driven by a position servo whose
    torque saturates at ``roll_torque_nm`` (equal to the core yaw servo's 15 N m). Controllers may
    not move the roll target faster than ``roll_speed_max_deg_s``, nor the pitch target faster
    than ``pitch_speed_dps`` or past +-``pitch_range_deg``.
    """

    roll_torque_nm: float = 15.0
    roll_kp: float = 20.0  # N m / rad (the core yaw servo's gain)
    roll_kv: float = 1.0  # N m s / rad
    roll_armature: float = 0.001  # kg m^2
    roll_speed_max_deg_s: float = 180.0
    rotor_mass_kg: float = 0.05  # roll stage between the pitch stage and the flange
    flange_mass_kg: float = 0.10  # the flange body the tools mount on
    pitch_range_deg: float = 90.0  # +- from straight down
    pitch_kp: float = 500.0  # N m / rad: stiff, so a held pitch barely deflects under contact
    pitch_kv: float = 5.0  # N m s / rad
    pitch_torque_nm: float = 15.0  # the core yaw servo's
    pitch_armature: float = 0.005  # kg m^2 (a geared joint's reflected rotor inertia)
    pitch_speed_dps: float = 180.0  # deg/s, the cap on the pitch target's speed
    pitch_mass_kg: float = 0.05  # pitch stage between the core wrist and the roll stage

    def __post_init__(self) -> None:
        for name in (
            "roll_torque_nm",
            "roll_kp",
            "roll_kv",
            "roll_armature",
            "roll_speed_max_deg_s",
            "rotor_mass_kg",
            "flange_mass_kg",
            "pitch_range_deg",
            "pitch_kp",
            "pitch_kv",
            "pitch_torque_nm",
            "pitch_armature",
            "pitch_speed_dps",
            "pitch_mass_kg",
        ):
            if _number(name, getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in sorted(vars(self).items())}


#: The one wrist every Studio candidate uses (:mod:`rtt.studio.fairness` asserts it).
STUDIO_WRIST: Final = WristLimits()

# --------------------------------------------------------------------------- setups


@dataclass(frozen=True)
class HeadMount:
    """Where one head of a multi-tool head sits on its bracket.

    The bracket's pivot is the wrist centre (the flange centre). The head's axis is tilted
    ``angle_deg`` about the bracket's x axis from straight down, its mount plane is
    ``standoff_mm`` from the pivot along that axis (``lateral_mm`` along x), and the head is
    turned ``spin_deg`` about its own axis. Presenting a head tilts the flange and bracket by
    ``-angle_deg`` so that head points straight down on the roll axis (coaxial when
    ``lateral_mm`` is 0, which a twist step requires).
    """

    angle_deg: float
    standoff_mm: float = 60.0
    lateral_mm: float = 0.0
    spin_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "angle_deg", _number("angle_deg", self.angle_deg, -120.0, 120.0))
        object.__setattr__(self, "standoff_mm", _number("standoff_mm", self.standoff_mm, 40.0, 200.0))
        object.__setattr__(self, "lateral_mm", _number("lateral_mm", self.lateral_mm, -80.0, 80.0))
        object.__setattr__(self, "spin_deg", _number("spin_deg", self.spin_deg, -180.0, 180.0))

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in sorted(vars(self).items())}


@dataclass(frozen=True)
class BracketSpec:
    """The generated multi-tool bracket (a hub under the flange, one arm and adapter per head).

    ``density_kg_m3`` defaults to a printed carbon-filled polymer [assumed]; the bracket's mass
    and collisions are simulated, and S5 exports its geometry in the build pack.
    """

    hub_thickness_mm: float = 10.0
    arm_width_mm: float = 24.0
    arm_thickness_mm: float = 8.0
    adapter_thickness_mm: float = 8.0
    density_kg_m3: float = 1300.0

    BOUNDS: ClassVar[Mapping[str, tuple[float, float]]] = MappingProxyType(
        {
            "hub_thickness_mm": (4.0, 30.0),
            "arm_width_mm": (10.0, 80.0),
            "arm_thickness_mm": (3.0, 30.0),
            "adapter_thickness_mm": (3.0, 25.0),
            "density_kg_m3": (800.0, 8000.0),
        }
    )

    def __post_init__(self) -> None:
        for name, (lo, hi) in self.BOUNDS.items():
            object.__setattr__(self, name, _number(name, getattr(self, name), lo, hi))

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.BOUNDS}


@dataclass(frozen=True)
class ToolChanger:
    """An automatic tool changer: a master plate on the flange, a tool plate on every tool.

    ``changeover_s`` is paid at every change; ``stack_mm`` lowers every tool's mount plane; the
    master plate stays on the wrist and the tool plate travels with its tool (the mass swap).
    Defaults [assumed] are a small pneumatic changer's: :data:`TOOL_CHANGE_S` (7.5 s [assumed]:
    sources 5–10 s; one changer measured 7.33 s) and a 52 mm stack (master and tool plate
    coupled) [assumed]. Fairness requires every tool-change candidate in one comparison to use
    the same changer.
    """

    changeover_s: float = TOOL_CHANGE_S
    stack_mm: float = 52.0  # [assumed]
    master_mass_kg: float = 0.25
    tool_plate_mass_kg: float = 0.10

    def __post_init__(self) -> None:
        object.__setattr__(self, "changeover_s", _number("changeover_s", self.changeover_s, 0.5, 60.0))
        object.__setattr__(self, "stack_mm", _number("stack_mm", self.stack_mm, 5.0, 80.0))
        object.__setattr__(self, "master_mass_kg", _number("master_mass_kg", self.master_mass_kg, 0.01, 2.0))
        object.__setattr__(
            self, "tool_plate_mass_kg", _number("tool_plate_mass_kg", self.tool_plate_mass_kg, 0.01, 2.0)
        )

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in sorted(vars(self).items())}


@dataclass(frozen=True)
class Segment:
    """Consecutive steps that run as one continuous episode with one active tool.

    ``change`` happened before it (``none`` for the first); ``change_s`` is its time.
    """

    index: int
    tool: int
    step_ids: tuple[str, ...]
    change: ChangeKind
    change_s: float


@dataclass(frozen=True, eq=False)
class Setup:
    """What gets recommended (see :class:`SetupKind`).

    * ``tools``: the effectors. ``single``: one; ``multi_head``: two (head order, one
      :class:`HeadMount` each in ``mounts`` plus a :class:`BracketSpec`); ``tool_change``: two
      to :data:`MAX_TOOLS` distinct effectors and a :class:`ToolChanger`.
    * ``assignment``: step id -> index into ``tools`` (the per-step effector).
    * ``present_s``: the multi-tool head's presentation time (:data:`PRESENT_S`).

    :attr:`setup_id` hashes everything but ``label``. :meth:`problems` checks the setup
    against a task; :meth:`segments` is the tool-change hook (module ``rtt.studio.episode``).
    """

    kind: SetupKind
    tools: tuple[EffectorSpec, ...]
    assignment: Mapping[str, int]
    mounts: tuple[HeadMount, ...] = ()
    bracket: BracketSpec | None = None
    changer: ToolChanger | None = None
    present_s: float = PRESENT_S
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", SetupKind(self.kind))
        tools = tuple(self.tools)
        if not tools or not all(isinstance(t, EffectorSpec) for t in tools):
            raise ValueError("tools must be EffectorSpec values")
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "mounts", tuple(self.mounts))
        assignment = {}
        for k, v in dict(self.assignment).items():
            _slug("assignment step id", k)
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < len(tools):
                raise ValueError(f"assignment[{k!r}] must be a tool index in [0, {len(tools)})")
            assignment[k] = int(v)
        if not assignment:
            raise ValueError("a setup assigns at least one step")
        object.__setattr__(self, "assignment", FrozenDict(assignment))
        if self.label:
            _text("setup label", self.label, _TEXT_MAX["label"])
        object.__setattr__(self, "present_s", _number("present_s", self.present_s, 0.1, 30.0))
        ids = [t.effector_id for t in tools]
        if self.kind is SetupKind.SINGLE:
            if len(tools) != 1 or self.mounts or self.bracket or self.changer:
                raise ValueError("a single setup has one tool and no mounts, bracket or changer")
        elif self.kind is SetupKind.MULTI_HEAD:
            if len(tools) != 2 or len(self.mounts) != 2 or self.changer is not None:
                raise ValueError("a multi_head setup has two tools, two HeadMounts and no changer")
            if self.bracket is None:
                object.__setattr__(self, "bracket", BracketSpec())
            a, b = (m.angle_deg for m in self.mounts)
            if abs(a - b) < 45.0:
                raise ValueError(
                    "the two heads must be at least 45 deg apart, or the idle head points at the work"
                )
        else:
            if not 2 <= len(tools) <= MAX_TOOLS or self.mounts or self.bracket is not None:
                raise ValueError(f"a tool_change setup has 2 to {MAX_TOOLS} tools and no mounts or bracket")
            if len(set(ids)) != len(ids):
                raise ValueError("tool_change tools must be distinct effectors")
            if self.changer is None:
                object.__setattr__(self, "changer", ToolChanger())

    # ------------------------------------------------------------------ constructors

    @classmethod
    def single(cls, effector: EffectorSpec, step_ids: Iterable[str], label: str = "") -> Setup:
        return cls(SetupKind.SINGLE, (effector,), {s: 0 for s in step_ids}, label=label)

    @classmethod
    def multi_head(
        cls,
        heads: Sequence[tuple[EffectorSpec, HeadMount]],
        assignment: Mapping[str, int],
        bracket: BracketSpec | None = None,
        label: str = "",
    ) -> Setup:
        return cls(
            SetupKind.MULTI_HEAD,
            tuple(h[0] for h in heads),
            assignment,
            mounts=tuple(h[1] for h in heads),
            bracket=bracket or BracketSpec(),
            label=label,
        )

    @classmethod
    def tool_change(
        cls, per_step: Mapping[str, EffectorSpec], changer: ToolChanger | None = None, label: str = ""
    ) -> Setup:
        """Distinct tools in first-use order, and the per-step assignment."""
        tools: list[EffectorSpec] = []
        index: dict[str, int] = {}
        assignment = {}
        for step_id, spec in per_step.items():
            if spec.effector_id not in index:
                index[spec.effector_id] = len(tools)
                tools.append(spec)
            assignment[step_id] = index[spec.effector_id]
        return cls(
            SetupKind.TOOL_CHANGE, tuple(tools), assignment, changer=changer or ToolChanger(), label=label
        )

    # ------------------------------------------------------------------ identity

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "contract": CONTRACT_VERSION,
            "kind": self.kind.value,
            "tools": [t.effector_id for t in self.tools],
            "assignment": dict(sorted(self.assignment.items())),
            "mounts": [m.to_dict() for m in self.mounts],
            "bracket": self.bracket.to_dict() if self.bracket else None,
            "changer": self.changer.to_dict() if self.changer else None,
            "present_s": self.present_s if self.kind is SetupKind.MULTI_HEAD else None,
        }

    @property
    def setup_id(self) -> str:
        return _sha256(canonical_json(self.canonical_dict()))

    def to_dict(self) -> dict[str, Any]:
        out = self.canonical_dict()
        out.update(
            setup_id=self.setup_id,
            label=self.label,
            tools=[t.to_dict() for t in self.tools],
            product_label=PRODUCT_MODE_LABEL,
        )
        return out

    def effector_for(self, step_id: str) -> EffectorSpec:
        return self.tools[self.assignment[step_id]]

    # ------------------------------------------------------------------ against a task

    def problems(self, task: TaskSpec) -> list[str]:
        """Everything that makes this setup unable to run ``task`` (empty when it can)."""
        out: list[str] = []
        ids = task.step_ids
        missing = [s for s in ids if s not in self.assignment]
        extra = [s for s in self.assignment if s not in ids]
        if missing:
            out.append(f"no tool assigned to steps {missing}")
        if extra:
            out.append(f"assignment names steps the task does not have: {extra}")
        if out:
            return out
        used = {self.assignment[s] for s in ids}
        unused = [i for i in range(len(self.tools)) if i not in used]
        if unused and self.kind is not SetupKind.SINGLE:
            out.append(f"tools {unused} are never used")
        for prev, nxt in zip(task.steps, task.steps[1:], strict=False):
            if prev.hold_after and self.assignment[prev.id] != self.assignment[nxt.id]:
                out.append(f"the tool changes between {prev.id} and {nxt.id} while {prev.object!r} is held")
        if self.kind is SetupKind.MULTI_HEAD:
            for s in task.steps:
                m = self.mounts[self.assignment[s.id]]
                if s.info.uses_roll and m.lateral_mm != 0.0:
                    out.append(f"{s.id} twists with head {self.assignment[s.id]}, which is off the roll axis")
        return out

    def segments(self, task: TaskSpec) -> tuple[Segment, ...]:
        """The episodes ``task`` runs as: consecutive steps with the same tool form one segment;
        a new segment starts at every tool change (``changer.changeover_s``) or head switch
        (``present_s``). Raises ``ValueError`` if :meth:`problems` is not empty."""
        problems = self.problems(task)
        if problems:
            raise ValueError("; ".join(problems))
        out: list[Segment] = []
        current: list[str] = []
        tool = self.assignment[task.steps[0].id]
        for s in task.steps:
            t = self.assignment[s.id]
            if t != tool:
                out.append(self._segment(len(out), tool, current))
                current, tool = [], t
            current.append(s.id)
        out.append(self._segment(len(out), tool, current))
        return tuple(out)

    def _segment(self, index: int, tool: int, step_ids: list[str]) -> Segment:
        if index == 0:
            return Segment(0, tool, tuple(step_ids), ChangeKind.NONE, 0.0)
        if self.kind is SetupKind.MULTI_HEAD:
            return Segment(index, tool, tuple(step_ids), ChangeKind.PRESENT, self.present_s)
        assert self.changer is not None
        return Segment(index, tool, tuple(step_ids), ChangeKind.TOOL_CHANGE, self.changer.changeover_s)


# --------------------------------------------------------------------------- results

#: Why a step failed (``None`` when it succeeded).
STEP_FAILURE_MODES: Final[tuple[str, ...]] = (
    "criterion_not_met",  # the step ended without meeting a success criterion
    "constraint_exceeded",  # a task limit (force, torque) was exceeded
    "slip",  # the tool slipped off the handle or in rotation beyond tolerance
    "jam",  # an insertion stalled short of its depth
    "dropped",  # a held object was lost
    "physics_invalid",  # MuJoCo reset the state (bad acceleration)
    "physics_unreliable",  # penetration above 1 mm for more than 10 ms (any pair but designed ones)
    "invalid_build",  # the setup could not be built for this task
    "not_simulated",  # no physics for this primitive or template yet (never scored as a pass)
    "not_reached",  # an earlier step failed, so the task stopped
    "out_of_reach",  # the step's target lies outside the palm's travel (Studio v2)
)


@dataclass(frozen=True)
class StepOutcome:
    """One step of one full-task run."""

    step_id: str
    primitive: PrimitiveKind
    object: str
    tool: int
    effector_id: str
    success: bool
    failure_mode: str | None
    quantities: Mapping[str, float]
    fidelity: FidelityTier
    caveats: tuple[str, ...] = ()
    start_s: float = 0.0
    duration_s: float = 0.0
    traj_sha: str = ""
    controller: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "primitive", PrimitiveKind(self.primitive))
        object.__setattr__(self, "fidelity", FidelityTier(self.fidelity))
        if self.failure_mode is not None and self.failure_mode not in STEP_FAILURE_MODES:
            raise ValueError(f"unknown step failure mode {self.failure_mode!r}")
        if self.success == (self.failure_mode is not None):
            raise ValueError("a step succeeds exactly when it has no failure mode")
        if self.success and self.fidelity is FidelityTier.NOT_SIMULATED:
            raise ValueError("a not-simulated step can never succeed")
        if self.fidelity is FidelityTier.APPROXIMATION and not self.caveats:
            raise ValueError("an approximation-tier step result needs its caveat")
        q = {k: float(v) for k, v in self.quantities.items()}
        object.__setattr__(self, "quantities", FrozenDict(q))
        object.__setattr__(self, "caveats", tuple(self.caveats))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "primitive": self.primitive.value,
            "object": self.object,
            "tool": self.tool,
            "effector_id": self.effector_id,
            "success": self.success,
            "failure_mode": self.failure_mode,
            "quantities": {k: (v if math.isfinite(v) else None) for k, v in sorted(self.quantities.items())},
            "fidelity": self.fidelity.value,
            "caveats": list(self.caveats),
            "start_s": round(self.start_s, 4),
            "duration_s": round(self.duration_s, 4),
            "traj_sha": self.traj_sha,
            "controller": self.controller,
        }


@dataclass(frozen=True)
class TaskResult:
    """One full-task run of one setup under one condition and seed.

    ``completed``: every step succeeded. ``cycle_time_s``: the task clock at the end, step
    durations plus every change (``change_time_s``); for an incomplete run it is the clock when
    the task stopped. ``tool_mass_kg``: the tool stack carried in each segment (the mass swap).
    ``fairness_sha``: the hash of the wrist, solver and Studio constants the run used
    (:mod:`rtt.studio.fairness`); ``controllers``: ``"default"`` or ``"reference"`` (the
    trivial S1 references, never a recommendation's evidence).
    """

    task_id: str
    setup_id: str
    setup_kind: SetupKind
    condition_id: str
    seed: int
    steps: tuple[StepOutcome, ...]
    cycle_time_s: float
    change_time_s: float
    n_tool_changes: int
    n_presentations: int
    tool_mass_kg: tuple[float, ...]
    traj_sha: str
    fairness_sha: str
    controllers: Literal["default", "reference"] = "default"
    wall_time_s: float = 0.0
    label: str = PRODUCT_MODE_LABEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup_kind", SetupKind(self.setup_kind))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "tool_mass_kg", tuple(float(m) for m in self.tool_mass_kg))
        if not self.steps:
            raise ValueError("a task result has at least one step outcome")
        if self.controllers not in ("default", "reference"):
            raise ValueError("controllers must be default or reference")

    @property
    def completed(self) -> bool:
        return all(s.success for s in self.steps)

    @property
    def fidelity(self) -> FidelityTier:
        return worst_tier(*(s.fidelity for s in self.steps))

    def to_record(self) -> dict[str, Any]:
        return {
            "contract": CONTRACT_VERSION,
            "label": self.label,
            "task_id": self.task_id,
            "setup_id": self.setup_id,
            "setup_kind": self.setup_kind.value,
            "condition_id": self.condition_id,
            "seed": int(self.seed),
            "completed": self.completed,
            "cycle_time_s": round(self.cycle_time_s, 4),
            "change_time_s": round(self.change_time_s, 4),
            "n_tool_changes": self.n_tool_changes,
            "n_presentations": self.n_presentations,
            "tool_mass_kg": [round(m, 5) for m in self.tool_mass_kg],
            "fidelity": self.fidelity.value,
            "traj_sha": self.traj_sha,
            "fairness_sha": self.fairness_sha,
            "controllers": self.controllers,
            "wall_time_s": round(self.wall_time_s, 3),
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass(frozen=True)
class ComplexityMetrics:
    """How complex a setup is to build and run (lower is simpler; the last ranking key)."""

    n_tools: int
    n_actuators: int
    tool_mass_kg: float  # heaviest stack the wrist carries in any segment
    total_mass_kg: float  # everything built: tools, bracket or changer plates
    n_changes: int  # tool changes plus presentations in one run
    bracket: bool = False

    def key(self) -> tuple[float, ...]:
        return (self.n_tools, self.n_actuators, round(self.total_mass_kg, 4), self.n_changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tools": self.n_tools,
            "n_actuators": self.n_actuators,
            "tool_mass_kg": round(self.tool_mass_kg, 5),
            "total_mass_kg": round(self.total_mass_kg, 5),
            "n_changes": self.n_changes,
            "bracket": self.bracket,
        }


@dataclass(frozen=True)
class CertificateRef:
    """Evidence behind a step: a v1 grasp certificate or a Studio step certificate.

    ``ref`` is a repository-relative path or a sha256; ``score`` its headline score, if any.
    """

    step_id: str
    effector_id: str
    kind: Literal["v1_grasp", "step"]
    ref: str
    score: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("v1_grasp", "step"):
            raise ValueError("certificate kind must be v1_grasp or step")
        if not self.ref or self.ref.startswith("/") or ".." in self.ref.split("/"):
            raise ValueError("certificate ref must be a relative path or a sha256")
        if self.score is not None:
            object.__setattr__(self, "score", _number("certificate score", self.score, 0.0, 1.0))


@dataclass(frozen=True)
class SetupResult:
    """A setup's runs on one task, aggregated for ranking (:meth:`rank_key`).

    Ranking (design section 5.3): full-task completion rate (higher first), then the weakest
    step's success rate, then median cycle time including changes, then complexity. The
    simplicity rule within a small margin is applied by the recommender (S5) on top.
    """

    setup: Setup
    task_id: str
    runs: tuple[TaskResult, ...]
    complexity: ComplexityMetrics
    certificates: tuple[CertificateRef, ...] = ()
    label: str = PRODUCT_MODE_LABEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs", tuple(self.runs))
        object.__setattr__(self, "certificates", tuple(self.certificates))
        if not self.runs:
            raise ValueError("a setup result needs at least one run")
        bad = [r for r in self.runs if r.task_id != self.task_id or r.setup_id != self.setup.setup_id]
        if bad:
            raise ValueError("every run must be of this task and this setup")

    @property
    def completion_rate(self) -> float:
        return sum(r.completed for r in self.runs) / len(self.runs)

    @property
    def step_success(self) -> dict[str, float]:
        ids = [s.step_id for s in self.runs[0].steps]
        return {
            i: sum(next(s for s in r.steps if s.step_id == i).success for r in self.runs) / len(self.runs)
            for i in ids
        }

    @property
    def weakest_step(self) -> tuple[str, float]:
        rates = self.step_success
        worst = min(rates, key=lambda k: (rates[k], k))
        return worst, rates[worst]

    @property
    def cycle_time_s(self) -> float:
        """Median cycle time of the completed runs (of all runs when none completed)."""
        done = [r.cycle_time_s for r in self.runs if r.completed] or [r.cycle_time_s for r in self.runs]
        return float(statistics.median(done))

    @property
    def fidelity(self) -> FidelityTier:
        return worst_tier(*(r.fidelity for r in self.runs))

    @property
    def caveats(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c for r in self.runs for s in r.steps for c in s.caveats))

    def rank_key(self) -> tuple[float, ...]:
        """Sort ascending: the best setup first."""
        return (
            -self.completion_rate,
            -self.weakest_step[1],
            round(self.cycle_time_s, 3),
            *self.complexity.key(),
        )

    def to_dict(self) -> dict[str, Any]:
        weakest, rate = self.weakest_step
        return {
            "contract": CONTRACT_VERSION,
            "label": self.label,
            "task_id": self.task_id,
            "setup": self.setup.to_dict(),
            "n_runs": len(self.runs),
            "completion_rate": round(self.completion_rate, 4),
            "step_success": {k: round(v, 4) for k, v in self.step_success.items()},
            "weakest_step": {"step_id": weakest, "success_rate": round(rate, 4)},
            "cycle_time_s": round(self.cycle_time_s, 3),
            "complexity": self.complexity.to_dict(),
            "fidelity": self.fidelity.value,
            "caveats": list(self.caveats),
            "certificates": [
                {
                    "step_id": c.step_id,
                    "effector_id": c.effector_id,
                    "kind": c.kind,
                    "ref": c.ref,
                    "score": c.score,
                }
                for c in self.certificates
            ],
        }


# --------------------------------------------------------------------------- build pack

BuildFileKind = Literal[
    "effector_cad", "adapter", "bracket", "changer_plate", "mjcf", "bom", "certificate", "notes", "licence"
]
BUILD_FILE_KINDS: Final[tuple[str, ...]] = (
    "effector_cad",
    "adapter",
    "bracket",
    "changer_plate",
    "mjcf",
    "bom",
    "certificate",
    "notes",
    "licence",
)
BUILD_FILE_FORMATS: Final[tuple[str, ...]] = ("stl", "step", "xml", "csv", "html", "md", "txt", "json")
_MESH_KINDS = ("effector_cad", "adapter", "bracket", "changer_plate")


@dataclass(frozen=True)
class BuildPackFile:
    """One redistributed file of a build pack.

    ``source`` says where it came from (``library``: an Effector Library item, ``generated``: our
    own geometry, ``studio``: a Studio artefact such as the MJCF or the certificate);
    ``source_ref`` names the library item (or the generator); ``watertight`` is required for
    every mesh of a printable part.
    """

    path: str
    sha256: str
    kind: BuildFileKind
    format: str
    license: str
    source: Literal["library", "generated", "studio"]
    source_ref: str
    attribution: str = ""
    watertight: bool | None = None

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith("/") or ".." in self.path.split("/"):
            raise ValueError(f"build pack path must be relative, got {self.path!r}")
        if not _SHA256.match(self.sha256):
            raise ValueError("sha256 must be lowercase hex")
        if self.kind not in BUILD_FILE_KINDS:
            raise ValueError(f"kind must be one of {BUILD_FILE_KINDS}")
        if self.format not in BUILD_FILE_FORMATS:
            raise ValueError(f"format must be one of {BUILD_FILE_FORMATS}")
        if not _SPDX.match(self.license) or self.license not in ALLOWED_LICENSES:
            raise ValueError(
                f"licence {self.license!r} may not be redistributed; allowed: {ALLOWED_LICENSES}"
            )
        if self.source not in ("library", "generated", "studio"):
            raise ValueError("source must be library, generated or studio")
        if not self.source_ref:
            raise ValueError("source_ref is required")
        if self.source == "library" and not self.attribution:
            raise ValueError(f"{self.path}: a library file needs its attribution")
        if self.kind in _MESH_KINDS and self.format == "stl" and self.watertight is None:
            raise ValueError(f"{self.path}: a printable mesh must state watertight")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class BuildPackLink:
    """An item whose files may not be redistributed: an upstream link instead of a file."""

    name: str
    url: str
    license: str
    reason: str

    def __post_init__(self) -> None:
        _text("link name", self.name, _TEXT_MAX["label"])
        if not self.url.startswith("https://"):
            raise ValueError("link url must be https")
        if not self.reason:
            raise ValueError("a link needs the reason its files are not included")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "url": self.url, "license": self.license, "reason": self.reason}


@dataclass(frozen=True)
class BomLine:
    """One bill-of-materials line (open hardware, servos, fasteners)."""

    item: str
    qty: int
    link: str | None = None
    license: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        _text("bom item", self.item, 120)
        if isinstance(self.qty, bool) or not isinstance(self.qty, int) or not 1 <= self.qty <= 1000:
            raise ValueError("bom qty must be an integer in [1, 1000]")
        if self.link is not None and not self.link.startswith("https://"):
            raise ValueError("bom link must be https")

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "qty": self.qty,
            "link": self.link,
            "license": self.license,
            "note": self.note,
        }


@dataclass(frozen=True)
class BuildPackManifest:
    """The manifest of a build pack (S5 writes it, S7 renders it); ``manifest.json`` in the pack.

    It must contain the full-setup MJCF, the certificate and the bill of materials as files,
    a CAD file or an upstream link for every tool, and every caveat of the setup's steps.
    """

    task_id: str
    setup_id: str
    title: str
    files: tuple[BuildPackFile, ...]
    bom: tuple[BomLine, ...]
    links: tuple[BuildPackLink, ...] = ()
    caveats: tuple[str, ...] = ()
    contract: str = CONTRACT_VERSION
    label: str = PRODUCT_MODE_LABEL

    def __post_init__(self) -> None:
        for name in ("files", "bom", "links", "caveats"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not (_SHA256.match(self.task_id) and _SHA256.match(self.setup_id)):
            raise ValueError("task_id and setup_id must be sha256 ids")
        _text("build pack title", self.title, _TEXT_MAX["title"])
        paths = [f.path for f in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("build pack paths must be unique")
        kinds = {f.kind for f in self.files}
        missing = [k for k in ("mjcf", "certificate", "bom") if k not in kinds]
        if missing:
            raise ValueError(f"a build pack needs {missing}")
        if self.contract != CONTRACT_VERSION or self.label != PRODUCT_MODE_LABEL:
            raise ValueError("a build pack carries this contract and the product-mode label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "label": self.label,
            "task_id": self.task_id,
            "setup_id": self.setup_id,
            "title": self.title,
            "files": [f.to_dict() for f in self.files],
            "links": [link.to_dict() for link in self.links],
            "bom": [b.to_dict() for b in self.bom],
            "caveats": list(self.caveats),
        }


def buildpack_manifest_json_schema() -> dict[str, Any]:
    """The JSON schema of :meth:`BuildPackManifest.to_dict` (closed objects, bounded strings)."""
    sha = {"type": "string", "pattern": _SHA256.pattern}
    lic = {"type": "string", "enum": list(ALLOWED_LICENSES)}
    file_schema = _object(
        {
            "path": {"type": "string", "pattern": r"^[^/.][^\0]*$", "maxLength": 200},
            "sha256": sha,
            "kind": {"type": "string", "enum": list(BUILD_FILE_KINDS)},
            "format": {"type": "string", "enum": list(BUILD_FILE_FORMATS)},
            "license": lic,
            "source": {"type": "string", "enum": ["library", "generated", "studio"]},
            "source_ref": {"type": "string", "minLength": 1, "maxLength": 200},
            "attribution": {"type": "string", "maxLength": 300},
            "watertight": {"type": ["boolean", "null"]},
        }
    )
    link_schema = _object(
        {
            "name": {"type": "string", "minLength": 1, "maxLength": 60},
            "url": {"type": "string", "pattern": r"^https://", "maxLength": 300},
            "license": {"type": "string", "maxLength": 60},
            "reason": {"type": "string", "minLength": 1, "maxLength": 200},
        }
    )
    bom_schema = _object(
        {
            "item": {"type": "string", "minLength": 1, "maxLength": 120},
            "qty": {"type": "integer", "minimum": 1, "maximum": 1000},
            "link": {"type": ["string", "null"], "pattern": r"^https://", "maxLength": 300},
            "license": {"type": ["string", "null"], "maxLength": 60},
            "note": {"type": "string", "maxLength": 200},
        }
    )
    return _object(
        {
            "contract": {"type": "string", "const": CONTRACT_VERSION},
            "label": {"type": "string", "const": PRODUCT_MODE_LABEL},
            "task_id": sha,
            "setup_id": sha,
            "title": {"type": "string", "minLength": 1, "maxLength": _TEXT_MAX["title"]},
            "files": {"type": "array", "items": file_schema, "minItems": 3, "maxItems": 200},
            "links": {"type": "array", "items": link_schema, "maxItems": 50},
            "bom": {"type": "array", "items": bom_schema, "maxItems": 200},
            "caveats": {"type": "array", "items": {"type": "string", "maxLength": 300}, "maxItems": 20},
        }
    )
