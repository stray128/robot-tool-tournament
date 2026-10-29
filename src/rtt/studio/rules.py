"""Integrator rules the Studio's search and recommender follow (track S5, product mode).

Published integrator practice, cited by rule id (``docs/STUDIO_SEARCH.md`` section 5):

* **R01** (single effector by default): the recommender keeps the best single tool unless it fails
  a step outright or a more complex setup clears a margin per added piece
  (:data:`rtt.studio.recommend.SIMPLICITY_MARGIN`).
* **R03** (articulated objects use the existing gripper): soft arms, tentacles and dexterous hands
  do not open lids, doors or drawers or press buttons unless every rigid option fails
  (:func:`rtt.studio.search.rule_ineligible`, with the search's readmit fallback).
* **R05** (process steps need process tools): a twist or insert on a threaded member that must
  reach :data:`PROCESS_TORQUE_NM` or more, and a push that presses a tile into its bed, are process
  steps (:func:`process_need`); only the process tool of that kind does them (a nut runner, a tamper
  pad: :func:`process_kind`), see :func:`rtt.studio.search.rule_ineligible`. The rule makes one
  exception, for light work at low torque: an ordinary hand tool held in a gripper, marked as a
  makeshift setup. The Studio does not model hand tools held in a gripper, so under the threshold
  the gripper turns the member directly and no mark is shown.
* **R07** (tool change only when kinds differ): a change costs
  :data:`~rtt.studio.types.TOOL_CHANGE_S`; the sensitivity report re-ranks over 3 to 12 s.
* **R09** (soft only in its niche): soft arms do not insert and do not carry parts over
  :data:`SOFT_ARM_MAX_PART_KG`; setups with one rank below every rigid setup.
* **R10** (dexterous hand last): setups with a dexterous hand rank below every rigid setup and
  below soft arms (the largest complexity penalty): eligible, never preferred.

"Rigid" here is the rules' sense: a setup with no soft arm and no dexterous hand (a suction cup
counts as rigid), not the fidelity tier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Final, Literal

from rtt.effectors.types import EffectorFamily, EffectorSpec
from rtt.studio.types import THREAD_KINDS, ObjectInstance, PrimitiveKind, Setup, Step

__all__ = [
    "ARTICULATED_TEMPLATES",
    "HOOK_MIN_MM",
    "HOOK_THROAT_MM",
    "PRESS_TEMPLATES",
    "PROCESS_TOOLS",
    "PROCESS_TORQUE_NM",
    "RULE_CLASSES",
    "SOCKET_FIT_MM",
    "SOFT_ARM_MAX_PART_KG",
    "THREADED_TEMPLATES",
    "TORQUE_QUANTITIES",
    "ProcessKind",
    "RuleClass",
    "articulated",
    "is_dexterous_hand",
    "is_soft_arm",
    "is_tentacle",
    "part_mass_kg",
    "process_kind",
    "process_need",
    "required_torque_nm",
    "rule_class",
    "setup_rule_class",
]

#: Object templates whose steps open or press a mechanism (R03), by value, so templates added
#: later (a sliding door, a spring button) are covered as soon as they exist.
ARTICULATED_TEMPLATES: Final = frozenset({"hinged_lid", "sliding_door", "spring_button", "drawer"})
#: Heaviest part a soft arm may carry (R09: commercial soft grippers are rated 2 to 3 kg) [assumed].
SOFT_ARM_MAX_PART_KG: Final = 2.5
#: A rigid tool hooks (R03's "gripper-held passive tool, hook") only if its tip reaches at least
#: this far sideways past the rest of the tool and the flange: the hook pull goes in 3 mm short of
#: a hook's reach, so this leaves 2 mm under a lip [assumed].
HOOK_MIN_MM: Final = 5.0
#: Room a lip or handle needs above a hook before the flange (the thinnest feature, a 4 mm lip,
#: plus the 2 mm the hook goes in under it) [assumed]; a flange closer than this blocks the hook.
HOOK_THROAT_MM: Final = 6.0

#: R05: a twist or insert on a threaded member that must reach this torque or more (N m) is a
#: fastening step for a nut runner [assumed]. Hand-tight, what a person reaches with the fingers, is
#: about 3 N m (the pipe coupling hero asks for at least 3 N m); the threshold sits 1 N m above that,
#: so a hand-tight fitting stays a gripper's job and anything tighter (a 5 N m clamp) is not.
PROCESS_TORQUE_NM: Final = 4.0
#: A nut runner's socket drives a hex within this much under its own size across flats (mm)
#: [assumed: standard hex sizes are 1 mm apart]; a bigger one does not enter its cavity
#: (:func:`rtt.studio.search.fit_problem`).
SOCKET_FIT_MM: Final = 0.5
#: Object templates with a threaded member (R05), by value: the kinds the twist and the insert
#: drive as a thread (:data:`rtt.studio.types.THREAD_KINDS`: the threaded joint and a panel's clamp
#: on its mounting rail), one set kept there.
THREADED_TEMPLATES: Final = frozenset(kind.value for kind in THREAD_KINDS)
#: Object templates a push presses into place as a process (R05), by value: a tile on its bed.
PRESS_TEMPLATES: Final = frozenset({"tile_bed"})
#: Criterion quantities that are a torque the member must reach (R05), by name: the wrist's peak
#: torque and a clamp's seating torque (covered as soon as that quantity exists).
TORQUE_QUANTITIES: Final = frozenset({"torque_nm", "clamp_torque_nm"})

ProcessKind = Literal["nut_runner", "tamper"]
#: The process tools (R05) by the primitive their spec names, and what each is, in words.
PROCESS_TOOLS: Final[Mapping[str, tuple[ProcessKind, str]]] = MappingProxyType(
    {"drive": ("nut_runner", "a nut runner"), "press": ("tamper", "a tamper pad")}
)

RuleClass = Literal["rigid", "soft_arm", "hand"]
#: Rank order of the classes (R09, R10): every rigid setup first, then soft arms, then hands.
RULE_CLASSES: Final[tuple[RuleClass, ...]] = ("rigid", "soft_arm", "hand")


def is_soft_arm(spec: EffectorSpec) -> bool:
    """A soft arm (the soft-arm family's spring-jointed tentacles)."""
    return spec.family is EffectorFamily.SOFT_ARM


def is_tentacle(spec: EffectorSpec) -> bool:
    """A soft arm or a designed segmented tentacle (R03 names tentacles beside soft arms)."""
    if is_soft_arm(spec):
        return True
    return spec.family is EffectorFamily.DESIGNED and _designed_kind(spec) == "segmented_tentacle"


def is_dexterous_hand(spec: EffectorSpec) -> bool:
    """A dexterous multi-finger hand: a catalog hand or a designed fingered hand (R03, R10)."""
    if spec.family is EffectorFamily.CATALOG_HAND:
        return True
    return spec.family is EffectorFamily.DESIGNED and _designed_kind(spec) == "fingered_hand"


def _designed_kind(spec: EffectorSpec) -> str | None:
    return spec.recipe.kind if spec.recipe is not None else None


def rule_class(spec: EffectorSpec) -> RuleClass:
    """``hand`` (R10), ``soft_arm`` (R09) or ``rigid``."""
    if is_dexterous_hand(spec):
        return "hand"
    if is_soft_arm(spec):
        return "soft_arm"
    return "rigid"


def setup_rule_class(setup: Setup | Iterable[EffectorSpec]) -> RuleClass:
    """The last-ranked class of any of a setup's tools (a hand outranks a soft arm)."""
    tools = setup.tools if isinstance(setup, Setup) else tuple(setup)
    return max((rule_class(t) for t in tools), key=RULE_CLASSES.index, default="rigid")


def articulated(obj: ObjectInstance) -> bool:
    """Whether a step on ``obj`` opens or presses a mechanism (R03)."""
    return obj.kind.value in ARTICULATED_TEMPLATES


def part_mass_kg(obj: ObjectInstance) -> float | None:
    """A box or cylinder part's stated mass (kg); ``None`` for anything else (a mesh part's mass
    comes from its material, known only once built)."""
    if obj.kind.value != "part" or obj.params.get("shape") not in ("box", "cylinder"):
        return None
    mass = obj.params.get("mass_g")
    return float(mass) / 1000.0 if isinstance(mass, int | float) else None


def process_kind(spec: EffectorSpec) -> ProcessKind | None:
    """Which process tool ``spec`` is (R05): ``nut_runner``, ``tamper``, or ``None`` for any tool
    that is not one (by the process family's primitive)."""
    if spec.family is not EffectorFamily.PROCESS_TOOL or spec.primitive not in PROCESS_TOOLS:
        return None
    return PROCESS_TOOLS[spec.primitive][0]


def required_torque_nm(step: Step) -> float | None:
    """The largest torque the step's criteria require the member to reach (N m), or ``None``."""
    need = [
        float(c.at_least) for c in step.success if c.quantity in TORQUE_QUANTITIES and c.at_least is not None
    ]
    return max(need, default=None)


def process_need(step: Step, obj: Any) -> ProcessKind | None:
    """The process tool a step needs (R05), or ``None`` when it is not a process step:

    * ``nut_runner``: a twist or insert on a threaded member (:data:`THREADED_TEMPLATES`) whose
      criteria require a torque of :data:`PROCESS_TORQUE_NM` or more [assumed];
    * ``tamper``: a push on a tile on its bed (:data:`PRESS_TEMPLATES`).

    ``obj`` is the step's object (its template kind is matched by value)."""
    kind = obj.kind.value
    if step.primitive in (PrimitiveKind.TWIST, PrimitiveKind.INSERT) and kind in THREADED_TEMPLATES:
        torque = required_torque_nm(step)
        if torque is not None and torque >= PROCESS_TORQUE_NM - 1e-9:
            return "nut_runner"
    if step.primitive is PrimitiveKind.PUSH and kind in PRESS_TEMPLATES:
        return "tamper"
    return None
