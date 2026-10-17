"""Design spaces: the :class:`~rtt.effectors.types.ParamSpec` tables of every generator.

Our own generators (family :attr:`~rtt.effectors.types.EffectorFamily.DESIGNED`, built by
:mod:`rtt.effectors.morphology.generator`):

* ``fingered_hand``: 2-5 fingers of 1-3 phalanges on a round palm, evenly spaced or with one
  thumb facing the others, flat or round pads, one servo per joint or one tendon per finger,
  a power (curl) or pinch (upright fingertip) synergy.
* ``hook_gripper``: a pad plus hook hybrid. 2-4 rigid fingers, each with a flat pad and an
  inward hook lip at its tip, that either swing about a knuckle or slide straight in toward
  the centre (a chuck).
* ``segmented_tentacle``: 1-4 curling digits of 3-8 capsule segments, tapered, each digit
  curled by one tendon; joint springs set which end curls first.

Parametric families of other tracks join through the registry: every family module that
loads and declares :meth:`~rtt.effectors.registry.EffectorFamilyImpl.recipe_kinds` adds its
spaces to :func:`design_spaces` (the soft arm and the suction cup, once they land). A family
that is not built yet is skipped.

Every space is a tuple of :class:`ParamSpec` rows in a fixed order; that order is the order
of the forced tool's JSON schema (:func:`~rtt.effectors.types.params_json_schema`) and of
the parameter tables in the prompt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from rtt.effectors.types import EffectorFamily, ParamSpec

__all__ = [
    "DESIGNED_KINDS",
    "FINGERED_HAND",
    "FOREIGN_FAMILIES",
    "HOOK_GRIPPER",
    "KIND_SUMMARIES",
    "MATERIAL_DENSITY",
    "SEGMENTED_TENTACLE",
    "SpaceKey",
    "design_spaces",
    "kind_summary",
    "parse_kind",
    "select_spaces",
    "space_for",
    "space_label",
]

SpaceKey = tuple[EffectorFamily, str]

#: Densities (kg/m^3) of the structural materials a designed effector can be made of.
MATERIAL_DENSITY: Final[Mapping[str, float]] = {"pla": 1240.0, "aluminium": 2700.0, "steel": 7850.0}

_MATERIAL = ParamSpec(
    "material",
    "choice",
    "Structural material: printed PLA, machined aluminium (2.2x PLA's mass) or steel (6.3x); "
    "it sets mass and inertia, never friction.",
    choices=tuple(MATERIAL_DENSITY),
    default="pla",
)
_LAYOUT = ParamSpec(
    "layout",
    "choice",
    "Where the digits sit around the palm: evenly spaced, or one thumb facing all the others.",
    choices=("even", "opposed"),
    default="even",
)
_SPREAD = ParamSpec(
    "spread_deg",
    "float",
    "Opposed layout with 3 or more digits: the angle spanned by the digits that face the thumb "
    "(ignored otherwise).",
    "deg",
    0.0,
    150.0,
    default=60.0,
)
_PALM_THICKNESS = ParamSpec(
    "palm_thickness_mm",
    "float",
    "Thickness of the palm plate under the wrist flange; the knuckles hang from its underside.",
    "mm",
    4.0,
    30.0,
    default=8.0,
)
_GRIP_FORCE = ParamSpec(
    "grip_force_n",
    "float",
    "Total closing force shared by the digits at their nominal contact point (budget: 50 N in "
    "total and 1.5 N m per joint).",
    "N",
    5.0,
    50.0,
    default=30.0,
)
_GRIP_HEIGHT = ParamSpec(
    "grip_height_frac",
    "float",
    "Height of the effector's lowest point above the table at the grasp, as a fraction of the "
    "part's height (never below 3 mm).",
    "",
    0.0,
    0.9,
    default=0.3,
)
_YAW = ParamSpec(
    "yaw_deg",
    "float",
    "Rotation of the whole effector about the vertical axis relative to the part (0 puts digit 1 "
    "on the part's +y side).",
    "deg",
    0.0,
    180.0,
    default=0.0,
)

FINGERED_HAND: Final[tuple[ParamSpec, ...]] = (
    ParamSpec("n_fingers", "int", "Number of fingers on the palm.", "count", 2, 5, default=3),
    ParamSpec(
        "n_phalanges",
        "int",
        "Links (and joints) per finger, from the knuckle to the tip.",
        "count",
        1,
        3,
        default=2,
    ),
    _LAYOUT,
    _SPREAD,
    ParamSpec(
        "palm_radius_mm",
        "float",
        "Radius of the circle the knuckles sit on; with the fingers hanging straight down, a "
        "finger's inner face is this minus half its thickness and the pad.",
        "mm",
        15.0,
        70.0,
        default=35.0,
    ),
    _PALM_THICKNESS,
    ParamSpec(
        "finger_length_mm",
        "float",
        "Length of one finger from the knuckle to the tip (all phalanges).",
        "mm",
        30.0,
        150.0,
        default=70.0,
    ),
    ParamSpec(
        "proximal_frac",
        "float",
        "Share of the finger length in the first phalanx; the rest is split evenly between the "
        "others (ignored with 1 phalanx).",
        "",
        0.3,
        0.8,
        default=0.5,
    ),
    ParamSpec(
        "finger_width_mm",
        "float",
        "Finger width across its closing plane (the contact patch width).",
        "mm",
        6.0,
        30.0,
        default=14.0,
    ),
    ParamSpec(
        "finger_thickness_mm",
        "float",
        "Finger thickness along the closing direction.",
        "mm",
        3.0,
        16.0,
        default=8.0,
    ),
    ParamSpec(
        "pad",
        "choice",
        "Contact pad on the inner face of every phalanx: none, a flat block, or a round bump "
        "across the finger.",
        choices=("none", "flat", "round"),
        default="flat",
    ),
    ParamSpec(
        "pad_thickness_mm",
        "float",
        "How far the pad stands out from the finger's inner face (ignored without a pad).",
        "mm",
        1.0,
        8.0,
        default=3.0,
    ),
    ParamSpec(
        "actuation",
        "choice",
        "per_joint: one position servo per joint; tendon: one servo per finger pulling a tendon "
        "over all its joints, so the outer phalanges keep closing after the first one is stopped.",
        choices=("per_joint", "tendon"),
        default="per_joint",
    ),
    ParamSpec(
        "grasp",
        "choice",
        "power: every joint curls to wrap the part; pinch: the knuckles close while the last "
        "phalanx stays upright (per_joint) or follows passively on a spring (tendon).",
        choices=("power", "pinch"),
        default="pinch",
    ),
    _MATERIAL,
    _GRIP_FORCE,
    _GRIP_HEIGHT,
    _YAW,
)

HOOK_GRIPPER: Final[tuple[ParamSpec, ...]] = (
    ParamSpec("n_fingers", "int", "Number of rigid fingers.", "count", 2, 4, default=3),
    _LAYOUT,
    _SPREAD,
    ParamSpec(
        "motion",
        "choice",
        "swing: each finger hinges at its knuckle and swings in; slide: each finger slides "
        "straight in toward the centre and stays upright (a chuck).",
        choices=("swing", "slide"),
        default="slide",
    ),
    ParamSpec(
        "palm_radius_mm",
        "float",
        "Radius of the circle the finger roots sit on in the open pose.",
        "mm",
        15.0,
        70.0,
        default=35.0,
    ),
    _PALM_THICKNESS,
    ParamSpec(
        "finger_length_mm",
        "float",
        "Length of a finger from its root to the underside of its hook.",
        "mm",
        20.0,
        150.0,
        default=60.0,
    ),
    ParamSpec(
        "finger_width_mm",
        "float",
        "Finger width across its closing plane.",
        "mm",
        6.0,
        30.0,
        default=14.0,
    ),
    ParamSpec(
        "finger_thickness_mm",
        "float",
        "Finger thickness along the closing direction.",
        "mm",
        3.0,
        16.0,
        default=8.0,
    ),
    ParamSpec(
        "pad_thickness_mm",
        "float",
        "How far the flat pad above the hook stands out from the finger's inner face.",
        "mm",
        1.0,
        8.0,
        default=2.0,
    ),
    ParamSpec(
        "hook_depth_mm",
        "float",
        "How far the hook lip at the fingertip reaches in beyond the pad (0: no hook, a plain "
        "padded finger).",
        "mm",
        0.0,
        10.0,
        default=3.0,
    ),
    ParamSpec(
        "hook_height_mm",
        "float",
        "Vertical thickness of the hook lip (ignored without a hook).",
        "mm",
        2.0,
        10.0,
        default=4.0,
    ),
    _MATERIAL,
    _GRIP_FORCE,
    _GRIP_HEIGHT,
    _YAW,
)

SEGMENTED_TENTACLE: Final[tuple[ParamSpec, ...]] = (
    ParamSpec("n_digits", "int", "Number of curling digits.", "count", 1, 4, default=3),
    ParamSpec(
        "n_segments", "int", "Rigid capsule segments per digit, one joint each.", "count", 3, 8, default=5
    ),
    _LAYOUT,
    _SPREAD,
    ParamSpec(
        "palm_radius_mm",
        "float",
        "Radius of the circle the digit roots sit on.",
        "mm",
        10.0,
        70.0,
        default=30.0,
    ),
    _PALM_THICKNESS,
    ParamSpec(
        "digit_length_mm",
        "float",
        "Length of one digit from its root to its tip.",
        "mm",
        40.0,
        160.0,
        default=90.0,
    ),
    ParamSpec(
        "root_radius_mm",
        "float",
        "Radius of the segments at the root of a digit.",
        "mm",
        3.0,
        12.0,
        default=6.0,
    ),
    ParamSpec(
        "taper",
        "float",
        "Tip segment radius as a fraction of the root radius (linear in between).",
        "",
        0.3,
        1.0,
        default=0.6,
    ),
    ParamSpec(
        "curl_bias",
        "float",
        "Which end curls first under the tendon: 0 the root segments, 1 the tip segments "
        "(the joint springs are graded along the digit).",
        "",
        0.0,
        1.0,
        default=0.5,
    ),
    _MATERIAL,
    _GRIP_FORCE,
    _GRIP_HEIGHT,
    _YAW,
)

#: Our own generators, in prompt order.
DESIGNED_KINDS: Final[tuple[str, ...]] = ("fingered_hand", "hook_gripper", "segmented_tentacle")
_DESIGNED: Final[Mapping[str, tuple[ParamSpec, ...]]] = {
    "fingered_hand": FINGERED_HAND,
    "hook_gripper": HOOK_GRIPPER,
    "segmented_tentacle": SEGMENTED_TENTACLE,
}
#: One paragraph per generator for the prompt (what it builds and how it closes).
KIND_SUMMARIES: Final[Mapping[str, str]] = {
    "fingered_hand": (
        "A round palm with 2-5 fingers hanging straight down from knuckles on a circle. Each "
        "finger is a chain of 1-3 rigid phalanges with hinge joints that close it toward the "
        "centre. Fingers are evenly spaced, or one thumb faces the others. The power synergy curls "
        "every joint; the pinch synergy closes the knuckles while the last phalanx stays upright."
    ),
    "hook_gripper": (
        "A round palm with 2-4 rigid fingers, each with a flat pad on its inner face and an inward "
        "hook lip at its tip that can catch under a rim or a flange. Fingers either swing in about "
        "a knuckle hinge or slide straight in toward the centre and stay upright, like a chuck."
    ),
    "segmented_tentacle": (
        "A round palm with 1-4 tapered digits of 3-8 rigid capsule segments. One tendon per digit "
        "curls every segment toward the centre; graded joint springs decide which end curls first, "
        "so a digit wraps around the part. A single digit can wrap a part on its own."
    ),
}
#: Parametric families of other tracks whose spaces join the designer when their modules load.
FOREIGN_FAMILIES: Final[tuple[EffectorFamily, ...]] = (EffectorFamily.SOFT_ARM, EffectorFamily.SUCTION)


def _foreign_spaces() -> dict[SpaceKey, tuple[ParamSpec, ...]]:
    """Spaces of other parametric families (skipped while their module is not built)."""
    from rtt.effectors.registry import UnknownFamilyError, get_family

    out: dict[SpaceKey, tuple[ParamSpec, ...]] = {}
    for fam in FOREIGN_FAMILIES:
        try:
            impl = get_family(fam)
        except UnknownFamilyError:
            continue
        for kind in impl.recipe_kinds():
            space = tuple(impl.param_space(kind))
            if space:
                out[(fam, kind)] = space
    return out


def design_spaces(*, include_foreign: bool = True) -> dict[SpaceKey, tuple[ParamSpec, ...]]:
    """Every designable space keyed by ``(family, kind)``: ours first, then other families'."""
    out: dict[SpaceKey, tuple[ParamSpec, ...]] = {
        (EffectorFamily.DESIGNED, kind): _DESIGNED[kind] for kind in DESIGNED_KINDS
    }
    if include_foreign:
        out.update(_foreign_spaces())
    return out


def space_label(key: SpaceKey) -> str:
    """The prompt name of a space: the kind for ours, ``family/kind`` for other families."""
    fam, kind = key
    return kind if fam is EffectorFamily.DESIGNED else f"{fam.value}/{kind}"


def parse_kind(value: str | SpaceKey) -> SpaceKey:
    """``"fingered_hand"``, ``"soft_arm/tentacle"`` or ``(family, kind)`` as a space key."""
    if isinstance(value, tuple):
        return (EffectorFamily(value[0]), str(value[1]))
    if "/" in value:
        fam, kind = value.split("/", 1)
        return (EffectorFamily(fam), kind)
    return (EffectorFamily.DESIGNED, value)


def space_for(family: EffectorFamily | str, kind: str) -> tuple[ParamSpec, ...]:
    """The space of one generator (``KeyError`` when it is unknown or its family is not built)."""
    fam = EffectorFamily(family)
    if fam is EffectorFamily.DESIGNED:
        return _DESIGNED[kind]
    spaces = _foreign_spaces()
    if (fam, kind) not in spaces:
        raise KeyError(f"no design space {fam.value}/{kind}")
    return spaces[(fam, kind)]


def kind_summary(key: SpaceKey) -> str:
    """What a generator builds, for the prompt (a generic line for other families' kinds)."""
    fam, kind = key
    if fam is EffectorFamily.DESIGNED:
        return KIND_SUMMARIES[kind]
    return f"A {fam.value.replace('_', ' ')} effector ({kind.replace('_', ' ')}), built by that family's generator."


def select_spaces(
    kinds: Sequence[str | SpaceKey] | None, *, include_foreign: bool = True
) -> dict[SpaceKey, tuple[ParamSpec, ...]]:
    """The spaces named by ``kinds`` (all of :func:`design_spaces` when ``None``)."""
    spaces = design_spaces(include_foreign=include_foreign)
    if kinds is None:
        return spaces
    out = {}
    for k in kinds:
        key = parse_kind(k)
        if key not in spaces:
            raise KeyError(
                f"unknown design space {space_label(key)!r}; known: {[space_label(s) for s in spaces]}"
            )
        out[key] = spaces[key]
    if not out:
        raise ValueError("kinds must name at least one design space")
    return out
