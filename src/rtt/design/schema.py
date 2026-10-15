"""Finger design schema and gripper task configuration.

``FingerDesign`` is the 10-parameter description of one gripper finger that every
search arm (random, LLM, TPE, manual entrants) proposes. It is a strict pydantic
model: out-of-range values, unknown categories and unknown fields are rejected.
``FingerDesign.model_json_schema()`` (or :func:`design_json_schema`) gives a JSON
schema with a physical description on every field, suitable for an LLM tool
definition.

``GripperTask`` holds the task-level constants that validation needs (stroke,
mounting gap, tool envelope, mass budget, widest training object). Defaults are
starting values; a YAML file can override them with :func:`load_task`.

Units: the design is in millimetres and degrees; generated geometry is in metres.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field

InnerFace = Literal["flat", "concave_arc", "v_groove"]
TipProfile = Literal["blunt", "rounded", "chamfer", "hook"]

#: Inclusive bounds of every continuous design parameter.
PARAM_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "finger_length_mm": (40.0, 120.0),
    "finger_thickness_mm": (4.0, 16.0),
    "finger_width_mm": (10.0, 30.0),
    "bend_angle_deg": (-15.0, 45.0),
    "bend_start_frac": (0.2, 0.9),
    "inner_face_depth_mm": (0.0, 8.0),
    "tip_size_mm": (0.0, 15.0),
    "taper_ratio": (0.5, 1.0),
}

#: Allowed values of every categorical design parameter.
CATEGORICAL_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "inner_face": get_args(InnerFace),
    "tip_profile": get_args(TipProfile),
}

#: The 10 parameters that define geometry, in canonical order.
GEOMETRY_PARAMS: Final[tuple[str, ...]] = (
    "finger_length_mm",
    "finger_thickness_mm",
    "finger_width_mm",
    "bend_angle_deg",
    "bend_start_frac",
    "inner_face",
    "inner_face_depth_mm",
    "tip_profile",
    "tip_size_mm",
    "taper_ratio",
)


def _bounded(name: str, description: str) -> Any:
    lo, hi = PARAM_BOUNDS[name]
    return Field(ge=lo, le=hi, description=description)


class FingerDesign(BaseModel):
    """One parallel-gripper finger, described by 10 physical parameters.

    The finger hangs below the palm and closes toward the opposite finger (which
    is its mirror image). Its *inner face* is the contact face that touches the
    object. Both fingers of a gripper use the same design. The finger is
    3D-printed in PLA and built from 14 boxes: 4 segments along its length, each
    with a left flank, a centre and a right flank across its width, plus 2 boxes
    that shape the tip.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = Field(
        default=None,
        max_length=64,
        description="Optional short label for this design (e.g. 'wide-hook-v2'). Does not affect geometry.",
    )
    rationale: str | None = Field(
        default=None,
        max_length=1000,
        description=(
            "Optional one- or two-sentence explanation of why this design should grasp well. "
            "Does not affect geometry."
        ),
    )
    finger_length_mm: float = _bounded(
        "finger_length_mm",
        "Length of the finger from its root at the palm to the tip end, in mm; for a bent finger it "
        "is measured along the outside of the bend (the back face for an inward bend, the contact "
        "face for an outward bend). Longer fingers reach lower on tall objects but are heavier and "
        "put a longer lever arm on the grasp.",
    )
    finger_thickness_mm: float = _bounded(
        "finger_thickness_mm",
        "Thickness of the finger at the root in the closing direction (from the inner contact face "
        "to the back face), in mm. Thicker fingers are stiffer and heavier; thickness also limits "
        "how deep the inner-face groove can be.",
    )
    finger_width_mm: float = _bounded(
        "finger_width_mm",
        "Width of the finger at the root, perpendicular to both the closing direction and the "
        "finger length, in mm. Wider fingers give a larger contact patch and resist object "
        "rotation, at the cost of mass.",
    )
    bend_angle_deg: float = _bounded(
        "bend_angle_deg",
        "Angle of a single bend in the finger, in degrees. Positive values bend the part after the "
        "bend inward, toward the other finger (the tips close in like pincers); negative values "
        "splay the tips outward; 0 is a straight finger. Inward bends narrow the opening at the "
        "tips.",
    )
    bend_start_frac: float = _bounded(
        "bend_start_frac",
        "Where the bend sits, as a fraction of finger_length_mm measured from the root (0.2 = near "
        "the palm, 0.9 = near the tip). Everything beyond this point is rotated by bend_angle_deg.",
    )
    inner_face: InnerFace = Field(
        description=(
            "Shape of the inner contact face across the finger width. 'flat': a plain flat pad. "
            "'concave_arc': a channel recessed into the face between two raised lips (approximates "
            "a concave arc; cradles round objects on two lines). 'v_groove': a V-shaped groove "
            "whose two sloped flanks centre cylinders and rods."
        )
    )
    inner_face_depth_mm: float = _bounded(
        "inner_face_depth_mm",
        "Depth of the concave_arc channel or v_groove below the face, in mm. Must be 0 for a flat "
        "face and at most half of finger_thickness_mm otherwise; a v_groove also needs depth <= "
        "0.35 x finger_width_mm. Scales down with the taper toward the tip.",
    )
    tip_profile: TipProfile = Field(
        description=(
            "Shape of the finger tip. 'blunt': a square end (tip_size_mm adds a solid end cap that "
            "closes any groove at the tip). 'rounded': both tip corners cut back into a blunt roof "
            "(tip_size_mm = how far the rounding reaches back). 'chamfer': a 45-degree lead-in bevel "
            "on the inner corner of the tip that guides objects into the grasp (tip_size_mm = bevel "
            "size). 'hook': a lip at the tip that sticks out inward, toward the other finger, by "
            "tip_size_mm, to catch under or behind objects."
        )
    )
    tip_size_mm: float = _bounded(
        "tip_size_mm",
        "Size of the tip feature in mm (see tip_profile). For a hook it is how far the lip "
        "protrudes inward and must be <= finger_thickness_mm + 6. Rounded and chamfer sizes are "
        "capped at half the tip thickness and half the length after the bend.",
    )
    taper_ratio: float = _bounded(
        "taper_ratio",
        "Ratio of tip to root cross-section (thickness and width both scale linearly along the "
        "length, in 4 steps). 1.0 = no taper; 0.5 = the tip is half as thick and half as wide as "
        "the root. Tapering saves mass and lets the tip slip into narrow gaps, but thin tips can "
        "violate the 3 mm minimum section.",
    )

    def geometry_params(self) -> dict[str, float | str]:
        """The 10 geometry-defining parameters (``name`` and ``rationale`` excluded)."""
        return {key: getattr(self, key) for key in GEOMETRY_PARAMS}

    def design_id(self) -> str:
        """Stable 12-hex-digit id of the geometry (ignores name and rationale).

        Floats are rounded to 6 decimals; ``+ 0.0`` folds ``-0.0`` (e.g. a bend of
        ``-1e-9`` degrees) into ``0.0`` so geometrically identical designs share an id.
        """
        canonical = {
            key: round(value, 6) + 0.0 if isinstance(value, float) else value
            for key, value in self.geometry_params().items()
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def design_json_schema(*, strict: bool = False, include_meta: bool = True) -> dict[str, Any]:
    """JSON schema of :class:`FingerDesign` for an LLM tool/function definition.

    Titles are removed (they only repeat field names). With ``strict=True`` every
    property is listed in ``required`` and optional strings become
    ``["string", "null"]``, which is what strict structured-output modes expect.
    With ``include_meta=False`` the ``name`` and ``rationale`` fields are dropped.
    """
    schema = FingerDesign.model_json_schema()
    schema.pop("title", None)
    properties: dict[str, Any] = {}
    for key, prop in schema["properties"].items():
        if not include_meta and key not in GEOMETRY_PARAMS:
            continue
        prop = dict(prop)
        prop.pop("title", None)
        if "anyOf" in prop:  # optional string -> flatten to a (nullable) string
            string_branch = next(b for b in prop.pop("anyOf") if b.get("type") == "string")
            prop.pop("default", None)
            prop["type"] = ["string", "null"] if strict else "string"
            if "maxLength" in string_branch:
                prop["maxLength"] = string_branch["maxLength"]
        properties[key] = prop
    schema["properties"] = properties
    schema["additionalProperties"] = False
    schema["required"] = list(properties) if strict else list(GEOMETRY_PARAMS)
    return schema


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolEnvelope:
    """Box (in the palm frame, mm) that the finger pair must stay inside at every stroke.

    The palm frame is Z-up with its origin at the centre between the finger roots on
    the mounting plane. Fingers hang toward -Z.
    """

    half_width_x_mm: float = 20.0
    """Maximum |x| of any finger point (x = finger width direction)."""
    max_reach_z_mm: float = 130.0
    """How far below the mounting plane (z = 0) a finger may reach."""
    max_rise_z_mm: float = 0.0
    """How far above the mounting plane a finger may rise (0: nothing above the mount)."""
    half_span_y_mm: float = 75.0
    """Maximum |y| of any finger point at full open (y = closing direction)."""

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "max_rise_z_mm":
                if value < 0:
                    raise ValueError(f"{f.name} must be >= 0, got {value}")
            elif value <= 0:
                raise ValueError(f"{f.name} must be > 0, got {value}")


@dataclass(frozen=True)
class GripperTask:
    """Task-level constants shared by every design (never tuned per design).

    Mounting convention (matches the simulator): a finger's root inner face lies
    on its local plane ``y = 0``; at full open the left finger's root sits at palm
    ``y = closed_gap_mm / 2 + stroke_mm`` and at full close at
    ``y = closed_gap_mm / 2``; the right finger is the mirror image. The defaults
    (0 mm closed gap, 42 mm stroke) give the 84 mm opening of the simulator's gripper
    (``configs/task_v1.yaml``); :func:`rtt.sim.task.gripper_task_for` derives both
    from the task file and ``tests/test_integration.py`` keeps them equal.
    """

    stroke_mm: float = 42.0
    """Slide travel of each finger (42 mm per finger gives the default 84 mm opening)."""
    closed_gap_mm: float = 0.0
    """Distance between the two root inner faces at full close."""
    finger_contact: bool = True
    """Whether the simulator models finger-finger contact. If so, closing fingers stop
    where they meet, so inward features (bends, hooks) only need to clear each other
    at full open; if not, they must not cross the centre plane even at full close."""
    widest_training_object_mm: float = 45.0
    """Widest training object, measured across the grasp."""
    open_gap_margin_mm: float = 5.0
    """Required clearance beyond the widest object at full open."""
    mass_budget_g: float = 50.0
    """Maximum mass of one finger."""
    min_section_mm: float = 3.0
    """Thinnest allowed wall/section anywhere in the finger."""
    hook_extra_mm: float = 6.0
    """A hook may protrude at most finger_thickness_mm + this much."""
    max_depth_to_thickness: float = 0.5
    """Groove depth may be at most this fraction of finger_thickness_mm."""
    max_v_groove_depth_to_width: float = 0.35
    """V-groove depth may be at most this fraction of finger_width_mm (flank angle ~35 deg)."""
    envelope: ToolEnvelope = field(default_factory=ToolEnvelope)

    def __post_init__(self) -> None:
        if self.stroke_mm <= 0.0:
            raise ValueError(f"stroke_mm must be > 0, got {self.stroke_mm}")
        if self.closed_gap_mm < 0.0:
            raise ValueError(f"closed_gap_mm must be >= 0, got {self.closed_gap_mm}")
        for name in ("widest_training_object_mm", "mass_budget_g", "min_section_mm"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("open_gap_margin_mm", "hook_extra_mm"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.0 < self.max_depth_to_thickness <= 1.0:
            raise ValueError("max_depth_to_thickness must be in (0, 1]")
        if not 0.0 < self.max_v_groove_depth_to_width <= 0.5:
            raise ValueError("max_v_groove_depth_to_width must be in (0, 0.5]")

    @property
    def max_opening_mm(self) -> float:
        """Distance between the root inner faces at full open."""
        return self.closed_gap_mm + 2.0 * self.stroke_mm

    @property
    def required_open_gap_mm(self) -> float:
        """Smallest acceptable finger-to-finger gap at full open."""
        return self.widest_training_object_mm + self.open_gap_margin_mm

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GripperTask:
        """Build a task from a plain mapping (e.g. parsed YAML); unknown keys are errors."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown GripperTask keys: {sorted(unknown)}; allowed: {sorted(known)}")
        values = dict(data)
        envelope = values.pop("envelope", None)
        if envelope is not None and not isinstance(envelope, ToolEnvelope):
            env_known = {f.name for f in fields(ToolEnvelope)}
            env_unknown = set(envelope) - env_known
            if env_unknown:
                raise ValueError(
                    f"unknown ToolEnvelope keys: {sorted(env_unknown)}; allowed: {sorted(env_known)}"
                )
            envelope = ToolEnvelope(**{k: float(v) for k, v in envelope.items()})
        values = {k: v if isinstance(v, bool) else float(v) for k, v in values.items()}
        if envelope is not None:
            return cls(envelope=envelope, **values)
        return cls(**values)


def load_task(path: str | Path | None = None, *, section: str | None = None) -> GripperTask:
    """Load a :class:`GripperTask` from YAML, or return the defaults when ``path`` is None.

    ``section`` selects a top-level key of the YAML file when the task lives inside a
    larger config file; a missing section is an error rather than silent defaults.
    """
    if path is None:
        return GripperTask()
    data = yaml.safe_load(Path(path).read_text()) or {}
    if section is not None:
        if not isinstance(data, Mapping) or section not in data:
            raise ValueError(f"{path}: no top-level '{section}' section")
        data = data[section] or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"{path}: expected a mapping of GripperTask fields")
    return GripperTask.from_mapping(data)


def load_entrants(path: str | Path) -> list[FingerDesign]:
    """Load manual entrant designs from a YAML file with a top-level ``entrants`` list."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    items = data.get("entrants") if isinstance(data, Mapping) else None
    if not isinstance(items, list):
        raise TypeError(f"{path}: expected a top-level 'entrants' list")
    return [FingerDesign.model_validate(item) for item in items]
