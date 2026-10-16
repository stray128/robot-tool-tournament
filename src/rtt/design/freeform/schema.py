"""FingerRecipe v2: a part-relative finger recipe (design space ``freeform-v2``).

A recipe does not describe a shape. It says how to shape a jaw *against a part*: which pad
strategy to use, where on the part to grip, how far the pad wraps around it, how steep its
walls are and how much clearance it leaves. The generator (:mod:`rtt.design.freeform.generate`)
compiles one recipe into a separate left and right jaw for every part, so the same recipe is
scored across the whole training suite and can be recompiled for parts it has never seen.

Scored parameters: 2 categoricals (``pad``, ``tip``) and 7 continuous ones. Some continuous
parameters do nothing for some strategies; they are *inactive* and canonicalisation
(:mod:`rtt.design.freeform.canon`) resets them to a fixed value:

==================  ===========================  ==============================
parameter           inactive when                fixed value
==================  ===========================  ==============================
``pad_depth_mm``    ``pad == "flat"``            0
``wall_draft_deg``  ``pad == "flat"``            0
``clearance_mm``    ``pad in ("flat", "v_funnel")``  0.2
``tip_size_mm``     ``tip == "blunt"``           0
==================  ===========================  ==============================

Only active parameters are required. Strategy rules (checked on the quantised values):
``v_funnel`` needs a draft in [15, 70] degrees and a depth of at least 3 mm, ``conform``
needs a depth of at least 1 mm and ``hook`` needs a tip size of at least 2 mm. Two fit rules
follow from the tip geometry: a hook (height ``min(tip_size, 4)`` plus its protrusion under a
45-degree top) must fit inside the pad, and a chamfer leg must leave 0.5 mm of the pad face.

Product-only options (never in a pre-registered arm, never scored): ``locator`` (a pin on a
declared socket) and ``body_style`` (lightening windows). Scored recipes keep the defaults.

Units: millimetres, degrees and fractions in the recipe; metres in generated geometry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

PadStrategy = Literal["flat", "conform", "v_funnel"]
TipStrategy = Literal["blunt", "chamfer", "hook"]
LocatorOption = Literal["none", "straight_pin", "conical_pin"]
BodyStyle = Literal["solid", "windowed", "truss"]

PAD_STRATEGIES: Final[tuple[str, ...]] = get_args(PadStrategy)
TIP_STRATEGIES: Final[tuple[str, ...]] = get_args(TipStrategy)
LOCATOR_OPTIONS: Final[tuple[str, ...]] = get_args(LocatorOption)
BODY_STYLES: Final[tuple[str, ...]] = get_args(BodyStyle)

#: Inclusive bounds of every continuous scored parameter.
PARAM_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "grip_height_frac": (0.15, 0.85),
    "pad_height_mm": (8.0, 30.0),
    "finger_width_mm": (14.0, 40.0),
    "pad_depth_mm": (0.0, 15.0),
    "wall_draft_deg": (0.0, 70.0),
    "clearance_mm": (0.2, 1.5),
    "tip_size_mm": (0.0, 8.0),
}

#: Allowed values of the two scored categoricals.
CATEGORICAL_OPTIONS: Final[dict[str, tuple[str, ...]]] = {"pad": PAD_STRATEGIES, "tip": TIP_STRATEGIES}

#: The 7 continuous scored parameters, in canonical order.
CONTINUOUS_PARAMS: Final[tuple[str, ...]] = tuple(PARAM_BOUNDS)

#: The 9 scored parameters, in canonical order (the distance vector and the Sobol dimensions).
SCORED_PARAMS: Final[tuple[str, ...]] = ("pad", "tip", *CONTINUOUS_PARAMS)

#: Product-only options and their defaults (the scored value).
PRODUCT_DEFAULTS: Final[dict[str, str]] = {"locator": "none", "body_style": "windowed"}

#: Parameters that may be inactive, with the value canonicalisation resets them to.
FIXED_VALUES: Final[dict[str, float]] = {
    "pad_depth_mm": 0.0,
    "wall_draft_deg": 0.0,
    "clearance_mm": 0.2,
    "tip_size_mm": 0.0,
}

#: Quantisation step of every continuous parameter (0.1 mm, 0.5 degrees, 0.01 for fractions).
QUANTUM: Final[dict[str, float]] = {
    "grip_height_frac": 0.01,
    "pad_height_mm": 0.1,
    "finger_width_mm": 0.1,
    "pad_depth_mm": 0.1,
    "wall_draft_deg": 0.5,
    "clearance_mm": 0.1,
    "tip_size_mm": 0.1,
}
_DECIMALS: Final[dict[str, int]] = {k: (2 if v == 0.01 else 1) for k, v in QUANTUM.items()}

#: Strategy-rule limits.
V_FUNNEL_DRAFT_DEG: Final = (15.0, 70.0)
V_FUNNEL_MIN_DEPTH_MM: Final = 3.0
CONFORM_MIN_DEPTH_MM: Final = 1.0
HOOK_MIN_TIP_MM: Final = 2.0
HOOK_MAX_HEIGHT_MM: Final = 4.0  # hook height = min(tip_size, 4) mm
CHAMFER_LAND_MM: Final = 0.5  # a chamfer leaves at least this much of the pad's front face

DESCRIPTIONS: Final[dict[str, str]] = {
    "pad": (
        "Pad strategy. 'flat': a plane touching the part's most protruding point (a plain parallel "
        "jaw). 'conform': the pad follows the part's outline seen from above, offset by clearance_mm, "
        "and wraps pad_depth_mm around it; wall_draft_deg tilts every wall away from the closing axis "
        "(0 = straight imprint walls). 'v_funnel': a V-groove whose flanks sit wall_draft_deg from "
        "the closing axis, just touching the part, at most pad_depth_mm deep."
    ),
    "tip": (
        "Tip strategy at the bottom of the pad. 'blunt': square end. 'chamfer': a 45-degree bevel of "
        "tip_size_mm on the pad's lower front edge, a lead-in for parts that sit a little too close. "
        "'hook': a lip that sticks out tip_size_mm toward the part at the bottom of the pad (height "
        "min(tip_size_mm, 4) mm, 45-degree top), staying clearance_mm off the part, to catch under "
        "an overhang such as a flange."
    ),
    "grip_height_frac": (
        "Height of the pad centre as a fraction of the part's height on the table (0.15 = low, "
        "0.85 = near the top). The pad bottom is never lower than 3 mm above the table."
    ),
    "pad_height_mm": "Height of the pad along the approach (vertical) axis, in mm.",
    "finger_width_mm": (
        "Width of the finger and pad across the jaw (perpendicular to the closing axis), in mm, "
        "centred on the part. The tool envelope is +-20 mm."
    ),
    "pad_depth_mm": (
        "How far the pad wraps around the part beyond its first-contact plane, in mm (conform: depth "
        "of the pocket walls; v_funnel: depth of the V). Inactive (0) for a flat pad."
    ),
    "wall_draft_deg": (
        "Angle of the pad walls from the closing axis, in degrees. 0 = straight walls parallel to the "
        "closing axis (a tight imprint); larger angles open the walls into a funnel that can push an "
        "off-centre part toward the middle. For v_funnel it is the flank angle. Inactive (0) for a "
        "flat pad."
    ),
    "clearance_mm": (
        "Gap the pad leaves around the part, in mm (print tolerance; 0.2-0.5 is typical). Active for "
        "conform only; fixed at 0.2 otherwise."
    ),
    "tip_size_mm": (
        "Chamfer leg (tip = chamfer) or hook protrusion (tip = hook), in mm. Inactive (0) for a blunt tip."
    ),
    "locator": (
        "Product mode only, never scored: a pin on a socket the part declares facing that jaw. "
        "'straight_pin' or 'conical_pin' (its taper equals wall_draft_deg)."
    ),
    "body_style": (
        "Product mode only, never scored: 'windowed' (default) cuts lightening windows wherever the "
        "body is thick enough, 'truss' cuts a larger truss pattern, 'solid' cuts none."
    ),
}


def quantize(name: str, value: float) -> float:
    """``value`` rounded half-up to the parameter's quantum, as an exact short decimal."""
    q = QUANTUM[name]
    steps = math.floor(float(value) / q + 0.5 + 1e-9)
    return round(steps * q, _DECIMALS[name])


def inactive_params(pad: str, tip: str) -> tuple[str, ...]:
    """The continuous parameters that have no effect for this ``pad`` / ``tip`` combination."""
    out: list[str] = []
    if pad == "flat":
        out += ["pad_depth_mm", "wall_draft_deg", "clearance_mm"]
    elif pad == "v_funnel":
        out.append("clearance_mm")
    if tip == "blunt":
        out.append("tip_size_mm")
    return tuple(out)


def strategy_violation(values: Mapping[str, Any]) -> str | None:
    """The first strategy rule the (quantised, active) values break, in words, or ``None``."""
    pad, tip = values["pad"], values["tip"]
    depth = values.get("pad_depth_mm")
    draft = values.get("wall_draft_deg")
    size = values.get("tip_size_mm")
    if pad == "v_funnel":
        lo, hi = V_FUNNEL_DRAFT_DEG
        if draft is None or not lo <= draft <= hi:
            return f"pad=v_funnel needs wall_draft_deg in [{lo:g}, {hi:g}], got {draft}"
        if depth is None or depth < V_FUNNEL_MIN_DEPTH_MM:
            return f"pad=v_funnel needs pad_depth_mm >= {V_FUNNEL_MIN_DEPTH_MM:g}, got {depth}"
    if pad == "conform" and (depth is None or depth < CONFORM_MIN_DEPTH_MM):
        return f"pad=conform needs pad_depth_mm >= {CONFORM_MIN_DEPTH_MM:g}, got {depth}"
    if tip == "hook" and (size is None or size < HOOK_MIN_TIP_MM):
        return f"tip=hook needs tip_size_mm >= {HOOK_MIN_TIP_MM:g}, got {size}"
    pad_h = values.get("pad_height_mm")
    if tip == "hook" and pad_h is not None and size is not None:
        need = min(size, HOOK_MAX_HEIGHT_MM) + size
        if pad_h < need - 1e-9:
            return (
                f"tip=hook with tip_size_mm {size:g} needs pad_height_mm >= {need:g} (hook height "
                f"{min(size, HOOK_MAX_HEIGHT_MM):g} + protrusion {size:g} under its 45-degree top), got {pad_h:g}"
            )
    if tip == "chamfer" and pad_h is not None and size is not None and size > pad_h - CHAMFER_LAND_MM + 1e-9:
        return f"tip=chamfer needs tip_size_mm <= pad_height_mm - {CHAMFER_LAND_MM:g}, got {size:g} on a {pad_h:g} mm pad"
    return None


def _bounded(name: str) -> Any:
    lo, hi = PARAM_BOUNDS[name]
    return Field(ge=lo, le=hi, description=DESCRIPTIONS[name])


def _optional_bounded(name: str) -> Any:
    lo, hi = PARAM_BOUNDS[name]
    return Field(default=None, ge=lo, le=hi, description=DESCRIPTIONS[name])


class FingerRecipe(BaseModel):
    """One part-relative finger recipe; the same recipe shapes both jaws on every part.

    The generator builds each jaw against its own side of the part (the right jaw against
    the part mirrored across the closing axis), so the two jaws differ wherever the part does.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = Field(
        default=None, max_length=64, description="Optional short label. Does not affect geometry."
    )
    rationale: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional one- or two-sentence reason for the design. Does not affect geometry.",
    )
    pad: PadStrategy = Field(description=DESCRIPTIONS["pad"])
    tip: TipStrategy = Field(description=DESCRIPTIONS["tip"])
    grip_height_frac: float = _bounded("grip_height_frac")
    pad_height_mm: float = _bounded("pad_height_mm")
    finger_width_mm: float = _bounded("finger_width_mm")
    pad_depth_mm: float | None = _optional_bounded("pad_depth_mm")
    wall_draft_deg: float | None = _optional_bounded("wall_draft_deg")
    clearance_mm: float | None = _optional_bounded("clearance_mm")
    tip_size_mm: float | None = _optional_bounded("tip_size_mm")
    locator: LocatorOption = Field(default="none", description=DESCRIPTIONS["locator"])
    body_style: BodyStyle = Field(default="windowed", description=DESCRIPTIONS["body_style"])

    @model_validator(mode="after")
    def _check_strategy(self) -> FingerRecipe:
        inactive = set(inactive_params(self.pad, self.tip))
        missing = [k for k in FIXED_VALUES if k not in inactive and getattr(self, k) is None]
        if missing:
            raise ValueError(
                f"pad={self.pad}, tip={self.tip} needs {', '.join(missing)} (only inactive parameters may be "
                f"left out)"
            )
        values = quantized_values(self)
        problem = strategy_violation(values)
        if problem:
            raise ValueError(problem)
        return self

    @property
    def is_product_mode(self) -> bool:
        """True when a product-only option differs from its scored default."""
        return any(getattr(self, k) != v for k, v in PRODUCT_DEFAULTS.items())


def quantized_values(recipe: FingerRecipe) -> dict[str, Any]:
    """The 9 scored values after resetting inactive parameters and quantising, plus the options."""
    inactive = set(inactive_params(recipe.pad, recipe.tip))
    out: dict[str, Any] = {"pad": recipe.pad, "tip": recipe.tip}
    for name in CONTINUOUS_PARAMS:
        raw = getattr(recipe, name)
        out[name] = FIXED_VALUES[name] if name in inactive else quantize(name, raw)
    for name in PRODUCT_DEFAULTS:
        out[name] = getattr(recipe, name)
    return out


def recipe_json_schema() -> dict[str, Any]:
    """JSON schema of :class:`FingerRecipe` (descriptions on every field)."""
    return FingerRecipe.model_json_schema()


def recipe_from_unit(u: Sequence[float] | np.ndarray, **extra: Any) -> FingerRecipe:
    """Map a point of the unit cube ``[0, 1)^9`` onto a recipe (Sobol and random sampling).

    Dimension order is :data:`SCORED_PARAMS`: the two categoricals take
    ``floor(3 u)`` and the continuous parameters scale linearly onto their bounds. Raises
    ``ValueError`` (pydantic's ``ValidationError``) when the point breaks a strategy rule,
    so callers can rejection-sample.
    """
    v = np.asarray(u, dtype=np.float64)
    if v.shape != (len(SCORED_PARAMS),) or np.any(v < 0) or np.any(v > 1) or not np.all(np.isfinite(v)):
        raise ValueError(f"expected {len(SCORED_PARAMS)} values in [0, 1], got {u!r}")
    pad = PAD_STRATEGIES[min(int(v[0] * 3), 2)]
    tip = TIP_STRATEGIES[min(int(v[1] * 3), 2)]
    values: dict[str, Any] = {"pad": pad, "tip": tip}
    inactive = set(inactive_params(pad, tip))
    for i, name in enumerate(CONTINUOUS_PARAMS, start=2):
        if name in inactive:
            continue
        lo, hi = PARAM_BOUNDS[name]
        values[name] = min(hi, lo + float(v[i]) * (hi - lo))
    values.update(extra)
    return FingerRecipe(**values)
