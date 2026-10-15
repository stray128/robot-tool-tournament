"""Design validators shared by every search arm.

:func:`validate` checks a design against the plan's rules and returns every
violation as a specific, actionable sentence (an LLM reads them to repair its
proposals). Rules:

* groove depth is 0 for a flat face, otherwise <= 0.5 x thickness;
* a v_groove is no steeper than ~35 degrees (depth <= 0.35 x width);
* thinnest section >= 3 mm (core thickness/width per segment, lips, hook lip);
* hook protrusion <= thickness + 6 mm;
* the swept finger envelope (over the whole stroke) stays inside the tool box;
* finger mass <= budget;
* gap between the fingers at full open >= widest training object + 5 mm;
* no finger-finger interpenetration at full close. When the simulator models
  finger-finger contact (``GripperTask.finger_contact``, the default) closing
  fingers stop where they meet, so this reduces to the fingers being clear of
  each other at full open (a warning reports how early they meet); otherwise no
  point may cross the centre plane at full close.

All task constants come from :class:`~rtt.design.schema.GripperTask`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from rtt.design.generate import build_finger, geometry_bounds, mass_and_inertia
from rtt.design.schema import CATEGORICAL_OPTIONS, GEOMETRY_PARAMS, PARAM_BOUNDS, FingerDesign, GripperTask

_M_TO_MM = 1000.0
_TOL_MM = 1e-6


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :func:`validate`.

    ``errors`` make a design invalid; ``warnings`` describe parameters that were capped
    or have no effect. ``metrics`` holds the derived quantities the checks used
    (empty when the input did not parse as a design).
    """

    ok: bool
    errors: list[str]
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    design: FingerDesign | None = None

    def summary(self) -> str:
        """One line per problem, for logs and LLM repair prompts."""
        if self.ok:
            return "valid" + "".join(f"\n  warning: {w}" for w in self.warnings)
        return "invalid:" + "".join(f"\n  - {e}" for e in self.errors)


def _short(value: object, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _schema_errors(exc: ValidationError) -> list[str]:
    """Turn pydantic errors into one plain sentence per problem."""
    messages = []
    for err in exc.errors():
        name = ".".join(str(part) for part in err["loc"]) or "design"
        kind = err["type"]
        value = _short(err.get("input"))
        if kind == "missing":
            messages.append(f"missing required field '{name}'")
        elif kind == "extra_forbidden":
            allowed = ", ".join(("name", "rationale", *GEOMETRY_PARAMS))
            messages.append(f"unknown field '{name}'; allowed fields are: {allowed}")
        elif kind == "literal_error" and name in CATEGORICAL_OPTIONS:
            options = ", ".join(repr(o) for o in CATEGORICAL_OPTIONS[name])
            messages.append(f"{name}={value} is not one of {options}")
        elif name in PARAM_BOUNDS and kind in ("greater_than_equal", "less_than_equal"):
            lo, hi = PARAM_BOUNDS[name]
            messages.append(f"{name}={value} is out of range [{lo:g}, {hi:g}]")
        else:
            messages.append(f"{name}: {err['msg']} (got {value})")
    return messages


def validate(design: FingerDesign | Mapping[str, Any], task: GripperTask | None = None) -> ValidationResult:
    """Check a design (a :class:`FingerDesign` or a raw mapping) against every rule."""
    task = task or GripperTask()
    if not isinstance(design, FingerDesign):
        try:
            design = FingerDesign.model_validate(design)
        except ValidationError as exc:
            return ValidationResult(ok=False, errors=_schema_errors(exc))

    errors: list[str] = []
    d = design
    thick, width, depth = d.finger_thickness_mm, d.finger_width_mm, d.inner_face_depth_mm

    # --- parameter rules ------------------------------------------------------
    if d.inner_face == "flat" and depth > 0.0:
        errors.append(
            f"inner_face='flat' requires inner_face_depth_mm=0 (got {depth:g}); set the depth to 0 "
            "or choose inner_face='concave_arc' or 'v_groove' for a recessed face"
        )
    if d.inner_face != "flat":
        max_depth = task.max_depth_to_thickness * thick
        if depth > max_depth + _TOL_MM:
            errors.append(
                f"inner_face_depth_mm={depth:g} is deeper than {task.max_depth_to_thickness:g} x "
                f"finger_thickness_mm (= {max_depth:.2f} mm); reduce the depth or increase thickness "
                f"to >= {depth / task.max_depth_to_thickness:.1f} mm"
            )
    if d.inner_face == "v_groove":
        max_v = task.max_v_groove_depth_to_width * width
        if depth > max_v + _TOL_MM:
            errors.append(
                f"v_groove is too steep: inner_face_depth_mm={depth:g} must be <= "
                f"{task.max_v_groove_depth_to_width:g} x finger_width_mm (= {max_v:.2f} mm); reduce the "
                "depth, widen the finger, or use inner_face='concave_arc'"
            )
    if d.tip_profile == "hook":
        max_hook = thick + task.hook_extra_mm
        if d.tip_size_mm > max_hook + _TOL_MM:
            errors.append(
                f"hook protrusion tip_size_mm={d.tip_size_mm:g} exceeds finger_thickness_mm + "
                f"{task.hook_extra_mm:g} (= {max_hook:.1f} mm); shorten the hook or thicken the finger"
            )

    # --- geometry rules -------------------------------------------------------
    build = build_finger(d)
    geom = build.geometry
    lo, hi = (v * _M_TO_MM for v in geometry_bounds(geom))
    mass_g = mass_and_inertia(geom).mass_kg * 1000.0
    weakest = build.min_section
    # Innermost point relative to the root inner-face plane (y = 0): negative when a bend or
    # hook reaches inward, positive when every face is recessed (bevelled v_groove lips).
    inner_offset = float(lo[1])
    protrusion = max(0.0, -inner_offset)  # inward reach past the root inner-face plane
    half_closed = task.closed_gap_mm / 2.0
    open_gap = task.max_opening_mm + 2.0 * inner_offset  # same clear gap the simulator computes
    outer_reach = half_closed + task.stroke_mm + float(hi[1])
    env = task.envelope

    if weakest.size_mm < task.min_section_mm - _TOL_MM:
        thin = [s for s in build.sections if s.size_mm < task.min_section_mm - _TOL_MM]
        more = f" ({len(thin) - 1} other section(s) also too thin)" if len(thin) > 1 else ""
        errors.append(
            f"thinnest section is {weakest.size_mm:.2f} mm (< {task.min_section_mm:g} mm) at "
            f"{weakest.where}{more}; {weakest.remedy}"
        )
    if mass_g > task.mass_budget_g + _TOL_MM:
        errors.append(
            f"finger mass {mass_g:.1f} g exceeds the {task.mass_budget_g:g} g budget; reduce "
            "finger_length_mm, finger_thickness_mm or finger_width_mm, or lower taper_ratio"
        )
    fix_inward = (
        "reduce bend_angle_deg, move bend_start_frac toward the tip, or shorten the hook (tip_size_mm)"
    )
    if task.finger_contact:
        # Closing fingers stop where they meet, so they only interpenetrate if they
        # already overlap at full open.
        half_open = half_closed + task.stroke_mm
        if protrusion >= half_open - _TOL_MM:
            errors.append(
                f"fingers would interpenetrate even at full open: each reaches {protrusion:.1f} mm inward "
                f"of its root face but the roots are only {half_open:.1f} mm from the centre; {fix_inward}"
            )
    elif protrusion > half_closed + _TOL_MM:
        errors.append(
            f"fingers would interpenetrate at full close: each reaches {protrusion:.1f} mm inward of its "
            f"root face but only {half_closed:.1f} mm (half the {task.closed_gap_mm:g} mm closed gap) is "
            f"free; {fix_inward}"
        )
    if open_gap < task.required_open_gap_mm - _TOL_MM:
        errors.append(
            f"opening at full open is only {open_gap:.1f} mm, less than the widest training object "
            f"({task.widest_training_object_mm:g} mm) + {task.open_gap_margin_mm:g} mm margin; the fingers "
            f"reach {protrusion:.1f} mm inward, so reduce the inward bend or hook"
        )
    if float(hi[2]) > env.max_rise_z_mm + _TOL_MM:
        errors.append(
            f"finger rises {hi[2]:.1f} mm above the mounting plane (allowed {env.max_rise_z_mm:g} mm): "
            "the inward bend is too close to the root for this thickness; increase bend_start_frac, "
            "or reduce bend_angle_deg or finger_thickness_mm"
        )
    if -float(lo[2]) > env.max_reach_z_mm + _TOL_MM:
        errors.append(
            f"finger reaches {-lo[2]:.1f} mm below the mounting plane, beyond the {env.max_reach_z_mm:g} mm "
            "tool envelope; shorten the finger"
        )
    half_x = max(-float(lo[0]), float(hi[0]))
    if half_x > env.half_width_x_mm + _TOL_MM:
        errors.append(
            f"finger is {2 * half_x:.1f} mm wide, beyond the {2 * env.half_width_x_mm:g} mm tool envelope; "
            "reduce finger_width_mm"
        )
    if outer_reach > env.half_span_y_mm + _TOL_MM:
        errors.append(
            f"at full open the finger's back reaches {outer_reach:.1f} mm from the gripper centre, beyond "
            f"the {env.half_span_y_mm:g} mm tool envelope; reduce the outward (negative) bend, "
            "finger_thickness_mm or finger_length_mm"
        )

    warnings = list(build.notes)
    if task.finger_contact and protrusion > half_closed + _TOL_MM:
        warnings.append(
            f"the fingers meet {protrusion - half_closed:.1f} mm before the end of their stroke, so objects "
            f"narrower than {2 * protrusion - task.closed_gap_mm:.1f} mm cannot be squeezed"
        )
    if d.inner_face != "flat" and depth == 0.0:
        warnings.append(f"inner_face='{d.inner_face}' with inner_face_depth_mm=0 is a flat face")

    metrics = {
        "mass_g": mass_g,
        "min_section_mm": weakest.size_mm,
        "max_inward_protrusion_mm": protrusion,
        "open_gap_mm": open_gap,
        "close_clearance_mm": half_closed + inner_offset,
        "outer_reach_y_mm": outer_reach,
        "max_rise_z_mm": float(hi[2]),
        "max_reach_z_mm": -float(lo[2]),
        "half_width_x_mm": half_x,
        "tip_feature_mm": build.tip_feature_mm,
    }
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, metrics=metrics, design=d)


def is_valid(design: FingerDesign | Mapping[str, Any], task: GripperTask | None = None) -> bool:
    """Shorthand for ``validate(design, task).ok``."""
    return validate(design, task).ok
