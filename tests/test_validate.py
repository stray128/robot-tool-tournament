"""Validators: entrants pass; specific, actionable errors for each rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from rtt.design.generate import generate_finger, geometry_bounds
from rtt.design.schema import FingerDesign, GripperTask, ToolEnvelope, load_entrants
from rtt.design.validate import ValidationResult, is_valid, validate

ENTRANTS = Path(__file__).resolve().parents[1] / "configs" / "entrants.yaml"
TASK = GripperTask()

BASE: dict[str, object] = {
    "finger_length_mm": 80.0,
    "finger_thickness_mm": 10.0,
    "finger_width_mm": 20.0,
    "bend_angle_deg": 0.0,
    "bend_start_frac": 0.5,
    "inner_face": "flat",
    "inner_face_depth_mm": 0.0,
    "tip_profile": "blunt",
    "tip_size_mm": 0.0,
    "taper_ratio": 1.0,
}


def make(**overrides: object) -> FingerDesign:
    return FingerDesign(**{**BASE, **overrides})


@pytest.mark.parametrize("entrant", load_entrants(ENTRANTS), ids=lambda e: e.name)
def test_manual_entrants_are_valid(entrant: FingerDesign) -> None:
    result = validate(entrant, TASK)
    assert result.ok, result.summary()
    assert result.errors == []
    assert result.design == entrant
    assert result.metrics["mass_g"] < TASK.mass_budget_g
    assert result.metrics["open_gap_mm"] >= TASK.required_open_gap_mm


STRICT = GripperTask(finger_contact=False, closed_gap_mm=20.0)

# (case id, design overrides, task, phrase that must appear in one error)
INVALID_CASES: list[tuple[str, dict[str, object], GripperTask, str]] = [
    (
        "flat_with_depth",
        {"inner_face_depth_mm": 2.0},
        TASK,
        "inner_face='flat' requires inner_face_depth_mm=0",
    ),
    (
        "groove_deeper_than_half_thickness",
        {"inner_face": "concave_arc", "finger_thickness_mm": 6.0, "inner_face_depth_mm": 4.0},
        TASK,
        "deeper than 0.5 x finger_thickness_mm",
    ),
    (
        "v_groove_too_steep",
        {
            "inner_face": "v_groove",
            "finger_width_mm": 10.0,
            "finger_thickness_mm": 16.0,
            "inner_face_depth_mm": 6.0,
        },
        TASK,
        "v_groove is too steep",
    ),
    (
        "thin_tapered_tip",
        {"finger_thickness_mm": 4.0, "taper_ratio": 0.5},
        TASK,
        "thinnest section is 2.25 mm",
    ),
    (
        "hook_too_long",
        {"tip_profile": "hook", "tip_size_mm": 12.0, "finger_thickness_mm": 5.0},
        TASK,
        "hook protrusion tip_size_mm=12 exceeds",
    ),
    (
        "fingers_overlap_when_open",
        {"finger_length_mm": 120.0, "bend_start_frac": 0.2, "bend_angle_deg": 45.0},
        TASK,
        "fingers would interpenetrate even at full open",
    ),
    (
        "fingers_cross_at_full_close_without_contact_model",
        {"bend_angle_deg": 30.0, "bend_start_frac": 0.4},
        STRICT,
        "fingers would interpenetrate at full close",
    ),
    (
        "opening_too_narrow",
        {"bend_angle_deg": 30.0, "bend_start_frac": 0.4},
        TASK,
        "opening at full open is only 38.7 mm",  # 84 mm opening - 2 x 22.7 mm inward reach
    ),
    (
        "over_mass_budget",
        {"finger_length_mm": 120.0, "finger_thickness_mm": 16.0, "finger_width_mm": 30.0},
        TASK,
        "exceeds the 50 g budget",
    ),
    (
        "rises_above_mount",
        {
            "finger_length_mm": 40.0,
            "bend_start_frac": 0.2,
            "bend_angle_deg": 45.0,
            "finger_thickness_mm": 16.0,
        },
        TASK,
        "above the mounting plane",
    ),
    (
        "splayed_beyond_envelope",
        {
            "finger_length_mm": 120.0,
            "bend_start_frac": 0.2,
            "bend_angle_deg": -15.0,
            "finger_thickness_mm": 16.0,
        },
        TASK,
        "finger's back reaches",
    ),
    (
        "too_wide_for_envelope",
        {"finger_width_mm": 30.0},
        GripperTask(envelope=ToolEnvelope(half_width_x_mm=12.0)),
        "mm wide, beyond",
    ),
    (
        "too_long_for_envelope",
        {"finger_length_mm": 120.0},
        GripperTask(envelope=ToolEnvelope(max_reach_z_mm=100.0)),
        "below the mounting plane, beyond",
    ),
]


@pytest.mark.parametrize(
    ("case", "overrides", "task", "phrase"), INVALID_CASES, ids=[c[0] for c in INVALID_CASES]
)
def test_invalid_designs_are_rejected_with_specific_messages(
    case: str, overrides: dict[str, object], task: GripperTask, phrase: str
) -> None:
    result = validate(make(**overrides), task)
    assert not result.ok
    assert any(phrase in error for error in result.errors), result.summary()


def test_rejection_messages_are_distinct() -> None:
    matched = set()
    for _, overrides, task, phrase in INVALID_CASES:
        errors = validate(make(**overrides), task).errors
        matched.add(next(e for e in errors if phrase in e))
    assert len(matched) == len(INVALID_CASES) >= 8


@pytest.mark.parametrize(
    ("raw", "phrase"),
    [
        ({**BASE, "finger_length_mm": 150.0}, "finger_length_mm=150.0 is out of range [40, 120]"),
        (
            {**BASE, "inner_face": "spiral"},
            "inner_face='spiral' is not one of 'flat', 'concave_arc', 'v_groove'",
        ),
        ({**BASE, "colour": "red"}, "unknown field 'colour'"),
        ({k: v for k, v in BASE.items() if k != "taper_ratio"}, "missing required field 'taper_ratio'"),
        ({**BASE, "finger_width_mm": "wide"}, "finger_width_mm"),
    ],
)
def test_raw_llm_output_errors_are_readable(raw: dict[str, object], phrase: str) -> None:
    result = validate(raw, TASK)
    assert not result.ok
    assert result.design is None and result.metrics == {}
    assert any(phrase in error for error in result.errors), result.errors


def test_long_inputs_are_truncated_in_messages() -> None:
    result = validate({**BASE, "rationale": "x" * 5000}, TASK)
    assert not result.ok
    assert result.errors[0].startswith("rationale: String should have at most 1000 characters")
    assert len(result.errors[0]) < 200


def test_raw_mapping_that_parses_is_validated_fully() -> None:
    result = validate({**BASE, "name": "baseline"}, TASK)
    assert result.ok
    assert result.design is not None and result.design.name == "baseline"


def test_task_constants_change_the_verdict() -> None:
    hooked = next(e for e in load_entrants(ENTRANTS) if e.name == "hooked")
    result = validate(hooked, TASK)
    assert result.ok
    assert any("fingers meet 13.6 mm before the end of their stroke" in w for w in result.warnings)
    assert not is_valid(hooked, GripperTask(finger_contact=False))
    assert is_valid(hooked, GripperTask(finger_contact=False, closed_gap_mm=30.0))
    # the hook leaves a 56.8 mm gap at the 84 mm opening: enough for 50 mm + 5 mm, not for 54 mm + 5 mm
    assert is_valid(hooked, GripperTask(widest_training_object_mm=50.0))
    assert not is_valid(hooked, GripperTask(widest_training_object_mm=54.0))


def test_metrics_report_the_checked_quantities() -> None:
    result = validate(make(tip_profile="hook", tip_size_mm=6.0), TASK)
    assert result.ok
    m = result.metrics
    assert m["max_inward_protrusion_mm"] == pytest.approx(6.0)
    assert m["open_gap_mm"] == pytest.approx(TASK.max_opening_mm - 12.0)
    assert m["close_clearance_mm"] == pytest.approx(TASK.closed_gap_mm / 2 - 6.0)
    assert m["outer_reach_y_mm"] == pytest.approx(TASK.closed_gap_mm / 2 + TASK.stroke_mm + 10.0)
    assert m["max_reach_z_mm"] == pytest.approx(80.0)
    assert m["mass_g"] == pytest.approx(1240.0 * (80 * 10 * 20 + 5 * 6 * 20) * 1e-9 * 1000, rel=1e-4)


def test_open_gap_counts_recessed_faces_like_the_simulator() -> None:
    """Bevelled v_groove lips sit behind the root plane, so the clear gap exceeds the opening."""
    design = make(inner_face="v_groove", inner_face_depth_mm=4.0)
    recess_mm = geometry_bounds(generate_finger(design))[0][1] * 1000.0
    assert recess_mm > 0.0
    m = validate(design, TASK).metrics
    assert m["max_inward_protrusion_mm"] == 0.0
    assert m["open_gap_mm"] == pytest.approx(TASK.max_opening_mm + 2.0 * recess_mm)
    assert m["close_clearance_mm"] == pytest.approx(TASK.closed_gap_mm / 2 + recess_mm)


def test_warnings_flag_capped_or_inert_parameters() -> None:
    capped = validate(make(tip_profile="chamfer", tip_size_mm=15.0), TASK)
    assert capped.ok and any("capped" in w for w in capped.warnings)
    flat_groove = validate(make(inner_face="concave_arc", inner_face_depth_mm=0.0), TASK)
    assert flat_groove.ok and any("is a flat face" in w for w in flat_groove.warnings)


def test_summary_text() -> None:
    assert validate(make(), TASK).summary().startswith("valid")
    bad = validate(make(inner_face_depth_mm=2.0), TASK)
    assert isinstance(bad, ValidationResult)
    assert bad.summary().startswith("invalid:") and "flat" in bad.summary()
