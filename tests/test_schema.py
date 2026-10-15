"""Design schema, task config, entrant loading and design distance."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from rtt.design.distance import design_distance, min_distance, pairwise_distances, parameter_distances
from rtt.design.schema import (
    CATEGORICAL_OPTIONS,
    GEOMETRY_PARAMS,
    PARAM_BOUNDS,
    FingerDesign,
    GripperTask,
    ToolEnvelope,
    design_json_schema,
    load_entrants,
    load_task,
)

ENTRANTS = Path(__file__).resolve().parents[1] / "configs" / "entrants.yaml"

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


def _walk(node: object):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


# --- FingerDesign ---------------------------------------------------------------


def test_ten_geometry_parameters() -> None:
    assert len(GEOMETRY_PARAMS) == 10
    assert set(GEOMETRY_PARAMS) == set(PARAM_BOUNDS) | set(CATEGORICAL_OPTIONS)
    assert CATEGORICAL_OPTIONS["inner_face"] == ("flat", "concave_arc", "v_groove")
    assert CATEGORICAL_OPTIONS["tip_profile"] == ("blunt", "rounded", "chamfer", "hook")


@pytest.mark.parametrize("key", sorted(PARAM_BOUNDS))
def test_bounds_are_inclusive_and_enforced(key: str) -> None:
    lo, hi = PARAM_BOUNDS[key]
    make(**{key: lo})
    make(**{key: hi})
    with pytest.raises(ValidationError):
        make(**{key: lo - 1e-3})
    with pytest.raises(ValidationError):
        make(**{key: hi + 1e-3})


def test_unknown_category_and_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        make(inner_face="spiral")
    with pytest.raises(ValidationError):
        make(tip_profile="pointy")
    with pytest.raises(ValidationError):
        make(colour="red")
    with pytest.raises(ValidationError):
        FingerDesign(**{k: v for k, v in BASE.items() if k != "taper_ratio"})


def test_design_is_immutable_and_hashable() -> None:
    design = make()
    with pytest.raises(ValidationError):
        design.finger_length_mm = 90.0  # type: ignore[misc]
    assert hash(design) == hash(make())


def test_meta_fields_do_not_affect_geometry_identity() -> None:
    plain = make()
    labelled = make(name="baseline", rationale="flat pads are simple")
    assert "name" not in labelled.geometry_params()
    assert "rationale" not in labelled.geometry_params()
    assert labelled.geometry_params() == plain.geometry_params()
    assert labelled.design_id() == plain.design_id()
    assert make(finger_length_mm=80.5).design_id() != plain.design_id()
    assert len(plain.design_id()) == 12


def test_design_id_ignores_signed_zero() -> None:
    plain = make()
    assert make(bend_angle_deg=-0.0).design_id() == plain.design_id()
    assert make(bend_angle_deg=-1e-9).design_id() == plain.design_id()  # rounds to -0.0


def test_model_json_schema_describes_every_field() -> None:
    schema = FingerDesign.model_json_schema()
    props = schema["properties"]
    assert set(GEOMETRY_PARAMS) <= set(props)
    for key in GEOMETRY_PARAMS:
        assert len(props[key].get("description", "")) > 60, key
    for key, (lo, hi) in PARAM_BOUNDS.items():
        assert props[key]["minimum"] == lo and props[key]["maximum"] == hi
    for key, options in CATEGORICAL_OPTIONS.items():
        assert tuple(props[key]["enum"]) == options
    assert set(schema["required"]) == set(GEOMETRY_PARAMS)


def test_llm_tool_schema_is_flat_and_strict() -> None:
    schema = design_json_schema(strict=True)
    json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert "$defs" not in schema and "$ref" not in json.dumps(schema)
    assert not any("title" in node for node in _walk(schema))
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["rationale"]["type"] == ["string", "null"]

    loose = design_json_schema()
    assert loose["required"] == list(GEOMETRY_PARAMS)
    assert loose["properties"]["name"]["type"] == "string"
    assert "name" not in design_json_schema(include_meta=False)["properties"]


# --- GripperTask -------------------------------------------------------------------


def test_task_defaults() -> None:
    task = load_task()
    assert task == GripperTask()
    assert task.stroke_mm == 42.0
    assert task.max_opening_mm == task.closed_gap_mm + 84.0
    assert task.required_open_gap_mm == task.widest_training_object_mm + 5.0
    assert task.envelope == ToolEnvelope()


def test_task_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text("gripper:\n  stroke_mm: 35\n  mass_budget_g: 40\n  envelope:\n    half_span_y_mm: 80\n")
    task = load_task(path, section="gripper")
    assert task.stroke_mm == 35.0
    assert task.mass_budget_g == 40.0
    assert task.envelope.half_span_y_mm == 80.0
    assert task.envelope.max_reach_z_mm == ToolEnvelope().max_reach_z_mm


@pytest.mark.parametrize(
    "data",
    [{"stroke_mm": 0}, {"stroke_mm": -5}, {"closed_gap_mm": -1}, {"mass_budget_g": 0}, {"bogus": 1}],
)
def test_task_rejects_bad_values(data: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GripperTask.from_mapping(data)


def test_task_accepts_a_wider_stroke() -> None:
    """The stroke is a calibration input (e.g. an 85 mm span), not capped at the default."""
    assert GripperTask(stroke_mm=42.5).max_opening_mm == pytest.approx(85.0)


def test_load_task_rejects_a_missing_section(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text("gripper:\n  stroke_mm: 35\n")
    with pytest.raises(ValueError, match="no top-level 'design' section"):
        load_task(path, section="design")


def test_load_entrants() -> None:
    entrants = load_entrants(ENTRANTS)
    assert [e.name for e in entrants] == ["straight", "concave", "hooked", "stock"]
    straight, concave, hooked, stock = entrants
    assert (straight.inner_face, straight.bend_angle_deg, straight.tip_profile) == ("flat", 0.0, "blunt")
    assert (concave.inner_face, concave.inner_face_depth_mm) == ("concave_arc", 4.0)
    assert (hooked.tip_profile, hooked.tip_size_mm, hooked.bend_angle_deg) == ("hook", 8.0, 15.0)
    assert (stock.finger_length_mm, stock.taper_ratio, stock.tip_profile) == (62.5, 0.5, "rounded")


# --- distance ------------------------------------------------------------------------


def test_distance_identity_symmetry_and_range() -> None:
    a = make()
    b = make(finger_length_mm=120.0, inner_face="v_groove", inner_face_depth_mm=3.0)
    assert design_distance(a, a) == 0.0
    assert design_distance(a, make(name="x", rationale="y")) == 0.0
    assert design_distance(a, b) == pytest.approx(design_distance(b, a))
    assert 0.0 < design_distance(a, b) <= 1.0


def test_distance_terms() -> None:
    a = make()
    b = make(finger_length_mm=120.0, tip_profile="hook")
    terms = parameter_distances(a, b)
    assert terms["finger_length_mm"] == pytest.approx(0.5)
    assert terms["tip_profile"] == 1.0
    assert terms["inner_face"] == 0.0
    assert design_distance(a, b) == pytest.approx(1.5 / 10)


def test_distance_extremes_reach_one() -> None:
    lo = {k: v[0] for k, v in PARAM_BOUNDS.items()}
    hi = {k: v[1] for k, v in PARAM_BOUNDS.items()}
    a = FingerDesign(**lo, inner_face="flat", tip_profile="blunt")
    b = FingerDesign(**hi, inner_face="v_groove", tip_profile="hook")
    assert design_distance(a, b) == pytest.approx(1.0)


def test_min_and_pairwise_distances() -> None:
    designs = [make(), make(finger_width_mm=30.0), make(taper_ratio=0.5)]
    assert min_distance(designs[0], []) == math.inf
    assert min_distance(designs[0], designs[1:]) == pytest.approx(min(0.05, 0.1))
    matrix = pairwise_distances(designs)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0.0)
