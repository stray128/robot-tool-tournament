"""FingerRecipe v2: bounds, strategy rules, inactive parameters, unit-cube mapping, JSON schema."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from rtt.design.freeform.schema import (
    CONTINUOUS_PARAMS,
    DESCRIPTIONS,
    FIXED_VALUES,
    PARAM_BOUNDS,
    SCORED_PARAMS,
    FingerRecipe,
    inactive_params,
    quantize,
    quantized_values,
    recipe_from_unit,
    recipe_json_schema,
    strategy_violation,
)

BASE = {
    "pad": "conform",
    "tip": "blunt",
    "grip_height_frac": 0.5,
    "pad_height_mm": 20.0,
    "finger_width_mm": 30.0,
}
CONFORM = dict(BASE, pad_depth_mm=4.0, wall_draft_deg=30.0, clearance_mm=0.4)


def test_nine_scored_parameters_two_categoricals() -> None:
    assert len(SCORED_PARAMS) == 9
    assert SCORED_PARAMS[:2] == ("pad", "tip")
    assert len(CONTINUOUS_PARAMS) == 7
    assert PARAM_BOUNDS["finger_width_mm"] == (14.0, 40.0)
    assert PARAM_BOUNDS["wall_draft_deg"] == (0.0, 70.0)


def test_valid_recipes_of_every_strategy() -> None:
    FingerRecipe(**CONFORM)
    FingerRecipe(**dict(BASE, pad="flat"))
    FingerRecipe(**dict(BASE, pad="v_funnel", pad_depth_mm=5.0, wall_draft_deg=40.0))
    FingerRecipe(**dict(CONFORM, tip="hook", tip_size_mm=4.0))
    FingerRecipe(**dict(CONFORM, tip="chamfer", tip_size_mm=3.0))


def test_extra_fields_and_out_of_bounds_are_rejected() -> None:
    with pytest.raises(ValidationError):
        FingerRecipe(**CONFORM, finger_length_mm=80.0)
    with pytest.raises(ValidationError):
        FingerRecipe(**dict(CONFORM, finger_width_mm=41.0))
    with pytest.raises(ValidationError):
        FingerRecipe(**dict(CONFORM, pad="imprint"))
    with pytest.raises(ValidationError):
        FingerRecipe(**dict(CONFORM, grip_height_frac=0.9))


def test_recipe_is_frozen() -> None:
    r = FingerRecipe(**CONFORM)
    with pytest.raises(ValidationError):
        r.pad_depth_mm = 5.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"pad": "v_funnel", "pad_depth_mm": 5.0, "wall_draft_deg": 10.0}, "v_funnel needs wall_draft_deg"),
        ({"pad": "v_funnel", "pad_depth_mm": 2.0, "wall_draft_deg": 40.0}, "v_funnel needs pad_depth_mm"),
        ({"pad_depth_mm": 0.5}, "conform needs pad_depth_mm"),
        ({"tip": "hook", "tip_size_mm": 1.5}, "hook needs tip_size_mm"),
        ({"tip": "hook", "tip_size_mm": 8.0, "pad_height_mm": 10.0}, "needs pad_height_mm >= 12"),
        ({"tip": "chamfer", "tip_size_mm": 8.0, "pad_height_mm": 8.0}, "chamfer needs tip_size_mm"),
    ],
)
def test_strategy_rules(changes: dict, words: str) -> None:
    with pytest.raises(ValidationError, match=words):
        FingerRecipe(**dict(CONFORM, **changes))


def test_strategy_rules_use_quantised_values() -> None:
    FingerRecipe(**dict(CONFORM, pad_depth_mm=0.96))  # quantises to 1.0
    with pytest.raises(ValidationError):
        FingerRecipe(**dict(CONFORM, pad_depth_mm=0.94))  # quantises to 0.9
    FingerRecipe(**dict(BASE, pad="v_funnel", pad_depth_mm=3.0, wall_draft_deg=14.8))  # -> 15.0


def test_active_parameters_are_required_inactive_ones_optional() -> None:
    with pytest.raises(ValidationError, match="needs clearance_mm"):
        FingerRecipe(**dict(BASE, pad_depth_mm=4.0, wall_draft_deg=30.0))
    with pytest.raises(ValidationError, match="needs tip_size_mm"):
        FingerRecipe(**dict(CONFORM, tip="hook"))
    flat = FingerRecipe(**dict(BASE, pad="flat"))
    assert flat.pad_depth_mm is None and flat.clearance_mm is None
    # inactive values may be given and are ignored by canonicalisation
    FingerRecipe(**dict(BASE, pad="flat", pad_depth_mm=12.0, wall_draft_deg=45.0, clearance_mm=1.0))


def test_inactive_values_are_ignored_even_outside_the_bounds() -> None:
    """Spec section 12: only active parameters are required; inactive ones are ignored and logged."""
    from rtt.design.freeform.canon import canonicalize
    from rtt.design.freeform.hashing import design_id

    flat = FingerRecipe(**dict(BASE, pad="flat"))
    odd = FingerRecipe(
        **dict(BASE, pad="flat", pad_depth_mm=20.0, wall_draft_deg=80.0, clearance_mm=0.05, tip_size_mm=9.0)
    )
    assert design_id(odd) == design_id(flat)
    _, log = canonicalize(odd)
    assert len(log) == 4 and all("inactive" in line for line in log)
    with pytest.raises(ValidationError, match="pad_depth_mm must be in"):
        FingerRecipe(**dict(CONFORM, pad_depth_mm=16.0))  # active: the bound binds
    with pytest.raises(ValidationError, match="finite"):
        FingerRecipe(**dict(BASE, pad="flat", pad_depth_mm=float("nan")))
    props = recipe_json_schema()["properties"]["pad_depth_mm"]["anyOf"]
    assert {"type": "number", "minimum": 0.0, "maximum": 15.0}.items() <= props[0].items()


def test_inactive_params_table() -> None:
    assert set(inactive_params("flat", "blunt")) == {
        "pad_depth_mm",
        "wall_draft_deg",
        "clearance_mm",
        "tip_size_mm",
    }
    assert set(inactive_params("v_funnel", "hook")) == {"clearance_mm"}
    assert inactive_params("conform", "chamfer") == ()
    assert FIXED_VALUES == {
        "pad_depth_mm": 0.0,
        "wall_draft_deg": 0.0,
        "clearance_mm": 0.2,
        "tip_size_mm": 0.0,
    }


def test_quantize_steps_and_half_up() -> None:
    assert quantize("pad_depth_mm", 4.04) == 4.0
    assert quantize("pad_depth_mm", 4.05) == 4.1
    assert quantize("wall_draft_deg", 30.24) == 30.0
    assert quantize("wall_draft_deg", 30.25) == 30.5
    assert quantize("grip_height_frac", 0.505) == 0.51
    assert quantize("clearance_mm", 0.25) == 0.3


def test_quantized_values_resets_inactive() -> None:
    r = FingerRecipe(**dict(BASE, pad="flat", pad_depth_mm=12.3, clearance_mm=1.1))
    v = quantized_values(r)
    assert v["pad_depth_mm"] == 0.0 and v["clearance_mm"] == 0.2 and v["tip_size_mm"] == 0.0
    assert v["locator"] == "none" and v["body_style"] == "windowed"


def test_product_options_default_to_scored_values() -> None:
    r = FingerRecipe(**CONFORM)
    assert r.locator == "none" and r.body_style == "windowed" and not r.is_product_mode
    assert FingerRecipe(**dict(CONFORM, locator="conical_pin")).is_product_mode
    with pytest.raises(ValidationError):
        FingerRecipe(**dict(CONFORM, body_style="finray"))


def test_strategy_violation_messages_name_the_rule() -> None:
    assert strategy_violation({"pad": "flat", "tip": "blunt"}) is None
    assert "v_funnel" in strategy_violation(
        {"pad": "v_funnel", "tip": "blunt", "wall_draft_deg": 5.0, "pad_depth_mm": 5.0}
    )


def test_recipe_from_unit_maps_bounds_and_categoricals() -> None:
    r = recipe_from_unit(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]))
    assert r.pad == "conform" and r.tip == "blunt"
    assert r.grip_height_frac == 0.15 and r.pad_height_mm == 8.0 and r.finger_width_mm == 14.0
    assert r.pad_depth_mm == 7.5 and r.wall_draft_deg == 35.0
    hi = recipe_from_unit(np.array([0.99] * 9))
    assert hi.pad == "v_funnel" and hi.tip == "hook"
    with pytest.raises(ValueError):
        recipe_from_unit(np.array([0.5] * 8))
    with pytest.raises(ValueError):
        recipe_from_unit(np.array([1.5] + [0.5] * 8))


def test_json_schema_describes_every_field() -> None:
    schema = recipe_json_schema()
    props = schema["properties"]
    for name in (*SCORED_PARAMS, "locator", "body_style"):
        assert name in props
        assert props[name].get("description") or any("description" in x for x in props[name].get("anyOf", []))
        assert name in DESCRIPTIONS
    assert schema["additionalProperties"] is False
