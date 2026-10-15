"""Suite files: per-size materials and explicit condition lists (rtt.sim.conditions)."""

from __future__ import annotations

import pytest

from rtt.sim.conditions import suite_spec_from_dict, training_suite

BASE = {
    "name": "t",
    "base_seed": 1,
    "materials": {"plastic": 1200.0, "aluminium": 2700.0},
    "families": {
        "box": {
            "small": [0.04, 0.03, 0.05],
            "bar": {"size": [0.16, 0.03, 0.03], "material": "aluminium"},
        },
        "rod": {"peg": {"size": [0.007, 0.14], "density": 7850.0}},
    },
    "placements": {"centred": {}, "handle": {"err_x_mm": 35.0}},
}


def test_cross_product_without_a_condition_list() -> None:
    suite = training_suite(suite_spec_from_dict(BASE))
    assert [c.condition_id for c in suite] == [
        "t/box-small/centred",
        "t/box-small/handle",
        "t/box-bar/centred",
        "t/box-bar/handle",
        "t/rod-peg/centred",
        "t/rod-peg/handle",
    ]
    dens = {c.condition_id: c.obj.density for c in suite}
    assert dens["t/box-small/centred"] is None  # the task's default density
    assert dens["t/box-bar/handle"] == 2700.0 and dens["t/rod-peg/centred"] == 7850.0
    handle = next(c for c in suite if c.placement_label == "handle")
    assert handle.placement.err_x == pytest.approx(0.035)


def test_explicit_conditions_keep_file_order_and_density_enters_the_digest() -> None:
    spec = suite_spec_from_dict(
        {**BASE, "conditions": ["rod/peg/handle", "box/bar/handle", "box/small/centred"]}
    )
    suite = training_suite(spec)
    assert [c.condition_id for c in suite] == ["t/rod-peg/handle", "t/box-bar/handle", "t/box-small/centred"]
    assert spec.family_densities("box") == (2700.0, None)
    lighter = {
        **BASE,
        "families": {**BASE["families"], "rod": {"peg": {"size": [0.007, 0.14], "density": 2700.0}}},
    }
    other = training_suite(
        suite_spec_from_dict(
            {**lighter, "conditions": ["rod/peg/handle", "box/bar/handle", "box/small/centred"]}
        )
    )
    assert other.digest() != suite.digest()


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"conditions": ["box/small"]}, "family/size/placement"),
        ({"conditions": ["box/huge/centred"]}, "unknown family or size"),
        ({"conditions": ["box/small/sideways"]}, "unknown placement"),
        ({"conditions": ["box/small/centred", "box/small/centred"]}, "without duplicates"),
        ({"conditions": ["box/small/centred"]}, "not used by any condition"),
        ({"families": {"box": {"x": {"size": [0.04, 0.03, 0.05], "material": "gold"}}}}, "unknown material"),
        ({"families": {"box": {"x": {"size": [0.04, 0.03, 0.05], "colour": "red"}}}}, "unknown keys"),
        ({"materials": {"foam": 0.0}}, "positive density"),
    ],
)
def test_bad_suite_files_are_rejected(patch: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        suite_spec_from_dict({**BASE, **patch})
