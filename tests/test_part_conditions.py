"""The per-part training suite: families per pose, grasp points, placements and seeds."""

from __future__ import annotations

import math

import pytest

from rtt.part.conditions import (
    DEFAULT_PLACEMENTS,
    SLOPPY_ERR_MM,
    SLOPPY_ERR_YAW_DEG,
    conditions_for_part,
    describe_pose,
)
from rtt.part.examples import example_path
from rtt.part.intake import PartSpec, intake
from rtt.search.sim_adapter import SimEvaluator, describe_condition


@pytest.fixture(scope="module")
def bracket() -> PartSpec:
    return intake(example_path("l_bracket"), decomposition="hull")


@pytest.fixture(scope="module")
def tub() -> PartSpec:
    return intake(example_path("rounded_tub"), decomposition="hull")


def test_one_family_per_pose_and_unique_ids(tub: PartSpec) -> None:
    suite = conditions_for_part(tub)
    assert suite.name == "part_rounded_tub"
    assert {c.family for c in suite} == {f"pose{p.rank}" for p in tub.poses}
    ids = [c.condition_id for c in suite]
    assert len(ids) == len(set(ids))
    assert all(c.placement_label in DEFAULT_PLACEMENTS for c in suite)
    assert all(c.obj.kind == "mesh" for c in suite)


def test_suite_is_deterministic_and_seeded_from_the_file(tub: PartSpec) -> None:
    a, b = conditions_for_part(tub), conditions_for_part(tub)
    assert a.digest() == b.digest()
    assert [c.seeds for c in a] == [c.seeds for c in b]
    other = conditions_for_part(tub, base_seed=7)
    assert [c.seeds for c in other] != [c.seeds for c in a]
    assert conditions_for_part(tub, seeds_per_condition=3).n_episodes_per_setting == 3 * len(a)


def test_placements_aim_at_the_grasp_point(bracket: PartSpec) -> None:
    suite = conditions_for_part(bracket, placements=("centred", "sloppy", "offcentre"), max_conditions=50)
    for cond in suite:
        pose = bracket.poses[int(cond.family.removeprefix("pose"))]
        cand = next(c for c in pose.candidates if f"-{c.label}/" in cond.condition_id)
        if cond.placement_label == "centred":
            assert cond.placement.err_x == pytest.approx(-cand.offset_mm / 1000.0)
            assert (cond.placement.x, cond.placement.y, cond.placement.err_y) == (0.0, 0.0, 0.0)
        elif cond.placement_label == "sloppy":
            assert cond.placement.err_x == pytest.approx(-cand.offset_mm / 1000.0 + SLOPPY_ERR_MM / 1000.0)
            assert cond.placement.err_y == pytest.approx(SLOPPY_ERR_MM / 1000.0)
            assert math.degrees(cond.placement.err_yaw) == pytest.approx(SLOPPY_ERR_YAW_DEG)
        else:
            assert cond.placement.err_x > -cand.offset_mm / 1000.0


def test_cap_drops_whole_unlikely_poses(tub: PartSpec) -> None:
    full = conditions_for_part(tub, max_conditions=100)
    n_first = sum(1 for c in full if c.family == "pose0")
    capped = conditions_for_part(tub, max_conditions=n_first + 1)
    assert {c.family for c in capped} == {"pose0"}  # the next pose does not fit whole
    assert len(capped) == n_first
    first = conditions_for_part(tub, max_conditions=1)
    assert len(first) == 1 and first.conditions[0].family == "pose0"
    assert len(full) >= len(capped)


def test_wide_axis_only_when_asked(bracket: PartSpec) -> None:
    narrow = conditions_for_part(bracket, max_conditions=100)
    both = conditions_for_part(bracket, axes="all", max_conditions=100)
    assert all("-narrow-" in c.condition_id for c in narrow)
    assert any("-wide-" in c.condition_id for c in both)


def test_bad_arguments(tub: PartSpec) -> None:
    with pytest.raises(ValueError):
        conditions_for_part(tub, placements=())
    with pytest.raises(ValueError):
        conditions_for_part(tub, placements=("centred", "centred"))
    with pytest.raises(ValueError):
        conditions_for_part(tub, placements=("upside_down",))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        conditions_for_part(tub, seeds_per_condition=0)


def test_the_existing_evaluator_takes_the_suite(tub: PartSpec) -> None:
    suite = conditions_for_part(tub)
    evaluator = SimEvaluator(suite, workers=1)
    info = evaluator.describe()
    assert info.families == tuple(dict.fromkeys(c.family for c in suite))
    assert info.episodes_per_design == len(suite) * 3
    widest = max(p.extents_mm[1] for p in tub.poses)
    assert info.gripper_task.widest_training_object_mm == pytest.approx(widest, abs=0.11)
    label = describe_condition(suite.conditions[0])
    assert "uploaded part 'rounded_tub'" in label and "% of drops" in label
    pose, cand = tub.poses[0], tub.poses[0].candidates[0]
    assert describe_pose(tub, pose, cand).endswith(";")
