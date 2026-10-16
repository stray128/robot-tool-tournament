"""Diversity metric in points and cells, the explore novelty rule, and the scale of the 0.08 threshold."""

from __future__ import annotations

import statistics

import pytest

from rtt.design.distance import design_distance
from rtt.design.schema import FingerDesign, load_entrants
from rtt.search.base import derive_seed
from rtt.search.diversity import (
    EXPLORE_MIN_DISTANCE,
    MIN_DISTANCE,
    NUMERIC_PARAMS,
    all_cells,
    cell_name,
    cell_of,
    distance_points,
    explore_check,
    nearest,
    neighbours,
    points_of,
    sequential_nn_distances,
    to_points,
)
from rtt.search.fakes import FakeEvaluator
from rtt.search.random_search import DesignSampler, round0_designs
from rtt.search.runner import DEFAULT_ENTRANTS
from rtt.sim.task import gripper_task_for, load_task

STRAIGHT = FingerDesign(
    finger_length_mm=80.0,
    finger_thickness_mm=10.0,
    finger_width_mm=20.0,
    bend_angle_deg=0.0,
    bend_start_frac=0.5,
    inner_face="flat",
    inner_face_depth_mm=0.0,
    tip_profile="blunt",
    tip_size_mm=0.0,
    taper_ratio=1.0,
)


def with_(design: FingerDesign, **changes: object) -> FingerDesign:
    return FingerDesign.model_validate({**design.geometry_params(), **changes})


def test_points_map_each_range_onto_0_to_100() -> None:
    assert points_of(STRAIGHT) == {
        "finger_length_mm": 50,
        "finger_thickness_mm": 50,
        "finger_width_mm": 50,
        "bend_angle_deg": 25,
        "bend_start_frac": 43,
        "inner_face_depth_mm": 0,
        "tip_size_mm": 0,
        "taper_ratio": 100,
    }
    assert to_points("finger_length_mm", 40.0) == 0.0 and to_points("finger_length_mm", 120.0) == 100.0
    assert len(NUMERIC_PARAMS) == 8


def test_distance_in_points_is_the_sum_of_point_differences() -> None:
    longer = with_(STRAIGHT, finger_length_mm=104.0, finger_width_mm=26.0)  # +30 and +30 points
    assert distance_points(STRAIGHT, longer) == 60
    assert distance_points(STRAIGHT, longer) == round(1000 * design_distance(STRAIGHT, longer))
    # One categorical change is 100 points: designs in different cells are never closer than that.
    hooked = with_(STRAIGHT, tip_profile="hook", tip_size_mm=0.0)
    assert distance_points(STRAIGHT, hooked) == 100 and cell_of(hooked) != cell_of(STRAIGHT)
    assert design_distance(STRAIGHT, hooked) >= MIN_DISTANCE


def test_cells_cover_every_face_and_tip_pair() -> None:
    cells = all_cells()
    assert len(cells) == 12 and len(set(cells)) == 12
    assert cell_name(cell_of(STRAIGHT)) == "flat + blunt"


def test_explore_check_wants_an_untried_cell_or_a_far_design() -> None:
    refs = [STRAIGHT, with_(STRAIGHT, tip_profile="rounded", tip_size_mm=2.0)]
    labels = ["d01", "d02"]
    untried = with_(STRAIGHT, inner_face="v_groove", inner_face_depth_mm=2.0)
    assert explore_check(untried, refs, labels) == (True, None)
    close = with_(STRAIGHT, finger_length_mm=112.0, finger_width_mm=28.0)  # 80 points from d01, same cell
    ok, near = explore_check(close, refs, labels)
    assert not ok and near is not None and near.label == "d01" and near.index == 0
    assert near.distance == pytest.approx(0.08)
    far = with_(
        STRAIGHT, finger_length_mm=120.0, finger_width_mm=30.0, finger_thickness_mm=16.0, taper_ratio=0.5
    )
    assert explore_check(far, refs, labels, EXPLORE_MIN_DISTANCE)[0]
    assert not explore_check(far, refs, labels, 0.5)[0]


def test_neighbours_are_sorted_and_bounded() -> None:
    refs = [
        with_(STRAIGHT, finger_length_mm=120.0),
        with_(STRAIGHT, finger_length_mm=88.0),
        STRAIGHT,
        with_(STRAIGHT, tip_profile="hook"),
    ]
    near = neighbours(STRAIGHT, refs, ["a", "b", "c", "d"], k=3)
    assert [n.label for n in near] == ["c", "b", "a"]
    assert [round(1000 * n.distance) for n in near] == [0, 10, 50]
    assert [n.label for n in neighbours(STRAIGHT, refs, ["a", "b", "c", "d"], k=4, max_distance=0.02)] == [
        "c",
        "b",
    ]
    assert neighbours(STRAIGHT, [], k=3) == []


def test_sequential_nn_distances_match_nearest() -> None:
    designs = DesignSampler(7, FakeEvaluator(0).describe().gripper_task).sample(6)
    got = sequential_nn_distances(designs, start=2)
    assert got == [nearest(designs[i], designs[:i]).distance for i in range(2, 6)]


def test_threshold_never_binds_on_random_search() -> None:
    """The documented scale of MIN_DISTANCE: random draws are almost never this close to the run."""
    task = gripper_task_for(load_task())  # the frozen task's validator constants
    entrants = load_entrants(DEFAULT_ENTRANTS)
    distances: list[float] = []
    for seed in (101, 102, 103, 104):
        designs = [p.design for p in round0_designs(entrants, seed, 8, task)]
        designs += DesignSampler(derive_seed(seed, "random"), task).sample(32)
        distances += sequential_nn_distances(designs, start=8)
    below = sum(d < MIN_DISTANCE for d in distances) / len(distances)
    assert below <= 0.03
    assert statistics.median(distances) > 2 * MIN_DISTANCE
