"""Training suite and the commit-reveal held-out suite."""

from __future__ import annotations

import inspect
import math
from collections import Counter

import numpy as np
import pytest

from rtt.design.schema import load_entrants
from rtt.design.validate import validate
from rtt.sim import objects as objs
from rtt.sim.conditions import (
    DEFAULT_TRAIN_SUITE,
    HELDOUT_EXTRAP,
    REPO_ROOT,
    derive_seed,
    heldout_suite,
    load_suite_spec,
    mesh_suite,
    new_secret_seed,
    overlapping_conditions,
    read_secret_seed,
    replace_seeds,
    seed_commitment,
    suite_spec_from_dict,
    training_suite,
)
from rtt.sim.m1 import DEFAULT_ENTRANTS
from rtt.sim.task import load_task

TASK = load_task()
SECRET = "0123456789abcdef" * 4  # a test secret (64 characters); never the real one


@pytest.fixture(scope="module")
def train():
    return training_suite()


@pytest.fixture(scope="module")
def heldout():
    return heldout_suite(SECRET, task=TASK)


def test_training_suite_is_the_calibrated_16_condition_parts_suite(train) -> None:
    assert train.name == "train_v1"
    assert len(train) == 16
    families = Counter(c.family for c in train)
    assert families == {"box": 8, "cylinder": 2, "rod": 6}
    assert [c.condition_id for c in train][:2] == ["train_v1/box-large/sloppy", "train_v1/box-long/handle20"]
    # one public, hash-derived jitter seed per condition: 16 x k = 3 = 48 episodes per design
    assert all(len(c.seeds) == 1 for c in train)
    assert train.n_episodes_per_setting == 16
    assert len({c.seeds[0] for c in train}) == 16
    spec = load_suite_spec(DEFAULT_TRAIN_SUITE)
    for c in train:
        assert c.seeds == (derive_seed(spec.base_seed, c.condition_id, 0),)
        assert c.obj.size == tuple(spec.families[c.family][c.size_label])
        assert c.obj.density == spec.density(c.family, c.size_label)
    dens = {c.condition_id: c.obj.density for c in train}
    assert dens["train_v1/box-long/handle35"] == 2700.0 and dens["train_v1/cylinder-thin/sloppy"] == 1200.0


def test_training_suite_is_deterministic(train) -> None:
    again = training_suite()
    assert again == train
    assert again.digest() == train.digest()
    # the seed derivation is a fixed SHA-256 function, identical on every platform
    assert derive_seed(1, "a", 0) == derive_seed(1, "a", 0) != derive_seed(1, "a", 1)
    assert 0 <= derive_seed("x") < 2**63


def test_placements_carry_pose_errors_and_grasp_offsets(train) -> None:
    sloppy = next(c for c in train if c.placement_label == "sloppy")
    p = sloppy.placement
    assert (p.x, p.y) == (-0.03, 0.04)
    assert math.degrees(p.yaw) == pytest.approx(-60.0)
    assert (p.err_x, p.err_y) == pytest.approx((0.006, 0.006))
    assert math.degrees(p.err_yaw) == pytest.approx(15.0)
    centred = next(c for c in train if c.placement_label == "centred")
    assert centred.placement == objs.Placement()
    for c in train:
        if c.placement_label.startswith("handle"):
            offset_mm = int(c.placement_label.removeprefix("handle"))
            assert c.placement.err_x == pytest.approx(offset_mm / 1000.0)
            # the grasp stays on the part: at least 15 mm of it beyond the grasp centre
            assert objs.half_extents(c.obj)[0] - c.placement.err_x >= 0.015 - 1e-9


def test_suite_gripper_task_admits_every_entrant(train) -> None:
    gripper = train.gripper_task(TASK)
    assert train.widest_grip_width == pytest.approx(0.05)
    assert gripper.widest_training_object_mm == pytest.approx(50.0)
    assert gripper.stroke_mm == pytest.approx(42.0) and gripper.closed_gap_mm == 0.0
    assert gripper.max_opening_mm == pytest.approx(84.0)
    for design in load_entrants(DEFAULT_ENTRANTS):
        assert validate(design, gripper).ok, design.name


def test_suite_spec_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        suite_spec_from_dict({"families": {"sphere": {"a": [0.02]}}, "placements": {"c": {}}})
    with pytest.raises(ValueError, match="unknown placement keys"):
        suite_spec_from_dict({"families": {"box": {"a": [0.03, 0.03, 0.03]}}, "placements": {"c": {"z": 1}}})
    with pytest.raises(ValueError, match="metres"):
        suite_spec_from_dict({"families": {"box": {"a": [30, 30, 30]}}, "placements": {"c": {}}})


def test_replace_seeds_keeps_conditions(train) -> None:
    swapped = replace_seeds(train, [5, 6])
    assert [c.condition_id for c in swapped] == [c.condition_id for c in train]
    assert all(c.seeds == (5, 6) for c in swapped)


# --------------------------------------------------------------------------- held-out


def test_heldout_needs_a_runtime_secret() -> None:
    # no default: the secret can only come from the caller, never from repository contents
    assert inspect.signature(heldout_suite).parameters["secret"].default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="at least 32"):
        heldout_suite("too-short")
    with pytest.raises(ValueError, match="at least 32"):
        heldout_suite(b"")


def test_heldout_is_100_conditions_x_2_seeds(heldout) -> None:
    assert len(heldout) == 100
    assert heldout.n_episodes_per_setting == 200
    assert all(len(c.seeds) == 2 for c in heldout)
    counts = Counter(c.family for c in heldout)
    assert sorted(counts.values()) == [33, 33, 34]
    modes = Counter(c.size_label for c in heldout)
    assert set(modes) == {"interp", "down", "up"}
    assert len({s for c in heldout for s in c.seeds}) == 200


def test_heldout_is_a_pure_function_of_the_secret(heldout) -> None:
    again = heldout_suite(SECRET, task=TASK)
    assert again.digest() == heldout.digest()
    assert heldout_suite(SECRET.encode(), task=TASK).digest() == heldout.digest()
    other = heldout_suite("f" * 64, task=TASK)
    assert other.digest() != heldout.digest()


def test_heldout_commitment_is_checked(heldout) -> None:
    commit = seed_commitment(SECRET)
    assert len(commit) == 64 and commit != SECRET
    assert heldout_suite(SECRET, commitment=commit, task=TASK).digest() == heldout.digest()
    with pytest.raises(ValueError, match="does not match"):
        heldout_suite(SECRET, commitment=seed_commitment("e" * 64), task=TASK)
    fresh = new_secret_seed()
    assert len(fresh) == 64 and fresh != new_secret_seed()


def test_heldout_sizes_extrapolate_15_to_30_percent(heldout) -> None:
    spec = load_suite_spec()
    max_width = 2 * TASK.gripper.open_half_gap - 0.005
    lo_f, hi_f = HELDOUT_EXTRAP
    for c in heldout:
        lo, hi = spec.size_range(c.family)
        size = np.array(c.obj.size)
        width = objs.grip_width(c.obj)
        assert width <= max_width + 1e-9
        capped = width >= max_width - 1e-9
        if c.family == "rod":
            assert size[1] >= 1.5 * 2 * size[0] - 1e-9
        if c.size_label == "down":
            ratio = size / lo
            if c.family != "rod":
                assert np.allclose(ratio, ratio[0], atol=1e-4)
            assert 1 - hi_f - 1e-4 <= ratio[0] <= 1 - lo_f + 1e-4
        elif c.size_label == "up" and not capped:
            ratio = size / hi
            assert np.allclose(ratio, ratio[0], atol=1e-4)
            assert 1 + lo_f - 1e-4 <= ratio[0] <= 1 + hi_f + 1e-4
        elif c.size_label == "interp" and not capped:
            assert np.all(size >= lo - 1e-9) and np.all(size <= hi + 1e-9)


def test_heldout_placements_are_new(train, heldout) -> None:
    assert overlapping_conditions(heldout, train) == []
    assert overlapping_conditions(train, train) == [c.condition_id for c in train]  # sanity
    yaws = {round(math.degrees(c.placement.yaw)) for c in heldout}
    assert len(yaws) > 50
    for c in heldout:
        p = c.placement
        assert max(abs(p.x), abs(p.y)) <= 0.08
        assert abs(p.err_y) <= 0.003
        if c.placement_label == "random-offcentre":  # an off-centre grasp, as in training
            half_x = objs.half_extents(c.obj)[0]
            assert 0.010 - 1e-9 <= abs(p.err_x) <= half_x - 0.015 + 1e-9
        else:
            assert c.placement_label == "random" and abs(p.err_x) <= 0.003
        assert abs(math.degrees(p.err_yaw)) <= 8.0 + 1e-6


def test_heldout_mirrors_training_materials_and_offcentre_grasps(heldout) -> None:
    spec = load_suite_spec()
    for c in heldout:
        assert c.obj.density in spec.family_densities(c.family)
    trained = {f for f, _, pl in spec.triples() if abs(spec.placements[pl].err_x) >= 0.010}
    offcentre = {c.family for c in heldout if c.placement_label == "random-offcentre"}
    assert offcentre <= trained


def test_read_secret_seed_refuses_tracked_paths(tmp_path) -> None:
    secret_file = tmp_path / "heldout_seed.txt"
    secret_file.write_text(SECRET + "\n")
    assert read_secret_seed(secret_file) == SECRET.encode()
    # a path inside the repository that git does not ignore is refused before it is read
    with pytest.raises(PermissionError, match="git-ignored"):
        read_secret_seed(REPO_ROOT / "configs" / "heldout_seed.txt")
    short = tmp_path / "short.txt"
    short.write_text("abc")
    with pytest.raises(ValueError, match="at least 32"):
        read_secret_seed(short)


def test_mesh_suite_builds_one_condition_per_object_and_placement() -> None:
    import trimesh

    from rtt.sim.mesh import mesh_object_from_trimesh

    part = mesh_object_from_trimesh(
        trimesh.creation.box((0.04, 0.03, 0.05)), name="block", decomposition="hull"
    )
    suite = mesh_suite("byp", [part], {"a": objs.Placement(), "b": objs.Placement(0.02, 0.0, 0.5)}, 2)
    assert [c.condition_id for c in suite] == ["byp/block/a", "byp/block/b"]
    assert all(c.family == "mesh" and len(c.seeds) == 2 for c in suite)
    assert suite.describe()["conditions"][0]["object"]["name"] == "block"
