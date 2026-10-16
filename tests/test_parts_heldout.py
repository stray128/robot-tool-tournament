"""v2 suites over parts: train_v2 (rtt.parts.suite) and heldout_procedural_v2 (rtt.parts.heldout)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from rtt.parts import heldout as H
from rtt.parts import library as lib
from rtt.parts.suite import PartSuite, load_train_spec, train_suite
from rtt.sim.conditions import heldout_suite
from rtt.sim.task import default_task

SECRET = "5" * 64  # a test secret; the real one is drawn at PREREG and never committed
OTHER = "6" * 64


@pytest.fixture(scope="module")
def suite() -> PartSuite:
    return H.heldout_procedural_suite(SECRET)


@pytest.fixture(scope="module")
def train() -> PartSuite:
    return train_suite()


# --------------------------------------------------------------------------- train_v2


def test_train_v2_has_48_episodes_per_design(train: PartSuite) -> None:
    assert len(train) == 24 and train.n_episodes == 48
    assert [p.name for p in train.parts()] == list(lib.load_parts_library().names())
    labels = {c.placement_label: c.placement for c in train.conditions[:4]}
    assert labels["c0"].err_x == 0.0 and labels["c0"].err_yaw == 0.0
    assert labels["c1"].err_x == pytest.approx(0.002)
    assert labels["c2"].err_x == pytest.approx(-0.003)
    assert labels["c3"].err_x == pytest.approx(0.001) and math.degrees(labels["c3"].err_yaw) == pytest.approx(
        6.0
    )
    assert all(
        c.err_z == 0.0 and c.placement.err_y == 0.0 for c in train
    )  # closing-axis and height errors: held-out only
    assert all(c.placement.x == c.placement.y == c.placement.yaw == 0.0 for c in train)


def test_train_v2_seeds_are_common_and_stable(train: PartSuite) -> None:
    again = train_suite()
    assert again.digest() == train.digest()
    seeds = [s for c in train for s in c.seeds]
    assert len(set(seeds)) == len(seeds)
    spec = load_train_spec()
    assert spec.base_seed == 20261018 and spec.jitter_seeds_per_condition == 2


def test_train_v2_runs_on_the_v1_evaluation_path(train: PartSuite) -> None:
    v1 = train.to_suite()
    assert len(v1) == 24
    assert v1.conditions[0].obj.geometry_sha256 == train.conditions[0].part.geometry_sha256
    assert v1.gripper_task().widest_training_object_mm == pytest.approx(44.3)
    assert v1.digest() != train.digest()


def test_train_v2_digest_follows_the_parts(tmp_path: Path) -> None:
    raw = lib.DEFAULT_PARTS_FILE.read_text()
    parts = tmp_path / "parts.yaml"
    parts.write_text(raw.replace("diameter: 40.0", "diameter: 41.0"))
    suite_file = tmp_path / "train.yaml"
    suite_file.write_text(
        (lib.REPO_ROOT / "configs/suites/train_v2.yaml")
        .read_text()
        .replace("parts_file: ../parts_v2.yaml", "parts_file: parts.yaml")
    )
    assert train_suite(suite_file).digest() != train_suite().digest()


def test_train_spec_rejects_bad_files(tmp_path: Path) -> None:
    bad = tmp_path / "t.yaml"
    bad.write_text(
        "name: t\nparts_file: " + str(lib.DEFAULT_PARTS_FILE) + "\nparts: [nope]\nplacements: {c0: {}}\n"
    )
    with pytest.raises(ValueError, match="unknown"):
        train_suite(bad)
    bad.write_text(
        "name: t\nparts_file: " + str(lib.DEFAULT_PARTS_FILE) + "\nplacements: {c0: {err_q_mm: 1}}\n"
    )
    with pytest.raises(ValueError, match="placement keys"):
        train_suite(bad)


# --------------------------------------------------------------------------- held-out


def test_heldout_is_a_pure_function_of_the_secret(suite: PartSuite) -> None:
    assert H.heldout_procedural_suite(SECRET).digest() == suite.digest()
    other = H.heldout_procedural_suite(OTHER, n=20)
    assert other.digest() != H.heldout_procedural_suite(SECRET, n=20).digest()


def test_heldout_refuses_a_wrong_or_short_secret() -> None:
    commit = hashlib.sha256(SECRET.encode()).hexdigest()
    assert len(H.heldout_procedural_suite(SECRET, commitment=commit, n=3)) == 3
    with pytest.raises(ValueError, match="does not match"):
        H.heldout_procedural_suite(OTHER, commitment=commit, n=3)
    with pytest.raises(ValueError, match="at least"):
        H.heldout_procedural_suite("short", n=3)


def test_heldout_composition(suite: PartSuite) -> None:
    summary = H.suite_summary(suite)
    assert summary["n"] == 100 and suite.n_episodes == 200
    fam = summary["families"]
    assert sum(fam[f] for f in lib.TRAINING_FAMILIES) == 70
    assert {fam[f] for f in lib.TRAINING_FAMILIES} <= {11, 12}  # stratified
    assert fam["pipe_elbow"] == 15 and fam["spur_gear"] == 15
    assert summary["modes"] == {"down": 21, "near": 28, "unseen": 30, "up": 21}
    assert summary["outside_training_dims"] >= 0.3 * summary["training_family_instances"]


def test_heldout_parts_fit_the_gripper_and_the_grip(suite: PartSuite) -> None:
    cfg = H.load_heldout_config()
    cap = cfg.mass_cap_kg(default_task())
    assert cap == pytest.approx(0.917, abs=1e-3)
    for c in suite:
        _sx, sy, sz = (1000 * s for s in c.part.size)
        assert sy <= 79.0 and 6.0 <= sz <= 100.0 and c.part.mass <= cap
        assert c.part.decomposition == "exact"
        assert c.part.name.startswith("hp2-")
        assert c.params_mm is not None


def test_heldout_placements_exceed_the_training_ranges(suite: PartSuite) -> None:
    for c in suite:
        p = c.placement
        assert math.hypot(p.err_x, p.err_y) <= 0.003 + 1e-6
        assert abs(math.degrees(p.err_yaw)) <= 8.0 + 1e-4
        assert abs(c.err_z) <= 0.0015 + 1e-9
        assert abs(p.x) <= 0.08 and abs(p.y) <= 0.08 and abs(p.yaw) <= math.pi
    assert max(abs(c.err_z) for c in suite) > 0.001
    assert max(math.hypot(c.placement.err_x, c.placement.err_y) for c in suite) > 0.0025


def test_heldout_parts_never_repeat_training_parts(suite: PartSuite, train: PartSuite) -> None:
    train_hashes = {p.geometry_sha256 for p in train.parts()}
    assert not train_hashes & {c.part.geometry_sha256 for c in suite}
    assert len({c.part.name for c in suite}) == 100


def test_heldout_unseen_families_are_well_formed(suite: PartSuite) -> None:
    from rtt.parts.features import check_features

    gears = [c for c in suite if c.part.family == "spur_gear"]
    elbows = [c for c in suite if c.part.family == "pipe_elbow"]
    for c in gears:
        assert 1.5 <= c.params_mm["module"] <= 2.0 and 16 <= c.params_mm["teeth"] <= 24
        assert c.part.yaw_symmetry == c.params_mm["teeth"]
        assert len(c.part.pieces) <= 40
    for c in gears[:2] + elbows[:2]:
        assert check_features(c.part) == []
    assert all(any(f.facing() == "left" for f in c.part.features) for c in elbows)


def test_heldout_is_domain_separated_from_v1() -> None:
    v1 = heldout_suite(SECRET)
    v2 = H.heldout_procedural_suite(SECRET, n=20)
    assert {c.condition_id.split("/")[0] for c in v1} != {c.condition_id.split("/")[0] for c in v2}
    v1_err = [round(c.placement.err_x, 6) for c in v1][:20]
    v2_err = [round(c.placement.err_x, 6) for c in v2]
    assert v1_err != v2_err


def test_heldout_config_is_validated(tmp_path: Path) -> None:
    raw = H.DEFAULT_CONFIG.read_text()
    bad = tmp_path / "h.yaml"
    bad.write_text(
        raw.replace("parts_file: ../parts_v2.yaml", f"parts_file: {lib.DEFAULT_PARTS_FILE}").replace(
            "err_z_mm: 1.5", "err_z_mm: 1.5\n  wobble: 2"
        )
    )
    with pytest.raises(ValueError, match="unknown keys in 'placement'"):
        H.load_heldout_config(bad)
    bad.write_text(raw.replace("{near: 0.4, down: 0.3, up: 0.3}", "{near: 0.5, down: 0.3, up: 0.3}"))
    with pytest.raises(ValueError, match="sum"):
        H.load_heldout_config(bad)
    bad.write_text(raw.replace("down: [0.70, 0.85]", "down: [0.70, 0.90]"))
    with pytest.raises(ValueError, match="overlap"):
        H.load_heldout_config(bad)


def test_heldout_conditions_describe_to_json(suite: PartSuite) -> None:
    d = suite.conditions[0].describe()
    json.dumps(d)
    assert set(d["placement"]) >= {"err_x", "err_y", "err_yaw_deg", "err_z"}
    assert d["part"]["geometry_sha256"] == suite.conditions[0].part.geometry_sha256
    with pytest.raises(ValueError, match="height error"):
        next(c for c in suite if c.err_z != 0).to_condition()
