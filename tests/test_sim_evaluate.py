"""Batch evaluation API: records, executors, Parquet and design scores."""

from __future__ import annotations

import dataclasses

import pytest

from rtt.design.generate import generate_finger
from rtt.design.schema import load_entrants
from rtt.sim.conditions import Suite, training_suite
from rtt.sim.evaluate import (
    EpisodeRecord,
    evaluate_design,
    evaluate_designs,
    rank_designs,
    records_to_table,
    score_designs,
    write_parquet,
)
from rtt.sim.exec_cpu import CpuExecutor, run_jobs
from rtt.sim.m1 import DEFAULT_ENTRANTS
from rtt.sim.task import load_task, task_digest

TASK = load_task()
ENTRANTS = {d.name: d for d in load_entrants(DEFAULT_ENTRANTS)}
TRAIN = training_suite()
MINI = Suite(
    "mini",
    tuple(c for c in TRAIN if c.condition_id in ("train_v1/box-large/sloppy", "train_v1/rod-peg/handle20")),
)


@pytest.fixture(scope="module")
def serial() -> list[EpisodeRecord]:
    return evaluate_designs(
        [ENTRANTS["stock"], ENTRANTS["straight"]], MINI, controller_settings=[0, 2], workers=1, task=TASK
    )


def test_records_cover_designs_conditions_settings_in_order(serial: list[EpisodeRecord]) -> None:
    assert len(serial) == 2 * 2 * 2
    keys = [(r.design_id, r.condition_id, r.k) for r in serial]
    expected = [
        (d.design_id(), c.condition_id, k)
        for d in (ENTRANTS["stock"], ENTRANTS["straight"])
        for c in MINI
        for k in (0, 2)
    ]
    assert keys == expected
    by_cond = {c.condition_id: c for c in MINI}
    for r in serial:
        cond = by_cond[r.condition_id]
        assert r.seed == cond.seeds[0]
        assert r.suite == "mini" and r.family == cond.family and r.object_kind == cond.obj.kind
        assert r.placement_label == cond.placement_label and r.size_label == cond.size_label
        assert r.grasp_frac == TASK.controller.grasp_height_fracs[r.k]
        assert r.task == "task_v1" and r.task_digest == task_digest(TASK) and r.executor == "cpu"
        assert isinstance(r.success, bool) and isinstance(r.flags, tuple)
        assert (r.failure_mode is None) == r.success
        assert 0.0 <= r.hold_frac <= 1.0
        assert r.palm_gap >= TASK.controller.palm_clearance - 1e-12
        assert r.n_steps > 0 and r.wall_time_s > 0
    stock = [r for r in serial if r.design_id == ENTRANTS["stock"].design_id()]
    assert stock[0].finger_mass_g == pytest.approx(10.85, abs=0.01)
    d = serial[0].to_dict()
    assert isinstance(d["flags"], list) and d["record_version"] == 1


def test_process_pool_and_open_executor_match_serial(serial: list[EpisodeRecord]) -> None:
    designs = [ENTRANTS["stock"], ENTRANTS["straight"]]
    pooled = evaluate_designs(designs, MINI, controller_settings=[0, 2], workers=2, task=TASK)
    assert pooled == serial  # wall_time_s is excluded from comparison
    with CpuExecutor(2) as ex:
        reused = evaluate_designs(designs, MINI, controller_settings=[0, 2], executor=ex, task=TASK)
    assert reused == serial
    custom = evaluate_designs(
        designs, MINI, controller_settings=[0, 2], executor=run_jobs, workers=1, task=TASK
    )
    assert [dataclasses.replace(r, executor="cpu") for r in custom] == serial
    assert custom[0].executor == "run_jobs"


def test_seed_override_and_raw_geometry() -> None:
    geom = generate_finger(ENTRANTS["stock"])
    recs = evaluate_design(geom, MINI, controller_settings=[1], seeds=[11, 12], workers=1, task=TASK)
    assert [r.seed for r in recs] == [11, 12, 11, 12]
    assert all(r.design_id.startswith("g") and len(r.design_id) == 12 for r in recs)
    assert len({r.design_id for r in recs}) == 1


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="controller settings"):
        evaluate_design(ENTRANTS["stock"], MINI, controller_settings=[3], task=TASK)
    with pytest.raises(ValueError, match="unknown executor"):
        evaluate_design(ENTRANTS["stock"], MINI, executor="tpu", task=TASK)
    with pytest.raises(TypeError):
        evaluate_design({"finger_length_mm": 60}, MINI, task=TASK)  # type: ignore[arg-type]


def test_parquet_round_trip(serial: list[EpisodeRecord], tmp_path) -> None:
    import pyarrow.parquet as pq

    table = records_to_table(serial)
    assert table.num_rows == len(serial)
    assert table.column_names[:10] == [
        "design_id",
        "condition_id",
        "k",
        "seed",
        "success",
        "failure_mode",
        "lift",
        "hold",
        "flags",
        "hold_frac",
    ]
    back = pq.read_table(write_parquet(serial, tmp_path / "eval.parquet")).to_pylist()
    for row, rec in zip(back, serial, strict=True):
        assert row == rec.to_dict()


def _record(design: str, k: int, success: bool, hold: float = 0.0, mass: float = 10.0, **kw) -> EpisodeRecord:
    base: dict[str, object] = {
        "design_id": design,
        "condition_id": "c",
        "k": k,
        "seed": 0,
        "success": success,
        "failure_mode": None if success else "missed_grasp",
        "lift": 0.0,
        "hold": hold,
        "flags": (),
        "hold_frac": 1.0 if success else hold / 2.0,
        "grasp_frac": 0.5,
        "suite": "s",
        "family": "box",
        "object_kind": "box",
        "size_label": "",
        "placement_label": "",
        "finger_mass_g": mass,
        "tip_height": 0.0,
        "palm_gap": 0.01,
        "penetration_max": 0.0,
        "decided_at": 1.0,
        "n_steps": 1,
        "task": "t",
        "task_digest": "d",
        "executor": "cpu",
        "wall_time_s": 0.0,
    }
    base.update(kw)
    return EpisodeRecord(**base)  # type: ignore[arg-type]


def test_score_is_the_best_setting_with_tie_breaks() -> None:
    records = [
        # design a: k0 1/2, k1 2/2 -> score 1.0 at k1
        _record("a", 0, True), _record("a", 0, False), _record("a", 1, True), _record("a", 1, True),
        # design b: k0 and k1 tie on success (1/2); k1 holds longer before failing -> k1
        _record("b", 0, True), _record("b", 0, False, hold=0.2),
        _record("b", 1, True), _record("b", 1, False, hold=1.0, family="rod"),
        # design c: same as a but heavier -> ranked after a
        _record("c", 0, True, mass=20.0), _record("c", 1, True, mass=20.0),
    ]  # fmt: skip
    scores = score_designs(records)
    assert scores["a"].score == 1.0 and scores["a"].best_k == 1
    assert scores["a"].success_by_k == {0: 0.5, 1: 1.0}
    assert scores["b"].score == 0.5 and scores["b"].best_k == 1
    assert scores["b"].success_by_family == {"box": 1.0, "rod": 0.0}
    assert scores["b"].failure_modes == {"missed_grasp": 1}
    assert [s.design_id for s in rank_designs(records)] == ["a", "c", "b"]
