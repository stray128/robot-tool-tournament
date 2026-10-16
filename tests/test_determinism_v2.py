"""Determinism of the v2 simulator path (gate GV2): identical trajectory hashes in-process,
across fresh processes and with or without recording."""

from __future__ import annotations

from freeform_fixtures import (
    block,
    build_fixture_pair,
    conform,
    determinism_designs,
    hex_standoff,
    pipe_tee,
    training_parts,
)
from rtt.sim.episode import run_episode_v2
from rtt.sim.exec_cpu import EpisodeJob, run_jobs
from rtt.sim.gates_v2 import determinism_jobs, fresh_process_hashes
from rtt.sim.objects import Placement
from rtt.sim.task import load_task_v2

TASK = load_task_v2()
TEE = pipe_tee()
TEE_PAIR = build_fixture_pair(conform(), TEE, TASK)


def test_the_trajectory_hash_repeats_in_process_and_depends_on_the_seed() -> None:
    a = run_episode_v2(TEE_PAIR, TEE, Placement(err_x=0.002), 3, record=False, task=TASK)
    b = run_episode_v2(TEE_PAIR, TEE, Placement(err_x=0.002), 3, record=False, task=TASK)
    c = run_episode_v2(TEE_PAIR, TEE, Placement(err_x=0.002), 4, record=False, task=TASK)
    assert len(a.traj_sha) == 64
    assert a.traj_sha == b.traj_sha and a.summary() == b.summary()
    assert a.traj_sha != c.traj_sha


def test_recording_does_not_change_the_trajectory() -> None:
    rec = run_episode_v2(TEE_PAIR, TEE, Placement(), 1, record=True, task=TASK)
    bare = run_episode_v2(TEE_PAIR, TEE, Placement(), 1, record=False, task=TASK)
    assert rec.traj_sha == bare.traj_sha


def test_rebuilding_the_pair_gives_the_same_geometry_and_trajectory() -> None:
    again = build_fixture_pair(conform(), TEE, TASK)
    assert again.geometry_hash == TEE_PAIR.geometry_hash and again.design_id == TEE_PAIR.design_id
    a = run_episode_v2(again, TEE, Placement(), 2, record=False, task=TASK)
    b = run_episode_v2(TEE_PAIR, TEE, Placement(), 2, record=False, task=TASK)
    assert a.traj_sha == b.traj_sha


def test_the_trajectory_hash_can_be_switched_off() -> None:
    task = TASK.with_overrides(checks={"trajectory_hash": False})
    res = run_episode_v2(TEE_PAIR, TEE, Placement(), 0, record=False, task=task)
    assert res.traj_sha == ""


def test_pool_workers_match_serial_hashes() -> None:
    blk, hexs = block(), hex_standoff()
    jobs = [
        EpisodeJob(build_fixture_pair(conform(), blk, TASK), blk, Placement(err_x=-0.003), -1, 7, task=TASK),
        EpisodeJob(build_fixture_pair(conform(), hexs, TASK), hexs, Placement(), -1, 8, task=TASK),
        EpisodeJob(TEE_PAIR, TEE, Placement(err_x=0.001, err_yaw=0.1), -1, 9, task=TASK),
    ]
    serial = [r.traj_sha for r in run_jobs(jobs, workers=1).results]
    pooled = [r.traj_sha for r in run_jobs(jobs, workers=3).results]
    assert serial == pooled and all(serial)


def test_ten_designs_give_one_episode_each() -> None:
    labels, jobs = determinism_jobs(build_fixture_pair, training_parts(), determinism_designs(), TASK)
    assert len(jobs) == len(determinism_designs()) == 10
    assert len(set(labels)) == 10


def test_fresh_processes_reproduce_the_hashes() -> None:
    _, jobs = determinism_jobs(build_fixture_pair, training_parts(), determinism_designs(), TASK)
    out = fresh_process_hashes(jobs[:3], processes=2)
    assert out["identical"], out
    assert out["mismatches"] == [] and len(out["hashes"]) == 3
