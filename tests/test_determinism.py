"""Determinism: same seed -> identical episode; batch execution == serial execution."""

from __future__ import annotations

import math

import numpy as np

from rtt.sim.episode import run_episode, sample_jitter
from rtt.sim.exec_cpu import EpisodeJob, run_jobs
from rtt.sim.m1 import G1_BOX, G1_SETTING
from rtt.sim.objects import ObjectSpec, Placement
from rtt.sim.task import load_task
from sim_fingers import straight_finger

TASK = load_task()


def test_three_same_seed_reruns_are_identical() -> None:
    finger = straight_finger()
    runs = [
        run_episode(finger, G1_BOX, Placement(0.03, -0.02, 0.3), G1_SETTING, seed=7, task=TASK)
        for _ in range(3)
    ]
    first = runs[0]
    assert first.success
    for other in runs[1:]:
        assert other.summary() == first.summary()
        assert other.replay is not None and first.replay is not None
        assert np.array_equal(other.replay.frames, first.replay.frames)
        assert np.array_equal(other.qpos, first.qpos)


def test_recording_does_not_change_the_outcome() -> None:
    finger = straight_finger()
    rec = run_episode(finger, G1_BOX, Placement(), G1_SETTING, seed=3, record=True, task=TASK)
    bare = run_episode(finger, G1_BOX, Placement(), G1_SETTING, seed=3, record=False, task=TASK)
    rec_summary, bare_summary = rec.summary(), bare.summary()
    assert rec_summary.pop("n_steps") > bare_summary.pop("n_steps")  # the recording tail
    assert rec_summary == bare_summary
    assert bare.replay is None and bare.qpos is None


def test_jitter_is_seeded_and_bounded() -> None:
    cfg = TASK.jitter
    assert sample_jitter(11, cfg) == sample_jitter(11, cfg)
    draws = [sample_jitter(s, cfg) for s in range(500)]
    assert len({(j.dx, j.dy) for j in draws}) == len(draws)
    assert max(abs(j.dx) for j in draws) <= cfg.pos
    assert max(abs(j.dy) for j in draws) <= cfg.pos
    assert max(abs(j.dyaw) for j in draws) <= math.radians(cfg.yaw_deg)
    assert all(abs(j.mu_scale - 1.0) <= cfg.mu_frac for j in draws)
    # the bounds are actually used, not just respected
    assert max(abs(j.dx) for j in draws) > 0.9 * cfg.pos
    assert max(abs(j.mu_scale - 1.0) for j in draws) > 0.9 * cfg.mu_frac


def test_different_seeds_change_the_episode() -> None:
    finger = straight_finger()
    a = run_episode(finger, G1_BOX, Placement(), G1_SETTING, seed=1, record=False, task=TASK)
    b = run_episode(finger, G1_BOX, Placement(), G1_SETTING, seed=2, record=False, task=TASK)
    assert a.metrics["mu"] != b.metrics["mu"]
    assert a.metrics["jitter_dx"] != b.metrics["jitter_dx"]


def test_process_pool_matches_serial_execution() -> None:
    finger = straight_finger()
    jobs = [
        EpisodeJob(finger, G1_BOX, Placement(), G1_SETTING, seed=0, task=TASK),
        EpisodeJob(
            finger, ObjectSpec("cylinder", (0.025, 0.09)), Placement(0.05, 0.0, 0.2), 0, seed=1, task=TASK
        ),
        EpisodeJob(
            finger, ObjectSpec("rod", (0.015, 0.14)), Placement(-0.04, 0.03, 1.0), 2, seed=2, task=TASK
        ),
    ]
    serial = run_jobs(jobs, workers=1)
    pooled = run_jobs(jobs, workers=2)
    assert pooled.workers == 2
    assert [r.summary() for r in pooled.results] == [r.summary() for r in serial.results]
    assert pooled.total_steps == serial.total_steps > 0
    assert serial.steps_per_sec > 0 and serial.steps_per_sec_per_worker > 0
