"""MuJoCo's silent NaN resets: the bad_qacc check (task v2) and the v2 failure-mode precedence."""

from __future__ import annotations

import math

import mujoco
import pytest

from rtt.sim.checks import FLAG_BAD_QACC, FLAG_PENETRATION, CheckEvent, PhysicsChecks, bad_state_count
from rtt.sim.controller import Phase, plan_grasp_v2
from rtt.sim.episode import run_episode, run_episode_v2
from rtt.sim.fingers_v2 import v1_pair
from rtt.sim.m1 import G1_BOX, G1_SETTING
from rtt.sim.objects import Placement
from rtt.sim.perturb import Push
from rtt.sim.register import probe_first_contact
from rtt.sim.scene import build_model, build_model_v2
from rtt.sim.score import FAILURE_MODES, FAILURE_MODES_V2, LiftTracker, classify_v2
from rtt.sim.task import load_task, load_task_v2
from sim_fingers import straight_finger

TASK = load_task_v2()
V1 = load_task()
#: 1e12 N s over 20 ms: the object's acceleration is ~1e14 m/s^2, far past MuJoCo's 1e10 limit.
BLAST = Push(axis="x", impulse=1e12, duration=0.02, delay=0.2)


def _pair():
    finger = straight_finger()
    rough = v1_pair(finger, part_id="box", tip_height=0.06, q_seat=0.01)
    model, ids = build_model_v2(rough, G1_BOX, TASK)
    q = min(probe_first_contact(model, ids, plan_grasp_v2(rough, G1_BOX, Placement(), TASK), G1_BOX))
    return v1_pair(finger, part_id="box", tip_height=0.06, q_seat=q)


def test_a_mujoco_reset_makes_the_v2_episode_physics_invalid() -> None:
    res = run_episode_v2(
        _pair(), G1_BOX, Placement(), 0, record=False, task=TASK, perturbation=BLAST, check_part=False
    )
    assert res.bad_qacc >= 1 and res.metrics["bad_qacc"] >= 1
    assert not res.success and not res.lift_success
    assert res.failure_mode == "physics_invalid"
    assert any(f.startswith(f"{FLAG_BAD_QACC}@") for f in res.flags)


def test_the_same_episode_without_the_blast_is_clean() -> None:
    res = run_episode_v2(_pair(), G1_BOX, Placement(), 0, record=False, task=TASK, check_part=False)
    assert res.success and res.bad_qacc == 0 and res.metrics["bad_qacc"] == 0


def test_the_check_can_be_switched_off_in_the_task() -> None:
    task = TASK.with_overrides(checks={"bad_qacc": False})
    res = run_episode_v2(
        _pair(), G1_BOX, Placement(), 0, record=False, task=task, perturbation=BLAST, check_part=False
    )
    assert res.failure_mode != "physics_invalid" and res.bad_qacc == 0


def test_v1_episodes_never_raise_it() -> None:
    """task_v1 is frozen: its episodes keep their v1 labels (the blast reads as a v1 failure)."""
    res = run_episode(
        straight_finger(), G1_BOX, Placement(), G1_SETTING, 0, record=False, task=V1, perturbation=BLAST
    )
    assert res.failure_mode in FAILURE_MODES
    assert not any(f.startswith(FLAG_BAD_QACC) for f in res.flags)
    assert "bad_qacc" not in res.metrics


def test_bad_state_count_reads_all_reset_warnings() -> None:
    model, ids = build_model(straight_finger(), G1_BOX, V1)
    data = mujoco.MjData(model)
    assert bad_state_count(data) == 0
    data.xfrc_applied[ids.object_body, 0] = 1e30
    mujoco.mj_step(model, data)
    assert bad_state_count(data) >= 1
    checks = PhysicsChecks(model=model, ids=ids, task=V1, bad_qacc=True)
    checks.update(data, 0.002, Phase.SETTLE)
    assert checks.flags[:1] == (FLAG_BAD_QACC,) and checks.bad_qacc_count >= 1
    quiet = PhysicsChecks(model=model, ids=ids, task=V1)
    quiet.update(data, 0.002, Phase.SETTLE)
    assert FLAG_BAD_QACC not in quiet.flags


# --------------------------------------------------------------------------- precedence


def _tracker(success_at: float | None, max_lift: float = 0.15) -> LiftTracker:
    t = LiftTracker(cfg=TASK.score, z0=0.0, max_lift=max_lift)
    t.success_at = success_at
    return t


def test_precedence_physics_flags_then_v1_modes_then_registration() -> None:
    reset = CheckEvent(FLAG_BAD_QACC, 5.0, Phase.HOLD)
    late_reset = CheckEvent(FLAG_BAD_QACC, 7.0, Phase.HOLD)  # after the confirmed success: tail only
    pen_close = CheckEvent(FLAG_PENETRATION, 2.0, Phase.CLOSE)
    assert classify_v2([reset], _tracker(6.0), True, True) == (False, False, "physics_invalid")
    assert classify_v2([reset, pen_close], _tracker(None), True, False) == (False, False, "physics_invalid")
    assert classify_v2([late_reset], _tracker(6.0), True, True) == (True, True, None)
    assert classify_v2([pen_close], _tracker(None), True, False) == (False, False, "ejected_on_close")
    assert classify_v2([], _tracker(None, 0.0), False, False) == (False, False, "missed_grasp")
    assert classify_v2([], _tracker(6.0), True, False) == (False, True, "not_registered")
    assert classify_v2([], _tracker(6.0), True, True) == (True, True, None)


def test_v2_failure_modes_extend_v1() -> None:
    assert FAILURE_MODES_V2[: len(FAILURE_MODES)] == FAILURE_MODES
    assert set(FAILURE_MODES_V2) - set(FAILURE_MODES) == {
        "not_registered",
        "physics_invalid",
        "invalid_build",
    }


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_nan_poses_reset_too(value: float) -> None:
    model, ids = build_model(straight_finger(), G1_BOX, V1)
    data = mujoco.MjData(model)
    data.qvel[ids.object_dof] = value
    mujoco.mj_step(model, data)
    assert bad_state_count(data) >= 1
