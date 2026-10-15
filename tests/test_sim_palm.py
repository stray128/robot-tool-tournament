"""Palm clearance: the controller keeps the palm off the object, and palm support never counts."""

from __future__ import annotations

import numpy as np
import pytest

from rtt.design.generate import generate_finger
from rtt.design.schema import load_entrants
from rtt.sim.bench import random_valid_designs
from rtt.sim.conditions import training_suite
from rtt.sim.controller import plan_grasp
from rtt.sim.episode import run_episode
from rtt.sim.m1 import DEFAULT_ENTRANTS
from rtt.sim.objects import ObjectSpec, Placement, height
from rtt.sim.scene import finger_tip_depth
from rtt.sim.task import load_task

TASK = load_task()
ENTRANTS = {d.name: d for d in load_entrants(DEFAULT_ENTRANTS)}
STOCK = generate_finger(ENTRANTS["stock"])
TALL_BOX = ObjectSpec("box", (0.05, 0.04, 0.10))


def test_short_finger_grasps_tall_objects_higher_up() -> None:
    reach = finger_tip_depth(STOCK)
    assert reach == pytest.approx(0.0625)
    h = height(TALL_BOX)
    floor = h + TASK.controller.palm_clearance - reach  # 47.5 mm
    tips = []
    for k, frac in enumerate(TASK.controller.grasp_height_fracs):
        plan = plan_grasp(STOCK, TALL_BOX, Placement(), k, TASK)
        assert plan.reach_floor == pytest.approx(floor)
        assert plan.tip_height == pytest.approx(floor + frac * (h - floor))
        assert plan.palm_gap == pytest.approx(plan.tip_height + reach - h)
        assert plan.palm_gap >= TASK.controller.palm_clearance
        tips.append(plan.tip_height)
    assert tips == sorted(tips) and len(set(tips)) == 3  # the k settings stay distinct


@pytest.mark.parametrize(
    "design", random_valid_designs(12, 7, training_suite().gripper_task(TASK)), ids=lambda d: d.design_id()
)
def test_palm_clears_every_training_object(design) -> None:
    finger = generate_finger(design)
    for cond in training_suite():
        for k in range(len(TASK.controller.grasp_height_fracs)):
            plan = plan_grasp(finger, cond.obj, cond.placement, k, TASK)
            assert plan.palm_gap >= TASK.controller.palm_clearance - 1e-12
            assert TASK.controller.min_tip_clearance <= plan.tip_height < height(cond.obj)


def test_clear_grasp_never_touches_the_palm() -> None:
    res = run_episode(STOCK, TALL_BOX, Placement(), 0, seed=0, record=False, task=TASK)
    assert res.success, res.summary()
    assert res.metrics["palm_object_force_max"] == 0.0
    assert res.metrics["carry_palm_force_max"] == 0.0


@pytest.mark.parametrize("intrusion", [0.0005, 0.002])
def test_palm_pressing_on_the_carried_object_fails(intrusion: float) -> None:
    """Drive the palm ``intrusion`` below the box's top: the palm still touches the box
    when the lift starts, which is flagged and fails the episode (it would otherwise lift)."""
    gap = plan_grasp(STOCK, TALL_BOX, Placement(), 0, TASK).palm_gap
    task = TASK.with_overrides(controller={"grasp_z_offset": -gap - intrusion})
    res = run_episode(STOCK, TALL_BOX, Placement(), 0, seed=0, record=False, task=task)
    assert res.metrics["palm_gap"] == pytest.approx(-intrusion)
    assert not res.success
    assert res.failure_mode == "slip_during_lift"
    assert len(res.flags) == 1 and res.flags[0].startswith("palm_contact@")
    assert ":lift" in res.flags[0]
    m = res.metrics
    assert m["carry_palm_force_max"] > TASK.score.palm_contact_force_max
    assert m["palm_object_force_max"] >= m["carry_palm_force_max"]  # it pressed before LIFT as well
    assert np.isfinite(m["decided_at"])
