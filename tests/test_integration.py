"""Design generator -> simulator integration: both modules must agree on every finger."""

from __future__ import annotations

import numpy as np
import pytest

from rtt.design.generate import generate_finger, mass_and_inertia
from rtt.design.schema import CATEGORICAL_OPTIONS, PARAM_BOUNDS, FingerDesign, GripperTask, load_entrants
from rtt.design.types import BOXES_PER_FINGER
from rtt.design.validate import validate
from rtt.sim.episode import run_episode
from rtt.sim.m1 import DEFAULT_ENTRANTS
from rtt.sim.objects import ObjectSpec, Placement
from rtt.sim.scene import build_model, finger_mass_properties, finger_tip_depth, open_gap
from rtt.sim.score import FAILURE_MODES
from rtt.sim.task import gripper_task_for, load_task

TASK = load_task()
GRIPPER = gripper_task_for(TASK)


def _random_valid_designs(n: int, seed: int) -> list[FingerDesign]:
    """Uniform random designs over the schema, kept when valid for the simulator's gripper."""
    rng = np.random.default_rng(seed)
    designs: list[FingerDesign] = []
    while len(designs) < n:
        params: dict[str, object] = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PARAM_BOUNDS.items()}
        params.update({k: str(rng.choice(opts)) for k, opts in CATEGORICAL_OPTIONS.items()})
        if params["inner_face"] == "flat":
            params["inner_face_depth_mm"] = 0.0
        design = FingerDesign.model_validate(params)
        if validate(design, GRIPPER).ok:
            designs.append(design)
    return designs


ENTRANTS = load_entrants(DEFAULT_ENTRANTS)
DESIGNS = ENTRANTS + _random_valid_designs(40, seed=20261015)


def test_validator_gripper_is_derived_from_the_sim_task() -> None:
    # The validator's defaults describe the same mount as configs/task_v1.yaml.
    assert GRIPPER == GripperTask()
    assert GRIPPER.max_opening_mm == pytest.approx(2000.0 * TASK.gripper.open_half_gap)
    custom = gripper_task_for(TASK, widest_training_object_mm=40.0)
    assert custom.widest_training_object_mm == 40.0
    with pytest.raises(ValueError, match="simulator task"):
        gripper_task_for(TASK, stroke_mm=42.5)
    wider = TASK.with_overrides(gripper={"stroke": 0.0425, "open_half_gap": 0.0425})
    assert gripper_task_for(wider).max_opening_mm == pytest.approx(85.0)


@pytest.mark.parametrize("design", ENTRANTS, ids=lambda d: d.name)
def test_entrants_are_valid_for_the_simulator_gripper(design: FingerDesign) -> None:
    result = validate(design, GRIPPER)
    assert result.ok, result.summary()


@pytest.mark.parametrize("design", DESIGNS, ids=lambda d: d.name or d.design_id())
def test_design_and_sim_agree_on_generated_fingers(design: FingerDesign) -> None:
    geom = generate_finger(design)
    metrics = validate(design, GRIPPER).metrics
    # open gap, reach and mass are computed independently by the two modules
    assert 1000.0 * open_gap(geom, TASK) == pytest.approx(metrics["open_gap_mm"], abs=1e-6)
    assert 1000.0 * finger_tip_depth(geom) == pytest.approx(metrics["max_reach_z_mm"], abs=1e-6)
    sim_props = finger_mass_properties(geom)
    design_props = mass_and_inertia(geom)
    assert sim_props.mass == pytest.approx(design_props.mass_kg, rel=1e-12)
    assert np.allclose(sim_props.com, design_props.com_m, atol=1e-12)
    assert np.allclose(sim_props.inertia, design_props.inertia_com, rtol=1e-9, atol=1e-15)

    model, ids = build_model(geom, ObjectSpec("cylinder", (0.015, 0.08)), TASK)
    for body in (ids.finger_left_body, ids.finger_right_body):
        assert (model.geom_bodyid == body).sum() == BOXES_PER_FINGER
        assert model.body_mass[body] == pytest.approx(design_props.mass_kg)


@pytest.mark.parametrize("design", DESIGNS[:8], ids=lambda d: d.name or d.design_id())
def test_generated_fingers_run_full_episodes(design: FingerDesign) -> None:
    geom = generate_finger(design)
    res = run_episode(
        geom, ObjectSpec("cylinder", (0.015, 0.08)), Placement(), 1, seed=0, record=False, task=TASK
    )
    assert res.n_steps > 0
    assert res.success or res.failure_mode in FAILURE_MODES
    # the object starts resting on the table, untouched by the open fingers
    assert res.metrics["rest_table_support"] == pytest.approx(1.0, abs=0.02)
    assert res.metrics["rest_finger_force"] == pytest.approx(0.0, abs=1e-6)
