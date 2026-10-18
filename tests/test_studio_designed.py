"""Designed effectors inside a Studio scene.

The Studio wrist has five servos (x, y, z, yaw, roll) where the v1 scene has four. The designed
family's per-joint pinch reads its actuators' control ranges to clamp the upright command; those
ids must come from the scene, not from a fixed offset, or the clamp reads the wrist roll.
"""

from __future__ import annotations

import numpy as np

from rtt.effectors.designed import DesignedFamily, reference_recipes, spec_from_recipe
from rtt.studio.episode import StudioCondition, StudioJob, run_task
from rtt.studio.scene import build_studio_scene
from rtt.studio.types import ObjectInstance, ObjectPose, Setup, Step, SuccessCriterion, TaskSpec


def _pick_task() -> TaskSpec:
    slug = ObjectInstance.of(
        "slug", "part", "Steel slug", ObjectPose(-80.0, -100.0, 0.0), ref="library:steel_slug", shape="mesh"
    )
    tote = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0.0, 150.0, 0.0))
    step = Step.of(
        "pick",
        "grasp",
        "slug",
        "part",
        "Pick the slug into the tote",
        [SuccessCriterion("in_target", at_least=1)],
        target="tote",
    )
    return TaskSpec("designed pinch in the Studio", "", (slug, tote), (step,), source="test")


def _upright_spec():  # a per-joint pinch hand, the only kind with upright rules
    for recipe in reference_recipes():
        spec = spec_from_recipe(recipe)
        if DesignedFamily()._layout(spec).upright_rules():
            return spec
    raise AssertionError("no reference recipe has upright rules")


def test_upright_clamp_reads_the_effectors_own_actuators_in_a_studio_scene() -> None:
    task = _pick_task()
    eff = _upright_spec()
    scene = build_studio_scene(Setup.single(eff, task.step_ids), task, 0)
    prim = DesignedFamily().primitive(eff, scene.heads[0].build)
    resolved = prim._resolve(scene.model, scene)
    acts = scene.ids.effector_actuators
    assert resolved
    for a, (_, lo, hi) in resolved.items():
        np.testing.assert_allclose((lo, hi), scene.model.actuator_ctrlrange[acts[a]])
    assert int(acts[0]) == 5  # five wrist servos in front, so a fixed "4 + a" would read the roll


def test_a_designed_pinch_hand_picks_a_part_in_the_studio() -> None:
    task = _pick_task()
    eff = _upright_spec()
    result = run_task(StudioJob(Setup.single(eff, task.step_ids), task, StudioCondition("seed0", 0)))
    (step,) = result.steps
    assert step.success, step.failure_mode
    assert step.quantities["penetration_mm"] <= 1.0
