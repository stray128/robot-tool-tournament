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


def test_a_tool_that_picks_nothing_is_built_against_the_probe_cube() -> None:
    from rtt.studio.scene import PROBE_PART_NAME, build_part_for

    hook = next(
        spec_from_recipe(r) for r in reference_recipes() if r.kind == "hook_gripper"
    )  # its 60 mm digits are too short for the 60 mm library block
    drawer = ObjectInstance.of("drawer", "drawer", "Drawer", ObjectPose(0.0, 0.0, 0.0))
    pull = Step.of(
        "pull",
        "hook_pull",
        "drawer",
        "handle",
        "Pull the drawer",
        [SuccessCriterion("travel_mm", at_least=100)],
    )
    task = TaskSpec("hook build", "", (drawer,), (pull,), source="test")
    setup = Setup.single(hook, task.step_ids)
    assert build_part_for(setup, task, 0).name == PROBE_PART_NAME
    scene = build_studio_scene(setup, task, 0)  # was invalid_build (too_short) against the block
    assert scene.active.build.mass > 0


def test_a_tool_that_picks_still_sees_its_part() -> None:
    from rtt.studio.scene import build_part_for

    task = _pick_task()
    assert build_part_for(Setup.single(_upright_spec(), task.step_ids), task, 0).name != "probe_block_10mm"


def test_every_designed_reference_tool_builds_for_a_task_it_picks_nothing_in() -> None:
    from rtt.studio.scene import PROBE_PART_NAME, build_part_for

    drawer = ObjectInstance.of("drawer", "drawer", "Drawer", ObjectPose(0.0, 0.0, 0.0))
    pull = Step.of(
        "pull",
        "hook_pull",
        "drawer",
        "handle",
        "Pull the drawer",
        [SuccessCriterion("travel_mm", at_least=100)],
    )
    task = TaskSpec("pull only", "", (drawer,), (pull,), source="test")
    seen = set()
    for recipe in reference_recipes():
        setup = Setup.single(spec_from_recipe(recipe), task.step_ids)
        seen.add(build_part_for(setup, task, 0).name)
        build_studio_scene(setup, task, 0)  # the probe if its family builds on it, else the block
    assert seen == {PROBE_PART_NAME, "block"}  # the hook gripper needs the probe; the hands refuse it
