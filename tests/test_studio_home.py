"""The wrist's home pose clears every object (CR-S2b-2)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rtt.effectors.registry import get_family
from rtt.sim.task import default_task_v2
from rtt.studio.episode import HOME_CLEARANCE, HOME_HEIGHT, home_pose
from rtt.studio.scene import build_studio_scene
from rtt.studio.types import ObjectInstance, ObjectPose, Setup, Step, SuccessCriterion, TaskSpec


def _tool(model_name: str):  # a catalog preset by model name
    for fam in ("catalog_gripper", "catalog_hand"):
        for s in get_family(fam).presets():
            if s.model == model_name:
                return s
    raise LookupError(model_name)


def _door_task(**params: object) -> TaskSpec:
    door = ObjectInstance.of(
        "door", "hinged_lid", "Door", ObjectPose(0.0, 0.0, 0.0), orientation="door", **params
    )
    pull = Step.of(
        "open", "hook_pull", "door", "handle", "Open the door", [SuccessCriterion("angle_deg", at_least=60)]
    )
    return TaskSpec("door at the table centre", "", (door,), (pull,), source="test")


def _gap_at(scene, home: np.ndarray) -> float:
    """The smallest distance between the tool stack and any object with the wrist at ``home``."""
    model, data = scene.model, mujoco.MjData(scene.model)
    data.qpos[scene.ids.palm_qpos] = home
    mujoco.mj_forward(model, data)
    root = model.body_rootid[model.geom_bodyid]
    wrist = model.body_rootid[scene.ids.palm_body]
    tool = np.flatnonzero(root == wrist)
    objects = np.flatnonzero((root != wrist) & (model.geom_bodyid != 0))
    fromto = np.zeros(6)
    return min(
        mujoco.mj_geomDistance(model, data, int(a), int(b), 1.0, fromto) for a in tool for b in objects
    )


@pytest.mark.parametrize("tool", ["adaptive-2f", "gripper-2f-finray", "hand-5f-22dof"])
def test_home_clears_a_tall_door_at_the_table_centre(tool: str) -> None:
    task = default_task_v2()
    spec = _door_task(feature="handle")
    scene = build_studio_scene(Setup.single(_tool(tool), spec.step_ids), spec, 0, task=task)
    home = home_pose(scene, task)
    base = np.array([0.0, 0.0, task.physics.table_top_z + HOME_HEIGHT, 0.0])
    assert home[2] >= base[2]
    assert _gap_at(scene, home) >= HOME_CLEARANCE - 0.005  # boxes are conservative; geoms are closer to exact


def test_home_stays_at_its_height_when_nothing_reaches_the_tool() -> None:
    task = default_task_v2()
    part = ObjectInstance.of(
        "slug", "part", "Slug", ObjectPose(-80.0, -100.0, 0.0), ref="library:steel_slug", shape="mesh"
    )
    pick = Step.of("pick", "grasp", "slug", "part", "Pick it", [SuccessCriterion("lift_mm", at_least=10)])
    spec = TaskSpec("low part", "", (part,), (pick,), source="test")
    scene = build_studio_scene(Setup.single(_tool("adaptive-2f"), spec.step_ids), spec, 0, task=task)
    assert home_pose(scene, task)[2] == pytest.approx(task.physics.table_top_z + HOME_HEIGHT)
