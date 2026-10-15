"""Gate G1 (Milestone 1 physics) plus unit checks of the scene construction.

The gate cases run on fingers from the real design generator (the manual entrants);
the scene and contact unit checks use the hand-built ``sim_fingers.straight_finger``.
"""

from __future__ import annotations

import dataclasses
import math

import mujoco
import numpy as np
import pytest

from rtt.design.types import BOXES_PER_FINGER, Box, FingerGeometry
from rtt.sim.controller import Phase, plan_grasp
from rtt.sim.episode import EpisodeResult, run_episode
from rtt.sim.m1 import G1_BOX, G1_SETTING, GateCase, g1_cases, run_case
from rtt.sim.objects import ObjectSpec, Placement, height
from rtt.sim.render import render_mp4
from rtt.sim.replay import read_replay, write_replay
from rtt.sim.scene import (
    REPLAY_BODIES,
    box_corners,
    build_model,
    finger_mass_properties,
    finger_tip_depth,
    mirror_box,
    open_gap,
    quat_to_mat,
)
from rtt.sim.task import DEFAULT_TASK_PATH, TaskConfig, load_task
from sim_fingers import straight_finger

TASK = load_task()
# 60 mm across the jaw: wider than a 20 deg inward-bent finger's 53 mm tip opening.
WIDE_TOP_BOX = ObjectSpec("box", (0.06, 0.06, 0.12))


@pytest.fixture(scope="module")
def cases() -> dict[str, GateCase]:
    return {c.name: c for c in g1_cases(TASK)}


@pytest.fixture(scope="module")
def nominal(cases: dict[str, GateCase]) -> EpisodeResult:
    return run_case(cases["straight"], seed=0, record=True)


def _bent_finger() -> FingerGeometry:
    """Straight finger with its lower half rotated 20 deg about X (exercises box quaternions)."""
    base = straight_finger()
    theta = math.radians(-20.0)
    quat = (math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0)
    rot = quat_to_mat(quat)
    pivot = np.array([0.0, 0.0, -0.04])
    boxes = []
    for i, b in enumerate(base.boxes):
        if 6 <= i < 12:
            c = pivot + rot @ (np.asarray(b.center) - pivot)
            boxes.append(Box(tuple(c), b.half_size, quat))
        else:
            boxes.append(b)
    return FingerGeometry(tuple(boxes))


def _concave_finger() -> FingerGeometry:
    """Flat-backed finger with a 4 mm channel between two 4 mm lips (12 x 3 boxes + 2 parked)."""
    boxes = []
    for s in range(4):
        cz = -0.01 - 0.02 * s
        boxes.append(Box((-0.008, 0.002, cz), (0.002, 0.002, 0.01)))
        boxes.append(Box((0.0, 0.007, cz), (0.01, 0.003, 0.01)))
        boxes.append(Box((0.008, 0.002, cz), (0.002, 0.002, 0.01)))
    boxes += [Box((0.0, 0.007, -0.01), (1e-4, 1e-4, 1e-4))] * 2
    return FingerGeometry(tuple(boxes))


# --------------------------------------------------------------------------- configuration and scene


def test_task_yaml_matches_dataclass_defaults() -> None:
    assert DEFAULT_TASK_PATH.exists()
    assert load_task() == TaskConfig()


def test_unknown_yaml_keys_are_rejected() -> None:
    from rtt.sim.task import task_from_dict

    with pytest.raises(ValueError, match="unknown keys"):
        task_from_dict({"physics": {"timestepp": 0.001}})


def test_fixture_finger_follows_contract() -> None:
    finger = straight_finger()
    assert len(finger.boxes) == BOXES_PER_FINGER
    assert finger_tip_depth(finger) == pytest.approx(0.08)
    assert open_gap(finger, TASK) == pytest.approx(0.08)  # inner faces 80 mm apart when open


def test_mirror_box_reflects_every_corner() -> None:
    q = np.array([0.9, 0.2, -0.3, 0.25])
    q /= np.linalg.norm(q)
    box = Box(center=(0.01, 0.02, -0.03), half_size=(0.004, 0.006, 0.01), quat_wxyz=tuple(q))
    mirrored = box_corners(mirror_box(box))
    expected = box_corners(box) * np.array([1.0, -1.0, 1.0])
    # same set of 8 points (order may differ)
    d = np.linalg.norm(mirrored[:, None, :] - expected[None, :, :], axis=-1)
    assert np.allclose(d.min(axis=1), 0.0, atol=1e-12)
    assert np.allclose(d.min(axis=0), 0.0, atol=1e-12)


def test_finger_mass_properties_match_mujoco_geom_inertia() -> None:
    finger = _bent_finger()
    props = finger_mass_properties(finger)
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="f")
    body.add_freejoint()
    for b in finger.boxes:
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(b.half_size),
            pos=list(b.center),
            quat=list(b.quat_wxyz),
            density=finger.density_kg_m3,
        )
    model = spec.compile()
    assert model.body_mass[1] == pytest.approx(props.mass, rel=1e-9)
    assert np.allclose(model.body_ipos[1], props.com, atol=1e-12)
    rot = quat_to_mat(model.body_iquat[1])
    full = rot @ np.diag(model.body_inertia[1]) @ rot.T
    assert np.allclose(full, props.inertia, rtol=1e-6, atol=1e-12)
    # PLA straight finger: 20 x 10 x 80 mm (+ two tiny parked boxes)
    assert finger_mass_properties(straight_finger()).mass == pytest.approx(16e-6 * 1240, rel=1e-3)


def test_scene_structure() -> None:
    finger = _bent_finger()
    model, ids = build_model(finger, G1_BOX, TASK)

    def act(name: str) -> np.ndarray:
        return model.actuator_forcerange[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)]

    assert act("palm_z").tolist() == [-150.0, 150.0]
    assert act("palm_yaw").tolist() == [-15.0, 15.0]
    assert act("grip").tolist() == [-40.0, 40.0]
    for name in ("finger_left", "finger_right"):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert model.jnt_range[j].tolist() == pytest.approx([0.0, 0.04])
    assert model.neq == 1 and model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert model.nmocap == 0
    # 14 boxes per finger, right finger mirrored, explicit PLA inertia on both
    left = model.geom_bodyid == ids.finger_left_body
    right = model.geom_bodyid == ids.finger_right_body
    assert left.sum() == right.sum() == BOXES_PER_FINGER
    assert np.allclose(model.geom_pos[right], model.geom_pos[left] * [1, -1, 1])
    props = finger_mass_properties(finger)
    assert model.body_mass[ids.finger_left_body] == pytest.approx(props.mass)
    assert model.body_mass[ids.finger_right_body] == pytest.approx(props.mass)
    # object density 700 kg/m^3: the 90 x 40 x 120 mm G1 box weighs 0.30 kg
    assert model.body_mass[ids.object_body] == pytest.approx(math.prod(G1_BOX.size) * 700)
    assert model.body_mass[ids.object_body] == pytest.approx(0.3024)
    # physics options
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == pytest.approx(10.0)
    assert set(model.geom_condim.tolist()) == {4}


def test_left_and_right_fingers_are_mirror_images_in_the_world() -> None:
    finger = _bent_finger()
    model, ids = build_model(finger, G1_BOX, TASK)
    data = mujoco.MjData(model)
    data.qpos[ids.palm_qpos] = [0.0, 0.0, 0.3, 0.0]
    data.qpos[ids.finger_qpos] = [0.01, 0.01]
    mujoco.mj_forward(model, data)
    left = model.geom_bodyid == ids.finger_left_body
    right = model.geom_bodyid == ids.finger_right_body
    assert np.allclose(data.geom_xpos[right], data.geom_xpos[left] * [1, -1, 1], atol=1e-12)
    flip = np.diag([1.0, -1.0, 1.0])
    for gl, gr in zip(np.flatnonzero(left), np.flatnonzero(right), strict=True):
        rl = data.geom_xmat[gl].reshape(3, 3)
        rr = data.geom_xmat[gr].reshape(3, 3)
        assert np.allclose(rr, flip @ rl @ flip, atol=1e-12)


def test_grasp_plan_places_finger_tips_at_k_times_object_height() -> None:
    finger = straight_finger()
    obj = ObjectSpec("cylinder", (0.025, 0.09))
    for k, frac in enumerate(TASK.controller.grasp_height_fracs):
        plan = plan_grasp(finger, obj, Placement(0.05, -0.02, 0.4), k, TASK)
        assert plan.tip_height == pytest.approx(frac * height(obj))
        palm_to_tip = TASK.gripper.palm_half_size[2] + finger_tip_depth(finger)
        assert plan.grasp_z - palm_to_tip == pytest.approx(frac * height(obj))
        # approach at least 8 cm above the grasp and clear of the object's top
        assert plan.approach_z - plan.grasp_z >= TASK.controller.approach_offset - 1e-12
        assert plan.approach_z - palm_to_tip >= height(obj) + TASK.controller.approach_clearance - 1e-12
        assert plan.lift_z - plan.grasp_z == pytest.approx(0.15)
    with pytest.raises(ValueError):
        plan_grasp(finger, obj, Placement(), 3, TASK)


def test_placement_reach_and_yaw_wrapping() -> None:
    finger = straight_finger()
    with pytest.raises(ValueError, match="reach"):
        plan_grasp(finger, G1_BOX, Placement(0.5, 0.0, 0.0), G1_SETTING, TASK)
    plan = plan_grasp(finger, G1_BOX, Placement(0.0, 0.0, 0.3 + 4 * math.pi), G1_SETTING, TASK)
    assert plan.yaw == pytest.approx(0.3)
    # a box that only fits the jaw across its object-frame Y axis, turned by 2.5 turns
    long_x = ObjectSpec("box", (0.10, 0.04, 0.10))
    res = run_episode(
        finger, long_x, Placement(0.03, 0.02, 5 * math.pi), G1_SETTING, 0, record=False, task=TASK
    )
    assert res.success, res.summary()


def test_millimetre_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="metres"):
        ObjectSpec("box", (60.0, 60.0, 120.0))
    mm = straight_finger(length=80.0, thickness=10.0, width=20.0)
    with pytest.raises(ValueError, match="metres"):
        build_model(mm, G1_BOX, TASK)


# --------------------------------------------------------------------------- gate G1


@pytest.mark.parametrize("entrant", ["straight", "concave", "hooked"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_g1_entrant_box_grasp_succeeds(cases: dict[str, GateCase], entrant: str, seed: int) -> None:
    res = run_case(cases[entrant], seed=seed)
    assert res.success, res.summary()
    assert res.failure_mode is None
    assert res.flags == ()
    assert res.max_lift >= TASK.score.success_lift
    assert res.hold_time >= TASK.score.success_hold
    assert res.metrics["penetration_max"] < 0.001
    assert res.metrics["object_speed_max"] < 0.5


@pytest.mark.parametrize(
    ("name", "mode"),
    [("too_high", "missed_grasp"), ("low_friction", "slip_during_lift"), ("too_wide", "missed_grasp")],
)
def test_g1_deliberate_failures_fail(cases: dict[str, GateCase], name: str, mode: str) -> None:
    case = cases[name]
    assert not case.expect_success
    res = run_case(case, seed=0)
    assert not res.success, res.summary()
    assert res.failure_mode == mode
    assert res.max_lift < TASK.score.success_lift


def test_g1_cases_are_what_they_claim(cases: dict[str, GateCase]) -> None:
    assert list(cases) == ["straight", "concave", "hooked", "too_high", "low_friction", "too_wide"]
    for name in ("straight", "concave", "hooked"):
        assert cases[name].expect_success and cases[name].obj == G1_BOX
        # every entrant opens wide enough for the box, with the validator's 5 mm margin
        assert open_gap(cases[name].finger, TASK) >= G1_BOX.size[1] + 0.005
    high = cases["too_high"]
    plan = plan_grasp(high.finger, G1_BOX, Placement(), G1_SETTING, high.task)
    assert plan.tip_height == pytest.approx(height(G1_BOX) + 0.03)
    low = cases["low_friction"]
    assert low.task.physics.friction == pytest.approx(0.05)
    # even with +10 % friction jitter, 2 * mu * F/2 stays below the box's weight
    capacity = 2 * 0.05 * 1.1 * TASK.gripper.grip_force / 2
    assert capacity < math.prod(G1_BOX.size) * 700 * 9.81
    wide = cases["too_wide"]
    assert sorted(wide.obj.size) == sorted(G1_BOX.size)  # the same box, turned
    assert wide.obj.size[1] > open_gap(wide.finger, TASK)
    assert {c.entrant for c in (high, low, wide)} == {"straight"}


def test_fingers_landing_on_the_object_top_are_a_missed_grasp() -> None:
    """Regression: tilted tips resting on a too-wide box read as a sideways squeeze and the
    failure was labelled ``slip_during_lift``; pushing the object down is not a grasp."""
    finger = _bent_finger()  # tips bent 20 deg inward: a 53 mm opening at the tips
    assert open_gap(finger, TASK) < WIDE_TOP_BOX.size[1]
    res = run_case(dataclasses.replace(g1_cases(TASK)[0], obj=WIDE_TOP_BOX), seed=1, finger=finger)
    assert not res.success
    assert res.failure_mode == "missed_grasp", res.summary()
    m = res.metrics
    assert m["grasped"] == 0.0
    # the tilted tip faces do push sideways, but they push down much harder
    assert min(m["squeeze_left"], m["squeeze_right"]) > 5.0
    assert m["press_down_left"] > m["squeeze_left"] and m["press_down_right"] > m["squeeze_right"]


def test_cylinder_on_concave_lips_gets_radial_contact_normals() -> None:
    """Regression: native CCD + multi-CCD produced sideways normals and phantom wedge forces here."""
    cyl = ObjectSpec("cylinder", (0.025, 0.09))
    res = run_case(dataclasses.replace(g1_cases(TASK)[0], obj=cyl), seed=0, finger=_concave_finger())
    assert res.success, res.summary()
    half_f = TASK.gripper.grip_force / 2
    assert res.metrics["squeeze_left"] == pytest.approx(half_f, rel=0.2)
    assert res.metrics["squeeze_right"] == pytest.approx(half_f, rel=0.2)
    assert res.metrics["penetration_max"] < 0.001


def test_object_rests_on_table_not_on_finger(nominal: EpisodeResult) -> None:
    m = nominal.metrics
    # after settling and again with the open fingers lowered around it, the
    # table carries the object's full weight and no finger touches it
    for prefix in ("rest", "preclose"):
        assert m[f"{prefix}_table_support"] == pytest.approx(1.0, abs=0.02)
        assert m[f"{prefix}_finger_force"] == pytest.approx(0.0, abs=1e-6)
    assert m["grasped"] == 1.0
    half_f = TASK.gripper.grip_force / 2
    assert m["squeeze_left"] == pytest.approx(half_f, rel=0.25)
    assert m["squeeze_right"] == pytest.approx(half_f, rel=0.25)
    # flat vertical pads pinch from the side; they do not push the box down
    assert max(m["press_down_left"], m["press_down_right"]) < 0.05 * half_f
    assert m["gripper_env_force_max"] == 0.0


def test_nominal_replay_round_trip(nominal: EpisodeResult, tmp_path) -> None:
    replay = nominal.replay
    assert replay is not None and nominal.qpos is not None
    frames = replay.frames
    assert frames.dtype == np.float32
    assert frames.shape == (nominal.qpos.shape[0], len(REPLAY_BODIES), 7)
    assert np.allclose(np.linalg.norm(frames[:, :, 3:], axis=-1), 1.0, atol=1e-5)
    # 30 fps over the decided episode plus the recording tail
    expected = (nominal.metrics["decided_at"] + TASK.score.record_tail) * TASK.replay.fps
    assert abs(frames.shape[0] - expected) <= 2
    # object starts at rest height (box centre at h/2) and ends lifted
    obj = REPLAY_BODIES.index("object")
    assert frames[0, obj, 2] == pytest.approx(0.06, abs=1e-3)
    assert frames[-1, obj, 2] - frames[0, obj, 2] >= TASK.score.success_lift
    # quaternions are xyzw: the replay's first object quaternion equals the qpos wxyz one reordered
    q_wxyz = nominal.qpos[0, 9:13]
    assert np.allclose(frames[0, obj, 3:], np.r_[q_wxyz[1:], q_wxyz[0]], atol=1e-6)

    bin_path, _ = write_replay(replay, tmp_path / "ep")
    assert bin_path.stat().st_size == frames.size * 4
    back = read_replay(tmp_path / "ep")
    assert np.array_equal(back.frames, frames)
    header = back.header
    assert header["fps"] == 30 and header["pose_fields"] == ["px", "py", "pz", "qx", "qy", "qz", "qw"]
    assert [b["name"] for b in header["bodies"]] == list(REPLAY_BODIES)
    geoms = {b["name"]: b["geoms"] for b in header["bodies"]}
    assert len(geoms["finger_left"]) == len(geoms["finger_right"]) == BOXES_PER_FINGER
    assert geoms["object"][0]["type"] == "box"
    assert {g["name"] for g in header["static"]} == {"table", "floor"}
    assert header["meta"]["success"] is True


def test_replay_header_describes_cylinders_and_rods(tmp_path) -> None:
    for obj, gtype in (
        (ObjectSpec("cylinder", (0.02, 0.07)), "cylinder"),
        (ObjectSpec("rod", (0.012, 0.1)), "capsule"),
    ):
        res = run_case(dataclasses.replace(g1_cases(TASK)[0], obj=obj), seed=0, record=True)
        assert res.replay is not None
        geom = res.replay.header["bodies"][-1]["geoms"][0]
        assert geom["type"] == gtype
        assert geom["radius"] == pytest.approx(obj.size[0])


def test_render_mp4_smoke(nominal: EpisodeResult, tmp_path) -> None:
    assert nominal.qpos is not None
    try:
        path = render_mp4(
            g1_cases(TASK)[0].finger,
            G1_BOX,
            TASK,
            nominal.qpos[::20],
            tmp_path / "clip.mp4",
            width=320,
            height=240,
        )
    except RuntimeError as exc:  # no EGL / OSMesa on this machine
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    assert path.exists() and path.stat().st_size > 1000


def test_phases_are_ordered() -> None:
    plan = plan_grasp(straight_finger(), G1_BOX, Placement(), G1_SETTING, TASK)
    assert 0 < plan.t_descend < plan.t_close < plan.t_lift < plan.t_hold < TASK.score.max_time
    assert list(Phase) == sorted(Phase)
