"""The v2 scene path: freeform jaw pairs with their own geometry and mass, the v2 plan, jobs and replays."""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np
import pytest

from freeform_fixtures import block, build_fixture_pair, conform, imprint_configurator, l_bracket, pipe_tee
from rtt.design.freeform.types import FingerPair, JawGeometry
from rtt.sim.controller import plan_grasp_v2
from rtt.sim.episode import OVERLAY_CHANNELS, EpisodeResultV2, run_episode, run_episode_v2
from rtt.sim.exec_cpu import EpisodeJob, run_jobs
from rtt.sim.fingers_v2 import check_pair_units, jaw_mass_properties, v1_pair
from rtt.sim.m1 import G1_BOX
from rtt.sim.objects import Placement
from rtt.sim.scene import build_model, build_model_v2, build_spec_v2
from rtt.sim.task import load_task, load_task_v2
from sim_fingers import straight_finger

TASK = load_task_v2()
TEE = pipe_tee()
TEE_PAIR = build_fixture_pair(imprint_configurator(), TEE, TASK)


def _model_bytes(model: mujoco.MjModel) -> bytes:
    buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return buf.tobytes()


def _body_frame_vertices(model: mujoco.MjModel, body: int) -> np.ndarray:
    """Every collision vertex of a body in its own frame (mesh frame -> geom pose -> body)."""
    pts = []
    for g in np.flatnonzero(model.geom_bodyid == body):
        mesh = int(model.geom_dataid[g])
        va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        pts.append(model.mesh_vert[va : va + vn] @ rot.reshape(3, 3).T + model.geom_pos[g])
    return np.concatenate(pts)


def _full_inertia(model: mujoco.MjModel, body: int) -> np.ndarray:
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[body])
    r = rot.reshape(3, 3)
    return r @ np.diag(model.body_inertia[body]) @ r.T


def test_the_tee_jaws_are_different_and_each_body_carries_its_own_jaw() -> None:
    model, ids = build_model_v2(TEE_PAIR, TEE.to_mesh_object(), TASK)
    left, right = TEE_PAIR.left, TEE_PAIR.right
    assert left.mass != pytest.approx(right.mass, rel=1e-3)  # the branch side and the run side differ
    for body, jaw in ((ids.finger_left_body, left), (ids.finger_right_body, right)):
        assert model.body_mass[body] == pytest.approx(jaw.mass, rel=1e-12)
        assert np.allclose(model.body_ipos[body], jaw.com, atol=1e-12)
        assert np.allclose(
            _full_inertia(model, body), jaw_mass_properties(jaw).inertia, rtol=1e-6, atol=1e-12
        )
        assert int(np.sum(model.geom_bodyid == body)) == jaw.n_pieces
        verts = _body_frame_vertices(model, body)
        assert np.allclose(verts.min(axis=0), jaw.bounds[0], atol=2e-6)  # MuJoCo stores float32 vertices
        assert np.allclose(verts.max(axis=0), jaw.bounds[1], atol=2e-6)


def test_nothing_is_mirrored_by_the_scene() -> None:
    model, ids = build_model_v2(TEE_PAIR, TEE.to_mesh_object(), TASK)
    right = _body_frame_vertices(model, ids.finger_right_body)
    mirrored_left = TEE_PAIR.left.bounds * np.array([1.0, -1.0, 1.0])
    assert not np.allclose(np.sort(mirrored_left[:, 1]), [right[:, 1].min(), right[:, 1].max()], atol=1e-4)
    # the right jaw faces +y: its pad is on its +y side, the body behind it toward -y
    assert TEE_PAIR.right.bounds[1, 1] > 0 > TEE_PAIR.right.bounds[0, 1]


def test_the_scored_scene_has_no_visual_only_geoms() -> None:
    model, _ = build_model_v2(TEE_PAIR, TEE.to_mesh_object(), TASK)
    collide = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    assert collide.all()


def test_a_v1_finger_as_a_pair_compiles_to_the_v1_scene_byte_for_byte() -> None:
    """The v2 scene is the v1 scene with the jaws swapped in: same bodies, joints, servos."""
    finger = straight_finger()
    v1_task = load_task()
    v1_model, _ = build_model(finger, G1_BOX, v1_task)
    pair = v1_pair(finger, part_id="box", tip_height=0.06, q_seat=0.022)
    v2_model, _ = build_model_v2(pair, G1_BOX, v1_task)
    assert hashlib.sha256(_model_bytes(v1_model)).digest() == hashlib.sha256(_model_bytes(v2_model)).digest()


def test_the_plan_puts_the_lowest_jaw_point_at_the_designed_tip_height() -> None:
    obj = TEE.to_mesh_object()
    model, ids = build_model_v2(TEE_PAIR, obj, TASK)
    plan = plan_grasp_v2(TEE_PAIR, obj, Placement(0.05, -0.02, 0.4), TASK)
    data = mujoco.MjData(model)
    data.qpos[ids.palm_qpos] = [plan.x, plan.y, plan.grasp_z, plan.yaw]
    mujoco.mj_kinematics(model, data)
    lowest = min(
        float((_body_frame_vertices(model, b) @ data.xmat[b].reshape(3, 3).T + data.xpos[b])[:, 2].min())
        for b in (ids.finger_left_body, ids.finger_right_body)
    )
    assert lowest == pytest.approx(TEE_PAIR.tip_height, abs=2e-6)
    assert plan.tip_height == TEE_PAIR.tip_height
    assert plan.palm_gap == pytest.approx(TEE_PAIR.tip_height + TEE_PAIR.finger_length - obj.size[2])
    assert plan.palm_gap >= TASK.controller.palm_clearance - 1e-9


def test_the_cpu_executor_runs_v2_jobs_like_serial_runs() -> None:
    blk = block()
    pair = build_fixture_pair(conform(), blk, TASK)
    jobs = [
        EpisodeJob(pair, blk, Placement(), -1, seed=0, task=TASK),
        EpisodeJob(pair, blk, Placement(err_x=0.002), -1, seed=1, task=TASK),
        EpisodeJob(TEE_PAIR, TEE, Placement(err_x=-0.003), -1, seed=2, task=TASK),
    ]
    serial = run_jobs(jobs, workers=1)
    pooled = run_jobs(jobs, workers=2)
    assert all(isinstance(r, EpisodeResultV2) for r in pooled.results)
    assert [r.summary() for r in pooled.results] == [r.summary() for r in serial.results]
    assert all(r.traj_sha for r in serial.results)
    direct = run_episode(pair, blk, Placement(), 99, 0, record=False, task=TASK)  # setting ignored
    assert direct.summary() == serial.results[0].summary()


def test_recording_carries_v2_meta_and_overlays() -> None:
    res = run_episode_v2(TEE_PAIR, TEE, Placement(err_x=0.002), 0, record=True, task=TASK)
    bare = run_episode_v2(TEE_PAIR, TEE, Placement(err_x=0.002), 0, record=False, task=TASK)
    rec, plain = res.summary(), bare.summary()
    assert rec.pop("n_steps") > plain.pop("n_steps")  # the recording tail
    assert rec == plain
    meta = res.replay.header["meta"]
    assert meta["design_space"] == "freeform-v2" and meta["part_id"] == TEE.part_id
    assert meta["design_id"] == TEE_PAIR.design_id and meta["object"]["family"] == "pipe_tee"
    assert set(res.overlays) == set(OVERLAY_CHANNELS)
    assert all(len(v) == res.replay.n_frames for v in res.overlays.values())
    assert res.overlays["n_contacts"].max() >= 2


def test_asymmetric_part_gets_asymmetric_jaws() -> None:
    bracket = l_bracket()
    pair = build_fixture_pair(imprint_configurator(), bracket, TASK)
    assert pair.left.n_pieces != pair.right.n_pieces or pair.left.mass != pytest.approx(pair.right.mass)
    res = run_episode_v2(pair, bracket, Placement(), 0, record=False, task=TASK)
    assert res.lift_success, res.summary()


def test_millimetre_jaws_are_rejected() -> None:
    jaw = TEE_PAIR.left
    big = JawGeometry(
        tuple(p * 1000 for p in jaw.pieces), jaw.mass, jaw.com, jaw.inertia6, jaw.print_mesh_sha
    )
    pair = FingerPair(
        big, TEE_PAIR.right, TEE_PAIR.tip_height, TEE_PAIR.finger_length, TEE_PAIR.q_seat,
        TEE_PAIR.design_id, TEE_PAIR.part_id, TEE_PAIR.geometry_hash, TEE_PAIR.capture, 0.0,
    )  # fmt: skip
    with pytest.raises(ValueError, match="metres"):
        check_pair_units(pair)
    with pytest.raises(ValueError, match="metres"):
        build_spec_v2(pair, TEE.to_mesh_object(), TASK)
