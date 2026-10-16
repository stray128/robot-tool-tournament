"""The v2 scene path: freeform jaw pairs with their own geometry and mass, the v2 plan, jobs and replays."""

from __future__ import annotations

import hashlib
import math

import mujoco
import numpy as np
import pytest

from freeform_fixtures import block, build_fixture_pair, conform, imprint_configurator, l_bracket, pipe_tee
from rtt.design.freeform.types import FingerPair, JawGeometry
from rtt.sim.controller import plan_grasp_v2
from rtt.sim.episode import OVERLAY_CHANNELS, EpisodeResultV2, run_episode, run_episode_v2
from rtt.sim.exec_cpu import EpisodeJob, run_jobs
from rtt.sim.fingers_v2 import check_convex_piece, check_pair_units, jaw_mass_properties, v1_pair
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


def _l_prism() -> np.ndarray:
    """An extruded L (non-convex): its inner corner lies inside the hull's cap faces."""
    outline = np.array([[0, 0], [20, 0], [20, 5], [5, 5], [5, 20], [0, 20]], dtype=float) * 1e-3
    return np.array([[x - 0.01, y + 0.001, z] for z in (-0.06, -0.04) for x, y in outline])


def _with_left_piece(pair: FingerPair, piece: np.ndarray) -> FingerPair:
    import dataclasses

    left = dataclasses.replace(pair.left, pieces=(piece, *pair.left.pieces[1:]))
    return dataclasses.replace(pair, left=left)


def test_a_non_convex_or_flat_piece_is_refused_not_filled_in() -> None:
    # MuJoCo would silently collide with the hull of the L, i.e. a pad cut into the part
    with pytest.raises(ValueError, match="left jaw piece 0 is not convex"):
        build_spec_v2(_with_left_piece(TEE_PAIR, _l_prism()), TEE.to_mesh_object(), TASK)
    flat = np.array([[x, y, -0.05] for x in (0.0, 0.01) for y in (0.0, 0.01)])
    with pytest.raises(ValueError, match="flat or degenerate"):
        build_spec_v2(_with_left_piece(TEE_PAIR, flat), TEE.to_mesh_object(), TASK)


def test_the_convexity_check_allows_corners_and_edge_points_only() -> None:
    cube = np.array([[x, y, z] for x in (0, 0.01) for y in (0, 0.01) for z in (0, 0.01)], dtype=float)
    check_convex_piece(cube, "cube")
    check_convex_piece(np.vstack([cube, cube[:1], [[0.005, 0.0, 0.0]]]), "duplicate and edge point")
    with pytest.raises(ValueError, match="inside a face"):
        check_convex_piece(np.vstack([cube, [[0.005, 0.005, 0.0]]]), "face point")
    with pytest.raises(ValueError, match="5.000 mm inside the hull"):
        check_convex_piece(np.vstack([cube, [[0.005, 0.005, 0.005]]]), "interior point")
    for pair in (TEE_PAIR, build_fixture_pair(conform(), block(), TASK)):
        for jaw in (pair.left, pair.right):
            for i, piece in enumerate(jaw.pieces):
                check_convex_piece(piece, f"piece {i}")


# --------------------------------------------------------------------------- evaluation and certificate


def _suite(parts, placements, seeds=(0,), err_z=0.0):
    from freeform_fixtures import fixture_suite

    return fixture_suite(parts, placements, seeds, err_z=err_z)


def test_evaluate_designs_v2_records_every_episode_and_invalid_builds() -> None:
    from rtt.sim.evaluate import RECORD_VERSION_V2, evaluate_designs_v2, score_designs_v2

    blk, tee = block(), TEE
    suite = _suite([blk, tee], {"c0": Placement(), "c1": Placement(err_x=0.002)})

    def builder(design, part, task):
        if design.pad == "flat" and part.part_id == tee.part_id:
            raise ValueError("planted: no flat jaws for the tee")
        return build_fixture_pair(design, part, task)

    def concave_builder(design, part, task):
        return _with_left_piece(build_fixture_pair(design, part, task), _l_prism())

    bad = evaluate_designs_v2(
        [conform()], _suite([blk], {"c0": Placement()}), builder=concave_builder, task=TASK
    )
    assert [r.failure_mode for r in bad] == ["invalid_build"]
    assert bad[0].flags[0].startswith("invalid_build: scene_check: left jaw piece 0 is not convex")

    designs = [conform(), conform(pad="flat", clearance_mm=0.0, pad_depth_mm=0.0, wall_draft_deg=0.0)]
    records = evaluate_designs_v2(designs, suite, builder=builder, workers=2, task=TASK)
    assert len(records) == len(designs) * len(suite)
    assert [r.design_id for r in records[:4]] == [designs[0].design_id()] * 4
    assert all(r.record_version == RECORD_VERSION_V2 and r.task == "task_v2" for r in records)
    invalid = [r for r in records if r.failure_mode == "invalid_build"]
    assert len(invalid) == 2 and all(r.part_id == tee.part_id and not r.success for r in invalid)
    assert invalid[0].flags[0].startswith("invalid_build: planted")
    sim = [r for r in records if r.failure_mode != "invalid_build"]
    assert all(r.geometry_hash and r.design_space == "freeform-v2" for r in sim)
    assert all(r.success <= r.lift_success and r.success == r.registered for r in sim)
    assert all(r.jam_on_close == ("jam_on_close" in r.registration_flags) for r in sim)
    for r in sim:
        measured = [t for t in (r.reg_tilt_deg, r.reg_hold_tilt_deg) if math.isfinite(t)]
        assert r.tilt_max_deg >= max(measured, default=0.0) - 1e-6
    d = sim[0].to_dict()
    assert d["registration_flags"] == list(sim[0].registration_flags) and d["record_version"] == 2
    scores = score_designs_v2(records)
    good, partial = scores[designs[0].design_id()], scores[designs[1].design_id()]
    assert good.n_episodes == partial.n_episodes == 4
    assert good.geometry_id is not None and partial.geometry_id is None
    assert partial.failure_modes.get("invalid_build") == 2
    assert set(good.success_by_part) == {blk.part_id, tee.part_id}


def test_v2_records_write_none_for_unmeasured_values_and_round_trip_through_parquet(tmp_path) -> None:
    import pyarrow.parquet as pq

    from rtt.sim.evaluate import invalid_record_v2, records_v2_to_table, write_parquet_v2

    suite = _suite([block()], {"c0": Placement()})
    rec = invalid_record_v2("ab" * 32, suite.conditions[0], suite.name, 3, "planted", TASK)
    d = rec.to_dict()
    assert d["seat_short_mm"] is None and d["finger_mass_g"] is None and d["failure_mode"] == "invalid_build"
    table = records_v2_to_table([rec, rec])
    assert table.num_rows == 2 and table.schema.field("registration_flags").type.value_type == "string"
    path = write_parquet_v2([rec], tmp_path / "v2.parquet")
    back = pq.read_table(path).to_pylist()[0]
    assert back["design_id"] == rec.design_id and back["flags"] == list(rec.flags)


def test_rank_designs_v2_breaks_ties_on_lift_then_mass() -> None:
    import dataclasses

    from rtt.sim.evaluate import invalid_record_v2, rank_designs_v2

    cond = _suite([block()], {"c0": Placement()}).conditions[0]
    base = invalid_record_v2("0" * 64, cond, "s", 0, "x", TASK)

    def rec(did: str, success: bool, lift: bool, mass: float):
        return dataclasses.replace(
            base, design_id=did, success=success, lift_success=lift, finger_mass_g=mass,
            failure_mode=None if success else "not_registered", geometry_hash="f" * 64,
        )  # fmt: skip

    records = [
        rec("a", True, True, 90.0), rec("a", False, False, 90.0),
        rec("b", True, True, 80.0), rec("b", False, True, 80.0),
        rec("c", True, True, 70.0), rec("c", False, True, 70.0),
    ]  # fmt: skip
    assert [s.design_id for s in rank_designs_v2(records)] == ["c", "b", "a"]


def test_a_height_error_lowers_the_jaws_on_the_part() -> None:
    from rtt.sim.evaluate import condition_task_v2

    cond = _suite([block()], {"c0": Placement()}, err_z=0.0015).conditions[0]
    task = condition_task_v2(TASK, cond)
    assert task.controller.grasp_z_offset == pytest.approx(-0.0015)
    assert condition_task_v2(TASK, _suite([block()], {"c0": Placement()}).conditions[0]) is TASK
    pair = build_fixture_pair(conform(), block(), TASK)
    plan = plan_grasp_v2(pair, block().to_mesh_object(), Placement(), task)
    assert plan.tip_height == pytest.approx(pair.tip_height - 0.0015)


def test_certify_v2_reports_held_and_registered_per_test() -> None:
    from rtt.sim.certify import certify_v2

    blk = block()
    pair = build_fixture_pair(conform(), blk, TASK)
    small = TASK.with_overrides(
        robustness={"placement_errors_mm": (2.0,), "shake_axes": ("x",), "push_axes": ("y",)}
    )
    cert = certify_v2(pair, blk, task=small, workers=2)
    names = [t.name for t in cert.tests]
    assert names[0] == "nominal" and len(names) == 1 + 4 + 1 + 1
    assert set(cert.held_scores) == set(cert.registered_scores) == {"placement", "shake", "push"}
    assert all(t.registered <= t.held for t in cert.tests)
    nominal = cert.tests[0]
    assert nominal.held and nominal.registered and nominal.offset_mm < 0.75
    assert cert.registered_tolerance_mm["x"] in (0.0, 2.0)
    d = cert.to_dict()
    assert d["registered"].endswith(f"/{len(names)}") and d["settings"]["jitter"] == "off"
    assert d["settings"]["registration"]["seat_short_max_mm"] == 1.0
    assert d["part_id"] == blk.part_id and d["geometry_hash"] == pair.geometry_hash
