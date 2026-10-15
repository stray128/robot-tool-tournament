"""Object placement errors and mesh objects (stable poses, convex pieces, mass, episodes)."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
import trimesh

from rtt.design.generate import generate_finger
from rtt.design.schema import load_entrants
from rtt.sim import objects as objs
from rtt.sim.episode import run_episode
from rtt.sim.m1 import DEFAULT_ENTRANTS
from rtt.sim.mesh import (
    coacd_available,
    load_mesh_object,
    mesh_object_from_trimesh,
    read_mesh_object,
    save_mesh_object,
    stable_poses,
)
from rtt.sim.scene import build_model
from rtt.sim.task import load_task

TASK = load_task()
STOCK = generate_finger({d.name: d for d in load_entrants(DEFAULT_ENTRANTS)}["stock"])


def _bracket_mm() -> trimesh.Trimesh:
    """An L bracket in millimetres: a 60 x 30 x 8 plate with an 8 x 30 x 50 upright."""
    plate = trimesh.creation.box((60, 30, 8))
    plate.apply_translation((0, 0, 4))
    upright = trimesh.creation.box((8, 30, 50))
    upright.apply_translation((26, 0, 25))
    return trimesh.boolean.union([plate, upright], engine="manifold")


# --------------------------------------------------------------------------- placement


def test_placement_error_is_in_the_nominal_object_frame() -> None:
    p = objs.Placement(x=0.1, y=-0.05, yaw=math.pi / 2, err_x=0.002, err_y=0.003, err_yaw=0.1)
    real = p.actual()
    # yaw 90 deg: object x (finger width) points along world +Y, object y (closing) along world -X
    assert real.x == pytest.approx(0.1 - 0.003)
    assert real.y == pytest.approx(-0.05 + 0.002)
    assert real.yaw == pytest.approx(math.pi / 2 + 0.1)
    assert (real.err_x, real.err_y, real.err_yaw) == (0.0, 0.0, 0.0)
    shifted = objs.Placement(0.01, 0.02, 0.3).actual(dx=0.001, dy=-0.002, dyaw=0.05)
    assert (shifted.x, shifted.y, shifted.yaw) == pytest.approx((0.011, 0.018, 0.35))


def test_pose_error_moves_the_object_but_not_the_controller() -> None:
    err = objs.Placement(0.0, 0.0, 0.0, err_y=0.003)
    a = run_episode(STOCK, objs.ObjectSpec("box", (0.05, 0.04, 0.06)), err, 1, seed=0, record=True, task=TASK)
    assert a.qpos is not None
    _, ids = build_model(STOCK, objs.ObjectSpec("box", (0.05, 0.04, 0.06)), TASK)
    start = a.qpos[0]
    assert start[ids.palm_qpos[1]] == pytest.approx(0.0)  # palm aims at the nominal pose
    assert start[ids.object_qpos + 1] == pytest.approx(0.003 + a.metrics["jitter_dy"])
    assert a.metrics["err_y"] == pytest.approx(0.003)


# --------------------------------------------------------------------------- mesh objects


def test_box_mesh_matches_the_primitive_box() -> None:
    box = trimesh.creation.box((0.04, 0.06, 0.05))
    obj = mesh_object_from_trimesh(box, name="box", decomposition="hull")
    # grasp frame: narrowest horizontal extent across the jaw (y), resting on a large face
    assert sorted(obj.size) == pytest.approx([0.04, 0.05, 0.06])
    assert obj.size[1] <= obj.size[0]
    assert obj.volume == pytest.approx(0.04 * 0.06 * 0.05)
    assert np.allclose(obj.com, 0.0, atol=1e-9)
    assert objs.height(obj) == pytest.approx(obj.size[2])
    assert objs.grip_width(obj) == pytest.approx(obj.size[1])
    model, ids = build_model(STOCK, obj, TASK)
    assert model.body_mass[ids.object_body] == pytest.approx(obj.volume * TASK.physics.object_density)
    assert len(ids.object_geoms) == 1
    assert model.geom_type[ids.object_geom] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_priority[ids.object_geom] == 1
    # same inertia as MuJoCo computes for a solid box of that mass
    m = obj.volume * TASK.physics.object_density
    sx, sy, sz = obj.size
    expected = m / 12 * np.array([sy**2 + sz**2, sx**2 + sz**2, sx**2 + sy**2])
    assert np.allclose(np.sort(model.body_inertia[ids.object_body]), np.sort(expected), rtol=1e-6)


def test_stable_poses_are_ranked_and_deterministic() -> None:
    mesh = _bracket_mm()
    mesh.apply_scale(0.001)
    poses = stable_poses(mesh)
    probs = [p for _, p in poses]
    assert len(poses) >= 3
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    again = stable_poses(mesh)
    assert all(np.array_equal(a, b) for (a, _), (b, _) in zip(poses, again, strict=True))


def test_load_stl_in_millimetres(tmp_path) -> None:
    path = tmp_path / "bracket.stl"
    _bracket_mm().export(path)
    with pytest.raises(ValueError, match="scale=0.001"):
        load_mesh_object(path)
    obj = load_mesh_object(path, scale=0.001, decomposition="hull")
    assert obj.name == "bracket" and len(obj.source_sha256) == 64
    assert obj.volume == pytest.approx((60 * 30 * 8 + 8 * 30 * 42) * 1e-9, rel=1e-6)
    assert max(obj.size) < 0.1
    # poses 0 and 1 lie the bracket on its two mirror-image sides; some other pose stands it up
    heights = {
        round(load_mesh_object(path, scale=0.001, decomposition="hull", pose_index=i).size[2], 4)
        for i in range(4)
    }
    assert len(heights) >= 2
    with pytest.raises(ValueError, match="pose_index"):
        load_mesh_object(path, scale=0.001, decomposition="hull", pose_index=99)


@pytest.mark.skipif(not coacd_available(), reason="coacd not installed")
def test_coacd_splits_a_concave_part_and_is_deterministic(tmp_path) -> None:
    path = tmp_path / "bracket.stl"
    _bracket_mm().export(path)
    a = load_mesh_object(path, scale=0.001, decomposition="coacd")
    b = load_mesh_object(path, scale=0.001, decomposition="coacd")
    assert a.decomposition == "coacd"
    assert len(a.parts) >= 2  # an L cannot be one convex piece
    assert len(a.parts) == len(b.parts)
    assert all(np.array_equal(x, y) for x, y in zip(a.parts, b.parts, strict=True))
    hull = load_mesh_object(path, scale=0.001, decomposition="hull")
    assert len(hull.parts) == 1
    # mass properties come from the solid mesh, not from the (overlapping) pieces
    assert a.volume == pytest.approx(hull.volume)


@pytest.mark.skipif(not coacd_available(), reason="coacd not installed")
def test_suite_digest_pins_the_collision_geometry() -> None:
    """Same file, name and size but a different decomposition is a different suite."""
    from rtt.sim.conditions import mesh_suite

    bracket = _bracket_mm()
    hull = mesh_object_from_trimesh(bracket, name="b", scale=0.001, decomposition="hull", source_sha256="f")
    fine = mesh_object_from_trimesh(bracket, name="b", scale=0.001, decomposition="coacd", source_sha256="f")
    coarse = mesh_object_from_trimesh(
        bracket, name="b", scale=0.001, decomposition="coacd", max_parts=1, source_sha256="f"
    )
    assert len(fine.parts) > len(coarse.parts)
    hashes = {o.geometry_sha256 for o in (hull, fine, coarse)}
    assert len(hashes) == 3
    digests = {mesh_suite("s", [o]).digest() for o in (hull, fine, coarse)}
    assert len(digests) == 3
    again = mesh_object_from_trimesh(bracket, name="b", scale=0.001, decomposition="coacd", source_sha256="f")
    assert again.geometry_sha256 == fine.geometry_sha256
    described = mesh_suite("s", [fine]).describe()["conditions"][0]["object"]
    assert described["geometry_sha256"] == fine.geometry_sha256
    assert described["n_parts"] == len(fine.parts) and described["decomposition"] == "coacd"


def test_mesh_object_cache_round_trip(tmp_path) -> None:
    obj = mesh_object_from_trimesh(trimesh.creation.cylinder(0.02, 0.06, sections=24), name="can")
    back = read_mesh_object(save_mesh_object(obj, tmp_path / "can.npz"))
    assert back.geometry_sha256 == obj.geometry_sha256
    assert back.name == "can" and back.decomposition == obj.decomposition
    assert back.volume == obj.volume and back.com == obj.com and back.unit_inertia == obj.unit_inertia
    assert all(np.array_equal(x, y) for x, y in zip(obj.parts, back.parts, strict=True))
    assert all(np.array_equal(x, y) for x, y in zip(obj.faces, back.faces, strict=True))


def test_mesh_object_is_grasped_and_replayed(tmp_path) -> None:
    obj = mesh_object_from_trimesh(
        trimesh.creation.box((0.05, 0.035, 0.07)), name="block", decomposition="hull"
    )
    res = run_episode(STOCK, obj, objs.Placement(0.02, -0.01, 0.4), 1, seed=3, record=True, task=TASK)
    assert res.success, res.summary()
    assert res.metrics["rest_table_support"] == pytest.approx(1.0, abs=0.02)
    assert res.replay is not None
    geom = res.replay.header["bodies"][-1]["geoms"][0]
    assert geom["type"] == "mesh"
    verts = np.array(geom["vertices"]).reshape(-1, 3)
    faces = np.array(geom["faces"]).reshape(-1, 3)
    assert len(verts) == 8 and len(faces) == 12 and faces.max() < len(verts)
    assert res.replay.header["meta"]["object"]["name"] == "block"


def test_mesh_object_validation() -> None:
    with pytest.raises(ValueError, match="at least one convex piece"):
        objs.MeshObject("x", (), (), 1e-6, (0, 0, 0), (1, 1, 1, 0, 0, 0))
    flat = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0.01, 0.01, 0]], dtype=float)
    with pytest.raises(ValueError, match="degenerate"):
        objs.MeshObject(
            "x", (flat,), (np.array([[0, 1, 2], [1, 3, 2]]),), 1e-6, (0, 0, 0), (1, 1, 1, 0, 0, 0)
        )
