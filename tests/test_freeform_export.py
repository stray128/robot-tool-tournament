"""Print STLs (mm, canonical), GLB visuals (m, Z up), mass properties and the pair export."""

from __future__ import annotations

import hashlib
import io
import json
import struct

import numpy as np
import pytest
import trimesh

from freeform_parts import block, mvp_tee
from rtt.design.freeform.export import (
    STL_HEADER,
    JawMesh,
    edge_report,
    export_pair,
    glb_bytes,
    inertia6,
    mass_properties,
    stl_bytes,
)
from rtt.design.freeform.generate import build_pair_full
from rtt.design.freeform.schema import FingerRecipe
from rtt.sim.task import default_task

TASK = default_task()
RECIPE = FingerRecipe(
    pad="conform",
    tip="chamfer",
    grip_height_frac=0.5,
    pad_height_mm=20.0,
    finger_width_mm=34.0,
    pad_depth_mm=5.0,
    wall_draft_deg=30.0,
    clearance_mm=0.4,
    tip_size_mm=2.0,
)


@pytest.fixture(scope="module")
def build():
    return build_pair_full(RECIPE, mvp_tee(), TASK)


def test_stl_is_binary_millimetres_and_matches_the_print_sha(build) -> None:
    for jaw, geom in ((build.left, build.pair.left), (build.right, build.pair.right)):
        data = stl_bytes(jaw.mesh.vertices, jaw.mesh.faces)
        assert data[:80] == STL_HEADER
        (n,) = struct.unpack("<I", data[80:84])
        assert n == len(jaw.mesh.faces) and len(data) == 84 + 50 * n
        assert hashlib.sha256(data).hexdigest() == geom.print_mesh_sha
        mesh = trimesh.load(io.BytesIO(data), file_type="stl")
        assert mesh.is_watertight
        assert np.allclose(mesh.bounds, jaw.mesh.bounds * 1000.0, atol=1e-3)  # mm


def test_stl_bytes_do_not_depend_on_triangle_order(build) -> None:
    mesh = build.left.mesh
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(mesh.faces))
    rolled = np.roll(mesh.faces[perm], 1, axis=1)  # same triangles, other start vertex
    assert stl_bytes(mesh.vertices, rolled) == stl_bytes(mesh.vertices, mesh.faces)


def test_glb_is_valid_and_z_up_metres(build) -> None:
    jaw = build.left
    data = glb_bytes(jaw.mesh, name="left")
    magic, version, total = struct.unpack("<4sII", data[:12])
    assert magic == b"glTF" and version == 2 and total == len(data)
    (json_len,) = struct.unpack("<I", data[12:16])
    doc = json.loads(data[20 : 20 + json_len])
    assert doc["extras"] == {"up_axis": "Z", "units": "m"}
    acc = doc["accessors"][0]
    assert acc["count"] == len(jaw.mesh.vertices)
    assert np.allclose(acc["min"], jaw.mesh.vertices.min(axis=0), atol=1e-6)
    scene = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh")
    assert len(scene.faces) == len(jaw.mesh.faces)
    assert glb_bytes(jaw.mesh, name="left") == data  # deterministic bytes


def test_mass_properties_match_trimesh(build) -> None:
    jaw = build.left
    m = trimesh.Trimesh(jaw.mesh.vertices, jaw.mesh.faces, process=False)
    m.density = 1240.0
    mass, com, inertia = mass_properties(jaw.mesh.vertices, jaw.mesh.faces, 1240.0)
    assert mass == pytest.approx(m.mass, rel=1e-9)
    assert np.allclose(com, m.center_mass, atol=1e-12)
    assert np.allclose(inertia, m.moment_inertia, rtol=1e-6, atol=1e-14)
    assert inertia6(inertia)[3] == pytest.approx(inertia[0, 1])
    assert np.allclose(build.pair.left.inertia6, inertia6(inertia))


def test_mirrored_mesh_keeps_outward_faces(build) -> None:
    m = build.left.mesh
    r = m.mirrored_y()
    vol_l = mass_properties(m.vertices, m.faces, 1.0)[0]
    vol_r = mass_properties(r.vertices, r.faces, 1.0)[0]
    assert vol_r == pytest.approx(vol_l)
    assert edge_report(r.faces) == (True, 1)


def test_edge_report_detects_holes_and_pieces() -> None:
    from rtt.design.freeform.tip import piece_from_box

    a = piece_from_box((0, 0, 0), (1, 1, 1), "pad")
    assert edge_report(a.faces) == (True, 1)
    assert edge_report(a.faces[:-1])[0] is False
    two = np.concatenate([a.faces, a.faces + len(a.vertices)])
    assert edge_report(two) == (True, 2)
    mesh = JawMesh(np.concatenate([a.vertices, a.vertices + 2.0]), two)
    assert mesh.bounds[1, 0] == pytest.approx(3.0)


def test_export_pair_writes_the_files(tmp_path, build) -> None:
    paths = export_pair(build, tmp_path / "pack")
    assert set(paths) == {"left_stl", "right_stl", "left_glb", "right_glb", "recipe"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    rec = json.loads(paths["recipe"].read_text())
    assert rec["design_id"] == build.pair.design_id and rec["part_id"] == "pipe_tee"
    assert rec["units"] == {"stl": "mm", "glb": "m"}
    assert rec["print_mesh_sha256"]["left"] == build.pair.left.print_mesh_sha
    assert hashlib.sha256(paths["left_stl"].read_bytes()).hexdigest() == build.pair.left.print_mesh_sha
    other = build_pair_full(RECIPE, block(), TASK)
    assert other.pair.left.print_mesh_sha != build.pair.left.print_mesh_sha
