"""GSO v2 intake: pinned CoACD rule, textured visuals in the intake frame, brand screen.

Offline: synthetic meshes and a local mirror; CoACD is stubbed except in one small run.
The v1 tests (tests/test_gso.py, tests/test_gso_pipeline.py) stay as they are and must pass.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from rtt.sim import gso
from rtt.sim import gso_visual as gv
from rtt.sim import mesh as M
from rtt.sim import objects as objs

V1_CONFIG_DIGEST = "ce272cfb376a7048396d360e291ae669e5794f0c3bc85c172729bfafab23be82"


# --------------------------------------------------------------------------- config


def test_v1_config_digest_is_unchanged() -> None:
    assert gso.load_gso_config().digest() == V1_CONFIG_DIGEST
    payload = gso.DecompositionRules().payload()
    assert set(payload) == {"method", "threshold", "max_parts", "max_part_vertices", "seed"}


def test_v2_config_changes_only_the_decomposition_and_exclusions() -> None:
    v1 = gso.load_gso_config(gso.DEFAULT_CONFIG).describe()
    v2 = gso.load_gso_config(gso.DEFAULT_CONFIG_V2).describe()
    assert v2["name"] == "heldout_real_v2"
    d = v2["rules"].pop("decomposition")
    v1["rules"].pop("decomposition")
    assert d == {
        "method": "coacd",
        "threshold": 0.02,
        "max_parts": 32,
        "max_part_vertices": 48,
        "seed": 0,
        "min_parts": 6,
        "retry_threshold": 0.01,
        "coacd_version": "1.0.14",
    }
    ex1, ex2 = v1["draw"].pop("exclude_names"), v2["draw"].pop("exclude_names")
    assert set(ex1) < set(ex2) and len(ex2) == len(ex1) + 8
    for key in ("source", "draw", "placement", "rules", "denylist"):
        assert v1[key] == v2[key], key


def test_v2_rules_key_differs_from_v1() -> None:
    from rtt.sim.task import default_task

    task = default_task()
    k1 = gso.load_gso_config(gso.DEFAULT_CONFIG).rules.key(task)
    k2 = gso.load_gso_config(gso.DEFAULT_CONFIG_V2).rules.key(task)
    assert k1 != k2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"min_parts": 0}, "min_parts"),
        ({"min_parts": 20, "max_parts": 16}, "min_parts"),
        ({"retry_threshold": 0.05}, "retry_threshold"),
        ({"retry_threshold": -0.01}, "retry_threshold"),
        ({"coacd_version": "latest"}, "coacd_version"),
    ],
)
def test_decomposition_rules_are_validated(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gso.DecompositionRules(**kwargs)


# --------------------------------------------------------------------------- decomposition rule


def _u_mesh() -> trimesh.Trimesh:
    """A concave U-shaped block (40 x 30 x 20 mm), watertight."""
    import manifold3d as m3d

    outer = m3d.Manifold.cube([0.04, 0.03, 0.02])
    slot = m3d.Manifold.cube([0.02, 0.04, 0.015]).translate([0.01, -0.005, 0.006])
    out = (outer - slot).to_mesh64()
    return trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)


def _stub_coacd(monkeypatch: pytest.MonkeyPatch, counts: dict[float, int]) -> list[float]:
    calls: list[float] = []

    def fake(mesh, threshold, max_parts, max_part_vertices, seed):
        calls.append(threshold)
        lo, hi = mesh.bounds
        n = counts[threshold]
        edges = np.linspace(lo[0], hi[0], n + 1)
        out = []
        for a, b in itertools.pairwise(edges):
            box = trimesh.creation.box(bounds=[[a, lo[1], lo[2]], [b, hi[1], hi[2]]])
            out.append(np.asarray(box.vertices))
        return out

    monkeypatch.setattr(M, "_decompose_coacd", fake)
    monkeypatch.setattr(M, "coacd_available", lambda: True)
    monkeypatch.setattr(M, "coacd_version", lambda: "1.0.14")
    return calls


def test_too_few_pieces_retry_at_the_finer_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_coacd(monkeypatch, {0.02: 3, 0.01: 7})
    obj = M.mesh_object_from_trimesh(
        _u_mesh(),
        name="u",
        decomposition="coacd",
        threshold=0.02,
        max_parts=32,
        min_parts=6,
        retry_threshold=0.01,
        pin_coacd="1.0.14",
    )
    assert calls == [0.02, 0.01]
    assert len(obj.parts) == 7
    assert obj.decomposition == "coacd-1.0.14:t=0.01:seed=0"


def test_enough_pieces_do_not_retry_and_the_retry_result_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_coacd(monkeypatch, {0.02: 9, 0.01: 2})
    obj = M.mesh_object_from_trimesh(
        _u_mesh(),
        name="u",
        decomposition="coacd",
        threshold=0.02,
        max_parts=32,
        min_parts=6,
        retry_threshold=0.01,
    )
    assert calls == [0.02] and len(obj.parts) == 9 and obj.decomposition == "coacd-1.0.14:t=0.02:seed=0"
    calls.clear()
    _stub_coacd(monkeypatch, {0.02: 2, 0.01: 3})
    obj = M.mesh_object_from_trimesh(
        _u_mesh(),
        name="u",
        decomposition="coacd",
        threshold=0.02,
        max_parts=32,
        min_parts=6,
        retry_threshold=0.01,
    )
    assert len(obj.parts) == 3  # still fewer than 6 after the retry: kept (the object is nearly convex)


def test_v1_calls_keep_the_v1_label(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_coacd(monkeypatch, {0.05: 2})
    obj = M.mesh_object_from_trimesh(_u_mesh(), name="u", decomposition="coacd")
    assert calls == [0.05] and obj.decomposition == "coacd"


def test_pinned_coacd_version_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_coacd(monkeypatch, {0.02: 6})
    monkeypatch.setattr(M, "coacd_version", lambda: "1.0.15")
    with pytest.raises(RuntimeError, match="pinned"):
        M.mesh_object_from_trimesh(
            _u_mesh(), name="u", decomposition="coacd", threshold=0.02, pin_coacd="1.0.14"
        )


def test_real_coacd_follows_the_v2_rule() -> None:
    pytest.importorskip("coacd")
    if M.coacd_version() != "1.0.14":
        pytest.skip("CoACD 1.0.14 is the pinned version")
    obj = M.mesh_object_from_trimesh(
        _u_mesh(),
        name="u",
        decomposition="coacd",
        threshold=0.02,
        max_parts=32,
        min_parts=6,
        retry_threshold=0.01,
        pin_coacd="1.0.14",
    )
    assert obj.decomposition in ("coacd-1.0.14:t=0.02:seed=0", "coacd-1.0.14:t=0.01:seed=0")
    again = M.mesh_object_from_trimesh(
        _u_mesh(),
        name="u",
        decomposition="coacd",
        threshold=0.02,
        max_parts=32,
        min_parts=6,
        retry_threshold=0.01,
        pin_coacd="1.0.14",
    )
    assert again.geometry_sha256 == obj.geometry_sha256  # deterministic with the seed pinned


def test_screen_passes_the_v2_rule_to_intake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict = {}

    def capture(mesh, **kwargs):
        seen.update(kwargs)
        raise ValueError("stop here")

    monkeypatch.setattr(gso, "mesh_object_from_trimesh", capture)
    root = tmp_path / "mirror"
    data = trimesh.creation.box((0.04, 0.03, 0.05)).export(file_type="obj").encode()
    (root / "models" / "Box").mkdir(parents=True)
    (root / "models" / "Box" / "model.obj").write_bytes(data)
    source = gso.GsoSource(repo="t/o", commit="a" * 40, models_tree="b" * 40, n_models=1)
    cache = gso.GsoCache(source, root=tmp_path / "cache", fetch=gso.LocalMirror(root))
    rules = gso.load_gso_config(gso.DEFAULT_CONFIG_V2).rules
    result = gso.evaluate_candidate("Box", cache, rules)
    assert result.reason == "decomposition_failed"
    assert seen["threshold"] == 0.02 and seen["min_parts"] == 6 and seen["retry_threshold"] == 0.01
    assert seen["pin_coacd"] == "1.0.14" and seen["max_parts"] == 32 and seen["seed"] == 0


# --------------------------------------------------------------------------- textured visuals


def _textured_obj() -> bytes:
    """An asymmetric textured scan (a wedge on a slab), with UV seams (per-face texture
    coordinates), in metres."""
    mesh = trimesh.util.concatenate(
        [
            trimesh.creation.box(bounds=[[-0.03, -0.01, 0.0], [0.03, 0.01, 0.01]]),
            trimesh.creation.box(bounds=[[0.0, -0.01, 0.01], [0.03, 0.01, 0.05]]),
        ]
    )
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
    rot = trimesh.transformations.rotation_matrix(0.4, [0.3, 0.2, 1.0])
    mesh.apply_transform(rot)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in mesh.vertices]
    vt = []
    faces = []
    for i, (a, b, c) in enumerate(mesh.faces):
        base = len(vt)
        u = (i % 10) / 10.0
        v = (i // 10 % 10) / 10.0
        vt += [(u, v), (u + 0.05, v), (u, v + 0.05)]
        faces.append(f"f {a + 1}/{base + 1} {b + 1}/{base + 2} {c + 1}/{base + 3}")
    lines += [f"vt {u:.4f} {v:.4f}" for u, v in vt] + faces
    return ("\n".join(lines) + "\n").encode()


def test_visual_lands_exactly_on_the_intake_frame() -> None:
    data = _textured_obj()
    geom = gso.geometry_mesh(data)
    obj = M.mesh_object_from_trimesh(geom, name="w", pose_index=0, decomposition="hull")
    vis = gv.load_scan_visual(data, name="w", pose_index=0)
    assert vis.bounds_error_mm == 0.0  # bit-identical replay of the intake steps
    assert gv.collision_bounds_gap_mm(vis, obj.parts) < 0.2
    assert vis.uv is not None and vis.uv.shape == (len(vis.vertices), 2)
    assert len(vis.vertices) > len(geom.vertices)  # seams stay split
    flipped = vis.uv_top_left()
    assert np.allclose(flipped[:, 0], vis.uv[:, 0]) and np.allclose(flipped[:, 1], 1.0 - vis.uv[:, 1])
    frame, rested = M.intake_frame(geom, name="w", pose_index=0)
    assert np.array_equal(frame.apply(geom).vertices, rested.vertices)
    pts = np.c_[geom.vertices, np.ones(len(geom.vertices))] @ frame.matrix.T
    assert np.allclose(pts[:, :3], rested.vertices, atol=1e-12)


def test_other_stable_poses_follow_the_same_steps() -> None:
    data = _textured_obj()
    fits = gso.pose_fits(gso.geometry_mesh(data))
    assert len(fits) >= 2
    vis = gv.load_scan_visual(data, name="w", pose_index=1)
    assert vis.bounds_error_mm == 0.0 and vis.frame.pose_index == 1
    assert vis.frame.probability == pytest.approx(fits[1].probability)


def test_visual_refuses_a_frame_mismatch() -> None:
    with pytest.raises(ValueError, match="would not sit"):
        gv.load_scan_visual(_textured_obj(), name="w", pose_index=0, bounds_tol=-1.0)


def test_glb_export_keeps_the_obj_uv_convention(tmp_path: Path) -> None:
    from PIL import Image

    tex = tmp_path / "texture.png"
    Image.new("RGB", (8, 8), (200, 40, 40)).save(tex)
    vis = gv.load_scan_visual(_textured_obj(), name="w", pose_index=0, texture=tex)
    mesh = vis.to_trimesh()
    blob = mesh.export(file_type="glb")
    back = trimesh.load(io.BytesIO(blob), file_type="glb", force="mesh", process=False)
    order = np.lexsort(np.round(back.vertices, 9).T)
    ref = np.lexsort(np.round(vis.vertices, 9).T)
    assert np.allclose(back.visual.uv[order], vis.uv[ref], atol=1e-6)  # trimesh flips on export and import
    assert len(vis.clay().faces) == len(vis.faces) and not hasattr(vis.clay().visual, "uv")


def test_textures_are_fetched_like_meshes(tmp_path: Path) -> None:
    root = tmp_path / "mirror"
    (root / "models" / "Box").mkdir(parents=True)
    (root / "models" / "Box" / "texture.png").write_bytes(b"\x89PNG fake")
    source = gso.GsoSource(repo="t/o", commit="a" * 40, models_tree="b" * 40, n_models=1)
    cache = gso.GsoCache(source, root=tmp_path / "cache", fetch=gso.LocalMirror(root))
    path = cache.texture_path("Box")
    assert path.read_bytes() == b"\x89PNG fake" and path.is_relative_to(cache.dir)
    with pytest.raises(ValueError):
        cache.texture_path("Box", "../evil.png")


def test_scan_description_and_visual_attribution() -> None:
    d = gv.describe_scan(_textured_obj())
    assert d["has_uv"] and d["n_faces"] > 0
    src = gso.load_gso_config(gso.DEFAULT_CONFIG_V2).source
    credit = gv.visual_attribution("Some_Object", src)
    assert credit["license"] == "CC-BY-4.0" and "texture kept" in credit["changes"]
    assert credit["source_url"].endswith("/models/Some_Object")


# --------------------------------------------------------------------------- brand screen


def test_shipped_brand_screen_is_conservative() -> None:
    screen = gv.load_visual_screen()
    assert screen.default_brand_visible is True
    assert all(not e.brand_visible for e in screen.entries.values())  # only cleared objects are named
    assert len(screen.screened_branded_sha256) >= 9
    assert gv.check_shot([gv.ShotObject("Any_Unscreened_Scan", "gso", True, 0)], screen)


def test_brand_screen_rules() -> None:
    name = "Plain_Grey_Thing"
    screen = gv.visual_screen_from_dict(
        {
            "objects": {name: {"brand_visible": False, "poses": [0], "method": "renders"}},
            "screened_branded_sha256": [hashlib.sha256(b"Loud_Label_Thing").hexdigest()],
        }
    )
    assert not screen.brand_visible(name, 0)
    assert screen.brand_visible(name, 1)  # not screened in that pose
    assert screen.brand_visible(name)  # pose unknown
    assert screen.was_screened("Loud_Label_Thing") and not screen.was_screened("Other")
    assert screen.textured_ok() == (name,)
    shot = [
        gv.ShotObject(name, "gso", True, 0),
        gv.ShotObject("pipe_tee", "pipe_tee", True),
        gv.ShotObject("Loud_Label_Thing", "gso", False, 0),  # clay render: fine
    ]
    assert gv.check_shot(shot, screen) == []
    bad = gv.check_shot([gv.ShotObject("Loud_Label_Thing", "gso", True, 0)], screen)
    assert len(bad) == 1 and "not cleared" in bad[0]


@pytest.mark.parametrize(
    "raw, error",
    [
        ({"objects": {"A": {"brand_visible": "no"}}}, TypeError),
        ({"objects": {"A": {"brand_visible": False, "poses": [-1]}}}, TypeError),
        ({"objects": {"A": {"brand_visible": False, "colour": 1}}}, ValueError),
        ({"screened_branded_sha256": ["abc"]}, ValueError),
        ({"version": 2}, ValueError),
        ({"extra": 1}, ValueError),
    ],
)
def test_brand_screen_rejects_malformed_files(raw: dict, error: type) -> None:
    with pytest.raises(error):
        gv.visual_screen_from_dict(raw)


# --------------------------------------------------------------------------- v2 suite of scans


def test_selection_part_suite_keeps_conditions_and_geometry() -> None:
    mesh = trimesh.creation.cylinder(radius=0.02, height=0.06, sections=48)
    obj = M.mesh_object_from_trimesh(mesh, name="Round_Tin", decomposition="hull")
    rec = gso.CandidateResult("Round_Tin", True, None, "ok", geometry_sha256=obj.geometry_sha256)
    cond = {
        "object": "Round_Tin",
        "x": 0.01,
        "y": -0.02,
        "yaw": 0.3,
        "err_x": 0.001,
        "err_y": -0.002,
        "err_yaw": 0.05,
        "seeds": [11, 12],
    }
    src = gso.GsoSource(repo="t/o", commit="a" * 40, models_tree="b" * 40, n_models=1)
    sel = gso.Selection(
        "heldout_real_v2", "dev", {}, "c", "k", "t", dataclasses.asdict(src), 1, 1, 1, 1, (rec,), (cond,), ()
    )
    v2 = gso.selection_part_suite(sel, [obj], visual_assets={"Round_Tin": "0123456789abcdef"})
    v1 = gso.selection_suite(sel, [obj])
    assert [c.condition_id for c in v2] == [c.condition_id for c in v1]
    part = v2.conditions[0].part
    assert part.family == "gso" and part.geometry_sha256 == obj.geometry_sha256
    assert part.yaw_symmetry == 0 and "yaw" not in part.register_axes  # round footprint
    assert part.visual_asset == "0123456789abcdef"
    assert v2.conditions[0].seeds == (11, 12) and v2.conditions[0].placement == objs.Placement(
        0.01, -0.02, 0.3, 0.001, -0.002, 0.05
    )
    json.dumps(v2.describe())
