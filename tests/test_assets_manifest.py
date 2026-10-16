"""Vendored gripper assets, the stock blade hulls, GLB files and the asset manifest (track R)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial import ConvexHull

from rtt.design.freeform import CARRIAGE_INNER_Y, MAX_PIECES_PER_JAW, PLA_DENSITY, JawGeometry
from rtt.render import glb, gripper
from rtt.render import materials as mat
from schema_lite import Validator

REPO = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((REPO / "src" / "rtt" / "sim" / "replay_v2.schema.json").read_text(encoding="utf-8"))
ASSET_SCHEMA = {"$defs": SCHEMA["$defs"], "$ref": "#/$defs/asset"}
LOCK = gripper.load_lock()


# --------------------------------------------------------------------------- vendored files


def test_vendored_files_match_their_lock() -> None:
    lock = gripper.verify_assets()
    assert lock["commit"] == "190eb162c9e59d7f6dceab850ea1b5c6a4539c14"
    assert lock["license"] == "CC-BY-4.0"
    for rel, entry in lock["files"].items():
        data = (gripper.ASSET_DIR / rel).read_bytes()
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], rel
        assert entry["path"].startswith(("simulation/", "DOCS-LICENSE")), "only CC BY 4.0 upstream files"


def test_licence_source_and_attribution_files_are_present() -> None:
    assert (
        (gripper.ASSET_DIR / "LICENSE-CC-BY-4.0.txt")
        .read_text(encoding="utf-8")
        .startswith("Attribution 4.0 International")
    )
    source = (gripper.ASSET_DIR / "SOURCE.md").read_text(encoding="utf-8")
    attribution = (REPO / "docs" / "ATTRIBUTION.md").read_text(encoding="utf-8")
    for doc in (source, attribution):
        assert LOCK["attribution"] in doc
        assert LOCK["commit"] in doc


def test_verify_assets_reports_a_tampered_file(tmp_path: Path) -> None:
    for rel in LOCK["files"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((gripper.ASSET_DIR / rel).read_bytes())
    (tmp_path / "lock.json").write_text(json.dumps(LOCK), encoding="utf-8")
    gripper.verify_assets(tmp_path)
    (tmp_path / "meshes" / "clamp1_1.stl").write_bytes(b"solid nope")
    with pytest.raises(gripper.AssetError, match="fetch_assets"):
        gripper.verify_assets(tmp_path)


# --------------------------------------------------------------------------- frames


def test_gripper_meshes_sit_in_the_contract_frames() -> None:
    g = gripper.load_gripper()
    # frame: its front face on the jaw root plane (palm z = -12 mm), centred in y
    assert g.frame.bounds[0][2] == pytest.approx(-0.012, abs=2e-5)
    assert abs(g.frame.bounds[:, 1].sum()) < 1e-5
    for side, sign in (("left", 1.0), ("right", -1.0)):
        blade, carriage = g.blade(side), g.carriage(side)
        assert blade.is_watertight and carriage.is_watertight
        # cut exactly at the root plane: blade below z = 0, carriage above
        assert blade.bounds[1][2] == pytest.approx(0.0, abs=1e-7)
        assert carriage.bounds[0][2] == pytest.approx(0.0, abs=1e-7)
        assert -blade.bounds[0][2] == pytest.approx(0.0625, abs=2e-4)  # 62.5 mm, as the v1 stock entrant
        inner = blade.bounds[0][1] if sign > 0 else blade.bounds[1][1]
        assert inner == pytest.approx(sign * CARRIAGE_INNER_Y, abs=1e-5)  # contract: +-0.4 mm at closure
    # the right jaw is its own geometry: the same printed part turned 180 deg about z, not a mirror
    from scipy.spatial import cKDTree

    left, right = np.asarray(g.blade_left.vertices), np.asarray(g.blade_right.vertices)
    mirror_gap = cKDTree(right).query(left * [1.0, -1.0, 1.0])[0].max()
    turned_gap = cKDTree(right).query(left * [-1.0, -1.0, 1.0])[0].max()
    assert turned_gap < 1e-3 and mirror_gap > 1e-5  # from clamp2 itself, never a mirrored left blade
    assert np.allclose(g.blade_left.volume, g.blade_right.volume, rtol=1e-6)


def test_cut_conserves_the_clamp_volume() -> None:
    g = gripper.load_gripper()
    for side in ("left", "right"):
        whole = g.blade(side).volume + g.carriage(side).volume
        assert whole == pytest.approx(10.54e-6, rel=2e-3)


# --------------------------------------------------------------------------- stock blade hulls


def test_hull_file_records_its_pinned_settings() -> None:
    hulls = gripper.load_stock_blade_hulls()
    assert hulls.meta["coacd"] == "1.0.14"
    assert hulls.meta["settings"] == gripper.HULL_SETTINGS
    assert hulls.meta["settings"]["threshold"] == 0.02 and hulls.meta["settings"]["seed"] == 0
    assert hulls.meta["upstream_commit"] == LOCK["commit"]
    g = gripper.load_gripper()
    for side in ("left", "right"):
        assert hulls.meta["inputs"][side] == gripper._mesh_digest(g.blade(side)), "hulls are stale: rebuild"
        pieces = hulls.pieces(side)
        assert 1 <= len(pieces) <= MAX_PIECES_PER_JAW
        for p in pieces:
            assert len(ConvexHull(p).vertices) == len(p), "every stored vertex is a hull vertex"


def test_hull_file_bytes_are_reproducible(tmp_path: Path) -> None:
    pytest.importorskip("coacd")
    out = gripper.build_stock_blade_hulls(tmp_path / "hulls.npz")
    assert hashlib.sha256(out.read_bytes()).hexdigest() == gripper.load_stock_blade_hulls().sha256


def test_stock_blade_report_gv1_numbers() -> None:
    rep = gripper.stock_blade_report()
    for side in ("left", "right"):
        r = rep[side]
        assert r["convex"] and not r["stale_inputs"]
        assert r["pieces"] <= MAX_PIECES_PER_JAW
        assert r["inner_face_y_mm"] == pytest.approx(0.4 if side == "left" else -0.4, abs=0.01)
        assert r["pad_area_mm2"] > 400.0  # the 22 mm tip pad
        assert r["bulge_pad_mm"] <= gripper.BULGE_LIMIT_MM  # the contact pad is reproduced
        assert r["over_limit_min_recess_mm"] is None or r["over_limit_min_recess_mm"] >= 2.0
    assert rep["pad_ok"] and rep["pieces_ok"]
    # Measured 2026-10-16: 0.69 / 0.75 mm on the recessed arms (>= 2.5 mm behind the pad plane).
    # GV1 asks for <= 0.2 mm on the inner face; this pins the number for the ruling.
    assert rep["bulge_mm"] < 1.0


def test_inner_face_deviation_of_exact_hulls_is_zero() -> None:
    box = trimesh.creation.box(extents=(0.02, 0.008, 0.05))
    box.apply_translation((0.0, 0.004 + 0.0004, -0.025))
    dev = gripper.inner_face_deviation(box, (np.asarray(box.vertices),), "left")
    assert dev["bulge_mm"] == 0.0 and dev["inset_mm"] == 0.0 and dev["spill_mm2"] == 0.0
    assert dev["inner_face_y_mm"] == pytest.approx(0.4, abs=1e-6)
    proud = np.asarray(box.vertices) - [0.0, 0.0005, 0.0]  # every hull point 0.5 mm toward -y
    dev2 = gripper.inner_face_deviation(box, (proud,), "left")
    assert dev2["bulge_mm"] == pytest.approx(0.5, abs=1e-3)


def test_stock_jaws_are_contract_jaw_geometry() -> None:
    left, right = gripper.stock_jaws()
    g = gripper.load_gripper()
    for jaw, side in ((left, "left"), (right, "right")):
        assert isinstance(jaw, JawGeometry)
        assert jaw.mass == pytest.approx(g.blade(side).volume * PLA_DENSITY)
        lo, hi = jaw.bounds
        assert np.all(jaw.com > lo) and np.all(jaw.com < hi)
        assert jaw.print_mesh_sha == hashlib.sha256(gripper.stl_bytes_mm(g.blade(side))).hexdigest()
    assert left.com[1] > 0 > right.com[1]  # left jaw on +y, right on -y


# --------------------------------------------------------------------------- GLB files


def _box_mesh(look: mat.Look = mat.FINGER_DESIGNED) -> glb.GlbMesh:
    return glb.mesh_from_trimesh(trimesh.creation.box(extents=(0.02, 0.01, 0.06)), look, name="box")


@pytest.mark.parametrize("quantize", [True, False])
def test_glb_round_trip_and_determinism(quantize: bool) -> None:
    mesh = _box_mesh()
    a = glb.glb_bytes([mesh], copyright="credit line", extras={"role": "finger_left"}, quantize=quantize)
    b = glb.glb_bytes(
        [_box_mesh()], copyright="credit line", extras={"role": "finger_left"}, quantize=quantize
    )
    assert a == b
    assert a[:4] == b"glTF" and len(a) % 4 == 0
    doc = glb.read_glb(a)
    assert doc.copyright == "credit line" and doc.extras == {"role": "finger_left"}
    (back,) = doc.meshes
    assert np.abs(back.vertices - mesh.vertices).max() < (2e-6 if quantize else 1e-7)
    assert np.array_equal(back.faces, mesh.faces)
    assert np.abs(back.normals - mesh.normals).max() < 0.02
    assert back.look.base_color == pytest.approx(mesh.look.base_color, abs=2e-3)
    assert back.look.roughness == pytest.approx(mesh.look.roughness)
    js = json.loads(a[20 : 20 + int.from_bytes(a[12:16], "little")])
    assert js["asset"]["generator"] == glb.GENERATOR
    assert ("KHR_mesh_quantization" in js.get("extensionsRequired", [])) == quantize


def test_glb_carries_a_texture_and_rejects_bad_input() -> None:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 30, 30)).save(buf, format="PNG")
    box = trimesh.creation.box()
    uv = np.zeros((len(box.vertices), 2))
    tex = glb.GlbMesh(box.vertices, box.faces, mat.PART_LOOKS["clay"], uv=uv, texture=buf.getvalue())
    doc = glb.read_glb(glb.glb_bytes([tex], copyright="x"))
    assert doc.meshes[0].texture == buf.getvalue()
    with pytest.raises(ValueError, match="texture needs uv"):
        glb.GlbMesh(box.vertices, box.faces, mat.FRAME, texture=buf.getvalue())
    with pytest.raises(ValueError, match="credit line"):
        glb.glb_bytes([tex], copyright="")
    with pytest.raises(ValueError, match="faces"):
        glb.GlbMesh(box.vertices, box.faces + 100, mat.FRAME)


def test_mesh_from_geoms_fuses_v1_boxes() -> None:
    geoms = [
        {"type": "box", "half_size": [0.01, 0.005, 0.02], "pos": [0, 0, -0.02], "quat_xyzw": [0, 0, 0, 1]},
        {"type": "box", "half_size": [0.01, 0.005, 0.02], "pos": [0, 0, -0.05], "quat_xyzw": [0, 0, 0, 1]},
        {
            "type": "cylinder",
            "radius": 0.004,
            "half_height": 0.01,
            "pos": [0, 0.007, -0.03],
            "quat_xyzw": [0, 0, 0, 1],
        },
        {"type": "plane", "half_size": [1, 1], "pos": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]},
    ]
    m = glb.mesh_from_geoms(geoms)
    assert m.is_watertight and m.body_count == 1
    assert m.bounds[0][2] == pytest.approx(-0.07, abs=2e-5)


def test_material_conversions() -> None:
    assert mat.srgb_to_linear(1.0) == pytest.approx(1.0)
    assert mat.linear_to_srgb(mat.srgb_to_linear(0.5)) == pytest.approx(0.5)
    g = mat.FINGER_DESIGNED.gltf()
    assert g["baseColorFactor"][0] == pytest.approx(mat.srgb_to_linear(0.96), abs=1e-6)
    m = mat.PART_LOOKS["brass"].mujoco()
    assert m["metallic"] == 1.0 and 0 < m["specular"] <= 1 and 0 < m["shininess"] <= 1
    with pytest.raises(ValueError):
        mat.Look("bad", (1.2, 0.0, 0.0))


# --------------------------------------------------------------------------- asset store and manifest


def test_store_is_content_addressed_and_validates(tmp_path: Path) -> None:
    store = glb.AssetStore(tmp_path / "assets")
    data = glb.glb_bytes([_box_mesh()], copyright=glb.OURS_ATTRIBUTION)
    aid = store.add(data, role="finger_left", label="box finger, left jaw")
    assert aid == hashlib.sha256(data).hexdigest()[:16]
    assert store.path(aid).read_bytes() == data
    assert store.add(data, role="finger_left", label="again") == aid  # idempotent
    with pytest.raises(ValueError, match="different metadata"):
        store.add(data, role="object")
    entry = store.entry(aid)
    assert Validator(ASSET_SCHEMA).errors(entry) == []
    assert entry["uri"] == f"assets/{aid}.glb" and entry["bytes"] == len(data)
    path = store.write_manifest()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format"] == glb.MANIFEST_FORMAT and list(doc["assets"]) == [aid]
    reopened = glb.AssetStore(tmp_path / "assets")
    assert reopened.entries == store.entries
    assert reopened.raw_bytes(aid) == data


def test_store_enforces_licence_and_privacy_rules(tmp_path: Path) -> None:
    store = glb.AssetStore(tmp_path)
    data = glb.glb_bytes([_box_mesh()], copyright="x")
    with pytest.raises(ValueError, match="must list its changes"):
        store.add(data, role="prop", license="CC-BY-4.0", attribution="someone")
    with pytest.raises(ValueError, match="local path"):
        store.add(data, role="prop", label="/home/someone/part.glb")
    with pytest.raises(ValueError, match="source.path"):
        store.add(
            data,
            role="prop",
            source={"kind": "upstream", "url": "https://x.org", "sha256": "0" * 64, "path": "/tmp/x"},
        )
    with pytest.raises(ValueError, match="role"):
        store.add(data, role="wheel")


def test_compressing_store_keeps_the_raw_file(tmp_path: Path) -> None:
    store = glb.AssetStore(tmp_path / "served", compressor=lambda b: b + b"\0" * 4, raw_root=tmp_path / "raw")
    data = glb.glb_bytes([_box_mesh()], copyright="x")
    aid = store.add(data, role="object")
    assert store.path(aid).read_bytes() == data + b"\0" * 4
    assert store.raw_bytes(aid) == data
    with pytest.raises(ValueError, match="raw_root"):
        glb.AssetStore(tmp_path / "x", compressor=lambda b: b)


def test_gripper_assets_carry_the_credit_without_naming_the_maker(tmp_path: Path) -> None:
    store = glb.AssetStore(tmp_path)
    ids = gripper.add_gripper_assets(store)
    assert len({ids.frame, ids.carriage_left, ids.carriage_right, ids.blade_left, ids.blade_right}) == 5
    maker = LOCK["attribution"].split(",")[0].strip().lower()
    for aid, entry in store.entries.items():
        assert Validator(ASSET_SCHEMA).errors(entry) == [], aid
        assert entry["license"] == "CC-BY-4.0" and entry["changes"]
        assert entry["attribution"] == glb.GRIPPER_ATTRIBUTION
        assert entry["source"]["commit"] == LOCK["commit"]
        doc = glb.read_glb(store.raw_bytes(aid))
        assert doc.copyright == glb.GRIPPER_ATTRIBUTION and doc.extras["license"] == "CC-BY-4.0"
        # the on-screen rule: the maker is credited in ATTRIBUTION.md / NOTICE / README / credits only
        assert maker not in json.dumps(entry).lower().replace(LOCK["repository"].lower(), "")
    assert store.entry(ids.blade_right)["role"] == "finger_right"
    # payload of the gripper visuals (uncompressed, quantised), for the 300 KB hero budget
    assert sum(e["bytes"] for e in store.entries.values()) < 200_000


def test_manifest_doc_rules_match_the_code() -> None:
    doc = (REPO / "docs" / "ASSET_MANIFEST.md").read_text(encoding="utf-8")
    assert glb.MANIFEST_FORMAT in doc
    for role in glb.ROLES:
        assert f"`{role}`" in doc
    assert re.search(r"short (on-screen )?credit", doc), "the doc explains the short credit form"
