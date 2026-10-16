"""Replay v2: header helpers, writer checks, overlays and the reader for versions 1 and 2."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from rtt.design.generate import generate_finger
from rtt.design.schema import load_entrants
from rtt.pipeline.replays import lift_to_v2
from rtt.render.glb import AssetStore
from rtt.sim import objects as objs
from rtt.sim import replay as rp
from rtt.sim.episode import run_episode
from rtt.sim.scene import build_model
from schema_lite import Validator

REPO = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(rp.SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = Validator(SCHEMA)
ENTRANTS = {e.name: e for e in load_entrants(REPO / "configs" / "entrants.yaml")}


@pytest.fixture(scope="module")
def episode() -> Any:
    finger = generate_finger(ENTRANTS["straight"])
    return run_episode(finger, objs.ObjectSpec("box", (0.04, 0.03, 0.06)), objs.Placement(), 1, seed=0)


@pytest.fixture(scope="module")
def lifted(episode: Any, tmp_path_factory: pytest.TempPathFactory) -> tuple[rp.Replay, AssetStore]:
    store = AssetStore(tmp_path_factory.mktemp("assets"))
    v1 = copy.deepcopy(episode.replay)
    v1.header["meta"]["design_id"] = ENTRANTS["straight"].design_id()
    return lift_to_v2(v1, store, run_id="test-run", code_version="10e9384", part_id="block"), store


def test_v1_recorder_still_writes_version_1(episode: Any, tmp_path: Path) -> None:
    assert episode.replay.header["version"] == rp.VERSION_V1 == 1
    assert rp.VERSION == 2 and rp.SUPPORTED_VERSIONS == (1, 2)
    rp.write_replay(episode.replay, tmp_path / "v1")
    back = rp.read_replay(tmp_path / "v1")
    assert back.version == 1 and np.array_equal(back.frames, episode.replay.frames)
    assert "overlay_channels" not in json.loads((tmp_path / "v1.json").read_text())


def test_lifted_header_passes_the_schema_and_cross_field_rules(lifted: Any, tmp_path: Path) -> None:
    replay, store = lifted
    bin_path, json_path = rp.write_replay(replay, tmp_path / "bout")
    header = json.loads(json_path.read_text(encoding="utf-8"))
    assert VALIDATOR.errors(header) == []
    assert rp.check_header(header) == []
    assert bin_path.stat().st_size == header["n_frames"] * header["n_bodies"] * 7 * 4
    for aid, entry in header["assets"].items():
        data = store.path(aid).read_bytes()
        assert entry["bytes"] == len(data) and entry == store.entry(aid)
    back = rp.read_replay(tmp_path / "bout")
    assert back.version == 2 and np.array_equal(back.frames, replay.frames)
    assert back.body_names == ["palm", "finger_left", "finger_right", "object"]


def test_lift_keeps_the_poses_and_gives_each_jaw_its_own_asset(lifted: Any, episode: Any) -> None:
    replay, store = lifted
    assert np.array_equal(replay.frames, episode.replay.frames)
    bodies = {b["name"]: b for b in replay.header["bodies"]}
    left = {store.entry(v["asset"])["role"]: v["asset"] for v in bodies["finger_left"]["visuals"]}
    right = {store.entry(v["asset"])["role"]: v["asset"] for v in bodies["finger_right"]["visuals"]}
    assert set(left) == {"carriage", "finger_left"} and set(right) == {"carriage", "finger_right"}
    assert left["finger_left"] != right["finger_right"] and left["carriage"] != right["carriage"]
    assert all(v["scale"] == [1.0, 1.0, 1.0] for b in bodies.values() for v in b["visuals"])
    assert bodies["finger_left"]["geoms"] == episode.replay.header["bodies"][1]["geoms"]  # collision overlay
    meta, prov = replay.header["meta"], replay.header["provenance"]
    assert meta["design_space"] == "v1" and meta["geometry_id"] is None and meta["part_id"] == "block"
    assert meta["success"] == episode.success  # the v1 meta is kept
    assert prov["engine_version"] == mujoco.__version__ and prov["source"] == "simulation"
    assert prov["product_mode"] is False and prov["seed"] == 0


def test_stock_blade_lift_draws_the_cad_blade(episode: Any, tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    v2 = lift_to_v2(episode.replay, store, run_id="r", code_version="10e9384", stock_blade=True)
    fingers = [store.entry(v["asset"]) for b in v2.header["bodies"][1:3] for v in b["visuals"]]
    assert all(e["license"] == "CC-BY-4.0" for e in fingers)
    assert v2.header["meta"]["finger_visual"].startswith("stock blade CAD")


def _mutations() -> list[tuple[str, Any]]:
    def unknown_asset(h: dict) -> None:
        h["bodies"][0]["visuals"][0]["asset"] = "f" * 16

    def mirrored(h: dict) -> None:
        h["bodies"][2]["visuals"][0]["scale"] = [1.0, -1.0, 1.0]

    def no_meta_key(h: dict) -> None:
        del h["meta"]["part_id"]

    def bad_code(h: dict) -> None:
        h["provenance"]["code_version"] = "main"

    def wrong_count(h: dict) -> None:
        h["n_bodies"] = 3

    def unused_asset(h: dict) -> None:
        aid = next(iter(h["assets"]))
        h["assets"]["0" * 16] = dict(h["assets"][aid], uri=f"assets/{'0' * 16}.glb", sha256="0" * 64)

    def not_addressed(h: dict) -> None:
        aid = next(iter(h["assets"]))
        h["assets"][aid]["uri"] = "assets/other.glb"

    def no_changes(h: dict) -> None:
        for e in h["assets"].values():
            if e["license"] != "Apache-2.0":
                e["changes"] = []

    def engine(h: dict) -> None:
        h["provenance"]["engine"] = "isaac"

    def stray_provenance(h: dict) -> None:
        h["provenance"]["gpu"] = "L4"

    return [
        ("unknown asset", unknown_asset),
        ("scale", mirrored),
        ("meta.part_id", no_meta_key),
        ("code_version", bad_code),
        ("n_bodies", wrong_count),
        ("not referenced", unused_asset),
        ("content addressed", not_addressed),
        ("changes", no_changes),
        ("engine", engine),
        ("does not allow", stray_provenance),
    ]


@pytest.mark.parametrize(("message", "mutate"), _mutations(), ids=[m for m, _ in _mutations()])
def test_writer_refuses_a_broken_header(lifted: Any, tmp_path: Path, message: str, mutate: Any) -> None:
    replay, _ = lifted
    header = copy.deepcopy(replay.header)
    mutate(header)
    with pytest.raises(ValueError, match=re.escape(message)):
        rp.write_replay(rp.Replay(frames=replay.frames, header=header), tmp_path / "bad")
    assert not list(tmp_path.iterdir()), "nothing is written for an invalid header"


def test_overlays_round_trip_as_float16(episode: Any, tmp_path: Path) -> None:
    finger = generate_finger(ENTRANTS["straight"])
    model, ids = build_model(finger, objs.ObjectSpec("box", (0.04, 0.03, 0.06)), episode_task())
    data = mujoco.MjData(model)
    rec = rp.ReplayRecorder(
        model,
        ids,
        30,
        overlay_channels=("offset_mm", "n_contacts"),
        overlay_fn=lambda d: {"offset_mm": 1000 * d.time, "n_contacts": d.ncon},
    )
    for _ in range(40):
        rec.maybe_record(data)
        mujoco.mj_step(model, data, nstep=10)
    v1, _qpos = rec.finish(meta={"design_id": "d"})
    assert (
        v1.overlays is not None and v1.overlays.dtype == np.float16 and v1.overlays.shape == (v1.n_frames, 2)
    )
    store = AssetStore(tmp_path / "a")
    v2 = lift_to_v2(v1, store, run_id="r", code_version="10e9384")
    assert v2.header["overlays"]["channels"] == ["offset_mm", "n_contacts"]
    rp.write_replay(v2, tmp_path / "ovl")
    assert (tmp_path / "ovl.ovl.bin").stat().st_size == v1.n_frames * 2 * 2
    header = json.loads((tmp_path / "ovl.json").read_text())
    assert header["overlays"]["bin"] == "ovl.ovl.bin" and VALIDATOR.errors(header) == []
    back = rp.read_replay(tmp_path / "ovl")
    assert np.array_equal(back.overlays, v1.overlays)
    with pytest.raises(ValueError, match="version 2"):
        rp.write_replay(v1, tmp_path / "v1ovl")


def episode_task() -> Any:
    from rtt.sim.task import default_task

    return default_task()


def test_recorder_rejects_unknown_overlay_channels() -> None:
    finger = generate_finger(ENTRANTS["straight"])
    model, ids = build_model(finger, objs.ObjectSpec("box", (0.04, 0.03, 0.06)), episode_task())
    with pytest.raises(ValueError, match="unknown overlay"):
        rp.ReplayRecorder(model, ids, 30, overlay_channels=("speed",), overlay_fn=lambda d: {})
    with pytest.raises(ValueError, match="go together"):
        rp.ReplayRecorder(model, ids, 30, overlay_channels=("hold_s",))


def test_reader_rejects_unknown_versions(lifted: Any, tmp_path: Path) -> None:
    replay, _ = lifted
    rp.write_replay(replay, tmp_path / "x")
    header = json.loads((tmp_path / "x.json").read_text())
    header["version"] = 3
    (tmp_path / "x.json").write_text(json.dumps(header))
    with pytest.raises(ValueError, match="supported version"):
        rp.read_replay(tmp_path / "x")


def test_visual_and_provenance_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    v = rp.visual("0" * 16, pos=(0.1, 0, 0))
    assert v == {
        "asset": "0" * 16,
        "pos": [0.1, 0.0, 0.0],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    }
    with pytest.raises(ValueError, match="never a mirror"):
        rp.visual("0" * 16, scale=(1, -1, 1))
    p = rp.provenance(
        run_id="pilot-a1", code_version="abcdef1+dirty", product_mode=True, seed=None, suite="train_v2"
    )
    assert p["product_mode"] is True and "seed" not in p and p["suite"] == "train_v2"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", p["created_utc"])
    assert re.fullmatch(r"[0-9a-f]{12}(\+dirty)?", rp.code_version(REPO))
    monkeypatch.setenv("RTT_GIT_SHA", "0123456789abcdef0123456789abcdef01234567-dirty")
    assert rp.code_version() == "0123456789ab+dirty"
    assert rp.normalise_code_version("ABCDEF1") == "abcdef1"
    for bad in ("main", "", None, "0123456-clean"):
        with pytest.raises(ValueError):
            rp.normalise_code_version(bad)


def test_upgrade_header_needs_known_bodies(episode: Any) -> None:
    with pytest.raises(ValueError, match="unknown bodies"):
        rp.upgrade_header(episode.replay.header, visuals={"wrist": []}, assets={}, meta={}, provenance={})


# --------------------------------------------------------------------------- freeform (v2) episodes


def _freeform_pair(part_id: str = "block_30x40x60") -> Any:
    """A FingerPair of two distinct two-piece jaws (left faces -y, right faces +y)."""
    import hashlib

    import trimesh

    from rtt.design.freeform import CaptureReport, FingerPair, JawGeometry

    def jaw(sign: float, pad_depth: float) -> JawGeometry:
        root = trimesh.creation.box(extents=(0.030, 0.012, 0.012))
        root.apply_translation((0.0, sign * 0.0064, -0.006))
        pad = trimesh.creation.box(extents=(0.020, pad_depth, 0.048))
        pad.apply_translation((0.0, sign * (0.0004 + pad_depth / 2.0), -0.036))
        return JawGeometry(
            pieces=(np.asarray(root.vertices), np.asarray(pad.vertices)),
            mass=0.01,
            com=np.zeros(3),
            inertia6=np.array([1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0]),
            print_mesh_sha=hashlib.sha256(f"{sign}{pad_depth}".encode()).hexdigest(),
        )

    return FingerPair(
        left=jaw(1.0, 0.006),
        right=jaw(-1.0, 0.008),
        tip_height=0.01,
        finger_length=0.06,
        q_seat=0.02,
        design_id="a" * 64,
        part_id=part_id,
        geometry_hash="b" * 64,
        capture=CaptureReport(no_jam_mm=1.0, centring_mm=0.5, mu=0.3),
        build_s=0.1,
    )


def _box_part() -> Any:
    from scipy.spatial import ConvexHull

    from rtt.parts.types import Material, PartSpec

    s = np.array([0.03, 0.04, 0.06])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float) * s / 2
    vol = float(np.prod(s))
    return PartSpec(
        name="block_30x40x60",
        family="block",
        pieces=(corners,),
        faces=(ConvexHull(corners).simplices,),
        volume=vol,
        com=(0.0, 0.0, 0.0),
        unit_inertia=(vol * 0.0052 / 12, vol * 0.0045 / 12, vol * 0.0025 / 12, 0.0, 0.0, 0.0),
        material=Material("steel", 7850.0, (0.62, 0.63, 0.65), metallic=0.9, roughness=0.3),
        decomposition="exact",
    )


def test_freeform_lift_draws_each_jaw_and_the_part_from_their_pieces(episode: Any, tmp_path: Path) -> None:
    from rtt.pipeline.replays import lift_freeform_to_v2
    from rtt.render.glb import read_glb

    pair, part = _freeform_pair(), _box_part()
    v1 = copy.deepcopy(episode.replay)
    v1.header["meta"].update(design_space="freeform-v2", part_id=part.name)
    n = v1.n_frames
    readouts = {"offset_mm": np.linspace(2.0, 0.05, n), "seat_short_mm": np.zeros(n), "hold_s": np.zeros(n)}
    store = AssetStore(tmp_path / "assets")
    v2 = lift_freeform_to_v2(
        v1, store, pair, part=part, overlays=readouts, run_id="pilot", code_version="10e9384"
    )
    assert np.array_equal(v2.frames, episode.replay.frames)
    meta = v2.header["meta"]
    assert meta["design_space"] == "freeform-v2" and meta["design_id"] == "a" * 64
    assert meta["part_id"] == part.name and meta["geometry_hash"] == "b" * 64 and meta["geometry_id"] is None
    bodies = {b["name"]: b for b in v2.header["bodies"]}
    jaws = {}
    for side in ("left", "right"):
        roles = {store.entry(v["asset"])["role"]: v["asset"] for v in bodies[f"finger_{side}"]["visuals"]}
        assert set(roles) == {"carriage", f"finger_{side}"}
        jaws[side] = roles[f"finger_{side}"]
        mesh = read_glb(store.raw_bytes(jaws[side])).meshes[0]
        lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        pieces = np.vstack(getattr(pair, side).pieces)
        assert np.allclose(lo, pieces.min(axis=0), atol=2e-5) and np.allclose(
            hi, pieces.max(axis=0), atol=2e-5
        )
    assert jaws["left"] != jaws["right"]
    obj = store.entry(bodies["object"]["visuals"][0]["asset"])
    assert obj["license"] == "Apache-2.0" and obj["label"] == part.name
    path = tmp_path / "ep"
    rp.write_replay(v2, path)
    header = json.loads(path.with_suffix(".json").read_text())
    assert VALIDATOR.errors(header) == []
    assert header["overlays"]["channels"] == ["offset_mm", "seat_short_mm", "hold_s"]
    back = rp.read_replay(path)
    assert back.overlays.shape == (n, 3) and abs(float(back.overlays[0, 0]) - 2.0) < 1e-3
    with pytest.raises(ValueError, match="replay frames"):
        lift_freeform_to_v2(
            v1, store, pair, overlays={"hold_s": np.zeros(n + 1)}, run_id="r", code_version="1234567"
        )


def test_freeform_lift_draws_a_stock_pair_with_the_blade_cad(episode: Any, tmp_path: Path) -> None:
    import dataclasses

    from rtt.pipeline.replays import lift_freeform_to_v2
    from rtt.render.gripper import add_gripper_assets, stock_jaws

    left, right = stock_jaws()
    pair = dataclasses.replace(_freeform_pair(), left=left, right=right)
    store = AssetStore(tmp_path / "assets")
    v2 = lift_freeform_to_v2(copy.deepcopy(episode.replay), store, pair, run_id="r", code_version="1234567")
    grip = add_gripper_assets(store)
    bodies = {b["name"]: b for b in v2.header["bodies"]}
    for side in ("left", "right"):
        assert grip.blade(side) in {v["asset"] for v in bodies[f"finger_{side}"]["visuals"]}
    assert v2.header["meta"]["finger_visual"].startswith("stock blade CAD")


def test_scan_part_assets_carry_the_scan_credit(tmp_path: Path) -> None:
    import dataclasses

    from rtt.pipeline.replays import part_asset, scan_asset
    from rtt.render.glb import GSO_ATTRIBUTION, read_glb

    store = AssetStore(tmp_path)
    scan_part = dataclasses.replace(_box_part(), name="scan_mug", family="gso")
    aid = part_asset(store, scan_part)
    entry = store.entry(aid)
    assert entry["license"] == "CC-BY-4.0" and entry["attribution"] == GSO_ATTRIBUTION and entry["changes"]
    assert entry["brand_visible"] is False

    class Visual:  # the fields of rtt.sim.gso_visual.ScanVisual that scan_asset reads
        name = "scan_mug"
        vertices = np.asarray(scan_part.pieces[0])
        faces = np.asarray(scan_part.faces[0])
        uv = np.tile([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], (2, 1))
        texture = tmp_path / "tex.png"

        def uv_top_left(self) -> np.ndarray:
            return np.stack([self.uv[:, 0], 1.0 - self.uv[:, 1]], axis=1)

    from PIL import Image

    Image.new("RGB", (8, 8), (200, 40, 40)).save(Visual.texture)
    fields = {"license": "CC-BY-4.0", "changes": ["re-posed", "re-encoded as GLB"], "brand_visible": False}
    tex = scan_asset(store, Visual(), fields, textured=True)
    doc = read_glb(store.raw_bytes(tex))
    assert doc.copyright == GSO_ATTRIBUTION and doc.meshes[0].texture is not None
    assert np.allclose(doc.meshes[0].uv, Visual().uv_top_left(), atol=1e-6)
    clay = scan_asset(store, Visual(), fields, textured=False)
    assert read_glb(store.raw_bytes(clay)).meshes[0].texture is None and clay != tex


def test_provenance_keys_match_the_schema() -> None:
    assert set(rp.PROVENANCE_KEYS) == set(SCHEMA["$defs"]["provenance"]["properties"])
    assert SCHEMA["$defs"]["provenance"]["additionalProperties"] is False
