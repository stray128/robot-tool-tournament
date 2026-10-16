"""Web export of replay v2 with GLB assets, and the v2 print pack (track R)."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

from rtt.design.freeform import CaptureReport, FingerPair, JawGeometry
from rtt.llm.client import TokenBucket
from rtt.part.pack import PACK_V2_FILES, make_print_pack_v2
from rtt.pipeline.export_web import export_web
from rtt.pipeline.replays import lift_to_v2
from rtt.pipeline.slice import SliceConfig, read_slice, run_slice
from rtt.render.compress import compression_available
from rtt.render.glb import AssetStore, read_glb, read_glb_json
from rtt.render.gripper import load_lock
from rtt.search.fakes import ScriptedProposer
from rtt.search.nemotron import make_proposer_client
from rtt.search.sim_adapter import SimEvaluator
from rtt.sim.conditions import Suite, training_suite
from rtt.sim.exec_cpu import CpuExecutor
from rtt.sim.replay import read_replay, write_replay
from schema_lite import Validator

REPO = Path(__file__).resolve().parents[1]
VALIDATOR = Validator(json.loads((REPO / "src" / "rtt" / "sim" / "replay_v2.schema.json").read_text()))
HERO_PAYLOAD_BYTES = 300_000  # GV-R: one lane's download (header, poses, overlays, GLBs) as served
needs_compressor = pytest.mark.skipif(
    not compression_available(), reason="GLB compression needs node and web/node_modules (pnpm install)"
)


@pytest.fixture(scope="module")
def slice_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("slice") / "slice-v2"
    client = make_proposer_client(out / "llm_cache", transport=ScriptedProposer(3), limiter=TokenBucket(1e12))
    suite = training_suite()
    with CpuExecutor(4) as pool:
        sim = SimEvaluator(Suite(suite.name, suite.conditions[:2]), executor=pool)
        run_slice(out, SliceConfig(seed=5), llm_client=client, evaluator=sim, executor=pool)
    return out


def _payload(data: Path, assets: Path, lane: dict[str, Any]) -> int:
    header = json.loads((data / f"{lane['replay']}.json").read_text())
    files = [data / f"{lane['replay']}.json", data / f"{lane['replay']}.bin"]
    if header.get("overlays"):
        files.append(data / f"{lane['replay']}.ovl.bin")
    files += [assets / f"{aid}.glb" for aid in header["assets"]]
    return sum(f.stat().st_size for f in files)


@needs_compressor
def test_export_v2_lifts_lanes_and_publishes_assets(slice_dir: Path, tmp_path: Path) -> None:
    data, assets = tmp_path / "public" / "data", tmp_path / "public" / "assets"
    result = export_web([slice_dir], data, replay_version=2)
    assert result.replays == 5 and result.assets
    assert set(result.payload_bytes) == {
        lane["replay"] for lane in json.loads((data / "bouts.json").read_text())["bouts"][0]["lanes"]
    }
    for aid in result.assets:  # served meshopt-compressed, credit kept in the file
        doc = read_glb_json((assets / f"{aid}.glb").read_bytes())
        assert "EXT_meshopt_compression" in doc["extensionsRequired"]
        assert doc["asset"]["copyright"]
    manifest = json.loads((assets / "manifest.json").read_text())
    assert set(result.assets) <= set(manifest["assets"])
    bout = json.loads((data / "bouts.json").read_text())["bouts"][0]
    for lane in bout["lanes"]:
        header = json.loads((data / f"{lane['replay']}.json").read_text())
        assert header["version"] == 2 and VALIDATOR.errors(header) == []
        assert header["n_frames"] == lane["n_frames"]
        assert header["provenance"]["run_id"] == bout["id"]
        for aid, entry in header["assets"].items():
            blob = (assets / f"{aid}.glb").read_bytes()
            assert hashlib.sha256(blob).hexdigest() == entry["sha256"] == manifest["assets"][aid]["sha256"]
        fingers = [header["assets"][v["asset"]] for b in header["bodies"][1:3] for v in b["visuals"]]
        licences = {e["role"]: e["license"] for e in fingers}
        stock = lane["entrant"] == "stock"
        assert licences["finger_left"] == ("CC-BY-4.0" if stock else "Apache-2.0")
        assert _payload(data, assets, lane) == result.payload_bytes[lane["replay"]] <= HERO_PAYLOAD_BYTES
    # the default export is unchanged: version 1 headers, no assets
    v1 = export_web([slice_dir], tmp_path / "v1" / "data")
    lane = json.loads((tmp_path / "v1" / "data" / "bouts.json").read_text())["bouts"][0]["lanes"][0]
    assert json.loads((tmp_path / "v1" / "data" / f"{lane['replay']}.json").read_text())["version"] == 1
    assert v1.assets == [] and not (tmp_path / "v1" / "assets").exists()


def test_v2_run_directories_export_with_their_own_assets(slice_dir: Path, tmp_path: Path) -> None:
    run = tmp_path / "slice-v2run"
    shutil.copytree(slice_dir, run)
    store = AssetStore(run / "assets")
    for lane in read_slice(run)["lanes"]:
        stem = run / lane["replay"]
        write_replay(lift_to_v2(read_replay(stem), store, run_id="v2run", code_version="10e9384"), stem)
    store.write_manifest()
    with pytest.raises(ValueError, match="replay_version=2"):
        export_web([run], tmp_path / "a" / "data")
    result = export_web([run], tmp_path / "b" / "data", replay_version=2, compress=False)
    assert result.replays == 5 and set(result.assets) <= set(store.entries)  # served raw: the store's ids
    (run / "assets" / f"{result.assets[0]}.glb").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        export_web([run], tmp_path / "c" / "data", replay_version=2, compress=False)


# --------------------------------------------------------------------------- v2 bouts

BOUT_EPISODES = (
    {"stem": "flat_block_c1", "design": "flat", "part": "block", "placement": "c1"},
    {
        "stem": "stock_block_c1",
        "design": "stock",
        "part": "block",
        "placement": "c1",
        "label": "Stock blade, as shipped",
    },
)


@pytest.fixture(scope="module")
def bout_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from rtt.pipeline.replays import record_bout_v2

    out = tmp_path_factory.mktemp("bout") / "v2-test-bout"
    record_bout_v2(out, BOUT_EPISODES, run_id="v2-test-bout", title="Two v2 jaws", summary="Test bout.")
    return out


def test_record_bout_v2_writes_lanes_with_scored_outcomes(bout_dir: Path) -> None:
    doc = json.loads((bout_dir / "bout.json").read_text())
    assert doc["format"] == "rtt-bout" and doc["version"] == 2 and doc["run_id"] == "v2-test-bout"
    assert [lane["stem"] for lane in doc["lanes"]] == ["flat_block_c1", "stock_block_c1"]
    flat, stock = doc["lanes"]
    assert flat["label"] == "Flat pad" and stock["label"] == "Stock blade, as shipped"
    assert flat["condition"] == "block, 2 mm off centre across the jaw"
    for lane in doc["lanes"]:
        header = read_replay(bout_dir / lane["stem"]).header
        assert header["version"] == 2 and VALIDATOR.errors(header) == []
        assert (
            header["meta"]["design_id"] == lane["design_id"] and header["meta"]["part_id"] == lane["part_id"]
        )
        assert (
            header["meta"]["success"] == lane["success"]
            and header["meta"]["failure_mode"] == lane["failure_mode"]
        )
        assert header["provenance"]["condition"] == lane["condition_id"] == "train_v2/block/c1"
        text = (bout_dir / f"{lane['stem']}.json").read_text()
        assert (
            text == json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False) + "\n"
        )  # compact


@needs_compressor
def test_export_v2_bout_publishes_labelled_part_lanes_within_budget(bout_dir: Path, tmp_path: Path) -> None:
    data, assets = tmp_path / "public" / "data", tmp_path / "public" / "assets"
    with pytest.raises(ValueError, match="replay_version=2"):
        export_web([bout_dir], data)
    result = export_web([bout_dir], data, replay_version=2)
    bout = json.loads((data / "bouts.json").read_text())["bouts"][0]
    assert (
        bout["id"] == "v2-test-bout" and bout["title"] == "Two v2 jaws" and bout["task"]["name"] == "task_v2"
    )
    manifest = json.loads((assets / "manifest.json").read_text())["assets"]
    labels = {lane["id"]: lane["label"] for lane in bout["lanes"]}
    assert labels == {"flat_block_c1": "Flat pad", "stock_block_c1": "Stock blade, as shipped"}
    for lane in bout["lanes"]:
        assert lane["object"]["kind"] == "part" and lane["object"]["part_id"] == "block"
        assert lane["controller_setting"] is None and lane["outcome"]["source"] == "scorer"
        header = json.loads((data / f"{lane['replay']}.json").read_text())
        assert VALIDATOR.errors(header) == []
        assert header["meta"]["design_id"] == lane["design_id"]  # the label describes this replay
        assert _payload(data, assets, lane) == result.payload_bytes[lane["replay"]] <= HERO_PAYLOAD_BYTES
        for aid, entry in header["assets"].items():
            blob = (assets / f"{aid}.glb").read_bytes()
            assert hashlib.sha256(blob).hexdigest() == entry["sha256"] and entry["attribution"]
            assert manifest[aid] == entry
            assert "EXT_meshopt_compression" in read_glb_json(blob)["extensionsRequired"]
        jaws = {b["name"]: b for b in header["bodies"]}
        if lane["entrant"] == "stock":  # the blade's 80 hulls are not published; the CAD is drawn
            assert "geoms" not in jaws["finger_left"] and "geoms" not in jaws["finger_right"]
            assert header["meta"]["collision_overlay_omitted"] == ["finger_left", "finger_right"]
        else:
            assert jaws["finger_left"]["geoms"] and jaws["finger_right"]["geoms"]


def test_export_v2_bout_refuses_a_lane_labelled_for_another_replay(bout_dir: Path, tmp_path: Path) -> None:
    run = tmp_path / "swapped"
    shutil.copytree(bout_dir, run)
    doc = json.loads((run / "bout.json").read_text())
    doc["lanes"][0]["stem"], doc["lanes"][1]["stem"] = doc["lanes"][1]["stem"], doc["lanes"][0]["stem"]
    (run / "bout.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="another lane's label"):
        export_web([run], tmp_path / "data", replay_version=2, compress=False)


def test_export_v2_refuses_a_lane_over_the_payload_budget(bout_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="byte budget"):
        export_web([bout_dir], tmp_path / "data", replay_version=2, compress=False, payload_budget=50_000)
    assert not (tmp_path / "data" / "bouts.json").exists()  # nothing half-published


@needs_compressor
def test_compression_is_deterministic_and_keeps_each_credit(bout_dir: Path) -> None:
    from rtt.render.compress import compress_assets, remap_header_assets

    store = AssetStore(bout_dir / "assets")
    raw = {aid: (store.path(aid).read_bytes(), store.entry(aid)) for aid in sorted(store.entries)}
    first, again = compress_assets(raw), compress_assets(raw)
    assert {k: v[0] for k, v in first.items()} == {k: v[0] for k, v in again.items()}
    assert sum(len(b) for b, _ in first.values()) < 0.6 * sum(len(b) for b, _ in raw.values())
    for aid, (blob, entry) in first.items():
        for key in ("role", "license", "attribution", "changes", "source"):
            assert entry[key] == raw[aid][1][key]
        doc = read_glb_json(blob)
        assert doc["asset"]["copyright"] == entry["attribution"]
        assert doc["asset"]["extras"]["license"] == entry["license"]
    header = read_replay(bout_dir / "stock_block_c1").header
    remap_header_assets(header, first)
    served = {e["sha256"][:16] for _, e in first.values()}
    assert set(header["assets"]) <= served
    assert all(v["asset"] in header["assets"] for b in header["bodies"] for v in b["visuals"])


# --------------------------------------------------------------------------- print pack v2


def _jaw(side: str) -> tuple[JawGeometry, bytes]:
    sign = 1.0 if side == "left" else -1.0
    root = trimesh.creation.box(extents=(0.030, 0.012, 0.012))
    root.apply_translation((0.0, sign * (0.0004 + 0.006), -0.006))
    pad = trimesh.creation.box(extents=(0.020, 0.006, 0.050))
    pad.apply_translation((0.0, sign * (0.0004 + 0.003 + (0.002 if side == "left" else 0.0)), -0.035))
    solid = trimesh.boolean.union([root, pad], engine="manifold")
    mm = solid.copy()
    mm.apply_scale(1000.0)
    stl = mm.export(file_type="stl")
    solid.density = 1240.0
    i = solid.moment_inertia
    jaw = JawGeometry(
        pieces=(np.asarray(root.vertices), np.asarray(pad.vertices)),
        mass=float(solid.mass),
        com=np.asarray(solid.center_mass),
        inertia6=np.array([i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]),
        print_mesh_sha=hashlib.sha256(stl).hexdigest(),
    )
    return jaw, stl


@pytest.fixture()
def pair() -> tuple[FingerPair, bytes, bytes]:
    (left, l_stl), (right, r_stl) = _jaw("left"), _jaw("right")
    fp = FingerPair(
        left=left,
        right=right,
        tip_height=0.01,
        finger_length=0.06,
        q_seat=0.02,
        design_id="d" * 64,
        part_id="block",
        geometry_hash="e" * 64,
        capture=CaptureReport(no_jam_mm=1.0, centring_mm=0.5, mu=0.3),
        build_s=0.1,
    )
    return fp, l_stl, r_stl


def test_print_pack_v2_ships_watertight_upright_jaws(pair: Any, tmp_path: Path) -> None:
    fp, l_stl, r_stl = pair
    zpath = make_print_pack_v2(fp, l_stl, r_stl, tmp_path, recipe={"pad": "flat"}, part={"family": "block"})
    with zipfile.ZipFile(zpath) as zf:
        names = sorted(n.split("/", 1)[1] for n in zf.namelist())
        assert names == sorted(PACK_V2_FILES)
        folder = zf.namelist()[0].split("/")[0]
        read = {n: zf.read(f"{folder}/{n}") for n in names}
    manifest = json.loads(read["manifest.json"])
    assert manifest["version"] == 2 and manifest["design"]["design_id"] == "d" * 64
    for name, blob in read.items():
        if name != "manifest.json":
            assert manifest["files"][name]["sha256"] == hashlib.sha256(blob).hexdigest()
    meshes = {}
    for side in ("left", "right"):
        m = trimesh.load(io.BytesIO(read[f"{side}.stl"]), file_type="stl", force="mesh")
        m.merge_vertices()
        assert m.is_watertight and m.is_volume
        assert m.bounds[0][2] == pytest.approx(0.0, abs=1e-6)  # root plane on the bed
        assert m.bounds[1][2] == pytest.approx(60.0, abs=1e-3)  # tip up
        meshes[side] = m
        mount = trimesh.load(io.BytesIO(read[f"mount_{side}.stl"]), file_type="stl", force="mesh")
        mount.merge_vertices()
        assert mount.is_watertight and mount.bounds[1][2] > m.bounds[1][2]
        doc = read_glb(read[f"{side}.glb"])
        assert doc.extras["role"] == f"finger_{side}"
        assert doc.extras["license"] == "CC0-1.0" and "CC0-1.0" in doc.copyright  # one licence per file
        assert manifest["files"][f"{side}.stl"]["jaw_frame_sha256"] == getattr(fp, side).print_mesh_sha
    assert meshes["left"].volume != pytest.approx(0.0)
    attribution = read["ATTRIBUTION.txt"].decode()
    assert load_lock()["attribution"] in attribution and "CC0-1.0" in attribution
    again = make_print_pack_v2(
        fp, l_stl, r_stl, tmp_path / "again", recipe={"pad": "flat"}, part={"family": "block"}
    )
    with zipfile.ZipFile(again) as zf:
        assert zf.read(f"{folder}/left.stl") == read["left.stl"]  # deterministic geometry


def test_print_pack_v2_refuses_mismatched_or_open_meshes(pair: Any, tmp_path: Path) -> None:
    fp, l_stl, r_stl = pair
    with pytest.raises(ValueError, match="print_mesh_sha"):
        make_print_pack_v2(fp, r_stl, l_stl, tmp_path, recipe={})
    open_mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    open_mesh.faces = open_mesh.faces[:-2]
    bad = open_mesh.export(file_type="stl")
    left = JawGeometry(
        fp.left.pieces, fp.left.mass, fp.left.com, fp.left.inertia6, hashlib.sha256(bad).hexdigest()
    )
    fp_bad = FingerPair(**{**{f: getattr(fp, f) for f in fp.__dataclass_fields__}, "left": left})
    with pytest.raises(ValueError, match="closed"):
        make_print_pack_v2(fp_bad, bad, r_stl, tmp_path, recipe={})
    with pytest.raises(ValueError, match="pack name"):
        make_print_pack_v2(fp, l_stl, r_stl, tmp_path, recipe={}, name="../escape")


def test_print_pack_v2_refuses_the_stock_blade(tmp_path: Path) -> None:
    """The stock blade is the gripper's own CC BY 4.0 part: never relabelled as a CC0-1.0 jaw."""
    from rtt.part.mount import ClampInterface
    from rtt.render.gripper import load_gripper, stl_bytes_mm, stock_jaws

    left, right = stock_jaws()
    grip = load_gripper()
    fp = FingerPair(
        left=left,
        right=right,
        tip_height=0.01,
        finger_length=0.0625,
        q_seat=0.02,
        design_id="a" * 64,
        part_id="block",
        geometry_hash="b" * 64,
        capture=CaptureReport(no_jam_mm=1.0, centring_mm=0.5, mu=0.3),
        build_s=0.0,
    )
    blades = stl_bytes_mm(grip.blade("left")), stl_bytes_mm(grip.blade("right"))
    wide = ClampInterface(block_width_mm=50.0)  # wide enough that only the licence rule stops it
    with pytest.raises(ValueError, match="CC BY 4.0"):
        make_print_pack_v2(fp, *blades, tmp_path, recipe={}, clamp=wide)
    assert not any(tmp_path.iterdir())


def test_print_pack_v2_refuses_a_jaw_that_misses_its_carriage(pair: Any, tmp_path: Path) -> None:
    """A jaw that does not reach the root plane would print as two loose solids."""
    fp, _, _ = pair
    jaws, stls = {}, {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        solid = trimesh.creation.box(extents=(0.030, 0.012, 0.010))
        solid.apply_translation((0.0, sign * 0.0064, -0.008))  # top face 3 mm below the root plane
        mm = solid.copy()
        mm.apply_scale(1000.0)
        stls[side] = mm.export(file_type="stl")
        solid.density = 1240.0
        i = solid.moment_inertia
        jaws[side] = JawGeometry(
            pieces=(np.asarray(solid.vertices),),
            mass=float(solid.mass),
            com=np.asarray(solid.center_mass),
            inertia6=np.array([i[0, 0], i[1, 1], i[2, 2], 0.0, 0.0, 0.0]),
            print_mesh_sha=hashlib.sha256(stls[side]).hexdigest(),
        )
    detached = FingerPair(
        **{
            **{f: getattr(fp, f) for f in fp.__dataclass_fields__},
            "left": jaws["left"],
            "right": jaws["right"],
        }
    )
    with pytest.raises(ValueError, match="carriage block"):
        make_print_pack_v2(detached, stls["left"], stls["right"], tmp_path, recipe={})
    assert not any(tmp_path.iterdir())  # nothing half-written


# --------------------------------------------------------------------------- print pack v2, real jaws

#: Real generator builds (rtt.design.freeform) on library parts (rtt.parts): every pad strategy,
#: a hook, a sweep and four different parts. Each jaw carries the 46 mm root block measured on
#: the gripper's clamp cut face and two blind M3 holes opening on its root plane.
REAL_PAIRS = (
    ("flat", "block"),
    ("imprint_configurator", "pipe_tee"),
    ("imprint_swept", "steel_slug"),
    ("ref_wrap", "flanged_bushing"),
    ("ref_hook", "hex_standoff"),
)


def _real_build(entrant: str, part: str) -> Any:
    from rtt.design.freeform import build_pair_full, load_entrants_v2
    from rtt.parts.library import get_part
    from rtt.sim.task import default_task_v2

    design = {e.name: e for e in load_entrants_v2()}[entrant]
    return build_pair_full(design, get_part(part), default_task_v2())


def _real_stls(build: Any) -> tuple[bytes, bytes]:
    from rtt.design.freeform import stl_bytes

    return (
        stl_bytes(build.left_mesh.vertices, build.left_mesh.faces),
        stl_bytes(build.right_mesh.vertices, build.right_mesh.faces),
    )


def _solid(blob: bytes) -> trimesh.Trimesh:
    m = trimesh.load(io.BytesIO(blob), file_type="stl", force="mesh")
    m.merge_vertices()
    return m


@pytest.mark.parametrize(("entrant", "part"), REAL_PAIRS)
def test_print_pack_v2_packs_real_jaws_as_single_watertight_solids(
    entrant: str, part: str, tmp_path: Path
) -> None:
    from manifold3d import Manifold, OpType

    from rtt.design.freeform import body as B
    from rtt.part.mount import manifold_from_trimesh
    from rtt.part.pack import V2_BLOCK_SOURCE

    build = _real_build(entrant, part)
    l_stl, r_stl = _real_stls(build)
    zpath = make_print_pack_v2(build.pair, l_stl, r_stl, tmp_path, recipe=build.recipe_record())
    with zipfile.ZipFile(zpath) as zf:
        folder = zf.namelist()[0].split("/")[0]
        read = {n.split("/", 1)[1]: zf.read(n) for n in zf.namelist()}
    assert sorted(read) == sorted(PACK_V2_FILES)
    manifest = json.loads(read["manifest.json"])
    assert manifest["clamp_interface"]["block_width_mm"] == {"value": 46.0, "source": V2_BLOCK_SOURCE}
    for side, jaw_stl in (("left", l_stl), ("right", r_stl)):
        jaw, mount = _solid(read[f"{side}.stl"]), _solid(read[f"mount_{side}.stl"])
        for name, m in ((f"{side}.stl", jaw), (f"mount_{side}.stl", mount)):
            assert m.is_watertight and m.is_winding_consistent and m.is_volume, name
            assert m.body_count == 1 and m.volume > 0.0, name
            assert manifest["files"][name]["volume_mm3"] == pytest.approx(m.volume, rel=1e-6)
        assert jaw.extents[0] == pytest.approx(46.0, abs=1e-3)  # the measured root block
        assert mount.extents[0] == pytest.approx(46.0, abs=1e-3)  # the block is as wide as the jaw
        # the bare jaw keeps its two bolt holes (filling them adds their volume) ...
        holes = []
        sy = 1.0 if side == "left" else -1.0
        r, depth = 500.0 * B.MOUNT_HOLE_DIAMETER, 1000.0 * B.MOUNT_HOLE_DEPTH
        for sx in (-1.0, 1.0):
            cyl = Manifold.cylinder(depth, r, r, 64)
            holes.append(cyl.translate((sx * 1000.0 * B.MOUNT_HOLE_X, sy * 1000.0 * B.MOUNT_HOLE_Y, -depth)))
        frame_jaw = _solid(jaw_stl)
        filled = Manifold.batch_boolean([manifold_from_trimesh(frame_jaw), *holes], OpType.Add).volume()
        assert filled - frame_jaw.volume == pytest.approx(2 * np.pi * r * r * depth, rel=0.05)
        # ... and the one-piece mount has them filled: no sealed void left inside the print
        assert manifest["files"][f"mount_{side}.stl"]["voids_filled"] == 2
    assert folder.startswith("print_pack_")


def test_print_pack_v2_refuses_a_block_narrower_than_the_jaw(tmp_path: Path) -> None:
    from rtt.part.mount import ClampInterface

    build = _real_build("flat", "block")
    with pytest.raises(ValueError, match="wider than the 42 mm carriage block"):
        make_print_pack_v2(build.pair, *_real_stls(build), tmp_path, recipe={}, clamp=ClampInterface())
    assert not any(tmp_path.iterdir())


def test_one_solid_check_names_an_enclosed_void() -> None:
    from rtt.part.pack import _check_one_solid, _fill_voids

    outer = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    cavity = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    cavity.invert()
    hollow = trimesh.util.concatenate([outer, cavity])
    with pytest.raises(ValueError, match="encloses 1 void"):
        _check_one_solid(hollow, "hollow")
    solid, n = _fill_voids(hollow, "hollow")
    assert n == 1 and solid.volume == pytest.approx(1000.0)
    _check_one_solid(solid, "filled")
