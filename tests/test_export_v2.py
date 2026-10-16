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
from rtt.render.glb import AssetStore, read_glb
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
HERO_PAYLOAD_BYTES = 300 * 1024  # GV-R: a hero replay with its GLBs, before web compression


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
    files += [assets / f"{aid}.glb" for aid in header["assets"]]
    return sum(f.stat().st_size for f in files)


def test_export_v2_lifts_lanes_and_publishes_assets(slice_dir: Path, tmp_path: Path) -> None:
    data, assets = tmp_path / "public" / "data", tmp_path / "public" / "assets"
    result = export_web([slice_dir], data, replay_version=2)
    assert result.replays == 5 and result.assets
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
        assert _payload(data, assets, lane) <= HERO_PAYLOAD_BYTES
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
    result = export_web([run], tmp_path / "b" / "data", replay_version=2)
    assert result.replays == 5 and set(result.assets) <= set(store.entries)
    (run / "assets" / f"{result.assets[0]}.glb").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        export_web([run], tmp_path / "c" / "data", replay_version=2)


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
