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
