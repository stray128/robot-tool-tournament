"""Render from the replay (gate GV-R): poses, model isolation, x-ray, captions, a real render."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from rtt.design.generate import generate_finger
from rtt.design.schema import load_entrants
from rtt.pipeline.replays import lift_to_v2
from rtt.render import playback
from rtt.render.glb import AssetStore
from rtt.render.studio import GROUP_COLLISION, GROUP_VISUAL, StudioConfig, build_render_scene
from rtt.render.xray import XRAY_ALPHA, xray
from rtt.sim import objects as objs
from rtt.sim.episode import run_episode
from rtt.sim.scene import build_model
from rtt.sim.task import default_task

REPO = Path(__file__).resolve().parents[1]
ENTRANTS = {e.name: e for e in load_entrants(REPO / "configs" / "entrants.yaml")}
SMALL = StudioConfig(width=320, height=180, shadowsize=1024, offsamples=2)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    finger = generate_finger(ENTRANTS["stock"])
    obj = objs.ObjectSpec("cylinder", (0.02, 0.05), density=7850.0)
    res = run_episode(finger, obj, objs.Placement(err_x=0.002), 1, seed=3)
    store = AssetStore(tmp_path_factory.mktemp("assets"))
    v1 = res.replay
    v1.header["meta"]["design_id"] = ENTRANTS["stock"].design_id()
    v2 = lift_to_v2(v1, store, run_id="test-run", code_version="10e9384", stock_blade=True)
    return {"res": res, "v1": v1, "v2": v2, "store": store, "finger": finger, "obj": obj}


def _model_digest(model: mujoco.MjModel) -> str:
    buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


def test_render_poses_equal_replay_poses(run: dict[str, Any]) -> None:
    for replay, assets in ((run["v2"], run["store"]), (run["v1"], None)):
        scene = build_render_scene(replay.header, assets, SMALL)
        res = playback.pose_residual(scene, replay.frames)
        assert res["frames"] == replay.n_frames > 100
        assert res["pos"] <= 1e-6 and res["quat"] <= 1e-6, res


def test_render_model_is_visual_only(run: dict[str, Any]) -> None:
    scene = build_render_scene(run["v2"].header, run["store"], StudioConfig(collision_overlay=True))
    m = scene.model
    assert m.njnt == 0 and m.nu == 0 and m.nmocap == 4
    assert np.all(m.geom_contype == 0) and np.all(m.geom_conaffinity == 0)
    assert set(np.unique(m.geom_group)) <= {GROUP_VISUAL, GROUP_COLLISION}
    assert scene.body_names == ("palm", "finger_left", "finger_right", "object")
    # the collision overlay draws the v1 boxes the scorer used, in group 3 only
    overlay = np.flatnonzero(m.geom_group == GROUP_COLLISION)
    assert len(overlay) == sum(len(b.get("geoms", [])) for b in run["v2"].header["bodies"])


def test_scored_model_is_unchanged_by_render_assets(run: dict[str, Any], tmp_path: Path) -> None:
    task = default_task()
    before = _model_digest(build_model(run["finger"], run["obj"], task)[0])
    build_render_scene(run["v2"].header, run["store"], SMALL)
    (tmp_path / "assets").mkdir()
    old = Path.cwd()
    try:
        os.chdir(tmp_path)  # an asset directory next to the process changes nothing
        after = _model_digest(build_model(run["finger"], run["obj"], task)[0])
    finally:
        os.chdir(old)
    assert before == after
    # and statically: the scored scene never imports the renderer or reads asset files
    tree = ast.parse((REPO / "src" / "rtt" / "sim" / "scene.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(mod.startswith("rtt.render") for mod in imported)
    assert "assets/" not in (REPO / "src" / "rtt" / "sim" / "scene.py").read_text(encoding="utf-8")


def test_xray_sets_and_restores_the_part_alpha(run: dict[str, Any]) -> None:
    scene = build_render_scene(run["v2"].header, run["store"], SMALL)
    mats = scene.body_materials["object"]
    others = np.setdiff1d(np.arange(scene.model.nmat), mats)
    before = scene.model.mat_rgba.copy()
    with xray(scene):
        assert np.allclose(scene.model.mat_rgba[mats, 3], XRAY_ALPHA)
        assert np.array_equal(scene.model.mat_rgba[others], before[others])
    assert np.array_equal(scene.model.mat_rgba, before)


def test_caption_comes_from_provenance(run: dict[str, Any]) -> None:
    header = run["v2"].header
    text = playback.caption_text(header)
    sep = playback.CAPTION_SEPARATOR
    assert text == sep.join(
        ["simulation replay", "MuJoCo", "run test-run", f"design {ENTRANTS['stock'].design_id()[:12]}"]
    )
    product = {**header, "provenance": {**header["provenance"], "product_mode": True}}
    assert playback.caption_text(product, offline=True).endswith(
        sep.join(["product mode", "rendered offline"])
    )
    img = np.full((180, 320, 3), 200, dtype=np.uint8)
    out = playback.burn_caption(img, text, readout="2.0 -> 0.1 mm")
    assert out.shape == img.shape and not np.array_equal(out, img)
    assert np.array_equal(out[60:120, 200:], img[60:120, 200:])  # only the corners are touched


def test_studio_renders_stills_and_a_clip(run: dict[str, Any], tmp_path: Path) -> None:
    replay, store = run["v2"], run["store"]
    n = replay.n_frames
    try:
        stills = playback.render_stills(
            replay,
            store,
            {
                "wide": {"frame": n - 1, "view": "wide", "target": "rest"},
                "xray": {"frame": n // 2, "view": "xray", "xray": True},
            },
            tmp_path / "png",
            cfg=SMALL,
        )
    except RuntimeError as exc:  # no EGL or OSMesa on this machine
        pytest.skip(f"no offscreen OpenGL: {str(exc)[:200]}")
    import imageio.v2 as imageio

    wide = imageio.imread(stills["wide"])
    assert wide.shape == (180, 320, 3)
    assert wide.std() > 10  # a lit scene, not a blank frame
    clip = playback.render_video(replay, store, tmp_path / "clip.mp4", start=0, stop=12, cfg=SMALL)
    assert clip.stat().st_size > 1000


def test_rig_and_views_turn_with_the_palm(run: dict[str, Any]) -> None:
    from dataclasses import replace

    from rtt.render.studio import AUTHORED_PALM_YAW_DEG, VIEWS, align_to_palm, free_camera

    replay = run["v2"]
    cfg = align_to_palm(SMALL, replay.header, replay.frames)
    # the episode ran at placement yaw 0: the rig turns by -180 deg from the authored palm yaw
    assert abs(abs(cfg.rig_yaw_deg) - AUTHORED_PALM_YAW_DEG) < 1e-6
    frames = replay.frames.copy()
    frames[:, 0, 3:] = [0.0, 0.0, 1.0, 0.0]  # palm yawed by 180 deg: no turn needed
    assert align_to_palm(SMALL, replay.header, frames).rig_yaw_deg == pytest.approx(0.0, abs=1e-6)
    scene = build_render_scene(replay.header, run["store"], replace(SMALL, rig_yaw_deg=90.0))
    cam = free_camera(scene, "detail", (0.0, 0.0, 0.0))
    v = VIEWS["detail"]
    assert cam.azimuth == pytest.approx(v.azimuth + 90.0)
    assert cam.lookat[0] == pytest.approx(-v.offset[1]) and cam.lookat[1] == pytest.approx(v.offset[0])
    key = scene.model.light_dir[0]
    base = build_render_scene(replay.header, run["store"], SMALL).model.light_dir[0]
    assert key == pytest.approx([-base[1], base[0], base[2]], abs=1e-9)


def test_iter_frames_refuses_without_a_headless_backend(
    run: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    with pytest.raises(RuntimeError, match="MUJOCO_GL"):
        next(playback.iter_frames(run["v2"], run["store"], [{"frame": 0}]))


def test_blender_hero_helpers_run_without_blender(run: dict[str, Any], tmp_path: Path) -> None:
    from rtt.render import blender_hero
    from rtt.sim.replay import write_replay

    write_replay(run["v2"], tmp_path / "r")
    header, poses = blender_hero.load_replay(tmp_path / "r")
    assert poses.shape == (header["n_frames"], 4, 7)
    assert np.array_equal(poses, run["v2"].frames)
    assert blender_hero.blender_quat([0.1, 0.2, 0.3, 0.9]) == (0.9, 0.1, 0.2, 0.3)
    assert blender_hero.caption(header) == playback.caption_text(header, offline=True)
    aid = next(iter(header["assets"]))
    assert blender_hero.asset_path(run["store"].root, aid) == run["store"].path(aid)
    with pytest.raises(FileNotFoundError):
        blender_hero.asset_path(tmp_path, aid)
    with pytest.raises(ValueError, match="replay v2"):
        write_replay(run["v1"], tmp_path / "v1")
        blender_hero.load_replay(tmp_path / "v1")
