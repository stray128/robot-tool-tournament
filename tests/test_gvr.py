"""Gate GV-R (render), one test per criterion, on the v2 path and the published site data.

* Render poses = replay poses within 1e-6.
* The scored v2 model's hash is unchanged with render assets present.
* The web plays v1 and v2: the site publishes both, and every v2 lane plays its own replay
  (the browser half is ``web/scripts/smoke_v2.mjs``).
* Hero payload <= 300 000 bytes per published v2 lane.
* Every GLB has a sha256 and an attribution (in its manifest entry and inside the file).
* Credits are present (README, NOTICE, docs/ATTRIBUTION.md, the site's credits page), and the
  gripper's maker is named only there and in the vendored upstream provenance.
* Print STLs are watertight single solids.
* The brand screen is clean on every hero frame, and captions carry no third-party credit.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
import trimesh

from rtt.render.glb import AssetStore, read_glb_json
from rtt.render.gripper import load_lock

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
DATA = WEB / "public" / "data"
ASSETS = WEB / "public" / "assets"
HERO_PAYLOAD_BYTES = 300_000
LOCK = load_lock()
MAKER = LOCK["attribution"].split(",")[0].strip()  # the gripper's maker, as credited upstream
EPISODES = (
    {"stem": "designed_bushing_c1", "design": "ref_wrap", "part": "flanged_bushing", "placement": "c1"},
    {"stem": "stock_bushing_c1", "design": "stock", "part": "flanged_bushing", "placement": "c1"},
)


@pytest.fixture(scope="module")
def bout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from rtt.pipeline.replays import record_bout_v2

    out = tmp_path_factory.mktemp("gvr") / "gvr-bout"
    record_bout_v2(out, EPISODES, run_id="gvr-bout")
    return out


def _published_lanes() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    bouts = json.loads((DATA / "bouts.json").read_text(encoding="utf-8"))["bouts"]
    return [
        (lane, json.loads((DATA / f"{lane['replay']}.json").read_text(encoding="utf-8")))
        for b in bouts
        for lane in b["lanes"]
    ]


def _model_digest(model: mujoco.MjModel) -> str:
    buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


def test_render_poses_equal_replay_poses(bout: Path) -> None:
    from rtt.render.playback import pose_residual
    from rtt.render.studio import build_render_scene
    from rtt.sim.replay import read_replay

    store = AssetStore(bout / "assets")
    for ep in EPISODES:
        rp = read_replay(bout / ep["stem"])
        res = pose_residual(build_render_scene(rp.header, store), rp.frames)
        assert res["frames"] == rp.n_frames
        assert res["pos"] <= 1e-6 and res["quat"] <= 1e-6, (ep["stem"], res)


def test_scored_v2_model_hash_is_unchanged_with_render_assets_present(bout: Path, tmp_path: Path) -> None:
    from rtt.design.freeform import build_pair, load_entrants_v2
    from rtt.parts.library import get_part
    from rtt.render.studio import build_render_scene
    from rtt.sim.replay import read_replay
    from rtt.sim.scene import build_model_v2
    from rtt.sim.task import default_task_v2

    task, part = default_task_v2(), get_part("flanged_bushing")
    entrants = {e.name: e for e in load_entrants_v2()}
    for ep in EPISODES:
        pair = build_pair(entrants[ep["design"]], part, task)
        obj = part.to_mesh_object()
        before = _model_digest(build_model_v2(pair, obj, task)[0])
        build_render_scene(read_replay(bout / ep["stem"]).header, AssetStore(bout / "assets"))
        (tmp_path / "assets").mkdir(exist_ok=True)
        old = Path.cwd()
        try:
            os.chdir(tmp_path)  # an asset directory next to the process changes nothing
            after = _model_digest(build_model_v2(pair, obj, task)[0])
        finally:
            os.chdir(old)
        assert before == after, ep["stem"]


def test_site_publishes_v1_and_v2_lanes_each_playing_its_own_replay() -> None:
    lanes = _published_lanes()
    versions = {h["version"] for _, h in lanes}
    assert versions == {1, 2}
    v2 = [(lane, h) for lane, h in lanes if h["version"] == 2]
    assert any(lane["object"]["kind"] == "part" for lane, _ in v2)
    for lane, h in v2:
        assert h["meta"]["design_id"] == lane["design_id"], lane["id"]
        if lane["object"]["kind"] == "part":
            assert h["meta"]["part_id"] == lane["object"]["part_id"], lane["id"]


def test_every_published_v2_lane_fits_the_hero_payload_budget() -> None:
    sizes = {}
    for lane, h in _published_lanes():
        if h["version"] != 2:
            continue
        stem = DATA / lane["replay"]
        files = [stem.with_suffix(".json"), stem.parent / h["bin"]]
        if h.get("overlays"):
            files.append(stem.parent / h["overlays"]["bin"])
        sizes[lane["replay"]] = sum(f.stat().st_size for f in files) + sum(
            a["bytes"] for a in h["assets"].values()
        )
    assert sizes and max(sizes.values()) <= HERO_PAYLOAD_BYTES, sizes


def test_every_glb_has_a_sha256_and_an_attribution(bout: Path) -> None:
    for root in (ASSETS, bout / "assets"):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["assets"]
        on_disk = {p.stem for p in root.glob("*.glb")}
        assert on_disk and on_disk == set(manifest), root
        for aid, entry in manifest.items():
            blob = (root / f"{aid}.glb").read_bytes()
            assert hashlib.sha256(blob).hexdigest() == entry["sha256"] and entry["sha256"].startswith(aid)
            assert entry["attribution"] and entry["license"]
            doc = read_glb_json(blob)  # the credit travels inside the file too
            assert doc["asset"]["copyright"] == entry["attribution"]
            assert doc["asset"]["extras"]["license"] == entry["license"]
            if entry["license"] != "Apache-2.0":
                assert entry["changes"]


def test_credits_are_present() -> None:
    credit_files = {
        "README.md": REPO / "README.md",
        "NOTICE": REPO / "NOTICE",
        "docs/ATTRIBUTION.md": REPO / "docs" / "ATTRIBUTION.md",
        "credits page": WEB / "app" / "credits" / "page.tsx",
    }
    for name, path in credit_files.items():
        text = " ".join(path.read_text(encoding="utf-8").split())
        assert f"{MAKER}, SO-ARM 102" in text, name
        assert LOCK["repository"] in text, name
        assert "CC BY 4.0" in text, name
        assert "Google Scanned Objects" in text, name


def test_the_maker_is_named_only_in_the_credit_files() -> None:
    """On screen the gripper is "an open-source parallel gripper"; the maker's name appears in
    the credit files, the vendored upstream provenance and the checks that look for it."""
    allowed = {
        "README.md",
        "NOTICE",
        "docs/ATTRIBUTION.md",
        "web/app/credits/page.tsx",
        "assets/gripper/so_arm_102/SOURCE.md",
        "assets/gripper/so_arm_102/lock.json",
        "web/scripts/smoke_v2.mjs",
        "src/rtt/library/brands.py",  # the open library's brand list (it names makers to keep them off screen)
    }
    # the open effector library's provenance (SOURCE.json, lock.json, upstream licence texts)
    allowed_prefixes = ("tests/", "assets/effectors/library/")
    files = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    # upstream repository URLs (source pins in asset metadata and page data) are not on-screen credits
    url = re.compile(r"https?://[^\s\"')>\]]+")
    hits = []
    for rel in files:
        path = REPO / rel
        if rel.startswith(allowed_prefixes) or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if MAKER.lower() in url.sub(" ", text).lower() and rel not in allowed:
            hits.append(rel)
    assert hits == []


def test_print_stls_are_watertight_single_solids(tmp_path: Path) -> None:
    from rtt.design.freeform import build_pair_full, load_entrants_v2, stl_bytes
    from rtt.part.pack import make_print_pack_v2
    from rtt.parts.library import get_part
    from rtt.sim.task import default_task_v2

    design = {e.name: e for e in load_entrants_v2()}["ref_wrap"]
    build = build_pair_full(design, get_part("flanged_bushing"), default_task_v2())
    stls = [stl_bytes(m.vertices, m.faces) for m in (build.left_mesh, build.right_mesh)]
    zpath = make_print_pack_v2(build.pair, *stls, tmp_path, recipe=build.recipe_record())
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.endswith(".stl")]
        assert len(names) == 4
        for name in names:
            m = trimesh.load(io.BytesIO(zf.read(name)), file_type="stl", force="mesh")
            m.merge_vertices()
            assert m.is_watertight and m.is_winding_consistent and m.is_volume, name
            assert m.body_count == 1 and m.volume > 0.0, name


def test_brand_screen_is_clean_on_every_hero_frame(bout: Path) -> None:
    from rtt.render.playback import brand_screen, caption_text
    from rtt.sim.gso_visual import ShotObject, check_shot
    from rtt.sim.replay import read_replay

    headers = [read_replay(bout / ep["stem"]).header for ep in EPISODES]
    headers += [h for _, h in _published_lanes() if h["version"] == 2]
    for h in headers:
        assert brand_screen(h) == []
        obj = h["meta"]["object"]
        textured = any(
            e.get("brand_visible") is not None for e in h["assets"].values() if e["role"] == "object"
        )
        shot = ShotObject(name=str(obj.get("name")), family=str(obj.get("family")), textured=textured)
        assert check_shot([shot]) == [], h["meta"]["part_id"]
        caption = caption_text(h)
        assert caption.startswith("simulation replay · MuJoCo · run ")
        assert MAKER.lower() not in caption.lower() and "CC BY" not in caption
