"""Catalog effectors (track B1; product mode, not part of the pre-registered experiment).

* vendoring: every shipped file re-hashes to its ``SOURCE.json`` record, every licence is
  permissive and pinned at the audited Menagerie commit, the upstream model and licence files
  are the ones the licence audit hashed (``docs/EFFECTORS.md`` section 9), and the excluded
  models are absent;
* names: generic keys and display names, and no brand word in any element name of a built
  scene (element names reach replay lanes);
* building: every preset compiles in the shared scene with its declared actuators; the
  procedural adaptive grippers are underactuated (one actuator drives two joints);
* determinism (gate C3): two in-process runs and a spawned worker give one ``traj_sha``;
* the hold gates that the grippers meet at nominal, the three-finger engagement count, and the
  c2 lane payload of every preset.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from rtt.effectors.catalog.adaptive import ADAPTIVE_MODELS, joint_coupling
from rtt.effectors.catalog.family import OPENING_MARGIN, check_fits, mount_geometry
from rtt.effectors.catalog.models import CATALOG_MODELS
from rtt.effectors.catalog.vendored import CATALOG_DIR, read_model, vendored_keys, verify_model
from rtt.effectors.registry import get_family
from rtt.effectors.runner import EffectorJob, run_effector_jobs
from rtt.effectors.scene import build_effector_scene
from rtt.effectors.types import (
    ALLOWED_LICENSES,
    PRODUCT_MODE_LABEL,
    EffectorFamily,
    InvalidEffector,
    check_display_name,
    find_brand_tokens,
)
from rtt.parts.library import get_part
from rtt.parts.suite import train_suite
from rtt.sim import objects as objs
from rtt.sim import replay as rp
from rtt.sim.certify import certificate_task
from rtt.sim.episode import OVERLAY_CHANNELS
from rtt.sim.task import default_task_v2

PINNED_COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
PARTS = ("block", "steel_slug", "hex_standoff", "l_bracket", "pipe_tee", "flanged_bushing")
#: Upstream directory -> (model file sha256, LICENSE sha256), as audited in docs/EFFECTORS.md 9.
AUDIT = {
    "wonik_allegro": (
        "cec3efe34be49c8220c23afdae0cfacbce10164a5bc7c34075b6668ee0a862a6",
        "09c187439ecece8b00330e71eb988d3d30f29373c580dc31ab54200d0aa4deb4",
    ),
    "shadow_hand": (
        "ac14d09dd12b62c3ca62233d0ac646101cc718146dda76e26d0b993a91abf399",
        "d42cae9a0f292b7a713597ba26e1e1e23d50df906c488f0d36fd3538af200bc1",
    ),
    "sharpa_wave": (
        "22208af2ee00bcb62a197dbba8e5ed6577ec41f30ba1fa13fdee375a299ab27e",
        "8d3d877d33afe58c5146e4304db9cc01e7550f843b025a461ea2804e90a3c6d3",
    ),
    "umi_gripper": (
        "12b6bbd5e94758164141d54417cad9cde6acdc073311c7c9657dca1ae3fde6ec",
        "304e99a48271e10820f71a1b86ef395c06c100d7f68dd1d308790d1064fc4d41",
    ),
    "i2rt_yam": (
        "5ae901629db944abd3e6cbb0eaf28ea78bbda3483c840d6db1167e999cc0348c",
        "dc46c3a7dc48221aa783e634d15db0f816ba83ba5cdb77bbaa91525c61c7552f",
    ),
    "agilex_piper": (
        "a7b5b5d3b2a68d5c553b2ee9665d54a422bd8bf1fa6f3251bc11834993d37098",
        "e32d78fd3761052ab0b2bb42fef8a7a9ad3f298d315d305333c349d2aff1f134",
    ),
}
#: Never vendored: the amber hand (founder decision) and the red one.
EXCLUDED_UPSTREAM = ("leap_hand", "tetheria_aero_hand_open")
LANE_BUDGET = 300_000  # bytes, the hero payload budget (docs/EFFECTORS.md section 6)


def _presets() -> list[Any]:
    return [*get_family("catalog_hand").presets(), *get_family("catalog_gripper").presets()]


def _workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


@pytest.fixture(scope="module")
def task() -> Any:
    return default_task_v2()


# --------------------------------------------------------------------------- vendoring


def test_vendored_files_rehash_to_their_records() -> None:
    keys = vendored_keys()
    assert set(keys) == {k for k, m in CATALOG_MODELS.items()} == set(keys)
    for key in keys:
        assert verify_model(key) == [], key


def test_licences_are_permissive_pinned_and_audited() -> None:
    seen_dirs = set()
    for key in vendored_keys():
        vm = read_model(key)
        assert vm.status == "green" and not vm.note
        assert vm.upstream["commit"] == PINNED_COMMIT
        assert vm.upstream["repository"] == "https://github.com/google-deepmind/mujoco_menagerie"
        directory = vm.upstream["directory"]
        seen_dirs.add(directory)
        assert directory not in EXCLUDED_UPSTREAM
        by_path = {s.upstream_path: s for s in vm.sources}
        model_sha, licence_sha = AUDIT[directory]
        assert by_path[f"{directory}/{vm.upstream['model_file']}"].upstream_sha256 == model_sha
        assert by_path[f"{directory}/LICENSE"].upstream_sha256 == licence_sha
        for src in vm.sources:
            assert src.license in ALLOWED_LICENSES and src.license == vm.license
            assert src.upstream_commit == PINNED_COMMIT and src.status == "green"
            assert src.path.startswith(f"assets/effectors/catalog/{key}/")
            assert (src.sha256 == src.upstream_sha256) == (not src.changes), src.path
        for mesh, rel in vm.mesh_files.items():  # every derived file says what it is
            shipped = next(s for s in vm.sources if s.path.endswith(rel))
            assert shipped.changes and "convex hull" in shipped.changes[0], mesh
    assert seen_dirs == set(AUDIT)
    for name in EXCLUDED_UPSTREAM:
        assert not any(name in p.read_text(encoding="utf-8") for p in CATALOG_DIR.glob("*/SOURCE.json"))


def test_keys_and_screen_names_are_generic() -> None:
    for key in vendored_keys():
        assert not find_brand_tokens(key.replace("-", " ")) and not find_brand_tokens(key)
        check_display_name(read_model(key).display_name)
    for spec in _presets():
        check_display_name(spec.display_name)
        assert spec.origin == "catalog" and spec.to_dict()["label"] == PRODUCT_MODE_LABEL
    assert len({s.effector_id for s in _presets()}) == len(_presets())


# --------------------------------------------------------------------------- building


def _element_names(model: mujoco.MjModel) -> list[str]:
    out = []
    for obj, n in (
        (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        (mujoco.mjtObj.mjOBJ_TENDON, model.ntendon),
        (mujoco.mjtObj.mjOBJ_EQUALITY, model.neq),
        (mujoco.mjtObj.mjOBJ_MESH, model.nmesh),
        (mujoco.mjtObj.mjOBJ_MATERIAL, model.nmat),
    ):
        out += [mujoco.mj_id2name(model, obj, i) or "" for i in range(n)]
    return out


@pytest.mark.parametrize("spec", _presets(), ids=lambda s: f"{s.model}-{s.primitive}")
def test_every_preset_builds_in_the_shared_scene(spec: Any, task: Any) -> None:
    impl = get_family(spec.family)
    scene = build_effector_scene(impl, spec, get_part("block"), task)
    b = scene.build
    assert scene.model.nu == 4 + b.n_actuators and len(b.ctrl_open) == b.n_actuators
    assert 0.0 < b.reach < 0.5 and b.mass is not None and 0.0 < b.mass <= 5.0
    assert b.actuation and all(isinstance(v, int | float) for v in b.actuation.values())
    names = " ".join(_element_names(scene.model))
    assert find_brand_tokens(names) == [] and find_brand_tokens(names.replace("_", " ")) == []
    if spec.model in CATALOG_MODELS:
        assert b.sources and {s.license for s in b.sources} <= set(ALLOWED_LICENSES)
    prim = impl.primitive(spec, b)
    u = prim.command(0.0, None, 1.0, scene.model, None, scene)  # type: ignore[arg-type]
    assert np.asarray(u).shape == (b.n_actuators,)


def test_adaptive_grippers_are_underactuated(task: Any) -> None:
    impl = get_family(EffectorFamily.CATALOG_GRIPPER)
    for spec in impl.presets():
        if spec.model not in ADAPTIVE_MODELS:
            continue
        scene = build_effector_scene(impl, spec, get_part("pipe_tee"), task)
        design = ADAPTIVE_MODELS[spec.model]
        assert scene.build.n_actuators == design.n_fingers
        assert all(joint_coupling(scene.model, int(a)) >= 2 for a in scene.ids.effector_actuators)
        hinges = [
            j for j in range(scene.model.njnt) if scene.model.jnt_bodyid[j] in scene.ids.effector_bodies
        ]
        assert len(hinges) == 2 * design.n_fingers


def test_a_gripper_refuses_a_part_wider_than_its_opening() -> None:
    cm = CATALOG_MODELS["gripper-2f-rigid-a"]
    geo = mount_geometry(cm.key, "pinch")
    assert geo.pose.opening is not None

    class Wide:
        size = (0.03, geo.pose.opening - OPENING_MARGIN + 0.001, 0.03)

    with pytest.raises(InvalidEffector, match="opens"):
        check_fits(cm, geo, Wide())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- determinism (C3)


def _c(part: str, label: str) -> Any:
    return next(c for c in train_suite() if c.part.name == part and c.placement_label == label)


def test_episodes_are_deterministic_in_process_and_in_workers(task: Any) -> None:
    cond = _c("block", "c0")
    jobs = [EffectorJob(s, cond.part, cond.placement, cond.seeds[0], task=task) for s in _presets()]
    here = run_effector_jobs(jobs, workers=1)
    again = run_effector_jobs(jobs, workers=1)
    there = run_effector_jobs(jobs, workers=_workers())
    for a, b, c in zip(here, again, there, strict=True):
        assert len(a.traj_sha) == 64 and a.traj_sha == b.traj_sha == c.traj_sha, a.spec.model
        assert a.bad_qacc == 0 and a.label == PRODUCT_MODE_LABEL
        json.dumps(a.to_record(), allow_nan=False)


# --------------------------------------------------------------------------- gates


def _nominal_jobs(specs: list[Any], task: Any) -> list[EffectorJob]:
    """The certificate's nominal test (jitter off, no placement error) on the six parts."""
    ctask = certificate_task(task)
    return [
        EffectorJob(s, get_part(p), objs.Placement(), 0, task=ctask, condition_id="certificate/nominal")
        for s in specs
        for p in PARTS
    ]


def test_grippers_hold_at_nominal(task: Any) -> None:
    """Each gripper that meets the B1 gate holds at least 5 of the 6 training parts at nominal,
    and the three-finger adaptive gripper engages with at least 3 bodies on the pipe tee and the
    flanged bushing. (The vendored rigid-pad grippers and the hands are reported, not gated:
    their upstream gains are too weak at the parts' friction; see the track report.)"""
    gated = [
        s
        for s in get_family("catalog_gripper").presets()
        if s.model in ADAPTIVE_MODELS or s.model == "gripper-2f-finray"
    ]
    results = run_effector_jobs(_nominal_jobs(gated, task), workers=_workers())
    by_spec: dict[str, list[Any]] = {}
    for r in results:
        by_spec.setdefault(r.spec.effector_id, []).append(r)
        assert r.bad_qacc == 0
    for rs in by_spec.values():
        held = sum(r.held for r in rs)
        assert held >= 5, (rs[0].spec.model, rs[0].spec.primitive, [(r.part_id, r.failure_mode) for r in rs])
        if rs[0].spec.model == "adaptive-3f":
            for r in rs:
                if r.part_id in ("pipe_tee", "flanged_bushing"):
                    assert r.metrics["eng_bodies"] >= 3, (r.part_id, r.metrics["eng_bodies"])


def _lane_bytes(result: Any, directory: Path) -> int:
    v1 = dict(result.replay.header, overlay_channels=list(OVERLAY_CHANNELS))
    header = rp.upgrade_header(
        v1,
        visuals={},
        assets={},
        meta={"geometry_id": None},
        provenance=rp.provenance(run_id="effectors-b1", code_version="0" * 7, product_mode=True),
    )
    overlays = np.stack([result.overlays[c] for c in OVERLAY_CHANNELS], axis=1).astype(np.float16)
    assert rp.check_header(header, result.replay.frames, overlays) == []
    stem = directory / f"{result.spec.model}-{result.spec.primitive}"
    rp.write_replay(rp.Replay(frames=result.replay.frames, header=header, overlays=overlays), stem)
    assert not find_brand_tokens(stem.with_suffix(".json").read_text(encoding="utf-8").replace("_", " "))
    return sum(p.stat().st_size for p in directory.glob(f"{stem.name}*"))


def test_c2_lanes_fit_the_payload_budget(task: Any, tmp_path: Path) -> None:
    jobs = []
    for spec in _presets():
        cond = _c("pipe_tee", "c2")
        jobs.append(EffectorJob(spec, cond.part, cond.placement, cond.seeds[0], task=task, record=True))
    for r in run_effector_jobs(jobs, workers=_workers()):
        if r.failure_mode == "invalid_build":
            continue
        size = _lane_bytes(r, tmp_path)
        assert 0 < size <= LANE_BUDGET, (r.spec.model, r.spec.primitive, size)
        assert math.isfinite(r.max_lift)
