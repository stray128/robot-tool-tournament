"""Track S7: the Studio's site data: replays of full-task runs, the blocker probe, the export.

Physics uses S1's ``part`` and ``bin`` templates with the S1 reference controllers (fast and
independent of S2's tuning); reference results are never evidence, but a replay must reproduce
any scored run, so they are what the recorder is tested on. The published files under
``web/public/data/studio/`` are checked when they exist (the export script writes them).
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rtt.effectors.types import PRODUCT_MODE_LABEL
from rtt.library.brands import find_library_brands
from rtt.sim.replay import check_header, read_replay, write_replay
from rtt.studio import replay as sr
from rtt.studio.buildpack import setup_from_dict, task_from_dict
from rtt.studio.episode import StudioCondition, StudioJob, run_task_jobs
from rtt.studio.search import candidate_of, run_task_tournament
from rtt.studio.types import ObjectInstance, ObjectPose, Step, SuccessCriterion, TaskSpec
from studio_fixtures import cup, gripper, two_press_task

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "web" / "public" / "data"
ASSETS = REPO / "web" / "public" / "assets"
STUDIO = DATA / "studio"
#: The Studio's whole published payload (pages, replays, packs, sessions; GLBs counted apart).
STUDIO_BUDGET_BYTES = 6_000_000


def _export_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "export_studio_web", REPO / "scripts" / "export_studio_web.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_studio_web"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ex() -> Any:
    return _export_module()


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """A reference-controller tournament (two presses, a gripper and a cup): finalists of every kind."""
    rep = run_task_tournament(
        two_press_task(),
        candidates=[gripper(), cup()],
        workers=1,
        n_conditions=2,
        n_screen=1,
        controllers="reference",
    )
    return json.loads(json.dumps(rep.to_dict()))


def _finalist(report: dict[str, Any], kind: str) -> dict[str, Any]:
    found = [f for f in report["finalists"] if f["setup"]["kind"] == kind]
    if not found:
        pytest.skip(f"no {kind} finalist in this tournament")
    return found[0]


def _record(report: dict[str, Any], finalist: dict[str, Any], i: int = 0) -> sr.Recording:
    spec = task_from_dict(report["task"])
    cond = report["conditions"][i]
    c = StudioCondition(
        cond["condition_id"], cond["seed"], {k: tuple(v) for k, v in cond["aim_offsets_mm"].items()}
    )
    run = next(r for r in finalist["runs"] if r["condition_id"] == c.condition_id)
    return sr.record_run(
        setup_from_dict(finalist["setup"]),
        spec,
        c,
        expected_traj_sha=run["traj_sha"],
        controllers="reference",
    )


# --------------------------------------------------------------------------- recording


@pytest.mark.parametrize("kind", ["single", "tool_change", "multi_head"])
def test_a_recording_reproduces_the_scored_run_bit_for_bit(report: dict[str, Any], kind: str) -> None:
    f = _finalist(report, kind)
    rec = _record(report, f)
    assert rec.result.traj_sha == f["runs"][0]["traj_sha"]
    assert rec.frames and len(rec.times) == len(rec.frames)
    assert all(b >= a for a, b in zip(rec.times, rec.times[1:], strict=False))
    assert rec.times[-1] <= rec.result.cycle_time_s + 1e-6
    keys = set(rec.bodies)
    assert "wrist_flange" in keys and any(k.startswith("obj_") for k in keys)
    assert {rec.bodies[k].tool for k in keys if rec.bodies[k].category == "tool"} <= set(
        range(len(f["setup"]["tools"]))
    )


def test_a_wrong_traj_sha_is_refused(report: dict[str, Any]) -> None:
    f = _finalist(report, "single")
    spec = task_from_dict(report["task"])
    with pytest.raises(sr.ReplayMismatchError, match="traj_sha"):
        sr.record_run(
            setup_from_dict(f["setup"]),
            spec,
            StudioCondition("nominal", 0),
            expected_traj_sha="0" * 64,
            controllers="reference",
        )


def test_recordings_are_deterministic_in_process_and_match_a_spawned_worker(report: dict[str, Any]) -> None:
    f = _finalist(report, "tool_change")
    a, b = _record(report, f), _record(report, f)
    assert a.result.traj_sha == b.result.traj_sha
    assert a.times == b.times
    for fa, fb in zip(a.frames, b.frames, strict=True):
        assert fa.keys() == fb.keys()
        assert all(np.array_equal(fa[k], fb[k]) for k in fa)
    job = StudioJob(a.setup, a.task_spec, a.condition, controllers="reference")
    spawned = run_task_jobs([job, job], workers=2)
    assert {r.traj_sha for r in spawned} == {a.result.traj_sha}


# --------------------------------------------------------------------------- lanes


def test_a_tool_change_lane_parks_the_idle_tool_on_the_stand_and_writes_a_valid_v2_header(
    report: dict[str, Any], tmp_path: Path
) -> None:
    f = _finalist(report, "tool_change")
    rec = _record(report, f)
    store = sr.staging_store(tmp_path / "raw")
    replay, used, fps = sr.build_lane(rec, store, hero="t1", lane="recommended", code_version="0123456789ab")
    h = replay.header
    assert fps in sr.FPS_LADDER and h["fps"] == fps
    assert h["provenance"]["product_mode"] is True and h["meta"]["label"] == PRODUCT_MODE_LABEL
    assert h["meta"]["design_id"] == f["setup"]["setup_id"] and h["meta"]["traj_sha"] == rec.result.traj_sha
    assert "created_utc" not in h["provenance"]  # a re-export is byte for byte the same
    assert used and set(h["assets"]) == used
    assert any(v["asset"] for v in h.get("static_visuals", []))  # the tool stand
    segs = h["meta"]["segments"]
    assert segs[0]["change"] == "none" and any(s["change"] == "tool_change" for s in segs)
    for s0, s1 in itertools.pairwise(segs):
        assert s1["start_s"] == pytest.approx(s0["end_s"] + s1["change_s"], abs=1e-3)
    assert not check_header(h, replay.frames)
    stem = tmp_path / "lane"
    write_replay(replay, stem)
    back = read_replay(stem)
    assert back.n_frames == h["n_frames"] == math.floor(rec.result.cycle_time_s * fps + 1e-6) + 1
    # the first tool rests on the stand while the second works (the stand is behind the table)
    second = next(s for s in segs if s["tool"] != segs[0]["tool"])
    k = round((second["start_s"] + 0.5) * fps)
    names = [b["name"] for b in h["bodies"]]
    first_tool = [i for i, n in enumerate(names) if n.startswith(f"finger_t{segs[0]['tool']}_")]
    assert first_tool and all(abs(back.frames[k, i, 1] - sr.STAND_Y_M) < 0.2 for i in first_tool)
    assert not find_library_brands(json.dumps(h))
    raw = json.dumps(h)
    assert "/home/" not in raw and "/tmp/" not in raw
    # a second export of the same run gives the same assets and the same header, byte for byte
    again, used2, _ = sr.build_lane(
        _record(report, f),
        sr.staging_store(tmp_path / "raw2"),
        hero="t1",
        lane="recommended",
        code_version="0123456789ab",
    )
    assert used2 == used and json.dumps(again.header) == raw
    assert np.array_equal(again.frames, replay.frames)


def test_body_meshes_are_the_same_on_every_call() -> None:
    geoms = [
        {"type": "box", "half_size": [0.01, 0.02, 0.005], "pos": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1]},
        {
            "type": "capsule",
            "radius": 0.004,
            "half_length": 0.01,
            "pos": [0, 0, 0.02],
            "quat_xyzw": [0, 0, 0, 1],
        },
        {
            "type": "capsule",
            "radius": 0.004,
            "half_length": 0.0,
            "pos": [0.02, 0, 0],
            "quat_xyzw": [0, 0, 0, 1],
        },
        {
            "type": "cylinder",
            "radius": 0.006,
            "half_height": 0.01,
            "pos": [0, 0.03, 0],
            "quat_xyzw": [0.7071068, 0, 0, 0.7071068],
        },
    ]
    a, b = sr.mesh_of_geoms(geoms), sr.mesh_of_geoms(geoms)
    assert np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces)
    lo, hi = a.bounds
    assert lo[2] == pytest.approx(-0.006, abs=1e-6) and hi[2] == pytest.approx(0.034, abs=1e-6)


def test_rigid_bodies_merge_and_resting_bodies_become_static() -> None:
    n = 20
    t = np.linspace(0.0, 1.0, n)
    moving = np.zeros((n, 7))
    moving[:, 0] = t
    moving[:, 6] = 1.0
    angle = 0.5 * t
    spun = moving.copy()
    spun[:, 3:] = np.stack([np.zeros(n), np.zeros(n), np.sin(angle / 2), np.cos(angle / 2)], axis=1)
    rider = moving.copy()
    rider[:, 2] = 0.1  # rides 10 cm above the moving body
    resting = np.zeros((n, 7))
    resting[:, 6] = 1.0
    tracks = np.stack([moving, rider, spun, resting], axis=1)
    info = {k: sr._BodyInfo(k, "tool", 0, [], False) for k in "abcd"}
    groups, static = sr._groups(tracks, list("abcd"), info)
    assert static == [3]
    assert [g.members for g in groups] == [[0, 1], [2]]


def test_lane_choices_and_the_shown_condition() -> None:
    def run(cid: str, done: bool) -> dict[str, Any]:
        return {"condition_id": cid, "completed": done}

    rep = {
        "recommendation": {"recommendation": {"setup_id": "r"}, "no_winner": None},
        "best_single": "b",
        "conditions": [{"condition_id": "nominal"}, {"condition_id": "place01"}],
        "finalists": [{"setup": {"setup_id": "r"}, "runs": [run("nominal", False), run("place01", True)]}],
    }
    assert sr.lane_choices(rep) == [
        ("recommended", "Recommended", "r"),
        ("best_single", "Best single off-the-shelf", "b"),
    ]
    assert sr._lane_condition(rep, ["r"])["condition_id"] == "place01"
    rep["recommendation"] = {
        "recommendation": None,
        "no_winner": {"closest": [{"setup_id": "b"}, {"setup_id": "c"}]},
    }
    assert [c[0] for c in sr.lane_choices(rep)] == ["closest", "best_single"]
    assert sr.lane_choices(rep)[0][2] == "c"


# --------------------------------------------------------------------------- blocker probe


def _push_task() -> TaskSpec:
    objects = (
        ObjectInstance.of(
            "box", "part", "Box", ObjectPose(80.0, 0.0, 0.0), size_x_mm=30, size_y_mm=40, size_z_mm=50
        ),
        ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0.0, -200.0, 0.0)),
    )
    steps = (
        Step.of(
            "press_box",
            "push",
            "box",
            "part",
            "Press the box down hard",
            [SuccessCriterion("force_n", at_least=900.0), SuccessCriterion("tilt_deg", at_most=30.0)],
        ),
    )
    return TaskSpec("A hard press", "A press nothing completes.", objects, steps, source="test")


def test_the_blocker_probe_reruns_the_screen_and_reports_the_peak_per_family() -> None:
    cands = [candidate_of(gripper(), "preset"), candidate_of(cup(), "preset")]
    rep = run_task_tournament(
        _push_task(), candidates=cands, workers=1, n_conditions=2, n_screen=2, controllers="reference"
    )
    d = json.loads(json.dumps(rep.to_dict()))
    # reference results rank nothing; give the probe the stop the ranking would report
    d["recommendation"]["recommendation"] = None
    d["recommendation"]["ranking"] = [{"stops_at": {"press_box": {"runs": 2, "failure_modes": {}}}}]
    doc = sr.blocker_probe(d, workers=1, candidates=cands, log=lambda _: None)
    assert doc is not None
    assert doc["step_id"] == "press_box" and doc["quantity"] == "force_n" and doc["at_least"] == 900.0
    assert doc["n_runs"] == len(doc["conditions"]) * doc["n_candidates"]
    fams = {r["family"] for r in doc["families"]}
    assert fams <= {"catalog_gripper", "suction"} and fams
    assert doc["max_peak"] == max(r["peak"] for r in doc["families"]) < 900.0
    assert "900 N" in doc["text"]


# --------------------------------------------------------------------------- export helpers


def test_criterion_text_reason_groups_and_headline(ex: Any) -> None:
    assert ex.criterion_text("angle_deg", 95.0, None) == "angle at least 95 deg"
    assert ex.criterion_text("torque_nm", 3.0, 8.0) == "torque 3 to 8 N m"
    assert ex.criterion_text("turns", 0.25, None) == "turns at least 0.25 turns"
    assert ex.criterion_text("in_target", 1.0, None) == "ends inside the target"
    assert ex.criterion_text("jam", None, 0.0) == "no jam"
    assert ex._reason_group("pack: the part is 100 mm across the closing axis; the gripper opens 85 mm") == (
        "Opens narrower than the part"
    )
    assert ex._reason_group("something new") == "something new"
    rec = {"setup_id": "r", "kind": "multi_head", "completed_runs": 8, "n_runs": 8}
    best = {"setup_id": "b", "completed_runs": 0, "n_runs": 8}
    assert ex.headline(rec, best, None, {}) == (
        "A two-tool head completes the task in 8 of 8 runs; the best single off-the-shelf tool in 0 of 8 runs."
    )
    assert ex.headline(None, best, {"stops_at": "x"}, {"x": "step 2, twist the coupling"}) == (
        "No setup completes the task; the closest setup stops at step 2, twist the coupling."
    )


def test_check_doc_refuses_brands_urls_outside_url_fields_and_local_paths(ex: Any) -> None:
    ok = {"label": PRODUCT_MODE_LABEL, "links": [{"name": "Tool 0", "url": "https://example.org/x"}]}
    assert ex.check_doc(ok, "page") == []
    assert ex.check_doc({"name": "a Robotiq gripper"}, "p")
    assert ex.check_doc({"text": "see https://example.org"}, "p")
    assert ex.check_doc({"text": "saved in /home/someone/x"}, "p")
    assert ex.check_doc({"url": "http://insecure.example"}, "p")


def test_a_session_is_scrubbed_before_it_is_published(ex: Any) -> None:
    doc = {
        "format": "rtt-agent-session",
        "label": PRODUCT_MODE_LABEL,
        "events": [{"type": "tool.result", "error": "file /home/someone/runs/x/report.json is missing"}],
        "messages": [{"role": "system", "content": "the system prompt"}],
    }
    out = ex.scrub_session(doc)
    assert "messages" not in out and out["kind"] == "recorded"
    assert "/home/" not in json.dumps(out)
    with pytest.raises(ex.ExportError):
        ex.scrub_session({"format": "something else"})


# --------------------------------------------------------------------------- published data


def _published() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not (STUDIO / "index.json").is_file():
        pytest.skip("no Studio data exported yet (scripts/export_studio_web.py)")
    index = json.loads((STUDIO / "index.json").read_text(encoding="utf-8"))
    pages = [json.loads((DATA / h["page"]).read_text(encoding="utf-8")) for h in index["heroes"]]
    return index, pages


def test_published_pages_are_labelled_brand_free_and_consistent(ex: Any) -> None:
    index, pages = _published()
    assert index["schema"] == "rtt-web-studio" and index["label"] == PRODUCT_MODE_LABEL
    assert [h["hero"] for h in index["heroes"]] == [p["hero"] for p in pages]
    assert ex.check_doc(index, "index") == []
    for entry, page in zip(index["heroes"], pages, strict=True):
        assert page["schema"] == "rtt-web-studio-hero" and page["label"] == PRODUCT_MODE_LABEL
        assert ex.check_doc(page, page["hero"]) == []
        assert entry["outcome"] == page["outcome"] and entry["headline"] == page["headline"]
        assert (page["recommendation"] is None) == (page["outcome"] == "no_recommendation")
        assert page["plan"]["task_id"] == entry["task_id"] == page["provenance"]["task_id"]
        steps = [s["id"] for s in page["plan"]["steps"]]
        assert page["matrix"]["steps"] == steps
        for row in page["matrix"]["rows"]:
            assert set(row["cells"]) == set(steps)
        if page["recommendation"] is not None:
            rec = page["recommendation"]
            assert [s["step_id"] for s in rec["steps"]] == steps
            assert rec["verify_on_hardware"] == any(s["fidelity"] == "approximation" for s in rec["steps"])
        else:
            assert page["no_winner"]["closest"]


def test_published_lanes_match_their_replays_and_assets() -> None:
    _, pages = _published()
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))["assets"]
    for page in pages:
        if not page["lanes"]:
            continue
        conds = {lane["condition_id"] for lane in page["lanes"]}
        assert len(conds) == 1  # one condition for every lane
        for lane in page["lanes"]:
            stem = DATA / lane["replay"]
            header = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            assert not check_header(header)
            assert header["provenance"]["product_mode"] is True
            assert header["meta"]["design_id"] == lane["setup_id"]
            assert header["meta"]["traj_sha"] == lane["traj_sha"]
            assert header["fps"] == lane["fps"] and header["n_frames"] == lane["n_frames"]
            served = stem.with_suffix(".json").stat().st_size + stem.with_suffix(".bin").stat().st_size
            assert served == lane["bytes"] <= sr.LANE_BUDGET_BYTES
            assert stem.with_suffix(".bin").stat().st_size == header["n_frames"] * header["n_bodies"] * 28
            total = 0
            for aid, entry in header["assets"].items():
                glb = ASSETS / f"{aid}.glb"
                data = glb.read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["sha256"] and len(data) == entry["bytes"]
                assert manifest.get(aid, {}).get("sha256") == entry["sha256"]
                assert not find_library_brands(entry["attribution"])
                total += len(data)
            assert total == lane["asset_bytes"]


def test_published_build_packs_and_sessions() -> None:
    _, pages = _published()
    for page in pages:
        pack = page["build_pack"]
        if pack is not None:
            path = DATA / pack["zip"]
            data = path.read_bytes()
            assert len(data) == pack["bytes"] <= 2_000_000
            assert hashlib.sha256(data).hexdigest() == pack["sha256"]
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
            folder = {n.split("/", 1)[1] for n in names if "/" in n} or names
            assert {f["path"] for f in pack["files"]} <= folder | names
            assert any(n.endswith("manifest.json") for n in names)
        else:
            assert page["build_pack_note"]
        if page["session"] is not None:
            doc = json.loads((DATA / page["session"]["file"]).read_text(encoding="utf-8"))
            assert doc["format"] == "rtt-agent-session" and doc["label"] == PRODUCT_MODE_LABEL
            assert "messages" not in doc
            raw = json.dumps(doc)
            assert "/home/" not in raw and not find_library_brands(raw)


def test_the_studio_payload_stays_within_its_budget() -> None:
    if not STUDIO.is_dir():
        pytest.skip("no Studio data exported yet")
    total = sum(p.stat().st_size for p in STUDIO.rglob("*") if p.is_file())
    assert total <= STUDIO_BUDGET_BYTES


def test_a_summary_cut_at_the_cap_is_shown_up_to_its_last_sentence(ex: Any) -> None:
    cut = ("A sentence that ends. " * 30)[: ex.SUMMARY_CAP_CHARS - 4] + " Verify on ha"
    assert ex.trim_capped(cut).endswith("ends.") and len(ex.trim_capped(cut)) < len(cut)
    assert ex.trim_capped("Short and whole.") == "Short and whole."
