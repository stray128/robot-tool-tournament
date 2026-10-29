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
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rtt.effectors.types import PRODUCT_MODE_LABEL
from rtt.library.brands import find_library_brands
from rtt.studio import replay as sr
from rtt.studio.buildpack import setup_from_dict, task_from_dict
from rtt.studio.episode import StudioCondition, StudioJob, run_task_jobs
from rtt.studio.search import candidate_of, run_task_tournament
from rtt.studio.types import HeadMount, ObjectInstance, ObjectPose, Setup, Step, SuccessCriterion, TaskSpec
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


def _record(
    report: dict[str, Any], finalist: dict[str, Any], i: int = 0, fps: int = sr.RECORD_FPS
) -> sr.Recording:
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
        fps=fps,
        controllers="reference",
    )


def _record_kind(report: dict[str, Any], kind: str, fps: int = sr.RECORD_FPS) -> sr.Recording:
    """A recording of a ``single`` or ``tool_change`` finalist of the tournament, or of a
    ``multi_head`` setup (a head switch) of the same task, which the tournament does not field."""
    if kind != "multi_head":
        return _record(report, _finalist(report, kind), fps=fps)
    setup = Setup.multi_head(
        [(gripper(), HeadMount(-45.0, 70.0)), (cup(), HeadMount(45.0, 70.0))],
        {"press_carton": 0, "press_flat": 1},
    )
    return sr.record_run(
        setup, two_press_task(), StudioCondition("nominal", 0), fps=fps, controllers="reference"
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
    assert fps == sr.RECORD_FPS == h["fps"] == 100  # the nominal rate the keys were taken from
    assert h["provenance"]["product_mode"] is True and h["meta"]["label"] == PRODUCT_MODE_LABEL
    assert h["meta"]["design_id"] == f["setup"]["setup_id"] and h["meta"]["traj_sha"] == rec.result.traj_sha
    assert "created_utc" not in h["provenance"]  # a re-export is byte for byte the same
    assert used and set(h["assets"]) == used
    assert any(v["asset"] for v in h.get("static_visuals", []))  # the tool stand
    segs = h["meta"]["segments"]
    assert segs[0]["change"] == "none" and any(s["change"] == "tool_change" for s in segs)
    for s0, s1 in itertools.pairwise(segs):
        assert s1["start_s"] == pytest.approx(s0["end_s"] + s1["change_s"], abs=1e-3)
    assert not sr.check_lane_header(h, replay.frames)
    stem = tmp_path / "lane"
    sr.write_lane(replay, stem)
    back = sr.read_lane(stem)
    assert back.n_frames == h["n_frames"] == len(h["t"]) < len(rec.frames)  # keyframes, not every sample
    assert h["t"][-1] == pytest.approx(rec.times[-1], abs=1e-6)
    # the first tool rests on the stand while the second works (the stand is behind the table)
    second = next(s for s in segs if s["tool"] != segs[0]["tool"])
    k = int(np.searchsorted(h["t"], second["start_s"] + 0.5))
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


# --------------------------------------------------------------------------- keyframes and cuts


def _vertex_error_m(drawn: np.ndarray, truth: np.ndarray, verts: np.ndarray) -> float:
    """Worst displacement of ``verts`` (body frame) between two world poses (7,)."""
    a = verts @ sr._rot(drawn[3:]).T + drawn[:3]
    b = verts @ sr._rot(truth[3:]).T + truth[:3]
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def test_the_recorder_samples_at_100_hz_and_at_every_contact_change(report: dict[str, Any]) -> None:
    assert sr.RECORD_FPS == 100
    rec = _record(report, _finalist(report, "single"))
    assert rec.fps == 100
    times = np.asarray(rec.times)
    assert np.all(np.diff(times) > 0)
    # a frame within one control tick (2 ms) after every 10 ms frame time ...
    grid = np.arange(0.0, times[-1], 0.01)
    after = np.searchsorted(times, grid - 1e-9)
    assert np.all(times[after] - grid < 0.0021)
    # ... and one at the very tick every contact starts or stops: an impact lands between frame times
    assert rec.contact_log
    assert all(np.min(np.abs(times - when)) < 1e-9 for when, *_ in rec.contact_log)
    assert len(times) > len(grid)


def test_select_keys_keeps_the_kinks_and_drops_the_straight_runs() -> None:
    n = 501
    t = np.arange(n) / 100.0
    x = np.interp(t, [0, 2, 3, 5], [0.0, 1.0, 1.0, 0.0])  # out, rest, back: kinks at 2 s and 3 s
    track = np.zeros((n, 1, 7))
    track[:, 0, 0] = x
    track[:, 0, 6] = 1.0
    keys = sr.select_keys(t, np.zeros(n, dtype=int), track, tol_m=1e-3)
    assert keys.tolist() == [0, 200, 300, 500]


def test_select_keys_bounds_the_error_keeps_forced_frames_and_never_spans_a_segment() -> None:
    n = 600
    t = np.arange(n) / 100.0
    w = 1.3  # a body going round a circle while turning: the error of a straight chord is large
    track = np.zeros((n, 2, 7))
    track[:, 0, 0] = 0.2 * np.cos(w * t)
    track[:, 0, 1] = 0.2 * np.sin(w * t)
    track[:, 0, 5] = np.sin(w * t / 2)
    track[:, 0, 6] = np.cos(w * t / 2)
    track[:, 1, 6] = 1.0
    track[:, 1, 2] = 0.3  # a body that never moves needs no frames of its own
    scene = (np.arange(n) >= 300).astype(int)
    pts = [np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]), np.zeros((1, 3))]
    keys = sr.select_keys(t, scene, track, points=pts, forced=[137], tol_m=1e-3)
    assert {0, 137, 299, 300, n - 1} <= set(keys.tolist())
    assert len(keys) < n // 4
    # an interpolation that follows the rules of the player stays within tolerance at every sample
    drawn = sr.sample_lane(track[keys].astype(np.float32), t[keys], [int(np.searchsorted(keys, 300))], t)
    for s in range(n):
        assert _vertex_error_m(drawn[s, 0], track[s, 0], pts[0]) <= 1e-3 + 1e-6
    # a looser tolerance keeps fewer frames
    assert len(sr.select_keys(t, scene, track, points=pts, forced=[137], tol_m=5e-3)) < len(keys)


def test_sample_lane_holds_a_pose_across_a_cut_and_interpolates_inside_a_segment() -> None:
    frames = np.zeros((4, 1, 7), np.float32)
    frames[:, 0, 6] = 1.0
    frames[:, 0, 0] = [0.0, 1.0, 10.0, 10.0]
    t = np.array([0.0, 1.0, 5.0, 6.0])
    got = sr.sample_lane(frames, t, [2], np.array([0.5, 1.0, 3.0, 4.999, 5.0, 5.5, 9.0, -1.0]))
    assert got[:, 0, 0].tolist() == pytest.approx([0.5, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 0.0])
    # the shortest way round: two key rotations on either side of a half turn do not spin the long way
    q = np.zeros((2, 1, 7), np.float32)
    q[0, 0, 3:] = [0.0, 0.0, np.sin(0.1), np.cos(0.1)]
    q[1, 0, 3:] = -np.array([0.0, 0.0, np.sin(-0.1), np.cos(-0.1)])  # the same rotation, other sign
    mid = sr.sample_lane(q, np.array([0.0, 1.0]), [], np.array([0.5]))[0, 0, 3:]
    assert abs(mid[2]) < 0.1 and abs(abs(mid[3]) - 1.0) < 0.01


def test_a_quantised_lane_writes_the_words_it_was_measured_with(
    report: dict[str, Any], tmp_path: Path
) -> None:

    rec = _record(report, _finalist(report, "single"))
    store = sr.staging_store(tmp_path / "raw")
    roomy = sr.build_lane(rec, store, hero="t1", lane="recommended", code_version="0123456789ab")[0]
    lane = sr.build_lane(
        rec,
        store,
        hero="t1",
        lane="recommended",
        code_version="0123456789ab",
        budget=sr.lane_bytes(roomy) - 1,
    )[0]
    assert lane.header["quantised"] is True and lane.words is not None
    assert "quat_encoding" not in lane.header and "qi" not in lane.header
    assert lane.words.shape == (lane.n_frames, lane.header["n_bodies"], 7)  # px py pz qa qb qc qi
    assert np.array_equal(sr.dequantise_poses(lane.words), lane.frames)
    sr.write_lane(lane, tmp_path / "recommended")
    on_disk = np.frombuffer((tmp_path / "recommended.bin").read_bytes(), dtype="<i2")
    assert np.array_equal(on_disk.reshape(lane.words.shape), lane.words)
    header = json.loads((tmp_path / "recommended.json").read_text(encoding="utf-8"))
    assert not sr.check_lane_header(header)


def test_contact_times_settle_chatter_into_onsets_and_releases() -> None:
    log = [
        (1.00, True, "a", "b"),
        (1.01, False, "a", "b"),  # a flicker of the same pair: one contact
        (1.02, True, "a", "b"),
        (2.00, False, "a", "b"),
        (3.00, True, "c", "d"),  # a blip shorter than the hold: no contact
        (3.01, False, "c", "d"),
        (4.00, True, "e", "world"),  # still touching at the end of the run: an onset only
    ]
    assert sr.contact_times(log) == pytest.approx([1.0, 2.0, 4.0])
    assert sr.contact_times(log, hold_s=0.0) == pytest.approx(sorted({t for t, *_ in log}))


@pytest.mark.parametrize("kind", ["tool_change", "multi_head"])
def test_a_key_lands_exactly_on_each_segment_start(report: dict[str, Any], tmp_path: Path, kind: str) -> None:
    rec = _record_kind(report, kind)
    replay, _, _ = sr.build_lane(
        rec, sr.staging_store(tmp_path / "raw"), hero="t1", lane="recommended", code_version="0123456789ab"
    )
    h = replay.header
    segs = h["meta"]["segments"]
    assert len(segs) >= 2 and len(h["cuts"]) == len(segs) - 1
    scene_starts = [i for i in range(1, len(rec.times)) if rec.scene_of_frame[i] != rec.scene_of_frame[i - 1]]
    assert len(scene_starts) == len(h["cuts"])
    for c, s, first in zip(h["cuts"], segs[1:], scene_starts, strict=True):
        assert h["t"][c] == pytest.approx(s["start_s"], abs=1e-3)  # a frame exactly at the segment's start
        assert h["t"][c] == pytest.approx(rec.times[first], abs=1e-6)
        assert h["t"][c - 1] == pytest.approx(
            rec.times[first - 1], abs=1e-6
        )  # the last frame of the one before
    assert h["t"][0] == 0.0 and h["cuts"] == sorted(h["cuts"]) and 0 not in h["cuts"]
    # at a cut the previous segment's tool and the next segment's tool are one frame apart in the
    # list and the player holds the first until the second: poses differ across it, never blend
    assert not sr.check_lane_header(h, replay.frames)


def test_keys_keep_every_body_within_1_mm_of_the_sim(report: dict[str, Any], tmp_path: Path) -> None:
    """The lane's interpolation at 60 Hz against the simulation at every 2 ms control tick: the
    farthest vertex of every drawn body, tool change and head switch included."""
    for kind in ("single", "tool_change", "multi_head"):
        rec = _record_kind(report, kind)
        replay, _, _ = sr.build_lane(
            rec, sr.staging_store(tmp_path / kind), hero="t1", lane="recommended", code_version="0123456789ab"
        )
        dense = _record_kind(report, kind, fps=500)  # the simulation itself: every control tick
        keys = sr._key_order(rec.bodies)
        tracks = sr._parked_tracks(rec, keys, range(len(rec.frames)))
        groups, static = sr._groups(tracks, keys, rec.bodies)
        truth = sr._parked_tracks(dense, keys, range(len(dense.frames)))
        h = replay.header
        t_dense = np.asarray(dense.times)
        verts = {k: sr.mesh_of_geoms(rec.bodies[k].geoms).vertices for k in keys}
        cut_times = [(h["t"][c - 1], h["t"][c]) for c in h["cuts"]]
        tau = np.arange(0.0, h["t"][-1], 1 / 60.0)
        tau = np.asarray(
            [x for x in tau if not any(a <= x < b for a, b in cut_times)]
        )  # nothing simulated there
        reps = sr.sample_lane(replay.frames, np.asarray(h["t"]), h["cuts"], tau)
        owner = {m: gi for gi, g in enumerate(groups) for m in g.members}
        worst = 0.0
        for i, x in enumerate(tau):
            j = int(np.clip(np.searchsorted(t_dense, x, side="right") - 1, 0, len(t_dense) - 2))
            a = (x - t_dense[j]) / (t_dense[j + 1] - t_dense[j])
            for b, key in enumerate(keys):
                if b not in owner and b not in static:
                    continue
                tr = sr.sample_lane(truth[j : j + 2, b : b + 1], np.array([0.0, 1.0]), [], np.array([a]))[
                    0, 0
                ]
                if b in static:
                    drawn = tracks[0, b]
                else:
                    g = groups[owner[b]]
                    pos, rot = sr._relative(tracks[0, g.rep], tracks[0, b])
                    drawn = sr._compose(reps[i, owner[b]].astype(np.float64), pos, rot)
                worst = max(worst, _vertex_error_m(drawn, tr, verts[key]))
        assert worst <= 1e-3, f"{kind}: the farthest vertex is {1000 * worst:.3f} mm from the simulation"
        assert h["meta"]["keyframes"]["max_error_vs_recording_mm"] <= 1.0


def test_contact_onset_and_release_get_a_frame(report: dict[str, Any], tmp_path: Path) -> None:
    f = _finalist(report, "single")
    rec = _record(report, f)
    assert rec.contact_log, "the presses touch their objects"
    settled = sr.contact_times(rec.contact_log)
    assert settled
    replay, _, _ = sr.build_lane(
        rec, sr.staging_store(tmp_path / "raw"), hero="t1", lane="recommended", code_version="0123456789ab"
    )
    t = np.asarray(replay.header["t"])
    for when in settled:
        assert np.min(np.abs(t - when)) <= 0.012, f"no frame near the contact event at {when:.3f} s"


def test_a_lane_is_quantised_before_its_keys_are_decimated(report: dict[str, Any], tmp_path: Path) -> None:
    f = _finalist(report, "tool_change")
    rec = _record(report, f)
    store = sr.staging_store(tmp_path / "raw")  # a recording remembers the assets it made in its store

    def lane(budget: int) -> sr.LaneReplay:
        return sr.build_lane(
            rec,
            store,
            hero="t1",
            lane="recommended",
            code_version="0123456789ab",
            budget=budget,
        )[0]

    roomy = lane(10_000_000)
    assert roomy.header.get("quantised", False) is False and roomy.header["dtype"] == "float32"
    assert roomy.header["meta"]["keyframes"]["tolerance_mm"] == 1.0
    size_f = sr.lane_bytes(roomy)
    squeezed = lane(size_f - 1)
    assert squeezed.header["quantised"] is True and squeezed.header["dtype"] == "int16"
    assert squeezed.header["pose_fields"] == ["px", "py", "pz", "qa", "qb", "qc", "qi"]
    assert roomy.n_frames <= squeezed.n_frames <= 1.1 * roomy.n_frames  # no frame was dropped for it
    assert squeezed.header["meta"]["keyframes"]["tolerance_mm"] == 1.0
    size_q = sr.lane_bytes(squeezed)
    n_b = squeezed.header["n_bodies"]
    assert size_q <= size_f - 1
    assert squeezed.n_frames * n_b * 14 <= 0.5 * roomy.n_frames * n_b * 28 * 1.1  # 14 bytes a pose, not 28
    assert squeezed.header["meta"]["keyframes"]["max_error_vs_recording_mm"] <= 1.0
    # only when int16 poses do not fit either is the tolerance relaxed
    tight = lane(size_q - 1)
    assert tight.header["quantised"] is True and tight.n_frames < squeezed.n_frames
    assert tight.header["meta"]["keyframes"]["tolerance_mm"] > 1.0
    assert (
        tight.header["meta"]["keyframes"]["max_error_vs_recording_mm"]
        <= tight.header["meta"]["keyframes"]["tolerance_mm"]
    )
    assert sr.lane_bytes(tight) <= size_q - 1
    # the file the writer makes is the size the budget counted
    sr.write_lane(squeezed, tmp_path / "recommended")  # the stem is the lane id, as the export names it
    served = (tmp_path / "recommended.json").stat().st_size + (tmp_path / "recommended.bin").stat().st_size
    assert served == sr.lane_bytes(squeezed)
    back = sr.read_lane(tmp_path / "recommended")
    assert np.array_equal(back.frames, squeezed.frames)  # what was measured is what a player reads
    with pytest.raises(sr.ReplayMismatchError, match="does not fit"):
        lane(2_000)


def test_check_lane_catches_a_page_entry_that_disagrees_with_its_replay(ex: Any) -> None:
    header = {"n_frames": 3, "fps": 100, "t": [0.0, 1.0, 5.0], "cuts": [2]}
    doc = {
        "bytes": 1000,
        "n_frames": 3,
        "fps": 100,
        "duration_s": 5.0,
        "segments": [{"start_s": 0.0}, {"start_s": 5.0}],
    }
    assert ex.check_lane(doc, header, "x") == []
    assert ex.check_lane(dict(doc, bytes=sr.LANE_BUDGET_BYTES + 1), header, "x")
    assert ex.check_lane(dict(doc, duration_s=4.0), header, "x")
    assert ex.check_lane(dict(doc, n_frames=4), header, "x")
    assert ex.check_lane(dict(doc, segments=[{"start_s": 0.0}, {"start_s": 4.0}]), header, "x")
    assert ex.check_lane(dict(doc, segments=[{"start_s": 0.0}]), header, "x")  # a cut without a change
    assert (
        ex.check_lane({k: v for k, v in doc.items() if k != "duration_s"}, {"n_frames": 3, "fps": 100}, "x")
        == []
    )


def test_exported_lanes_describe_their_keyframed_replays(
    report: dict[str, Any], ex: Any, tmp_path: Path
) -> None:
    lanes = sr.export_lanes(
        "t1", report, sr.staging_store(tmp_path / "raw"), code_version="0123456789ab", log=lambda _: None
    )
    assert lanes
    for built in lanes:
        h, d = built.replay.header, built.doc
        assert d["n_frames"] == h["n_frames"] == len(h["t"]) and d["fps"] == sr.RECORD_FPS == h["fps"]
        assert d["duration_s"] == pytest.approx(h["t"][-1], abs=1e-3)
        assert (
            d["quantised"] is h["quantised"]
            and d["max_error_vs_recording_mm"] == h["meta"]["keyframes"]["max_error_vs_recording_mm"]
        )
        d["bytes"] = sr.lane_bytes(built.replay)
        assert ex.check_lane(d, h, f"t1/{built.stem}") == []


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


def test_published_lanes_match_their_replays_and_assets(ex: Any) -> None:
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
            assert not sr.check_lane_header(header)
            assert header["provenance"]["product_mode"] is True
            assert header["meta"]["design_id"] == lane["setup_id"]
            assert header["meta"]["traj_sha"] == lane["traj_sha"]
            assert header["fps"] == lane["fps"] and header["n_frames"] == lane["n_frames"]
            served = stem.with_suffix(".json").stat().st_size + stem.with_suffix(".bin").stat().st_size
            assert served == lane["bytes"] <= sr.LANE_BUDGET_BYTES
            words = 7 * 2 if header.get("quantised") else 7 * 4  # int16 x 7, or float32 x 7, per pose
            assert stem.with_suffix(".bin").stat().st_size == header["n_frames"] * header["n_bodies"] * words
            if "t" in header:  # a keyframed lane: the page entry agrees with it, with a frame at each cut
                assert not ex.check_lane(lane, header, lane["replay"])
                assert (
                    len(header["t"]) == header["n_frames"]
                    and header["meta"]["keyframes"]["max_error_vs_recording_mm"]
                    <= header["meta"]["keyframes"]["tolerance_mm"]
                )
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


def test_the_export_serves_the_v2_heroes_from_the_v2_recording(ex: Any) -> None:
    """Each hero is named by its case id, and its plan comes from the Studio v2 recording, where it is
    valid; the published pages are only those heroes (a v1 route the site redirects has no page)."""
    from rtt.studio.engineer.gate import RECORDED_GATE

    rows = {
        r["case_id"]: r
        for r in json.loads((RECORDED_GATE / "manifest.json").read_text(encoding="utf-8"))["cases"]
    }
    assert ex.HEROES and all(hero == case_id for hero, case_id in ex.HEROES)
    for _, case_id in ex.HEROES:
        assert rows[case_id]["status"] == "valid"
        assert ex.load_case(case_id)["task_id"] == rows[case_id]["task_id"]
    if STUDIO.is_dir():
        assert ex.stale_heroes(DATA) == []
        index = json.loads((STUDIO / "index.json").read_text(encoding="utf-8"))
        assert [h["hero"] for h in index["heroes"]] == [
            h for h, _ in ex.HEROES if (STUDIO / f"{h}.json").is_file()
        ]


def test_a_hero_no_longer_listed_is_stale(ex: Any, tmp_path: Path) -> None:
    studio = tmp_path / "studio"
    studio.mkdir()
    for name in ("index.json", "h1.json", "h3.json", f"{ex.HEROES[0][0]}.json"):
        (studio / name).write_text("{}", encoding="utf-8")
    assert ex.stale_heroes(tmp_path) == ["h1", "h3"]
    assert ex.stale_heroes(tmp_path / "nowhere") == []


def test_a_partial_export_keeps_the_assets_of_the_heroes_it_leaves(ex: Any, tmp_path: Path) -> None:
    """A partial export (``--heroes`` a subset) keeps the other heroes' pages, so it keeps the GLBs
    their published lanes draw: only a Studio asset no published lane draws any more is pruned. It
    used to prune every Studio asset the exported heroes did not serve, the other heroes' too."""
    import hashlib

    from rtt.studio.replay import STUDIO_LABEL_PREFIXES, publish_assets

    data, assets = tmp_path / "data", tmp_path / "assets"
    kept, served_hero = ex.HEROES[0][0], ex.HEROES[-1][0]
    (data / "studio" / "replays" / kept).mkdir(parents=True)
    (data / "studio" / f"{kept}.json").write_text(
        json.dumps({"lanes": [{"replay": f"studio/replays/{kept}/recommended"}]}), encoding="utf-8"
    )
    blobs = {name: name.encode() * 8 for name in ("kept", "orphan", "new")}
    ids = {name: hashlib.sha256(b).hexdigest()[:16] for name, b in blobs.items()}
    (data / "studio" / "replays" / kept / "recommended.json").write_text(
        json.dumps({"assets": {ids["kept"]: {}}}), encoding="utf-8"
    )
    keep = ex.kept_assets(data, [h for h, _ in ex.HEROES if h != served_hero])
    assert keep == {ids["kept"]}
    label = {"label": f"{STUDIO_LABEL_PREFIXES[0]}x"}
    assets.mkdir()
    for name in ("kept", "orphan"):
        (assets / f"{ids[name]}.glb").write_bytes(blobs[name])
    (assets / "manifest.json").write_text(
        json.dumps({"assets": {ids["kept"]: label, ids["orphan"]: label}}), encoding="utf-8"
    )
    publish_assets({ids["new"]: (blobs["new"], label)}, assets, keep=keep)
    left = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))["assets"]
    assert sorted(left) == sorted([ids["kept"], ids["new"]])
    assert sorted(p.stem for p in assets.glob("*.glb")) == sorted([ids["kept"], ids["new"]])
    (data / "studio" / "replays" / kept / "recommended.json").unlink()
    with pytest.raises(ex.ExportError, match="has no header"):
        ex.kept_assets(data, [kept])
