"""The replay quality gate (Studio v2): a lane that does not faithfully show the simulation
is never exported, and the advisory vision review that needs two independent models to agree.

Physics uses the same reference-controller tournament as ``tests/test_studio_web.py`` (two presses,
a gripper and a cup): a replay must reproduce any scored run, so the reference controllers are what
the gate is tested on. Nothing here calls a network service: the reviewers are stubs.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rtt.studio import replay as sr
from rtt.studio import replay_qa as qa
from rtt.studio.buildpack import setup_from_dict, task_from_dict
from rtt.studio.episode import StudioCondition
from rtt.studio.search import run_task_tournament
from studio_fixtures import cup, gripper, two_press_task

REPO = Path(__file__).resolve().parents[1]


def _script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ex() -> Any:
    return _script("export_studio_web")


@pytest.fixture(scope="module")
def vision() -> Any:
    return _script("replay_vision_review")


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
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


def _record(report: dict[str, Any], finalist: dict[str, Any], fps: int = sr.RECORD_FPS) -> sr.Recording:
    cond = report["conditions"][0]
    c = StudioCondition(
        cond["condition_id"], cond["seed"], {k: tuple(v) for k, v in cond["aim_offsets_mm"].items()}
    )
    return sr.record_run(
        setup_from_dict(finalist["setup"]),
        task_from_dict(report["task"]),
        c,
        fps=fps,
        controllers="reference",
    )


def _lane_build(rec: sr.Recording, store: Any, lane: str = "recommended", **kw: Any) -> sr.LaneBuild:
    replay, used, _ = sr.build_lane(rec, store, hero="t1", lane=lane, code_version="0123456789ab", **kw)
    doc = {
        "id": lane,
        "setup_id": rec.setup.setup_id,
        "condition_id": rec.condition.condition_id,
        "traj_sha": rec.result.traj_sha,
    }
    return sr.LaneBuild(replay=replay, doc=doc, assets=used, stem=lane, recording=rec)


def _assets(lane: sr.LaneBuild, store: Any) -> dict[str, bytes]:
    return {aid: store.raw_bytes(aid) for aid in lane.assets}


def _tampered(
    lane: sr.LaneBuild, *, header: dict[str, Any] | None = None, frames: Any = None
) -> sr.LaneBuild:
    replay = sr.LaneReplay(
        frames=np.array(lane.replay.frames if frames is None else frames),
        header=copy.deepcopy(lane.replay.header if header is None else header),
    )
    return dataclasses.replace(lane, replay=replay)


@pytest.fixture(scope="module")
def built(
    report: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, tuple[sr.LaneBuild, Any]]:
    out = {}
    for kind in ("single", "tool_change"):
        store = sr.staging_store(tmp_path_factory.mktemp(f"raw_{kind}"))
        out[kind] = (_lane_build(_record(report, _finalist(report, kind)), store), store)
    return out


@pytest.fixture(scope="module")
def verdicts(report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]]) -> dict[str, qa.QAResult]:
    return {k: qa.check_lane(report, lane, assets=_assets(lane, store)) for k, (lane, store) in built.items()}


# --------------------------------------------------------------------------- the gate passes a good lane


@pytest.mark.parametrize("kind", ["single", "tool_change"])
def test_a_faithful_lane_passes_every_check(verdicts: dict[str, qa.QAResult], kind: str) -> None:
    res = verdicts[kind]
    assert res.passed, res.problems
    assert res.problems == []
    m = res.metrics
    assert m["interpolation"]["max_error_vs_sim_mm"] <= 2.0
    assert "max_error_mm" not in m["interpolation"]
    # measured against the simulation itself: a 500 Hz re-run (every 2 ms control tick), not the 100 Hz recording
    assert m["interpolation"]["against_hz"] == 500
    assert m["interpolation"]["truth_tick_ms"] == pytest.approx(2.0, abs=0.05)
    assert m["jumps"]["max_jump_mm"] <= 50.0 and m["jumps"]["max_drawn_minus_sim_mm"] <= 5.0
    assert m["glb"]["max_deviation_mm"] <= 0.05
    assert m["traj_sha"]["reproduced"] is True
    assert set(m) >= set(qa.CHECKS)


def test_the_thresholds_are_the_briefed_ones() -> None:
    assert qa.INTERP_TOL_M == 2e-3 and qa.JUMP_TOL_M == 50e-3 and qa.JUMP_SLACK_M == 5e-3
    assert qa.PEN_FLOOR_M == 1e-3 and qa.PEN_SLACK_M == 0.5e-3 and qa.GLB_TOL_M == 5e-5
    assert qa.DISPLAY_FPS == 60 and qa.QA_RECORD_FPS == 500
    assert qa.CHECKS == ("traj_sha", "cuts", "interpolation", "jumps", "penetration", "glb")


# --------------------------------------------------------------------------- each check fails a bad lane


def test_a_lane_interpolated_across_a_cut_fails(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]]
) -> None:
    lane, store = built["tool_change"]
    header = lane.replay.header
    assert header["cuts"], "the tool change lane has a cut"
    # the header forgets the cut: the player would slide the tool across the change
    forgotten = _tampered(lane, header=dict(copy.deepcopy(header), cuts=[]))
    res = qa.check_lane(report, forgotten, assets=_assets(lane, store), checks=["cuts"])
    assert not res.passed
    assert any(p.startswith("cuts:") and "interpolat" in p for p in res.problems), res.problems
    # the cut is listed one frame late: the player would interpolate over the first frame of the new tool
    late = _tampered(lane, header=dict(copy.deepcopy(header), cuts=[header["cuts"][0] + 1]))
    assert not qa.check_lane(report, late, checks=["cuts"]).passed
    # a lane with no frame at the end of the segment (the keyframer thinned across the cut) fails too
    c = header["cuts"][0]
    t = [x for i, x in enumerate(header["t"]) if i != c - 1]
    thinned = _tampered(
        lane,
        header=dict(copy.deepcopy(header), t=t, n_frames=len(t), cuts=[c - 1]),
        frames=np.delete(lane.replay.frames, c - 1, axis=0),
    )
    assert not qa.check_lane(report, thinned, checks=["cuts"]).passed
    # and the real one is fine
    assert qa.check_lane(report, lane, checks=["cuts"]).passed


def test_a_jump_between_display_frames_fails(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]]
) -> None:
    lane, _ = built["single"]
    frames = np.array(lane.replay.frames)
    mid = len(frames) // 2
    frames[mid:, 0, 0] += 0.2  # a body teleports 200 mm and stays there, in the middle of a segment
    res = qa.check_lane(
        report, _tampered(lane, frames=frames), checks=["jumps", "interpolation"], dense=lane.recording
    )
    assert not res.passed
    assert any(p.startswith("jumps:") for p in res.problems), res.problems
    assert res.metrics["jumps"]["max_jump_mm"] > 50.0 and res.metrics["jumps"]["max_drawn_minus_sim_mm"] > 5.0
    assert any(p.startswith("interpolation:") for p in res.problems), res.problems


def test_a_jump_at_a_cut_is_not_a_jump(verdicts: dict[str, qa.QAResult]) -> None:
    res = verdicts["tool_change"]
    assert res.metrics["jumps"]["cut_frames_exempt"] >= 1
    assert not [p for p in res.problems if p.startswith("jumps:")]


def test_the_jump_rule_is_on_display_frames_not_on_keys(
    built: dict[str, tuple[sr.LaneBuild, Any]], report: dict[str, Any]
) -> None:
    """Keys of a healthy lane can be far apart (a falling body, a long straight move): the rule is
    on the 60 Hz frames the player shows, so a long glide between two keys is not a jump."""
    lane, _ = built["single"]
    header = copy.deepcopy(lane.replay.header)
    n = 2
    header.update(t=[0.0, 2.0], n_frames=n, cuts=[])
    frames = np.array(lane.replay.frames[:n])
    frames[1] = frames[0]
    frames[1, :, 0] += 0.3  # every body glides 300 mm in two seconds: 2.5 mm a display frame
    res = qa.check_lane(
        report, _tampered(lane, header=header, frames=frames), checks=["jumps"], dense=lane.recording
    )
    assert res.passed, res.problems
    assert res.metrics["jumps"]["max_jump_mm"] < 5.0


def test_drawn_geometry_sunk_into_another_solid_fails(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]]
) -> None:
    lane, _ = built["single"]
    header = lane.replay.header
    fingers = [i for i, b in enumerate(header["bodies"]) if b["name"].startswith("finger")]
    frames = np.array(lane.replay.frames)
    frames[:, fingers, 2] -= (
        0.03  # the tool is drawn 30 mm lower than it was: through the table and the objects
    )
    res = qa.check_lane(report, _tampered(lane, frames=frames), checks=["penetration"], dense=lane.recording)
    assert not res.passed
    assert any(p.startswith("penetration:") for p in res.problems), res.problems
    assert res.metrics["penetration"]["max_excess_mm"] > 0.0


def _sunk(rec: sr.Recording, dz: float) -> sr.Recording:
    """``rec`` with every tool body ``dz`` metres lower (a simulation that lets the tool sink)."""
    shift = np.array([0.0, 0.0, dz, 0.0, 0.0, 0.0, 0.0])
    frames = [
        {k: (p + shift if rec.bodies[k].category == "tool" else p) for k, p in f.items()} for f in rec.frames
    ]
    return dataclasses.replace(rec, frames=frames)


def test_the_drawn_world_may_overlap_as_much_as_the_simulation_does_plus_half_a_millimetre(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]]
) -> None:
    """The tool tips rest 2.4 mm above the table in the unshifted run. Shift the simulation 5 mm down
    (2.6 mm of overlap): a replay drawn exactly that deep is faithful, one 1.5 mm deeper is not, and
    the same 2.6 mm against a simulation that never overlapped is."""
    lane, _ = built["single"]
    fingers = [i for i, b in enumerate(lane.replay.header["bodies"]) if b["name"].startswith("finger")]

    def drawn(dz: float) -> sr.LaneBuild:
        frames = np.array(lane.replay.frames)
        frames[:, fingers, 2] += dz
        return _tampered(lane, frames=frames)

    sim = _sunk(lane.recording, -0.005)
    same = qa.check_lane(report, drawn(-0.005), checks=["penetration"], dense=sim)
    assert same.passed, same.problems
    assert same.metrics["penetration"]["max_drawn_mm"] > 2.0  # it does overlap: as the simulation does
    deeper = qa.check_lane(report, drawn(-0.0065), checks=["penetration"], dense=sim)
    assert not deeper.passed and deeper.problems[0].startswith("penetration:")
    against_none = qa.check_lane(report, drawn(-0.005), checks=["penetration"], dense=lane.recording)
    assert not against_none.passed


def test_penetration_is_judged_against_the_simulation_not_against_zero(
    verdicts: dict[str, qa.QAResult],
) -> None:
    """The simulation itself lets solids sink a fraction of a millimetre into each other; a replay
    that shows exactly that is faithful. The allowance is max(1 mm, the simulation's own + 0.5 mm)."""
    pen = verdicts["single"].metrics["penetration"]
    assert pen["allowed_floor_mm"] == 1.0 and pen["slack_mm"] == 0.5
    assert pen["violations"] == 0


def test_a_wrong_traj_sha_fails_the_gate(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]]
) -> None:
    lane, _ = built["single"]
    other = copy.deepcopy(report)
    for f in other["finalists"]:
        if f["setup"]["setup_id"] == lane.recording.setup.setup_id:
            for r in f["runs"]:
                r["traj_sha"] = "0" * 64
    res = qa.check_lane(other, lane)
    assert not res.passed
    assert any(p.startswith("traj_sha:") for p in res.problems)
    # a failed reproduction ends the check: nothing measured against a run that is not the scored one
    assert not any(p.startswith(("interpolation:", "penetration:")) for p in res.problems)
    # a setup the report does not hold cannot be checked either
    gone = copy.deepcopy(report)
    gone["finalists"] = []
    assert not qa.check_lane(gone, lane).passed


def test_a_glb_that_differs_from_its_geoms_fails(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]]
) -> None:
    lane, store = built["single"]
    header = copy.deepcopy(lane.replay.header)
    moved = next(b for b in header["bodies"] if b["visuals"])
    moved["visuals"][0]["pos"][0] += 0.001  # drawn 1 mm from where its collision geoms are
    res = qa.check_lane(report, _tampered(lane, header=header), assets=_assets(lane, store), checks=["glb"])
    assert not res.passed
    assert any(p.startswith("glb:") for p in res.problems), res.problems
    assert res.metrics["glb"]["max_deviation_mm"] > 0.05
    # and a gate that was given no GLB bytes cannot vouch for them
    assert not qa.check_lane(report, lane, checks=["glb"]).passed


def test_check_lane_refuses_a_lane_with_no_recording(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]]
) -> None:
    lane, _ = built["single"]
    with pytest.raises(ValueError, match="recording"):
        qa.check_lane(report, dataclasses.replace(lane, recording=None))


# --------------------------------------------------------------------------- long lanes


def _long_recording(rec: sr.Recording, tiles: int, clones: int) -> sr.Recording:
    """``rec`` played forwards and backwards ``tiles`` times (continuous at the joins), with ``clones``
    more bodies that follow the wrist's track shifted in time (so none is rigid with another): the long,
    many-bodied lane a solar or tile job makes."""
    n = len(rec.frames)
    period = rec.times[-1] + 0.01
    src = "wrist_flange"
    times: list[float] = []
    frames: list[dict[str, np.ndarray]] = []
    for tile in range(tiles):
        order = range(n) if tile % 2 == 0 else range(n - 1, -1, -1)
        for j in order:
            times.append(tile * period + (rec.times[j] if tile % 2 == 0 else rec.times[-1] - rec.times[j]))
            frames.append(rec.frames[j])
    total = len(frames)
    out_frames = []
    for i, frame in enumerate(frames):
        f = dict(frame)
        for c in range(clones):
            pose = np.array(frames[(i + 37 * (c + 1)) % total][src])
            pose[0] += 0.03 * (c + 1)
            f[f"obj_clone{c}"] = pose
        out_frames.append(f)
    bodies = dict(rec.bodies)
    for c in range(clones):
        bodies[f"obj_clone{c}"] = dataclasses.replace(
            rec.bodies[src], key=f"obj_clone{c}", category="object", tool=None, label=f"clone{c}"
        )
    return dataclasses.replace(
        rec, times=times, frames=out_frames, scene_of_frame=[0] * total, bodies=bodies, contact_log=[]
    )


def test_a_long_lane_passes_after_quantisation(report: dict[str, Any], tmp_path: Path) -> None:
    """A long, many-bodied recording does not fit the lane budget as float32; quantised to int16 at
    the same 1 mm keys it does, and the lane still passes the gate's 2 mm fidelity check."""
    base = _record(report, _finalist(report, "single"))
    long = _long_recording(base, tiles=8, clones=20)
    store = sr.staging_store(tmp_path / "raw")
    roomy = _lane_build(long, store, budget=10**9)
    assert roomy.replay.header["quantised"] is False
    assert sr.lane_bytes(roomy.replay) > sr.LANE_BUDGET_BYTES, "the lane is over budget as float32"
    lane = _lane_build(long, store)
    h = lane.replay.header
    assert h["quantised"] is True and h["dtype"] == "int16"
    assert sr.lane_bytes(lane.replay) <= sr.LANE_BUDGET_BYTES
    assert h["meta"]["keyframes"]["tolerance_mm"] == 1.0  # quantised first, not decimated
    # it was measured the way a player reads it (int16 poses), against the recording it was made from
    res = qa.check_lane(
        report,
        lane,
        assets=_assets(lane, store),
        checks=["cuts", "interpolation", "jumps", "glb"],
        dense=long,
    )
    assert res.passed, res.problems
    assert res.metrics["interpolation"]["max_error_vs_sim_mm"] <= 2.0
    assert res.metrics["interpolation"]["quantised"] is True


# --------------------------------------------------------------------------- recording ticks, cuts and contacts


def test_every_control_tick_is_kept_for_30_ms_after_a_contact_change_and_a_segment_start(
    report: dict[str, Any],
) -> None:
    assert sr.SETTLE_S == 0.030
    rec = _record(report, _finalist(report, "tool_change"))
    tick = float(np.min(np.diff(_record(report, _finalist(report, "single"), fps=500).times)))
    times = np.asarray(rec.times)
    scene = np.asarray(rec.scene_of_frame)
    starts = [times[0], *times[1:][scene[1:] != scene[:-1]]]
    onsets = [t for t, *_ in rec.contact_log]
    assert onsets and len(starts) >= 2
    for t0 in [*starts, *onsets[:40]]:
        inside = times[(times >= t0 - 1e-9) & (times <= t0 + sr.SETTLE_S - 1.5 * tick)]
        assert len(inside) >= 2 and np.max(np.diff(inside)) < 1.5 * tick, f"ticks missing after {t0:.3f} s"


def test_quantising_a_body_beyond_the_int16_range_skips_that_candidate(
    report: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(sr.QuantiseRangeError, match="3.27 m"):
        sr.quantise_poses(np.array([[4.0, 0, 0, 0, 0, 0, 1]]))
    assert issubclass(sr.QuantiseRangeError, ValueError)
    rec = _record(report, _finalist(report, "single"))
    far = next(k for k in sorted(rec.bodies) if rec.bodies[k].category == "tool")
    for frame in rec.frames:  # one tool body works 5 m away: float32 holds that, int16 cannot
        frame[far] = frame[far] + np.array([5.0, 0, 0, 0, 0, 0, 0])
    store = sr.staging_store(tmp_path / "raw")

    def lane(budget: int) -> sr.LaneReplay:
        return sr.build_lane(
            rec, store, hero="t1", lane="recommended", code_version="0123456789ab", budget=budget
        )[0]

    roomy = lane(sr.LANE_BUDGET_BYTES)
    assert roomy.header["quantised"] is False  # it fits, so the unusable int16 candidate is never needed
    # too tight for 1 mm float32: the ladder goes on down as float32 (int16 is skipped, not an error)
    tight = lane(sr.lane_bytes(roomy) - 1)
    assert tight.header["quantised"] is False and tight.header["meta"]["keyframes"]["tolerance_mm"] > 1.0
    with pytest.raises(sr.ReplayMismatchError, match=r"t1/recommended.*does not fit.*3\.27 m"):
        lane(500)


def test_quantise_poses_refuses_non_finite_poses() -> None:
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            sr.quantise_poses(np.array([[bad, 0, 0, 0, 0, 0, 1]]))
        with pytest.raises(ValueError, match="finite"):
            sr.quantise_poses(np.array([[0, 0, 0, bad, 0, 0, 1]]))


def test_a_lane_carries_the_rate_it_was_recorded_at(report: dict[str, Any], tmp_path: Path) -> None:
    rec = _record(report, _finalist(report, "single"), fps=50)
    assert rec.fps == 50
    replay, _, fps = sr.build_lane(
        rec, sr.staging_store(tmp_path / "raw"), hero="t1", lane="recommended", code_version="0123456789ab"
    )
    assert fps == 50 and replay.header["fps"] == 50
    assert replay.header["meta"]["keyframes"]["recorded_fps"] == 50


def test_the_tolerance_ladder_stops_at_2_mm_and_then_refuses(report: dict[str, Any], tmp_path: Path) -> None:
    assert sr.KEY_TOLERANCES_M[-1] == 2e-3 and max(sr.KEY_TOLERANCES_M) == 2e-3
    assert sr.KEY_TOLERANCES_M[0] == sr.KEY_TOL_M
    rec = _record(report, _finalist(report, "single"))
    store = sr.staging_store(tmp_path / "raw")
    with pytest.raises(sr.ReplayMismatchError, match=r"t1/recommended.*does not fit.*2\.0 mm"):
        sr.build_lane(rec, store, hero="t1", lane="recommended", code_version="0123456789ab", budget=500)


def test_the_error_against_the_recording_is_named_for_what_it_measures(
    built: dict[str, tuple[sr.LaneBuild, Any]],
) -> None:
    kf = built["single"][0].replay.header["meta"]["keyframes"]
    assert "max_error_mm" not in kf
    assert 0.0 < kf["max_error_vs_recording_mm"] <= kf["tolerance_mm"]


def test_exported_lane_entries_use_the_renamed_field(report: dict[str, Any], tmp_path: Path) -> None:
    lanes = sr.export_lanes(
        "t1", report, sr.staging_store(tmp_path / "raw"), code_version="0123456789ab", log=lambda _: None
    )
    assert lanes
    for built_lane in lanes:
        assert "max_error_mm" not in built_lane.doc
        assert (
            built_lane.doc["max_error_vs_recording_mm"]
            == built_lane.replay.header["meta"]["keyframes"]["max_error_vs_recording_mm"]
        )
        assert built_lane.recording is not None and built_lane.doc["fps"] == built_lane.recording.fps


def test_export_lanes_collects_a_sha_mismatch_when_asked(report: dict[str, Any], tmp_path: Path) -> None:
    stale = copy.deepcopy(report)
    for f in stale["finalists"]:
        for r in f["runs"]:
            r["traj_sha"] = "0" * 64
    store = sr.staging_store(tmp_path / "raw")
    with pytest.raises(sr.ReplayMismatchError, match="t1"):
        sr.export_lanes("t1", stale, store, code_version="0123456789ab", log=lambda _: None)
    failures: list[str] = []
    lanes = sr.export_lanes(
        "t1", stale, store, code_version="0123456789ab", log=lambda _: None, failures=failures
    )
    assert lanes == [] and failures
    assert all(f.startswith("t1/") and "traj_sha" in f for f in failures)


def test_a_lane_that_cannot_be_built_is_named_not_fatal_when_failures_are_collected(
    report: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def too_big(rec: Any, store: Any, *, hero: str, lane: str, **kw: Any) -> Any:
        raise sr.ReplayMismatchError(f"lane {hero}/{lane} does not fit 480000 bytes: the smallest is 600000")

    monkeypatch.setattr(sr, "build_lane", too_big)
    failures: list[str] = []
    lanes = sr.export_lanes(
        "t1",
        report,
        sr.staging_store(tmp_path / "raw"),
        code_version="0123456789ab",
        log=lambda _: None,
        failures=failures,
    )
    assert lanes == [] and failures
    assert all(f.startswith("t1/") and ": build: " in f and "does not fit" in f for f in failures)
    with pytest.raises(sr.ReplayMismatchError, match="does not fit"):  # without failures it is an error
        sr.export_lanes(
            "t1", report, sr.staging_store(tmp_path / "raw2"), code_version="0123456789ab", log=lambda _: None
        )


def test_every_tick_is_kept_while_a_body_moves_fast(
    report: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A body that moves faster than FAST_M_S between control ticks opens the settle window: the ticks
    a straight line between 100 Hz samples would not follow are all recorded."""
    monkeypatch.setattr(sr, "FAST_M_S", 0.03)
    finalist = _finalist(report, "single")
    rec = _record(report, finalist)
    dense = _record(report, finalist, fps=500)
    t = np.asarray(dense.times)
    keys = sorted(k for k in dense.bodies if k in dense.frames[0])
    pos = np.stack([np.stack([f[k][:3] for k in keys]) for f in dense.frames])
    speed = np.max(np.linalg.norm(np.diff(pos, axis=0), axis=2), axis=1) / np.diff(t)
    fast_ticks = t[1:][speed > 0.03]
    assert len(fast_ticks) > 20, "the presses move faster than 3 cm/s for a while"
    have = np.asarray(rec.times)
    missing = [x for x in fast_ticks if np.min(np.abs(have - x)) > 1e-9]
    assert not missing, f"{len(missing)} fast ticks are not in the recording"
    monkeypatch.setattr(sr, "FAST_M_S", float("inf"))
    sparse = _record(report, finalist)
    assert len(sparse.frames) < len(rec.frames)


def test_the_gate_passes_lanes_as_the_export_serves_them(
    ex: Any, report: dict[str, Any], tmp_path: Path
) -> None:
    store = sr.staging_store(tmp_path / "raw")
    lanes = sr.export_lanes("t1", report, store, code_version="0123456789ab", log=lambda _: None)
    served = sr.serve_assets(store, lanes, compress=False)  # the headers now point at served ids
    problems, results = ex.run_gate(
        [{"hero": "t1", "report": report, "lanes": lanes}], served, log=lambda _: None
    )
    assert problems == [], problems
    assert set(results["t1"]) == {lane.stem for lane in lanes}
    ex._write_qa(tmp_path / "qa", results)
    doc = json.loads((tmp_path / "qa" / "replay_qa.json").read_text(encoding="utf-8"))
    assert all(lane["passed"] for lane in doc["heroes"]["t1"].values())


def test_the_gate_decodes_the_meshopt_compressed_glbs_the_site_serves(
    report: dict[str, Any], tmp_path: Path
) -> None:
    from rtt.render.compress import compression_available

    if not compression_available():
        pytest.skip("needs Node and the web app's packages (pnpm install in web/)")
    store = sr.staging_store(tmp_path / "raw")
    lanes = sr.export_lanes("t1", report, store, code_version="0123456789ab", log=lambda _: None)
    served = sr.serve_assets(store, lanes, compress=True)
    assets = {aid: data for aid, (data, _) in served.items()}
    assert all(qa._compressed(d) for d in assets.values())
    for lane in lanes:
        res = qa.check_lane(report, lane, assets=assets, checks=["glb"])
        assert res.passed and res.metrics["glb"]["max_deviation_mm"] <= 0.05, res.problems


def test_a_compressed_glb_without_the_web_packages_is_a_clear_failure(
    report: dict[str, Any],
    built: dict[str, tuple[sr.LaneBuild, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane, store = built["single"]
    monkeypatch.setattr(qa, "_compressed", lambda data: True)  # as if the build had compressed them
    res = qa.check_lane(
        report, lane, assets=_assets(lane, store), checks=["glb"], web_dir=tmp_path / "no_web"
    )
    assert not res.passed
    assert res.problems[0].startswith("glb:") and "Node" in res.problems[0]


def test_the_schema_comment_points_at_the_lane_header_check() -> None:
    schema = json.loads((REPO / "src" / "rtt" / "sim" / "replay_v2.schema.json").read_text(encoding="utf-8"))
    comment = schema["$comment"]
    assert "checked by rtt.studio.replay.check_lane_header" in comment
    # the v1 header check is named as the step that check_lane_header applies first
    assert "rtt.sim.replay.check_header, which check_lane_header applies first" in comment


# --------------------------------------------------------------------------- the export script


def test_main_exits_cleanly_when_a_replay_does_not_reproduce(
    ex: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise sr.ReplayMismatchError("h1/recommended: traj_sha: the re-run gives 123, the report 456")

    monkeypatch.setattr(ex, "export", boom)
    assert ex.main(["--reports", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "traj_sha" in err and "Traceback" not in err


def _fake_item(hero: str, lanes: list[Any]) -> dict[str, Any]:
    return {"hero": hero, "report": {}, "lanes": lanes}


def test_the_export_gate_names_every_failing_hero_lane_and_check(ex: Any) -> None:
    class L:
        def __init__(self, stem: str) -> None:
            self.stem = stem

    def check(report: Any, lane: Any, **kw: Any) -> qa.QAResult:
        if lane.stem == "recommended":
            return qa.QAResult(False, ["interpolation: 3.1 mm", "glb: 0.2 mm"], {})
        return qa.QAResult(True, [], {"x": 1})

    items = [_fake_item("h1", [L("recommended"), L("best_single")]), _fake_item("h2", [L("best_single")])]
    problems, results = ex.run_gate(items, {}, check=check, log=lambda _: None, stop_on_failure=False)
    assert problems == ["h1/recommended: interpolation: 3.1 mm", "h1/recommended: glb: 0.2 mm"]
    assert results["h1"]["best_single"].passed and results["h2"]["best_single"].passed


def test_the_export_writes_nothing_when_the_gate_fails(
    ex: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``export`` runs the gate before anything is published: a failing lane leaves the data and
    assets folders exactly as they were and names hero, lane and check."""
    data, assets = tmp_path / "data", tmp_path / "assets"
    data.mkdir()
    (data / "keep.txt").write_text("kept", encoding="utf-8")

    class Lane:
        stem = "recommended"
        replay = None
        recording = None

        def __init__(self) -> None:
            self.doc: dict[str, Any] = {}
            self.assets: set[str] = set()

    def fake_export_lanes(hero: str, rep: Any, store: Any, **kw: Any) -> list[Any]:
        return [Lane()]

    monkeypatch.setattr(sr, "export_lanes", fake_export_lanes)
    monkeypatch.setattr(sr, "serve_assets", lambda store, lanes, compress=True: {})
    monkeypatch.setattr(
        sr, "write_lane", lambda replay, stem: (_ for _ in ()).throw(AssertionError("wrote a lane"))
    )
    monkeypatch.setattr(
        ex,
        "load_report",
        lambda reports, case: ({"task_id": "0" * 64, "recommendation": {}}, tmp_path / "r.json"),
    )
    monkeypatch.setattr(ex, "load_case", lambda case_id: {})
    monkeypatch.setattr(
        ex,
        "run_gate",
        lambda items, served, **kw: (["h1_tote/recommended: interpolation: 3.1 mm over 2 mm"], {}),
    )
    with pytest.raises(ex.ExportError, match=r"h1_tote/recommended: interpolation: 3\.1 mm"):
        ex.export(
            tmp_path,
            data_dir=data,
            assets_dir=assets,
            heroes=["h1_tote"],
            packs=False,
            blocker=False,
            log=lambda _: None,
        )
    assert sorted(p.name for p in data.iterdir()) == ["keep.txt"] and not assets.exists()


# --------------------------------------------------------------------------- the vision review (advisory)


class _Stub:
    """A reviewer that flags the frames it is told to, makes no network call and costs ``cost`` a look."""

    def __init__(
        self,
        name: str,
        flags: set[str],
        cost: float = 0.0,
        fail_on: set[str] = frozenset(),  # type: ignore[assignment]
        severity: int = 2,
    ) -> None:
        self.name, self.flags, self.cost, self.fail_on = name, flags, cost, set(fail_on)
        self.severity = severity
        self.seen: list[str] = []

    def review(self, frame: Any, prompt: str) -> Any:
        self.seen.append(frame.frame_id)
        if frame.frame_id in self.fail_on:
            raise RuntimeError("the service is down")
        flagged = frame.frame_id in self.flags
        return {
            "interpenetration": flagged,
            "detached": False,
            "severity": self.severity if flagged else 0,
            "where": "the finger passes through the carton" if flagged else "",
            "confidence": 0.9,
            "cost_usd": self.cost,
        }


def _frames(vision: Any, n: int = 5) -> list[Any]:
    return [vision.FrameRef(f"f{i}", f"frame {i}", f"lane/{i}", 0.5 * i, b"\x89PNG") for i in range(n)]


def test_vision_review_needs_two_reviewers_to_flag(vision: Any, tmp_path: Path) -> None:
    a = _Stub("a", {"f1", "f2"})
    b = _Stub("b", {"f2", "f3"})
    guard = vision.SpendGuard(tmp_path / "spend.json", caps={"a": 10.0, "b": 10.0})
    review = vision.review_frames(_frames(vision), [a, b], guards={"a": guard, "b": guard})
    flagged = [f["frame_id"] for f in review["flagged"]]
    assert flagged == ["f2"]  # only the frame both reviewers flag
    assert review["advisory"] is True and review["blocks_export"] is False
    assert review["summary"]["frames"] == 5 and review["summary"]["flagged"] == 1
    assert review["summary"]["reviewers_disagreed"] == 2  # f1 (only a) and f3 (only b): not flags
    assert a.seen == b.seen == [f"f{i}" for i in range(5)]
    assert set(review["flagged"][0]["reviewers"]) == {"A", "B"}
    text = json.dumps(review)
    assert "claude" not in text.lower() and "openai" not in text.lower() and "gpt" not in text.lower()


def test_a_single_reviewer_cannot_flag_and_a_failing_one_does_not_block(vision: Any, tmp_path: Path) -> None:
    a = _Stub("a", {"f0", "f1", "f2"})
    b = _Stub("b", {"f0", "f1", "f2"}, fail_on={"f1"})
    g = vision.SpendGuard(tmp_path / "spend.json", caps={"a": 10.0, "b": 10.0})
    review = vision.review_frames(_frames(vision, 3), [a, b], guards={"a": g, "b": g})
    assert [f["frame_id"] for f in review["flagged"]] == ["f0", "f2"]  # f1: no second opinion, no flag
    assert review["summary"]["errors"] == 1 and review["blocks_export"] is False
    one = vision.review_frames(_frames(vision, 3), [a], guards={"a": g})
    assert one["flagged"] == [] and one["summary"]["reviewers"] == 1  # one reviewer never flags


def test_the_spend_guard_stops_before_a_cap_and_remembers_across_runs(vision: Any, tmp_path: Path) -> None:
    path = tmp_path / "runs" / "spend.json"
    guard = vision.SpendGuard(path, caps={"a": 1.0, "b": 1.0})
    guard.reserve("a", 0.4)
    guard.commit("a", 0.4, 0.3)
    assert guard.spent("a") == pytest.approx(0.3)
    again = vision.SpendGuard(path, caps={"a": 1.0, "b": 1.0})  # a new process reads the running total
    assert again.spent("a") == pytest.approx(0.3)
    again.reserve("a", 0.6)  # 0.3 + 0.6 <= 1.0
    with pytest.raises(vision.SpendCapReached):
        again.reserve("a", 0.5)  # 0.3 + 0.6 + 0.5 would pass the cap: refused before the call
    again.release("a", 0.6)
    again.reserve("a", 0.5)
    assert json.loads(path.read_text(encoding="utf-8"))["a"]["spent_usd"] == pytest.approx(0.3)
    # the reviews stop at the cap and say so; nothing is raised
    a = _Stub("a", {"f0"}, cost=0.4)
    b = _Stub("b", {"f0"}, cost=0.1)
    g = vision.SpendGuard(tmp_path / "s2.json", caps={"a": 1.0, "b": 5.0})
    review = vision.review_frames(
        _frames(vision, 6), [a, b], guards={"a": g, "b": g}, estimate_usd={"a": 0.4, "b": 0.1}
    )
    assert review["summary"]["stopped_by_spend_cap"] is True
    assert g.spent("a") <= 1.0 and len(a.seen) < 6
    assert review["blocks_export"] is False


def test_the_vision_script_reads_keys_at_runtime_and_never_stores_them(vision: Any, tmp_path: Path) -> None:
    keys = tmp_path / "keys.json"
    keys.write_text(
        json.dumps({"claude": "sk-ant-secret-value", "openai": "sk-openai-secret-value"}), encoding="utf-8"
    )
    loaded = vision.load_keys(keys)
    assert set(loaded) == {"claude", "openai"}
    assert "secret" not in repr(loaded) and "secret" not in str(loaded)  # the keys do not show up in reprs
    with pytest.raises(vision.ReviewSetupError, match="keys"):
        vision.load_keys(tmp_path / "missing.json")
    source = (REPO / "scripts" / "replay_vision_review.py").read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9_-]{16,}", source)  # no key is written into the script


def test_an_openai_model_is_chosen_from_the_listing_not_guessed(vision: Any) -> None:
    listing = ["gpt-4o", "gpt-5.4", "gpt-5.5-pro", "gpt-5.5", "text-embedding-3-large"]
    assert vision.choose_openai_model(listing) == "gpt-5.5"
    assert vision.choose_openai_model(listing, prefer="gpt-4o") == "gpt-4o"
    with pytest.raises(vision.ReviewSetupError, match="no vision-capable model"):
        vision.choose_openai_model(["text-embedding-3-large", "whisper-1"])
    with pytest.raises(vision.ReviewSetupError, match="gpt-9"):
        vision.choose_openai_model(listing, prefer="gpt-9")


def test_frames_are_chosen_around_cuts_and_contact_onsets(vision: Any) -> None:
    times = vision.frame_times(cuts_s=[(5.0, 6.0)], onsets_s=[1.0, 1.02, 3.0], duration_s=8.0, max_frames=10)
    assert 4.95 <= min(t for t in times if t > 4) and any(abs(t - 6.0) < 0.1 for t in times)
    assert any(abs(t - 1.0) < 0.1 for t in times) and any(abs(t - 3.0) < 0.1 for t in times)
    assert len(times) <= 10 and times == sorted(times) and len(set(times)) == len(times)
    assert len(vision.frame_times([], [], 8.0, max_frames=4)) <= 4


def test_lanes_are_drawn_around_their_cuts_and_contacts_for_review(
    vision: Any, built: dict[str, tuple[sr.LaneBuild, Any]], tmp_path: Path
) -> None:
    lane, _ = built["tool_change"]
    guard = vision.SpendGuard(tmp_path / "spend.json", caps={"a": 10.0, "b": 10.0})
    seen: list[Any] = []

    class Looker(_Stub):
        def review(self, frame: Any, prompt: str) -> Any:
            seen.append(frame)
            return super().review(frame, prompt)

    a, b = Looker("a", set()), Looker("b", set())
    try:
        review = vision.review_lanes(
            [("t1", lane)], [a, b], guards={"a": guard, "b": guard}, max_frames_per_lane=6, log=lambda _: None
        )
    except RuntimeError as exc:
        pytest.skip(f"no offscreen GL here: {exc}")
    frames = {f.frame_id: f for f in seen}
    assert 2 <= len(frames) <= 6
    assert all(f.png.startswith(b"\x89PNG") and len(f.png) > 2000 for f in frames.values())
    assert {f.reason for f in frames.values()} <= {"cut", "contact"} and "cut" in {
        f.reason for f in frames.values()
    }
    cut_time = lane.replay.header["t"][lane.replay.header["cuts"][0]]
    assert any(abs(f.time_s - cut_time) < 0.2 for f in frames.values())  # a frame just after the cut
    assert review["flagged"] == [] and review["blocks_export"] is False and review["advisory"] is True


# --------------------------------------------------------------------------- the relative jump rule


def _falling(rec: sr.Recording, drop: float) -> sr.Recording:
    """``rec`` with the carton, which rests, dropped ``drop`` metres in free fall from the start."""
    key, g = "obj_carton_body", 9.81
    t_fall = (2 * drop / g) ** 0.5
    frames = []
    for t, frame in zip(rec.times, rec.frames, strict=True):
        f = dict(frame)
        pose = np.array(rec.frames[0][key])
        pose[2] += max(drop - 0.5 * g * min(t, t_fall) ** 2, 0.0)
        f[key] = pose
        frames.append(f)
    return dataclasses.replace(rec, frames=frames, contact_log=[])


def test_an_honest_free_fall_passes_the_jump_rule_and_a_teleport_still_fails(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]], tmp_path: Path
) -> None:
    """A carton in free fall from 1 m moves 70 mm and more between display frames and a faithful lane
    shows exactly that: the rule is against the simulation's own displacement, not an absolute 50 mm."""
    base = _record(report, _finalist(report, "single"))
    store = sr.staging_store(tmp_path / "raw")
    for drop in (0.6, 1.0):
        fall = _falling(base, drop)
        lane = _lane_build(fall, store, lane=f"fall{int(drop * 10)}")
        res = qa.check_lane(report, lane, checks=["jumps", "interpolation"], dense=fall)
        assert res.passed, res.problems
        m = res.metrics["jumps"]
        assert m["max_jump_mm"] > 50.0, "the drawn fall is faster than an absolute 50 mm rule allows"
        assert m["jump_body"] == "obj_carton_body"
        assert m["max_drawn_minus_sim_mm"] < 5.0 and m["max_sim_jump_mm"] > 50.0
    assert m["max_jump_mm"] > 65.0  # from 1 m
    # a teleport is far over the simulation's displacement and still fails (and is not excused by speed)
    lane, _ = built["single"]
    frames = np.array(lane.replay.frames)
    frames[len(frames) // 2 :, 0, 0] += 0.2
    res = qa.check_lane(report, _tampered(lane, frames=frames), checks=["jumps"], dense=lane.recording)
    assert not res.passed and res.problems[0].startswith("jumps:")
    assert res.metrics["jumps"]["max_drawn_minus_sim_mm"] > 50.0


# --------------------------------------------------------------------------- pins for the gate thresholds


def test_a_27_mm_20_ms_snap_passes_against_the_recording_and_fails_against_the_500_hz_truth(
    report: dict[str, Any], tmp_path: Path
) -> None:
    """A jaw snapping open (27 mm in 20 ms) that the 100 Hz samples follow is 2.4 mm off in
    between: only a measure against the 500 Hz re-run sees it."""
    finalist = _finalist(report, "single")
    rec, dense = _record(report, finalist), _record(report, finalist, fps=500)

    def snap(r: sr.Recording) -> sr.Recording:
        frames = []
        for t, frame in zip(r.times, r.frames, strict=True):
            x = min(max((t - 5.0) / 0.02, 0.0), 1.0)
            f = dict(frame)
            pose = np.array(frame["obj_carton_body"])
            pose[2] += 0.027 * (3 * x * x - 2 * x**3)
            f["obj_carton_body"] = pose
            frames.append(f)
        return dataclasses.replace(r, frames=frames, contact_log=[])

    rec, dense = snap(rec), snap(dense)
    lane = _lane_build(rec, sr.staging_store(tmp_path / "raw"))
    against_recording = qa.check_lane(report, lane, checks=["interpolation"], dense=rec)
    assert against_recording.passed and against_recording.metrics["interpolation"]["against_hz"] == 100
    against_truth = qa.check_lane(report, lane, checks=["interpolation"], dense=dense)
    assert not against_truth.passed
    m = against_truth.metrics["interpolation"]
    assert (
        m["against_hz"] == 500
        and m["worst_body"] == "obj_carton_body"
        and 2.0 < m["max_error_vs_sim_mm"] < 3.0
    )


def test_a_1_5_mm_shift_passes_and_a_2_5_mm_shift_fails_the_interpolation_check(
    report: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2 mm threshold itself: with keys within 0.2 mm, a rigid 1.5 mm shift stays under it and a
    2.5 mm one does not."""
    monkeypatch.setattr(sr, "KEY_TOLERANCES_M", (2e-4,))
    rec = _record(report, _finalist(report, "single"))
    lane = _lane_build(rec, sr.staging_store(tmp_path / "raw"))

    def shifted(delta: float) -> qa.QAResult:
        frames = np.array(lane.replay.frames)
        frames[:, :, 2] += delta
        return qa.check_lane(report, _tampered(lane, frames=frames), checks=["interpolation"], dense=rec)

    assert (
        qa.check_lane(report, lane, checks=["interpolation"], dense=rec).metrics["interpolation"][
            "max_error_vs_sim_mm"
        ]
        < 0.3
    )
    under, over = shifted(0.0015), shifted(0.0025)
    assert under.passed and 1.3 < under.metrics["interpolation"]["max_error_vs_sim_mm"] < 1.7
    assert not over.passed and 2.3 < over.metrics["interpolation"]["max_error_vs_sim_mm"] < 2.7


def _sink_probe(lane: sr.LaneBuild) -> Any:
    """The lane's drawn poses, with a function that puts the first finger inside the tote at chosen
    display frames and runs the penetration check on the result."""
    d = qa._Drawn(lane)
    truth = qa._Truth(d, lane.recording)
    instants = d.instants()
    sim = truth.at(instants)
    drawn = d.keys_at(d.reps(instants))
    finger, tote = d.keys.index("t0/eff_f0_distal"), d.keys.index("obj_tote_frame")
    geom = lane.recording.bodies["obj_tote_frame"].geoms[0]
    mid = len(instants) // 2
    centre = drawn[mid, tote, :3] + sr._rot(drawn[mid, tote, 3:]) @ np.array(geom["pos"])

    def run(frames: list[int]) -> dict[str, Any]:
        poses = drawn.copy()
        for f in frames:
            poses[f, finger, :3] = centre
        problems, metrics = qa._check_penetration(d, truth, instants, poses, sim)
        return {"problems": problems, **metrics}

    return mid, run


def test_one_sunk_display_frame_passes_and_two_within_three_frames_fail(
    built: dict[str, tuple[sr.LaneBuild, Any]],
) -> None:
    """Beyond one display frame: a single frame deep in another solid is tolerated, two over the
    allowance within any three consecutive display frames (even with one between) fail."""
    mid, run = _sink_probe(built["single"][0])
    one = run([mid])
    assert (
        one["max_drawn_mm"] > 10.0
        and one["violations"] >= 1
        and one["failing_pairs"] == 0
        and not one["problems"]
    )
    two = run([mid, mid + 1])
    assert two["failing_pairs"] >= 1 and two["problems"][0].startswith("penetration:")
    apart = run([mid, mid + 2])  # one frame between them: still within three consecutive frames
    assert apart["failing_pairs"] >= 1
    far = run([mid, mid + 3])  # two frames between them: two separate blips
    assert far["failing_pairs"] == 0
    assert run([mid + 2 * i for i in range(6)])["failing_pairs"] >= 1  # alternating frames


def test_a_pair_of_bodies_is_one_pair_however_many_geoms_it_has(
    built: dict[str, tuple[sr.LaneBuild, Any]],
) -> None:
    lane, _ = built["single"]
    d = qa._Drawn(lane)
    world = qa._DrawnWorld(lane.recording, d.keys)
    pairs = {world.body_pair(i) for i in range(len(world.pairs))}
    assert len(pairs) <= len(world.pairs)
    assert all(a <= b for a, b in pairs)
    geoms_of = {k: sum(1 for g in world.gkey if g == k) for k in d.keys}
    assert max(geoms_of.values()) >= 1
    # the tote frame is drawn from several box geoms: its pair with the finger is one pair, not several
    tote = [
        i for i in range(len(world.pairs)) if world.body_pair(i) == ("obj_tote_frame", "t0/eff_f0_distal")
    ]
    assert len(tote) >= 1 and len({world.body_pair(i) for i in tote}) == 1


def test_a_drawn_overlap_under_the_one_millimetre_floor_passes_where_the_simulation_has_none(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]]
) -> None:
    """The floor of the allowance: drawn 0.8 mm into the table against a simulation that never touched it
    passes, 1.2 mm fails (the allowance is max(1 mm, simulated + 0.5 mm), not simulated + 0.5 mm)."""
    lane, _ = built["single"]
    fingers = [i for i, b in enumerate(lane.replay.header["bodies"]) if b["name"].startswith("finger")]

    def sunk(dz: float) -> qa.QAResult:
        frames = np.array(lane.replay.frames)
        frames[:, fingers, 2] += dz
        return qa.check_lane(
            report, _tampered(lane, frames=frames), checks=["penetration"], dense=lane.recording
        )

    shallow = sunk(-0.0032)  # the finger tips rest 2.4 mm above the table
    assert shallow.passed, shallow.problems
    assert 0.7 < shallow.metrics["penetration"]["max_drawn_mm"] < 0.9
    deeper = sunk(-0.0036)
    assert not deeper.passed and 1.1 < deeper.metrics["penetration"]["max_drawn_mm"] < 1.3


def test_a_500_hz_rerun_that_does_not_reproduce_fails_the_gate_on_its_own(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]]
) -> None:
    lane, _ = built["single"]
    rec = lane.recording
    assert qa.check_lane(
        report, lane, checks=["traj_sha"], dense=rec
    ).passed  # the lane's own sha is the scored one
    other = dataclasses.replace(rec, result=dataclasses.replace(rec.result, traj_sha="f" * 64))
    res = qa.check_lane(report, lane, checks=["traj_sha"], dense=other)
    assert not res.passed
    assert len(res.problems) == 1 and "500 Hz re-run" in res.problems[0]  # only the re-run differs


def test_a_failure_before_the_gate_stops_the_export_and_is_written_down(
    ex: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lane that did not reproduce its run (collected by export_lanes) refuses the export, writes
    nothing to the data or assets folders, does not spend the gate's re-runs on the other lanes, and
    still leaves replay_qa.json behind."""
    data, assets, qa_dir = tmp_path / "data", tmp_path / "assets", tmp_path / "qa"
    data.mkdir()
    (data / "keep.txt").write_text("kept", encoding="utf-8")

    def fake_export_lanes(hero: str, rep: Any, store: Any, *, failures: list[str], **kw: Any) -> list[Any]:
        failures.append(f"{hero}/recommended: traj_sha: the re-run gives 123, the report 456")
        return []

    def no_gate(*a: Any, **k: Any) -> Any:
        raise AssertionError("the gate ran after a lane had already failed")

    monkeypatch.setattr(sr, "export_lanes", fake_export_lanes)
    monkeypatch.setattr(sr, "serve_assets", lambda store, lanes, compress=True: {})
    monkeypatch.setattr(
        ex,
        "load_report",
        lambda reports, case: ({"task_id": "0" * 64, "recommendation": {}}, tmp_path / "r.json"),
    )
    monkeypatch.setattr(ex, "load_case", lambda case_id: {})
    monkeypatch.setattr(ex, "run_gate", no_gate)
    with pytest.raises(ex.ExportError, match=r"h1_tote/recommended: traj_sha: the re-run gives 123"):
        ex.export(
            tmp_path,
            data_dir=data,
            assets_dir=assets,
            qa_dir=qa_dir,
            heroes=["h1_tote"],
            packs=False,
            blocker=False,
            log=lambda _: None,
        )
    assert sorted(p.name for p in data.iterdir()) == ["keep.txt"] and not assets.exists()
    doc = json.loads((qa_dir / "replay_qa.json").read_text(encoding="utf-8"))
    assert (
        doc["failures"] == ["h1_tote/recommended: traj_sha: the re-run gives 123, the report 456"]
        and doc["heroes"] == {}
    )


def test_the_gate_stops_at_the_first_failing_lane(ex: Any) -> None:
    calls: list[str] = []

    class L:
        def __init__(self, stem: str) -> None:
            self.stem = stem

    def check(report: Any, lane: Any, **kw: Any) -> qa.QAResult:
        calls.append(lane.stem)
        return qa.QAResult(lane.stem != "a", [] if lane.stem != "a" else ["cuts: nope"], {"x": 1})

    items = [_fake_item("h1", [L("a"), L("b")]), _fake_item("h2", [L("c")])]
    problems, results = ex.run_gate(items, {}, check=check, log=lambda _: None)
    assert calls == ["a"] and problems == ["h1/a: cuts: nope"]
    assert results["h1"]["b"].metrics == {"skipped": "an earlier lane failed"} and "c" in results["h2"]
    calls.clear()
    problems, _ = ex.run_gate(items, {}, check=check, log=lambda _: None, stop_on_failure=False)
    assert calls == ["a", "b", "c"]


def test_the_glb_check_reports_a_hung_or_missing_node_as_a_problem(
    report: dict[str, Any], built: dict[str, tuple[sr.LaneBuild, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    lane, store = built["single"]
    for exc in (subprocess.TimeoutExpired("node", 300), FileNotFoundError("node"), ValueError("not a GLB")):

        def broken(assets: Any, web_dir: Any = None, _exc: Exception = exc) -> Any:
            raise _exc

        monkeypatch.setattr(qa, "glb_vertices", broken)
        res = qa.check_lane(report, lane, assets=_assets(lane, store), checks=["glb"])
        assert not res.passed and res.problems[0].startswith("glb:") and type(exc).__name__ in res.problems[0]


def test_a_reply_that_says_false_in_words_is_not_a_flag(vision: Any) -> None:
    parse = vision.parse_verdict
    base = '{"interpenetration": %s, "detached": %s, "severity": 3, "where": "x", "confidence": 0.9}'
    got = parse(base % ('"false"', '"No"'))
    assert got["interpenetration"] is False and got["detached"] is False
    got = parse(base % ('"true"', "false"))
    assert got["interpenetration"] is True and got["detached"] is False
    assert parse(base % ("0", "1"))["detached"] is True
    with pytest.raises(ValueError):
        parse(base % ('"maybe"', "false"))


def test_a_subtle_finding_is_not_a_flag_even_when_both_reviewers_make_it(vision: Any, tmp_path: Path) -> None:
    """MIN_SEVERITY: interpenetration true at severity 1 ('subtle') from both reviewers does not flag;
    at 2 ('clearly visible') it does."""
    g = vision.SpendGuard(tmp_path / "s.json", caps={"a": 1.0, "b": 1.0})
    guards = {"a": g, "b": g}
    subtle = vision.review_frames(
        _frames(vision, 2),
        [_Stub("a", {"f0", "f1"}, severity=1), _Stub("b", {"f0", "f1"}, severity=1)],
        guards=guards,
    )
    assert subtle["flagged"] == [] and subtle["summary"]["reviewers_disagreed"] == 0
    clear = vision.review_frames(
        _frames(vision, 2),
        [_Stub("a", {"f0"}, severity=2), _Stub("b", {"f0"}, severity=3)],
        guards=guards,
    )
    assert [f["frame_id"] for f in clear["flagged"]] == ["f0"]


# --------------------------------------------------------------------------- the spend guard fails closed


@pytest.mark.parametrize(
    "content",
    [
        '{"claude": {"spent_usd": 59.9, "calls": 5000}, "openai": {"spent_usd": 29.9, "ca',  # truncated
        "",
        "[1, 2, 3]",
        '{"claude": 12}',
        '{"claude": {"spent_usd": NaN}}',
        '{"claude": {"spent_usd": Infinity}}',
        '{"claude": {"spent_usd": -1.0}}',
        '{"claude": {"spent_usd": "0.5"}}',
        '{"claude": {"spent_usd": true}}',
        '{"claude": {"spent_usd": 1.0, "calls": -3}}',
    ],
)
def test_a_damaged_spend_file_stops_everything_and_is_never_reset(
    vision: Any, tmp_path: Path, content: str
) -> None:
    path = tmp_path / "spend.json"
    path.write_text(content, encoding="utf-8")
    g = vision.SpendGuard(path, caps={"claude": 60.0, "a": 1.0})
    for call in (
        lambda: g.spent("claude"),
        lambda: g.reserve("claude", 0.01),
        lambda: g.commit("claude", 0.0, 0.01),
    ):
        with pytest.raises(vision.SpendFileError, match="spend file unreadable: nothing is spent"):
            call()
    assert path.read_text(encoding="utf-8") == content  # the damaged total is left for a person to look at
    # the cost of a look that was paid for is not lost: it is counted in memory
    assert g._unrecorded.get("claude") == pytest.approx(0.01)
    # and a review does not make a single look
    a, b = _Stub("a", {"f0"}), _Stub("b", {"f0"})
    gg = vision.SpendGuard(path, caps={"a": 1.0, "b": 1.0})
    review = vision.review_frames(_frames(vision, 3), [a, b], guards={"a": gg, "b": gg})
    assert a.seen == b.seen == [] and review["summary"]["stopped_by_spend_cap"] is True
    assert "spend file unreadable" in review["summary"]["stop_reason"] and review["blocks_export"] is False


def test_a_missing_spend_file_is_zero_spent_and_is_made_on_the_first_commit(
    vision: Any, tmp_path: Path
) -> None:
    path = tmp_path / "deep" / "spend.json"
    g = vision.SpendGuard(path, caps={"a": 1.0})
    assert g.spent("a") == 0.0 and not path.exists()
    g.reserve("a", 0.5)
    g.commit("a", 0.5, 0.25)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": {"calls": 1, "spent_usd": 0.25}}


def test_a_commit_is_a_locked_read_modify_write(vision: Any, tmp_path: Path) -> None:
    import threading

    path = tmp_path / "spend.json"
    guard_for = lambda: vision.SpendGuard(path, caps={"a": 1e9})
    errors: list[BaseException] = []

    def work() -> None:
        try:
            g = guard_for()
            for _ in range(25):
                g.commit("a", 0.0, 0.001)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["a"]["calls"] == 200 and doc["a"]["spent_usd"] == pytest.approx(0.2)  # no update lost
    assert (tmp_path / "spend.json.lock").exists()  # the lock file next to it


def test_a_model_without_a_listed_price_is_refused_unless_a_price_is_given(
    vision: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(vision.ReviewSetupError, match="no price is listed for the model 'some-new-model'"):
        vision.price_of("some-new-model")
    assert vision.price_of("some-new-model", (1.0, 2.0)) == (1.0, 2.0)
    for bad in ((0.0, 1.0), (float("nan"), 1.0), (-1.0, 2.0)):
        with pytest.raises(vision.ReviewSetupError, match="two positive numbers"):
            vision.price_of("some-new-model", bad)
    # every model the second reviewer may be chosen from is listed, explicitly, at a deliberately high price
    for model in vision.OPENAI_PREFERENCE:
        assert vision.price_of(model) == (10.0, 40.0)
    source = (REPO / "scripts" / "replay_vision_review.py").read_text(encoding="utf-8")
    assert all(
        "[assumed]" in line for line in source.splitlines() if '"gpt-' in line and "(10.0, 40.0)" in line
    )
    # the first reviewer's model is priced before any network call is made
    monkeypatch.setattr(vision, "list_openai_models", lambda key: pytest.fail("the service was asked"))
    with pytest.raises(vision.ReviewSetupError, match="no price is listed"):
        vision.make_reviewers(
            {"claude": vision._Secret("k"), "openai": vision._Secret("k")}, claude_model="new-model"
        )
    assert vision._parse_price("1.5, 6") == (1.5, 6.0) and vision._parse_price(None) is None
    with pytest.raises(vision.ReviewSetupError):
        vision._parse_price("cheap")


def _git(*args: str, cwd: Path) -> None:
    import subprocess

    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_the_default_spend_file_is_shared_by_every_worktree_and_seeded_from_the_old_ones(
    vision: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    if shutil.which("git") is None:
        pytest.skip("needs git")
    monkeypatch.delenv("RTT_VISION_SPEND_FILE", raising=False)
    main, tree = tmp_path / "main", tmp_path / "tree"
    main.mkdir()
    _git("init", "-q", cwd=main)
    _git("commit", "-q", "--allow-empty", "-m", "start", cwd=main)
    _git("worktree", "add", "-q", str(tree), "-b", "side", cwd=main)
    want = (main / ".git" / "rtt" / "vision_spend.json").resolve()
    assert vision.default_spend_path(main).resolve() == vision.default_spend_path(tree).resolve() == want
    # the older per-checkout totals (this checkout's and the main one's) are added, once, into the shared file
    for root, spent in ((main, 0.0138), (tree, 0.5)):
        old = root / "runs" / "vision_review"
        old.mkdir(parents=True)
        (old / "spend.json").write_text(
            json.dumps({"claude": {"spent_usd": spent, "calls": 1}}), encoding="utf-8"
        )
    g = vision.default_spend_guard(tree)
    assert g.path.resolve() == want and g.spent("claude") == pytest.approx(0.5138)
    assert json.loads(want.read_text(encoding="utf-8"))["claude"]["calls"] == 2
    g.commit("claude", 0.0, 0.1)
    again = vision.default_spend_guard(main)  # the other worktree sees the same total, and does not re-seed
    assert again.spent("claude") == pytest.approx(0.6138)
    # an explicit file is used as it is and seeded from nothing
    monkeypatch.setenv("RTT_VISION_SPEND_FILE", str(tmp_path / "own.json"))
    assert vision.default_spend_path(tree) == tmp_path / "own.json"
    assert vision.default_spend_guard(tree).spent("claude") == 0.0
    # a damaged old file is not guessed at
    monkeypatch.delenv("RTT_VISION_SPEND_FILE")
    want.unlink()
    (tree / "runs" / "vision_review" / "spend.json").write_text("{", encoding="utf-8")
    with pytest.raises(vision.SpendFileError):
        vision.default_spend_guard(tree).spent("claude")
    assert not want.exists()


def test_the_export_passes_the_vision_keys_and_spend_file_through_and_never_stops_for_them(
    ex: Any, vision: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(ex, "export", lambda *a, **k: seen.update(k))
    missing, spend = tmp_path / "nokeys.json", tmp_path / "spend.json"
    # a missing keys file is a warning (naming the flag the export has), not a stop
    assert ex.main(["--reports", str(tmp_path), "--vision-review", "--vision-keys", str(missing)]) == 0
    err = capsys.readouterr().err
    assert "warning" in err and "--vision-keys" in err and "--keys " not in err
    review = seen["vision_review"]
    with pytest.raises(vision.ReviewSetupError, match="--vision-keys"):
        review([])  # when it runs it says why it cannot: export() logs that and goes on
    # with keys and a spend file given, the review uses exactly those
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"claude": "k1", "openai": "k2"}), encoding="utf-8")
    got: dict[str, Any] = {}
    monkeypatch.setattr(
        vision, "make_reviewers", lambda k, **kw: [_Stub("claude", set()), _Stub("openai", set())]
    )
    monkeypatch.setattr(
        vision, "review_lanes", lambda lanes, reviewers, guards, **kw: got.update(guards=guards) or {}
    )
    assert (
        ex.main(
            [
                "--reports",
                str(tmp_path),
                "--vision-review",
                "--vision-keys",
                str(keys),
                "--vision-spend-file",
                str(spend),
            ]
        )
        == 0
    )
    seen["vision_review"]([])
    assert {g.path for g in got["guards"].values()} == {spend} and set(got["guards"]) == {"claude", "openai"}
    # and no vision flags, no review
    seen.clear()
    assert ex.main(["--reports", str(tmp_path)]) == 0 and seen["vision_review"] is None


def test_a_review_needs_a_spend_guard_for_every_reviewer(vision: Any, tmp_path: Path) -> None:
    g = vision.SpendGuard(tmp_path / "s.json", caps={"a": 10.0, "b": 10.0})
    a, b = _Stub("a", {"f0"}), _Stub("b", {"f0"})
    for guards in (None, {}, {"a": g}):
        with pytest.raises(vision.ReviewSetupError, match="no spend guard"):
            vision.review_frames(_frames(vision, 2), [a, b], guards=guards)
    assert a.seen == b.seen == []  # not one look was made
    assert vision.review_frames(_frames(vision, 2), [a, b], guards={"a": g, "b": g})["summary"]["frames"] == 2
