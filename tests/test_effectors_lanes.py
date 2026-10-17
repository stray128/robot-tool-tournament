"""Web lanes and page data of the effector tournament (product mode; ``docs/EFFECTORS.md`` section 6).

A lane is a recorded screen episode lifted to replay version 2 (collision geoms, no GLBs), labelled,
brand-free and at most 300 KB as served. The published data under ``web/public/data/effectors*``
and ``web/public/data/replays/effectors-*`` is checked here too, so a regenerated export that
breaks a rule fails the suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from effector_toy import ToyTwoPad, toy_spec
from rtt.effectors import registry
from rtt.effectors.lanes import (
    LANE_BUDGET_BYTES,
    LaneError,
    _check_brands,
    export_web,
    fit_budget,
    lane_payload,
    lane_plan,
    lift_lane,
    record_lane,
)
from rtt.effectors.parallel_jaw import jaw_spec
from rtt.effectors.runner import run_effector_episode
from rtt.effectors.tournament import run_tournament, write_outputs
from rtt.effectors.types import PRODUCT_MODE_LABEL, find_brand_tokens
from rtt.parts.library import get_part
from rtt.parts.suite import train_suite
from rtt.sim import replay as rp
from rtt.sim.task import default_task_v2
from schema_lite import Validator

REPO = Path(__file__).resolve().parents[1]
WEB_DATA = REPO / "web" / "public" / "data"


@pytest.fixture(scope="module")
def block() -> Any:
    return get_part("block")


@pytest.fixture(scope="module")
def c2() -> Any:
    return next(c for c in train_suite() if c.part.name == "block" and c.placement_label == "c2")


def _schema_errors(header: dict[str, Any]) -> list[str]:
    return Validator(json.loads(rp.SCHEMA_PATH.read_text(encoding="utf-8"))).errors(header)


def test_a_native_lane_is_a_labelled_v2_replay_under_budget(tmp_path: Path, block: Any, c2: Any) -> None:
    with registry.override_family(ToyTwoPad()):
        r = run_effector_episode(
            toy_spec(), block, c2.placement, c2.seeds[0], record=True, condition_id=c2.condition_id
        )
    lane = lift_lane(
        r,
        block,
        run_id="effectors-test",
        code_version="815ac94",
        condition_id=c2.condition_id,
        suite="train_v2",
        episode="recommended-test",
        role="recommended",
    )
    h = lane.header
    assert (
        h["version"] == 2 and rp.check_header(h, lane.frames, lane.overlays) == [] and _schema_errors(h) == []
    )
    assert (
        h["meta"]["design_space"] == "effector-catalog_gripper"
        and h["meta"]["design_id"] == toy_spec().effector_id
    )
    assert h["meta"]["geometry_id"] is None and h["meta"]["part_id"] == "block"
    assert (
        h["meta"]["effector"]["display_name"] == "Two-pad test gripper"
        and h["meta"]["label"] == PRODUCT_MODE_LABEL
    )
    assert h["provenance"]["product_mode"] is True and h["provenance"]["condition"] == c2.condition_id
    assert h["assets"] == {} and all(b["visuals"] == [] and b["geoms"] for b in h["bodies"])
    assert [b["name"] for b in h["bodies"]] == ["palm", "eff_base", "eff_left", "eff_right", "object"]
    channels = h["overlays"]["channels"]
    assert channels == list(rp.OVERLAY_CHANNELS)
    seat = lane.overlays[:, channels.index("seat_short_mm")]
    assert np.isnan(seat.astype(np.float32)).all()  # no seat outside the jaw family
    assert lane_payload(lane)["total"] <= LANE_BUDGET_BYTES and not find_brand_tokens(json.dumps(h))
    rp.write_replay(lane, tmp_path / "lane")
    back = rp.read_replay(tmp_path / "lane")
    assert back.n_frames == lane.n_frames and back.header["meta"]["design_id"] == h["meta"]["design_id"]
    served = sum(p.stat().st_size for p in tmp_path.glob("lane*"))
    assert served <= LANE_BUDGET_BYTES


def test_a_lane_must_reproduce_its_screen_record(block: Any, c2: Any) -> None:
    task = default_task_v2()
    scored = run_effector_episode(jaw_spec("stock"), block, c2.placement, c2.seeds[0], task=task)
    r = record_lane(
        jaw_spec("stock"),
        block,
        c2.placement,
        c2.seeds[0],
        reference=None,
        condition_id=c2.condition_id,
        expect_traj_sha=scored.traj_sha,
        task=task,
    )
    assert r.traj_sha == scored.traj_sha and r.replay is not None
    lane = lift_lane(
        r,
        block,
        run_id="effectors-test",
        code_version="815ac94",
        condition_id=c2.condition_id,
        suite="train_v2",
        episode="jaw",
        role="recommended",
    )
    assert [b["name"] for b in lane.header["bodies"]] == ["palm", "finger_left", "finger_right", "object"]
    assert (
        lane.header["meta"]["design_space"] == "effector-parallel_jaw" and _schema_errors(lane.header) == []
    )
    assert lane_payload(lane)["total"] <= LANE_BUDGET_BYTES
    with pytest.raises(LaneError, match="disagrees with the score"):
        record_lane(
            jaw_spec("stock"),
            block,
            c2.placement,
            c2.seeds[0],
            reference=None,
            condition_id=c2.condition_id,
            expect_traj_sha="0" * 64,
            task=task,
        )


def test_an_oversized_lane_drops_its_frame_rate_or_is_refused() -> None:
    header = {
        "format": "rtt-replay",
        "version": 2,
        "fps": 30,
        "dtype": "float32",
        "byte_order": "little",
        "layout": ["frames", "bodies", "pose"],
        "pose_fields": ["px", "py", "pz", "qx", "qy", "qz", "qw"],
        "up_axis": "z",
        "units": "m",
        "n_frames": 360,
        "n_bodies": 40,
        "bin": "x.bin",
        "assets": {},
        "bodies": [],
        "meta": {},
        "provenance": {},
    }
    frames = np.zeros((360, 40, 7), dtype=np.float32)
    big = rp.Replay(frames=frames, header=header)
    assert lane_payload(big)["total"] > LANE_BUDGET_BYTES
    fitted = fit_budget(big)
    assert (
        fitted.header["fps"] == 15
        and fitted.n_frames == 180
        and fitted.header["meta"]["decimated_from_fps"] == 30
    )
    assert lane_payload(fitted)["total"] <= LANE_BUDGET_BYTES
    huge = rp.Replay(frames=np.zeros((360, 160, 7), dtype=np.float32), header={**header, "n_bodies": 160})
    with pytest.raises(LaneError, match="over"):
        fit_budget(huge)


def test_brands_are_refused_in_lanes_and_pages() -> None:
    with pytest.raises(LaneError, match="names a brand"):
        _check_brands({"bodies": [{"name": "eff_allegro_palm"}]}, "lane x")
    _check_brands({"bodies": [{"name": "eff_palm"}], "label": PRODUCT_MODE_LABEL}, "lane y")


def test_lane_plan_puts_the_recommendation_first() -> None:
    rec = {
        "winner": {"effector": {"effector_id": "b" * 64}},
        "runner_up_rigid": {"effector": {"effector_id": "c" * 64}},
        "alternatives": {"held": {"effector_id": "a" * 64}},
        "by_group": {"reference": "a" * 64, "suction": "b" * 64, "catalog": "c" * 64},
        "ranking": [{"effector_id": x * 64} for x in "bcad"],
    }
    tour = {
        "conditions": [
            {"label": "c0", "condition_id": "train_v2/block/c0", "seeds": [1, 2]},
            {"label": "c2", "condition_id": "train_v2/block/c2", "seeds": [7, 8]},
        ],
        "groups": {"x": {"finalists": ["a" * 64, "b" * 64, "c" * 64, "d" * 64]}},
    }
    plan = lane_plan(rec, tour)
    assert [p.effector_id[0] for p in plan] == ["b", "c", "a", "d"]
    assert (
        plan[0].roles == ("recommended", "best_suction", "finalist")
        and plan[0].stem == f"recommended-{'b' * 12}"
    )
    assert plan[1].roles[0] == "runner_up" and plan[2].roles[0] == "recommended_held"
    assert {(p.condition_id, p.seed) for p in plan} == {("train_v2/block/c2", 7)}


def test_export_web_writes_labelled_pages_and_lanes(tmp_path: Path, block: Any) -> None:
    with registry.override_family(ToyTwoPad()):
        res = run_tournament(block, groups=["catalog"], families=["catalog_gripper"], workers=1)
        write_outputs(res, tmp_path / "runs" / "block")
        index = export_web(tmp_path / "runs", tmp_path / "web")
    assert index["label"] == PRODUCT_MODE_LABEL and [p["part_id"] for p in index["parts"]] == ["block"]
    page = json.loads((tmp_path / "web" / "effectors" / "block.json").read_text())
    assert page["schema"] == "rtt-web-effector-part" and page["label"] == PRODUCT_MODE_LABEL
    assert page["requirement"] == "registered" and set(page["by_requirement"]) == {"held", "registered"}
    assert [f["family"] for f in page["families"]][:1] == ["parallel_jaw"] and all(
        f["label"] == PRODUCT_MODE_LABEL for f in page["families"]
    )
    assert page["fairness"]["identical"] and page["caveats"]["common"]
    lanes = page["lanes"]
    assert len(lanes) == 1 and lanes[0]["roles"][0] == "recommended"
    lane = lanes[0]
    stem = tmp_path / "web" / lane["replay"]
    header = json.loads(stem.with_suffix(".json").read_text())
    assert (
        header["meta"]["design_id"] == lane["effector"]["effector_id"]
        and header["provenance"]["product_mode"]
    )
    served = sum(p.stat().st_size for p in stem.parent.glob(f"{stem.name}.*"))
    assert served == lane["bytes"] <= LANE_BUDGET_BYTES
    assert lane["condition_id"] == "train_v2/block/c2" and lane["outcome"]["traj_sha"]
    for path in (tmp_path / "web").rglob("*.json"):
        assert not find_brand_tokens(path.read_text()), path


# --------------------------------------------------------------------------- published data


def _pages() -> list[Path]:
    return sorted(p for p in (WEB_DATA / "effectors").glob("*.json") if p.name != "index.json")


@pytest.mark.skipif(
    not (WEB_DATA / "effectors" / "index.json").is_file(), reason="no effector data published"
)
def test_published_effector_data_keeps_every_rule() -> None:
    index = json.loads((WEB_DATA / "effectors" / "index.json").read_text())
    assert index["schema"] == "rtt-web-effectors" and index["label"] == PRODUCT_MODE_LABEL
    pages = _pages()
    assert {p.stem for p in pages} == {e["part_id"] for e in index["parts"]}
    for path in pages:
        page = json.loads(path.read_text())
        assert page["label"] == PRODUCT_MODE_LABEL and page["fairness"]["identical"], path.name
        assert page["lanes"] and page["lanes"][0]["roles"][0] in ("recommended", "best_reference", "finalist")
        for lane in page["lanes"]:
            stem = WEB_DATA / lane["replay"]
            files = sorted(stem.parent.glob(f"{stem.name}.*"))
            assert {f.name for f in files} >= {f"{stem.name}.json", f"{stem.name}.bin"}
            served = sum(f.stat().st_size for f in files)
            assert served == lane["bytes"] <= LANE_BUDGET_BYTES, lane["replay"]
            replay = rp.read_replay(stem)
            h = replay.header
            assert rp.check_header(h, replay.frames, replay.overlays) == [] and _schema_errors(h) == []
            assert h["provenance"]["product_mode"] is True and h["meta"]["label"] == PRODUCT_MODE_LABEL
            assert (
                h["meta"]["design_id"] == lane["effector"]["effector_id"]
                and h["meta"]["part_id"] == path.stem
            )
    for root in (WEB_DATA / "effectors", *WEB_DATA.glob("replays/effectors-*")):
        for f in root.rglob("*.json"):
            assert not find_brand_tokens(f.read_text(encoding="utf-8")), f
