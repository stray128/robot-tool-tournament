"""The Studio lane format on both sides: what the Python exporter writes is what the site reads.

``web/tests/fixtures/studio_lanes/`` holds two lanes made by the current exporter
(:func:`rtt.studio.replay.export_lanes` on a reference-controller tournament): a tool change with
a cut, stored as float32, and the same lane forced to int16 by a smaller budget, with the page
entries the exporter writes for them. ``web/tests/studio_lanes.test.ts`` reads those files with the
site's own parsers. This file re-runs the exporter and checks that what it writes now has exactly
the fixture's shape (every key and the type of every value), so a format change on either side
fails a test until both sides agree.

Rewrite the fixture after a deliberate format change:
``RTT_WRITE_WEB_FIXTURES=1 pytest tests/test_web_lane_contract.py``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from rtt.studio import replay as sr
from rtt.studio.search import run_task_tournament
from studio_fixtures import cup, gripper, two_press_task

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "web" / "tests" / "fixtures" / "studio_lanes"
HERO = "fixture"
CODE_VERSION = "0123456789ab"
#: The stems of the two fixture lanes.
STEMS = ("recommended", "recommended_int16")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[dict[str, Any], dict[str, Any], Path]]:
    """The two lanes as the exporter writes them now: stem -> (header, page entry, folder)."""
    rep = run_task_tournament(
        two_press_task(),
        candidates=[gripper(), cup()],
        workers=1,
        n_conditions=2,
        n_screen=1,
        controllers="reference",
    )
    report = json.loads(json.dumps(rep.to_dict()))
    change = next(f for f in report["finalists"] if f["setup"]["kind"] == "tool_change")
    # Reference results are never recommended (they are not evidence); the harness names the tool
    # change as the pick so the exporter records the lane a recommended tool change would get.
    report["recommendation"]["recommendation"] = {"setup_id": change["setup"]["setup_id"]}
    out = tmp_path_factory.mktemp("lanes")
    store = sr.staging_store(out / "raw")

    def lane(budget: int) -> sr.LaneBuild:
        lanes = sr.export_lanes(
            HERO, report, store, code_version=CODE_VERSION, log=lambda _: None, budget=budget
        )
        return next(x for x in lanes if x.stem == "recommended")

    wide = lane(sr.LANE_BUDGET_BYTES)
    narrow = lane(sr.lane_bytes(wide.replay) - 1)
    result = {}
    for stem, build in zip(STEMS, (wide, narrow), strict=True):
        bin_path, json_path = sr.write_lane(build.replay, out / stem)
        doc = dict(build.doc)
        # what the export script fills in once the lane is written (scripts/export_studio_web.py)
        doc["replay"] = f"studio/replays/{HERO}/{stem}"
        doc["bytes"] = json_path.stat().st_size + bin_path.stat().st_size
        doc["asset_bytes"] = 0
        result[stem] = (json.loads(json_path.read_text(encoding="utf-8")), doc, out)
    if os.environ.get("RTT_WRITE_WEB_FIXTURES") == "1":
        FIXTURE.mkdir(parents=True, exist_ok=True)
        for stem in STEMS:
            for ext in (".json", ".bin"):
                shutil.copyfile(out / f"{stem}{ext}", FIXTURE / f"{stem}{ext}")
        entries = {stem: result[stem][1] for stem in STEMS}
        (FIXTURE / "entries.json").write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")
    return result


def shape(value: Any) -> Any:
    """The shape of a JSON value: every key, and the type of every value (numbers are one type;
    a list is the shape of its items, which must agree)."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, dict):
        return {k: shape(v) for k, v in sorted(value.items())}
    items = [shape(v) for v in value]
    first = next((s for s in items if s != "null"), "null")
    return ["list", first]


def same_shape(a: Any, b: Any, where: str = "$") -> list[str]:
    """Where two shapes differ. ``null`` matches anything (a value may be null in one run)."""
    if a == "null" or b == "null":
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        out = [
            f"{where}: key {k} only in the {'fixture' if k in b else 'export'}"
            for k in sorted(set(a) ^ set(b))
        ]
        for k in sorted(set(a) & set(b)):
            out += same_shape(a[k], b[k], f"{where}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        return same_shape(a[1], b[1], f"{where}[]")
    return [] if a == b else [f"{where}: {a} in the export, {b} in the fixture"]


def _fixture(stem: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not (FIXTURE / "entries.json").is_file():
        pytest.fail(f"no fixture in {FIXTURE}: run with RTT_WRITE_WEB_FIXTURES=1 to write it")
    header = json.loads((FIXTURE / f"{stem}.json").read_text(encoding="utf-8"))
    entry = json.loads((FIXTURE / "entries.json").read_text(encoding="utf-8"))[stem]
    return header, entry


@pytest.mark.parametrize("stem", STEMS)
def test_the_exporter_writes_the_shape_the_site_is_tested_on(built: dict[str, Any], stem: str) -> None:
    header, entry, _ = built[stem]
    fixed_header, fixed_entry = _fixture(stem)
    assert same_shape(shape(header), shape(fixed_header)) == []
    assert same_shape(shape(entry), shape(fixed_entry)) == []


def test_the_fixture_is_a_tool_change_lane_in_both_encodings() -> None:
    wide, wide_entry = _fixture("recommended")
    narrow, narrow_entry = _fixture("recommended_int16")
    assert wide["quantised"] is False and wide["dtype"] == "float32"
    assert narrow["quantised"] is True and narrow["dtype"] == "int16"
    assert narrow["pose_fields"] == list(sr.QUANT_POSE_FIELDS)
    for header, entry in ((wide, wide_entry), (narrow, narrow_entry)):
        assert header["cuts"], "the fixture lane must cross a tool change"
        kf = header["meta"]["keyframes"]
        assert set(kf) == {"recorded_fps", "samples", "tolerance_mm", "max_error_vs_recording_mm"}
        assert "max_error_mm" not in kf and "max_error_mm" not in entry
        assert kf["tolerance_mm"] in (1.0, 1.5, 2.0)
        assert header["fps"] == kf["recorded_fps"] == entry["fps"] == sr.RECORD_FPS
        assert entry["max_error_vs_recording_mm"] == kf["max_error_vs_recording_mm"] <= kf["tolerance_mm"]
        assert entry["duration_s"] == pytest.approx(header["t"][-1], abs=1e-3)
        assert entry["quantised"] is header["quantised"]
        assert entry["n_frames"] == header["n_frames"] == len(header["t"])


@pytest.mark.parametrize("stem", STEMS)
def test_the_fixture_reads_back_in_python(stem: str) -> None:
    header, _ = _fixture(stem)
    lane = sr.read_lane(FIXTURE / stem)
    assert lane.frames.shape == (header["n_frames"], header["n_bodies"], 7)
    assert sr.check_lane_header(header, lane.frames) == []


def test_the_v2_templates_have_their_own_looks() -> None:
    from rtt.render.materials import PART_LOOKS

    looks = {
        ("panel_rail", "panel"): "glass_panel",
        ("panel_rail", "frame"): "aluminium_anodised",
        ("tile_bed", "tile"): "ceramic_tile",
        ("tile_bed", "bed"): "mortar",
        ("spring_button", "cap"): "plastic_blue",
        ("spring_button", "mount"): "plastic_grey",
        ("sliding_door", "door"): "steel_stamped",
        ("hinged_lid", "lid"): "pvc_white",
        (None, ""): "clay",
    }
    for (template, part), want in looks.items():
        assert sr.object_look(template, part) == want, (template, part)
        assert want in PART_LOOKS
    from rtt.studio.types import TemplateKind

    for kind in TemplateKind:
        assert sr.object_look(kind.value) != "clay", f"{kind.value} falls back to clay"
