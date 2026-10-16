"""The v2 path end to end, on the real generator, parts library, simulator and replay writer.

* ``scripts/v2_bout.py``: recipe -> jaw pair -> v2 episode -> replay v2 with GLB assets, read
  back and checked pose for pose against the render model.
* ``rtt.analysis.gv3_run``: the screening sample's Sobol draws are valid on every training
  part, distinct and reproducible.
* ``rtt.analysis.gvh_run``: the designed candidates are fixed by the GV3 report's ranking.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _bout_module():
    spec = importlib.util.spec_from_file_location("v2_bout", REPO / "scripts" / "v2_bout.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_one_designed_episode_goes_from_recipe_to_a_web_ready_replay(tmp_path: Path) -> None:
    from rtt.render.glb import AssetStore
    from rtt.sim.replay import read_replay

    bout = _bout_module()
    summary = bout.run(tmp_path, (("imprint_block_c1", "imprint_configurator", "block", "c1"),), "test-bout")
    ep = summary["episodes"]["imprint_block_c1"]
    assert ep["condition"] == "train_v2/block/c1"
    assert ep["pose_residual"]["pos"] <= 1e-6 and ep["pose_residual"]["quat"] <= 1e-6  # GV-R
    rp = read_replay(tmp_path / "imprint_block_c1")
    assert rp.version == 2 and rp.overlays is not None
    meta = rp.header["meta"]
    assert meta["design_space"] == "freeform-v2" and meta["design_id"] == ep["design_id"]
    store = AssetStore(tmp_path / "assets")
    roles = {store.entries[a]["role"] for a in rp.header["assets"]}
    assert {"finger_left", "finger_right", "gripper_frame", "carriage", "object"} <= roles
    for aid in rp.header["assets"]:
        entry = store.entries[aid]
        assert entry["sha256"] and entry["attribution"]  # GV-R: every GLB has a sha and a credit
        assert (tmp_path / "assets" / f"{aid}.glb").stat().st_size == entry["bytes"]
    left = [v["asset"] for b in rp.header["bodies"] if b["name"] == "finger_left" for v in b["visuals"]]
    right = [v["asset"] for b in rp.header["bodies"] if b["name"] == "finger_right" for v in b["visuals"]]
    assert set(left) != set(right)  # the right jaw is its own asset, never a mirrored left one


def test_screening_sobol_designs_are_valid_distinct_and_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    from rtt.analysis import gv3_run
    from rtt.design.freeform import validate_design
    from rtt.parts.library import training_parts
    from rtt.sim.task import default_task_v2

    monkeypatch.setattr(gv3_run, "N_SOBOL", 3)
    parts, task = training_parts(), default_task_v2()
    a, log = gv3_run.screening_designs(20261090, parts, task)
    b, _ = gv3_run.screening_designs(20261090, parts, task)
    assert len(a) == 3 and [d.model_dump() for d in a] == [d.model_dump() for d in b]
    assert log["accepted"] == 3 and 0 < log["valid_rate"] <= 1
    ids = set()
    for d in a:
        report = validate_design(d, parts, task)
        assert report.valid, report.message
        ids.add(report.geometry_id)
    assert len(ids) == 3


def test_screening_seed_rules_are_enforced() -> None:
    from rtt.analysis import calibrate_v2 as cal

    with pytest.raises(ValueError, match="gate, search or development seed"):
        cal.check_screening_seed(20261019)


def test_hero_candidates_follow_the_gv3_ranking(tmp_path: Path) -> None:
    from rtt.analysis import gvh_run

    common = {"tip": "blunt", "grip_height_frac": 0.5, "pad_height_mm": 20.0, "finger_width_mm": 30.0}
    designs = [
        {"name": "sobol_00", "origin": "sobol", "design": {"pad": "flat", **common}},
        {
            "name": "sobol_01",
            "origin": "sobol",
            "design": {
                "pad": "conform",
                "pad_depth_mm": 4.0,
                "wall_draft_deg": 30.0,
                "clearance_mm": 0.4,
                **common,
            },
        },
    ]
    report = {
        "sobol": [
            {"name": "sobol_00", "score": 0.5, "lift_score": 0.9, "mass_g": 20.0},
            {"name": "sobol_01", "score": 0.5, "lift_score": 0.95, "mass_g": 30.0},
        ]
    }
    (tmp_path / "gv3_designs.json").write_text(json.dumps(designs))
    (tmp_path / "gv3_report.json").write_text(json.dumps(report))
    names = [n for n, _ in gvh_run.candidates(tmp_path, top=1)]
    assert names == ["ref_wrap", "ref_hook", "sobol_01"]  # references first, then the ranking
