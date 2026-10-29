"""What the Studio export adds for v2 (``scripts/export_studio_web.py``), checked against the site.

* Studio-only tools (vacuum area frames, the nut runner, the tamper pad) carry ``studio_only``, a
  rating computed by their family's own geometry and their own caveat; every other tool carries
  none of these. ``web/tests/fixtures/studio_tools.json`` holds what the exporter writes for the
  four presets, and the site's tests build their Studio-only candidates from it.
* The recommender's baseline (the best eligible single tool, rule R01) is on the page as
  ``baseline`` (its finalist by id, or why there is none) with ``versus_baseline`` on the
  recommendation and a ``baseline`` role on its finalist row.

Rewrite the fixture after a deliberate change: ``RTT_WRITE_WEB_FIXTURES=1 pytest <this file>``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from rtt.effectors.registry import get_family
from rtt.effectors.types import EffectorFamily
from rtt.studio.recommend import recommend_setups
from test_studio_rules import _result, _runs, _setups

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "web" / "tests" / "fixtures" / "studio_tools.json"
STUDIO_ONLY = ("nut-runner-socket-13", "tamper-pad-200", "vacuum-frame-2x2-300", "vacuum-frame-2x3-600")


@pytest.fixture(scope="module")
def ex() -> Any:
    spec = importlib.util.spec_from_file_location(
        "export_studio_web", REPO / "scripts" / "export_studio_web.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_studio_web"] = mod
    spec.loader.exec_module(mod)
    return mod


def _presets() -> dict[str, Any]:
    return {
        s.model: s
        for fam in (EffectorFamily.PROCESS_TOOL, EffectorFamily.SUCTION)
        for s in get_family(fam).presets()
    }


def test_studio_only_tools_carry_their_rating_and_caveat(ex: Any) -> None:
    presets = _presets()
    tools = []
    for model in STUDIO_ONLY:
        spec = presets[model]
        facts = ex.studio_tool(spec.family.value, model)
        assert facts == ex.studio_tool(spec.family.value, model, spec.params)  # preset or params alike
        assert facts["studio_only"] is True and facts["rating"].startswith("rated ")
        assert "(approximation)" in facts["caveat"] and "verify on hardware" in facts["caveat"].lower()
        tools.append(
            {
                "effector_id": spec.effector_id,
                "display_name": spec.display_name,
                "family": spec.family.value,
                "model": model,
                "fidelity": spec.fidelity.value,
                **facts,
            }
        )
    by_model = {t["model"]: t for t in tools}
    # the ratings are the family's own (docs/EFFECTORS.md 13)
    assert by_model["nut-runner-socket-13"]["rating"].startswith("rated 6.7 N m (clutch cap 10 N m)")
    assert by_model["tamper-pad-200"]["rating"] == "rated 40 N (the stage slips at 60 N)"
    assert by_model["vacuum-frame-2x2-300"]["rating"].startswith("rated 4.15 kg on a smooth face")
    assert by_model["vacuum-frame-2x3-600"]["rating"].startswith("rated 8.47 kg on a smooth face")
    assert by_model["vacuum-frame-2x2-300"]["family_title"] == "Vacuum area frame"
    assert by_model["tamper-pad-200"]["family_title"] == "Process tool"
    for doc in tools:
        assert ex.check_doc(doc, doc["model"]) == []
    # every other preset is an ordinary tool: no flag, no rating, no caveat of its own
    for model, spec in presets.items():
        if model not in STUDIO_ONLY:
            assert ex.studio_tool(spec.family.value, model) == {}, model
    assert ex.studio_tool("soft_arm", "tentacle-12") == {}
    if os.environ.get("RTT_WRITE_WEB_FIXTURES") == "1":
        FIXTURE.write_text(json.dumps(tools, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == tools, (
        "web/tests/fixtures/studio_tools.json is stale: rewrite it with RTT_WRITE_WEB_FIXTURES=1"
    )


def _rows(rec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["setup_id"]: r for r in rec["ranking"]}


def test_the_baseline_is_on_the_page_beside_the_pick(ex: Any) -> None:
    task, single, _, head = _setups()
    rec = recommend_setups(
        [_result(single, task, _runs(6, 8), 40.0), _result(head, task, _runs(7, 8), 45.0, presentations=1)]
    )
    assert rec["recommendation"]["setup_id"] == head.setup_id
    report = {"recommendation": rec}
    assert ex.baseline_doc(report, _rows(rec)) == {"setup_id": single.setup_id, "basis": None}
    # the baseline's row is marked in the ranking the finalist rows are built from
    assert [r["setup_id"] for r in rec["ranking"] if r.get("baseline")] == [single.setup_id]
    vs = ex._rd(rec["versus_baseline"], 4)
    assert vs["basis"] == "margin" and vs["same_setup"] is False and vs["added_pieces"] == 1
    # a baseline that is not among the page's finalists is refused
    with pytest.raises(ex.ExportError, match="not among the finalists"):
        ex.baseline_doc(report, {})


def test_no_baseline_says_why(ex: Any) -> None:
    task, single, _, head = _setups()
    rec = recommend_setups([_result(single, task, [False] * 8, 40.0), _result(head, task, [False] * 8, 45.0)])
    assert rec["recommendation"] is None
    doc = ex.baseline_doc({"recommendation": rec}, _rows(rec))
    assert doc["setup_id"] is None and doc["basis"].startswith("no single tool is eligible")
    # a report from before the recommender had a baseline has none on the page
    assert ex.baseline_doc({"recommendation": {}}, {}) is None


def test_a_studio_only_tool_that_is_not_a_preset_is_rated_from_its_own_params(
    ex: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report lists its screened candidates without their params; the exporter finds them in the
    candidate pool by effector id, so a designed frame variant still says it is Studio-only, with its
    rating and caveat (without them the site's parser refuses the page)."""
    import dataclasses

    from rtt.studio import search

    preset = _presets()["vacuum-frame-2x2-300"]
    variant = dataclasses.replace(
        preset,
        model="vacuum-frame-2x2-400",
        params={**preset.params, "spacing_x_mm": 300, "spacing_y_mm": 300},
    )
    cand = search.candidate_of(variant, "designed")
    assert "params" not in cand.to_dict()
    assert ex.studio_tool("suction", variant.model) == {}  # no preset of that model
    monkeypatch.setattr(search, "default_candidates", lambda **kw: (cand,))
    task = _setups()[0].to_dict()
    params = ex.candidate_params({"task": task, "finalists": []})
    facts = ex.studio_tool("suction", variant.model, params[variant.effector_id])
    assert facts["studio_only"] is True and facts["family_title"] == "Vacuum area frame"
    assert facts["rating"].startswith("rated ") and facts["caveat"]


def test_pruning_reasons_reach_the_page_in_plain_words(ex: Any) -> None:
    labels = {"parcel": "Parcel", "test_port": "Test port", "qt_latch": "Quarter-turn latch"}
    cases = {
        "parcel: too_narrow: open, digit 1 has its inner face 35.0 mm from the centre (3.5 mm placement "
        "error + 1.5 mm clearance; palm_radius_mm sets it)": "Parcel: too narrow: open, digit 1 has its inner "
        "face 35.0 mm from the centre (3.5 mm placement error + 1.5 mm clearance; the palm radius sets it)",
        "test_port: digit_collision: closing over studio_probe, digits 1 and 2 run into each other": "Test port: "
        "digits collide: closing over the probe part, digits 1 and 2 run into each other",
        "cannot be built: Parallel jaw, configurator imprint for qt_latch: q_seat_range: part "
        "'studio_qt_latch_head': the pads reach too far in (reduce pad_depth_mm, tip_size_mm, or raise "
        "wall_draft_deg)": "cannot be built: Parallel jaw, configurator imprint for Quarter-turn latch: seat "
        "range: part Quarter-turn latch head: the pads reach too far in (reduce the pad depth, the tip size, "
        "or raise the wall draft)",
        "its capability tags do not include hook_pull": "its capability tags do not include hook pull",
    }
    for raw, plain in cases.items():
        assert ex.plain_reason(raw, labels) == plain
        assert "_" not in plain
    # the summary still groups by the reason as the search wrote it
    assert (
        ex._reason_group("parcel: too_narrow: open", labels) == "Its open digits stand too close for the part"
    )


def test_a_rigid_pick_names_no_rigid_runner_up(ex: Any) -> None:
    """The site requires a basis for a runner-up it shows: a rigid pick (the recommender gives no
    runner-up) is null on the page, never an empty basis."""
    assert ex.runner_up_doc(None) is None
    assert ex.runner_up_doc({"setup": None, "basis": "no rigid-tier setup is eligible"}) == {
        "label": None,
        "basis": "no rigid-tier setup is eligible",
    }
    card = ex.runner_up_doc({"setup": {"label": "Two-finger gripper"}, "why": None})
    assert card is not None and card["label"] == "Two-finger gripper" and card["basis"]
