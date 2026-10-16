"""Arms C and Z on freeform-v2: tool schema, prompt content, notes block, repair, guard and hashes."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from rtt.llm.cache import canonical_json
from rtt.search import prompts
from rtt.search.fakes_v2 import DEFAULT_FAKE_PARTS_V2, FakeEvaluatorV2, ScriptedProposerV2, stub_entrants
from rtt.search.nemotron import PROPOSER_SLOTS, PromptLeakError, prompt_spec_hashes, submit_designs_tool
from rtt.search.nemotron_v2 import (
    FORBIDDEN_PROMPT_TEXT_V2,
    NemotronProposerV2,
    prompt_spec_hashes_v2,
    render_notes,
)
from rtt.search.runner import run_search
from rtt.search.space import V2, get_space
from rtt.search.zeroshot import ZEROSHOT_SLOTS, ZeroShotProposerV2
from search_v2_support import (
    ReplyTransport,
    client_for,
    diverse_recipes,
    recipe_slot,
    submission_v2,
    v2_round0,
)

SPACE = get_space(V2)


def test_tool_schema_keeps_four_slots_and_four_reserves_with_pad_tip_params() -> None:
    tool = submit_designs_tool(PROPOSER_SLOTS, explain=True, reserves=4, space=SPACE)
    fn = tool["function"]
    assert fn["description"] == (
        "Submit 4 gripper jaw recipes for simulation, one per slot, plus 4 reserve designs that replace any rejected one."
    )
    props = fn["parameters"]["properties"]
    assert list(props) == [
        "analysis",
        "mutate_1",
        "mutate_2",
        "crossover",
        "explore",
        "reserve_1",
        "reserve_2",
        "reserve_3",
        "reserve_4",
    ]
    assert list(props["mutate_1"]["properties"]) == [
        "parents",
        "hypothesis",
        "changes",
        "same_cell_distances",
        "pad",
        "tip",
        "params",
    ]
    assert list(props["reserve_1"]["properties"])[:2] == ["operator", "parents"]
    assert props["crossover"]["properties"]["parents"]["minItems"] == 2
    assert "parents" not in props["explore"]["properties"]
    assert props["mutate_1"]["properties"]["params"]["required"] == [
        "grip_height_frac",
        "pad_height_mm",
        "finger_width_mm",
    ]
    z = submit_designs_tool(ZEROSHOT_SLOTS, space=SPACE)["function"]["parameters"]["properties"]
    assert list(z["design_1"]["properties"]) == ["hypothesis", "pad", "tip", "params"]


@pytest.fixture(scope="module")
def prompts_c_and_z(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[dict]]:
    root = tmp_path_factory.mktemp("v2prompts")
    out = {}
    for arm in ("nemotron", "zeroshot"):
        model = ScriptedProposerV2(3, invalid_rate=0.3, similar_rate=0.3)
        run_search(
            arm, 1, 24, [1], FakeEvaluatorV2(0), root / arm, client_for(model), entrants=stub_entrants()
        )
        out[arm] = model.payloads
    return out


def _user(payload: dict) -> str:
    return payload["messages"][-1]["content"]


def test_arm_c_prompt_has_every_v2_section(prompts_c_and_z) -> None:
    text = _user(prompts_c_and_z["nemotron"][0])
    sections = [line for line in text.splitlines() if line.startswith("# ")]
    assert sections == [
        "# Task",
        "# Training parts",
        "# Placements",
        "# Recipe parameters",
        "# Validator rules",
        "# Engineering notes",
        "# Distance rule",
        "# All 8 designs tested in this run, by cell",
        "# Your team's best designs (top 8 of 8, worst first, best last)",
        "# Informative failures",
        "# Validator rejections in your last round",
        "# Target",
        "# Round 1 of 2: what to submit",
    ]
    for part in DEFAULT_FAKE_PARTS_V2:
        assert f"| {part.part_id} |" in text
    for placement in ("c0: centred", "c1: 2 mm off centre", "c2: 3 mm off centre", "c3: 1 mm off centre"):
        assert placement in text
    for tolerance in (
        "at most 1 mm short of its seat",
        "within 0.75 mm of centre",
        "within 2 deg of its yaw",
    ):
        assert tolerance in text
    assert "48 training episodes (6 parts x 4 placements x 2 jitter seeds)" in text
    elites = text.split("# Your team's best designs")[1].split("# Informative failures")[0]
    for marker in (
        "per part: pipe_tee",
        "per placement: c0",
        "jam_on_close",
        "off_centre",
        "capture no-jam/centring",
        "pieces per jaw",
        "mass ",
    ):
        assert marker in elites
    scores = [int(m) for m in re.findall(r"^d\d\d \[[^]]+\] score (\d+)%", elites, re.MULTILINE)]
    assert scores == sorted(scores) and len(scores) == 8  # worst first, best last
    assert "at least 72 points" in text and "duplicates" in text


def test_the_engineering_notes_block_is_shared_by_c_and_z_and_hashed(prompts_c_and_z) -> None:
    info = FakeEvaluatorV2(0).describe()
    notes = render_notes(info)
    assert "keep wall_draft_deg at or below about 68 deg" in notes
    assert "capture range of at least 3.5 mm" in notes and "at least 2.4 mm from the part's centre" in notes
    assert "hex_standoff 6.1 mm" in notes and "l_bracket 17.6 mm" in notes  # reach - 2.4 mm
    assert "flanged_bushing 7.6 mm" in notes  # the 20 mm body, not the 36 mm flange
    assert "guidance only: the validator does not enforce them" in notes
    for arm in ("nemotron", "zeroshot"):
        for payload in prompts_c_and_z[arm]:
            assert notes in _user(payload)
    hashes = prompt_spec_hashes_v2(info)
    assert hashes["engineering_notes"] == hashlib.sha256(canonical_json(notes).encode("utf-8")).hexdigest()
    assert "template:v2/engineering_notes.md" in hashes


def test_v1_prompt_hashes_do_not_see_the_v2_templates() -> None:
    v1 = prompt_spec_hashes()
    assert set(v1) == {f"template:{n}" for n in prompts.TEMPLATE_NAMES} | {
        "tool:proposer",
        "tool:zeroshot",
        "parameter_table",
        "points_rule",
    }
    v2 = prompt_spec_hashes_v2()
    assert all(k.startswith("template:v2/") for k in v2 if k.startswith("template:"))
    assert v2["tool:proposer"] != v1["tool:proposer"]


def test_inactive_parameters_are_ignored_and_logged_on_the_proposal() -> None:
    ctx, items = v2_round0()
    recipes = diverse_recipes(ctx, 4)
    args = submission_v2(recipes)
    flat = next((r for r in recipes if SPACE.params(r)["pad"] == "flat"), None)
    target = "mutate_1"
    if flat is None:
        flat = SPACE.recipe(
            pad="flat", tip="blunt", grip_height_frac=0.83, pad_height_mm=9.0, finger_width_mm=39.0
        )
    args[target] = recipe_slot(flat, ("d05",), "flat with extras", pad_depth_mm=7.0, wall_draft_deg=33.0)
    proposer = NemotronProposerV2(ctx, client_for(ReplyTransport(args, args, args, args)))
    proposer.tell(items)
    out = proposer.ask(1, 8)
    first = next(p for p in out if p.slot == target and p.island == 0)
    assert SPACE.design_id(first.design) == SPACE.design_id(flat)
    assert any("pad_depth_mm is inactive for a flat pad" in n for n in first.notes)
    assert first.to_json()["canon_notes"] == list(first.notes)


def test_invalid_slot_is_repaired_with_the_v2_messages() -> None:
    ctx, items = v2_round0()
    recipes = diverse_recipes(ctx, 8)
    bad = submission_v2(recipes[:4])
    bad["crossover"] = {
        "parents": ["d05", "d06"],
        "hypothesis": "a V too shallow",
        "pad": "v_funnel",
        "tip": "blunt",
        "params": {
            "grip_height_frac": 0.4,
            "pad_height_mm": 12.0,
            "finger_width_mm": 30.0,
            "pad_depth_mm": 6.0,
            "wall_draft_deg": 8.0,
        },
    }
    fixed = submission_v2(recipes[:4])
    fixed["crossover"] = recipe_slot(recipes[4], ("d05", "d06"), "fixed V")
    second = submission_v2(recipes[4:])
    transport = ReplyTransport(bad, fixed, second, second, second)
    proposer = NemotronProposerV2(ctx, client_for(transport))
    proposer.tell(items)
    proposer.ask(1, 8)
    repair = _user(transport.payloads[1])
    assert "# Correction needed" in repair
    assert "crossover: a v_funnel pad needs wall_draft_deg between 15 and 70 (got 8)" in repair
    assert "Give every number that is active for the pad and tip you choose." in repair
    assert ctx.rejections[0].reason == "invalid" and ctx.rejections[0].params["pad"] == "v_funnel"


def test_the_v2_guard_refuses_held_out_families_and_scrubs_model_text() -> None:
    for word in ("a pipe elbow", "spur gear", "Gears", "held-out"):
        assert FORBIDDEN_PROMPT_TEXT_V2.search(word)
    for word in ("pipe tee", "flanged bushing", "steel slug", "engineering", "L-bracket"):
        assert not FORBIDDEN_PROMPT_TEXT_V2.search(word)
    ctx, items = v2_round0()
    proposer = NemotronProposerV2(ctx, client_for(ReplyTransport()))
    with pytest.raises(PromptLeakError):
        proposer._call("main", 1, 0, [{"role": "user", "content": "like a spur gear"}], keep=0)
    assert proposer._scrub("grips it like a gear tooth") == "grips it like a [removed] tooth"
    recipes = diverse_recipes(ctx, 8)
    args = submission_v2(recipes[:4])
    args["mutate_1"]["hypothesis"] = "Hold it like a gear in an elbow joint."
    transport = ReplyTransport(args, submission_v2(recipes[4:]), args, args)
    proposer = NemotronProposerV2(ctx, client_for(transport))
    proposer.tell(items)
    proposer.ask(1, 8)  # quoting the hypothesis back later is safe: it was scrubbed


def test_zeroshot_v2_prompts_never_depend_on_results(tmp_path: Path) -> None:
    payloads = []
    for seed in (0, 1):  # two landscapes: different scores, same designs proposed
        model = ScriptedProposerV2(4)
        run_search(
            "zeroshot",
            1,
            24,
            [9],
            FakeEvaluatorV2(seed),
            tmp_path / f"z{seed}",
            client_for(model),
            entrants=stub_entrants(),
        )
        payloads.append([_user(p) for p in model.payloads])
    assert payloads[0] == payloads[1]
    proposed = payloads[0][-1].split("# Designs already submitted")[1].split("# Call")[0]
    assert "{" in proposed and "%" not in proposed and "lift" not in proposed
    assert ZeroShotProposerV2.forbidden is FORBIDDEN_PROMPT_TEXT_V2


def test_v2_runs_record_v2_prompt_hashes(tmp_path: Path) -> None:
    out = run_search(
        "nemotron",
        1,
        16,
        [3],
        FakeEvaluatorV2(0),
        tmp_path / "c",
        client_for(ScriptedProposerV2(1)),
        entrants=stub_entrants(),
    )
    hashes = out.manifest["prompt_hashes"]
    assert "template:v2/proposer_user.md" in hashes and "engineering_notes" in hashes
    assert out.manifest["gripper_task"]["design_space"] == V2
    assert out.manifest["entrants"][0]["name"] == "stock"


def test_probe_v2_offline_scores_slots_and_replays_without_calls(tmp_path) -> None:
    """The GV5 probe harness on a scripted model: 5 calls, 20 slots, then a free replay."""
    from rtt.llm.client import TokenBucket
    from rtt.search.probe_v2 import run_probe

    model = ScriptedProposerV2(5, invalid_rate=0.25, similar_rate=0.2)
    report = run_probe(tmp_path, transport=model, limiter=TokenBucket(1e12))
    assert model.calls == 5 and report["slots"] == 20 and len(report["calls"]) == 5
    assert report["validators"] == "schema" and report["live_calls_ledger_total"] == 5
    assert report["live_cost_usd_ledger_total"] == pytest.approx(report["cost_usd_this_report"], abs=1e-6)
    for call in report["calls"]:
        assert call["ok"] and call["valid_primary"] <= 4 and call["valid_reserves"] <= 4
        assert call["accepted_after_reserves"] <= call["valid_after_reserves"] <= 4
        assert call["needs_repair"] == (call["valid_after_reserves"] < 4)
    rate = sum(c["valid_after_reserves"] for c in report["calls"]) / 20
    assert report["valid_after_reserves_rate"] == rate
    repairs = sum(c["needs_repair"] for c in report["calls"])
    assert report["pass"] == (rate >= 0.9 and repairs <= 1)

    class NoNetwork:
        def chat_completions(self, payload, timeout_s):  # type: ignore[no-untyped-def]
            raise AssertionError("no API call is allowed during replay")

    again = run_probe(tmp_path, mode="replay", transport=NoNetwork(), limiter=TokenBucket(1e12))
    keys = (
        "valid_primary",
        "valid_reserves",
        "valid_after_reserves",
        "accepted_after_reserves",
        "errors",
        "cells",
    )
    assert [{k: c[k] for k in keys} for c in again["calls"]] == [
        {k: c[k] for k in keys} for c in report["calls"]
    ]
    assert all(c["cached"] for c in again["calls"]) and again["live_calls_ledger_total"] == 5
