"""The Studio's Task Engineer (track S4, product mode): prompts, checks, repair, refusals, uploads,
the recorded gate run, the command line and the HTTP routes. Offline: scripted or recorded
responses only, no API key, no network.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rtt.effectors.types import PRODUCT_MODE_LABEL, find_brand_tokens
from rtt.llm.cache import CacheMiss, ResponseCache, wire_json
from rtt.llm.client import TokenBucket
from rtt.llm.ledger import Ledger, read_ledger
from rtt.llm.providers import TransportResponse
from rtt.search.nemotron_v2 import FORBIDDEN_PROMPT_TEXT_V2
from rtt.studio.engineer import (
    ENGINEER_MODEL,
    GATE_CASES,
    HERO_CASES,
    UploadInfo,
    UploadRegistry,
    build_messages,
    check_plan,
    engineer_tool,
    install_upload_resolver,
    judge_args,
    make_engineer_client,
    nearest_primitive,
    plan_task,
    sanitize_use_case,
    web_catalog,
    worked_example,
)
from rtt.studio.engineer import prompts as eprompts
from rtt.studio.engineer.__main__ import main as cli_main
from rtt.studio.engineer.catalog import render_primitives, render_rules, render_templates
from rtt.studio.engineer.engineer import (
    EngineerCache,
    assert_clean,
    explain_problems,
    normalise_nulls,
    render_example,
    tool_sha256,
)
from rtt.studio.engineer.examples import UPLOAD_CASE, UPLOAD_EXAMPLE_STL, V1_GATE_CASES, V1_HERO_CASES
from rtt.studio.engineer.gate import (
    GATE_VALID_RATE,
    RECORDED_GATE,
    RECORDED_GATE_V1,
    gate_manifest,
    recorded_versions,
    replay_plan,
    replay_recorded,
    responses_client,
    run_gate,
)
from rtt.studio.engineer.webdata import FILES, RECORDED_UPLOAD, parity_cases, recorded_plans
from rtt.studio.types import (
    PRIMITIVES,
    ObjectInstance,
    ObjectPose,
    PrimitiveKind,
    Step,
    StepConstraints,
    SuccessCriterion,
    TaskSpec,
    TemplateKind,
    planner_templates,
    submit_task_plan_tool,
    task_plan_json_schema,
)
from schema_lite import Validator, unsupported_keywords

REPO = Path(__file__).resolve().parents[1]
WEB_INTAKE = REPO / "web" / "components" / "studio" / "intake"
HELDOUT = FORBIDDEN_PROMPT_TEXT_V2


# --------------------------------------------------------------------------- scripted model


class ScriptedPlanner:
    """A fake transport: answers each call with the next scripted item and records the payloads.

    An item is tool arguments (a mapping), raw argument text (a string), or an int HTTP status.
    """

    def __init__(
        self,
        answers: Sequence[Any],
        *,
        reasoning: str = "",
        content: str | None = None,
        finish_reason: str = "tool_calls",
    ) -> None:
        self.answers = list(answers)
        self.payloads: list[dict[str, Any]] = []
        self.reasoning = reasoning
        self.content = content
        self.finish_reason = finish_reason

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        self.payloads.append(json.loads(json.dumps(payload)))
        item = self.answers.pop(0)
        if isinstance(item, int):
            return TransportResponse(item, {}, {"error": {"message": "scripted failure"}})
        arguments = item if isinstance(item, str) else json.dumps(item)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "submit_task_plan", "arguments": arguments},
                }
            ],
        }
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        body = {
            "id": f"resp-{len(self.payloads)}",
            "choices": [{"index": 0, "finish_reason": self.finish_reason, "message": message}],
            "usage": {"prompt_tokens": 18000, "completion_tokens": 2000},
        }
        return TransportResponse(200, {}, body)

    def list_models(self, timeout_s: float) -> TransportResponse:
        return TransportResponse(200, {}, {"data": []})


def client_for(planner: ScriptedPlanner, cache: Path | None = None, budget: float | None = None) -> Any:
    return make_engineer_client(
        cache, ledger=Ledger(None, budget_usd=budget), transport=planner, limiter=TokenBucket(1e12)
    )


def good_args() -> dict[str, Any]:
    return worked_example().to_tool_args()


def hero_args(case_id: str) -> dict[str, Any]:
    example = next(e for e in recorded_plans()["examples"] if e["case_id"] == case_id)
    keys = (
        "title",
        "description",
        "objects",
        "steps",
        "assumptions",
        "clarifications",
        "unsupported",
        "refusal",
    )
    return copy.deepcopy({k: example["plan"][k] for k in keys})


def refusal_args(nearest: str | None = "grasp") -> dict[str, Any]:
    return {
        "title": "Fold a shirt",
        "description": "Fold a shirt.",
        "objects": [],
        "steps": [],
        "assumptions": [],
        "clarifications": [],
        "unsupported": [
            {"what": "Folding cloth", "reason": "Cloth is deformable.", "nearest_primitive": nearest}
        ],
        "refusal": "Cloth is deformable and is not simulated.",
    }


# --------------------------------------------------------------------------- tool and prompts


def test_the_default_tool_is_the_contracts_and_compact_only_drops_descriptions() -> None:
    assert engineer_tool() == submit_task_plan_tool()
    full, compact = engineer_tool(compact=False), engineer_tool(compact=True)

    def strip(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "description"}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    assert compact["function"]["parameters"] == strip(full["function"]["parameters"])
    assert compact["function"]["name"] == full["function"]["name"] == "submit_task_plan"
    assert not unsupported_keywords(compact["function"]["parameters"])
    assert len(json.dumps(compact)) < len(json.dumps(full))


def test_every_gate_prompt_is_clean_and_names_every_template_and_primitive() -> None:
    for case in (*GATE_CASES, *V1_GATE_CASES, UPLOAD_CASE):
        messages = build_messages(sanitize_use_case(case.text)[0])
        assert_clean(messages)
        text = "\n".join(m["content"] for m in messages)
        assert not HELDOUT.search(text) and not find_brand_tokens(text), case.case_id
        assert case.text.split(":")[0] in text
    user = build_messages("Open the lid.")[1]["content"]
    for t in planner_templates():  # the kinds it is offered (the Studio v2 job templates come later)
        assert f"## {t.kind.value}:" in user
        assert all(f"| {p.name} |" in user for p in t.params)
    for k in PrimitiveKind:
        assert f"## {k.value}\n" in user
    assert "No part was uploaded" in user
    assert set(eprompts.template_hashes()) == {"system.md", "user.md", "repair.md"}


def test_prompts_carry_no_held_out_words_brands_or_local_paths() -> None:
    root = REPO / "src" / "rtt" / "studio" / "engineer"
    private = re.compile(r"/home/|/Users/")  # no absolute local paths
    for path in [*root.glob("prompts/*.md"), *root.glob("*.py")]:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".md":
            assert not HELDOUT.search(text) and not find_brand_tokens(text), path.name
        assert not private.search(text), path.name
    for text in (render_templates(), render_primitives(), render_rules(), render_example()):
        assert not HELDOUT.search(text) and not find_brand_tokens(text)


def test_the_worked_example_is_a_valid_plan_in_schema_order() -> None:
    spec = worked_example()
    assert check_plan(spec) == ([], [])
    rendered = json.loads(render_example())
    assert rendered == json.loads(json.dumps(spec.to_tool_args()))
    validator = Validator(task_plan_json_schema())
    assert validator.errors(rendered) == []
    step_keys = list(task_plan_json_schema()["properties"]["steps"]["items"]["anyOf"][0]["properties"])
    assert list(rendered["steps"][0]) == step_keys
    # the example is a different use case from every gate description
    for case in (*GATE_CASES, *V1_GATE_CASES):
        assert "small box" not in case.text and "parts drawer" not in case.text


def test_sanitize_scrubs_brands_and_held_out_words_and_bounds_length() -> None:
    clean, notes = sanitize_use_case("  Pick the pipe elbow  with a Allegro\tgripper \n\n and a spur gear ")
    assert "elbow" not in clean and "Allegro" not in clean and "gear" not in clean
    assert clean.count("[removed]") == 4 and "\n" in clean and "  " not in clean
    assert notes and "replaced" in notes[0]
    long, notes = sanitize_use_case("a" * 5000)
    assert len(long) == 2000 and "cut" in notes[0]
    with pytest.raises(ValueError):
        sanitize_use_case(" \n ")


# --------------------------------------------------------------------------- checks


def _spec(objects: Sequence[ObjectInstance], steps: Sequence[Step], **kwargs: Any) -> TaskSpec:
    return TaskSpec("Test", "A test plan.", tuple(objects), tuple(steps), source="test", **kwargs)


def _crit(q: str, lo: float | None = None, hi: float | None = None) -> SuccessCriterion:
    return SuccessCriterion(q, at_least=lo, at_most=hi)


def test_checks_need_a_goal_and_keep_targets_and_holding_consistent() -> None:
    part = ObjectInstance.of("box", "part", "Box", ObjectPose(0, 0, 0))
    tote = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(200, 150, 0))
    only_limits = Step.of("pick", "grasp", "box", "part", "Pick it.", [_crit("force_n", hi=20)])
    problems, _ = check_plan(_spec([part, tote], [only_limits]))
    assert len(problems) == 1 and "add a criterion on what the step achieves" in problems[0]
    no_target = Step.of("pick", "grasp", "box", "part", "Pick it.", [_crit("in_target", 1)])
    assert "in_target needs a target bin" in check_plan(_spec([part, tote], [no_target]))[0][0]
    ok = Step.of("pick", "grasp", "box", "part", "Pick it.", [_crit("in_target", 1)], target="tote")
    assert check_plan(_spec([part, tote], [ok])) == ([], [])
    lifted = Step.of(
        "pick", "grasp", "box", "part", "Pick it.", [_crit("lift_mm", 150)], params={"lift_mm": 100.0}
    )
    assert "more than the step's own lift_mm" in check_plan(_spec([part, tote], [lifted]))[0][0]
    warnings = check_plan(_spec([part, tote], [lifted]))[1]
    assert warnings == ["objects[1] (tote): no step acts on it or places into it"]


def test_a_plug_insert_picks_its_plug_and_criteria_must_be_reachable() -> None:
    port = ObjectInstance.of("port", "port_plug", "Port", ObjectPose(0, 0, 0), depth_mm=15)
    insert = Step.of("plug_in", "insert", "port", "plug", "Plug it in.", [_crit("depth_mm", 12)])
    grasp = Step.of(
        "take", "grasp", "port", "plug", "Take the plug.", [_crit("lift_mm", 20)], hold_after=True
    )
    assert check_plan(_spec([port], [insert])) == ([], [])  # S2's insert picks the plug itself
    assert check_plan(_spec([port], [grasp, insert])) == ([], [])  # or uses the held plug
    deep = Step.of("plug_in", "insert", "port", "plug", "Plug it in.", [_crit("depth_mm", 20)])
    assert "cannot be reached" in check_plan(_spec([port], [grasp, deep]))[0][0]
    lid = ObjectInstance.of("lid", "hinged_lid", "Lid", ObjectPose(0, 0, 0), max_angle_deg=100)
    for value, n in ((95, 0), (120, 1)):
        step = Step.of("open", "hook_pull", "lid", "handle", "Open it.", [_crit("angle_deg", value)])
        assert len(check_plan(_spec([lid], [step]))[0]) == n
    latch = ObjectInstance.of("latch", "quarter_turn_latch", "Latch", ObjectPose(0, 0, 0), turn_deg=90)
    far = Step.of(
        "turn", "twist", "latch", "head", "Turn it.", [_crit("angle_deg", 120)], params={"direction": "ccw"}
    )
    assert "120 cannot be reached" in check_plan(_spec([latch], [far]))[0][0]
    drawer = ObjectInstance.of(  # its handle within reach out to its travel (rule set v2)
        "drawer", "drawer", "Drawer", ObjectPose(0, 150, 0), travel_mm=200, pull_force_n=30
    )
    pull = Step.of(
        "open",
        "hook_pull",
        "drawer",
        "handle",
        "Open it.",
        [_crit("travel_mm", 150)],
        constraints=StepConstraints(max_force_n=20),
    )
    problems = check_plan(_spec([drawer], [pull]))[0]
    assert problems == [
        "steps[0] (open): max_force_n 20 is below the 30 N it takes (the drawer's pull_force_n)"
    ]
    shut = Step.of("shut", "push", "drawer", "drawer", "Shut it.", [_crit("travel_mm", hi=5)])
    assert check_plan(_spec([drawer], [shut]))[0] == []  # 30 N fits the default 30 N push
    heavy = ObjectInstance.of("drawer", "drawer", "Drawer", ObjectPose(0, 150, 0), pull_force_n=35)
    assert "without a max_force_n the push stops at 30 N" in check_plan(_spec([heavy], [shut]))[0][0]
    capped = Step.of(
        "shut",
        "push",
        "drawer",
        "drawer",
        "Shut it.",
        [_crit("travel_mm", hi=5)],
        constraints=StepConstraints(max_force_n=40),
    )
    assert "lets the push reach 32 N (80 %)" in check_plan(_spec([heavy], [capped]))[0][0]


def _thread(**params: Any) -> ObjectInstance:
    base = {"engage_turns": 3, "start_torque_nm": 0.2, "final_torque_nm": 6, "pitch_mm": 1.5, "chamfer_mm": 1}
    return ObjectInstance.of("nut", "threaded_joint", "Nut", ObjectPose(0, 0, 0), **{**base, **params})


def _insert(**kwargs: Any) -> Step:
    return Step.of(
        "start", "insert", "nut", "member", "Push it onto the thread.", [_crit("depth_mm", 4)], **kwargs
    )


def _turn(rows: Sequence[SuccessCriterion], direction: str = "cw", **kwargs: Any) -> Step:
    return Step.of(
        "tighten", "twist", "nut", "member", "Run it down.", rows, params={"direction": direction}, **kwargs
    )


def test_a_threaded_member_is_inserted_then_twisted_clockwise_and_never_picked() -> None:
    nut = _thread()
    assert check_plan(_spec([nut], [_insert(), _turn([_crit("turns", 3)])])) == ([], [])
    pick = Step.of("take", "grasp", "nut", "member", "Take the nut.", [_crit("lift_mm", 20)])
    problems = check_plan(_spec([nut], [pick, _insert(), _turn([_crit("turns", 3)])]))[0]
    assert problems == [
        (
            "steps[0] (take): a threaded member lives on its stub and cannot be picked up or moved; leave the pick "
            "out (insert and twist grip the member in place)"
        )
    ]
    alone = check_plan(_spec([nut], [_turn([_crit("turns", 3)])]))[0]
    assert alone == [
        (
            "steps[0] (tighten): a threaded member starts on its lead-in, free of the thread; insert 'nut' to the "
            "thread start before this twist"
        )
    ]
    backwards = check_plan(_spec([nut], [_insert(), _turn([_crit("turns", 3)], "ccw")]))[0]
    assert backwards == [
        (
            "steps[1] (tighten): a threaded member tightens clockwise from its start; twist it cw (ccw meets the "
            "thread start)"
        )
    ]
    held = check_plan(_spec([nut], [_insert(hold_after=True), _turn([_crit("turns", 3)])]))[0]
    assert held == [
        (
            "steps[0] (start): end the insert of a threaded member with hold_after false; the twist grips it "
            "again in place"
        )
    ]
    deep = Step.of("start", "insert", "nut", "member", "Push it.", [_crit("depth_mm", 6)])
    assert check_plan(_spec([nut], [deep]))[0] == [
        (
            "steps[0] (start): depth_mm 6 cannot be reached; 'nut' stops at 4 (the thread start, chamfer_mm + 3 mm; "
            "a twist runs it further)"
        )
    ]
    seated = _turn([_crit("depth_mm", 8.5)])  # 4 mm lead-in + 3 turns x 1.5 mm
    assert check_plan(_spec([nut], [_insert(), seated])) == ([], [])
    snug = _turn([_crit("torque_nm", 5)])  # tighten until snug: a goal on its own
    assert check_plan(_spec([nut], [_insert(), snug])) == ([], [])


def test_latches_and_valves_open_counter_clockwise_and_take_no_torque_stop() -> None:
    latch = ObjectInstance.of("latch", "quarter_turn_latch", "Latch", ObjectPose(0, 0, 0), torque_nm=0.8)
    opens = Step.of(
        "open", "twist", "latch", "head", "Open it.", [_crit("angle_deg", 85)], params={"direction": "ccw"}
    )
    assert check_plan(_spec([latch], [opens])) == ([], [])
    wrong = Step.of(
        "open", "twist", "latch", "head", "Open it.", [_crit("angle_deg", 85)], params={"direction": "cw"}
    )
    assert check_plan(_spec([latch], [wrong]))[0] == [
        (
            "steps[0] (open): a latch starts closed at its stop and opens counter-clockwise; twist it ccw (cw "
            "meets the stop)"
        )
    ]
    back = Step.of(
        "lock",
        "twist",
        "latch",
        "head",
        "Lock it again.",
        [_crit("angle_deg", hi=5)],
        params={"direction": "cw"},
    )
    assert check_plan(_spec([latch], [opens, back])) == ([], [])  # turning back what an earlier twist opened
    stop = Step.of(
        "open",
        "twist",
        "latch",
        "head",
        "Open it.",
        [_crit("angle_deg", 85), _crit("torque_nm", 0.8)],
        params={"direction": "ccw"},
    )
    assert (
        "a twist stops as soon as the roll torque reaches a torque_nm at_least"
        in check_plan(_spec([latch], [stop]))[0][0]
    )
    valve = ObjectInstance.of("valve", "handwheel_valve", "Valve", ObjectPose(0, 0, 0), travel_deg=90)
    shut = Step.of(
        "shut", "twist", "valve", "wheel", "Close it.", [_crit("angle_deg", hi=5)], params={"direction": "cw"}
    )
    assert "a valve starts closed at its stop" in check_plan(_spec([valve], [shut]))[0][0]


def test_twist_torque_is_checked_against_its_share_the_wrist_and_the_criterion() -> None:
    nut = _thread()
    start = _turn([_crit("turns", 1)], constraints=StepConstraints(max_torque_nm=3))
    # one turn of three: 0.2 + (6 - 0.2) / 3 = 2.13 N m, under 90 % of 3 N m
    assert check_plan(_spec([nut], [_insert(), start])) == ([], [])
    seat = _turn([_crit("turns", 3), _crit("torque_nm", hi=4)], constraints=StepConstraints(max_torque_nm=6))
    problems = check_plan(_spec([nut], [_insert(), seat]))[0]
    why = "the thread's torque there, start_torque_nm rising to final_torque_nm"
    assert problems == [
        f"steps[1] (tighten): max_torque_nm 6 lets the twist turn up to 5.4 N m (90 %), below the 6 N m it takes ({why})",
        f"steps[1] (tighten): torque_nm at_most 4 is below the 6 N m it takes ({why})",
    ]
    valve = ObjectInstance.of(
        "valve", "handwheel_valve", "Valve", ObjectPose(0, 0, 0), torque_nm=14.5, travel_deg=360
    )
    turn = Step.of(
        "turn", "twist", "valve", "wheel", "Turn it.", [_crit("angle_deg", 360)], params={"direction": "ccw"}
    )
    assert (
        "but a twist turns at most 14.25 N m (95 % of the wrist roll's 15 N m)"
        in check_plan(_spec([valve], [turn]))[0][0]
    )


def test_a_lid_needs_a_120_mm_bin_under_it_and_a_low_stop_is_warned_about() -> None:
    tote = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0, 0, 0), height_mm=200)
    lid = ObjectInstance.of(
        "lid", "hinged_lid", "Lid", ObjectPose(0, 0, 0), max_angle_deg=85, width_mm=330, depth_mm=230
    )
    open_lid = Step.of("open", "hook_pull", "lid", "handle", "Open it.", [_crit("angle_deg", 80)])
    problems, warnings = check_plan(_spec([tote, lid], [open_lid]))
    assert problems == [
        (
            "objects[1] (lid): the lid template's hinge height is fixed at 120 mm, so the bin 'tote' it covers must "
            "be 120 mm tall (it is 200 mm); if the use case states another height, put that lid on that bin in "
            "unsupported (not modelled) instead of changing it"
        )
    ]
    assert warnings == [
        "objects[0] (tote): no step acts on it or places into it",
        (
            "objects[1] (lid): a lid stays open only past vertical; with max_angle_deg 85 it falls shut when the "
            "tool lets go"
        ),
    ]
    fits = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0, 0, 0), height_mm=120)
    assert check_plan(_spec([fits, lid], [open_lid]))[0] == []
    apart = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0, -260, 0), height_mm=200)
    assert check_plan(_spec([apart, lid], [open_lid]))[0] == []  # not under the lid, and clear of it
    door = ObjectInstance.of("door", "hinged_lid", "Door", ObjectPose(0, 0, 0), orientation="door")
    swing = Step.of("open", "hook_pull", "door", "handle", "Open it.", [_crit("angle_deg", 80)])
    on_door = check_plan(_spec([tote, door], [swing]))[0]  # a door is not a lid on a bin ...
    assert (
        len(on_door) == 1 and "overlaps the footprint of objects[0] (tote)" in on_door[0]
    )  # ... it overlaps


def test_a_lip_lid_overhangs_its_bin_by_feature_mm_on_every_side_and_a_handle_lid_need_not() -> None:
    tote = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0, 0, 0), height_mm=120)  # 300 x 200 mm

    def lid(feature: str = "lip", w: float = 300, d: float = 200, lip: float = 15) -> ObjectInstance:
        return ObjectInstance.of(
            "lid",
            "hinged_lid",
            "Lid",
            ObjectPose(0, 0, 0),
            feature=feature,
            feature_mm=lip,
            width_mm=w,
            depth_mm=d,
        )

    open_lid = Step.of("open", "hook_pull", "lid", "handle", "Open it.", [_crit("angle_deg", 110)])
    assert check_plan(_spec([tote, lid()], [open_lid]))[0] == [
        (
            "objects[1] (lid): a lip lid is opened by a hook under its free edge, so it overhangs its bin by "
            "feature_mm (15 mm) on every side: at least 330 x 230 mm over the bin 'tote' (300 x 200 mm), it is "
            "300 x 200 mm; make it larger or give it a handle"
        )
    ]
    assert check_plan(_spec([tote, lid("handle")], [open_lid]))[0] == []  # a handle lid may match its bin
    assert check_plan(_spec([tote, lid(w=330, d=230)], [open_lid]))[0] == []
    assert check_plan(_spec([tote, lid(w=310, d=210, lip=5)], [open_lid]))[0] == []
    assert len(check_plan(_spec([tote, lid(w=329, d=230)], [open_lid]))[0]) == 1  # 1 mm short along x
    assert len(check_plan(_spec([tote, lid(w=330, d=229)], [open_lid]))[0]) == 1  # ... and along y
    # a bin turned a quarter turn under the lid: its 300 mm side now runs along the lid's depth
    turned = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0, 0, 90), height_mm=120)
    assert (
        "at least 230 x 330 mm over the bin 'tote' (200 x 300 mm)"
        in check_plan(_spec([turned, lid(w=330, d=230)], [open_lid]))[0][0]
    )
    assert check_plan(_spec([turned, lid(w=230, d=330)], [open_lid]))[0] == []
    # a lid that is not over the bin owes it nothing
    apart = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0, -260, 0), height_mm=120)
    assert check_plan(_spec([apart, lid()], [open_lid]))[0] == []


def _inside(px: float, py: float, poly: list[tuple[float, float]], tol: float) -> bool:
    """``(px, py)`` lies in the convex polygon or within ``tol`` mm of it."""
    import math as _m

    n = len(poly)
    signs = []
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        signs.append((x1 - x0) * (py - y0) - (y1 - y0) * (px - x0))
    if all(v >= 0 for v in signs) or all(v <= 0 for v in signs):
        return True
    best = _m.inf
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        den = dx * dx + dy * dy or 1.0
        u = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / den))
        best = min(best, _m.hypot(px - x0 - u * dx, py - y0 - u * dy))
    return best <= tol


_FOOTPRINT_VARIANTS = {
    "bin": ("bin", {"length_mm": 260, "width_mm": 180}),
    "box": ("part", {"size_x_mm": 70, "size_y_mm": 30}),
    "cylinder": ("part", {"shape": "cylinder", "size_x_mm": 50}),
    "latch": ("quarter_turn_latch", {"head_diameter_mm": 40}),
    "thread": ("threaded_joint", {"diameter_mm": 50}),
    "valve": ("handwheel_valve", {"rim_diameter_mm": 160}),
    "port": ("port_plug", {"plug_diameter_mm": 20, "chamfer_mm": 2}),
    "drawer_bar": ("drawer", {"travel_mm": 150}),
    "drawer_knob": ("drawer", {"handle": "knob", "travel_mm": 150}),
    "lid_lip": ("hinged_lid", {"width_mm": 200, "depth_mm": 150}),
    "lid_handle": (
        "hinged_lid",
        {"width_mm": 200, "depth_mm": 150, "feature": "handle", "max_angle_deg": 120},
    ),
    "door_lip": ("hinged_lid", {"orientation": "door", "width_mm": 200, "depth_mm": 160}),
    "door_handle": (
        "hinged_lid",
        {"orientation": "door", "width_mm": 200, "depth_mm": 160, "feature": "handle"},
    ),
    # Studio v2's job templates
    "panel_rail": ("panel_rail", {"length_mm": 600, "width_mm": 400}),
    "tile_bed": ("tile_bed", {"tile_mm": 200, "mass_g": 1000}),
    "button_wall": ("spring_button", {"diameter_mm": 80, "proud_mm": 6}),
    "button_desk": ("spring_button", {"mount": "desk"}),
    "sliding_door": ("sliding_door", {"travel_mm": 200}),
}


@pytest.mark.parametrize("variant", sorted(_FOOTPRINT_VARIANTS))
@pytest.mark.parametrize("yaw", [0.0, 30.0])
def test_footprints_and_swings_hold_the_built_geometry(variant: str, yaw: float) -> None:
    """The plan-view footprint (closed) and footprint plus swing (at the stop) contain every geom S2's
    builder makes, within 5 mm (the swing's arc is drawn as chords)."""
    import mujoco
    import numpy as np

    from rtt.studio.engineer.layout import regions
    from rtt.studio.scene import build_studio_scene
    from rtt.studio.types import Setup
    from studio_fixtures import gripper

    kind, params = _FOOTPRINT_VARIANTS[variant]
    obj = ObjectInstance.of("thing", kind, "Thing", ObjectPose(20.0, -30.0, yaw), **params)
    probe = ObjectInstance.of(
        "probe", "part", "Probe", ObjectPose(-280.0, -280.0, 0.0), size_x_mm=10, size_y_mm=10
    )
    step = Step.of("s", "push", "probe", "part", "Probe.", [_crit("tilt_deg", hi=30)])
    spec = TaskSpec("Footprint", "", (obj, probe), (step,), source="test")
    scene = build_studio_scene(Setup.single(gripper(), spec.step_ids), spec, 0)
    m, d = scene.model, mujoco.MjData(scene.model)
    handles = scene.objects["thing"]
    polys = {name: poly for name, poly in regions(obj)}
    states = [("closed", ["footprint"])]
    joint = next((j for role, j in handles.handles.joints.items() if role in ("hinge", "slide")), None)
    if kind in ("drawer", "hinged_lid", "sliding_door") and joint is not None:
        states.append(("open", ["footprint", "swing"]))
    for state, names in states:
        mujoco.mj_resetData(m, d)
        if state == "open":
            jid = m.joint(joint).id
            d.qpos[m.jnt_qposadr[jid]] = m.jnt_range[jid][1]
        mujoco.mj_forward(m, d)
        for g in handles.geom_ids:
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        p = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (c + h * np.array([sx, sy, sz]))
                        px, py = 1000.0 * p[0], 1000.0 * p[1]
                        assert any(_inside(px, py, polys[n], 5.0) for n in names if n in polys), (
                            variant,
                            state,
                            m.geom(g).name,
                            round(px, 1),
                            round(py, 1),
                        )


def test_layout_refuses_overlaps_and_swings_but_not_a_lid_on_its_bin() -> None:
    from rtt.studio.engineer.layout import CLEARANCE_MM, layout_problems

    tote = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0, 0, 0), height_mm=120)
    lid = ObjectInstance.of("lid", "hinged_lid", "Lid", ObjectPose(0, 0, 0), width_mm=300, depth_mm=200)
    inside = ObjectInstance.of("box", "part", "Box", ObjectPose(0, 0, 0))
    pick = Step.of("pick", "grasp", "box", "part", "Pick it.", [_crit("in_target", 1)], target="tote")
    problems = layout_problems([tote, lid, inside], [pick])
    assert problems == [
        (
            "objects[2] (box): its footprint overlaps the footprint of objects[0] (tote); keep them at least 20 mm "
            "apart; a part picked into a bin starts outside it"
        ),
        "objects[2] (box): its footprint overlaps the footprint of objects[1] (lid); keep them at least 20 mm apart",
    ]
    beside = ObjectInstance.of(
        "box", "part", "Box", ObjectPose(150 + 20 + 15, -200, 0), size_x_mm=30, size_y_mm=30
    )
    assert layout_problems([tote, lid, beside], [pick]) == []  # the lid on its tote is allowed
    close = ObjectInstance.of(
        "box", "part", "Box", ObjectPose(150 + 10 + 15, 0, 0), size_x_mm=30, size_y_mm=30
    )
    assert "comes within 20 mm of the footprint of objects[0] (tote)" in layout_problems([tote, close], [])[0]
    door = ObjectInstance.of(
        "door", "hinged_lid", "Door", ObjectPose(0, 0, 0), orientation="door", depth_mm=200
    )
    latch = ObjectInstance.of("latch", "quarter_turn_latch", "Latch", ObjectPose(0, -100, 0))
    assert layout_problems([door, latch], []) == [
        "objects[1] (latch): its footprint overlaps the swing of objects[0] (door); keep them at least 20 mm apart"
    ]
    clear = ObjectInstance.of("latch", "quarter_turn_latch", "Latch", ObjectPose(150, 60, 0))
    assert layout_problems([door, clear], []) == []
    drawer = ObjectInstance.of(
        "drawer", "drawer", "Drawer", ObjectPose(0, 150, 0), depth_mm=200, travel_mm=150
    )
    in_travel = ObjectInstance.of("box", "part", "Box", ObjectPose(0, -100, 0))
    assert "overlaps the swing of objects[0] (drawer)" in layout_problems([drawer, in_travel], [])[0]
    assert CLEARANCE_MM == 20.0


def test_windows_leave_room_for_measurement_and_flags_are_exempt() -> None:
    from rtt.studio.engineer.checks import WINDOW_MIN

    port = ObjectInstance.of("port", "port_plug", "Port", ObjectPose(0, 0, 0), depth_mm=15)
    exact = Step.of("plug_in", "insert", "port", "plug", "Plug it in.", [_crit("depth_mm", 15, 15)])
    assert check_plan(_spec([port], [exact]))[0] == [
        (
            "steps[0] (plug_in): the depth_mm window 15 to 15 is narrower than 1 mm; a measurement never lands "
            "exactly, so widen it or keep one bound"
        )
    ]
    ok = Step.of("plug_in", "insert", "port", "plug", "Plug it in.", [_crit("depth_mm", 14, 15)])
    assert check_plan(_spec([port], [ok]))[0] == []
    tote = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(150, 0, 0), length_mm=200, width_mm=200)
    box = ObjectInstance.of("box", "part", "Box", ObjectPose(-150, 0, 0))
    flag = Step.of("pick", "grasp", "box", "part", "Pick it.", [_crit("in_target", 1, 1)], target="tote")
    assert check_plan(_spec([tote, box], [flag]))[0] == []
    assert WINDOW_MIN["angle_deg"] == 5.0 and WINDOW_MIN["turns"] == 0.25 and "in_target" not in WINDOW_MIN


def test_the_simulator_numbers_match_the_template_and_primitive_modules() -> None:
    from types import SimpleNamespace

    from rtt.studio.engineer import catalog as cat
    from rtt.studio.engineer.checks import DEPTH_TOLERANCE_MM
    from rtt.studio.primitives import insert, push, twist
    from rtt.studio.primitives._common import force_limit, torque_limit
    from rtt.studio.templates.handwheel_valve import WHEEL_Z
    from rtt.studio.templates.hinged_lid import LID_HINGE_HEIGHT
    from rtt.studio.templates.threaded_joint import lead_of

    for kind, sense in cat.TWIST_SENSE.items():  # a panel's clamp turns as a threaded member
        assert sense == ("ccw" if twist.axis_site(kind)[2] > 0 else "cw"), kind
    assert set(cat.TWIST_SENSE) == set(twist.AXIS_SITE) | {TemplateKind.PANEL_RAIL}
    assert lead_of(0.0) * 1000.0 == pytest.approx(cat.THREAD_LEAD_EXTRA_MM)
    assert insert.PUSH_N == push.PUSH_N == cat.DEFAULT_PUSH_N
    ctx = SimpleNamespace(
        step=SimpleNamespace(constraints=StepConstraints(max_force_n=10, max_torque_nm=10)),
        scene=SimpleNamespace(wrist=SimpleNamespace(roll_torque_nm=15.0)),
    )
    assert force_limit(ctx, 30.0) == pytest.approx(cat.FORCE_SHARE * 10)  # type: ignore[arg-type]
    assert torque_limit(ctx) == pytest.approx(cat.TORQUE_SHARE * 10)  # type: ignore[arg-type]
    ctx.step.constraints = StepConstraints()
    assert torque_limit(ctx) == pytest.approx(cat.wrist_twist_limit())  # type: ignore[arg-type]
    assert LID_HINGE_HEIGHT * 1000.0 == pytest.approx(cat.LID_HINGE_MM)
    assert WHEEL_Z * 1000.0 == pytest.approx(cat.WHEEL_HEIGHT_MM)
    assert web_catalog()["sim"]["depth_tolerance_mm"] == DEPTH_TOLERANCE_MM


def test_refs_and_refusals() -> None:
    mesh = ObjectInstance.of(
        "tub", "part", "Tub", ObjectPose(0, 0, 0), ref="upload:stl-0123456789ab", shape="mesh"
    )
    pick = Step.of("pick", "grasp", "tub", "part", "Pick it.", [_crit("lift_mm", 50)])
    spec = _spec([mesh], [pick])
    assert "was not uploaded" in check_plan(spec)[0][0]
    assert check_plan(spec, uploads=["stl-0123456789ab"]) == ([], [])
    box = ObjectInstance.of("box", "part", "Box", ObjectPose(0, 0, 0))
    unused = check_plan(
        _spec([box], [Step.of("pick", "grasp", "box", "part", "Pick it.", [_crit("lift_mm", 50)])]),
        uploads=["stl-0123456789ab"],
        required_uploads=["stl-0123456789ab"],
    )[0]
    assert unused and "is not used" in unused[0]
    scan = ObjectInstance.of("scan", "part", "Scan", ObjectPose(0, 0, 0), ref="scan:abc", shape="mesh")
    assert (
        "scan refs"
        in check_plan(_spec([scan], [Step.of("p", "grasp", "scan", "part", "P.", [_crit("lift_mm", 9)])]))[0][
            0
        ]
    )
    lib = ObjectInstance.of("tee", "part", "Tee", ObjectPose(0, 0, 0), ref="library:pipe_tee", shape="mesh")
    assert (
        check_plan(_spec([lib], [Step.of("p", "grasp", "tee", "part", "P.", [_crit("lift_mm", 9)])]))[0] == []
    )
    bad = ObjectInstance.of("x", "part", "X", ObjectPose(0, 0, 0), ref="library:no_such_part", shape="mesh")
    assert (
        "no library part"
        in check_plan(_spec([bad], [Step.of("p", "grasp", "x", "part", "P.", [_crit("lift_mm", 9)])]))[0][0]
    )
    refused = TaskSpec("No", "", (), (), refusal="Cloth.", source="test")
    assert "a refusal lists what is not simulated" in check_plan(refused)[0][0]
    ok = judge_args(refusal_args())
    assert ok[0] is not None and ok[1] == [] and ok[0].refusal
    assert "name the nearest primitive" in judge_args(refusal_args(None))[1][0]


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Folding a T-shirt", "grasp"),
        ("Pour the coffee", "grasp"),
        ("Unscrew the jar cap", "twist"),
        ("Plug in the cable connector", "insert"),
        ("Open the hatch", "hook_pull"),
        ("Close the drawer", "push"),
        ("Peel the film off the glass", "suction"),
        ("Weld the seam", None),
    ],
)
def test_nearest_primitive_keywords(text: str, kind: str | None) -> None:
    got = nearest_primitive(text)
    assert (got.value if got else None) == kind


def test_null_text_is_read_as_null_and_terse_problems_are_explained() -> None:
    args = good_args()
    args["refusal"] = "None"
    args["steps"][0]["target"] = "null"
    fixed, notes = normalise_nulls(args)
    assert fixed["refusal"] is None and fixed["steps"][0]["target"] is None and "refusal" in notes[0]
    assert args["refusal"] == "None"  # the input is not changed
    assert judge_args(fixed)[1] == []
    broken = good_args()
    del broken["steps"][1]["description"]
    _, problems, _ = judge_args(broken)
    assert problems[0] == "steps[1] (pick_box): missing 'description': One plain sentence for the step list"
    assert explain_problems(["a refused task has no steps"], {})[0].startswith("refusal: must be null")


def test_judge_lists_parse_problems_and_plan_checks_together() -> None:
    args = good_args()
    args["steps"][1]["target"] = "nowhere"
    args["steps"][2]["success"] = [{"quantity": "force_n", "at_least": None, "at_most": 40}]
    spec, problems, _ = judge_args(args)
    assert spec is None
    assert problems == [
        "step pick_box: unknown target 'nowhere'",
        (
            "steps[2] (close_drawer): add a criterion on what the step achieves (travel_mm); force, torque, slip "
            "and penetration limits alone do not say when it succeeded"
        ),
    ]
    nonsense = judge_args({"title": "x", "objects": 5, "steps": "no"})
    assert nonsense[0] is None and nonsense[1]


# --------------------------------------------------------------------------- planning (scripted)


def test_a_valid_first_call_is_one_call(tmp_path: Path) -> None:
    planner = ScriptedPlanner([good_args()])
    result = plan_task(
        "Pull the drawer open, put the box in the tote, shut the drawer.", client=client_for(planner)
    )
    assert result.status == "valid" and result.first_try and not result.repaired
    assert len(planner.payloads) == 1 and result.spec is not None
    payload = planner.payloads[0]
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "submit_task_plan"}}
    assert payload["model"] == ENGINEER_MODEL and payload["tools"][0] == engineer_tool()
    record = result.to_dict()
    assert record["label"] == PRODUCT_MODE_LABEL and record["plan"]["task_id"] == result.spec.task_id
    assert record["status"] == "valid" and record["attempts"][0]["cost_usd"] > 0


def test_one_repair_quotes_the_previous_call_and_every_problem() -> None:
    bad = good_args()
    bad["steps"][1]["target"] = "nowhere"
    bad["objects"][0]["params"]["travel_mm"] = 900
    planner = ScriptedPlanner([bad, good_args()])
    result = plan_task("Drawer, box, tote.", client=client_for(planner))
    assert result.status == "valid" and result.repaired and len(result.attempts) == 2
    assert result.attempts[0].outcome == "invalid" and len(result.attempts[0].problems) == 1
    repair = planner.payloads[1]["messages"][-1]["content"]
    assert "# Your plan was refused" in repair and '"target":"nowhere"' in repair
    assert "travel_mm must be in [20, 500] mm" in repair
    assert planner.payloads[1]["messages"][0] == planner.payloads[0]["messages"][0]


def test_still_invalid_after_the_repair_is_reported_not_fixed() -> None:
    bad = good_args()
    bad["steps"][2]["hold_after"] = True
    planner = ScriptedPlanner([bad, bad])
    result = plan_task("Drawer, box, tote.", client=client_for(planner))
    assert result.status == "invalid" and not result.ok and len(planner.payloads) == 2
    assert result.problems == ("steps[2]: step close_drawer: a push step cannot end holding its object",)
    one = plan_task("Drawer, box, tote.", client=client_for(ScriptedPlanner([bad])), repairs=0)
    assert one.status == "invalid" and len(one.attempts) == 1


def test_unreadable_arguments_get_a_repair_and_failures_are_failed() -> None:
    planner = ScriptedPlanner(["{not json at all", good_args()])
    result = plan_task("Drawer, box, tote.", client=client_for(planner))
    assert result.status == "valid" and result.attempts[0].outcome == "parse_error"
    assert "no usable submit_task_plan call" in planner.payloads[1]["messages"][-1]["content"]
    failed = plan_task("Drawer, box, tote.", client=client_for(ScriptedPlanner([400, 400])))
    assert failed.status == "failed" and [a.outcome for a in failed.attempts] == [
        "call_failed",
        "call_failed",
    ]
    # a failed repair keeps the first plan's problems
    bad = good_args()
    bad["steps"][2]["hold_after"] = True
    kept = plan_task("Drawer, box, tote.", client=client_for(ScriptedPlanner([bad, 400])))
    assert kept.status == "invalid" and kept.problems and "repair call failed" in kept.notes[0]


def test_the_spend_cap_stops_before_any_call() -> None:
    planner = ScriptedPlanner([good_args()])
    result = plan_task("Drawer, box, tote.", client=client_for(planner, budget=0.001))
    assert result.status == "failed" and planner.payloads == []
    assert [a.outcome for a in result.attempts] == ["spend_cap"]


def test_a_refusal_without_a_nearest_primitive_is_completed_from_the_keywords() -> None:
    planner = ScriptedPlanner([refusal_args(None), refusal_args(None)])
    result = plan_task("Fold the shirt and put it away.", client=client_for(planner))
    assert result.status == "refused" and result.spec is not None
    assert result.spec.unsupported[0].nearest_primitive is PrimitiveKind.GRASP
    assert "keyword table" in result.notes[-1]
    direct = plan_task("Fold the shirt.", client=client_for(ScriptedPlanner([refusal_args("wrap")])))
    assert direct.status == "refused" and direct.first_try


def test_model_text_is_scrubbed_reasoning_dropped_and_replays_are_exact(tmp_path: Path) -> None:
    args = good_args()
    args["description"] = "Like a Allegro gripper would, near the pipe elbow."
    args["assumptions"] = ["A Festo cup works."]
    planner = ScriptedPlanner([args], reasoning="I think Allegro ...", content="Schunk")
    live = plan_task("Drawer, box, tote.", client=client_for(planner, cache=tmp_path / "cache"))
    assert live.status == "valid" and live.spec is not None
    assert "[removed]" in live.spec.description and "[removed]" in live.spec.assumptions[0]
    files = list((tmp_path / "cache").rglob("*.json"))
    assert len(files) == 1
    stored = json.loads(files[0].read_text(encoding="utf-8"))["response"]
    stored_text = json.dumps(stored)
    assert "reasoning_content" not in stored_text and not find_brand_tokens(stored_text)
    assert not HELDOUT.search(stored_text)
    replay = plan_task(
        "Drawer, box, tote.", client=make_engineer_client(tmp_path / "cache", cache_mode="replay")
    )
    assert replay.to_dict()["plan"] == live.to_dict()["plan"] and replay.attempts[0].cached
    with pytest.raises(CacheMiss):
        plan_task("Something new.", client=make_engineer_client(tmp_path / "cache", cache_mode="replay"))
    assert isinstance(make_engineer_client(tmp_path / "c").cache, EngineerCache)
    assert isinstance(make_engineer_client(tmp_path / "c").cache, ResponseCache)


def test_user_text_with_reserved_words_is_scrubbed_before_sending() -> None:
    planner = ScriptedPlanner([good_args()])
    result = plan_task(
        "Thread a fitting onto the pipe elbow with an OnRobot tool.", client=client_for(planner)
    )
    sent = json.dumps(planner.payloads[0]["messages"])
    assert not HELDOUT.search(sent) and not find_brand_tokens(sent)
    assert "[removed]" in result.use_case and result.notes


def test_an_uploaded_part_is_described_and_must_be_used() -> None:
    upload = UploadInfo("stl-0123456789ab", "bracket", (60.0, 30.0, 50.0), 42.0, "plastic", 3)
    messages = build_messages("Put the bracket in the tote.", [upload])
    assert "upload:stl-0123456789ab: bracket, 60 x 30 x 50 mm" in messages[1]["content"]
    args = good_args()
    unused = plan_task(
        "Put it in the tote.", client=client_for(ScriptedPlanner([args, args])), uploads=[upload]
    )
    assert unused.status == "invalid" and "upload:stl-0123456789ab is not used" in unused.problems[0]
    used = copy.deepcopy(args)
    used["objects"][1]["params"] = upload.part_params()
    used["objects"][1]["ref"] = upload.ref
    ok = plan_task("Put it in the tote.", client=client_for(ScriptedPlanner([used])), uploads=[upload])
    assert ok.status == "valid" and ok.to_dict()["uploads"][0]["ref"] == upload.ref


# --------------------------------------------------------------------------- uploads (real intake)


@pytest.fixture(scope="module")
def tub(tmp_path_factory: pytest.TempPathFactory) -> tuple[UploadRegistry, UploadInfo]:
    registry = UploadRegistry(tmp_path_factory.mktemp("uploads"))
    info = registry.add_stl(REPO / UPLOAD_EXAMPLE_STL, decomposition="hull")
    return registry, info


def test_an_stl_goes_through_the_part_intake_and_resolves_for_the_part_template(tub: Any) -> None:
    from rtt.studio.templates import TemplateUnavailable
    from rtt.studio.templates.part import part_spec_of

    registry, info = tub
    assert info.upload_id.startswith("stl-") and info.upload_id in registry
    assert registry.add_stl(REPO / UPLOAD_EXAMPLE_STL, decomposition="hull") == info  # content addressed
    assert info.part_params()["shape"] == "mesh" and info.mass_g > 0
    install_upload_resolver(registry)
    obj = ObjectInstance.of("tub", "part", "Tub", ObjectPose(0, 0, 0), ref=info.ref, **info.part_params())
    part = part_spec_of(obj)
    assert part.name == info.upload_id and part.family == "upload" and part.pieces
    missing = ObjectInstance.of(
        "x", "part", "X", ObjectPose(0, 0, 0), ref="upload:stl-ffffffffffff", shape="mesh"
    )
    with pytest.raises(TemplateUnavailable):
        part_spec_of(missing)
    with pytest.raises(ValueError):
        registry.part_dir("../etc")


def test_the_recorded_upload_plan_replays_and_builds_a_studio_scene(tub: Any) -> None:
    from rtt.studio.scene import build_studio_scene
    from rtt.studio.types import Setup
    from studio_fixtures import gripper

    registry, info = tub
    recorded = json.loads((RECORDED_UPLOAD / "plan.json").read_text(encoding="utf-8"))
    assert recorded["uploads"][0]["upload_id"] == info.upload_id
    assert recorded["uploads"][0]["size_mm"] == list(info.size_mm)
    # recorded on the v1 prompt: replayed from the model's recorded responses (rtt.studio.engineer.gate)
    client = responses_client(RECORDED_UPLOAD / "cache", [a["request_hash"] for a in recorded["attempts"]])
    result = plan_task(UPLOAD_CASE.text, client=client, uploads=[info])
    assert result.status == "valid" and result.spec is not None
    assert result.to_dict()["plan"] == recorded["plan"]
    assert any(o.ref == info.ref for o in result.spec.objects)
    install_upload_resolver(registry)
    pick = next(s for s in result.spec.steps if s.object in {o.name for o in result.spec.objects if o.ref})
    objects = tuple(o for o in result.spec.objects if o.name in (pick.object, pick.target))
    single = TaskSpec(
        "Pick the upload",
        "",
        objects,
        (
            Step.of(
                "pick", pick.primitive, pick.object, pick.handle, "Pick it.", pick.success, target=pick.target
            ),
        ),
        source="test",
    )
    scene = build_studio_scene(Setup.single(gripper(), single.step_ids), single, 0)
    assert scene.model.nbody > 0


# --------------------------------------------------------------------------- lid-open rule and layout fixer (S4e)


def _tote_scene(open_deg: float | None = 100.0) -> tuple[list[ObjectInstance], list[Step]]:
    tote = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0, 0, 0), height_mm=120)  # 300 x 200 mm
    lid = ObjectInstance.of("lid", "hinged_lid", "Lid", ObjectPose(0, 0, 0), width_mm=330, depth_mm=230)
    box = ObjectInstance.of("box", "part", "Box", ObjectPose(-250, -250, 0), size_x_mm=60, size_y_mm=40)
    crit = [_crit("angle_deg", open_deg)] if open_deg is not None else [_crit("angle_deg", None, 120)]
    steps = [
        Step.of("open", "hook_pull", "lid", "handle", "Open it.", crit),
        Step.of("pick", "grasp", "box", "part", "Pick it.", [_crit("in_target", 1)], target="tote"),
        Step.of("shut", "push", "lid", "lid", "Shut it.", [_crit("angle_deg", None, 5)]),
    ]
    return [tote, lid, box], steps


def test_a_pick_into_a_lidded_bin_needs_the_lid_opened_past_vertical_first() -> None:
    from rtt.studio.engineer.catalog import LID_OPEN_DEG

    assert LID_OPEN_DEG == 95.0
    objects, steps = _tote_scene(100)
    assert check_plan(_spec(objects, steps))[0] == []
    low = check_plan(_spec(*_tote_scene(30)))[0]
    assert low == [
        (
            "steps[0] (open): 'pick' picks into 'tote' under this lid later, so open it past vertical: angle_deg "
            "at_least 95 (it asks for 30); a lid opened less falls shut when the tool lets go"
        )
    ]
    assert "(it asks for no angle_deg at_least)" in check_plan(_spec(*_tote_scene(None)))[0][0]
    objects, steps = _tote_scene(100)
    never = check_plan(_spec(objects, [steps[1]]))[0]
    assert never == [
        (
            "steps[0] (pick): it picks into 'tote', which the lid 'lid' covers, and no earlier step opens that lid; "
            "hook_pull 'lid' first with angle_deg at_least 95 (past vertical: a lid opened less falls shut)"
        )
    ]
    shut_first = check_plan(_spec(objects, [steps[0], steps[2], steps[1]]))[0]
    assert shut_first == [
        (
            "steps[2] (pick): 'shut' pushes the lid 'lid' shut after 'open' opened it, so this pick into 'tote' "
            "meets a closed lid; hook_pull it open again (angle_deg at_least 95) after that push"
        )
    ]
    # one problem per hook pull, at its first pick
    other = ObjectInstance.of("tin", "part", "Tin", ObjectPose(250, -250, 0), size_x_mm=60, size_y_mm=40)
    two = [
        *_tote_scene(30)[1][:2],
        Step.of("pick2", "grasp", "tin", "part", "Pick.", [_crit("in_target", 1)], target="tote"),
    ]
    assert len(check_plan(_spec([*_tote_scene(30)[0], other], two))[0]) == 1
    # a bin without a lid, and a lid not over the bin, need nothing
    tote, _, box = objects
    plain = check_plan(_spec([tote, box], [steps[1]]))[0]
    assert plain == []
    away = ObjectInstance.of("lid", "hinged_lid", "Lid", ObjectPose(0, 250, 0), width_mm=200, depth_mm=40)
    assert not any("past vertical" in p for p in check_plan(_spec([tote, away, box], steps))[0])


def test_the_layout_fixer_moves_free_objects_to_the_nearest_clear_spot_deterministically() -> None:
    from rtt.studio.engineer.layout import fix_layout, layout_problems

    objects, steps = _tote_scene(100)
    tote, lid, _ = objects
    box = ObjectInstance.of("box", "part", "Box", ObjectPose(-150, 0, 0), size_x_mm=60, size_y_mm=40)
    scene = [tote, lid, box]
    assert layout_problems(scene, steps)
    fixed = fix_layout(scene, steps)
    again = fix_layout(scene, steps)
    assert fixed is not None and again is not None
    moved, moves = fixed
    assert moves == again[1] and [o.pose for o in moved] == [o.pose for o in again[0]]  # deterministic
    assert [m.text() for m in moves] == [
        "Moved box from (-150, 0) to (-220, 0) mm to clear the footprint of tote"
    ]
    assert moved[0].pose == tote.pose and moved[1].pose == lid.pose  # the target bin and its lid stay
    assert layout_problems(moved, steps) == []
    # a bin and its lid move together, when nothing a step places into has to stay
    door = ObjectInstance.of(
        "door", "hinged_lid", "Door", ObjectPose(0, 0, 0), orientation="door", depth_mm=200
    )
    crate = ObjectInstance.of(
        "crate", "bin", "Crate", ObjectPose(0, -150, 0), height_mm=120, length_mm=200, width_mm=150
    )
    cover = ObjectInstance.of(
        "cover", "hinged_lid", "Cover", ObjectPose(0, -150, 0), width_mm=200, depth_mm=150, feature="handle"
    )
    swing = [Step.of("open", "hook_pull", "door", "handle", "Open it.", [_crit("angle_deg", 80)])]
    group = fix_layout([door, crate, cover], swing)
    assert group is not None
    (d2, c2, v2), (move,) = group
    assert d2.pose == door.pose and (c2.pose.x_mm, c2.pose.y_mm) == (v2.pose.x_mm, v2.pose.y_mm)
    assert move.text().startswith("Moved crate (with cover) from (0, -150) to (")
    # no spot within reach: two bins too big to stand 20 mm apart
    big = [
        ObjectInstance.of(f"bin{i}", "bin", "Bin", ObjectPose(0, 0, 0), length_mm=800, width_mm=600)
        for i in (1, 2)
    ]
    assert fix_layout(big, []) is None


def test_layout_problems_alone_after_the_repair_are_fixed_in_code_and_stated() -> None:
    args = good_args()
    box = next(o for o in args["objects"] if o["name"] == "small_box")
    box["pose"] = {"x_mm": 200.0, "y_mm": 60.0, "yaw_deg": 0.0}  # inside the tote
    planner = ScriptedPlanner([args, args])
    result = plan_task("Drawer, box, tote.", client=client_for(planner))
    assert len(planner.payloads) == 2  # the model gets its repair call first
    assert result.status == "valid" and result.layout_fixed and not result.repaired and not result.first_try
    assert result.result == "valid_after_layout_fix"
    move = "Moved small_box from (200, 60) to (200, 200) mm to clear the footprint of tote"
    assert result.layout_moves == (move,)
    assert result.spec is not None and result.spec.assumptions[-1] == move
    moved = next(o for o in result.spec.objects if o.name == "small_box")
    assert (moved.pose.x_mm, moved.pose.y_mm) == (200.0, 200.0)
    assert any("layout fixer moved 1 object" in n for n in result.notes)
    record = result.to_dict()
    assert record["result"] == "valid_after_layout_fix" and record["layout_moves"] == [move]
    assert judge_args(result.spec.to_tool_args())[1] == []
    # anything besides layout problems: no fix
    args["steps"][2]["hold_after"] = True
    mixed = plan_task("Drawer, box, tote.", client=client_for(ScriptedPlanner([args, args])))
    assert mixed.status == "invalid" and not mixed.layout_moves


def test_the_gate_summary_counts_each_result_once() -> None:
    from rtt.studio.engineer.examples import GATE_CASES as cases

    args = good_args()
    inside = copy.deepcopy(args)
    next(o for o in inside["objects"] if o["name"] == "small_box")["pose"] = {
        "x_mm": 200.0,
        "y_mm": 60.0,
        "yaw_deg": 0.0,
    }
    broken = copy.deepcopy(args)
    broken["steps"][2]["hold_after"] = True
    planner = ScriptedPlanner([args, broken, args, inside, inside, broken, broken])
    picked = [c for c in cases if c.expect == "plan"][:4]
    results = run_gate(client_for(planner), picked)
    m = gate_manifest(results, mode="replay")
    assert [r["result"] for r in m["cases"]] == [
        "valid_first_call",
        "valid_after_repair",
        "valid_after_layout_fix",
        "failed",
    ]
    plans = m["summary"]["plans"]
    assert (plans["valid_first_call"], plans["valid_after_repair"], plans["valid_after_layout_fix"]) == (
        1,
        1,
        1,
    )
    assert plans["failed"] == 1 and plans["valid"] == 3 and plans["valid_rate"] == 0.75


# --------------------------------------------------------------------------- Studio v2 job templates


def _panel(x: float = 0.0, y: float = 0.0, yaw: float = 0.0, **params: Any) -> ObjectInstance:
    return ObjectInstance.of("panel", "panel_rail", "Solar panel", ObjectPose(x, y, yaw), **params)


def _clamp_steps(*success: SuccessCriterion, cap: float | None = 10.0) -> list[Step]:
    return [
        Step.of("start", "insert", "panel", "member", "Start the clamp.", [_crit("depth_mm", 3.5)]),
        Step.of(
            "tighten",
            "twist",
            "panel",
            "member",
            "Tighten the clamp.",
            list(success) or [_crit("clamp_torque_nm", 6.0)],
            constraints=StepConstraints(max_torque_nm=cap),
        ),
    ]


def test_wall_objects_outside_reach_are_layout_errors() -> None:
    """A wall button (pressed by a tool tilted to point at it), a machine's sliding door
    (its handle slides with the door) and a panel (its centre and its clamp) placed past the wrist's reach
    are layout problems with a path to the pose, which the layout fixer moves into reach; never a plan
    the simulator only finds it cannot reach."""
    from rtt.studio.engineer.layout import FRONT_REACH_MM, WORK_LIMIT_MM, fix_layout, layout_problems

    assert (WORK_LIMIT_MM, FRONT_REACH_MM) == (338.0, 400.0)
    press = Step.of(
        "press",
        "push",
        "call",
        "button",
        "Press the call button.",
        [_crit("pressed", 1)],
        params={"approach": "front"},
        constraints=StepConstraints(max_force_n=20.0),
    )

    def button(x: float, y: float, yaw: float = 0.0, mount: str = "wall") -> ObjectInstance:
        return ObjectInstance.of("call", "spring_button", "Call button", ObjectPose(x, y, yaw), mount=mount)

    near = button(0.0, -250.0)  # facing -y at the near edge: the tilted wrist would sit 400 mm further out
    problems = layout_problems([near], [press])
    assert problems == [
        (
            "objects[0].pose (call): a wall button is pressed by a tool tilted to point at its face, which puts "
            "the wrist up to 400 mm in front of it; facing (0, -1) from (0, -251) the wrist would reach (0, -651), "
            "past -338 to 338 mm; stand it so its face looks toward the table's centre with that room in front of it"
        )
    ]
    spec, found, _ = judge_args(_spec([near], [press]).to_tool_args())
    assert spec is not None and found == problems  # the plan parses: the problem is the layout's, on the pose
    assert layout_problems([button(0.0, 250.0)], [press]) == []  # far side, facing the centre
    assert layout_problems([button(0.0, 250.0, 180.0)], [press])  # far side, facing away
    assert layout_problems([button(-250.0, 0.0, -90.0)], [press])  # facing -x from the -x edge
    assert layout_problems([button(-250.0, 0.0, 90.0)], [press]) == []  # facing +x, toward the centre
    assert layout_problems([button(0.0, -250.0, mount="desk")], []) == []  # pressed from above
    # the layout fixer moves it within reach, after the repair
    args = _spec([near], [press]).to_tool_args()
    result = plan_task("Press the call button.", client=client_for(ScriptedPlanner([args, args])))
    assert result.status == "valid" and result.result == "valid_after_layout_fix"
    assert result.layout_moves == (
        "Moved call from (0, -250) to (0, 70) mm to bring it within the wrist's reach",
    )
    # a machine's sliding door: its handle slides travel_mm toward -y with the door
    door = ObjectInstance.of(
        "machine", "sliding_door", "Machine", ObjectPose(0.0, -100.0, 0.0), travel_mm=300
    )
    assert layout_problems([door], []) == [
        (
            "objects[0].pose (machine): its handle slides travel_mm (300 mm) toward local -y as the door opens, "
            "from (0, -100) to (0, -400), past -338 to 338 mm; move it or shorten the travel"
        )
    ]
    moved = fix_layout([door], [])
    assert moved is not None and moved[0][0].pose.y_mm == -30.0
    # a panel: its centre and its clamp lie (width / 2 + 25) / 2 either side of the pose
    assert layout_problems([_panel(175.0, -175.0)], []) == []
    far = layout_problems([_panel(176.0)], [])
    assert len(far) == 1 and far[0].startswith(
        "objects[0].pose (panel): the panel's centre and its clamp lie"
    )
    assert "keep x_mm and y_mm within -175.5 to 175.5" in far[0]
    assert layout_problems([_panel(0.0, 200.0, width_mm=400)], []) == []  # a narrower panel reaches further


def test_a_panel_clamp_is_a_threaded_member_judged_by_its_clamp_torque() -> None:
    lift = Step.of(
        "lay",
        "suction",
        "panel",
        "panel",
        "Lift the panel and set it on its rails.",
        [_crit("lift_mm", 25), _crit("hold_s", 1), _crit("gap_mm", hi=5)],
        params={"lift_mm": 60, "hold_s": 1.5},
    )
    assert check_plan(_spec([_panel()], [lift, *_clamp_steps()])) == ([], [])

    def problems(*steps: Step, objects: Sequence[ObjectInstance] = (_panel(),)) -> list[str]:
        return check_plan(_spec(objects, steps))[0]

    start, tighten = _clamp_steps()
    assert problems(tighten) == [
        (
            "steps[0] (tighten): a threaded member starts on its lead-in, free of the thread; insert 'panel' to "
            "the thread start before this twist"
        )
    ]
    assert problems(*_clamp_steps(_crit("torque_nm", 6.0))) == [
        (
            "steps[1] (tighten): torque_nm 6 cannot be reached; 'panel' stops at 5.7 (a seated clamp keeps turning "
            "at its clamp_torque_nm, so the wrist's torque stays just under it; judge the tightening by "
            "clamp_torque_nm at_least)"
        )
    ]
    assert problems(*_clamp_steps(_crit("torque_nm", 5.5))) == []  # within 95 % of the clamp's 6 N m
    assert problems(*_clamp_steps(_crit("clamp_torque_nm", 8.0))) == [
        (
            "steps[1] (tighten): clamp_torque_nm 8 cannot be reached; 'panel' stops at 6 (its clamp_torque_nm, the "
            "torque that seats it)"
        )
    ]
    assert problems(*_clamp_steps(cap=5.0)) == [
        (
            "steps[1] (tighten): max_torque_nm 5 lets the twist turn up to 4.5 N m (90 %), below the 6 N m it takes "
            "(its clamp_torque_nm at_least criterion)"
        )
    ]
    ccw = Step.of(
        "tighten",
        "twist",
        "panel",
        "member",
        "Tighten.",
        [_crit("clamp_torque_nm", 6.0)],
        params={"direction": "ccw"},
    )
    assert problems(start, ccw) == [
        "steps[1] (tighten): a panel clamp tightens clockwise from its start; twist it cw (ccw meets the thread start)"
    ]
    deep = Step.of("start", "insert", "panel", "member", "Start.", [_crit("depth_mm", 5)])
    assert problems(deep, tighten)[0] == (
        "steps[0] (start): depth_mm 5 cannot be reached; 'panel' stops at 4 (the clamp's thread start, 4 mm; a "
        "twist runs it further)"
    )
    held = Step.of("start", "insert", "panel", "member", "Start.", [_crit("depth_mm", 3.5)], hold_after=True)
    assert "end the insert of a threaded member with hold_after false" in problems(held, tighten)[0]
    grab = Step.of("grab", "grasp", "panel", "member", "Grab the clamp.", [_crit("lift_mm", 10)])
    assert problems(grab) == [
        (
            "steps[0] (grab): the panel's clamp lives on its post and cannot be picked up; a pick lifts the panel "
            "(handle panel), insert and twist work the clamp"
        )
    ]
    crate = ObjectInstance.of("crate", "bin", "Crate", ObjectPose(0, -270, 0), length_mm=100, width_mm=60)
    into = Step.of("lay", "suction", "panel", "panel", "Lay it.", [_crit("in_target", 1)], target="crate")
    assert any(
        "a pick on a panel on its rails lifts it and sets it back where it was; target null" in p
        for p in problems(into, objects=(_panel(), crate))
    )


def test_buttons_are_pressed_from_their_face_judged_by_pressed_with_room_for_the_press() -> None:
    wall = ObjectInstance.of("call", "spring_button", "Call", ObjectPose(0, 250, 0), actuation_force_n=15.0)
    desk = ObjectInstance.of("bell", "spring_button", "Bell", ObjectPose(0, 0, 0), mount="desk")

    def press(
        obj: str = "call", approach: str = "front", cap: float | None = 30.0, *crit: SuccessCriterion
    ) -> Step:
        return Step.of(
            "press",
            "push",
            obj,
            "button",
            "Press it.",
            list(crit) or [_crit("pressed", 1)],
            params={"approach": approach},
            constraints=StepConstraints(max_force_n=cap),
        )

    assert check_plan(_spec([wall], [press()]))[0] == []
    assert check_plan(_spec([desk], [press("bell", "top")]))[0] == []
    assert check_plan(_spec([wall], [press(approach="top")]))[0] == [
        (
            "steps[0] (press): a wall button faces sideways; push it with approach front (the tool tilted to point "
            "at its face)"
        )
    ]
    assert check_plan(_spec([desk], [press("bell")]))[0] == [
        "steps[0] (press): a desk button faces up; push it from the top (approach top)"
    ]
    assert check_plan(_spec([wall], [press("call", "front", 30.0, _crit("travel_mm", 2))]))[0] == [
        (
            "steps[0] (press): a button springs back when the tool lets go, so its travel_mm reads 0 when the step "
            "ends; judge the press by pressed at_least 1"
        )
    ]
    # the push presses with 1.5 x the 15 N actuation force, within 80 % of its limit: a limit of 28.13 N at least
    assert check_plan(_spec([wall], [press(cap=20.0)]))[0] == [
        (
            "steps[0] (press): a press on a button pushes with at least 1.5 x its actuation_force_n (22.5 N, past "
            "its bottom-out), but this push keeps under 16 N (80 % of max_force_n 20); set "
            "steps[0].constraints.max_force_n to at least 28.13 N"
        )
    ]
    assert check_plan(_spec([wall], [press(cap=28.2)]))[0] == []
    assert check_plan(_spec([wall], [press(cap=None)]))[0] == []  # 30 N without a limit
    capped = check_plan(
        _spec([wall], [press("call", "front", None, _crit("pressed", 1), _crit("force_n", hi=20))])
    )
    assert "keeps under 16 N (80 % of its force_n at_most 20)" in capped[0][0]


def test_a_tile_press_asks_for_the_force_that_seats_it_and_no_more() -> None:
    tile = ObjectInstance.of("tile", "tile_bed", "Tile", ObjectPose(0, 0, 0), mass_g=2200.0)

    def tamp(force: float | None, cap: float | None = 60.0, level: float = 1.0) -> Step:
        crit = [_crit("height_err_mm", hi=level), _crit("tilt_deg", hi=0.5)]
        if force is not None:
            crit.insert(0, _crit("force_n", force))
        return Step.of(
            "tamp", "push", "tile", "part", "Tamp it.", crit, constraints=StepConstraints(max_force_n=cap)
        )

    lay = Step.of(
        "lay",
        "suction",
        "tile",
        "part",
        "Lay it.",
        [_crit("lift_mm", 20)],
        params={"lift_mm": 40, "hold_s": 0.8},
    )
    assert check_plan(_spec([tile], [lay, tamp(40.0)]))[0] == []
    # 30 N yield + 15 N/mm x (2 - 1 + 0.3) mm - 21.58 N of tile = 27.92 N at least
    assert check_plan(_spec([tile], [tamp(None)]))[0] == [
        (
            "steps[0] (tamp): a push on a tile presses with 10 N unless it asks for a force; add force_n at_least "
            "28 (the bed yields at bed_yield_n and hardens by bed_hardening_n_mm per mm it sinks; the tile's weight "
            "helps)"
        )
    ]
    assert check_plan(_spec([tile], [tamp(20.0)]))[0] == [
        (
            "steps[0] (tamp): force_n at_least 20 leaves the tile proud: sinking it to within 1 mm of its level (it "
            "stands 2 mm proud and springs back up to 0.3 mm) takes 27.92 N on this bed (bed_yield_n + "
            "bed_hardening_n_mm x the sink, less the tile's weight)"
        )
    ]
    over = check_plan(_spec([tile], [tamp(55.0, cap=100.0)]))[0]
    assert over == [
        (
            "steps[0] (tamp): force_n at_least 55 (pressed to 56.65 N) sinks the tile more than 1 mm below its "
            "level on this bed (at most 53.42 N keeps it within)"
        )
    ]
    assert check_plan(_spec([tile], [tamp(40.0, cap=45.0)]))[0] == [
        (
            "steps[0] (tamp): the push presses to 41.2 N (force_n at_least 40 and 3 %) but keeps under 36 N (80 % "
            "of max_force_n 45); raise max_force_n to at least 51.5 N"
        )
    ]
    bed = ObjectInstance.of("bin", "bin", "Bin", ObjectPose(250, 250, 0), length_mm=80, width_mm=80)
    into = Step.of("lay", "suction", "tile", "part", "Lay it.", [_crit("in_target", 1)], target="bin")
    assert any("a pick on a tile on its bed lifts it" in p for p in check_plan(_spec([tile, bed], [into]))[0])


def _machine_scene(
    open_mm: float | None = 280.0, slip: float | None = None
) -> tuple[list[ObjectInstance], list[Step]]:
    machine = ObjectInstance.of(
        "machine", "sliding_door", "Machine", ObjectPose(0, 100, 0), travel_mm=300, pull_force_n=10
    )
    fixture = ObjectInstance.of(
        "fixture", "bin", "Fixture", ObjectPose(0, 35, 0), length_mm=120, width_mm=100, height_mm=40
    )
    done = ObjectInstance.of(
        "done", "part", "Finished part", ObjectPose(0, 150, 0), size_x_mm=50, size_y_mm=50
    )
    out = ObjectInstance.of("out", "bin", "Out bin", ObjectPose(250, 100, 0), length_mm=100, width_mm=150)
    blank = ObjectInstance.of("blank", "part", "Blank", ObjectPose(-240, 100, 0), size_x_mm=50, size_y_mm=50)
    crit = [_crit("travel_mm", open_mm)] if open_mm is not None else [_crit("travel_mm", hi=300)]
    if slip is not None:
        crit.append(_crit("slip_mm", hi=slip))
    steps = [
        Step.of(
            "open",
            "hook_pull",
            "machine",
            "handle",
            "Open.",
            crit,
            constraints=StepConstraints(max_force_n=30),
        ),
        Step.of("unload", "grasp", "done", "part", "Unload.", [_crit("in_target", 1)], target="out"),
        Step.of("load", "grasp", "blank", "part", "Load.", [_crit("in_target", 1)], target="fixture"),
        Step.of(
            "shut",
            "push",
            "machine",
            "door",
            "Shut.",
            [_crit("travel_mm", hi=5)],
            constraints=StepConstraints(max_force_n=30),
        ),
    ]
    return [machine, fixture, done, out, blank], steps


def test_a_machine_door_opens_before_picks_inside_it_and_what_stands_inside_may_overlap_it() -> None:
    from rtt.studio.engineer.layout import inside_machine, layout_problems

    objects, steps = _machine_scene()
    machine, fixture, done, out, _ = objects
    assert check_plan(_spec(objects, steps)) == ([], [])
    assert (
        inside_machine(machine, fixture)
        and inside_machine(machine, done)
        and not inside_machine(machine, out)
    )
    # the fixture's far side is 115 mm on the -y side: the door opens 125 + 10 + 115 + 20 mm to uncover it
    short = check_plan(_spec(*_machine_scene(200.0)))[0]
    assert short == [
        (
            "steps[0] (open): 'load' picks into 'machine' later, so open its door far enough to uncover 'fixture': "
            "travel_mm at_least 270 (it asks for 200; set steps[0].success[0].at_least)"
        )
    ]
    shut_first = check_plan(_spec(objects, [steps[0], steps[3], steps[1]]))[0]
    assert shut_first == [
        (
            "steps[2] (unload): 'shut' slides the door of 'machine' shut after 'open' opened it, so this pick meets "
            "a closed door; open it again after that push"
        )
    ]
    assert check_plan(_spec(objects, steps[1:]))[0][0] == (
        "steps[0] (unload): 'done' stands inside 'machine' under its closed door; hook_pull 'machine' open first "
        "with travel_mm at_least 150"
    )
    assert check_plan(_spec(*_machine_scene(slip=15.0)))[0] == [
        (
            "steps[0] (open): a pinch on a sliding door's fin gives about 33 mm before a 10 N door moves; bound "
            "slip_mm at 40 mm or more, or leave it out"
        )
    ]
    assert check_plan(_spec(*_machine_scene(slip=40.0)))[0] == []
    # half in, half out of the machine is an overlap; standing outside it beside a part keeps 20 mm
    poking = ObjectInstance.of(
        "done", "part", "Finished part", ObjectPose(0, 225, 0), size_x_mm=50, size_y_mm=50
    )
    assert layout_problems([machine, fixture, poking], []) == [
        (
            "objects[2] (done): its footprint overlaps the footprint of objects[0] (machine); keep them at least "
            "20 mm apart"
        )
    ]
    tall = ObjectInstance.of(
        "done", "part", "Tall part", ObjectPose(0, 150, 0), size_x_mm=50, size_y_mm=50, size_z_mm=240
    )
    assert layout_problems([machine, tall], [])  # too tall for the interior
    touching = ObjectInstance.of("done", "part", "Part", ObjectPose(0, 70, 0), size_x_mm=50, size_y_mm=50)
    assert (
        "overlaps the footprint of objects[1] (fixture)"
        in layout_problems([machine, fixture, touching], [])[0]
    )
    # a sliding door's forces are a drawer's
    heavy = ObjectInstance.of("machine", "sliding_door", "Machine", ObjectPose(0, 100, 0), pull_force_n=50)
    pull = Step.of(
        "open",
        "hook_pull",
        "machine",
        "handle",
        "Open.",
        [_crit("travel_mm", 100)],
        constraints=StepConstraints(max_force_n=40),
    )
    assert check_plan(_spec([heavy], [pull]))[0] == [
        (
            "steps[0] (open): steps[0].constraints.max_force_n 40 N stops the pull: a gripper drags a sliding door at "
            "up to 3 x its pull_force_n (150 N for this one; 23 to 26 N measured on a 10 N door); set "
            "steps[0].constraints.max_force_n to at least 150, or null"
        ),
        "steps[0] (open): max_force_n 40 is below the 50 N it takes (the door's pull_force_n)",
    ]


def test_the_prompt_teaches_the_job_templates_and_names_no_winning_tool() -> None:
    from rtt.studio.engineer.catalog import TOOL_ROWS, render_tools

    user = build_messages("Open the lid.")[1]["content"]
    for kind in ("panel_rail", "tile_bed", "spring_button", "sliding_door"):
        assert f"## {kind}:" in user
    for q in ("gap_mm", "clamp_torque_nm", "height_err_mm", "pressed"):
        assert f"| {q} |" in user
    assert "| approach | top, front | top |" in user  # push's taught optional param
    assert "It does not tilt" not in user and "pitches the tool up to 90 deg" in user
    assert "# The tools the search tries" in user and render_tools() in user
    assert "never name, choose or assume a tool" in user
    for row in TOOL_ROWS:
        assert row in user
    # the hero descriptions say what the job is, never which tool to use
    tool_words = re.compile(
        r"gripper|suction|vacuum|nut runner|tamper|socket|finger|\bhands?\b(?!-)|soft arm", re.IGNORECASE
    )
    for case in HERO_CASES:
        assert not tool_words.search(case.text), case.case_id


def test_the_job_template_numbers_match_their_modules() -> None:
    import inspect

    from rtt.sim.scene import PALM_XY_LIMIT
    from rtt.studio.engineer import catalog as cat
    from rtt.studio.engineer import layout as lay
    from rtt.studio.primitives import _common, push, twist
    from rtt.studio.templates import panel_rail, sliding_door, spring_button, tile_bed

    assert (
        twist.axis_site(TemplateKind.PANEL_RAIL)[2] == -1.0
        and cat.TWIST_SENSE[TemplateKind.PANEL_RAIL] == "cw"
    )
    assert cat.CLAMP_TURNS == panel_rail.CLAMP_TURNS and cat.CLAMP_PITCH_MM == pytest.approx(
        1000 * panel_rail.CLAMP_PITCH
    )
    assert (
        cat.CLAMP_CHAMFER_MM == panel_rail.CLAMP_CHAMFER_MM
        and cat.CLAMP_START_NM == panel_rail.CLAMP_START_NM
    )
    assert lay.PANEL_CLAMP_OUT_MM == pytest.approx(1000 * panel_rail.CLAMP_OUT)
    assert lay.PANEL_RAIL_PAST_MM == pytest.approx(1000 * panel_rail.RAIL_PAST)
    assert lay.PANEL_STOP_MM == pytest.approx(1000 * panel_rail.STOP_T)
    assert lay.PANEL_POST_MM == pytest.approx(1000 * panel_rail.POST_W)
    assert lay.panel_reach_mm(600) == pytest.approx(1000 * panel_rail.reach_of(0.6))
    assert lay.TILE_BORDER_MM == pytest.approx(1000 * tile_bed.BORDER)
    assert cat.TILE_OVER_MM == pytest.approx(1000 * tile_bed.OVER)
    assert lay.BUTTON_PLATE_MM[:2] == pytest.approx(
        (1000 * spring_button.PLATE_W, 1000 * spring_button.PLATE_T)
    )
    assert lay.BUTTON_CONSOLE_MM[0] == pytest.approx(1000 * spring_button.CONSOLE)
    src = inspect.getsource(spring_button.SpringButtonBuilder.add)
    assert "max(PLATE_W, d + 0.08)" in src and "max(CONSOLE, d + 0.04)" in src
    for name, value in (
        ("WALL", "MACHINE_WALL_MM"),
        ("ROOF_T", "MACHINE_WALL_MM"),
        ("LEAF_T", "MACHINE_LEAF_MM"),
    ):
        assert getattr(lay, value) == pytest.approx(1000 * getattr(sliding_door, name)), name
    assert lay.MACHINE_GAP_MM == pytest.approx(1000 * sliding_door.GAP)
    assert lay.MACHINE_OVERLAP_MM == pytest.approx(1000 * sliding_door.OVERLAP)
    assert lay.PALM_LIMIT_MM == pytest.approx(1000 * PALM_XY_LIMIT)
    assert lay.FRONT_REACH_MM >= 1000 * (push.STANDOFF + push.APPROACH)
    assert cat.PRESS_N == push.PRESS_N and cat.FORCE_GOAL_MARGIN == _common.FORCE_MARGIN
    assert f'max(press, {cat.BUTTON_PRESS_FACTOR} * info["force_n"])' in inspect.getsource(push._pushes)
    assert spring_button.ACTUATE_FRAC == 0.7  # the prompt says "past 70 % of its travel"


def test_the_front_reach_covers_the_longest_studio_tool() -> None:
    """:data:`FRONT_REACH_MM` puts every Studio tool's tilted wrist inside it: the longest tool (flange
    to tip) plus the push's standoff and approach."""
    from rtt.studio.engineer.layout import FRONT_REACH_MM
    from rtt.studio.primitives import push
    from rtt.studio.search import default_candidates, probe_tool

    longest = max(abs(probe_tool(c.spec).tcp_mm[2]) for c in default_candidates())
    assert 300.0 < longest <= FRONT_REACH_MM - 1000 * (push.STANDOFF + push.APPROACH)


# --------------------------------------------------------------------------- the recorded gate runs
#
# Two recordings (rtt.studio.engineer.gate): Studio v2's (RECORDED_GATE: the six hero jobs, on the
# current prompt; a replay rebuilds the same requests) and v1's (RECORDED_GATE_V1: frozen by the
# use-case benchmark v1; its prompt is no longer rendered, so it replays from its recorded responses).


def _manifest(root: Path = RECORDED_GATE) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _v1_hero_args(case_id: str) -> dict[str, Any]:
    row = next(r for r in _manifest(RECORDED_GATE_V1)["cases"] if r["case_id"] == case_id)
    keys = (
        "title",
        "description",
        "objects",
        "steps",
        "assumptions",
        "clarifications",
        "unsupported",
        "refusal",
    )
    return copy.deepcopy({k: row["plan"]["plan"][k] for k in keys})


def _check_one_run(root: Path) -> dict[str, Any]:
    """The checks every recorded gate run passes: one live run under its cap, counted honestly."""
    m = _manifest(root)
    assert m["mode"] == "live" and m["model"] == ENGINEER_MODEL and m["label"] == PRODUCT_MODE_LABEL
    assert m["llm"]["cost_usd"] <= m["budget_usd"] <= 0.50 and m["llm"]["live_calls"] == m["llm"]["calls"]
    assert "runs" not in m  # one run (and its listed re-runs, gate.merge_rerun), not assembled from draws
    assert all(not a["cached"] for r in m["cases"] for a in r["attempts"])
    ledger = read_ledger(root / "ledger.jsonl")
    # a call the run's spend cap refused is an attempt, but it was never sent (no ledger line)
    every = [a for r in m["cases"] for a in r["attempts"]]
    every += [a for x in m.get("reruns", []) for a in x["superseded"]["attempts"]]
    refused = sum(1 for a in every if a["outcome"] == "spend_cap")
    assert len(ledger) == m["llm"]["calls"] and m["llm"].get("refused_calls", 0) == refused
    assert abs(sum(e.cost_usd for e in ledger) - m["llm"]["cost_usd"]) < 1e-6
    # the ledger holds the first run in case order (a re-run case's first attempts in its place), then
    # each re-run in turn, its cases in case order: nothing recorded is dropped
    reruns = m.get("reruns", [])
    chains = {
        r["case_id"]: [*(x["superseded"] for x in reruns if x["case_id"] == r["case_id"]), r]
        for r in m["cases"]
    }
    numbers = {
        r["case_id"]: [x.get("rerun", 1) for x in reruns if x["case_id"] == r["case_id"]] for r in m["cases"]
    }
    first = [a for r in m["cases"] for a in chains[r["case_id"]][0]["attempts"]]
    again = [
        a
        for k in sorted({x.get("rerun", 1) for x in reruns})
        for r in m["cases"]
        if k in numbers[r["case_id"]]
        for a in chains[r["case_id"]][numbers[r["case_id"]].index(k) + 1]["attempts"]
    ]
    hashes = [a["request_hash"] for a in (*first, *again) if a["outcome"] != "spend_cap"]
    assert [e.request_hash for e in ledger] == hashes
    # a request sent again (a re-run of the same description on the same prompt) keeps each earlier
    # answer under superseded/rerun-<n>/ (the re-run that sent it again)
    from rtt.studio.engineer.gate import superseded_cache

    for h in {h for h in hashes if hashes.count(h) > 1}:
        kept = [
            x["rerun"] for x in reruns if (superseded_cache(root, x["rerun"]) / h[:2] / f"{h}.json").is_file()
        ]
        assert len(kept) == hashes.count(h) - 1, h[:12]
    assert m["llm"].get("superseded_calls", 0) == sum(
        1 for x in reruns for a in x["superseded"]["attempts"] if a["outcome"] != "spend_cap"
    )
    for x in reruns:
        assert x["reason"] and x["superseded"]["case_id"] == x["case_id"]
    # every call cut off at max_tokens is marked, on its row and in the manifest
    if "truncated" in m:
        cut = [
            (r["case_id"], a["stage"])
            for r in m["cases"]
            for a in r["attempts"]
            if a["finish_reason"] == "length"
        ]
        assert m["truncated"] == {"calls": len(cut), "cases": list(dict.fromkeys(c for c, _ in cut))}
        for r in m["cases"]:
            assert r["truncated"] == [a["stage"] for a in r["attempts"] if a["finish_reason"] == "length"]
    plans = m["summary"]["plans"]
    rows = [r for r in m["cases"] if r["expect"] == "plan"]
    for key in ("valid_first_call", "valid_after_repair", "valid_after_layout_fix"):
        assert plans[key] == sum(1 for r in rows if r["result"] == key), key
    assert plans["failed"] == sum(1 for r in rows if r["result"] == "failed")
    assert plans["valid"] + plans["failed"] == plans["n"] == len(rows)
    assert plans["valid_rate"] == round(plans["valid"] / plans["n"], 4)
    assert m["gate"]["passed"] == (
        plans["valid_rate"] >= GATE_VALID_RATE and m["summary"]["refusals"]["failed"] == 0
    )
    for row in m["cases"]:
        assert row["calls"] <= 2, row["case_id"]
        if not row["passed"]:
            assert row["result"] == "failed", row["case_id"]  # a failed case stays recorded as failed
        elif row["expect"] == "refusal":
            assert (
                row["result"] == "refused"
                and row["refusal"]
                and row["nearest_primitives"]
                and not row["steps"]
            )
        else:
            assert row["result"].startswith("valid_") and row["steps"] and row["task_id"]
            assert bool(row["layout_moves"]) == (row["result"] == "valid_after_layout_fix")
    return m


def _replays_like(m: dict[str, Any], replay: dict[str, Any]) -> None:
    assert (
        replay["llm"]["live_calls"] == 0 and replay["gate"] == m["gate"] and replay["summary"] == m["summary"]
    )
    for a, b in zip(m["cases"], replay["cases"], strict=True):
        for key in (
            "status",
            "result",
            "layout_moves",
            "steps",
            "task_id",
            "problems",
            "first_problems",
            "warnings",
            "notes",
            "refusal",
        ):
            assert a[key] == b[key], (a["case_id"], key)
        assert a["plan"]["plan"] == b["plan"]["plan"]


#: The v1 hero plans, frozen by the pre-registered use-case benchmark v1 (configs/usecase_v1.yaml) and
#: the recorded v1 Studio sessions (S7): a change to any of them is deliberate.
FROZEN_V1_HERO_TASK_IDS = {
    "h1_tote": "e4586f3612e11b0eb490349c52a109bec75f43fdb0bcf9f1d91127f9e01a4ad5",
    "h2_coupling": "33599e99ac277cf20e73d6acd349b9422330a255b3fe5ea48c6442a26c3decd6",
    "h3_cabinet": "3297a111d7c61392e9a99a08d3a80a8e14c7046b80d277925ed3a78788ff651d",
}

#: Studio v2's hero plans, as Nemotron returned them in the recorded v2 gate run with rule set v2 (the
#: Studio search runs them): a change to any of them is deliberate. h1, h6, h2 and h7 answered prompt v2 (frozen);
#: h4 and h5 were re-recorded on v2.1 (rule set v2-r2) and on v2.2 (v2-r3). h4_cnc failed its last
#: run (its repair left the door opened 200 of the 205 mm the tray needs) and has no plan to pin.
FROZEN_HERO_TASK_IDS = {
    "h1_tote": "351218d6c47d46bba1e14ecb4cf230c592618cbe28e9bf1d8be29cb88085e4f8",
    "h5_solar": "cfd24f6d1ed0eb6b47202ff99d6aacb9cbc029e0960a0e30d16370f63c5adbcd",
    "h6_tile": "f9553b22ce12fc141c274d5532f25c97083ff5cd41faa49fc1927d9487fbc5a3",
    "h2_coupling": "bbedc596166617a245983eee4ee20f2e7fe0266ace56d1ccb954bf7d316aeb01",
    "h7_hotel": "dc0ae4f82a5123cc7039f7ac2e35fe3b54399c63dd4443c5cc50cb24c33e3243",
}
#: The v2 heroes the recorded run failed (each stays recorded as failed).
FAILED_V2_HEROES = ("h4_cnc",)
#: The prompt each v2 hero answered.
V2_HERO_PROMPTS = {
    "h1_tote": "v2",
    "h4_cnc": "v2.2",
    "h5_solar": "v2.2",
    "h6_tile": "v2",
    "h2_coupling": "v2",
    "h7_hotel": "v2",
}


def test_the_v1_recorded_gate_passed_under_its_cap_on_the_v1_prompt() -> None:
    m = _check_one_run(RECORDED_GATE_V1)
    assert m["gate"]["passed"] and m["summary"]["plans"]["valid_rate"] >= GATE_VALID_RATE
    assert m["summary"]["plans"]["n"] == 8 and m["summary"]["refusals"]["n"] == 2
    assert m["summary"]["refusals"]["refused_with_nearest"] == 2
    assert [c["case_id"] for c in m["cases"]] == [c.case_id for c in V1_GATE_CASES]
    assert [c["text"] for c in m["cases"]] == [c.text for c in V1_GATE_CASES]
    # every request it recorded was sent with the tool its manifest names (the v1 one, not today's)
    cache = ResponseCache(RECORDED_GATE_V1 / "cache", mode="replay")
    for row in m["cases"]:
        for a in row["attempts"]:
            tools = cache.get(a["request_hash"]).request["payload"]["tools"]  # type: ignore[union-attr]
            assert hashlib.sha256(wire_json(tools[0]).encode()).hexdigest() == m["tool"]["sha256"]
    assert m["tool"]["sha256"] != tool_sha256() and set(m["template_hashes"]) == set(
        eprompts.template_hashes()
    )


def test_the_v1_hero_plans_are_frozen_valid_and_open_the_lid_past_vertical() -> None:
    from rtt.studio.engineer.catalog import LID_OPEN_DEG

    rows = {r["case_id"]: r for r in _manifest(RECORDED_GATE_V1)["cases"]}
    assert {k: rows[k]["task_id"] for k in FROZEN_V1_HERO_TASK_IDS} == FROZEN_V1_HERO_TASK_IDS
    for case_id, task_id in FROZEN_V1_HERO_TASK_IDS.items():
        assert rows[case_id]["passed"] and rows[case_id]["plan"]["status"] == "valid", case_id
        spec, problems, _ = judge_args(_v1_hero_args(case_id))
        assert spec is not None and problems == [] and spec.task_id == task_id
    h1 = _v1_hero_args("h1_tote")
    opens = [s for s in h1["steps"] if s["primitive"] == "hook_pull"]
    assert opens and all(
        any(c["quantity"] == "angle_deg" and (c["at_least"] or 0) >= LID_OPEN_DEG for c in s["success"])
        for s in opens
    )
    assert all(
        o["kind"] != "part" or abs(o["pose"]["x_mm"]) > 150 or abs(o["pose"]["y_mm"]) > 100
        for o in h1["objects"]
    )


def test_the_v1_recorded_gate_replays_from_its_responses_with_zero_calls() -> None:
    m = _manifest(RECORDED_GATE_V1)
    replay = gate_manifest(replay_recorded(RECORDED_GATE_V1), mode="replay", recorded=RECORDED_GATE_V1)
    _replays_like(m, replay)
    assert replay["tool"] == m["tool"] and replay["template_hashes"] == m["template_hashes"]
    # its cache replays too, request for request, through the frozen v1 prompt and rule set; the v2
    # prompt renders other requests, which the v1 cache does not hold
    client = make_engineer_client(RECORDED_GATE_V1 / "cache", cache_mode="replay")
    exact = gate_manifest(run_gate(client, V1_GATE_CASES, version="v1"), mode="replay", version="v1")
    _replays_like(m, exact)
    hashes = [a["request_hash"] for r in m["cases"] for a in r["attempts"]]
    assert [a["request_hash"] for r in exact["cases"] for a in r["attempts"]] == hashes
    assert exact["tool"] == m["tool"] and exact["template_hashes"] == m["template_hashes"]
    with pytest.raises(CacheMiss):
        run_gate(make_engineer_client(RECORDED_GATE_V1 / "cache", cache_mode="replay"), V1_GATE_CASES[:1])


def test_the_v2_recorded_gate_is_one_live_run_of_the_six_heroes_on_the_v2_prompts() -> None:
    m = _check_one_run(RECORDED_GATE)
    assert [c["case_id"] for c in m["cases"]] == [c.case_id for c in GATE_CASES]
    assert [c["text"] for c in m["cases"]] == [c.text for c in HERO_CASES]
    assert m["summary"]["plans"]["n"] == 6 and m["summary"]["refusals"]["n"] == 0
    # the first run answered prompt v2 (frozen); the re-runs of h4 and h5 answered v2.1 and v2.2 (both
    # frozen); one rule set (v2) judges all six
    assert m["prompt_version"] == "v2" and m["rules_version"] == "v2" == eprompts.rules_of("v2.2")
    assert {r["case_id"]: r["prompt_version"] for r in m["cases"]} == V2_HERO_PROMPTS
    assert m["prompts"] == {
        v: {
            "tool": {**m["tool"], "sha256": tool_sha256(version=v)},
            "template_hashes": eprompts.template_hashes(v),
        }
        for v in ("v2", "v2.1", "v2.2")
    }
    assert m["tool"]["sha256"] == tool_sha256(version="v2") and eprompts.CURRENT == "v2.3"
    # every call is counted: the recording's ledger (25 sent, 1 refused by a run's spend cap) and the two
    # draft recordings' ledger lines kept under superseded/
    drafts = read_ledger(RECORDED_GATE / m["drafts"]["ledger"])
    assert m["llm"]["calls"] == 25 and m["llm"]["refused_calls"] == 1
    assert m["drafts"]["calls"] == len(drafts) == 36
    assert abs(m["drafts"]["cost_usd"] - sum(e.cost_usd for e in drafts)) < 1e-6
    spent = sum(e.cost_usd for e in read_ledger(RECORDED_GATE / "ledger.jsonl")) + sum(
        e.cost_usd for e in drafts
    )
    assert m["llm_total_usd"] == round(spent, 6) == 0.560558
    # recorded as it came: 5 of 6 valid is under the 90 % the gate needs, so the gate failed
    assert m["summary"]["plans"]["valid"] == len(FROZEN_HERO_TASK_IDS) == 5 and not m["gate"]["passed"]
    # the one call cut off at max_tokens (h7's repair, which did not parse) is marked
    assert m["truncated"] == {"calls": 1, "cases": ["h7_hotel"]}
    h7 = next(r for r in m["cases"] if r["case_id"] == "h7_hotel")
    assert h7["truncated"] == ["repair"] and h7["attempts"][1]["outcome"] == "parse_error"


def test_each_v2_hero_plan_is_valid_on_the_recorded_run_and_frozen() -> None:
    rows = {r["case_id"]: r for r in _manifest()["cases"]}
    assert {k: rows[k]["task_id"] for k in FROZEN_HERO_TASK_IDS} == FROZEN_HERO_TASK_IDS
    for case_id, task_id in FROZEN_HERO_TASK_IDS.items():
        assert rows[case_id]["passed"] and rows[case_id]["plan"]["status"] == "valid", case_id
        spec, problems, _ = judge_args(hero_args(case_id))
        assert spec is not None and problems == [] and spec.task_id == task_id, case_id
    for case_id in FAILED_V2_HEROES:  # failed, and the problems it was left with stand under the v2 rules
        assert rows[case_id]["result"] == "failed" and rows[case_id]["problems"], case_id
        assert judge_args(hero_args(case_id))[1] == rows[case_id]["problems"]
    # h4's last repair left the door opened 200 of the 205 mm the tray needs, a step criterion the layout
    # fixer does not change, so the fixer did not run
    assert any("set steps[0].success[0].at_least" in p for p in rows["h4_cnc"]["problems"])


def test_the_v2_hero_plans_keep_the_measured_limits() -> None:
    """The limits Studio v2 measured hold on the recorded plans, by kind, whatever the model
    named things: every object in reach, a panel within +-175 mm and its clamp judged by
    clamp_torque_nm at_least <= 6 N m (never torque_nm at_least 6), a wall button pushed from the front
    with a limit of at least 1.9 x its actuation force and judged by pressed, a door's slip not bounded
    under 40 mm, a hand-tight coupling under the nut runner's threshold, and every pick, hook pull and
    push target inside the palm box (+-350 mm, 12 mm kept)."""
    from rtt.studio.engineer.layout import WORK_LIMIT_MM, reach_problems
    from rtt.studio.rules import PROCESS_TORQUE_NM, required_torque_nm

    for case_id in FROZEN_HERO_TASK_IDS:
        spec = judge_args(hero_args(case_id))[0]
        assert spec is not None, case_id
        assert reach_problems(spec.objects) == [], case_id
        by_name = {o.name: o for o in spec.objects}
        for st in spec.steps:
            obj = by_name[st.object]
            assert max(abs(obj.pose.x_mm), abs(obj.pose.y_mm)) <= WORK_LIMIT_MM, (case_id, st.id)
            kind = obj.kind.value
            if kind == "panel_rail":
                assert max(abs(obj.pose.x_mm), abs(obj.pose.y_mm)) <= 175.0, case_id
                if st.primitive is PrimitiveKind.TWIST:
                    clamp = [c.at_least for c in st.success if c.quantity == "clamp_torque_nm" and c.at_least]
                    assert clamp and max(clamp) <= min(6.0, float(obj.params["clamp_torque_nm"])), case_id
                    assert not any(
                        c.quantity == "torque_nm" and (c.at_least or 0) >= 6.0 for c in st.success
                    ), case_id
            if kind == "spring_button" and obj.params["mount"] == "wall":
                assert st.params.get("approach") == "front", case_id
                limit = st.constraints.max_force_n
                assert limit is None or limit >= 1.875 * float(obj.params["actuation_force_n"]), case_id
                assert any(c.quantity == "pressed" and c.at_least == 1.0 for c in st.success), case_id
            if kind == "sliding_door" and st.primitive is PrimitiveKind.HOOK_PULL:
                assert all(
                    c.at_most is None or c.at_most >= 40.0 for c in st.success if c.quantity == "slip_mm"
                )
            if kind == "threaded_joint":
                torque = required_torque_nm(st)
                assert torque is None or torque < PROCESS_TORQUE_NM, case_id  # hand-tight: a gripper's job


def test_a_set_back_pick_says_it_lifted_and_a_panel_clamp_is_inserted_to_its_thread_start() -> None:
    """Found by a physics probe of the first recorded solar plan: its insert asked depth_mm at_least 0, so
    it stopped 0.5 mm down and the nut runner reached 4.0 of 6 N m (asking 3.5 mm seated it at 6.0), and
    its lift asked lift_mm at_least 0, which a tool that lifts nothing meets."""
    lay = Step.of(
        "lay",
        "suction",
        "panel",
        "panel",
        "Lay it.",
        [_crit("gap_mm", hi=5), _crit("lift_mm", 0)],
        params={"lift_mm": 50, "hold_s": 1},
    )
    assert check_plan(_spec([_panel()], [lay]))[0] == [
        (
            "steps[0] (lay): a pick that sets the panel back must say it lifted it: add lift_mm at_least (more "
            "than 0, within the step's lift_mm) or hold_s at_least; without one the step passes when nothing lifts it"
        )
    ]
    shallow = Step.of("start", "insert", "panel", "member", "Start.", [_crit("depth_mm", 0, 5)])
    tighten = _clamp_steps()[1]
    assert check_plan(_spec([_panel()], [shallow, tighten]))[0] == [
        (
            "steps[0] (start): an insert stops 0.5 mm past its depth_mm at_least, so at_least 0 leaves the member "
            "0.5 mm down, short of the thread start (4 mm): the thread never catches; set "
            "steps[0].success[0].at_least to 3.5 or leave at_least out (the insert then goes to the thread start)"
        )
    ]
    deep = Step.of("start", "insert", "panel", "member", "Start.", [_crit("depth_mm", 3.5)])
    assert check_plan(_spec([_panel()], [deep, tighten]))[0] == []
    # a threaded joint's member too in rule set v2; the frozen v1 rule set keeps the v1 verdicts (the v1
    # recording's e2 nut asks depth_mm at_least 0)
    nut = ObjectInstance.of("nut", "threaded_joint", "Nut", ObjectPose(0, 0, 0))
    loose = Step.of("start", "insert", "nut", "member", "Start.", [_crit("depth_mm", 0, 5)])
    assert check_plan(_spec([nut], [loose]))[0] == [
        (
            "steps[0] (start): an insert stops 0.5 mm past its depth_mm at_least, so at_least 0 leaves the member "
            "0.5 mm down, short of the thread start (4 mm): the thread never catches; set "
            "steps[0].success[0].at_least to 3.5 or leave at_least out (the insert then goes to the thread start)"
        )
    ]
    assert check_plan(_spec([nut], [loose]), rules="v1")[0] == []


def test_nothing_stands_in_a_wall_buttons_approach() -> None:
    """Found by a physics probe of the first recorded hotel plan: its closed room door stood between the
    wall button and the side a tilted tool presses it from, and the button was never pressed (0 N on it;
    without the door, 10 N)."""
    from rtt.studio.engineer.layout import fix_layout, layout_problems, regions

    call = ObjectInstance.of("call", "spring_button", "Call", ObjectPose(0, 130, 0))
    door = ObjectInstance.of(
        "door", "hinged_lid", "Door", ObjectPose(0, 0, 0), orientation="door", depth_mm=300, width_mm=400
    )
    names = [name for name, _ in regions(call)]
    assert names == ["footprint", "approach"]
    assert layout_problems([call, door], []) == [
        (
            "objects[1] (door): its footprint overlaps the approach of objects[0] (call); keep them at least 20 mm "
            "apart"
        )
    ]
    moved = fix_layout([call, door], [])
    assert moved is not None and layout_problems(moved[0], []) == []
    desk = ObjectInstance.of("call", "spring_button", "Call", ObjectPose(0, 130, 0), mount="desk")
    assert [name for name, _ in regions(desk)] == ["footprint"]  # pressed from above


def test_merge_rerun_keeps_the_superseded_attempts_and_refuses_another_prompt(tmp_path: Path) -> None:
    import shutil

    from rtt.studio.engineer.gate import merge_rerun

    m = _manifest()
    assert [(x["case_id"], x["rerun"]) for x in m["reruns"]] == [
        ("h4_cnc", 1),
        ("h4_cnc", 2),
        ("h5_solar", 2),
        ("h4_cnc", 3),
        ("h5_solar", 3),
        ("h4_cnc", 4),
        ("h4_cnc", 5),
    ]
    texts = {c.case_id: c.text for c in HERO_CASES}
    for x in m["reruns"]:  # each superseded plan stays on record, as recorded, with why it was run again
        assert x["superseded"]["attempts"]
        if x["rerun"] == 1:  # its description made clearer
            assert (
                "description was made clearer" in x["reason"]
                and x["superseded"]["text"] != texts[x["case_id"]]
            )
        else:  # the same description, after the fixer or the rules changed, or a run cut short
            assert x["reason"].startswith(("Rule set v2-r2", "Rule set v2-r3", "Superseded:"))
            assert x["superseded"]["text"] == texts[x["case_id"]]
    assert [(x["superseded"]["prompt_version"], x["superseded"]["result"]) for x in m["reruns"]] == [
        ("v2", "failed"),
        ("v2", "failed"),
        ("v2", "valid_after_repair"),
        ("v2.1", "valid_after_layout_fix"),
        ("v2.1", "failed"),
        ("v2.2", "failed"),  # the layout fixer moved the tray past the plan's door opening (fixed since)
        ("v2.2", "failed"),  # its repair call refused by the run's own spend cap, set too tight
    ]
    assert m["budget_usd"] == 0.35 and m["llm"]["cost_usd"] <= m["budget_usd"]
    from rtt.studio.engineer.gate import superseded_cache

    for x in m["reruns"]:
        moved = ResponseCache(superseded_cache(RECORDED_GATE, x["rerun"]), mode="replay")
        cache = ResponseCache(RECORDED_GATE / "cache", mode="replay")
        for a in x["superseded"]["attempts"]:
            if a["request_hash"] is None:  # a call the spend cap refused was never sent
                continue
            h = a["request_hash"]
            kept = (superseded_cache(RECORDED_GATE, x["rerun"]) / h[:2] / f"{h}.json").is_file()
            entry = (moved if kept else cache).get(h)  # each hashing to its key
            assert (
                entry is not None and entry.response["usage"]["completion_tokens"] == a["completion_tokens"]
            )
    base, rerun = tmp_path / "base", tmp_path / "rerun"
    shutil.copytree(RECORDED_GATE, base)
    rerun.mkdir()
    (rerun / "cache").mkdir()
    (rerun / "ledger.jsonl").write_text("", encoding="utf-8")
    other = {**m, "cases": [m["cases"][0]], "tool": {**m["tool"], "sha256": "0" * 64}}
    (rerun / "manifest.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ValueError, match="another prompt"):  # another prompt under a recorded name
        merge_rerun(base, rerun, {"h1_tote": "again"})
    (rerun / "manifest.json").write_text(json.dumps({**other, "rules_version": "v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="another rule set"):
        merge_rerun(base, rerun, {"h1_tote": "again"})
    (rerun / "manifest.json").write_text(json.dumps({**m, "cases": [m["cases"][0]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="reason"):
        merge_rerun(base, rerun, {})
    entry = next((RECORDED_GATE / "cache").glob("??/*.json"))
    (rerun / "cache" / entry.parent.name).mkdir()
    shutil.copy(entry, rerun / "cache" / entry.parent.name / entry.name)
    with pytest.raises(ValueError, match="repeats a recorded request"):
        merge_rerun(base, rerun, {"h1_tote": "again"})


@pytest.mark.xfail(
    strict=True,
    reason=(
        "h4 failed the gate after its last re-record (door opened 200 of the 205 mm its tray needs); on that "
        "invalid plan the adaptive-85 gripper opens the door and does both picks, and the shut stops 10.7 mm "
        "open"
    ),
)
def test_every_v2_hero_plan_can_succeed_with_a_sensible_tool() -> None:
    rows = {r["case_id"]: r for r in _manifest()["cases"]}
    failing = [c.case_id for c in HERO_CASES if rows[c.case_id]["result"] == "failed"]
    assert failing == []


def test_v2_criteria_are_reachable_from_the_start_and_limits_are_never_goals() -> None:
    """Rule set v2-r1: a recorded h5 re-run asked penetration_mm at_least 4 on its insert and
    gap_mm at_most 5 from a 10 mm start gap (a pick sets the panel back where it was)."""
    panel = _panel(start_gap_mm=10)
    lay = Step.of(
        "lay",
        "grasp",
        "panel",
        "panel",
        "Lay it.",
        [_crit("gap_mm", hi=5), _crit("lift_mm", 10)],
        params={"lift_mm": 120, "hold_s": 1},
    )
    start = Step.of(
        "start",
        "insert",
        "panel",
        "member",
        "Start.",
        [_crit("penetration_mm", 4, 6), _crit("depth_mm", 4, 6)],
    )
    problems = check_plan(_spec([panel], [lay, start, _clamp_steps()[1]]))[0]
    assert problems == [
        (
            "steps[0] (lay): gap_mm at_most 5 cannot be met: the panel starts start_gap_mm (10 mm) from the stops "
            "and a pick sets it back where it was; set objects[0].params.start_gap_mm to the gap the use case "
            "allows, at most 5, or raise the bound (steps[0].success[0].at_most)"
        ),
        (
            "steps[1] (start): penetration_mm at_least 4 asks the step to reach a limit; penetration_mm only bounds "
            "how a step went (physics validity, slip or jam), so give it at_most or leave it out"
        ),
    ]
    assert check_plan(_spec([panel], [lay, start, _clamp_steps()[1]]), rules="v1")[0] == []
    tile = ObjectInstance.of("tile", "tile_bed", "Tile", ObjectPose(0, 0, 0), mass_g=2200.0)
    flat = Step.of(
        "lay", "suction", "tile", "part", "Lay it.", [_crit("lift_mm", 20), _crit("height_err_mm", hi=1)]
    )
    assert check_plan(_spec([tile], [flat]))[0] == [
        (
            "steps[0] (lay): height_err_mm at_most 1 cannot be met by a pick: it lays the tile back 2 mm proud "
            "(proud_mm); a push presses it to its level"
        )
    ]
    exact = _clamp_steps(_crit("clamp_torque_nm", 6.0, 6.0))
    assert (
        "the clamp_torque_nm window 6 to 6 is narrower than 0.2 N m"
        in check_plan(_spec([_panel()], exact))[0][0]
    )


def test_v2_every_handle_or_free_edge_a_step_reaches_stays_in_the_palm_box() -> None:
    """Rule set v2-r1, every kind: a recorded h7 re-run put a door's free edge at x 350 mm."""
    from rtt.studio.engineer.layout import layout_problems

    door = ObjectInstance.of(
        "door",
        "hinged_lid",
        "Door",
        ObjectPose(200, 0, 0),
        orientation="door",
        depth_mm=300,
        width_mm=400,
        feature="handle",
        feature_mm=15,
        max_angle_deg=90,
        thickness_mm=20,
    )
    problems = layout_problems([door], [])
    assert len(problems) == 1 and problems[0].startswith(
        "objects[0].pose (door): its free edge and handle swing open toward local -y, to (350"
    )
    assert problems[0].endswith(
        "past -338 to 338 mm where the wrist works; move it toward the table's centre"
    )
    assert layout_problems([door], [], rules="v1") == []
    near = ObjectInstance.of(
        "door", "hinged_lid", "Door", ObjectPose(0, 0, 0), orientation="door", depth_mm=300, max_angle_deg=90
    )
    assert layout_problems([near], []) == []
    drawer = ObjectInstance.of("drawer", "drawer", "Drawer", ObjectPose(0, 0, 0), travel_mm=250)
    assert (
        "its handle pulls out by travel_mm toward local -y, to (0, -473)" in layout_problems([drawer], [])[0]
    )
    lid = ObjectInstance.of("lid", "hinged_lid", "Lid", ObjectPose(0, 220, 0), width_mm=300, depth_mm=300)
    assert (
        "its free edge (the lid's lip or handle) swings up and over toward its hinge"
        in layout_problems([lid], [])[0]
    )
    port = ObjectInstance.of("port", "port_plug", "Port", ObjectPose(290, 0, 0), plug_diameter_mm=20)
    assert "its plug stands beside its socket along local +x" in layout_problems([port], [])[0]


def test_v2_a_layout_claim_in_words_must_match_the_poses() -> None:
    """Rule set v2-r1: a recorded h4 plan said its fixture tray stood inside the machine, beside it."""
    objects, steps = _machine_scene()
    machine, fixture, done, out, blank = objects
    outside = replace(fixture, pose=ObjectPose(-260, 0, 0))
    claim = TaskSpec(
        "Machine",
        "",
        (machine, outside, done, out, blank),
        tuple(steps),
        assumptions=("The fixture tray is a 120 x 100 x 40 mm bin fixed inside the machine interior.",),
        source="test",
    )
    problems = check_plan(claim)[0]
    assert problems[-1] == (
        "assumptions[0]: it puts 'fixture' inside 'machine', but at (-260, 0) its footprint is not inside the "
        "machine's interior (depth_mm - 40 x width_mm - 20 mm about 'machine''s pose, at most height_mm - 20 mm "
        "tall); move it there or correct the words"
    )
    assert check_plan(claim, rules="v1")[0] == []
    inside = replace(claim, objects=tuple(objects))
    assert check_plan(inside)[0] == []
    beside = replace(claim, assumptions=("The out bin stands beside the machine, outside its interior.",))
    assert not any(p.startswith("assumptions") for p in check_plan(beside)[0])
    labelled = replace(
        claim,
        assumptions=(),
        objects=(machine, replace(outside, label="Fixture inside the machine"), done, out, blank),
    )
    assert any(
        p.startswith("objects[1] (fixture).label: it puts 'fixture' inside") for p in check_plan(labelled)[0]
    )


def test_the_v1_prompt_and_rules_are_frozen_and_rebuild_the_v1_requests() -> None:
    """The v1 prompt (its templates, its rendered sections and its tool, as the v1 code rendered them)
    rebuilds every request of the v1 recording byte for byte, and the v2 prompt is another one."""
    m = _manifest(RECORDED_GATE_V1)
    assert (
        eprompts.template_hashes("v1") == m["template_hashes"]
        and tool_sha256(version="v1") == m["tool"]["sha256"]
    )
    cache = ResponseCache(RECORDED_GATE_V1 / "cache", mode="replay")
    case = V1_GATE_CASES[0]
    first = cache.get(m["cases"][0]["attempts"][0]["request_hash"])
    assert first is not None
    sent = first.request["payload"]["messages"]
    assert build_messages(sanitize_use_case(case.text)[0], version="v1") == sent
    assert build_messages(sanitize_use_case(case.text)[0]) != sent
    assert eprompts.template_hashes() != eprompts.template_hashes("v1") and tool_sha256() != tool_sha256(
        version="v1"
    )
    v2 = build_messages("x")[1]["content"]
    for words in (
        "ask depth_mm at_least the thread start less 0.5 mm",
        "says it lifted it (lift_mm or hold_s at_least above 0)",
        "stays clear: no door (open or closed), bin or part stands in it",
        "Every handle, lip or free edge a step must reach stays within reach along its whole motion",
        "give them at_most, never at_least",
        "an assumption or a label that says where something stands",
    ):
        assert words in v2, words


def test_v2_a_sliding_door_is_pushed_shut_at_40_mm_s_or_less() -> None:
    """After a pick, a push shuts a 10 N sliding door within its time budget only at
    speed_mm_s 40 or less (at 60 to 100 mm/s it ran out of time 6 to 9 mm open)."""
    from rtt.studio.engineer.catalog import SLIDE_PUSH_MAX_MM_S

    assert SLIDE_PUSH_MAX_MM_S == 40.0
    objects, steps = _machine_scene()
    fast = [*steps[:3], replace(steps[3], params={**steps[3].params, "speed_mm_s": 60.0})]
    assert check_plan(_spec(objects, fast))[0] == [
        (
            "steps[3] (shut): a push shutting a sliding door reaches its closed stop in time only at speed_mm_s 40 "
            "or less (faster, it runs out of time a few mm open); set steps[3].params.speed_mm_s to at most 40"
        )
    ]
    slow = [*steps[:3], replace(steps[3], params={**steps[3].params, "speed_mm_s": 40.0})]
    assert check_plan(_spec(objects, slow))[0] == []
    assert check_plan(_spec(objects, fast), rules="v1")[0] == []


def test_v2_picks_leave_room_for_the_tools_grip_and_give_and_a_held_object_keeps_its_handle() -> None:
    """Found by a physics smoke run of the recorded heroes: a 20 N or 30 N cap on a pick stopped the
    vacuum frame (66.7 N on the tile) and the cup (31.8 N on the parcel); a 10 mm lift left the tile 1.5 mm
    up (the frame gave 8.5 mm); and a panel held by its panel blocked the change to the nut runner."""
    tile = ObjectInstance.of("tile", "tile_bed", "Tile", ObjectPose(0, 0, 0), mass_g=2200.0)
    lay = Step.of(
        "lay",
        "grasp",
        "tile",
        "part",
        "Lay it.",
        [_crit("lift_mm", 5, 20)],
        params={"lift_mm": 10, "hold_s": 0.5},
        constraints=StepConstraints(max_force_n=20),
    )
    assert check_plan(_spec([tile], [lay]))[0] == [
        (
            "steps[0] (lay): max_force_n 20 on a pick stops most tools: they grip or seal a part with 30 to 92 N of "
            "their own, whatever it weighs; leave it null unless the use case limits the squeeze, then at least 100"
        ),
        (
            "steps[0] (lay): its criterion lift_mm at_least 5 leaves under 10 mm of steps[0].params.lift_mm (10, how "
            "far the wrist rises) for the tool's give (a vacuum frame gives about 8.5 mm before the part rises); raise "
            "steps[0].params.lift_mm to at least 15 and keep the criterion at 5, or lower the criterion"
        ),
    ]
    roomy = replace(lay, params={"lift_mm": 40, "hold_s": 0.5}, constraints=StepConstraints())
    assert check_plan(_spec([tile], [roomy]))[0] == []
    held = Step.of(
        "lay",
        "suction",
        "panel",
        "panel",
        "Lay it.",
        [_crit("lift_mm", 25)],
        params={"lift_mm": 60, "hold_s": 1},
        hold_after=True,
    )
    assert check_plan(_spec([_panel()], [held, *_clamp_steps()]))[0] == [
        (
            "steps[0] (lay): it ends holding 'panel' by its panel, but the next step works its member; a held "
            "object goes on by the same handle, so let go (hold_after false)"
        )
    ]
    port = ObjectInstance.of("port", "port_plug", "Port", ObjectPose(0, 0, 0), depth_mm=15)
    take = Step.of(
        "take",
        "grasp",
        "port",
        "plug",
        "Take it.",
        [_crit("lift_mm", 20)],
        params={"lift_mm": 40},
        hold_after=True,
    )
    push_in = Step.of("plug_in", "insert", "port", "plug", "Plug it in.", [_crit("depth_mm", 12)])
    assert check_plan(_spec([port], [take, push_in]))[0] == []  # the same handle: the plug goes on held


def test_the_fixer_keeps_a_machines_contents_apart_inside_it_on_the_failing_recorded_h4() -> None:
    """Rule sets v2-r2 and v2-r3, from the recorded output: h4's first re-run put the fixture tray, the
    finished part and the blank inside the machine, the tray 5 mm from each part, and the fixer, which
    moved them only with the machine, left them so (failed). Today's rules fail that plan on its 20 N
    hook-pull cap (a step limit, which the fixer does not change), but its layout alone is fixed: the
    fixer places the tray in a corner of the interior and keeps the gripper's 40 mm of finger room
    around both parts, inside the walls and the opening's edge, within the plan's door opening."""
    from rtt.studio.engineer.checks import claim_problems
    from rtt.studio.engineer.layout import fix_layout, inside_machine, layout_problems

    row = next(x["superseded"] for x in _manifest()["reruns"] if x["case_id"] == "h4_cnc" and x["rerun"] == 2)
    assert row["result"] == "failed" and row["task_id"].startswith("d0dfea0c")
    assert row["problems"] == [
        (
            "objects[4] (fixture): its footprint comes within 20 mm of the footprint of objects[1] (finished_part); "
            "keep them at least 20 mm apart"
        ),
        (
            "objects[4] (fixture): its footprint comes within 20 mm of the footprint of objects[3] (blank_part); "
            "keep them at least 20 mm apart; a part picked into a bin starts outside it"
        ),
    ]
    client = responses_client(RECORDED_GATE / "cache", [a["request_hash"] for a in row["attempts"]])
    result = plan_task(row["text"], client=client, version="v2")
    assert result.result == "failed" and result.cost_usd == 0.0
    assert any("steps[0].constraints.max_force_n 20 N stops the pull" in p for p in result.problems)
    spec = result.spec
    assert spec is not None
    fixed = fix_layout(spec.objects, spec.steps)
    assert fixed is not None
    objects, moves = fixed
    assert [m.text() for m in moves] == [
        (
            "Moved finished_part from (-200, -80) to (-145, -80) mm, inside guard_door under its loading opening, "
            "to clear the footprint of fixture"
        ),
        (
            "Moved blank_part from (-200, 80) to (-200, 60) mm, inside guard_door under its loading opening, to "
            "clear the footprint of finished_part"
        ),
        (
            "Moved fixture from (-200, 0) to (-270, -65) mm, inside guard_door under its loading opening, to clear "
            "the footprint of finished_part"
        ),
    ]
    machine = next(o for o in objects if o.kind is TemplateKind.SLIDING_DOOR)
    assert {o.name for o in objects if inside_machine(machine, o)} == {
        "fixture",
        "finished_part",
        "blank_part",
    }
    assert layout_problems(objects, spec.steps) == []
    assert claim_problems(replace(spec, objects=tuple(objects))) == []


def test_the_fixer_places_what_the_plan_says_stands_inside_a_machine() -> None:
    """Rule set v2-r2: the h4 re-run on prompt v2.1 said its fixture tray stands inside the machine but
    put it beside the door; the fixer places it in the interior, as the words say, 20 mm or more from what
    stands there. Rule set v1 has no claims and leaves it."""
    from rtt.studio.engineer.checks import claim_problems, claimed_inside
    from rtt.studio.engineer.engineer import _fix_layout
    from rtt.studio.engineer.layout import fix_layout, inside_machine

    objects, steps = _machine_scene()
    beside = [replace(o, pose=ObjectPose(0, -150, 0)) if o.name == "fixture" else o for o in objects]
    spec = _spec(
        beside, steps, assumptions=("The fixture tray stands inside the machine, under its opening.",)
    )
    problems = check_plan(spec)[0]
    assert claim_problems(spec) and set(claim_problems(spec)) <= set(problems)
    assert claimed_inside(spec) == {"fixture": "machine"}
    fixed = _fix_layout(spec, problems, ())
    assert not isinstance(fixed, str)
    moved, _, lines = fixed
    assert lines == (
        "Moved fixture from (0, -150) to (0, 35) mm, inside machine under its loading opening, as the plan says",
    )
    fixture = next(o for o in moved.objects if o.name == "fixture")
    assert inside_machine(moved.objects[0], fixture) and check_plan(moved)[0] == []
    # rule set v1: no claims, so nothing is placed by words (the layout alone is fixed, or not)
    words, plain = (
        fix_layout(beside, steps, rules="v1", inside={"fixture": "machine"}),
        fix_layout(beside, steps, rules="v1"),
    )
    assert words is not None and plain is not None
    assert [m.text() for m in words[1]] == [m.text() for m in plain[1]] and not any(
        m.inside for m in words[1]
    )


def test_v2_a_pick_leaves_the_tools_give_under_the_parts_weight() -> None:
    """Rule set v2-r2: h5's pick lifted the wrist 20 mm and the 8 kg panel never rose. Measured: both
    vacuum frames give 3 mm plus 2.5 mm per kg before the part rises, whatever the lift (2.2 to 10 kg,
    lift_mm 40 and 60; the 2 x 2 frame holds nothing over 5 kg). The rule keeps 1.5 mm spare, and
    never less than the 10 mm the light-part rule leaves (whose words h6's recorded repair answered); a
    hold counts only above the pick's lift threshold. The quantity is not the problem: lift_mm measures
    how far the part rose from where it started."""
    import inspect

    from rtt.studio.engineer.catalog import HOLD_LIFT_MIN_MM, HOLD_LIFT_SHARE, PICK_GIVE_MM, lift_give_mm
    from rtt.studio.primitives import _pick

    measured = {2.2: 8.54, 4.0: 13.05, 6.0: 18.04, 8.0: 22.98, 10.0: 27.89}  # kg: mm, measured
    for kg, give in measured.items():
        assert give <= lift_give_mm(1000 * kg) <= give + 2.0, kg
    assert lift_give_mm(300) == lift_give_mm(2200) == PICK_GIVE_MM and lift_give_mm(8000) == 24.5
    assert 'max(10.0, 0.5 * float(step.params["lift_mm"]))' in inspect.getsource(_pick)
    assert (HOLD_LIFT_MIN_MM, HOLD_LIFT_SHARE) == (10.0, 0.5)
    assert 'lift = 1000.0 * (z - v["z0"])' in inspect.getsource(_pick)  # the part's rise from its start
    panel = _panel(mass_g=8000)
    lay = Step.of(
        "lay",
        "suction",
        "panel",
        "panel",
        "Lay it.",
        [_crit("gap_mm", hi=5), _crit("lift_mm", 10)],
        params={"lift_mm": 20, "hold_s": 1},
    )
    assert check_plan(_spec([panel], [lay, *_clamp_steps()]))[0] == [
        (
            "steps[0] (lay): its criterion lift_mm at_least 10 leaves under 24.5 mm of steps[0].params.lift_mm (20, "
            "how far the wrist rises) for the tool's give under panel's 8 kg (a vacuum frame gives 3 mm plus 2.5 mm "
            "per kg before the part rises, 23 mm under 8 kg, and 1.5 mm is kept spare); raise steps[0].params.lift_mm "
            "to at least 34.5 and keep the criterion at 10 (raising both leaves the same shortfall)"
        )
    ]
    assert (
        check_plan(_spec([panel], [replace(lay, params={"lift_mm": 34.5, "hold_s": 1}), *_clamp_steps()]))[0]
        == []
    )
    held = replace(
        lay, success=(_crit("gap_mm", hi=5), _crit("hold_s", 1)), params={"lift_mm": 40, "hold_s": 2}
    )
    assert check_plan(_spec([panel], [held, *_clamp_steps()]))[0] == [
        (
            "steps[0] (lay): hold_s at_least 1 counts only while the part is up at least 20 mm (10 mm or 50 % of "
            "steps[0].params.lift_mm, whichever is more), and the tool's give takes up to 24.5 mm of "
            "steps[0].params.lift_mm (40) before panel rises; raise steps[0].params.lift_mm to at least 49"
        )
    ]
    assert (
        check_plan(_spec([panel], [replace(held, params={"lift_mm": 49, "hold_s": 2}), *_clamp_steps()]))[0]
        == []
    )
    assert check_plan(_spec([panel], [lay, *_clamp_steps()]), rules="v1")[0] == []  # v1 is frozen
    # the current prompt states it; the frozen v2 prompt (four recorded heroes answered it) does not
    v21, v2 = build_messages("x")[1]["content"], build_messages("x", version="v2")[1]["content"]
    assert "A pick's lift_mm (its params.lift_mm, not a criterion) is how far the wrist rises" in v21
    for words in (
        "the tool gives under the part's weight before the part itself rises, about 3 mm plus 2.5 mm per kg",
        "a 4 kg part with lift_mm at_least 10 needs lift_mm 24.5 or more",
        "A hold_s at_least counts only while the part is up at least 10 mm or 50 % of the step's lift_mm",
    ):
        assert words in v21 and words not in v2, words


def test_v2_a_hook_pull_leaves_room_over_a_sliding_doors_pull() -> None:
    """Rule set v2-r3: h4's hook pull capped its force at 20 N, and the adaptive-85 gripper pulled the
    10 N door at 23 N (constraint exceeded; 26 N with no cap). A limit under 3 x pull_force_n is refused,
    with the field to change."""
    from rtt.studio.engineer.catalog import SLIDE_PULL_FACTOR

    assert SLIDE_PULL_FACTOR == 3.0
    objects, steps = _machine_scene()
    capped = [
        replace(
            steps[0],
            constraints=StepConstraints(max_force_n=20),
            success=(*steps[0].success, _crit("force_n", hi=25)),
        ),
        *steps[1:],
    ]
    drag = (
        "a gripper drags a sliding door at up to 3 x its pull_force_n (30 N for this one; 23 to 26 N measured on a "
        "10 N door)"
    )
    assert check_plan(_spec(objects, capped))[0] == [
        (
            f"steps[0] (open): steps[0].constraints.max_force_n 20 N stops the pull: {drag}; set "
            "steps[0].constraints.max_force_n to at least 30, or null"
        ),
        (
            f"steps[0] (open): steps[0].success[1] force_n at_most 25 stops the pull: {drag}; set "
            "steps[0].success[1].at_most to at least 30, or leave the bound out"
        ),
    ]
    assert check_plan(_spec(objects, steps))[0] == []  # max_force_n 30 on the 10 N door
    assert check_plan(_spec(objects, capped), rules="v1")[0] == []
    v22, v21 = build_messages("x")[1]["content"], build_messages("x", version="v2.1")[1]["content"]
    words = "its max_force_n (and any force_n at_most) is at least 3 x pull_force_n"
    assert words in v22 and words not in v21


def test_v2_a_part_picked_inside_a_machine_keeps_a_grippers_finger_room() -> None:
    """Rule set v2-r3: inside the machine the adaptive-85 gripper's fingers met a tray and the open
    door's edge 20 mm from a 50 mm part, and cleared both at 30 mm (measured). A part a step picks there
    keeps 40 mm from every other bin or part inside, from the walls and the opening's +y edge, and from
    the open door's edge; the layout fixer keeps that room and the plan's door opening."""
    from rtt.studio.engineer.engineer import _fix_layout
    from rtt.studio.engineer.layout import FINGER_ROOM_MM

    assert FINGER_ROOM_MM == 40.0
    objects, steps = _machine_scene()
    near = [replace(o, pose=ObjectPose(0, 115, 0)) if o.name == "done" else o for o in objects]
    spec = _spec(near, steps)
    problems = check_plan(spec)[0]
    assert problems == [
        (
            "objects[2] (done): its footprint comes within 20 mm of the footprint of objects[1] (fixture); keep them "
            "at least 20 mm apart"
        ),
        (
            "objects[1] (fixture): it stands 5 mm from objects[2] (done), which steps[1] (unload) picks inside "
            "machine; a gripper's open fingers need 40 mm around it there: move objects[1].pose or objects[2].pose "
            "to keep them 40 mm apart"
        ),
    ]
    fixed = _fix_layout(spec, problems, ())
    assert not isinstance(fixed, str) and check_plan(fixed[0])[0] == []
    assert fixed[2] == (
        (
            "Moved done from (0, 115) to (0, 150) mm, inside machine under its loading opening, to clear the "
            "footprint of fixture"
        ),
    )
    wall = [replace(o, pose=ObjectPose(90, 150, 0)) if o.name == "done" else o for o in objects]
    spec = _spec(wall, steps)
    problems = check_plan(spec)[0]
    assert problems == [
        (
            "objects[2].pose (done): steps[1] (unload) picks it inside machine, where a gripper's open fingers need "
            "40 mm around it; keep its footprint 40 mm from the walls and the opening's +y edge: within 100 mm of "
            "machine's pose along its local x and at most 85 mm toward its local +y"
        )
    ]
    fixed = _fix_layout(spec, problems, ())
    assert not isinstance(fixed, str) and check_plan(fixed[0])[0] == []
    assert fixed[2] == (
        (
            "Moved done from (90, 150) to (75, 150) mm, inside machine under its loading opening, to leave room for "
            "a gripper's fingers"
        ),
    )
    # the open door's edge: 40 mm past the part it uncovers for a pick (20 mm for a bin picked into)
    short = [replace(steps[0], success=(_crit("travel_mm", 140),)), steps[1]]
    assert check_plan(_spec(objects, short))[0] == [
        (
            "steps[0] (open): 'unload' picks out of 'machine' later, so open its door far enough to uncover 'done' "
            "and 40 mm past it for a gripper's open fingers: travel_mm at_least 150 (it asks for 140; set "
            "steps[0].success[0].at_least)"
        )
    ]
    assert check_plan(_spec(near, steps), rules="v1")[0] == check_plan(_spec(near, steps), rules="v1")[0]
    assert not any("open fingers" in p for p in check_plan(_spec(near, steps), rules="v1")[0])
    v22, v21 = build_messages("x")[1]["content"], build_messages("x", version="v2.1")[1]["content"]
    words = "A part a step picks inside a machine leaves a gripper's open fingers 40 mm around it"
    assert words in v22 and words not in v21


def test_prompt_v2_is_frozen_and_each_v2_hero_rebuilds_its_first_request() -> None:
    assert eprompts.VERSIONS == ("v1", "v2", "v2.1", "v2.2", "v2.3")
    assert eprompts.FROZEN == ("v1", "v2", "v2.1", "v2.2") and eprompts.CURRENT == "v2.3"
    assert eprompts.RULES_OF == {"v1": "v1", "v2": "v2", "v2.1": "v2", "v2.2": "v2", "v2.3": "v2"}
    assert eprompts.FIELD_PATHS == {"v2.3"} and not eprompts.field_paths("v2.2")
    # v2.3 asks the same first question as v2.2; its repair says that problems name the field
    assert build_messages("x", version="v2.3") == build_messages("x", version="v2.2")
    assert "the field to change" in eprompts.render("repair.md", "v2.3", previous="p", problems="q")
    assert "the field to change" not in eprompts.render("repair.md", "v2.2", previous="p", problems="q")
    with pytest.raises(KeyError):
        eprompts.frozen("v2.3")
    with pytest.raises(KeyError):
        eprompts.rules_of("v3")
    m = _manifest()
    cache = ResponseCache(RECORDED_GATE / "cache", mode="replay")
    for row in m["cases"]:
        entry = cache.get(row["attempts"][0]["request_hash"])
        assert entry is not None
        sent = entry.request["payload"]["messages"]
        assert build_messages(sanitize_use_case(row["text"])[0], version=row["prompt_version"]) == sent, row[
            "case_id"
        ]
        for other in ("v2", "v2.1", "v2.2"):
            if other != row["prompt_version"]:
                assert build_messages(sanitize_use_case(row["text"])[0], version=other) != sent, row[
                    "case_id"
                ]
    # the superseded v2.1 rows rebuild through the frozen v2.1 prompt
    for x in m["reruns"]:
        if x["superseded"]["prompt_version"] == "v2.1":
            entry = cache.get(x["superseded"]["attempts"][0]["request_hash"])
            assert entry is not None
            sent = entry.request["payload"]["messages"]
            assert build_messages(sanitize_use_case(x["superseded"]["text"])[0], version="v2.1") == sent
    v2, now = eprompts.template_hashes("v2"), eprompts.template_hashes()
    assert tool_sha256(version="v2") == tool_sha256() and v2["system.md"] == now["system.md"]
    assert v2["user.md"] == now["user.md"] and v2["repair.md"] != now["repair.md"]


def test_every_current_recorded_row_rejudges_to_its_recorded_verdicts() -> None:
    """Rule set v2 was revised in place (``checks.RULES_REVISION``): every current row of both recordings
    is judged again, attempt by attempt, from its recorded responses with the rule set and wording of the
    prompt it answered, and gives its recorded outcomes, problems, verdict and plan. A rules edit that
    changes any of them fails here; the v2 manifest names the revision it was verified with."""
    from rtt.studio.engineer.checks import RULES_REVISION

    assert RULES_REVISION == {"v1": "v1", "v2": "v2-r5"}
    for root in (RECORDED_GATE_V1, RECORDED_GATE):
        m = _manifest(root)
        assert m.get("rules_revision", "v1") == RULES_REVISION[m.get("rules_version", "v1")], root.name
        for (case, result), row in zip(replay_recorded(root), m["cases"], strict=True):
            where = (root.name, case.case_id)
            assert [a.outcome for a in result.attempts] == [a["outcome"] for a in row["attempts"]], where
            assert [list(a.problems) for a in result.attempts] == [a["problems"] for a in row["attempts"]], (
                where
            )
            assert result.status == row["status"] and result.result == row["result"], where
            assert (
                list(result.problems) == row["problems"] and list(result.layout_moves) == row["layout_moves"]
            ), where
            task_id = result.spec.task_id if result.spec is not None and result.spec.steps else None
            assert task_id == row["task_id"], where


def test_v2_claims_bind_only_what_a_clause_names_with_the_machine() -> None:
    """A claim pinned "inside" on every object its line named: "The finished part is inside the out bin."
    put the out bin inside the machine, and the fixer tried to move it there. A claim now binds only what
    a clause names together with the machine (its name, "machine", "enclosure", "inside the door")."""
    from rtt.studio.engineer.checks import claim_problems, claimed_inside

    objects, steps = _machine_scene()
    ok = _spec(objects, steps, assumptions=("The finished part is inside the out bin.",))
    assert claimed_inside(ok) == {} and check_plan(ok)[0] == []
    real = _spec(objects, steps, assumptions=("The out bin stands inside the machine.",))
    assert claimed_inside(real) == {"out": "machine"}
    assert claim_problems(real) == [
        (
            "assumptions[0]: it puts 'out' inside 'machine', but at (250, 100) its footprint is not inside the "
            "machine's interior (depth_mm - 40 x width_mm - 20 mm about 'machine''s pose, at most height_mm - 20 "
            "mm tall); move it there or correct the words"
        )
    ]
    # clause by clause: the blank waits on the bench, the out bin is inside the machine
    two = _spec(
        objects, steps, assumptions=("The blank waits on the bench; the out bin is inside the machine.",)
    )
    assert claimed_inside(two) == {"out": "machine"}
    door = _spec(objects, steps, assumptions=("The finished part sits inside the door of the enclosure.",))
    assert claimed_inside(door) == {"done": "machine"}


def test_v2_repairs_name_the_field_to_change() -> None:
    """Every rule set v2 repair that asks for a value names the field it lives in (prompt v2.3 and on):
    the h5 repair that raised a criterion with its lift read "lift_mm" as the criterion. Prompts recorded
    before v2.3 keep the words their recorded repairs answered."""
    field = re.compile(r"\b(steps|objects)\[\d+\]\.(params\.|constraints\.|success\[\d+\]|pose\b)")
    objects, steps = _machine_scene()
    tile = ObjectInstance.of("tile", "tile_bed", "Tile", ObjectPose(0, 0, 0), mass_g=2200.0)
    lay = Step.of("lay", "grasp", "tile", "part", "Lay it.", [_crit("lift_mm", 5)], params={"lift_mm": 10})
    button = ObjectInstance.of(
        "button", "spring_button", "Button", ObjectPose(0, 100, 0), mount="desk", actuation_force_n=25.0
    )
    press = Step.of(
        "press",
        "push",
        "button",
        "button",
        "Press it.",
        [_crit("pressed", 1)],
        constraints=StepConstraints(max_force_n=30),
    )
    panel = _panel(start_gap_mm=10)
    gap = Step.of(
        "lay",
        "suction",
        "panel",
        "panel",
        "Lay it.",
        [_crit("gap_mm", hi=5), _crit("lift_mm", 10)],
        params={"lift_mm": 60, "hold_s": 1},
    )
    shallow = [Step.of("start", "insert", "panel", "member", "Start the clamp.", [_crit("depth_mm", 2)])]
    fast = [*steps[:3], replace(steps[3], params={**steps[3].params, "speed_mm_s": 60.0})]
    capped = [replace(steps[0], constraints=StepConstraints(max_force_n=20)), *steps[1:]]
    heavy = Step.of(
        "lay", "suction", "panel", "panel", "Lay it.", [_crit("lift_mm", 10)], params={"lift_mm": 20}
    )
    held = Step.of(
        "lay", "suction", "panel", "panel", "Lay it.", [_crit("hold_s", 1)], params={"lift_mm": 40}
    )
    cases = {
        "light lift": ([tile], [lay]),
        "button force": ([button], [press]),
        "start gap": ([panel], [gap, *_clamp_steps()]),
        "start gap at_least": (
            [panel],
            [replace(gap, success=(_crit("gap_mm", 12), _crit("lift_mm", 10))), *_clamp_steps()],
        ),
        "insert at_least": ([_panel()], [shallow[0], _clamp_steps()[1]]),
        "door speed": (objects, fast),
        "door pull": (objects, capped),
        "heavy lift": ([_panel()], [heavy, *_clamp_steps()]),
        "hold": ([_panel()], [held, *_clamp_steps()]),
        "door travel": (objects, [replace(steps[0], success=(_crit("travel_mm", 140),)), steps[1]]),
    }
    old = {
        "light lift": "raise lift_mm to at least 15 or lower the criterion",
        "button force": "set max_force_n to at least",
        "start gap": "set start_gap_mm to the gap the use case allows",
        "insert at_least": "ask depth_mm at_least 3.5",
        "door speed": "set speed_mm_s to at most 40",
    }
    for name, (objs, sts) in cases.items():
        problems = check_plan(_spec(objs, sts))[0]
        assert problems, name
        assert all(field.search(p) for p in problems), (name, problems)
        if name in old:  # the recorded prompts' words
            before = check_plan(_spec(objs, sts), paths=False)[0]
            assert any(old[name] in p for p in before) and not all(field.search(p) for p in before), name
    assert any(
        "set steps[3].params.speed_mm_s to at most 40" in p for p in check_plan(_spec(objects, fast))[0]
    )


def test_a_truncated_call_is_never_judged_under_v2_and_the_recorder_marks_it() -> None:
    """Rule set v2-r2: h7's repair was cut off at max_tokens and did not parse (a parse error, so the
    first plan's problems stood and the layout fixer made that plan valid). A cut-off call that does
    parse is a prefix of the plan the model meant: under rule set v2 it is never judged; v1 is frozen."""
    from rtt.studio.engineer.examples import UseCase
    from rtt.studio.engineer.gate import case_row

    text = "Pick the box off the table and drop it in the tote."
    planner = ScriptedPlanner([good_args(), good_args()], finish_reason="length")
    result = plan_task(text, client=client_for(planner))
    assert result.status == "failed" and [a.outcome for a in result.attempts] == ["truncated", "truncated"]
    assert result.truncated == ("main", "repair")
    assert all(
        a.problems[0].startswith("truncated: the call was cut off at max_tokens") for a in result.attempts
    )
    case = UseCase("t1", "test", "Truncated", text, "plan")
    assert case_row(case, result)["truncated"] == ["main", "repair"]
    assert gate_manifest([(case, result)], mode="replay")["truncated"] == {"calls": 2, "cases": ["t1"]}
    whole = plan_task(text, client=client_for(ScriptedPlanner([good_args()])))
    assert whole.status == "valid" and whole.truncated == ()
    v1 = plan_task(
        text, client=client_for(ScriptedPlanner([good_args()], finish_reason="length")), version="v1"
    )
    assert v1.attempts[0].outcome != "truncated" and v1.truncated == ("main",)


def test_the_v2_recorded_gate_replays_exactly_with_zero_calls() -> None:
    """Request for request, each case on the prompt it answered (v2 frozen, v2.1 current), and from
    its responses; the current prompt alone misses the cases recorded on v2."""
    m = _manifest()
    client = make_engineer_client(RECORDED_GATE / "cache", cache_mode="replay")
    exact = gate_manifest(
        run_gate(client, versions=recorded_versions(RECORDED_GATE)),
        mode="replay",
        recorded=RECORDED_GATE,
        version="v2",
    )
    _replays_like(m, exact)
    hashes = [a["request_hash"] for r in m["cases"] for a in r["attempts"]]
    assert [a["request_hash"] for r in exact["cases"] for a in r["attempts"]] == hashes
    assert [r["prompt_version"] for r in exact["cases"]] == list(V2_HERO_PROMPTS.values())
    # and from its responses, as the v1 run replays
    _replays_like(m, gate_manifest(replay_recorded(RECORDED_GATE), mode="replay", recorded=RECORDED_GATE))
    with pytest.raises(CacheMiss):
        run_gate(make_engineer_client(RECORDED_GATE / "cache", cache_mode="replay"), GATE_CASES[:1])
    # a recorded description is served with the prompt it was recorded on (the API, the agent, the CLI)
    for case in GATE_CASES:
        result = replay_plan(case.text, RECORDED_GATE / "cache")
        assert result.version == V2_HERO_PROMPTS[case.case_id] and result.cost_usd == 0.0
    with pytest.raises(CacheMiss):
        replay_plan("A description nobody recorded.", RECORDED_GATE / "cache")


def test_recorded_and_web_files_carry_no_reasoning_brands_or_held_out_words() -> None:
    paths = [
        *RECORDED_GATE.rglob("*.json*"),
        *RECORDED_GATE_V1.rglob("*.json*"),
        *RECORDED_UPLOAD.rglob("*.json*"),
        *WEB_INTAKE.glob("*.json"),
    ]
    assert len(paths) > 20
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "reasoning_content" not in text, path.name
        assert not find_brand_tokens(text) and not HELDOUT.search(text), path.name
        assert "NEBIUS_API_KEY" not in text and "Bearer" not in text, path.name


def test_the_hero_cases_are_six_jobs_and_the_v1_set_is_kept_as_recorded() -> None:
    assert [c.case_id for c in HERO_CASES] == [
        "h1_tote",
        "h4_cnc",
        "h5_solar",
        "h6_tile",
        "h2_coupling",
        "h7_hotel",
    ]
    assert [c.setting for c in HERO_CASES] == [
        "warehouse",
        "manufacturing",
        "construction",
        "construction",
        "construction",
        "hospitality",
    ]
    assert GATE_CASES == HERO_CASES and all(c.expect == "plan" for c in GATE_CASES)
    assert [c.setting for c in V1_HERO_CASES] == ["warehouse", "construction", "maintenance"]
    assert {c.setting for c in V1_GATE_CASES} == {
        "warehouse",
        "construction",
        "maintenance",
        "lab",
        "retail",
        "unsupported",
    }
    assert [c.expect for c in V1_GATE_CASES].count("refusal") == 2
    h2 = {c.case_id: c for c in HERO_CASES}["h2_coupling"]
    assert h2 == {c.case_id: c for c in V1_HERO_CASES}["h2_coupling"]  # the coupling is described as in v1


# --------------------------------------------------------------------------- web data


def test_the_committed_web_data_is_regenerated_exactly(tmp_path: Path) -> None:
    from rtt.studio.engineer.webdata import write_web_data

    write_web_data(None, tmp_path)
    for name in FILES:
        assert json.loads((tmp_path / name).read_text()) == json.loads((WEB_INTAKE / name).read_text()), name
    catalog = web_catalog()
    assert [t["kind"] for t in catalog["templates"]] == [t.kind.value for t in planner_templates()]
    assert [p["kind"] for p in catalog["primitives"]] == [k.value for k in PrimitiveKind]
    assert "brand_tokens" not in catalog


def test_parity_cases_cover_valid_refused_and_every_kind_of_problem() -> None:
    cases = parity_cases()["cases"]
    statuses = {c["status"] for c in cases}
    assert statuses == {"valid", "refused", "invalid"}
    assert len(cases) >= 40
    joined = " ".join(p for c in cases for p in c["problems"])
    for fragment in (
        "must be in [",
        "unknown target",
        "is not measured here",
        "cannot be reached",
        "the wrist roll",
        "still held",
        "must be unique",
        "at most 12 steps",
        "opens counter-clockwise",
        "lives on its stub",
        "before this twist",
        "label is longer",
        "a lip lid is opened by a hook",
        "under this lid later, so open it past vertical",
        "no earlier step opens that lid",
        "meets a closed lid",
    ):
        assert fragment in joined, fragment
    recorded = recorded_plans()
    # a hero whose recorded plan is not valid (h4_cnc) is listed, not offered as a hero example
    heroes = [e["case_id"] for e in recorded["examples"] if e["hero"]]
    assert heroes == [c.case_id for c in HERO_CASES if c.case_id != "h4_cnc"]
    assert next(e for e in recorded["examples"] if e["case_id"] == "h4_cnc")["status"] == "invalid"
    assert recorded["examples"][-1]["uploads"]


# --------------------------------------------------------------------------- command line


def test_cli_plans_a_recorded_case_and_stops_on_a_new_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "plan.json"
    assert cli_main(["plan", "--case", "h1_tote", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "status: valid" in printed and PRODUCT_MODE_LABEL in printed
    assert json.loads(out.read_text())["status"] == "valid"
    assert cli_main(["plan", "--text", "Juggle three balls."]) == 2
    with pytest.raises(SystemExit):
        cli_main(["plan", "--text", "x", "--live"])  # no --env-file


def test_cli_gate_replay_and_web_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = {r["case_id"]: r for r in _manifest()["cases"]}
    passing = [
        c.case_id for c in GATE_CASES if c.case_id in ("h6_tile", "h2_coupling") and rows[c.case_id]["passed"]
    ]
    assert cli_main(["gate", "--out", str(tmp_path / "gate"), "--cases", ",".join(passing)]) == 0
    m = json.loads((tmp_path / "gate" / "manifest.json").read_text())
    assert [c["case_id"] for c in m["cases"]] == passing and m["mode"] == "replay"
    assert "gate PASSED" in capsys.readouterr().out
    # the v1 run replays from its recorded responses when its folder is named
    v1 = [
        "gate",
        "--out",
        str(tmp_path / "v1"),
        "--recorded",
        str(RECORDED_GATE_V1),
        "--cases",
        "h2_coupling,u2_pour",
    ]
    assert cli_main(v1) == 0
    m = json.loads((tmp_path / "v1" / "manifest.json").read_text())
    assert [c["case_id"] for c in m["cases"]] == ["h2_coupling", "u2_pour"]
    assert m["cases"][0]["task_id"] == FROZEN_V1_HERO_TASK_IDS["h2_coupling"]
    assert m["tool"] == _manifest(RECORDED_GATE_V1)["tool"] and "gate PASSED" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli_main([*v1, "--live"])
    # and from its cache through the frozen v1 prompt and rule set
    v1_cache = [
        "gate",
        "--out",
        str(tmp_path / "v1c"),
        "--prompt-version",
        "v1",
        "--cache",
        str(RECORDED_GATE_V1 / "cache"),
    ]
    assert cli_main([*v1_cache, "--cases", "h3_cabinet,e2_rod_nut"]) == 0
    m = json.loads((tmp_path / "v1c" / "manifest.json").read_text())
    assert m["prompt_version"] == "v1" and m["tool"] == _manifest(RECORDED_GATE_V1)["tool"]
    rows_v1 = {r["case_id"]: r for r in _manifest(RECORDED_GATE_V1)["cases"]}
    assert [(c["case_id"], c["result"]) for c in m["cases"]] == [
        (k, rows_v1[k]["result"]) for k in ("h3_cabinet", "e2_rod_nut")
    ]
    assert cli_main(["export-web", "--out", str(tmp_path / "web")]) == 0
    assert sorted(p.name for p in (tmp_path / "web").iterdir()) == sorted(FILES)


# --------------------------------------------------------------------------- HTTP routes


PASSCODE = "correct horse"


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[Any, Any, list[ScriptedPlanner]]]:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rtt.agent.caps import AgentCaps, PasscodeGate, passcode_digest
    from rtt.agent.fakes import FakePartBackend
    from rtt.api.service import AgentService, ApiSettings
    from rtt.studio.engineer.api import StudioPlanSettings, studio_router

    settings = ApiSettings(root=tmp_path / "agent", caps=AgentCaps(poll_wait_s=5.0))
    svc = AgentService(
        settings,
        backend=FakePartBackend(tmp_path / "agent" / "parts", delay_s=0.0),
        gate=PasscodeGate(passcode_digest(PASSCODE)),
    )
    planners: list[ScriptedPlanner] = []

    def factory() -> ScriptedPlanner:
        planner = ScriptedPlanner([good_args(), good_args()])
        planners.append(planner)
        return planner

    app = FastAPI()
    app.include_router(studio_router(svc, StudioPlanSettings(plan_usd=0.05), transport_factory=factory))
    with TestClient(app) as client:
        yield client, svc, planners
    svc.shutdown()


def test_api_serves_recorded_plans_free_and_without_a_passcode(api: Any) -> None:
    client, _, planners = api
    res = client.post("/studio/plan", json={"text": HERO_CASES[0].text})
    assert res.status_code == 200, res.text
    body = res.json()
    assert (
        body["source"] == "recorded"
        and body["result"]["status"] == "valid"
        and body["label"] == PRODUCT_MODE_LABEL
    )
    assert planners == []


def test_api_live_plans_need_the_passcode_and_stay_under_the_cap(api: Any) -> None:
    client, svc, planners = api
    text = {"text": "Pull the drawer open, put the box in the tote and push it shut."}
    assert client.post("/studio/plan", json=text).status_code == 403
    assert client.post("/studio/plan", json={**text, "passcode": "wrong"}).status_code == 403
    res = client.post("/studio/plan", json=text, headers={"X-RTT-Passcode": PASSCODE})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["source"] == "live" and body["result"]["status"] == "valid"
    assert body["spend"]["session_cap_usd"] == 0.05 and 0 < body["spend"]["spent_usd"] <= 0.05
    assert len(planners) == 1 and len(planners[0].payloads) == 1
    assert svc.day_book.spent_today() == pytest.approx(body["spend"]["spent_usd"])
    assert client.post("/studio/plan", json={"text": "  "}).status_code == 422
    assert client.post("/studio/plan", json={"text": "x", "extra": 1}).status_code == 422


def test_api_spend_cap_and_offline_mode(tmp_path: Path, api: Any) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from rtt.agent.caps import PasscodeGate, passcode_digest
    from rtt.agent.fakes import FakePartBackend
    from rtt.api.service import AgentService, ApiSettings
    from rtt.studio.engineer.api import StudioPlanSettings, studio_router

    _, svc, _ = api
    tiny = FastAPI()
    tiny.include_router(
        studio_router(svc, StudioPlanSettings(plan_usd=0.001), transport_factory=lambda: ScriptedPlanner([]))
    )
    with TestClient(tiny) as c:
        res = c.post("/studio/plan", json={"text": "Something new", "passcode": PASSCODE})
        assert res.status_code == 429 and "spend cap" in res.json()["detail"]
    off = AgentService(
        ApiSettings(root=tmp_path / "off", offline=True),
        backend=FakePartBackend(tmp_path / "off" / "parts"),
        gate=PasscodeGate(passcode_digest(PASSCODE)),
    )
    app = FastAPI()
    app.include_router(studio_router(off, StudioPlanSettings()))
    recorded = {r["case_id"]: r["status"] for r in _manifest()["cases"]}
    with TestClient(app) as c:
        hero = HERO_CASES[-1]
        assert (
            c.post("/studio/plan", json={"text": hero.text}).json()["result"]["status"]
            == recorded[hero.case_id]
        )
        assert c.post("/studio/plan", json={"text": "Something new", "passcode": PASSCODE}).status_code == 409
    off.shutdown()


def test_api_plans_with_an_uploaded_part_through_its_intake_record(api: Any) -> None:
    import trimesh

    from rtt.part.intake import intake, save_part

    client, svc, planners = api
    info = svc.upload_part(trimesh.creation.box(extents=(40.0, 30.0, 60.0)).export(file_type="stl"))
    part_id = info.part_id
    no_record = client.post(
        "/studio/plan", json={"text": "Put my part in the tote.", "part_id": part_id, "passcode": PASSCODE}
    )
    assert no_record.status_code == 422
    stl = svc.store.parts_dir / part_id / "source.stl"
    save_part(intake(stl, units="mm", decomposition="hull"), svc.store.parts_dir / part_id / "part")
    unknown = client.post(
        "/studio/plan", json={"text": "x", "part_id": "part-000000000000", "passcode": PASSCODE}
    )
    assert unknown.status_code == 404
    res = client.post(
        "/studio/plan", json={"text": "Put my part in the tote.", "part_id": part_id, "passcode": PASSCODE}
    )
    assert res.status_code == 200, res.text
    result = res.json()["result"]
    assert result["status"] == "invalid" and f"upload:{part_id} is not used" in result["problems"][0]
    prompt = planners[-1].payloads[0]["messages"][1]["content"]
    assert f"upload:{part_id}:" in prompt


def test_api_validates_edited_plans(api: Any) -> None:
    client, _, _ = api
    args = hero_args("h3_cabinet")
    res = client.post("/studio/validate", json={"plan": args}).json()
    assert (
        res["status"] == "valid"
        and res["task_id"]
        and res["fidelity"][args["steps"][0]["id"]] == "approximation"
    )
    args["steps"][0]["params"]["direction"] = "cw"
    res = client.post("/studio/validate", json={"plan": args}).json()
    assert res["status"] == "invalid" and any("twist it ccw" in p for p in res["problems"])
    big = {"plan": {"title": "x" * (300 * 1024)}}
    assert client.post("/studio/validate", json=big).status_code == 413
    assert PRIMITIVES  # the contract is importable next to the router


# --------------------------------------------------------------------------- a door's bar, gripped from the front


def _bar_door(x: float, y: float = 290.0, *, depth: float = 300.0, yaw: float = 0.0) -> ObjectInstance:
    """The hotel job's room door: 400 mm tall, a 10 mm bar 20 mm off its face, hinged on its left."""
    return ObjectInstance.of(
        "room_door",
        "hinged_lid",
        "Room door",
        ObjectPose(x, y, yaw),
        orientation="door",
        feature="handle",
        feature_mm=20,
        depth_mm=depth,
        width_mm=400,
        thickness_mm=20,
        max_angle_deg=90,
        open_torque_nm=0.2,
        damping_nms=0.05,
    )


def _bar_pull() -> Step:
    return Step.of(
        "open_door",
        "hook_pull",
        "room_door",
        "handle",
        "Pull the door open by its handle",
        [SuccessCriterion("angle_deg", at_least=80.0)],
        params={"approach": "front", "speed_mm_s": 50},
    )


def _hotel_scene(door_x: float = -140.0) -> tuple[list[ObjectInstance], list[Step]]:
    """The hotel job as its recorded plan stood after the first layout fix (the door at x -140 mm)."""
    button = ObjectInstance.of(
        "call_button",
        "spring_button",
        "Elevator call button",
        ObjectPose(280.0, 70.0, 0.0),
        actuation_force_n=5,
        diameter_mm=40,
        height_mm=150,
        mount="wall",
        proud_mm=1,
        travel_mm=3,
    )
    parcel = ObjectInstance.of(
        "parcel", "part", "Parcel", ObjectPose(100.0, -160.0, 0.0),
        mass_g=500, shape="box", size_x_mm=160, size_y_mm=120, size_z_mm=80,
    )  # fmt: skip
    tray = ObjectInstance.of(
        "tray", "bin", "Delivery tray", ObjectPose(50.0, 0.0, 0.0),
        fixed=True, height_mm=100, length_mm=200, wall_mm=5, width_mm=150,
    )  # fmt: skip
    steps = [
        Step.of(
            "press_button", "push", "call_button", "button", "Press the call button",
            [SuccessCriterion("pressed", at_least=1.0)], params={"approach": "front", "speed_mm_s": 20},
        ),
        _bar_pull(),
        Step.of(
            "place_parcel", "grasp", "parcel", "part", "Place the parcel into the tray",
            [SuccessCriterion("in_target", at_least=1.0), SuccessCriterion("tilt_deg", at_most=10.0)],
            target="tray", params={"hold_s": 1, "lift_mm": 150},
        ),
    ]  # fmt: skip
    return [button, _bar_door(door_x), parcel, tray], steps


def test_the_fixer_moves_a_door_along_x_into_the_front_grip_of_a_tool_changer_gripper() -> None:
    """A door's bar is gripped from the front only, the tool normal to the door, which puts the palm
    the tool's length out from the bar. The hotel door at x -140 mm swings its face toward the table's
    edge: neither reference gripper on a tool changer could follow it to its 88 deg goal. The fixer
    moves it along x only (y and yaw kept) to the nearest position where the preferred 110 mm servo
    gripper's front grip stays in the wrist's travel over the engagement, the swing and the retreat
    (x 90 mm), and moves the tray out of the door's new swing under the usual clearance rules. Both
    moves are stated plainly, and the result is the same every time."""
    from rtt.studio.engineer.layout import BAR_GRIPPERS, bar_grip_reach, fix_door_bars, layout_problems

    objects, steps = _hotel_scene()
    door = objects[1]
    assert all(bar_grip_reach(door, steps, g) is not None for g in BAR_GRIPPERS)
    assert layout_problems(objects, steps) == []  # the clearance and reach rules held before
    fixed = fix_door_bars(objects, steps)
    assert not isinstance(fixed, str), fixed
    moved, moves = fixed
    new = next(o for o in moved if o.name == "room_door")
    assert (new.pose.x_mm, new.pose.y_mm, new.pose.yaw_deg) == (90.0, 290.0, 0.0)
    assert bar_grip_reach(new, steps, BAR_GRIPPERS[0]) is None
    assert bar_grip_reach(replace(new, pose=ObjectPose(80.0, 290.0, 0.0)), steps, BAR_GRIPPERS[0]) is not None
    door_move = (
        "Moved room_door from (-140, 290) to (90, 290) mm along x so the 110 mm servo gripper on a tool changer "
        "grips its bar handle from the front through the whole pull"
    )
    assert [m.text() for m in moves] == [
        door_move,
        "Moved tray from (50, 0) to (-100, -110) mm to clear the swing of room_door",
    ]
    assert layout_problems(moved, steps) == []
    again = fix_door_bars(objects, steps)
    assert not isinstance(again, str)  # deterministic
    assert again[1] == moves and [o.pose for o in again[0]] == [o.pose for o in moved]
    # a door the second reference gripper already reaches stays where it is
    near, _ = _hotel_scene(door_x=50.0)
    assert fix_door_bars(near, steps) == (near, [])


def test_a_door_only_the_adaptive_gripper_can_front_grip_is_placed_for_it_or_left_with_a_reason() -> None:
    """The fallback order: a 400 mm door at y 200 mm has no position along x where the servo gripper's
    front grip stays in the travel, so the fixer places it for the two-finger adaptive gripper; a 500 mm
    door has none for either, and a door turned to face the table's +y or -x edge has none along x:
    those stay where they are and the fixer says why (the step will fail plainly in the simulator)."""
    from rtt.studio.engineer.layout import fix_door_bars

    pull = [_bar_pull()]
    moved, moves = fix_door_bars([_bar_door(-140.0, 200.0, depth=400.0)], pull)  # type: ignore[misc]
    assert moved[0].pose == ObjectPose(90.0, 200.0, 0.0)
    door_move = (
        "Moved room_door from (-140, 200) to (90, 200) mm along x so the two-finger adaptive gripper on a tool "
        "changer grips its bar handle from the front through the whole pull"
    )
    assert [m.text() for m in moves] == [door_move]
    for door in (
        _bar_door(-140.0, 200.0, depth=500.0),
        _bar_door(-140.0, 200.0, yaw=180.0),
        _bar_door(-140.0, 200.0, yaw=-90.0),
    ):
        why = fix_door_bars([door], pull)
        assert isinstance(why, str) and why.startswith("No position of room_door along x"), why


def test_the_door_bar_reference_grippers_are_measured_on_the_tool_changer() -> None:
    """The fixer's reference grippers (``layout.BAR_GRIPPERS``) are the Studio's own tools on its tool
    changer: their TCP below the palm and their tips past it match the calibrated tools, and the fixer's
    reach agrees with the hook pull's controller on the hotel door either side of the servo gripper's
    first position (x 80 mm: the controller does not attempt it; x 90 mm: it grips the bar from the front)."""
    import mujoco
    import numpy as np

    from rtt.sim.task import default_task_v2
    from rtt.studio.engineer.layout import BAR_GRIPPERS, bar_grip_reach
    from rtt.studio.plan import StepContext
    from rtt.studio.primitives._common import tool_of
    from rtt.studio.primitives.hook_pull import HookPullController, bar_drop
    from rtt.studio.scene import build_studio_scene
    from rtt.studio.search import default_candidates
    from rtt.studio.types import Setup, TaskSpec

    task = default_task_v2()
    pool = {(c.spec.model, c.spec.primitive): c.spec for c in default_candidates()}
    cup = pool[("cup-25mm", "seal")]
    for model, _, tcp_mm, drop_mm in BAR_GRIPPERS:
        gripper = pool[(model, "pinch")]
        for x, front in ((80.0, False), (90.0, True)):
            objects, steps = _hotel_scene(door_x=x)
            spec = TaskSpec("hotel", "", tuple(objects), tuple(steps), source="test")
            setup = Setup.tool_change({"press_button": gripper, "open_door": gripper, "place_parcel": cup})
            tool = next(sg for sg in setup.segments(spec) if "open_door" in sg.step_ids).tool
            scene = build_studio_scene(setup, spec, tool, task=task)
            data = mujoco.MjData(scene.model)
            mujoco.mj_kinematics(scene.model, data)
            ctx = StepContext(
                step=steps[1], task_spec=spec, scene=scene, obj=scene.objects["room_door"], target=None,
                head=scene.active, wrist_start=np.zeros(5), task=task, seed=0,
            )  # fmt: skip
            measured = tool_of(ctx)
            assert -1000.0 * float(measured.tcp[2]) == pytest.approx(tcp_mm, abs=0.1), model
            drop = bar_drop(measured.footprint, float(measured.tcp[2]), measured.tip_drop, 0.010)
            assert 1000.0 * drop == pytest.approx(drop_mm, abs=0.1), model
            if model != BAR_GRIPPERS[0][0]:
                continue
            plan = HookPullController().plan(ctx, scene.model, data)
            assert (plan.values["run"].get("mode") == "bar") is front, (x, plan.note)
            assert (bar_grip_reach(objects[1], steps, BAR_GRIPPERS[0]) is None) is front, x


def test_the_door_bar_rule_leaves_every_recorded_plan_without_a_door_bar_as_it_is() -> None:
    """The fixer's door-bar rule only moves a door whose bar a step pulls: every recorded plan of
    both recordings without one replays exactly as recorded (the re-judge test compares every row),
    and the rule returns its objects unchanged; of the current recording only the hotel job has one."""
    from rtt.studio.engineer.layout import bar_steps, fix_door_bars

    with_bar = []
    for root in (RECORDED_GATE, RECORDED_GATE_V1):
        for case, result in replay_recorded(root):
            spec = result.spec
            if spec is None:
                continue
            if any(bar_steps(o, spec.steps) for o in spec.objects):
                with_bar.append((root.name, case.case_id))
                continue
            assert fix_door_bars(spec.objects, spec.steps) == (list(spec.objects), []), case.case_id
    assert ("gate-2026-10-27-v2c", "h7_hotel") in with_bar
    assert [c for r, c in with_bar if r == "gate-2026-10-27-v2c"] == ["h7_hotel"]
