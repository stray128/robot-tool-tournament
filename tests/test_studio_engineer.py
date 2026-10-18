"""The Studio's Task Engineer (track S4, product mode): prompts, checks, repair, refusals, uploads,
the recorded gate run, the command line and the HTTP routes. Offline: scripted or recorded
responses only, no API key, no network.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from rtt.effectors.types import PRODUCT_MODE_LABEL, find_brand_tokens
from rtt.llm.cache import CacheMiss, ResponseCache
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
from rtt.studio.engineer.examples import UPLOAD_CASE, UPLOAD_EXAMPLE_STL
from rtt.studio.engineer.gate import GATE_VALID_RATE, RECORDED_GATE, gate_manifest, run_gate
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
    submit_task_plan_tool,
    task_plan_json_schema,
    templates,
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

    def __init__(self, answers: Sequence[Any], *, reasoning: str = "", content: str | None = None) -> None:
        self.answers = list(answers)
        self.payloads: list[dict[str, Any]] = []
        self.reasoning = reasoning
        self.content = content

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
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
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
    for case in (*GATE_CASES, UPLOAD_CASE):
        messages = build_messages(sanitize_use_case(case.text)[0])
        assert_clean(messages)
        text = "\n".join(m["content"] for m in messages)
        assert not HELDOUT.search(text) and not find_brand_tokens(text), case.case_id
        assert case.text.split(":")[0] in text
    user = build_messages("Open the lid.")[1]["content"]
    for t in templates():
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
    for case in GATE_CASES:
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
    drawer = ObjectInstance.of(
        "drawer", "drawer", "Drawer", ObjectPose(0, 0, 0), travel_mm=200, pull_force_n=30
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
    heavy = ObjectInstance.of("drawer", "drawer", "Drawer", ObjectPose(0, 0, 0), pull_force_n=35)
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
    if kind in ("drawer", "hinged_lid") and joint is not None:
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

    for kind, (_, _, w) in twist.AXIS_SITE.items():
        assert cat.TWIST_SENSE[kind] == ("ccw" if w > 0 else "cw"), kind
    assert set(cat.TWIST_SENSE) == set(twist.AXIS_SITE)
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
    client = make_engineer_client(RECORDED_UPLOAD / "cache", cache_mode="replay")
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


# --------------------------------------------------------------------------- the recorded gate run


def _manifest() -> dict[str, Any]:
    return json.loads((RECORDED_GATE / "manifest.json").read_text(encoding="utf-8"))


def test_the_recorded_gate_passed_under_its_cap_with_the_label() -> None:
    m = _manifest()
    assert m["mode"] == "live" and m["model"] == ENGINEER_MODEL and m["label"] == PRODUCT_MODE_LABEL
    assert m["gate"]["passed"] and m["summary"]["plans"]["valid_rate"] >= GATE_VALID_RATE
    assert m["summary"]["plans"]["n"] == 8 and m["summary"]["refusals"]["n"] == 2
    assert m["summary"]["refusals"]["refused_with_nearest"] == 2
    assert m["llm"]["cost_usd"] <= m["budget_usd"] <= 0.50 and m["llm"]["live_calls"] == m["llm"]["calls"]
    ledger = read_ledger(RECORDED_GATE / "ledger.jsonl")
    assert len(ledger) == m["llm"]["calls"]
    assert abs(sum(e.cost_usd for e in ledger) - m["llm"]["cost_usd"]) < 1e-6
    assert {c["case_id"] for c in m["cases"]} == {c.case_id for c in GATE_CASES}
    for row in m["cases"]:
        assert row["calls"] <= 2 and row["passed"], row["case_id"]
        if row["expect"] == "refusal":
            assert row["refusal"] and row["nearest_primitives"] and not row["steps"]
        else:
            assert row["steps"] and row["task_id"]
    assert m["tool"]["sha256"] == tool_sha256() and m["template_hashes"] == eprompts.template_hashes()


def test_the_recorded_gate_replays_exactly_with_zero_calls() -> None:
    m = _manifest()
    client = make_engineer_client(RECORDED_GATE / "cache", cache_mode="replay")
    replay = gate_manifest(run_gate(client), mode="replay")
    assert (
        replay["llm"]["live_calls"] == 0 and replay["gate"] == m["gate"] and replay["summary"] == m["summary"]
    )
    for a, b in zip(m["cases"], replay["cases"], strict=True):
        for key in (
            "status",
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


def test_recorded_and_web_files_carry_no_reasoning_brands_or_held_out_words() -> None:
    paths = [*RECORDED_GATE.rglob("*.json*"), *RECORDED_UPLOAD.rglob("*.json*"), *WEB_INTAKE.glob("*.json")]
    assert len(paths) > 10
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "reasoning_content" not in text, path.name
        assert not find_brand_tokens(text) and not HELDOUT.search(text), path.name
        assert "NEBIUS_API_KEY" not in text and "Bearer" not in text, path.name


def test_the_hero_cases_are_the_three_settings() -> None:
    assert [c.setting for c in HERO_CASES] == ["warehouse", "construction", "maintenance"]
    assert {c.setting for c in GATE_CASES} == {
        "warehouse",
        "construction",
        "maintenance",
        "lab",
        "retail",
        "unsupported",
    }
    assert [c.expect for c in GATE_CASES].count("refusal") == 2


# --------------------------------------------------------------------------- web data


def test_the_committed_web_data_is_regenerated_exactly(tmp_path: Path) -> None:
    from rtt.studio.engineer.webdata import write_web_data

    write_web_data(None, tmp_path)
    for name in FILES:
        assert json.loads((tmp_path / name).read_text()) == json.loads((WEB_INTAKE / name).read_text()), name
    catalog = web_catalog()
    assert [t["kind"] for t in catalog["templates"]] == [k.value for k in TemplateKind]
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
    ):
        assert fragment in joined, fragment
    recorded = recorded_plans()
    assert [e["case_id"] for e in recorded["examples"] if e["hero"]] == [c.case_id for c in HERO_CASES]
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
    assert cli_main(["gate", "--out", str(tmp_path / "gate"), "--cases", "h2_coupling,u2_pour"]) == 0
    m = json.loads((tmp_path / "gate" / "manifest.json").read_text())
    assert [c["case_id"] for c in m["cases"]] == ["h2_coupling", "u2_pour"] and m["mode"] == "replay"
    assert "gate PASSED" in capsys.readouterr().out
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
    with TestClient(app) as c:
        assert (
            c.post("/studio/plan", json={"text": GATE_CASES[8].text}).json()["result"]["status"] == "refused"
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
