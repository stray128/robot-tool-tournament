"""Morphology designer (product mode, track B3): spaces, canonical recipes, validators, random
control and the Nemotron proposer, all offline (cached or scripted responses only).

The physics of designed effectors is in ``tests/test_effectors_designed.py``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from rtt.effectors.designed import reference_recipes
from rtt.effectors.morphology import (
    CHECK_CODES,
    DESIGNED_KINDS,
    MIN_POINTS,
    DesignOutcome,
    canonical_recipe,
    design_spaces,
    distance_points,
    make_designer_client,
    propose_recipes,
    random_recipes,
    random_recipes_with_stats,
    screen,
    space_label,
    spec_for,
    submit_morphologies_tool,
    validate_recipe,
    validation_codes,
)
from rtt.effectors.morphology import prompts as mprompts
from rtt.effectors.morphology.diversity import inactive_params, quantise
from rtt.effectors.morphology.layout import layout_of
from rtt.effectors.morphology.proposer import (
    ScrubbingCache,
    assert_clean,
    build_messages,
    recipe_problems,
    scrub,
)
from rtt.effectors.morphology.spaces import parse_kind, select_spaces, space_for
from rtt.effectors.morphology.validators import fit_numbers, validate_structure
from rtt.effectors.registry import EffectorFamilyImpl, override_family
from rtt.effectors.types import (
    EffectorFamily,
    EffectorSpec,
    MorphologyRecipe,
    ParamSpec,
    find_brand_tokens,
    params_json_schema,
)
from rtt.llm.cache import CacheMiss, ResponseCache
from rtt.llm.client import TokenBucket
from rtt.llm.providers import TransportResponse
from rtt.parts.library import get_part, training_parts
from rtt.search.nemotron import PromptLeakError
from rtt.search.nemotron_v2 import FORBIDDEN_PROMPT_TEXT_V2
from rtt.sim.task import default_task_v2

REPO = Path(__file__).resolve().parents[1]
RECORDED = REPO / "src/rtt/effectors/morphology/recorded/smoke-2026-10-17"
HELDOUT_WORDS = FORBIDDEN_PROMPT_TEXT_V2


@pytest.fixture(scope="module")
def task() -> Any:
    return default_task_v2()


def _recipe(kind: str, **changes: Any) -> MorphologyRecipe:
    base = next(r for r in reference_recipes() if r.kind == kind)
    return MorphologyRecipe(EffectorFamily.DESIGNED, kind, {**base.params, **changes})


# --------------------------------------------------------------------------- spaces


def test_design_spaces_cover_our_generators_with_strict_schemas() -> None:
    spaces = design_spaces()
    assert [k for k in spaces if k[0] is EffectorFamily.DESIGNED] == [
        (EffectorFamily.DESIGNED, k) for k in DESIGNED_KINDS
    ]
    for key, space in spaces.items():
        assert space and all(isinstance(p, ParamSpec) for p in space)
        schema = params_json_schema(space)
        assert schema["additionalProperties"] is False and schema["required"] == [p.name for p in space]
        for p in space:
            assert p.default is not None and not p.problems(p.default), (key, p.name)
            assert not find_brand_tokens(p.description) and not HELDOUT_WORDS.search(p.description)
    assert parse_kind("fingered_hand") == (EffectorFamily.DESIGNED, "fingered_hand")
    assert parse_kind("soft_arm/tentacle") == (EffectorFamily.SOFT_ARM, "tentacle")
    with pytest.raises(KeyError):
        select_spaces(["no_such_kind"])


def test_other_families_join_the_designer_through_the_registry() -> None:
    """A parametric family that declares recipe kinds (track B2's soft arm) adds its spaces."""

    class FakeSoftArm(EffectorFamilyImpl):
        family = EffectorFamily.SOFT_ARM

        def presets(self) -> tuple[EffectorSpec, ...]:
            return ()

        def recipe_kinds(self) -> tuple[str, ...]:
            return ("tentacle",)

        def param_space(self, kind: str) -> tuple[ParamSpec, ...]:
            return (ParamSpec("segments", "int", "Segments of the arm.", "count", 12, 24, default=16),)

    with override_family(FakeSoftArm()):
        spaces = design_spaces()
        assert spaces[(EffectorFamily.SOFT_ARM, "tentacle")][0].name == "segments"
        assert space_label((EffectorFamily.SOFT_ARM, "tentacle")) == "soft_arm/tentacle"
        tool = submit_morphologies_tool(spaces, 2)
        kinds = tool["function"]["parameters"]["properties"]["designs"]["items"]["properties"]["kind"]["enum"]
        assert "soft_arm/tentacle" in kinds and "fingered_hand" in kinds
    assert (EffectorFamily.SOFT_ARM, "tentacle") not in design_spaces(include_foreign=False)


# --------------------------------------------------------------------------- canonical recipes and diversity


def test_recipe_id_is_stable_across_processes_and_ignores_hypothesis_and_origin() -> None:
    r = reference_recipes()[0]
    other = MorphologyRecipe(
        r.family, r.kind, dict(reversed(list(r.params.items()))), "another idea", "random"
    )
    assert other.recipe_id == r.recipe_id
    code = (
        "from rtt.effectors.designed import reference_recipes;"
        "print(','.join(x.recipe_id for x in reference_recipes()))"
    )
    env_path = str(REPO / "src")
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin"},
    ).stdout.strip()
    assert out.split(",") == [x.recipe_id for x in reference_recipes()]


def test_canonical_form_resets_unused_parameters_and_rounds_numbers() -> None:
    a = _recipe("fingered_hand", layout="even", spread_deg=12.3, pad="none", pad_thickness_mm=7.0)
    c = canonical_recipe(a)
    space = space_for(EffectorFamily.DESIGNED, "fingered_hand")
    defaults = {p.name: p.default for p in space}
    assert (
        c.params["spread_deg"] == defaults["spread_deg"]
        and c.params["pad_thickness_mm"] == defaults["pad_thickness_mm"]
    )
    assert set(inactive_params((a.family, a.kind), a.params)) == {"spread_deg", "pad_thickness_mm"}
    b = _recipe("fingered_hand", layout="even", spread_deg=99.0, pad="none", pad_thickness_mm=2.0)
    assert canonical_recipe(b).recipe_id == c.recipe_id
    radius = next(p for p in space if p.name == "palm_radius_mm")
    assert quantise(radius, 42.26) == 42.5 and quantise(radius, 99.0) == 70
    assert canonical_recipe(c) is c


def test_distance_points_and_the_screen() -> None:
    a = _recipe("fingered_hand")
    assert distance_points(a, a) == 0.0
    assert distance_points(a, _recipe("hook_gripper")) == math.inf
    one_choice = _recipe("fingered_hand", pad="round")
    assert distance_points(a, one_choice) == 100.0
    tweak = _recipe("fingered_hand", palm_radius_mm=float(a.params["palm_radius_mm"]) + 5.5)  # 10 points
    assert math.isclose(distance_points(a, tweak), 10.0)
    assert screen(a, [a]).reason == "duplicate"  # type: ignore[union-attr]
    verdict = screen(tweak, [a], labels=["t1"])
    assert verdict is not None and verdict.reason == "too_similar" and "t1" in verdict.message
    assert screen(one_choice, [a]) is None and MIN_POINTS == 100


# --------------------------------------------------------------------------- layout


def test_layout_budget_and_card() -> None:
    lay = layout_of(
        _recipe("fingered_hand", n_fingers=3, n_phalanges=2, grip_force_n=30.0, finger_length_mm=80.0)
    )
    assert lay.n_actuators == 6 and lay.n_digits == 3
    assert math.isclose(lay.digit_force, 10.0)
    # proximal torque = force per digit x 0.75 x length; the distal one only its own lever
    d = lay.digits[0]
    assert math.isclose(d.torque[0], 10.0 * 0.75 * 0.08) and d.torque[1] < d.torque[0]
    card = lay.actuation_card()
    assert (
        card["grip_force_n_total"] == 30.0 and card["actuators"] == 6.0 and card["joint_torque_nm_max"] <= 1.5
    )
    tendon = layout_of(_recipe("fingered_hand", actuation="tendon"))
    assert tendon.n_actuators == tendon.n_digits and tendon.tendon_coefs
    chuck = layout_of(_recipe("hook_gripper"))
    assert (
        chuck.digits[0].joint == "slide" and card and chuck.actuation_card()["slide_force_n_per_finger"] > 0
    )
    assert chuck.upright_rules() == ()
    pinch = layout_of(_recipe("fingered_hand", grasp="pinch", actuation="per_joint"))
    assert len(pinch.upright_rules()) == pinch.n_digits


# --------------------------------------------------------------------------- validators


#: The fixed adversarial set: (description, recipe, part, expected code). Every one must be refused.
def _adversarial() -> list[tuple[str, MorphologyRecipe, str, str]]:
    fh = "fingered_hand"
    return [
        ("seven fingers", _recipe(fh, n_fingers=7), "block", "out_of_bounds"),
        ("a 500 mm finger", _recipe(fh, finger_length_mm=500.0), "block", "out_of_bounds"),
        (
            "an unknown parameter",
            MorphologyRecipe("designed", fh, {**_recipe(fh).params, "wings": 2}),
            "block",
            "out_of_bounds",
        ),
        (
            "a missing parameter",
            MorphologyRecipe("designed", fh, {k: v for k, v in _recipe(fh).params.items() if k != "pad"}),
            "block",
            "out_of_bounds",
        ),
        ("a word for a number", _recipe(fh, n_fingers="three"), "block", "out_of_bounds"),
        ("an unknown pad", _recipe(fh, pad="velcro"), "block", "out_of_bounds"),
        (
            "an unknown generator",
            MorphologyRecipe("designed", "gecko_pad", {"x": 1}),
            "block",
            "out_of_bounds",
        ),
        (
            "five wide fingers on a small palm",
            _recipe(fh, n_fingers=5, finger_width_mm=30.0, palm_radius_mm=20.0),
            "hex_standoff",
            "self_intersecting",
        ),
        (
            "opposed fingers stacked at 0 deg spread",
            _recipe(fh, n_fingers=4, layout="opposed", spread_deg=0.0),
            "block",
            "self_intersecting",
        ),
        (
            "pads that cross the centre",
            _recipe(fh, palm_radius_mm=15.0, finger_thickness_mm=16.0, pad_thickness_mm=8.0, n_fingers=2),
            "block",
            "self_intersecting",
        ),
        (
            "a 4 mm thick 150 mm finger in PLA",
            _recipe(fh, finger_length_mm=150.0, finger_thickness_mm=4.0, n_phalanges=1),
            "block",
            "unprintable",
        ),
        (
            "a hook lip too thin for its reach",
            _recipe("hook_gripper", hook_depth_mm=10.0, hook_height_mm=2.0),
            "steel_slug",
            "unprintable",
        ),
        (
            "pencil-thin tentacle tips",
            _recipe("segmented_tentacle", root_radius_mm=3.0, taper=0.3),
            "block",
            "unprintable",
        ),
        (
            "a steel five-finger hand",
            _recipe(
                fh,
                material="steel",
                n_fingers=5,
                n_phalanges=1,
                finger_length_mm=150.0,
                finger_width_mm=20.0,
                finger_thickness_mm=16.0,
                palm_radius_mm=70.0,
                palm_thickness_mm=30.0,
            ),
            "block",
            "too_heavy",
        ),
        (
            "15 servos",
            _recipe(fh, n_fingers=5, n_phalanges=3, palm_radius_mm=60.0, finger_width_mm=10.0),
            "block",
            "over_budget",
        ),
        (
            "50 N on two long fingers",
            _recipe(fh, n_fingers=2, grip_force_n=50.0, finger_length_mm=140.0, finger_thickness_mm=10.0),
            "block",
            "over_budget",
        ),
        ("a narrow palm on the tee", _recipe(fh, palm_radius_mm=20.0), "pipe_tee", "too_narrow"),
        (
            "a narrow chuck on the slug",
            _recipe("hook_gripper", palm_radius_mm=24.0),
            "steel_slug",
            "too_narrow",
        ),
        (
            "short fingers on the block",
            _recipe(fh, finger_length_mm=40.0, grip_height_frac=0.1),
            "block",
            "too_short",
        ),
        (
            "closing above the tee",
            _recipe(fh, grip_height_frac=0.9, finger_length_mm=60.0),
            "pipe_tee",
            "above_part",
        ),
        ("converging fingers on a small part", reference_recipes()[1], "hex_standoff", "digit_collision"),
        (
            "one tentacle digit that never reaches",
            _recipe(
                "segmented_tentacle", n_digits=1, n_segments=3, palm_radius_mm=70.0, digit_length_mm=60.0
            ),
            "hex_standoff",
            "cannot_reach",
        ),
        (
            "only the neighbouring fingers reach the tee",
            _recipe(
                fh,
                actuation="tendon",
                finger_length_mm=38.0,
                finger_thickness_mm=8.5,
                finger_width_mm=7.0,
                grasp="power",
                grip_force_n=35.0,
                grip_height_frac=0.31,
                layout="opposed",
                n_fingers=5,
                n_phalanges=1,
                pad="flat",
                pad_thickness_mm=7.0,
                palm_radius_mm=55.5,
                palm_thickness_mm=23.0,
                spread_deg=78.0,
                yaw_deg=7.0,
            ),
            "pipe_tee",
            "no_opposition",
        ),
        (
            "fingers that swing over the top of the standoff",
            _recipe(
                fh,
                n_fingers=5,
                palm_radius_mm=64.0,
                finger_length_mm=62.0,
                finger_width_mm=18.0,
                finger_thickness_mm=12.0,
                proximal_frac=0.74,
                grip_height_frac=0.18,
                pad_thickness_mm=1.5,
                palm_thickness_mm=14.0,
                grip_force_n=27.5,
                yaw_deg=76.0,
            ),
            "hex_standoff",
            "cannot_reach",
        ),
    ]


def test_validators_reject_the_whole_adversarial_set_with_readable_reasons(task: Any) -> None:
    rows = _adversarial()
    seen_codes = set()
    for what, recipe, part_name, code in rows:
        problems = validate_recipe(recipe, get_part(part_name), task)
        assert problems, f"{what}: accepted"
        codes = validation_codes(problems)
        assert code in codes, f"{what}: {problems}"
        for p in problems:
            c, sentence = p.split(": ", 1)
            assert c in CHECK_CODES and len(sentence) > 20, p
        seen_codes.add(code)
    assert len(rows) >= 20 and seen_codes >= {
        "out_of_bounds",
        "self_intersecting",
        "too_narrow",
        "too_heavy",
        "over_budget",
        "unprintable",
    }


def test_validators_accept_the_reference_designs_where_they_fit(task: Any) -> None:
    ok = {
        (r.kind, p.name)
        for r in reference_recipes()
        for p in training_parts()
        if not validate_recipe(r, p, task)
    }
    assert ("hook_gripper", "steel_slug") in ok and ("fingered_hand", "hex_standoff") in ok
    for r in reference_recipes():
        assert validate_structure(r) == []


def test_fit_numbers_explain_the_opening_rule(task: Any) -> None:
    lay = layout_of(_recipe("fingered_hand", palm_radius_mm=20.0))
    rows = fit_numbers(lay, get_part("pipe_tee"))
    assert len(rows) == lay.n_digits
    assert all(math.isclose(r["needed_mm"] - r["outline_mm"], 5.0) for r in rows)
    assert any(r["inner_mm"] < r["needed_mm"] for r in rows)


def test_multi_part_rule_needs_two_thirds(task: Any) -> None:
    parts = list(training_parts())
    chuck = reference_recipes()[2]
    problems = recipe_problems(chuck, parts, task, 4)
    built = [p.name for p in parts if not validate_recipe(chuck, p, task)]
    assert (problems == []) == (len(built) >= 4)
    assert recipe_problems(chuck, parts, task, 1) == []
    if problems:
        assert problems[0].startswith(f"buildable on {len(built)} of 6")


# --------------------------------------------------------------------------- random control


def test_random_control_is_valid_seeded_and_diverse(task: Any) -> None:
    part = get_part("block")
    a = random_recipes(part, 6, seed=3, task=task)
    b = random_recipes(part, 6, seed=3, task=task)
    c = random_recipes(part, 6, seed=4, task=task)
    assert [r.recipe_id for r in a] == [r.recipe_id for r in b] != [r.recipe_id for r in c]
    assert len(a) == 6 and all(r.origin == "random" for r in a)
    for i, r in enumerate(a):
        assert validate_recipe(r, part, task) == [] and canonical_recipe(r) is r
        assert screen(r, a[:i]) is None
        assert spec_for(r).group.value == "random"
    only = random_recipes(part, 3, seed=0, kinds=["hook_gripper"], task=task)
    assert {r.kind for r in only} == {"hook_gripper"}
    recs, stats = random_recipes_with_stats(list(training_parts()), 2, seed=0, task=task)
    assert (
        len(recs) == 2
        and stats.draws >= 2
        and set(stats.refused) <= set(CHECK_CODES) | {"duplicate", "too_similar"}
    )
    assert all(recipe_problems(r, list(training_parts()), task, 4) == [] for r in recs)


# --------------------------------------------------------------------------- proposer (scripted transport)


def _completion(
    args: Any, *, reasoning: str | None = None, prompt: int = 6000, completion: int = 1500
) -> dict:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "submit_morphologies",
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            }
        ],
    }
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


class ScriptedDesigner:
    """A fake Token Factory transport answering each request with the next scripted tool call."""

    def __init__(self, scripts: list[Any]) -> None:
        self.scripts = list(scripts)
        self.payloads: list[dict] = []

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        self.payloads.append(json.loads(json.dumps(payload)))
        item = self.scripts.pop(0)
        if isinstance(item, TransportResponse):
            return item
        return TransportResponse(
            200, {}, _completion(item, reasoning="I compare grippers, such as an Allegro hand.")
        )

    def list_models(self, timeout_s: float) -> TransportResponse:
        raise NotImplementedError


def _design(recipe: MorphologyRecipe, hypothesis: str = "it should hold") -> dict[str, Any]:
    return {
        "hypothesis": hypothesis,
        "kind": space_label((recipe.family, recipe.kind)),
        "params": dict(recipe.params),
    }


def _client(tmp_path: Path, transport: Any, **kw: Any) -> Any:
    return make_designer_client(tmp_path / "cache", transport=transport, limiter=TokenBucket(1e12), **kw)


def test_tool_schema_puts_analysis_first_and_fixes_the_count() -> None:
    tool = submit_morphologies_tool(design_spaces(), 4)
    fn = tool["function"]
    assert fn["name"] == "submit_morphologies"
    params = fn["parameters"]
    assert list(params["properties"]) == ["analysis", "designs"] and params["additionalProperties"] is False
    designs = params["properties"]["designs"]
    assert designs["minItems"] == designs["maxItems"] == 4
    item = designs["items"]
    assert list(item["properties"]) == ["hypothesis", "kind", "params"]
    schemas = item["properties"]["params"]["anyOf"]
    assert len(schemas) == len(design_spaces())
    one = submit_morphologies_tool(select_spaces(["hook_gripper"]), 2)
    p1 = one["function"]["parameters"]["properties"]["designs"]["items"]["properties"]["params"]
    assert "anyOf" not in p1 and p1["required"] == [p.name for p in space_for("designed", "hook_gripper")]


def test_prompt_is_part_aware_clean_and_brand_free(task: Any) -> None:
    part = get_part("flanged_bushing")
    chuck = reference_recipes()[2]
    history = [DesignOutcome(chuck, 0.75, 0.5, {"slip_during_lift": 2}, n_episodes=8)]
    msgs = build_messages(part, history, 4, design_spaces(), task)
    assert_clean(msgs)
    text = "\n".join(m["content"] for m in msgs)
    assert "flanged_bushing" in text and "36 x 36 x 35 mm" in text and "flange" in text
    assert "Opening needed" in text and "held 6/8, registered 4/8" in text and "slip_during_lift 2" in text
    assert not find_brand_tokens(text) and not HELDOUT_WORDS.search(text)
    others = {p.name for p in training_parts()} - {part.name}
    assert not any(f"- {name} (" in text or f"## {name}" in text for name in others)
    assert not any(name in text for name in others if "_" in name)
    multi = "\n".join(
        m["content"] for m in build_messages(list(training_parts()), [], 4, design_spaces(), task)
    )
    assert all(f"## {p.name}" in multi for p in training_parts()) and "at least 4 of them" in multi
    with pytest.raises(PromptLeakError):
        assert_clean([{"role": "user", "content": "the held-out suite"}])
    with pytest.raises(PromptLeakError):
        assert_clean([{"role": "user", "content": "copy the Allegro hand"}])
    assert scrub("like an Allegro hand on a spur gear") == "like an [removed] hand on a [removed] [removed]"
    assert set(mprompts.template_hashes()) == set(mprompts.TEMPLATE_NAMES)


def test_proposer_validates_screens_and_repairs_once(tmp_path: Path, task: Any) -> None:
    part = get_part("steel_slug")
    ref = reference_recipes()
    bad = {**ref[0].params, "n_fingers": 7}
    main = {
        "analysis": "Round parts suit a chuck.",
        "designs": [
            _design(ref[2], "a chuck centres the slug"),
            {"hypothesis": "too many", "kind": "fingered_hand", "params": bad},
            _design(ref[2], "the same chuck again"),
            _design(ref[0], "three upright fingers"),
        ],
    }
    repair = {
        "analysis": "Replacing designs 2 and 3.",
        "designs": [
            _design(ref[2]),
            _design(ref[3], "three curling digits"),
            {"hypothesis": "unknown", "kind": "gecko_pad", "params": {}},
            _design(ref[0]),
        ],
    }
    fake = ScriptedDesigner([main, repair])
    client = _client(tmp_path, fake, budget_usd=0.05)
    history: list[DesignOutcome] = []
    batch = propose_recipes(part, history, 4, client=client, task=task)
    assert len(fake.payloads) == 2 and len(batch.calls) == 2  # main + exactly one repair
    assert [r.kind for r in batch.recipes] == ["hook_gripper", "segmented_tentacle", "fingered_hand"]
    assert batch.shortfall == 1 and all(r.origin == "nemotron" for r in batch.recipes)
    reasons = [(r.slot, r.stage, r.reason) for r in batch.rejections]
    assert reasons == [(2, "main", "invalid"), (3, "main", "duplicate"), (3, "repair", "invalid")]
    repair_prompt = fake.payloads[1]["messages"][-1]["content"]
    assert "design 2:" in repair_prompt and "n_fingers must be in [2, 5]" in repair_prompt
    assert "Keep design 1, design 4 exactly" in repair_prompt
    for payload in fake.payloads:
        assert payload["tool_choice"] == {"type": "function", "function": {"name": "submit_morphologies"}}
        assert payload["model"] == "nvidia/nemotron-3-super-120b-a12b"
        assert_clean(payload["messages"])
    d = batch.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["cost_usd"] > 0 and len(d["ledger"]) == 2

    # replay: the same batch from the cache, zero calls, no key needed
    replay = make_designer_client(tmp_path / "cache", cache_mode="replay")
    again = propose_recipes(part, history, 4, client=replay, task=task)
    assert [r.recipe_id for r in again.recipes] == [r.recipe_id for r in batch.recipes]
    assert all(c.cached for c in again.calls) and again.cost_usd == 0.0
    with pytest.raises(CacheMiss):
        propose_recipes(get_part("block"), history, 4, client=replay, task=task)


def test_second_round_sees_the_first_rounds_outcomes(tmp_path: Path, task: Any) -> None:
    part = get_part("hex_standoff")
    ref = reference_recipes()
    fake = ScriptedDesigner(
        [{"analysis": "a", "designs": [_design(ref[2])]}, {"analysis": "b", "designs": [_design(ref[3])]}]
    )
    client = _client(tmp_path, fake)
    first = propose_recipes(part, [], 1, client=client, task=task)
    outcomes = [DesignOutcome(r, 1.0, 0.25, {"not_registered": 6}, n_episodes=8) for r in first.recipes]
    second = propose_recipes(part, outcomes, 1, client=client, task=task)
    assert [r.kind for r in second.recipes] == ["segmented_tentacle"]
    prompt = fake.payloads[1]["messages"][1]["content"]
    assert "t1 hook_gripper [nemotron]: held 8/8, registered 2/8; failures: not_registered 6" in prompt
    assert "None yet" in fake.payloads[0]["messages"][1]["content"]


def test_unusable_calls_spend_caps_and_billing(tmp_path: Path, task: Any) -> None:
    part = get_part("block")
    ref = reference_recipes()
    garbage = ScriptedDesigner(["{not json at all", {"analysis": "ok", "designs": [_design(ref[0])]}])
    batch = propose_recipes(part, [], 1, client=_client(tmp_path / "a", garbage), task=task)
    assert [c.stage for c in batch.calls] == ["main", "repair"] and len(batch.recipes) == 1
    assert batch.rejections[0].reason == "parse_error"
    # a spend cap smaller than one call's worst case: nothing is sent, nothing crashes
    capped = ScriptedDesigner([])
    batch = propose_recipes(part, [], 4, client=_client(tmp_path / "b", capped, budget_usd=0.001), task=task)
    assert capped.payloads == [] and batch.recipes == () and batch.rejections[0].reason == "spend_cap"
    assert len(batch.calls) == 1  # no repair after a spend cap
    from rtt.llm.client import LLMBillingError

    broke = ScriptedDesigner([TransportResponse(402, {}, {"error": {"message": "insufficient balance"}})])
    with pytest.raises(LLMBillingError):
        propose_recipes(part, [], 1, client=_client(tmp_path / "c", broke), task=task)


def test_scrubbing_cache_keeps_replays_exact_and_drops_reasoning(tmp_path: Path, task: Any) -> None:
    part = get_part("steel_slug")
    ref = reference_recipes()
    args = {
        "analysis": "Unlike a Shadow Hand style or Allegro clone, a chuck.",
        "designs": [_design(ref[2], "beats the Allegro hand")],
    }
    fake = ScriptedDesigner([args])
    client = _client(tmp_path, fake)
    assert isinstance(client.cache, ScrubbingCache)
    live = propose_recipes(part, [], 1, client=client, task=task)
    files = list((tmp_path / "cache").rglob("*.json"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "reasoning_content" not in text and not find_brand_tokens(text)
    replay = propose_recipes(
        part, [], 1, client=make_designer_client(tmp_path / "cache", cache_mode="replay"), task=task
    )
    assert [r.recipe_id for r in replay.recipes] == [r.recipe_id for r in live.recipes]
    assert (
        [r.hypothesis for r in replay.recipes]
        == [r.hypothesis for r in live.recipes]
        == ["beats the [removed] hand"]
    )
    assert replay.analysis == live.analysis and not find_brand_tokens(live.analysis or "")
    assert isinstance(
        ResponseCache(tmp_path / "cache", mode="replay").get(json.loads(text)["request_hash"]), object
    )


def test_design_outcome_validates_rates() -> None:
    r = reference_recipes()[0]
    with pytest.raises(ValueError):
        DesignOutcome(r, 1.5, 0.0, {})
    o = DesignOutcome(r, 0.5, 0.25, {"slip_during_lift": 3}, n_episodes=8, per_part={"block": (0.5, 0.25)})
    assert o.per_part == {"block": (0.5, 0.25)} and o.failure_modes == {"slip_during_lift": 3}


# --------------------------------------------------------------------------- the recorded live smoke run


def _manifest() -> dict[str, Any]:
    path = RECORDED / "manifest.json"
    if not path.is_file():
        pytest.skip("no recorded smoke run")
    return json.loads(path.read_text(encoding="utf-8"))


def test_recorded_smoke_run_stayed_under_its_cap_and_carries_the_label() -> None:
    m = _manifest()
    assert m["mode"] == "live" and m["budget_usd"] == 0.05
    assert m["llm"]["cost_usd"] <= 0.05 and m["llm"]["live_calls"] >= 2
    assert m["label"] == "Product mode: not part of the pre-registered experiment"
    assert m["model"] == "nvidia/nemotron-3-super-120b-a12b"
    ledger = [
        json.loads(line) for line in (RECORDED / "ledger.jsonl").read_text().splitlines() if line.strip()
    ]
    assert len(ledger) == m["llm"]["calls"] and sum(e["cost_usd"] for e in ledger) <= 0.05
    for path in RECORDED.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert not find_brand_tokens(text) and not HELDOUT_WORDS.search(text), path.name
        assert "reasoning_content" not in text


def test_recorded_smoke_run_replays_exactly_with_zero_calls(task: Any) -> None:
    """Both design rounds re-run from the recorded cache: same recipes, no API key, no calls.

    The run offered the designed family's own generators only (the suction and soft-arm
    families were not built yet), so the replay asks for exactly those: once other families
    register their spaces, the default prompt offers them too and is a different request.
    """
    m = _manifest()
    parts = [get_part(name) for name in m["parts"]]
    client = make_designer_client(RECORDED / "cache", cache_mode="replay")
    by_id = {d["recipe"]["recipe_id"]: d for d in m["arms"]["nemotron"]["designs"]}
    history: list[DesignOutcome] = []
    for rnd, recorded in enumerate(m["batches"], start=1):
        batch = propose_recipes(
            parts,
            history,
            recorded["n_requested"],
            client=client,
            task=task,
            tag=f"smoke/round{rnd}",
            kinds=DESIGNED_KINDS,
        )
        assert [r.recipe_id for r in batch.recipes] == [r["recipe_id"] for r in recorded["recipes"]]
        assert all(c.cached for c in batch.calls) and batch.cost_usd == 0.0
        for r in batch.recipes:
            d = by_id[r.recipe_id]
            history.append(
                DesignOutcome(
                    r,
                    d["held_rate"],
                    d["registered_rate"],
                    d["failure_modes"],
                    n_episodes=d["n_episodes"],
                    # in the run's part order (the manifest's keys are sorted)
                    per_part={k: tuple(d["per_part"][k]) for k in m["parts"]},  # type: ignore[misc]
                )
            )
    assert client.ledger.spent_usd == 0.0
