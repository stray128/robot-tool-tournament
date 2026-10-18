"""The Studio data contract ``rtt-effectors/2`` (``docs/STUDIO.md``; no physics here).

* guard rails: PREREG A5 code listings and files, and the v1 effector contract files, unchanged;
  the Studio never reads held-out material;
* task specs: construction, cross-reference validation (every problem listed at once), ids,
  round trips through the Task Engineer's tool form;
* the Task Engineer's JSON schema: strict (closed objects, every property required, bounded
  numbers and strings), inside the repository's schema subset, valid and invalid arguments;
* object templates and primitives: param spaces, defaults, handles, quantities, fidelity tiers;
* setups: the three kinds, ids, checks against a task, segments (the tool-change hook);
* results: step outcomes, task results, setup aggregation and ranking; the build pack manifest.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from rtt import studio
from rtt.effectors.types import BRAND_TOKENS, PRODUCT_MODE_LABEL, ParamSpec
from rtt.studio.types import (
    CONTRACT_VERSION,
    MAX_STEPS,
    PRIMITIVES,
    QUANTITIES,
    STUDIO_WRIST,
    TASK_PLAN_TOOL,
    TOOL_CHANGE_S,
    BomLine,
    BuildPackFile,
    BuildPackLink,
    BuildPackManifest,
    CertificateRef,
    ChangeKind,
    Clarification,
    ComplexityMetrics,
    FidelityTier,
    HeadMount,
    ObjectInstance,
    ObjectPose,
    PrimitiveKind,
    Setup,
    SetupKind,
    SetupResult,
    Step,
    StepConstraints,
    StepOutcome,
    SuccessCriterion,
    TaskResult,
    TaskSpec,
    TaskSpecError,
    TemplateKind,
    ToolChanger,
    UnsupportedItem,
    buildpack_manifest_json_schema,
    get_template,
    submit_task_plan_tool,
    task_plan_json_schema,
    task_spec_from_tool_args,
    templates,
    tier_of_effector,
    worst_tier,
)
from schema_lite import Validator, unsupported_keywords
from studio_fixtures import cup, gripper
from test_effectors_contract import CONTRACT_FILES, _listing

REPO = Path(__file__).resolve().parents[1]
SHA = "a" * 64


# --------------------------------------------------------------------------- guard rails


def test_prereg_a5_frozen_inputs_are_unchanged() -> None:
    """PREREG A5: the C-locale code listings and the frozen files, recomputed from disk."""
    listings = {
        "src/rtt/design/freeform": "f62376d69fa331ccb66da5d78828bd17e1f5f0d7d4fba60661f54f51a27e9422",
        "src/rtt/parts": "5f80c1776d413fc1c4150ddc61b936f8145e0904ec640c9049cd957ecc5a3f99",
        "src/rtt/analysis": "c78462aadacfc24a714085cd4b9c5476fbfa94d050289e76c572cf6cd2914043",
        "src/rtt/search": "42861eee09114842c0af0dd7404add70b762e7254c9d90ae0cf20682052d138d",
        "src/rtt/sim": "069763de5090a66cef204186a81bb221433ca7ed1e2b54b447dfb728a48d04ea",
    }
    for rel, want in listings.items():
        assert _listing(REPO / rel) == want, f"{rel} changed: it is frozen by PREREG A5"
    assert _listing(REPO / "src/rtt/search/prompts/v2", "*") == (
        "f9d2b5ac0c39fb1af6f7e56663e32209bfca9e573605fab1d46f2ffda7d5575e"
    )
    files = {
        "configs/task_v2.yaml": "bef7297f8e1ff2a6270cd3f1ffb6de8690928f46485828a58fd1ce2313d87487",
        "configs/suites/train_v2.yaml": "ebd292b98fb46d220589a96e1d18d465e9b3152657401b11c3efd323e2625c63",
        "configs/entrants_v2.yaml": "7d9ac3922a34f3904e9ffb1cd868442197c3b94e475682cad2bbba3851faf421",
        "configs/parts_v2.yaml": "43b35533a91746d3333714a648f0d3ca300b7dc2d88297749f6bdbd07dbfff70",
        "src/rtt/analysis/hero_rule.py": "c4746149ceb6ada0bf11aa241748a3e68ee6b32a0b37fc4cc3f3fca94f946aad",
        "src/rtt/analysis/tau.py": "2bac2736b3f1a52c16454b2ff118d711993af3c3514abb176eb5c8b9955b9cc3",
        "src/rtt/analysis/power.py": "af86b018d39bf9f46949fae6a7b860dd346b95dbe81ccd5f1485b8e66b47ff72",
        "src/rtt/analysis/s1.py": "d6f4e3b1611bdc1a682cdf800f411aaddc4e57262906adcdc5f00abfead6209e",
        "configs/diversity_v2.json": "37ec231e27758f1a649544588863ac1dbbd4d0ae0b0c380b622d3421b611a7eb",
        "configs/seeds_main.yaml": "e09bc94071e4ceff97acf3cd1b068454feafc2e553febbcb24c18f593830cb9f",
        "configs/suites/heldout_procedural_v2.yaml": "626141b45c6462783aa7d699a9aad932472d2f548dfd243e346eaeff939325f0",
        "uv.lock": "e876d63062a8c3d88f748b28b1618f9c474b75fdf56c3b9011b3ca61df24e922",
    }
    for rel, want in files.items():
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == want, f"{rel} is frozen by PREREG A5"


def test_the_v1_effector_contract_is_unchanged() -> None:
    root = REPO / "src/rtt/effectors"
    current = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in CONTRACT_FILES}
    assert current == CONTRACT_FILES
    from rtt.effectors.types import CONTRACT_VERSION as V1

    assert V1 == "rtt-effectors/1" and CONTRACT_VERSION == "rtt-effectors/2"


def test_studio_code_never_reads_held_out_material() -> None:
    root = Path(studio.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "vault" not in node.module and "heldout" not in node.module, path.name
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                low = node.value.lower()
                assert "heldout" not in low and "secrets/" not in low, path.name


def test_importing_the_package_loads_no_mujoco_module() -> None:
    import subprocess
    import sys

    code = "import sys, rtt.studio; print(any(m.startswith('rtt.studio.scene') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------- a hero-like spec (H1)


def h1_spec(**overrides: Any) -> TaskSpec:
    """Warehouse returns and kitting: open the tote lid, pick a carton and a flat item, close the lid."""
    objects = (
        ObjectInstance.of("tote_lid", "hinged_lid", "Tote lid", ObjectPose(0.0, 120.0, 0.0), width_mm=400),
        ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0.0, 120.0, 0.0), length_mm=400, width_mm=300),
        ObjectInstance.of(
            "carton",
            "part",
            "Carton",
            ObjectPose(-150.0, -150.0, 0.0),
            size_x_mm=120,
            size_y_mm=80,
            size_z_mm=60,
            mass_g=300,
        ),
        ObjectInstance.of(
            "flat_item",
            "part",
            "Flat wrapped item",
            ObjectPose(150.0, -150.0, 0.0),
            size_x_mm=200,
            size_y_mm=150,
            size_z_mm=15,
            mass_g=120,
        ),
    )
    steps = (
        Step.of(
            "open_lid",
            "hook_pull",
            "tote_lid",
            "handle",
            "Hook the lid lip and pull it open",
            [SuccessCriterion("angle_deg", at_least=80.0), SuccessCriterion("slip_mm", at_most=10.0)],
        ),
        Step.of(
            "pick_carton",
            "grasp",
            "carton",
            "part",
            "Pick the carton into the tote",
            [SuccessCriterion("in_target", at_least=1.0)],
            target="tote",
            after=("open_lid",),
        ),
        Step.of(
            "pick_flat",
            "suction",
            "flat_item",
            "part",
            "Pick the flat item into the tote",
            [SuccessCriterion("in_target", at_least=1.0)],
            target="tote",
            after=("open_lid",),
        ),
        Step.of(
            "close_lid",
            "push",
            "tote_lid",
            "lid",
            "Push the lid closed",
            [SuccessCriterion("angle_deg", at_most=5.0)],
            constraints=StepConstraints(max_force_n=40.0),
            after=("pick_carton", "pick_flat"),
        ),
    )
    kwargs: dict[str, Any] = {
        "title": "Returns kitting with a lidded tote",
        "description": "Open a hinged tote lid, place a carton and a flat item inside, close the lid.",
        "objects": objects,
        "steps": steps,
        "assumptions": ("The tote is fixed to the table.",),
        "clarifications": (Clarification("How heavy is the carton?", "About 300 g."),),
        "unsupported": (UnsupportedItem("Shrink wrap film", "Deformables are not simulated yet", "suction"),),
        "source": "test",
    }
    kwargs.update(overrides)
    return TaskSpec(**kwargs)


def test_task_spec_construction_ids_and_fidelity() -> None:
    spec = h1_spec()
    assert spec.step_ids == ("open_lid", "pick_carton", "pick_flat", "close_lid")
    assert len(spec.task_id) == 64
    assert h1_spec(title="Another title", description="").task_id == spec.task_id  # free text is not hashed
    assert spec.fidelity("pick_flat") is FidelityTier.APPROXIMATION  # suction
    assert spec.fidelity("open_lid") is FidelityTier.RIGID
    assert spec.caveats("pick_flat") and not spec.caveats("pick_carton")
    d = spec.to_dict()
    assert d["contract"] == CONTRACT_VERSION and d["label"] == PRODUCT_MODE_LABEL
    json.dumps(d, allow_nan=False)
    again = pickle.loads(pickle.dumps(spec))
    assert again.task_id == spec.task_id
    with pytest.raises(TypeError):
        spec.steps[0].params["speed_mm_s"] = 1.0  # type: ignore[index]


def test_task_spec_lists_every_cross_reference_problem_at_once() -> None:
    good = h1_spec()
    s = list(good.steps)
    bad = [
        Step.of(
            "open_lid",
            "twist",
            "tote_lid",
            "handle",
            "Turn the lid",
            [SuccessCriterion("angle_deg", at_least=1)],
        ),
        Step.of(
            "pick_carton",
            "grasp",
            "ghost",
            "part",
            "Pick a ghost",
            [SuccessCriterion("lift_mm", at_least=10)],
        ),
        Step.of(
            "pick_flat",
            "suction",
            "flat_item",
            "lid",
            "Wrong handle",
            [SuccessCriterion("turns", at_least=1)],
            target="carton",
            after=("close_lid",),
        ),
        s[3],
    ]
    with pytest.raises(TaskSpecError) as err:
        h1_spec(steps=tuple(bad))
    text = " | ".join(err.value.problems)
    for needle in (
        "twist cannot act on a hinged_lid",
        "unknown object 'ghost'",
        "handle 'lid' is not one of",
        "'turns' is not measured here",
        "target 'carton' must be one of",
        "depends on 'close_lid'",
    ):
        assert needle in text, needle
    assert len(err.value.problems) >= 6


def test_holding_rules() -> None:
    port = ObjectInstance.of("port", "port_plug", "Test port")
    other = ObjectInstance.of("box", "part", "Box")
    grab = Step.of(
        "grab",
        "grasp",
        "port",
        "plug",
        "Pick the plug",
        [SuccessCriterion("lift_mm", at_least=10)],
        hold_after=True,
    )
    plug_in = Step.of(
        "plug_in", "insert", "port", "plug", "Insert it", [SuccessCriterion("depth_mm", at_least=10)]
    )
    ok = TaskSpec("Plug in", "", (port, other), (grab, plug_in), source="test")
    assert ok.steps[0].hold_after
    poke = Step.of("poke", "push", "box", "part", "Push the box", [SuccessCriterion("force_n", at_least=1)])
    with pytest.raises(TaskSpecError, match="still held"):
        TaskSpec("Drop", "", (port, other), (grab, poke), source="test")
    with pytest.raises(TaskSpecError, match="last step must release"):
        TaskSpec("Hold", "", (port,), (grab,), source="test")
    with pytest.raises(ValueError, match="cannot end holding"):
        Step.of("p", "push", "box", "part", "x", [SuccessCriterion("force_n", at_least=1)], hold_after=True)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: ObjectInstance.of("Bad Name", "part", "x"), "snake_case"),
        (lambda: ObjectInstance.of("lid", "hinged_lid", "Lid", width_mm=5000), "width_mm must be in"),
        (lambda: ObjectInstance.of("lid", "hinged_lid", "Lid", colour="red"), "unknown param"),
        (lambda: ObjectInstance.of("p", "part", "Part", shape="mesh"), "needs its ref"),
        (lambda: ObjectInstance.of("p", "part", "Part", ref="library:block"), "only a mesh part has a ref"),
        (lambda: ObjectInstance.of("p", "part", "Part", ref="http://x", shape="mesh"), "ref must match"),
        (lambda: ObjectInstance.of("p", "part", f"{BRAND_TOKENS[0].title()} part"), "brand"),
        (lambda: ObjectPose(x_mm=900.0), "pose.x_mm"),
        (lambda: SuccessCriterion("angle_deg"), "needs at_least or at_most"),
        (lambda: SuccessCriterion("angle_deg", at_least=10, at_most=5), "at_least 10 > at_most 5"),
        (lambda: SuccessCriterion("speed", at_least=1), "unknown quantity"),
        (lambda: StepConstraints(max_force_n=0.0), "max_force_n must be in"),
        (
            lambda: Step.of(
                "s",
                "twist",
                "x",
                "head",
                "x",
                [SuccessCriterion("angle_deg", at_least=1)],
                params={"direction": "sideways"},
            ),
            "direction must be one of",
        ),
        (lambda: Step.of("s", "push", "x", "part", "x", []), "1 to 4 SuccessCriterion"),
    ],
)
def test_field_validation(factory: Any, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


def test_refusals_and_limits() -> None:
    refused = TaskSpec(
        "Fold a shirt", "", (), (), refusal="Cloth is deformable and not simulated yet.", source="test"
    )
    assert refused.refusal and not refused.steps
    with pytest.raises(TaskSpecError, match="a refused task has no steps"):
        h1_spec(refusal="no")
    with pytest.raises(TaskSpecError, match="at least one step"):
        h1_spec(steps=())
    many = tuple(
        Step.of(f"s{i}", "push", "carton", "part", "Push", [SuccessCriterion("force_n", at_least=1)])
        for i in range(MAX_STEPS + 1)
    )
    with pytest.raises(TaskSpecError, match=f"at most {MAX_STEPS} steps"):
        h1_spec(steps=many)


# --------------------------------------------------------------------------- templates and primitives


def test_templates_cover_every_kind_with_bounded_defaults() -> None:
    kinds = [t.kind for t in templates()]
    assert kinds == list(TemplateKind)
    for t in templates():
        assert t.problems(t.defaults()) == [], t.kind
        assert all(isinstance(p, ParamSpec) for p in t.params)
        assert all(q in QUANTITIES for q in t.quantities)
        assert t.fidelity is FidelityTier.RIGID or t.caveat, t.kind
    approx = {t.kind for t in templates() if t.fidelity is FidelityTier.APPROXIMATION}
    assert approx == {
        TemplateKind.QUARTER_TURN_LATCH,
        TemplateKind.THREADED_JOINT,
        TemplateKind.HANDWHEEL_VALVE,
    }


def test_primitives_reference_real_templates_handles_and_quantities() -> None:
    assert set(PRIMITIVES) == set(PrimitiveKind)
    for kind, info in PRIMITIVES.items():
        for tk in info.objects:
            assert set(info.handles) & set(get_template(tk).handles), (kind, tk)
        assert all(q in QUANTITIES for q in info.quantities)
        assert all(p.default is not None for p in info.params)
        assert info.fidelity is FidelityTier.RIGID or info.caveat
    assert [k for k, v in PRIMITIVES.items() if v.uses_roll] == [PrimitiveKind.TWIST]
    assert {k for k, v in PRIMITIVES.items() if v.status == "new"} == {
        PrimitiveKind.HOOK_PULL,
        PrimitiveKind.TWIST,
        PrimitiveKind.INSERT,
        PrimitiveKind.PUSH,
    }


def test_fidelity_tiers() -> None:
    assert worst_tier() is FidelityTier.RIGID
    assert worst_tier("rigid", "approximation") is FidelityTier.APPROXIMATION
    assert worst_tier("approximation", "not_simulated", "rigid") is FidelityTier.NOT_SIMULATED
    assert tier_of_effector("core") is FidelityTier.RIGID
    assert tier_of_effector("approximation") is FidelityTier.APPROXIMATION
    assert tier_of_effector(cup().fidelity) is FidelityTier.APPROXIMATION


# --------------------------------------------------------------------------- the Task Engineer's schema


def _walk(schema: Any, path: str = "#") -> list[str]:
    """Every object schema that is not closed with all properties required, and every unbounded
    number or string."""
    bad: list[str] = []
    if isinstance(schema, dict):
        types = schema.get("type")
        types = types if isinstance(types, list) else [types]
        if "object" in types:
            props = schema.get("properties", {})
            if schema.get("additionalProperties") is not False or sorted(
                schema.get("required", [])
            ) != sorted(props):
                bad.append(f"{path}: not strict")
        if "string" in types and "enum" not in schema and "const" not in schema and "maxLength" not in schema:
            bad.append(f"{path}: unbounded string")
        numeric = "number" in types or "integer" in types
        per_quantity = path.split("/")[-1] in ("at_least", "at_most")  # bounded per quantity by the validator
        if numeric and not per_quantity and not ({"minimum", "maximum"} <= schema.keys()):
            bad.append(f"{path}: unbounded number")
        if "array" in types and "maxItems" not in schema:
            bad.append(f"{path}: unbounded array")
        for k, v in schema.items():
            if k in ("properties", "$defs"):
                for name, sub in v.items():
                    bad += _walk(sub, f"{path}/{k}/{name}")
            elif k == "anyOf":
                for i, sub in enumerate(v):
                    bad += _walk(sub, f"{path}/anyOf/{i}")
            elif k == "items":
                bad += _walk(v, f"{path}/items")
    return bad


def test_task_plan_schema_is_strict_bounded_and_in_the_supported_subset() -> None:
    schema = task_plan_json_schema()
    assert unsupported_keywords(schema) == []
    assert _walk(schema) == []
    tool = submit_task_plan_tool()
    assert tool["type"] == "function" and tool["function"]["name"] == TASK_PLAN_TOOL
    assert tool["function"]["parameters"] == schema
    json.dumps(tool, allow_nan=False)
    assert len(schema["properties"]["objects"]["items"]["anyOf"]) == len(TemplateKind)
    assert len(schema["properties"]["steps"]["items"]["anyOf"]) == len(PrimitiveKind)


def test_tool_args_round_trip_and_validate() -> None:
    spec = h1_spec()
    args = spec.to_tool_args()
    validator = Validator(task_plan_json_schema())
    assert validator.errors(json.loads(json.dumps(args))) == []
    again = task_spec_from_tool_args(args, source="recorded")
    assert again.task_id == spec.task_id and again.source == "recorded"
    assert again.to_tool_args() == args
    for t in templates():  # every default instance fits the schema
        inst = ObjectInstance.of(
            "thing",
            t.kind,
            "Thing",
            ref="library:block" if t.needs_ref else None,
            **({"shape": "mesh"} if t.needs_ref else {}),
        )
        one = {**args, "objects": [inst.to_dict()]}
        assert validator.errors(json.loads(json.dumps(one))) == [], t.kind


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.update(extra=1),
        lambda a: a["objects"][0]["params"].update(width_mm=5000),
        lambda a: a["objects"][0].update(kind="sofa"),
        lambda a: a.update(title="x" * 81),
        lambda a: a["steps"][0]["params"].update(speed_mm_s=-1),
        lambda a: a["steps"][0].update(hold_after=True),  # hook_pull may not hold
        lambda a: a["steps"][0]["success"][0].update(quantity="speed"),
        lambda a: a["steps"][0]["constraints"].update(max_torque_nm=99.0),
    ],
)
def test_schema_rejects_bad_arguments(mutate: Any) -> None:
    args = copy.deepcopy(h1_spec().to_tool_args())
    mutate(args)
    assert Validator(task_plan_json_schema()).errors(json.loads(json.dumps(args)))


def test_parser_reports_every_problem_with_its_path() -> None:
    args = copy.deepcopy(h1_spec().to_tool_args())
    args["objects"][2]["params"]["mass_g"] = 99999
    args["steps"][1]["params"]["lift_mm"] = 999
    del args["steps"][3]["success"]
    with pytest.raises(TaskSpecError) as err:
        task_spec_from_tool_args(args)
    probs = err.value.problems
    assert any(p.startswith("objects[2]") and "mass_g" in p for p in probs)
    assert any(p.startswith("steps[1]") and "lift_mm" in p for p in probs)
    assert any(p.startswith("steps[3]") for p in probs)
    with pytest.raises(TaskSpecError):
        task_spec_from_tool_args("not an object")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- setups


def two_step_spec() -> TaskSpec:
    from studio_fixtures import two_press_task

    return two_press_task()


def test_setup_kinds_ids_and_validation() -> None:
    spec = two_step_spec()
    g, c = gripper(), cup()
    single = Setup.single(g, spec.step_ids)
    multi = Setup.multi_head(
        ((g, HeadMount(60.0)), (c, HeadMount(-60.0))), {"press_carton": 0, "press_flat": 1}
    )
    change = Setup.tool_change({"press_carton": g, "press_flat": c})
    ids = {single.setup_id, multi.setup_id, change.setup_id}
    assert len(ids) == 3 and all(len(i) == 64 for i in ids)
    assert multi.bracket is not None and change.changer == ToolChanger()
    assert change.changer.changeover_s == TOOL_CHANGE_S and change.tools == (g, c)
    assert Setup.single(g, spec.step_ids, label="Another name").setup_id == single.setup_id
    assert change.effector_for("press_flat").effector_id == c.effector_id
    for s in (single, multi, change):
        assert s.problems(spec) == []
        assert pickle.loads(pickle.dumps(s)).setup_id == s.setup_id
        json.dumps(s.to_dict(), allow_nan=False)
    with pytest.raises(ValueError, match="45 deg apart"):
        Setup.multi_head(((g, HeadMount(10.0)), (c, HeadMount(-10.0))), {"press_carton": 0})
    with pytest.raises(ValueError, match="distinct"):
        Setup(SetupKind.TOOL_CHANGE, (g, g), {"press_carton": 0, "press_flat": 1})
    with pytest.raises(ValueError, match="tool index"):
        Setup(SetupKind.SINGLE, (g,), {"press_carton": 3})
    with pytest.raises(ValueError, match="one tool"):
        Setup(SetupKind.SINGLE, (g, c), {"press_carton": 0})
    with pytest.raises(ValueError, match="angle_deg"):
        HeadMount(170.0)


def test_setup_problems_against_a_task() -> None:
    spec = two_step_spec()
    g, c = gripper(), cup()
    assert "no tool assigned" in Setup.single(g, ["press_carton"]).problems(spec)[0]
    assert (
        "never used"
        in Setup.multi_head(
            ((g, HeadMount(60)), (c, HeadMount(-60))), dict.fromkeys(spec.step_ids, 0)
        ).problems(spec)[0]
    )
    port = ObjectInstance.of("port", "port_plug", "Port")
    grab = Step.of(
        "grab", "grasp", "port", "plug", "Pick", [SuccessCriterion("lift_mm", at_least=1)], hold_after=True
    )
    plug = Step.of("plug_in", "insert", "port", "plug", "Insert", [SuccessCriterion("depth_mm", at_least=1)])
    held = TaskSpec("Plug", "", (port,), (grab, plug), source="test")
    assert "while 'port' is held" in Setup.tool_change({"grab": g, "plug_in": c}).problems(held)[0]
    latch = ObjectInstance.of("latch", "quarter_turn_latch", "Latch")
    turn = Step.of("turn", "twist", "latch", "head", "Turn", [SuccessCriterion("angle_deg", at_least=80)])
    poke = Step.of("poke", "push", "box", "part", "Push", [SuccessCriterion("force_n", at_least=1)])
    twist = TaskSpec(
        "Turn", "", (latch, ObjectInstance.of("box", "part", "Box")), (turn, poke), source="test"
    )
    off_axis = Setup.multi_head(
        ((g, HeadMount(60, lateral_mm=20)), (c, HeadMount(-60))), {"turn": 0, "poke": 1}
    )
    assert "off the roll axis" in off_axis.problems(twist)[0]
    with pytest.raises(ValueError):
        off_axis.segments(twist)


def test_segments_are_the_tool_change_hook() -> None:
    spec = h1_spec()
    g, c = gripper(), cup()
    by_step = {"open_lid": g, "pick_carton": g, "pick_flat": c, "close_lid": g}
    change = Setup.tool_change(by_step)
    segs = change.segments(spec)
    assert [s.step_ids for s in segs] == [("open_lid", "pick_carton"), ("pick_flat",), ("close_lid",)]
    assert [s.change for s in segs] == [ChangeKind.NONE, ChangeKind.TOOL_CHANGE, ChangeKind.TOOL_CHANGE]
    assert [s.change_s for s in segs] == [0.0, TOOL_CHANGE_S, TOOL_CHANGE_S]
    assert [s.tool for s in segs] == [0, 1, 0]
    multi = Setup.multi_head(
        ((g, HeadMount(60)), (c, HeadMount(-60))), {k: int(v is c) for k, v in by_step.items()}
    )
    assert [s.change for s in multi.segments(spec)][1:] == [ChangeKind.PRESENT, ChangeKind.PRESENT]
    assert [s.change_s for s in multi.segments(spec)][1:] == [multi.present_s] * 2
    single = Setup.single(g, spec.step_ids).segments(spec)
    assert len(single) == 1 and single[0].step_ids == spec.step_ids


def test_wrist_limits_are_one_shared_constant() -> None:
    d = STUDIO_WRIST.to_dict()
    assert d["roll_torque_nm"] == 15.0 and d["roll_speed_max_deg_s"] == 180.0
    with pytest.raises(ValueError):
        type(STUDIO_WRIST)(roll_torque_nm=0.0)


# --------------------------------------------------------------------------- results


def _outcome(step_id: str, ok: bool, fidelity: str = "rigid") -> StepOutcome:
    return StepOutcome(
        step_id=step_id,
        primitive="push",
        object="carton",
        tool=0,
        effector_id=SHA,
        success=ok,
        failure_mode=None if ok else "criterion_not_met",
        quantities={"force_n": 3.0},
        fidelity=fidelity,
        caveats=("approximation caveat",) if fidelity == "approximation" else (),
    )


def _run(setup: Setup, spec: TaskSpec, seed: int, oks: tuple[bool, bool], cycle: float) -> TaskResult:
    return TaskResult(
        task_id=spec.task_id,
        setup_id=setup.setup_id,
        setup_kind=setup.kind,
        condition_id="nominal",
        seed=seed,
        steps=(_outcome("press_carton", oks[0]), _outcome("press_flat", oks[1])),
        cycle_time_s=cycle,
        change_time_s=0.0,
        n_tool_changes=0,
        n_presentations=0,
        tool_mass_kg=(0.4,),
        traj_sha=SHA,
        fairness_sha=SHA,
    )


def test_step_outcome_rules() -> None:
    with pytest.raises(ValueError, match="exactly when"):
        StepOutcome("s", "push", "o", 0, SHA, True, "jam", {}, "rigid")
    with pytest.raises(ValueError, match="needs its caveat"):
        StepOutcome("s", "suction", "o", 0, SHA, True, None, {}, "approximation")
    with pytest.raises(ValueError, match="never succeed"):
        StepOutcome("s", "push", "o", 0, SHA, True, None, {}, "not_simulated", caveats=("x",))
    with pytest.raises(ValueError, match="unknown step failure"):
        StepOutcome("s", "push", "o", 0, SHA, False, "oops", {}, "rigid")
    rec = _outcome("s", True).to_dict()
    assert rec["success"] and rec["fidelity"] == "rigid"


def test_setup_results_aggregate_and_rank() -> None:
    spec = two_step_spec()
    a = Setup.single(gripper(), spec.step_ids)
    b = Setup.single(cup(), spec.step_ids)
    ra = SetupResult(
        a,
        spec.task_id,
        (_run(a, spec, 0, (True, True), 10.0), _run(a, spec, 1, (True, False), 12.0)),
        ComplexityMetrics(1, 2, 0.37, 0.37, 0),
        (CertificateRef("press_carton", SHA, "step", "runs/certs/a.json", 0.9),),
    )
    rb = SetupResult(
        b,
        spec.task_id,
        (_run(b, spec, 0, (True, True), 9.0), _run(b, spec, 1, (True, True), 9.5)),
        ComplexityMetrics(1, 1, 0.06, 0.06, 0),
    )
    assert ra.completion_rate == 0.5 and rb.completion_rate == 1.0
    assert ra.weakest_step == ("press_flat", 0.5)
    assert ra.cycle_time_s == 10.0  # median of completed runs only
    assert min([ra, rb], key=SetupResult.rank_key) is rb
    d = ra.to_dict()
    assert d["label"] == PRODUCT_MODE_LABEL and d["weakest_step"]["step_id"] == "press_flat"
    json.dumps(d, allow_nan=False)
    with pytest.raises(ValueError, match="this task and this setup"):
        SetupResult(b, spec.task_id, ra.runs, ComplexityMetrics(1, 1, 0.1, 0.1, 0))
    rec = ra.runs[0].to_record()
    assert rec["completed"] and rec["contract"] == CONTRACT_VERSION and len(rec["steps"]) == 2


# --------------------------------------------------------------------------- build pack


def _file(path: str, kind: str, fmt: str, **kw: Any) -> BuildPackFile:
    base = {"license": "Apache-2.0", "source": "generated", "source_ref": "studio-bracket/1"}
    base.update(kw)
    return BuildPackFile(path, SHA, kind, fmt, **base)  # type: ignore[arg-type]


def test_build_pack_manifest() -> None:
    files = (
        _file("setup.xml", "mjcf", "xml", source="studio"),
        _file("certificate.html", "certificate", "html", source="studio"),
        _file("bom.csv", "bom", "csv", source="studio"),
        _file("bracket.stl", "bracket", "stl", watertight=True),
        _file(
            "gripper/base.stl",
            "effector_cad",
            "stl",
            source="library",
            source_ref="lib:adaptive-2f",
            attribution="Our own procedural design",
            watertight=True,
        ),
    )
    man = BuildPackManifest(
        SHA,
        SHA,
        "Returns kitting setup",
        files,
        (BomLine("Serial-bus servo", 2, "https://example.org/servo"),),
        (BuildPackLink("Cup", "https://example.org/cup", "MIT", "Upstream files are linked, not copied"),),
        ("Suction is an approximation.",),
    )
    d = man.to_dict()
    validator = Validator(buildpack_manifest_json_schema())
    assert unsupported_keywords(buildpack_manifest_json_schema()) == []
    assert validator.errors(json.loads(json.dumps(d))) == []
    bad = copy.deepcopy(d)
    bad["files"][0]["license"] = "GPL-3.0-only"
    assert validator.errors(bad)
    with pytest.raises(ValueError, match="may not be redistributed"):
        _file("x.stl", "bracket", "stl", license="CC-BY-NC-4.0", watertight=True)
    with pytest.raises(ValueError, match="watertight"):
        _file("x.stl", "bracket", "stl")
    with pytest.raises(ValueError, match="attribution"):
        _file("x.stl", "effector_cad", "stl", source="library", watertight=True)
    with pytest.raises(ValueError, match="relative"):
        _file("/abs.stl", "bracket", "stl", watertight=True)
    with pytest.raises(ValueError, match="needs"):
        BuildPackManifest(SHA, SHA, "x", files[3:], ())
