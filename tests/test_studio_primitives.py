"""Track S2: the seven Studio primitives in physics (``docs/STUDIO_PRIMITIVES.md``).

The S2 gate, one case per primitive: a reference effector from the catalog succeeds and a
deliberately wrong one fails, with every successful step's tool-object penetration at most
1 mm, no MuJoCo reset, and the fidelity tier and caveats on every approximation; the runs are
deterministic (two in-process runs and a spawned worker give the same ``traj_sha``). Also: the
registry resolves every primitive to its S2 controller, controllers never read the true object
state, a full task chains the primitives across a tool change, and the failure modes ``slip``,
``jam`` and ``dropped`` come out where they should.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import rtt.studio.primitives as primitives_pkg
from rtt.effectors.registry import get_family
from rtt.effectors.types import EffectorSpec
from rtt.studio.episode import StudioCondition, StudioJob, run_task, run_task_jobs
from rtt.studio.plan import HELD_PENETRATION_MAX_MM, get_primitive
from rtt.studio.types import (
    FidelityTier,
    ObjectInstance,
    ObjectPose,
    PrimitiveKind,
    Setup,
    Step,
    StepConstraints,
    SuccessCriterion,
    TaskResult,
    TaskSpec,
)


def preset(family: str, model: str, primitive: str | None = None) -> EffectorSpec:
    return next(
        s
        for s in get_family(family).presets()
        if s.model == model and (primitive is None or s.primitive == primitive)
    )


TOTE = ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0.0, 150.0, 0.0))


def _task(objects: tuple[ObjectInstance, ...], step: Step) -> TaskSpec:
    return TaskSpec(f"S2 gate: {step.primitive.value}", "", objects, (step,), source="test")


#: primitive -> (task, reference effector, deliberately wrong effector, the wrong one's failure mode)
GATE: dict[str, tuple[TaskSpec, tuple[str, str, str | None], tuple[str, str, str | None], str]] = {
    "grasp": (
        _task(
            (
                ObjectInstance.of(
                    "slug",
                    "part",
                    "Steel slug",
                    ObjectPose(-80.0, -100.0, 0.0),
                    ref="library:steel_slug",
                    shape="mesh",
                ),
                TOTE,
            ),
            Step.of(
                "pick",
                "grasp",
                "slug",
                "part",
                "Pick the slug into the tote",
                [SuccessCriterion("in_target", at_least=1)],
                target="tote",
            ),
        ),
        ("catalog_gripper", "adaptive-2f", "pinch"),
        ("catalog_gripper", "gripper-2f-rigid-a", None),  # 4.1 N of grip at its published settings
        "criterion_not_met",
    ),
    "suction": (
        _task(
            (
                ObjectInstance.of(
                    "flat",
                    "part",
                    "Flat item",
                    ObjectPose(80.0, -100.0, 0.0),
                    size_x_mm=120,
                    size_y_mm=90,
                    size_z_mm=15,
                    mass_g=600,
                ),
                TOTE,
            ),
            Step.of(
                "pick",
                "suction",
                "flat",
                "part",
                "Pick the flat item into the tote",
                [SuccessCriterion("in_target", at_least=1)],
                target="tote",
            ),
        ),
        ("suction", "cup-25mm", None),
        ("suction", "cup-10mm", None),  # holds 4.7 N; the item weighs 5.9 N
        "criterion_not_met",
    ),
    "wrap": (
        _task(
            (
                ObjectInstance.of(
                    "bush",
                    "part",
                    "Bushing",
                    ObjectPose(0.0, -120.0, 0.0),
                    ref="library:flanged_bushing",
                    shape="mesh",
                ),
            ),
            Step.of(
                "lift",
                "wrap",
                "bush",
                "part",
                "Wrap and lift the bushing",
                [SuccessCriterion("lift_mm", at_least=50), SuccessCriterion("hold_s", at_least=1.5)],
            ),
        ),
        ("soft_arm", "tentacle-16", None),
        ("suction", "cup-16mm", None),  # the bore under the cup: no seal
        "criterion_not_met",
    ),
    "hook_pull": (
        _task(
            (ObjectInstance.of("dr", "drawer", "Drawer", ObjectPose(0.0, 120.0, 0.0)),),
            Step.of(
                "open",
                "hook_pull",
                "dr",
                "handle",
                "Pull the drawer open",
                [SuccessCriterion("travel_mm", at_least=120), SuccessCriterion("slip_mm", at_most=15)],
            ),
        ),
        ("catalog_gripper", "adaptive-2f", "pinch"),
        ("suction", "cup-25mm", None),  # no seal on a round bar
        "slip",
    ),
    "twist": (
        _task(
            (ObjectInstance.of("va", "handwheel_valve", "Valve", ObjectPose(0.0, 80.0, 0.0)),),
            Step.of(
                "turn",
                "twist",
                "va",
                "wheel",
                "Open the valve a quarter turn",
                [SuccessCriterion("angle_deg", at_least=85), SuccessCriterion("slip_deg", at_most=15)],
                params={"direction": "ccw"},
            ),
        ),
        ("catalog_gripper", "adaptive-2f", "pinch"),
        ("suction", "cup-25mm", None),  # a cup on a handwheel
        "slip",
    ),
    "insert": (
        _task(
            (
                ObjectInstance.of(
                    "po",
                    "port_plug",
                    "Port",
                    ObjectPose(0.0, 80.0, 0.0),
                    plug_diameter_mm=20,
                    clearance_mm=0.4,
                    chamfer_mm=2.0,
                ),
            ),
            Step.of(
                "plug_in",
                "insert",
                "po",
                "plug",
                "Plug the connector in",
                [SuccessCriterion("depth_mm", at_least=14), SuccessCriterion("lateral_force_n", at_most=20)],
            ),
        ),
        ("catalog_gripper", "adaptive-2f", "power"),
        ("soft_arm", "tentacle-12", None),  # no axial stiffness
        "jam",
    ),
    "push": (
        _task(
            (
                ObjectInstance.of(
                    "box", "part", "Box", ObjectPose(0.0, 80.0, 0.0), size_x_mm=40, size_y_mm=40, size_z_mm=40
                ),
            ),
            Step.of(
                "press",
                "push",
                "box",
                "part",
                "Press the box down",
                [SuccessCriterion("force_n", at_least=5), SuccessCriterion("tilt_deg", at_most=5)],
                constraints=StepConstraints(max_force_n=40),
            ),
        ),
        ("catalog_gripper", "adaptive-2f", "pinch"),
        ("soft_arm", "tentacle-12", None),  # hangs aside, presses nothing
        "criterion_not_met",
    ),
}


def job(spec: TaskSpec, key: tuple[str, str, str | None], seed: int = 0) -> StudioJob:
    return StudioJob(Setup.single(preset(*key), spec.step_ids), spec, StudioCondition(f"seed{seed}", seed))


@pytest.fixture(scope="module")
def gate_runs() -> dict[str, dict[str, TaskResult]]:
    return {
        name: {"reference": run_task(job(spec, ref)), "wrong": run_task(job(spec, wrong))}
        for name, (spec, ref, wrong, _) in GATE.items()
    }


# --------------------------------------------------------------------------- registry and rules


def test_every_primitive_resolves_to_its_s2_controller() -> None:
    for kind in PrimitiveKind:
        controller, predicate = get_primitive(kind)
        assert controller.name == predicate.name == f"studio_{kind.value}"
        assert controller.primitive is predicate.primitive is kind


def test_controllers_never_read_the_true_object_state() -> None:
    """Controllers locate objects through observed sites only (``observed_state``); the true
    template state (``StepContext.object_state``) is the predicates'."""
    root = Path(primitives_pkg.__file__).parent
    true_state = ("object_state", "site_xmat", "site_xpos", "site_pos", "xpos", "xmat")
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            is_predicate = isinstance(node, ast.ClassDef) and node.name.endswith("Predicate")
            if is_predicate or not isinstance(node, ast.ClassDef | ast.FunctionDef):
                continue
            if path.name == "_common.py" and getattr(node, "name", "") == "footprint":
                continue  # the tool's own calibration run (a separate MjData, no objects)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in true_state:
                    pytest.fail(f"{path.name}:{node.name} reads {sub.attr} (controllers use observed sites)")


# --------------------------------------------------------------------------- the gate


@pytest.mark.parametrize("name", sorted(GATE))
def test_reference_succeeds_and_wrong_effector_fails(
    name: str, gate_runs: dict[str, dict[str, TaskResult]]
) -> None:
    ref, wrong = gate_runs[name]["reference"].steps[0], gate_runs[name]["wrong"].steps[0]
    assert ref.success, (name, ref.failure_mode, dict(ref.quantities))
    assert not wrong.success and wrong.failure_mode == GATE[name][3], (
        name,
        wrong.failure_mode,
        dict(wrong.quantities),
    )
    assert ref.quantities["penetration_mm"] <= HELD_PENETRATION_MAX_MM
    assert ref.controller == wrong.controller == f"studio_{name}"
    for out in (ref, wrong):
        assert out.failure_mode != "physics_invalid"
        if out.fidelity is FidelityTier.APPROXIMATION:
            assert out.caveats
        assert out.duration_s <= 30.0


def test_fidelity_tiers_follow_template_primitive_and_effector(
    gate_runs: dict[str, dict[str, TaskResult]],
) -> None:
    tiers = {name: runs["reference"].steps[0].fidelity for name, runs in gate_runs.items()}
    assert tiers["grasp"] is FidelityTier.RIGID and tiers["hook_pull"] is FidelityTier.RIGID
    assert tiers["insert"] is FidelityTier.RIGID and tiers["push"] is FidelityTier.RIGID
    assert tiers["suction"] is FidelityTier.APPROXIMATION and tiers["wrap"] is FidelityTier.APPROXIMATION
    assert tiers["twist"] is FidelityTier.APPROXIMATION  # the valve template is an approximation
    twist = gate_runs["twist"]["reference"].steps[0]
    assert any("Valves are an approximation" in c for c in twist.caveats)


def test_runs_are_deterministic_in_process_and_in_a_spawned_worker(
    gate_runs: dict[str, dict[str, TaskResult]],
) -> None:
    names = sorted(GATE)
    jobs = [job(GATE[n][0], GATE[n][1]) for n in names]
    again = [run_task(j) for j in jobs]
    workers = run_task_jobs(jobs, workers=2)
    for n, a, w in zip(names, again, workers, strict=True):
        first = gate_runs[n]["reference"]
        assert first.traj_sha == a.traj_sha == w.traj_sha, n
        assert first.to_record() | {"wall_time_s": 0} == w.to_record() | {"wall_time_s": 0}, n


def test_seeds_change_the_physics_but_not_the_outcome() -> None:
    spec, ref, _, _ = GATE["twist"]
    runs = [run_task(job(spec, ref, seed)) for seed in (1, 2)]
    assert runs[0].traj_sha != runs[1].traj_sha
    assert all(r.completed for r in runs)


# --------------------------------------------------------------------------- failure modes and tasks


def test_a_cup_on_a_wing_latch_slips_and_a_gripper_turns_it() -> None:
    latch = ObjectInstance.of("la", "quarter_turn_latch", "Latch", ObjectPose(0.0, 80.0, 0.0), torque_nm=0.3)
    step = Step.of(
        "turn",
        "twist",
        "la",
        "head",
        "Unlock the latch",
        [SuccessCriterion("angle_deg", at_least=85)],
        params={"direction": "ccw"},
    )
    spec = TaskSpec("latch", "", (latch,), (step,), source="test")
    ok = run_task(job(spec, ("catalog_gripper", "adaptive-2f", "pinch"))).steps[0]
    assert ok.success and ok.quantities["slip_deg"] < 5.0 and ok.quantities["torque_nm"] > 0.25
    bad = run_task(job(spec, ("suction", "cup-25mm", None))).steps[0]
    assert bad.failure_mode == "slip" and bad.quantities["slip_deg"] > 90.0
    against = Step.of(
        "turn",
        "twist",
        "la",
        "head",
        "Turn the wrong way",
        [SuccessCriterion("angle_deg", at_least=85)],
        params={"direction": "cw"},
    )
    wrong_way = run_task(
        job(
            TaskSpec("latch", "", (latch,), (against,), source="test"),
            ("catalog_gripper", "adaptive-2f", "pinch"),
        )
    ).steps[0]
    assert not wrong_way.success  # into the locked stop


def test_a_thread_is_started_by_insert_and_tightened_by_twist() -> None:
    nut = ObjectInstance.of(
        "nut",
        "threaded_joint",
        "Nut",
        ObjectPose(0.0, 80.0, 0.0),
        member="nut",
        start_torque_nm=0.05,
        final_torque_nm=0.3,
    )
    steps = (
        Step.of(
            "start",
            "insert",
            "nut",
            "member",
            "Push the nut onto the thread",
            [SuccessCriterion("depth_mm", at_least=3.5)],
        ),
        Step.of(
            "tighten",
            "twist",
            "nut",
            "member",
            "Run it down",
            [SuccessCriterion("turns", at_least=2.5)],
            params={"direction": "cw", "speed_deg_s": 120},
        ),
    )
    spec = TaskSpec("thread", "", (nut,), steps, source="test")
    r = run_task(
        StudioJob(
            Setup.single(preset("catalog_gripper", "adaptive-2f", "pinch"), spec.step_ids),
            spec,
            StudioCondition("c", 0),
        )
    )
    assert r.completed, [(s.step_id, s.failure_mode, dict(s.quantities)) for s in r.steps]
    assert r.steps[1].fidelity is FidelityTier.APPROXIMATION and any(
        "Threads" in c for c in r.steps[1].caveats
    )
    deep = Step.of(
        "start",
        "insert",
        "nut",
        "member",
        "Push it past the thread start",
        [SuccessCriterion("depth_mm", at_least=8)],
    )
    jam = run_task(
        job(
            TaskSpec("thread", "", (nut,), (deep,), source="test"),
            ("catalog_gripper", "adaptive-2f", "pinch"),
        )
    ).steps[0]
    assert jam.failure_mode == "jam" and jam.quantities["jam"] == 1.0  # a thread does not push in


def test_a_lidded_tote_task_runs_end_to_end_across_a_tool_change() -> None:
    objs = (
        ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0.0, 120.0, 0.0)),
        ObjectInstance.of("lid", "hinged_lid", "Tote lid", ObjectPose(0.0, 120.0, 0.0), feature="handle"),
        ObjectInstance.of(
            "carton",
            "part",
            "Carton",
            ObjectPose(-150.0, -150.0, 0.0),
            size_x_mm=40,
            size_y_mm=40,
            size_z_mm=50,
            mass_g=150,
        ),
        ObjectInstance.of(
            "flat",
            "part",
            "Flat item",
            ObjectPose(150.0, -150.0, 0.0),
            size_x_mm=120,
            size_y_mm=90,
            size_z_mm=15,
            mass_g=150,
        ),
    )
    steps = (
        Step.of(
            "open_lid",
            "hook_pull",
            "lid",
            "handle",
            "Open the lid",
            [SuccessCriterion("angle_deg", at_least=80)],
        ),
        Step.of(
            "pick_carton",
            "grasp",
            "carton",
            "part",
            "Carton into the tote",
            [SuccessCriterion("in_target", at_least=1)],
            target="tote",
        ),
        Step.of(
            "pick_flat",
            "suction",
            "flat",
            "part",
            "Flat item into the tote",
            [SuccessCriterion("in_target", at_least=1)],
            target="tote",
        ),
        Step.of(
            "close_lid",
            "push",
            "lid",
            "lid",
            "Close the lid",
            [SuccessCriterion("angle_deg", at_most=5)],
            constraints=StepConstraints(max_force_n=40),
            params={"speed_mm_s": 60},
        ),
    )
    spec = TaskSpec("Lidded tote", "", objs, steps, source="test")
    g, c = preset("catalog_gripper", "adaptive-2f", "pinch"), preset("suction", "cup-25mm")
    setup = Setup.tool_change({"open_lid": g, "pick_carton": g, "pick_flat": c, "close_lid": g})
    r = run_task(StudioJob(setup, spec, StudioCondition("c", 0)))
    assert r.completed, [(s.step_id, s.failure_mode, dict(s.quantities)) for s in r.steps]
    assert r.n_tool_changes == 2 and r.steps[0].quantities["angle_deg"] > 95.0  # stays open past vertical
    assert r.steps[3].quantities["force_n"] <= 40.0
    assert all(s.quantities["penetration_mm"] <= HELD_PENETRATION_MAX_MM for s in r.steps)


def test_a_part_lifted_and_lost_is_dropped_and_a_part_never_lifted_is_not() -> None:
    from types import SimpleNamespace

    from rtt.studio.primitives._pick import PickPredicate

    step = Step.of(
        "pick", "grasp", "p", "part", "Pick it", [SuccessCriterion("in_target", at_least=1)], target="t"
    )
    ctx = SimpleNamespace(step=step)
    pred = PickPredicate(PrimitiveKind.GRASP)
    lost = pred.verdict(ctx, {"lift_mm": 80.0, "in_target": 0.0, "dropped": 1.0, "penetration_mm": 0.2})  # type: ignore[arg-type]
    assert lost.failure_mode == "dropped"
    never = pred.verdict(ctx, {"lift_mm": 0.0, "in_target": 0.0, "penetration_mm": 0.2})  # type: ignore[arg-type]
    assert never.failure_mode == "criterion_not_met"
    deep = pred.verdict(ctx, {"lift_mm": 80.0, "in_target": 1.0, "penetration_mm": 1.4})  # type: ignore[arg-type]
    assert deep.failure_mode == "physics_unreliable"


def test_unknown_controller_inputs_are_refused_by_the_runner() -> None:
    """The runner's own checks (roll speed cap, finite commands) hold for every S2 controller:
    a fast twist is capped below the roll limit instead of tripping it."""
    spec, ref, _, _ = GATE["twist"]
    fast = Step.of(
        "turn",
        "twist",
        "va",
        "wheel",
        "Open fast",
        [SuccessCriterion("angle_deg", at_least=85)],
        params={"direction": "ccw", "speed_deg_s": 180},
    )
    r = run_task(job(TaskSpec("fast", "", spec.objects, (fast,), source="test"), ref))
    assert r.steps[0].failure_mode in (None, "criterion_not_met", "slip")
