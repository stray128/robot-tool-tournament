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

import numpy as np
import pytest

import rtt.studio.primitives as primitives_pkg
from rtt.effectors.registry import get_family
from rtt.effectors.types import EffectorSpec
from rtt.studio.episode import StudioCondition, StudioJob, run_task, run_task_jobs
from rtt.studio.plan import HELD_PENETRATION_MAX_MM, PENETRATION_TRANSIENT_MS, get_primitive
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


def gate_conditions(spec: TaskSpec) -> list[StudioCondition]:
    """The S2 gate's 14 conditions (``docs/STUDIO_PRIMITIVES.md`` section 6): 10 seeds (the v1
    jitter) and 4 placement errors of 2 mm (+x, -x, +y, -y) on every object, hidden from the
    controller."""
    out = [StudioCondition(f"seed{s}", s) for s in range(10)]
    for i, (dx, dy) in enumerate(((2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0))):
        out.append(StudioCondition(f"place{i}", 10 + i, {o.name: (dx, dy) for o in spec.objects}))
    return out


@pytest.mark.parametrize("name", sorted(GATE))
def test_the_s2_gate_table_holds_on_its_14_conditions(name: str) -> None:
    """The S2 gate table: every reference 13 or 14 of 14 (a successful step within 1 mm of
    penetration), every wrong effector 0 of 14 with its row's failure mode, no MuJoCo reset."""
    spec, ref, wrong, mode = GATE[name]
    conds = gate_conditions(spec)
    jobs = [
        StudioJob(Setup.single(preset(*key), spec.step_ids), spec, c) for key in (ref, wrong) for c in conds
    ]
    runs = [r.steps[0] for r in run_task_jobs(jobs, workers=4)]
    refs, wrongs = runs[: len(conds)], runs[len(conds) :]
    assert sum(o.success for o in refs) >= 13, [(o.failure_mode, dict(o.quantities)) for o in refs]
    assert all(not o.success and o.failure_mode == mode for o in wrongs), [o.failure_mode for o in wrongs]
    assert all(o.quantities["penetration_mm"] <= HELD_PENETRATION_MAX_MM for o in refs if o.success)
    assert all(o.failure_mode != "physics_invalid" for o in runs)


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
    # a flagged impact transient above 1 mm may pass (docs/STUDIO.md 3.3), a held penetration may not
    assert all(s.failure_mode != "physics_unreliable" for s in r.steps)
    assert all(s.quantities["penetration_ms"] <= PENETRATION_TRANSIENT_MS for s in r.steps)


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


# --------------------------------------------------------------------------- S2b: the hero-task limits

#: the equipment-cabinet connector's port (8 mm plug, 0.2 mm clearance, 1 mm chamfer, 15 mm deep)
SMALL_PORT = {
    "chamfer_mm": 1,
    "clearance_mm": 0.2,
    "depth_mm": 15,
    "insert_force_n": 5,
    "latch": False,
    "plug_length_mm": 30,
}


def _plug_task(diameter_mm: float) -> TaskSpec:
    port = ObjectInstance.of(
        "port", "port_plug", "Port", ObjectPose(200.0, 0.0, 0.0), plug_diameter_mm=diameter_mm, **SMALL_PORT
    )
    step = Step.of(
        "plug_in",
        "insert",
        "port",
        "plug",
        "Plug the connector in",
        [SuccessCriterion("depth_mm", at_least=15.0), SuccessCriterion("force_n", at_most=20)],
        constraints=StepConstraints(max_force_n=20),
        params={"speed_mm_s": 10, "standoff_mm": 5},
    )
    return TaskSpec(f"small plug {diameter_mm:g} mm", "", (port,), (step,), source="test")


@pytest.mark.parametrize(
    ("diameter_mm", "key"),
    [
        (8, ("catalog_gripper", "gripper-2f-finray", None)),  # pads mounted off the TCP
        (12, ("catalog_gripper", "gripper-2f-rigid-a", None)),
        (20, ("catalog_gripper", "adaptive-2f", "power")),
    ],
)
def test_a_small_plug_is_gripped_low_started_and_pressed_home(
    diameter_mm: float, key: tuple[str, str, str | None]
) -> None:
    """A slim plug is gripped below its tip-over height, started past the chamfer by the grip,
    let go and pressed home by the fist, within the step's force limit."""
    out = run_task(job(_plug_task(diameter_mm), key)).steps[0]
    assert out.success, (diameter_mm, key, out.failure_mode, dict(out.quantities))
    assert out.quantities["depth_mm"] == pytest.approx(15.0)
    assert out.quantities["force_n"] <= 20.0
    assert out.quantities["penetration_mm"] <= HELD_PENETRATION_MAX_MM


def test_curling_fingers_cannot_carry_a_slim_plug_upright() -> None:
    """An honest limit: the adaptive gripper's fingers curl into a V around an 8 mm pin and
    squeeze it out of the grasp; the plug never reaches the socket."""
    out = run_task(job(_plug_task(8), ("catalog_gripper", "adaptive-2f", "power"))).steps[0]
    assert not out.success
    assert out.quantities["depth_mm"] == 0.0


def test_pads_mounted_off_the_tcp_grip_a_plug_between_them() -> None:
    from rtt.studio.primitives._common import footprint
    from rtt.studio.scene import build_studio_scene

    spec = _plug_task(8)
    centres = {}
    for model in ("gripper-2f-finray", "gripper-2f-rigid-a"):
        scene = build_studio_scene(Setup.single(preset("catalog_gripper", model), spec.step_ids), spec, 0)
        centres[model] = footprint(scene, 0.0).grip_centre(0.004)
    assert centres["gripper-2f-finray"] > 0.010  # its pads span 0.6 to 26 mm along the tool's x
    assert centres["gripper-2f-rigid-a"] == 0.0  # its pads cover the plug on the axis


def _lip_task(lid_depth_mm: int, tote_width_mm: int | None) -> TaskSpec:
    lid = ObjectInstance.of(
        "lid",
        "hinged_lid",
        "Tote lid",
        ObjectPose(0.0, 0.0, 0.0),
        depth_mm=lid_depth_mm,
        width_mm=lid_depth_mm + 100,
        thickness_mm=5,
        feature="lip",
        feature_mm=15,
        max_angle_deg=110,
        open_torque_nm=0.3,
        damping_nms=0.05,
    )
    objects: tuple[ObjectInstance, ...] = (lid,)
    if tote_width_mm is not None:
        tote = ObjectInstance.of(
            "tote",
            "bin",
            "Tote",
            ObjectPose(0.0, 0.0, 0.0),
            fixed=True,
            width_mm=tote_width_mm,
            height_mm=120,
            wall_mm=5,
            length_mm=400,
        )
        objects = (tote, lid)
    step = Step.of(
        "open_lid",
        "hook_pull",
        "lid",
        "handle",
        "Hook the lip and open the lid",
        [SuccessCriterion("angle_deg", at_least=90, at_most=110), SuccessCriterion("force_n", at_most=20)],
        constraints=StepConstraints(max_force_n=20),
        params={"approach": "top", "speed_mm_s": 50},
    )
    return TaskSpec("lid lip", "", objects, (step,), source="test")


@pytest.mark.parametrize("tote_width_mm", [None, 260])
def test_a_soft_arm_hooks_under_a_lid_lip_and_opens_it(tote_width_mm: int | None) -> None:
    """The soft arm's horizontal arm goes under the lip (a free lid, or one overhanging its
    tote) and carries it along the arc to its stop."""
    out = run_task(job(_lip_task(280, tote_width_mm), ("soft_arm", "tentacle-12", None))).steps[0]
    assert out.success, (tote_width_mm, out.failure_mode, dict(out.quantities))
    assert out.quantities["slip_mm"] <= 10.0
    assert out.quantities["penetration_mm"] <= HELD_PENETRATION_MAX_MM


@pytest.mark.parametrize(
    ("key", "tote_width_mm"),
    [
        (("soft_arm", "tentacle-12", None), 300),  # the lid inset in the rim: no gap to hook into
    ],
)
def test_a_lip_without_a_hook_or_without_room_under_it_stays_shut(
    key: tuple[str, str, str | None], tote_width_mm: int | None
) -> None:
    out = run_task(job(_lip_task(280, tote_width_mm), key)).steps[0]
    assert not out.success and out.failure_mode == "slip", (out.failure_mode, dict(out.quantities))
    assert out.quantities["angle_deg"] < 5.0


#: The frozen hero plans (Task Engineer gate S4e); H1 is the returns tote with its lipped lid.
HERO_MANIFEST = (
    Path(__file__).resolve().parents[1] / "src/rtt/studio/engineer/recorded/gate-2026-10-18-s4e/manifest.json"
)


def _h1_task() -> TaskSpec:
    import json

    from rtt.studio.types import task_spec_from_tool_args

    cases = json.loads(HERO_MANIFEST.read_text(encoding="utf-8"))["cases"]
    plan = next(c for c in cases if c["case_id"] == "h1_tote")["plan"]["plan"]
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
    return task_spec_from_tool_args({k: plan[k] for k in keys}, source="recorded")


def test_a_two_finger_gripper_tilts_and_pinches_a_lid_lip() -> None:
    """Wrist pitch: a parallel gripper whose opening fits the lip tilts its wrist (the pitch axis
    along the hinge, the tool pointing at the hinge), comes in from the front under the overhang,
    pinches the lip and carries it along the arc, yielding along the tool axis. The frozen H1 lid
    (6 mm thick, a 20 mm lip over its tote, at least 95 deg within 20 N) opens with the 110 mm servo
    gripper, which has no sideways hook, on all 8 conditions of the hero grid, deterministically
    (two in-process runs and a spawned worker give the same trajectory)."""
    from rtt.studio.runner import condition_grid
    from rtt.studio.search import default_candidates, sub_task

    full = _h1_task()
    spec = sub_task(full, ("open_lid",))
    servo = next(c.spec for c in default_candidates() if c.spec.model == "gripper-2f-servo-110")
    conds = condition_grid(full, 8)
    jobs = [StudioJob(Setup.single(servo, spec.step_ids), spec, c) for c in conds]
    runs = run_task_jobs(jobs, workers=4)
    # deterministic: two in-process runs give the spawned worker's trajectory
    assert run_task(jobs[0]).traj_sha == run_task(jobs[0]).traj_sha == runs[0].traj_sha
    for c, r in zip(conds, runs, strict=True):
        out = r.steps[0]
        assert out.success, (c.condition_id, out.failure_mode, dict(out.quantities))
        assert out.quantities["angle_deg"] >= 95.0 and out.quantities["force_n"] <= 20.0, c.condition_id
        assert out.quantities["penetration_mm"] <= HELD_PENETRATION_MAX_MM, c.condition_id


def test_a_door_pulled_by_its_handle_stays_within_1_mm_of_penetration() -> None:
    """The equipment-cabinet door with nothing else in its footprint and clear of the wrist's home
    pose (a door under the home pose has the tool start inside it)."""
    door = ObjectInstance.of(
        "door",
        "hinged_lid",
        "Cabinet door",
        ObjectPose(0.0, 150.0, 0.0),
        orientation="door",
        width_mm=400,
        depth_mm=300,
        thickness_mm=10,
        max_angle_deg=110,
        open_torque_nm=0.3,
        damping_nms=0.05,
        feature="handle",
        feature_mm=20,
    )
    step = Step.of(
        "open_door",
        "hook_pull",
        "door",
        "handle",
        "Hook the handle and pull the door open",
        [SuccessCriterion("angle_deg", at_least=80.0), SuccessCriterion("force_n", at_most=30.0)],
        constraints=StepConstraints(max_force_n=30.0),
        params={"speed_mm_s": 80, "approach": "front"},
    )
    spec = TaskSpec("door", "", (door,), (step,), source="test")
    out = run_task(job(spec, ("catalog_gripper", "adaptive-2f", "pinch"))).steps[0]
    assert out.success, (out.failure_mode, dict(out.quantities))
    assert out.quantities["penetration_mm"] <= HELD_PENETRATION_MAX_MM


def _candidate(name: str) -> EffectorSpec:
    from rtt.studio.search import default_candidates

    return next(c.spec for c in default_candidates() if c.name.startswith(name))


def _twist_task(obj: ObjectInstance, handle: str, lo: float, hi: float) -> TaskSpec:
    step = Step.of(
        "turn",
        "twist",
        obj.name,
        handle,
        "Turn it open",
        [SuccessCriterion("angle_deg", at_least=lo, at_most=hi)],
        constraints=StepConstraints(max_force_n=10.0, max_torque_nm=5.0),
        params={"direction": "ccw", "speed_deg_s": 60},
    )
    return TaskSpec("twist under a force limit", "", (obj,), (step,), source="test")


def test_a_wing_latch_is_turned_within_the_step_force_limit() -> None:
    """The equipment-cabinet latch (0.8 N m detent on a 25 mm wing, 10 N limit): a linkage
    gripper's tips swing down as it closes, so it is lowered for its lowest closing point and does
    not land on the panel; its wing grip turns the latch open."""
    latch = ObjectInstance.of(
        "latch",
        "quarter_turn_latch",
        "Latch",
        ObjectPose(-200.0, 0.0, 0.0),
        detent=True,
        head="wing",
        head_diameter_mm=25,
        head_height_mm=12,
        torque_nm=0.8,
        turn_deg=90,
    )
    spec = _twist_task(latch, "head", 85.0, 95.0)
    eff = _candidate("Two-finger adaptive linkage gripper")
    out = run_task(StudioJob(Setup.single(eff, spec.step_ids), spec, StudioCondition("seed0", 0))).steps[0]
    assert out.success, (out.failure_mode, dict(out.quantities))
    assert out.quantities["force_n"] <= 10.0


def test_a_spoked_valve_is_left_without_lifting_a_spoke() -> None:
    """After the turn, fingers in the spoke gaps turn back by their free play before they rise,
    so a hooked finger does not lift the wheel on the way out (10 N limit)."""
    valve = ObjectInstance.of(
        "valve",
        "handwheel_valve",
        "Valve",
        ObjectPose(200.0, 0.0, 0.0),
        rim_diameter_mm=120,
        rim_section_mm=12,
        spokes=3,
        torque_nm=3,
        travel_deg=90,
    )
    spec = _twist_task(valve, "wheel", 80.0, 100.0)
    eff = _candidate("Designed three-finger sliding hook gripper")
    out = run_task(StudioJob(Setup.single(eff, spec.step_ids), spec, StudioCondition("seed0", 0))).steps[0]
    assert out.success, (out.failure_mode, dict(out.quantities))
    assert out.quantities["force_n"] <= 10.0


def test_a_closing_linkage_finger_reaches_below_its_open_tip() -> None:
    from rtt.studio.scene import build_studio_scene

    spec = _plug_task(8)
    scene = build_studio_scene(
        Setup.single(_candidate("Two-finger adaptive linkage gripper"), spec.step_ids), spec, 0
    )
    from rtt.studio.primitives._common import footprint

    assert footprint(scene, 1.0).tip_z < footprint(scene, 0.0).tip_z - 0.010


# --------------------------------------------------------------------------- S2c: tool stacks

#: one tool per build path: an adaptive gripper (honours ``mount_z``), vendored catalog models
#: (they level their housing at ``MOUNT_Z`` whatever ``mount_z`` says), a cup, a soft arm
STACK_TOOLS = [
    ("catalog_gripper", "adaptive-2f", "pinch"),
    ("catalog_gripper", "gripper-2f-finray", None),
    ("catalog_hand", "hand-5f-22dof", "power"),
    ("suction", "cup-25mm", None),
    ("soft_arm", "tentacle-12", None),
]


def _with_spare_step(spec: TaskSpec) -> TaskSpec:
    """The task plus a later push on a far-away part, so a tool-change setup has its two tools."""
    spare = ObjectInstance.of(
        "spare", "part", "Spare", ObjectPose(-260.0, -160.0, 0.0), size_x_mm=30, size_y_mm=30, size_z_mm=30
    )
    second = Step.of(
        "spare_push", "push", "spare", "part", "Push the spare", [SuccessCriterion("force_n", at_least=0)]
    )
    return TaskSpec(spec.title, "", (*spec.objects, spare), (*spec.steps, second), source="test")


def _mounted(eff: EffectorSpec, spec: TaskSpec, stack_mm: float | None) -> Setup:
    if stack_mm is None:
        return Setup.single(eff, spec.step_ids)
    from rtt.studio.types import ToolChanger

    other = (
        preset("soft_arm", "tentacle-12") if eff.family.value == "suction" else preset("suction", "cup-25mm")
    )
    return Setup.tool_change({spec.steps[0].id: eff, "spare_push": other}, ToolChanger(stack_mm=stack_mm))


@pytest.mark.parametrize("key", STACK_TOOLS, ids=lambda k: k[1])
def test_a_tool_on_a_changer_hangs_below_it_and_the_planner_knows_where_its_tip_is(
    key: tuple[str, str, str | None],
) -> None:
    """S2c: every family's tool hangs the changer's stack height lower than on the flange, and the
    tip the controllers plan with (``Tool.tip``) moves with it (a vendored model used to stay put
    inside a 40 mm changer while the planner moved its tip 28 mm down)."""
    from rtt.sim.task import default_task_v2
    from rtt.studio.plan import StepContext
    from rtt.studio.primitives._common import footprint, tool_of
    from rtt.studio.scene import build_studio_scene

    task = default_task_v2()
    spec = _with_spare_step(GATE["twist"][0])
    eff = preset(*key)
    lowest, planned = {}, {}
    for stack in (None, 5.0, 40.0):
        scene = build_studio_scene(_mounted(eff, spec, stack), spec, 0, task=task)
        step = spec.steps[0]
        ctx = StepContext(
            step=step,
            task_spec=spec,
            scene=scene,
            obj=scene.objects[step.object],
            target=None,
            head=scene.active,
            wrist_start=np.zeros(5),
            task=task,
            seed=0,
            aim_offsets={},
        )
        lowest[stack] = footprint(scene, 0.0).tip_z  # measured: the head's lowest point
        planned[stack] = float(tool_of(ctx).tip[2])
    for stack in (5.0, 40.0):
        drop = -stack / 1000.0  # the flange's mount plane is MOUNT_Z, the changer's MOUNT_Z - stack
        # measured by a settling calibration run: a hand's fingers settle within a few hundredths
        assert lowest[stack] - lowest[None] == pytest.approx(drop, abs=0.0005), (key, stack)
        assert planned[stack] - planned[None] == pytest.approx(drop, abs=1e-6), (key, stack)


#: the S2 gate references (they honour ``mount_z``) and vendored tools that pass the same gate
#: tasks on the flange (they did not on a changer before S2c)
CHANGER_CASES = [(name, GATE[name][1]) for name in sorted(GATE)] + [
    ("grasp", ("catalog_gripper", "gripper-2f-finray", None)),
    ("wrap", ("catalog_gripper", "gripper-2f-finray", None)),
    ("insert", ("catalog_gripper", "gripper-2f-rigid-a", None)),
    ("push", ("catalog_hand", "hand-5f-22dof", "power")),
]


@pytest.mark.parametrize(("name", "key"), CHANGER_CASES, ids=lambda v: v if isinstance(v, str) else v[1])
def test_a_gate_task_on_a_tool_changer_ends_as_on_the_flange(
    name: str, key: tuple[str, str, str | None]
) -> None:
    spec = _with_spare_step(GATE[name][0])
    eff = preset(*key)
    jobs = [
        StudioJob(_mounted(eff, spec, stack), spec, StudioCondition("seed0", 0))
        for stack in (None, 25.0, 40.0)
    ]
    flange, *changers = (r.steps[0] for r in run_task_jobs(jobs, workers=3))
    assert flange.success, (name, key, flange.failure_mode, dict(flange.quantities))
    for out in changers:
        assert out.success == flange.success and out.failure_mode == flange.failure_mode, (
            name,
            key,
            out.failure_mode,
            dict(out.quantities),
        )
        assert out.quantities["penetration_mm"] <= HELD_PENETRATION_MAX_MM
        for q in ("angle_deg", "travel_mm", "depth_mm", "lift_mm"):
            if q in flange.quantities:
                assert out.quantities[q] == pytest.approx(flange.quantities[q], abs=3.0), (name, key, q)


def test_a_thread_twist_is_deterministic_in_process_and_in_a_spawned_worker() -> None:
    """S2c: the thread's friction anchor keeps its state in the actuator's control (MjData), so a
    run does not depend on what ran before it in the process."""
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
            "start", "insert", "nut", "member", "Push it on", [SuccessCriterion("depth_mm", at_least=3.5)]
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
    j = StudioJob(
        Setup.single(preset("catalog_gripper", "adaptive-2f", "pinch"), spec.step_ids),
        spec,
        StudioCondition("c", 0),
    )
    first, second = run_task(j), run_task(j)
    workers = run_task_jobs([j, j], workers=2)  # two spawned processes
    assert first.traj_sha == second.traj_sha == workers[0].traj_sha == workers[1].traj_sha
    assert first.completed


# --------------------------------------------------------------------------- S2d: every goal of a step


def _tighten(final_nm: float, torque_lo: float, torque_hi: float) -> TaskSpec:
    nut = ObjectInstance.of(
        "nut",
        "threaded_joint",
        "Nut",
        ObjectPose(0.0, 80.0, 0.0),
        member="nut",
        start_torque_nm=0.05,
        final_torque_nm=final_nm,
    )
    steps = (
        Step.of(
            "start", "insert", "nut", "member", "Push it on", [SuccessCriterion("depth_mm", at_least=3.5)]
        ),
        Step.of(
            "tighten",
            "twist",
            "nut",
            "member",
            "Tighten it a quarter turn and to a torque",
            [
                SuccessCriterion("turns", at_least=0.25),
                SuccessCriterion("torque_nm", at_least=torque_lo, at_most=torque_hi),
                SuccessCriterion("slip_deg", at_most=15),
            ],
            params={"direction": "cw", "speed_deg_s": 120},
        ),
    )
    return TaskSpec("tighten to turns and torque", "", (nut,), steps, source="test")


def test_a_twist_with_turns_and_torque_goals_turns_until_both_are_met() -> None:
    """S2d: a tightening step that asks for a quarter turn and a torque on the thread's ramp runs
    past the quarter turn, up the ramp, until the torque is reached too; it used to stop at the
    first goal met (0.26 turns, under half the torque). A torque the grip cannot hold ends it as
    slip after it has turned well past the quarter turn, not as an early stop."""
    key = ("catalog_gripper", "adaptive-2f", "pinch")
    ok = run_task(job(_tighten(0.3, 0.2, 1.0), key)).steps[1]
    assert ok.success, (ok.failure_mode, dict(ok.quantities))
    assert ok.quantities["turns"] > 1.0 and 0.2 <= ok.quantities["torque_nm"] <= 1.0
    over = run_task(job(_tighten(0.6, 0.5, 1.2), key)).steps[1]  # the pinch grip holds about 0.35 N m
    assert over.failure_mode == "slip" and over.quantities["turns"] > 1.0, dict(over.quantities)


def test_a_push_meets_its_force_goal_and_its_observed_travel_goal() -> None:
    """S2d: a press asked for more than the default 10 N presses on to it, and a drawer pushed
    shut ends when the observed drawer meets its travel criterion, not when the command arrives
    (the wrist servo lags behind it under the drawer's friction)."""
    box = ObjectInstance.of(
        "box", "part", "Box", ObjectPose(0.0, 80.0, 0.0), size_x_mm=40, size_y_mm=40, size_z_mm=40
    )
    press = Step.of(
        "press",
        "push",
        "box",
        "part",
        "Press the box down hard",
        [SuccessCriterion("force_n", at_least=20), SuccessCriterion("tilt_deg", at_most=5)],
        constraints=StepConstraints(max_force_n=40),
    )
    key = ("catalog_gripper", "adaptive-2f", "pinch")
    out = run_task(job(_task((box,), press), key)).steps[0]
    assert out.success and 20.0 <= out.quantities["force_n"] <= 40.0, (out.failure_mode, dict(out.quantities))
    drawer = ObjectInstance.of("dr", "drawer", "Drawer", ObjectPose(0.0, 120.0, 0.0))
    steps = (
        Step.of(
            "open",
            "hook_pull",
            "dr",
            "handle",
            "Pull the drawer open",
            [SuccessCriterion("travel_mm", at_least=120), SuccessCriterion("slip_mm", at_most=15)],
        ),
        Step.of(
            "close",
            "push",
            "dr",
            "drawer",
            "Push the drawer shut",
            [SuccessCriterion("travel_mm", at_most=5)],
            constraints=StepConstraints(max_force_n=40),
        ),
    )
    r = run_task(job(TaskSpec("drawer shut", "", (drawer,), steps, source="test"), key))
    assert r.completed, [(s.step_id, s.failure_mode, dict(s.quantities)) for s in r.steps]


def test_a_push_that_loses_travel_to_its_fist_goes_on_until_the_drawer_is_shut() -> None:
    """The drawer of the S2d test on seed 3. The adaptive gripper's fist (both tendons at
    their limit, the tips pressed together) has no lateral stiffness: on contact it swings and uses
    up most of the push's 18 mm ``over``, so the push let go with the drawer 37.8 mm open. A push to
    an observed-state criterion now goes on past ``over`` while the state is unmet and the force is
    under the limit (:data:`~rtt.studio.primitives.push.PUSH_ON_MAX` at most, within its budget)."""
    from rtt.studio.primitives.push import PUSH_ON_MAX

    drawer = ObjectInstance.of("dr", "drawer", "Drawer", ObjectPose(0.0, 120.0, 0.0))
    steps = (
        Step.of(
            "open",
            "hook_pull",
            "dr",
            "handle",
            "Pull the drawer open",
            [SuccessCriterion("travel_mm", at_least=120), SuccessCriterion("slip_mm", at_most=15)],
        ),
        Step.of(
            "close",
            "push",
            "dr",
            "drawer",
            "Push the drawer shut",
            [SuccessCriterion("travel_mm", at_most=5)],
            constraints=StepConstraints(max_force_n=40),
        ),
    )
    spec = TaskSpec("drawer shut", "", (drawer,), steps, source="test")
    r = run_task(job(spec, ("catalog_gripper", "adaptive-2f", "pinch"), seed=3))
    assert r.completed, [(s.step_id, s.failure_mode, dict(s.quantities)) for s in r.steps]
    q = r.steps[1].quantities
    assert q["travel_mm"] <= 5.0 and q["force_n"] <= 40.0 and q["penetration_mm"] <= 1.0, dict(q)
    assert PUSH_ON_MAX == pytest.approx(0.060)


def test_a_held_part_stays_held_through_a_step_that_is_not_tried() -> None:
    """A step that is not tried holds still for 0.2 s. Its tool used to open,
    dropping what the previous step holds (``hold_after``). A frame lifts a panel at (0, -300 mm)
    and holds it; the next step starts its clamp, 462.5 mm out, which is refused as out of reach.
    The panel stays up: the idle tool keeps holding what the previous step holds."""
    from rtt.studio.runner import run_segments
    from rtt.studio.search import default_candidates

    lift = Step.of(
        "lift",
        "suction",
        "panel",
        "panel",
        "Lift the panel and hold it",
        [SuccessCriterion("lift_mm", at_least=25.0)],
        params={"lift_mm": 60.0, "hold_s": 0.5},
        hold_after=True,
    )
    start = Step.of(
        "start", "insert", "panel", "member", "Start the clamp", [SuccessCriterion("depth_mm", at_least=3.5)]
    )
    panel = ObjectInstance.of("panel", "panel_rail", "Solar panel on its rails", ObjectPose(0.0, -300.0, 0.0))
    spec = TaskSpec("held", "", (panel,), (lift, start), source="test")
    frame = next(c.spec for c in default_candidates() if c.spec.model == "vacuum-frame-2x3-600")
    setup = Setup.single(frame, spec.step_ids)
    r, state = run_segments(setup, spec, StudioCondition("nominal", 0))
    held, refused = r.steps
    assert held.success and held.quantities["lift_mm"] >= 25.0, (held.failure_mode, dict(held.quantities))
    assert refused.failure_mode == "out_of_reach" and refused.duration_s <= 0.25
    free = next(name for name in state.qpos if name.endswith("free"))
    rest_z = 0.0575  # m: the panel's free joint at rest on its rails
    assert state.qpos[free][2] >= rest_z + 0.025, state.qpos[free]  # still up after the 0.2 s


def test_the_reach_helpers_live_in_common_and_keep_their_margin() -> None:
    """One ``out_of_reach`` for every primitive, in :mod:`~rtt.studio.primitives._common`:
    a pose is out of reach past the palm's +-350 mm travel less a margin (5 mm by default; the lid
    pinch and hook keep 12 mm)."""
    from rtt.studio.primitives import _common, insert, twist
    from rtt.studio.primitives.hook_pull import PINCH_REACH_MARGIN

    assert twist.reach_problem is _common.reach_problem and insert.reach_problem is _common.reach_problem
    assert _common.REACH_MARGIN == pytest.approx(0.005)
    pose = np.array([0.0, 0.341, 0.4, 0.0, 0.0, 0.0])
    assert _common.reach_problem(pose) is None
    far = _common.reach_problem(pose, margin=PINCH_REACH_MARGIN)
    assert far is not None and far["reach_y_mm"] == pytest.approx(341.0) and far["palm_limit_mm"] == 350.0


def test_a_retreat_stops_at_the_palms_travel_and_is_whole_inside_it() -> None:
    """A hook pull's retreat is never refused: it is cut where the palm meets its travel less the
    margin, and left whole (the same float, so an in-reach run is unchanged) everywhere else."""
    from rtt.studio.primitives._common import REACH_MARGIN
    from rtt.studio.primitives.hook_pull import RETRACT_OUT, _retreat_within

    out = np.array([-1.0, 0.0, 0.0])
    near = np.array([-0.2, 0.1, 0.3, 0.0, 0.0, 0.0])
    assert _retreat_within(near, out, RETRACT_OUT, REACH_MARGIN) == RETRACT_OUT
    edge = np.array([-0.3152, 0.0123, 0.3, 0.0, 0.0, 0.0])  # the hotel door's pull end
    assert _retreat_within(edge, out, RETRACT_OUT, REACH_MARGIN) == pytest.approx(0.0298)
    slant = np.array([0.6, -0.8, 0.0])
    corner = np.array([0.33, -0.33, 0.3, 0.0, 0.0, 0.0])
    assert _retreat_within(corner, slant, RETRACT_OUT, REACH_MARGIN) == pytest.approx(0.015 / 0.8)
    beyond = np.array([0.35, 0.0, 0.3, 0.0, 0.0, 0.0])
    assert _retreat_within(beyond, np.array([1.0, 0.0, 0.0]), RETRACT_OUT, REACH_MARGIN) == 0.0


# --------------------------------------------------------------------------- S2e: a push's approach


def test_a_lid_opened_past_vertical_is_pushed_shut_within_the_step_force_limit() -> None:
    """S2e: a lid with a bar handle, opened to 110 deg, pushed shut (angle_deg <= 3, 40 N limit).
    The move into the push used to be planned off the lid's face, so the tool ran into the bar
    standing off it: 58 N (the rigid-pad gripper), 57 N (the servo gripper) and 104 N (the linkage
    gripper) before the push started. The push now stands off the bar and the whole tool, moves in
    guarded and meets the face at the touch speed, and each tool closes the lid under the limit."""
    from rtt.studio.runner import run_segments
    from rtt.studio.search import sub_task

    lid = ObjectInstance.of("lid", "hinged_lid", "Lid", ObjectPose(0.0, 120.0, 0.0), feature="handle")
    steps = (
        Step.of(
            "open", "hook_pull", "lid", "handle", "Open the lid", [SuccessCriterion("angle_deg", at_least=60)]
        ),
        Step.of(
            "close",
            "push",
            "lid",
            "lid",
            "Push the lid shut",
            [SuccessCriterion("angle_deg", at_most=3)],
            constraints=StepConstraints(max_force_n=40),
        ),
    )
    spec = TaskSpec("lid shut from past vertical", "", (lid,), steps, source="test")
    cond = StudioCondition("seed0", 0)
    opened, state = run_segments(
        Setup.single(preset("catalog_gripper", "adaptive-2f", "pinch"), ("open",)),
        sub_task(spec, ("open",)),
        cond,
    )
    assert opened.steps[0].quantities["angle_deg"] > 100.0  # past vertical: gravity holds it open
    tools = (
        preset("catalog_gripper", "gripper-2f-rigid-a"),
        _candidate("Industrial servo parallel gripper"),
        _candidate("Two-finger linkage gripper"),
    )
    for tool in tools:
        r, _ = run_segments(Setup.single(tool, ("close",)), sub_task(spec, ("close",)), cond, initial=state)
        out = r.steps[0]
        assert out.success, (tool.model, out.failure_mode, dict(out.quantities))
        assert out.quantities["force_n"] <= 40.0 and out.quantities["angle_deg"] <= 3.0


# --------------------------------------------------------------------------- S2f: set down, close gently


def _gap_below_mm(model: object, data: object, body: int, skip: np.ndarray) -> float:
    """How far (mm) a part's lowest point is above what it rests on or will land on: the highest
    top of the colliding geoms (not the part's, not ``skip``) whose boxes overlap the part's in x
    and y and do not reach more than 2 mm above its bottom (true state, the test's own reading)."""
    m, d = model, data

    def box(g: int) -> tuple[np.ndarray, np.ndarray]:
        c = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g][:3]  # type: ignore[attr-defined]
        e = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_aabb[g][3:]  # type: ignore[attr-defined]
        return c - e, c + e

    own = np.flatnonzero(m.geom_bodyid == body)  # type: ignore[attr-defined]
    lo = np.min([box(int(g))[0] for g in own], axis=0)
    hi = np.max([box(int(g))[1] for g in own], axis=0)
    live = np.flatnonzero((m.geom_contype != 0) | (m.geom_conaffinity != 0))  # type: ignore[attr-defined]
    top = -np.inf
    for g in live:
        if g in own or g in skip:
            continue
        a, b = box(int(g))
        if a[0] < hi[0] and b[0] > lo[0] and a[1] < hi[1] and b[1] > lo[1] and b[2] <= lo[2] + 0.002:
            top = max(top, float(b[2]))
    return 1000.0 * (float(lo[2]) - top)


#: step id -> the part's gap above its support at the release (mm), filled by :class:`_ReleaseWatch`
RELEASES: dict[str, float] = {}


def _release_watch(kind: PrimitiveKind) -> object:
    from dataclasses import dataclass

    from rtt.studio.primitives._common import schedule, tool_geoms
    from rtt.studio.primitives._pick import PickPredicate, pick_body

    @dataclass(frozen=True)
    class _ReleaseWatch(PickPredicate):
        """The pick predicate, also reading the part's gap above its support when it is let go."""

        def observe(self, t, ctx, model, data, acc) -> None:  # type: ignore[no-untyped-def]
            super().observe(t, ctx, model, data, acc)
            rel = schedule(ctx).released
            if not np.isnan(rel) and t >= rel and ctx.step.id not in RELEASES:
                RELEASES[ctx.step.id] = _gap_below_mm(model, data, pick_body(ctx), tool_geoms(ctx.scene))

    return _ReleaseWatch(kind, name=f"studio_{kind.value}")


@pytest.mark.parametrize("name", ["grasp", "suction", "wrap"])
def test_a_placed_part_is_released_within_20_mm_of_its_support(name: str) -> None:
    """Every acquire primitive sets its part down before it lets go: at the release the part's
    lowest point is at most RELEASE_HEIGHT_MAX_MM above what it rests on (the bin floor for grasp
    and suction, the table it was lifted from for wrap). The 600 g suction item used to be let go
    145 mm above the tote floor: the cup's lip, pressing the item away from the cup while the
    vacuum holds it, read as "landed" to the old guard, which never lowered it."""
    from rtt.studio.plan import override_primitive
    from rtt.studio.primitives._common import RELEASE_HEIGHT_MAX_MM
    from rtt.studio.primitives._pick import PickController

    spec, ref, _, _ = GATE[name]
    kind = PrimitiveKind(name)
    RELEASES.clear()
    with override_primitive(PickController(kind, name=f"studio_{name}"), _release_watch(kind)):  # type: ignore[arg-type]
        out = run_task(job(spec, ref)).steps[0]
    gap = RELEASES[spec.steps[0].id]
    assert -1.0 <= gap <= RELEASE_HEIGHT_MAX_MM, (name, gap)
    assert out.success, (name, out.failure_mode, dict(out.quantities))
    assert out.quantities["penetration_mm"] <= HELD_PENETRATION_MAX_MM, (name, dict(out.quantities))


def test_a_lid_pushed_shut_is_not_dropped() -> None:
    """A push that closes a lid open past vertical used to push it over to 80 deg and let it fall
    (the frozen H1 tote lid met its stop at 340 to 400 deg/s, with a 6 to 10 ms penetration
    transient on the tote's rim). A two-finger gripper now pinches the lid's lip, which overhangs
    the tote, carries it down its arc and lets go only once it is shut, so it meets its stop under
    30 deg/s (the push's own ``lid_release_deg`` and ``lid_stop_dps``, sampled every physics step)
    and the step carries no fall caveat; on all 8 conditions of the hero grid, within 20 N."""
    from rtt.studio.primitives.push import LID_STOP_DPS_MAX
    from rtt.studio.runner import condition_grid
    from rtt.studio.search import default_candidates, sub_task

    full = _h1_task()
    spec = sub_task(full, ("open_lid", "close_lid"))
    servo = next(c.spec for c in default_candidates() if c.spec.model == "gripper-2f-servo-110")
    jobs = [StudioJob(Setup.single(servo, spec.step_ids), spec, c) for c in condition_grid(full, 8)]
    for r in run_task_jobs(jobs, workers=4):
        assert r.completed, [(s.step_id, s.failure_mode, dict(s.quantities)) for s in r.steps]
        close = r.steps[1]
        q = close.quantities
        assert q["lid_release_deg"] <= 1.0 and q["lid_stop_dps"] < LID_STOP_DPS_MAX, dict(q)
        assert not [c for c in close.caveats if "fell shut" in c], close.caveats
        assert q["force_n"] <= 20.0 and q["angle_deg"] <= 5.0, dict(q)
        assert q["penetration_mm"] <= HELD_PENETRATION_MAX_MM, dict(q)


def _s2e_lid(
    y_mm: float,
    *,
    depth_mm: float = 200.0,
    box: bool = False,
    feature: str = "handle",
    feature_mm: float = 15.0,
) -> TaskSpec:
    """A lid to push shut (at most 3 deg, 40 N); ``box``: a default bin under it, flush with its
    edge (the S2e lid with ``feature="handle"`` and no box)."""
    lid = ObjectInstance.of(
        "lid",
        "hinged_lid",
        "Lid",
        ObjectPose(0.0, y_mm, 0.0),
        depth_mm=depth_mm,
        feature=feature,
        feature_mm=feature_mm,
    )
    objs = (ObjectInstance.of("bin", "bin", "Bin", ObjectPose(0.0, y_mm, 0.0)), lid) if box else (lid,)
    close = Step.of(
        "close",
        "push",
        "lid",
        "lid",
        "Push the lid shut",
        [SuccessCriterion("angle_deg", at_most=3)],
        constraints=StepConstraints(max_force_n=40),
    )
    return TaskSpec("lid shut", "", objs, (close,), source="test")


def _close(spec: TaskSpec, tool: EffectorSpec) -> tuple[object, list[str]]:
    """Push the lid of ``spec`` shut with ``tool`` from its 110 deg stop: the step's outcome and the
    push plan's notes."""
    import math

    from rtt.sim.task import default_task_v2
    from rtt.studio.episode import WorldState
    from rtt.studio.primitives.push import PushController
    from rtt.studio.runner import run_segments
    from rtt.studio.scene import build_studio_scene

    setup = Setup.single(tool, spec.step_ids)
    scene = build_studio_scene(setup, spec, 0, task=default_task_v2())
    hinge = scene.objects["lid"].handles.joints["hinge"]
    state = WorldState({hinge: (math.radians(110.0),)}, 0.0)
    notes: list[str] = []
    plan = PushController.plan

    def spy(self: PushController, ctx: object, model: object, data: object) -> object:
        out = plan(self, ctx, model, data)  # type: ignore[arg-type]
        notes.append(out.note)
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PushController, "plan", spy)
        r, _ = run_segments(setup, spec, StudioCondition("seed0", 0), initial=state)
    return r.steps[0], notes


def test_a_lid_that_falls_shut_is_flagged_on_its_step() -> None:
    """A lid the tool cannot hold (a cup pushes it over from behind) falls onto its
    stop. That is not a failure (a lid with a catch snaps shut), but the step says so: its
    ``lid_release_deg`` (where it fell from) and ``lid_stop_dps`` (sampled every physics step: a
    fall crosses its last 2 deg in about 5 ms) and a plain caveat on the outcome."""
    from rtt.studio.primitives.push import LID_STOP_DPS_MAX

    out, notes = _close(_s2e_lid(120.0), preset("suction", "cup-25mm"))
    q = out.quantities  # type: ignore[attr-defined]
    assert out.success, (out.failure_mode, dict(q))  # type: ignore[attr-defined]
    assert q["lid_release_deg"] > 80.0 and q["lid_stop_dps"] > LID_STOP_DPS_MAX, dict(q)
    fell = [c for c in out.caveats if "The lid fell shut from" in c]  # type: ignore[attr-defined]
    assert fell and "deg/s" in fell[0], out.caveats  # type: ignore[attr-defined]
    assert "pushed over, not carried shut: the tool cannot hold it" in notes[-1], notes


def test_a_carry_the_palm_cannot_follow_is_not_taken() -> None:
    """At the lid's stop the palm sits the tool's length beyond its free edge, so a
    deep lid at the front of the table took it past its 350 mm travel mid-carry (a 360 mm lid at
    y = -80 mm: the palm stopped and the turn back reopened the lid to 31 deg). The carry is checked
    along its whole path, in the controller and in the search's tilted poses alike: such a lid is
    pushed over (and falls, flagged), and the search sees no tilted pose for it; the H1 lid, whose
    carry stays in reach, is carried, and the search checks that arc."""
    import math

    import mujoco

    from rtt.sim.task import default_task_v2
    from rtt.studio.scene import build_studio_scene
    from rtt.studio.search import default_candidates, sub_task, tilted_poses

    servo = next(c.spec for c in default_candidates() if c.spec.model == "gripper-2f-servo-110")
    deep = _s2e_lid(-80.0, depth_mm=360.0, feature="lip")
    out, notes = _close(deep, servo)
    assert out.success and out.quantities["angle_deg"] <= 3.0, dict(out.quantities)  # type: ignore[attr-defined]
    assert "not carried shut: the carry would take the palm" in notes[-1], notes
    assert [c for c in out.caveats if "The lid fell shut from" in c], out.caveats  # type: ignore[attr-defined]
    task = default_task_v2()
    for spec, carried in ((deep, False), (sub_task(_h1_task(), ("open_lid", "close_lid")), True)):
        setup = Setup.single(servo, spec.step_ids)
        scene = build_studio_scene(setup, spec, 0, task=task)
        data = mujoco.MjData(scene.model)
        mujoco.mj_kinematics(scene.model, data)
        step = next(st for st in spec.steps if st.primitive is PrimitiveKind.PUSH)
        poses = tilted_poses(setup, step, 0, scene, data, task, scene.active.tcp_wrist)
        assert bool(poses) is carried, spec.title
        if carried:  # from the lid's stop (110 deg: pitch -20) to shut (pitch 90)
            pitches = [math.degrees(p) for _, _, _, p in poses]
            assert min(pitches) == pytest.approx(-20.0, abs=0.5) and max(pitches) == pytest.approx(90.0)


def test_a_lid_flush_with_its_box_is_let_go_above_the_wall_and_flagged() -> None:
    """A lid flush with its box leaves no room under its free edge, so the finger
    under the edge used to be carried onto the box's wall (12 to 37 N on it) before the lid was let
    go. The hold is now sized from what lies under the lid: nothing of the edge clears the wall here
    and the bar handle is too close to the face for these fingers, so the lid is carried down to
    just above where a finger would meet the wall, let go there and flagged when it falls the rest
    (the plan says so); the finger never presses on the wall."""
    out, notes = _close(_s2e_lid(0.0, box=True), preset("catalog_gripper", "adaptive-2f", "pinch"))
    q = out.quantities  # type: ignore[attr-defined]
    assert out.success and q["angle_deg"] <= 3.0, (out.failure_mode, dict(q))  # type: ignore[attr-defined]
    assert "where a finger would meet what lies under it" in notes[-1], notes
    assert 0.0 < q["lid_release_deg"] <= 15.0, dict(q)
    assert [c for c in out.caveats if "The lid fell shut from" in c], out.caveats  # type: ignore[attr-defined]
    assert q["force_n"] <= 10.0 and q["penetration_ms"] <= PENETRATION_TRANSIENT_MS, dict(q)


def test_a_part_stopped_on_a_bin_rim_is_not_dropped() -> None:
    """The set-down stops when the wrist feels a support take the load, which a
    tool landing on a bin's rim also does. A 110 x 80 mm box into a 130 x 100 mm bin stopped 96 mm
    above the floor and was let go there (it fell 98 mm at 1.3 m/s, scored a success). Before it
    opens, the controller now compares where the part is (its observed site) with where it was to
    be set down: more than RELEASE_HEIGHT_MAX_MM above it, it is not let go, and the step fails
    with a plain reason."""
    from rtt.studio.primitives._common import RELEASE_HEIGHT_MAX_MM
    from rtt.studio.search import default_candidates

    box = ObjectInstance.of(
        "box",
        "part",
        "Box",
        ObjectPose(-80.0, -100.0, 0.0),
        size_x_mm=110,
        size_y_mm=80,
        size_z_mm=40,
        mass_g=300,
    )
    tight = ObjectInstance.of(
        "tote", "bin", "Tote", ObjectPose(0.0, 150.0, 0.0), length_mm=130, width_mm=100, height_mm=100
    )
    step = Step.of(
        "pick",
        "grasp",
        "box",
        "part",
        "Box into the tight bin",
        [SuccessCriterion("in_target", at_least=1)],
        target="tote",
        params={"lift_mm": 150, "hold_s": 1.0},
    )
    spec = TaskSpec("tight bin", "", (box, tight), (step,), source="test")
    servo = next(c.spec for c in default_candidates() if c.spec.model == "gripper-2f-servo-110")
    out = run_task(StudioJob(Setup.single(servo, spec.step_ids), spec, StudioCondition("seed0", 0))).steps[0]
    assert not out.success and out.failure_mode == "criterion_not_met", (
        out.failure_mode,
        dict(out.quantities),
    )
    assert out.quantities["held_high_mm"] > RELEASE_HEIGHT_MAX_MM
    assert [c for c in out.caveats if "it was not let go, since releasing it would drop it" in c], out.caveats
    assert out.quantities["penetration_ms"] <= PENETRATION_TRANSIENT_MS


def test_a_flush_lid_with_room_under_its_bar_is_carried_by_the_bar() -> None:
    """Where the fingers fit between a bar handle and the face (here a 50 mm
    clearance), the gripper pinches the bar and rides above the lid, so a lid flush with its box
    is carried all the way to its stop: no fall, no caveat."""
    from rtt.studio.primitives.push import LID_STOP_DPS_MAX

    spec = _s2e_lid(0.0, box=True, feature_mm=50.0)
    out, notes = _close(spec, preset("catalog_gripper", "adaptive-2f", "pinch"))
    q = out.quantities  # type: ignore[attr-defined]
    assert out.success and q["angle_deg"] <= 3.0, (out.failure_mode, dict(q))  # type: ignore[attr-defined]
    assert "pinch the bar handle" in notes[-1] and "carry the lid to its stop" in notes[-1], notes
    assert q["lid_release_deg"] <= 1.0 and q["lid_stop_dps"] < LID_STOP_DPS_MAX, dict(q)
    assert not [c for c in out.caveats if "fell shut" in c], out.caveats  # type: ignore[attr-defined]


def test_a_heavy_part_on_a_cup_is_released_resting() -> None:
    """A guard that holds the descent back for a moment lets it go on once the
    load is off again. Ending the set-down on the first trip let a 4 kg item on a 40 mm cup go
    about 9 mm up (it fell 9.6 mm at 0.36 m/s); it is let go resting on the tote floor."""
    from rtt.studio.plan import override_primitive
    from rtt.studio.primitives._pick import PickController
    from rtt.studio.search import default_candidates

    flat = ObjectInstance.of(
        "flat",
        "part",
        "Heavy flat item",
        ObjectPose(80.0, -100.0, 0.0),
        size_x_mm=150,
        size_y_mm=120,
        size_z_mm=20,
        mass_g=4000,
    )
    step = Step.of(
        "pick",
        "suction",
        "flat",
        "part",
        "Heavy item into the tote",
        [SuccessCriterion("in_target", at_least=1)],
        target="tote",
        params={"lift_mm": 150, "hold_s": 1.0},
    )
    spec = TaskSpec("heavy cup", "", (flat, TOTE), (step,), source="test")
    cup = next(c.spec for c in default_candidates() if c.spec.model == "cup-40mm")
    kind = PrimitiveKind.SUCTION
    RELEASES.clear()
    with override_primitive(PickController(kind, name="studio_suction"), _release_watch(kind)):  # type: ignore[arg-type]
        out = run_task(StudioJob(Setup.single(cup, spec.step_ids), spec, StudioCondition("seed0", 0))).steps[
            0
        ]
    assert out.success, (out.failure_mode, dict(out.quantities))
    assert -1.0 <= RELEASES["pick"] <= 1.0, RELEASES
    assert out.quantities["penetration_mm"] <= HELD_PENETRATION_MAX_MM
