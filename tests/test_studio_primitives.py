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
        (("catalog_gripper", "adaptive-2f", "pinch"), None),  # straight fingers: no hook
        (("soft_arm", "tentacle-12", None), 300),  # the lid inset in the rim: no gap to hook into
    ],
)
def test_a_lip_without_a_hook_or_without_room_under_it_stays_shut(
    key: tuple[str, str, str | None], tote_width_mm: int | None
) -> None:
    out = run_task(job(_lip_task(280, tote_width_mm), key)).steps[0]
    assert not out.success and out.failure_mode == "slip", (out.failure_mode, dict(out.quantities))
    assert out.quantities["angle_deg"] < 5.0


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
