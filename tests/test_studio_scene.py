"""The Studio scene and runner (``rtt-effectors/2``, ``docs/STUDIO.md``): physics-side contract tests.

* the continuous wrist roll: unlimited, torque-limited, the same in every scene;
* single setups reproduce the v1 effector scene's tool geometry and mass;
* a multi-tool head with two existing effectors (a two-finger adaptive gripper and a 16 mm cup):
  both heads, the idle head's collisions and mass, the presented head on the roll axis, the
  bracket unchanged whichever head is active, the active head's v1 primitive and criterion;
* tool-change stacks and the mass swap; the contract checks on families and templates;
* smoke runs over two dummy steps (tool change and multi-tool head), every S1 reference
  controller once, ``not_simulated`` steps, carried world state;
* determinism (in-process twice and in a spawned worker) and the fairness assertions.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
import pytest

from rtt.effectors.registry import get_family
from rtt.effectors.scene import build_effector_scene
from rtt.effectors.types import ContractError, InvalidEffector
from rtt.sim.task import default_task_v2
from rtt.studio.episode import (
    HOME_HEIGHT,
    StudioCondition,
    StudioJob,
    apply_world_state,
    capture_world_state,
    run_task,
    run_task_jobs,
)
from rtt.studio.fairness import assert_fair, fairness_problems, fairness_sha, wrist_signature
from rtt.studio.plan import PrimitiveNotSimulated, V1SceneView, get_primitive
from rtt.studio.scene import FLANGE_BODY, ROLL_JOINT, STUDIO_PREFIX, build_part_for, build_studio_scene
from rtt.studio.templates import TemplateUnavailable, get_template_builder, override_template_builder
from rtt.studio.types import (
    PRESENT_S,
    STUDIO_WRIST,
    TOOL_CHANGE_S,
    FidelityTier,
    HeadMount,
    ObjectInstance,
    ObjectPose,
    Setup,
    Step,
    SuccessCriterion,
    TaskSpec,
    ToolChanger,
)
from studio_fixtures import cup, gripper, toy_task, two_press_task

STEPS = ("press_carton", "press_flat")


@pytest.fixture(scope="module")
def task() -> Any:
    return default_task_v2()


@pytest.fixture(scope="module")
def spec() -> TaskSpec:
    return two_press_task()


def setups() -> dict[str, Setup]:
    g, c = gripper(), cup()
    return {
        "single": Setup.single(g, STEPS),
        "multi_head": Setup.multi_head(
            ((g, HeadMount(60.0)), (c, HeadMount(-60.0))), {STEPS[0]: 0, STEPS[1]: 1}
        ),
        "tool_change": Setup.tool_change({STEPS[0]: g, STEPS[1]: c}),
    }


def _job(setup: Setup, spec: TaskSpec, seed: int = 0, cid: str = "nominal") -> StudioJob:
    return StudioJob(setup, spec, StudioCondition(cid, seed), controllers="reference")


# --------------------------------------------------------------------------- the wrist roll


def test_wrist_roll_is_continuous_torque_limited_and_shared(spec: TaskSpec, task: Any) -> None:
    sigs = []
    for setup in setups().values():
        for tool in range(len(setup.tools)):
            scene = build_studio_scene(setup, spec, tool, task=task)
            m = scene.model
            j, a = m.joint(ROLL_JOINT), m.actuator(ROLL_JOINT)
            assert int(m.jnt_limited[j.id]) == 0 and int(j.type[0]) == mujoco.mjtJoint.mjJNT_HINGE
            assert int(m.actuator_ctrllimited[a.id]) == 0
            assert list(a.forcerange) == [-STUDIO_WRIST.roll_torque_nm, STUDIO_WRIST.roll_torque_nm]
            assert list(scene.ids.wrist_actuators) == [0, 1, 2, 3, 4, 5]  # x, y, z, yaw, roll, pitch
            sigs.append(wrist_signature(m))
            assert fairness_sha(scene) == fairness_sha(build_studio_scene(setup, spec, tool, task=task))
    assert all(s == sigs[0] for s in sigs), "every candidate must get the same wrist"


def test_roll_winds_several_turns_and_saturates_at_its_torque(spec: TaskSpec, task: Any) -> None:
    scene = build_studio_scene(setups()["single"], spec, 0, task=task)
    m, ids = scene.model, scene.ids
    d = mujoco.MjData(m)
    d.qpos[ids.palm_qpos] = [0.0, 0.0, HOME_HEIGHT, 0.0]
    mujoco.mj_forward(m, d)
    ctrl = np.zeros(m.nu)
    ctrl[:4] = d.qpos[ids.palm_qpos]
    ctrl[ids.effector_actuators] = scene.active.build.ctrl_open
    turns, rate = 3.0, math.radians(STUDIO_WRIST.roll_speed_max_deg_s)
    t_end = 2 * math.pi * turns / rate
    for k in range(int((t_end + 1.0) / 0.002)):
        ctrl[4] = min(k * 0.002 * rate, 2 * math.pi * turns)
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d, nstep=2)
    assert d.qpos[ids.roll_qpos] == pytest.approx(2 * math.pi * turns, abs=0.02)  # no joint limit
    # an opposing load larger than the limit: the servo saturates and the roll is driven back
    d.qfrc_applied[ids.roll_dof] = -(STUDIO_WRIST.roll_torque_nm + 5.0)
    for _ in range(200):
        mujoco.mj_step(m, d, nstep=2)
    assert abs(d.actuator_force[4]) == pytest.approx(STUDIO_WRIST.roll_torque_nm, rel=1e-6)
    assert d.qpos[ids.roll_qpos] < 2 * math.pi * turns - 0.1


# --------------------------------------------------------------------------- scenes


def _rel_poses(model: mujoco.MjModel, names: list[str], palm: str = "palm") -> np.ndarray:
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    p = model.body(palm).id
    r = d.xmat[p].reshape(3, 3)
    return np.array([r.T @ (d.xpos[model.body(n).id] - d.xpos[p]) for n in names])


def test_single_setup_reproduces_the_v1_tool_geometry_and_mass(spec: TaskSpec, task: Any) -> None:
    g = gripper()
    scene = build_studio_scene(setups()["single"], spec, 0, task=task)
    part = build_part_for(setups()["single"], spec, 0)
    v1 = build_effector_scene(get_family(g.family), g, part, task)
    bodies = list(v1.build.bodies)
    assert np.allclose(_rel_poses(scene.model, bodies), _rel_poses(v1.model, bodies), atol=1e-12)
    assert scene.active.build.mass == pytest.approx(v1.build.mass, rel=1e-12)
    assert scene.active.build.reach == v1.build.reach
    assert scene.active.prefix == "" and scene.active.tcp_wrist == pytest.approx(v1.build.tcp)


def test_multi_head_scene_with_two_existing_effectors(spec: TaskSpec, task: Any) -> None:
    setup = setups()["multi_head"]
    scenes = [build_studio_scene(setup, spec, k, task=task) for k in (0, 1)]
    for k, scene in enumerate(scenes):
        m, ids = scene.model, scene.ids
        assert [h.spec.model for h in scene.heads] == ["adaptive-2f", "cup-16mm"]
        assert [h.active for h in scene.heads] == [k == 0, k == 1]
        active, idle = scene.heads[k], scene.heads[1 - k]
        assert active.prefix == "" and idle.prefix == f"eff_h{1 - k}_"
        # the presented head points straight down on the roll axis; the idle one is tilted away
        assert active.axis_wrist == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)
        assert active.tcp_wrist[:2] == pytest.approx((0.0, 0.0), abs=1e-9)
        assert idle.axis_wrist[2] > 0.4 and idle.tcp_wrist[2] > active.tcp_wrist[2] + 0.1
        # the idle head is simulated: its collision geoms and its mass are in the scene
        assert len(ids.idle_geoms) > 0 and len(ids.idle_actuators) == len(idle.build.actuators)
        assert np.all((m.geom_contype[ids.idle_geoms] != 0) | (m.geom_conaffinity[ids.idle_geoms] != 0))
        assert idle.build.mass is not None and idle.build.mass > 0
        stack = scene.stack_mass_kg
        assert stack > 0.05  # bracket, hub, arms and two adapters
        assert scene.tool_mass_kg == pytest.approx(active.build.mass + idle.build.mass + stack, rel=1e-9)
        assert len(scene.ids.effector_actuators) == len(active.build.actuators)
        assert m.nu == 6 + sum(len(h.build.actuators) for h in scene.heads)
    # which head is active changes the names and the presentation, never the bracket
    a, b = scenes
    assert a.tool_mass_kg == pytest.approx(b.tool_mass_kg, rel=1e-12)
    assert (a.model.nbody, a.model.ngeom, a.model.nu) == (b.model.nbody, b.model.ngeom, b.model.nu)
    names_a = [n for n in (a.model.body(i).name for i in range(a.model.nbody)) if n.startswith("eff_")]
    names_b = [n for n in (b.model.body(i).name for i in range(b.model.nbody)) if n.startswith("eff_")]

    def canon(n: str) -> str:
        return n.removeprefix("eff_h0_").removeprefix("eff_h1_")

    assert [canon(n) for n in names_a] == [canon(n) for n in names_b]
    assert np.allclose(
        _rel_poses(a.model, names_a, FLANGE_BODY), _rel_poses(b.model, names_b, FLANGE_BODY), atol=1e-12
    )


def test_the_active_head_keeps_its_family_names_for_v1_primitives(spec: TaskSpec, task: Any) -> None:
    from rtt.studio.plan import StepContext

    for k, model_key in ((0, "adaptive-2f"), (1, "cup-16mm")):
        scene = build_studio_scene(setups()["multi_head"], spec, k, task=task)
        head = scene.active
        impl = get_family(head.spec.family)
        assert head.build.actuators == tuple(
            scene.model.actuator(i).name for i in scene.ids.effector_actuators
        )
        ctx = StepContext(
            step=spec.steps[k],
            task_spec=spec,
            scene=scene,
            obj=scene.objects[spec.steps[k].object],
            target=None,
            head=head,
            wrist_start=np.zeros(5),
            task=task,
            seed=0,
        )
        view = V1SceneView.of(ctx)
        d = mujoco.MjData(scene.model)
        mujoco.mj_forward(scene.model, d)
        criterion = impl.criterion(head.spec, head.build)
        primitive = impl.primitive(head.spec, head.build)
        assert isinstance(criterion.grasped(scene.model, d, view), bool), model_key
        u = primitive.command(0.0, 0, 0.0, scene.model, d, view)
        assert np.asarray(u).shape == (len(head.build.actuators),)


def test_tool_change_scenes_hold_one_tool_on_the_changer(spec: TaskSpec, task: Any) -> None:
    setup = setups()["tool_change"]
    single = build_studio_scene(setups()["single"], spec, 0, task=task)
    a, b = (build_studio_scene(setup, spec, k, task=task) for k in (0, 1))
    assert [h.spec.model for h in a.heads] == ["adaptive-2f"] and [h.spec.model for h in b.heads] == [
        "cup-16mm"
    ]
    ch = ToolChanger()
    for scene in (a, b):
        assert scene.stack_mass_kg == pytest.approx(ch.master_mass_kg + ch.tool_plate_mass_kg)
        assert any(scene.model.body(i).name == f"{STUDIO_PREFIX}changer" for i in range(scene.model.nbody))
    # the changer stack lowers the tool by its height; the carried mass swaps with the tool
    assert a.active.tcp_wrist[2] == pytest.approx(single.active.tcp_wrist[2] - ch.stack_mm / 1000.0)
    assert a.tool_mass_kg - b.tool_mass_kg == pytest.approx(a.active.build.mass - b.active.build.mass)


def test_contract_checks_refuse_broken_tools_and_objects(spec: TaskSpec, task: Any) -> None:
    setup = setups()["single"]

    def sticky_tool(s: mujoco.MjSpec) -> None:
        next(g for g in s.geoms if g.name.startswith("eff_")).priority = 2

    def slippery_object(s: mujoco.MjSpec) -> None:
        next(g for g in s.geoms if g.name.startswith("obj_carton_")).priority = 0

    def tied(s: mujoco.MjSpec) -> None:
        s.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1="eff_base",
            name2="obj_carton_body",
        )

    for edit, match in ((sticky_tool, "priority"), (slippery_object, "priority 1"), (tied, "ties a tool")):
        with pytest.raises(ContractError, match=match):
            build_studio_scene(setup, spec, 0, task=task, edit=edit)
    wide = TaskSpec(
        "Too wide",
        "",
        (ObjectInstance.of("slab", "part", "Slab", size_x_mm=300, size_y_mm=300, size_z_mm=20, mass_g=500),),
        (
            Step.of(
                "pick", "grasp", "slab", "part", "Pick the slab", [SuccessCriterion("lift_mm", at_least=50)]
            ),
        ),
        source="test",
    )
    with pytest.raises(InvalidEffector):
        build_studio_scene(Setup.single(gripper(), ("pick",)), wide, 0, task=task)


def _hide(monkeypatch: pytest.MonkeyPatch, module: str, registry: dict[Any, Any], key: Any) -> None:
    """Make ``module`` unimportable and drop its registration (S2's modules exist now)."""
    import sys

    monkeypatch.setitem(sys.modules, module, None)
    if key in registry:
        monkeypatch.delitem(registry, key)


def test_templates_without_physics_and_primitives_without_controllers_are_not_simulated(
    task: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rtt.studio import plan as plan_mod
    from rtt.studio import templates as templates_mod
    from rtt.studio.types import PrimitiveKind, TemplateKind

    _hide(monkeypatch, "rtt.studio.templates.drawer", templates_mod._BUILDERS, TemplateKind.DRAWER)
    _hide(monkeypatch, "rtt.studio.primitives.push", plan_mod._SETS["default"], PrimitiveKind.PUSH)
    with pytest.raises(TemplateUnavailable):
        get_template_builder("drawer")
    with pytest.raises(PrimitiveNotSimulated):
        get_primitive("push")
    with pytest.raises(PrimitiveNotSimulated):
        get_primitive("grasp", "reference")
    drawer = ObjectInstance.of("drawer", "drawer", "Drawer")
    step = Step.of(
        "pull", "hook_pull", "drawer", "handle", "Pull it", [SuccessCriterion("travel_mm", at_least=100)]
    )
    spec = TaskSpec("Drawer", "", (drawer,), (step,), source="test")
    r = run_task(_job(Setup.single(gripper(), ("pull",)), spec))
    assert r.steps[0].failure_mode == "not_simulated" and r.steps[0].fidelity is FidelityTier.NOT_SIMULATED
    assert not r.completed and r.steps[0].caveats
    part_spec = two_press_task()
    r = run_task(StudioJob(Setup.single(gripper(), STEPS), part_spec, StudioCondition("nominal", 0)))
    assert [s.failure_mode for s in r.steps] == ["not_simulated", "not_reached"]


# --------------------------------------------------------------------------- smoke runs


def test_tool_change_sequence_over_two_dummy_steps(spec: TaskSpec, task: Any) -> None:
    r = run_task(_job(setups()["tool_change"], spec))
    assert r.completed and [s.step_id for s in r.steps] == list(STEPS)
    assert r.n_tool_changes == 1 and r.n_presentations == 0 and r.change_time_s == TOOL_CHANGE_S
    assert r.cycle_time_s == pytest.approx(sum(s.duration_s for s in r.steps) + TOOL_CHANGE_S)
    assert r.steps[1].start_s == pytest.approx(r.steps[0].duration_s + TOOL_CHANGE_S)
    assert [s.tool for s in r.steps] == [0, 1]
    assert len(r.tool_mass_kg) == 2 and r.tool_mass_kg[0] > r.tool_mass_kg[1]  # the mass swap
    assert r.steps[1].fidelity is FidelityTier.APPROXIMATION and r.steps[1].caveats  # the cup
    assert r.controllers == "reference"
    rec = r.to_record()
    assert rec["label"] and rec["n_tool_changes"] == 1


def test_multi_head_sequence_presents_the_other_head(spec: TaskSpec, task: Any) -> None:
    r = run_task(_job(setups()["multi_head"], spec))
    assert r.completed and r.n_presentations == 1 and r.n_tool_changes == 0
    assert r.change_time_s == PRESENT_S
    assert r.tool_mass_kg[0] == pytest.approx(r.tool_mass_kg[1])  # both heads always carried
    single = run_task(_job(setups()["single"], spec))
    assert single.completed and single.change_time_s == 0.0 and len(single.tool_mass_kg) == 1


@pytest.mark.parametrize("kind", ["twist", "hook_pull", "insert"])
def test_every_reference_controller_runs(kind: str, task: Any) -> None:
    spec, builder = toy_task(kind)
    with override_template_builder(builder):
        for eff in (gripper(), cup()):
            r = run_task(_job(Setup.single(eff, spec.step_ids), spec))
            s = r.steps[0]
            assert s.controller == "reference" and s.duration_s > 0 and s.traj_sha
            assert s.failure_mode in (None, "criterion_not_met")
            if kind == "twist":
                assert s.quantities["torque_nm"] > 0.0  # the roll was driven


def test_world_state_carries_across_scenes(spec: TaskSpec, task: Any) -> None:
    setup = setups()["tool_change"]
    a, b = (build_studio_scene(setup, spec, k, task=task) for k in (0, 1))
    da = mujoco.MjData(a.model)
    adr = a.objects["carton"].joint_qpos["free"]
    da.qpos[adr : adr + 3] += [0.01, -0.02, 0.0]
    state = capture_world_state(a, da, 3.5)
    db = mujoco.MjData(b.model)
    apply_world_state(b, db, state)
    adr_b = b.objects["carton"].joint_qpos["free"]
    assert np.array_equal(db.qpos[adr_b : adr_b + 7], da.qpos[adr : adr + 7])
    assert state.clock_s == 3.5


# --------------------------------------------------------------------------- determinism and fairness


def test_traj_hashes_are_deterministic_in_process_and_in_workers(spec: TaskSpec) -> None:
    jobs = [_job(s, spec, seed=3) for s in setups().values()]
    first = [run_task(j) for j in jobs]
    again = [run_task(j) for j in jobs]
    workers = run_task_jobs(jobs, workers=2)
    for a, b, c in zip(first, again, workers, strict=True):
        assert a.traj_sha == b.traj_sha == c.traj_sha
        assert [s.traj_sha for s in a.steps] == [s.traj_sha for s in c.steps]
        assert a.to_record() | {"wall_time_s": 0} == c.to_record() | {"wall_time_s": 0}
    other_seed = run_task(_job(setups()["single"], spec, seed=4))
    assert other_seed.traj_sha != first[0].traj_sha  # the seed's jitter matters


def test_fairness_is_asserted_across_candidates(spec: TaskSpec) -> None:
    grid = [("nominal", 0), ("nominal", 1)]
    s = setups()
    runs = [run_task(_job(setup, spec, seed, cid)) for setup in s.values() for cid, seed in grid]
    assert fairness_problems(runs, list(s.values())) == []
    assert_fair(runs, list(s.values()))
    assert len({r.fairness_sha for r in runs}) == 1
    lopsided = runs[:-1] + [run_task(_job(s["tool_change"], spec, 7))]
    assert any("other conditions or seeds" in p for p in fairness_problems(lopsided))
    with pytest.raises(ContractError, match="unfair"):
        assert_fair(lopsided)
    slow = Setup.tool_change(
        dict(zip(STEPS, (gripper(), cup()), strict=True)), ToolChanger(changeover_s=12.0)
    )
    assert any("tool changers" in p for p in fairness_problems(runs, [s["tool_change"], slow]))
    mixed = runs[:2] + [run_task(StudioJob(s["single"], spec, StudioCondition("nominal", 0)))]
    assert any("controllers" in p for p in fairness_problems(mixed))


def test_objects_sit_where_the_spec_says(spec: TaskSpec, task: Any) -> None:
    scene = build_studio_scene(setups()["single"], spec, 0, task=task)
    d = mujoco.MjData(scene.model)
    mujoco.mj_forward(scene.model, d)
    carton = scene.objects["carton"]
    assert carton.site_pos(d, "part") == pytest.approx([0.1, 0.0, 0.03], abs=1e-9)
    tote = scene.objects["tote"]
    assert tote.site_pos(d, "bin")[:2] == pytest.approx([0.0, -0.2], abs=1e-9)
    assert ObjectPose(100, 0, 0).x_mm == 100.0
