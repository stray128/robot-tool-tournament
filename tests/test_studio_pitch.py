"""The wrist pitch in the Studio contract (``rtt-effectors/2``, ``docs/STUDIO.md`` section 4.1).

* the six-servo wrist: x, y, z, yaw, roll, pitch (the roll keeps index 4, the pitch is 5), the
  pitch hinge between the yaw and the roll, the same in every scene;
* :func:`~rtt.studio.scene.world_tcp`: palm . Rz(yaw) . Rx(pitch) . Rz(roll) . tcp, which is the
  old Rz(yaw + roll) at pitch 0 and MuJoCo's own kinematics at any pitch;
* the pitch limit (+-90 deg in the joint, the servo and the runner) and its speed cap;
* a five-target controller holds the pitch where its step began; a segment starts at pitch 0;
* ``fairness_sha`` covers the pitch; a held part still blocks a tool change or a head switch;
* the tool-change constants (7.5 s, a 52 mm changer).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import mujoco
import numpy as np
import pytest

from rtt.effectors.types import ContractError
from rtt.sim.task import default_task_v2
from rtt.studio.episode import HOME_HEIGHT, StudioCondition, StudioJob, run_task
from rtt.studio.fairness import fairness_sha, wrist_signature
from rtt.studio.plan import PrimitiveCommand, StepContext, StepPlan, StepVerdict, override_primitive
from rtt.studio.scene import (
    PITCH_ACTUATOR,
    PITCH_JOINT,
    ROLL_JOINT,
    WRIST_JOINTS,
    build_studio_scene,
    world_tcp,
)
from rtt.studio.types import (
    STUDIO_WRIST,
    TOOL_CHANGE_S,
    HeadMount,
    ObjectInstance,
    ObjectPose,
    PrimitiveKind,
    Setup,
    Step,
    StepConstraints,
    SuccessCriterion,
    TaskSpec,
    ToolChanger,
)
from studio_fixtures import cup, gripper, preset, two_press_task

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


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# --------------------------------------------------------------------------- layout and kinematics


def test_six_servo_layout(spec: TaskSpec, task: Any) -> None:
    assert WRIST_JOINTS == ("palm_x", "palm_y", "palm_z", "palm_yaw", ROLL_JOINT, PITCH_JOINT)
    for setup in setups().values():
        for tool in range(len(setup.tools)):
            scene = build_studio_scene(setup, spec, tool, task=task)
            m, ids = scene.model, scene.ids
            assert list(ids.wrist_actuators) == [0, 1, 2, 3, 4, 5]
            for i, name in enumerate(WRIST_JOINTS):  # actuator i drives wrist joint i
                assert int(m.actuator_trnid[i, 0]) == m.joint(name).id
            assert m.actuator(4).name == ROLL_JOINT and m.actuator(5).name == PITCH_ACTUATOR
            heads = sum(len(h.build.actuators) for h in scene.heads)
            assert m.nu == 6 + heads + len(ids.object_actuators)
            assert min(np.concatenate([ids.effector_actuators, ids.idle_actuators])) == 6  # heads next
            # the chain: palm (yaw) -> pitch about the palm's x -> roll about the tool axis -> flange
            assert m.body_parentid[ids.pitch_body] == ids.palm_body
            assert m.body_parentid[ids.roll_body] == ids.pitch_body
            assert m.body_parentid[ids.flange_body] == ids.roll_body
            pitch, roll = m.joint(PITCH_JOINT).id, m.joint(ROLL_JOINT).id
            assert m.jnt_bodyid[pitch] == ids.pitch_body and m.jnt_bodyid[roll] == ids.roll_body
            np.testing.assert_array_equal(m.jnt_axis[pitch], [1.0, 0.0, 0.0])
            np.testing.assert_array_equal(m.jnt_axis[roll], [0.0, 0.0, 1.0])
            assert ids.pitch_qpos == m.jnt_qposadr[pitch] and ids.pitch_dof == m.jnt_dofadr[pitch]


def test_pitch_zero_matches_the_old_tcp(spec: TaskSpec, task: Any) -> None:
    scene = build_studio_scene(setups()["multi_head"], spec, 0, task=task)
    ids = scene.ids
    d = mujoco.MjData(scene.model)
    idle = next(h for h in scene.heads if not h.active)
    assert np.linalg.norm(idle.tcp_wrist[:2]) > 0.02  # an off-axis point, so Rz is really tested
    rng = np.random.default_rng(7)
    for _ in range(6):
        x, y, z = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.2, 0.6)
        yaw, roll = rng.uniform(-3.0, 3.0), rng.uniform(-12.0, 12.0)
        d.qpos[ids.palm_qpos] = [x, y, z, yaw]
        d.qpos[ids.roll_qpos] = roll
        d.qpos[ids.pitch_qpos] = 0.0
        for point in (scene.active.tcp_wrist, idle.tcp_wrist):
            old = np.array([x, y, z]) + _rz(yaw + roll) @ np.asarray(point)
            np.testing.assert_allclose(world_tcp(scene, d, point), old, rtol=0.0, atol=1e-9)
        old = np.array([x, y, z]) + _rz(yaw + roll) @ np.asarray(scene.active.tcp_wrist)
        np.testing.assert_allclose(world_tcp(scene, d), old, rtol=0.0, atol=1e-9)


def test_world_tcp_is_the_compiled_chain_at_any_pitch(spec: TaskSpec, task: Any) -> None:
    for name in ("multi_head", "tool_change"):
        scene = build_studio_scene(setups()[name], spec, 0, task=task)
        m, ids = scene.model, scene.ids
        d = mujoco.MjData(m)
        for k, pitch in enumerate((-math.pi / 2, -0.7, 0.4, math.pi / 2)):
            x, y, z, yaw, roll = 0.05 * k, -0.03, 0.4, 0.3 * k - 0.5, 2.0 - k
            d.qpos[ids.palm_qpos] = [x, y, z, yaw]
            d.qpos[ids.roll_qpos] = roll
            d.qpos[ids.pitch_qpos] = pitch
            mujoco.mj_kinematics(m, d)
            for h in scene.heads:
                mount = m.body(h.mount_body).id
                truth = d.xpos[mount] + d.xmat[mount].reshape(3, 3) @ np.asarray(h.build.tcp)
                np.testing.assert_allclose(world_tcp(scene, d, h.tcp_wrist), truth, rtol=0.0, atol=1e-9)
            tcp = np.asarray(scene.active.tcp_wrist)
            chain = np.array([x, y, z]) + _rz(yaw) @ _rx(pitch) @ _rz(roll) @ tcp
            np.testing.assert_allclose(world_tcp(scene, d), chain, rtol=0.0, atol=1e-12)
            # a controller's TCP offset (roll 0) is the same chain
            ctx = StepContext(
                step=spec.steps[0],
                task_spec=spec,
                scene=scene,
                obj=scene.objects[spec.steps[0].object],
                target=None,
                head=scene.active,
                wrist_start=np.array([x, y, z, yaw, 0.0]),
                task=task,
                seed=0,
                wrist_pitch_start=pitch,
            )
            d.qpos[ids.roll_qpos] = 0.0
            np.testing.assert_allclose(
                ctx.tcp_offset(yaw, pitch), world_tcp(scene, d) - [x, y, z], rtol=0.0, atol=1e-12
            )


# --------------------------------------------------------------------------- the limit and the runner


@dataclass(frozen=True)
class _Tilt:
    """A test controller: the wrist stays where the step began while the pitch target turns at
    ``rate_dps`` toward a step's goal (deg); a step without a goal gets five targets only."""

    goals: tuple[tuple[str, float], ...]
    rate_dps: float = 90.0
    duration_s: float = 1.0
    primitive: PrimitiveKind = PrimitiveKind.PUSH
    name: str = "tilt"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        return StepPlan(self.duration_s)

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        u = np.asarray(ctx.head.build.ctrl_open, dtype=np.float64)
        start = np.asarray(ctx.wrist_start, dtype=np.float64)
        goal = dict(self.goals).get(ctx.step.id)
        if goal is None:
            return PrimitiveCommand(start.copy(), u)
        d = math.radians(goal) - ctx.wrist_pitch_start
        pitch = ctx.wrist_pitch_start + math.copysign(min(abs(d), math.radians(self.rate_dps) * t), d)
        return PrimitiveCommand(np.append(start, pitch), u)


@dataclass(frozen=True)
class _PitchProbe:
    """Records the pitch where the step began, its target and the joint, at the step's end."""

    primitive: PrimitiveKind = PrimitiveKind.PUSH
    name: str = "pitch_probe"

    def observe(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None:
        acc["start_deg"] = math.degrees(ctx.wrist_pitch_start)
        acc["target_deg"] = math.degrees(float(data.ctrl[ctx.scene.ids.wrist_actuators[5]]))
        acc["pitch_deg"] = math.degrees(float(data.qpos[ctx.scene.ids.pitch_qpos]))

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict:
        return StepVerdict(True, None, acc)


def _run(setup: Setup, spec: TaskSpec, controller: _Tilt) -> Any:
    with override_primitive(controller, _PitchProbe(), controllers="reference"):
        return run_task(StudioJob(setup, spec, StudioCondition("nominal", 0), controllers="reference"))


def test_pitch_is_limited_to_90_degrees(spec: TaskSpec, task: Any) -> None:
    assert STUDIO_WRIST.pitch_range_deg == 90.0
    lim = math.radians(90.0)
    scene = build_studio_scene(setups()["single"], spec, 0, task=task)
    m, ids = scene.model, scene.ids
    j, a = m.joint(PITCH_JOINT), m.actuator(PITCH_ACTUATOR)
    assert int(m.jnt_limited[j.id]) == 1 and int(j.type[0]) == mujoco.mjtJoint.mjJNT_HINGE
    np.testing.assert_allclose(j.range, [-lim, lim])
    assert int(m.actuator_ctrllimited[a.id]) == 1
    np.testing.assert_allclose(m.actuator_ctrlrange[a.id], [-lim, lim])
    assert list(a.forcerange) == [-STUDIO_WRIST.pitch_torque_nm, STUDIO_WRIST.pitch_torque_nm]
    # a servo target past the limit is clamped and the joint stops at 90 deg
    d = mujoco.MjData(m)
    d.qpos[ids.palm_qpos] = [0.0, 0.0, HOME_HEIGHT + 0.2, 0.0]
    mujoco.mj_forward(m, d)
    ctrl = np.zeros(m.nu)
    ctrl[:4] = d.qpos[ids.palm_qpos]
    ctrl[ids.effector_actuators] = scene.active.build.ctrl_open
    ctrl[5] = 2.5
    for _ in range(750):
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d, nstep=2)
    assert d.qpos[ids.pitch_qpos] == pytest.approx(lim, abs=0.02)
    assert d.qpos[ids.pitch_qpos] <= lim + 0.01
    # the runner refuses a target past the limit, however slowly it gets there
    with pytest.raises(ContractError, match=r"past \+-90 deg"):
        _run(setups()["single"], spec, _Tilt(((STEPS[0], 95.0),), duration_s=1.5))
    with pytest.raises(ContractError, match=r"past \+-90 deg"):
        _run(setups()["single"], spec, _Tilt(((STEPS[0], -95.0),), duration_s=1.5))


def test_the_pitch_target_may_not_move_faster_than_its_cap(spec: TaskSpec) -> None:
    assert STUDIO_WRIST.pitch_speed_dps == 180.0
    with pytest.raises(ContractError, match="faster than 180 deg/s"):
        _run(setups()["single"], spec, _Tilt(((STEPS[0], 60.0),), rate_dps=270.0))
    r = _run(setups()["single"], spec, _Tilt(((STEPS[0], 60.0),), rate_dps=179.0))
    assert r.steps[0].success and r.steps[0].quantities["target_deg"] == pytest.approx(60.0)


def test_a_five_target_controller_holds_the_pitch_where_its_step_began(spec: TaskSpec) -> None:
    tilt = _Tilt(((STEPS[0], 30.0),))  # the first step tilts, the second returns five targets
    one = _run(setups()["single"], spec, tilt)  # one segment: the tilt carries into step 2
    s0, s1 = one.steps
    assert s0.quantities["start_deg"] == 0.0 and s0.quantities["target_deg"] == pytest.approx(30.0)
    assert s0.quantities["pitch_deg"] == pytest.approx(30.0, abs=0.5)
    assert s1.quantities["start_deg"] == pytest.approx(30.0)
    assert s1.quantities["target_deg"] == pytest.approx(30.0)
    assert s1.quantities["pitch_deg"] == pytest.approx(30.0, abs=0.5)
    two = _run(setups()["tool_change"], spec, tilt)  # a new segment starts at home, pitch 0
    assert two.steps[1].quantities["start_deg"] == 0.0
    assert abs(two.steps[1].quantities["pitch_deg"]) < 0.1


# --------------------------------------------------------------------------- fairness and holding


def test_fairness_sha_changes_with_pitch(spec: TaskSpec, task: Any) -> None:
    setup = setups()["single"]
    base = build_studio_scene(setup, spec, 0, task=task)
    sig = wrist_signature(base.model)
    assert PITCH_JOINT in sig and f"body:{PITCH_JOINT}" in sig
    assert sig[PITCH_JOINT]["limited"] == 1
    assert sig[PITCH_JOINT]["range"] == pytest.approx([-math.pi / 2, math.pi / 2])
    limits = base.wrist.to_dict()
    for key in ("pitch_range_deg", "pitch_kp", "pitch_kv", "pitch_torque_nm", "pitch_speed_dps"):
        assert key in limits
    shas = {fairness_sha(base)}
    for change in (
        {"pitch_range_deg": 60.0},
        {"pitch_kp": 400.0},
        {"pitch_torque_nm": 10.0},
        {"pitch_speed_dps": 90.0},  # not in the model, but a runner rule: in the constants
    ):
        wrist = replace(STUDIO_WRIST, **change)
        shas.add(fairness_sha(build_studio_scene(setup, spec, 0, task=task, wrist=wrist)))
    assert len(shas) == 5
    # every setup kind still gets one wrist, pitch included
    assert len({fairness_sha(build_studio_scene(s, spec, 0, task=task)) for s in setups().values()}) == 1


def test_held_part_still_blocks_a_tool_change(task: Any) -> None:
    g, c = gripper(), cup()
    port = ObjectInstance.of("port", "port_plug", "Port")
    grab = Step.of(
        "grab", "grasp", "port", "plug", "Pick", [SuccessCriterion("lift_mm", at_least=1)], hold_after=True
    )
    plug = Step.of("plug_in", "insert", "port", "plug", "Insert", [SuccessCriterion("depth_mm", at_least=1)])
    held = TaskSpec("Plug", "", (port,), (grab, plug), source="test")
    change = Setup.tool_change({"grab": g, "plug_in": c})
    switch = Setup.multi_head(((g, HeadMount(60.0)), (c, HeadMount(-60.0))), {"grab": 0, "plug_in": 1})
    for setup in (change, switch):
        assert any("while 'port' is held" in p for p in setup.problems(held))
        with pytest.raises(ValueError, match="while 'port' is held"):
            setup.segments(held)
        with pytest.raises(ValueError, match="cannot run this task"):
            build_studio_scene(setup, held, 0, task=task)
        with pytest.raises(ValueError, match="while 'port' is held"):
            run_task(StudioJob(setup, held, StudioCondition("nominal", 0), task=task))
    assert Setup.single(g, held.step_ids).problems(held) == []


def test_tool_change_takes_7_5_s_on_a_52_mm_changer() -> None:
    assert TOOL_CHANGE_S == 7.5
    ch = ToolChanger()
    assert ch.changeover_s == 7.5 and ch.stack_mm == 52.0


# --------------------------------------------------------------------------- the search sees the tilt


def _lip_and_flat(feature: str = "lip", y_mm: float = 100.0) -> TaskSpec:
    """A lid with a 20 mm lip (pinched by a two-finger head at a 90 deg tilt; ``feature="handle"``:
    a bar gripped from above instead) and a flat item for a cup, far apart on the table. At
    y = 100 mm the pinch's palm stays within its travel all the way to 103 deg (about 314 mm out);
    at 150 mm it would pass it (365 mm), so the lid is hooked instead."""
    lid = ObjectInstance.of(
        "lid",
        "hinged_lid",
        "Lid",
        ObjectPose(0.0, y_mm, 0.0),
        depth_mm=240,
        width_mm=340,
        thickness_mm=6,
        feature=feature,
        feature_mm=20,
        max_angle_deg=110,
        open_torque_nm=0.3,
        damping_nms=0.05,
    )
    flat = ObjectInstance.of(
        "flat",
        "part",
        "Flat item",
        ObjectPose(-150.0, -150.0, 0.0),
        size_x_mm=120,
        size_y_mm=90,
        size_z_mm=15,
    )
    steps = (
        Step.of(
            "open_lid",
            "hook_pull",
            "lid",
            "handle",
            "Open the lid by its lip",
            [SuccessCriterion("angle_deg", at_least=95)],
            constraints=StepConstraints(max_force_n=20),
        ),
        Step.of(
            "lift",
            "suction",
            "flat",
            "part",
            "Lift the flat item",
            [SuccessCriterion("lift_mm", at_least=50)],
        ),
    )
    return TaskSpec("tilt sweep", "", (lid, flat), steps, source="test")


def test_a_tilted_head_near_90_deg_is_refused_by_the_swept_volume_check(task: Any) -> None:
    """Review focus 1: the servo gripper pinches the lip with the wrist tilted 90 deg. On a bracket
    with the cup 150 deg from it, the tilt swings the idle cup up through the arm's link above the
    pitch joint (which does not tilt): the swept-volume check refuses the bracket for that reason
    alone, where the check with the tool straight down passed it. The same heads with the cup
    straight down and the gripper 120 deg off it keep clear of the arm while tilted and pass."""
    from rtt.studio.search import default_candidates, swept_volume_problems

    spec = _lip_and_flat()
    servo = next(c.spec for c in default_candidates() if c.spec.model == "gripper-2f-servo-110")
    cup25 = preset("suction", "cup-25mm", "seal")
    plan = {"open_lid": 0, "lift": 1}
    refused = Setup.multi_head(((servo, HeadMount(-75.0, 100.0)), (cup25, HeadMount(75.0, 100.0))), plan)
    problems, _, _ = swept_volume_problems(refused, spec, task=task)
    assert problems, "a tilted head swinging through the arm must be refused"
    assert all("during open_lid the wrist tilted" in p for p in problems), problems
    assert any("into the arm's link above the pitch joint" in p for p in problems), problems
    clear = Setup.multi_head(((servo, HeadMount(-120.0, 150.0)), (cup25, HeadMount(0.0, 150.0))), plan)
    assert swept_volume_problems(clear, spec, task=task)[0] == []


@dataclass(frozen=True)
class _TiltWatch:
    """The push predicate, also recording the wrist pitch's peak and its value at the step's end."""

    primitive: PrimitiveKind = PrimitiveKind.PUSH
    name: str = "studio_push"

    def observe(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None:
        from rtt.studio.primitives.push import PushPredicate

        PushPredicate().observe(t, ctx, model, data, acc)
        pitch = math.degrees(float(data.qpos[ctx.scene.ids.pitch_qpos]))
        acc["pitch_peak_deg"] = max(acc.get("pitch_peak_deg", 0.0), abs(pitch))
        acc["pitch_end_deg"] = pitch

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict:
        from rtt.studio.primitives.push import PushPredicate

        return PushPredicate().verdict(ctx, acc)


def _drawer_from_the_front(y_mm: float) -> TaskSpec:
    drawer = ObjectInstance.of("dr", "drawer", "Drawer", ObjectPose(0.0, y_mm, 0.0))
    close = Step.of(
        "close",
        "push",
        "dr",
        "drawer",
        "Push the drawer shut from the front",
        [SuccessCriterion("travel_mm", at_most=5)],
        params={"approach": "front"},  # push's optional param (omitted at its default, "top")
        constraints=StepConstraints(max_force_n=40),
    )
    pull = Step.of(
        "open",
        "hook_pull",
        "dr",
        "handle",
        "Pull the drawer open",
        [SuccessCriterion("travel_mm", at_least=120), SuccessCriterion("slip_mm", at_most=15)],
    )
    return TaskSpec("drawer from the front", "", (drawer,), (pull, close), source="test")


def test_a_push_from_the_front_tilts_the_tool_square_to_an_upright_face(task: Any) -> None:
    """``approach="front"`` on a push: a drawer pulled 160 mm open is pushed shut with the wrist
    tilted 90 deg, the tool's tip square on the drawer's front, the guarded move in horizontal; the
    retract rises clear and turns the tool back straight down. A drawer whose open front would put
    the tilted palm past its travel (the palm sits the tool's length out) is pushed from the top."""
    from rtt.studio.primitives.push import PushController

    for y_mm, tilted in ((250.0, True), (120.0, False)):
        spec = _drawer_from_the_front(y_mm)
        with override_primitive(PushController(), _TiltWatch()):
            r = run_task(
                StudioJob(
                    Setup.single(gripper(), spec.step_ids), spec, StudioCondition("nominal", 0), task=task
                )
            )
        out = r.steps[1]
        assert r.completed, (y_mm, [(s.step_id, s.failure_mode, dict(s.quantities)) for s in r.steps])
        assert out.quantities["travel_mm"] <= 5.0 and out.quantities["force_n"] <= 40.0, y_mm
        assert (out.quantities["pitch_peak_deg"] >= 85.0) is tilted, (y_mm, out.quantities["pitch_peak_deg"])
        assert abs(out.quantities["pitch_end_deg"]) <= 2.0  # straight down again
        assert out.quantities["penetration_mm"] <= 1.0


def _servo() -> Any:
    from rtt.studio.search import default_candidates

    return next(c.spec for c in default_candidates() if c.spec.model == "gripper-2f-servo-110")


def test_a_tilted_idle_head_that_reaches_the_table_is_refused(task: Any) -> None:
    """Review focus 1, the wall or table half: on a bracket with the cup 60 deg from the servo
    gripper (100 mm standoff), pinching the lip at a 90 deg tilt brings the idle cup down to the
    table, and the swept-volume check refuses the step for that alone. The same bracket opening a
    lid by a bar gripped from above (no tilt) has no problem at that step."""
    from rtt.studio.search import swept_volume_problems

    setup = Setup.multi_head(
        (
            (_servo(), HeadMount(30.0, 100.0)),
            (preset("suction", "cup-25mm", "seal"), HeadMount(-30.0, 100.0)),
        ),
        {"open_lid": 0, "lift": 1},
    )
    tilted = [p for p in swept_volume_problems(setup, _lip_and_flat(), task=task)[0] if "open_lid" in p]
    assert tilted == [
        "during open_lid (the wrist tilted 90 deg) the idle head 1 or the bracket comes within 5 mm of the table"
    ]
    straight = swept_volume_problems(setup, _lip_and_flat("handle"), task=task)[0]
    assert not [p for p in straight if "open_lid" in p], straight


#: tools that hook a lip (straight down) although their fingers would clear a 6 mm lip: each of
#: them pinched the lip under the old rule (apertures alone)
HOOKERS = (
    "hand-4f-16dof",
    "fingered-hand-d55e79eb9c71",
    "hook-gripper-da49099c0b38",
    "segmented-tentacle-f293609d8563",
)


def test_only_two_finger_grippers_pinch_a_lid_lip(task: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pinch is for two-finger grippers (parallel, angular or adaptive two-finger, jaws): a
    four- or three-finger hand, a hook gripper and a tentacle, whose calibrated apertures would
    clear the lip, take the hook path, in the controller and in the search's tilted poses alike."""
    import rtt.studio.primitives.hook_pull as hp
    from rtt.studio.search import default_candidates, tilted_poses

    specs = {c.spec.model: c.spec for c in default_candidates()}
    for model in ("gripper-2f-servo-110", "gripper-2f-adaptive-85", "gripper-2f-rigid-a", "adaptive-2f"):
        assert hp.two_finger(specs[model]), model
    for model in (*HOOKERS, "adaptive-3f", "tool-button-pusher", "hand-3f-12dof-industrial"):
        assert not hp.two_finger(specs[model]), model
    modes: list[str] = []
    plan = hp.HookPullController.plan

    def spy(self: Any, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        out = plan(self, ctx, model, data)
        modes.append(out.values["run"].get("mode", "grip"))
        return out

    monkeypatch.setattr(hp.HookPullController, "plan", spy)
    spec = TaskSpec("lip", "", (_lip_and_flat().objects[0],), (_lip_and_flat().steps[0],), source="test")
    for model in ("gripper-2f-servo-110", *HOOKERS):
        tool = specs[model]
        modes.clear()
        run_task(StudioJob(Setup.single(tool, spec.step_ids), spec, StudioCondition("nominal", 0), task=task))
        scene = build_studio_scene(Setup.single(tool, spec.step_ids), spec, 0, task=task)
        data = mujoco.MjData(scene.model)
        mujoco.mj_kinematics(scene.model, data)
        poses = tilted_poses(
            Setup.single(tool, spec.step_ids), spec.steps[0], 0, scene, data, task, scene.active.tcp_wrist
        )
        pinch = model == "gripper-2f-servo-110"
        assert modes == ["pinch" if pinch else "hook"], (model, modes)
        assert bool(poses) is pinch, model


def test_a_lip_pinch_whose_palm_would_leave_its_travel_falls_back_to_the_hook(task: Any) -> None:
    """The lip pinch is checked along its whole path, opening as well as closing (at
    103 deg the palm sits the tool's length beyond the free edge). The 240 mm lid at y = 150 mm
    commanded the palm to 365 mm while the slide stops at 350 mm, for 3.4 s; it is now hooked
    instead (the servo gripper has no hook, so it fails honestly), the plan says why, and the
    search's tilted poses agree. At y = 100 mm the same lid is pinched."""
    import rtt.studio.primitives.hook_pull as hp
    from rtt.studio.search import tilted_poses

    notes: list[str] = []
    plan = hp.HookPullController.plan

    def spy(self: Any, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        out = plan(self, ctx, model, data)
        notes.append(f"{out.values['run'].get('mode', 'grip')}: {out.note}")
        return out

    servo = _servo()
    for y_mm, pinch in ((150.0, False), (100.0, True)):
        full = _lip_and_flat(y_mm=y_mm)
        spec = TaskSpec("lip", "", (full.objects[0],), (full.steps[0],), source="test")
        setup = Setup.single(servo, spec.step_ids)
        notes.clear()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(hp.HookPullController, "plan", spy)
            out = run_task(StudioJob(setup, spec, StudioCondition("nominal", 0), task=task)).steps[0]
        scene = build_studio_scene(setup, spec, 0, task=task)
        data = mujoco.MjData(scene.model)
        mujoco.mj_kinematics(scene.model, data)
        poses = tilted_poses(setup, spec.steps[0], 0, scene, data, task, scene.active.tcp_wrist)
        assert bool(poses) is pinch, y_mm
        if pinch:
            assert notes[-1].startswith("pinch:") and out.success, (notes, out.failure_mode)
        else:
            assert notes[-1].startswith("hook:") and "not pinched: the carry would take the palm" in notes[-1]
            assert not out.success, (y_mm, out.failure_mode)


def test_a_hook_whose_pull_on_would_leave_the_palms_travel_is_out_of_reach(task: Any) -> None:
    """The same 240 mm lid at y = 150 mm with a 20 N force goal. The pinch is declined
    (its palm would leave its travel) and the hook takes over, but the hook's pull on to the force
    goal goes on along the lid's arc to its stop and past it: it commanded the palm to 368.7 mm for
    865 ticks while the slide stops at 350 mm. The hook is now checked along its whole path, the pull
    on included, with the pinch's 12 mm margin: the step fails ``out_of_reach``, not attempted,
    and says why. Without the force goal the hook stays in reach and is taken (above)."""
    import rtt.studio.primitives.hook_pull as hp

    lid = _lip_and_flat(y_mm=150.0).objects[0]
    forced = Step.of(
        "open_lid",
        "hook_pull",
        "lid",
        "handle",
        "Open the lid by its lip and hold it open with 20 N",
        [SuccessCriterion("angle_deg", at_least=95), SuccessCriterion("force_n", at_least=20)],
        params={"speed_mm_s": 80.0},
    )
    spec = TaskSpec("lip", "", (lid,), (forced,), source="test")
    out = run_task(
        StudioJob(Setup.single(_servo(), spec.step_ids), spec, StudioCondition("nominal", 0), task=task)
    ).steps[0]
    assert not out.success and out.failure_mode == "out_of_reach", (out.failure_mode, dict(out.quantities))
    q = out.quantities
    limit = 1000.0 * (0.35 - hp.PINCH_REACH_MARGIN)
    assert max(abs(q["reach_x_mm"]), abs(q["reach_y_mm"])) > limit and q["palm_limit_mm"] == pytest.approx(
        350.0
    )
    assert out.duration_s <= 0.25
    why = [c for c in out.caveats if c.startswith("Out of reach:")]
    assert why and "12 mm" in why[0], out.caveats


@pytest.mark.parametrize("depth_mm", [60, 40])
def test_a_fast_pull_on_a_small_lid_stays_within_the_pitch_cap(task: Any, depth_mm: int) -> None:
    """The pitch turns with the lid, so a fast pull on a small lid at 200 mm/s (its peak speed:
    the arc alone would turn the pitch 1.06x faster than its 180 deg/s cap on a 60 mm lid, 1.6x on
    a 40 mm one) slows to the cap instead of tripping the runner's contract check, and still opens
    the lid."""
    lid = ObjectInstance.of(
        "lid",
        "hinged_lid",
        "Lid",
        ObjectPose(0.0, 150.0, 0.0),
        depth_mm=depth_mm,
        width_mm=200,
        thickness_mm=6,
        feature="lip",
        feature_mm=15,
        max_angle_deg=110,
        open_torque_nm=0.05,
        damping_nms=0.01,
    )
    step = Step.of(
        "open_lid",
        "hook_pull",
        "lid",
        "handle",
        "Open the small lid fast",
        [SuccessCriterion("angle_deg", at_least=95)],
        params={"speed_mm_s": 200.0},
    )
    spec = TaskSpec("fast lip", "", (lid,), (step,), source="test")
    out = run_task(
        StudioJob(Setup.single(_servo(), spec.step_ids), spec, StudioCondition("nominal", 0), task=task)
    )
    assert out.steps[0].success, (out.steps[0].failure_mode, dict(out.steps[0].quantities))


def test_a_pinch_start_a_hair_past_the_range_is_clamped() -> None:
    """A lid observed a hair below closed puts the pinch's start a hair past +90 deg: within 2 deg
    it is clamped to the range (the pinch goes ahead); further, it is not (the hook then)."""
    from rtt.studio.primitives.hook_pull import pinch_pitches

    lim, goal = math.radians(90.0), math.radians(103.0)
    p0, p1 = pinch_pitches(-1.0, math.radians(-1.5), goal, lim)
    assert p0 == lim and math.isclose(math.degrees(p1), -13.0)
    assert math.isclose(math.degrees(pinch_pitches(-1.0, math.radians(-3.0), goal, lim)[0]), 93.0)


@dataclass(frozen=True)
class _PitchLog:
    """The hook-pull controller, logging every pitch target it commands into ``log``."""

    log: list[float]
    primitive: PrimitiveKind = PrimitiveKind.HOOK_PULL
    name: str = "studio_hook_pull"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        from rtt.studio.primitives.hook_pull import HookPullController

        return HookPullController().plan(ctx, model, data)

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        from rtt.studio.primitives.hook_pull import HookPullController

        out = HookPullController().command(t, ctx, plan, model, data)
        self.log.append(float(np.asarray(out.wrist)[5]))
        return out


def test_a_pull_on_to_a_force_goal_keeps_to_the_pitch_the_search_checked(task: Any) -> None:
    """With a ``force_n`` at_least goal the pinch pulls on past its goal into the
    lid's stop, and its pitch used to turn on with the lid (a 40 mm lid, 8 N: -44 deg commanded,
    while the search's tilted swept-volume arc ends at -13 deg), so the swept-volume check never saw
    that rotation. The pull-on now holds the pitch at the goal's: every pitch target the controller
    commands lies within the pitches the search checks (its tilted poses, and straight down)."""
    from rtt.studio.primitives.hook_pull import HookPullPredicate
    from rtt.studio.search import tilted_poses

    lid = ObjectInstance.of(
        "lid",
        "hinged_lid",
        "Lid",
        ObjectPose(0.0, 150.0, 0.0),
        depth_mm=40,
        width_mm=200,
        thickness_mm=6,
        feature="lip",
        feature_mm=15,
        max_angle_deg=110,
        open_torque_nm=0.05,
        damping_nms=0.01,
    )
    step = Step.of(
        "open_lid",
        "hook_pull",
        "lid",
        "handle",
        "Open the small lid and hold it open with 8 N",
        [SuccessCriterion("angle_deg", at_least=95), SuccessCriterion("force_n", at_least=8)],
        params={"speed_mm_s": 50.0},
    )
    spec = TaskSpec("pull on", "", (lid,), (step,), source="test")
    setup = Setup.single(_servo(), spec.step_ids)
    scene = build_studio_scene(setup, spec, 0, task=task)
    data = mujoco.MjData(scene.model)
    mujoco.mj_kinematics(scene.model, data)
    checked = [p for _, _, _, p in tilted_poses(setup, step, 0, scene, data, task, scene.active.tcp_wrist)]
    assert checked, "the search sees the pinch"
    lo, hi = min(0.0, *checked), max(0.0, *checked)
    log: list[float] = []
    with override_primitive(_PitchLog(log), HookPullPredicate()):
        r = run_task(StudioJob(setup, spec, StudioCondition("nominal", 0), task=task))
    out = r.steps[0]
    assert out.quantities["angle_deg"] >= 95.0, (out.failure_mode, dict(out.quantities))
    assert lo - 1e-9 <= min(log) and max(log) <= hi + 1e-9, (
        math.degrees(min(log)),
        math.degrees(max(log)),
        math.degrees(lo),
        math.degrees(hi),
    )


def test_every_catalog_gripper_is_a_two_finger_pincher() -> None:
    """The pinch is for two-finger grippers, and a vendored model says it is one by
    its catalog kind. Every catalog and library model of kind ``gripper`` (today all two-finger,
    parallel or angular) is classified two-finger, so a gripper added later cannot silently lose the
    pinch; every hand and tool is not."""
    from rtt.effectors.catalog.family import vendored_spec
    from rtt.effectors.catalog.library_models import LIBRARY_MODELS
    from rtt.effectors.catalog.models import CATALOG_MODELS
    from rtt.effectors.types import EffectorFamily
    from rtt.studio.primitives.hook_pull import two_finger

    models = [*CATALOG_MODELS.values(), *LIBRARY_MODELS.values()]
    assert sum(m.kind == "gripper" for m in models) >= 12
    for cm in models:
        family = EffectorFamily.CATALOG_HAND if cm.kind == "hand" else EffectorFamily.CATALOG_GRIPPER
        for syn in cm.synergies:
            assert two_finger(vendored_spec(family, cm, syn)) is (cm.kind == "gripper"), (cm.key, cm.kind)


def test_a_path_turns_the_pitch_within_the_scenes_cap() -> None:
    """``Path.to`` bounds a pitch move by the scene's own cap (``Path.start``), not the default
    wrist's; a path without the cap refuses to turn the pitch."""
    from rtt.studio.primitives._common import TURN_MARGIN, Path

    a = np.zeros(6)
    b = a.copy()
    b[5] = 0.5 * math.pi
    for dps in (180.0, 60.0):
        path = Path.start(a, dps)
        assert path.to(b, 0.1) == pytest.approx(TURN_MARGIN * 1.875 * 0.5 * math.pi / math.radians(dps))
    with pytest.raises(ValueError, match="pitch cap"):
        Path([0.0], [a]).to(b, 0.1)


# --------------------------------------------------------------------------- a door's bar gripped from the front


def _door(x_mm: float, y_mm: float, thickness_mm: float, *, yaw_deg: float = 0.0) -> TaskSpec:
    """A 400 mm tall, 300 mm wide door with a bar handle 20 mm off its face, pulled 80 deg open
    from the front: at (0, 150 mm), 10 mm thick, it is the S2b cabinet door; at (-140, 290 mm),
    20 mm thick, the hotel hero's room door where its plan stands."""
    door = ObjectInstance.of(
        "door",
        "hinged_lid",
        "Door",
        ObjectPose(x_mm, y_mm, yaw_deg),
        orientation="door",
        width_mm=400,
        depth_mm=300,
        thickness_mm=thickness_mm,
        max_angle_deg=110 if thickness_mm < 15 else 90,
        open_torque_nm=0.3,
        damping_nms=0.05,
        feature="handle",
        feature_mm=20,
    )
    pull = Step.of(
        "open_door",
        "hook_pull",
        "door",
        "handle",
        "Pull the door open by its handle",
        [SuccessCriterion("angle_deg", at_least=80.0)],
        params={"speed_mm_s": 80, "approach": "front"},
    )
    return TaskSpec("door", "", (door,), (pull,), source="test")


def _door_ctx(spec: TaskSpec, tool: Any, task: Any) -> tuple[StepContext, mujoco.MjData]:
    scene = build_studio_scene(Setup.single(tool, spec.step_ids), spec, 0, task=task)
    data = mujoco.MjData(scene.model)
    mujoco.mj_kinematics(scene.model, data)
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
    )
    return ctx, data


def _pool() -> dict[str, Any]:
    from rtt.studio.search import default_candidates

    return {f"{c.spec.model}/{c.spec.primitive}": c.spec for c in default_candidates()}


def test_a_door_bar_front_grip_points_the_tool_at_the_face_and_closes_across_the_bar() -> None:
    """The pose math: at the pinch's 90 deg pitch, the yaw points the tool along the door's inward
    face normal and the quarter-turn roll lays the closing axis (the tool's y) horizontal along the
    face, across the vertical bar; turning the yaw with the door keeps the tool normal to the face.
    The yaw and roll are the equivalents nearest the wrist's."""
    from rtt.studio.primitives._common import rotate
    from rtt.studio.primitives.hook_pull import PINCH_PITCH, bar_roll, bar_yaw
    from rtt.studio.scene import wrist_rotation

    axis = np.array([0.0, 0.0, -1.0])  # a door's hinge: positive opens it clockwise from above
    up, down = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    for door_yaw in (0.0, 30.0, -90.0, 180.0):
        out0 = _rz(math.radians(door_yaw)) @ np.array([0.0, -1.0, 0.0])
        for angle in (0.0, 40.0, 85.0):
            n_out = rotate(out0, axis, math.radians(angle))
            for yaw_now, roll_now in ((0.0, 0.0), (5.0, 2.0), (-2.0, -3.5)):
                yaw, roll = bar_yaw(n_out, yaw_now), bar_roll(roll_now)
                assert abs(yaw - yaw_now) <= math.pi + 1e-9 and abs(roll - roll_now) <= 0.5 * math.pi + 1e-9
                r = wrist_rotation(yaw, roll, PINCH_PITCH)
                tool_axis, closing = r @ down, r @ np.array([0.0, 1.0, 0.0])
                assert np.allclose(tool_axis, -n_out, atol=1e-9), (door_yaw, angle, tool_axis)
                assert abs(float(closing @ up)) < 1e-9 and abs(float(closing @ n_out)) < 1e-9
                for s in (0.3, 1.2):  # the heading turns with the door
                    turned = wrist_rotation(yaw + s * float(axis[2]), roll, PINCH_PITCH) @ down
                    assert np.allclose(turned, -rotate(n_out, axis, s), atol=1e-9)


def test_a_door_bar_front_grip_stops_its_tips_4_mm_off_the_face() -> None:
    """The tips stand 25 mm out from the bar's front, then go in past it by the bar's depth off
    the face (its clearance and its diameter) less 4 mm: they stop 4 mm short of the door's face."""
    from rtt.studio.primitives.hook_pull import BAR_STANDOFF, bar_depth, bar_points

    info = {"feature_m": 0.020, "bar_radius_m": 0.005}
    depth = bar_depth(info)
    assert depth == pytest.approx(0.026)
    n_out = np.array([0.6, -0.8, 0.0])
    bar = np.array([0.1, 0.2, 0.35])  # the bar's centre at mid-height
    drop = 0.0125  # the tips beyond the TCP along the tool axis
    stand, engaged = bar_points(bar, n_out, 0.005, depth, drop)
    front, face = bar + 0.005 * n_out, bar - 0.025 * n_out
    for tcp in (stand, engaged):
        assert np.allclose(np.cross(tcp - bar, n_out), 0.0, atol=1e-12)  # on the bar's normal at mid-height
    assert float((stand - drop * n_out - front) @ n_out) == pytest.approx(BAR_STANDOFF)
    assert float((engaged - drop * n_out - face) @ n_out) == pytest.approx(0.004)
    assert BAR_STANDOFF == pytest.approx(0.025)


def test_only_two_finger_grippers_whose_fingers_clear_the_bar_grip_it_from_the_front(task: Any) -> None:
    """Eligibility, as for a lid's lip: a two-finger gripper whose calibrated open aperture clears
    the 10 mm bar by 6 mm a side and whose shut aperture is under it. The servo-110 and the adaptive
    two-finger grippers do; a cup, a three-finger hand and a tool without a pair of fingers about its
    axis (the angular-jaw gripper's jaws close off its axis) do not; and a lid's bar never does."""
    from rtt.studio.primitives._common import tool_of
    from rtt.studio.primitives.hook_pull import bar_fits, bar_pinch_fits

    assert bar_pinch_fits(0.0669, -0.0012, 0.010) and bar_pinch_fits(0.022, 0.0, 0.010)
    assert not bar_pinch_fits(0.0219, 0.0, 0.010)  # opens less than 6 mm a side
    assert not bar_pinch_fits(0.040, 0.010, 0.010)  # does not close under the bar
    assert not bar_pinch_fits(None, None, 0.010)
    pool = _pool()
    spec = _door(0.0, 150.0, 10)
    for key, fits in (
        ("gripper-2f-servo-110/pinch", True),
        ("adaptive-2f/pinch", True),
        ("gripper-angular-jaw-hobby/pinch", False),
        ("adaptive-3f/pinch", False),
        ("cup-25mm/seal", False),
    ):
        ctx, _ = _door_ctx(spec, pool[key], task)
        assert bar_fits(ctx, tool_of(ctx)) is fits, key
    full = _lip_and_flat("handle")
    lid = TaskSpec("lid", "", (full.objects[0],), (full.steps[0],), source="test")
    ctx, _ = _door_ctx(lid, pool["gripper-2f-servo-110/pinch"], task)
    assert not bar_fits(ctx, tool_of(ctx))


def test_a_door_bar_front_grip_is_made_only_where_the_palm_reaches_it(task: Any) -> None:
    """The front grip puts the palm the tool's length out from the bar along the face normal, and
    a door opened toward a table edge carries it there: the grip, the whole pull (the heading turning
    with the door) and the draw back are checked against the palm's travel less 12 mm, in the
    controller and in the search's tilted poses alike. The S2b door at (0, 150 mm) is gripped from the
    front all the way to its goal (88 deg). The hotel door at x -140 mm is not (by 80 deg the palm
    would be past -350 mm): that step is not attempted and fails out of reach, and a tool that does not
    fit the bar is not tried either. Nothing grips a door's bar from above."""
    import rtt.studio.primitives.hook_pull as hp
    from rtt.studio.search import tilted_poses

    pool = _pool()
    cases = (
        (_door(0.0, 150.0, 10), "adaptive-2f/pinch", "bar"),
        (_door(0.0, 150.0, 10), "gripper-angular-jaw-hobby/pinch", "unfit"),
        (_door(-140.0, 290.0, 20), "gripper-2f-servo-110/pinch", "far"),
        (_door(-140.0, 290.0, 20), "adaptive-2f/pinch", "far"),
    )
    for spec, key, want in cases:
        setup = Setup.single(pool[key], spec.step_ids)
        ctx, data = _door_ctx(spec, pool[key], task)
        plan = hp.HookPullController().plan(ctx, ctx.scene.model, data)
        run = plan.values["run"]
        poses = tilted_poses(setup, spec.steps[0], 0, ctx.scene, data, task, ctx.scene.active.tcp_wrist)
        assert bool(poses) is (want == "bar"), (key, plan.note)
        if want == "bar":
            assert run.get("mode") == "bar" and "grip the bar from the front" in plan.note, plan.note
            pitches = {round(math.degrees(p), 6) for _, _, _, p in poses}
            yaws = [y for _, y, _, _ in poses]
            assert pitches == {90.0} and math.degrees(yaws[0] - yaws[-1]) == pytest.approx(88.0, abs=0.5)
            continue
        assert "idle" in run and plan.duration_s == pytest.approx(0.2), (key, plan.note)  # not attempted
        values = plan.values["schedule"].values
        if want == "far":
            assert values["out_of_reach"] == 1.0 and values["reach_what"] == "the front grip's pull", key
            assert values["reach_x_mm"] < -338.0 and values["reach_margin_mm"] == pytest.approx(12.0), key
        else:
            assert values["no_front_grip"] == 1.0, key
            assert "no pair of fingers either side of its axis" in values["front_grip_why"], key
