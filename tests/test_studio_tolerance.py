"""Honest success tolerances and all-pairs penetration (Studio v2, ``docs/STUDIO.md`` sections 3.3, 4.5).

* a success bound is met within its quantity's tolerance (:data:`CRITERION_TOLERANCE`): the
  published H2 coupling pass sat at 3.9999936 mm against an "at most 4.0 mm", the thread's own
  start, and 4.0000012 mm after an unrelated physics change must still pass; a part 3 mm short
  must still fail;
* the runner measures penetration over every contact pair except the designed ones (a tool's own
  links, a template's spring-loaded pads), not only the tool on its object;
* more than 1 mm for more than 10 ms is ``physics_unreliable``; more than 1 mm for 10 ms or
  less is flagged in the step's quantities and not failed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import mujoco
import numpy as np
import pytest

from rtt.effectors.registry import get_family
from rtt.sim.task import default_task_v2
from rtt.studio import episode, fairness
from rtt.studio.episode import (
    PenetrationMonitor,
    StudioCondition,
    StudioJob,
    designed_pairs,
    run_task,
)
from rtt.studio.plan import (
    HELD_PENETRATION_MAX_MM,
    PENETRATION_TRANSIENT_MS,
    PrimitiveCommand,
    StepContext,
    StepPlan,
    StepVerdict,
    criteria_verdict,
    override_primitive,
    penetration_unreliable,
)
from rtt.studio.scene import build_studio_scene
from rtt.studio.types import (
    CRITERION_TOLERANCE,
    ObjectInstance,
    ObjectPose,
    PrimitiveKind,
    Setup,
    Step,
    SuccessCriterion,
    TaskSpec,
)
from studio_fixtures import gripper, two_press_task

# --------------------------------------------------------------------------- the tolerance table


def _insert(*criteria: SuccessCriterion) -> Step:
    return Step.of("start", "insert", "joint", "member", "Start the thread", list(criteria))


def test_the_tolerance_table_is_the_contracts() -> None:
    assert dict(CRITERION_TOLERANCE) == {
        "depth_mm": 0.1,
        "travel_mm": 0.5,
        "lift_mm": 0.5,
        "angle_deg": 0.5,
        "turns": 0.01,
        "torque_nm": 0.02,
        "force_n": 0.2,
    }
    with pytest.raises(TypeError):
        CRITERION_TOLERANCE["depth_mm"] = 1.0  # type: ignore[index]


def test_insert_depth_at_the_thread_start_passes_within_0_1_mm() -> None:
    """The H2 coupling: the thread starts at 4.0 mm; the run read 3.9999936, then 4.0000012."""
    step = _insert(SuccessCriterion("depth_mm", at_most=4.0))
    for depth in (3.9999936, 4.0, 4.0000012, 4.05):
        v = criteria_verdict(step, {"depth_mm": depth})
        assert v.success and v.failure_mode is None, depth
    for depth in (4.15, 4.5, 6.0):
        v = criteria_verdict(step, {"depth_mm": depth})
        assert not v.success and v.failure_mode == "criterion_not_met", depth


def test_a_part_3_mm_short_still_fails() -> None:
    """The tolerance never hides a real miss, on any quantity that has one."""
    goals = {
        "depth_mm": 20.0,
        "travel_mm": 120.0,
        "lift_mm": 50.0,
        "angle_deg": 85.0,
        "force_n": 5.0,
        "torque_nm": 3.0,
    }
    shorts = {
        "depth_mm": 3.0,
        "travel_mm": 3.0,
        "lift_mm": 3.0,
        "angle_deg": 3.0,
        "force_n": 1.0,
        "torque_nm": 0.5,
    }
    for quantity, goal in goals.items():
        step = _insert(SuccessCriterion(quantity, at_least=goal))
        near = criteria_verdict(step, {quantity: goal - 0.5 * CRITERION_TOLERANCE[quantity]})
        assert near.success, quantity
        short = criteria_verdict(step, {quantity: goal - shorts[quantity]})
        assert not short.success and short.failure_mode == "criterion_not_met", quantity
    step = _insert(SuccessCriterion("depth_mm", at_least=20.0))
    short = criteria_verdict(step, {"depth_mm": 17.0})
    assert not short.success and short.failure_mode == "criterion_not_met"


def test_a_quantity_without_a_tolerance_is_compared_exactly() -> None:
    for quantity in ("hold_s", "in_target", "tilt_deg", "slip_deg", "slip_mm", "lateral_force_n", "jam"):
        assert quantity not in CRITERION_TOLERANCE
        assert SuccessCriterion(quantity, at_least=1.0).tolerance == 0.0
    step = _insert(SuccessCriterion("hold_s", at_least=2.0))
    assert criteria_verdict(step, {"hold_s": 2.0}).success
    assert not criteria_verdict(step, {"hold_s": 1.9999}).success
    step = _insert(SuccessCriterion("tilt_deg", at_most=5.0))
    assert not criteria_verdict(step, {"tilt_deg": 5.0001}).success


def test_the_tolerance_applies_to_both_bounds_and_never_to_a_missing_value() -> None:
    step = _insert(SuccessCriterion("travel_mm", at_least=100.0, at_most=110.0))
    assert criteria_verdict(step, {"travel_mm": 99.6}).success
    assert criteria_verdict(step, {"travel_mm": 110.4}).success
    assert not criteria_verdict(step, {"travel_mm": 99.4}).success
    assert not criteria_verdict(step, {"travel_mm": 110.6}).success
    assert not criteria_verdict(step, {}).success
    assert not criteria_verdict(step, {"travel_mm": float("nan")}).success


@pytest.mark.parametrize("member", ["nut", "coupling"])
def test_a_thread_start_in_physics_meets_an_at_most_bound_set_at_the_thread_start(member: str) -> None:
    """The H2 coupling, in physics. The thread starts at the lead, 4.0 mm for the default 1 mm
    chamfer, and an insert pushes the member onto it: the run reads 4.00000005 mm (nut) and
    4.00000015 mm (coupling), a hair past an "at most 4.0 mm" that an exact comparison fails."""
    obj = ObjectInstance.of("joint", "threaded_joint", "Joint", ObjectPose(0.0, 80.0, 0.0), member=member)
    step = _insert(SuccessCriterion("depth_mm", at_most=4.0))
    spec = TaskSpec("thread start", "", (obj,), (step,), source="test")
    setup = Setup.single(gripper(), spec.step_ids)
    out = run_task(StudioJob(setup, spec, StudioCondition("tol", 0))).steps[0]
    depth = out.quantities["depth_mm"]
    assert 4.0 - 0.01 <= depth <= 4.0 + CRITERION_TOLERANCE["depth_mm"], depth
    assert out.success and out.failure_mode is None, (out.failure_mode, dict(out.quantities))


# --------------------------------------------------------------------------- the monitor, unit level

_XML = """
<mujoco><option timestep="0.002"/>
<worldbody>
  <geom name="table" type="box" size="1 1 0.05" pos="0 0 -0.05"/>
  <body name="a" pos="0 0 0.0495"><freejoint name="ja"/><geom name="ga" type="box" size="0.02 0.02 0.0495"/></body>
  <body name="b" pos="0 0 0.17"><freejoint name="jb"/><geom name="gb" type="box" size="0.02 0.02 0.02"/></body>
</worldbody></mujoco>
"""
#: two boxes that overlap by 5 mm: a pair a template or a tool makes by design
_LINKS = """
<mujoco><worldbody>
  <body name="link1" pos="0 0 0.5"><freejoint/><geom name="l1" type="box" size="0.02 0.02 0.02"/></body>
  <body name="link2" pos="0 0 0.465"><freejoint/><geom name="l2" type="box" size="0.02 0.02 0.02"/></body>
</worldbody></mujoco>
"""


def _world() -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_string(_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _place(m: mujoco.MjModel, d: mujoco.MjData, a_sink_mm: float, b_over_a_mm: float) -> None:
    """Box ``a`` sinks into the table by ``a_sink_mm``; box ``b`` overlaps the top of ``a`` by
    ``b_over_a_mm`` (the overlap is the contact's penetration)."""
    a_z = 0.0495 - 1e-3 * a_sink_mm
    d.qpos[m.joint("ja").qposadr[0] + 2] = a_z
    d.qpos[m.joint("jb").qposadr[0] + 2] = a_z + 0.0495 + 0.02 - 1e-3 * b_over_a_mm
    mujoco.mj_forward(m, d)


def test_object_object_penetration_is_measured() -> None:
    m, d = _world()
    _place(m, d, a_sink_mm=0.0, b_over_a_mm=2.0)  # b into a: object against object
    mon = PenetrationMonitor(m)
    mon.observe(d)
    assert mon.peak_m * 1000.0 == pytest.approx(2.0, abs=1e-6)
    assert mon.quantities(0.002)["penetration_mm"] == pytest.approx(2.0, abs=1e-6)


def test_object_wall_and_object_fixture_penetration_are_measured() -> None:
    m, d = _world()
    _place(m, d, a_sink_mm=3.0, b_over_a_mm=0.0)  # a into the table (a fixture, body 0)
    mon = PenetrationMonitor(m)
    mon.observe(d)
    assert mon.peak_m * 1000.0 == pytest.approx(3.0, abs=1e-6)


def test_the_peak_is_the_deepest_pair_and_resting_contact_is_not_a_penetration() -> None:
    m, d = _world()
    _place(m, d, a_sink_mm=1.0, b_over_a_mm=2.5)
    mon = PenetrationMonitor(m)
    mon.observe(d)
    assert mon.peak_m * 1000.0 == pytest.approx(2.5, abs=1e-6)
    m, d = _world()
    _place(m, d, a_sink_mm=0.0, b_over_a_mm=0.0)  # touching exactly: no overlap
    mon = PenetrationMonitor(m)
    mon.observe(d)
    assert mon.peak_m == 0.0 and mon.longest == 0


def test_designed_pairs_are_not_counted() -> None:
    m = mujoco.MjModel.from_xml_string(_LINKS)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    counted = PenetrationMonitor(m)
    counted.observe(d)
    assert counted.peak_m * 1000.0 == pytest.approx(5.0, abs=1e-6)
    designed = np.zeros((m.ngeom, m.ngeom), dtype=bool)
    designed[m.geom("l1").id, m.geom("l2").id] = designed[m.geom("l2").id, m.geom("l1").id] = True
    skipped = PenetrationMonitor(m, designed)
    skipped.observe(d)
    assert skipped.peak_m == 0.0 and skipped.longest == 0


# --------------------------------------------------------------------------- the transient rule


def _ticks(n_over: float, k: int, *, gap: bool = False) -> PenetrationMonitor:
    """``k`` control ticks of ``b`` ``n_over`` mm into ``a`` (optionally with one clear tick between)."""
    m, d = _world()
    mon = PenetrationMonitor(m)
    for i in range(k):
        _place(m, d, 0.0, n_over)
        mon.observe(d)
        if gap and i == k // 2:
            _place(m, d, 0.0, 0.0)
            mon.observe(d)
    return mon


def test_a_brief_impact_spike_is_flagged_not_failed() -> None:
    step = _insert(SuccessCriterion("depth_mm", at_most=4.0))
    q = _ticks(3.8, 5).quantities(0.002)  # 3.8 mm for 5 ticks = 10 ms: the limit of "brief"
    assert q["penetration_mm"] == pytest.approx(3.8, abs=1e-6)
    assert q["penetration_ms"] == pytest.approx(10.0)
    assert q["penetration_transient"] == 1.0
    assert not penetration_unreliable(q)
    v = criteria_verdict(step, {**q, "depth_mm": 4.0})
    assert v.success and v.failure_mode is None
    assert v.quantities["penetration_transient"] == 1.0 and v.quantities["penetration_mm"] > 1.0
    assert HELD_PENETRATION_MAX_MM == 1.0 and PENETRATION_TRANSIENT_MS == 10.0


def test_more_than_10_ms_over_1_mm_is_physics_unreliable() -> None:
    step = _insert(SuccessCriterion("depth_mm", at_most=4.0))
    q = _ticks(3.8, 6).quantities(0.002)  # 12 ms: the H1 carton into the tote floor
    assert q["penetration_ms"] == pytest.approx(12.0) and q["penetration_transient"] == 0.0
    assert penetration_unreliable(q)
    v = criteria_verdict(step, {**q, "depth_mm": 4.0})
    assert not v.success and v.failure_mode == "physics_unreliable"


def test_a_pair_that_stays_under_1_mm_is_neither_flagged_nor_failed() -> None:
    q = _ticks(0.9, 200).quantities(0.002)
    assert q["penetration_mm"] == pytest.approx(0.9, abs=1e-6)
    assert q["penetration_ms"] == 0.0 and q["penetration_transient"] == 0.0
    assert not penetration_unreliable(q)


def test_a_run_is_unbroken_ticks_of_one_pair() -> None:
    """A tick back under the limit ends the run (two 3-tick impacts are two transients) and a
    different pair does not extend it."""
    q = _ticks(2.0, 7, gap=True).quantities(0.002)  # 4 ticks over, one clear, 3 over
    assert q["penetration_ms"] == pytest.approx(8.0) and q["penetration_transient"] == 1.0
    assert not penetration_unreliable(q)
    m, d = _world()
    mon = PenetrationMonitor(m)
    for i in range(8):  # b into a for 4 ticks, then a into the table for 4: two pairs, 4 ticks each
        _place(m, d, 2.0 if i >= 4 else 0.0, 0.0 if i >= 4 else 2.0)
        mon.observe(d)
    q = mon.quantities(0.002)
    assert q["penetration_ms"] == pytest.approx(8.0) and not penetration_unreliable(q)


def test_without_a_measured_duration_the_peak_alone_decides() -> None:
    """The v1 rule, for a caller that is not the runner (no ``penetration_ms``)."""
    assert penetration_unreliable({"penetration_mm": 1.5})
    assert not penetration_unreliable({"penetration_mm": 1.0})
    assert not penetration_unreliable({})
    assert penetration_unreliable({"penetration_mm": float("nan")})


# --------------------------------------------------------------------------- designed pairs, in a scene


def _port_task() -> tuple[Setup, TaskSpec]:
    port = ObjectInstance.of(
        "po",
        "port_plug",
        "Port",
        ObjectPose(0.0, 80.0, 0.0),
        plug_diameter_mm=20,
        clearance_mm=0.4,
        chamfer_mm=2.0,
    )
    step = Step.of(
        "plug_in",
        "insert",
        "po",
        "plug",
        "Plug the connector in",
        [SuccessCriterion("depth_mm", at_least=14)],
    )
    spec = TaskSpec("Tolerance: port", "", (port,), (step,), source="test")
    eff = next(
        s
        for s in get_family("catalog_gripper").presets()
        if s.model == "adaptive-2f" and s.primitive == "power"
    )
    return Setup.single(eff, spec.step_ids), spec


def test_a_tools_own_links_and_a_ports_pads_are_designed_contacts() -> None:
    setup, spec = _port_task()
    scene = build_studio_scene(setup, spec, 0, task=default_task_v2())
    m = scene.model
    mask = designed_pairs(scene)
    assert mask.shape == (m.ngeom, m.ngeom) and (mask == mask.T).all()
    tool = scene.ids.effector_geoms
    assert mask[np.ix_(tool, tool)].all()  # every pair of tool geoms
    obj = scene.objects["po"]
    names = list(obj.handles.geoms)
    pads = [obj.geom_ids[i] for i, n in enumerate(names) if n.endswith(("pad0", "pad1", "pad2"))]
    assert len(pads) == 3
    plug = [obj.geom_ids[i] for i, n in enumerate(names) if "plug" in n and "pad" not in n]
    assert plug and all(mask[p, g] for p in pads for g in plug)
    # but the plug against the table, the plug against the socket wall and the tool against the
    # plug are all counted
    table = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
    walls = [obj.geom_ids[i] for i, n in enumerate(names) if "wall" in n]
    assert walls and not any(mask[g, t] for g in plug for t in table)
    assert not any(mask[g, w] for g in plug for w in walls)
    assert not any(mask[t, g] for t in tool for g in plug)
    assert designed_pairs(scene) is mask  # one mask per scene
    assert not mask.flags.writeable  # shared, so read-only
    with pytest.raises(ValueError, match="read-only"):
        mask[0, 0] = True


# --------------------------------------------------------------------------- the runner


@dataclass(frozen=True)
class _Hold:
    """A test controller: the wrist stays at home for ``duration_s`` with the head open."""

    duration_s: float = 0.5
    primitive: PrimitiveKind = PrimitiveKind.PUSH
    name: str = "hold"

    def plan(self, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData) -> StepPlan:
        return StepPlan(self.duration_s)

    def command(
        self, t: float, ctx: StepContext, plan: StepPlan, model: mujoco.MjModel, data: mujoco.MjData
    ) -> PrimitiveCommand:
        return PrimitiveCommand(
            np.append(ctx.wrist_start, ctx.wrist_pitch_start),
            np.asarray(ctx.head.build.ctrl_open, dtype=np.float64),
        )


@dataclass
class _Drive:
    """A test predicate that sinks the first step's part into ``into`` (``"table"``: a fixture;
    ``"flat"``: the other part) by ``depth_mm`` at its first observation (``repeat``: at every
    observation, which keeps the part over 1 mm deep), then ends the step with the shared rule.

    The contact pushes a sunk part out within a few ticks whatever the depth, so one drive is an
    impact transient and a repeated one is a sustained excursion."""

    depth_mm: float
    into: str = "table"
    repeat: bool = False
    primitive: PrimitiveKind = PrimitiveKind.PUSH
    name: str = "drive"
    _done: bool = False

    def observe(
        self, t: float, ctx: StepContext, model: mujoco.MjModel, data: mujoco.MjData, acc: dict[str, float]
    ) -> None:
        acc["tilt_deg"] = 0.0  # the dummy steps ask the part to stay upright
        if (self._done and not self.repeat) or ctx.step.id != "press_carton" or self.depth_mm <= 0.0:
            return
        self._done = True
        adr = ctx.scene.objects["carton"].joint_qpos["free"]
        if self.into == "table":
            data.qpos[adr + 2] = ctx.task.physics.table_top_z + 0.030 - 1e-3 * self.depth_mm  # 60 mm high
        else:  # onto the flat item, its top overlapped by depth_mm
            flat = ctx.scene.objects["flat"]
            top = max(
                data.geom_xpos[g][2] + np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
                for g in flat.geom_ids
            )
            data.qpos[adr : adr + 2] = data.xpos[flat.body_ids[0]][:2]
            data.qpos[adr + 2] = top + 0.030 - 1e-3 * self.depth_mm
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def verdict(self, ctx: StepContext, acc: Mapping[str, float]) -> StepVerdict:
        return criteria_verdict(ctx.step, dict(acc))


def _drive(depth_mm: float, into: str, repeat: bool = False) -> tuple[Mapping[str, float], bool, str | None]:
    spec = two_press_task()
    setup = Setup.single(gripper(), ("press_carton", "press_flat"))
    with override_primitive(_Hold(), _Drive(depth_mm, into, repeat), controllers="reference"):
        r = run_task(StudioJob(setup, spec, StudioCondition("tol", 0), controllers="reference"))
    out = r.steps[0]
    return out.quantities, out.success, out.failure_mode


@pytest.mark.parametrize("into", ["table", "flat"])
def test_the_runner_measures_a_part_driven_into_a_fixture_or_another_part(into: str) -> None:
    q, _, _ = _drive(4.0, into)
    assert q["penetration_mm"] > 3.0, q  # no tool anywhere near: only the all-pairs rule can see this
    assert q["penetration_ms"] > 0.0


def test_a_part_resting_in_a_scene_reads_no_penetration_and_no_flag() -> None:
    q, success, mode = _drive(0.0, "table")
    assert q["penetration_mm"] < 0.5 and q["penetration_ms"] == 0.0 and q["penetration_transient"] == 0.0
    assert success and mode is None


@pytest.mark.parametrize("into", ["table", "flat"])
def test_the_runner_flags_one_impact_and_does_not_fail_it(into: str) -> None:
    q, success, mode = _drive(3.0, into)  # pushed out again within a few ticks
    assert q["penetration_mm"] > 1.0 and 0.0 < q["penetration_ms"] <= PENETRATION_TRANSIENT_MS, q
    assert q["penetration_transient"] == 1.0
    assert success and mode is None


@pytest.mark.parametrize("into", ["table", "flat"])
def test_the_runner_fails_a_penetration_that_is_held_past_10_ms(into: str) -> None:
    q, success, mode = _drive(6.0, into, repeat=True)
    assert q["penetration_ms"] > PENETRATION_TRANSIENT_MS and q["penetration_transient"] == 0.0, q
    assert not success and mode == "physics_unreliable"


def test_the_monitor_leaves_the_trajectory_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measuring is read-only: with ``PenetrationMonitor.observe`` a no-op the run has the same
    ``traj_sha`` and the same outcome, and only the penetration quantities differ."""

    def run(depth_mm: float) -> tuple[str, Mapping[str, float]]:
        spec = two_press_task()
        setup = Setup.single(gripper(), ("press_carton", "press_flat"))
        with override_primitive(_Hold(), _Drive(depth_mm, "flat"), controllers="reference"):
            r = run_task(StudioJob(setup, spec, StudioCondition("tol", 0), controllers="reference"))
        return r.traj_sha, r.steps[0].quantities

    sha, q = run(3.0)  # a part driven into another: contacts to measure
    assert q["penetration_mm"] > 1.0 and q["penetration_ms"] > 0.0
    with monkeypatch.context() as mp:
        mp.setattr(episode.PenetrationMonitor, "observe", lambda self, data: None)
        silent_sha, silent_q = run(3.0)
    assert silent_q["penetration_mm"] == 0.0 and silent_q["penetration_ms"] == 0.0
    assert silent_sha == sha
    assert {k: v for k, v in silent_q.items() if not k.startswith("penetration")} == {
        k: v for k, v in q.items() if not k.startswith("penetration")
    }


def test_fairness_sha_covers_the_scoring_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tolerance table, the 1 mm limit and the 10 ms transient allowance are in the fairness
    record: change any of them and ``fairness_sha`` changes, so results scored under other rules
    never mix."""
    spec = two_press_task()
    scene = build_studio_scene(
        Setup.single(gripper(), ("press_carton", "press_flat")), spec, 0, task=default_task_v2()
    )
    sig = fairness.scoring_signature()
    assert sig == {
        "criterion_tolerance": dict(sorted(CRITERION_TOLERANCE.items())),
        "penetration_limit_mm": HELD_PENETRATION_MAX_MM,
        "penetration_transient_ms": PENETRATION_TRANSIENT_MS,
    }
    base = fairness.fairness_sha(scene)
    assert fairness.fairness_sha(scene) == base
    widened = MappingProxyType({**CRITERION_TOLERANCE, "depth_mm": 0.2})
    fewer = MappingProxyType({k: v for k, v in CRITERION_TOLERANCE.items() if k != "force_n"})
    shas = {base}
    for name, value in (
        ("CRITERION_TOLERANCE", widened),
        ("CRITERION_TOLERANCE", fewer),
        ("HELD_PENETRATION_MAX_MM", 2.0),
        ("PENETRATION_TRANSIENT_MS", 20.0),
    ):
        with monkeypatch.context() as mp:
            mp.setattr(fairness, name, value)
            shas.add(fairness.fairness_sha(scene))
    assert len(shas) == 5
    assert fairness.fairness_sha(scene) == base  # patches undone
