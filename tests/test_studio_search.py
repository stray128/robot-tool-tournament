"""Track S5: capability pruning, screening, combination search, the runner and the recommender.

Physics here uses S1's ``part`` and ``bin`` templates, the test-only stand-ins of
``tests/studio_fixtures.py`` and the S1 reference controllers (in-process overrides), so the tests
are fast and independent of S2's tuning. Reference-controller results are never evidence: the
recommender refuses them, and the ranking rules are tested on hand-built results.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

import numpy as np
import pytest

from rtt.effectors.types import PRODUCT_MODE_LABEL, ContractError, find_brand_tokens
from rtt.studio import reference
from rtt.studio.episode import StudioCondition, StudioJob, run_task
from rtt.studio.plan import override_primitive
from rtt.studio.recommend import (
    SIMPLICITY_MARGIN,
    eligibility,
    rank_setups,
    recommend_setups,
    sensitivity,
)
from rtt.studio.runner import WorkerPool, condition_grid, run_full_tasks, run_segments
from rtt.studio.search import (
    ACQUIRE,
    BracketGeometry,
    acquire_primitive,
    bracket_search,
    candidate_of,
    choose_finalists,
    combine,
    default_candidates,
    hold_chains,
    last_link_intrusion,
    prune,
    run_task_tournament,
    screen,
    sub_task,
    swept_volume_problems,
)
from rtt.studio.types import (
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
    StepOutcome,
    SuccessCriterion,
    TaskResult,
    TaskSpec,
)
from studio_fixtures import cup, gripper, preset, two_press_task

UPRIGHT = [SuccessCriterion("tilt_deg", at_most=30.0)]


def _jaw():
    return preset("parallel_jaw", "stock", "jaw")


def _arm():
    return preset("soft_arm", "tentacle-12", "wrap")


def _hand():
    return preset("catalog_hand", "hand-5f-18act", "power")


def _pick_task(width_mm: float = 30.0) -> TaskSpec:
    objects = (
        ObjectInstance.of(
            "small",
            "part",
            "Small box",
            ObjectPose(100.0, 0.0, 0.0),
            size_x_mm=30,
            size_y_mm=30,
            size_z_mm=40,
        ),
        ObjectInstance.of(
            "wide",
            "part",
            "Wide box",
            ObjectPose(-100.0, 60.0, 0.0),
            size_x_mm=120,
            size_y_mm=width_mm,
            size_z_mm=30,
            mass_g=80,
        ),
        ObjectInstance.of("tote", "bin", "Tote", ObjectPose(0.0, -200.0, 0.0)),
    )
    lift = [SuccessCriterion("lift_mm", at_least=5.0)]
    steps = (
        Step.of("pick_small", "grasp", "small", "part", "Pick the small box", lift),
        Step.of("pick_wide", "grasp", "wide", "part", "Pick the wide box", lift),
        Step.of("press_small", "push", "small", "part", "Press the small box", UPRIGHT),
    )
    return TaskSpec("Picks and a press", "Synthetic S5 task.", objects, steps, source="test")


# --------------------------------------------------------------------------- chains and pruning


def test_hold_chains_group_held_steps_and_sub_tasks_keep_every_object() -> None:
    objs = (
        ObjectInstance.of("box", "part", "Box", ObjectPose(80.0, 0.0, 0.0)),
        ObjectInstance.of("other", "part", "Other", ObjectPose(-80.0, 0.0, 0.0)),
    )
    lift = [SuccessCriterion("lift_mm", at_least=5.0)]
    steps = (
        Step.of("a", "push", "other", "part", "Press the other box", UPRIGHT),
        Step.of("b", "grasp", "box", "part", "Hold the box", lift, hold_after=True, after=("a",)),
        Step.of("c", "push", "box", "part", "Press it down", UPRIGHT, after=("b",)),
        Step.of("d", "push", "other", "part", "Press the other again", UPRIGHT, after=("c",)),
    )
    task = TaskSpec("Chains", "", objs, steps, source="test")
    assert hold_chains(task) == (("a",), ("b", "c"), ("d",))
    sub = sub_task(task, ("b", "c"))
    assert sub.step_ids == ("b", "c") and [o.name for o in sub.objects] == ["box", "other"]
    assert sub.step("b").after == () and sub.step("c").after == ("b",)


def test_pruning_lists_a_reason_for_every_left_out_candidate() -> None:
    task = _pick_task(width_mm=90.0)
    cands = [candidate_of(s, "preset") for s in (_jaw(), _arm(), cup(), gripper(), _hand())]
    eligible, pruned, infos = prune(task, cands, payload_kg=0.6)
    reasons = {(p.display_name, p.step_id): p.reason for p in pruned}
    assert "frozen v2 episode" in reasons[(_jaw().display_name, None)]
    assert (
        "weighs" in reasons[(_hand().display_name, None)]
        and "[assumed]" in reasons[(_hand().display_name, None)]
    )
    # ruling 4: the gripper fits the first part it picks but not the 90 mm one
    gr = gripper().display_name
    assert (gr, "pick_small") not in reasons
    assert "wide" in reasons[(gr, "pick_wide")] and "opens" in reasons[(gr, "pick_wide")]
    # a slack soft arm cannot push (chain 2), a suction cup is too wide for nothing here
    assert "soft arm" in reasons[(_arm().display_name, "press_small")]
    names = {j: {c.name for c in cs} for j, cs in eligible.items()}
    assert cup().display_name in names[0] and cup().display_name in names[1]
    assert infos[gripper().effector_id].max_opening_mm == pytest.approx(52.0, abs=0.5)
    for p in pruned:
        assert p.reason and not find_brand_tokens(p.reason)


def test_hook_pull_prunes_a_cup_and_a_too_wide_cup_is_pruned_from_a_narrow_face() -> None:
    lid = ObjectInstance.of("lid", "hinged_lid", "Lid", ObjectPose(0.0, 0.0, 0.0), width_mm=150, depth_mm=100)
    narrow = ObjectInstance.of(
        "pin", "part", "Pin", ObjectPose(120.0, 0.0, 0.0), shape="cylinder", size_x_mm=12, size_z_mm=40
    )
    steps = (
        Step.of(
            "open", "hook_pull", "lid", "handle", "Open the lid", [SuccessCriterion("angle_deg", at_least=30)]
        ),
        Step.of("pick", "grasp", "pin", "part", "Pick the pin", [SuccessCriterion("lift_mm", at_least=5)]),
    )
    task = TaskSpec("Lid and pin", "", (lid, narrow), steps, source="test")
    big = preset("suction", "cup-25mm", "seal")
    _, pruned, _ = prune(task, [candidate_of(cup(), "preset"), candidate_of(big, "preset")])
    reasons = {(p.display_name, p.step_id): p.reason for p in pruned}
    assert "no hook" in reasons[(cup().display_name, "open")]
    assert "25 mm cup is wider than the 12 mm" in reasons[(big.display_name, "pick")]


@dataclass(frozen=True)
class _PressAsPick:
    """A test-only stand-in for a pick controller: the S1 reference press, registered for grasp."""

    primitive: PrimitiveKind = PrimitiveKind.GRASP
    name: str = "test_press_as_pick"

    def plan(self, ctx, model, data):  # type: ignore[no-untyped-def]
        return reference.ReferenceController(PrimitiveKind.PUSH).plan(ctx, model, data)

    def command(self, t, ctx, plan, model, data):  # type: ignore[no-untyped-def]
        return reference.ReferenceController(PrimitiveKind.PUSH).command(t, ctx, plan, model, data)


def test_acquire_class_every_acquire_tool_competes_and_the_screen_records_its_primitive() -> None:
    task = _pick_task()
    cands = [candidate_of(s, "preset") for s in (cup(), gripper())]
    assert all(c.capabilities >= ACQUIRE for c in cands)
    assert (
        acquire_primitive(cup()) is PrimitiveKind.SUCTION and acquire_primitive(_arm()) is PrimitiveKind.WRAP
    )
    pred = reference.ReferencePredicate(PrimitiveKind.GRASP)
    with override_primitive(_PressAsPick(), pred, controllers="reference"), WorkerPool(1) as pool:
        sc = screen(task, cands, condition_grid(task, 1), pool=pool, controllers="reference")
    used = {(c.effector_id, c.step_id): c.primitive_used for c in sc.cells.values()}
    assert used[(cup().effector_id, "pick_small")] == "suction"
    assert used[(gripper().effector_id, "pick_small")] == "grasp"
    cell = sc.cells[(cup().effector_id, "pick_small")]
    assert cell.primitive == "grasp" and cell.fidelity == "approximation" and cell.caveats
    assert sc.cells[(gripper().effector_id, "press_small")].context == "carried"
    matrix = sc.matrix()
    assert matrix["steps"] == ["pick_small", "pick_wide", "press_small"]
    assert {r["display_name"] for r in matrix["rows"]} == {c.name for c in cands}


# --------------------------------------------------------------------------- runner


def test_run_segments_is_run_task_and_carries_the_world_state() -> None:
    task = two_press_task()
    setup = Setup.single(gripper(), task.step_ids)
    cond = StudioCondition("c1", 1, {"carton": (1.0, -1.0)})
    ref = run_task(StudioJob(setup, task, cond, controllers="reference"))
    res, state = run_segments(setup, task, cond, controllers="reference")
    assert res.traj_sha == ref.traj_sha and [s.to_dict() for s in res.steps] == [
        s.to_dict() for s in ref.steps
    ]
    assert res.fairness_sha == ref.fairness_sha and state is not None
    sub = sub_task(task, ("press_flat",))
    one = Setup.single(cup(), sub.step_ids)
    a, _ = run_segments(one, sub, cond, controllers="reference", initial=state)
    b, _ = run_segments(one, sub, cond, controllers="reference", initial=state)
    assert a.traj_sha == b.traj_sha
    moved = dict(state.qpos)
    q = list(moved["obj_flat_free"])
    q[0] += 0.03
    moved["obj_flat_free"] = tuple(q)
    c, _ = run_segments(one, sub, cond, controllers="reference", initial=replace(state, qpos=moved))
    assert c.traj_sha != a.traj_sha


def test_condition_grid_is_deterministic_and_bounded() -> None:
    task = two_press_task()
    a, b = condition_grid(task, 6), condition_grid(task, 6)
    assert [(c.condition_id, c.seed, dict(c.aim_offsets_mm)) for c in a] == [
        (c.condition_id, c.seed, dict(c.aim_offsets_mm)) for c in b
    ]
    assert a[0].condition_id == "nominal" and not a[0].aim_offsets_mm
    assert all(abs(v) <= 2.0 for c in a for xy in c.aim_offsets_mm.values() for v in xy)
    assert len({c.seed for c in a}) == 6
    with pytest.raises(ValueError):
        condition_grid(task, 0)


def test_determinism_two_in_process_runs_and_a_spawned_worker_agree() -> None:
    """G3: the same traj_sha in-process twice and on a spawned worker pool."""
    task = two_press_task()
    setups = [
        Setup.single(gripper(), task.step_ids),
        Setup.tool_change({"press_carton": cup(), "press_flat": gripper()}),
    ]
    grid = condition_grid(task, 2)
    with WorkerPool(1) as pool:
        one = run_full_tasks(setups, task, grid, pool=pool, controllers="reference")
        two = run_full_tasks(setups, task, grid, pool=pool, controllers="reference")
    with WorkerPool(2) as pool:
        spawned = run_full_tasks(setups, task, grid, pool=pool, controllers="reference")
    shas = {k: [r.traj_sha for r in v] for k, v in one.items()}
    assert shas == {k: [r.traj_sha for r in v] for k, v in two.items()}
    assert shas == {k: [r.traj_sha for r in v] for k, v in spawned.items()}


# --------------------------------------------------------------------------- multi-tool heads


def test_bracket_search_finds_a_geometry_within_the_head_mount_bounds() -> None:
    task = two_press_task()
    geo, reasons = bracket_search((_hand(), gripper()), {"press_carton": 0, "press_flat": 1}, task)
    assert isinstance(geo, BracketGeometry) and not reasons
    a, b = geo.mounts
    assert abs(a.angle_deg - b.angle_deg) >= 45 and a.lateral_mm == 0 == b.lateral_mm
    assert 40 <= a.standoff_mm <= 200 and geo.clearance_mm >= 5.0
    assert geo.carried_kg <= 3.0 and geo.stack_mass_kg > 0


def test_swept_volume_check_rejects_a_head_that_sweeps_through_an_object_or_the_arm() -> None:
    task = two_press_task()
    heads = [(_hand(), HeadMount(-45.0, 70.0)), (gripper(), HeadMount(45.0, 70.0))]
    setup = Setup.multi_head(heads, {"press_carton": 0, "press_flat": 1})
    problems, gap, _ = swept_volume_problems(setup, task)
    assert problems == [] and gap > 0.005
    # a tall bin right beside the second press: the idle head swings into it
    objs = (
        *task.objects[:2],
        ObjectInstance.of("wall", "bin", "Tall bin", ObjectPose(-100.0, 230.0, 0.0), height_mm=300),
    )
    blocked = TaskSpec("Blocked", "", objs, task.steps, source="test")
    problems, _, _ = swept_volume_problems(setup, blocked)
    assert any("wall" in p for p in problems)
    # nothing may enter the arm's last link above the flange
    assert last_link_intrusion(np.array([[0.0, 0.0, -0.05, 0.01]])) == 0.0  # well below the face
    assert last_link_intrusion(np.array([[0.10, 0.0, 0.02, 0.01]])) == 0.0  # beside the link
    assert last_link_intrusion(np.array([[0.02, 0.0, -0.015, 0.012]])) == pytest.approx(0.009)
    heavy, reasons = bracket_search(
        (_hand(), gripper()), {"press_carton": 0, "press_flat": 1}, task, payload_kg=0.8
    )
    assert heavy is None and any("the arm carries at most" in r for r in reasons)


# --------------------------------------------------------------------------- combinations and the tournament


def test_combination_search_scores_every_kind_refuses_payload_and_keeps_the_best_single() -> None:
    task = two_press_task()
    cands = [candidate_of(s, "preset") for s in (gripper(), cup(), _hand())]
    with WorkerPool(1) as pool:
        sc = screen(task, cands, condition_grid(task, 1), pool=pool, controllers="reference")
    combos = combine(sc)
    assert combos.counts["single"] == 3 and combos.counts["tool_change"] >= 1
    for p in combos.plans:
        assert p.setup.problems(task) == [] and len(p.setup.tools) <= 4
        assert 0.0 <= p.predicted <= 1.0 and p.cycle_s > 0
    picks, best = choose_finalists(combos, sc, 4)
    assert best is not None and best.setup.kind is SetupKind.SINGLE
    assert any(p.setup.setup_id == best.setup.setup_id for p in picks)
    light = combine(sc, payload_kg=0.45)
    assert any("the arm carries at most" in r["reason"] for r in light.rejected)


def test_reference_tournament_runs_end_to_end_but_is_never_evidence() -> None:
    task = two_press_task()
    rep = run_task_tournament(
        task, candidates=[gripper(), cup()], workers=1, n_conditions=2, n_screen=1, controllers="reference"
    )
    d = json.loads(json.dumps(rep.to_dict()))
    assert d["label"] == PRODUCT_MODE_LABEL and d["evidence"] is False
    assert d["recommendation"]["evidence"] == "refused" and d["recommendation"]["recommendation"] is None
    assert d["recommendation"]["ranking"] == [] and d["recommendation"]["reference_results"]
    assert d["best_single"] in {f["setup"]["setup_id"] for f in d["finalists"]}
    for f in d["finalists"]:
        assert all(r["controllers"] == "reference" and r["label"] == PRODUCT_MODE_LABEL for r in f["runs"])
        assert [r["condition_id"] for r in f["runs"]] == [c["condition_id"] for c in d["conditions"]]
    assert not find_brand_tokens(json.dumps(d))
    assert set(d["constants"]) >= {"roll_torque_nm", "tool_change_s", "present_s", "arm_payload_kg"}
    assert all(v["tag"] == "assumed" for v in d["constants"].values())


def test_default_candidates_include_presets_and_designed_effectors() -> None:
    cands = default_candidates(library=False)
    sources = {c.source for c in cands}
    assert {"preset", "designed"} <= sources
    designed = [c for c in cands if c.source == "designed"]
    assert designed and not any(c.off_the_shelf for c in designed)
    assert any(c.spec.origin == "nemotron" for c in designed)
    assert len({c.effector_id for c in cands}) == len(cands)


# --------------------------------------------------------------------------- the recommender (synthetic results)

_FSHA = "f" * 64


def _outcome(
    step_id: str, ok: bool, effector_id: str, tier: str = "rigid", mode: str | None = None
) -> StepOutcome:
    return StepOutcome(
        step_id=step_id,
        primitive=PrimitiveKind.PUSH,
        object="carton",
        tool=0,
        effector_id=effector_id,
        success=ok,
        failure_mode=None if ok else (mode or "criterion_not_met"),
        quantities={},
        fidelity=FidelityTier(tier),
        caveats=("an approximation",) if tier == "approximation" else (),
        duration_s=5.0,
    )


def _result(
    setup: Setup,
    task: TaskSpec,
    oks: list[bool],
    cycle: float,
    *,
    changes: int = 0,
    tier: str = "rigid",
    seeds: list[int] | None = None,
    controllers: str = "default",
    mode: str | None = None,
) -> SetupResult:
    runs = []
    for i, ok in enumerate(oks):
        steps = (
            _outcome("press_carton", True, setup.tools[0].effector_id, tier),
            _outcome("press_flat", ok, setup.tools[-1].effector_id, tier, mode),
        )
        runs.append(
            TaskResult(
                task_id=task.task_id,
                setup_id=setup.setup_id,
                setup_kind=setup.kind,
                condition_id=f"c{i}",
                seed=(seeds or list(range(len(oks))))[i],
                steps=steps,
                cycle_time_s=cycle,
                change_time_s=6.0 * changes,
                n_tool_changes=changes,
                n_presentations=0,
                tool_mass_kg=(0.4,),
                traj_sha=f"{i:064x}",
                fairness_sha=_FSHA,
                controllers=controllers,  # type: ignore[arg-type]
            )
        )
    n_tools = len(setup.tools)
    cx = ComplexityMetrics(n_tools, 2 * n_tools, 0.4, 0.4 * n_tools + (0.45 if changes else 0.0), changes)
    return SetupResult(setup, task.task_id, tuple(runs), cx)


def _pair() -> tuple[TaskSpec, Setup, Setup]:
    task = two_press_task()
    single = Setup.single(gripper(), task.step_ids, label="Gripper")
    change = Setup.tool_change({"press_carton": cup(), "press_flat": gripper()}, label="Cup then gripper")
    return task, single, change


def test_simplicity_rule_picks_the_simpler_setup_within_the_margins() -> None:
    task, single, change = _pair()
    oks = [True] * 20
    near = [True] * 19 + [False]  # 0.95: within the 0.05 margin
    tc = _result(change, task, oks, 20.0, changes=1)
    sg = _result(single, task, near, 21.0)
    rec = recommend_setups([tc, sg], best_single_id=single.setup_id)
    assert rec["evidence"] == "accepted"
    assert rec["recommendation"]["setup_id"] == single.setup_id
    assert (
        "simpler" in rec["recommendation"]["why"] and str(SIMPLICITY_MARGIN) in rec["recommendation"]["why"]
    )
    assert rec["versus_best_single"]["same_setup"] is True
    # outside the margin the better setup wins
    worse = _result(single, task, [True] * 15 + [False] * 5, 21.0)
    assert recommend_setups([tc, worse])["recommendation"]["setup_id"] == change.setup_id
    # much slower is not "close" either
    slow = _result(single, task, oks, 30.0)
    assert next(sr.setup.setup_id for sr, _ in rank_setups([tc, slow])) == change.setup_id


def test_reference_results_are_refused_and_unfair_ones_raise() -> None:
    task, single, change = _pair()
    ref = _result(single, task, [True, True], 20.0, controllers="reference")
    rec = recommend_setups([ref])
    assert rec["evidence"] == "refused" and rec["recommendation"] is None and rec["ranking"] == []
    a = _result(single, task, [True, True], 20.0, seeds=[0, 1])
    b = _result(change, task, [True, True], 20.0, seeds=[0, 2])
    with pytest.raises(ContractError, match="unfair"):
        recommend_setups([a, b])


def test_no_winner_is_reported_and_an_approximation_winner_carries_a_rigid_runner_up() -> None:
    task, single, change = _pair()
    none = recommend_setups(
        [_result(single, task, [False] * 4, 20.0), _result(change, task, [False] * 4, 25.0, changes=1)]
    )
    assert none["recommendation"] is None and none["no_winner"]["closest"]
    assert not eligibility(_result(single, task, [False] * 4, 20.0))[0]
    approx = _result(change, task, [True] * 4, 18.0, changes=1, tier="approximation")
    rigid = _result(single, task, [True, True, False, False], 20.0)
    rec = recommend_setups([approx, rigid])
    assert rec["recommendation"]["setup_id"] == change.setup_id
    assert rec["recommendation"]["caveat"] == "approximation: verify on hardware"
    assert rec["runner_up_rigid"]["setup_id"] == single.setup_id
    reset = _result(single, task, [True, False], 20.0, mode="physics_invalid")
    assert eligibility(reset) == (False, "a run reset the simulator (physics_invalid)")


def test_sensitivity_recomputes_cycle_times_and_reports_rank_changes() -> None:
    task, single, change = _pair()
    sg = _result(single, task, [True] * 4, 30.0)
    tc = _result(change, task, [True] * 4, 27.0, changes=1)  # 21 s of steps + one 6 s change
    sens = sensitivity([sg, tc])
    assert sens["baseline"]["order"][0] == change.setup_id
    by = {(c["tool_change_s"], c["present_s"]): c for c in sens["combos"]}
    assert len(by) == 9
    assert by[(3.0, 1.5)]["recommended"] == change.setup_id
    assert by[(12.0, 1.5)]["recommended"] == single.setup_id and by[(12.0, 1.5)]["changed"]
    assert math.isclose(by[(12.0, 0.75)]["cycle_time_s"][change.setup_id], 33.0)
    assert sens["recommendation_changes"] is True
