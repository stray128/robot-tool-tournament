"""Integrator rules in the Studio search and recommender (track S5, product mode).

The recommender follows published integrator practice: one gripper is the default, and a second
tool, a tool change or a two-tool head wins only when the single tool fails a step outright or the
gain clears a margin per added piece (integrator rule R01). Soft arms, tentacles and dexterous hands
are not used on articulated steps unless every rigid option fails (R03); soft arms stay out of
inserts and heavy parts (R09); setups with a soft arm or a dexterous hand rank below every rigid
setup (R09, R10). A tool's hook comes from its measured geometry, not its tags. Process steps
(tightening past hand-tight, pressing a tile into its bed) need the process tool of their kind,
and process tools only drive, start or press (R05). A vacuum frame fits a part only when the part's
face covers enough of its cups.

The ranking rules are tested on hand-built results; the pruning rules on the real presets, the
designed reference recipes and the open library's tools.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from rtt.effectors.suction import FramePreset, frame_spec
from rtt.studio.buildpack.documents import assembly_notes, bom_lines
from rtt.studio.recommend import (
    ASSUMED_CONSTANTS,
    SENSITIVITY_CHANGE_S,
    SIMPLICITY_MARGIN,
    added_pieces,
    rank_setups,
    recommend_setups,
    sensitivity,
)
from rtt.studio.rules import HOOK_MIN_MM, HOOK_THROAT_MM, PROCESS_TORQUE_NM, process_need, setup_rule_class
from rtt.studio.search import (
    N_FINALISTS,
    _library_candidates,
    candidate_of,
    capability_problem,
    choose_finalists,
    fit_problem,
    probe_tool,
    prune,
    rule_ineligible,
)
from rtt.studio.types import (
    PRESENT_S,
    STUDIO_WRIST,
    TOOL_CHANGE_S,
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
    ToolChanger,
)
from studio_fixtures import cup, gripper, preset, two_press_task

_FSHA = "f" * 64
LIFT = [SuccessCriterion("lift_mm", at_least=5.0)]
OPEN = [SuccessCriterion("angle_deg", at_least=30.0)]


def _arm():
    return preset("soft_arm", "tentacle-12", "wrap")


def _hand():
    return preset("catalog_hand", "hand-5f-18act", "power")


# --------------------------------------------------------------------------- synthetic results


def _result(
    setup: Setup,
    task: TaskSpec,
    oks: list[bool],
    cycle: float,
    *,
    changes: int = 0,
    presentations: int = 0,
) -> SetupResult:
    """``len(oks)`` runs: every step succeeds but the last, which succeeds where ``oks`` says (so
    the completion rate is the weakest step's rate)."""
    last = task.step_ids[-1]
    runs = []
    for i, ok in enumerate(oks):
        steps = tuple(
            StepOutcome(
                step_id=sid,
                primitive=PrimitiveKind.PUSH,
                object=task.step(sid).object,
                tool=setup.assignment[sid],
                effector_id=setup.effector_for(sid).effector_id,
                success=ok if sid == last else True,
                failure_mode=None if (ok or sid != last) else "criterion_not_met",
                quantities={},
                fidelity=FidelityTier.RIGID,
                duration_s=5.0,
            )
            for sid in task.step_ids
        )
        runs.append(
            TaskResult(
                task_id=task.task_id,
                setup_id=setup.setup_id,
                setup_kind=setup.kind,
                condition_id=f"c{i}",
                seed=i,
                steps=steps,
                cycle_time_s=cycle,
                change_time_s=TOOL_CHANGE_S * changes + PRESENT_S * presentations,
                n_tool_changes=changes,
                n_presentations=presentations,
                tool_mass_kg=(0.4,),
                traj_sha=f"{i:064x}",
                fairness_sha=_FSHA,
            )
        )
    n = len(setup.tools)
    cx = ComplexityMetrics(
        n, 2 * n, 0.4, 0.4 * n, changes + presentations, bracket=setup.kind is SetupKind.MULTI_HEAD
    )
    return SetupResult(setup, task.task_id, tuple(runs), cx)


def _runs(done: int, of: int) -> list[bool]:
    return [True] * done + [False] * (of - done)


def _setups() -> tuple[TaskSpec, Setup, Setup, Setup]:
    task = two_press_task()
    single = Setup.single(gripper(), task.step_ids, label="Gripper")
    change = Setup.tool_change({"press_carton": cup(), "press_flat": gripper()}, label="Cup then gripper")
    head = Setup.multi_head(
        [(cup(), HeadMount(-45.0, 70.0)), (gripper(), HeadMount(45.0, 70.0))],
        {"press_carton": 0, "press_flat": 1},
        label="Cup and gripper head",
    )
    return task, single, change, head


def _pick(results: list[SetupResult]) -> str:
    rec = recommend_setups(results)
    assert rec["evidence"] == "accepted"
    return rec["recommendation"]["setup_id"]


# --------------------------------------------------------------------------- R01: single tool by default


def test_the_simplicity_margin_is_the_published_rule() -> None:
    assert dict(SIMPLICITY_MARGIN) == {"per_extra_effector": 0.05, "step_floor": 0.5}
    _, single, change, head = _setups()
    assert (added_pieces(single), added_pieces(head), added_pieces(change)) == (0, 1, 2)
    three = Setup.tool_change({"press_carton": cup(), "press_flat": _hand()})
    assert added_pieces(three) == 2


def test_a_single_tool_wins_when_it_finishes() -> None:
    task, single, change, _ = _setups()
    tc = _result(change, task, [True] * 8, 62.0, changes=1)
    # the plan's case: 66 s against 62 s (the old 10 % cycle rule also gave the single)
    rec = recommend_setups([tc, _result(single, task, [True] * 8, 66.0)])
    assert rec["recommendation"]["setup_id"] == single.setup_id
    # 80 s against 62 s: cycle time no longer lets the more complex setup beat the single tool
    slow = _result(single, task, [True] * 8, 80.0)
    rec = recommend_setups([tc, slow])
    assert rec["recommendation"]["setup_id"] == single.setup_id
    why = rec["recommendation"]["why"]
    assert "R01" in why and "One tool finishes the job" in why
    assert "not more than 10 points better" in why and "80.0 s against 62.0 s" in why
    # the baseline is printed beside the pick; here it is the pick
    assert rec["baseline"]["setup_id"] == single.setup_id
    assert rec["versus_baseline"]["same_setup"] is True and rec["versus_baseline"]["basis"] == "baseline"
    assert [r["setup_id"] for r in rec["ranking"]] == [single.setup_id, change.setup_id]
    assert [r["baseline"] for r in rec["ranking"]] == [True, False]


def test_a_combination_wins_only_past_the_margin() -> None:
    task, single, change, head = _setups()
    # one added piece (a two-tool head, no changer): 7/8 against 6/8 is 12.5 points > 5
    rec = recommend_setups(
        [_result(single, task, _runs(6, 8), 40.0), _result(head, task, _runs(7, 8), 45.0, presentations=1)]
    )
    assert rec["recommendation"]["setup_id"] == head.setup_id
    assert "R01" in rec["recommendation"]["why"] and "more than 5 points" in rec["recommendation"]["why"]
    assert rec["baseline"]["setup_id"] == single.setup_id  # printed beside the pick
    vs = rec["versus_baseline"]
    assert vs["same_setup"] is False and vs["added_pieces"] == 1 and vs["basis"] == "margin"
    assert vs["completion_delta"] == pytest.approx(0.125) and vs["needed_gain"] == pytest.approx(0.05)
    # exactly the margin is not more than it: 20/20 against 19/20 keeps the single tool
    assert (
        _pick(
            [
                _result(single, task, _runs(19, 20), 40.0),
                _result(head, task, [True] * 20, 30.0, presentations=1),
            ]
        )
        == single.setup_id
    )
    # two added pieces (a second tool and the changer): 8/8 against 7/8 is 12.5 points > 10
    assert (
        _pick([_result(single, task, _runs(7, 8), 40.0), _result(change, task, [True] * 8, 45.0, changes=1)])
        == change.setup_id
    )
    # 16/16 against 15/16 is 6.25 points, not more than 10: the single tool stays
    rec = recommend_setups(
        [_result(single, task, _runs(15, 16), 40.0), _result(change, task, [True] * 16, 30.0, changes=1)]
    )
    assert rec["recommendation"]["setup_id"] == single.setup_id
    assert "not more than 10 points better" in rec["recommendation"]["why"]
    # among setups of more pieces the same margin applies per piece of difference: the head
    # (1 piece) is 15 points better than the single, the tool change (2 pieces) 20 points better
    # than the single but only 5 better than the head, which is not more than 5
    three = [
        _result(single, task, _runs(15, 20), 40.0),
        _result(head, task, _runs(18, 20), 41.0, presentations=1),
        _result(change, task, _runs(19, 20), 39.0, changes=1),
    ]
    order = [sr.setup.setup_id for sr, _ in rank_setups(three)]
    assert order == [head.setup_id, change.setup_id, single.setup_id]


def test_a_single_tool_that_fails_a_step_outright_loses_to_any_setup_that_completes_more() -> None:
    task, single, change, _ = _setups()
    # weakest step 6/16 (< 0.5): the tool change completes more (7/16), within the margin, and wins
    rec = recommend_setups(
        [_result(single, task, _runs(6, 16), 40.0), _result(change, task, _runs(7, 16), 50.0, changes=1)]
    )
    assert rec["recommendation"]["setup_id"] == change.setup_id
    why = rec["recommendation"]["why"]
    assert "R01" in why and "fails press_flat outright" in why
    assert rec["versus_baseline"]["basis"] == "step_floor"
    # at the floor (8/16 = 0.5) the single tool finishes the job and the same gain is too small
    assert (
        _pick(
            [_result(single, task, _runs(8, 16), 40.0), _result(change, task, _runs(9, 16), 30.0, changes=1)]
        )
        == single.setup_id
    )
    # completing no more is not enough even below the floor
    assert (
        _pick(
            [_result(single, task, _runs(6, 16), 40.0), _result(change, task, _runs(6, 16), 30.0, changes=1)]
        )
        == single.setup_id
    )


def test_r01_compares_the_best_single_with_every_more_complex_setup() -> None:
    """Regressions: R01 must not run over the heads of the same-complexity tie."""
    task, single, change, head = _setups()
    simple_single = ComplexityMetrics(1, 1, 0.2, 0.2, 0)
    single_cx = ComplexityMetrics(1, 2, 0.4, 0.4, 0)
    best = Setup.single(gripper(), task.step_ids, label="Best single")
    simpler = Setup.single(
        preset("catalog_gripper", "gripper-2f-finray", "pinch"), task.step_ids, label="Simpler"
    )
    # (a) the simpler single ties with the best one (within 0.05, under 10 % slower), but the
    # baseline is the best single: the head is only 5 points better than it, so the best single wins
    results = [
        replace(_result(best, task, _runs(18, 20), 40.0), complexity=single_cx),
        replace(_result(simpler, task, _runs(17, 20), 42.0), complexity=simple_single),
        _result(head, task, _runs(19, 20), 41.0, presentations=1),
    ]
    rec = recommend_setups(results)
    assert rec["recommendation"]["setup_id"] == best.setup_id
    why = rec["recommendation"]["why"]
    assert "One tool finishes the job (90% of the runs" in why
    assert "(Cup and gripper head, 95%) is not more than 5 points better" in why
    assert [sr.setup.label for sr, _ in rank_setups(results)] == ["Best single", head.label, "Simpler"]
    # (b) the simpler head ties with the better one, but only the better one clears the margin
    # over the single, so it is the pick, and the reason names the single it was compared with
    other = Setup.multi_head(
        [
            (preset("suction", "cup-25mm", "seal"), HeadMount(-45.0, 70.0)),
            (simpler.tools[0], HeadMount(45.0, 70.0)),
        ],
        {"press_carton": 0, "press_flat": 1},
        label="Simpler head",
    )
    results = [
        replace(_result(single, task, _runs(80, 100), 40.0), complexity=single_cx),
        replace(
            _result(head, task, _runs(86, 100), 41.0, presentations=1),
            complexity=ComplexityMetrics(2, 4, 0.4, 0.8, 1, bracket=True),
        ),
        replace(
            _result(other, task, _runs(82, 100), 42.0, presentations=1),
            complexity=ComplexityMetrics(2, 2, 0.4, 0.6, 1, bracket=True),
        ),
    ]
    rec = recommend_setups(results)
    assert rec["recommendation"]["setup_id"] == head.setup_id
    why = rec["recommendation"]["why"]
    assert "more than 5 points better than one tool (Gripper, 80%)" in why and "Simpler head" not in why
    assert rec["versus_baseline"]["basis"] == "margin" and rec["baseline"]["setup_id"] == single.setup_id
    # the chain: a tool change is kept behind a head it is not 5 points better than, and the
    # reason names that head
    results = [
        _result(single, task, _runs(70, 100), 40.0),
        _result(head, task, _runs(80, 100), 41.0, presentations=1),
        _result(change, task, _runs(85, 100), 39.0, changes=1),
    ]
    why = recommend_setups(results)["recommendation"]["why"]
    assert "more than 5 points better than one tool (Gripper, 70%)" in why
    assert "The 2-tool change (Cup then gripper, 85%) is not more than 5 points better than it" in why


def test_the_baseline_stays_the_best_single_when_the_tie_picks_a_simpler_one() -> None:
    """Single A completes 90 %, single B 86 % and is simpler, a head
    91 %. R01 keeps one tool and the same-complexity tie picks B; the baseline printed beside it is
    still A, the best single by completion, and the basis says the tie picked the simpler B."""
    task, _, _, head = _setups()
    best = Setup.single(gripper(), task.step_ids, label="Single A")
    simpler = Setup.single(
        preset("catalog_gripper", "gripper-2f-finray", "pinch"), task.step_ids, label="Single B"
    )
    results = [
        replace(_result(best, task, _runs(90, 100), 40.0), complexity=ComplexityMetrics(1, 2, 0.4, 0.4, 0)),
        replace(
            _result(simpler, task, _runs(86, 100), 42.0), complexity=ComplexityMetrics(1, 1, 0.2, 0.2, 0)
        ),
        _result(head, task, _runs(91, 100), 41.0, presentations=1),
    ]
    rec = recommend_setups(results)
    assert rec["recommendation"]["setup_id"] == simpler.setup_id
    assert rec["baseline"]["setup_id"] == best.setup_id
    vs = rec["versus_baseline"]
    assert vs["basis"] == "tie" and vs["same_setup"] is False and vs["added_pieces"] == 0
    assert vs["completion_delta"] == pytest.approx(-0.04)
    assert "Single B" in vs["note"] and "Single A" in vs["note"]
    rows = {r["setup_id"]: r for r in rec["ranking"]}
    assert rows[best.setup_id]["baseline"] is True and rows[simpler.setup_id]["baseline"] is False


def test_the_demotion_sentence_names_the_soft_arm_when_the_pick_is_one() -> None:
    """With no rigid setup, a soft arm is picked ahead of a hand that
    completes more; the sentence says hands rank below soft arms, not only below rigid setups."""
    task = two_press_task()
    hand = Setup.single(_hand(), task.step_ids, label="Hand")
    arm = Setup.single(_arm(), task.step_ids, label="Soft arm")
    rigid = Setup.single(gripper(), task.step_ids, label="Gripper")
    results = [
        _result(hand, task, [True] * 8, 30.0),
        _result(arm, task, _runs(6, 8), 30.0),
        _result(rigid, task, [False] * 8, 40.0),
    ]
    rec = recommend_setups(results)
    assert rec["recommendation"]["setup_id"] == arm.setup_id
    why = rec["recommendation"]["why"]
    assert (
        "Setups with a dexterous hand rank below every rigid setup and every setup with a soft arm "
        "(integrator rule R10), so Hand (100%) comes later." in why
    )
    assert "a soft arm is the last resort (integrator rule R09)" in why
    # a rigid pick keeps the plain sentence
    results = [_result(hand, task, [True] * 8, 30.0), _result(rigid, task, _runs(6, 8), 40.0)]
    why = recommend_setups(results)["recommendation"]["why"]
    assert "Setups with a dexterous hand rank below every rigid setup (integrator rule R10)" in why


# --------------------------------------------------------------------------- R09, R10: last resorts rank last


def test_setups_with_a_soft_arm_or_a_dexterous_hand_rank_below_every_rigid_setup() -> None:
    task = two_press_task()
    rigid = Setup.single(gripper(), task.step_ids, label="Gripper")
    hand = Setup.single(_hand(), task.step_ids, label="Hand")
    arm = Setup.single(_arm(), task.step_ids, label="Soft arm")
    results = [
        _result(hand, task, [True] * 8, 30.0),
        _result(arm, task, [True] * 8, 30.0),
        _result(rigid, task, _runs(3, 8), 40.0),
    ]
    order = [sr.setup.setup_id for sr, _ in rank_setups(results)]
    assert order == [rigid.setup_id, arm.setup_id, hand.setup_id]
    rec = recommend_setups(results)
    why = rec["recommendation"]["why"]
    assert rec["recommendation"]["setup_id"] == rigid.setup_id
    assert "R10" in why or "R09" in why
    # eligible, never preferred: with no rigid setup completing, the hand is still recommended
    rec = recommend_setups([_result(hand, task, [True] * 8, 30.0), _result(rigid, task, [False] * 8, 40.0)])
    assert rec["recommendation"]["setup_id"] == hand.setup_id
    assert "No rigid setup completed the task" in rec["recommendation"]["why"]
    assert "R10" in rec["recommendation"]["why"]
    # ... but when no rigid setup ran at all, the reason does not claim one failed
    why = recommend_setups([_result(hand, task, [True] * 8, 30.0)])["recommendation"]["why"]
    assert "No rigid setup was among the finalists, so none was run" in why
    assert "completed" not in why.split("finalists")[1]
    # a rigid head beside a hand single: the hand is printed as the baseline, ranked below by R10
    head = Setup.multi_head(
        [(cup(), HeadMount(-45.0, 70.0)), (gripper(), HeadMount(45.0, 70.0))],
        {"press_carton": 0, "press_flat": 1},
    )
    rec = recommend_setups(
        [_result(head, task, [True] * 8, 40.0, presentations=1), _result(hand, task, [True] * 8, 30.0)]
    )
    assert rec["recommendation"]["setup_id"] == head.setup_id
    assert rec["baseline"]["setup_id"] == hand.setup_id
    assert rec["versus_baseline"]["basis"] == "R10" and rec["versus_baseline"]["needed_gain"] is None


def test_the_baseline_says_why_when_no_single_tool_is_eligible() -> None:
    task, single, change, head = _setups()
    rec = recommend_setups(
        [_result(single, task, [False] * 8, 40.0), _result(head, task, _runs(5, 8), 45.0, presentations=1)]
    )
    assert rec["recommendation"]["setup_id"] == head.setup_id
    assert rec["baseline"]["setup"] is None and rec["versus_baseline"] is None
    basis = rec["baseline"]["basis"]
    assert basis.startswith("no single tool is eligible: Gripper: never completed the task")
    rec = recommend_setups([_result(change, task, [True] * 8, 40.0, changes=1)])
    assert rec["baseline"]["basis"] == "no single tool is eligible: no single tool was among the finalists"


def _plan(setup: Setup, predicted: float, *, off_the_shelf: bool = True) -> SimpleNamespace:
    eid = setup.tools[0].effector_id
    return SimpleNamespace(
        setup=setup,
        chain_tools=(eid,),
        predicted=predicted,
        key=lambda p=predicted, s=setup: (-p, -p, -p, 30.0, len(s.tools)),
        off_the_shelf=off_the_shelf,
    )


def _finalists(plans: list[SimpleNamespace], n: int = N_FINALISTS) -> tuple[list, object]:
    cands = {p.chain_tools[0]: SimpleNamespace(off_the_shelf=p.off_the_shelf) for p in plans}
    return choose_finalists(SimpleNamespace(plans=plans), SimpleNamespace(candidates=cands), n)


def test_finalists_go_rigid_first_and_keep_a_rigid_single() -> None:
    """Regression: soft-arm setups predicted best must not crowd
    the rigid tool change and head out of the finals."""
    task = two_press_task()
    ids = task.step_ids
    arm, arm2 = _arm(), preset("soft_arm", "tentacle-16", "wrap")
    finray = preset("catalog_gripper", "gripper-2f-finray", "pinch")

    def hd(a, b):  # type: ignore[no-untyped-def]
        return Setup.multi_head(
            [(a, HeadMount(-45.0, 70.0)), (b, HeadMount(45.0, 70.0))], {ids[0]: 0, ids[1]: 1}
        )

    def tc(a, b):  # type: ignore[no-untyped-def]
        return Setup.tool_change({ids[0]: a, ids[1]: b})

    plans = [
        _plan(hd(arm, cup()), 1.00),
        _plan(hd(arm2, cup()), 0.99),
        _plan(tc(arm, finray), 0.98),
        _plan(tc(cup(), gripper()), 0.95),
        _plan(hd(cup(), gripper()), 0.94),
        _plan(Setup.single(gripper(), ids), 0.50),
    ]
    picks, best = _finalists(plans)
    kinds = [(p.setup.kind, setup_rule_class(p.setup)) for p in picks]
    assert kinds[:3] == [
        (SetupKind.TOOL_CHANGE, "rigid"),
        (SetupKind.SINGLE, "rigid"),
        (SetupKind.MULTI_HEAD, "rigid"),
    ]
    assert len(picks) == 4 and kinds[3] == (SetupKind.MULTI_HEAD, "soft_arm") and picks[3].predicted == 1.0
    assert best is plans[5]
    # the reserved slot is the best rigid single from any source (a designed single counts), and
    # the best single off-the-shelf effector still runs
    designed = _plan(Setup.single(finray, ids), 0.60, off_the_shelf=False)
    picks, best = _finalists([*plans, designed])
    assert designed in picks and plans[5] in picks and best is plans[5]
    assert {p.setup.kind for p in picks} >= {SetupKind.TOOL_CHANGE, SetupKind.MULTI_HEAD}
    # a soft arm still runs ahead of rigid setups predicted never to complete
    hopeless = [_plan(tc(cup(), gripper()), 0.0), _plan(Setup.single(gripper(), ids), 0.0)]
    picks, _ = _finalists([*hopeless, _plan(Setup.single(arm, ids), 0.9)], n=2)
    assert {setup_rule_class(p.setup) for p in picks} == {"rigid", "soft_arm"}


# --------------------------------------------------------------------------- R03, R09: who may do a step


def _lid_task(*, mass_g: float = 100.0) -> TaskSpec:
    lid = ObjectInstance.of("lid", "hinged_lid", "Lid", ObjectPose(0.0, 0.0, 0.0), width_mm=150, depth_mm=100)
    box = ObjectInstance.of("box", "part", "Box", ObjectPose(150.0, 0.0, 0.0), mass_g=mass_g)
    steps = (
        Step.of("open", "hook_pull", "lid", "handle", "Open the lid", OPEN),
        Step.of("pick", "grasp", "box", "part", "Pick the box", LIFT, after=("open",)),
    )
    return TaskSpec("Lid and box", "", (lid, box), steps, source="test")


def test_soft_arms_are_ineligible_on_lid_steps() -> None:
    from rtt.effectors.designed import reference_recipes, spec_from_recipe

    task = _lid_task()
    designed = {r.kind: spec_from_recipe(r) for r in reference_recipes()}
    arm, hand, grip = (candidate_of(s, "preset") for s in (_arm(), _hand(), gripper()))
    fingered = candidate_of(designed["fingered_hand"], "designed")
    tentacle = candidate_of(designed["segmented_tentacle"], "designed")
    hook = candidate_of(designed["hook_gripper"], "designed")
    lid, pick = task.step("open"), task.step("pick")
    for c in (arm, hand, fingered, tentacle):
        reason = rule_ineligible(c, lid, task)
        assert reason is not None and 'integrator rule R03: "Lid" is a mechanism' in reason, c.name
        assert rule_ineligible(c, pick, task) is None  # a part is not articulated
    for c in (grip, hook):
        assert rule_ineligible(c, lid, task) is None
    eligible, pruned, _ = prune(task, [arm, hand, grip, fingered, tentacle, hook])
    reasons = {(p.effector_id, p.step_id): p.reason for p in pruned}
    for c in (arm, hand, fingered, tentacle):
        assert "R03" in reasons[(c.effector_id, "open")]
    assert {c.effector_id for c in eligible[0]} == {grip.effector_id, hook.effector_id}
    assert arm in eligible[1] and hand in eligible[1]  # still eligible to pick the box


def test_soft_arms_stay_out_of_inserts_and_heavy_parts() -> None:
    arm = candidate_of(_arm(), "preset")
    heavy = _lid_task(mass_g=3000.0)
    reason = rule_ineligible(arm, heavy.step("pick"), heavy)
    assert reason is not None and "R09" in reason and "2.5 kg" in reason
    assert rule_ineligible(candidate_of(gripper(), "preset"), heavy.step("pick"), heavy) is None
    port = ObjectInstance.of("port", "port_plug", "Port", ObjectPose(-150.0, 0.0, 0.0), plug_diameter_mm=8)
    plug = Step.of("plug", "insert", "port", "plug", "Plug in", [SuccessCriterion("depth_mm", at_least=10)])
    task = TaskSpec("Port", "", (port,), (plug,), source="test")
    reason = rule_ineligible(arm, plug, task)
    assert reason is not None and "R09" in reason


def test_a_soft_arm_or_a_hand_is_readmitted_when_every_rigid_option_fails() -> None:
    task = _lid_task()
    arm, hand, suction = (candidate_of(s, "preset") for s in (_arm(), _hand(), cup()))
    notes: list[dict] = []
    eligible, pruned, _ = prune(task, [arm, hand, suction], notes=notes)
    # the cup cannot hook the lid at all, so no rigid option is left: the soft arm and the hand
    # come back for that step, and the report says so
    assert {c.effector_id for c in eligible[0]} == {arm.effector_id, hand.effector_id}
    reasons = {(p.effector_id, p.step_id): p.reason for p in pruned}
    assert "no hook" in reasons[(suction.effector_id, "open")]
    assert (arm.effector_id, "open") not in reasons and (hand.effector_id, "open") not in reasons
    assert len(notes) == 1
    note = notes[0]
    assert note["rule"] == "R03" and note["steps"] == ["open"]
    assert set(note["readmitted"]) == {arm.effector_id, hand.effector_id}
    assert "every rigid option" in note["note"]
    # with a rigid option left, nothing is readmitted
    notes = []
    eligible, _, _ = prune(task, [arm, hand, suction, candidate_of(gripper(), "preset")], notes=notes)
    assert {c.effector_id for c in eligible[0]} == {gripper().effector_id} and notes == []


# --------------------------------------------------------------------------- hook by geometry


def _library(item: str):
    return next(c for c in _library_candidates() if c.item == item)


def test_a_rigid_l_hook_is_not_pruned_by_tags() -> None:
    """The library's button pusher is tagged ``push`` only, but its pad is offset on an arm that
    reaches 24 mm sideways past its shank and past the flange: an L by geometry, so it may hook a
    lid. The drawer-pull hook is tagged ``hook``, but on the flange its plate is flush with the
    flange face, so nothing can get between them: it is no hook here. A cup is never one."""
    lid = ObjectInstance.of(
        "lid", "hinged_lid", "Lid", ObjectPose(0.0, 0.0, 0.0), width_mm=150, depth_mm=100, feature="lip"
    )
    step = Step.of("open", "hook_pull", "lid", "handle", "Open the lid", OPEN)
    task = TaskSpec("Lid", "", (lid,), (step,), source="test")
    pusher, drawer_hook = _library("tool-button-pusher"), _library("tool-hook-drawer")
    assert PrimitiveKind.HOOK_PULL not in pusher.capabilities  # its tags do not say hook
    info = probe_tool(pusher.spec)
    assert info.hook_mm >= 20.0
    assert capability_problem(pusher, step, info) is None
    reason = capability_problem(drawer_hook, step, probe_tool(drawer_hook.spec))
    assert reason is not None and "measured" in reason and "flange" in reason
    eligible, pruned, _ = prune(task, [pusher, drawer_hook, candidate_of(cup(), "preset")])
    assert [c.effector_id for c in eligible[0]] == [pusher.effector_id]
    reasons = {(p.effector_id, p.step_id): p.reason for p in pruned}
    assert "no hook" in reasons[(cup().effector_id, "open")]
    # grippers and hands keep hook_pull from their fingers, whatever their measured hook
    assert capability_problem(candidate_of(gripper(), "preset"), step, probe_tool(gripper())) is None


def test_rule_reasons_use_the_object_label_as_written() -> None:
    """A label is quoted as the user wrote it, with no case change
    (an acronym, a label that starts with "The")."""
    lid = ObjectInstance.of(
        "lid", "hinged_lid", "The PCB tray lid", ObjectPose(0.0, 0.0, 0.0), width_mm=150, depth_mm=100
    )
    rack = ObjectInstance.of("rack", "part", "PCB rack", ObjectPose(150.0, 0.0, 0.0), mass_g=3000)
    steps = (
        Step.of("open", "hook_pull", "lid", "handle", "Open the lid", OPEN),
        Step.of("pick", "grasp", "rack", "part", "Pick the rack", LIFT),
    )
    task = TaskSpec("Rack", "", (lid, rack), steps, source="test")
    arm = candidate_of(_arm(), "preset")
    r03 = rule_ineligible(arm, task.step("open"), task)
    assert r03 is not None and r03.startswith('integrator rule R03: "The PCB tray lid" is a mechanism')
    r09 = rule_ineligible(arm, task.step("pick"), task)
    assert r09 is not None and r09.startswith('integrator rule R09: "PCB rack" weighs 3 kg')


# --------------------------------------------------------------------------- process tools (R05)


def _nut_runner():
    return preset("process_tool", "nut-runner-socket-13", "drive")


def _tamper():
    return preset("process_tool", "tamper-pad-200", "press")


def test_the_threaded_templates_are_the_thread_kinds() -> None:
    """R05's threaded members are the kinds the twist and the insert drive as a thread
    (:data:`rtt.studio.types.THREAD_KINDS`), one set, read by value."""
    import inspect

    from rtt.studio import rules
    from rtt.studio.types import THREAD_KINDS

    assert rules.THREADED_TEMPLATES == frozenset(k.value for k in THREAD_KINDS)
    assert rules.THREADED_TEMPLATES == {"threaded_joint", "panel_rail"}
    assert "THREAD_KINDS" in inspect.getsource(rules)  # derived from it, not a second list


def test_process_tools_have_their_own_mechanism_and_serve_only_their_process() -> None:
    """A nut runner drives and starts a fastener, a tamper presses a face: neither picks, holds or
    hooks a part. The nut runner measures no hook; the tamper's 200 mm pad overhangs its post like a
    T, but that is its press face, not a hook."""
    nut, tamper = candidate_of(_nut_runner(), "preset"), candidate_of(_tamper(), "preset")
    assert nut.mechanism == tamper.mechanism == "process"
    assert nut.capabilities == {PrimitiveKind.TWIST, PrimitiveKind.INSERT}
    assert tamper.capabilities == {PrimitiveKind.PUSH}
    assert nut.to_dict()["mechanism"] == "process" and nut.to_dict()["acquire_primitive"] is None
    task = _lid_task()
    lid, pick = task.step("open"), task.step("pick")
    assert capability_problem(nut, pick) == "a nut runner turns a fastener; it cannot pick or hold a part"
    assert capability_problem(tamper, pick) == "a tamper pad presses a face; it cannot pick or hold a part"
    assert probe_tool(nut.spec).hook_mm < HOOK_MIN_MM < probe_tool(tamper.spec).hook_mm
    for c in (nut, tamper):
        reason = capability_problem(c, lid)
        assert reason is not None and "it is not a hook" in reason, c.name
    close = Step.of(
        "close", "push", "lid", "lid", "Close the lid", [SuccessCriterion("angle_deg", at_most=5.0)]
    )
    assert capability_problem(tamper, close) is None
    assert capability_problem(nut, close) == "a nut runner turns a fastener; it cannot push"


def test_process_tools_do_not_keep_a_soft_arm_or_a_hand_off_an_articulated_step() -> None:
    """The 5a merge review: the process tools counted as fingers, so on a lid they were the rigid
    options left and R03 never readmitted the soft arm (measured: the soft arm and a cup alone gave a
    readmission; adding the process tools left the nut runner and the tamper eligible, no note)."""
    task = _lid_task()
    arm, suction, nut, tamper = (candidate_of(s, "preset") for s in (_arm(), cup(), _nut_runner(), _tamper()))
    notes: list[dict] = []
    eligible, pruned, _ = prune(task, [arm, suction, nut, tamper], notes=notes)
    assert [c.effector_id for c in eligible[0]] == [arm.effector_id]
    assert len(notes) == 1 and notes[0]["readmitted"] == [arm.effector_id]
    reasons = {(p.effector_id, p.step_id): p.reason for p in pruned}
    assert "not a hook" in reasons[(nut.effector_id, "open")]
    assert "not a hook" in reasons[(tamper.effector_id, "open")]
    # the same on a push: the nut runner does not push, so the hand comes back for the lid
    lid = task.object("lid")
    close = Step.of(
        "close", "push", "lid", "lid", "Close the lid", [SuccessCriterion("angle_deg", at_most=5.0)]
    )
    push_task = TaskSpec("Close", "", (lid,), (close,), source="test")
    hand = candidate_of(_hand(), "preset")
    notes = []
    eligible, pruned, _ = prune(push_task, [hand, nut], notes=notes)
    assert [c.effector_id for c in eligible[0]] == [hand.effector_id] and len(notes) == 1
    assert {(p.effector_id, p.reason) for p in pruned} == {
        (nut.effector_id, "a nut runner turns a fastener; it cannot push")
    }


def _clamp(torque_nm: float) -> TaskSpec:
    """A 13 mm hex clamp nut on a stud (15.011 mm across corners, 16 mm tall): start it, then
    tighten it to ``torque_nm``."""
    nut = ObjectInstance.of(
        "clamp_nut",
        "threaded_joint",
        "Clamp nut",
        ObjectPose(0.0, 0.0, 0.0),
        member="nut",
        diameter_mm=15.011,
        length_mm=16.0,
        pitch_mm=1.25,
        engage_turns=3.0,
        start_torque_nm=0.3,
        final_torque_nm=6.0,
        chamfer_mm=1.0,
    )
    steps = (
        Step.of(
            "start",
            "insert",
            "clamp_nut",
            "member",
            "Start the nut",
            [SuccessCriterion("depth_mm", at_least=3.5)],
        ),
        Step.of(
            "tighten",
            "twist",
            "clamp_nut",
            "member",
            "Tighten the clamp",
            [SuccessCriterion("torque_nm", at_least=torque_nm)],
            params={"direction": "cw", "speed_deg_s": 180},
            constraints=StepConstraints(max_torque_nm=15.0),
        ),
    )
    return TaskSpec("Clamp", "", (nut,), steps, source="test")


def test_tightening_a_clamp_to_5_nm_needs_the_nut_runner() -> None:
    task = _clamp(5.0)
    start, tighten = task.step("start"), task.step("tighten")
    grip, nut, tamper, suction = (
        candidate_of(s, "preset") for s in (gripper(), _nut_runner(), _tamper(), cup())
    )
    reason = rule_ineligible(grip, tighten, task)
    assert reason is not None and reason.startswith('integrator rule R05: tightening "Clamp nut" to 5 N m')
    assert "nut runner" in reason and f"{PROCESS_TORQUE_NM:g} N m [assumed]" in reason
    assert rule_ineligible(nut, tighten, task) is None
    other = rule_ineligible(tamper, tighten, task)
    assert other is not None and other.startswith("integrator rule R05") and "nut runner" in other
    assert rule_ineligible(grip, start, task) is None  # starting it asks for no torque: not a process step
    eligible, pruned, _ = prune(task, [grip, nut, tamper, suction])
    assert [c.effector_id for c in eligible[1]] == [nut.effector_id]
    assert grip in eligible[0] and nut in eligible[0]
    reasons = {(p.effector_id, p.step_id): p.reason for p in pruned}
    assert reasons[(grip.effector_id, "tighten")].startswith("integrator rule R05")
    assert reasons[(suction.effector_id, "tighten")].startswith("integrator rule R05")
    # hand-tight is a gripper's job: under the threshold R05 does not apply, and the gripper turns
    # the member itself (the Studio does not model hand tools held in a gripper)
    light = _clamp(PROCESS_TORQUE_NM - 1.0)
    assert rule_ineligible(grip, light.step("tighten"), light) is None


def test_the_hand_tight_pipe_coupling_stays_a_gripper_job() -> None:
    """H2 (the recorded v1 hero plan; the v2 one is checked in test_studio_engineer): the coupling is turned on to at least 3 N m (hand-tight), under
    the R05 threshold, so the gripper still turns it. The nut runner's hex socket does not fit the
    round coupling."""
    from rtt.studio.engineer.gate import RECORDED_GATE_V1
    from rtt.studio.engineer.webdata import recorded_plans
    from rtt.studio.types import task_spec_from_tool_args

    plan = next(
        e["plan"] for e in recorded_plans(RECORDED_GATE_V1)["examples"] if e["case_id"] == "h2_coupling"
    )
    keys = ("title", "description", "objects", "steps", "assumptions")
    task = task_spec_from_tool_args({k: plan[k] for k in keys}, source="recorded")
    tighten = task.step("tighten_coupling")
    need = max(c.at_least for c in tighten.success if c.quantity == "torque_nm" and c.at_least is not None)
    assert need == 3.0 and PROCESS_TORQUE_NM > need
    assert ASSUMED_CONSTANTS["process_torque_nm"]["value"] == PROCESS_TORQUE_NM
    assert rule_ineligible(candidate_of(gripper(), "preset"), tighten, task) is None
    nut = candidate_of(_nut_runner(), "preset")
    reason = fit_problem(nut, probe_tool(nut.spec), tighten, task, _v2())
    assert reason is not None and "coupling" in reason and "hex" in reason


def _v2():
    from rtt.sim.task import default_task_v2

    return default_task_v2()


def test_the_nut_runner_fits_only_a_13_mm_hex_nut_tall_enough_for_its_socket() -> None:
    from rtt.effectors.process_tool import MIN_MEMBER_MM

    nut = candidate_of(_nut_runner(), "preset")
    info = probe_tool(nut.spec)
    task = _clamp(5.0)
    assert fit_problem(nut, info, task.step("tighten"), task, _v2()) is None

    def member(**params: float) -> TaskSpec:
        obj = replace(task.object("clamp_nut"), params={**task.object("clamp_nut").params, **params})
        return TaskSpec("Clamp", "", (obj,), task.steps, source="test")

    small = member(diameter_mm=11.547)  # a 10 mm hex
    reason = fit_problem(nut, info, small.step("tighten"), small, _v2())
    assert reason == "its 13 mm hex socket drives a 13 mm hex; the nut clamp_nut is 10 mm across flats"
    short = member(length_mm=8.0)
    reason = fit_problem(nut, info, short.step("start"), short, _v2())
    assert reason == (
        f"the nut clamp_nut is 8 mm tall; the nut runner's socket seats on members at least "
        f"{MIN_MEMBER_MM:g} mm tall (measured)"
    )


def test_the_r05_threshold_is_4_nm_and_inclusive() -> None:
    grip, nut = candidate_of(gripper(), "preset"), candidate_of(_nut_runner(), "preset")
    assert PROCESS_TORQUE_NM == 4.0
    for torque, process in ((3.99, False), (4.0, True)):
        task = _clamp(torque)
        step = task.step("tighten")
        assert (process_need(step, task.object("clamp_nut")) == "nut_runner") is process, torque
        assert (rule_ineligible(grip, step, task) is not None) is process, torque
        assert rule_ineligible(nut, step, task) is None


def _stand_in_step(primitive: str, quantity: str, torque: float) -> SimpleNamespace:
    """A step on the panel clamp with one at-least criterion (``clamp_torque_nm`` is another task's
    quantity; the rule matches it by name, so a stand-in criterion is enough here)."""
    criterion = SimpleNamespace(quantity=quantity, at_least=torque, at_most=None)
    return SimpleNamespace(
        id="clamp", primitive=PrimitiveKind(primitive), object="panel", success=(criterion,)
    )


def test_r05_matches_a_panel_clamp_and_its_clamp_torque_by_name() -> None:
    """The panel-on-rails template and its clamp-torque quantity are another task's; R05 matches
    them by value and by name, so they are covered as soon as they exist."""
    clamp = SimpleNamespace(
        name="panel", label="Panel clamp", kind=SimpleNamespace(value="panel_rail"), params={}
    )
    task = SimpleNamespace(object=lambda name: clamp)
    grip, nut = candidate_of(gripper(), "preset"), candidate_of(_nut_runner(), "preset")
    for quantity in ("torque_nm", "clamp_torque_nm"):
        for primitive in ("twist", "insert"):
            step = _stand_in_step(primitive, quantity, 5.0)
            assert process_need(step, clamp) == "nut_runner", (quantity, primitive)
            reason = rule_ineligible(grip, step, task)
            assert reason is not None and reason.startswith(
                'integrator rule R05: tightening "Panel clamp" to 5 N m'
            )
            assert rule_ineligible(nut, step, task) is None
        assert process_need(_stand_in_step("twist", quantity, 3.0), clamp) is None  # hand-tight
    assert process_need(_stand_in_step("twist", "turns", 5.0), clamp) is None  # not a torque
    assert process_need(_stand_in_step("push", "torque_nm", 5.0), clamp) is None  # not a tightening


def test_the_nut_runner_is_not_screened_on_an_object_without_a_thread() -> None:
    nut = candidate_of(_nut_runner(), "preset")
    info = probe_tool(nut.spec)
    port = ObjectInstance.of("port", "port_plug", "Port", ObjectPose(-150.0, 0.0, 0.0), plug_diameter_mm=8)
    latch = ObjectInstance.of("latch", "quarter_turn_latch", "Latch", ObjectPose(150.0, 0.0, 0.0))
    valve = ObjectInstance.of("valve", "handwheel_valve", "Valve", ObjectPose(0.0, 150.0, 0.0))
    turn = {"direction": "ccw", "speed_deg_s": 60}
    steps = (
        Step.of("plug", "insert", "port", "plug", "Plug in", [SuccessCriterion("depth_mm", at_least=5)]),
        Step.of(
            "unlock",
            "twist",
            "latch",
            "head",
            "Unlock",
            [SuccessCriterion("angle_deg", at_least=80)],
            params=turn,
        ),
        Step.of(
            "open",
            "twist",
            "valve",
            "wheel",
            "Open",
            [SuccessCriterion("angle_deg", at_least=80)],
            params=turn,
        ),
    )
    task = TaskSpec("Cabinet", "", (port, latch, valve), steps, source="test")
    want = {
        "plug": "a nut runner drives only a threaded nut or clamp; port is a plug pushed into its port",
        "unlock": "a nut runner drives only a threaded nut or clamp; latch is a quarter-turn latch",
        "open": "a nut runner drives only a threaded nut or clamp; valve is a handwheel valve",
    }
    for step in steps:
        assert capability_problem(nut, step) is None  # it does turn and insert ...
        assert fit_problem(nut, info, step, task, _v2()) == want[step.id]  # ... but only a thread
    _, pruned, _ = prune(task, [nut])
    assert {p.step_id: p.reason for p in pruned} == want
    # a panel's clamp is a threaded member: the socket is not refused for the kind of object
    clamp = SimpleNamespace(
        name="panel", label="Panel clamp", kind=SimpleNamespace(value="panel_rail"), params={}
    )
    stand_in = SimpleNamespace(object=lambda name: clamp)
    step = _stand_in_step("twist", "torque_nm", 5.0)
    assert fit_problem(nut, info, step, stand_in, _v2()) is None


def _tile_bed() -> tuple[SimpleNamespace, Step]:
    """A push that presses a tile into its bed. The tile-on-a-bed template is another task's; the
    rule matches the kind by its value, so a stand-in object is enough here."""
    tile = SimpleNamespace(name="tile", label="Floor tile", kind=SimpleNamespace(value="tile_bed"), params={})
    step = Step.of(
        "tamp",
        "push",
        "tile",
        "part",
        "Press the tile into its bed",
        [SuccessCriterion("force_n", at_least=30)],
    )
    return SimpleNamespace(object=lambda name: tile), step


def test_pressing_a_tile_into_its_bed_needs_the_tamper() -> None:
    task, step = _tile_bed()
    grip, nut, tamper, suction = (
        candidate_of(s, "preset") for s in (gripper(), _nut_runner(), _tamper(), cup())
    )
    assert rule_ineligible(tamper, step, task) is None
    for c in (grip, nut, suction):
        reason = rule_ineligible(c, step, task)
        assert reason is not None and reason.startswith(
            'integrator rule R05: pressing "Floor tile" into its bed'
        )
        assert "tamper" in reason, c.name
    # pressing a plain part is not a process step: every pushing tool may do it
    plain = two_press_task()
    assert rule_ineligible(grip, plain.step("press_carton"), plain) is None
    assert rule_ineligible(tamper, plain.step("press_carton"), plain) is None


# --------------------------------------------------------------------------- vacuum frames fit by their cups


def _slab(
    size_mm: float, height_mm: float, mass_g: float, shape: str = "box", size_y_mm: float | None = None
) -> TaskSpec:
    part = ObjectInstance.of(
        "slab",
        "part",
        "Slab",
        ObjectPose(0.0, 0.0, 0.0),
        shape=shape,
        size_x_mm=size_mm,
        size_y_mm=size_mm if size_y_mm is None else size_y_mm,
        size_z_mm=height_mm,
        mass_g=mass_g,
    )
    step = Step.of("lift", "suction", "slab", "part", "Lift the slab", LIFT)
    return TaskSpec("Lift", "", (part,), (step,), source="test")


def test_a_frame_fits_a_part_only_when_its_face_covers_enough_cups() -> None:
    """A frame holds only with half its cups sealed, at least three (the frame's own engagement
    rule): a part whose top face covers fewer of its cups is refused before any physics."""
    frame = candidate_of(preset("suction", "vacuum-frame-2x2-300", "seal"), "preset")
    info = probe_tool(frame.spec)
    small = _slab(100.0, 20.0, 200.0)
    assert fit_problem(frame, info, small.step("lift"), small, _v2()) == (
        "the 100 x 100 mm top face of slab covers 0 of the frame's 4 cups; the frame holds a part only "
        "with at least 3 of them sealed [assumed]"
    )
    tile = _slab(300.0, 10.0, 2160.0)
    assert fit_problem(frame, info, tile.step("lift"), tile, _v2()) is None
    # the lip counts: the 18 mm cups stand 100 mm off each axis, so a square face needs
    # 2 x (100 + 9) = 218 mm, and a round one 2 x (141.4 + 9) = 300.8 mm across
    for size, shape, ok in (
        (217.8, "box", False),
        (218.0, "box", True),
        (290.0, "cylinder", False),
        (301.0, "cylinder", True),
    ):
        part = _slab(size, 10.0, 1000.0, shape=shape)
        reason = fit_problem(frame, info, part.step("lift"), part, _v2())
        if ok:
            assert reason is None, (size, shape, reason)
        else:
            face = f"{size:g} x {size:g} mm top face" if shape == "box" else f"{size:g} mm round top face"
            assert reason is not None and reason.startswith(
                f"the {face} of slab covers 0 of the frame's 4 cups"
            )


def test_a_frame_counts_the_cups_its_face_covers_along_each_axis() -> None:
    """The 2 x 3 frame: 3 columns of 21 mm cups 250 mm apart along x, 2 rows 250 mm apart along y; it
    holds with 3 of its 6 cups. A part (at most 400 mm across) under it covers only the middle
    column. A frame's x is aimed along the part's x, so a long face fits one way round only, and a
    frame of 8 cups holds with 4 of them."""
    frame = candidate_of(preset("suction", "vacuum-frame-2x3-600", "seal"), "preset")
    info = probe_tool(frame.spec)
    square = _slab(400.0, 30.0, 3000.0)  # only the middle column's 2 cups are on it
    assert fit_problem(frame, info, square.step("lift"), square, _v2()) == (
        "the 400 x 400 mm top face of slab covers 2 of the frame's 6 cups; the frame holds a part only "
        "with at least 3 of them sealed [assumed]"
    )
    # 3 x 2 cups of 18 mm, 150 mm apart along x and 100 mm along y: 6 cups, holds with 3
    six = candidate_of(frame_spec(FramePreset(3, 2, 150.0, 100.0, 18.0, "a test frame")), "preset")
    info = probe_tool(six.spec)
    along = _slab(340.0, 20.0, 1000.0, size_y_mm=130.0)  # all 6: 150 + 9 <= 170, 50 + 9 <= 65
    assert fit_problem(six, info, along.step("lift"), along, _v2()) is None
    across = _slab(130.0, 20.0, 1000.0, size_y_mm=340.0)  # the same face turned: the middle 2 only
    reason = fit_problem(six, info, across.step("lift"), across, _v2())
    assert reason is not None and reason.startswith(
        "the 130 x 340 mm top face of slab covers 2 of the frame's 6"
    )
    # 4 x 2 cups of 18 mm, 100 mm apart: 8 cups, holds with 4, so a face over the two middle
    # columns is enough although it leaves half the cups off
    eight = candidate_of(frame_spec(FramePreset(4, 2, 100.0, 100.0, 18.0, "a test frame")), "preset")
    info = probe_tool(eight.spec)
    middle = _slab(120.0, 20.0, 500.0, size_y_mm=140.0)  # 50 + 9 <= 60 along x, 50 + 9 <= 70 along y
    assert fit_problem(eight, info, middle.step("lift"), middle, _v2()) is None
    narrow = _slab(116.0, 20.0, 500.0, size_y_mm=140.0)
    reason = fit_problem(eight, info, narrow.step("lift"), narrow, _v2())
    assert reason is not None and "covers 0 of the frame's 8 cups" in reason and "at least 4" in reason


# --------------------------------------------------------------------------- R07: change times


def test_sensitivity_uses_7_5_s() -> None:
    assert SENSITIVITY_CHANGE_S == (3.0, 6.0, 7.5, 10.0, 12.0)
    assert TOOL_CHANGE_S == 7.5 and TOOL_CHANGE_S in SENSITIVITY_CHANGE_S
    # four presses: one tool change setup changes once, the other three times
    base = two_press_task()
    again = (
        replace(base.step("press_carton"), id="press_carton_2", after=()),
        replace(base.step("press_flat"), id="press_flat_2", after=()),
    )
    task = TaskSpec("Four presses", "", base.objects, (*base.steps, *again), source="test")
    ids = task.step_ids
    single = Setup.single(gripper(), ids, label="Gripper")
    once = Setup.tool_change(
        {ids[0]: cup(), ids[1]: cup(), ids[2]: gripper(), ids[3]: gripper()}, label="Cup, then gripper"
    )
    alternate = Setup.tool_change({s: (cup() if i % 2 == 0 else gripper()) for i, s in enumerate(ids)})
    assert once.setup_id != alternate.setup_id
    results = [
        _result(single, task, [True] * 4, 60.0),
        _result(once, task, [True] * 4, 24.0 + TOOL_CHANGE_S, changes=1),  # 31.5 s
        _result(alternate, task, [True] * 4, 8.0 + 3 * TOOL_CHANGE_S, changes=3),  # 30.5 s
    ]
    sens = sensitivity(results)
    by = {(c["tool_change_s"], c["present_s"]): c for c in sens["combos"]}
    assert len(by) == 15 and sens["baseline"]["tool_change_s"] == 7.5
    # at 7.5 s the two tool changes are within 10 % and tie, so the one with fewer changes is next
    assert sens["baseline"]["order"] == [single.setup_id, once.setup_id, alternate.setup_id]
    assert by[(7.5, 1.5)]["order"] == sens["baseline"]["order"]
    assert by[(7.5, 1.5)]["cycle_time_s"][once.setup_id] == pytest.approx(31.5)
    # the single tool is recommended at every change time (R01: cycle time never decides it) ...
    assert all(c["recommended"] == single.setup_id for c in sens["combos"])
    assert sens["recommendation_changes"] is False
    # ... while the two tool changes swap places when changes are cheap
    assert by[(6.0, 1.5)]["order"] == [single.setup_id, alternate.setup_id, once.setup_id]
    assert by[(6.0, 1.5)]["cycle_time_s"][alternate.setup_id] == pytest.approx(26.0)
    assert by[(6.0, 1.5)]["order_changed"] and not by[(10.0, 1.5)]["order_changed"]


# --------------------------------------------------------------------------- stale constants


def test_the_constants_table_has_the_wrist_pitch_and_the_rule_margins() -> None:
    c = ASSUMED_CONSTANTS
    assert c["pitch_range_deg"]["value"] == STUDIO_WRIST.pitch_range_deg == 90.0
    assert c["pitch_speed_dps"]["value"] == STUDIO_WRIST.pitch_speed_dps == 180.0
    assert c["simplicity_per_extra_effector"]["value"] == 0.05
    assert c["simplicity_step_floor"]["value"] == 0.5
    assert c["tool_change_s"]["value"] == 7.5 and c["changer_stack_mm"]["value"] == 52.0
    assert c["hook_min_mm"]["value"] == HOOK_MIN_MM and c["hook_throat_mm"]["value"] == HOOK_THROAT_MM
    assert all(v["tag"] == "assumed" for v in c.values())


def test_build_pack_documents_fall_back_to_the_contract_changer() -> None:
    tools = [
        {
            "effector_id": "a",
            "display_name": "Tool A",
            "family": "suction",
            "primitive": "seal",
            "fidelity": "approximation",
        },
        {
            "effector_id": "b",
            "display_name": "Tool B",
            "family": "catalog_gripper",
            "primitive": "pinch",
            "fidelity": "rigid",
        },
    ]
    setup = {
        "kind": "tool_change",
        "tools": tools,
        "changer": {},
        "assignment": {"press": 0},
        "setup_id": "0" * 64,
        "label": "Two tools",
    }
    ch = ToolChanger()
    notes = " ".join(b.note or "" for b in bom_lines(setup, {}, [], {}))
    assert f"{TOOL_CHANGE_S:g} s changeover" in notes and f"stack {ch.stack_mm:g} mm" in notes
    assert "6 s" not in notes and "25 mm" not in notes
    result = {"step_success": {"press": 1.0}, "completion_rate": 1.0, "n_runs": 8, "cycle_time_s": 20.0}
    finalist = {"setup": setup, "result": result}
    report = {"task": {"title": "Press", "steps": [{"id": "press", "primitive": "push", "object": "box"}]}}
    text = assembly_notes(finalist, report, [])
    assert f"Changer stack {ch.stack_mm:g} mm, changeover {TOOL_CHANGE_S:g} s" in text
