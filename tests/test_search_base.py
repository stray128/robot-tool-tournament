"""Search core: seeds, round plan, score aggregation, budget ledger, history and JSON round trips."""

from __future__ import annotations

import pytest

from rtt.design.schema import FingerDesign
from rtt.search.base import (
    FAILURE_MODES,
    BudgetError,
    BudgetLedger,
    DesignScore,
    EpisodeOutcome,
    Evaluated,
    Proposal,
    RunHistory,
    derive_seed,
    round_plan,
)
from rtt.search.fakes import DEFAULT_FAKE_CONDITIONS, FakeEvaluator

BASE = {
    "finger_length_mm": 80.0,
    "finger_thickness_mm": 10.0,
    "finger_width_mm": 20.0,
    "bend_angle_deg": 0.0,
    "bend_start_frac": 0.5,
    "inner_face": "flat",
    "inner_face_depth_mm": 0.0,
    "tip_profile": "blunt",
    "tip_size_mm": 0.0,
    "taper_ratio": 1.0,
}


def design(**overrides: object) -> FingerDesign:
    return FingerDesign(**{**BASE, **overrides})


def ep(
    cid: str, family: str, k: int, success: bool, mode: str | None = None, hold: float = 0.0
) -> EpisodeOutcome:
    return EpisodeOutcome(
        cid, family, k, 0, success, None if success else mode, 0.15 if success else 0.0, hold
    )


def test_derive_seed_is_stable_and_label_sensitive() -> None:
    assert derive_seed(7, "round0") == derive_seed(7, "round0")
    assert derive_seed(7, "round0") != derive_seed(7, "random")
    assert derive_seed(7, "a", 1) != derive_seed(8, "a", 1)
    assert 0 <= derive_seed(123, "x") < 2**63


def test_round_plan() -> None:
    assert round_plan(40, 8, 8) == (8, 8, 8, 8, 8)
    assert round_plan(24, 8, 8) == (8, 8, 8)
    with pytest.raises(ValueError):
        round_plan(41, 8, 8)
    with pytest.raises(ValueError):
        round_plan(4, 8, 8)


def test_failure_modes_match_the_simulator() -> None:
    from rtt.sim.score import FAILURE_MODES as SIM_MODES

    assert FAILURE_MODES == tuple(SIM_MODES)


def test_score_uses_best_setting_for_every_breakdown() -> None:
    episodes = [
        # k = 0: 1/4 successes
        ep("a", "box", 0, True, hold=1.0),
        ep("b", "box", 0, False, "missed_grasp"),
        ep("c", "rod", 0, False, "missed_grasp"),
        ep("d", "rod", 0, False, "slip_during_lift"),
        # k = 1: 3/4 successes
        ep("a", "box", 1, True, hold=1.0),
        ep("b", "box", 1, True, hold=1.0),
        ep("c", "rod", 1, True, hold=1.0),
        ep("d", "rod", 1, False, "unstable_hold", hold=0.4),
    ]
    score = DesignScore.from_episodes("abc", 12.5, episodes, n_settings=2)
    assert score.best_k == 1
    assert score.success == pytest.approx(0.75)
    assert score.success_by_k == pytest.approx((0.25, 0.75))
    assert score.per_family == {"box": 1.0, "rod": 0.5}
    assert score.per_condition == {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}
    assert score.failure_modes == {"unstable_hold": 1}
    assert score.condition_failures == {"d": {"unstable_hold": 1}}
    assert score.hold_fraction == pytest.approx((3 + 0.4) / 4)
    assert score.n_episodes == 8


def test_score_ties_break_on_hold_then_lower_setting() -> None:
    episodes = [
        ep("a", "box", 0, False, "slip_during_lift", 0.2),
        ep("a", "box", 1, False, "slip_during_lift", 0.6),
    ]
    assert DesignScore.from_episodes("x", 1.0, episodes, 2).best_k == 1
    episodes = [ep("a", "box", 0, True, hold=1.0), ep("a", "box", 1, True, hold=1.0)]
    assert DesignScore.from_episodes("x", 1.0, episodes, 2).best_k == 0


def test_rank_key_prefers_success_then_hold_then_lighter() -> None:
    def score(success: bool, hold: float, mass: float) -> DesignScore:
        return DesignScore.from_episodes("x", mass, [ep("a", "box", 0, success, "missed_grasp", hold)], 1)

    assert score(True, 1.0, 30.0).rank_key() > score(False, 0.9, 5.0).rank_key()
    assert score(False, 0.6, 30.0).rank_key() > score(False, 0.5, 5.0).rank_key()
    assert score(False, 0.5, 5.0).rank_key() > score(False, 0.5, 6.0).rank_key()


def test_score_rejects_bad_settings_and_empty_input() -> None:
    with pytest.raises(ValueError):
        DesignScore.from_episodes("x", 1.0, [], 3)
    with pytest.raises(ValueError):
        DesignScore.from_episodes("x", 1.0, [ep("a", "box", 3, True)], 3)


def test_budget_ledger_charges_each_round_exactly_once_in_order() -> None:
    ledger = BudgetLedger(40, 8, 8, llm_call_cap=16)
    assert ledger.round_sizes == (8, 8, 8, 8, 8)
    with pytest.raises(BudgetError):
        ledger.charge(1, 8)  # round 0 first
    with pytest.raises(BudgetError):
        ledger.charge(0, 7)  # wrong size
    ledger.charge(0, 8)
    with pytest.raises(BudgetError):
        ledger.charge(0, 8)  # twice
    for r in range(1, 5):
        ledger.charge(r, 8)
    assert ledger.complete and ledger.spent == 40 and ledger.remaining == 0
    with pytest.raises(BudgetError):
        ledger.charge(5, 8)


def test_llm_call_cap_keeps_reserved_calls() -> None:
    ledger = BudgetLedger(llm_call_cap=3)
    assert ledger.take_llm_call(keep=2)  # 1 used, 2 left
    assert not ledger.take_llm_call(keep=2)  # would leave only 1
    assert ledger.take_llm_call(keep=1)
    assert ledger.take_llm_call()
    assert not ledger.take_llm_call()
    assert ledger.llm_calls == 3 and ledger.llm_calls_left() == 0


def _scored(designs: list[FingerDesign]) -> list[DesignScore]:
    return FakeEvaluator(0).evaluate(designs)


def test_history_labels_ranking_and_best_so_far() -> None:
    designs = [design(finger_length_mm=60.0 + 10 * i) for i in range(4)]
    scores = _scored(designs)
    history = RunHistory()
    for d, s in zip(designs, scores, strict=True):
        history.add(0, Proposal(d, "sobol"), s)
    assert [e.label for e in history] == ["d01", "d02", "d03", "d04"]
    assert history.by_label("d03") is history[2]
    ranked = history.ranked()
    assert ranked[0] is history.best()
    keys = [e.score.rank_key() for e in ranked]
    assert keys == sorted(keys, reverse=True)
    curve = history.best_so_far()
    assert curve == sorted(curve) and curve[-1] == max(s.success for s in scores)
    with pytest.raises(ValueError):
        history.add(1, Proposal(designs[0], "random"), scores[1])  # score of another design


def test_json_round_trips() -> None:
    d = design(inner_face="v_groove", inner_face_depth_mm=3.0, rationale="groove")
    score = _scored([d])[0]
    proposal = Proposal(d, "nemotron", "mutate", 1, "mutate_1", ("d03",), "groove helps", "repair", "ab" * 32)
    item = Evaluated(9, 2, proposal, score)
    again = Evaluated.from_json(item.to_json())
    assert again == item
    assert again.label == "d09"
    assert DesignScore.from_json(score.to_json()) == score


def test_fake_evaluator_info_and_batch_invariance() -> None:
    ev = FakeEvaluator(3)
    info = ev.describe()
    assert len(info.conditions) == len(DEFAULT_FAKE_CONDITIONS) == 12
    assert info.families == ("box", "cylinder", "rod")
    assert info.episodes_per_design == 36
    designs = [design(finger_length_mm=50.0 + 5 * i) for i in range(6)]
    alone = [ev.evaluate([d])[0] for d in designs]
    batch = ev.evaluate(list(reversed(designs)))[::-1]
    assert alone == batch
    assert all(s.n_episodes == 36 for s in batch)
