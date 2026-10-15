"""SimEvaluator: the search evaluator contract on top of the simulator's batch API (one real episode)."""

from __future__ import annotations

import pytest

from rtt.design.schema import load_entrants
from rtt.search.base import Evaluator
from rtt.search.runner import DEFAULT_ENTRANTS

sim_evaluate = pytest.importorskip("rtt.sim.evaluate")
conditions = pytest.importorskip("rtt.sim.conditions")


@pytest.fixture(scope="module")
def one_condition_evaluator():
    from rtt.search.sim_adapter import SimEvaluator

    full = conditions.training_suite()
    suite = conditions.Suite("probe", (full.conditions[0],))
    return SimEvaluator(suite, controller_settings=(1,), workers=1)


def test_describe_exposes_training_facts_only(one_condition_evaluator) -> None:
    ev = one_condition_evaluator
    assert isinstance(ev, Evaluator)
    info = ev.describe()
    cond = ev.suite.conditions[0]
    assert [c.condition_id for c in info.conditions] == [cond.condition_id]
    assert info.conditions[0].family == cond.family
    assert "mm" in info.conditions[0].label
    assert info.controller_settings == (ev.task.controller.grasp_height_fracs[1],)
    assert info.seeds_per_condition == len(cond.seeds)
    assert info.suite_hash == ev.suite.digest()
    assert info.gripper_task.widest_training_object_mm >= 1000.0 * ev.suite.widest_grip_width - 0.1
    assert info.executor == "cpu" and info.episodes_per_design == len(cond.seeds)
    assert info.palm_clearance_m == ev.task.controller.palm_clearance  # the prompt's grasp-band rule


def test_scores_match_the_simulator_records(one_condition_evaluator) -> None:
    ev = one_condition_evaluator
    straight = load_entrants(DEFAULT_ENTRANTS)[0]
    scores = ev.evaluate([straight, straight])  # a duplicate is simulated once
    assert len(scores) == 2 and scores[0] == scores[1]
    records = ev.last_records
    assert len(records) == ev.describe().episodes_per_design
    score = scores[0]
    assert score.design_id == straight.design_id() and score.best_k == 0
    assert score.success == pytest.approx(sum(r.success for r in records) / len(records))
    assert score.mass_g == pytest.approx(records[0].finger_mass_g)
    assert score.n_episodes == len(records)
    episode = score.episodes[0]
    assert (episode.condition_id, episode.seed, episode.success) == (
        records[0].condition_id,
        records[0].seed,
        records[0].success,
    )
    assert episode.failure_mode == records[0].failure_mode
