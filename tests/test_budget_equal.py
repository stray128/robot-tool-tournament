"""Equal budgets: every run of every arm simulates exactly B = 40 valid designs, and round 0 is shared.

Invalid or unusable model output must never cost simulator budget, and round 0 (manual
entrants + Sobol fill) must be byte-identical across arms for the same run seed.
"""

from __future__ import annotations

import pytest

from rtt.design.validate import validate
from rtt.llm.client import TokenBucket
from rtt.llm.providers import TransportResponse
from rtt.search.fakes import FakeEvaluator, ScriptedProposer
from rtt.search.nemotron import make_proposer_client
from rtt.search.runner import SearchOutcome, run_search
from rtt.store.runs import read_jsonl

SEEDS = [101, 202]
BUDGET = 40


class Hostile:
    """A model that never produces a usable design: prose, invalid designs, or errors."""

    def __init__(self) -> None:
        self.calls = 0

    def list_models(self, timeout_s: float) -> TransportResponse:
        return TransportResponse(200, {}, {"data": []})

    def chat_completions(self, payload, timeout_s):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls % 3 == 0:
            return TransportResponse(400, {}, {"error": {"message": "bad request"}})
        message = {"role": "assistant", "content": "I prefer to describe fingers in prose."}
        usage = {"prompt_tokens": 10, "completion_tokens": 10}
        return TransportResponse(200, {}, {"choices": [{"index": 0, "message": message}], "usage": usage})


def client(transport):  # type: ignore[no-untyped-def]
    return make_proposer_client(None, transport=transport, limiter=TokenBucket(1e12))


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, SearchOutcome]:
    root = tmp_path_factory.mktemp("budget")
    noisy = ScriptedProposer(7, invalid_rate=0.4, similar_rate=0.3, garbage_rate=0.2)
    return {
        "random": run_search("random", 2, BUDGET, SEEDS, FakeEvaluator(0), root / "random"),
        "nemotron": run_search(
            "nemotron", 2, BUDGET, SEEDS, FakeEvaluator(0), root / "nemotron", client(noisy)
        ),
        "zeroshot": run_search(
            "zeroshot", 2, BUDGET, SEEDS, FakeEvaluator(0), root / "zeroshot", client(ScriptedProposer(8))
        ),
        "hostile": run_search(
            "nemotron", 2, BUDGET, SEEDS, FakeEvaluator(0), root / "hostile", client(Hostile())
        ),
    }


@pytest.mark.parametrize("arm", ["random", "nemotron", "zeroshot", "hostile"])
def test_every_run_simulates_exactly_the_budget_of_valid_designs(outcomes, arm: str) -> None:
    out = outcomes[arm]
    for run in out.runs:
        ctx = run.ctx
        assert len(ctx.history) == BUDGET
        assert ctx.budget.charged == [8, 8, 8, 8, 8] and ctx.budget.complete
        assert [sum(1 for e in ctx.history if e.round == r) for r in range(5)] == [8] * 5
        assert all(validate(e.design, ctx.gripper_task).ok for e in ctx.history)
        assert ctx.budget.llm_calls <= 16
        lines = read_jsonl(out.out_dir / "runs" / ctx.run_id / "history.jsonl")
        assert len(lines) == BUDGET
    assert out.manifest["sim"]["designs_charged"] == BUDGET * len(SEEDS)


def test_invalid_model_output_costs_no_budget(outcomes) -> None:
    for arm in ("nemotron", "hostile"):
        for run in outcomes[arm].runs:
            ctx = run.ctx
            assert ctx.rejections, "the noisy model should have produced rejected proposals"
            assert len(ctx.history) == BUDGET  # rejections are on top of, not inside, the budget
            simulated = {e.design.design_id() for e in ctx.history}
            for rejection in ctx.rejections:
                if rejection.reason == "invalid":
                    assert (
                        rejection.params is None or not validate(dict(rejection.params), ctx.gripper_task).ok
                    )
            assert ctx.budget.fallbacks == sum(
                1 for e in ctx.history if e.proposal.origin == "fallback_random"
            )
            assert len(simulated) == BUDGET  # the diversity check keeps every simulated design distinct
    hostile = outcomes["hostile"].runs[0].ctx
    assert {e.proposal.origin for e in hostile.history if e.round > 0} == {"fallback_random"}


def test_round0_is_byte_identical_across_arms(outcomes) -> None:
    for i, seed in enumerate(SEEDS):
        blobs = []
        for arm, out in outcomes.items():
            run_id = out.runs[i].ctx.run_id
            assert out.runs[i].ctx.run_seed == seed
            lines = (out.out_dir / "runs" / run_id / "history.jsonl").read_text().splitlines()
            round0 = lines[:8]
            assert all('"round": 0' in line for line in round0)
            # Compare the proposals (the score part is identical too, since the evaluator is shared).
            blobs.append((arm, "\n".join(round0)))
        assert len({blob for _, blob in blobs}) == 1, [arm for arm, _ in blobs]
    n = len(outcomes["random"].manifest["entrants"])
    first = [e.design.design_id() for e in outcomes["random"].runs[0].ctx.history if e.round == 0]
    second = [e.design.design_id() for e in outcomes["random"].runs[1].ctx.history if e.round == 0]
    assert first[:n] == second[:n] and first[n:] != second[n:]  # entrants shared, Sobol fill per seed
