"""Arm A: the Sobol sampler, the shared round 0 and run determinism."""

from __future__ import annotations

from pathlib import Path

import pytest

from rtt.design.schema import CATEGORICAL_OPTIONS, PARAM_BOUNDS, FingerDesign, GripperTask, load_entrants
from rtt.design.validate import validate
from rtt.search.fakes import FakeEvaluator
from rtt.search.random_search import DesignSampler, round0_designs
from rtt.search.runner import DEFAULT_ENTRANTS, run_search

TASK = GripperTask()


def test_sampler_is_deterministic_per_seed_and_always_valid() -> None:
    a = DesignSampler(42, TASK).sample(24)
    b = DesignSampler(42, TASK).sample(24)
    c = DesignSampler(43, TASK).sample(24)
    assert [d.model_dump() for d in a] == [d.model_dump() for d in b]
    assert [d.design_id() for d in a] != [d.design_id() for d in c]
    assert all(validate(d, TASK).ok for d in a + c)


def test_sampler_covers_the_space() -> None:
    sampler = DesignSampler(5, TASK)
    designs = sampler.sample(200)
    assert sampler.draws == 200 + sampler.rejected
    for name, options in CATEGORICAL_OPTIONS.items():
        assert {getattr(d, name) for d in designs} == set(options)
    for name, (lo, hi) in PARAM_BOUNDS.items():
        values = [getattr(d, name) for d in designs]
        assert min(values) >= lo and max(values) <= hi
        if name not in ("inner_face_depth_mm", "tip_size_mm"):  # depth is 0 on flat faces
            assert max(values) - min(values) > 0.5 * (hi - lo), name
    assert all(d.inner_face_depth_mm == 0.0 for d in designs if d.inner_face == "flat")


def test_sampler_accept_filter_and_exhaustion() -> None:
    sampler = DesignSampler(1, TASK, max_draws=50)
    picked = sampler.sample(3, accept=lambda d: d.inner_face == "v_groove")
    assert all(d.inner_face == "v_groove" for d in picked)
    from rtt.search.random_search import SamplerExhaustedError

    with pytest.raises(SamplerExhaustedError):
        sampler.sample(1, accept=lambda d: False)


def test_round0_is_entrants_then_sobol_and_depends_only_on_the_seed() -> None:
    entrants = load_entrants(DEFAULT_ENTRANTS)
    r0 = round0_designs(entrants, 1001, 8, TASK)
    assert len(r0) == 8
    assert [p.design for p in r0[: len(entrants)]] == entrants
    assert {p.origin for p in r0[len(entrants) :]} == {"sobol"}
    again = round0_designs(entrants, 1001, 8, TASK)
    assert [p.to_json() for p in again] == [p.to_json() for p in r0]
    other = round0_designs(entrants, 1002, 8, TASK)
    assert [p.design for p in other[: len(entrants)]] == entrants
    assert [p.to_json() for p in other] != [p.to_json() for p in r0]


def test_round0_rejects_invalid_entrants_and_overflow() -> None:
    bad = FingerDesign(**{**load_entrants(DEFAULT_ENTRANTS)[0].geometry_params(), "inner_face_depth_mm": 2.0})
    with pytest.raises(ValueError, match="invalid"):
        round0_designs([bad], 1, 8, TASK)
    with pytest.raises(ValueError, match="do not fit"):
        round0_designs(load_entrants(DEFAULT_ENTRANTS) * 3, 1, 8, TASK)


def test_random_arm_is_deterministic(tmp_path: Path) -> None:
    first = run_search("random", 2, 40, [7, 8], FakeEvaluator(0), tmp_path / "a")
    second = run_search("random", 2, 40, [7, 8], FakeEvaluator(0), tmp_path / "b")
    assert [r.fingerprint for r in first.runs] == [r.fingerprint for r in second.runs]
    for run in ("random-r00", "random-r01"):
        a = (tmp_path / "a" / "runs" / run / "history.jsonl").read_bytes()
        b = (tmp_path / "b" / "runs" / run / "history.jsonl").read_bytes()
        assert a == b
    assert (tmp_path / "a/results/designs.jsonl").read_bytes() == (
        tmp_path / "b/results/designs.jsonl"
    ).read_bytes()
    assert first.runs[0].fingerprint != first.runs[1].fingerprint
    assert all(
        item.proposal.origin in ("entrant", "sobol", "random") for r in first.runs for item in r.ctx.history
    )
