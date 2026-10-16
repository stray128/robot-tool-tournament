"""Random arm A on freeform-v2: scrambled Sobol over 7 numbers, canonicalised, validated, screened."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

from rtt.search.diversity import MIN_DISTANCE_V2
from rtt.search.fakes_v2 import FakeEvaluatorV2, stub_entrants
from rtt.search.random_search import SpaceSampler, round0_designs
from rtt.search.runner import run_search
from rtt.search.space import V2, get_space
from rtt.search.space_v2 import NUMERIC_BOUNDS, canonical_params
from rtt.store.runs import read_table

SPACE = get_space(V2)
CTX = FakeEvaluatorV2(0).describe().gripper_task


def test_sampler_is_deterministic_canonical_and_logs_its_valid_rate() -> None:
    a = SPACE.sample_sobol(11, CTX)
    b = SPACE.sample_sobol(11, CTX)
    assert isinstance(a, SpaceSampler)
    da, db = a.sample(64), b.sample(64)
    assert [SPACE.design_id(d) for d in da] == [SPACE.design_id(d) for d in db]
    assert [SPACE.design_id(d) for d in SPACE.sample_sobol(12, CTX).sample(8)] != [
        SPACE.design_id(d) for d in da[:8]
    ]
    for d in da:
        params = SPACE.params(d)
        assert canonical_params(params) == (params, (), ())  # canonical and valid
    stats = a.stats()
    assert stats["draws"] == a.draws >= 64 and stats["accepted"] == 64
    assert 0.6 < stats["valid_rate"] < 0.95  # strategy rules reject some draws, nothing else
    assert set(stats["rejections_by_reason"]) == {"schema"}


def test_sampler_covers_the_space_and_every_cell() -> None:
    designs = SPACE.sample_sobol(3, CTX).sample(900)
    cells = Counter(SPACE.cell(d) for d in designs)
    assert set(cells) == set(SPACE.cells())
    assert min(cells.values()) > 60  # uniform categoricals: about 100 each, fewer for rule-heavy cells
    for name, (lo, hi) in NUMERIC_BOUNDS.items():
        values = [
            SPACE.params(d)[name]
            for d in designs
            if SPACE.params(d)[name] not in (0.0, 0.2)
            or name in ("grip_height_frac", "pad_height_mm", "finger_width_mm")
        ]
        assert min(values) < lo + 0.1 * (hi - lo) and max(values) > hi - 0.1 * (hi - lo), name


def test_accept_filter_reasons_and_exhaustion() -> None:
    sampler = SPACE.sample_sobol(5, CTX, max_draws=50)
    picked = sampler.sample(3, accept=lambda d: "too_similar" if SPACE.params(d)["pad"] != "flat" else None)
    assert all(SPACE.params(d)["pad"] == "flat" for d in picked)
    assert sampler.reasons["too_similar"] > 0
    with pytest.raises(RuntimeError):
        sampler.sample(1, accept=lambda d: False)


def test_round0_is_entrants_then_screened_sobol_and_depends_only_on_the_seed() -> None:
    entrants = stub_entrants()
    a = round0_designs(entrants, 101, 8, CTX, space=SPACE, min_distance=MIN_DISTANCE_V2)
    b = round0_designs(entrants, 101, 8, CTX, space=SPACE, min_distance=MIN_DISTANCE_V2)
    c = round0_designs(entrants, 102, 8, CTX, space=SPACE, min_distance=MIN_DISTANCE_V2)
    assert [p.to_json() for p in a] == [p.to_json() for p in b]
    assert [p.origin for p in a] == ["entrant"] * 4 + ["sobol"] * 4
    assert [p.to_json() for p in a[:4]] == [p.to_json() for p in c[:4]] and a[4:] != c[4:]
    designs = [p.design for p in a]
    for x, y in combinations(designs, 2):
        if {SPACE.label(x), SPACE.label(y)} == {"imprint_configurator", "imprint_swept"}:
            continue  # the two imprints share one nominal recipe point by construction
        assert SPACE.distance(x, y) >= MIN_DISTANCE_V2
    with pytest.raises(ValueError):
        round0_designs(entrants * 3, 101, 8, CTX, space=SPACE)


def test_round0_rejects_an_invalid_entrant() -> None:
    def reject_stock(design, context):  # type: ignore[no-untyped-def]
        from rtt.search.space import SpaceCheck

        return (
            SpaceCheck(False, None, ("the stock blade bulges by 0.4 mm",), stage="build")
            if SPACE.label(design) == "stock"
            else None
        )

    from rtt.search.space_v2 import V2Context

    ctx = V2Context(part_ids=CTX.part_ids, validator=reject_stock)
    with pytest.raises(ValueError, match="bulges"):
        round0_designs(stub_entrants(), 1, 8, ctx, space=SPACE)


def test_random_arm_v2_is_deterministic_screened_and_logged(tmp_path: Path) -> None:
    outs = [
        run_search("random", 2, 24, [7, 8], FakeEvaluatorV2(0), tmp_path / f"r{i}", entrants=stub_entrants())
        for i in range(2)
    ]
    assert [r.fingerprint for r in outs[0].runs] == [r.fingerprint for r in outs[1].runs]
    manifest = outs[0].manifest
    assert manifest["design_space"] == V2 and manifest["config"]["shared_diversity"] is True
    assert manifest["config"]["proposer"]["min_distance"] == MIN_DISTANCE_V2
    for run in manifest["per_run"]:
        assert run["sampler"]["accepted"] == 16 and 0.6 < run["sampler"]["valid_rate"] < 1.0
        assert len(run["lift_best_so_far"]) == 24
    for run in outs[0].runs:
        recipes = [e.design for e in run.ctx.history if SPACE.cell(e.design) is not None and e.round > 0]
        others = [e.design for e in run.ctx.history if SPACE.cell(e.design) is not None]
        assert len({SPACE.design_id(e.design) for e in run.ctx.history}) == 24
        for r in recipes:
            assert min(SPACE.distance(r, o) for o in others if o is not r) >= MIN_DISTANCE_V2
    rows = read_table(tmp_path / "r0" / "results" / "designs.parquet")
    assert len(rows) == 48 and {"pad", "tip", "lift_success", "jam_on_close", "success_c2"} <= set(rows[0])
    assert {r["name"] for r in rows} == {"stock", "flat", "imprint_configurator", "imprint_swept", None}
    assert {r["baseline_kind"] for r in rows} == {
        "stock_blade",
        "recipe",
        "imprint_configurator",
        "imprint_swept",
        None,
    }
    frozen = outs[0].runs[0].frozen
    assert frozen.design_space == V2 and frozen.controller_setting == 0
