"""S1 analysis: endpoint definition, tau rule, exact permutation test, power, run loading."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from rtt.analysis import s1
from rtt.analysis.s1 import (
    GRID_V1,
    RunRecord,
    best_so_far,
    derive_tau,
    evals_to_tau,
    exact_sum_null,
    hit_rate,
    load_runs,
    logrank_permutation_test,
    permutation_test,
    power,
    reach_curve,
    restricted_mean_ci,
    round_to_grid,
)


def _run(arm: str, scores: list[float], seed: int = 1, round0: int = 8) -> RunRecord:
    origins = (
        ["entrant"] * 4
        + ["sobol"] * (round0 - 4)
        + [arm if arm == "random" else "nemotron"] * (len(scores) - round0)
    )
    return RunRecord(
        arm=arm,
        run_id=f"{arm}-r00",
        run_seed=seed,
        scores=tuple(scores),
        origins=tuple(origins),
        round0_size=round0,
        budget=len(scores),
        fallback_random=0,
        llm_calls=None,
        llm_cost_usd=None,
        source="test",
    )


def test_best_so_far_and_evals_to_tau() -> None:
    scores = [0.5, 0.9, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.6, 0.8125, 0.95]
    assert best_so_far(scores).tolist()[:3] == [0.5, 0.9, 0.9]
    # round 0 (designs 1-8) never counts for the arm endpoint, even when it holds a hit
    assert evals_to_tau(scores, 0.8125, round0_size=8) == 10
    assert evals_to_tau(scores, 0.8125, round0_size=8, count="all") == 2
    # censored: B + 1
    assert evals_to_tau(scores, 0.99, round0_size=8) == len(scores) + 1
    # a score a hair below tau because of float noise still counts
    assert evals_to_tau([0.0] * 8 + [13 / 16 - 1e-12], 13 / 16) == 9


def test_round_to_grid_rounds_halves_up() -> None:
    assert round_to_grid(0.8125, GRID_V1) == pytest.approx(0.8125)
    assert round_to_grid(0.84375, GRID_V1) == pytest.approx(0.875)  # exactly half a step
    assert round_to_grid(0.84, GRID_V1) == pytest.approx(0.8125)
    assert hit_rate([0.5, 0.8125, 0.9], 0.8125) == pytest.approx(2 / 3)
    assert hit_rate([], 0.5) == 0.0


def _brute_p(a: list[int], c: list[int]) -> float:
    pooled = a + c
    observed = sum(c)
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), len(c)):
        total += 1
        hits += sum(pooled[i] for i in idx) <= observed
    return hits / total


@pytest.mark.parametrize("seed", range(6))
def test_exact_permutation_p_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    r = 4 + seed % 3
    a = rng.integers(9, 42, r).tolist()
    c = rng.integers(9, 42, r).tolist()
    assert permutation_test(a, c).p_value == pytest.approx(_brute_p(a, c), abs=1e-12)


def test_exact_sum_null_counts_all_subsets() -> None:
    values = [3, 0, 5, 5, 2, 9]
    counts = exact_sum_null(values, 3)
    assert counts.sum() == math.comb(6, 3)
    brute = np.zeros_like(counts)
    for idx in itertools.combinations(values, 3):
        brute[sum(idx)] += 1
    assert counts.tolist() == brute.tolist()
    with pytest.raises(ValueError):
        exact_sum_null([-1, 2], 1)


def test_permutation_test_extremes() -> None:
    a = [41] * 10
    c = [9] * 10
    res = permutation_test(a, c)
    assert res.statistic == 32.0
    assert res.p_value == pytest.approx(1 / math.comb(20, 10))
    same = permutation_test([20] * 8, [20] * 8)
    assert same.p_value == 1.0  # every relabelling ties, and ties count against H1
    worse = permutation_test(c, a)
    assert worse.p_value == 1.0


def test_logrank_sensitivity_favours_the_faster_arm() -> None:
    fast = [9, 10, 12, 15, 11, 9, 13, 14]
    slow = [41, 30, 41, 41, 28, 41, 35, 41]
    assert logrank_permutation_test(slow, fast, 41, n_perm=2000).p_value < 0.01
    assert logrank_permutation_test(fast, slow, 41, n_perm=2000).p_value > 0.9


def test_restricted_mean_ci_and_reach_curve() -> None:
    ci = restricted_mean_ci([41, 41, 30, 20], [10, 12, 41, 9], n_boot=2000)
    assert ci["diff"] == pytest.approx((132 - 72) / 4)
    assert ci["ci_low"] <= ci["diff"] <= ci["ci_high"]
    curve = reach_curve([9, 41, 20], budget=40)
    assert (
        curve[8] == pytest.approx(1 / 3)
        and curve[19] == pytest.approx(2 / 3)
        and curve[-1] == pytest.approx(2 / 3)
    )


def test_tau_rule_median_floor_and_random_only() -> None:
    entrants = [0.6875, 0.75, 0.5, 0.6875]
    # a pool whose bootstrap best-at-40 median is ~0.875
    pool = [0.5] * 60 + [0.875] * 2
    res = derive_tau([], pool, entrants)
    assert res.tau == pytest.approx(0.875)
    assert res.floor == pytest.approx(0.8125)
    # weak pool: the median best is the best entrant, so the floor (one grid step above) binds
    weak = derive_tau([], [0.5] * 100, entrants)
    assert weak.tau == pytest.approx(0.8125) and weak.q_hits == 0
    # pilots' entrant rows never enter the pool, their random designs do
    pilot = _run("random", entrants + [0.5] * 4 + [0.9375] + [0.5] * 31)
    with_pilot = derive_tau([pilot], [0.5] * 100, entrants)
    assert with_pilot.pool_sources == {"calibration_sobol": 100, "random_pilot_designs": 36}
    assert with_pilot.pilot_best_at_b == [0.9375]
    with pytest.raises(ValueError, match="random-arm"):
        derive_tau([_run("nemotron", pilot.scores)], pool, entrants)
    # deterministic: same inputs, same tau and quantiles
    assert derive_tau([], pool, entrants).boot_quantiles == res.boot_quantiles


def test_power_null_is_calibrated_and_large_effects_are_detected() -> None:
    null = power(0.05, 1.0, 10, nsim=600, seed=3)
    assert null.power <= 0.08  # one-sided alpha 0.05 (exact test, discrete: conservative)
    big = power(0.05, 6.0, 10, nsim=300, seed=4)
    assert big.power > 0.8 and big.mean_t_c < big.mean_t_a
    het = power(0.05, 2.0, 10, nsim=200, seed=5, sigma=0.5)
    assert het.sigma == 0.5 and 0.0 <= het.power <= 1.0
    unc = power(0.05, 2.0, 10, nsim=200, seed=6, q_posterior=(5.5, 100.5))
    assert unc.q_uncertain


def _fake_search(tmp: Path, arm: str, runs: dict[str, list[float]], *, nested: bool) -> Path:
    root = tmp / f"{arm}-search"
    search = root / "search" if nested else root
    per_run = []
    for i, (run_id, scores) in enumerate(runs.items()):
        d = search / "runs" / run_id
        d.mkdir(parents=True)
        lines = []
        for j, sc in enumerate(scores, start=1):
            origin = "entrant" if j <= 4 else "sobol" if j <= 8 else arm
            lines.append(json.dumps({"index": j, "proposal": {"origin": origin}, "score": {"success": sc}}))
        (d / "history.jsonl").write_text("\n".join(reversed(lines)) + "\n")  # order is restored by index
        per_run.append(
            {
                "run_id": run_id,
                "run_seed": 100 + i,
                "fallback_random": 1,
                "llm": {"calls": 3, "cost_usd": 0.01},
            }
        )
    manifest = {
        "arm": arm,
        "budget": len(next(iter(runs.values()))),
        "round_sizes": [8, 8],
        "status": "complete",
    }
    manifest["per_run"] = per_run
    (search / "manifest.json").write_text(json.dumps(manifest))
    if nested:
        (root / "manifest.json").write_text(json.dumps({"files": []}))  # the cloud file manifest
        (root / "job.json").write_text(
            json.dumps(
                {
                    "job_id": "aijob-test",
                    "platform": "cpu-d3",
                    "wall_s": 120.0,
                    "attempts": [{"est_usd": 0.02}],
                }
            )
        )
    return root


def test_load_runs_reads_local_and_fetched_cloud_layouts(tmp_path: Path) -> None:
    scores = [0.5] * 9 + [0.875] + [0.5] * 6
    local = load_runs(_fake_search(tmp_path, "random", {"random-r00": scores}, nested=False))
    cloud = load_runs(_fake_search(tmp_path, "nemotron", {"nemotron-r00": scores}, nested=True))
    assert local[0].scores == tuple(scores) and local[0].job is None
    assert cloud[0].job == {
        "job_id": "aijob-test",
        "platform": "cpu-d3",
        "preset": None,
        "region": None,
        "wall_s": 120.0,
        "run_s": None,
        "est_usd": 0.02,
    }
    assert cloud[0].llm_calls == 3 and cloud[0].random_design_scores() == [0.5] * 4  # round-0 Sobol fill only
    assert local[0].random_design_scores() == [0.5] * 4 + [0.5, 0.875] + [0.5] * 6


def test_cli_writes_a_report(tmp_path: Path, capsys) -> None:
    scores_a = [0.6875, 0.75, 0.5, 0.6875] + [0.5] * 36
    scores_c = [0.6875, 0.75, 0.5, 0.6875] + [0.5] * 6 + [0.875] + [0.5] * 29
    ra = _fake_search(tmp_path, "random", {"random-r00": scores_a, "random-r01": scores_a}, nested=False)
    rc = _fake_search(tmp_path, "nemotron", {"nemotron-r00": scores_c, "nemotron-r01": scores_c}, nested=True)
    cal = tmp_path / "calibration.json"
    cal.write_text(
        json.dumps(
            {
                "sobol": [{"score": 0.5}] * 60 + [{"score": 0.875}] * 2,
                "entrants": [{"score": x} for x in scores_a[:4]],
            }
        )
    )
    out = tmp_path / "out"
    assert (
        s1.main(
            [
                "--random",
                str(ra),
                "--nemotron",
                str(rc),
                "--calibration",
                str(cal),
                "--compare",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads((out / "s1.json").read_text())
    # the random pilots' 0.5 designs join the pool, so the median best is the best entrant
    # and the floor (one grid step above it) sets tau
    assert report["tau"] == pytest.approx(0.8125)
    assert [r["t_arm"] for r in report["runs"]["C"]] == [11, 11]
    assert [r["t_arm"] for r in report["runs"]["A"]] == [41, 41]
    assert report["comparison"]["primary"]["p_value"] == pytest.approx(1 / 6)
    assert "S1 test" in capsys.readouterr().out


def test_geometric_mle_counts_exposure_up_to_the_first_hit() -> None:
    fit = s1.geometric_mle([9, 18, 41], round0_size=8, budget=40)
    # exposures 1, 10 and 32 (censored); 2 events
    assert fit["events"] == 2 and fit["exposure"] == 43
    assert fit["q"] == pytest.approx(2 / 43)
    assert fit["posterior"] == [2.5, 41.5]
    # the fit recovers a known rate from simulated runs
    rng = np.random.default_rng(0)
    q = 0.07
    g = rng.geometric(q, 4000)
    ts = np.where(g <= 32, 8 + g, 41).tolist()
    assert s1.geometric_mle(ts)["q"] == pytest.approx(q, rel=0.08)


def test_permutation_test_means_exact_and_monte_carlo_agree() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.1, 6).tolist()
    c = rng.normal(0.6, 0.1, 6).tolist()
    exact = s1.permutation_test_means(a, c)
    mc = s1.permutation_test_means(a, c, exact_limit=0, n_perm=40_000)
    assert exact.method.startswith("exact") and mc.method.startswith("Monte Carlo")
    assert mc.p_value == pytest.approx(exact.p_value, abs=0.01)
    assert exact.statistic == pytest.approx(np.mean(c) - np.mean(a))
    # brute force on a tiny case
    a2, c2 = [0.1, 0.4, 0.2], [0.5, 0.3, 0.9]
    pooled = a2 + c2
    obs = np.mean(c2) - np.mean(a2)
    hits = sum(
        (np.mean([pooled[i] for i in idx]) - np.mean([pooled[i] for i in range(6) if i not in idx]))
        >= obs - 1e-9
        for idx in itertools.combinations(range(6), 3)
    )
    assert s1.permutation_test_means(a2, c2).p_value == pytest.approx(hits / 20)


def test_holm_adjustment_and_best_so_far_area() -> None:
    adj = s1.holm({"H1": 0.01, "H2": 0.04, "S2": 0.03})
    assert adj == pytest.approx({"H1": 0.03, "S2": 0.06, "H2": 0.06})
    assert s1.holm({"x": 0.6, "y": 0.9}) == pytest.approx({"x": 1.0, "y": 1.0})
    scores = [0.5] * 8 + [0.6, 0.4, 0.8, 0.7]
    assert s1.best_so_far_area(scores) == pytest.approx((0.6 + 0.6 + 0.8 + 0.8) / 4)
