"""v2 analysis: tau rule, R rule, GV6 checks, GV3 calibration procedure, hero rule."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rtt.analysis import calibrate_v2, hero_rule, power, tau
from rtt.analysis.calibrate_v2 import (
    CalibrationLog,
    CalibrationV2Report,
    DesignScoreV2,
    check_knob,
    check_screening_seed,
    freeze_hashes,
    gv3_decision,
    score_designs_v2,
)
from rtt.analysis.hero_rule import (
    HERO_CONDITIONS,
    TRANSLATION_CONDITIONS,
    HeroRule,
    ScreenEpisode,
    apply_hero_rule,
    evaluate_part,
)
from rtt.analysis.s1 import RunRecord, permutation_test
from rtt.analysis.tau import GRID_V2, TauRuleV2, check_on_grid, derive_tau_v2, round_half_up

G = GRID_V2


def _run(arm: str, scores: list[float], seed: int = 101, round0: int = 8) -> RunRecord:
    origins = ["entrant"] * 4 + ["sobol"] * (round0 - 4) + [arm] * (len(scores) - round0)
    return RunRecord(
        arm=arm,
        run_id=f"{arm}-s{seed}",
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


ENTRANTS = [10 * G, 20 * G, 24 * G, 18 * G]  # stock, flat, configurator imprint, swept imprint


def _pilot(seed: int, arm_steps: list[int], fill: list[int] | None = None) -> RunRecord:
    fill = fill or [12, 14, 16, 20]
    steps = [10, 20, 24, 18, *fill, *arm_steps]
    assert len(steps) == 40
    return _run("random", [s * G for s in steps], seed=seed)


# --------------------------------------------------------------------------- tau rule


def test_grid_rounding_and_checks() -> None:
    assert round_half_up(0.5 * G + 10 * G) == pytest.approx(11 * G)  # half a step rounds up
    assert round_half_up(10.49 * G) == pytest.approx(10 * G)
    check_on_grid([0.0, 1.0, 17 * G])
    with pytest.raises(ValueError, match="grid"):
        check_on_grid([0.1])
    with pytest.raises(ValueError, match="outside"):
        check_on_grid([1.5])


def test_tau_is_the_median_of_bootstrapped_best_at_b() -> None:
    # pool almost all at 20/48, a thin tail at 40/48: P(best of 36 hits the tail) ~ 1 - (1-p)^36
    sobol = [20 * G] * 58 + [40 * G] * 6
    pilots = [_pilot(101 + i, [20] * 32) for i in range(3)]
    res = derive_tau_v2(pilots, sobol, ENTRANTS)
    p_tail = 6 / (64 + 3 * 36)
    p_best_tail = 1 - (1 - p_tail) ** 36
    assert p_best_tail > 0.5  # so the median best-at-B is the tail value
    assert res.tau == pytest.approx(40 * G) and res.binding == "median"
    assert res.tau_steps == 40 and not res.degenerate
    assert res.pool_size == 64 + 3 * 36
    assert res.pool_sources == {"calibration_sobol": 64, "random_pilot_designs": 108}
    # deterministic: same inputs, same answer
    assert derive_tau_v2(pilots, sobol, ENTRANTS).to_json() == res.to_json()


def test_tau_floor_is_two_steps_above_the_best_round0_score() -> None:
    sobol = [10 * G] * 64
    # one pilot's round-0 Sobol fill holds a 30/48 design: the floor is 32/48
    pilots = [_pilot(101, [10] * 32, fill=[30, 10, 10, 10]), _pilot(102, [10] * 32), _pilot(103, [10] * 32)]
    res = derive_tau_v2(pilots, sobol, ENTRANTS)
    assert res.best_round0 == pytest.approx(30 * G)
    assert res.tau == pytest.approx(32 * G) and res.binding == "floor"
    # without that design the best round-0 score is the configurator imprint (24/48)
    plain = [_pilot(101 + i, [10] * 32) for i in range(3)]
    assert derive_tau_v2(plain, sobol, ENTRANTS).tau == pytest.approx(26 * G)


def test_tau_cap_is_flagged_degenerate() -> None:
    sobol = [47 * G] * 64
    pilots = [_pilot(101 + i, [47] * 32, fill=[47, 47, 47, 47]) for i in range(3)]
    res = derive_tau_v2(pilots, sobol, ENTRANTS)
    assert res.tau == 1.0 and res.degenerate and res.binding == "cap"


def test_tau_rule_refuses_wrong_inputs() -> None:
    sobol = [10 * G] * 64
    good = [_pilot(101 + i, [10] * 32) for i in range(3)]
    with pytest.raises(ValueError, match="exactly 3"):
        derive_tau_v2(good[:2], sobol, ENTRANTS)
    nemo = _run("nemotron", list(good[0].scores), seed=104)
    with pytest.raises(ValueError, match="random-arm"):
        derive_tau_v2([*good[:2], nemo], sobol, ENTRANTS)
    with pytest.raises(ValueError, match="distinct"):
        derive_tau_v2([good[0], good[0], good[1]], sobol, ENTRANTS)
    with pytest.raises(ValueError, match="confirmatory"):
        derive_tau_v2(good, sobol, ENTRANTS, forbidden_seeds=[102, 1001])
    with pytest.raises(ValueError, match="Sobol-64"):
        derive_tau_v2(good, sobol[:63], ENTRANTS)
    with pytest.raises(ValueError, match="entrant"):
        derive_tau_v2(good, sobol, ENTRANTS[:3])
    off_grid = _run("random", [0.1] * 40, seed=109)
    with pytest.raises(ValueError, match="grid"):
        derive_tau_v2([*good[:2], off_grid], sobol, ENTRANTS)


def test_rule_digests_are_stable_and_sensitive() -> None:
    assert TauRuleV2().digest() == TauRuleV2().digest()
    assert TauRuleV2(margin_steps=1).digest() != TauRuleV2().digest()
    assert power.RRuleV2().digest() != power.RRuleV2(speedup=1.5).digest()
    assert HeroRule().digest() != HeroRule(designed_offset_mm=0.6).digest()


# --------------------------------------------------------------------------- run loading


def _fake_search(
    root: Path,
    arm: str,
    runs: dict[str, list[float]],
    *,
    design_space: str | None,
    task_hash: str = "task-a",
    lift_bonus: int = 4,
    per_design: int | None = 48,
) -> Path:
    out = root / f"{arm}-{len(list(root.iterdir()))}"
    per_run = []
    for i, (run_id, scores) in enumerate(runs.items()):
        hist = out / "runs" / run_id / "history.jsonl"
        hist.parent.mkdir(parents=True)
        origins = ["entrant"] * 4 + ["sobol"] * 4 + [arm] * (len(scores) - 8)
        lines = [
            json.dumps(
                {
                    "index": j + 1,
                    "score": {"success": s, "lift_success": min(1.0, max(0.0, s + lift_bonus * G))},
                    "proposal": {"origin": o},
                }
            )
            for j, (s, o) in enumerate(zip(scores, origins, strict=True))
        ]
        hist.write_text("\n".join(lines) + "\n")
        per_run.append({"run_id": run_id, "run_seed": 1001 + i, "fallback_random": 0, "llm": {}})
    manifest = {
        "kind": "search",
        "arm": arm,
        "status": "complete",
        "budget": len(next(iter(runs.values()))),
        "round_sizes": [8, 8, 8, 8, 8],
        "evaluator": {"episodes_per_design": per_design, "task_hash": task_hash, "suite_hash": "suite-a"},
        "per_run": per_run,
    }
    if design_space is not None:
        manifest["design_space"] = design_space
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def _scores_with_t(t: int, rng: np.random.Generator, tau_steps: int = 36) -> list[float]:
    steps = rng.integers(0, tau_steps, 40)
    steps[:8] = rng.integers(0, 49, 8)
    if t <= 40:
        steps[t - 1] = tau_steps + 2
    return [float(s) * G for s in steps]


def test_load_runs_v2_requires_the_v2_design_space(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    runs = {"r0": _scores_with_t(12, rng)}
    good = _fake_search(tmp_path, "random", runs, design_space="freeform-v2")
    loaded = tau.load_runs_v2(good)
    assert len(loaded) == 1 and loaded[0].scores == tuple(runs["r0"])
    lift = tau.load_runs_v2(good, score_key="lift_success")
    assert lift[0].scores[0] == pytest.approx(min(1.0, runs["r0"][0] + 4 * G))
    with pytest.raises(ValueError, match="design_space"):
        tau.load_runs_v2(_fake_search(tmp_path, "random", runs, design_space=None))
    with pytest.raises(ValueError, match="design_space"):
        tau.load_runs_v2(_fake_search(tmp_path, "random", runs, design_space="box14-v1"))
    with pytest.raises(KeyError, match="no_such"):
        tau.load_runs_v2(good, score_key="no_such")


def test_load_runs_v2_refuses_lift_only_scores_filed_as_primary(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    runs = {"r0": _scores_with_t(12, rng)}
    # registered above lift-only is impossible: the history holds a lift-only score as "success"
    swapped = _fake_search(tmp_path, "random", runs, design_space="freeform-v2", lift_bonus=-4)
    with pytest.raises(ValueError, match="higher registered than lift-only"):
        tau.load_runs_v2(swapped)
    with pytest.raises(ValueError, match="episodes per design"):
        tau.load_runs_v2(_fake_search(tmp_path, "random", runs, design_space="freeform-v2", per_design=None))


def test_analyze_refuses_arms_scored_on_different_tasks(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    a = _fake_search(tmp_path, "random", {"a0": _scores_with_t(30, rng)}, design_space="freeform-v2")
    c = _fake_search(
        tmp_path, "nemotron", {"c0": _scores_with_t(12, rng)}, design_space="freeform-v2", task_hash="task-b"
    )
    with pytest.raises(ValueError, match="different task/suite"):
        tau.main(["analyze", "--random", str(a), "--nemotron", str(c), "--tau", str(36 * G)])


def test_analyze_cli_recovers_a_planted_effect_end_to_end(tmp_path: Path) -> None:
    """GV6 'planted effect recovered': run dirs on disk -> CLI -> exact test."""
    rng = np.random.default_rng(7)
    slow = [41, 41, 30, 41, 25, 41, 41, 36, 41, 33]
    fast = [12, 15, 9, 20, 41, 11, 14, 10, 18, 13]
    a = _fake_search(
        tmp_path,
        "random",
        {f"a{i}": _scores_with_t(t, rng) for i, t in enumerate(slow)},
        design_space="freeform-v2",
    )
    c = _fake_search(
        tmp_path,
        "nemotron",
        {f"c{i}": _scores_with_t(t, rng) for i, t in enumerate(fast)},
        design_space="freeform-v2",
    )
    out = tmp_path / "out"
    rc = tau.main(
        ["analyze", "--random", str(a), "--nemotron", str(c), "--tau", str(36 * G), "--out", str(out)]
    )
    assert rc == 0
    report = json.loads((out / "s1_v2.json").read_text())
    primary = report["comparison"]["primary"]
    assert primary["statistic"] == pytest.approx(np.mean(slow) - np.mean(fast))
    assert primary["p_value"] == pytest.approx(permutation_test(slow, fast).p_value)
    assert primary["p_value"] < 0.001
    assert report["lift_only"]["status"] == "exploratory"
    assert report["design_space"] == "freeform-v2" and report["version"] == 2


def test_derive_cli_uses_the_rule_and_vets_seeds(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    runs = {f"p{i}": _scores_with_t(t, rng) for i, t in enumerate([20, 41, 33])}
    pilots = _fake_search(tmp_path, "random", runs, design_space="freeform-v2")
    cal = tmp_path / "calibration_v2.json"
    cal.write_text(
        json.dumps(
            {
                "schema": "rtt.calibration/2",
                "entrants": [{"score": s} for s in ENTRANTS],
                "sobol": [{"score": float(s) * G} for s in rng.integers(0, 40, 64)],
            }
        )
    )
    seeds = tmp_path / "seeds_main.yaml"
    seeds.write_text("seeds: [5001, 5002]\n")
    out = tmp_path / "out"
    argv = ["derive", "--random", str(pilots), "--calibration", str(cal), "--seeds-main", str(seeds)]
    assert tau.main([*argv, "--out", str(out)]) == 0
    report = json.loads((out / "s1_v2.json").read_text())
    assert report["tau_rule"]["rule"]["digest"] == TauRuleV2().digest()
    assert report["tau"] == pytest.approx(report["tau_rule"]["tau"])
    # the fake pilots carry seeds 1001-1003: confirmatory seeds are refused
    seeds.write_text("seeds: [1001, 1002]\n")
    with pytest.raises(ValueError, match="confirmatory"):
        tau.main(argv)


def test_load_calibration_v2_checks_schema(tmp_path: Path) -> None:
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"schema": "rtt.calibration/1", "entrants": [], "sobol": []}))
    with pytest.raises(ValueError, match="schema"):
        tau.load_calibration_v2(f)


# --------------------------------------------------------------------------- R rule and GV6


def test_r_rule_picks_the_smallest_powered_r() -> None:
    # at the pre-registered 2x no R in {8, 10, 12} reaches 0.8 for any q (about 0.47 at R = 12),
    # so the "chosen" branch is exercised with a large assumed speed-up
    easy = power.apply_r_rule(0.08, power.RRuleV2(nsim=300, speedup=8.0))
    assert easy.chosen in (8, 10) and easy.r_used == easy.chosen and easy.mde_speedup is None
    assert easy.cells[[8, 10, 12].index(easy.chosen)].power >= 0.8
    hard = power.apply_r_rule(0.01, power.RRuleV2(nsim=100))
    assert hard.chosen is None and hard.r_used == 12
    assert hard.mde_speedup is None or hard.mde_speedup > 2.0
    assert [c.runs_per_arm for c in hard.cells] == [8, 10, 12]
    assert all(c.speedup == 2.0 for c in hard.cells)


def test_q_estimates_and_report_shape() -> None:
    rng = np.random.default_rng(1)
    runs = [_run("random", _scores_with_t(t, rng), seed=101 + i) for i, t in enumerate([15, 41, 30])]
    pool = [36 * G] * 5 + [10 * G] * 95
    est = power.q_estimates(runs, pool, 36 * G)
    assert est["q_design"] == pytest.approx(0.05) and est["q_design_hits"] == 5
    assert est["random_t"] == [15, 41, 30]
    assert est["q_run"]["events"] == 2
    rep = power.r_rule_report(runs, pool, 36 * G, nsim=100)
    assert rep["r_rule"]["q_source"] == "q_design"
    assert len(rep["sensitivity"]["q_posterior"]) == 3 and len(rep["sensitivity"]["q_run_level"]) == 3


def test_expected_t_matches_simulation() -> None:
    rng = np.random.default_rng(5)
    q = 0.05
    g = rng.geometric(q, 200_000)
    t = np.where(g <= 32, 8 + g, 41)
    assert power.expected_t(q) == pytest.approx(t.mean(), abs=0.05)
    assert power.expected_t(0.0) == 41.0
    assert power.expected_t(1.0) == pytest.approx(9.0)


def test_randomized_p_brackets_the_conservative_p() -> None:
    a, c = [41, 30, 20, 41], [12, 41, 9, 15]
    cons = permutation_test(a, c).p_value
    assert power.randomized_p(a, c, 1.0) == pytest.approx(cons)
    assert power.randomized_p(a, c, 0.0) < cons


def test_gv6_null_p_values_are_uniform() -> None:
    res = power.null_calibration(0.04, 10, nsim=800)
    assert res["passed"], res
    assert res["type1_conservative"] <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 800)


def test_gv6_planted_effect_is_recovered() -> None:
    res = power.planted_effect(0.04, 2.5, 12, nsim=300, n_boot=1000)
    assert res["passed"], res
    assert res["rejection_rate"] > 0.2
    # and a null "effect" is not invented
    null = power.planted_effect(0.04, 1.0, 12, nsim=300, n_boot=1000)
    assert null["planted_diff"] == pytest.approx(0.0)
    assert null["passed"] and null["rejection_rate"] < 0.1


# --------------------------------------------------------------------------- GV3 calibration


def _row(name: str, origin: str, score: float, *, invalid: int = 0) -> DesignScoreV2:
    return DesignScoreV2(
        name=name,
        design_id=f"id-{name}",
        origin=origin,  # type: ignore[arg-type]
        score=score,
        lift_score=min(1.0, score + 0.1),
        n_episodes=48,
        mass_g=40.0,
        by_part={},
        by_placement={},
        failure_modes={},
        jam_on_close=0,
        off_centre=0,
        physics_invalid=invalid,
    )


def _report(sobol_steps: list[int], entrants: list[float] | None = None, **kw: object) -> CalibrationV2Report:
    ents = entrants or [0.25, 0.40, 0.50, 0.45]
    rows = [
        _row(n, "entrant", s) for n, s in zip(("stock", "flat", "imprint_c", "imprint_s"), ents, strict=True)
    ]
    rows += [_row("ref_wrap", "reference", 0.9), _row("ref_hook", "reference", 0.95)]
    rows += [_row(f"sobol-{i:02d}", "sobol", s * G) for i, s in enumerate(sobol_steps)]
    return CalibrationV2Report(rows=rows, screening_seed=calibrate_v2.SCREENING_SEED_V2, **kw)  # type: ignore[arg-type]


GOOD_SOBOL = [0] * 10 + [10] * 20 + [24] * 20 + [30] * 6 + [40] * 6 + [48] * 2


def test_gv3_passes_and_fails_on_the_right_criteria() -> None:
    rep = _report(GOOD_SOBOL)
    assert rep.passed, rep.table()
    assert rep.sobol_stats()["n_beat_best_entrant"] == 14
    assert gv3_decision(rep, 1) == ("freeze", [])
    data = rep.to_dict()
    assert (
        data["schema"] == "rtt.calibration/2" and len(data["entrants"]) == 4 and len(data["references"]) == 2
    )
    # references never gate, even far out of range
    assert all(c["passed"] for c in data["checks"] if c["gating"])

    out_of_range = _report(GOOD_SOBOL, entrants=[0.10, 0.40, 0.50, 0.45])
    assert out_of_range.failing() == ["entrants_in_range"]
    flat = _report([20] * 64)
    assert "sobol_sd" in flat.failing()
    ceiling = _report([0] * 20 + [48] * 44)
    assert "ceiling" in ceiling.failing()
    no_headroom = _report([0] * 30 + [10] * 30 + [25] * 4)
    assert no_headroom.failing() == ["headroom", "sobol_sd"] or "headroom" in no_headroom.failing()


def test_gv3_decision_after_the_last_iteration() -> None:
    headroom_only = _report([0] * 22 + [20] * 22 + [24] * 16 + [30] * 4)
    assert headroom_only.failing() == ["headroom"]
    assert gv3_decision(headroom_only, 2)[0] == "iterate"
    assert gv3_decision(headroom_only, 3) == ("declare_miss_and_freeze", ["headroom"])
    entrants_off = _report(GOOD_SOBOL, entrants=[0.05, 0.40, 0.50, 0.45])
    assert gv3_decision(entrants_off, 3)[0] == "f1"


def test_physics_invalid_share_is_reported_not_gating() -> None:
    rows_rep = _report(GOOD_SOBOL)
    rows_rep.rows[0] = _row("stock", "entrant", 0.25, invalid=48)
    check = {c.name: c for c in rows_rep.checks()}["physics_invalid_share"]
    assert not check.passed and not check.gating and rows_rep.passed


def test_score_designs_v2_from_records() -> None:
    recs = []
    for part in ("tee", "block", "hex", "bushing", "bracket", "slug"):
        for place in ("c0", "c1", "c2", "c3"):
            for seed in (0, 1):
                registered = place == "c0" or part == "slug"
                recs.append(
                    {
                        "design_id": "d1",
                        "part_id": part,
                        "placement": place,
                        "seed": seed,
                        "success": True,
                        "registered": registered,
                        "jam_on_close": not registered and part == "block",
                        "failure_mode": None if registered else "not_registered",
                    }
                )
    recs[0]["bad_qacc"] = 1  # physics_invalid: a failure even though it lifted and registered
    (row,) = score_designs_v2(recs, {"d1": ("imprint_c", "entrant")})
    assert row.n_episodes == 48
    assert row.score == pytest.approx((5 * 2 + 8 - 1) / 48)  # 5 parts at c0 + all of the slug, minus 1
    assert row.lift_score == pytest.approx(47 / 48)
    assert row.by_part["slug"] == pytest.approx(1.0) and row.by_placement["c0"] == pytest.approx(11 / 12)
    assert row.jam_on_close == 6 and row.physics_invalid == 1
    assert row.failure_modes == {"not_registered": 30}
    with pytest.raises(ValueError, match="47 episodes"):
        score_designs_v2(recs[1:], {"d1": ("x", "entrant")})
    with pytest.raises(ValueError, match="no episode"):
        score_designs_v2(recs, {"zz": ("x", "sobol")})
    # an explicit success_v2 wins over success & registered
    (alt,) = score_designs_v2([{**r, "success_v2": False} for r in recs], {"d1": ("x", "entrant")})
    assert alt.score == 0.0


def test_episode_outcome_reads_both_record_conventions() -> None:
    eo = calibrate_v2.episode_outcome
    # search-harness convention: success is success_v2, lift_success is lift-only
    assert eo({"success": False, "lift_success": True, "registered": False, "jam_on_close": True}) == (
        False,
        True,
    )
    assert eo({"success": True, "lift_success": True, "registered": True}) == (True, True)
    # sim-record convention: success is lift-only
    assert eo({"success": True, "registered": False}) == (False, True)
    assert eo({"success": True, "registered": True}) == (True, True)
    assert eo({"success_v2": True}) == (True, None)
    with pytest.raises(KeyError):
        eo({"success": True})  # neither registered nor success_v2: never guessed


@pytest.mark.parametrize(
    "rec",
    [
        {"success": True, "lift_success": True, "jam_on_close": True},  # jammed, yet registered
        {"success": True, "lift_success": True, "off_centre": True},
        {"success_v2": True, "success": True, "jam_on_close": True},
        {"success": True, "lift_success": False},  # registered without a lift
        {"success": True, "lift_success": True, "registered": False},
        {"success": True, "lift_success": True, "success_v2": False},
        {"success_v2": True, "success": False},
    ],
)
def test_contradictory_records_are_refused_not_scored(rec: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        calibrate_v2.episode_outcome(rec)


def test_score_designs_v2_with_harness_records() -> None:
    """Harness records (success = success_v2): lift-only must come from lift_success."""
    recs = [
        {
            "design_id": "imp",
            "success": i < 12,
            "lift_success": True,
            "registered": i < 12,
            "jam_on_close": i >= 12,
            "failure_mode": None if i < 12 else "not_registered",
        }
        for i in range(48)
    ]
    (row,) = score_designs_v2(recs, {"imp": ("imprint_configurator", "entrant")})
    assert row.score == pytest.approx(12 / 48) and row.lift_score == pytest.approx(1.0)
    assert row.jam_on_close == 36
    with pytest.raises(ValueError, match="lift-only"):
        score_designs_v2([{"design_id": "x", "success_v2": True}] * 48, {"x": ("x", "sobol")})


def test_screening_seed_and_knob_rules() -> None:
    check_screening_seed(calibrate_v2.SCREENING_SEED_V2, forbidden=[1001, 1002])
    for bad in (101, 20261018, 205):
        with pytest.raises(ValueError):
            check_screening_seed(bad)
    with pytest.raises(ValueError):
        check_screening_seed(777, forbidden=[777])
    assert check_knob("part:block.length_mm") == "part"
    assert check_knob("placement:c2") == "placement"
    assert check_knob("tolerance:seat_short_mm") == "tolerance"
    for bad in (
        # friction and grip force are fixed: the claims (PREREG A13) and notes are stated at them
        "task:mu",
        "task:grip_force_n",
        "entrant:stock.width",
        "design:pad_depth_mm",
        "task:timestep",
        "physics:impratio",
        "tolerance:x",
        "mu",
    ):
        with pytest.raises(ValueError):
            check_knob(bad)


def test_calibration_log_enforces_the_procedure(tmp_path: Path) -> None:
    path = tmp_path / "calibration_log.json"
    log = CalibrationLog.open(path)
    bad = _report([0] * 22 + [20] * 22 + [24] * 16 + [30] * 4)  # headroom fails
    with pytest.raises(ValueError, match="iteration 1 is the task as drafted"):
        log.record(bad, changes={"placement:c2": (-3.0, -2.5)})
    assert log.record(bad)[0] == "iterate"
    with pytest.raises(ValueError, match="must log what changed"):
        log.record(_report(GOOD_SOBOL, iteration=2))
    with pytest.raises(ValueError, match="may not be tuned"):
        log.record(_report(GOOD_SOBOL, iteration=2), changes={"entrant:flat.width": (20, 22)})
    with pytest.raises(ValueError, match="may not be tuned"):
        log.record(_report(GOOD_SOBOL, iteration=2), changes={"task:mu": (0.3, 0.35)})
    with pytest.raises(ValueError, match="iteration 3"):
        log.record(_report(GOOD_SOBOL, iteration=3), changes={"placement:c2": (-3.0, -2.5)})
    second = _report([0] * 22 + [20] * 22 + [24] * 16 + [30] * 4, iteration=2)
    assert log.record(second, changes={"tolerance:seat_short_mm": (1.0, 1.2)})[0] == "iterate"
    # tolerances may change only once
    with pytest.raises(ValueError, match="at most one iteration"):
        log.record(_report(GOOD_SOBOL, iteration=3), changes={"tolerance:yaw_deg": (2.0, 2.5)})
    # reopened from disk, the log still knows where it is
    log2 = CalibrationLog.open(path)
    assert len(log2.iterations) == 2 and log2.tolerance_changes() == 1
    decision, failing = log2.record(
        _report([0] * 22 + [20] * 22 + [24] * 16 + [30] * 4, iteration=3),
        changes={"part:block.width_mm": (40, 38)},
    )
    assert (decision, failing) == ("declare_miss_and_freeze", ["headroom"])
    with pytest.raises(ValueError, match="already closed"):
        log2.record(_report(GOOD_SOBOL, iteration=4), changes={"placement:c1": (2.0, 2.0)})
    saved = json.loads(path.read_text())
    assert [it["decision"] for it in saved["iterations"]] == ["iterate", "iterate", "declare_miss_and_freeze"]


def test_calibration_log_closes_on_pass_and_keeps_one_sample(tmp_path: Path) -> None:
    log = CalibrationLog.open(tmp_path / "log.json")
    assert log.record(_report(GOOD_SOBOL))[0] == "freeze"
    with pytest.raises(ValueError, match="already closed"):
        log.record(_report(GOOD_SOBOL, iteration=2), changes={"placement:c1": (2.0, 1.5)})
    other = CalibrationLog.open(tmp_path / "log2.json")
    other.record(_report([0] * 22 + [20] * 22 + [24] * 16 + [30] * 4))
    moved = _report(GOOD_SOBOL, iteration=2)
    moved.screening_seed = 20261030
    with pytest.raises(ValueError, match="same screening sample"):
        other.record(moved, changes={"placement:c1": (2.0, 1.5)})


def test_freeze_hashes(tmp_path: Path) -> None:
    (tmp_path / "configs" / "suites").mkdir(parents=True)
    for f in calibrate_v2.FREEZE_FILES[:3]:
        (tmp_path / f).write_text(f)
    with pytest.raises(FileNotFoundError, match="parts_v2"):
        freeze_hashes(tmp_path)
    (tmp_path / calibrate_v2.FREEZE_FILES[3]).write_text("parts")
    hashes = freeze_hashes(tmp_path)
    assert set(hashes) == set(calibrate_v2.FREEZE_FILES) and all(len(h) == 64 for h in hashes.values())


# --------------------------------------------------------------------------- hero rule


def _eps(**by_cond: dict[str, object]) -> list[ScreenEpisode]:
    """Episodes for every hero condition; ``by_cond`` overrides fields per condition."""
    base: dict[str, object] = {"success_v2": True, "offset_mm": 0.1}
    return [ScreenEpisode(condition=c, **{**base, **by_cond.get(c, {})}) for c in HERO_CONDITIONS]  # type: ignore[arg-type]


def _stock_fails(n_fail: int, n_visible: int) -> list[ScreenEpisode]:
    over: dict[str, dict[str, object]] = {}
    for i, c in enumerate(HERO_CONDITIONS):
        if i < n_visible:
            over[c] = {"success_v2": False, "failure_mode": "ejected_on_close"}
        elif i < n_fail:
            over[c] = {"success_v2": False, "failure_mode": "not_registered", "offset_mm": 2.0}
    return _eps(**over)


def _imprint(n_jam: int) -> list[ScreenEpisode]:
    eps = []
    for i, c in enumerate(TRANSLATION_CONDITIONS):
        jam = i < n_jam
        eps.append(
            ScreenEpisode(condition=c, success_v2=not jam, seat_short_mm=5.0 if jam else 0.1, offset_mm=0.2)
        )
    return eps


def test_tee_like_part_is_an_imprint_hero() -> None:
    v = evaluate_part("tee", _stock_fails(5, 3), {"d-good": _eps(), "d-bad": _stock_fails(6, 0)}, _imprint(5))
    assert v.a_ok and v.b_ok and v.c_ok and v.hero and v.imprint_hero
    assert v.best_design == "d-good" and v.design_passes == 7 and v.imprint_hits == 5


def test_round_part_is_a_hero_but_not_an_imprint_hero() -> None:
    v = evaluate_part("slug", _stock_fails(4, 2), {"d": _eps()}, _imprint(0))
    assert v.hero and not v.imprint_hero and not v.b_ok


def test_invisible_or_too_few_stock_failures_do_not_qualify() -> None:
    assert not evaluate_part("p", _stock_fails(6, 1), {"d": _eps()}).hero  # only 1 visible
    assert not evaluate_part("p", _stock_fails(3, 3), {"d": _eps()}).hero  # only 3 fails
    tilted = _eps(**{c: {"success_v2": False, "tilt_deg": 12.0} for c in list(HERO_CONDITIONS)[:4]})
    assert evaluate_part("p", tilted, {"d": _eps()}).a_ok  # tilt > 10 deg is visible


def test_designed_offset_and_pass_count() -> None:
    loose = _eps(**{c: {"offset_mm": 0.6} for c in list(HERO_CONDITIONS)[:2]})
    v = evaluate_part("p", _stock_fails(5, 3), {"d": loose})
    assert v.design_passes == 5 and not v.c_ok and not v.hero
    one_off = _eps(**{"yaw+5": {"success_v2": False}})
    assert evaluate_part("p", _stock_fails(5, 3), {"d": one_off}).c_ok


def test_ties_and_physics_invalid_count_against_qualification() -> None:
    # two episodes per condition, one failing: a tie, so the stock does not "fail" there
    tie = [
        e
        for c in HERO_CONDITIONS
        for e in (
            ScreenEpisode(condition=c, success_v2=False, failure_mode="ejected_on_close"),
            ScreenEpisode(condition=c, success_v2=True),
        )
    ]
    assert evaluate_part("p", tie, {"d": _eps()}).stock_fails == 0
    invalid = [
        ScreenEpisode(condition=c, success_v2=False, failure_mode="physics_invalid", physics_invalid=True)
        for c in HERO_CONDITIONS
    ]
    assert evaluate_part("p", invalid, {"d": _eps()}).stock_fails == 0
    good_but_invalid = [
        ScreenEpisode(condition=c, success_v2=True, physics_invalid=True) for c in HERO_CONDITIONS
    ]
    assert evaluate_part("p", _stock_fails(5, 3), {"d": good_but_invalid}).design_passes == 0


def test_brand_screen_and_missing_conditions() -> None:
    v = evaluate_part("scan", _stock_fails(5, 3), {"d": _eps()}, eligible=False)
    assert v.a_ok and v.c_ok and not v.hero
    with pytest.raises(ValueError, match="no episodes"):
        evaluate_part("p", _stock_fails(5, 3)[:-1], {"d": _eps()})
    with pytest.raises(ValueError, match="unknown hero condition"):
        ScreenEpisode(condition="x+4", success_v2=True)


def test_hook_mode_needs_three_heroes() -> None:
    hero = evaluate_part("tee", _stock_fails(5, 3), {"d": _eps()}, _imprint(5))
    other = evaluate_part("block", _stock_fails(5, 3), {"d": _eps()}, _imprint(6))
    dud = evaluate_part("hex", _stock_fails(1, 0), {"d": _eps()})
    two = apply_hero_rule([hero, other, dud])
    assert two.heroes == ("tee", "block") and two.hook_mode == "offset_contrast_only"
    three = apply_hero_rule([hero, other, evaluate_part("slug", _stock_fails(4, 2), {"d": _eps()}), dud])
    assert three.hook_mode == "hero_triptych" and three.imprint_heroes == ("tee", "block")
    assert three.label == "showcase parts chosen because the stock finger fails"
    assert three.to_json()["rule_digest"] == HeroRule().digest()
    with pytest.raises(ValueError, match="duplicate"):
        apply_hero_rule([hero, hero])


def test_screen_episode_from_v2_record() -> None:
    rec = {
        "condition": "x+2",
        "success": True,
        "registered": False,
        "reg_dx_mm": 1.2,
        "reg_dy_mm": 0.5,
        "seat_short_mm": 4.8,
        "reg_tilt_deg": 2.0,
        "hold_tilt_deg": 11.0,
        "failure_mode": "slip",
    }
    ep = ScreenEpisode.from_record(rec)
    assert not ep.success_v2 and ep.offset_mm == pytest.approx(1.3) and ep.tilt_deg == 11.0
    assert ep.mode == "slip_during_lift" and not ep.physics_invalid
    bad = ScreenEpisode.from_record({**rec, "registered": True, "bad_qacc": 3})
    assert bad.physics_invalid and not bad.success_v2


def test_unmeasured_values_count_against_qualification() -> None:
    # a designed episode with no registration pose is never a centred pass
    blind = [ScreenEpisode.from_record({"condition": c, "success_v2": True}) for c in HERO_CONDITIONS]
    assert all(e.offset_mm is None for e in blind)
    assert evaluate_part("p", _stock_fails(5, 3), {"d": blind}).design_passes == 0
    # an imprint episode with no shortfall or offset is never a jam
    unmeasured = [ScreenEpisode(condition=c, success_v2=False) for c in TRANSLATION_CONDITIONS]
    assert evaluate_part("p", _stock_fails(5, 3), {"d": _eps()}, unmeasured).imprint_hits == 0
    # an unmeasured tilt is never a visible failure
    quiet = [
        ScreenEpisode(condition=c, success_v2=False, failure_mode="not_registered") for c in HERO_CONDITIONS
    ]
    assert evaluate_part("p", quiet, {"d": _eps()}).stock_visible == 0


def test_non_finite_measurements() -> None:
    rec = {
        "condition": "x+1",
        "success": False,
        "registered": False,
        "reg_dx_mm": float("nan"),
        "reg_dy_mm": 0.0,
    }
    ep = ScreenEpisode.from_record({**rec, "bad_qacc": 2})  # a NaN reset: unmeasured, not a crash
    assert ep.physics_invalid and ep.offset_mm is None and not ep.success_v2
    with pytest.raises(ValueError, match="valid episode"):
        ScreenEpisode.from_record(rec)


def test_hero_cli_round_trip(tmp_path: Path) -> None:
    def recs(eps: list[ScreenEpisode]) -> list[dict[str, object]]:
        return [
            {
                "condition": e.condition,
                "success_v2": e.success_v2,
                "failure_mode": e.failure_mode,
                "seat_short_mm": e.seat_short_mm,
                "reg_dx_mm": e.offset_mm,
                "reg_dy_mm": 0.0,
                "tilt_max_deg": e.tilt_deg,
            }
            for e in eps
        ]

    screen = {
        "parts": [
            {
                "part_id": "tee",
                "stock": recs(_stock_fails(5, 3)),
                "imprint": recs(_imprint(5)),
                "designed": {"d": recs(_eps())},
            },
            {
                "part_id": "hex",
                "stock": recs(_stock_fails(1, 0)),
                "imprint": None,
                "designed": {"d": recs(_eps())},
            },
        ]
    }
    f = tmp_path / "screen.json"
    f.write_text(json.dumps(screen))
    assert hero_rule.main([str(f), "--out", str(tmp_path / "o")]) == 0
    out = json.loads((tmp_path / "o" / "hero_screen.json").read_text())
    assert (
        out["heroes"] == ["tee"]
        and out["imprint_heroes"] == ["tee"]
        and out["hook_mode"] == "offset_contrast_only"
    )


# --------------------------------------------------------------------------- hygiene


V2_MODULES = ("calibrate_v2", "tau", "power", "hero_rule")


@pytest.mark.parametrize("name", V2_MODULES)
def test_v2_analysis_modules_stay_light(name: str) -> None:
    """No simulator, model client or v1 design import: the analysis runs on any laptop."""
    src = (Path(tau.__file__).parent / f"{name}.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    heavy = {"mujoco", "trimesh", "coacd", "manifold3d", "openai", "warp", "mujoco_warp"}
    assert not imported & heavy
    assert not any(
        m.startswith(("rtt.sim", "rtt.design", "rtt.search", "rtt.llm", "rtt.cloud")) for m in imported
    )
    code = f"import sys, rtt.analysis.{name}; print(int(any(m in sys.modules for m in {sorted(heavy)!r})))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "0"
