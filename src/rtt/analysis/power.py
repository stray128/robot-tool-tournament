"""The v2 R rule and the GV6 checks of the S1 analysis (``freeform-v2`` design space).

The model, the test and the Monte Carlo power function are those of :mod:`rtt.analysis.s1`:
after round 0, each random design reaches tau independently with probability ``q``; a
speed-up ``s`` gives arm C ``q_C = 1 - (1 - q)^s``; every simulated experiment is analysed
with the exact permutation test of the restricted mean of ``T``.

**R rule v2** (:func:`apply_r_rule`, fixed before the pilot, founder decision D6)

1. ``q`` is the per-design hit rate at tau of the random pool that set tau: the Sobol-64
   calibration sample plus every non-entrant design of the three random pilots
   (:func:`q_estimates`). With only three pilot runs a run-level fit has at most three
   events, so the censored-geometric fit to the pilots' ``T`` is reported as a sensitivity
   analysis, as is power with ``q`` drawn from its Jeffreys posterior.
2. R is the smallest value in {8, 10, 12} whose Monte Carlo power (4,000 experiments,
   one-sided alpha 0.05) is at least 0.8 at an assumed 2x speed-up **[assumed]**.
3. If none qualifies, R = 12 and the minimum detectable speed-up at R = 12 (80 % power,
   on a 0.25 grid) is declared in PREREG.

**GV6 checks** (gate GV6, before the ``prereg-v1`` tag)

* :func:`null_calibration`: under s = 1 the randomised exact p-values must be uniform
  (Kolmogorov-Smirnov p >= 0.01), and the conservative test's rejection rate at alpha must
  not exceed alpha by more than 3 Monte Carlo standard errors.
* :func:`planted_effect`: synthetic runs with a planted speed-up, written as score
  sequences on the 1/48 grid and analysed by the same functions as the real runs
  (``evals_to_tau``, ``permutation_test``, ``restricted_mean_ci``), must recover the planted
  difference in restricted means without bias (within 3 Monte Carlo standard errors), and
  its 95 % percentile-bootstrap intervals must cover the planted difference in at least
  90 % of experiments (less 3 Monte Carlo standard errors; the percentile bootstrap
  under-covers slightly at R = 8-12).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

from rtt.analysis.s1 import (
    PowerCell,
    RunRecord,
    evals_to_tau,
    exact_sum_null,
    geometric_mle,
    hit_rate,
    minimum_detectable_speedup,
    permutation_test,
    power,
    restricted_mean_ci,
)
from rtt.analysis.tau import GRID_V2

__all__ = [
    "RRuleResult",
    "RRuleV2",
    "apply_r_rule",
    "expected_t",
    "main",
    "null_calibration",
    "planted_effect",
    "q_estimates",
    "r_rule_report",
    "randomized_p",
]

_EPS: Final = 1e-9


@dataclass(frozen=True)
class RRuleV2:
    """Constants of the v2 R rule (module docstring); hashed into PREREG."""

    candidates: tuple[int, ...] = (8, 10, 12)
    speedup: float = 2.0
    target: float = 0.8
    alpha: float = 0.05
    fallback: int = 12
    nsim: int = 4000
    seed: int = 20261017
    budget: int = 40
    round0_size: int = 8

    def digest(self) -> str:
        blob = json.dumps({"rule": "R-v2", "version": 1, **asdict(self)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class RRuleResult:
    q: float
    q_source: str
    chosen: int | None
    r_used: int
    cells: tuple[PowerCell, ...]
    mde_speedup: float | None
    rule: RRuleV2

    def to_json(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "q_source": self.q_source,
            "chosen": self.chosen,
            "r_used": self.r_used,
            "cells": [asdict(c) for c in self.cells],
            "mde_speedup_at_r_used": self.mde_speedup,
            "rule": {**asdict(self.rule), "digest": self.rule.digest()},
        }


def q_estimates(
    random_runs: Sequence[RunRecord], pool_scores: Sequence[float], tau: float, rule: RRuleV2 | None = None
) -> dict[str, Any]:
    """Primary ``q`` (pooled per-design hit rate) and the run-level fit (sensitivity)."""
    rule = rule or RRuleV2()
    hits = int(sum(s >= tau - _EPS for s in pool_scores))
    n = len(pool_scores)
    ts = [evals_to_tau(r.scores, tau, round0_size=rule.round0_size) for r in random_runs]
    return {
        "q_design": hit_rate(pool_scores, tau),
        "q_design_hits": hits,
        "q_design_n": n,
        "q_design_posterior": [hits + 0.5, n - hits + 0.5],
        "q_run": geometric_mle(ts, round0_size=rule.round0_size, budget=rule.budget),
        "random_t": ts,
    }


def apply_r_rule(q: float, rule: RRuleV2 | None = None, *, q_source: str = "q_design") -> RRuleResult:
    """The R rule: smallest candidate with power >= target at the assumed speed-up, else the fallback."""
    rule = rule or RRuleV2()
    cells = tuple(
        power(
            q,
            rule.speedup,
            r,
            budget=rule.budget,
            round0_size=rule.round0_size,
            alpha=rule.alpha,
            nsim=rule.nsim,
            seed=rule.seed + i,
        )
        for i, r in enumerate(rule.candidates)
    )
    chosen = next((c.runs_per_arm for c in cells if c.power >= rule.target), None)
    r_used = chosen if chosen is not None else rule.fallback
    mde = None
    if chosen is None and q > 0:
        mde = minimum_detectable_speedup(
            q, r_used, budget=rule.budget, round0_size=rule.round0_size, nsim=max(50, rule.nsim // 2)
        )
    return RRuleResult(q, q_source, chosen, r_used, cells, mde, rule)


def r_rule_report(
    random_runs: Sequence[RunRecord],
    pool_scores: Sequence[float],
    tau: float,
    *,
    nsim: int | None = None,
    rule: RRuleV2 | None = None,
) -> dict[str, Any]:
    """The R rule on the primary ``q``, plus power at the run-level ``q`` and under ``q`` uncertainty."""
    rule = rule or RRuleV2()
    if nsim is not None:
        rule = RRuleV2(**{**asdict(rule), "nsim": nsim})
    est = q_estimates(random_runs, pool_scores, tau, rule)
    result = apply_r_rule(est["q_design"], rule)
    a, b = est["q_design_posterior"]
    sens_posterior = [
        asdict(
            power(
                est["q_design"],
                rule.speedup,
                r,
                budget=rule.budget,
                round0_size=rule.round0_size,
                nsim=max(1000, rule.nsim // 2),
                q_posterior=(a, b),
                seed=rule.seed + 100 + r,
            )
        )
        for r in rule.candidates
    ]
    q_run = float(est["q_run"]["q"])
    sens_run = [
        asdict(power(q_run, rule.speedup, r, nsim=max(1000, rule.nsim // 2), seed=rule.seed + 200 + r))
        for r in rule.candidates
    ]
    return {
        "model": "censored geometric after round 0; arm C hit rate 1-(1-q)^s",
        "q": est,
        "r_rule": {**result.to_json(), "q": result.q},
        "sensitivity": {"q_posterior": sens_posterior, "q_run_level": sens_run},
    }


# --------------------------------------------------------------------------- GV6 checks


def randomized_p(t_a: Sequence[int], t_c: Sequence[int], u: float) -> float:
    """Randomised exact permutation p-value: P(S < s) + u P(S = s), S the arm-C sum under the null.

    For GV6 only. With ``u`` uniform on [0, 1) it is exactly uniform under exchangeability;
    the confirmatory test (:func:`rtt.analysis.s1.permutation_test`) is its conservative
    ``u = 1`` form.
    """
    a = [int(x) for x in t_a]
    c = [int(x) for x in t_c]
    counts = exact_sum_null(a + c, len(c))
    observed = sum(c)
    total = counts.sum()
    return float((counts[:observed].sum() + u * counts[observed]) / total)


def expected_t(q: float, *, budget: int = 40, round0_size: int = 8) -> float:
    """E[T] under the censored geometric: round0 + sum_{k=0}^{n} (1-q)^k, n = budget - round0."""
    n = budget - round0_size
    if q <= 0:
        return float(budget + 1)
    return round0_size + (1.0 - (1.0 - q) ** (n + 1)) / q


def _simulate(q: float, runs: int, n_arm: int, round0_size: int, rng: np.random.Generator) -> np.ndarray:
    g = rng.geometric(np.clip(np.full(runs, q), 1e-12, 1.0))
    return np.where(g <= n_arm, round0_size + g, round0_size + n_arm + 1)


def null_calibration(
    q: float,
    runs_per_arm: int,
    *,
    nsim: int = 2000,
    alpha: float = 0.05,
    budget: int = 40,
    round0_size: int = 8,
    seed: int = 20261021,
) -> dict[str, Any]:
    """GV6: p-values under the null (both arms at ``q``) are uniform and the test holds its size."""
    from scipy import stats

    rng = np.random.default_rng(seed)
    n_arm = budget - round0_size
    p_cons = np.empty(nsim)
    p_rand = np.empty(nsim)
    for i in range(nsim):
        t_a = _simulate(q, runs_per_arm, n_arm, round0_size, rng)
        t_c = _simulate(q, runs_per_arm, n_arm, round0_size, rng)
        p_cons[i] = permutation_test(t_a, t_c).p_value
        p_rand[i] = randomized_p(t_a, t_c, float(rng.random()))
    ks = stats.kstest(p_rand, "uniform")
    type1 = float(np.mean(p_cons <= alpha + _EPS))
    se = math.sqrt(alpha * (1 - alpha) / nsim)
    return {
        "q": q,
        "runs_per_arm": runs_per_arm,
        "nsim": nsim,
        "ks_statistic": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "type1_conservative": type1,
        "type1_limit": alpha + 3 * se,
        "passed": bool(ks.pvalue >= 0.01 and type1 <= alpha + 3 * se),
    }


def _score_sequence(
    t: int, tau_steps: int, budget: int, round0_size: int, rng: np.random.Generator
) -> list[float]:
    """Scores on the 1/48 grid whose first arm design at or above tau is design ``t`` (B + 1: none).

    Round 0 may hold anything, including tau-level designs: it never counts toward ``T``.
    """
    below = rng.integers(0, tau_steps, budget)  # strictly below tau
    above = rng.integers(tau_steps, 49, budget)  # at or above tau
    steps = below.copy()
    steps[:round0_size] = rng.integers(0, 49, round0_size)
    if t <= budget:
        steps[t - 1] = above[t - 1]
        # designs after the first hit may score anything
        later = rng.random(budget) < 0.3
        steps[t:] = np.where(later[t:], above[t:], below[t:])
    return [float(s) * GRID_V2 for s in steps]


def planted_effect(
    q: float,
    speedup: float,
    runs_per_arm: int,
    *,
    tau_steps: int = 36,
    nsim: int = 500,
    n_boot: int = 2000,
    alpha: float = 0.05,
    budget: int = 40,
    round0_size: int = 8,
    seed: int = 20261022,
) -> dict[str, Any]:
    """GV6: synthetic runs with a planted speed-up; the analysis must recover the planted effect.

    Each simulated run is a score sequence on the 1/48 grid (round 0 below tau, then
    geometric hits), so ``T`` is recomputed by :func:`~rtt.analysis.s1.evals_to_tau` exactly
    as for real runs.
    """
    if not 0 < tau_steps <= 48:
        raise ValueError("tau_steps must be in 1..48")
    rng = np.random.default_rng(seed)
    n_arm = budget - round0_size
    q_c = 1.0 - (1.0 - q) ** speedup
    tau = tau_steps * GRID_V2
    planted = expected_t(q, budget=budget, round0_size=round0_size) - expected_t(
        q_c, budget=budget, round0_size=round0_size
    )
    est = np.empty(nsim)
    covered = rejected = 0
    for i in range(nsim):
        t_true_a = _simulate(q, runs_per_arm, n_arm, round0_size, rng)
        t_true_c = _simulate(q_c, runs_per_arm, n_arm, round0_size, rng)
        t_a = [
            evals_to_tau(
                _score_sequence(int(t), tau_steps, budget, round0_size, rng), tau, round0_size=round0_size
            )
            for t in t_true_a
        ]
        t_c = [
            evals_to_tau(
                _score_sequence(int(t), tau_steps, budget, round0_size, rng), tau, round0_size=round0_size
            )
            for t in t_true_c
        ]
        if t_a != t_true_a.tolist() or t_c != t_true_c.tolist():
            raise AssertionError("evals_to_tau did not recover the planted T")  # pipeline bug
        ci = restricted_mean_ci(t_a, t_c, n_boot=n_boot, seed=int(rng.integers(2**31)))
        est[i] = ci["diff"]
        covered += ci["ci_low"] - _EPS <= planted <= ci["ci_high"] + _EPS
        rejected += permutation_test(t_a, t_c).p_value <= alpha + _EPS
    bias = float(est.mean() - planted)
    se = float(est.std(ddof=1) / math.sqrt(nsim))
    coverage = covered / nsim
    coverage_limit = 0.90 - 3 * math.sqrt(0.90 * 0.10 / nsim)
    return {
        "q": q,
        "q_c": q_c,
        "speedup": speedup,
        "runs_per_arm": runs_per_arm,
        "nsim": nsim,
        "planted_diff": planted,
        "mean_estimate": float(est.mean()),
        "bias": bias,
        "bias_se": se,
        "ci_coverage": coverage,
        "rejection_rate": rejected / nsim,
        "coverage_limit": coverage_limit,
        "passed": bool(abs(bias) <= 3 * se + _EPS and coverage >= coverage_limit),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rtt.analysis.power", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("r-rule", help="apply the R rule at a given q")
    r.add_argument("--q", type=float, required=True)
    g = sub.add_parser("gv6", help="null calibration and planted-effect checks")
    g.add_argument("--q", type=float, required=True)
    g.add_argument("--runs", type=int, required=True, help="R per arm")
    g.add_argument("--speedup", type=float, default=2.0)
    g.add_argument("--nsim", type=int, default=2000)
    sub.add_parser("rule", help="print the R rule constants and digest")
    args = ap.parse_args(argv)
    if args.cmd == "rule":
        rule = RRuleV2()
        print(json.dumps({**asdict(rule), "digest": rule.digest()}, indent=1))
    elif args.cmd == "r-rule":
        res = apply_r_rule(args.q)
        for c in res.cells:
            print(f"R = {c.runs_per_arm:>2}: power {c.power:.3f} at s = {c.speedup:g}")
        print(f"chosen {res.chosen}, R used {res.r_used}, MDE {res.mde_speedup}")
    else:
        null = null_calibration(args.q, args.runs, nsim=args.nsim)
        planted = planted_effect(args.q, args.speedup, args.runs, nsim=max(200, args.nsim // 4))
        print(json.dumps({"null": null, "planted": planted}, indent=1))
        return 0 if null["passed"] and planted["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
