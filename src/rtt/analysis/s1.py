"""S1, the primary endpoint: simulated designs needed to reach a training-success threshold.

Everything here reads finished search outputs (``manifest.json`` plus
``runs/<run_id>/history.jsonl``); nothing simulates or calls a model. The same code
serves the pilot (derive tau, estimate power, choose R) and the confirmatory analysis
(tau and R read from ``docs/PREREG.md``).

**Definitions** (written into ``docs/PREREG.md``)

* A run's designs are numbered 1..B in evaluation order (the ``index`` field). Designs
  1..``round0_size`` are the shared round 0: the manual entrants plus a Sobol fill seeded
  only by the run seed, byte-identical in every arm for a given run seed.
* ``T`` (evaluations to tau) is the index of the first design **proposed by the arm**
  (index > ``round0_size``) whose training score is at least tau. A run that never gets
  there is censored at ``B + 1``. Round-0 designs are excluded because they are the same
  for both arms and so carry no information about the arm; ``count="all"`` (round 0
  included) is reported as a sensitivity analysis only.
* The best-so-far curve is the running maximum of the training score.

**tau rule** (:func:`derive_tau`; the v1 form of the rule fixed before the pilot)

1. Pool every random-arm design score available before the confirmatory runs: the
   Sobol-64 calibration sample plus every non-entrant design of the random pilot runs.
2. Bootstrap ``n_boot`` random runs of budget B: the manual entrants' scores plus
   ``B - n_entrants`` scores drawn with replacement from the pool; best-at-B is the max.
3. ``tau_raw`` is the median of the bootstrapped best-at-B. tau is ``tau_raw`` rounded to
   the score grid (1/16 on ``train_v1``: 16 conditions, best controller setting), half up.
4. tau is raised, if needed, to one grid step above the best manual entrant (the fixed
   part of round 0), and capped at 1.

The Nemotron pilots never enter the rule; they are used only for valid and fallback rates.

**Test** (:func:`permutation_test`): a one-sided exact permutation test of the
restricted mean of ``T`` (all censoring is administrative, at B + 1), H1: arm C needs
fewer evaluations than arm A. With equal arm sizes the statistic ``mean(T_A) - mean(T_C)``
is a monotone function of ``sum(T_C)``, and ``T`` is an integer, so the exact null
distribution over all ``C(2R, R)`` relabellings is a subset-sum count (dynamic
programming), exact for any R in microseconds. It needs no proportional-hazards
assumption, has no large-sample approximation, and its effect estimate reads directly
as "designs saved". A permutation log-rank test is reported as a sensitivity analysis.

**Power** (:func:`power`): Monte Carlo under a geometric model. After round 0, each
random design reaches tau independently with probability ``q``, fitted to the random
runs' ``T`` values (:func:`geometric_mle`; the per-design hit rate of the pool is a
sensitivity analysis). A speed-up ``s`` means each arm-C design is worth ``s`` random designs:
``q_C = 1 - (1 - q)^s``. Optional run-to-run heterogeneity draws each arm-C run's
speed-up from a log-normal with median ``s`` and log-SD ``sigma``; optional ``q``
uncertainty draws ``q`` per simulated experiment from its beta posterior.

Command line: ``rtt analyze-s1 --random DIR... [--nemotron DIR...] --calibration FILE``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

__all__ = [
    "GRID_V1",
    "PowerCell",
    "RunRecord",
    "TauResult",
    "TauRule",
    "TestResult",
    "best_so_far",
    "best_so_far_area",
    "derive_tau",
    "evals_to_tau",
    "exact_sum_null",
    "geometric_mle",
    "hit_rate",
    "holm",
    "load_calibration_scores",
    "load_runs",
    "logrank_permutation_test",
    "main",
    "permutation_test",
    "permutation_test_means",
    "power",
    "power_report",
    "power_table",
    "reach_curve",
    "restricted_mean_ci",
    "round_to_grid",
]

#: Score grid of ``train_v1``: success is the best controller setting's mean over 16 conditions.
GRID_V1: Final = 1.0 / 16.0
_EPS: Final = 1e-9
Count = Literal["arm", "all"]


# --------------------------------------------------------------------------- run records


@dataclass(frozen=True)
class RunRecord:
    """One search run, reduced to what S1 needs (scores in evaluation order)."""

    arm: str
    run_id: str
    run_seed: int
    scores: tuple[float, ...]
    origins: tuple[str, ...]
    round0_size: int
    budget: int
    fallback_random: int
    llm_calls: int | None
    llm_cost_usd: float | None
    source: str
    job: dict[str, Any] | None = None

    @property
    def best(self) -> float:
        return max(self.scores)

    def entrant_scores(self) -> list[float]:
        return [s for s, o in zip(self.scores, self.origins, strict=True) if o == "entrant"]

    def random_design_scores(self) -> list[float]:
        """Scores of the run's random-arm draws (round-0 Sobol fill plus later random designs)."""
        return [s for s, o in zip(self.scores, self.origins, strict=True) if o in ("sobol", "random")]


def _manifest_path(directory: Path) -> Path:
    # a fetched Nebius run keeps its own file manifest at the top and the search under search/
    for candidate in (directory / "search" / "manifest.json", directory / "manifest.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{directory}: no manifest.json (or search/manifest.json)")


def _job_facts(directory: Path) -> dict[str, Any] | None:
    """Job ID, platform, wall time and wall-based cost of a fetched Nebius run, if any."""
    job_file = directory / "job.json"
    if not job_file.is_file():
        return None
    job = json.loads(job_file.read_text(encoding="utf-8"))
    attempts = job.get("attempts") or []
    est = sum(float(a.get("est_usd") or 0.0) for a in attempts)
    return {
        "job_id": job.get("job_id"),
        "platform": job.get("platform"),
        "preset": job.get("preset"),
        "region": job.get("region"),
        "wall_s": job.get("wall_s"),
        "run_s": job.get("run_s"),
        "est_usd": round(est, 4) if attempts else None,
    }


def load_runs(directory: str | Path) -> list[RunRecord]:
    """Every run of one search output (a ``rtt search`` dir or a fetched ``runs/cloud/<id>``)."""
    root = Path(directory)
    manifest_file = _manifest_path(root)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("status") not in (None, "complete"):
        raise ValueError(f"{manifest_file}: status {manifest.get('status')!r}, not a finished search")
    search_dir = manifest_file.parent
    sizes = manifest.get("round_sizes") or []
    round0 = int(sizes[0]) if sizes else int(manifest.get("config", {}).get("round0_size", 8))
    budget = int(manifest["budget"])
    job = _job_facts(root)
    out: list[RunRecord] = []
    for run in manifest["per_run"]:
        history = search_dir / "runs" / run["run_id"] / "history.jsonl"
        items = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line]
        items.sort(key=lambda item: int(item["index"]))
        if [int(i["index"]) for i in items] != list(range(1, len(items) + 1)):
            raise ValueError(f"{history}: design indices are not 1..n")
        if len(items) != budget:
            raise ValueError(f"{history}: {len(items)} designs, budget {budget}")
        llm = run.get("llm") or {}
        out.append(
            RunRecord(
                arm=str(manifest["arm"]),
                run_id=str(run["run_id"]),
                run_seed=int(run["run_seed"]),
                scores=tuple(float(i["score"]["success"]) for i in items),
                origins=tuple(str(i["proposal"].get("origin")) for i in items),
                round0_size=round0,
                budget=budget,
                fallback_random=int(run.get("fallback_random") or 0),
                llm_calls=int(llm["calls"]) if "calls" in llm else None,
                llm_cost_usd=float(llm["cost_usd"]) if "cost_usd" in llm else None,
                source=str(root),
                job=job,
            )
        )
    return out


def load_calibration_scores(path: str | Path) -> tuple[list[float], list[float]]:
    """``(sobol_scores, entrant_scores)`` from a ``rtt calibrate`` ``calibration.json``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sobol = [float(d["score"]) for d in data["sobol"]]
    entrants = [float(d["score"]) for d in data["entrants"]]
    return sobol, entrants


# --------------------------------------------------------------------------- endpoint


def best_so_far(scores: Sequence[float]) -> np.ndarray:
    """Running maximum of ``scores`` (index i holds the best of designs 1..i+1)."""
    return np.maximum.accumulate(np.asarray(scores, dtype=float))


def evals_to_tau(scores: Sequence[float], tau: float, *, round0_size: int = 8, count: Count = "arm") -> int:
    """S1's ``T``: index (1-based) of the first counted design scoring >= ``tau``; ``B + 1`` if none."""
    start = round0_size if count == "arm" else 0
    for i in range(start, len(scores)):
        if scores[i] >= tau - _EPS:
            return i + 1
    return len(scores) + 1


def reach_curve(ts: Sequence[int], budget: int) -> list[float]:
    """Share of runs that reached tau within n evaluations, for n = 1..budget."""
    arr = np.asarray(ts)
    return [float(np.mean(arr <= n)) for n in range(1, budget + 1)]


def hit_rate(scores: Sequence[float], tau: float) -> float:
    """Share of ``scores`` at or above ``tau`` (0 for an empty list)."""
    arr = np.asarray(scores, dtype=float)
    return float(np.mean(arr >= tau - _EPS)) if arr.size else 0.0


def round_to_grid(x: float, grid: float) -> float:
    """``x`` rounded to the nearest multiple of ``grid``, halves rounded up."""
    return math.floor(x / grid + 0.5 + _EPS) * grid


# --------------------------------------------------------------------------- tau rule


@dataclass(frozen=True)
class TauRule:
    """Parameters of the tau rule (hashed into PREREG with this file)."""

    budget: int = 40
    grid: float = GRID_V1
    n_boot: int = 20_000
    boot_seed: int = 20261016
    margin_steps: int = 1


@dataclass(frozen=True)
class TauResult:
    tau: float
    tau_raw: float
    floor: float
    best_entrant: float
    pool_size: int
    pool_sources: dict[str, int]
    boot_quantiles: dict[str, float]
    pilot_best_at_b: list[float]
    q_hat: float
    q_hits: int
    rule: TauRule = field(default_factory=TauRule)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["rule"] = asdict(self.rule)
        return d


def derive_tau(
    random_pilots: Sequence[RunRecord],
    calibration_scores: Sequence[float],
    entrant_scores: Sequence[float],
    rule: TauRule | None = None,
) -> TauResult:
    """Apply the tau rule (module docstring) to the random pilots and the calibration sample."""
    rule = rule or TauRule()
    if any(r.arm != "random" for r in random_pilots):
        raise ValueError("the tau rule uses random-arm runs only")
    pilot_pool = [s for r in random_pilots for s in r.random_design_scores()]
    pool = np.asarray([*calibration_scores, *pilot_pool], dtype=float)
    if pool.size == 0 or not entrant_scores:
        raise ValueError("the tau rule needs a non-empty pool and the entrants' scores")
    n_draw = rule.budget - len(entrant_scores)
    rng = np.random.default_rng(rule.boot_seed)
    draws = rng.choice(pool, size=(rule.n_boot, n_draw), replace=True)
    best = np.maximum(draws.max(axis=1), max(entrant_scores))
    tau_raw = float(np.median(best))
    best_entrant = float(max(entrant_scores))
    floor = best_entrant + rule.margin_steps * rule.grid
    tau = min(1.0, max(round_to_grid(tau_raw, rule.grid), round_to_grid(floor, rule.grid)))
    hits = int(np.sum(pool >= tau - _EPS))
    return TauResult(
        tau=tau,
        tau_raw=tau_raw,
        floor=floor,
        best_entrant=best_entrant,
        pool_size=int(pool.size),
        pool_sources={"calibration_sobol": len(calibration_scores), "random_pilot_designs": len(pilot_pool)},
        boot_quantiles={f"q{int(100 * q)}": float(np.quantile(best, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        pilot_best_at_b=[r.best for r in random_pilots],
        q_hat=hits / pool.size,
        q_hits=hits,
        rule=rule,
    )


# --------------------------------------------------------------------------- tests


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    method: str
    n_a: int
    n_c: int

    __test__ = False  # not a pytest class


def exact_sum_null(values: Sequence[int], k: int) -> np.ndarray:
    """``counts[s]`` = number of size-``k`` subsets of ``values`` (non-negative ints) summing to ``s``."""
    vals = [int(v) for v in values]
    if any(v < 0 for v in vals):
        raise ValueError("values must be non-negative integers")
    total = sum(vals)
    # float counts: exact up to 2**53, far above C(2R, R) for any R used here
    table = np.zeros((k + 1, total + 1), dtype=float)
    table[0, 0] = 1.0
    for v in vals:
        for j in range(k, 0, -1):
            if v == 0:
                table[j] += table[j - 1]
            else:
                table[j, v:] += table[j - 1, :-v]
    return table[k]


def permutation_test(t_a: Sequence[int], t_c: Sequence[int]) -> TestResult:
    """One-sided exact permutation test, H1: arm C reaches tau in fewer evaluations than arm A.

    The statistic is ``mean(T_A) - mean(T_C)``; the p-value is the share of all relabellings
    of the pooled runs into groups of the observed sizes whose arm-C sum is at most the
    observed arm-C sum (ties count against H1).
    """
    a = [int(x) for x in t_a]
    c = [int(x) for x in t_c]
    if not a or not c:
        raise ValueError("both arms need at least one run")
    pooled = a + c
    counts = exact_sum_null(pooled, len(c))
    observed = sum(c)
    p = float(counts[: observed + 1].sum() / counts.sum())
    stat = float(np.mean(a) - np.mean(c))
    return TestResult(stat, p, f"exact permutation over C({len(pooled)},{len(c)}) splits", len(a), len(c))


def permutation_test_means(
    a: Sequence[float],
    c: Sequence[float],
    *,
    exact_limit: int = 1_000_000,
    n_perm: int = 200_000,
    seed: int = 20261016,
) -> TestResult:
    """One-sided permutation test of ``mean(c) - mean(a)`` (H1: arm C is higher), real values.

    For the secondary endpoints (held-out success, best-so-far area). Exact over all
    relabellings when there are at most ``exact_limit`` of them, else a seeded Monte Carlo
    with the ``(hits + 1) / (n + 1)`` estimator.
    """
    av = np.asarray(a, dtype=float)
    cv = np.asarray(c, dtype=float)
    pooled = np.concatenate([av, cv])
    k = cv.size
    observed = float(cv.mean() - av.mean())
    total = pooled.sum()

    def stat_from_c_sum(c_sum: np.ndarray) -> np.ndarray:
        return c_sum / k - (total - c_sum) / av.size

    n_splits = math.comb(pooled.size, k)
    if n_splits <= exact_limit:
        sums = np.fromiter(
            (pooled[list(idx)].sum() for idx in itertools.combinations(range(pooled.size), k)),
            dtype=float,
            count=n_splits,
        )
        p = float(np.mean(stat_from_c_sum(sums) >= observed - _EPS))
        method = f"exact permutation over C({pooled.size},{k}) splits"
    else:
        rng = np.random.default_rng(seed)
        idx = np.argsort(rng.random((n_perm, pooled.size)), axis=1)[:, :k]
        hits = int(np.sum(stat_from_c_sum(pooled[idx].sum(axis=1)) >= observed - _EPS))
        p = (hits + 1) / (n_perm + 1)
        method = f"Monte Carlo permutation, {n_perm} relabellings"
    return TestResult(observed, p, method, int(av.size), int(k))


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone, capped at 1) for a family of tests."""
    order = sorted(p_values, key=lambda name: p_values[name])
    m = len(order)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[name]))
        adjusted[name] = running
    return adjusted


def best_so_far_area(scores: Sequence[float], *, round0_size: int = 8) -> float:
    """S2: mean best-so-far score over the arm's designs (index > round0_size), round 0 counted in the max."""
    curve = best_so_far(scores)
    return float(curve[round0_size:].mean())


def _logrank_stat(t: np.ndarray, is_c: np.ndarray, censor_at: int) -> float:
    """Sum over event times of (observed - expected) arm-C events; positive favours C."""
    stat = 0.0
    for time in np.unique(t[t < censor_at]):
        at_risk = t >= time
        n = at_risk.sum()
        n_c = (at_risk & is_c).sum()
        d = (t == time).sum()
        d_c = ((t == time) & is_c).sum()
        stat += d_c - d * n_c / n
    return float(stat)


def logrank_permutation_test(
    t_a: Sequence[int], t_c: Sequence[int], censor_at: int, *, n_perm: int = 20_000, seed: int = 0
) -> TestResult:
    """Sensitivity: one-sided permutation log-rank test (Monte Carlo relabellings)."""
    t = np.asarray([*t_a, *t_c])
    is_c = np.zeros(t.size, dtype=bool)
    is_c[len(t_a) :] = True
    observed = _logrank_stat(t, is_c, censor_at)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        hits += _logrank_stat(t, rng.permutation(is_c), censor_at) >= observed - _EPS
    p = (hits + 1) / (n_perm + 1)
    return TestResult(observed, float(p), f"permutation log-rank, {n_perm} relabellings", len(t_a), len(t_c))


def restricted_mean_ci(
    t_a: Sequence[int], t_c: Sequence[int], *, n_boot: int = 10_000, seed: int = 0, level: float = 0.95
) -> dict[str, float]:
    """Difference in restricted means (A minus C, designs saved) with a percentile bootstrap CI."""
    a = np.asarray(t_a, dtype=float)
    c = np.asarray(t_c, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = rng.choice(a, (n_boot, a.size)).mean(axis=1) - rng.choice(c, (n_boot, c.size)).mean(axis=1)
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"diff": float(a.mean() - c.mean()), "ci_low": float(lo), "ci_high": float(hi), "level": level}


# --------------------------------------------------------------------------- power


def _simulate_t(q: np.ndarray, n_arm: int, round0_size: int, rng: np.random.Generator) -> np.ndarray:
    """``T`` for runs whose per-design hit probabilities are ``q`` (geometric, censored)."""
    q = np.clip(q, 1e-12, 1.0)
    g = rng.geometric(q)
    return np.where(g <= n_arm, round0_size + g, round0_size + n_arm + 1)


@dataclass(frozen=True)
class PowerCell:
    runs_per_arm: int
    speedup: float
    sigma: float | None
    q_uncertain: bool
    power: float
    mean_t_a: float
    mean_t_c: float
    reach_a: float
    reach_c: float
    nsim: int


def power(
    q: float,
    speedup: float,
    runs_per_arm: int,
    *,
    budget: int = 40,
    round0_size: int = 8,
    alpha: float = 0.05,
    nsim: int = 4000,
    sigma: float | None = None,
    q_posterior: tuple[float, float] | None = None,
    seed: int = 0,
) -> PowerCell:
    """Monte Carlo power of :func:`permutation_test` under the geometric model.

    ``sigma`` (log-SD of a per-run log-normal speed-up) adds run-to-run variability to arm C;
    ``q_posterior`` = (alpha, beta) draws ``q`` per simulated experiment instead of fixing it.
    """
    rng = np.random.default_rng(seed)
    n_arm = budget - round0_size
    rejections = 0
    sum_a = sum_c = reach_a = reach_c = 0.0
    for _ in range(nsim):
        q_i = float(rng.beta(*q_posterior)) if q_posterior else q
        q_c = 1.0 - (1.0 - q_i) ** speedup
        qa = np.full(runs_per_arm, q_i)
        if sigma is None:
            qc = np.full(runs_per_arm, q_c)
        else:
            s_r = speedup * np.exp(sigma * rng.standard_normal(runs_per_arm))
            qc = 1.0 - (1.0 - q_i) ** s_r
        t_a = _simulate_t(qa, n_arm, round0_size, rng)
        t_c = _simulate_t(qc, n_arm, round0_size, rng)
        rejections += permutation_test(t_a, t_c).p_value <= alpha
        sum_a += t_a.mean()
        sum_c += t_c.mean()
        reach_a += np.mean(t_a <= budget)
        reach_c += np.mean(t_c <= budget)
    return PowerCell(
        runs_per_arm=runs_per_arm,
        speedup=speedup,
        sigma=sigma,
        q_uncertain=q_posterior is not None,
        power=rejections / nsim,
        mean_t_a=sum_a / nsim,
        mean_t_c=sum_c / nsim,
        reach_a=reach_a / nsim,
        reach_c=reach_c / nsim,
        nsim=nsim,
    )


def power_table(
    q: float,
    *,
    runs: Iterable[int] = (8, 10, 12, 16, 24, 32),
    speedups: Iterable[float] = (1.0, 1.5, 2.0, 3.0),
    sigmas: Iterable[float | None] = (None, 0.5),
    q_posterior: tuple[float, float] | None = None,
    budget: int = 40,
    round0_size: int = 8,
    nsim: int = 4000,
    seed: int = 20261016,
) -> list[PowerCell]:
    """Power over the grid; speed-up 1.0 is the null (its 'power' is the type-I error)."""
    cells = []
    for i, (r, s, k) in enumerate((r, s, k) for r in runs for s in speedups for k in sigmas):
        cells.append(
            power(q, s, r, budget=budget, round0_size=round0_size, nsim=nsim, sigma=k, seed=seed + i)
        )
        if q_posterior is not None and k is None:
            cells.append(
                power(
                    q,
                    s,
                    r,
                    budget=budget,
                    round0_size=round0_size,
                    nsim=nsim,
                    q_posterior=q_posterior,
                    seed=seed + 1000 + i,
                )
            )
    return cells


def choose_r(
    cells: Sequence[PowerCell], *, speedup: float = 2.0, candidates: Sequence[int] = (8, 10, 12)
) -> int | None:
    """The R rule: smallest candidate with power >= 0.8 at ``speedup`` (fixed q, no overdispersion)."""
    for r in candidates:
        for c in cells:
            plain = c.sigma is None and not c.q_uncertain
            if c.runs_per_arm == r and c.speedup == speedup and plain and c.power >= 0.8:
                return r
    return None


def minimum_detectable_speedup(
    q: float, runs_per_arm: int, *, budget: int = 40, round0_size: int = 8, nsim: int = 2000, seed: int = 7
) -> float | None:
    """Smallest speed-up on a 0.25 grid (1.25-6) with power >= 0.8 at ``runs_per_arm``."""
    for s in np.arange(1.25, 6.01, 0.25):
        cell = power(q, float(s), runs_per_arm, budget=budget, round0_size=round0_size, nsim=nsim, seed=seed)
        if cell.power >= 0.8:
            return float(s)
    return None


def geometric_mle(ts: Sequence[int], *, round0_size: int = 8, budget: int = 40) -> dict[str, float]:
    """Per-design hit probability of a censored geometric fitted to run-level ``T`` values.

    Each run contributes its arm-proposed designs up to and including the first hit
    (``T - round0_size``) or all ``budget - round0_size`` of them when censored; the MLE
    is hits / exposure. This targets the endpoint directly, so it absorbs any regularity
    of the Sobol stream that a per-design hit rate would miss.
    """
    n_arm = budget - round0_size
    events = sum(1 for t in ts if t <= budget)
    exposure = sum(min(t - round0_size, n_arm) for t in ts)
    q = events / exposure if exposure else 0.0
    return {
        "q": q,
        "events": events,
        "runs": len(ts),
        "exposure": exposure,
        "mean_t": float(np.mean(ts)) if len(ts) else float("nan"),
        "posterior": [events + 0.5, exposure - events + 0.5],
    }


def power_report(
    random_runs: Sequence[RunRecord],
    random_scores: Sequence[float],
    tau: float,
    *,
    budget: int = 40,
    round0_size: int = 8,
    nsim: int = 4000,
) -> dict[str, Any]:
    """Power table under the run-level fit (primary) and the per-design hit rate (sensitivity)."""
    ts = [evals_to_tau(r.scores, tau, round0_size=round0_size) for r in random_runs]
    fit = geometric_mle(ts, round0_size=round0_size, budget=budget)
    posterior = (float(fit["posterior"][0]), float(fit["posterior"][1]))
    cells = power_table(fit["q"], q_posterior=posterior, budget=budget, round0_size=round0_size, nsim=nsim)
    q_design = hit_rate(random_scores, tau)
    cells_design = power_table(
        q_design,
        speedups=(1.0, 2.0, 3.0),
        sigmas=(None,),
        budget=budget,
        round0_size=round0_size,
        nsim=max(1000, nsim // 2),
        seed=20261116,
    )
    chosen = choose_r(cells)
    return {
        "model": "censored geometric after round 0; arm C hit rate 1-(1-q)^s",
        "q_run": fit,
        "q_design": {"q": q_design, "n": len(random_scores)},
        "random_t": ts,
        "cells": [asdict(c) for c in cells],
        "cells_design_q": [asdict(c) for c in cells_design],
        "r_rule": {"speedup_assumed": 2.0, "candidates": [8, 10, 12], "chosen": chosen},
        "mde_speedup_at_r12": None
        if chosen is not None
        else minimum_detectable_speedup(fit["q"], 12, budget=budget, round0_size=round0_size),
        "mde_speedup_at_r16": minimum_detectable_speedup(
            fit["q"], 16, budget=budget, round0_size=round0_size
        ),
        "nsim": nsim,
    }


# --------------------------------------------------------------------------- report


def run_rows(runs: Sequence[RunRecord], tau: float) -> list[dict[str, Any]]:
    """Per-run table: best score, T (arm-only and all-designs), model calls, fallbacks, job."""
    rows = []
    for r in runs:
        rows.append(
            {
                "arm": r.arm,
                "run_id": r.run_id,
                "run_seed": r.run_seed,
                "best": r.best,
                "best_round0": max(r.scores[: r.round0_size]),
                "t_arm": evals_to_tau(r.scores, tau, round0_size=r.round0_size, count="arm"),
                "t_all": evals_to_tau(r.scores, tau, round0_size=r.round0_size, count="all"),
                "llm_calls": r.llm_calls,
                "llm_cost_usd": r.llm_cost_usd,
                "fallback_random": r.fallback_random,
                "source": r.source,
                "job": r.job,
                "best_so_far": [float(x) for x in best_so_far(r.scores)],
            }
        )
    return rows


def compare_arms(
    rows_a: Sequence[Mapping[str, Any]], rows_c: Sequence[Mapping[str, Any]], budget: int
) -> dict[str, Any]:
    """Primary test, effect CI and sensitivity analyses for two arms' run rows."""
    t_a = [int(r["t_arm"]) for r in rows_a]
    t_c = [int(r["t_arm"]) for r in rows_c]
    primary = permutation_test(t_a, t_c)
    t_a_all = [int(r["t_all"]) for r in rows_a]
    t_c_all = [int(r["t_all"]) for r in rows_c]
    return {
        "primary": asdict(primary),
        "effect": restricted_mean_ci(t_a, t_c),
        "reach_curve": {"A": reach_curve(t_a, budget), "C": reach_curve(t_c, budget)},
        "sensitivity": {
            "logrank": asdict(logrank_permutation_test(t_a, t_c, budget + 1)),
            "count_all": asdict(permutation_test(t_a_all, t_c_all)),
        },
    }


def _fmt_t(t: int, budget: int) -> str:
    return f">{budget}" if t > budget else str(t)


def main(argv: Sequence[str] | None = None, *, prog: str = "python -m rtt.analysis.s1") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    ap.add_argument("--random", nargs="+", type=Path, required=True, help="random-arm search dirs (pilot)")
    ap.add_argument("--nemotron", nargs="*", type=Path, default=[], help="arm-C search dirs")
    ap.add_argument("--calibration", type=Path, required=True, help="calibration.json (Sobol-64)")
    ap.add_argument("--tau", type=float, default=None, help="fixed tau (confirmatory: from PREREG)")
    ap.add_argument(
        "--extra-random",
        nargs="*",
        type=Path,
        default=[],
        help="random runs used only for the power sensitivity (never for tau)",
    )
    ap.add_argument("--power", action="store_true", help="Monte Carlo power table and the R rule")
    ap.add_argument("--nsim", type=int, default=4000)
    ap.add_argument("--compare", action="store_true", help="run the S1 test (confirmatory runs only)")
    ap.add_argument("--out", type=Path, default=None, help="write s1.json here")
    args = ap.parse_args(argv)

    random_runs = [r for d in args.random for r in load_runs(d)]
    nemo_runs = [r for d in args.nemotron for r in load_runs(d)]
    cal_scores, entrant_scores = load_calibration_scores(args.calibration)
    tau_result = derive_tau(random_runs, cal_scores, entrant_scores)
    tau = args.tau if args.tau is not None else tau_result.tau
    budget = random_runs[0].budget
    rows_a = run_rows(random_runs, tau)
    rows_c = run_rows(nemo_runs, tau)
    report: dict[str, Any] = {
        "format": "rtt-s1-analysis",
        "version": 1,
        "tau": tau,
        "tau_rule": tau_result.to_json(),
        "tau_source": "argument" if args.tau is not None else "rule",
        "budget": budget,
        "runs": {"A": rows_a, "C": rows_c},
    }
    print(
        f"tau = {tau:.4f} (rule: median bootstrap best-at-B {tau_result.tau_raw:.4f}, floor "
        f"{tau_result.floor:.4f}; pool {tau_result.pool_size} random designs, q_hat {tau_result.q_hat:.4f})"
    )
    print(
        f"{'arm':<9}{'run':<14}{'seed':>6}{'best':>7}{'r0 best':>8}{'T':>5}{'T_all':>6}{'calls':>6}{'fb':>4}"
    )
    for row in rows_a + rows_c:
        calls = "-" if row["llm_calls"] is None else str(row["llm_calls"])
        print(
            f"{row['arm']:<9}{row['run_id']:<14}{row['run_seed']:>6}{row['best']:>7.3f}{row['best_round0']:>8.3f}"
            f"{_fmt_t(row['t_arm'], budget):>5}{_fmt_t(row['t_all'], budget):>6}{calls:>6}{row['fallback_random']:>4}"
        )
    if args.compare:
        if not rows_c:
            ap.error("--compare needs --nemotron runs")
        report["comparison"] = compare_arms(rows_a, rows_c, budget)
        pr = report["comparison"]["primary"]
        print(f"S1 test: mean T_A - mean T_C = {pr['statistic']:.2f}, one-sided p = {pr['p_value']:.4f}")
    if args.power:
        extra = [r for d in args.extra_random for r in load_runs(d)]
        report["power"] = power_report(
            random_runs + extra,
            [*cal_scores, *(x for r in random_runs + extra for x in r.random_design_scores())],
            tau,
            budget=budget,
            round0_size=random_runs[0].round0_size,
            nsim=args.nsim,
        )
        pw = report["power"]
        fit = pw["q_run"]
        print(
            f"power: q_run = {fit['q']:.4f} (censored-geometric fit to {fit['runs']} random runs: {fit['events']} reached"
            f" tau, mean T {fit['mean_t']:.1f}); q_design = {pw['q_design']['q']:.4f} from {pw['q_design']['n']} designs"
        )
        print(
            f"{'q':>6}{'R':>4}{'s':>6}{'sigma':>7}{'q unc':>7}{'power':>7}{'E T_A':>7}{'E T_C':>7}"
            f"{'reach A':>8}{'reach C':>8}"
        )
        for label, key in (("run", "cells"), ("design", "cells_design_q")):
            for c in pw[key]:
                sig = "-" if c["sigma"] is None else f"{c['sigma']:g}"
                print(
                    f"{label:>6}{c['runs_per_arm']:>4}{c['speedup']:>6.1f}{sig:>7}"
                    f"{'yes' if c['q_uncertain'] else 'no':>7}{c['power']:>7.3f}{c['mean_t_a']:>7.1f}"
                    f"{c['mean_t_c']:>7.1f}{c['reach_a']:>8.2f}{c['reach_c']:>8.2f}"
                )
        rr = pw["r_rule"]
        print(
            f"R rule (smallest of 8/10/12 with power >= 0.8 at 2x): {rr['chosen']}; MDE at R=12: {pw['mde_speedup_at_r12']}"
        )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "s1.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out / 's1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
