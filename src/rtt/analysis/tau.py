"""The v2 tau rule and the v2 S1 analysis entry point (``freeform-v2`` design space).

Everything here reads finished search outputs and calibration files. Nothing simulates
or calls a model. The rule below is fixed before the v2 pilot (founder decision D6) and
this file's SHA-256 is recorded in ``docs/PREREG.md``.

**Score.** A v2 design's training score is its mean registered-grasp success
(``success_v2``) over the 48 episodes of ``train_v2`` (6 parts x 4 placements x 2 jitter
seeds). There is no best-of-k controller step any more, so scores lie on a 1/48 grid
(:data:`GRID_V2`).

**S1 is unchanged from v1** (:mod:`rtt.analysis.s1`): ``T`` is the index of the first
design proposed by the arm (index 9 or later) scoring at least tau, censored at B + 1 = 41;
the test is the one-sided exact permutation test of the restricted mean of ``T``.

**tau rule v2** (:func:`derive_tau_v2`)

1. Inputs: exactly three random-arm pilot runs (``TauRuleV2.n_random_pilots``), each
   with budget B = 40, and the Sobol-64 calibration sample of gate GV3 (the screening
   sample, ``calibration_v2.json``). The Nemotron pilots never enter the rule; they are
   used only for valid and fallback rates.
2. The pool is every Sobol-64 calibration score plus every non-entrant design score of
   the three random pilots (their round-0 Sobol fill and their arm designs).
3. Bootstrap ``n_boot`` = 20,000 random runs of B: the round-0 entrants' scores plus
   B - 4 = 36 scores drawn with replacement from the pool (seed 20261021). A run's
   best-at-B is its maximum.
4. ``tau_raw`` is the median of the bootstrapped best-at-B. tau is ``tau_raw`` rounded to
   the 1/48 grid, halves up.
5. tau is raised, if needed, to 2/48 above the best round-0 score observed before the
   tag: the maximum over the four round-0 entrants and the round-0 designs (entrants and
   Sobol fill) of the three random pilots. tau is capped at 1; a capped tau is reported as
   ``degenerate`` (gate GV3 should have failed first).

The three pilots' own best-at-B values and the pooled per-design hit rate at tau are
reported alongside; neither changes tau.

Command line (``python -m rtt.analysis.tau``):

* ``derive --random DIR... --calibration FILE [--power] --out DIR``: the pilot step
  (tau, and with ``--power`` the R rule of :mod:`rtt.analysis.power`).
* ``analyze --random DIR... --nemotron DIR... --tau X --out DIR``: the confirmatory S1
  analysis with tau read from PREREG. It refuses runs that are not the confirmatory design
  (:func:`check_confirmatory`): arm labels other than ``random`` and ``nemotron``, a seed
  used twice in an arm, arms with different seeds, or seeds other than the first R of
  ``configs/seeds_main.yaml``.
* ``rule``: print the rule constants and the digest recorded in PREREG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from rtt.analysis.s1 import RunRecord, compare_arms, evals_to_tau, hit_rate, load_runs, run_rows

__all__ = [
    "DESIGN_SPACE",
    "EPISODES_PER_DESIGN_V2",
    "GRID_V2",
    "N_ROUND0_ENTRANTS",
    "TauResultV2",
    "TauRuleV2",
    "check_calibration_task",
    "check_confirmatory",
    "check_on_grid",
    "check_same_task",
    "derive_tau_v2",
    "evaluator_hashes",
    "load_calibration_v2",
    "load_runs_v2",
    "main",
    "round_half_up",
]

DESIGN_SPACE: Final = "freeform-v2"
EPISODES_PER_DESIGN_V2: Final = 48
#: Score grid of ``train_v2``: mean ``success_v2`` over 48 episodes.
GRID_V2: Final = 1.0 / EPISODES_PER_DESIGN_V2
#: Round 0 = 4 entrants (stock, flat, configurator imprint, swept imprint) + 4 Sobol designs.
N_ROUND0_ENTRANTS: Final = 4
CALIBRATION_SCHEMA_V2: Final = "rtt.calibration/2"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_EPS: Final = 1e-9


# --------------------------------------------------------------------------- grid helpers


def round_half_up(x: float, grid: float = GRID_V2) -> float:
    """``x`` rounded to the nearest multiple of ``grid``, halves rounded up."""
    return math.floor(x / grid + 0.5 + _EPS) * grid


def check_on_grid(scores: Sequence[float], grid: float = GRID_V2, *, what: str = "scores") -> None:
    """Raise unless every score is in [0, 1] and a multiple of ``grid`` (within 1e-9)."""
    for s in scores:
        if not (-_EPS <= s <= 1.0 + _EPS):
            raise ValueError(f"{what}: {s!r} is outside [0, 1]")
        steps = s / grid
        if abs(steps - round(steps)) > 1e-6:
            raise ValueError(f"{what}: {s!r} is not on the 1/{round(1 / grid)} grid")


# --------------------------------------------------------------------------- loading


def _manifest_file(directory: Path) -> Path:
    for candidate in (directory / "search" / "manifest.json", directory / "manifest.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{directory}: no manifest.json (or search/manifest.json)")


def evaluator_hashes(directory: str | Path) -> tuple[str | None, str | None]:
    """``(task_hash, suite_hash)`` from a search output's manifest (``None`` where absent)."""
    manifest = json.loads(_manifest_file(Path(directory)).read_text(encoding="utf-8"))
    ev = manifest.get("evaluator") or {}
    return ev.get("task_hash"), ev.get("suite_hash")


def check_same_task(directories: Sequence[str | Path], runs: Sequence[RunRecord]) -> None:
    """Refuse to pool search outputs scored on different tasks or suites, or with another budget.

    Every output must report the same evaluator ``task_hash`` and ``suite_hash`` (a run
    scored before a GV3 iteration is not comparable with one scored after it), and every run
    must have B = 40 with a round 0 of 8.
    """
    rule = TauRuleV2()
    hashes = {str(d): evaluator_hashes(d) for d in directories}
    if len(set(hashes.values())) > 1:
        raise ValueError(f"search outputs scored on different task/suite hashes: {hashes}")
    for r in runs:
        if r.budget != rule.budget or r.round0_size != rule.round0_size or len(r.scores) != rule.budget:
            raise ValueError(
                f"{r.run_id}: budget {r.budget}, round 0 {r.round0_size}; v2 runs use {rule.budget}/{rule.round0_size}"
            )


def check_calibration_task(calibration: str | Path, directories: Sequence[str | Path]) -> None:
    """Refuse a calibration sample scored on another task or suite than the pilots.

    ``calibration_v2.json`` records ``task_digest`` and ``suite_digest`` (the evaluator
    digests, as in a search manifest's ``evaluator``). Where both sides carry a digest they
    must agree, so a screening sample scored before a GV3 iteration is never pooled with
    pilots run on the frozen task.
    """
    data = json.loads(Path(calibration).read_text(encoding="utf-8"))
    cal = (data.get("task_digest"), data.get("suite_digest"))
    for d in directories:
        for name, mine, theirs in zip(("task", "suite"), cal, evaluator_hashes(d), strict=True):
            if mine is not None and theirs is not None and mine != theirs:
                raise ValueError(
                    f"{calibration}: {name} digest {mine!r}, but {d} was scored on {theirs!r}; "
                    "the calibration sample and the pilots must use the same frozen task and suite"
                )


def check_confirmatory(
    random_runs: Sequence[RunRecord], nemotron_runs: Sequence[RunRecord], confirmatory_seeds: Sequence[int]
) -> list[int]:
    """Refuse anything but the pre-registered design for the confirmatory S1 test.

    Arm A runs must be labelled ``random`` and arm C runs ``nemotron`` (a swap would flip
    the one-sided test); no seed may appear twice in an arm (a directory passed twice);
    both arms must use the same seeds (round 0 is paired run by run); and those seeds must
    be exactly the first R confirmatory seeds. Returns the seeds, in order.
    """
    for arm, runs in (("random", random_runs), ("nemotron", nemotron_runs)):
        if not runs:
            raise ValueError(f"no {arm} runs")
        if wrong := sorted({r.arm for r in runs} - {arm}):
            raise ValueError(f"the {arm} arm holds runs labelled {wrong}")
        seeds = [r.run_seed for r in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"the {arm} arm uses a seed more than once: {sorted(seeds)}")
    seeds_a = sorted(r.run_seed for r in random_runs)
    seeds_c = sorted(r.run_seed for r in nemotron_runs)
    if seeds_a != seeds_c:
        raise ValueError(f"the arms use different seeds: random {seeds_a}, nemotron {seeds_c}")
    expected = [int(s) for s in confirmatory_seeds[: len(seeds_a)]]
    if len(expected) < len(seeds_a) or seeds_a != sorted(expected):
        raise ValueError(
            f"run seeds {seeds_a} are not the first {len(seeds_a)} confirmatory seeds {expected}"
        )
    return expected


def _declared_design_space(manifest: dict[str, Any]) -> str | None:
    for holder in (manifest, manifest.get("config") or {}, manifest.get("evaluator") or {}):
        value = holder.get("design_space") if isinstance(holder, dict) else None
        if value is not None:
            return str(value)
    return None


def load_runs_v2(directory: str | Path, *, score_key: str = "success") -> list[RunRecord]:
    """The runs of one v2 search output; refuses outputs that do not declare ``freeform-v2``.

    The manifest must carry ``design_space`` (top level, ``config`` or ``evaluator``) and the
    evaluator's ``episodes_per_design`` (48), and every score must lie on the 1/48 grid.
    ``score_key`` picks the history score: ``"success"`` is the primary ``success_v2`` mean,
    ``"lift_success"`` the lift-only mean reported for every endpoint. Every design must
    carry both, and its registered score may never exceed its lift-only score (a registered
    grasp is a lift-and-hold success first), which catches a lift-only score filed as the
    primary one.
    """
    root = Path(directory)
    manifest_file = _manifest_file(root)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    space = _declared_design_space(manifest)
    if space != DESIGN_SPACE:
        raise ValueError(f"{root}: design_space {space!r}, expected {DESIGN_SPACE!r}")
    per_design = (manifest.get("evaluator") or {}).get("episodes_per_design")
    if per_design is None or int(per_design) != EPISODES_PER_DESIGN_V2:
        raise ValueError(f"{root}: {per_design} episodes per design, expected {EPISODES_PER_DESIGN_V2}")
    primary = load_runs(root)
    out = []
    for r in primary:
        lift = _history_scores(manifest_file.parent, r, "lift_success")
        if len(lift) != len(r.scores):
            raise ValueError(f"{r.run_id}: {len(lift)} lift-only scores, expected {len(r.scores)}")
        bad = [i + 1 for i, (s, ls) in enumerate(zip(r.scores, lift, strict=True)) if s > ls + _EPS]
        if bad:
            raise ValueError(
                f"{root} {r.run_id}: designs {bad[:5]} score higher registered than lift-only; "
                "score.success must be the mean success_v2"
            )
        out.append(
            r
            if score_key == "success"
            else _with_scores(r, _history_scores(manifest_file.parent, r, score_key))
        )
    for r in out:
        check_on_grid(r.scores, what=f"{root} {r.run_id} {score_key}")
    return out


def _history_scores(search_dir: Path, run: RunRecord, key: str) -> tuple[float, ...]:
    history = search_dir / "runs" / run.run_id / "history.jsonl"
    items = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line]
    items.sort(key=lambda item: int(item["index"]))
    try:
        return tuple(float(i["score"][key]) for i in items)
    except KeyError as exc:
        raise KeyError(f"{history}: no score.{key} in the history") from exc


def _with_scores(run: RunRecord, scores: tuple[float, ...]) -> RunRecord:
    if len(scores) != len(run.scores):
        raise ValueError(f"{run.run_id}: {len(scores)} scores, expected {len(run.scores)}")
    return RunRecord(**{**asdict(run), "scores": scores})


#: The GV3 report writes scores rounded to 6 decimals (``rtt.analysis.calibrate_v2``), so a
#: score there lies within 5e-7 of its 1/48 grid point; the loader snaps it back to the point.
CALIBRATION_ROUNDING: Final = 5e-7


def snap_to_grid(scores: Sequence[float], grid: float = GRID_V2, *, what: str = "scores") -> list[float]:
    """Each score as the exact multiple of ``grid`` it was rounded from (within
    :data:`CALIBRATION_ROUNDING`); anything farther from the grid is refused."""
    out = []
    for s in scores:
        k = round(s / grid)
        if abs(s - k * grid) > CALIBRATION_ROUNDING + 1e-12:
            raise ValueError(f"{what}: {s!r} is not on the 1/{round(1 / grid)} grid")
        out.append(k * grid)
    check_on_grid(out, grid, what=what)
    return out


def load_calibration_v2(path: str | Path) -> tuple[list[float], list[float]]:
    """``(sobol_scores, round0_entrant_scores)`` from a GV3 ``calibration_v2.json``.

    Reference entrants (scored outside the budget) are not round 0 and are not returned.
    Scores are snapped to the 1/48 grid (the report rounds them to 6 decimals).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != CALIBRATION_SCHEMA_V2:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, expected {CALIBRATION_SCHEMA_V2!r}")
    sobol = snap_to_grid([float(d["score"]) for d in data["sobol"]], what=f"{path} sobol")
    entrants = snap_to_grid([float(d["score"]) for d in data["entrants"]], what=f"{path} entrants")
    return sobol, entrants


# --------------------------------------------------------------------------- the rule


@dataclass(frozen=True)
class TauRuleV2:
    """Constants of the v2 tau rule (module docstring); hashed into PREREG."""

    budget: int = 40
    round0_size: int = 8
    n_round0_entrants: int = N_ROUND0_ENTRANTS
    n_random_pilots: int = 3
    grid: float = GRID_V2
    n_boot: int = 20_000
    boot_seed: int = 20261021
    margin_steps: int = 2
    n_calibration: int = 64

    def digest(self) -> str:
        """SHA-256 of the rule constants (canonical JSON), recorded in PREREG."""
        blob = json.dumps({"rule": "tau-v2", "version": 1, **asdict(self)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class TauResultV2:
    tau: float
    tau_steps: int
    tau_raw: float
    floor: float
    best_round0: float
    best_entrant: float
    degenerate: bool
    pool_size: int
    pool_sources: dict[str, int]
    boot_quantiles: dict[str, float]
    pilot_best_at_b: list[float]
    pilot_seeds: list[int]
    q_design: float
    q_design_hits: int
    binding: str
    rule: TauRuleV2 = field(default_factory=TauRuleV2)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["rule"] = {**asdict(self.rule), "digest": self.rule.digest()}
        return d


def _round0_scores(run: RunRecord) -> list[float]:
    return list(run.scores[: run.round0_size])


def derive_tau_v2(
    random_pilots: Sequence[RunRecord],
    calibration_scores: Sequence[float],
    entrant_scores: Sequence[float],
    rule: TauRuleV2 | None = None,
    *,
    forbidden_seeds: Iterable[int] = (),
) -> TauResultV2:
    """Apply the v2 tau rule (module docstring) to the random pilots and the GV3 sample.

    ``forbidden_seeds`` are the confirmatory seeds (``configs/seeds_main.yaml``); a pilot
    that used one is refused.
    """
    rule = rule or TauRuleV2()
    if len(random_pilots) != rule.n_random_pilots:
        raise ValueError(
            f"the tau rule takes exactly {rule.n_random_pilots} random pilots, got {len(random_pilots)}"
        )
    if any(r.arm != "random" for r in random_pilots):
        raise ValueError(
            "the tau rule uses random-arm runs only (Nemotron pilots give valid and fallback rates)"
        )
    seeds = [r.run_seed for r in random_pilots]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"the random pilots must use distinct seeds, got {seeds}")
    if clash := sorted(set(seeds) & {int(s) for s in forbidden_seeds}):
        raise ValueError(f"pilot seeds {clash} are confirmatory seeds; pilots must never use them")
    for r in random_pilots:
        if r.budget != rule.budget or len(r.scores) != rule.budget or r.round0_size != rule.round0_size:
            raise ValueError(
                f"{r.run_id}: budget {r.budget}/round 0 {r.round0_size}, rule wants {rule.budget}/{rule.round0_size}"
            )
        check_on_grid(r.scores, rule.grid, what=r.run_id)
    if len(entrant_scores) != rule.n_round0_entrants:
        raise ValueError(
            f"expected {rule.n_round0_entrants} round-0 entrant scores, got {len(entrant_scores)}"
        )
    if len(calibration_scores) != rule.n_calibration:
        raise ValueError(
            f"expected the Sobol-{rule.n_calibration} calibration sample, got {len(calibration_scores)}"
        )
    check_on_grid(calibration_scores, rule.grid, what="calibration")
    check_on_grid(entrant_scores, rule.grid, what="entrants")

    pilot_pool = [s for r in random_pilots for s in r.random_design_scores()]
    pool = np.asarray([*calibration_scores, *pilot_pool], dtype=float)
    n_draw = rule.budget - rule.n_round0_entrants
    rng = np.random.default_rng(rule.boot_seed)
    draws = rng.choice(pool, size=(rule.n_boot, n_draw), replace=True)
    best_entrant = float(max(entrant_scores))
    best = np.maximum(draws.max(axis=1), best_entrant)
    tau_raw = float(np.median(best))

    round0_seen = [*entrant_scores, *(s for r in random_pilots for s in _round0_scores(r))]
    best_round0 = float(max(round0_seen))
    floor = best_round0 + rule.margin_steps * rule.grid
    from_median = round_half_up(tau_raw, rule.grid)
    from_floor = round_half_up(floor, rule.grid)
    uncapped = max(from_median, from_floor)
    tau = min(1.0, uncapped)
    binding = "median" if from_median >= from_floor else "floor"
    if uncapped > 1.0 + _EPS:
        binding = "cap"
    hits = int(np.sum(pool >= tau - _EPS))
    return TauResultV2(
        tau=tau,
        tau_steps=round(tau / rule.grid),
        tau_raw=tau_raw,
        floor=floor,
        best_round0=best_round0,
        best_entrant=best_entrant,
        degenerate=uncapped > 1.0 + _EPS or tau >= 1.0 - _EPS,
        pool_size=int(pool.size),
        pool_sources={"calibration_sobol": len(calibration_scores), "random_pilot_designs": len(pilot_pool)},
        boot_quantiles={f"q{int(100 * q)}": float(np.quantile(best, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        pilot_best_at_b=[r.best for r in random_pilots],
        pilot_seeds=[r.run_seed for r in random_pilots],
        q_design=hit_rate(pool.tolist(), tau),
        q_design_hits=hits,
        binding=binding,
        rule=rule,
    )


# --------------------------------------------------------------------------- command line


def _print_tau(res: TauResultV2) -> None:
    print(
        f"tau = {res.tau:.4f} ({res.tau_steps}/48; binding: {res.binding}); median bootstrap best-at-B "
        f"{res.tau_raw:.4f}, floor {res.floor:.4f} (best round 0 {res.best_round0:.4f} + 2/48)"
    )
    print(
        f"pool {res.pool_size} random designs ({res.pool_sources}); per-design hit rate at tau "
        f"{res.q_design:.4f}; pilot best-at-B {res.pilot_best_at_b} (seeds {res.pilot_seeds})"
    )


def _confirmatory_seeds(path: Path) -> list[int]:
    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"{path}: the confirmatory seed list is needed to vet the pilots")
    return [int(s) for s in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("seeds", [])]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rtt.analysis.tau", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive", help="pilot step: tau by rule (and the R rule with --power)")
    d.add_argument("--random", nargs="+", type=Path, required=True, help="the random pilot search dirs")
    d.add_argument("--calibration", type=Path, required=True, help="GV3 calibration_v2.json")
    d.add_argument("--power", action="store_true", help="also apply the R rule (rtt.analysis.power)")
    d.add_argument("--nsim", type=int, default=4000)
    d.add_argument(
        "--seeds-main",
        type=Path,
        default=REPO_ROOT / "configs" / "seeds_main.yaml",
        help="confirmatory seeds; a pilot that used one is refused",
    )
    d.add_argument("--out", type=Path, default=None)
    a = sub.add_parser("analyze", help="confirmatory S1 with tau from PREREG")
    a.add_argument("--random", nargs="+", type=Path, required=True)
    a.add_argument("--nemotron", nargs="+", type=Path, required=True)
    a.add_argument("--tau", type=float, required=True, help="tau as frozen in docs/PREREG.md")
    a.add_argument(
        "--seeds-main",
        type=Path,
        default=REPO_ROOT / "configs" / "seeds_main.yaml",
        help="confirmatory seeds; the runs must use exactly the first R of them, in both arms",
    )
    a.add_argument("--out", type=Path, default=None)
    sub.add_parser("rule", help="print the rule constants and digest")
    args = ap.parse_args(argv)

    if args.cmd == "rule":
        rule = TauRuleV2()
        print(json.dumps({**asdict(rule), "digest": rule.digest()}, indent=1))
        return 0

    report: dict[str, Any] = {"format": "rtt-s1-analysis", "version": 2, "design_space": DESIGN_SPACE}
    random_runs = [r for p in args.random for r in load_runs_v2(p)]
    if args.cmd == "derive":
        check_same_task(args.random, random_runs)
        check_calibration_task(args.calibration, args.random)
        sobol, entrants = load_calibration_v2(args.calibration)
        res = derive_tau_v2(
            random_runs, sobol, entrants, forbidden_seeds=_confirmatory_seeds(args.seeds_main)
        )
        _print_tau(res)
        report["tau"] = res.tau
        report["tau_rule"] = res.to_json()
        if args.power:
            from rtt.analysis.power import r_rule_report

            pool = [*sobol, *(s for r in random_runs for s in r.random_design_scores())]
            report["power"] = r_rule_report(random_runs, pool, res.tau, nsim=args.nsim)
            rr = report["power"]["r_rule"]
            print(f"R rule: chosen {rr['chosen']}, R used {rr['r_used']}, q {rr['q']:.4f} ({rr['q_source']})")
    else:
        check_on_grid([args.tau], what="--tau")
        nemo_runs = [r for p in args.nemotron for r in load_runs_v2(p)]
        check_same_task([*args.random, *args.nemotron], [*random_runs, *nemo_runs])
        report["seeds"] = check_confirmatory(random_runs, nemo_runs, _confirmatory_seeds(args.seeds_main))
        budget = random_runs[0].budget
        rows_a = run_rows(random_runs, args.tau)
        rows_c = run_rows(nemo_runs, args.tau)
        report.update(tau=args.tau, tau_source="PREREG", budget=budget, runs={"A": rows_a, "C": rows_c})
        report["comparison"] = compare_arms(rows_a, rows_c, budget)
        pr = report["comparison"]["primary"]
        t_a = [evals_to_tau(r.scores, args.tau, round0_size=r.round0_size) for r in random_runs]
        t_c = [evals_to_tau(r.scores, args.tau, round0_size=r.round0_size) for r in nemo_runs]
        print(f"T_A = {t_a}\nT_C = {t_c}")
        print(f"S1: mean T_A - mean T_C = {pr['statistic']:.2f}, one-sided exact p = {pr['p_value']:.4f}")
        # exploratory: the same endpoint on lift-only success, at the same tau
        try:
            lift_a = [r for p in args.random for r in load_runs_v2(p, score_key="lift_success")]
            lift_c = [r for p in args.nemotron for r in load_runs_v2(p, score_key="lift_success")]
        except KeyError as exc:
            report["lift_only"] = {"status": "not available", "reason": str(exc)}
        else:
            report["lift_only"] = {
                "status": "exploratory",
                "comparison": compare_arms(run_rows(lift_a, args.tau), run_rows(lift_c, args.tau), budget),
            }
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "s1_v2.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out / 's1_v2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
