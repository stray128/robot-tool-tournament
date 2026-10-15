"""Gate G3: is the training task hard enough, and geometry-sensitive enough, to rank designs?

The plan's calibration gate (section 3.5, day 4) runs the manual entrants plus a
64-point scrambled-Sobol sample of valid random designs on the training suite and
asks three questions of the design scores (each design's best controller setting's
mean success, :func:`rtt.sim.evaluate.score_designs`):

1. **Entrants in range.** Every manual entrant scores 30-70 %, and at least two of
   them differ by 15 points or more on at least one object family. A benchmark where
   the hand-designed fingers all score 100 % (or 0 %) cannot show improvement.
2. **Spread.** The standard deviation (``ddof=1``) of the 64 random designs' scores is
   at least 0.15, so the score responds to geometry. Below 0.10 the plan's kill
   trigger 4 applies (add geometry-sensitive conditions, else make sample efficiency
   the primary endpoint).
3. **Headroom.** At least 5 random designs beat the best manual entrant outright
   (strictly higher score), so better fingers exist for a search to find.

Only task-level knobs may be tuned to pass (object sizes and density, friction, grip
force within the gripper's spec, placements); never anything per design. The random
designs come from :class:`~rtt.search.random_search.DesignSampler` with a public
calibration seed, validated against the suite's own gripper task, so the sample is
reproducible and disjoint from every search run's seeds.

:func:`run_calibration` evaluates everything in one batch on an executor and
returns a :class:`CalibrationReport` (``table()`` for people, ``to_dict()`` for
``docs/TASK.md`` and ``docs/BENCH.md``); ``rtt calibrate`` wraps it.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from rtt.design.schema import FingerDesign, load_entrants
from rtt.search.random_search import DesignSampler
from rtt.sim.conditions import Suite, training_suite
from rtt.sim.evaluate import EpisodeRecord, Executor, evaluate_designs, score_designs, write_parquet
from rtt.sim.task import TaskConfig, default_task, task_digest

__all__ = [
    "CALIBRATION_SEED",
    "CalibrationReport",
    "Check",
    "DesignRow",
    "G3Targets",
    "calibration_designs",
    "file_sha256",
    "run_calibration",
]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_ENTRANTS: Final = REPO_ROOT / "configs" / "entrants.yaml"
#: Public seed of the calibration sample (not a search-run seed, not a secret).
CALIBRATION_SEED: Final = 20261018
DEFAULT_N_SOBOL: Final = 64

Origin = Literal["entrant", "sobol"]


@dataclass(frozen=True)
class G3Targets:
    """Gate G3 thresholds (plan section 3.5)."""

    entrant_min: float = 0.30
    entrant_max: float = 0.70
    family_gap: float = 0.15
    sobol_sd_min: float = 0.15
    headroom_min: int = 5
    kill_sd: float = 0.10  # plan kill trigger 4: below this the score barely responds to geometry


@dataclass(frozen=True)
class DesignRow:
    """One design's training score and breakdown (fractions in 0..1)."""

    name: str
    design_id: str
    origin: Origin
    score: float
    best_k: int
    hold_frac: float
    mass_g: float
    success_by_k: dict[int, float]
    success_by_family: dict[str, float]
    failure_modes: dict[str, int]
    flags: dict[str, int]
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "design_id": self.design_id,
            "origin": self.origin,
            "score": round(self.score, 4),
            "best_k": self.best_k,
            "hold_frac": round(self.hold_frac, 4),
            "mass_g": round(self.mass_g, 2),
            "success_by_k": {str(k): round(v, 4) for k, v in self.success_by_k.items()},
            "success_by_family": {k: round(v, 4) for k, v in self.success_by_family.items()},
            "failure_modes": dict(sorted(self.failure_modes.items())),
            "flags": dict(sorted(self.flags.items())),
            "params": self.params,
        }


@dataclass(frozen=True)
class Check:
    """One G3 criterion: what was measured, the target, and whether it passed."""

    name: str
    value: str
    target: str
    passed: bool


@dataclass
class CalibrationReport:
    """Result of :func:`run_calibration`."""

    task_name: str
    task_digest: str
    suite_name: str
    suite_digest: str
    families: tuple[str, ...]
    n_conditions: int
    episodes_per_design: int
    sobol_seed: int
    executor: str
    workers: int | None
    wall_s: float
    episodes: int
    entrants: list[DesignRow]
    sobol: list[DesignRow]
    targets: G3Targets = field(default_factory=G3Targets)
    task_file_sha256: str | None = None
    suite_file_sha256: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ statistics

    @property
    def best_entrant(self) -> DesignRow:
        return max(self.entrants, key=lambda r: (r.score, r.hold_frac, -r.mass_g))

    def sobol_scores(self) -> list[float]:
        return [r.score for r in self.sobol]

    def sobol_stats(self) -> dict[str, float | int]:
        """Mean, SD (``ddof=1``), quartiles and the count beating the best entrant."""
        scores = sorted(self.sobol_scores())
        if not scores:
            return {"n": 0}
        q = statistics.quantiles(scores, n=4, method="inclusive") if len(scores) > 1 else [scores[0]] * 3
        best = self.best_entrant.score
        return {
            "n": len(scores),
            "mean": statistics.fmean(scores),
            "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": scores[0],
            "q1": q[0],
            "median": q[1],
            "q3": q[2],
            "max": scores[-1],
            "n_zero": sum(s == 0.0 for s in scores),
            "n_perfect": sum(s == 1.0 for s in scores),
            "n_beat_best_entrant": sum(s > best for s in scores),
        }

    def family_gaps(self) -> dict[str, float]:
        """Largest minus smallest entrant success per family (at each entrant's best k)."""
        out: dict[str, float] = {}
        for fam in self.families:
            values = [r.success_by_family.get(fam, 0.0) for r in self.entrants]
            out[fam] = max(values) - min(values) if values else 0.0
        return out

    def checks(self) -> list[Check]:
        """The five G3 criteria, in plan order."""
        t = self.targets
        in_range = [r for r in self.entrants if t.entrant_min - 1e-9 <= r.score <= t.entrant_max + 1e-9]
        gaps = self.family_gaps()
        widest = max(gaps.items(), key=lambda kv: kv[1]) if gaps else ("-", 0.0)
        stats = self.sobol_stats()
        sd = float(stats.get("sd", 0.0))
        beat = int(stats.get("n_beat_best_entrant", 0))
        scores = ", ".join(f"{r.name} {100 * r.score:.0f}%" for r in self.entrants)
        return [
            Check(
                "entrants_in_range",
                f"{len(in_range)}/{len(self.entrants)} in range ({scores})",
                f"every entrant {100 * t.entrant_min:.0f}-{100 * t.entrant_max:.0f}%",
                len(in_range) == len(self.entrants),
            ),
            Check(
                "entrant_family_gap",
                f"widest gap {100 * widest[1]:.0f} pts on {widest[0]}",
                f">= {100 * t.family_gap:.0f} pts on at least one family",
                widest[1] >= t.family_gap - 1e-9,
            ),
            Check("sobol_sd", f"SD {sd:.3f}", f">= {t.sobol_sd_min:.2f}", sd >= t.sobol_sd_min),
            Check(
                "headroom",
                f"{beat}/{stats.get('n', 0)} beat {self.best_entrant.name} ({100 * self.best_entrant.score:.0f}%)",
                f">= {t.headroom_min} random designs beat the best entrant",
                beat >= t.headroom_min,
            ),
            Check(
                "kill_trigger_4",
                f"SD {sd:.3f}",
                f"SD >= {t.kill_sd:.2f} (else add geometry-sensitive conditions)",
                sd >= t.kill_sd,
            ),
        ]

    @property
    def passed(self) -> bool:
        """Whether gate G3 passes (the kill-trigger line is implied by the SD target)."""
        return all(c.passed for c in self.checks() if c.name != "kill_trigger_4")

    # ------------------------------------------------------------------ output

    def to_dict(self) -> dict[str, Any]:
        stats = self.sobol_stats()
        return {
            "schema": "rtt.calibration/1",
            "passed": self.passed,
            "task": {
                "name": self.task_name,
                "digest": self.task_digest,
                "file_sha256": self.task_file_sha256,
            },
            "suite": {
                "name": self.suite_name,
                "digest": self.suite_digest,
                "file_sha256": self.suite_file_sha256,
                "n_conditions": self.n_conditions,
                "families": list(self.families),
            },
            "episodes_per_design": self.episodes_per_design,
            "episodes": self.episodes,
            "sobol_seed": self.sobol_seed,
            "executor": self.executor,
            "workers": self.workers,
            "wall_s": round(self.wall_s, 2),
            "targets": vars(self.targets),
            "checks": [vars(c) for c in self.checks()],
            "family_gaps": {k: round(v, 4) for k, v in self.family_gaps().items()},
            "sobol_stats": {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
            "entrants": [r.to_dict() for r in self.entrants],
            "sobol": [r.to_dict() for r in self.sobol],
            "notes": self.notes,
        }

    def table(self) -> str:
        """Plain-text summary: per-entrant per-family success, Sobol statistics, the checks."""
        fams = self.families
        head = f"{'design':<22}{'score':>7}{'k':>3}" + "".join(f"{f:>10}" for f in fams) + f"{'mass g':>9}"
        lines = [
            (
                f"G3 calibration: task {self.task_name} ({self.task_digest}), suite {self.suite_name} "
                f"({self.n_conditions} conditions, {self.episodes_per_design} episodes per design), "
                f"{self.episodes} episodes in {self.wall_s:.0f} s on {self.executor}"
            ),
            "",
            head,
        ]
        for r in self.entrants:
            fam = "".join(f"{100 * r.success_by_family.get(f, 0.0):>9.0f}%" for f in fams)
            lines.append(f"{r.name:<22}{100 * r.score:>6.0f}%{r.best_k:>3}{fam}{r.mass_g:>9.1f}")
        stats = self.sobol_stats()
        if stats.get("n"):
            fam_means = {
                f: statistics.fmean(r.success_by_family.get(f, 0.0) for r in self.sobol) for f in fams
            }
            fam = "".join(f"{100 * fam_means[f]:>9.0f}%" for f in fams)
            lines.append(
                f"{'sobol mean (n=' + str(stats['n']) + ')':<22}{100 * float(stats['mean']):>6.0f}%   {fam}"
            )
            lines += [
                "",
                (
                    "Sobol scores: "
                    f"mean {stats['mean']:.3f}, SD {stats['sd']:.3f}, min {stats['min']:.3f}, "
                    f"q1 {stats['q1']:.3f}, median {stats['median']:.3f}, q3 {stats['q3']:.3f}, "
                    f"max {stats['max']:.3f}; {stats['n_perfect']} at 100 %, {stats['n_zero']} at 0 %"
                ),
            ]
        lines.append("")
        for c in self.checks():
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.value} (target {c.target})")
        lines.append(f"G3 {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- running


def file_sha256(path: str | Path) -> str:
    """SHA-256 (hex) of a file's bytes, as frozen into ``docs/TASK.md``."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_designs(
    entrants: Sequence[FingerDesign], suite: Suite, task: TaskConfig, n_sobol: int, seed: int
) -> list[tuple[str, Origin, FingerDesign]]:
    """The entrants, then ``n_sobol`` valid Sobol designs, as ``(name, origin, design)``."""
    gripper = suite.gripper_task(task)
    sampler = DesignSampler(seed, gripper)
    out: list[tuple[str, Origin, FingerDesign]] = [(d.name or d.design_id(), "entrant", d) for d in entrants]
    out += [(f"sobol-{i:02d}", "sobol", d) for i, d in enumerate(sampler.sample(n_sobol))]
    return out


def _row(name: str, origin: Origin, design: FingerDesign, score: Any) -> DesignRow:
    return DesignRow(
        name=name,
        design_id=score.design_id,
        origin=origin,
        score=float(score.score),
        best_k=int(score.best_k),
        hold_frac=float(score.hold_frac),
        mass_g=float(score.mass_g),
        success_by_k={int(k): float(v) for k, v in score.success_by_k.items()},
        success_by_family={str(k): float(v) for k, v in score.success_by_family.items()},
        failure_modes=dict(score.failure_modes),
        flags=dict(score.flags),
        params=design.geometry_params(),
    )


def run_calibration(
    task: TaskConfig | None = None,
    suite: Suite | None = None,
    entrants: Sequence[FingerDesign] | None = None,
    *,
    n_sobol: int = DEFAULT_N_SOBOL,
    seed: int = CALIBRATION_SEED,
    executor: Executor = "cpu",
    workers: int | None = None,
    targets: G3Targets | None = None,
    records_out: str | Path | None = None,
    task_file: str | Path | None = None,
    suite_file: str | Path | None = None,
) -> CalibrationReport:
    """Score the entrants and ``n_sobol`` random designs on the training suite (gate G3).

    Args:
        task: Simulator task (default ``configs/task_v1.yaml``).
        suite: Training suite (default ``configs/suites/train_v1.yaml``).
        entrants: Manual entrants (default ``configs/entrants.yaml``).
        n_sobol: Number of valid scrambled-Sobol designs (64 in the plan).
        seed: Calibration sample seed (public, fixed).
        executor: Simulator executor (``"cpu"``, a ``CpuExecutor``, ...).
        workers: Worker processes for the CPU executor (default: all CPUs).
        targets: G3 thresholds.
        records_out: Optional Parquet path for every episode record.
        task_file, suite_file: Files whose SHA-256 the report records.
    """
    task = task or default_task()
    suite = suite or training_suite()
    entrant_designs = list(entrants) if entrants is not None else load_entrants(DEFAULT_ENTRANTS)
    plan = calibration_designs(entrant_designs, suite, task, n_sobol, seed)
    unique = list({d.design_id(): d for _, _, d in plan}.values())
    t0 = time.perf_counter()
    records: list[EpisodeRecord] = evaluate_designs(
        unique, suite, executor=executor, workers=workers, task=task
    )
    wall = time.perf_counter() - t0
    if records_out is not None:
        write_parquet(records, records_out)
    scores = score_designs(records)
    rows = [_row(name, origin, d, scores[d.design_id()]) for name, origin, d in plan]
    n_settings = len(task.controller.grasp_height_fracs)
    exec_name = executor if isinstance(executor, str) else type(executor).__name__
    return CalibrationReport(
        task_name=task.name,
        task_digest=task_digest(task),
        suite_name=suite.name,
        suite_digest=suite.digest(),
        families=tuple(dict.fromkeys(c.family for c in suite)),
        n_conditions=len(suite),
        episodes_per_design=suite.n_episodes_per_setting * n_settings,
        sobol_seed=seed,
        executor=exec_name,
        workers=workers,
        wall_s=wall,
        episodes=len(records),
        entrants=[r for r in rows if r.origin == "entrant"],
        sobol=[r for r in rows if r.origin == "sobol"],
        targets=targets or G3Targets(),
        task_file_sha256=file_sha256(task_file) if task_file is not None else None,
        suite_file_sha256=file_sha256(suite_file) if suite_file is not None else None,
        notes={"steps_per_episode": _mean_steps(records)},
    )


def _mean_steps(records: Sequence[EpisodeRecord]) -> float:
    return round(statistics.fmean(r.n_steps for r in records), 1) if records else math.nan
