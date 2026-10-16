"""Gate GV3: calibration of ``task_v2`` / ``train_v2`` on the screening sample.

GV3 asks of the design scores (mean registered-grasp success over the 48 episodes of
``train_v2``):

1. **Entrants in range.** Every round-0 entrant (stock, flat, configurator imprint,
   swept imprint) scores 20-70 % registered. Reference entrants are reported only.
2. **Spread.** The standard deviation (``ddof=1``) of the 64 Sobol designs' scores is at
   least 0.15.
3. **Headroom.** At least 5 of the 64 Sobol designs beat the best round-0 entrant
   outright. A miss here may be declared in PREREG, as with v1 gate G3.
4. **Ceiling.** At most 10 % of the Sobol designs score 100 %.

Reported, not gating: lift-only scores, ``jam_on_close`` and ``off_centre`` counts, and the
share of ``physics_invalid`` episodes (above 1 % it is logged in ``docs/DEVIATIONS.md``).

**Procedure rules** (fixed before the pilot, enforced by :class:`CalibrationLog`):

* The screening sample's Sobol seed differs from every gate seed and every search seed
  (:func:`check_screening_seed`).
* At most 3 logged iterations. Each iteration after the first may change task-level knobs
  only: part dimensions, density or material (``part:``), placements (``placement:``),
  friction or grip force within the gripper's spec (``task:mu``, ``task:grip_force_n``), and
  the registration tolerances (``tolerance:``). Nothing per design, no entrant, no physics
  setting.
* The registration tolerances change in at most one iteration.
* After the last iteration: pass -> freeze; only the headroom criterion failing -> declare
  the miss and freeze; anything else -> fallback F1 (v1 protocol), the founder's go/no-go.

The freeze records the SHA-256 of ``configs/task_v2.yaml``, ``configs/suites/train_v2.yaml``,
``configs/entrants_v2.yaml`` and ``configs/parts_v2.yaml`` (:func:`freeze_hashes`).

This module reads episode records (dicts with ``RECORD_VERSION = 2`` field names) and
imports only the standard library, so it runs anywhere the records are.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

__all__ = [
    "CALIBRATION_SCHEMA_V2",
    "DECLARABLE",
    "FREEZE_FILES",
    "MAX_ITERATIONS",
    "SCREENING_SEED_V2",
    "CalibrationLog",
    "CalibrationV2Report",
    "Check",
    "DesignScoreV2",
    "GV3Targets",
    "check_knob",
    "check_screening_seed",
    "freeze_hashes",
    "gv3_decision",
    "score_designs_v2",
]

CALIBRATION_SCHEMA_V2: Final = "rtt.calibration/2"
#: Public seed of the v2 screening sample (not a run seed, not a gate seed, not a secret).
SCREENING_SEED_V2: Final = 20261019
#: Seeds used before the freeze that a screening sample must not reuse (see seeds_pilot.yaml).
DEV_SEEDS: Final[frozenset[int]] = frozenset({17, 101, 102, 103, *range(201, 221), 20261015, 20261018})
MAX_ITERATIONS: Final = 3
MAX_TOLERANCE_CHANGES: Final = 1
#: Criteria whose miss may be declared instead of failing over to F1.
DECLARABLE: Final[frozenset[str]] = frozenset({"headroom"})
FREEZE_FILES: Final[tuple[str, ...]] = (
    "configs/task_v2.yaml",
    "configs/suites/train_v2.yaml",
    "configs/entrants_v2.yaml",
    "configs/parts_v2.yaml",
)
_TASK_KNOBS: Final[frozenset[str]] = frozenset({"mu", "grip_force_n"})
_TOLERANCES: Final[frozenset[str]] = frozenset(
    {"seat_short_mm", "centre_dx_mm", "centre_dy_mm", "yaw_deg", "tilt_close_deg", "tilt_hold_deg"}
)
Origin = Literal["entrant", "reference", "sobol"]
Decision = Literal["freeze", "iterate", "declare_miss_and_freeze", "f1"]


# --------------------------------------------------------------------------- scores


@dataclass(frozen=True)
class DesignScoreV2:
    """One design's ``train_v2`` score and breakdown (fractions in 0..1)."""

    name: str
    design_id: str
    origin: Origin
    score: float
    lift_score: float
    n_episodes: int
    mass_g: float | None
    by_part: dict[str, float]
    by_placement: dict[str, float]
    failure_modes: dict[str, int]
    jam_on_close: int
    off_centre: int
    physics_invalid: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("score", "lift_score"):
            d[key] = round(d[key], 6)
        return d


def _success_v2(rec: Mapping[str, Any]) -> bool:
    if "success_v2" in rec:
        return bool(rec["success_v2"])
    return bool(rec["success"]) and bool(rec["registered"])


def _invalid(rec: Mapping[str, Any]) -> bool:
    return bool(rec.get("bad_qacc")) or rec.get("failure_mode") == "physics_invalid"


def score_designs_v2(
    records: Iterable[Mapping[str, Any]],
    names: Mapping[str, tuple[str, Origin]],
    *,
    episodes_per_design: int | None = 48,
    masses_g: Mapping[str, float] | None = None,
) -> list[DesignScoreV2]:
    """Score every design in ``names`` (design_id -> (name, origin)) from its episode records.

    ``success_v2`` is read from the record, or computed as lift-and-hold ``success`` and
    ``registered``; a ``physics_invalid`` episode (``bad_qacc`` or that failure mode) is a
    failure. Every design must have ``episodes_per_design`` episodes (``None``: any number).
    """
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["design_id"])].append(rec)
    out = []
    for design_id, (name, origin) in names.items():
        recs = grouped.get(design_id, [])
        if not recs:
            raise ValueError(f"{name} ({design_id}): no episode records")
        if episodes_per_design is not None and len(recs) != episodes_per_design:
            raise ValueError(f"{name} ({design_id}): {len(recs)} episodes, expected {episodes_per_design}")
        ok = [_success_v2(r) and not _invalid(r) for r in recs]
        by_part: dict[str, list[bool]] = defaultdict(list)
        by_place: dict[str, list[bool]] = defaultdict(list)
        for r, s in zip(recs, ok, strict=True):
            if r.get("part_id") is not None:
                by_part[str(r["part_id"])].append(s)
            if r.get("placement") is not None:
                by_place[str(r["placement"])].append(s)
        modes = Counter(str(r["failure_mode"]) for r in recs if r.get("failure_mode"))
        out.append(
            DesignScoreV2(
                name=name,
                design_id=design_id,
                origin=origin,
                score=sum(ok) / len(ok),
                lift_score=sum(bool(r["success"]) and not _invalid(r) for r in recs) / len(recs),
                n_episodes=len(recs),
                mass_g=None if masses_g is None else masses_g.get(design_id),
                by_part={k: sum(v) / len(v) for k, v in sorted(by_part.items())},
                by_placement={k: sum(v) / len(v) for k, v in sorted(by_place.items())},
                failure_modes=dict(sorted(modes.items())),
                jam_on_close=sum(bool(r.get("jam_on_close")) for r in recs),
                off_centre=sum(bool(r.get("off_centre")) for r in recs),
                physics_invalid=sum(_invalid(r) for r in recs),
            )
        )
    return out


# --------------------------------------------------------------------------- the gate


@dataclass(frozen=True)
class GV3Targets:
    """Gate GV3 thresholds (spec section 14)."""

    entrant_min: float = 0.20
    entrant_max: float = 0.70
    sobol_sd_min: float = 0.15
    headroom_min: int = 5
    perfect_share_max: float = 0.10
    n_sobol: int = 64
    n_round0_entrants: int = 4
    physics_invalid_max: float = 0.01  # reported; above it a deviation is logged


@dataclass(frozen=True)
class Check:
    name: str
    value: str
    target: str
    passed: bool
    gating: bool = True


@dataclass
class CalibrationV2Report:
    """GV3 on one screening sample (one iteration)."""

    rows: list[DesignScoreV2]
    screening_seed: int
    iteration: int = 1
    targets: GV3Targets = field(default_factory=GV3Targets)
    task_digest: str | None = None
    suite_digest: str | None = None
    file_sha256: dict[str, str] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [r.design_id for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate design IDs in the calibration rows")

    @property
    def entrants(self) -> list[DesignScoreV2]:
        return [r for r in self.rows if r.origin == "entrant"]

    @property
    def references(self) -> list[DesignScoreV2]:
        return [r for r in self.rows if r.origin == "reference"]

    @property
    def sobol(self) -> list[DesignScoreV2]:
        return [r for r in self.rows if r.origin == "sobol"]

    @property
    def best_entrant(self) -> DesignScoreV2:
        return max(self.entrants, key=lambda r: (r.score, r.lift_score, -(r.mass_g or 0.0)))

    def sobol_stats(self) -> dict[str, float | int]:
        scores = sorted(r.score for r in self.sobol)
        if not scores:
            return {"n": 0}
        best = self.best_entrant.score
        return {
            "n": len(scores),
            "mean": statistics.fmean(scores),
            "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": scores[0],
            "median": statistics.median(scores),
            "max": scores[-1],
            "n_perfect": sum(s >= 1.0 - 1e-9 for s in scores),
            "n_zero": sum(s <= 1e-9 for s in scores),
            "n_beat_best_entrant": sum(s > best + 1e-9 for s in scores),
            "lift_mean": statistics.fmean(r.lift_score for r in self.sobol),
        }

    def checks(self) -> list[Check]:
        t = self.targets
        if len(self.entrants) != t.n_round0_entrants:
            raise ValueError(f"expected {t.n_round0_entrants} round-0 entrants, got {len(self.entrants)}")
        if len(self.sobol) != t.n_sobol:
            raise ValueError(f"expected {t.n_sobol} Sobol designs, got {len(self.sobol)}")
        stats = self.sobol_stats()
        in_range = [r for r in self.entrants if t.entrant_min - 1e-9 <= r.score <= t.entrant_max + 1e-9]
        listing = ", ".join(f"{r.name} {100 * r.score:.1f}%" for r in self.entrants)
        sd = float(stats["sd"])
        beat = int(stats["n_beat_best_entrant"])
        perfect = int(stats["n_perfect"])
        n_eps = sum(r.n_episodes for r in self.rows)
        invalid = sum(r.physics_invalid for r in self.rows)
        share = invalid / n_eps if n_eps else 0.0
        return [
            Check(
                "entrants_in_range",
                f"{len(in_range)}/{len(self.entrants)} in range ({listing})",
                f"every round-0 entrant {100 * t.entrant_min:.0f}-{100 * t.entrant_max:.0f}% registered",
                len(in_range) == len(self.entrants),
            ),
            Check("sobol_sd", f"SD {sd:.3f}", f">= {t.sobol_sd_min:.2f}", sd >= t.sobol_sd_min - 1e-12),
            Check(
                "headroom",
                f"{beat}/{stats['n']} beat {self.best_entrant.name} ({100 * self.best_entrant.score:.1f}%)",
                f">= {t.headroom_min} Sobol designs beat the best round-0 entrant",
                beat >= t.headroom_min,
            ),
            Check(
                "ceiling",
                f"{perfect}/{stats['n']} at 100%",
                f"<= {100 * t.perfect_share_max:.0f}% of Sobol designs at 100%",
                perfect <= t.perfect_share_max * t.n_sobol + 1e-9,
            ),
            Check(
                "physics_invalid_share",
                f"{invalid}/{n_eps} episodes ({100 * share:.2f}%)",
                f"<= {100 * t.physics_invalid_max:.0f}% (else log a deviation)",
                share <= t.physics_invalid_max + 1e-12,
                gating=False,
            ),
        ]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks() if c.gating)

    def failing(self) -> list[str]:
        return [c.name for c in self.checks() if c.gating and not c.passed]

    def to_dict(self) -> dict[str, Any]:
        stats = self.sobol_stats()
        return {
            "schema": CALIBRATION_SCHEMA_V2,
            "design_space": "freeform-v2",
            "passed": self.passed,
            "iteration": self.iteration,
            "screening_seed": self.screening_seed,
            "task_digest": self.task_digest,
            "suite_digest": self.suite_digest,
            "file_sha256": dict(sorted(self.file_sha256.items())),
            "targets": asdict(self.targets),
            "checks": [asdict(c) for c in self.checks()],
            "sobol_stats": {k: round(v, 6) if isinstance(v, float) else v for k, v in stats.items()},
            "entrants": [r.to_dict() for r in self.entrants],
            "references": [r.to_dict() for r in self.references],
            "sobol": [r.to_dict() for r in self.sobol],
            "notes": self.notes,
        }

    def table(self) -> str:
        lines = [f"GV3 calibration, iteration {self.iteration}, screening seed {self.screening_seed}", ""]
        lines.append(f"{'design':<24}{'registered':>11}{'lift':>8}{'jam':>6}{'off':>6}{'invalid':>8}")
        for r in [*self.entrants, *self.references]:
            tag = r.name + (" (ref)" if r.origin == "reference" else "")
            lines.append(
                f"{tag:<24}{100 * r.score:>10.1f}%{100 * r.lift_score:>7.1f}%{r.jam_on_close:>6}"
                f"{r.off_centre:>6}{r.physics_invalid:>8}"
            )
        stats = self.sobol_stats()
        if stats.get("n"):
            lines.append(
                f"Sobol-{stats['n']}: mean {stats['mean']:.3f}, SD {stats['sd']:.3f}, median {stats['median']:.3f}, "
                f"max {stats['max']:.3f}; {stats['n_perfect']} at 100 %, {stats['n_zero']} at 0 %; "
                f"lift-only mean {stats['lift_mean']:.3f}"
            )
        lines.append("")
        for c in self.checks():
            mark = "PASS" if c.passed else ("FAIL" if c.gating else "NOTE")
            lines.append(f"  [{mark}] {c.name}: {c.value} (target {c.target})")
        lines.append(f"GV3 {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def gv3_decision(report: CalibrationV2Report, iterations_used: int) -> tuple[Decision, list[str]]:
    """What happens after an iteration: freeze, iterate, declare the miss, or fall back to F1."""
    failing = report.failing()
    if not failing:
        return "freeze", []
    if iterations_used < MAX_ITERATIONS:
        return "iterate", failing
    if set(failing) <= DECLARABLE:
        return "declare_miss_and_freeze", failing
    return "f1", failing


# --------------------------------------------------------------------------- procedure


def check_screening_seed(seed: int, forbidden: Iterable[int] = ()) -> None:
    """Refuse a screening seed that equals a gate seed, a search seed or a development seed."""
    bad = set(DEV_SEEDS) | {int(s) for s in forbidden}
    if int(seed) in bad:
        raise ValueError(f"screening seed {seed} is also a gate, search or development seed")


def check_knob(name: str) -> str:
    """Return the category of a task-level knob, or raise for anything a calibration may not touch."""
    category, sep, rest = name.partition(":")
    if not sep or not rest:
        raise ValueError(f"knob {name!r}: expected '<category>:<name>'")
    if category in ("part", "placement"):
        return category
    if category == "task" and rest in _TASK_KNOBS:
        return category
    if category == "tolerance" and rest in _TOLERANCES:
        return category
    raise ValueError(
        f"knob {name!r} may not be tuned in calibration (allowed: part:*, placement:*, "
        f"task:{{{','.join(sorted(_TASK_KNOBS))}}}, tolerance:{{{','.join(sorted(_TOLERANCES))}}})"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_hashes(repo_root: str | Path, files: Sequence[str] = FREEZE_FILES) -> dict[str, str]:
    """SHA-256 of each frozen file, keyed by its repo-relative path; every file must exist."""
    root = Path(repo_root)
    missing = [f for f in files if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze, missing: {missing}")
    return {f: _sha256(root / f) for f in files}


@dataclass
class CalibrationLog:
    """Append-only log of GV3 iterations (``calibration_log.json``); enforces the procedure rules."""

    path: Path
    iterations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def open(cls, path: str | Path) -> CalibrationLog:
        p = Path(path)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("schema") != "rtt.calibration_log/2":
                raise ValueError(f"{p}: not a v2 calibration log")
            return cls(p, list(data["iterations"]))
        return cls(p)

    def tolerance_changes(self) -> int:
        return sum(1 for it in self.iterations if any(k.startswith("tolerance:") for k in it["changes"]))

    def record(
        self,
        report: CalibrationV2Report,
        *,
        changes: Mapping[str, tuple[Any, Any]] | None = None,
        note: str = "",
        forbidden_seeds: Iterable[int] = (),
        now: _dt.datetime | None = None,
    ) -> tuple[Decision, list[str]]:
        """Log one iteration and return the decision; refuses anything the procedure forbids."""
        changes = dict(changes or {})
        n = len(self.iterations) + 1
        if self.iterations and self.iterations[-1]["decision"] != "iterate":
            raise ValueError(f"calibration already closed with {self.iterations[-1]['decision']!r}")
        if n > MAX_ITERATIONS:
            raise ValueError(f"at most {MAX_ITERATIONS} calibration iterations")
        if report.iteration != n:
            raise ValueError(f"report is iteration {report.iteration}, the log expects {n}")
        if n == 1 and changes:
            raise ValueError("iteration 1 is the task as drafted; changes start at iteration 2")
        if n > 1 and not changes:
            raise ValueError(f"iteration {n} must log what changed")
        for knob in changes:
            check_knob(knob)
        touches_tol = any(k.startswith("tolerance:") for k in changes)
        if touches_tol and self.tolerance_changes() >= MAX_TOLERANCE_CHANGES:
            raise ValueError("the registration tolerances may change in at most one iteration")
        check_screening_seed(report.screening_seed, forbidden_seeds)
        if self.iterations and report.screening_seed != self.iterations[0]["screening_seed"]:
            raise ValueError("every iteration uses the same screening sample (same seed)")
        decision, failing = gv3_decision(report, n)
        when = (now or _dt.datetime.now(_dt.UTC)).isoformat(timespec="seconds")
        self.iterations.append(
            {
                "iteration": n,
                "at": when,
                "screening_seed": report.screening_seed,
                "changes": {k: list(v) for k, v in sorted(changes.items())},
                "file_sha256": dict(sorted(report.file_sha256.items())),
                "checks": [asdict(c) for c in report.checks()],
                "passed": report.passed,
                "decision": decision,
                "failing": failing,
                "note": note,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"schema": "rtt.calibration_log/2", "iterations": self.iterations}
        self.path.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
        return decision, failing
