"""Batch evaluation: designs x suite conditions x controller settings x jitter seeds.

:func:`evaluate_designs` expands every combination into episode jobs, runs them on an
executor and returns one flat :class:`EpisodeRecord` per episode, in a fixed order
(design, condition, controller setting, seed), ready for a Parquet table
(:func:`records_to_table`, :func:`write_parquet`). :func:`score_designs` reduces the
records to the plan's design score: the best controller setting's mean success, ties
broken by its mean hold fraction, then by lower finger mass.

Executors: ``"cpu"`` runs :mod:`rtt.sim.exec_cpu` (a process pool); a
:class:`~rtt.sim.exec_cpu.CpuExecutor` instance reuses an open pool across calls;
``"mjwarp"`` uses ``rtt.sim.exec_mjwarp.run_jobs(jobs, workers)`` when that module is
present; any callable with the same signature works too. Every executor must return
:class:`~rtt.sim.exec_cpu.BatchResult` with results in job order.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rtt.design.generate import generate_finger
from rtt.design.schema import FingerDesign
from rtt.design.types import FingerGeometry
from rtt.sim.conditions import Condition, Suite
from rtt.sim.episode import EpisodeResult
from rtt.sim.exec_cpu import BatchResult, CpuExecutor, EpisodeJob
from rtt.sim.exec_cpu import run_jobs as run_jobs_cpu
from rtt.sim.scene import finger_mass_properties
from rtt.sim.task import TaskConfig, default_task, task_digest

Executor = str | CpuExecutor | Callable[[Sequence[EpisodeJob], int | None], BatchResult]
DesignLike = FingerDesign | FingerGeometry
RECORD_VERSION = 1


def geometry_id(finger: FingerGeometry) -> str:
    """Stable 12-hex-digit id of raw finger geometry (for fingers without a FingerDesign)."""
    payload = json.dumps(
        {"boxes": [asdict(b) for b in finger.boxes], "density": finger.density_kg_m3},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "g" + hashlib.sha256(payload.encode()).hexdigest()[:11]


def resolve_design(design: DesignLike) -> tuple[str, FingerGeometry]:
    """``(design_id, geometry)`` for a FingerDesign (generated) or raw geometry."""
    if isinstance(design, FingerDesign):
        return design.design_id(), generate_finger(design)
    if isinstance(design, FingerGeometry):
        return geometry_id(design), design
    raise TypeError(f"expected a FingerDesign or FingerGeometry, got {type(design).__name__}")


@dataclass(frozen=True)
class EpisodeRecord:
    """One evaluated episode, flat and JSON/Parquet friendly (SI units).

    ``lift`` is the object's maximum rise (m), ``hold`` the longest continuous time
    above the success height (s), ``hold_frac`` that time over the required hold
    (capped at 1). ``flags`` are physics-check names (e.g. ``penetration``,
    ``palm_contact``); ``wall_time_s`` is the only field that is not deterministic.
    """

    design_id: str
    condition_id: str
    k: int
    seed: int
    success: bool
    failure_mode: str | None
    lift: float
    hold: float
    flags: tuple[str, ...]
    hold_frac: float
    grasp_frac: float
    suite: str
    family: str
    object_kind: str
    size_label: str
    placement_label: str
    finger_mass_g: float
    tip_height: float
    palm_gap: float
    penetration_max: float
    decided_at: float
    n_steps: int
    task: str
    task_digest: str
    executor: str
    wall_time_s: float = field(compare=False)
    record_version: int = RECORD_VERSION

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["flags"] = list(self.flags)
        return out


def _flag_names(result: EpisodeResult) -> tuple[str, ...]:
    return tuple(f.split("@", 1)[0] for f in result.flags)


def _finite(value: float) -> float:
    return float(value) if math.isfinite(value) else float("nan")


def make_record(
    design_id: str,
    condition: Condition,
    suite_name: str,
    result: EpisodeResult,
    task: TaskConfig,
    executor: str,
    finger_mass_g: float,
) -> EpisodeRecord:
    """Flatten one episode result into a record."""
    m = result.metrics
    return EpisodeRecord(
        design_id=design_id,
        condition_id=condition.condition_id,
        k=result.controller_setting,
        seed=result.seed,
        success=result.success,
        failure_mode=result.failure_mode,
        lift=result.max_lift,
        hold=result.hold_time,
        flags=_flag_names(result),
        hold_frac=min(1.0, result.hold_time / task.score.success_hold),
        grasp_frac=m["grasp_fraction"],
        suite=suite_name,
        family=condition.family,
        object_kind=condition.obj.kind,
        size_label=condition.size_label,
        placement_label=condition.placement_label,
        finger_mass_g=finger_mass_g,
        tip_height=m["tip_height"],
        palm_gap=m["palm_gap"],
        penetration_max=m["penetration_max"],
        decided_at=_finite(m["decided_at"]),
        n_steps=result.n_steps,
        task=task.name,
        task_digest=task_digest(task),
        executor=executor,
        wall_time_s=result.wall_time_s,
    )


def _resolve_executor(
    executor: Executor,
) -> tuple[str, Callable[[Sequence[EpisodeJob], int | None], BatchResult]]:
    if isinstance(executor, CpuExecutor):
        return "cpu", lambda jobs, _workers: executor.run(jobs)
    if callable(executor):
        return getattr(executor, "__name__", "custom"), executor
    if executor == "cpu":
        return "cpu", run_jobs_cpu
    if executor == "mjwarp":
        try:
            module = importlib.import_module("rtt.sim.exec_mjwarp")
        except ImportError as exc:
            raise ValueError(f"executor 'mjwarp' is not available: {exc}") from exc
        run = getattr(module, "run_jobs", None)
        if run is None:
            raise ValueError("rtt.sim.exec_mjwarp has no run_jobs(jobs, workers) function")
        return "mjwarp", run
    raise ValueError(f"unknown executor {executor!r}; use 'cpu', 'mjwarp', a CpuExecutor or a callable")


def evaluate_designs(
    designs: Sequence[DesignLike],
    suite: Suite,
    controller_settings: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    executor: Executor = "cpu",
    workers: int | None = None,
    task: TaskConfig | None = None,
) -> list[EpisodeRecord]:
    """Evaluate every design on every condition, controller setting and seed.

    ``controller_settings`` defaults to every setting of the task (k = 0..2);
    ``seeds`` replaces each condition's own jitter seeds when given. Records come back
    ordered by design, condition (suite order), setting, then seed.
    """
    task = task or default_task()
    settings = (
        tuple(range(len(task.controller.grasp_height_fracs)))
        if controller_settings is None
        else tuple(controller_settings)
    )
    n_settings = len(task.controller.grasp_height_fracs)
    if not settings or any(not 0 <= k < n_settings for k in settings):
        raise ValueError(f"controller settings must be in [0, {n_settings}), got {settings}")
    name, run = _resolve_executor(executor)

    resolved = [resolve_design(d) for d in designs]
    masses = {did: 1000.0 * finger_mass_properties(geom).mass for did, geom in resolved}
    jobs: list[EpisodeJob] = []
    keys: list[tuple[str, Condition]] = []
    for did, geom in resolved:
        for cond in suite:
            for k in settings:
                for seed in seeds if seeds is not None else cond.seeds:
                    jobs.append(EpisodeJob(geom, cond.obj, cond.placement, k, int(seed), task=task))
                    keys.append((did, cond))
    batch = run(jobs, workers)
    if len(batch.results) != len(jobs):
        raise RuntimeError(f"executor {name!r} returned {len(batch.results)} results for {len(jobs)} jobs")
    return [
        make_record(did, cond, suite.name, res, task, name, masses[did])
        for (did, cond), res in zip(keys, batch.results, strict=True)
    ]


def evaluate_design(
    design: DesignLike,
    suite: Suite,
    controller_settings: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    executor: Executor = "cpu",
    workers: int | None = None,
    task: TaskConfig | None = None,
) -> list[EpisodeRecord]:
    """:func:`evaluate_designs` for a single design."""
    return evaluate_designs([design], suite, controller_settings, seeds, executor, workers, task)


# --------------------------------------------------------------------------- tables and scores


def records_to_table(records: Sequence[EpisodeRecord]) -> Any:
    """A ``pyarrow.Table`` with one row per record and a fixed schema."""
    import pyarrow as pa

    schema = pa.schema(
        [
            ("design_id", pa.string()),
            ("condition_id", pa.string()),
            ("k", pa.int16()),
            ("seed", pa.int64()),
            ("success", pa.bool_()),
            ("failure_mode", pa.string()),
            ("lift", pa.float64()),
            ("hold", pa.float64()),
            ("flags", pa.list_(pa.string())),
            ("hold_frac", pa.float64()),
            ("grasp_frac", pa.float64()),
            ("suite", pa.string()),
            ("family", pa.string()),
            ("object_kind", pa.string()),
            ("size_label", pa.string()),
            ("placement_label", pa.string()),
            ("finger_mass_g", pa.float64()),
            ("tip_height", pa.float64()),
            ("palm_gap", pa.float64()),
            ("penetration_max", pa.float64()),
            ("decided_at", pa.float64()),
            ("n_steps", pa.int64()),
            ("task", pa.string()),
            ("task_digest", pa.string()),
            ("executor", pa.string()),
            ("wall_time_s", pa.float64()),
            ("record_version", pa.int16()),
        ]
    )
    columns = {f.name: [getattr(r, f.name) for r in records] for f in schema}
    columns["flags"] = [list(r.flags) for r in records]
    return pa.table(columns, schema=schema)


def write_parquet(records: Sequence[EpisodeRecord], path: str | Path) -> Path:
    """Write records to a Parquet file (zstd)."""
    import pyarrow.parquet as pq

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(records_to_table(records), p, compression="zstd")
    return p


@dataclass(frozen=True)
class DesignScore:
    """A design's training score and the breakdown an LLM proposer is shown."""

    design_id: str
    score: float  # mean success of the best controller setting
    best_k: int
    hold_frac: float  # mean hold fraction at best_k (first tie-break)
    mass_g: float  # second tie-break (lower is better)
    success_by_k: dict[int, float]
    success_by_family: dict[str, float]  # at best_k
    failure_modes: dict[str, int]  # at best_k
    flags: dict[str, int]  # at best_k
    n_episodes: int

    def sort_key(self) -> tuple[float, float, float]:
        """Ascending key that ranks better designs first."""
        return (-self.score, -self.hold_frac, self.mass_g)


def score_designs(records: Sequence[EpisodeRecord]) -> dict[str, DesignScore]:
    """Reduce episode records to one :class:`DesignScore` per design (in first-seen order)."""
    by_design: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for r in records:
        by_design[r.design_id].append(r)
    out: dict[str, DesignScore] = {}
    for did, recs in by_design.items():
        by_k: dict[int, list[EpisodeRecord]] = defaultdict(list)
        for r in recs:
            by_k[r.k].append(r)

        def stats(rs: list[EpisodeRecord]) -> tuple[float, float]:
            return (sum(r.success for r in rs) / len(rs), sum(r.hold_frac for r in rs) / len(rs))

        best_k = min(by_k, key=lambda k: (-stats(by_k[k])[0], -stats(by_k[k])[1], k))
        best = by_k[best_k]
        fam: dict[str, list[bool]] = defaultdict(list)
        for r in best:
            fam[r.family].append(r.success)
        out[did] = DesignScore(
            design_id=did,
            score=stats(best)[0],
            best_k=best_k,
            hold_frac=stats(best)[1],
            mass_g=recs[0].finger_mass_g,
            success_by_k={k: stats(v)[0] for k, v in sorted(by_k.items())},
            success_by_family={f: sum(v) / len(v) for f, v in fam.items()},
            failure_modes=dict(Counter(r.failure_mode for r in best if r.failure_mode)),
            flags=dict(Counter(f for r in best for f in r.flags)),
            n_episodes=len(recs),
        )
    return out


def rank_designs(records: Sequence[EpisodeRecord]) -> list[DesignScore]:
    """Design scores, best first (score, then hold fraction, then lower mass, then id)."""
    return sorted(score_designs(records).values(), key=lambda s: (*s.sort_key(), s.design_id))
