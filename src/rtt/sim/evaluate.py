"""Batch evaluation: designs x suite conditions x controller settings x jitter seeds.

:func:`evaluate_designs` expands every combination into episode jobs, runs them on an
executor and returns one flat :class:`EpisodeRecord` per episode, in a fixed order
(design, condition, controller setting, seed), ready for a Parquet table
(:func:`records_to_table`, :func:`write_parquet`). :func:`score_designs` reduces the
records to the plan's design score: the best controller setting's mean success, ties
broken by its mean hold fraction, then by lower finger mass.

Task v2 (:func:`evaluate_designs_v2`): every design is compiled against every part of a
:class:`~rtt.parts.suite.PartSuite` by a builder (default
:func:`rtt.design.freeform.build_pair`), each pair runs on that part's placements and seeds,
and every episode becomes an :class:`EpisodeRecordV2` (``RECORD_VERSION_V2 = 2``). A design
that cannot be built for a part scores ``invalid_build`` failures there without simulating.
:func:`score_designs_v2` reduces them: score = mean registered-grasp success over every
episode (no best-of-k), ties broken by lift-only success, then lower mass.

Executors: ``"cpu"`` runs :mod:`rtt.sim.exec_cpu` (a process pool); a
:class:`~rtt.sim.exec_cpu.CpuExecutor` instance reuses an open pool across calls;
``"mjwarp"`` uses ``rtt.sim.exec_mjwarp.run_jobs(jobs, workers)`` when that module is
present; any callable with the same signature works too. Every executor must return
:class:`~rtt.sim.exec_cpu.BatchResult` with results in job order.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
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


# --------------------------------------------------------------------------- task v2

RECORD_VERSION_V2 = 2
Builder = Callable[[Any, Any, Any], Any]  # (design, part, task) -> FingerPair
_GEOMETRY_ID_DOMAIN = b"rtt-freeform-geometry-id/1\n"


def suite_parts(suite: Any) -> tuple[Any, ...]:
    """The distinct parts of a v2 suite (:class:`rtt.parts.suite.PartSuite`), in suite order."""
    parts = suite.parts
    return tuple(parts() if callable(parts) else parts)


def condition_task_v2(task: TaskConfig, condition: Any) -> TaskConfig:
    """``task`` for one v2 condition.

    A held-out condition may carry a height error ``err_z`` (m): the part sits that much
    higher than the controller assumes, so the jaws end up ``err_z`` lower on it. It is
    applied as ``controller.grasp_z_offset - err_z``; ``task`` is returned unchanged when the
    condition has none.
    """
    err_z = float(getattr(condition, "err_z", 0.0) or 0.0)
    if err_z == 0.0:
        return task
    offset = task.controller.grasp_z_offset - err_z
    return task.with_overrides(controller={"grasp_z_offset": offset})


def suite_geometry_id(hashes: Mapping[str, str]) -> str:
    """SHA-256 over ``{part_id: geometry_hash}`` (replay ``meta.geometry_id``).

    The generator's definition (:func:`rtt.design.freeform.hashing.geometry_id`) is used when
    it is installed; the fallback computes the same digest.
    """
    try:
        from rtt.design.freeform.hashing import geometry_id as generator_geometry_id
    except ImportError:
        items = sorted(dict(hashes).items())
        if not items:
            raise ValueError("geometry_id needs at least one part") from None
        h = hashlib.sha256(_GEOMETRY_ID_DOMAIN)
        for part_id, value in items:
            h.update(f"{part_id}={value}\n".encode())
        return h.hexdigest()
    return generator_geometry_id(hashes)


@dataclass(frozen=True)
class EpisodeRecordV2:
    """One evaluated v2 episode (spec section 7), flat and JSON/Parquet friendly.

    ``success`` is the registered grasp (``success_v2``), ``lift_success`` the v1 lift-and-hold
    criterion. Registration values are millimetres and degrees at the registration moment
    (NaN when the episode ended before it; :meth:`to_dict` writes ``None`` for every NaN).
    ``jam_on_close`` and ``off_centre`` are the registration sub-flags as booleans (also in
    ``registration_flags``). ``tilt_max_deg`` is the part's largest tilt from vertical before the
    outcome was decided (the hero rule's visible-failure test). ``hold_creep_mm`` is logged, never
    scored. ``bad_qacc`` counts
    MuJoCo state resets. ``geometry_hash`` identifies the jaw pair built for this part (empty
    for an ``invalid_build``). ``err_z_mm`` is a held-out height error (0 in training).
    ``wall_time_s`` is the only field that is not deterministic.
    """

    design_id: str
    part_id: str
    condition_id: str
    seed: int
    success: bool
    lift_success: bool
    failure_mode: str | None
    registered: bool
    jam_on_close: bool
    off_centre: bool
    seat_short_mm: float
    reg_dx_mm: float
    reg_dy_mm: float
    reg_dyaw_deg: float
    reg_tilt_deg: float
    reg_hold_tilt_deg: float
    tilt_max_deg: float
    registration_flags: tuple[str, ...]
    hold_creep_mm: float
    bad_qacc: int
    lift: float
    hold: float
    hold_frac: float
    flags: tuple[str, ...]
    suite: str
    family: str
    size_label: str
    placement_label: str
    err_z_mm: float
    finger_mass_g: float
    q_seat_mm: float
    tip_height: float
    palm_gap: float
    penetration_max: float
    obj_finger_pen_max: float
    decided_at: float
    n_steps: int
    geometry_hash: str
    design_space: str
    task: str
    task_digest: str
    executor: str
    wall_time_s: float = field(compare=False)
    record_version: int = RECORD_VERSION_V2

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["flags"] = list(self.flags)
        out["registration_flags"] = list(self.registration_flags)
        for key, value in out.items():
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
        return out


def _condition_fields(condition: Any) -> dict[str, Any]:
    part = condition.part
    return {
        "part_id": str(part.part_id),
        "condition_id": str(condition.condition_id),
        "family": str(part.family),
        "size_label": str(getattr(condition, "size_label", "") or ""),
        "placement_label": str(getattr(condition, "placement_label", "") or ""),
        "err_z_mm": 1000.0 * float(getattr(condition, "err_z", 0.0) or 0.0),
    }


def make_record_v2(
    design_id: str,
    condition: Any,
    suite_name: str,
    result: Any,
    pair: Any,
    task: TaskConfig,
    executor: str,
) -> EpisodeRecordV2:
    """Flatten one :class:`~rtt.sim.episode.EpisodeResultV2` into a record."""
    m = result.metrics
    reg = result.registration
    return EpisodeRecordV2(
        design_id=design_id,
        seed=result.seed,
        success=result.success,
        lift_success=result.lift_success,
        failure_mode=result.failure_mode,
        registered=reg.registered,
        jam_on_close="jam_on_close" in reg.sub_flags,
        off_centre="off_centre" in reg.sub_flags,
        seat_short_mm=_finite(reg.seat_short_mm),
        reg_dx_mm=_finite(reg.dx_mm),
        reg_dy_mm=_finite(reg.dy_mm),
        reg_dyaw_deg=_finite(reg.dyaw_deg),
        reg_tilt_deg=_finite(reg.tilt_deg),
        reg_hold_tilt_deg=_finite(reg.hold_tilt_deg),
        tilt_max_deg=_finite(m["tilt_max_deg"]),
        registration_flags=tuple(reg.sub_flags),
        hold_creep_mm=_finite(m["hold_creep_mm"]),
        bad_qacc=int(result.bad_qacc),
        lift=result.max_lift,
        hold=result.hold_time,
        hold_frac=min(1.0, result.hold_time / task.score.success_hold),
        flags=_flag_names(result),
        suite=suite_name,
        finger_mass_g=1000.0 * (pair.left.mass + pair.right.mass),
        q_seat_mm=1000.0 * pair.q_seat,
        tip_height=m["tip_height"],
        palm_gap=m["palm_gap"],
        penetration_max=m["penetration_max"],
        obj_finger_pen_max=m["obj_finger_pen_max"],
        decided_at=_finite(m["decided_at"]),
        n_steps=result.n_steps,
        geometry_hash=pair.geometry_hash,
        design_space="box-v1" if getattr(pair, "boxes", None) is not None else "freeform-v2",
        task=task.name,
        task_digest=task_digest(task),
        executor=executor,
        wall_time_s=result.wall_time_s,
        **_condition_fields(condition),
    )


def invalid_record_v2(
    design_id: str, condition: Any, suite_name: str, seed: int, reason: str, task: TaskConfig
) -> EpisodeRecordV2:
    """The record of an episode that could not run: the design has no jaws for this part."""
    nan = float("nan")
    return EpisodeRecordV2(
        design_id=design_id,
        seed=seed,
        success=False,
        lift_success=False,
        failure_mode="invalid_build",
        registered=False,
        jam_on_close=False,
        off_centre=False,
        seat_short_mm=nan,
        reg_dx_mm=nan,
        reg_dy_mm=nan,
        reg_dyaw_deg=nan,
        reg_tilt_deg=nan,
        reg_hold_tilt_deg=nan,
        tilt_max_deg=nan,
        registration_flags=(),
        hold_creep_mm=nan,
        bad_qacc=0,
        lift=0.0,
        hold=0.0,
        hold_frac=0.0,
        flags=(f"invalid_build: {reason}"[:200],),
        suite=suite_name,
        finger_mass_g=nan,
        q_seat_mm=nan,
        tip_height=nan,
        palm_gap=nan,
        penetration_max=nan,
        obj_finger_pen_max=nan,
        decided_at=nan,
        n_steps=0,
        geometry_hash="",
        design_space="freeform-v2",
        task=task.name,
        task_digest=task_digest(task),
        executor="none",
        wall_time_s=0.0,
        **_condition_fields(condition),
    )


def _default_builder() -> Builder:
    from rtt.design.freeform import build_pair

    return build_pair


def build_pairs(
    designs: Sequence[Any], parts: Sequence[Any], builder: Builder, task: TaskConfig
) -> list[dict[str, Any]]:
    """Per design, ``{part_id: FingerPair, or the build error message}`` (in part order).

    A ``ValueError`` from the builder (the generator's ``InvalidDesign`` is one) means the
    design has no jaws for that part; any other exception is a bug and propagates.
    """
    out = []
    for design in designs:
        built: dict[str, Any] = {}
        for part in parts:
            try:
                built[part.part_id] = builder(design, part, task)
            except ValueError as exc:
                built[part.part_id] = str(exc) or type(exc).__name__
        out.append(built)
    return out


def _design_id_of(design: Any, built: Mapping[str, Any]) -> str:
    for pair in built.values():
        if not isinstance(pair, str):
            return str(pair.design_id)
    did = getattr(design, "design_id", None)
    if callable(did):
        did = did()
    if isinstance(did, str) and did:
        return did
    try:
        from rtt.design.freeform.hashing import design_id as generator_design_id

        return str(generator_design_id(design))
    except (ImportError, TypeError, ValueError):
        return hashlib.sha256(repr(design).encode()).hexdigest()


def evaluate_designs_v2(
    designs: Sequence[Any],
    suite: Any,
    *,
    builder: Builder | None = None,
    seeds: Sequence[int] | None = None,
    executor: Executor = "cpu",
    workers: int | None = None,
    task: TaskConfig | None = None,
) -> list[EpisodeRecordV2]:
    """Compile every design against every suite part and run every condition and seed.

    ``suite`` is a v2 part suite (:class:`rtt.parts.suite.PartSuite`: conditions with a
    ``part``, ``placement``, ``seeds``, ``placement_label`` and optional ``err_z``).
    ``builder`` defaults to :func:`rtt.design.freeform.build_pair`. A design that cannot be
    built for a part gets ``invalid_build`` records there, without simulating. Records come
    back ordered by design, condition (suite order), then seed; ``seeds`` replaces each
    condition's own seeds when given.
    """
    from rtt.sim.task import default_task_v2

    task = task or default_task_v2()
    builder = builder or _default_builder()
    name, run = _resolve_executor(executor)
    built = build_pairs(designs, suite_parts(suite), builder, task)
    jobs: list[EpisodeJob] = []
    keys: list[tuple[str, Any, Any, TaskConfig]] = []  # (design id, condition, pair, task)
    out_slots: list[EpisodeRecordV2 | None] = []
    for design, pairs in zip(designs, built, strict=True):
        did = _design_id_of(design, pairs)
        for cond in suite:
            pair = pairs[cond.part.part_id]
            ctask = condition_task_v2(task, cond)
            for seed in seeds if seeds is not None else cond.seeds:
                if isinstance(pair, str):
                    out_slots.append(invalid_record_v2(did, cond, suite.name, int(seed), pair, task))
                    continue
                jobs.append(EpisodeJob(pair, cond.part, cond.placement, -1, int(seed), task=ctask))
                keys.append((did, cond, pair, ctask))
                out_slots.append(None)
    batch = run(jobs, workers) if jobs else BatchResult([], 0.0, 0, 0)
    if len(batch.results) != len(jobs):
        raise RuntimeError(f"executor {name!r} returned {len(batch.results)} results for {len(jobs)} jobs")
    results = iter(zip(keys, batch.results, strict=True))
    records: list[EpisodeRecordV2] = []
    for slot in out_slots:
        if slot is not None:
            records.append(slot)
            continue
        (did, cond, pair, ctask), res = next(results)
        records.append(make_record_v2(did, cond, suite.name, res, pair, ctask, name))
    return records


def physics_invalid_share(records: Sequence[EpisodeRecordV2]) -> float:
    """Share of simulated episodes labelled ``physics_invalid`` (> 1 % is a logged deviation)."""
    simulated = [r for r in records if r.failure_mode != "invalid_build"]
    if not simulated:
        return 0.0
    return sum(r.failure_mode == "physics_invalid" for r in simulated) / len(simulated)


_ARROW_TYPES = {"bool": "bool_", "int": "int64", "float": "float64", "str": "string", "str | None": "string"}


def records_v2_to_table(records: Sequence[EpisodeRecordV2]) -> Any:
    """A ``pyarrow.Table`` with one row per v2 record (schema from the dataclass fields)."""
    import pyarrow as pa

    fields = []
    for f in dataclasses.fields(EpisodeRecordV2):
        if f.type == "tuple[str, ...]":
            fields.append(pa.field(f.name, pa.list_(pa.string())))
        else:
            fields.append(pa.field(f.name, getattr(pa, _ARROW_TYPES[str(f.type)])()))
    schema = pa.schema(fields)
    columns: dict[str, list[Any]] = {f.name: [getattr(r, f.name) for r in records] for f in fields}
    for name in ("flags", "registration_flags"):
        columns[name] = [list(v) for v in columns[name]]
    return pa.table(columns, schema=schema)


def write_parquet_v2(records: Sequence[EpisodeRecordV2], path: str | Path) -> Path:
    import pyarrow.parquet as pq

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(records_v2_to_table(records), p, compression="zstd")
    return p


@dataclass(frozen=True)
class DesignScoreV2:
    """A v2 design's training score and the breakdown a proposer is shown (spec section 7)."""

    design_id: str
    score: float  # mean registered-grasp success over every episode
    lift_score: float  # mean lift-only success (first tie-break, always reported)
    mass_g: float  # mean jaw-pair mass over the built parts (second tie-break, lower is better)
    n_episodes: int
    success_by_part: dict[str, float]
    lift_by_part: dict[str, float]
    success_by_placement: dict[str, float]
    failure_modes: dict[str, int]
    registration_flags: dict[str, int]  # jam_on_close / off_centre / tilt_after_hold / not_measured
    flags: dict[str, int]
    geometry_id: str | None  # over the per-part geometry hashes; None if a part failed to build

    def sort_key(self) -> tuple[float, float, float]:
        return (-self.score, -self.lift_score, self.mass_g if math.isfinite(self.mass_g) else math.inf)


def score_designs_v2(records: Sequence[EpisodeRecordV2]) -> dict[str, DesignScoreV2]:
    """Reduce v2 records to one :class:`DesignScoreV2` per design (first-seen order).

    Score = mean ``success`` (registered grasp) over every episode; there is no best-of-k.
    """
    by_design: dict[str, list[EpisodeRecordV2]] = defaultdict(list)
    for r in records:
        by_design[r.design_id].append(r)

    def mean(rs: Sequence[EpisodeRecordV2], attr: str) -> float:
        return sum(bool(getattr(r, attr)) for r in rs) / len(rs)

    out: dict[str, DesignScoreV2] = {}
    for did, recs in by_design.items():
        by_part: dict[str, list[EpisodeRecordV2]] = defaultdict(list)
        by_place: dict[str, list[EpisodeRecordV2]] = defaultdict(list)
        hashes: dict[str, str] = {}
        masses: dict[str, float] = {}
        for r in recs:
            by_part[r.part_id].append(r)
            by_place[r.placement_label].append(r)
            hashes.setdefault(r.part_id, r.geometry_hash)
            masses.setdefault(r.part_id, r.finger_mass_g)
        finite = [m for m in masses.values() if math.isfinite(m)]
        out[did] = DesignScoreV2(
            design_id=did,
            score=mean(recs, "success"),
            lift_score=mean(recs, "lift_success"),
            mass_g=sum(finite) / len(finite) if finite else math.nan,
            n_episodes=len(recs),
            success_by_part={k: mean(v, "success") for k, v in by_part.items()},
            lift_by_part={k: mean(v, "lift_success") for k, v in by_part.items()},
            success_by_placement={k: mean(v, "success") for k, v in by_place.items()},
            failure_modes=dict(Counter(r.failure_mode for r in recs if r.failure_mode)),
            registration_flags=dict(Counter(f for r in recs for f in r.registration_flags)),
            flags=dict(Counter(f.split(":", 1)[0] for r in recs for f in r.flags)),
            geometry_id=None if any(not h for h in hashes.values()) else suite_geometry_id(hashes),
        )
    return out


def rank_designs_v2(records: Sequence[EpisodeRecordV2]) -> list[DesignScoreV2]:
    """v2 design scores, best first (score, lift-only success, lower mass, then id)."""
    return sorted(score_designs_v2(records).values(), key=lambda s: (*s.sort_key(), s.design_id))
