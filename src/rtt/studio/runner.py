"""Full-task runner of the Studio search (track S5, product mode): screening and finalist runs.

Built on :mod:`rtt.studio.episode` (the frozen S1 core), never beside it:

* :func:`run_segments` is :func:`rtt.studio.episode.run_task` with one addition: an optional
  ``initial`` :class:`~rtt.studio.episode.WorldState` that puts the objects where an earlier run
  left them before the first segment, and the world state at the end. It calls the episode's own
  step runner, scene cache and outcome builder, so with ``initial=None`` it gives the same
  :class:`~rtt.studio.types.TaskResult` (``traj_sha`` included) as ``run_task``; a test checks it.
  Starting from a carried state is exactly the contract's tool-change model (``docs/STUDIO.md``
  section 4.4): the wrist starts at home with the new tool and the objects keep their joint
  positions. The per-step screen (:mod:`rtt.studio.search`) uses it to screen a step in the state
  the earlier steps leave behind.
* :func:`condition_grid`: the default condition grid of a comparison (a nominal condition and
  per-object placement errors, each with its own seed); every candidate runs the same list in
  the same order (common random numbers, ``docs/STUDIO.md`` section 5).
* :class:`WorkerPool`: one spawned process pool for a whole tournament, capped at
  :data:`MAX_WORKERS` processes (the host is shared); ``workers <= 1`` runs in-process. Every job
  is independent, so results do not depend on how jobs land on workers.
* :func:`run_full_tasks`: the finalists' full-task runs on the grid, through
  :func:`rtt.studio.episode.run_task` itself.

Every record carries "Product mode: not part of the pre-registered experiment".
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Final, Self, TypeVar

import mujoco
import numpy as np

from rtt.effectors.types import InvalidEffector
from rtt.sim.episode import sample_jitter
from rtt.sim.task import TaskV2, default_task_v2
from rtt.studio import episode as _ep
from rtt.studio.episode import (
    StudioCondition,
    StudioJob,
    WorldState,
    apply_world_state,
    capture_world_state,
    home_pose,
    run_task,
)
from rtt.studio.plan import ControllerSet
from rtt.studio.templates import TemplateUnavailable
from rtt.studio.types import (
    ChangeKind,
    FidelityTier,
    Setup,
    StepOutcome,
    TaskResult,
    TaskSpec,
)

__all__ = [
    "MAX_WORKERS",
    "PLACEMENT_RADIUS_MM",
    "SegmentJob",
    "WorkerPool",
    "condition_grid",
    "run_full_tasks",
    "run_segment_job",
    "run_segments",
]

#: The machine is shared: never more worker processes than this.
MAX_WORKERS: Final = 6
#: Default per-object placement error of the condition grid (mm, uniform in a square) [assumed].
PLACEMENT_RADIUS_MM: Final = 2.0

T = TypeVar("T")
R = TypeVar("R")


# --------------------------------------------------------------------------- conditions


def condition_grid(
    task_spec: TaskSpec, n: int = 8, *, radius_mm: float = PLACEMENT_RADIUS_MM, seed0: int = 0
) -> tuple[StudioCondition, ...]:
    """``n`` conditions: ``nominal`` (seed ``seed0``, no placement error), then ``place01``...
    with seed ``seed0 + i`` and, for every object in task order, an aim offset drawn uniformly
    in ``[-radius_mm, radius_mm]`` on x and y (deterministic on every platform)."""
    if not 1 <= n <= 64:
        raise ValueError("a condition grid has 1 to 64 conditions")
    if not 0.0 <= radius_mm <= 50.0:
        raise ValueError("radius_mm must be in [0, 50]")
    out = [StudioCondition("nominal", seed0)]
    for i in range(1, n):
        rng = np.random.Generator(np.random.PCG64(10_007 * (seed0 + 1) + i))
        offsets = {}
        for obj in task_spec.objects:
            u = rng.uniform(-1.0, 1.0, size=2)
            offsets[obj.name] = (round(float(u[0]) * radius_mm, 3), round(float(u[1]) * radius_mm, 3))
        out.append(StudioCondition(f"place{i:02d}", seed0 + i, offsets))
    return tuple(out)


def condition_dict(cond: StudioCondition) -> dict[str, Any]:
    return {
        "condition_id": cond.condition_id,
        "seed": int(cond.seed),
        "aim_offsets_mm": {k: [float(v[0]), float(v[1])] for k, v in sorted(cond.aim_offsets_mm.items())},
    }


def condition_from_dict(d: Mapping[str, Any]) -> StudioCondition:
    return StudioCondition(
        str(d["condition_id"]),
        int(d["seed"]),
        {k: (float(v[0]), float(v[1])) for k, v in dict(d.get("aim_offsets_mm") or {}).items()},
    )


# --------------------------------------------------------------------------- one run, carried state


def run_segments(
    setup: Setup,
    task_spec: TaskSpec,
    condition: StudioCondition,
    *,
    task: TaskV2 | None = None,
    controllers: ControllerSet = "default",
    initial: WorldState | None = None,
) -> tuple[TaskResult, WorldState | None]:
    """:func:`rtt.studio.episode.run_task`, optionally from a carried world state (module docstring).

    Returns the result and the world state after the last segment that ran (``initial`` when no
    segment could be built). The task clock starts at 0 whatever ``initial.clock_s`` says.
    """
    from rtt.studio.fairness import fairness_sha

    wall0 = time.perf_counter()
    task = task or default_task_v2()
    job = StudioJob(setup, task_spec, condition, task=task, controllers=controllers)
    spec, cond = task_spec, condition
    segments = setup.segments(spec)
    clock = 0.0
    outcomes: list[StepOutcome] = []
    seg_shas: list[str] = []
    masses: list[float] = []
    change_time = 0.0
    n_changes = {ChangeKind.TOOL_CHANGE: 0, ChangeKind.PRESENT: 0}
    state: WorldState | None = initial
    fsha = ""
    stopped = False
    jit = sample_jitter(cond.seed, task.jitter)
    for seg in segments:
        steps = [spec.step(s) for s in seg.step_ids]
        if stopped:
            outcomes += [
                _ep._outcome(s, setup, spec, failure_mode="not_reached", start_s=clock) for s in steps
            ]
            continue
        if seg.change is not ChangeKind.NONE:
            clock += seg.change_s
            change_time += seg.change_s
            n_changes[seg.change] += 1
        try:
            scene = _ep._scene(setup, spec, seg.tool, task)
        except InvalidEffector:
            outcomes += [
                _ep._outcome(s, setup, spec, failure_mode="invalid_build", start_s=clock) for s in steps
            ]
            stopped = True
            continue
        except TemplateUnavailable:
            outcomes += [
                _ep._outcome(
                    s,
                    setup,
                    spec,
                    failure_mode="not_simulated",
                    fidelity=FidelityTier.NOT_SIMULATED,
                    start_s=clock,
                )
                for s in steps
            ]
            stopped = True
            continue
        fsha = fsha or fairness_sha(scene)
        masses.append(scene.tool_mass_kg)
        model = scene.model
        data = mujoco.MjData(model)
        model.geom_friction[scene.ids.object_geoms, 0] = scene.object_friction * jit.mu_scale
        home = home_pose(scene, task, state)
        data.qpos[scene.ids.palm_qpos] = home
        data.qpos[scene.ids.roll_qpos] = 0.0
        data.qpos[scene.ids.pitch_qpos] = 0.0
        if state is not None:
            apply_world_state(scene, data, state)
        mujoco.mj_forward(model, data)
        ctrl = np.zeros(model.nu)
        ctrl[:6] = [*home, 0.0, 0.0]  # x, y, z, yaw, roll, pitch
        for h in scene.heads:
            if not h.active:
                ctrl[[model.actuator(a).id for a in h.actuators]] = h.build.ctrl_open
        ctrl[scene.ids.effector_actuators] = scene.active.build.ctrl_open
        aims = _ep._aim_offsets(scene, cond, task)
        seg_sha = hashlib.sha256()
        for step in steps:
            if stopped:
                outcomes.append(_ep._outcome(step, setup, spec, failure_mode="not_reached", start_s=clock))
                continue
            out, ctrl = _ep._run_step(scene, data, step, ctrl, aims, job, task, clock, seg_sha)
            outcomes.append(out)
            clock += out.duration_s
            stopped = not out.success
        seg_shas.append(seg_sha.hexdigest())
        state = capture_world_state(scene, data, clock)
    fsha = fsha or "unbuilt"
    traj = hashlib.sha256("|".join(seg_shas).encode()).hexdigest()
    result = TaskResult(
        task_id=spec.task_id,
        setup_id=setup.setup_id,
        setup_kind=setup.kind,
        condition_id=cond.condition_id,
        seed=cond.seed,
        steps=tuple(outcomes),
        cycle_time_s=clock,
        change_time_s=change_time,
        n_tool_changes=n_changes[ChangeKind.TOOL_CHANGE],
        n_presentations=n_changes[ChangeKind.PRESENT],
        tool_mass_kg=tuple(masses),
        traj_sha=traj,
        fairness_sha=fsha,
        controllers=controllers,
        wall_time_s=time.perf_counter() - wall0,
    )
    return result, state


@dataclass(frozen=True)
class SegmentJob:
    """One screening run (picklable): a setup on a (sub-)task from a carried state."""

    setup: Setup
    task_spec: TaskSpec
    condition: StudioCondition
    initial: WorldState | None = None
    task: TaskV2 | None = None
    controllers: ControllerSet = "default"
    tag: tuple[Any, ...] = field(default=())


def run_segment_job(job: SegmentJob) -> tuple[TaskResult, WorldState | None]:
    """Worker entry point of :class:`SegmentJob` (top level, so a spawned pool can import it)."""
    return run_segments(
        job.setup,
        job.task_spec,
        job.condition,
        task=job.task,
        controllers=job.controllers,
        initial=job.initial,
    )


# --------------------------------------------------------------------------- the pool


class WorkerPool:
    """A spawned process pool shared by every phase of one tournament (module docstring).

    ``workers`` is capped at :data:`MAX_WORKERS`; ``workers <= 1`` runs in-process (tests use it,
    and it is what an override registered in this process needs). The pool starts on first use.
    """

    def __init__(self, workers: int | None = 1) -> None:
        if workers is None:
            workers = MAX_WORKERS
        self.workers = max(1, min(int(workers), MAX_WORKERS))
        self._pool: Any = None
        self.jobs_run = 0
        self.busy_s = 0.0

    def map(self, fn: Callable[[T], R], items: Sequence[T], chunksize: int = 1) -> list[R]:
        items = list(items)
        t0 = time.perf_counter()
        if self.workers <= 1 or len(items) <= 1:
            out = [fn(i) for i in items]
        else:
            if self._pool is None:
                self._pool = mp.get_context("spawn").Pool(processes=self.workers)
            out = self._pool.map(fn, items, chunksize=max(1, chunksize))
        self.jobs_run += len(items)
        self.busy_s += time.perf_counter() - t0
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if exc_type is not None and self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        self.close()


# --------------------------------------------------------------------------- finalists


def run_full_tasks(
    setups: Iterable[Setup],
    task_spec: TaskSpec,
    conditions: Sequence[StudioCondition],
    *,
    pool: WorkerPool,
    task: TaskV2 | None = None,
    controllers: ControllerSet = "default",
) -> dict[str, list[TaskResult]]:
    """Every setup on every condition, in condition order (common random numbers), through
    :func:`rtt.studio.episode.run_task`; returns setup id -> its runs."""
    setups = list(setups)
    jobs = [
        StudioJob(s, task_spec, c, task=task, controllers=controllers) for s in setups for c in conditions
    ]
    results = pool.map(run_task, jobs, chunksize=max(1, len(conditions) // 2))
    out: dict[str, list[TaskResult]] = {s.setup_id: [] for s in setups}
    for job, res in zip(jobs, results, strict=True):
        out[job.setup.setup_id].append(res)
    return out


def result_digest(results: Iterable[TaskResult]) -> str:
    """sha256 over the runs' trajectory hashes, in order (a fingerprint of a set of runs)."""
    h = hashlib.sha256()
    for r in results:
        h.update(f"{r.setup_id}:{r.condition_id}:{r.seed}:{r.traj_sha}|".encode())
    return h.hexdigest()
