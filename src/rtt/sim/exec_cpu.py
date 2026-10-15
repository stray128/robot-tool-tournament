"""CPU executor: run many independent episodes across processes.

Each episode is a separate single-threaded MuJoCo simulation, so episodes are
farmed out to a ``spawn`` process pool (no fork: safe with GL contexts and
threads). Results come back in job order and are bit-identical to running
:func:`~rtt.sim.episode.run_episode` serially, because every episode is fully
determined by its job.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

from rtt.design.types import FingerGeometry
from rtt.sim.episode import EpisodeResult, run_episode
from rtt.sim.objects import ObjectSpec, Placement
from rtt.sim.task import TaskConfig


@dataclass(frozen=True)
class EpisodeJob:
    """Everything needed to reproduce one episode."""

    finger: FingerGeometry
    obj: ObjectSpec
    placement: Placement
    controller_setting: int
    seed: int
    record: bool = False
    task: TaskConfig | None = None


@dataclass
class BatchResult:
    """Ordered episode results plus throughput accounting."""

    results: list[EpisodeResult]
    wall_time_s: float
    total_steps: int
    workers: int

    @property
    def steps_per_sec(self) -> float:
        """Aggregate physics steps per wall-clock second across all workers."""
        return self.total_steps / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def episodes_per_sec(self) -> float:
        return len(self.results) / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def steps_per_sec_per_worker(self) -> float:
        """Mean in-episode physics speed of a single worker (excludes pool overhead)."""
        busy = sum(r.wall_time_s for r in self.results)
        return self.total_steps / busy if busy > 0 else 0.0


def run_job(job: EpisodeJob) -> EpisodeResult:
    """Run one job (module-level so spawned workers can import it)."""
    return run_episode(
        job.finger,
        job.obj,
        job.placement,
        job.controller_setting,
        job.seed,
        record=job.record,
        task=job.task,
    )


def run_jobs(jobs: Sequence[EpisodeJob], workers: int | None = None) -> BatchResult:
    """Run ``jobs`` on up to ``workers`` processes (default: all CPUs; 1 = in-process)."""
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, len(jobs) or 1))
    t0 = time.perf_counter()
    if workers == 1:
        results = [run_job(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(run_job, jobs, chunksize=max(1, len(jobs) // (workers * 4)))
    wall = time.perf_counter() - t0
    return BatchResult(
        results=list(results),
        wall_time_s=wall,
        total_steps=sum(r.n_steps for r in results),
        workers=workers,
    )
