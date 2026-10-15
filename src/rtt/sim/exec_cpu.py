"""CPU executor: run many independent episodes across processes.

Each episode is a separate single-threaded MuJoCo simulation, so episodes are
farmed out to a ``spawn`` process pool (no fork: safe with GL contexts and
threads). Results come back in job order and are bit-identical to running
:func:`~rtt.sim.episode.run_episode` serially, because every episode is fully
determined by its job.

:func:`run_jobs` starts a pool per call. A search loop that evaluates a batch per
round can keep one :class:`CpuExecutor` open instead, so the ~1 s pool start-up
(spawning workers and importing MuJoCo) is paid once.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.pool import Pool
from types import TracebackType
from typing import Self

from rtt.design.types import FingerGeometry
from rtt.sim.episode import EpisodeResult, run_episode
from rtt.sim.objects import AnyObject, Placement
from rtt.sim.perturb import Perturbation
from rtt.sim.task import TaskConfig


@dataclass(frozen=True)
class EpisodeJob:
    """Everything needed to reproduce one episode."""

    finger: FingerGeometry
    obj: AnyObject
    placement: Placement
    controller_setting: int
    seed: int
    record: bool = False
    task: TaskConfig | None = None
    perturbation: Perturbation | None = None


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
        perturbation=job.perturbation,
    )


def default_workers() -> int:
    """All logical CPUs (``os.cpu_count()``); 1 if unknown."""
    return os.cpu_count() or 1


class CpuExecutor:
    """A reusable pool of ``workers`` spawned processes (``workers == 1``: run in-process).

    Use as a context manager, or call :meth:`close` when done::

        with CpuExecutor(16) as ex:
            batch = ex.run(jobs)
    """

    def __init__(self, workers: int | None = None) -> None:
        self.workers = max(1, workers if workers is not None else default_workers())
        self._pool: Pool | None = None

    def _ensure_pool(self) -> Pool:
        if self._pool is None:
            self._pool = mp.get_context("spawn").Pool(processes=self.workers)
        return self._pool

    def run(self, jobs: Sequence[EpisodeJob]) -> BatchResult:
        """Run ``jobs`` and return their results in job order."""
        t0 = time.perf_counter()
        if self.workers == 1 or len(jobs) <= 1:
            results = [run_job(j) for j in jobs]
            used = 1
        else:
            chunk = max(1, len(jobs) // (self.workers * 8))
            results = self._ensure_pool().map(run_job, jobs, chunksize=chunk)
            used = min(self.workers, len(jobs))
        return BatchResult(
            results=list(results),
            wall_time_s=time.perf_counter() - t0,
            total_steps=sum(r.n_steps for r in results),
            workers=used,
        )

    def close(self) -> None:
        """Shut the worker processes down (idempotent)."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None and self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        self.close()


def run_jobs(jobs: Sequence[EpisodeJob], workers: int | None = None) -> BatchResult:
    """Run ``jobs`` on up to ``workers`` processes (default: all CPUs; 1 = in-process)."""
    workers = max(1, min(workers if workers is not None else default_workers(), len(jobs) or 1))
    with CpuExecutor(workers) as ex:
        return ex.run(jobs)
