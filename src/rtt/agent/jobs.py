"""Simulation jobs the Grasp Engineer launches: local threads or Nebius Serverless AI Jobs.

Both runners share one contract (:class:`JobRunner`): ``launch`` returns at once with a
:class:`JobRecord`, ``wait`` blocks up to a timeout, ``results`` returns one
:class:`~rtt.search.base.DesignScore` per design, ``cancel`` stops a job.

* :class:`LocalJobRunner` evaluates designs on this machine (CPU MuJoCo, the executor of
  record) in a small thread pool. It costs nothing.
* :class:`NebiusJobRunner` submits a job through :class:`rtt.cloud.nebius_jobs.NebiusJobs`
  (the ``nebius`` CLI). The part and the designs are uploaded to the results bucket
  first; inside the container :mod:`rtt.agent.remote` runs the same backend and writes
  ``results.json``, which the runner reads back from the bucket. Jobs run on the CPU
  target by default (``cpu-d3`` in uk-south2, the executor of record, with a small
  preset), never on a GPU unless asked.

**Spend bound.** A Nebius job's timeout is whole hours (at least ``1h``), so its true
worst case is the timeout plus provisioning, not a typical wall time. The runner
reserves ``usd_per_hour x max_wall_s`` before submission, where ``max_wall_s`` is the
job timeout plus :data:`PROVISIONING_ALLOWANCE_S`, and cancels any job still running
when that much time has passed since it was created. The wall-based estimate (created
to finished) is charged when the job ends, so it can never exceed the reservation by
more than one poll interval. Job IDs are public provenance by design.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from rtt.agent.recorder import redact_paths
from rtt.design.schema import FingerDesign
from rtt.search.base import DesignScore

__all__ = [
    "AGENT_CPU_VCPUS",
    "JOB_TIMEOUT",
    "PROVISIONING_ALLOWANCE_S",
    "TERMINAL_STATES",
    "JobRecord",
    "JobRunner",
    "LocalJobRunner",
    "NebiusJobRunner",
    "UnknownJobError",
    "job_timeout_s",
    "part_job_factory",
]

JobState = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
Target = Literal["local", "nebius"]
TERMINAL_STATES: Final = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
REMOTE_MODULE: Final = "rtt.agent.remote"
#: Timeout of an agent job on Nebius (the platform accepts whole hours only, from 1h).
JOB_TIMEOUT: Final = "1h"
#: Wall time allowed for provisioning on top of the job timeout, in the spend reservation.
PROVISIONING_ALLOWANCE_S: Final = 900.0
#: vCPUs of the CPU preset agent jobs use by default (``4vcpu-16gb``: two jobs fit the
#: default $0.50 session cap even at their worst case).
AGENT_CPU_VCPUS: Final = 4
_HOURS: Final = re.compile(r"^(\d+)h$")


def job_timeout_s(timeout: str) -> float:
    """Seconds in a Nebius job timeout such as ``"1h"``."""
    match = _HOURS.match(timeout)
    if not match or int(match.group(1)) < 1:
        raise ValueError(f"timeout must be whole hours such as '1h', got {timeout!r}")
    return 3600.0 * int(match.group(1))


class UnknownJobError(KeyError):
    """No job with this ID was launched by this runner."""


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class JobRecord:
    """A launched job as the agent and the UI see it."""

    job_id: str
    target: Target
    part_id: str
    design_ids: tuple[str, ...]
    state: JobState
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    est_usd: float = 0.0
    """Worst case while the job is in flight; the wall-based estimate once it ended."""
    error: str | None = None
    platform: str | None = None
    preset: str | None = None
    run_name: str | None = None
    """Bucket prefix name of a Nebius job's inputs and outputs (``runs/<run_name>/``)."""

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def elapsed_s(self, now: float | None = None) -> float:
        end = self.finished_at if self.finished_at is not None else (now if now is not None else time.time())
        return max(0.0, end - self.created_at)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_id": self.job_id,
            "target": self.target,
            "part_id": self.part_id,
            "designs": list(self.design_ids),
            "state": self.state,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "elapsed_s": round(self.elapsed_s(), 1),
            "est_usd": round(self.est_usd, 4),
        }
        if self.error:
            data["error"] = self.error
        if self.platform:
            data["platform"] = self.platform
            data["preset"] = self.preset
        return data


class JobRunner(Protocol):
    """What :class:`rtt.agent.tools.Toolbox` needs from a job backend."""

    target: Target

    def estimate_usd(self, n_designs: int) -> float:
        """Worst-case cost of one job (reserved before launch)."""
        ...

    def launch(self, part_id: str, designs: Sequence[FingerDesign]) -> JobRecord: ...

    def status(self, job_id: str) -> JobRecord: ...

    def wait(self, job_id: str, timeout_s: float) -> JobRecord: ...

    def results(self, job_id: str) -> list[DesignScore]: ...

    def cancel(self, job_id: str) -> JobRecord: ...


# --------------------------------------------------------------------------- local


def _new_local_id() -> str:
    return f"local-{uuid.uuid4().hex[:12]}"


@dataclass
class _LocalJob:
    record: JobRecord
    future: Future[list[DesignScore]] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class LocalJobRunner:
    """Runs ``evaluate(part_id, designs)`` in a thread pool on this machine (no cost).

    Args:
        evaluate: Usually :meth:`rtt.agent.backends.PartBackend.evaluate`.
        max_workers: Jobs that may run at once (others queue).
        clock: Wall clock in seconds since the epoch.
    """

    target: Target = "local"

    def __init__(
        self,
        evaluate: Callable[[str, Sequence[FingerDesign]], list[DesignScore]],
        *,
        max_workers: int = 2,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._evaluate = evaluate
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rtt-agent-job")
        self._clock = clock
        self._jobs: dict[str, _LocalJob] = {}
        self._lock = threading.Lock()

    def estimate_usd(self, n_designs: int) -> float:
        return 0.0

    def launch(self, part_id: str, designs: Sequence[FingerDesign]) -> JobRecord:
        job_id = _new_local_id()
        record = JobRecord(
            job_id=job_id,
            target="local",
            part_id=part_id,
            design_ids=tuple(d.design_id() for d in designs),
            state="QUEUED",
            created_at=self._clock(),
        )
        job = _LocalJob(record)
        with self._lock:
            self._jobs[job_id] = job
        designs = list(designs)

        def run() -> list[DesignScore]:
            with job.lock:
                job.record = replace(job.record, state="RUNNING", started_at=self._clock())
            return self._evaluate(part_id, designs)

        job.future = self._pool.submit(run)
        job.future.add_done_callback(lambda fut: self._finish(job, fut))
        return job.record

    def _finish(self, job: _LocalJob, fut: Future[list[DesignScore]]) -> None:
        with job.lock:
            if fut.cancelled():
                job.record = replace(job.record, state="CANCELLED", finished_at=self._clock())
                return
            exc = fut.exception()
            if exc is not None:
                job.record = replace(
                    job.record,
                    state="FAILED",
                    finished_at=self._clock(),
                    error=redact_paths(f"{type(exc).__name__}: {exc}")[:400],
                )
            else:
                job.record = replace(job.record, state="COMPLETED", finished_at=self._clock())

    def _job(self, job_id: str) -> _LocalJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise UnknownJobError(job_id)
        return job

    def status(self, job_id: str) -> JobRecord:
        job = self._job(job_id)
        with job.lock:
            return job.record

    def wait(self, job_id: str, timeout_s: float) -> JobRecord:
        job = self._job(job_id)
        if job.future is not None:
            wait_futures([job.future], timeout=max(0.0, timeout_s))
            # The done-callback may still be running; give it a moment to publish the state.
            deadline = time.monotonic() + 1.0
            while job.future.done() and not self.status(job_id).terminal and time.monotonic() < deadline:
                time.sleep(0.005)
        return self.status(job_id)

    def results(self, job_id: str) -> list[DesignScore]:
        job = self._job(job_id)
        record = self.status(job_id)
        if record.state != "COMPLETED" or job.future is None:
            raise RuntimeError(f"job {job_id} is {record.state}, not COMPLETED")
        return job.future.result()

    def cancel(self, job_id: str) -> JobRecord:
        job = self._job(job_id)
        if job.future is not None and job.future.cancel():
            with job.lock:
                job.record = replace(job.record, state="CANCELLED", finished_at=self._clock())
        return self.status(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------- nebius


_NEBIUS_STATE: Final[dict[str, JobState]] = {
    "COMPLETED": "COMPLETED",
    "FAILED": "FAILED",
    "ERROR": "FAILED",
    "CANCELLED": "CANCELLED",
    "DELETING": "CANCELLED",
    "RUNNING": "RUNNING",
}


def _map_state(state: str) -> JobState:
    return _NEBIUS_STATE.get(state.upper(), "QUEUED")


SpecFactory = Callable[[str, str, Sequence[FingerDesign]], Any]
"""``(run_name, part_id, designs) -> JobSpec``: stage inputs and describe the job."""
ResultsFetcher = Callable[[str], list[DesignScore]]
"""``run_name -> scores``: read ``runs/<run_name>/results.json`` back."""


class NebiusJobRunner:
    """Part evaluations as Nebius Serverless AI Jobs (see the module docstring).

    Args:
        jobs: A configured :class:`rtt.cloud.nebius_jobs.NebiusJobs`.
        spec_factory: Stages inputs and returns the ``JobSpec`` (:func:`part_job_factory`).
        fetch_results: Reads a finished run's scores.
        usd_per_hour: On-demand price of the job's platform and preset.
        max_wall_s: Worst-case wall time of one job, from creation to its end: the job
            timeout plus provisioning. It is what a launch reserves, and a job still
            running after it is cancelled.
        poll_s: Poll interval while waiting.
    """

    target: Target = "nebius"

    def __init__(
        self,
        jobs: Any,
        *,
        spec_factory: SpecFactory,
        fetch_results: ResultsFetcher,
        usd_per_hour: float,
        max_wall_s: float = job_timeout_s(JOB_TIMEOUT) + PROVISIONING_ALLOWANCE_S,
        poll_s: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # ``not (x >= 0)`` also refuses NaN, which would make every spend comparison false
        if not (math.isfinite(usd_per_hour) and usd_per_hour >= 0):
            raise ValueError("usd_per_hour must be a finite number >= 0")
        if not (math.isfinite(max_wall_s) and max_wall_s > 0):
            raise ValueError("max_wall_s must be a finite number > 0")
        self.jobs = jobs
        self.spec_factory = spec_factory
        self.fetch_results = fetch_results
        self.usd_per_hour = usd_per_hour
        self.max_wall_s = max_wall_s
        self.poll_s = poll_s
        self._sleep = sleep
        self._clock = clock
        self._records: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def estimate_usd(self, n_designs: int) -> float:
        """The worst case of one job: ``usd_per_hour`` for ``max_wall_s``."""
        return self.usd_per_hour * self.max_wall_s / 3600.0

    def _wall_usd(self, record: JobRecord) -> float:
        return self.usd_per_hour * record.elapsed_s(self._clock()) / 3600.0

    def launch(self, part_id: str, designs: Sequence[FingerDesign]) -> JobRecord:
        run_name = f"agent-{uuid.uuid4().hex[:12]}"
        spec = self.spec_factory(run_name, part_id, designs)
        job = self.jobs.submit(spec)
        created = job.created_at.timestamp() if job.created_at else self._clock()
        record = JobRecord(
            job_id=job.id,
            target="nebius",
            part_id=part_id,
            design_ids=tuple(d.design_id() for d in designs),
            state=_map_state(job.state),
            created_at=created,
            est_usd=self.estimate_usd(len(designs)),
            platform=getattr(spec, "platform", None),
            preset=getattr(spec, "preset", None),
            run_name=run_name,
        )
        with self._lock:
            self._records[job.id] = record
        return record

    def _record(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise UnknownJobError(job_id)
        return record

    def status(self, job_id: str) -> JobRecord:
        """The job's state; a job past ``max_wall_s`` is cancelled here (the spend bound)."""
        record = self._record(job_id)
        if record.terminal:
            return record
        job = self.jobs.get(job_id)
        state = _map_state(job.state)
        started = job.started_at.timestamp() if job.started_at else record.started_at
        finished = job.finished_at.timestamp() if job.finished_at else None
        if state in TERMINAL_STATES and finished is None:
            finished = self._clock()
        updated = replace(
            record,
            state=state,
            started_at=started,
            finished_at=finished,
            error=job.state_details if state == "FAILED" else record.error,
        )
        if not updated.terminal and updated.elapsed_s(self._clock()) >= self.max_wall_s:
            with self._lock:
                self._records[job_id] = updated
            cancelled = self.cancel(job_id)
            return self._remember(
                replace(cancelled, error=f"cancelled after {self.max_wall_s:.0f} s, the reserved wall time")
            )
        if updated.terminal:
            updated = replace(updated, est_usd=self._wall_usd(updated))
        return self._remember(updated)

    def _remember(self, record: JobRecord) -> JobRecord:
        with self._lock:
            self._records[record.job_id] = record
        return record

    def wait(self, job_id: str, timeout_s: float) -> JobRecord:
        deadline = self._clock() + max(0.0, timeout_s)
        record = self.status(job_id)
        while not record.terminal and self._clock() < deadline:
            self._sleep(min(self.poll_s, max(0.0, deadline - self._clock())))
            record = self.status(job_id)
        return record

    def results(self, job_id: str) -> list[DesignScore]:
        record = self.status(job_id)
        if record.state != "COMPLETED" or record.run_name is None:
            raise RuntimeError(f"job {job_id} is {record.state}, not COMPLETED")
        return self.fetch_results(record.run_name)

    def cancel(self, job_id: str) -> JobRecord:
        record = self._record(job_id)
        if not record.terminal:
            self.jobs.cancel(job_id)
            record = replace(record, state="CANCELLED", finished_at=self._clock())
            record = self._remember(replace(record, est_usd=self._wall_usd(record)))
        return record

    @classmethod
    def from_config(
        cls,
        part_root: str | Path,
        *,
        config: Any | None = None,
        target: str | None = None,
        vcpus: int = AGENT_CPU_VCPUS,
        platform: str | None = None,
        preset: str | None = None,
        usd_per_hour: float | None = None,
        timeout: str = JOB_TIMEOUT,
        provisioning_s: float = PROVISIONING_ALLOWANCE_S,
    ) -> NebiusJobRunner:
        """Runner for the account in ``~/.config/rtt/`` (IDs never enter the repository).

        By default jobs go to the CPU target of :mod:`rtt.cloud.nebius_jobs`
        (``cpu-d3`` in uk-south2, the executor of record) with a ``vcpus`` preset, using
        that region's project, subnet, bucket and image
        (:meth:`rtt.cloud.config.NebiusConfig.for_region`). ``target`` names another
        target; ``platform``/``preset`` pin one platform. ``usd_per_hour`` defaults to
        the price table in :mod:`rtt.cloud.cost` (a platform without a listed price
        needs it passed explicitly, because the spend cap depends on it). The
        reservation per job covers ``timeout`` plus ``provisioning_s``.
        """
        from rtt.cloud.config import load_config
        from rtt.cloud.cost import hourly_usd
        from rtt.cloud.nebius_jobs import DEFAULT_TARGET, NebiusJobs, get_target, target_for_platform
        from rtt.cloud.storage import BucketStore

        home = config or load_config()
        if platform is not None:
            region_target = target_for_platform(platform)
        else:
            region_target = get_target(target or DEFAULT_TARGET, vcpus)
            platform = region_target.candidates[0].platform
            preset = preset or region_target.candidates[0].preset
        cfg = home.for_region(region_target.region)
        cfg.require("image", "bucket_id", "subnet_id", "project_id")
        if usd_per_hour is None:
            try:
                usd_per_hour = hourly_usd(platform, preset)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"no price for {platform}/{preset}; pass usd_per_hour") from exc
        store = BucketStore.from_config(cfg)
        return cls(
            NebiusJobs.from_config(cfg),
            spec_factory=part_job_factory(
                cfg, store, Path(part_root), platform=platform, preset=preset, timeout=timeout
            ),
            fetch_results=bucket_results_fetcher(store),
            usd_per_hour=usd_per_hour,
            max_wall_s=job_timeout_s(timeout) + provisioning_s,
        )


def part_job_factory(
    config: Any,
    store: Any,
    part_root: Path,
    *,
    platform: str,
    preset: str | None,
    timeout: str = JOB_TIMEOUT,
) -> SpecFactory:
    """Spec factory: upload the part and designs, then run :mod:`rtt.agent.remote` in the image.

    Inputs go to ``runs/<run_name>/inputs/`` (``part.stl``, ``part.json`` with the intake
    options, ``designs.json``); the container reads them from the bucket mount and writes
    ``results.json`` and ``manifest.json`` next to them.
    """
    from rtt.cloud.nebius_jobs import JobSpec
    from rtt.cloud.runs import CONTAINER_PYTHON, LAUNCHER, MOUNT, run_prefix, split_image_ref

    def factory(run_name: str, part_id: str, designs: Sequence[FingerDesign]) -> Any:
        config.require("image", "bucket_id", "subnet_id", "project_id")
        part_dir = Path(part_root) / part_id
        meta = json.loads((part_dir / "part.json").read_text(encoding="utf-8"))
        prefix = run_prefix(run_name)
        store.upload_file(part_dir / "source.stl", f"{prefix}/inputs/part.stl")
        store.put_json(f"{prefix}/inputs/part.json", {"part_id": part_id, **dict(meta.get("intake") or {})})
        store.put_json(
            f"{prefix}/inputs/designs.json", [{"design_id": d.design_id(), **d.model_dump()} for d in designs]
        )
        image_name, digest = split_image_ref(config.image)
        name = f"rtt-{run_name}"
        return JobSpec(
            name=name,
            image=config.image,
            command=CONTAINER_PYTHON,
            args=["-m", REMOTE_MODULE, "--run-dir", f"{MOUNT}/{prefix}"],
            env={
                "RTT_RUN_ID": run_name,
                "RTT_JOB_NAME": name,
                "RTT_IMAGE_NAME": image_name,
                "RTT_IMAGE_DIGEST": digest,
                "RTT_LAUNCHER": LAUNCHER,
                "RTT_PLATFORM": platform,
                "RTT_PRESET": preset or "",
                "RTT_REGION": config.region,
            },
            env_secrets={},
            volumes=[f"{config.bucket_id}:{MOUNT}:rw"],
            platform=platform,
            preset=preset,
            timeout=timeout,
            restart_policy="never",
            subnet_id=config.subnet_id,
            parent_id=config.project_id,
        )

    return factory


def bucket_results_fetcher(store: Any) -> ResultsFetcher:
    """Read ``runs/<run_name>/results.json`` (written by :mod:`rtt.agent.remote`)."""
    from rtt.cloud.runs import run_prefix

    def fetch(run_name: str) -> list[DesignScore]:
        data = store.get_json(f"{run_prefix(run_name)}/results.json")
        if not isinstance(data, dict) or data.get("status") != "complete":
            raise RuntimeError(f"run {run_name} has no complete results.json")
        return [DesignScore.from_json(item) for item in data["scores"]]

    return fetch
