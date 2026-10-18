"""Nebius fan-out of the Studio search (track S6, product mode).

**Product mode, not part of the pre-registered experiment.** Every record written here carries
:data:`~rtt.effectors.types.PRODUCT_MODE_LABEL`.

S5's search (:func:`rtt.studio.search.run_task_tournament`) does all of its physics through one
pool's ``map``: the per-chain screen (:func:`rtt.studio.runner.run_segment_job` over
:class:`~rtt.studio.runner.SegmentJob`) and the finalists' full-task runs
(:func:`rtt.studio.episode.run_task` over :class:`~rtt.studio.episode.StudioJob`). :class:`CloudPool`
stands in for :class:`rtt.studio.runner.WorkerPool` and sends each ``map`` to an executor as jobs,
so the search itself runs unchanged (``run_task_tournament(task, pool=CloudPool(...))``):

* **Phase**: one ``map`` call (one hold chain of the screen, or the finalists' runs). Phases run in
  order, because chain *j* of the screen starts from the state the best tool of chain *j - 1* left.
* **Shard**: one candidate on one hold chain (its screen conditions), or one setup under one
  condition (a finalist or a held-out run). Shards are independent; each holds the exact work items
  S5 built, pickled.
* **Job**: one or more shards of a phase, run by ``python -m rtt.studio.cloud shards`` on its own
  process pool. Inside a Nebius job the phase folder is on the bucket mount
  (``/out/runs/<run_id>/<phase>/``); the job writes ``jobs/<job>/shards/<shard>.{pkl,json}`` and its
  ``manifest.json`` last, through the staging and publishing of :mod:`rtt.cloud.entry`.
* **Executors**: :class:`InProcessExecutor` (this process, no job), :class:`SubprocessExecutor`
  (the local "fake cloud": the job entry point in subprocesses against a local folder that stands
  for the bucket) and :class:`NebiusExecutor` (Serverless Jobs on the existing :mod:`rtt.cloud`
  machinery: ``uks2-cpu`` ``cpu-d3`` first, ``eun1-gpu`` H100 as the fallback, the spend ledger and
  its cap, the inputs uploaded to and the outputs fetched from the region's bucket).

**Merge.** Each item's result goes back to its index in the ``map`` call, after the result file's
SHA-256 is checked against the job manifest and each result's task, setup and condition are checked
against its item. Items run exactly as in a local pool (same functions, same objects), so a cloud
run gives the local run's ``traj_sha`` for every run (``tests/test_studio_cloud.py``).

**Records.** Every run is a row (``rows.jsonl``): its :meth:`~rtt.studio.types.TaskResult.to_record`
plus ``cloud``: the job id, which keys that job's :class:`~rtt.cloud.cost.JobSpend` line in the spend
ledger (the existing ledger format), and ``est_usd``, the run's share of that line's wall-based
estimate (split by the runs' own wall times).

**Trust boundary.** Work items and results travel as pickles (:mod:`multiprocessing`'s pickler, so
a part-shaped jaw restores its build path as in S5's spawned pool). They are written and read only
by this module, in the project's private bucket or a local folder. Each file's SHA-256 is checked
before loading, and :func:`safe_loads` resolves only the Studio, effector and simulator classes (plus
the part-jaw restore hook), never an arbitrary global.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import re
import resource
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Protocol, Self

from rtt.cloud.cost import JobSpend, SpendLedger, estimate_usd, hourly_usd
from rtt.effectors.types import PRODUCT_MODE_LABEL

__all__ = [
    "FANOUT_FORMAT",
    "WORK",
    "CloudPool",
    "Executor",
    "InProcessExecutor",
    "JobPlan",
    "JobRecord",
    "NebiusExecutor",
    "Shard",
    "SubprocessExecutor",
    "dumps",
    "pack_jobs",
    "plan_shards",
    "run_shards",
    "safe_loads",
    "studio_job_spec",
]

FANOUT_FORMAT: Final = "rtt-studio-fanout/1"
#: The functions a job may run, by name (S5's pool entry points).
WORK: Final[Mapping[str, str]] = {
    "run_segment_job": "rtt.studio.runner:run_segment_job",
    "run_task": "rtt.studio.episode:run_task",
}
PICKLE_PROTOCOL: Final = 5
JOB_KIND: Final = "studio-shards"
#: The container's interpreter and the bucket mount (:mod:`rtt.cloud.runs`).
CONTAINER_PYTHON: Final = "/app/.venv/bin/python"
MOUNT: Final = "/out"
LAUNCHER: Final = "nebius-serverless-jobs"
LOCAL_LAUNCHER: Final = "local-fake-cloud"
#: Most local job subprocesses at once (the host is shared).
MAX_LOCAL_PARALLEL: Final = 4

_JOB_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]{0,56}[a-z0-9])?$")  # 58 at most: retries append "-rN"


# --------------------------------------------------------------------------- pickles


def dumps(obj: Any) -> bytes:
    """Pickle with :mod:`multiprocessing`'s pickler (its reducers: a part-shaped jaw spec restores
    through :mod:`rtt.effectors.part_jaw`, exactly as in S5's spawned pool)."""
    import rtt.effectors.part_jaw  # noqa: F401 - registers the jaw spec's reducer

    buf = io.BytesIO()
    ForkingPickler(buf, PICKLE_PROTOCOL).dump(obj)
    return buf.getvalue()


_SAFE_MODULES: Final = ("rtt.studio.", "rtt.effectors.", "rtt.sim.")
_SAFE_FUNCTIONS: Final = frozenset(
    {
        ("rtt.effectors.part_jaw", "_restore_spec"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }
)
_SAFE_TYPES: Final = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        *(
            ("builtins", n)
            for n in ("tuple", "list", "dict", "set", "frozenset", "float", "int", "complex", "bool", "str")
        ),
    }
)


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _SAFE_FUNCTIONS or (module, name) in _SAFE_TYPES:
            return super().find_class(module, name)
        if module.startswith(_SAFE_MODULES):
            obj = super().find_class(module, name)
            if isinstance(obj, type):
                return obj
        raise pickle.UnpicklingError(f"refused global {module}.{name}: not a Studio work item or result")


def safe_loads(data: bytes) -> Any:
    """Unpickle a work item or result: only Studio, effector and simulator classes resolve."""
    return _SafeUnpickler(io.BytesIO(data)).load()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- work, shards, jobs


def work_name(fn: Callable[..., Any]) -> str:
    """The :data:`WORK` name of ``fn`` (S5's pool entry points only)."""
    for name in WORK:
        if work_fn(name) is fn:
            return name
    raise ValueError(f"{getattr(fn, '__qualname__', fn)!r} is not a Studio work function: {sorted(WORK)}")


def work_fn(name: str) -> Callable[..., Any]:
    if name not in WORK:
        raise ValueError(f"unknown work function {name!r}; known: {sorted(WORK)}")
    import importlib

    module, attr = WORK[name].split(":")
    return getattr(importlib.import_module(module), attr)


def _item_parts(item: Any) -> tuple[Any, Any, Any]:
    """``(setup, task_spec, condition)`` of a work item."""
    return item.setup, item.task_spec, item.condition


def shard_key(item: Any) -> tuple[str, ...]:
    """What groups items into one shard: a candidate on one hold chain (S5's screen tags each
    :class:`~rtt.studio.runner.SegmentJob` with ``(effector_id, chain, condition_id)``), else one
    setup under one condition."""
    from rtt.studio.runner import SegmentJob

    setup, _, cond = _item_parts(item)
    if isinstance(item, SegmentJob) and len(item.tag) >= 2:
        return ("screen", str(item.tag[0]), f"chain{int(item.tag[1])}")
    return ("run", setup.setup_id, cond.condition_id)


def describe_item(item: Any) -> dict[str, Any]:
    """A work item as JSON (for people reading a phase folder; the pickle is what runs)."""
    from rtt.studio.runner import SegmentJob

    setup, spec, cond = _item_parts(item)
    out = {
        "task_id": spec.task_id,
        "setup_id": setup.setup_id,
        "setup_kind": setup.kind.value,
        "condition_id": cond.condition_id,
        "seed": int(cond.seed),
        "controllers": item.controllers,
    }
    if isinstance(item, SegmentJob):
        out["tag"] = [str(t) for t in item.tag]
        out["carried_state"] = item.initial is not None
    return out


@dataclass(frozen=True)
class Shard:
    """Items of one phase that belong together (:func:`shard_key`), by index into the ``map`` call."""

    shard_id: str
    key: tuple[str, ...]
    indices: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.indices)


def plan_shards(items: Sequence[Any]) -> list[Shard]:
    """One shard per :func:`shard_key`, in order of first appearance; indices keep the item order."""
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, item in enumerate(items):
        groups.setdefault(shard_key(item), []).append(i)
    return [Shard(f"s{k:04d}", key, tuple(idx)) for k, (key, idx) in enumerate(groups.items())]


@dataclass(frozen=True)
class JobPlan:
    """One job of a phase: its folder name under ``jobs/``, its Nebius job name, its shards and
    worker processes (``0``: one per CPU of the machine it lands on)."""

    dir: str
    name: str
    shards: tuple[str, ...]
    workers: int = 0


def job_name(run_id: str, phase_id: str, job_dir: str) -> str:
    """``rtt-<run>-<phase>-<job>``, shortened to a valid job name of at most 58 characters."""
    name = f"rtt-{run_id[:30]}-{phase_id}-{job_dir}".lower()
    name = re.sub(r"[^a-z0-9-]", "-", name).strip("-")[:58].strip("-")
    if not _JOB_NAME.match(name):
        name = "rtt-" + _sha256(f"{run_id}/{phase_id}/{job_dir}".encode())[:24]
    return name


def pack_jobs(
    run_id: str, phase_id: str, shards: Sequence[Shard], n_jobs: int, workers: int = 0
) -> list[JobPlan]:
    """Shards dealt round-robin into ``n_jobs`` jobs (at most one job per shard)."""
    n_jobs = max(1, min(int(n_jobs), len(shards)))
    out = []
    for k in range(n_jobs):
        ids = tuple(s.shard_id for s in shards[k::n_jobs])
        d = f"j{k:02d}"
        out.append(JobPlan(d, job_name(run_id, phase_id, d), ids, workers))
    return out


def phase_kind(items: Sequence[Any]) -> str:
    from rtt.studio.runner import SegmentJob

    if items and all(isinstance(i, SegmentJob) and len(i.tag) >= 2 for i in items):
        chains = sorted({int(i.tag[1]) for i in items})
        return "screen-c" + "-".join(str(c) for c in chains)
    return "runs"


def write_phase(
    phase_dir: Path,
    run_id: str,
    phase_id: str,
    fn_name: str,
    items: Sequence[Any],
    shards: Sequence[Shard],
    jobs: Sequence[JobPlan],
) -> dict[str, Any]:
    """``phase.json`` and ``inputs/<shard>.pkl``; returns the phase record."""
    rows = []
    for s in shards:
        data = dumps([items[i] for i in s.indices])
        path = phase_dir / "inputs" / f"{s.shard_id}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append(
            {
                "shard_id": s.shard_id,
                "key": list(s.key),
                "indices": list(s.indices),
                "n": s.n,
                "input": f"inputs/{s.shard_id}.pkl",
                "sha256": _sha256(data),
                "bytes": len(data),
                "items": [describe_item(items[i]) for i in s.indices],
            }
        )
    record = {
        "format": FANOUT_FORMAT,
        "label": PRODUCT_MODE_LABEL,
        "run_id": run_id,
        "phase": phase_id,
        "fn": fn_name,
        "n_items": len(items),
        "pickle": {"protocol": PICKLE_PROTOCOL, "python": platform.python_version()},
        "shards": rows,
        "jobs": [
            {"dir": j.dir, "name": j.name, "shards": list(j.shards), "workers": j.workers} for j in jobs
        ],
    }
    _write_json(phase_dir / "phase.json", record)
    return record


# --------------------------------------------------------------------------- the job entry point


def _execute(fn: Callable[[Any], Any], items: Sequence[Any], workers: int) -> list[Any]:
    """``fn`` over ``items`` in order: in-process for one worker, else a spawned pool (as S5's)."""
    if workers <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    pool = mp.get_context("spawn").Pool(processes=min(workers, len(items)))
    try:
        out = pool.map(fn, items, chunksize=1)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    return out


def _cpu_s() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


def _task_result(res: Any) -> Any:
    return res[0] if isinstance(res, tuple) else res


def run_shards(
    phase_dir: str | Path,
    job: str,
    shard_ids: Sequence[str],
    *,
    workers: int = 0,
    argv: Sequence[str] = (),
    stage_dir: str | Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Run shards of a phase and write ``jobs/<job>/`` (staged, then published: the bucket mount
    rejects seeks and renames); return the manifest and the exit status (0 ok, 2 error)."""
    import shutil

    from rtt.cloud import entry

    phase_dir = Path(phase_dir)
    out = phase_dir / "jobs" / job
    out.mkdir(parents=True, exist_ok=True)
    stage = entry._stage_for(out, stage_dir)
    manifest = entry.base_manifest(JOB_KIND, out, argv)
    manifest["label"] = PRODUCT_MODE_LABEL
    manifest["format"] = FANOUT_FORMAT
    t0 = time.time()
    manifest["started_at"] = entry.utc_iso(t0)
    manifest["environment"] = entry.environment(probe_gl=False, probe_warp=False)
    t_env = time.time()
    status = 2
    try:
        phase_path = phase_dir / "phase.json"
        phase = json.loads(phase_path.read_text(encoding="utf-8"))
        if phase.get("format") != FANOUT_FORMAT:
            raise ValueError(f"{phase_path} is not a {FANOUT_FORMAT} phase")
        if not os.environ.get("RTT_RUN_ID"):
            manifest["run_id"] = phase["run_id"]  # a local job: name the run, not the job folder
        manifest["inputs"] = {
            "run_id": phase["run_id"],
            "phase": phase["phase"],
            "fn": phase["fn"],
            "shards": list(shard_ids),
            "phase_sha256": _file_sha256(phase_path),
        }
        import rtt.studio.search  # noqa: F401 - the laptop's import graph (families, library, jaws)

        fn = work_fn(phase["fn"])
        by_id = {s["shard_id"]: s for s in phase["shards"]}
        unknown = [s for s in shard_ids if s not in by_id]
        if unknown or not shard_ids:
            raise ValueError(f"shards not in the phase: {unknown or 'none given'}")
        flat: list[tuple[str, Any]] = []
        for sid in shard_ids:
            entry_ = by_id[sid]
            data = (phase_dir / entry_["input"]).read_bytes()
            if _sha256(data) != entry_["sha256"]:
                raise ValueError(f"input of {sid} does not match its SHA-256 in phase.json")
            items = safe_loads(data)
            if len(items) != entry_["n"]:
                raise ValueError(f"input of {sid} holds {len(items)} items, phase.json says {entry_['n']}")
            flat += [(sid, it) for it in items]
        n_workers = workers if workers > 0 else (os.cpu_count() or 1)
        cpu0, w0 = _cpu_s(), time.perf_counter()
        results = _execute(fn, [it for _, it in flat], n_workers)
        cpu1, w1 = _cpu_s(), time.perf_counter()
        rows = []
        for sid in shard_ids:
            mine = [r for (s, _), r in zip(flat, results, strict=True) if s == sid]
            data = dumps(mine)
            (stage / "shards").mkdir(parents=True, exist_ok=True)
            (stage / "shards" / f"{sid}.pkl").write_bytes(data)
            records = [_task_result(r).to_record() for r in mine]
            entry.write_json(
                stage / "shards" / f"{sid}.json",
                {"label": PRODUCT_MODE_LABEL, "shard_id": sid, "sha256": _sha256(data), "runs": records},
            )
            rows.append(
                {
                    "shard_id": sid,
                    "n": len(mine),
                    "sha256": _sha256(data),
                    "bytes": len(data),
                    "run_wall_s": round(sum(_task_result(r).wall_time_s for r in mine), 3),
                }
            )
        manifest["result"] = {
            "shards": rows,
            "runs": len(results),
            "workers": n_workers,
            "work_wall_s": round(w1 - w0, 3),
            "work_cpu_s": round(cpu1 - cpu0, 3),
            "run_wall_s": round(sum(r["run_wall_s"] for r in rows), 3),
        }
        status = 0
    except Exception:  # noqa: BLE001 - recorded in the manifest, then exit 2
        manifest["error"] = traceback.format_exc()[-4000:]
        print(manifest["error"], file=sys.stderr, flush=True)
    try:
        manifest = entry._finish(manifest, stage, out, t0, t_env, status)
    finally:
        if stage_dir is None and stage.resolve() != out.resolve():
            shutil.rmtree(stage, ignore_errors=True)
    return manifest, manifest["exit_status"]


# --------------------------------------------------------------------------- executors


@dataclass(frozen=True)
class JobRecord:
    """One finished job: its plan, the ledger line of the attempt that ran and its local output."""

    plan: JobPlan
    job_id: str
    spend: JobSpend
    attempts: tuple[JobSpend, ...]
    out_dir: Path
    executor: str
    cpu_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": PRODUCT_MODE_LABEL,
            "dir": self.plan.dir,
            "name": self.plan.name,
            "shards": list(self.plan.shards),
            "job_id": self.job_id,
            "executor": self.executor,
            "cpu_s": None if self.cpu_s is None else round(self.cpu_s, 3),
            "spend": self.spend.to_json(),
            "attempts": [a.to_json() for a in self.attempts],
        }


class Executor(Protocol):
    """Runs the jobs of one phase and leaves each job's output at ``<phase_dir>/jobs/<job>/``."""

    name: str
    workers: int  # worker processes it offers in total (shown as the report's ``workers``)

    def phase_dir(self, run_id: str, phase_id: str) -> Path: ...

    def plan_jobs(self, run_id: str, phase_id: str, shards: Sequence[Shard]) -> list[JobPlan]: ...

    def run_phase(self, run_id: str, phase_dir: Path, jobs: Sequence[JobPlan]) -> list[JobRecord]: ...


def _local_spend(
    job_id: str, name: str, wall: float, workers: int, run_id: str, ok: bool, note: str
) -> JobSpend:
    """A local job's ledger line, priced as ``cpu-d3`` with at least as many vCPUs as it used
    (what that wall time would cost there; nothing is billed)."""
    from rtt.cloud.nebius_jobs import CPU_VCPU_CHOICES, cpu_preset

    vcpus = next((c for c in CPU_VCPU_CHOICES if c >= max(1, workers)), CPU_VCPU_CHOICES[-1])
    preset = cpu_preset(vcpus)
    return JobSpend(
        job_id=job_id,
        name=name,
        platform="cpu-d3",
        preset=preset,
        region="local",
        state="COMPLETED" if ok else "FAILED",
        preemptible=False,
        provisioning_s=0.0,
        run_s=wall,
        wall_s=wall,
        usd_per_hour=hourly_usd("cpu-d3", preset),
        est_usd=round(estimate_usd("cpu-d3", preset, wall), 6),
        est_usd_run=round(estimate_usd("cpu-d3", preset, wall), 6),
        run_id=run_id,
        note=note,
    )


def _local_job_id(run_id: str, phase_dir: Path, job: str) -> str:
    return "local-" + _sha256(f"{run_id}/{phase_dir.name}/{job}/{time.time_ns()}".encode())[:16]


class InProcessExecutor:
    """Runs each job's shards in this process (one job per phase, jobs one after another: S5's
    scene cache is per process and not thread-safe); no subprocess, no cloud.

    ``root`` stands for the bucket: phases go to ``root/runs/<run_id>/<phase>/``.
    """

    name = "in-process"

    def __init__(self, root: str | Path, workers: int = 1, ledger: SpendLedger | None = None) -> None:
        self.root = Path(root)
        self.workers = max(1, min(int(workers), MAX_LOCAL_PARALLEL))
        self.ledger = ledger

    def phase_dir(self, run_id: str, phase_id: str) -> Path:
        return self.root / "runs" / run_id / phase_id

    def plan_jobs(self, run_id: str, phase_id: str, shards: Sequence[Shard]) -> list[JobPlan]:
        return pack_jobs(run_id, phase_id, shards, 1, self.workers)

    def run_phase(self, run_id: str, phase_dir: Path, jobs: Sequence[JobPlan]) -> list[JobRecord]:
        out = []
        for job in jobs:
            t = time.perf_counter()
            manifest, status = run_shards(phase_dir, job.dir, job.shards, workers=job.workers, stage_dir=None)
            wall = time.perf_counter() - t
            jid = _local_job_id(run_id, phase_dir, job.dir)
            spend = _local_spend(
                jid, job.name, wall, job.workers, run_id, status == 0, "in-process: not billed"
            )
            if self.ledger is not None:
                self.ledger.append(spend)
            if status != 0:
                raise RuntimeError(f"job {job.name} failed: {str(manifest.get('error'))[-800:]}")
            cpu = (manifest.get("result") or {}).get("work_cpu_s")
            out.append(JobRecord(job, jid, spend, (spend,), phase_dir / "jobs" / job.dir, self.name, cpu))
        return out


class SubprocessExecutor:
    """The local fake cloud: each job is ``python -m rtt.studio.cloud shards`` in a subprocess, with
    the same arguments a Nebius job gets except that the phase folder is under ``root`` instead of
    the bucket mount. At most ``parallel`` (<= 4) jobs run at once, each with ``workers`` processes.
    Each job gets a ledger line priced as ``cpu-d3`` for its wall time (nothing is billed) and its
    measured CPU seconds (the child's resource usage)."""

    name = "fake-cloud"

    def __init__(
        self,
        root: str | Path,
        *,
        parallel: int = MAX_LOCAL_PARALLEL,
        workers: int = 1,
        ledger: SpendLedger | None = None,
        python: str = sys.executable,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.parallel = max(1, min(int(parallel), MAX_LOCAL_PARALLEL))
        self.job_workers = max(1, int(workers))
        if self.parallel * self.job_workers > MAX_LOCAL_PARALLEL:
            raise ValueError(f"at most {MAX_LOCAL_PARALLEL} local processes (parallel x workers)")
        self.workers = self.parallel * self.job_workers
        self.ledger = ledger
        self.python = python
        self.env = dict(env or {})

    def phase_dir(self, run_id: str, phase_id: str) -> Path:
        return self.root / "runs" / run_id / phase_id

    def plan_jobs(self, run_id: str, phase_id: str, shards: Sequence[Shard]) -> list[JobPlan]:
        return pack_jobs(run_id, phase_id, shards, self.parallel, self.job_workers)

    def _argv(self, phase_dir: Path, job: JobPlan) -> list[str]:
        return [
            self.python,
            "-m",
            "rtt.studio.cloud",
            "shards",
            "--phase-dir",
            str(phase_dir),
            "--job",
            job.dir,
            "--shards",
            ",".join(job.shards),
            "--workers",
            str(job.workers),
        ]

    def _run_one(self, run_id: str, phase_dir: Path, job: JobPlan) -> JobRecord:
        env = dict(os.environ)
        env.update(self.env)
        env.update(RTT_RUN_ID=run_id, RTT_JOB_NAME=job.name, RTT_LAUNCHER=LOCAL_LAUNCHER)
        for k in ("RTT_PLATFORM", "RTT_PRESET", "RTT_REGION", "RTT_IMAGE_NAME", "RTT_IMAGE_DIGEST"):
            env.pop(k, None)
        src = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(
            [src, *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]]
        )
        log = phase_dir / "logs" / f"{job.dir}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        with log.open("wb") as fh:
            proc = subprocess.Popen(self._argv(phase_dir, job), stdout=fh, stderr=subprocess.STDOUT, env=env)
            _, wait_status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(wait_status)
        wall = time.perf_counter() - t
        jid = _local_job_id(run_id, phase_dir, job.dir)
        ok = proc.returncode == 0
        spend = _local_spend(jid, job.name, wall, job.workers, run_id, ok, "local fake cloud: not billed")
        if self.ledger is not None:
            self.ledger.append(spend)
        if not ok:
            tail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"job {job.name} exited {proc.returncode}:\n{tail}")
        cpu = usage.ru_utime + usage.ru_stime
        return JobRecord(job, jid, spend, (spend,), phase_dir / "jobs" / job.dir, self.name, cpu)

    def run_phase(self, run_id: str, phase_dir: Path, jobs: Sequence[JobPlan]) -> list[JobRecord]:
        if len(jobs) == 1:
            return [self._run_one(run_id, phase_dir, jobs[0])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel) as ex:
            futures = [ex.submit(self._run_one, run_id, phase_dir, j) for j in jobs]
            return [f.result() for f in futures]


# --------------------------------------------------------------------------- Nebius


def studio_job_spec(
    config: Any, run_id: str, phase_id: str, job: JobPlan, candidate: Any, *, timeout: str = "1h"
) -> Any:
    """The Serverless Job that runs ``job`` of a phase on the region's bucket mount.

    The same job hygiene as :func:`rtt.cloud.runs.job_spec_for`: image by digest, the bucket by ID
    at ``/out``, ``--restart-policy never``, the job facts the manifest records, never the image
    path (it holds the registry ID). No secret of any kind: these jobs never call Token Factory and
    never see a held-out secret (held-out conditions arrive as work items).
    """
    from rtt.cloud.nebius_jobs import JobSpec
    from rtt.cloud.runs import run_prefix, split_image_ref

    config.require("image", "bucket_id", "subnet_id", "project_id")
    image_name, digest = split_image_ref(config.image)
    args = [
        "-m",
        "rtt.studio.cloud",
        "shards",
        "--phase-dir",
        f"{MOUNT}/{run_prefix(run_id)}/{phase_id}",
        "--job",
        job.dir,
        "--shards",
        ",".join(job.shards),
        "--workers",
        str(job.workers),
    ]
    env = {
        "RTT_RUN_ID": run_id,
        "RTT_JOB_NAME": job.name,
        "RTT_IMAGE_NAME": image_name,
        "RTT_IMAGE_DIGEST": digest,
        "RTT_LAUNCHER": LAUNCHER,
    }
    return JobSpec(
        name=job.name,
        image=config.image,
        command=CONTAINER_PYTHON,
        args=args,
        env=env,
        env_secrets={},
        volumes=[f"{config.bucket_id}:{MOUNT}:rw"],
        platform=candidate.platform,
        preset=candidate.preset,
        timeout=timeout,
        restart_policy="never",
        subnet_id=config.subnet_id,
        parent_id=config.project_id,
    )


class NebiusExecutor:
    """Phases as Nebius Serverless Jobs on the existing :mod:`rtt.cloud` machinery.

    Per phase: ``phase.json`` and the inputs go to ``runs/<run_id>/<phase>/`` in the bucket of the
    region a job runs in; each job is submitted with :meth:`rtt.cloud.nebius_jobs.NebiusJobs.run` on
    ``target`` (default ``uks2-cpu``: ``cpu-d3`` with ``vcpus`` vCPUs), moving to ``eun1-gpu`` (H100)
    only when ``fallback`` and the target cannot start it (not configured, no capacity or vCPU quota,
    or the job ended before its container started), as ``python -m rtt.cloud run`` does. Every
    attempt goes to ``ledger`` (its cap is enforced before each submission and while a job runs)
    and, when given, to ``project_ledger`` (the project's ledger, for the record). The job's output
    is fetched to ``<staging>/runs/<run_id>/<phase>/jobs/<job>/`` and verified. At most
    ``max_concurrent`` jobs are in flight (each holds ``vcpus`` of the region's quota).
    """

    name = "nebius"

    def __init__(
        self,
        config: Any,
        *,
        staging: str | Path,
        ledger: SpendLedger,
        project_ledger: SpendLedger | None = None,
        jobs_for: Callable[[Any], Any] | None = None,
        store_for: Callable[[Any], Any] | None = None,
        target: str = "uks2-cpu",
        vcpus: int = 32,
        fallback: bool = True,
        jobs_per_phase: int = 1,
        max_concurrent: int = 3,
        expected_run_s: float = 900.0,
        provisioning_timeout_s: float = 900.0,
        poll_s: float = 15.0,
        timeout: str = "1h",
        check_capacity: bool = True,
        on_update: Callable[[Any], None] | None = None,
    ) -> None:
        from rtt.cloud.nebius_jobs import NebiusJobs, cpu_preset
        from rtt.cloud.storage import BucketStore

        if ledger.cap_usd is None:
            raise ValueError("a Nebius fan-out needs a capped spend ledger")
        cpu_preset(vcpus)  # refuse an unknown size up front
        self.config = config
        self.staging = Path(staging)
        self.ledger = ledger
        self.project_ledger = project_ledger
        self.jobs_for = jobs_for or NebiusJobs.from_config
        self.store_for = store_for or BucketStore.from_config
        self.target = target
        self.vcpus = int(vcpus)
        self.fallback = fallback
        self.jobs_per_phase = max(1, int(jobs_per_phase))
        self.max_concurrent = max(1, int(max_concurrent))
        self.workers = self.vcpus * self.jobs_per_phase
        self.expected_run_s = expected_run_s
        self.provisioning_timeout_s = provisioning_timeout_s
        self.poll_s = poll_s
        self.timeout = timeout
        self.check_capacity = check_capacity
        self.on_update = on_update
        self._uploaded: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(self.max_concurrent)

    def phase_dir(self, run_id: str, phase_id: str) -> Path:
        return self.staging / "runs" / run_id / phase_id

    def plan_jobs(self, run_id: str, phase_id: str, shards: Sequence[Shard]) -> list[JobPlan]:
        return pack_jobs(run_id, phase_id, shards, self.jobs_per_phase, 0)

    def _chain(self) -> list[Any]:
        from rtt.cloud.nebius_jobs import FALLBACK_TARGET, TARGETS, get_target

        first = get_target(self.target, self.vcpus if TARGETS[self.target].kind == "cpu" else None)
        chain = [first]
        if self.fallback and first.name != FALLBACK_TARGET:
            chain.append(get_target(FALLBACK_TARGET))
        return chain

    def _upload(self, store: Any, region: str, run_id: str, phase_dir: Path) -> None:
        from rtt.cloud.runs import run_prefix

        with self._lock:
            key = (region, f"{run_id}/{phase_dir.name}")
            if key in self._uploaded:
                return
            prefix = f"{run_prefix(run_id)}/{phase_dir.name}"
            store.upload_file(phase_dir / "phase.json", f"{prefix}/phase.json")
            store.upload_dir(phase_dir / "inputs", f"{prefix}/inputs")
            self._uploaded.add(key)

    def _run_one(self, run_id: str, phase_dir: Path, job: JobPlan) -> JobRecord:
        from rtt.cloud.__main__ import check_same_image, never_started
        from rtt.cloud.config import ConfigError
        from rtt.cloud.nebius_jobs import NoCapacityError
        from rtt.cloud.runs import run_prefix, verify_run

        chain = self._chain()
        attempts: list[JobSpend] = []
        with self._slots:
            for i, target in enumerate(chain):
                last = i + 1 == len(chain)
                try:
                    cfg = self.config.for_region(target.region)
                    check_same_image(self.config, cfg)
                    spec = studio_job_spec(
                        cfg, run_id, phase_dir.name, job, target.candidates[0], timeout=self.timeout
                    )
                    store = self.store_for(cfg)
                except ConfigError:
                    if last:
                        raise
                    continue
                self._upload(store, target.region, run_id, phase_dir)
                jobs = self.jobs_for(cfg)
                try:
                    outcome = jobs.run(
                        spec,
                        target.candidates,
                        ledger=self.ledger,
                        expected_run_s=self.expected_run_s,
                        provisioning_timeout_s=self.provisioning_timeout_s,
                        poll_s=self.poll_s,
                        check_capacity=self.check_capacity,
                        run_id=run_id,
                        on_update=self.on_update,
                    )
                except NoCapacityError as exc:
                    attempts += exc.attempts
                    self._record(exc.attempts)
                    if last:
                        raise
                    continue
                attempts += outcome.attempts
                self._record(outcome.attempts)
                if never_started(outcome.job) and not last:
                    continue
                if not outcome.job.succeeded:
                    raise RuntimeError(
                        f"job {outcome.job.name} ended {outcome.job.state}: {outcome.job.state_details}"
                    )
                local = phase_dir / "jobs" / job.dir
                store.download_prefix(f"{run_prefix(run_id)}/{phase_dir.name}/jobs/{job.dir}", local)
                check = verify_run(local)
                if not check.ok:
                    raise RuntimeError(f"job {outcome.job.name}: {check.describe()}")
                cpu = ((check.manifest or {}).get("result") or {}).get("work_cpu_s")
                return JobRecord(
                    job, outcome.job.id, outcome.attempts[-1], tuple(attempts), local, self.name, cpu
                )
        raise RuntimeError(f"job {job.name} found no target")  # pragma: no cover - the loop raises

    def _record(self, attempts: Iterable[JobSpend]) -> None:
        if self.project_ledger is None:
            return
        for a in attempts:
            self.project_ledger.append(a)

    def run_phase(self, run_id: str, phase_dir: Path, jobs: Sequence[JobPlan]) -> list[JobRecord]:
        if len(jobs) == 1:
            return [self._run_one(run_id, phase_dir, jobs[0])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as ex:
            futures = [ex.submit(self._run_one, run_id, phase_dir, j) for j in jobs]
            return [f.result() for f in futures]


# --------------------------------------------------------------------------- merge and the pool


def merge_phase(phase_dir: Path, items: Sequence[Any], records: Sequence[JobRecord]) -> list[Any]:
    """Every item's result at its index, after the checks in the module docstring."""
    phase = json.loads((phase_dir / "phase.json").read_text(encoding="utf-8"))
    shards = {s["shard_id"]: s for s in phase["shards"]}
    results: list[Any] = [None] * len(items)
    filled = [False] * len(items)
    for rec in records:
        manifest = json.loads((rec.out_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("exit_status") != 0:
            raise RuntimeError(f"job {rec.plan.name}: exit status {manifest.get('exit_status')}")
        listed = {r["shard_id"]: r for r in (manifest.get("result") or {}).get("shards", [])}
        for sid in rec.plan.shards:
            if sid not in listed:
                raise RuntimeError(f"job {rec.plan.name} did not report shard {sid}")
            data = (rec.out_dir / "shards" / f"{sid}.pkl").read_bytes()
            if _sha256(data) != listed[sid]["sha256"]:
                raise RuntimeError(f"shard {sid}: the result file does not match the job manifest")
            res = safe_loads(data)
            idx = shards[sid]["indices"]
            if len(res) != len(idx):
                raise RuntimeError(f"shard {sid}: {len(res)} results for {len(idx)} items")
            for i, r in zip(idx, res, strict=True):
                tr = _task_result(r)
                setup, spec, cond = _item_parts(items[i])
                if (tr.task_id, tr.setup_id, tr.condition_id, tr.seed) != (
                    spec.task_id,
                    setup.setup_id,
                    cond.condition_id,
                    cond.seed,
                ):
                    raise RuntimeError(f"shard {sid}: a result does not belong to its item {i}")
                if filled[i]:
                    raise RuntimeError(f"item {i} has two results")
                results[i], filled[i] = r, True
    missing = [i for i, f in enumerate(filled) if not f]
    if missing:
        raise RuntimeError(f"no result for items {missing[:10]}")
    return results


def phase_rows(
    phase_dir: Path, items: Sequence[Any], results: Sequence[Any], records: Sequence[JobRecord]
) -> list[dict[str, Any]]:
    """One row per run: its record and its job (id, ledger price, share of the job's estimate)."""
    phase = json.loads((phase_dir / "phase.json").read_text(encoding="utf-8"))
    job_of = {sid: rec for rec in records for sid in rec.plan.shards}
    by_item: dict[int, tuple[str, JobRecord]] = {
        i: (s["shard_id"], job_of[s["shard_id"]]) for s in phase["shards"] for i in s["indices"]
    }
    walls: dict[str, float] = {}
    counts: dict[str, int] = {}
    for i, r in enumerate(results):
        jid = by_item[i][1].job_id
        walls[jid] = walls.get(jid, 0.0) + _task_result(r).wall_time_s
        counts[jid] = counts.get(jid, 0) + 1
    phase_id = phase["phase"]
    rows = []
    for i, (item, r) in enumerate(zip(items, results, strict=True)):
        sid, rec = by_item[i]
        tr = _task_result(r)
        total = walls[rec.job_id]
        share = tr.wall_time_s / total if total > 0 else 1.0 / counts[rec.job_id]
        rows.append(
            {
                "label": PRODUCT_MODE_LABEL,
                "phase": phase_id,
                "index": i,
                "shard_id": sid,
                "item": describe_item(item),
                **tr.to_record(),
                "cloud": {
                    "job_id": rec.job_id,
                    "job_name": rec.spend.name,
                    "executor": rec.executor,
                    "platform": rec.spend.platform,
                    "preset": rec.spend.preset,
                    "region": rec.spend.region,
                    "usd_per_hour": rec.spend.usd_per_hour,
                    "job_wall_s": round(rec.spend.wall_s, 3),
                    "job_est_usd": rec.spend.est_usd,
                    "est_usd": round(rec.spend.est_usd * share, 8),
                },
            }
        )
    return rows


@dataclass
class PhaseRecord:
    phase_id: str
    fn: str
    n_items: int
    n_shards: int
    jobs: list[JobRecord] = field(default_factory=list)
    wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase_id,
            "fn": self.fn,
            "n_items": self.n_items,
            "n_shards": self.n_shards,
            "jobs": [j.to_dict() for j in self.jobs],
            "wall_s": round(self.wall_s, 3),
            "est_usd": round(sum(j.spend.est_usd for j in self.jobs), 6),
        }


class CloudPool:
    """A :class:`rtt.studio.runner.WorkerPool` stand-in whose ``map`` is one phase of jobs.

    ``session_dir`` collects ``rows.jsonl`` (one row per run, with its job id and dollar share),
    ``jobs.jsonl`` (one line per job) and ``phases.json``. ``prefix`` names the phases
    (``<prefix>p00-screen-c0``...), so several searches can share one run id.
    """

    def __init__(self, executor: Executor, run_id: str, *, session_dir: str | Path, prefix: str = "") -> None:
        from rtt.cloud.runs import validate_run_id

        self.executor = executor
        self.run_id = validate_run_id(run_id)
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if prefix and not re.match(r"^[a-z0-9-]{1,24}$", prefix):
            raise ValueError("a phase prefix is 1-24 lowercase letters, digits and dashes")
        self.prefix = prefix
        self.phases: list[PhaseRecord] = []
        self.rows: list[dict[str, Any]] = []
        self.workers = executor.workers
        self.jobs_run = 0
        self.busy_s = 0.0

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any], chunksize: int = 1) -> list[Any]:
        del chunksize  # sharding decides how work is grouped
        items = list(items)
        if not items:
            return []
        t0 = time.perf_counter()
        name = work_name(fn)
        phase_id = f"{self.prefix}p{len(self.phases):02d}-{phase_kind(items)}"
        phase_dir = self.executor.phase_dir(self.run_id, phase_id)
        if (phase_dir / "phase.json").exists():
            raise FileExistsError(f"{phase_dir} already holds a phase: use a new run id or prefix")
        shards = plan_shards(items)
        jobs = self.executor.plan_jobs(self.run_id, phase_id, shards)
        write_phase(phase_dir, self.run_id, phase_id, name, items, shards, jobs)
        records = self.executor.run_phase(self.run_id, phase_dir, jobs)
        results = merge_phase(phase_dir, items, records)
        rows = phase_rows(phase_dir, items, results, records)
        rec = PhaseRecord(phase_id, name, len(items), len(shards), list(records), time.perf_counter() - t0)
        self.phases.append(rec)
        self.rows += rows
        with (self.session_dir / "rows.jsonl").open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        with (self.session_dir / "jobs.jsonl").open("a", encoding="utf-8") as fh:
            for j in records:
                fh.write(json.dumps({"phase": phase_id, **j.to_dict()}, sort_keys=True) + "\n")
        _write_json(self.session_dir / "phases.json", self.summary())
        self.jobs_run += len(items)
        self.busy_s += rec.wall_s
        return results

    def summary(self) -> dict[str, Any]:
        jobs = [j for p in self.phases for j in p.jobs]
        cpu = [j.cpu_s for j in jobs if j.cpu_s is not None]
        return {
            "format": FANOUT_FORMAT,
            "label": PRODUCT_MODE_LABEL,
            "run_id": self.run_id,
            "executor": self.executor.name,
            "phases": [p.to_dict() for p in self.phases],
            "runs": len(self.rows),
            "jobs": len(jobs),
            "est_usd": round(sum(j.spend.est_usd for j in jobs), 6),
            "est_usd_run": round(sum(j.spend.est_usd_run for j in jobs), 6),
            "cpu_s": round(sum(cpu), 3) if cpu else None,
            "run_wall_s": round(sum(r["wall_time_s"] for r in self.rows), 3),
        }

    def close(self) -> None:
        """Nothing to release: every job has finished when ``map`` returns."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def cost_per_run(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Measured seconds per run by kind (screen or full run): mean and total of the runs' own wall
    times (one process per run, so close to CPU seconds), from ``rows.jsonl``."""
    out: dict[str, Any] = {}
    for r in rows:
        kind = "screen" if "-screen-c" in r["phase"] else "run"
        d = out.setdefault(kind, {"runs": 0, "wall_s": 0.0})
        d["runs"] += 1
        d["wall_s"] += float(r["wall_time_s"])
    for d in out.values():
        d["mean_s"] = round(d["wall_s"] / d["runs"], 3) if d["runs"] else math.nan
        d["wall_s"] = round(d["wall_s"], 3)
    return out


# --------------------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rtt.studio.cloud", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("shards", help="run shards of a phase (the job entry point)")
    sh.add_argument("--phase-dir", type=Path, required=True, help="the phase folder (on the bucket mount)")
    sh.add_argument("--job", required=True, help="this job's folder name under jobs/")
    sh.add_argument("--shards", required=True, help="comma-separated shard ids")
    sh.add_argument("--workers", type=int, default=0, help="worker processes (0: one per CPU)")
    sh.add_argument(
        "--stage", type=Path, default=None, help="local staging folder (default: a temporary one)"
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    from rtt.cloud.entry import _summary_line

    args_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(args_list)
    if not re.match(r"^j[0-9]{2,4}$", args.job):
        raise SystemExit("--job must be a job folder name such as j00")
    manifest, status = run_shards(
        args.phase_dir,
        args.job,
        [s.strip() for s in args.shards.split(",") if s.strip()],
        workers=args.workers,
        argv=args_list,
        stage_dir=args.stage,
    )
    print(_summary_line(manifest), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
