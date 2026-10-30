"""Inside the Nebius job: run the Studio search on one submitted plan and publish the results.

``python -m rtt.runner.job --run-dir /out/runs/<run_id> --plan-sha256 <hex>``

1. Read ``<run-dir>/input/plan.json`` (written by the runner), check its SHA-256 against the one
   the runner put on the command line, and check the plan again
   (:func:`rtt.runner.validate.check_run_plan`, with the limits the runner passes as
   ``RUNNER_MAX_OBJECTS``, ``RUNNER_MAX_STEPS`` and ``RUNNER_MAX_PLAN_BYTES``).
2. Run :func:`rtt.studio.search.run_task_tournament` with the product's default settings (the
   default candidates and library, 8 conditions, 3 screened, 4 finalists, the default
   controllers). Only the worker count differs: a pool as wide as the job's vCPUs, since the
   search's results do not depend on how its items are spread over processes.
3. Stage ``report.json`` and ``summary.json`` (:func:`rtt.runner.summary.summarize_report`)
   locally, publish them to the run directory front to back, and write ``manifest.json`` last
   (:mod:`rtt.cloud.entry`'s staging, because the bucket mount cannot seek or rename).

A failure still publishes a ``summary.json`` with ``status: failed`` and a short error, and the
manifest's ``exit_status`` is not 0. The search stops itself at ``--deadline-s`` after the
container starts (the runner passes its wall limit less about 4 minutes for provisioning and
publishing: 960 s by default), so a slow search ends with its own error rather than the runner's
cancel.

No secret reaches this job: it never calls a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any, Final

from rtt.runner.config import RunnerConfig
from rtt.runner.summary import PRODUCT_MODE_LABEL, SUMMARY_FORMAT, summarize_report
from rtt.runner.validate import check_run_plan
from rtt.studio.runner import WorkerPool

__all__ = ["JobPool", "main", "run_job"]

MAX_PLAN_BYTES: Final = 256 * 1024
DEFAULT_DEADLINE_S: Final = 960


class JobPool(WorkerPool):
    """:class:`rtt.studio.runner.WorkerPool` without its six-process cap (one job owns the VM)."""

    def __init__(self, workers: int) -> None:
        super().__init__(1)
        self.workers = max(1, int(workers))


class DeadlineExceeded(RuntimeError):
    pass


def _on_alarm(signum: int, frame: FrameType | None) -> None:
    raise DeadlineExceeded("the search passed its time limit")


def _cpus() -> int:
    """The CPUs this process may use (the job's vCPUs)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _failed(run_id: str, error: str) -> dict[str, Any]:
    return {
        "format": SUMMARY_FORMAT,
        "version": 1,
        "label": PRODUCT_MODE_LABEL,
        "run_id": run_id,
        "status": "failed",
        "error": error[:300],
    }


def _code_version() -> str:
    try:
        from rtt.sim.replay import code_version

        return code_version()
    except Exception:  # noqa: BLE001
        return os.environ.get("RTT_GIT_SHA") or "unknown"


def run_job(
    run_dir: str | Path,
    plan_sha256: str,
    *,
    workers: int | None = None,
    deadline_s: int = DEFAULT_DEADLINE_S,
    stage_dir: str | Path | None = None,
    search_kwargs: Mapping[str, Any] | None = None,
    argv: Sequence[str] = (),
) -> tuple[dict[str, Any], int]:
    """Run one job (module docstring); returns the manifest and the exit status.

    ``search_kwargs`` overrides search settings (tests only; the command line has no such flag).
    """
    from rtt.cloud.entry import _finish, _stage_for, base_manifest, write_json

    t0 = time.time()
    out = Path(run_dir)
    run_id = os.environ.get("RTT_RUN_ID") or out.name
    stage = _stage_for(out, stage_dir)
    manifest = base_manifest("effector-lab-run", out, argv)
    manifest["run_id"] = run_id
    workers = workers or _cpus()
    status = 1
    t_mid = time.time()
    try:
        path = out / "input" / "plan.json"
        raw = path.read_bytes() if path.is_file() else b""
        if not raw or len(raw) > MAX_PLAN_BYTES:
            raise ValueError("the plan is missing or too large")
        sha = hashlib.sha256(raw).hexdigest()
        if sha != plan_sha256:
            raise ValueError("the plan does not match the SHA-256 the runner sent")
        # the runner passes its limits (RUNNER_MAX_*) in the job's environment
        spec, problems = check_run_plan(RunnerConfig.from_env(), json.loads(raw))
        if spec is None:
            raise ValueError("the plan did not pass the checks: " + "; ".join(problems[:3]))
        manifest["inputs"] = {"plan_sha256": sha, "task_id": spec.task_id}
        from rtt.studio.search import run_task_tournament

        if deadline_s > 0 and hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(int(deadline_s))
        t_mid = time.time()
        try:
            # on an exception (the deadline included) the pool's __exit__ terminates its workers
            with JobPool(workers) as pool:
                report = run_task_tournament(spec, pool=pool, **dict(search_kwargs or {}))
        finally:
            if deadline_s > 0 and hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        data = json.loads(json.dumps(report.to_dict(), default=str))
        write_json(stage / "report.json", data)
        summary = summarize_report(data, run_id=run_id, code_version=_code_version())
        write_json(stage / "summary.json", summary)
        manifest["result"] = {
            "outcome": summary["outcome"],
            "workers": pool.workers,
            "timings": data.get("timings"),
        }
        status = 0
    except Exception as exc:  # noqa: BLE001 - every failure is reported in the summary
        error = str(exc) if isinstance(exc, ValueError | DeadlineExceeded) else f"{type(exc).__name__}"
        write_json(stage / "summary.json", _failed(run_id, error))
        manifest["error"] = error[:300]
    manifest["label"] = PRODUCT_MODE_LABEL
    manifest = _finish(manifest, stage, out, t0, t_mid, status)
    return manifest, int(manifest.get("exit_status", status))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rtt.runner.job", description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--plan-sha256", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--deadline-s", type=int, default=DEFAULT_DEADLINE_S)
    args = ap.parse_args(argv)
    manifest, status = run_job(
        args.run_dir,
        args.plan_sha256,
        workers=args.workers,
        deadline_s=args.deadline_s,
        argv=list(sys.argv[1:] if argv is None else argv),
    )
    print(json.dumps({"run_id": manifest.get("run_id"), "exit_status": status}))
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
