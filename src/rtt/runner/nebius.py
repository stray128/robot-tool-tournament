"""One Nebius Serverless Job per run, and its results from Object Storage.

:class:`NebiusBackend` reuses the repository's job tooling (:mod:`rtt.cloud.nebius_jobs`: the
``nebius`` CLI wrapper, :class:`~rtt.cloud.nebius_jobs.JobSpec` with its hygiene checks, the job
states). Each run:

1. The validated plan goes to ``runs/<run_id>/input/plan.json`` in the results bucket (S3 API,
   with the runner's access key).
2. ``nebius ai job create --async``: the image pinned by digest, ``cpu-d3`` with 32 vCPUs in
   uk-south2, the bucket mounted at ``/out``, ``--restart-policy never`` and no secret of any
   kind. The container runs ``python -m rtt.runner.job --run-dir /out/runs/<run_id>
   --plan-sha256 <hex>`` (:mod:`rtt.runner.job`).
3. The runner polls ``nebius ai job get`` and, once the job ends, reads ``manifest.json`` and
   ``summary.json`` back. The full ``report.json`` is offered through a short-lived presigned URL.

In AWS the CLI authenticates as the runner's service account: :meth:`NebiusBackend.from_credentials`
writes the service-account JSON (from Secrets Manager) to a private file under ``/tmp`` and
creates a CLI profile that points at it (``nebius profile create --service-account-file-path
... --skip-auth``), all under a private ``HOME``.

:class:`FakeNebius` is the same interface in memory, for tests and ``rtt runner serve --fake``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol

from rtt.cloud.config import NebiusConfig
from rtt.cloud.nebius_jobs import (
    TERMINAL_STATES,
    CommandResult,
    Job,
    JobSpec,
    NebiusCLI,
    NebiusCLIError,
    NebiusJobs,
)
from rtt.runner.config import NebiusJobConfig
from rtt.runner.secrets import NebiusCredentials

__all__ = [
    "CONTAINER_PYTHON",
    "JOB_MODULE",
    "FakeNebius",
    "JobBackend",
    "JobStatus",
    "NebiusBackend",
    "SubmitNotStarted",
    "SubmitRefused",
    "Submitted",
    "grpc_status",
    "grpc_statuses",
    "job_name",
]

CONTAINER_PYTHON: Final = "/app/.venv/bin/python"
JOB_MODULE: Final = "rtt.runner.job"
MOUNT: Final = "/out"
PROFILE: Final = "runner"
LAUNCHER: Final = "effector-lab-runner"
_MAX_SUMMARY_BYTES: Final = 512 * 1024


class SubmitNotStarted(RuntimeError):
    """Submission failed before the job was requested: no job exists."""


class SubmitRefused(RuntimeError):
    """The platform answered ``create`` with a refusal (``code``: a gRPC status such as
    ``ResourceExhausted`` or ``InvalidArgument``), not a timeout. The job was most likely not
    created; the caller confirms with a lookup by name before giving anything back."""

    def __init__(self, code: str) -> None:
        super().__init__(f"create refused: {code}")
        self.code = code


#: gRPC status in the CLI's error text: ``rpc error: code = NotFound desc = ...`` (also accepts
#: ``code: NOT_FOUND`` and ``"code": "NOT_FOUND"``). Only this status is read; free text such as
#: "token not found" is not.
_GRPC_CODE: Final = re.compile(r"\bcode\"?\s*[=:]\s*\"?([A-Za-z_]+)\"?")
#: Statuses with which the platform refuses a request it evaluated (so it created nothing). Not
#: AlreadyExists (a retried create whose first attempt succeeded), nor DeadlineExceeded,
#: Unavailable, Unknown, Internal, Aborted or Canceled (the request may have gone through).
REFUSED: Final = frozenset(
    {
        "invalidargument",
        "failedprecondition",
        "outofrange",
        "permissiondenied",
        "unauthenticated",
        "resourceexhausted",
        "notfound",
        "unimplemented",
    }
)
#: The exit status :func:`rtt.cloud.nebius_jobs.subprocess_runner` gives a CLI call it timed out.
TIMED_OUT: Final = 124


def grpc_statuses(text: str) -> list[str]:
    """Every gRPC status named in ``text``, in order, normalised (``NotFound`` and ``NOT_FOUND``
    give ``notfound``)."""
    return [c.replace("_", "").lower() for c in _GRPC_CODE.findall(text)]


def grpc_status(text: str) -> str | None:
    """The last gRPC status named in ``text`` (for messages; decisions read every status with
    :func:`grpc_statuses`); ``None`` when there is none."""
    codes = grpc_statuses(text)
    return codes[-1] if codes else None


def _not_found(exc: NebiusCLIError) -> bool:
    """The platform said the job does not exist: every status in the error is NotFound. An
    outer error quoting an inner NotFound (``code = Unavailable ... code = NotFound``), or a retry
    log that ends in NotFound after another failure, is not an answer."""
    codes = grpc_statuses(exc.detail)
    return exc.returncode != TIMED_OUT and bool(codes) and all(c == "notfound" for c in codes)


def _refused(exc: NebiusCLIError) -> str | None:
    """The refusal status of a failed ``create``: every status in the error is a refusal (an
    earlier attempt that failed otherwise may have created the job). ``None``: a timeout, no
    status, or any status that does not rule out a created job."""
    if exc.command != "ai job create" or exc.returncode == TIMED_OUT:
        return None
    codes = grpc_statuses(exc.detail)
    return codes[-1] if codes and all(c in REFUSED for c in codes) else None


def job_name(run_id: str) -> str:
    """``elab-<run_id>``: 37 characters, a valid job name for a 32-character run id."""
    return f"elab-{run_id}"


def prefix(run_id: str) -> str:
    return f"runs/{run_id}"


@dataclass(frozen=True)
class Submitted:
    job_id: str
    job_name: str
    created_at: datetime | None


@dataclass(frozen=True)
class JobStatus:
    state: str
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def succeeded(self) -> bool:
        return self.state == "COMPLETED"

    @classmethod
    def of(cls, job: Job) -> JobStatus:
        return cls(job.state, job.created_at, job.started_at, job.finished_at)


class JobBackend(Protocol):
    def submit(self, run_id: str, plan_json: bytes) -> Submitted: ...

    def find(self, run_id: str) -> Submitted | None: ...

    def status(self, job_id: str) -> JobStatus: ...

    def cancel(self, job_id: str) -> None: ...

    def read_result(self, run_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """``(manifest, summary)``; either is ``None`` while missing or unreadable."""
        ...

    def report_url(self, run_id: str, ttl_s: int) -> str | None: ...


# --------------------------------------------------------------------------- the real backend


def _env_runner(env: Mapping[str, str]) -> Callable[[Sequence[str], float | None], CommandResult]:
    """A :data:`rtt.cloud.nebius_jobs.Runner` that runs the CLI with ``env`` (a private HOME)."""

    def run(argv: Sequence[str], timeout_s: float | None) -> CommandResult:
        try:
            proc = subprocess.run(
                list(argv), capture_output=True, text=True, timeout=timeout_s, check=False, env=dict(env)
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(124, "", f"timed out after {exc.timeout} s")
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)

    return run


class NebiusBackend:
    """Runs and reads jobs (module docstring). ``cli`` and ``s3`` are ready clients."""

    def __init__(
        self, cfg: NebiusJobConfig, cli: NebiusCLI, s3: Any, job_env: Mapping[str, str] | None = None
    ) -> None:
        missing = cfg.missing()
        if missing:
            raise ValueError(f"the Nebius job settings are incomplete: {', '.join(missing)}")
        self.cfg = cfg
        self.cli = cli
        self.s3 = s3
        self.job_env = dict(job_env or {})
        self.jobs = NebiusJobs(
            cli, NebiusConfig(profile=cli.profile, project_id=cfg.project_id, cli=cli.binary)
        )

    def __repr__(self) -> str:
        return f"NebiusBackend({self.cfg!r})"

    @classmethod
    def from_credentials(
        cls,
        cfg: NebiusJobConfig,
        creds: NebiusCredentials,
        home: str | Path = "/tmp/nebius-home",
        *,
        s3_client: Any | None = None,
        runner: Callable[[Sequence[str], float | None], CommandResult] | None = None,
        job_env: Mapping[str, str] | None = None,
    ) -> NebiusBackend:
        """Write the service-account file (0600) and a CLI profile under ``home``; build the S3
        client from the access key. Nothing secret is passed on a command line."""
        home = Path(home)
        conf = home / ".nebius"
        conf.mkdir(parents=True, exist_ok=True, mode=0o700)
        sa_file = conf / "runner-sa.json"
        fd = os.open(sa_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(dict(creds.service_account_credentials), fh)
        env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        run = runner or _env_runner(env)
        cli = NebiusCLI(cfg.cli, PROFILE, run, timeout_s=60.0)
        args = [
            "profile",
            "create",
            PROFILE,
            "--endpoint",
            "api.nebius.cloud",
            "--service-account-file-path",
            str(sa_file),
            "--parent-id",
            str(cfg.project_id),
            "--skip-auth",
        ]
        result = run([cfg.cli, *args, "--no-check-update"], 60.0)
        if result.returncode != 0 and "already exists" not in (result.stderr + result.stdout):
            raise NebiusCLIError("profile create", result.returncode, "could not create the CLI profile")
        if s3_client is None:
            import boto3
            from botocore.config import Config

            s3_client = boto3.client(
                "s3",
                endpoint_url=cfg.s3_endpoint,
                region_name=cfg.s3_region,
                aws_access_key_id=creds.s3_access_key_id,
                aws_secret_access_key=creds.s3_secret_access_key,
                config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
            )
        return cls(cfg, cli, s3_client, job_env)

    def job_spec(self, run_id: str, plan_sha256: str) -> JobSpec:
        from rtt.cloud.runs import split_image_ref

        assert self.cfg.image is not None
        image_name, digest = split_image_ref(self.cfg.image)
        name = job_name(run_id)
        return JobSpec(
            name=name,
            image=self.cfg.image,
            command=CONTAINER_PYTHON,
            args=[
                "-m",
                JOB_MODULE,
                "--run-dir",
                f"{MOUNT}/{prefix(run_id)}",
                "--plan-sha256",
                plan_sha256,
                "--deadline-s",
                str(int(self.cfg.deadline_s)),
            ],
            env={
                "RTT_RUN_ID": run_id,
                "RTT_JOB_NAME": name,
                "RTT_IMAGE_NAME": image_name,
                "RTT_IMAGE_DIGEST": digest,
                "RTT_PLATFORM": self.cfg.platform,
                "RTT_PRESET": self.cfg.preset,
                "RTT_REGION": self.cfg.region,
                "RTT_LAUNCHER": LAUNCHER,
                **self.job_env,
            },
            env_secrets={},
            volumes=[f"{self.cfg.bucket_id}:{MOUNT}:rw"],
            platform=self.cfg.platform,
            preset=self.cfg.preset,
            timeout=self.cfg.job_timeout,
            restart_policy="never",
            subnet_id=self.cfg.subnet_id,
            parent_id=self.cfg.project_id,
        )

    def submit(self, run_id: str, plan_json: bytes) -> Submitted:
        """Upload the plan, then create the job. A failure before ``create`` was sent raises
        :class:`SubmitNotStarted` (no job can exist); a refusal of ``create`` raises
        :class:`SubmitRefused`; any other failure is raised as it is, and the job may or may not
        exist."""
        sha = hashlib.sha256(plan_json).hexdigest()
        try:
            spec = self.job_spec(run_id, sha)  # validated before anything is written
            self.s3.put_object(
                Bucket=self.cfg.bucket_name,
                Key=f"{prefix(run_id)}/input/plan.json",
                Body=plan_json,
                ContentType="application/json",
            )
        except Exception as exc:  # noqa: BLE001
            raise SubmitNotStarted(type(exc).__name__) from None
        try:
            job = self.jobs.submit(spec)
        except NebiusCLIError as exc:
            code = _refused(exc)
            if code is not None:
                raise SubmitRefused(code) from None
            raise
        return Submitted(job.id, job.name, job.created_at)

    def find(self, run_id: str) -> Submitted | None:
        """The run's job by name; ``None`` only when the platform says it does not exist. Any
        other failure (timeout, unavailable) is raised: the caller must not read it as absence."""
        try:
            job = self.jobs.get_by_name(job_name(run_id))
        except NebiusCLIError as exc:
            if _not_found(exc):
                return None
            raise
        return Submitted(job.id, job.name, job.created_at) if job.id else None

    def status(self, job_id: str) -> JobStatus:
        return JobStatus.of(self.jobs.get(job_id))

    def cancel(self, job_id: str) -> None:
        self.jobs.cancel(job_id)

    def _get_json(self, key: str) -> dict[str, Any] | None:
        try:
            obj = self.s3.get_object(Bucket=self.cfg.bucket_name, Key=key)
        except Exception:  # noqa: BLE001 - missing or unreadable is the same answer here
            return None
        body = obj["Body"].read(_MAX_SUMMARY_BYTES + 1)
        if len(body) > _MAX_SUMMARY_BYTES:
            return None
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def read_result(self, run_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        manifest = self._get_json(f"{prefix(run_id)}/manifest.json")
        summary = self._get_json(f"{prefix(run_id)}/summary.json")
        return manifest, summary

    def report_url(self, run_id: str, ttl_s: int) -> str | None:
        try:
            return str(
                self.s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.cfg.bucket_name, "Key": f"{prefix(run_id)}/report.json"},
                    ExpiresIn=int(ttl_s),
                )
            )
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------- the fake


@dataclass
class _FakeJob:
    job_id: str
    run_id: str
    created_at: datetime
    state: str = "PROVISIONING"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancelled: bool = False
    plan: dict[str, Any] = field(default_factory=dict)


class FakeNebius:
    """Jobs in memory. Tests move them with :meth:`set_state`; with ``timeline`` (seconds in
    ``PROVISIONING``, then in ``RUNNING``) they move by themselves and finish with ``summary_for``.

    ``fail_submit``: submissions fail before ``create`` (:class:`SubmitNotStarted`);
    ``fail_create``: ``create`` times out and no job exists, though the caller cannot know;
    ``refuse_create``: the platform refuses ``create`` (:class:`SubmitRefused`), no job exists;
    ``fail_after_create``: the job is created, then the call raises (a timeout after ``create``);
    ``fail_find``, ``fail_status`` and ``fail_cancel``: those calls raise (the platform is
    unreachable).
    """

    def __init__(
        self,
        clock: Callable[[], datetime] | None = None,
        *,
        timeline: tuple[float, float] | None = None,
        summary_for: Callable[[str, Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.clock = clock or (lambda: datetime.now(UTC))
        self.timeline = timeline
        self.summary_for = summary_for
        self.jobs: dict[str, _FakeJob] = {}
        self.results: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
        self.cancels: list[str] = []
        self.submissions: list[tuple[str, bytes]] = []
        self.fail_submit = False
        self.fail_create = False
        self.refuse_create = False
        self.fail_after_create = False
        self.fail_find = False
        self.fail_cancel = False
        self.fail_status = False
        self._lock = threading.Lock()

    def submit(self, run_id: str, plan_json: bytes) -> Submitted:
        with self._lock:
            if self.fail_submit:
                raise SubmitNotStarted("scripted failure before create")
            if self.refuse_create:
                raise SubmitRefused("resourceexhausted")
            if self.fail_create:
                raise NebiusCLIError("ai job create", 124, "timed out: the job may or may not exist")
            self.submissions.append((run_id, plan_json))
            job_id = f"aijob-fake{len(self.jobs):04d}"
            self.jobs[job_id] = _FakeJob(job_id, run_id, self.clock(), plan=json.loads(plan_json))
            if self.fail_after_create:
                raise NebiusCLIError("ai job get", 124, "timed out after the job was created")
            return Submitted(job_id, job_name(run_id), self.jobs[job_id].created_at)

    def find(self, run_id: str) -> Submitted | None:
        if self.fail_find:
            raise NebiusCLIError("ai job get-by-name", 1, "rpc error: code = Unavailable")
        for job in self.jobs.values():
            if job.run_id == run_id:
                return Submitted(job.job_id, job_name(run_id), job.created_at)
        return None

    def set_state(self, job_id: str, state: str, at: datetime | None = None) -> None:
        job = self.jobs[job_id]
        now = at or self.clock()
        job.state = state
        if state == "RUNNING" and job.started_at is None:
            job.started_at = now
        if state in TERMINAL_STATES and job.finished_at is None:
            job.finished_at = now

    def _advance(self, job: _FakeJob) -> None:
        if self.timeline is None or job.state in TERMINAL_STATES:
            return
        age = (self.clock() - job.created_at).total_seconds()
        wait, run = self.timeline
        if age >= wait + run:
            job.started_at = job.started_at or job.created_at + timedelta(seconds=wait)
            job.state, job.finished_at = "COMPLETED", job.created_at + timedelta(seconds=wait + run)
            if self.summary_for is not None and job.run_id not in self.results:
                summary = self.summary_for(job.run_id, job.plan)
                finished = job.finished_at.isoformat().replace("+00:00", "Z")
                self.results[job.run_id] = (
                    {"exit_status": 0, "run_id": job.run_id, "finished_at": finished},
                    summary,
                )
        elif age >= wait:
            job.state, job.started_at = "RUNNING", job.created_at + timedelta(seconds=wait)

    def status(self, job_id: str) -> JobStatus:
        with self._lock:
            if self.fail_status:
                raise NebiusCLIError("ai job get", 1, "scripted failure")
            job = self.jobs[job_id]
            self._advance(job)
            return JobStatus(job.state, job.created_at, job.started_at, job.finished_at)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            if self.fail_cancel:
                raise NebiusCLIError("ai job cancel", 1, "rpc error: code = Unavailable desc = scripted")
            self.cancels.append(job_id)
            self.jobs[job_id].cancelled = True

    def read_result(self, run_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        return self.results.get(run_id, (None, None))

    def report_url(self, run_id: str, ttl_s: int) -> str | None:
        return f"https://storage.example.invalid/{prefix(run_id)}/report.json?expires={ttl_s}"
