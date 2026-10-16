"""Typed wrapper over the ``nebius`` CLI for Serverless AI Jobs.

Every call shells out to ``nebius ... --format json`` under the configured profile and
parses the result, so this module needs no SDK and sees exactly what the CLI sees.

* :class:`JobSpec` describes a job; :meth:`JobSpec.create_args` renders the
  ``nebius ai job create`` flags (image by digest, ``--env-secret`` for credentials,
  bucket volume by ID, platform/preset, timeout, ``--restart-policy never``).
* :class:`Target` is a region profile: where a job runs and on what. ``uks2-cpu``
  (:data:`DEFAULT_TARGET`, CPU-only ``cpu-d3`` in uk-south2, the executor of record for
  scored jobs) and ``eun1-gpu`` (:data:`FALLBACK_TARGET`, single-GPU H100/H200/L40S in
  eu-north1, used only when uk-south2 cannot run the job). :func:`cpu_preset` maps a
  vCPU count to the CPU preset name.
* :class:`NebiusJobs` submits, polls, streams logs, cancels and lists jobs, reads the
  tenant's capacity advice (GPU platforms) and vCPU quota (CPU platforms; capacity advice
  has no CPU rows), and :meth:`NebiusJobs.run` walks a platform fallback list: a job
  stuck before start for longer than ``provisioning_timeout_s`` is cancelled and the
  next available platform is tried. Each attempt is appended to a
  :class:`~rtt.cloud.cost.SpendLedger`, and the ledger's cap is enforced before every
  submission and while a job runs, across processes (each job holds a reservation in
  the ledger while it is in flight).

Credentials never pass through here: the Token Factory key reaches a job only as a
MysteryBox reference (``--env-secret``), and :class:`JobSpec` refuses plain ``--env``
values whose names look like secrets.
"""

from __future__ import annotations

import contextlib
import json
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from rtt.cloud.config import NebiusConfig
from rtt.cloud.cost import (
    JobSpend,
    SpendCapError,
    SpendLedger,
    estimate_usd,
    hourly_usd,
    is_cpu_platform,
    preset_resources,
)

# --------------------------------------------------------------------------- states

PRE_START_STATES = frozenset({"CREATING", "PENDING", "PROVISIONING", "STARTING", "IMAGE_PULLING"})
"""States in which no container runs yet; time here counts toward the provisioning timeout.

The CLI's job state enum (``nebius.ai.v1.JobStatus.State``) has ``PROVISIONING``,
``STARTING`` and ``IMAGE_PULLING`` before ``RUNNING``; a job stuck pulling its image must
time out like one stuck waiting for capacity. ``CREATING``/``PENDING`` are kept for
older CLI versions."""
TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "ERROR", "DELETING"})
SUCCESS_STATE = "COMPLETED"

_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.IGNORECASE)
_DURATION = re.compile(r"^(\d+)h$")
_JOB_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")


class NebiusCLIError(RuntimeError):
    """The ``nebius`` CLI exited non-zero or printed something unparseable."""

    def __init__(self, command: str, returncode: int, detail: str) -> None:
        self.command = command
        self.returncode = returncode
        self.detail = detail
        super().__init__(f"`nebius {command}` failed (exit {returncode}): {detail}")


class NoCapacityError(RuntimeError):
    """No candidate could run the job: none had capacity, or every attempt timed out before start.

    Raised before any of our code ran, so the caller may try another :class:`Target`.
    ``attempts`` holds the spend of the cancelled attempts (already in the ledger).
    """

    def __init__(self, message: str, attempts: tuple[JobSpend, ...] = ()) -> None:
        super().__init__(message)
        self.attempts = attempts


class ProvisioningTimeoutError(RuntimeError):
    """A job did not leave the pre-start states within the allowed time."""

    def __init__(self, job: Job, waited_s: float) -> None:
        self.job = job
        self.waited_s = waited_s
        super().__init__(f"job {job.name} still {job.state} after {waited_s:.0f} s on {job.platform}")


# --------------------------------------------------------------------------- CLI runner


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str], float | None], CommandResult]
"""Executes an argv with a timeout in seconds; tests substitute a fake."""


def subprocess_runner(argv: Sequence[str], timeout_s: float | None) -> CommandResult:
    """Run ``argv`` and capture its output (no shell)."""
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(124, "", f"timed out after {exc.timeout} s")
    return CommandResult(proc.returncode, proc.stdout, proc.stderr)


class NebiusCLI:
    """Invokes the ``nebius`` binary with a fixed profile.

    Args:
        binary: Path of the ``nebius`` executable.
        profile: CLI profile (``--profile``); ``None`` uses the CLI's default profile.
        runner: Command executor (default :func:`subprocess_runner`).
        timeout_s: Default per-call timeout.
    """

    def __init__(
        self,
        binary: str = "nebius",
        profile: str | None = None,
        runner: Runner = subprocess_runner,
        timeout_s: float = 300.0,
    ) -> None:
        self.binary = binary
        self.profile = profile
        self.runner = runner
        self.timeout_s = timeout_s

    def __repr__(self) -> str:
        return f"NebiusCLI(binary={self.binary!r}, profile={self.profile!r})"

    def argv(self, args: Sequence[str], fmt: str | None = "json") -> list[str]:
        out = [self.binary, *args]
        if fmt:
            out += ["--format", fmt]
        if self.profile:
            out += ["--profile", self.profile]
        return out + ["--no-check-update"]

    def run(self, args: Sequence[str], fmt: str | None = "json", timeout_s: float | None = None) -> str:
        """Run ``nebius <args>`` and return stdout; raise :class:`NebiusCLIError` on failure."""
        result = self.runner(self.argv(args, fmt), timeout_s if timeout_s is not None else self.timeout_s)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1500:]
            raise NebiusCLIError(_command_label(args), result.returncode, detail or "no output")
        return result.stdout

    def run_json(self, args: Sequence[str], timeout_s: float | None = None) -> Any:
        """Run with ``--format json`` and parse stdout."""
        out = self.run(args, "json", timeout_s)
        try:
            return json.loads(out) if out.strip() else {}
        except json.JSONDecodeError as exc:
            raise NebiusCLIError(_command_label(args), 0, f"unparseable JSON output: {exc}") from exc


def _command_label(args: Sequence[str]) -> str:
    """The subcommand words of ``args`` (stops at the first flag, so no values leak)."""
    words = []
    for a in args:
        if a.startswith("-"):
            break
        words.append(a)
    return " ".join(words)


# --------------------------------------------------------------------------- job model


def _parse_time(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


@dataclass(frozen=True)
class Job:
    """A job as returned by ``nebius ai job get``/``list`` (the raw JSON is kept)."""

    id: str
    name: str
    state: str
    platform: str | None = None
    preset: str | None = None
    image: str | None = field(default=None, repr=False)  # its path holds the registry ID
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    state_details: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Job:
        meta = data.get("metadata") or {}
        spec = data.get("spec") or {}
        status = data.get("status") or {}
        details = status.get("state_details")
        if isinstance(details, Mapping):
            details = json.dumps(details, sort_keys=True)
        return cls(
            id=str(meta.get("id", "")),
            name=str(meta.get("name", "")),
            state=str(status.get("state", "UNKNOWN")).upper(),
            platform=spec.get("platform"),
            preset=spec.get("preset"),
            image=spec.get("image"),
            created_at=_parse_time(meta.get("created_at")),
            started_at=_parse_time(status.get("started_at")),
            finished_at=_parse_time(status.get("finished_at")),
            state_details=str(details) if details else None,
            raw=dict(data),
        )

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def succeeded(self) -> bool:
        return self.state == SUCCESS_STATE

    @property
    def provisioning_s(self) -> float | None:
        """Creation to container start."""
        return _seconds(self.created_at, self.started_at)

    @property
    def run_s(self) -> float | None:
        """Container start to finish."""
        return _seconds(self.started_at, self.finished_at)

    def wall_s(self, now: datetime | None = None) -> float:
        """Creation to finish (or to ``now`` while the job is still active)."""
        end = self.finished_at or now or datetime.now(UTC)
        return _seconds(self.created_at, end) or 0.0


@dataclass(frozen=True)
class JobSpec:
    """Arguments for ``nebius ai job create``.

    ``env_secrets`` maps a container variable to a MysteryBox secret selector; it is the
    only way credentials reach a job. ``volumes`` use ``SOURCE:/container/path[:mode]``
    with the bucket *ID* as source (a bucket name is rejected by the API).
    """

    name: str
    image: str
    command: str | None = None
    args: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    env_secrets: Mapping[str, str] = field(default_factory=dict)
    volumes: Sequence[str] = ()
    platform: str = "gpu-h100-sxm"
    preset: str | None = "1gpu-16vcpu-200gb"
    timeout: str = "1h"
    restart_policy: str = "never"
    subnet_id: str | None = None
    parent_id: str | None = None
    disk_size: str | None = None
    preemptible: bool = False
    working_dir: str | None = None
    allow_tag: bool = False  # permit a tag instead of a digest (smoke tests of public images)

    def __repr__(self) -> str:
        """Names, platform and env keys only: the image path, secret selectors, bucket
        volume, subnet and project are account IDs and never appear in logs or tracebacks."""
        return (
            f"JobSpec(name={self.name!r}, image=<set>, platform={self.platform!r}, preset={self.preset!r},"
            f" env={sorted(self.env)!r}, env_secrets={sorted(self.env_secrets)!r},"
            f" volumes={len(self.volumes)}, timeout={self.timeout!r}, preemptible={self.preemptible!r})"
        )

    def validate(self) -> None:
        """Raise ``ValueError`` for a spec the platform or our hygiene rules would reject."""
        if not _JOB_NAME.match(self.name):
            raise ValueError(f"job name {self.name!r}: use 1-63 lowercase letters, digits and dashes")
        if "@sha256:" not in self.image and not self.allow_tag:
            # only the last path part: the registry path is an account ID
            raise ValueError(
                f"image must be pinned by digest (…@sha256:…), got …/{self.image.rsplit('/', 1)[-1]}"
            )
        for key in self.env:
            if _SECRET_NAME.search(key):
                raise ValueError(f"env {key!r} looks like a credential; pass it with env_secrets instead")
        match = _DURATION.match(self.timeout)
        if not match or not 1 <= int(match.group(1)) <= 168:
            raise ValueError(f"timeout must be whole hours between 1h and 168h, got {self.timeout!r}")
        if self.restart_policy not in ("never", "on-failure"):
            raise ValueError(f"restart_policy must be 'never' or 'on-failure', got {self.restart_policy!r}")
        for volume in self.volumes:
            if ":" not in volume:
                # the source is a bucket ID, so the value itself is not shown
                raise ValueError("a volume has no ':': expected SOURCE:/container/path[:mode]")

    def create_args(self, dry_run: bool = False) -> list[str]:
        """The ``ai job create`` argv (without the binary, ``--format`` or ``--profile``)."""
        self.validate()
        argv = ["ai", "job", "create", "--name", self.name, "--image", self.image]
        if self.command:
            argv += ["--container-command", self.command]
        if self.args:
            argv += ["--args", shlex.join(self.args)]
        for key, value in self.env.items():
            argv += ["--env", f"{key}={value}"]
        for key, selector in self.env_secrets.items():
            argv += ["--env-secret", f"{key}={selector}"]
        for volume in self.volumes:
            argv += ["--volume", volume]
        argv += ["--platform", self.platform]
        if self.preset:
            argv += ["--preset", self.preset]
        argv += ["--timeout", self.timeout, "--restart-policy", self.restart_policy]
        if self.subnet_id:
            argv += ["--subnet-id", self.subnet_id]
        if self.parent_id:
            argv += ["--parent-id", self.parent_id]
        if self.disk_size:
            argv += ["--disk-size", self.disk_size]
        if self.working_dir:
            argv += ["--working-dir", self.working_dir]
        if self.preemptible:
            argv += ["--preemptible", "--follows-spot-price"]
        argv += ["--dry-run"] if dry_run else ["--async"]
        return argv


# --------------------------------------------------------------------------- capacity

AVAILABILITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "LIMIT_REACHED": 0, "UNSPECIFIED": 0}


@dataclass(frozen=True)
class Candidate:
    """A platform and preset to try, in a region."""

    platform: str
    preset: str
    region: str = "eu-north1"

    @property
    def is_cpu(self) -> bool:
        """A CPU-only platform (no GPU; capacity is governed by the vCPU quota)."""
        return is_cpu_platform(self.platform)

    @property
    def vcpus(self) -> int | None:
        """vCPUs of the preset (``16vcpu-64gb`` -> 16)."""
        resources = preset_resources(self.preset)
        return resources[0] if resources else None


GPU_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("gpu-h100-sxm", "1gpu-16vcpu-200gb"),
    Candidate("gpu-h200-sxm", "1gpu-16vcpu-200gb"),
    Candidate("gpu-l40s-a", "1gpu-8vcpu-32gb"),
)
"""Fallback order for single-GPU jobs next to the eu-north1 bucket."""

DEFAULT_CANDIDATES = GPU_CANDIDATES
"""Kept for callers of the GPU-only API; new code picks a :class:`Target`."""

CPU_VCPU_CHOICES: tuple[int, ...] = (2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256)
"""vCPU counts of the ``cpu-d3`` presets (``nebius compute platform list``, uk-south2)."""
CPU_GIB_PER_VCPU = 4
DEFAULT_CPU_VCPUS = 16
"""16 vCPUs: the same CPU count as the 1-GPU H100 preset, so wall times compare directly."""
CPU_VCPU_QUOTA = "compute.instance.non-gpu.vcpu"


def cpu_preset(vcpus: int = DEFAULT_CPU_VCPUS) -> str:
    """The CPU-only preset with ``vcpus`` vCPUs and 4 GiB each, e.g. ``16vcpu-64gb``.

    Raises:
        ValueError: no preset has that many vCPUs.
    """
    if vcpus not in CPU_VCPU_CHOICES:
        raise ValueError(f"no CPU preset with {vcpus} vCPUs; choose one of {CPU_VCPU_CHOICES}")
    return f"{vcpus}vcpu-{vcpus * CPU_GIB_PER_VCPU}gb"


@dataclass(frozen=True)
class Target:
    """A region profile: the region a job runs in and the platforms to try there, in order.

    Each target uses that region's project, subnet and bucket
    (:meth:`rtt.cloud.config.NebiusConfig.for_region`).
    """

    name: str
    region: str
    kind: str  # "cpu" | "gpu"
    candidates: tuple[Candidate, ...]
    description: str = ""

    def with_vcpus(self, vcpus: int) -> Target:
        """A CPU target whose candidates all use the ``vcpus`` preset (GPU targets unchanged)."""
        if self.kind != "cpu":
            return self
        preset = cpu_preset(vcpus)
        return replace(self, candidates=tuple(replace(c, preset=preset) for c in self.candidates))

    def only(self, platform: str, preset: str | None = None) -> Target:
        """This target restricted to one platform (no fallback within the region)."""
        if preset is None:
            match = [c for c in self.candidates if c.platform == platform]
            if not match:
                raise ValueError(f"{platform} is not a platform of target {self.name}; pass a preset")
            preset = match[0].preset
        return replace(self, candidates=(Candidate(platform, preset, self.region),))


TARGETS: dict[str, Target] = {
    t.name: t
    for t in (
        Target(
            "uks2-cpu",
            "uk-south2",
            "cpu",
            (Candidate("cpu-d3", cpu_preset(DEFAULT_CPU_VCPUS), "uk-south2"),),
            "CPU-only AMD EPYC Genoa in uk-south2: the executor of record for scored jobs",
        ),
        Target(
            "eun1-gpu",
            "eu-north1",
            "gpu",
            GPU_CANDIDATES,
            "single-GPU H100/H200/L40S in eu-north1: fallback (eu-north1 has no CPU-only quota)",
        ),
    )
}
DEFAULT_TARGET = "uks2-cpu"
FALLBACK_TARGET = "eun1-gpu"


def get_target(name: str, vcpus: int | None = None) -> Target:
    """The named target, with its CPU preset set to ``vcpus`` when given.

    Raises:
        ValueError: unknown target or unsupported vCPU count.
    """
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r}; known: {', '.join(TARGETS)}")
    target = TARGETS[name]
    return target.with_vcpus(vcpus) if vcpus is not None else target


def target_for_platform(platform: str) -> Target:
    """The first target that lists ``platform`` (CPU platforms map to the CPU target)."""
    for target in TARGETS.values():
        if any(c.platform == platform for c in target.candidates):
            return target
    if is_cpu_platform(platform):
        return TARGETS[DEFAULT_TARGET]
    raise ValueError(f"no target runs platform {platform!r}; known targets: {', '.join(TARGETS)}")


@dataclass(frozen=True)
class QuotaAllowance:
    """One quota in one region (``nebius quotas quota-allowance get-by-name``)."""

    name: str
    region: str
    limit: int
    usage: int

    @property
    def free(self) -> int:
        return max(0, self.limit - self.usage)

    @classmethod
    def from_json(cls, data: Mapping[str, Any], name: str, region: str) -> QuotaAllowance:
        spec = data.get("spec") or {}
        status = data.get("status") or {}
        limit = int(spec.get("limit") or 0)
        usage = status.get("usage")
        if usage is None:  # the API omits zero usage; derive from the percentage if present
            pct = float(status.get("usage_percentage") or 0.0)
            usage = round(limit * pct / 100.0)
        return cls(name, str(spec.get("region") or region), limit, int(usage))


@dataclass(frozen=True)
class Availability:
    level: str  # HIGH | MEDIUM | LOW | LIMIT_REACHED | UNSPECIFIED
    available: int | None
    limit: int | None

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> Availability:
        data = data or {}
        level = str(data.get("availability_level", "UNSPECIFIED")).removeprefix("AVAILABILITY_LEVEL_")
        avail = data.get("available")
        limit = data.get("limit")
        return cls(
            level, int(avail) if avail is not None else None, int(limit) if limit is not None else None
        )

    def usable(self, min_level: str = "MEDIUM") -> bool:
        """At least one instance is free and the level is ``min_level`` or better."""
        return (self.available or 0) > 0 and AVAILABILITY_RANK.get(self.level, 0) >= AVAILABILITY_RANK[
            min_level
        ]


@dataclass(frozen=True)
class CapacityAdvice:
    """One ``capacity resource-advice`` row: a platform/preset in a region (and fabric)."""

    region: str
    fabric: str | None
    platform: str
    preset: str
    on_demand: Availability
    preemptible: Availability

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CapacityAdvice | None:
        spec = data.get("spec") or {}
        inst = spec.get("compute_instance")
        if not inst:
            return None
        preset = inst.get("preset") or {}
        status = data.get("status") or {}
        return cls(
            region=str(spec.get("region", "")),
            fabric=spec.get("fabric"),
            platform=str(inst.get("platform", "")),
            preset=str(preset.get("name", "")),
            on_demand=Availability.from_json(status.get("on_demand")),
            preemptible=Availability.from_json(status.get("preemptible")),
        )

    def availability(self, preemptible: bool = False) -> Availability:
        return self.preemptible if preemptible else self.on_demand


def parse_capacity(data: Mapping[str, Any]) -> list[CapacityAdvice]:
    """Rows of ``nebius capacity resource-advice list --format json`` (non-compute rows dropped)."""
    rows = (CapacityAdvice.from_json(item) for item in data.get("items", []))
    return [r for r in rows if r is not None]


def rank_candidates(
    advice: Iterable[CapacityAdvice],
    candidates: Sequence[Candidate] = DEFAULT_CANDIDATES,
    preemptible: bool = False,
    min_level: str = "MEDIUM",
    cpu_free_vcpus: Mapping[str, int] | None = None,
) -> list[Candidate]:
    """The candidates with usable capacity, in fallback order.

    A GPU candidate is usable when any fabric in its region reports a free instance at
    ``min_level`` or better for the requested allocation type. Capacity advice has no
    rows for CPU-only platforms, so a CPU candidate is usable when ``cpu_free_vcpus``
    (region -> free ``compute.instance.non-gpu.vcpu`` quota) covers its preset.
    """
    rows = list(advice)
    free = cpu_free_vcpus or {}
    out = []
    for cand in candidates:
        if cand.is_cpu:
            if (cand.vcpus or 0) <= free.get(cand.region, 0):
                out.append(cand)
        elif any(
            r.platform == cand.platform
            and r.preset == cand.preset
            and r.region == cand.region
            and r.availability(preemptible).usable(min_level)
            for r in rows
        ):
            out.append(cand)
    return out


# --------------------------------------------------------------------------- service


@dataclass(frozen=True)
class JobUpdate:
    """Passed to ``on_update`` callbacks while waiting."""

    job: Job
    elapsed_s: float


@dataclass(frozen=True)
class RunOutcome:
    """Result of :meth:`NebiusJobs.run`: the final job and every attempt's spend."""

    job: Job
    candidate: Candidate
    attempts: tuple[JobSpend, ...]

    @property
    def est_usd(self) -> float:
        return sum(a.est_usd for a in self.attempts)


def _extract_job_id(data: Any) -> str | None:
    """The ``aijob-…`` ID in a create/operation response, wherever the CLI put it."""
    if isinstance(data, Mapping):
        for key in ("resource_id", "id"):
            value = data.get(key)
            if isinstance(value, str) and value.startswith("aijob-"):
                return value
        for value in data.values():
            found = _extract_job_id(value)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _extract_job_id(value)
            if found:
                return found
    return None


def _budget_limit(
    ledger: SpendLedger | None, token: str | None, worst_usd: float, usd_per_s: float
) -> Callable[[float], float | None] | None:
    """The wall-time limit of a job holding reservation ``token``, re-read at every poll.

    Each call first raises the reservation to the job's live spend (when that exceeds
    the worst case reserved at submission), so other processes see it, then returns the
    seconds that the cap minus recorded spend minus every *other* reservation pays for.
    ``None`` (no limit) without a capped ledger.
    """
    if ledger is None or token is None or ledger.cap_usd is None or usd_per_s <= 0:
        return None

    def limit(elapsed_s: float) -> float | None:
        live = elapsed_s * usd_per_s
        if live > worst_usd:
            ledger.hold(token, live)
        remaining = ledger.remaining_usd(exclude=token)
        return None if remaining is None else remaining / usd_per_s

    return limit


class NebiusJobs:
    """Serverless AI Jobs through the CLI.

    Args:
        cli: Configured :class:`NebiusCLI`.
        config: Account settings (tenant for capacity advice, project, subnet, bucket).
        sleep: Sleep function (tests pass a fake).
        clock: Monotonic clock in seconds (tests pass a fake).
    """

    def __init__(
        self,
        cli: NebiusCLI,
        config: NebiusConfig,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cli = cli
        self.config = config
        self.sleep = sleep
        self.clock = clock

    @classmethod
    def from_config(cls, config: NebiusConfig, runner: Runner = subprocess_runner) -> NebiusJobs:
        return cls(NebiusCLI(config.cli, config.profile, runner), config)

    # -- single calls

    def submit(self, spec: JobSpec) -> Job:
        """Create the job (asynchronously) and return its first observed state."""
        data = self.cli.run_json(spec.create_args())
        job_id = _extract_job_id(data)
        if job_id is None:
            job = self.get_by_name(spec.name)
        else:
            job = self.get(job_id)
        return job

    def dry_run(self, spec: JobSpec) -> str:
        """Validate ``spec`` server-side without creating anything; returns the CLI message."""
        return self.cli.run(spec.create_args(dry_run=True), fmt=None).strip()

    def get(self, job_id: str) -> Job:
        return Job.from_json(self.cli.run_json(["ai", "job", "get", "--id", job_id]))

    def get_by_name(self, name: str) -> Job:
        """Look a job up by name in the configured project (needs ``NEBIUS_PROJECT_ID``)."""
        self.config.require("project_id")
        assert self.config.project_id is not None
        args = ["ai", "job", "get-by-name", "--name", name, "--parent-id", self.config.project_id]
        return Job.from_json(self.cli.run_json(args))

    def status(self, job_id: str) -> str:
        return self.get(job_id).state

    def list_jobs(self) -> list[Job]:
        args = ["ai", "job", "list"]
        if self.config.project_id:
            args += ["--parent-id", self.config.project_id]
        data = self.cli.run_json(args)
        return [Job.from_json(item) for item in data.get("items", [])]

    def logs(self, job_id: str, since: str = "24h", tail: int | None = None, timestamps: bool = False) -> str:
        """The job's container logs as text."""
        args = ["ai", "job", "logs", job_id, "--since", since]
        if tail is not None:
            args += ["--tail", str(tail)]
        if timestamps:
            args.append("--timestamps")
        return self.cli.run(args, fmt=None)

    def cancel(self, job_id: str) -> None:
        """Request cancellation and return at once (the CLI's blocking cancel can take minutes)."""
        self.cli.run(["ai", "job", "cancel", "--id", job_id, "--async"], fmt="json")

    def capacity(self) -> list[CapacityAdvice]:
        """The tenant's capacity advice (needs ``NEBIUS_TENANT_ID``)."""
        self.config.require("tenant_id")
        assert self.config.tenant_id is not None
        data = self.cli.run_json(
            ["capacity", "resource-advice", "list", "--parent-id", self.config.tenant_id, "--all"]
        )
        return parse_capacity(data)

    def quota(self, name: str, region: str | None = None) -> QuotaAllowance:
        """One tenant quota in ``region`` (default: the config's region)."""
        self.config.require("tenant_id")
        assert self.config.tenant_id is not None
        reg = region or self.config.region
        args = ["quotas", "quota-allowance", "get-by-name", "--parent-id", self.config.tenant_id]
        data = self.cli.run_json([*args, "--name", name, "--region", reg])
        return QuotaAllowance.from_json(data, name, reg)

    def cpu_free_vcpus(self, region: str | None = None) -> int:
        """Free ``compute.instance.non-gpu.vcpu`` quota in ``region`` (0 when the quota is absent)."""
        try:
            return self.quota(CPU_VCPU_QUOTA, region).free
        except NebiusCLIError as exc:
            if "not found" in exc.detail.lower() or "notfound" in exc.detail.lower():
                return 0
            raise

    def usable_candidates(
        self, candidates: Sequence[Candidate], preemptible: bool = False, min_level: str = "MEDIUM"
    ) -> list[Candidate]:
        """``candidates`` with capacity now: advice for GPU platforms, quota for CPU ones.

        Each source is queried only when some candidate needs it.
        """
        advice = self.capacity() if any(not c.is_cpu for c in candidates) else []
        free = {r: self.cpu_free_vcpus(r) for r in sorted({c.region for c in candidates if c.is_cpu})}
        return rank_candidates(advice, candidates, preemptible, min_level, free)

    # -- waiting

    def wait(
        self,
        job_id: str,
        poll_s: float = 15.0,
        provisioning_timeout_s: float | None = 900.0,
        max_wall_s: float | Callable[[float], float | None] | None = None,
        on_update: Callable[[JobUpdate], None] | None = None,
        max_poll_errors: int = 5,
    ) -> Job:
        """Poll until the job is terminal.

        A failed ``get`` (network blip, CLI token refresh) is retried on the next poll;
        only ``max_poll_errors`` failures in a row abort the wait, because giving up
        leaves a billed job running unobserved.

        ``max_wall_s`` is a fixed limit, or a function of the elapsed seconds that returns
        the current limit at every poll (:meth:`run` uses one so that jobs of other
        processes sharing the spend ledger shrink this job's allowance as they spend).

        Raises:
            ProvisioningTimeoutError: the job spent ``provisioning_timeout_s`` in
                :data:`PRE_START_STATES` (the caller decides whether to cancel).
            SpendCapError: the job ran past ``max_wall_s``, or a callable ``max_wall_s``
                raised (the cap can no longer be checked); the job is cancelled first.
            NebiusCLIError: ``max_poll_errors`` polls in a row failed.
        """
        t0 = self.clock()
        last_state = None
        poll_errors = 0
        while True:
            try:
                job = self.get(job_id)
            except NebiusCLIError:
                poll_errors += 1
                if poll_errors >= max_poll_errors:
                    raise
                self.sleep(poll_s)
                continue
            poll_errors = 0
            elapsed = self.clock() - t0
            if on_update is not None and job.state != last_state:
                on_update(JobUpdate(job, elapsed))
            last_state = job.state
            if job.terminal:
                return job
            if (
                provisioning_timeout_s is not None
                and job.state in PRE_START_STATES
                and elapsed >= provisioning_timeout_s
            ):
                raise ProvisioningTimeoutError(job, elapsed)
            try:
                limit = max_wall_s(elapsed) if callable(max_wall_s) else max_wall_s
            except Exception as exc:
                # Fail closed: when the budget can no longer be read (an unreadable in-flight
                # file, a ledger write error), the cap cannot be enforced, so the job stops
                # instead of running on unwatched until its timeout.
                self.cancel(job_id)
                raise SpendCapError(
                    f"job {job.name} cancelled after {elapsed:.0f} s: the spend cap could not be"
                    f" checked ({exc.__class__.__name__}: {exc})"
                ) from exc
            if limit is not None and elapsed >= limit:
                self.cancel(job_id)
                raise SpendCapError(
                    f"job {job.name} cancelled after {elapsed:.0f} s: the spend cap allows {max(0.0, limit):.0f} s"
                )
            self.sleep(poll_s)

    def wait_cancelled(self, job_id: str, poll_s: float = 10.0, timeout_s: float = 300.0) -> Job:
        """Poll a cancelled job until it is terminal (or ``timeout_s`` passes)."""
        t0 = self.clock()
        job = self.get(job_id)
        while not job.terminal and self.clock() - t0 < timeout_s:
            self.sleep(poll_s)
            job = self.get(job_id)
        return job

    # -- orchestration

    def run(
        self,
        spec: JobSpec,
        candidates: Sequence[Candidate] = DEFAULT_CANDIDATES,
        ledger: SpendLedger | None = None,
        expected_run_s: float = 600.0,
        provisioning_timeout_s: float = 900.0,
        poll_s: float = 15.0,
        check_capacity: bool = True,
        min_level: str = "MEDIUM",
        run_id: str | None = None,
        on_update: Callable[[JobUpdate], None] | None = None,
    ) -> RunOutcome:
        """Submit ``spec`` on the first available candidate and wait for it to finish.

        Each candidate gets the platform/preset written into the spec (and into the
        ``RTT_PLATFORM``/``RTT_PRESET``/``RTT_REGION`` env of the container). A job stuck
        before start for ``provisioning_timeout_s`` is cancelled and the next candidate is
        tried. With a capped ``ledger``, a submission is refused unless the worst case
        (provisioning timeout plus ``expected_run_s``) fits next to the recorded spend and
        every other job in flight, and that worst case stays reserved in the ledger while
        the job runs (raised to the job's live spend when it runs longer). A running job is
        cancelled when its wall time reaches what the budget left by everything else pays
        for, re-read at every poll, so concurrent ``rtt job`` processes cannot together
        pass the cap.

        Raises:
            NoCapacityError: no candidate had capacity or every attempt timed out.
            SpendCapError: see above.
        """
        if check_capacity:
            ranked = self.usable_candidates(candidates, spec.preemptible, min_level)
        else:
            ranked = list(candidates)
        ranked = [c for c in ranked if c.region == self.config.region]
        if not ranked:
            names = ", ".join(f"{c.platform}/{c.preset}" for c in candidates)
            raise NoCapacityError(f"no capacity in {self.config.region} for any of: {names}")

        attempts: list[JobSpend] = []
        for i, cand in enumerate(ranked):
            usd_per_s = hourly_usd(cand.platform, cand.preset) / 3600.0
            worst = (provisioning_timeout_s + expected_run_s) * usd_per_s
            name = spec.name if i == 0 else f"{spec.name[:58]}-r{i}"
            token = ledger.reserve(worst, name) if ledger is not None else None
            try:
                max_wall_s = _budget_limit(ledger, token, worst, usd_per_s)
                env = dict(spec.env)
                env.update(RTT_PLATFORM=cand.platform, RTT_PRESET=cand.preset, RTT_REGION=cand.region)
                if "RTT_JOB_NAME" in env:
                    env["RTT_JOB_NAME"] = name  # the manifest must name the attempt that ran
                attempt = replace(spec, name=name, platform=cand.platform, preset=cand.preset, env=env)
                job = self.submit(attempt)
                try:
                    final = self.wait(job.id, poll_s, provisioning_timeout_s, max_wall_s, on_update)
                except ProvisioningTimeoutError as exc:
                    self.cancel(job.id)
                    final = self.wait_cancelled(job.id)
                    attempts.append(self._spend(final, cand, attempt, run_id, f"cancelled: {exc}"))
                    if ledger is not None:
                        ledger.append(attempts[-1], release=token)
                    continue
                except SpendCapError as exc:
                    final = self.wait_cancelled(job.id)
                    attempts.append(self._spend(final, cand, attempt, run_id, str(exc)))
                    if ledger is not None:
                        ledger.append(attempts[-1], release=token)
                    raise
                except BaseException as exc:
                    # Lost track of a live job (poll failures, Ctrl-C): record what it has cost
                    # so far, so the cap still sees it, and leave it running for the caller.
                    self._record_abandoned(job, cand, attempt, run_id, ledger, exc)
                    raise
                attempts.append(self._spend(final, cand, attempt, run_id, final.state_details))
                if ledger is not None:
                    ledger.append(attempts[-1], release=token)
                return RunOutcome(final, cand, tuple(attempts))
            finally:
                if ledger is not None and token is not None:
                    with contextlib.suppress(Exception):  # never mask the original exception
                        ledger.release(token)  # no-op when append already released it
        raise NoCapacityError(
            f"every candidate timed out in provisioning ({len(attempts)} attempts, "
            f"${sum(a.est_usd for a in attempts):.2f} estimated)",
            tuple(attempts),
        )

    def _record_abandoned(
        self,
        job: Job,
        cand: Candidate,
        spec: JobSpec,
        run_id: str | None,
        ledger: SpendLedger | None,
        exc: BaseException,
    ) -> None:
        """Best effort: append the spend of a job ``run`` stopped watching, then return."""
        if ledger is None:
            return
        try:
            latest = self.get(job.id)
        except Exception:  # noqa: BLE001 - the submit-time snapshot still has created_at
            latest = job
        note = (
            f"abandoned while {latest.state} ({exc.__class__.__name__}); the job may still be"
            f" running: check `python -m rtt.cloud status {job.id}`"
        )
        with contextlib.suppress(Exception):  # never mask the original exception
            ledger.append(self._spend(latest, cand, spec, run_id, note))

    def _spend(
        self, job: Job, cand: Candidate, spec: JobSpec, run_id: str | None, note: str | None
    ) -> JobSpend:
        wall = job.wall_s()
        run = job.run_s
        return JobSpend(
            job_id=job.id,
            name=job.name,
            platform=cand.platform,
            preset=cand.preset,
            region=cand.region,
            state=job.state,
            preemptible=spec.preemptible,
            provisioning_s=job.provisioning_s,
            run_s=run,
            wall_s=wall,
            usd_per_hour=hourly_usd(cand.platform, cand.preset),
            est_usd=round(estimate_usd(cand.platform, cand.preset, wall), 4),
            est_usd_run=round(estimate_usd(cand.platform, cand.preset, run or 0.0), 4),
            run_id=run_id,
            note=note,
        )
