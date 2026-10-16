"""Price table, per-job cost estimates and an append-only spend ledger for Nebius jobs.

GPU platforms are priced per GPU-hour for the on-demand 1-GPU presets as listed on
https://nebius.com/prices (read 2026-10-15). A preset with ``N`` GPUs costs ``N`` times
the GPU-hour price. "from" prices (L40S) are the smallest preset, so estimates for larger
L40S presets are lower bounds; those rows are tagged ``assumed``.

CPU-only platforms are priced per vCPU-hour plus per GiB-of-RAM-hour, from
https://docs.nebius.com/compute/resources/pricing (prices from 2026-10-01, read
2026-10-15). A ``16vcpu-64gb`` preset on ``cpu-d3`` (AMD EPYC Genoa) costs
16 x $0.015 + 64 x $0.0045 = $0.528 per hour. The marketing page's "from $0.13" for Genoa
matches the 4 vCPU / 16 GiB preset under the same rates.

Preemptible VMs follow the spot price, which the page lists only as a floor ("from
$0.79"). Estimates for preemptible jobs therefore use the on-demand price as a ceiling,
so a spend cap enforced on them is conservative.

What a job is billed for is estimated two ways [assumed]:

* ``wall``: from job creation to finish, covering provisioning, image pull and the run.
  This is the upper bound and the number the spend cap is enforced on.
* ``run``: container start to finish. The jobs documentation says a job is "billed only
  while the job is running", so the true charge is expected to be near this lower bound.

The spend cap holds across processes. Every job reserves its worst case in
``<ledger>.inflight.json`` before it is submitted, under an exclusive ``flock`` on
``<ledger>.lock``, and keeps its reservation at least as large as its live spend while it
runs. A second ``rtt job`` started while the first is running therefore sees the first
one's money as spent: it is refused when the two together could pass the cap, and a
running job is cancelled when its own spend reaches the cap minus everything else.
Reservations of processes that died on this host are dropped.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import socket
import threading
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PRICES_SOURCE = (
    "https://nebius.com/prices and https://docs.nebius.com/compute/resources/pricing (read 2026-10-15)"
)


class SpendCapError(RuntimeError):
    """Submitting or continuing a job could push estimated spend past the cap."""


@dataclass(frozen=True)
class Price:
    """On-demand price of one GPU-hour on a platform, with the spot floor where listed."""

    platform: str
    gpu_hour_usd: float
    spot_floor_usd: float | None
    tag: str = "sourced"  # "sourced" (exact list price) or "assumed" (a "from" price or a guess)
    note: str = ""


PRICES: dict[str, Price] = {
    p.platform: p
    for p in (
        Price("gpu-h100-sxm", 4.50, 0.79, note="NVIDIA HGX H100, 16 vCPU, 200 GB per GPU"),
        Price("gpu-h200-sxm", 5.40, 0.79, note="NVIDIA HGX H200, 16 vCPU, 200 GB per GPU"),
        Price("gpu-b200-sxm", 8.50, 0.99, note="NVIDIA HGX B200, 20 vCPU, 224 GB per GPU"),
        Price("gpu-b200-sxm-a", 8.50, 0.99, "assumed", "assumed equal to gpu-b200-sxm"),
        Price("gpu-b300-sxm", 9.50, 0.99, note="NVIDIA HGX B300, 24 vCPU, 346 GB per GPU"),
        Price("gpu-rtx6000", 1.80, 0.79, note="NVIDIA RTX PRO 6000, 24 vCPU, 218 GB per GPU"),
        Price("gpu-rtx6000-a", 1.80, 0.79, "assumed", "assumed equal to gpu-rtx6000"),
        Price("gpu-l40s-a", 1.55, 0.74, "assumed", "L40S with Intel CPU: 'from' price (8 vCPU preset)"),
        Price("gpu-l40s-d", 1.82, 0.90, "assumed", "L40S with AMD CPU: 'from' price (smallest preset)"),
    )
}


@dataclass(frozen=True)
class CpuPrice:
    """On-demand price of a CPU-only platform: per vCPU-hour plus per GiB-of-RAM-hour."""

    platform: str
    vcpu_hour_usd: float
    gib_hour_usd: float
    tag: str = "sourced"
    note: str = ""


CPU_PRICES: dict[str, CpuPrice] = {
    p.platform: p
    for p in (
        CpuPrice("cpu-d3", 0.015, 0.0045, note="Non-GPU AMD EPYC Genoa; all regions; from 2026-10-01"),
        CpuPrice("cpu-e2", 0.012, 0.0045, note="Non-GPU Intel Ice Lake; eu-north1 only; from 2026-10-01"),
    )
}

OBJECT_STORAGE_USD_PER_GIB_MONTH = {"standard": 0.0147, "enhanced_throughput": 0.11}
"""Object Storage volume prices [sourced], https://nebius.com/prices. A run is a few MB,
so storage is left out of job estimates (well under $0.01 per month per run)."""

_GPU_COUNT = re.compile(r"^(\d+)gpu-")
_RESOURCES = re.compile(r"(?:^|-)(\d+)vcpu-(\d+)gb$")


def gpu_count(preset: str | None) -> int:
    """GPUs in a preset name such as ``8gpu-128vcpu-1600gb`` (1 when the name has no count)."""
    match = _GPU_COUNT.match(preset or "")
    return int(match.group(1)) if match else 1


def preset_resources(preset: str | None) -> tuple[int, int] | None:
    """``(vCPUs, GiB)`` of a preset name such as ``16vcpu-64gb`` or ``1gpu-16vcpu-200gb``."""
    match = _RESOURCES.search(preset or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def is_cpu_platform(platform: str) -> bool:
    """Whether ``platform`` is a CPU-only platform (``cpu-d3``, ``cpu-e2``)."""
    return platform.startswith("cpu-")


def hourly_usd(platform: str, preset: str | None = None) -> float:
    """Estimated on-demand USD per hour for ``platform`` / ``preset``.

    Raises:
        KeyError: the platform is in neither :data:`PRICES` nor :data:`CPU_PRICES`.
        ValueError: a CPU platform without a ``<n>vcpu-<m>gb`` preset.
    """
    if platform in CPU_PRICES:
        resources = preset_resources(preset)
        if resources is None:
            raise ValueError(f"CPU platform {platform!r} needs a '<n>vcpu-<m>gb' preset, got {preset!r}")
        price = CPU_PRICES[platform]
        return resources[0] * price.vcpu_hour_usd + resources[1] * price.gib_hour_usd
    if platform not in PRICES:
        known = ", ".join(sorted([*PRICES, *CPU_PRICES]))
        raise KeyError(f"no price for platform {platform!r}; known: {known}")
    return PRICES[platform].gpu_hour_usd * gpu_count(preset)


def price_tag(platform: str) -> str:
    """``sourced`` or ``assumed`` for a priced platform (``unpriced`` otherwise)."""
    if platform in CPU_PRICES:
        return CPU_PRICES[platform].tag
    return PRICES[platform].tag if platform in PRICES else "unpriced"


def price_label(platform: str, preset: str | None) -> str:
    """Human-readable price, e.g. ``$0.528/h (16 x $0.015/vCPU-h + 64 x $0.0045/GiB-h) [sourced]``."""
    hourly = hourly_usd(platform, preset)
    if platform in CPU_PRICES:
        cpu = CPU_PRICES[platform]
        vcpus, gib = preset_resources(preset) or (0, 0)
        detail = f"{vcpus} x ${cpu.vcpu_hour_usd:g}/vCPU-h + {gib} x ${cpu.gib_hour_usd:g}/GiB-h"
    else:
        detail = f"{gpu_count(preset)} x ${PRICES[platform].gpu_hour_usd:.2f}/GPU-h"
    return f"${hourly:.3f}/h ({detail}) [{price_tag(platform)}]"


def estimate_usd(platform: str, preset: str | None, seconds: float) -> float:
    """Estimated cost of ``seconds`` on ``platform`` / ``preset`` (on-demand price)."""
    return hourly_usd(platform, preset) * max(0.0, seconds) / 3600.0


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class JobSpend:
    """One finished (or abandoned) job as recorded in the spend ledger."""

    job_id: str
    name: str
    platform: str
    preset: str | None
    region: str
    state: str
    preemptible: bool
    provisioning_s: float | None
    run_s: float | None
    wall_s: float
    usd_per_hour: float
    est_usd: float  # wall-based upper bound; the spend cap uses this
    est_usd_run: float  # run-based lower bound
    run_id: str | None = None
    note: str | None = None
    ts: str = field(default_factory=utc_now_iso)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> JobSpend:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class Reservation:
    """Money held for a job that is about to run or running (``<ledger>.inflight.json``)."""

    token: str
    usd: float
    label: str
    pid: int
    host: str
    ts: str = field(default_factory=utc_now_iso)


def _pid_alive(pid: int) -> bool:
    """Whether a process with ``pid`` exists on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    except (OverflowError, ValueError):
        return False
    return True


class SpendLedger:
    """Append-only JSONL ledger of job spend with an optional cap, safe across processes.

    Recorded spend is the sum of ``est_usd`` in the file. Jobs in flight hold
    :class:`Reservation` s in ``<path>.inflight.json``; :meth:`remaining_usd` and
    :meth:`check` subtract them, and :meth:`reserve` checks and records one atomically
    under an exclusive ``flock`` on ``<path>.lock`` (see the module docstring).

    Args:
        path: JSONL file (created on first append).
        cap_usd: Maximum total estimated spend (``est_usd``) over the whole file plus what
            is in flight; ``None`` disables the cap (reservations are still recorded).
    """

    def __init__(self, path: str | Path, cap_usd: float | None = None) -> None:
        if cap_usd is not None and cap_usd < 0:
            raise ValueError("cap_usd must be non-negative")
        self.path = Path(path)
        self.cap_usd = cap_usd
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"SpendLedger(path={str(self.path)!r}, cap_usd={self.cap_usd!r})"

    @property
    def inflight_path(self) -> Path:
        return self.path.with_name(self.path.name + ".inflight.json")

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Exclusive across threads (``threading.Lock``) and processes (``flock``)."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    # -- in-flight reservations (call with the lock held)

    def _read_inflight(self) -> dict[str, Reservation]:
        """Live reservations; those of dead processes on this host are dropped."""
        if not self.inflight_path.is_file():
            return {}
        try:
            data = json.loads(self.inflight_path.read_text(encoding="utf-8") or "{}")
            rows = {k: Reservation(**v) for k, v in data.items()}
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            # fail closed: an unreadable file could hide money that is being spent
            raise SpendCapError(
                f"unreadable in-flight file {self.inflight_path} ({exc}); delete it once no job is running"
            ) from exc
        host = socket.gethostname()
        return {k: r for k, r in rows.items() if r.host != host or _pid_alive(r.pid)}

    def _write_inflight(self, rows: dict[str, Reservation]) -> None:
        if not rows:
            self.inflight_path.unlink(missing_ok=True)
            return
        tmp = self.inflight_path.with_name(self.inflight_path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({k: asdict(r) for k, r in rows.items()}, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.inflight_path)

    def _total(self) -> float:
        return sum(e.est_usd for e in read_spend(self.path))

    # -- public API

    def append(self, entry: JobSpend, release: str | None = None) -> None:
        """Record ``entry``; with ``release``, drop that reservation in the same step."""
        line = json.dumps(entry.to_json(), sort_keys=True)
        with self._locked():
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if release is not None:
                rows = self._read_inflight()
                if rows.pop(release, None) is not None:
                    self._write_inflight(rows)

    def entries(self) -> list[JobSpend]:
        if not self.path.parent.is_dir():
            return []
        with self._locked():
            return read_spend(self.path)

    def inflight(self) -> list[Reservation]:
        """Reservations of jobs that are about to run or running."""
        if not self.path.parent.is_dir():
            return []
        with self._locked():
            return sorted(self._read_inflight().values(), key=lambda r: r.ts)

    def total_usd(self) -> float:
        """Sum of ``est_usd`` over every recorded entry (reservations not included)."""
        return sum(e.est_usd for e in self.entries())

    def reserved_usd(self, exclude: str | None = None) -> float:
        """Money held by jobs in flight, except reservation ``exclude``."""
        return sum(r.usd for r in self.inflight() if r.token != exclude)

    def remaining_usd(self, exclude: str | None = None) -> float | None:
        """Cap minus recorded spend minus reservations (except ``exclude``); ``None`` when uncapped."""
        if self.cap_usd is None:
            return None
        if not self.path.parent.is_dir():
            return self.cap_usd
        with self._locked():
            held = sum(r.usd for r in self._read_inflight().values() if r.token != exclude)
            return self.cap_usd - self._total() - held

    def check(self, reserve_usd: float) -> None:
        """Raise :class:`SpendCapError` if ``reserve_usd`` more would exceed the cap."""
        remaining = self.remaining_usd()
        if remaining is not None and reserve_usd > remaining:
            raise SpendCapError(
                f"refused: ${reserve_usd:.2f} more would exceed the ${self.cap_usd:.2f} cap"
                f" (${self.total_usd():.2f} recorded in {self.path},"
                f" ${self.reserved_usd():.2f} held by jobs in flight)"
            )

    def reserve(self, usd: float, label: str) -> str:
        """Check ``usd`` against the cap and hold it, atomically; returns the reservation token.

        Raises:
            SpendCapError: recorded spend, other reservations and ``usd`` exceed the cap.
        """
        if usd < 0:
            raise ValueError("a reservation must be non-negative")
        with self._locked():
            rows = self._read_inflight()
            total = self._total()
            held = sum(r.usd for r in rows.values())
            if self.cap_usd is not None and usd > self.cap_usd - total - held:
                raise SpendCapError(
                    f"refused: ${usd:.2f} more would exceed the ${self.cap_usd:.2f} cap"
                    f" (${total:.2f} recorded in {self.path}, ${held:.2f} held by"
                    f" {len(rows)} job(s) in flight)"
                )
            token = uuid.uuid4().hex
            rows[token] = Reservation(token, round(usd, 6), label, os.getpid(), socket.gethostname())
            self._write_inflight(rows)
            return token

    def hold(self, token: str, usd: float) -> None:
        """Set reservation ``token`` to ``usd`` (a running job keeps it at least its live spend)."""
        with self._locked():
            rows = self._read_inflight()
            old = rows.get(token)
            label = old.label if old else "re-held"
            rows[token] = Reservation(token, round(usd, 6), label, os.getpid(), socket.gethostname())
            self._write_inflight(rows)

    def release(self, token: str) -> None:
        """Drop reservation ``token`` (no-op when it is gone)."""
        with self._locked():
            rows = self._read_inflight()
            if rows.pop(token, None) is not None:
                self._write_inflight(rows)


def read_spend(path: str | Path) -> list[JobSpend]:
    """Parse a spend ledger (missing file -> empty list)."""
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(JobSpend.from_json(json.loads(line)))
    return out


def summarize(entries: Iterable[JobSpend]) -> dict[str, float]:
    """Totals: jobs, wall/run seconds and estimated USD (upper and lower bound)."""
    rows = list(entries)
    return {
        "jobs": float(len(rows)),
        "wall_s": sum(e.wall_s for e in rows),
        "run_s": sum(e.run_s or 0.0 for e in rows),
        "est_usd": sum(e.est_usd for e in rows),
        "est_usd_run": sum(e.est_usd_run for e in rows),
    }
