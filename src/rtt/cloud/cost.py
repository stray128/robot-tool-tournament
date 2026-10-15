"""Price table, per-job cost estimates and an append-only spend ledger for Nebius jobs.

Prices are USD per GPU-hour for the on-demand 1-GPU presets as listed on
https://nebius.com/prices (read 2026-10-15). A preset with ``N`` GPUs costs ``N`` times
the GPU-hour price. "from" prices (L40S) are the smallest preset, so estimates for larger
L40S presets are lower bounds; those rows are tagged ``assumed``.

Preemptible VMs follow the spot price, which the page lists only as a floor ("from
$0.79"). Estimates for preemptible jobs therefore use the on-demand price as a ceiling,
so a spend cap enforced on them is conservative.

What a job is billed for is estimated two ways [assumed]:

* ``wall``: from job creation to finish, covering provisioning, image pull and the run.
  This is the upper bound and the number the spend cap is enforced on.
* ``run``: container start to finish. The jobs documentation says a job is "billed only
  while the job is running", so the true charge is expected to be near this lower bound.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PRICES_SOURCE = "https://nebius.com/prices (read 2026-10-15)"


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

_GPU_COUNT = re.compile(r"^(\d+)gpu-")


def gpu_count(preset: str | None) -> int:
    """GPUs in a preset name such as ``8gpu-128vcpu-1600gb`` (1 when the name has no count)."""
    match = _GPU_COUNT.match(preset or "")
    return int(match.group(1)) if match else 1


def hourly_usd(platform: str, preset: str | None = None) -> float:
    """Estimated on-demand USD per hour for ``platform`` / ``preset``.

    Raises:
        KeyError: the platform is not in :data:`PRICES` (CPU platforms are not priced).
    """
    if platform not in PRICES:
        raise KeyError(f"no price for platform {platform!r}; known: {', '.join(sorted(PRICES))}")
    return PRICES[platform].gpu_hour_usd * gpu_count(preset)


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


class SpendLedger:
    """Append-only JSONL ledger of job spend with an optional cap.

    Args:
        path: JSONL file (created on first append).
        cap_usd: Maximum total estimated spend (``est_usd``) over the whole file; ``None``
            disables :meth:`check`.
    """

    def __init__(self, path: str | Path, cap_usd: float | None = None) -> None:
        if cap_usd is not None and cap_usd < 0:
            raise ValueError("cap_usd must be non-negative")
        self.path = Path(path)
        self.cap_usd = cap_usd
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"SpendLedger(path={str(self.path)!r}, cap_usd={self.cap_usd!r})"

    def append(self, entry: JobSpend) -> None:
        line = json.dumps(entry.to_json(), sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def entries(self) -> list[JobSpend]:
        with self._lock:
            return read_spend(self.path)

    def total_usd(self) -> float:
        """Sum of ``est_usd`` over every entry."""
        return sum(e.est_usd for e in self.entries())

    def remaining_usd(self) -> float | None:
        """Cap minus recorded spend, or ``None`` when uncapped."""
        if self.cap_usd is None:
            return None
        return self.cap_usd - self.total_usd()

    def check(self, reserve_usd: float) -> None:
        """Raise :class:`SpendCapError` if ``reserve_usd`` more would exceed the cap."""
        remaining = self.remaining_usd()
        if remaining is not None and reserve_usd > remaining:
            raise SpendCapError(
                f"refused: ${reserve_usd:.2f} more would exceed the ${self.cap_usd:.2f} cap"
                f" (${self.total_usd():.2f} already recorded in {self.path})"
            )


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
