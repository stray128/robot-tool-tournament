"""Run manifests: everything needed to trace a reported number back to how it was made.

A manifest is one JSON document per search job (one arm, R runs). It records the
platform (Nebius job ID, platform, preset, region, image digest), the code (git SHA,
``uv.lock`` hash, prompt-template hashes), the experiment (arm, budget, seeds,
task and suite hashes, executor), the model (ID, sampling parameters, provider) and
the cost of each run (tokens, dollars, latency, validator rejections and the
``fallback_random`` count).

Job facts are never hard-coded: :meth:`JobContext.from_env` reads them from
``RTT_*`` environment variables that the job launcher sets, so no resource ID is
ever written into the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST_VERSION",
    "JobContext",
    "LLMUsage",
    "file_sha256",
    "git_sha",
    "read_manifest",
    "write_json_atomic",
    "write_manifest",
]

MANIFEST_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Environment variables read by :meth:`JobContext.from_env` (set by the job launcher).
JOB_ENV = {
    "job_id": "RTT_JOB_ID",
    "platform": "RTT_PLATFORM",
    "preset": "RTT_PRESET",
    "region": "RTT_REGION",
    "image_digest": "RTT_IMAGE_DIGEST",
    "launcher": "RTT_LAUNCHER",
}


@dataclass(frozen=True)
class JobContext:
    """Where this process runs. ``None`` fields are unknown (e.g. a laptop run)."""

    job_id: str | None = None
    platform: str | None = None
    preset: str | None = None
    region: str | None = None
    image_digest: str | None = None
    launcher: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> JobContext:
        env = os.environ if environ is None else environ
        return cls(**{field: (env.get(var) or None) for field, var in JOB_ENV.items()})

    @property
    def is_cloud_job(self) -> bool:
        return self.job_id is not None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMUsage:
    """Aggregate LLM usage of one run (or one arm). Costs and latency count live calls only."""

    calls: int = 0
    live_calls: int = 0
    cached_calls: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s_total: float = 0.0
    latency_s_p50: float | None = None
    latency_s_max: float | None = None

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> LLMUsage:
        """Aggregate call records (dicts with ``cached``, ``ok``, token, cost and latency keys).

        Token counts and cost come from the original live call even when a record
        was served from cache, so a replayed arm reports the same totals as the
        run that paid for them; ``live_calls`` / ``cached_calls`` tell them apart.
        """
        rows = list(records)
        latencies = [float(r.get("latency_s") or 0.0) for r in rows if r.get("request_hash")]
        return cls(
            calls=len(rows),
            live_calls=sum(1 for r in rows if r.get("request_hash") and not r.get("cached")),
            cached_calls=sum(1 for r in rows if r.get("cached")),
            errors=sum(1 for r in rows if not r.get("ok", True)),
            prompt_tokens=sum(int(r.get("prompt_tokens") or 0) for r in rows),
            completion_tokens=sum(int(r.get("completion_tokens") or 0) for r in rows),
            reasoning_tokens=sum(int(r.get("reasoning_tokens") or 0) for r in rows),
            cost_usd=round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 8),
            latency_s_total=round(sum(latencies), 4),
            latency_s_p50=round(statistics.median(latencies), 4) if latencies else None,
            latency_s_max=round(max(latencies), 4) if latencies else None,
        )

    @classmethod
    def combine(cls, parts: Iterable[LLMUsage]) -> LLMUsage:
        items = list(parts)
        maxima = [p.latency_s_max for p in items if p.latency_s_max is not None]
        return cls(
            calls=sum(p.calls for p in items),
            live_calls=sum(p.live_calls for p in items),
            cached_calls=sum(p.cached_calls for p in items),
            errors=sum(p.errors for p in items),
            prompt_tokens=sum(p.prompt_tokens for p in items),
            completion_tokens=sum(p.completion_tokens for p in items),
            reasoning_tokens=sum(p.reasoning_tokens for p in items),
            cost_usd=round(sum(p.cost_usd for p in items), 8),
            latency_s_total=round(sum(p.latency_s_total for p in items), 4),
            latency_s_p50=None,  # a median of medians is not a median; per-run values stay in the runs
            latency_s_max=max(maxima) if maxima else None,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: str | Path) -> str | None:
    """SHA-256 of a file's bytes, or ``None`` if it does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo: str | Path = _REPO_ROOT) -> str | None:
    """Commit of the code that is running.

    ``RTT_GIT_SHA`` wins (container images have no ``.git``); otherwise ``git
    rev-parse HEAD`` in ``repo``, with a ``-dirty`` suffix when tracked files have
    uncommitted changes. ``None`` when neither is available.
    """
    env = os.environ.get("RTT_GIT_SHA", "").strip()
    if env:
        return env
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha or None


def code_facts(repo: str | Path = _REPO_ROOT) -> dict[str, Any]:
    """Git SHA, ``uv.lock`` hash and package version for the manifest."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        pkg_version = version("robot-tool-tournament")
    except PackageNotFoundError:
        pkg_version = None
    return {
        "git_sha": git_sha(repo),
        "uv_lock_sha256": os.environ.get("RTT_UV_LOCK_SHA256") or file_sha256(Path(repo) / "uv.lock"),
        "package_version": pkg_version,
    }


def write_json_atomic(path: str | Path, data: Any) -> Path:
    """Write ``data`` as indented JSON via a temporary file and an atomic rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write a manifest (adds ``manifest_version``)."""
    return write_json_atomic(path, {"manifest_version": MANIFEST_VERSION, **manifest})


def read_manifest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"{path}: unsupported manifest version {data.get('manifest_version')!r}")
    return data
