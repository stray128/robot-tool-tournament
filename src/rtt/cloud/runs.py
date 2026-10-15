"""Run layout shared by the laptop and the job container.

A run is one job's output: the container writes ``/out/runs/<run_id>/`` on the mounted
bucket, which is the object prefix ``runs/<run_id>/``, and the laptop fetches it to
``runs/cloud/<run_id>/`` (gitignored). This module names runs, builds the job spec that
executes :mod:`rtt.cloud.entry` in the pinned image, and checks a fetched run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rtt.cloud.config import NebiusConfig
from rtt.cloud.nebius_jobs import JobSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_RUNS = REPO_ROOT / "runs" / "cloud"
DEFAULT_LEDGER = LOCAL_RUNS / "spend.jsonl"
CONTAINER_PYTHON = "/app/.venv/bin/python"
MOUNT = "/out"
KINDS = ("m1", "probe", "search")
LAUNCHER = "nebius-serverless-jobs"


@dataclass(frozen=True)
class SearchJobArgs:
    """``rtt search`` options for a ``search`` job (see :func:`rtt.cloud.entry.run_search_job`)."""

    arm: str
    runs: int
    budget: int
    seeds: str | None = None  # comma-separated run seeds; default 1..runs
    executor: str = "cpu"
    workers: int | None = None
    spend_cap_usd: float = 2.0

    def argv(self) -> list[str]:
        args = ["--arm", self.arm, "--runs", str(self.runs), "--budget", str(self.budget)]
        args += ["--executor", self.executor, "--spend-cap", f"{self.spend_cap_usd:g}"]
        if self.seeds:
            args += ["--seeds", self.seeds]
        if self.workers:
            args += ["--workers", str(self.workers)]
        return args


def split_image_ref(ref: str) -> tuple[str, str]:
    """``host/<registry path>/rtt-sim@sha256:…`` -> ``("rtt-sim", "sha256:…")``.

    Raises:
        ValueError: ``ref`` is not pinned by digest.
    """
    repo, sep, digest = ref.partition("@")
    if not sep or not digest.startswith("sha256:"):
        raise ValueError("image must be pinned by digest (…@sha256:…)")
    return repo.rsplit("/", 1)[-1], digest


def new_run_id(kind: str, now: datetime | None = None) -> str:
    """``<kind>-YYYYMMDD-HHMMSS`` in UTC (lowercase, so it is also a valid job name part)."""
    ts = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{kind}-{ts}"


def run_prefix(run_id: str) -> str:
    """Object-storage prefix of a run (no trailing slash)."""
    return f"runs/{run_id}"


def entry_args(
    kind: str,
    run_id: str,
    seed: int = 0,
    video: bool = True,
    probe_warp: bool = True,
    search: SearchJobArgs | None = None,
) -> list[str]:
    """Arguments for ``python -m rtt.cloud.entry`` inside the container."""
    if kind not in KINDS:
        raise ValueError(f"unknown run kind {kind!r}; expected one of {KINDS}")
    args = ["-m", "rtt.cloud.entry", kind, "--out", f"{MOUNT}/{run_prefix(run_id)}"]
    if kind == "search":
        if search is None:
            raise ValueError("a search job needs SearchJobArgs")
        return args + search.argv()
    if kind == "m1":
        args += ["--seed", str(seed)]
        if not video:
            args.append("--no-video")
        if probe_warp:
            args.append("--probe-warp")
    elif not probe_warp:
        args.append("--no-warp")
    return args


def job_spec_for(
    config: NebiusConfig,
    kind: str,
    run_id: str,
    seed: int = 0,
    video: bool = True,
    probe_warp: bool = True,
    preemptible: bool = False,
    timeout: str = "1h",
    extra_env: Mapping[str, str] | None = None,
    search: SearchJobArgs | None = None,
) -> JobSpec:
    """The job that runs ``kind`` in the configured image and writes to the bucket.

    Needs ``RTT_IMAGE`` (a digest reference), ``NEBIUS_BUCKET_ID``, ``NEBIUS_SUBNET_ID``
    and ``NEBIUS_PROJECT_ID``. The Token Factory key is attached by secret reference.

    The container gets the job facts that run manifests record (``RTT_RUN_ID``,
    ``RTT_JOB_NAME``, ``RTT_IMAGE_NAME``, ``RTT_IMAGE_DIGEST``, ``RTT_LAUNCHER``; the
    platform, preset and region are added per attempt by :meth:`NebiusJobs.run`). The
    full image reference is not passed: its path holds the registry ID, which must not
    reach published manifests.
    """
    config.require("image", "bucket_id", "subnet_id", "project_id")
    assert config.image is not None and config.bucket_id is not None
    image_name, digest = split_image_ref(config.image)
    name = f"rtt-{run_id}"
    env = {
        "RTT_RUN_ID": run_id,
        "RTT_JOB_NAME": name,
        "RTT_IMAGE_NAME": image_name,
        "RTT_IMAGE_DIGEST": digest,
        "RTT_LAUNCHER": LAUNCHER,
    }
    env.update(extra_env or {})
    return JobSpec(
        name=name,
        image=config.image,
        command=CONTAINER_PYTHON,
        args=entry_args(kind, run_id, seed, video, probe_warp, search),
        env=env,
        env_secrets={"NEBIUS_API_KEY": config.secret_name},
        volumes=[f"{config.bucket_id}:{MOUNT}:rw"],
        platform=config.gpu_platform,
        preset=config.gpu_preset,
        timeout=timeout,
        restart_policy="never",
        subnet_id=config.subnet_id,
        parent_id=config.project_id,
        preemptible=preemptible,
    )


@dataclass(frozen=True)
class RunCheck:
    """Outcome of :func:`verify_run`."""

    ok: bool
    problems: tuple[str, ...]
    manifest: Mapping[str, Any] | None

    def describe(self) -> str:
        if self.ok:
            return "run verified: manifest, report and every listed file are present"
        return "run NOT verified:\n" + "\n".join(f"  - {p}" for p in self.problems)


def verify_run(run_dir: str | Path, expect_video: bool | None = None) -> RunCheck:
    """Check a fetched run directory.

    * ``manifest.json`` exists, has exit status 0 and lists only files that exist, and the
      listed files add up to the bytes the job says it published (a failed write on the
      bucket mount leaves 0-byte files; without a published total, any empty file fails);
    * for ``m1``: ``report.json`` says the gate passed with at least one case, and every
      case's replay files (and a complete MP4 when ``expect_video``; default: as the
      manifest's inputs say) exist.
    """
    root = Path(run_dir)
    problems: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return RunCheck(False, (f"missing {manifest_path.name}",), None)
    manifest = _read_json(manifest_path)
    if manifest is None:
        return RunCheck(False, (f"{manifest_path.name} is empty or not JSON",), None)
    if manifest.get("exit_status") != 0:
        problems.append(f"exit_status {manifest.get('exit_status')}")
    if manifest.get("error"):
        problems.append("manifest records an error")
    listed_bytes = 0
    published = (manifest.get("publish") or {}).get("bytes")
    for rel in manifest.get("files", []):
        path = root / rel
        if not path.is_file():
            problems.append(f"listed file missing: {rel}")
            continue
        size = path.stat().st_size
        # With a published byte total, a truncated file shows up as a total mismatch, and a
        # legitimately empty file (e.g. a run with no rejections) is fine. Without one, any
        # empty file is suspect: a failed rename on the bucket mount leaves exactly that.
        if size == 0 and (published is None or rel == manifest_path.name):
            problems.append(f"listed file empty: {rel}")
        elif rel != manifest_path.name:
            listed_bytes += size
    if published is not None and published != listed_bytes:
        problems.append(f"files total {listed_bytes} bytes, the job published {published}")
    if manifest.get("kind") == "m1":
        problems += _verify_m1(root, manifest, expect_video)
    elif manifest.get("kind") == "search":
        problems += _verify_search(root)
    return RunCheck(not problems, tuple(problems), manifest)


def _read_json(path: Path) -> Any:
    """Parsed JSON object, or ``None`` for an empty or malformed file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _verify_m1(root: Path, manifest: Mapping[str, Any], expect_video: bool | None) -> Sequence[str]:
    problems = []
    report_path = root / "report.json"
    if not report_path.is_file():
        return ["missing report.json"]
    report = _read_json(report_path)
    if report is None:
        return ["report.json is empty or not JSON"]
    if report.get("passed") is not True:
        problems.append("gate report says FAILED")
    if not report.get("cases"):
        problems.append("gate report has no cases")
    video = (manifest.get("inputs") or {}).get("video", True) if expect_video is None else expect_video
    suffixes = (".bin", ".json", ".mp4") if video else (".bin", ".json")
    for name, case in report.get("cases", {}).items():
        if not case.get("passed"):
            problems.append(f"case {name} not as expected")
        for suffix in suffixes:
            path = root / f"{name}{suffix}"
            if not path.is_file():
                problems.append(f"case {name}: missing {name}{suffix}")
            elif suffix == ".mp4" and not mp4_complete(path):
                problems.append(f"case {name}: {name}.mp4 is truncated (no ftyp/moov box)")
    return problems


def _verify_search(root: Path) -> Sequence[str]:
    """A search job's ``search/manifest.json`` is complete and every run froze its top-1."""
    search = _read_json(root / "search" / "manifest.json")
    if search is None:
        return ["missing or unreadable search/manifest.json"]
    problems = []
    if search.get("status") != "complete":
        problems.append(f"search status {search.get('status')!r}")
    for run in search.get("per_run", []):
        frozen = root / "search" / "frozen" / f"{run.get('run_id')}.json"
        if not frozen.is_file() or frozen.stat().st_size == 0:
            problems.append(f"missing frozen pick for {run.get('run_id')}")
    if len(search.get("per_run", [])) != search.get("runs"):
        problems.append("per-run results do not cover every run")
    return problems


def mp4_complete(path: str | Path) -> bool:
    """Whether an MP4 has its ``ftyp`` header and ``moov`` index (written last by the muxer).

    Walks the top-level boxes: every box must fit in the file. A muxer that cannot seek
    back (the bucket mount) leaves a file without ``moov`` or with an ``mdat`` whose size
    was never patched, which players reject although the file is not empty.
    """
    data = Path(path).read_bytes()
    pos, seen = 0, set()
    while pos + 8 <= len(data):
        size = int.from_bytes(data[pos : pos + 4], "big")
        box = data[pos + 4 : pos + 8]
        if size == 1:  # 64-bit size follows the type
            size = int.from_bytes(data[pos + 8 : pos + 16], "big")
        elif size == 0:  # box runs to the end of the file
            size = len(data) - pos
        if size < 8 or pos + size > len(data):
            return False
        seen.add(box)
        pos += size
    return pos == len(data) and {b"ftyp", b"moov"} <= seen
