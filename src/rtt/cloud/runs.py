"""Run layout shared by the laptop and the job container.

A run is one job's output: the container writes ``/out/runs/<run_id>/`` on the mounted
bucket, which is the object prefix ``runs/<run_id>/``, and the laptop fetches it to
``runs/cloud/<run_id>/`` (gitignored). This module names runs, builds the job spec that
executes :mod:`rtt.cloud.entry` in the pinned image, and checks a fetched run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rtt.cloud.config import NebiusConfig
from rtt.cloud.nebius_jobs import Candidate, JobSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_RUNS = REPO_ROOT / "runs" / "cloud"
DEFAULT_LEDGER = LOCAL_RUNS / "spend.jsonl"
CONTAINER_PYTHON = "/app/.venv/bin/python"
MOUNT = "/out"
KINDS = ("m1", "probe", "search", "gv4", "heldout")
#: Where the held-out job finds the injected secret (``--inject-file``); no other job gets it.
SEED_MOUNT = "/secret/heldout_seed"
#: Steps of a heldout job: ``selection`` draws the scored scans (and records their hull
#: hashes) without scoring anything; ``procedural`` is H1; ``real`` is H2 (drawing the scans
#: first unless an earlier run's selection is given).
HELDOUT_STEPS = ("selection", "procedural", "real")
HELDOUT_SUITES = ("procedural", "real")
LAUNCHER = "nebius-serverless-jobs"
LLM_ARMS = frozenset({"nemotron", "zeroshot"})
"""Search arms that call Token Factory from inside the job."""


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
    space: str | None = None  # box-v1 (default) or freeform-v2

    def argv(self) -> list[str]:
        args = ["--arm", self.arm, "--runs", str(self.runs), "--budget", str(self.budget)]
        args += ["--executor", self.executor, "--spend-cap", f"{self.spend_cap_usd:g}"]
        if self.space:
            args += ["--space", self.space]
        if self.seeds:
            args += ["--seeds", self.seeds]
        if self.workers:
            args += ["--workers", str(self.workers)]
        return args


@dataclass(frozen=True)
class HeldoutJobArgs:
    """Options of a ``heldout`` job (:func:`rtt.cloud.entry.run_heldout_job`).

    ``frozen_runs`` are the run IDs of finished search jobs in the same region's bucket; the
    job reads their ``search/frozen/`` records from the mount and refuses to read the secret
    unless their digest is ``frozen_digest`` (computed on the laptop from the fetched
    copies). ``selection_run`` names an earlier held-out run whose ``real/selection`` the job
    scores instead of drawing it again. ``dev`` runs a small exploratory job on a throwaway
    secret (infrastructure check; it reveals nothing about the held-out suites).
    """

    frozen_runs: tuple[str, ...]
    frozen_digest: str
    suites: tuple[str, ...] = HELDOUT_SUITES  # any of HELDOUT_STEPS
    selection_run: str | None = None
    workers: int | None = None
    select_workers: int | None = None
    dev: bool = False

    def argv(self) -> list[str]:
        if not self.frozen_runs:
            raise ValueError("a heldout job needs the search runs whose frozen picks it scores")
        unknown = set(self.suites) - set(HELDOUT_STEPS)
        if unknown or not self.suites:
            raise ValueError(f"heldout suites must be a non-empty subset of {HELDOUT_STEPS}")
        if self.selection_run and "selection" in self.suites:
            raise ValueError("--selection-run reuses a selection; do not also ask for a new one")
        args = ["--seed-file", SEED_MOUNT, "--expect-frozen-digest", self.frozen_digest]
        args += ["--suites", ",".join(s for s in HELDOUT_STEPS if s in self.suites)]
        args += ["--frozen", *(f"{MOUNT}/{run_prefix(r)}/search" for r in self.frozen_runs)]
        if self.selection_run:
            args += ["--selection", f"{MOUNT}/{run_prefix(self.selection_run)}/real/selection"]
        if self.workers:
            args += ["--workers", str(self.workers)]
        if self.select_workers:
            args += ["--select-workers", str(self.select_workers)]
        if self.dev:
            args.append("--dev")
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


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,58}$")


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` if it is safe as a bucket prefix, a local folder and a job-name part.

    1-59 lowercase letters, digits and dashes, starting with a letter or digit, so that
    ``rtt-<run_id>`` is a valid job name and no ``/``, ``..`` or absolute path can reach
    the bucket prefix or ``runs/cloud/<run_id>/`` on the laptop.

    Raises:
        ValueError: anything else.
    """
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise ValueError(f"run ID {run_id!r}: use 1-59 lowercase letters, digits and dashes")
    return run_id


def run_prefix(run_id: str) -> str:
    """Object-storage prefix of a run (no trailing slash); ``run_id`` is validated."""
    return f"runs/{validate_run_id(run_id)}"


def entry_args(
    kind: str,
    run_id: str,
    seed: int = 0,
    video: bool = True,
    probe_warp: bool = True,
    search: SearchJobArgs | None = None,
    heldout: HeldoutJobArgs | None = None,
) -> list[str]:
    """Arguments for ``python -m rtt.cloud.entry`` inside the container."""
    if kind not in KINDS:
        raise ValueError(f"unknown run kind {kind!r}; expected one of {KINDS}")
    args = ["-m", "rtt.cloud.entry", kind, "--out", f"{MOUNT}/{run_prefix(run_id)}"]
    if kind == "search":
        if search is None:
            raise ValueError("a search job needs SearchJobArgs")
        return args + search.argv()
    if kind == "heldout":
        if heldout is None:
            raise ValueError("a heldout job needs HeldoutJobArgs")
        return args + heldout.argv()
    if kind == "gv4":  # gate GV4 (rtt.sim.gv4) on every vCPU of the job
        return args
    if kind == "m1":
        args += ["--seed", str(seed)]
        if not video:
            args.append("--no-video")
        if probe_warp:
            args.append("--probe-warp")
    elif not probe_warp:
        args.append("--no-warp")
    return args


def needs_token_factory(kind: str, search: SearchJobArgs | None = None) -> bool:
    """Whether a job of ``kind`` needs the Token Factory key.

    Only the LLM search arms call the model; the probe checks that the secret arrives.
    Gates and random search run without it, so they never see the key (and run in a
    region whose project has no copy of the secret).
    """
    return kind == "probe" or (kind == "search" and search is not None and search.arm in LLM_ARMS)


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
    candidate: Candidate | None = None,
    attach_key: bool | None = None,
    heldout: HeldoutJobArgs | None = None,
    inject_files: Sequence[str] = (),
) -> JobSpec:
    """The job that runs ``kind`` in the configured image and writes to the bucket.

    ``config`` is the config of the region the job runs in
    (:meth:`~rtt.cloud.config.NebiusConfig.for_region`). Needs ``RTT_IMAGE`` (a digest
    reference) and that region's bucket ID, subnet ID and project ID. ``candidate`` sets
    the platform and preset (default: the config's GPU platform and preset; :meth:`NebiusJobs.run`
    overrides them per attempt anyway, but a dry run validates exactly this spec). The
    Token Factory key is attached by secret reference when ``attach_key`` is true, or by
    default when :func:`needs_token_factory` says the job uses it.

    The container gets the job facts that run manifests record (``RTT_RUN_ID``,
    ``RTT_JOB_NAME``, ``RTT_IMAGE_NAME``, ``RTT_IMAGE_DIGEST``, ``RTT_LAUNCHER``; the
    platform, preset and region are added per attempt by :meth:`NebiusJobs.run`). The
    full image reference is not passed: its path holds the registry ID, which must not
    reach published manifests.

    The held-out secret reaches the ``heldout`` job only, as ``inject_files`` (exactly one
    file, at :data:`SEED_MOUNT`); every other kind refuses ``inject_files``, and a
    ``heldout`` job refuses to run without it.
    """
    inject = tuple(inject_files)
    if kind != "heldout" and inject:
        raise ValueError(f"only the heldout job may receive injected files, not a {kind} job")
    if kind == "heldout" and (len(inject) != 1 or not inject[0].endswith(f":{SEED_MOUNT}")):
        raise ValueError(f"a heldout job needs exactly one injected file, the secret at {SEED_MOUNT}")
    config.require("image", "bucket_id", "subnet_id", "project_id")
    assert config.image is not None and config.bucket_id is not None
    key = needs_token_factory(kind, search) if attach_key is None else attach_key
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
        args=entry_args(kind, run_id, seed, video, probe_warp, search, heldout),
        env=env,
        env_secrets={"NEBIUS_API_KEY": config.secret_name} if key else {},
        volumes=[f"{config.bucket_id}:{MOUNT}:rw"],
        platform=candidate.platform if candidate else config.gpu_platform,
        preset=candidate.preset if candidate else config.gpu_preset,
        timeout=timeout,
        restart_policy="never",
        subnet_id=config.subnet_id,
        parent_id=config.project_id,
        preemptible=preemptible,
        inject_files=inject,
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
    elif manifest.get("kind") == "gv4":
        problems += _verify_gv4(root, manifest)
    elif manifest.get("kind") == "heldout":
        problems += _verify_heldout(root, manifest)
    return RunCheck(not problems, tuple(problems), manifest)


HELDOUT_REPORTS = {
    "procedural": ("procedural/heldout_v2.json", "rtt.heldout/2"),
    "real": ("real/heldout_real_v2.json", "rtt.heldout_real/2"),
}


def _verify_heldout(root: Path, manifest: Mapping[str, Any]) -> Sequence[str]:
    """A held-out job wrote one report per suite it ran, of the right schema; outside a dev
    run each is an H1 or H2 result (commitment verified, full suite: not exploratory). A scan
    draw made in the job left its selection and hull hashes under ``real/selection/``."""
    inputs = manifest.get("inputs") or {}
    suites = list(inputs.get("suites") or ())
    problems = []
    if inputs.get("selection") is None and {"selection", "real"} & set(suites):
        for name in ("selection.json", "hull_hashes.json"):
            if _read_json(root / "real" / "selection" / name) is None:
                problems.append(f"missing or unreadable real/selection/{name}")
    for suite in suites:
        if suite not in HELDOUT_REPORTS:
            continue
        rel, schema = HELDOUT_REPORTS[suite]
        report = _read_json(root / rel)
        if report is None:
            problems.append(f"missing or unreadable {rel}")
            continue
        if report.get("schema") != schema:
            problems.append(f"{rel}: schema {report.get('schema')!r}, expected {schema}")
        if not inputs.get("dev") and report.get("exploratory"):
            problems.append(f"{rel}: exploratory ({report.get('exploratory_reasons')})")
    return problems


def _verify_gv4(root: Path, manifest: Mapping[str, Any]) -> Sequence[str]:
    """A GV4 job's ``gv4.json`` holds the episodes and the records digest its manifest names."""
    report = _read_json(root / "gv4.json")
    if report is None:
        return ["missing or unreadable gv4.json"]
    result = manifest.get("result") or {}
    problems = []
    if len(report.get("records") or ()) != report.get("episodes") or not report.get("episodes"):
        problems.append("gv4.json: record count does not match its episode count")
    if result.get("records_sha256") != report.get("records_sha256"):
        problems.append("gv4.json: records digest differs from the manifest's")
    return problems


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
