"""In-container entry point for Nebius jobs: ``python -m rtt.cloud.entry <kind> --out DIR``.

Kinds:

* ``m1``: runs gate G1 (:func:`rtt.sim.m1.run_gate`) on the CPU executor and writes one
  replay (``.bin`` + ``.json``) and, unless ``--no-video``, one MP4 per case, then
  ``report.json`` (the gate report) and ``manifest.json``.
* ``probe``: records the job environment only (GPU and driver, the OpenGL renderer that
  headless rendering gets, Warp's CUDA devices) in ``probe.json`` and ``manifest.json``.
* ``search``: runs ``rtt search`` (:func:`rtt.pipeline.search.run_search_command`) with the
  given arm, runs, budget and design space (``--space freeform-v2`` for the v2 protocol) on
  the job's CPUs; the search's own outputs (its ``manifest.json``, histories, Parquet tables,
  frozen picks and, for the LLM arms, the response cache and spend ledger) go to
  ``search/``, the job manifest to the top level.
* ``gv4``: gate GV4 (:mod:`rtt.sim.gv4`): builds and simulates the fixed v2 parity episodes
  on the job's CPUs and writes ``gv4.json``, which the laptop compares with its own run.

Every file is first written to a local stage directory and then copied whole to
``--out``: the bucket mount rejects seeks (MP4 muxing) and renames (see :func:`publish`).
``manifest.json`` is published last, so its presence marks a finished run. It records the
image name and digest (never the registry path, which holds the registry ID) and build
info (git SHA, ``uv.lock`` hash), the job's platform, preset and region (passed in as
``RTT_*`` variables by :mod:`rtt.cloud.runs` and :mod:`rtt.cloud.nebius_jobs`), the
input file hashes, timings and the environment probe. It never records environment
values: only variable names, plus whether the Token Factory key is present.

Exit status: 0 when the run passed, 1 when the gate failed, 2 on an exception (the
manifest then carries the traceback tail).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_INFO = REPO_ROOT / "BUILD_INFO.json"
MANIFEST_SCHEMA = "rtt.job-manifest/1"
JOB_ENV = (
    "RTT_RUN_ID",
    "RTT_JOB_NAME",
    "RTT_IMAGE_NAME",
    "RTT_IMAGE_DIGEST",
    "RTT_PLATFORM",
    "RTT_PRESET",
    "RTT_REGION",
    "RTT_LAUNCHER",
)
KEY_ENV = "NEBIUS_API_KEY"
MANIFEST = "manifest.json"


def utc_iso(ts: float | None = None) -> str:
    dt = datetime.fromtimestamp(ts, UTC) if ts is not None else datetime.now(UTC)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> Path:
    """Write JSON atomically (temp file + rename); for local disks only, see :func:`publish`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- probes


def build_info() -> dict[str, Any]:
    """``/app/BUILD_INFO.json`` inside the image; from git and ``uv.lock`` elsewhere."""
    if BUILD_INFO.is_file():
        baked = json.loads(BUILD_INFO.read_text(encoding="utf-8"))
        dirty = baked.get("git_dirty")
        baked["git_dirty"] = (
            {"true": True, "false": False}.get(dirty, None) if isinstance(dirty, str) else dirty
        )
        return dict(baked, source="image")
    info: dict[str, Any] = {"source": "checkout", "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock")}
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        info["git_sha"] = sha.stdout.strip() or None
        info["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        info["git_sha"] = None
    return info


def gpu_info(timeout_s: float = 30.0) -> list[dict[str, str]] | dict[str, str]:
    """One dict per GPU from ``nvidia-smi``, or ``{"error": ...}`` when there is none."""
    query = "name,driver_version,memory.total,compute_cap"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"nvidia-smi unavailable: {exc.__class__.__name__}"}
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()[-300:] or f"exit {proc.returncode}"}
    keys = query.split(",")
    return [
        dict(zip(keys, (v.strip() for v in line.split(",")), strict=False))
        for line in proc.stdout.splitlines()
        if line.strip()
    ]


def gl_info() -> dict[str, str]:
    """Vendor and renderer of the headless OpenGL context MuJoCo gets under ``MUJOCO_GL``."""
    backend = os.environ.get("MUJOCO_GL", "")
    if backend.lower() not in ("egl", "osmesa"):
        return {"backend": backend or "unset", "error": "set MUJOCO_GL=egl or osmesa to probe"}
    try:
        import mujoco
        from OpenGL import GL

        ctx = mujoco.GLContext(64, 64)
        try:
            ctx.make_current()

            def text(name: int) -> str:
                raw = GL.glGetString(name)
                return raw.decode() if isinstance(raw, bytes) else str(raw)

            return {
                "backend": backend,
                "vendor": text(GL.GL_VENDOR),
                "renderer": text(GL.GL_RENDERER),
                "version": text(GL.GL_VERSION),
            }
        finally:
            ctx.free()
    except Exception as exc:  # noqa: BLE001 - a probe reports, it never fails the run
        return {"backend": backend, "error": f"{exc.__class__.__name__}: {exc}"[:300]}


def warp_info() -> dict[str, Any]:
    """Warp's view of the CUDA driver and devices (initialises Warp; a few seconds)."""
    try:
        import warp as wp

        wp.config.quiet = True
        wp.init()
        cuda = wp.get_cuda_device_count()
        driver = wp.get_cuda_driver_version() if cuda else None
        return {
            "warp": wp.config.version,
            "cuda_driver": ".".join(str(x) for x in driver) if isinstance(driver, tuple) else driver,
            "devices": [
                {"alias": d.alias, "name": d.name, "arch": getattr(d, "arch", None)} for d in wp.get_devices()
            ],
            "cuda_device_count": cuda,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{exc.__class__.__name__}: {exc}"[:300]}


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def cpu_info(cpuinfo: str | Path = "/proc/cpuinfo") -> dict[str, Any]:
    """CPU model name and memory size, for comparing executors across platforms.

    Reads ``/proc/cpuinfo`` and ``/proc/meminfo`` (Linux); missing files give ``None``.
    """
    model = None
    try:
        for line in Path(cpuinfo).read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() == "model name":
                model = value.strip()
                break
    except OSError:
        pass
    mem_gib = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_gib = round(int(line.split()[1]) / (1024 * 1024), 1)
                break
    except (OSError, ValueError, IndexError):
        pass
    return {"model": model, "mem_total_gib": mem_gib}


def environment(probe_gl: bool = True, probe_warp: bool = False) -> dict[str, Any]:
    """Host, interpreter, package versions, CPU, GPU and (optionally) GL and Warp probes."""
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    env: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu": cpu_info(),
        "packages": {d: _version(d) for d in ("mujoco", "mujoco-warp", "warp-lang", "numpy")},
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "gpu": gpu_info(),
        "env_names": sorted(os.environ),
        "token_factory_key_present": bool(os.environ.get(KEY_ENV)),
    }
    if probe_gl:
        env["gl"] = gl_info()
    if probe_warp:
        env["warp"] = warp_info()
    return env


# --------------------------------------------------------------------------- runs


def _relative_files(report: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    for case in report.get("cases", {}).values():
        case["files"] = [
            str(Path(f).resolve().relative_to(out_dir.resolve())) if Path(f).is_absolute() else f
            for f in case.get("files", [])
        ]
    return report


def base_manifest(kind: str, out_dir: Path, argv: Sequence[str]) -> dict[str, Any]:
    job = {k.removeprefix("RTT_").lower(): os.environ.get(k) for k in JOB_ENV}
    image_name, digest = job.pop("image_name"), job.pop("image_digest")
    return {
        "schema": MANIFEST_SCHEMA,
        "kind": kind,
        "run_id": job.pop("run_id") or out_dir.name,
        "argv": list(argv),
        "out_dir": str(out_dir),
        "image": f"{image_name or 'rtt-sim'}@{digest}" if digest else None,
        "image_digest": digest,
        "build": build_info(),
        "nebius": job,
        "executor": "cpu",
        "provider": "nebius-serverless-jobs" if job.get("platform") else "local",
    }


def copy_file(src: Path, dst: Path) -> int:
    """Copy with one sequential write stream (no seek, no rename); return the bytes written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout, 1 << 20)
    return src.stat().st_size


def publish(stage: Path, out: Path, exclude: frozenset[str] = frozenset({MANIFEST})) -> dict[str, Any]:
    """Copy every staged file except ``exclude`` (top-level names) to ``out``.

    Jobs write to a local stage first because the bucket mount (a FUSE filesystem over
    object storage) does not support what local files do: an MP4 muxer seeking back to
    write its index fails with EINVAL, and a rename leaves an empty target behind. Whole
    files copied front to back are safe. Sizes are checked after each copy.
    """
    copied: list[str] = []
    mismatched: list[str] = []
    total = 0
    for src in sorted(p for p in stage.rglob("*") if p.is_file()):
        rel = src.relative_to(stage)
        if len(rel.parts) == 1 and rel.name in exclude:
            continue
        dst = out / rel
        total += copy_file(src, dst)
        copied.append(rel.as_posix())
        if dst.stat().st_size != src.stat().st_size:
            mismatched.append(rel.as_posix())
    return {"files": len(copied), "bytes": total, "size_mismatches": mismatched, "names": copied}


def _finish(
    manifest: dict[str, Any], stage: Path, out: Path, t0: float, t_mid: float, status: int
) -> dict[str, Any]:
    """Publish the staged run, then write ``manifest.json`` (staged and published) last."""
    t_pub = time.time()
    published = publish(stage, out) if stage.resolve() != out.resolve() else None
    t1 = time.time()
    manifest["timing_s"] = {
        "probe": round(t_mid - t0, 2),
        "run": round(t_pub - t_mid, 2),
        "publish": round(t1 - t_pub, 2),
        "total": round(t1 - t0, 2),
    }
    manifest["finished_at"] = utc_iso(t1)
    if published is not None:
        names = published.pop("names")
        manifest["publish"] = published
        if published["size_mismatches"] and status == 0:
            status = 2
    else:
        names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        names = [n for n in names if n != MANIFEST]
    manifest["exit_status"] = status
    manifest["files"] = sorted(names) + [MANIFEST]
    write_json(stage / MANIFEST, manifest)
    if published is not None:
        copy_file(stage / MANIFEST, out / MANIFEST)
    return manifest


def _stage_for(out: Path, stage_dir: str | Path | None) -> Path:
    if stage_dir is not None:
        stage = Path(stage_dir)
        stage.mkdir(parents=True, exist_ok=True)
        return stage
    return Path(tempfile.mkdtemp(prefix=f"rtt-{out.name}-"))


def run_m1(
    out_dir: str | Path,
    seed: int = 0,
    video: bool = True,
    probe_warp: bool = False,
    argv: Sequence[str] = (),
    stage_dir: str | Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Run gate G1, staged locally, then publish to ``out_dir``; return the manifest and status.

    ``stage_dir`` defaults to a fresh temporary directory, removed afterwards; pass
    ``out_dir`` itself to write in place (only safe on a local filesystem).
    """
    from rtt.sim.m1 import DEFAULT_ENTRANTS, run_gate
    from rtt.sim.task import DEFAULT_TASK_PATH

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage = _stage_for(out, stage_dir)
    manifest = base_manifest("m1", out, argv)
    manifest["inputs"] = {
        "task_sha256": file_sha256(DEFAULT_TASK_PATH),
        "entrants_sha256": file_sha256(DEFAULT_ENTRANTS),
        "seed": seed,
        "video": video,
    }
    t0 = time.time()
    manifest["started_at"] = utc_iso(t0)
    manifest["environment"] = environment(probe_gl=video, probe_warp=probe_warp)
    t_env = time.time()
    status = 2
    try:
        report = run_gate(stage, seed=seed, video=video)
        print(report.table(), flush=True)
        write_json(stage / "report.json", _relative_files(report.to_dict(), stage))
        manifest["result"] = {
            "passed": report.passed,
            "cases": len(report.outcomes),
            "cases_as_expected": sum(o.passed for o in report.outcomes),
            "sim_steps": report.sim_steps,
            "sim_wall_s": round(report.sim_wall_s, 3),
            "gate_wall_s": round(report.wall_time_s, 2),
        }
        status = 0 if report.passed else 1
    except Exception:  # noqa: BLE001 - record the failure in the manifest, then exit 2
        manifest["error"] = traceback.format_exc()[-4000:]
        print(manifest["error"], file=sys.stderr, flush=True)
    try:
        manifest = _finish(manifest, stage, out, t0, t_env, status)
    finally:
        if stage_dir is None:
            shutil.rmtree(stage, ignore_errors=True)
    return manifest, manifest["exit_status"]


def run_probe(
    out_dir: str | Path,
    probe_warp: bool = True,
    argv: Sequence[str] = (),
    stage_dir: str | Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Probe the environment only (staged like :func:`run_m1`); return the manifest and status."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage = _stage_for(out, stage_dir)
    manifest = base_manifest("probe", out, argv)
    t0 = time.time()
    manifest["started_at"] = utc_iso(t0)
    env = environment(probe_gl=True, probe_warp=probe_warp)
    write_json(stage / "probe.json", env)
    manifest["environment"] = env
    try:
        manifest = _finish(manifest, stage, out, t0, time.time(), 0)
    finally:
        if stage_dir is None:
            shutil.rmtree(stage, ignore_errors=True)
    return manifest, manifest["exit_status"]


def run_search_job(
    out_dir: str | Path,
    arm: str,
    runs: int,
    budget: int,
    seeds: str | None = None,
    executor: str = "cpu",
    workers: int | None = None,
    spend_cap_usd: float = 2.0,
    argv: Sequence[str] = (),
    stage_dir: str | Path | None = None,
    space: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Run one search (staged locally, like :func:`run_m1`); return the manifest and status."""
    from rtt.parts.library import DEFAULT_PARTS_FILE
    from rtt.parts.suite import DEFAULT_TRAIN_V2
    from rtt.pipeline.search import SearchRequest, parse_seeds, parse_space, run_search_command
    from rtt.sim.conditions import DEFAULT_TRAIN_SUITE
    from rtt.sim.task import DEFAULT_TASK_PATH, DEFAULT_TASK_V2_PATH

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage = _stage_for(out, stage_dir)
    manifest = base_manifest("search", out, argv)
    run_seeds = parse_seeds(seeds, runs)
    design_space = parse_space(space)
    v2 = design_space == "freeform-v2"
    manifest["executor"] = executor
    manifest["inputs"] = {
        "design_space": design_space,
        "task_sha256": file_sha256(DEFAULT_TASK_V2_PATH if v2 else DEFAULT_TASK_PATH),
        "suite_sha256": file_sha256(DEFAULT_TRAIN_V2 if v2 else DEFAULT_TRAIN_SUITE),
        **({"parts_sha256": file_sha256(DEFAULT_PARTS_FILE)} if v2 else {}),
        "arm": arm,
        "runs": runs,
        "budget": budget,
        "seeds": run_seeds,
        "spend_cap_usd": spend_cap_usd,
    }
    t0 = time.time()
    manifest["started_at"] = utc_iso(t0)
    manifest["environment"] = environment(probe_gl=False, probe_warp=executor == "mjwarp")
    t_env = time.time()
    status = 2
    try:
        outcome = run_search_command(
            SearchRequest(
                arm=arm,
                runs=runs,
                budget=budget,
                out=stage / "search",
                seeds=run_seeds,
                executor=executor,
                workers=workers,
                spend_cap_usd=spend_cap_usd,
                space=design_space,
            )
        )
        search = outcome.manifest
        manifest["result"] = {
            "status": search.get("status"),
            "runs": search.get("runs"),
            "best": [r["best"] for r in search.get("per_run", [])],
            "designs_simulated": (search.get("sim") or {}).get("designs_simulated"),
            "episodes_simulated": (search.get("sim") or {}).get("episodes_simulated"),
            "evaluator_wall_s": (search.get("sim") or {}).get("evaluator_wall_s"),
            "llm": search.get("llm"),
        }
        status = 0 if search.get("status") == "complete" else 1
    except Exception:  # noqa: BLE001 - record the failure in the manifest, then exit 2
        manifest["error"] = traceback.format_exc()[-4000:]
        print(manifest["error"], file=sys.stderr, flush=True)
    try:
        manifest = _finish(manifest, stage, out, t0, t_env, status)
    finally:
        if stage_dir is None:
            shutil.rmtree(stage, ignore_errors=True)
    return manifest, manifest["exit_status"]


def run_gv4_job(
    out_dir: str | Path,
    workers: int | None = None,
    argv: Sequence[str] = (),
    stage_dir: str | Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Gate GV4's parity episodes on the job's CPUs (:func:`rtt.sim.gv4.run_parity`), staged
    like :func:`run_m1`; ``gv4.json`` is compared with the laptop's report afterwards."""
    from rtt.parts.library import DEFAULT_PARTS_FILE
    from rtt.parts.suite import DEFAULT_TRAIN_V2
    from rtt.sim.gv4 import run_parity
    from rtt.sim.task import DEFAULT_TASK_V2_PATH

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage = _stage_for(out, stage_dir)
    manifest = base_manifest("gv4", out, argv)
    manifest["inputs"] = {
        "task_sha256": file_sha256(DEFAULT_TASK_V2_PATH),
        "suite_sha256": file_sha256(DEFAULT_TRAIN_V2),
        "parts_sha256": file_sha256(DEFAULT_PARTS_FILE),
    }
    t0 = time.time()
    manifest["started_at"] = utc_iso(t0)
    manifest["environment"] = environment(probe_gl=False, probe_warp=False)
    t_env = time.time()
    status = 2
    try:
        report = run_parity(workers)
        write_json(stage / "gv4.json", report)
        manifest["result"] = {
            "episodes": report["episodes"],
            "records_sha256": report["records_sha256"],
            "designs": [{k: d[k] for k in ("label", "score")} for d in report["designs"]],
            "timing_s": report["timing_s"],
        }
        status = 0
    except Exception:  # noqa: BLE001 - record the failure in the manifest, then exit 2
        manifest["error"] = traceback.format_exc()[-4000:]
        print(manifest["error"], file=sys.stderr, flush=True)
    try:
        manifest = _finish(manifest, stage, out, t0, t_env, status)
    finally:
        if stage_dir is None:
            shutil.rmtree(stage, ignore_errors=True)
    return manifest, manifest["exit_status"]


def _summary_line(manifest: dict[str, Any]) -> str:
    env = manifest.get("environment", {})
    gpu = env.get("gpu")
    gpu0 = gpu[0] if isinstance(gpu, list) and gpu else {}
    summary = {
        "run_id": manifest.get("run_id"),
        "kind": manifest.get("kind"),
        "exit_status": manifest.get("exit_status"),
        "result": manifest.get("result"),
        "timing_s": manifest.get("timing_s"),
        "gpu": gpu0.get("name"),
        "driver": gpu0.get("driver_version"),
        "gl_renderer": (env.get("gl") or {}).get("renderer"),
        "warp_cuda_devices": (env.get("warp") or {}).get("cuda_device_count"),
    }
    return "RTT_SUMMARY " + json.dumps(summary, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rtt.cloud.entry", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    m1 = sub.add_parser("m1", help="run gate G1 and write replays, MP4s, report.json and manifest.json")
    m1.add_argument("--out", type=Path, required=True, help="run directory, e.g. /out/runs/m1-<ts>")
    m1.add_argument("--seed", type=int, default=0)
    m1.add_argument("--no-video", action="store_true", help="skip MP4 rendering")
    m1.add_argument("--probe-warp", action="store_true", help="also initialise Warp and list CUDA devices")
    probe = sub.add_parser("probe", help="record the environment only")
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--no-warp", action="store_true", help="skip the Warp/CUDA probe")
    search = sub.add_parser("search", help="run one search arm on the job's CPUs")
    search.add_argument("--out", type=Path, required=True)
    search.add_argument("--arm", required=True, choices=("random", "nemotron", "zeroshot"))
    search.add_argument("--runs", type=int, required=True)
    search.add_argument("--budget", type=int, required=True)
    search.add_argument("--seeds", default=None, help="comma-separated run seeds (default 1..runs)")
    search.add_argument("--executor", default="cpu", choices=("cpu", "mjwarp"))
    search.add_argument("--workers", type=int, default=None)
    search.add_argument("--spend-cap", type=float, default=2.0, help="live model spend cap, USD")
    search.add_argument(
        "--space", default=None, choices=("box-v1", "freeform-v2", "v1", "v2"), help="design space"
    )
    gv4 = sub.add_parser("gv4", help="gate GV4: the v2 parity episodes, gv4.json and manifest.json")
    gv4.add_argument("--out", type=Path, required=True)
    gv4.add_argument("--workers", type=int, default=None)
    for p in (m1, probe, search, gv4):
        p.add_argument(
            "--stage",
            type=Path,
            default=None,
            help="local staging directory (default: a temporary directory, removed afterwards)",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(args_list)
    if args.kind == "m1":
        manifest, status = run_m1(
            args.out,
            seed=args.seed,
            video=not args.no_video,
            probe_warp=args.probe_warp,
            argv=args_list,
            stage_dir=args.stage,
        )
    elif args.kind == "search":
        manifest, status = run_search_job(
            args.out,
            args.arm,
            args.runs,
            args.budget,
            seeds=args.seeds,
            executor=args.executor,
            workers=args.workers,
            spend_cap_usd=args.spend_cap,
            argv=args_list,
            stage_dir=args.stage,
            space=args.space,
        )
    elif args.kind == "gv4":
        manifest, status = run_gv4_job(args.out, workers=args.workers, argv=args_list, stage_dir=args.stage)
    else:
        manifest, status = run_probe(
            args.out, probe_warp=not args.no_warp, argv=args_list, stage_dir=args.stage
        )
    print(_summary_line(manifest), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
