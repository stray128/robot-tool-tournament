"""Laptop-side command line for Nebius jobs: ``python -m rtt.cloud <command>``.

Commands:

* ``capacity``: the fallback platforms and whether each has capacity now.
* ``run {m1,probe,search}``: submit a run in the pinned image (``RTT_IMAGE``), pick the first
  platform with capacity, cancel and fall back if provisioning stalls, wait, record the
  spend, fetch ``runs/<run_id>/`` from the bucket to ``runs/cloud/<run_id>/`` and verify it.
  ``--dry-run`` validates the job spec server-side and creates nothing.
* ``status``, ``logs``, ``cancel``, ``list``: one job, by ID (``list``: the project's jobs).
* ``fetch RUN_ID``: download and verify a run.
* ``spend``: the spend ledger and its totals.

Settings come from :func:`rtt.cloud.config.load_config`; no resource ID is ever printed
except a job's own ID.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rtt.cloud.config import ConfigError, NebiusConfig, load_config
from rtt.cloud.cost import PRICES, PRICES_SOURCE, SpendCapError, SpendLedger, read_spend, summarize
from rtt.cloud.nebius_jobs import (
    DEFAULT_CANDIDATES,
    Candidate,
    Job,
    JobUpdate,
    NebiusCLIError,
    NebiusJobs,
    rank_candidates,
)
from rtt.cloud.runs import (
    DEFAULT_LEDGER,
    KINDS,
    LOCAL_RUNS,
    SearchJobArgs,
    job_spec_for,
    new_run_id,
    run_prefix,
    verify_run,
)
from rtt.cloud.storage import BucketStore


def _fmt_s(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} s"


def _job_line(job: Job) -> str:
    return (
        f"{job.id}  {job.name:32}  {job.state:12}  {job.platform or '-':14}"
        f"  prov {_fmt_s(job.provisioning_s):>7}  run {_fmt_s(job.run_s):>7}"
        f"  created {job.created_at.isoformat(timespec='seconds') if job.created_at else '-'}"
    )


def _job_record(job: Job) -> dict[str, object]:
    """What we keep about a job next to its run (no project, subnet or bucket IDs)."""
    return {
        "job_id": job.id,
        "name": job.name,
        "state": job.state,
        "state_details": job.state_details,
        "platform": job.platform,
        "preset": job.preset,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "provisioning_s": job.provisioning_s,
        "run_s": job.run_s,
        "wall_s": job.wall_s(),
    }


# --------------------------------------------------------------------------- commands


def cmd_capacity(cfg: NebiusConfig, jobs: NebiusJobs, args: argparse.Namespace) -> int:
    advice = jobs.capacity()
    usable = set(rank_candidates(advice, DEFAULT_CANDIDATES, args.preemptible, args.min_level))
    kind = "preemptible" if args.preemptible else "on-demand"
    print(f"{kind} capacity in {cfg.region} (minimum level {args.min_level}); prices: {PRICES_SOURCE}")
    for cand in DEFAULT_CANDIDATES:
        rows = [
            r for r in advice if (r.platform, r.preset, r.region) == (cand.platform, cand.preset, cand.region)
        ]
        levels = ", ".join(
            f"{r.fabric or '-'}:{r.availability(args.preemptible).level}"
            f"({r.availability(args.preemptible).available or 0}/{r.availability(args.preemptible).limit or 0})"
            for r in rows
        )
        price = PRICES[cand.platform]
        mark = "usable" if cand in usable else "skip"
        print(
            f"  {mark:6}  {cand.platform:14} {cand.preset:20} ${price.gpu_hour_usd:.2f}/GPU-h"
            f" [{price.tag}]  {levels or 'no advice row'}"
        )
    return 0


def _print_update(update: JobUpdate) -> None:
    job = update.job
    print(f"  [{update.elapsed_s:6.0f} s] {job.name}: {job.state} on {job.platform}", flush=True)


def cmd_run(cfg: NebiusConfig, jobs: NebiusJobs, args: argparse.Namespace) -> int:
    run_id = args.run_id or new_run_id(args.kind)
    search = None
    if args.kind == "search":
        if not (args.arm and args.runs and args.budget):
            raise ValueError("a search run needs --arm, --runs and --budget")
        search = SearchJobArgs(
            arm=args.arm,
            runs=args.runs,
            budget=args.budget,
            seeds=args.seeds,
            spend_cap_usd=args.spend_cap,
        )
    spec = job_spec_for(
        cfg,
        args.kind,
        run_id,
        seed=args.seed,
        video=not args.no_video,
        probe_warp=not args.no_warp,
        preemptible=args.preemptible,
        timeout=args.timeout,
        search=search,
    )
    if args.dry_run:
        print(f"{spec.name}: {jobs.dry_run(spec)}")
        return 0

    ledger = SpendLedger(args.ledger, cap_usd=args.cap_usd)
    candidates = DEFAULT_CANDIDATES
    if args.platform:
        candidates = (Candidate(args.platform, args.preset or cfg.gpu_preset, cfg.region),)
    print(
        f"run {run_id}: submitting {spec.name} ({args.kind}); spend so far ${ledger.total_usd():.2f}",
        flush=True,
    )
    outcome = jobs.run(
        spec,
        candidates,
        ledger=ledger,
        expected_run_s=args.expected_run_s,
        provisioning_timeout_s=args.provisioning_timeout_s,
        poll_s=args.poll_s,
        check_capacity=not args.no_capacity_check,
        run_id=run_id,
        on_update=_print_update,
    )
    job = outcome.job
    print(_job_line(job))
    for a in outcome.attempts:
        print(
            f"  attempt {a.name} on {a.platform}: {a.state}, provisioning {_fmt_s(a.provisioning_s)},"
            f" run {_fmt_s(a.run_s)}, wall {_fmt_s(a.wall_s)}, est ${a.est_usd:.3f}"
            f" (run-only ${a.est_usd_run:.3f})"
        )
    print(f"spend ledger total ${ledger.total_usd():.2f} (cap ${args.cap_usd:.2f})")

    local = Path(args.fetch_dir) / run_id
    local.mkdir(parents=True, exist_ok=True)
    try:
        log_text = jobs.logs(job.id, since="24h")
    except NebiusCLIError as exc:
        log_text = f"(logs unavailable: {exc})\n"
    (local / "job.log").write_text(log_text, encoding="utf-8")
    record = _job_record(job) | {"attempts": [a.to_json() for a in outcome.attempts], "run_id": run_id}
    (local / "job.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.no_fetch:
        return 0 if job.succeeded else 1

    store = BucketStore.from_config(cfg)
    store.put_json(f"{run_prefix(run_id)}/job.json", record)
    files = store.download_prefix(run_prefix(run_id), local)
    print(f"fetched {len(files)} files to {local}")
    check = verify_run(local)
    print(check.describe())
    return 0 if job.succeeded and check.ok else 1


def cmd_status(cfg: NebiusConfig, jobs: NebiusJobs, args: argparse.Namespace) -> int:
    job = jobs.get(args.job_id)
    print(json.dumps(_job_record(job), indent=2) if args.json else _job_line(job))
    return 0


def cmd_logs(cfg: NebiusConfig, jobs: NebiusJobs, args: argparse.Namespace) -> int:
    sys.stdout.write(jobs.logs(args.job_id, since=args.since, tail=args.tail, timestamps=args.timestamps))
    return 0


def cmd_cancel(cfg: NebiusConfig, jobs: NebiusJobs, args: argparse.Namespace) -> int:
    jobs.cancel(args.job_id)
    print(f"cancellation requested for {args.job_id}")
    if args.wait:
        print(_job_line(jobs.wait_cancelled(args.job_id)))
    return 0


def cmd_list(cfg: NebiusConfig, jobs: NebiusJobs, args: argparse.Namespace) -> int:
    for job in sorted(jobs.list_jobs(), key=lambda j: j.created_at.isoformat() if j.created_at else ""):
        print(_job_line(job))
    return 0


def cmd_fetch(cfg: NebiusConfig, jobs: NebiusJobs, args: argparse.Namespace) -> int:
    local = Path(args.fetch_dir) / args.run_id
    files = BucketStore.from_config(cfg).download_prefix(run_prefix(args.run_id), local)
    print(f"fetched {len(files)} files to {local}")
    check = verify_run(local)
    print(check.describe())
    return 0 if check.ok else 1


def cmd_spend(cfg: NebiusConfig, jobs: NebiusJobs, args: argparse.Namespace) -> int:
    entries = read_spend(args.ledger)
    for e in entries:
        print(
            f"{e.ts}  {e.name:32}  {e.platform:14}  {e.state:10}  wall {e.wall_s:6.0f} s"
            f"  run {e.run_s or 0:6.0f} s  est ${e.est_usd:.3f} (run-only ${e.est_usd_run:.3f})"
        )
    totals = summarize(entries)
    print(
        f"{int(totals['jobs'])} jobs, wall {totals['wall_s']:.0f} s, run {totals['run_s']:.0f} s,"
        f" est ${totals['est_usd']:.2f} (run-only ${totals['est_usd_run']:.2f})"
    )
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtt.cloud", description="Nebius Serverless AI Jobs for rtt"
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="command")

    cap = sub.add_parser("capacity", help="show which fallback platforms have capacity now")
    cap.add_argument("--preemptible", action="store_true")
    cap.add_argument("--min-level", default="MEDIUM", choices=("HIGH", "MEDIUM", "LOW"))
    cap.set_defaults(handler=cmd_capacity)

    run = sub.add_parser("run", help="submit a run, wait, record spend, fetch and verify")
    run.add_argument("kind", choices=KINDS)
    run.add_argument("--run-id", help="default: <kind>-<UTC timestamp>")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-video", action="store_true", help="m1: skip MP4 rendering")
    run.add_argument("--no-warp", action="store_true", help="skip the Warp/CUDA probe")
    run.add_argument("--arm", choices=("random", "nemotron", "zeroshot"), help="search: arm")
    run.add_argument("--runs", type=int, help="search: independent runs")
    run.add_argument("--budget", type=int, help="search: designs per run")
    run.add_argument("--seeds", default=None, help="search: comma-separated run seeds (default 1..runs)")
    run.add_argument("--spend-cap", type=float, default=2.0, help="search: live model spend cap, USD")
    run.add_argument("--platform", help="force one platform (no fallback), e.g. gpu-h100-sxm")
    run.add_argument("--preset", help="preset for --platform (default NEBIUS_GPU_PRESET)")
    run.add_argument("--preemptible", action="store_true", help="spot VM (follows the spot price)")
    run.add_argument("--timeout", default="1h", help="job timeout, whole hours 1h-168h (default 1h)")
    run.add_argument("--cap-usd", type=float, default=3.0, help="spend cap over the ledger (default 3)")
    run.add_argument(
        "--expected-run-s", type=float, default=600.0, help="run-time estimate for the cap check"
    )
    run.add_argument("--provisioning-timeout-s", type=float, default=900.0)
    run.add_argument("--poll-s", type=float, default=15.0)
    run.add_argument("--no-capacity-check", action="store_true", help="skip capacity advice")
    run.add_argument("--dry-run", action="store_true", help="validate the job server-side; create nothing")
    run.add_argument("--no-fetch", action="store_true", help="do not download the run afterwards")
    run.add_argument("--fetch-dir", type=Path, default=LOCAL_RUNS)
    run.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    run.set_defaults(handler=cmd_run)

    st = sub.add_parser("status", help="one job's state and timings")
    st.add_argument("job_id")
    st.add_argument("--json", action="store_true")
    st.set_defaults(handler=cmd_status)

    lg = sub.add_parser("logs", help="a job's container logs")
    lg.add_argument("job_id")
    lg.add_argument("--since", default="24h")
    lg.add_argument("--tail", type=int)
    lg.add_argument("--timestamps", action="store_true")
    lg.set_defaults(handler=cmd_logs)

    cn = sub.add_parser("cancel", help="cancel a job")
    cn.add_argument("job_id")
    cn.add_argument("--wait", action="store_true", help="poll until the job is terminal")
    cn.set_defaults(handler=cmd_cancel)

    ls = sub.add_parser("list", help="the project's jobs, oldest first")
    ls.set_defaults(handler=cmd_list)

    fe = sub.add_parser("fetch", help="download runs/<run_id>/ from the bucket and verify it")
    fe.add_argument("run_id")
    fe.add_argument("--fetch-dir", type=Path, default=LOCAL_RUNS)
    fe.set_defaults(handler=cmd_fetch)

    sp = sub.add_parser("spend", help="print the spend ledger")
    sp.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    sp.set_defaults(handler=cmd_spend)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config()
        jobs = NebiusJobs.from_config(cfg)
        return int(args.handler(cfg, jobs, args))
    except (ConfigError, NebiusCLIError, SpendCapError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
