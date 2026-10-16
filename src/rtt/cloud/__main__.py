"""Laptop-side command line for Nebius jobs: ``python -m rtt.cloud <command>``.

Commands:

* ``capacity``: for each target, which platforms can take a job now (GPU capacity
  advice, CPU vCPU quota) and what they cost.
* ``run {m1,probe,search}``: submit a run in the pinned image (``RTT_IMAGE``) on a
  :class:`~rtt.cloud.nebius_jobs.Target`, pick the first platform with capacity, cancel
  and fall back if provisioning stalls, wait, record the spend, fetch ``runs/<run_id>/``
  from that region's bucket to ``runs/cloud/<run_id>/`` and verify it. The default target
  is ``uks2-cpu`` (CPU-only, uk-south2: the executor of record). When it has no capacity
  or every attempt stalls before start, the job moves to ``eun1-gpu`` (H100 fallback)
  unless ``--no-fallback`` or an explicit ``--vcpus`` is given. ``--dry-run`` validates the job spec server-side and creates
  nothing.
* ``status``, ``logs``, ``cancel``: one job, by ID. ``list``: the jobs of every target's
  project.
* ``fetch RUN_ID``: download and verify a run (from the region recorded next to it).
* ``spend``: the spend ledger and its totals.

Settings come from :func:`rtt.cloud.config.load_config`; no resource ID is ever printed
except a job's own ID.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rtt.cloud.config import ConfigError, NebiusConfig, load_config
from rtt.cloud.cost import (
    PRICES_SOURCE,
    JobSpend,
    SpendCapError,
    SpendLedger,
    price_label,
    read_spend,
    summarize,
)
from rtt.cloud.nebius_jobs import (
    CPU_VCPU_CHOICES,
    CPU_VCPU_QUOTA,
    DEFAULT_TARGET,
    FALLBACK_TARGET,
    TARGETS,
    Job,
    JobUpdate,
    NebiusCLIError,
    NebiusJobs,
    NoCapacityError,
    Runner,
    RunOutcome,
    Target,
    get_target,
    rank_candidates,
    subprocess_runner,
    target_for_platform,
)
from rtt.cloud.runs import (
    DEFAULT_LEDGER,
    KINDS,
    LOCAL_RUNS,
    SearchJobArgs,
    job_spec_for,
    new_run_id,
    run_prefix,
    validate_run_id,
    verify_run,
)
from rtt.cloud.storage import BucketStore


@dataclass(frozen=True)
class Context:
    """How handlers reach the cloud; tests substitute fakes for both factories."""

    config: NebiusConfig  # home-region config
    jobs_for: Callable[[NebiusConfig], NebiusJobs]
    store_for: Callable[[NebiusConfig], BucketStore] = BucketStore.from_config

    def region(self, target: Target) -> tuple[NebiusConfig, NebiusJobs]:
        cfg = self.config.for_region(target.region)
        return cfg, self.jobs_for(cfg)


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


def never_started(job: Job) -> bool:
    """A job that ended without its container ever starting (image pull or VM start failed)."""
    return job.terminal and not job.succeeded and job.started_at is None and job.state != "CANCELLED"


def check_same_image(home: NebiusConfig, region: NebiusConfig) -> None:
    """Refuse a region whose image (``RTT_IMAGE_<SUFFIX>``) is another build than the home one.

    Every registry must hold the same digest, so a scored job and its fallback run the
    same code wherever they land.

    Raises:
        ValueError: both images are set and their digests differ.
    """
    if not (home.image and region.image) or home.image == region.image:
        return
    home_digest = home.image.partition("@")[2]
    region_digest = region.image.partition("@")[2]
    if home_digest != region_digest:
        raise ValueError(
            f"the {region.region} image ({region_digest[:19]}…) is not the home image"
            f" ({home_digest[:19]}…): push one build to every registry with jobs/nebius/push_image.sh"
        )


def _resolve_target(args: argparse.Namespace) -> Target:
    """``--target`` (default: inferred from ``--platform``, else :data:`DEFAULT_TARGET`)."""
    name = args.target or (target_for_platform(args.platform).name if args.platform else DEFAULT_TARGET)
    target = get_target(name, args.vcpus if TARGETS[name].kind == "cpu" else None)
    if args.platform:
        target = target.only(args.platform, args.preset)
    return target


# --------------------------------------------------------------------------- commands


def cmd_capacity(ctx: Context, args: argparse.Namespace) -> int:
    kind = "preemptible" if args.preemptible else "on-demand"
    print(f"{kind} capacity (GPU minimum level {args.min_level}); prices: {PRICES_SOURCE}")
    for name in TARGETS:
        target = get_target(name, args.vcpus if TARGETS[name].kind == "cpu" else None)
        _, jobs = ctx.region(target)
        tag = " (default)" if name == DEFAULT_TARGET else " (fallback)" if name == FALLBACK_TARGET else ""
        print(f"{name}{tag}: {target.region}, {target.description}")
        if target.kind == "cpu":
            quota = jobs.quota(CPU_VCPU_QUOTA, target.region)
            usable = set(rank_candidates([], target.candidates, cpu_free_vcpus={target.region: quota.free}))
            for cand in target.candidates:
                mark = "usable" if cand in usable else "skip"
                print(
                    f"  {mark:6}  {cand.platform:14} {cand.preset:20} {price_label(cand.platform, cand.preset)}"
                    f"  vCPU quota {quota.usage}/{quota.limit} used"
                )
            continue
        advice = jobs.capacity()
        usable = set(rank_candidates(advice, target.candidates, args.preemptible, args.min_level))
        for cand in target.candidates:
            rows = [
                r
                for r in advice
                if (r.platform, r.preset, r.region) == (cand.platform, cand.preset, cand.region)
            ]
            levels = ", ".join(
                f"{r.fabric or '-'}:{r.availability(args.preemptible).level}"
                f"({r.availability(args.preemptible).available or 0}/{r.availability(args.preemptible).limit or 0})"
                for r in rows
            )
            mark = "usable" if cand in usable else "skip"
            print(
                f"  {mark:6}  {cand.platform:14} {cand.preset:20} {price_label(cand.platform, cand.preset)}"
                f"  {levels or 'no advice row'}"
            )
    return 0


def _print_update(update: JobUpdate) -> None:
    job = update.job
    print(f"  [{update.elapsed_s:6.0f} s] {job.name}: {job.state} on {job.platform}", flush=True)


def cmd_run(ctx: Context, args: argparse.Namespace) -> int:
    run_id = validate_run_id(args.run_id or new_run_id(args.kind))
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
            workers=args.workers,
        )
    first = _resolve_target(args)
    chain = [first]
    # An explicit --vcpus is a request for that machine size: never swap it silently for a
    # 16-vCPU H100 (about 8.5x the price of the CPU preset) when the quota cannot fit it.
    if args.fallback and not args.platform and args.vcpus is None and first.name != FALLBACK_TARGET:
        chain.append(get_target(FALLBACK_TARGET))

    ledger = SpendLedger(args.ledger, cap_usd=args.cap_usd)
    attempts: list[JobSpend] = []
    outcome: RunOutcome | None = None
    ran_on: Target | None = None
    region_cfg: NebiusConfig | None = None
    for i, target in enumerate(chain):
        region_cfg, jobs = ctx.region(target)
        try:
            check_same_image(ctx.config, region_cfg)
            spec = job_spec_for(
                region_cfg,
                args.kind,
                run_id,
                seed=args.seed,
                video=not args.no_video,
                # Warp needs a GPU; on a CPU target the probe would only time its own failure
                probe_warp=not args.no_warp and target.kind == "gpu",
                preemptible=args.preemptible,
                timeout=args.timeout,
                search=search,
                candidate=target.candidates[0],
            )
        except ConfigError as exc:
            if i + 1 < len(chain):
                print(f"{target.name}: not configured ({exc}); falling back to {chain[i + 1].name}")
                continue
            raise
        if args.dry_run:
            print(f"{target.name} ({target.region}) {spec.name}: {jobs.dry_run(spec)}")
            continue
        print(
            f"run {run_id}: submitting {spec.name} ({args.kind}) to {target.name} ({target.region});"
            f" spend so far ${ledger.total_usd():.2f}",
            flush=True,
        )
        try:
            outcome = jobs.run(
                spec,
                target.candidates,
                ledger=ledger,
                expected_run_s=args.expected_run_s,
                provisioning_timeout_s=args.provisioning_timeout_s,
                poll_s=args.poll_s,
                check_capacity=not args.no_capacity_check,
                run_id=run_id,
                on_update=_print_update,
            )
        except NoCapacityError as exc:
            attempts += exc.attempts
            if i + 1 < len(chain):
                print(f"{target.name}: {exc}; falling back to {chain[i + 1].name}", flush=True)
                continue
            raise
        attempts += outcome.attempts
        if never_started(outcome.job) and i + 1 < len(chain):
            # the platform failed before our code ran (e.g. StartFailed on image pull)
            print(
                f"{target.name}: {outcome.job.name} {outcome.job.state} before start"
                f" ({outcome.job.state_details}); falling back to {chain[i + 1].name}",
                flush=True,
            )
            continue
        ran_on = target
        break
    if args.dry_run:
        return 0
    assert outcome is not None and ran_on is not None and region_cfg is not None

    job = outcome.job
    print(_job_line(job))
    for a in attempts:
        print(
            f"  attempt {a.name} on {a.platform}/{a.preset} ({a.region}): {a.state},"
            f" provisioning {_fmt_s(a.provisioning_s)}, run {_fmt_s(a.run_s)}, wall {_fmt_s(a.wall_s)},"
            f" est ${a.est_usd:.3f} (run-only ${a.est_usd_run:.3f})"
        )
    print(f"spend ledger total ${ledger.total_usd():.2f} (cap ${args.cap_usd:.2f})")

    local = Path(args.fetch_dir) / run_id
    local.mkdir(parents=True, exist_ok=True)
    _, jobs = ctx.region(ran_on)
    try:
        log_text = jobs.logs(job.id, since="24h")
    except NebiusCLIError as exc:
        log_text = f"(logs unavailable: {exc})\n"
    (local / "job.log").write_text(log_text, encoding="utf-8")
    record = _job_record(job) | {
        "attempts": [a.to_json() for a in attempts],
        "run_id": run_id,
        "target": ran_on.name,
        "region": outcome.candidate.region,
        "fallback_from": first.name if ran_on.name != first.name else None,
        "price": price_label(outcome.candidate.platform, outcome.candidate.preset),
    }
    (local / "job.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.no_fetch:
        return 0 if job.succeeded else 1

    store = ctx.store_for(region_cfg)
    store.put_json(f"{run_prefix(run_id)}/job.json", record)
    files = store.download_prefix(run_prefix(run_id), local)
    print(f"fetched {len(files)} files to {local} from the {region_cfg.region} bucket")
    check = verify_run(local)
    print(check.describe())
    return 0 if job.succeeded and check.ok else 1


def cmd_status(ctx: Context, args: argparse.Namespace) -> int:
    job = ctx.jobs_for(ctx.config).get(args.job_id)
    print(json.dumps(_job_record(job), indent=2) if args.json else _job_line(job))
    return 0


def cmd_logs(ctx: Context, args: argparse.Namespace) -> int:
    jobs = ctx.jobs_for(ctx.config)
    sys.stdout.write(jobs.logs(args.job_id, since=args.since, tail=args.tail, timestamps=args.timestamps))
    return 0


def cmd_cancel(ctx: Context, args: argparse.Namespace) -> int:
    jobs = ctx.jobs_for(ctx.config)
    jobs.cancel(args.job_id)
    print(f"cancellation requested for {args.job_id}")
    if args.wait:
        print(_job_line(jobs.wait_cancelled(args.job_id)))
    return 0


def cmd_list(ctx: Context, args: argparse.Namespace) -> int:
    names = [args.target] if args.target else list(TARGETS)
    for name in names:
        target = TARGETS[name]
        cfg, jobs = ctx.region(target)
        if not cfg.project_id:
            print(f"# {name} ({target.region}): no project configured ({cfg.env_var('project_id')})")
            continue
        print(f"# {name} ({target.region})")
        for job in sorted(jobs.list_jobs(), key=lambda j: j.created_at.isoformat() if j.created_at else ""):
            print(_job_line(job))
    return 0


def _run_region(local: Path, target: str | None) -> str:
    """Region of a run: ``--target``, else the ``job.json`` fetched with it, else the default target's."""
    if target:
        return TARGETS[target].region
    record = local / "job.json"
    if record.is_file():
        try:
            region = json.loads(record.read_text(encoding="utf-8")).get("region")
        except json.JSONDecodeError:
            region = None
        if region:
            return str(region)
    return TARGETS[DEFAULT_TARGET].region


def cmd_fetch(ctx: Context, args: argparse.Namespace) -> int:
    local = Path(args.fetch_dir) / validate_run_id(args.run_id)
    cfg = ctx.config.for_region(_run_region(local, args.target))
    files = ctx.store_for(cfg).download_prefix(run_prefix(args.run_id), local)
    print(f"fetched {len(files)} files to {local} from the {cfg.region} bucket")
    check = verify_run(local)
    print(check.describe())
    return 0 if check.ok else 1


def cmd_spend(ctx: Context, args: argparse.Namespace) -> int:
    entries = read_spend(args.ledger)
    for e in entries:
        print(
            f"{e.ts}  {e.name:32}  {e.platform:14}  {e.region:10}  {e.state:10}  wall {e.wall_s:6.0f} s"
            f"  run {e.run_s or 0:6.0f} s  est ${e.est_usd:.3f} (run-only ${e.est_usd_run:.3f})"
        )
    totals = summarize(entries)
    print(
        f"{int(totals['jobs'])} jobs, wall {totals['wall_s']:.0f} s, run {totals['run_s']:.0f} s,"
        f" est ${totals['est_usd']:.2f} (run-only ${totals['est_usd_run']:.2f})"
    )
    inflight = SpendLedger(args.ledger).inflight()
    for r in inflight:
        print(f"in flight: {r.label} holds ${r.usd:.3f} (pid {r.pid}, since {r.ts})")
    if inflight:
        print(f"{len(inflight)} job(s) in flight hold ${sum(r.usd for r in inflight):.2f} against the cap")
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rtt.cloud", description="Nebius Serverless AI Jobs for rtt"
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="command")
    targets = tuple(TARGETS)

    cap = sub.add_parser("capacity", help="show which platforms of each target can take a job now")
    cap.add_argument("--preemptible", action="store_true")
    cap.add_argument("--min-level", default="MEDIUM", choices=("HIGH", "MEDIUM", "LOW"))
    cap.add_argument("--vcpus", type=int, default=None, choices=CPU_VCPU_CHOICES, help="CPU preset size")
    cap.set_defaults(handler=cmd_capacity)

    run = sub.add_parser("run", help="submit a run, wait, record spend, fetch and verify")
    run.add_argument("kind", choices=KINDS)
    run.add_argument("--run-id", help="default: <kind>-<UTC timestamp>")
    run.add_argument(
        "--target",
        choices=targets,
        default=None,
        help=f"region profile (default {DEFAULT_TARGET}; inferred from --platform when given)",
    )
    run.add_argument(
        "--vcpus",
        type=int,
        default=None,
        choices=CPU_VCPU_CHOICES,
        help="CPU target: vCPUs of the preset (default 16, 4 GiB RAM each)",
    )
    run.add_argument(
        "--no-fallback",
        dest="fallback",
        action="store_false",
        help=f"do not move to {FALLBACK_TARGET} when the target has no capacity",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-video", action="store_true", help="m1: skip MP4 rendering")
    run.add_argument("--no-warp", action="store_true", help="skip the Warp/CUDA probe (always off on CPU)")
    run.add_argument("--arm", choices=("random", "nemotron", "zeroshot"), help="search: arm")
    run.add_argument("--runs", type=int, help="search: independent runs")
    run.add_argument("--budget", type=int, help="search: designs per run")
    run.add_argument("--seeds", default=None, help="search: comma-separated run seeds (default 1..runs)")
    run.add_argument("--workers", type=int, default=None, help="search: CPU processes (default: all vCPUs)")
    run.add_argument("--spend-cap", type=float, default=2.0, help="search: live model spend cap, USD")
    run.add_argument("--platform", help="force one platform (no fallback), e.g. cpu-d3 or gpu-h100-sxm")
    run.add_argument("--preset", help="preset for --platform (default: the target's preset)")
    run.add_argument("--preemptible", action="store_true", help="spot VM (follows the spot price)")
    run.add_argument("--timeout", default="1h", help="job timeout, whole hours 1h-168h (default 1h)")
    run.add_argument("--cap-usd", type=float, default=3.0, help="spend cap over the ledger (default 3)")
    run.add_argument(
        "--expected-run-s", type=float, default=600.0, help="run-time estimate for the cap check"
    )
    run.add_argument("--provisioning-timeout-s", type=float, default=900.0)
    run.add_argument("--poll-s", type=float, default=15.0)
    run.add_argument("--no-capacity-check", action="store_true", help="skip capacity advice and quota")
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

    ls = sub.add_parser("list", help="the jobs of every target's project, oldest first")
    ls.add_argument("--target", choices=targets, default=None, help="one target only")
    ls.set_defaults(handler=cmd_list)

    fe = sub.add_parser("fetch", help="download runs/<run_id>/ from the bucket and verify it")
    fe.add_argument("run_id")
    fe.add_argument("--target", choices=targets, default=None, help="default: the region in job.json")
    fe.add_argument("--fetch-dir", type=Path, default=LOCAL_RUNS)
    fe.set_defaults(handler=cmd_fetch)

    sp = sub.add_parser("spend", help="print the spend ledger")
    sp.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    sp.set_defaults(handler=cmd_spend)
    return parser


def main(argv: Sequence[str] | None = None, runner: Runner = subprocess_runner) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config()
        ctx = Context(cfg, lambda c: NebiusJobs.from_config(c, runner))
        return int(args.handler(ctx, args))
    except (ConfigError, NebiusCLIError, SpendCapError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
