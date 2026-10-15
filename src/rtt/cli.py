"""Command-line entry point: ``rtt <command>``.

* ``rtt m1``: gate G1. Runs every manual entrant plus three deliberate failures on
  the G1 box, prints a results table and writes a replay and an MP4 per case to
  ``data/demo/m1/``. Exit status 0 when the gate passes.
* ``rtt probe-llm``: prints a summary of ``docs/PROBE_LLM.md`` without calling any
  API. ``--live`` re-runs the Token Factory probe (cache first, under a hard spend
  cap); ``--replay`` rebuilds the report from the response cache only.
* ``rtt export-geom <entrant> --out finger.stl``: writes a manual entrant's finger
  as an STL in millimetres, or a palm-and-fingers preview for a ``.glb`` path.
* ``rtt calibrate``: gate G3. Scores the manual entrants and 64 Sobol designs on the
  training suite and checks the calibration targets (:mod:`rtt.analysis.calibrate`).
* ``rtt search --arm {random,nemotron,zeroshot} --runs R --budget B --out DIR``: R
  independent search runs in lockstep (:func:`rtt.search.runner.run_search`).
* ``rtt replay-llm --from DIR --out DIR2``: re-runs an LLM search from its response
  cache with zero API calls and checks every run reproduces.
* ``rtt heldout --seed-file PATH --frozen DIR...``: scores frozen designs on the
  commit-reveal held-out suite (:mod:`rtt.pipeline.heldout`).
* ``rtt certify``: the robustness certificate of a design on training objects.
* ``rtt slice``: the vertical slice, Nemotron proposes, the simulator scores, replays
  and a manifest are written, and the web data is exported (:mod:`rtt.pipeline.slice`).
* ``rtt export-web RUN_DIR...``: publishes run directories into ``web/public/data``.
* ``rtt job {smoke,m1,search}``: runs the same work as a Nebius Serverless AI Job in the
  pinned image, waits, records the spend and fetches the result (:mod:`rtt.cloud`);
  ``rtt cloud ...`` exposes the rest of the job tooling (status, logs, list, spend).
* ``rtt version``: prints the package version.

Heavy dependencies (MuJoCo, trimesh) are imported only by the command that needs them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rtt.llm import probe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENTRANTS = REPO_ROOT / "configs" / "entrants.yaml"
DEFAULT_M1_OUT = REPO_ROOT / "data" / "demo" / "m1"
DEFAULT_TASK = REPO_ROOT / "configs" / "task_v1.yaml"
DEFAULT_SUITE = REPO_ROOT / "configs" / "suites" / "train_v1.yaml"
DEFAULT_COMMITMENT = REPO_ROOT / "configs" / "suites" / "heldout_v1.sha256"
RUNS_DIR = REPO_ROOT / "runs"
ARM_CHOICES = ("random", "nemotron", "zeroshot")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _display(path: Path) -> str:
    """``path`` relative to the working directory when it lies below it."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- commands


def _cmd_version(_: argparse.Namespace) -> int:
    from importlib.metadata import version

    print(version("robot-tool-tournament"))
    return 0


def _cmd_m1(args: argparse.Namespace) -> int:
    from rtt.sim.m1 import run_gate
    from rtt.sim.task import load_task

    out = None if args.no_files else args.out
    report = run_gate(
        out,
        seed=args.seed,
        video=not args.no_video,
        task=load_task(args.task) if args.task else None,
        entrants_path=args.entrants,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.table())
        if out is not None:
            n_files = sum(len(o.files) for o in report.outcomes)
            kinds = "replays" if args.no_video else "replays and MP4s"
            print(f"Wrote {n_files} files ({kinds}, one set per case) to {_display(Path(out))}/")
    return 0 if report.passed else 1


def _cmd_probe_llm(args: argparse.Namespace) -> int:
    if args.live or args.replay:
        return probe.run_probe(args)
    if not args.doc.is_file():
        print(f"error: no probe report at {_display(args.doc)}; run `rtt probe-llm --live` to create it")
        return 1
    print(probe.report_summary(args.doc))
    return 0


def _cmd_export_geom(args: argparse.Namespace) -> int:
    from rtt.design.export import export_glb, export_stl, to_trimesh
    from rtt.design.generate import generate_finger, mass_and_inertia
    from rtt.design.schema import load_entrants
    from rtt.design.validate import validate
    from rtt.sim.task import gripper_task_for, load_task

    suffix = args.out.suffix.lower()
    if suffix not in (".stl", ".glb"):
        args.subparser.error(f"--out must end in .stl or .glb, got {args.out.name!r}")
    designs = {d.name: d for d in load_entrants(args.entrants)}
    design = designs.get(args.entrant)
    if design is None:
        names = ", ".join(str(n) for n in designs)
        args.subparser.error(f"unknown entrant {args.entrant!r}; {_display(args.entrants)} has: {names}")

    gripper = gripper_task_for(load_task(args.task))
    check = validate(design, gripper)
    print(f"{args.entrant} (design {design.design_id()}): {'valid' if check.ok else 'INVALID'}")
    for message in check.errors:
        print(f"  error: {message}")
    for message in check.warnings:
        print(f"  warning: {message}")
    if not check.ok and not args.allow_invalid:
        print("  not exported (pass --allow-invalid to export anyway)")
        return 1

    geom = generate_finger(design)
    mesh = to_trimesh(geom, mirror=args.mirror)
    extents_mm = " x ".join(f"{1000.0 * e:.1f}" for e in mesh.extents)
    printed_g = 1000.0 * abs(mesh.volume) * geom.density_kg_m3
    sim_g = 1000.0 * mass_and_inertia(geom).mass_kg
    print(
        f"  mass {printed_g:.1f} g printed (mesh volume), {sim_g:.1f} g in simulation (sum of boxes);"
        f" bounding box {extents_mm} mm (x y z); {'watertight' if mesh.is_watertight else 'NOT watertight'}"
    )
    if suffix == ".stl":
        path = export_stl(geom, args.out, mirror=args.mirror, units=args.units)
        side = "right (mirrored)" if args.mirror else "left"
        print(f"  wrote {_display(path)}: {side} finger in {args.units}")
    else:
        path = export_glb(geom, args.out, gripper)
        print(f"  wrote {_display(path)}: palm and both fingers, fully open, metres, Y-up")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from rtt.analysis.calibrate import G3Targets, run_calibration
    from rtt.sim.conditions import training_suite
    from rtt.sim.exec_cpu import CpuExecutor
    from rtt.sim.task import load_task

    out = args.out or RUNS_DIR / "calibrate" / _stamp()
    out.mkdir(parents=True, exist_ok=True)
    with CpuExecutor(args.workers) as pool:
        report = run_calibration(
            load_task(args.task),
            training_suite(args.suite),
            n_sobol=args.n_sobol,
            seed=args.seed,
            executor=pool,
            workers=args.workers,
            targets=G3Targets(),
            records_out=out / "episodes.parquet",
            task_file=args.task,
            suite_file=args.suite,
        )
    (out / "calibration.json").write_text(json.dumps(report.to_dict(), indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=1) if args.json else report.table())
    print(f"Wrote {_display(out / 'calibration.json')} and {_display(out / 'episodes.parquet')}")
    return 0 if report.passed else 1


def _cmd_search(args: argparse.Namespace) -> int:
    from rtt.pipeline.search import SearchRequest, parse_seeds, run_search_command

    outcome = run_search_command(
        SearchRequest(
            arm=args.arm,
            runs=args.runs,
            budget=args.budget,
            out=args.out,
            seeds=parse_seeds(args.seeds, args.runs),
            executor=args.executor,
            workers=args.workers,
            spend_cap_usd=args.spend_cap,
            cache_mode=args.cache_mode,
            cache_dir=args.cache,
        )
    )
    m = outcome.manifest
    print(
        f"search {args.arm}: {m['runs']} runs x {m['budget']} designs, {m['sim']['episodes_simulated']} episodes"
    )
    for run in m["per_run"]:
        best = run["best"]
        llm = run["llm"]
        extra = f", {llm['calls']} model calls ${llm['cost_usd']:.4f}" if m.get("model") else ""
        print(
            f"  {run['run_id']} (seed {run['run_seed']}): best {best['label']} {best['design_id']}"
            f" {100 * best['success']:.0f}%{extra}, {run['fallback_random']} random fallbacks"
        )
    print(f"Wrote {_display(outcome.paths['manifest'])}")
    return 0


def _cmd_replay_llm(args: argparse.Namespace) -> int:
    from rtt.pipeline.search import SearchRequest, run_search_command
    from rtt.store.manifest import read_manifest

    source = read_manifest(args.source / "manifest.json")
    if not source.get("model"):
        print(f"error: {args.source} is a {source.get('arm')} search; only LLM arms have a response cache")
        return 2
    outcome = run_search_command(
        SearchRequest(
            arm=source["arm"],
            runs=source["runs"],
            budget=source["budget"],
            out=args.out,
            seeds=source["seeds"],
            executor=args.executor,
            workers=args.workers,
            cache_mode="replay",
            cache_dir=args.source / "llm_cache",
        )
    )
    want = {r["run_id"]: r["fingerprint"] for r in source["per_run"]}
    got = {r["run_id"]: r["fingerprint"] for r in outcome.manifest["per_run"]}
    live = outcome.manifest["llm"]["live_calls"]
    same = want == got
    print(f"replayed {len(got)} runs from {_display(args.source / 'llm_cache')}: {live} live API calls")
    for run_id in sorted(want):
        print(f"  {run_id}: {'identical' if want[run_id] == got.get(run_id) else 'DIFFERENT'}")
    return 0 if same and live == 0 else 1


def _cmd_heldout(args: argparse.Namespace) -> int:
    from rtt.design.schema import load_entrants
    from rtt.pipeline.heldout import collect_frozen, run_heldout

    commitment = args.commitment
    if commitment is None and args.commitment_file and args.commitment_file.is_file():
        commitment = args.commitment_file.read_text(encoding="utf-8").split()[0]
    if commitment is None and not args.allow_uncommitted:
        print(
            f"error: no commitment ({_display(args.commitment_file)} missing); pass --commitment, "
            "or --allow-uncommitted for a development run"
        )
        return 2
    frozen = collect_frozen(args.frozen)
    entrants = [] if args.no_entrants else load_entrants(args.entrants)
    report = run_heldout(
        args.seed_file,
        args.out,
        frozen,
        commitment=commitment,
        entrants=entrants,
        n_conditions=args.n,
        seeds_per_condition=args.seeds_per_condition,
        workers=args.workers,
    )
    print(
        f"held-out suite {report['suite']['digest'][:12]} ({report['suite']['n_conditions']} conditions x "
        f"{report['suite']['seeds_per_condition']} seeds), commitment "
        f"{'verified' if report['commitment_verified'] else 'NOT checked'}"
    )
    for d in report["designs"]:
        print(
            f"  {d['key']:<20} {d['design_id']}  k={d['controller_setting']}  {100 * d['heldout_success']:.1f}%"
        )
    for arm, a in report["arms"].items():
        print(f"  arm {arm}: mean {100 * a['mean']:.1f}% over {len(a['runs'])} runs")
    print(f"Wrote {_display(args.out / 'heldout.json')}")
    return 0


def _parse_object(spec: str) -> object:
    """``box:80x50x110`` / ``cylinder:25x110`` / ``rod:18x180`` (millimetres) to an ObjectSpec."""
    from rtt.sim.objects import ObjectSpec

    kind, _, dims = spec.partition(":")
    try:
        size = tuple(float(v) / 1000.0 for v in dims.lower().split("x"))
        return ObjectSpec(kind, size)  # type: ignore[arg-type]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad object {spec!r} ({exc}); e.g. box:80x50x110") from exc


def _load_design(args: argparse.Namespace) -> object:
    from rtt.design.schema import FingerDesign, load_entrants

    if args.entrant:
        designs = {d.name: d for d in load_entrants(args.entrants)}
        if args.entrant not in designs:
            args.subparser.error(
                f"unknown entrant {args.entrant!r}; choose from {', '.join(map(str, designs))}"
            )
        return designs[args.entrant]
    data = json.loads(args.design.read_text(encoding="utf-8"))
    return FingerDesign.model_validate(data.get("design", data))


def _cmd_certify(args: argparse.Namespace) -> int:
    from rtt.sim.certify import certify
    from rtt.sim.conditions import training_suite
    from rtt.sim.exec_cpu import CpuExecutor

    design = _load_design(args)
    if args.object:
        objects = [(spec.label, spec) for spec in args.object]
    else:
        seen: dict[str, object] = {}
        for cond in training_suite():
            seen.setdefault(f"{cond.family}-{cond.size_label}", cond.obj)
        objects = list(seen.items())
    certificates = []
    with CpuExecutor(args.workers) as pool:
        for label, obj in objects:
            cert = certify(design, obj, controller_setting=args.k, executor=pool)  # type: ignore[arg-type]
            certificates.append(cert)
            tol = ", ".join(f"{a} {v if v is not None else '-'} mm" for a, v in cert.tolerance_mm.items())
            cats = ", ".join(f"{k} {100 * v:.0f}%" for k, v in cert.category_scores.items())
            print(
                f"{label:<18} k={cert.controller_setting}  {cert.passed}/{len(cert.tests)} passed  {cats}  tolerance {tol}"
            )
    doc = {
        "schema": "rtt.certificates/1",
        "design_id": design.design_id(),  # type: ignore[attr-defined]
        "design": design.model_dump(mode="json"),  # type: ignore[attr-defined]
        "issued_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "certificates": [c.to_dict() for c in certificates],
    }
    if args.out:
        out = args.out
        if out.suffix != ".json":
            out = out / "certificates" / f"{doc['design_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {_display(out)}")
    return 0


def _cmd_slice(args: argparse.Namespace) -> int:
    from rtt.pipeline.export_web import export_web
    from rtt.pipeline.slice import SliceConfig, run_slice

    out = args.out or RUNS_DIR / "slice" / f"slice-{_stamp()}"
    config = SliceConfig(
        seed=args.seed,
        spend_cap_usd=args.spend_cap,
        baseline=args.baseline,
        workers=args.workers,
        cache_mode="replay" if args.replay else "readwrite",
        cache_dir=args.cache,
    )
    if args.replay and args.cache is None:
        print("error: --replay needs --cache DIR (the llm_cache of the slice to replay)")
        return 2
    outcome = run_slice(out, config)
    s = outcome.summary
    llm = s["llm"]
    print(f"slice {s['slice_id']}: model {(s.get('model') or {}).get('model_id')} via {s.get('provider')}")
    print(
        f"  {llm['calls']} model calls ({llm['live_calls']} live), {llm['prompt_tokens']:,} prompt + "
        f"{llm['completion_tokens']:,} completion tokens, ${llm['cost_usd']:.4f}, "
        f"max latency {llm['latency_s_max']} s; {len(s['rejections'])} rejections, "
        f"{s['fallback_random']} random fallbacks"
    )
    print(f"  {'design':<14}{'id':<14}{'origin':<16}{'score':>7}{'k':>3}  per family")
    for d in s["designs"]:
        sc = d["score"]
        fam = " ".join(f"{f} {100 * v:.0f}%" for f, v in sc["per_family"].items())
        print(
            f"  {d['label']:<14}{d['design_id']:<14}{d['origin']:<16}{100 * sc['success']:>6.0f}%{sc['best_k']:>3}  {fam}"
        )
    print(f"  replays on {s['condition']['condition_id']} ({s['condition']['description']}):")
    for lane in s["lanes"]:
        verdict = "success" if lane["success"] else lane["failure_mode"]
        print(
            f"    {lane['stem']:<20} k={lane['controller_setting']}  {verdict}, lift {1000 * lane['max_lift_m']:.0f} mm"
        )
    print(f"  wall {s['wall_s']:.0f} s; wrote {_display(outcome.summary_path)}")
    if not args.no_export:
        result = export_web([out], args.web_data, featured=None)
        print(result.describe())
    return 0


def _cmd_export_web(args: argparse.Namespace) -> int:
    from rtt.pipeline.export_web import export_web

    result = export_web(args.runs, args.data_dir, featured=args.featured)
    print(result.describe())
    return 0 if not result.skipped else 1


def _cmd_job(args: argparse.Namespace) -> int:
    from rtt.cloud.__main__ import main as cloud_main

    kind = "probe" if args.kind == "smoke" else args.kind
    argv = ["run", kind, "--cap-usd", f"{args.cap_usd:g}", "--expected-run-s", f"{args.expected_run_s:g}"]
    if args.dry_run:
        argv.append("--dry-run")
    if args.platform:
        argv += ["--platform", args.platform]
    if args.run_id:
        argv += ["--run-id", args.run_id]
    if kind == "m1" and args.no_video:
        argv.append("--no-video")
    if kind == "search":
        if not (args.arm and args.runs and args.budget):
            print("error: rtt job search needs --arm, --runs and --budget")
            return 2
        argv += ["--arm", args.arm, "--runs", str(args.runs), "--budget", str(args.budget)]
        argv += ["--spend-cap", f"{args.spend_cap:g}"]
        if args.seeds:
            argv += ["--seeds", args.seeds]
    return cloud_main(argv)


def _cmd_cloud(args: argparse.Namespace) -> int:
    from rtt.cloud.__main__ import main as cloud_main

    return cloud_main(args.rest)


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """The ``rtt`` argument parser; each subcommand sets ``handler`` (and ``subparser``)."""
    parser = argparse.ArgumentParser(prog="rtt", description="Robot Tool Tournament")
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="command")

    version = sub.add_parser("version", help="print the package version")
    version.set_defaults(handler=_cmd_version)

    m1 = sub.add_parser(
        "m1",
        help="run gate G1 (manual entrants + deliberate failures); write replays and MP4s",
        description=(
            "Gate G1: every manual entrant must lift and hold the G1 box, and three deliberate failures"
            " (closing too high, mu = 0.05, an object wider than the opening) must fail as expected."
        ),
    )
    m1.add_argument(
        "--out", type=Path, default=DEFAULT_M1_OUT, help="output directory (default: data/demo/m1)"
    )
    m1.add_argument("--seed", type=int, default=0, help="episode seed for the object jitter (default 0)")
    m1.add_argument("--no-video", action="store_true", help="write replays only; skip MP4 rendering")
    m1.add_argument("--no-files", action="store_true", help="write nothing; run and print only")
    m1.add_argument("--json", action="store_true", help="print the full report as JSON")
    m1.add_argument("--entrants", type=Path, default=DEFAULT_ENTRANTS, help="entrants YAML")
    m1.add_argument("--task", type=Path, default=None, help="task YAML (default: configs/task_v1.yaml)")
    m1.set_defaults(handler=_cmd_m1)

    pl = sub.add_parser(
        "probe-llm",
        help="summarise docs/PROBE_LLM.md; --live re-runs the Token Factory probe",
        description="Without --live or --replay, print a summary of the probe report; no API calls are made.",
    )
    mode = pl.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help="run the probe against Token Factory (needs the API key; cache first, spend-capped)",
    )
    mode.add_argument("--replay", action="store_true", help="rebuild the report from the cache only")
    probe.add_probe_arguments(pl)
    pl.set_defaults(handler=_cmd_probe_llm)

    eg = sub.add_parser(
        "export-geom",
        help="export a manual entrant's finger as STL (mm) or a gripper preview as GLB",
        description="Validate a manual entrant against the simulator's gripper, then export its geometry.",
    )
    eg.add_argument("entrant", help="entrant name in the entrants YAML, e.g. straight")
    eg.add_argument("--out", type=Path, required=True, help="output path ending in .stl or .glb")
    eg.add_argument("--mirror", action="store_true", help="STL of the right (mirrored) finger")
    eg.add_argument("--units", choices=("mm", "m"), default="mm", help="STL units (default mm)")
    eg.add_argument("--allow-invalid", action="store_true", help="export even if validation fails")
    eg.add_argument("--entrants", type=Path, default=DEFAULT_ENTRANTS, help="entrants YAML")
    eg.add_argument("--task", type=Path, default=None, help="task YAML (default: configs/task_v1.yaml)")
    eg.set_defaults(handler=_cmd_export_geom, subparser=eg)

    cal = sub.add_parser(
        "calibrate",
        help="gate G3: entrants + Sobol-64 on the training suite against the calibration targets",
        description="Score the manual entrants and n Sobol designs on the training suite (gate G3).",
    )
    cal.add_argument("--task", type=Path, default=DEFAULT_TASK, help="task YAML")
    cal.add_argument("--suite", type=Path, default=DEFAULT_SUITE, help="training suite YAML")
    cal.add_argument("--n-sobol", type=int, default=64, help="random designs (default 64)")
    cal.add_argument("--seed", type=int, default=20261018, help="calibration sample seed (public)")
    cal.add_argument("--workers", type=int, default=None, help="CPU workers (default: all)")
    cal.add_argument("--out", type=Path, default=None, help="output directory (default runs/calibrate/<ts>)")
    cal.add_argument("--json", action="store_true", help="print the full report as JSON")
    cal.set_defaults(handler=_cmd_calibrate)

    se = sub.add_parser(
        "search",
        help="R independent search runs of one arm, in lockstep, on the training suite",
        description="Run R independent runs of one arm; each simulates exactly B valid designs.",
    )
    se.add_argument("--arm", required=True, choices=ARM_CHOICES)
    se.add_argument("--runs", type=int, required=True, help="independent runs R")
    se.add_argument("--budget", type=int, default=40, help="valid designs simulated per run (default 40)")
    se.add_argument("--out", type=Path, required=True, help="output directory (must not hold a search)")
    se.add_argument("--seeds", default=None, help="seeds YAML or comma-separated run seeds (default 1..R)")
    se.add_argument("--executor", choices=("cpu", "mjwarp"), default="cpu", help="simulator executor")
    se.add_argument("--workers", type=int, default=None, help="CPU workers (default: all)")
    se.add_argument("--spend-cap", type=float, default=2.0, help="live model spend cap in USD (LLM arms)")
    se.add_argument(
        "--cache-mode",
        choices=("readwrite", "replay"),
        default="readwrite",
        help="replay: answer every model call from the cache (no key, no spend)",
    )
    se.add_argument("--cache", type=Path, default=None, help="response cache (default OUT/llm_cache)")
    se.set_defaults(handler=_cmd_search)

    rl = sub.add_parser(
        "replay-llm",
        help="re-run an LLM search from its response cache (0 API calls) and compare",
    )
    rl.add_argument("--from", dest="source", type=Path, required=True, help="search output directory")
    rl.add_argument("--out", type=Path, required=True, help="new output directory")
    rl.add_argument("--executor", choices=("cpu", "mjwarp"), default="cpu")
    rl.add_argument("--workers", type=int, default=None)
    rl.set_defaults(handler=_cmd_replay_llm)

    ho = sub.add_parser(
        "heldout",
        help="score frozen designs on the commit-reveal held-out suite",
        description="The only command that reads the held-out seed. Never run it inside a search job.",
    )
    ho.add_argument(
        "--seed-file", type=Path, required=True, help="held-out seed file (git-ignored or outside)"
    )
    ho.add_argument("--frozen", type=Path, nargs="*", default=[], help="search dirs, frozen/ dirs or files")
    ho.add_argument("--out", type=Path, required=True, help="output directory")
    ho.add_argument("--commitment", default=None, help="committed SHA-256 of the seed")
    ho.add_argument(
        "--commitment-file", type=Path, default=DEFAULT_COMMITMENT, help="file holding the commitment"
    )
    ho.add_argument("--allow-uncommitted", action="store_true", help="development only: skip the check")
    ho.add_argument("--no-entrants", action="store_true", help="score frozen designs only")
    ho.add_argument("--entrants", type=Path, default=DEFAULT_ENTRANTS)
    ho.add_argument("--n", type=int, default=None, help="conditions (default 100)")
    ho.add_argument("--seeds-per-condition", type=int, default=None, help="jitter seeds (default 2)")
    ho.add_argument("--workers", type=int, default=None)
    ho.set_defaults(handler=_cmd_heldout)

    ce = sub.add_parser(
        "certify",
        help="robustness certificate: placement error, shake and push on training objects",
    )
    which = ce.add_mutually_exclusive_group(required=True)
    which.add_argument("--entrant", help="manual entrant name")
    which.add_argument("--design", type=Path, help="design JSON (a design, or a frozen record)")
    ce.add_argument(
        "--object",
        type=_parse_object,
        action="append",
        help="object in mm, e.g. box:80x50x110, cylinder:25x110, rod:18x180 (repeatable;"
        " default: every training object)",
    )
    ce.add_argument("--k", type=int, default=None, help="controller setting (default: best nominal)")
    ce.add_argument("--entrants", type=Path, default=DEFAULT_ENTRANTS)
    ce.add_argument("--workers", type=int, default=None)
    ce.add_argument(
        "--out", type=Path, default=None, help="JSON file, or a run dir (writes certificates/<id>.json)"
    )
    ce.set_defaults(handler=_cmd_certify, subparser=ce)

    sl = sub.add_parser(
        "slice",
        help="vertical slice: Nemotron proposes 4 designs, simulate, score, replays, export to the web",
    )
    sl.add_argument("--out", type=Path, default=None, help="output directory (default runs/slice/slice-<ts>)")
    sl.add_argument("--seed", type=int, default=17, help="run seed (round-0 Sobol fill)")
    sl.add_argument(
        "--spend-cap", type=float, default=0.20, help="live model spend cap in USD (default 0.20)"
    )
    sl.add_argument("--baseline", default="stock", help="entrant replayed beside the new designs")
    sl.add_argument("--workers", type=int, default=None)
    sl.add_argument("--replay", action="store_true", help="answer every model call from --cache (no key)")
    sl.add_argument("--cache", type=Path, default=None, help="response cache (default OUT/llm_cache)")
    sl.add_argument("--no-export", action="store_true", help="do not write web/public/data")
    sl.add_argument("--web-data", type=Path, default=REPO_ROOT / "web" / "public" / "data")
    sl.set_defaults(handler=_cmd_slice)

    ew = sub.add_parser("export-web", help="publish run directories into web/public/data")
    ew.add_argument("runs", type=Path, nargs="+", help="slice, search or held-out run directories")
    ew.add_argument("--data-dir", type=Path, default=REPO_ROOT / "web" / "public" / "data")
    ew.add_argument("--featured", default=None, help="bout id to feature on the landing page")
    ew.set_defaults(handler=_cmd_export_web)

    jb = sub.add_parser(
        "job",
        help="run smoke / m1 / search as a Nebius Serverless AI Job (waits, records spend, fetches)",
        description="Settings come from ~/.config/rtt/nebius.env and image.env; no IDs are stored in the repo.",
    )
    jb.add_argument("kind", choices=("smoke", "m1", "search"))
    jb.add_argument("--arm", choices=ARM_CHOICES, help="search: arm")
    jb.add_argument("--runs", type=int, help="search: independent runs")
    jb.add_argument("--budget", type=int, help="search: designs per run")
    jb.add_argument("--seeds", default=None, help="search: comma-separated run seeds")
    jb.add_argument("--spend-cap", type=float, default=2.0, help="search: live model spend cap in USD")
    jb.add_argument("--no-video", action="store_true", help="m1: skip MP4 rendering")
    jb.add_argument("--platform", default=None, help="force one platform (no fallback)")
    jb.add_argument("--run-id", default=None)
    jb.add_argument("--cap-usd", type=float, default=3.0, help="Nebius spend cap over the ledger")
    jb.add_argument("--expected-run-s", type=float, default=600.0, help="run-time estimate for the cap check")
    jb.add_argument("--dry-run", action="store_true", help="validate the job server-side; create nothing")
    jb.set_defaults(handler=_cmd_job)

    cl = sub.add_parser("cloud", help="Nebius job tooling: capacity, status, logs, list, fetch, spend, ...")
    cl.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for python -m rtt.cloud")
    cl.set_defaults(handler=_cmd_cloud)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
