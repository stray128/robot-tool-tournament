"""Command-line entry point: ``rtt <command>``.

* ``rtt m1``: gate G1. Runs every manual entrant plus three deliberate failures on
  the G1 box, prints a results table and writes a replay and an MP4 per case to
  ``data/demo/m1/``. Exit status 0 when the gate passes.
* ``rtt probe-llm``: prints a summary of ``docs/PROBE_LLM.md`` without calling any
  API. ``--live`` re-runs the Token Factory probe (cache first, under a hard spend
  cap); ``--replay`` rebuilds the report from the response cache only.
* ``rtt export-geom <entrant> --out finger.stl``: writes a manual entrant's finger
  as an STL in millimetres, or a palm-and-fingers preview for a ``.glb`` path.
* ``rtt version``: prints the package version.

Heavy dependencies (MuJoCo, trimesh) are imported only by the command that needs them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rtt.llm import probe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENTRANTS = REPO_ROOT / "configs" / "entrants.yaml"
DEFAULT_M1_OUT = REPO_ROOT / "data" / "demo" / "m1"


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
