"""Command line for Bring Your Part (``python -m rtt.part``).

* ``intake PART.stl`` prints what the simulator will see (poses, grasps, warnings).
* ``run PART.stl --out DIR`` runs intake, the per-part search, the certificate and
  the print pack, and prints the stage timings.
* ``certify DIR`` re-runs the robustness certificate of a run's best design on its part
  (``--design FILE --out FILE`` certifies another design on the same part instead).
* ``pack DIR`` (alias ``printpack``) rebuilds the print pack of an existing run directory.

Exit status 2 means the part was refused; the message says why.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from rtt.part.intake import MATERIALS, PartError


def _spend_cap(text: str) -> float:
    """``--spend-cap``: a finite number of dollars within the part run's ceiling."""
    from rtt.part.search import check_spend_cap

    try:
        return check_spend_cap(float(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser(prog: str = "python -m rtt.part") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def part_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("stl", help="part mesh (STL, OBJ, PLY or OFF)")
        p.add_argument("--units", default="auto", choices=["auto", "mm", "cm", "m", "in"])
        p.add_argument("--material", choices=sorted(MATERIALS))
        p.add_argument("--density", type=float, help="kg/m^3 (instead of --material)")
        p.add_argument("--name")
        p.add_argument("--top-k", type=int, default=3, help="most likely resting poses to test")
        p.add_argument(
            "--decomposition",
            default="auto",
            choices=["auto", "coacd", "hull"],
            help="convex decomposition (hull: one piece per body, fast but loses concavities)",
        )

    p_in = sub.add_parser("intake", help="validate a part and show its poses")
    part_args(p_in)
    p_in.add_argument("--json", action="store_true", help="print the PartSpec as JSON")

    p_run = sub.add_parser("run", help="intake, search, certify and pack")
    part_args(p_run)
    p_run.add_argument("--out", required=True, help="run directory")
    p_run.add_argument("--arm", default="nemotron", choices=["nemotron", "random", "zeroshot"])
    p_run.add_argument("--budget", type=int, default=24, help="designs simulated per run (8 + 8n)")
    p_run.add_argument("--runs", type=int, default=1)
    p_run.add_argument("--workers", type=int)
    p_run.add_argument(
        "--spend-cap", type=_spend_cap, default=1.0, help="live model spend cap (USD, at most 5)"
    )
    p_run.add_argument("--cache-mode", default="readwrite", choices=["readwrite", "replay", "off"])

    p_cert = sub.add_parser("certify", help="re-run the robustness certificate on a run's part")
    p_cert.add_argument("run_dir", help="a `run` output directory (holds part/ and search/)")
    p_cert.add_argument(
        "--design", help="design JSON to certify instead of the run's best (needs --out)", default=None
    )
    p_cert.add_argument("--out", default=None, help="certificate JSON (default RUN_DIR/certificate.json)")
    p_cert.add_argument("--k", type=int, default=None, help="controller setting (default: the frozen one)")
    p_cert.add_argument(
        "--poses", choices=["tested", "primary"], default="tested", help="all tested poses or the likeliest"
    )
    p_cert.add_argument("--workers", type=int)

    p_pack = sub.add_parser("pack", aliases=["printpack"], help="(re)build the print pack of a run directory")
    p_pack.add_argument("run_dir")
    p_pack.add_argument("--rack", action="store_true", help="add the (assumed) drive rack to the jaw")
    return ap


def _certify(args: argparse.Namespace) -> int:
    """``certify``: the run's best design (or ``--design``) on the run's saved part."""
    from pathlib import Path

    from rtt.design.schema import FingerDesign
    from rtt.part.intake import load_part
    from rtt.part.search import _per_condition, certify_part, frozen_from_run_dir
    from rtt.store.manifest import write_json_atomic

    run_dir = Path(args.run_dir)
    part = load_part(run_dir / "part")
    if args.design is None:
        best = frozen_from_run_dir(run_dir)
        design, k = best.design, best.controller_setting
        per_condition = _per_condition(run_dir / "search", best)
        out = Path(args.out) if args.out else run_dir / "certificate.json"
    else:
        if not args.out:
            print("error: --design needs --out (the run's certificate.json belongs to its best design)")
            return 2
        data = json.loads(Path(args.design).read_text(encoding="utf-8"))
        design = FingerDesign.model_validate(data.get("design", data))
        k, per_condition, out = None, None, Path(args.out)
    cert = certify_part(
        design,
        part,
        controller_setting=args.k if args.k is not None else k,
        per_condition=per_condition,
        poses=args.poses,
        workers=args.workers,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, cert.to_json())
    print(
        f"design {cert.design_id} on {part.name}: certificate {100 * cert.score:.0f}% "
        f"({cert.passed}/{cert.n_tests} tests held), k={cert.controller_setting}"
    )
    for pose in cert.poses:
        c = pose.certificate
        print(
            f"  pose{pose.pose_rank} (p={pose.probability:.2f}, grasp {pose.candidate}): {c.passed}/{len(c.tests)} held"
        )
    print(f"wrote {out}")
    return 0


def main(argv: Sequence[str] | None = None, *, prog: str = "python -m rtt.part") -> int:
    args = _parser(prog).parse_args(argv)
    try:
        if args.cmd == "intake":
            from rtt.part.intake import intake

            part = intake(
                args.stl,
                units=args.units,
                material=args.material,
                density=args.density,
                name=args.name,
                top_k=args.top_k,
                decomposition=args.decomposition,
            )
            print(json.dumps(part.to_json(), indent=2) if args.json else part.summary())
        elif args.cmd == "run":
            from rtt.part.search import run_part

            result = run_part(
                args.stl,
                args.out,
                arm=args.arm,
                budget=args.budget,
                runs=args.runs,
                units=args.units,
                material=args.material,
                density=args.density,
                name=args.name,
                top_k=args.top_k,
                workers=args.workers,
                spend_cap_usd=args.spend_cap,
                cache_mode=args.cache_mode,
                intake_kwargs={"decomposition": args.decomposition},
            )
            best = result.search.best
            print(result.part.summary())
            print(
                f"best design {best.design_id} ({best.origin}): training {100 * best.training_success:.0f}%, "
                f"certificate {100 * result.certificate.score:.0f}% "
                f"({result.certificate.passed}/{result.certificate.n_tests} tests held)"
            )
            print("timings: " + ", ".join(f"{k} {v:.1f}" for k, v in result.timings.items()))
            print(f"print pack: {result.pack}")
        elif args.cmd == "certify":
            return _certify(args)
        else:
            from rtt.part.mount import ClampInterface
            from rtt.part.pack import make_print_pack

            print(make_print_pack(args.run_dir, clamp=ClampInterface(rack=args.rack)))
    except PartError as exc:
        print(f"part refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
