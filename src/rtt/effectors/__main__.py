"""Command line of the effector layer (product mode, not pre-registered).

::

    python -m rtt.effectors families
    python -m rtt.effectors tournament --part block --requirement registered --workers 8
    python -m rtt.effectors tournament --part all --workers 8 --compare
    python -m rtt.effectors export-web --runs runs/effectors --web web/public/data

``tournament`` writes ``runs/effectors/<part>/`` (``records.parquet``, ``certificates/``,
``tournament.json``, ``recommendation.json``); ``--compare`` re-runs a part that already has
outputs and fails unless the records, ``traj_sha`` values and recommendation are identical.
``--design`` asks Nemotron for the designed group's proposals: it spends real money on Token
Factory and needs ``--spend-usd`` (the cap for the whole run) and track B3's designer.
``export-web`` records the lanes and writes the site's effector data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rtt.effectors.types import PRODUCT_MODE_LABEL, EntrantGroup, Requirement

DEFAULT_OUT = Path("runs/effectors")


def _parts(name: str) -> list[Any]:
    from rtt.parts.library import get_part, training_parts

    if name == "all":
        return list(training_parts())
    return [get_part(n) for n in name.split(",")]


def _designer(args: argparse.Namespace, out: Path) -> Any:
    if not args.design:
        return None
    if args.spend_usd is None or args.spend_usd <= 0:
        raise SystemExit("--design spends money on Token Factory: pass --spend-usd (for example 0.05)")
    from rtt.effectors.tournament import default_designer

    try:
        from rtt.effectors.morphology import make_designer_client
    except ModuleNotFoundError:
        raise SystemExit(
            "--design needs the morphology designer (rtt.effectors.morphology, track B3)"
        ) from None
    # track B3's client: Nemotron 3 Super at the designer's sampling, the scrubbing cache (no
    # reasoning text, brand-free model text) and a ledger capped at --spend-usd for this run
    client = make_designer_client(
        out / "llm_cache", ledger_path=out / "llm_ledger.jsonl", budget_usd=args.spend_usd
    )
    designer = default_designer(client)
    if designer is None:
        raise SystemExit("--design needs the morphology designer (rtt.effectors.morphology, track B3)")
    return designer


def _tournament(args: argparse.Namespace) -> int:
    from rtt.effectors.tournament import TournamentExecutor, run_tournament, write_outputs

    out = Path(args.out)
    groups = [EntrantGroup(g) for g in args.groups.split(",")] if args.groups else None
    designer = _designer(args, out)
    failures = 0
    with TournamentExecutor(args.workers) as ex:
        for part in _parts(args.part):
            t0 = time.perf_counter()
            previous = None
            target = out / part.name
            if args.compare and (target / "tournament.json").is_file():
                previous = json.loads((target / "tournament.json").read_text(encoding="utf-8")).get("digest")

            def progress(stage: str, done: int, total: int, *, _p: str = part.name, _t: float = t0) -> None:
                print(f"  {_p}: {stage} ({total} jobs) at {time.perf_counter() - _t:.0f} s", flush=True)

            result = run_tournament(
                part,
                requirement=args.requirement,
                groups=groups,
                families=args.families.split(",") if args.families else None,
                seed=args.seed,
                executor=ex,
                design=args.design,
                designer=designer,
                progress=progress,
            )
            paths = write_outputs(result, target)
            rec = result.recommendation
            fair = result.fairness()
            digest = result.digest()
            win = rec["winner"]
            name = win["effector"]["display_name"] if win else "no winner"
            print(
                f"{part.name}: {args.requirement} -> {name}; fair={fair['identical']}; "
                f"{result.physics()['episodes'] + result.physics()['certificate_episodes']} episodes in "
                f"{result.wall_s:.0f} s; digest {digest[:12]}; {paths['tournament'].parent}",
                flush=True,
            )
            if not fair["identical"]:
                failures += 1
                print("  fairness problems: " + "; ".join(fair["problems"]), file=sys.stderr)
            if previous is not None:
                same = previous == digest
                print(f"  re-run identical to the previous outputs: {same}", flush=True)
                failures += 0 if same else 1
    print(PRODUCT_MODE_LABEL)
    return 1 if failures else 0


def _export(args: argparse.Namespace) -> int:
    from rtt.effectors.lanes import export_web

    index = export_web(Path(args.runs), Path(args.web), parts=args.parts.split(",") if args.parts else None)
    for entry in index["parts"]:
        w = entry["winner"]
        print(f"{entry['part_id']}: {entry['lanes']} lanes; {w['display_name'] if w else 'no winner'}")
    print(PRODUCT_MODE_LABEL)
    return 0


def _families(_: argparse.Namespace) -> int:
    from rtt.effectors.registry import get_family
    from rtt.effectors.tournament import family_status

    fams, status = family_status(None)
    for fam, state in status.items():
        print(f"{fam}: {state}")
    for fam in fams:
        for spec in get_family(fam).presets():
            print(f"  {spec.group.value:<9} {spec.short_id} {spec.display_name}")
    print(PRODUCT_MODE_LABEL)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rtt.effectors", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tournament", help="run the per-part tournament and write runs/effectors/<part>/")
    t.add_argument("--part", required=True, help="a train_v2 part name, a comma list, or 'all'")
    t.add_argument(
        "--requirement", default=Requirement.REGISTERED.value, choices=[r.value for r in Requirement]
    )
    t.add_argument("--groups", default=None, help="comma list of groups (default: all)")
    t.add_argument("--families", default=None, help="comma list of families (default: every available one)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--workers", type=int, default=1)
    t.add_argument("--out", default=str(DEFAULT_OUT))
    t.add_argument(
        "--compare", action="store_true", help="fail unless a re-run reproduces the existing outputs"
    )
    t.add_argument(
        "--design", action="store_true", help="Nemotron proposals for the designed group (spends money)"
    )
    t.add_argument("--spend-usd", type=float, default=None)
    t.set_defaults(func=_tournament)
    e = sub.add_parser("export-web", help="record the lanes and write web/public/data/effectors/")
    e.add_argument("--runs", default=str(DEFAULT_OUT))
    e.add_argument("--web", default="web/public/data")
    e.add_argument("--parts", default=None)
    e.set_defaults(func=_export)
    f = sub.add_parser("families", help="list the families and presets this build can run")
    f.set_defaults(func=_families)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
