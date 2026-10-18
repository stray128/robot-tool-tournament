"""Command line of the Task Engineer (product mode).

Usage::

    python -m rtt.studio.engineer plan --text "Open the tote lid, ..." [--stl PART.stl]
    python -m rtt.studio.engineer plan --case h1_tote            # a gate case, replayed
    python -m rtt.studio.engineer plan --text "..." --live --env-file PATH/.env --budget 0.05
    python -m rtt.studio.engineer gate --out DIR [--live --env-file PATH/.env --budget 0.50]
    python -m rtt.studio.engineer export-web [--recorded DIR] [--out web/components/studio/intake]

Without ``--live`` every call is replayed from a cache (by default the recorded gate run), with
zero API calls and no key; a description that is not in the cache stops with a message.
``--live`` calls Token Factory under a hard cap on everything recorded in the run's ledger
(``--budget``), reading ``NEBIUS_API_KEY`` from ``--env-file`` (never printed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rtt.effectors.types import PRODUCT_MODE_LABEL
from rtt.llm.cache import CacheMiss
from rtt.llm.client import BudgetExceededError, LLMAuthError, LLMBillingError
from rtt.llm.ledger import Ledger, read_ledger
from rtt.search.nemotron import PromptLeakError

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUNS = REPO_ROOT / "runs" / "studio"
WEB_INTAKE = REPO_ROOT / "web" / "components" / "studio" / "intake"


def _spent(ledger_path: Path) -> float:
    return sum(e.cost_usd for e in read_ledger(ledger_path) if not e.cached)


def _client(args: argparse.Namespace, cache_dir: Path, ledger_path: Path | None) -> Any:
    from rtt.studio.engineer.engineer import make_engineer_client

    if not args.live:
        return make_engineer_client(cache_dir, cache_mode="replay")
    if args.env_file is None:
        raise SystemExit("--live needs --env-file (the .env that holds NEBIUS_API_KEY)")
    assert ledger_path is not None
    remaining = max(0.0, args.budget - _spent(ledger_path))
    ledger = Ledger(ledger_path, budget_usd=remaining)
    return make_engineer_client(cache_dir, cache_mode="readwrite", ledger=ledger, env_file=args.env_file)


def _print_plan(result: Any) -> None:
    spec = result.spec
    print(f"status: {result.status} ({'first try' if result.first_try else f'{len(result.attempts)} calls'})")
    if spec is not None:
        print(f"title: {spec.title}")
        if spec.refusal:
            print(f"refusal: {spec.refusal}")
        for o in spec.objects:
            print(f"  object {o.name}: {o.kind.value} ({o.label})")
        for i, s in enumerate(spec.steps, start=1):
            tgt = f" -> {s.target}" if s.target else ""
            print(f"  {i}. {s.primitive.value} {s.object}.{s.handle}{tgt}: {s.description}")
        for u in spec.unsupported:
            near = u.nearest_primitive.value if u.nearest_primitive else "none"
            print(f"  not simulated: {u.what} ({u.reason}); nearest primitive: {near}")
    for p in result.problems:
        print(f"  problem: {p}")
    for w in result.warnings:
        print(f"  warning: {w}")
    for n in result.notes:
        print(f"  note: {n}")
    print(f"cost: ${result.cost_usd:.4f} live; {PRODUCT_MODE_LABEL}")


def cmd_plan(args: argparse.Namespace) -> int:
    from rtt.studio.engineer.engineer import plan_task
    from rtt.studio.engineer.examples import GATE_CASES
    from rtt.studio.engineer.gate import RECORDED_GATE
    from rtt.studio.engineer.uploads import UploadRegistry, install_upload_resolver

    if (args.text is None) == (args.case is None):
        raise SystemExit("give exactly one of --text or --case")
    text = args.text
    if args.case is not None:
        case = next((c for c in GATE_CASES if c.case_id == args.case), None)
        if case is None:
            raise SystemExit(f"no gate case {args.case!r}; cases: {', '.join(c.case_id for c in GATE_CASES)}")
        text = case.text
    uploads = []
    if args.stl is not None:
        registry = UploadRegistry(args.uploads)
        install_upload_resolver(registry)
        info = registry.add_stl(args.stl, units=args.units)
        print(f"uploaded: {info.ref} {info.name} {info.size_mm} mm, {info.mass_g:g} g")
        uploads.append(info)
    cache_dir = args.cache or (DEFAULT_RUNS / "engineer" / "cache" if args.live else RECORDED_GATE / "cache")
    ledger_path = args.ledger or DEFAULT_RUNS / "engineer" / "ledger.jsonl"
    client = _client(args, cache_dir, ledger_path if args.live else None)
    try:
        result = plan_task(text, client=client, uploads=uploads, compact=args.compact_tool)
    except CacheMiss:
        print(
            "this description is not in the replay cache; run with --live (spends) or pick a --case",
            file=sys.stderr,
        )
        return 2
    except (LLMAuthError, LLMBillingError) as exc:
        print(f"Token Factory refused the call: {type(exc).__name__}", file=sys.stderr)
        return 3
    except PromptLeakError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    _print_plan(result)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if result.ok else 1


def cmd_gate(args: argparse.Namespace) -> int:
    from rtt.studio.engineer.examples import GATE_CASES
    from rtt.studio.engineer.gate import RECORDED_GATE, gate_manifest, run_gate

    out: Path = args.out
    cases = GATE_CASES
    if args.cases:
        wanted = args.cases.split(",")
        cases = tuple(c for c in GATE_CASES if c.case_id in wanted)
    cache_dir = out / "cache" if args.live else (args.cache or RECORDED_GATE / "cache")
    out.mkdir(parents=True, exist_ok=True)
    client = _client(args, cache_dir, out / "ledger.jsonl" if args.live else None)
    try:
        results = run_gate(client, cases, compact=args.compact_tool)
    except CacheMiss as exc:
        print(f"replay cache miss ({exc}); run with --live", file=sys.stderr)
        return 2
    except BudgetExceededError as exc:
        print(f"spend cap: {exc}", file=sys.stderr)
        return 3
    manifest = gate_manifest(
        results,
        mode="live" if args.live else "replay",
        budget_usd=args.budget if args.live else None,
        compact=args.compact_tool,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("| case | expect | status | calls | steps |")
    print("|---|---|---|---|---|")
    for row in manifest["cases"]:
        steps = (
            ", ".join(f"{s['primitive']}:{s['object']}" for s in row["steps"]) or (row["refusal"] or "")[:60]
        )
        verdict = row["status"] + (" (repaired)" if row["repaired"] else "")
        print(f"| {row['case_id']} | {row['expect']} | {verdict} | {row['calls']} | {steps} |")
    s = manifest["summary"]
    print(
        f"plans valid {s['plans']['valid']}/{s['plans']['n']} (first try {s['plans']['valid_first_try']}), "
        f"refusals {s['refusals']['refused_with_nearest']}/{s['refusals']['n']}; "
        f"gate {'PASSED' if manifest['gate']['passed'] else 'FAILED'}; "
        f"recorded cost ${manifest['llm']['cost_usd']:.4f}"
    )
    return 0 if manifest["gate"]["passed"] else 1


def cmd_export_web(args: argparse.Namespace) -> int:
    from rtt.studio.engineer.webdata import write_web_data

    for path in write_web_data(args.recorded, args.out):
        print(f"wrote {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rtt.studio.engineer", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def live_flags(p: argparse.ArgumentParser, budget: float) -> None:
        p.add_argument("--live", action="store_true", help="call Token Factory (spend-capped)")
        p.add_argument("--env-file", type=Path, default=None, help=".env holding NEBIUS_API_KEY (live only)")
        p.add_argument("--budget", type=float, default=budget, help="hard cap in USD over the run's ledger")
        p.add_argument("--cache", type=Path, default=None, help="response cache directory")
        p.add_argument("--compact-tool", action="store_true", help="send the schema without its descriptions")

    p = sub.add_parser("plan", help="plan one use case")
    p.add_argument("--text", default=None)
    p.add_argument("--case", default=None, help="a gate case id instead of --text")
    p.add_argument("--stl", type=Path, default=None, help="an STL of the use case's part")
    p.add_argument("--units", default="mm", choices=("auto", "mm", "cm", "m", "in"))
    p.add_argument("--uploads", type=Path, default=DEFAULT_RUNS / "uploads", help="upload registry folder")
    p.add_argument("--ledger", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="write the plan JSON here")
    live_flags(p, 0.05)
    p.set_defaults(func=cmd_plan)

    g = sub.add_parser("gate", help="run the gate descriptions")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--cases", default=None, help="comma-separated case ids (default: all)")
    live_flags(g, 0.50)
    g.set_defaults(func=cmd_gate)

    w = sub.add_parser("export-web", help="write the intake page's catalog, recorded plans and parity cases")
    w.add_argument(
        "--recorded", type=Path, default=None, help="recorded gate run (default: the packaged one)"
    )
    w.add_argument("--out", type=Path, default=WEB_INTAKE)
    w.set_defaults(func=cmd_export_web)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
