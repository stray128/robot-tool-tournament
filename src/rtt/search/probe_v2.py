"""Probe Nemotron 3 Super on the v2 ``submit_designs`` tool before GV5 (spec section 12).

Five forced tool calls, each the real arm-C v2 main prompt for round 1 of an independent run
(its own run seed and round 0: the four entrants plus four screened Sobol recipes). The
round-0 scores come from the synthetic :class:`~rtt.search.fakes_v2.FakeEvaluatorV2`, so the
feedback in the prompt is realistic in form but not in value; the probe measures the
**model's output**, not the search:

* ``valid_primary``: of the 4 slots, how many hold a recipe that passes the validators;
* ``valid_after_reserves``: how many of the 4 slots can be filled by a valid design once the
  call's 4 reserves replace invalid slots in order (the arm's own rule);
* ``accepted_after_reserves``: the same, also passing the duplicate and diversity screen;
* ``needs_repair``: the call leaves an invalid slot that no reserve fills (the arm would
  then make a repair call).

Pass (spec section 12): at least 90 % of slots valid after reserves and at most 1 call that
needs a repair. Fallback if it fails: 3 slots + 3 reserves.

Every response is cached and every call ledgered under ``--out``; ``--replay`` rebuilds the
report from the cache with no API key and no spend. ``--cap-usd`` is a hard cap on all live
calls in this probe's ledger (default $0.30).

``--real`` (gate GV5) makes each call the exact arm-C round-1 prompt of a run: the six
training parts and ``task_v2`` in the prompt, the entrants of ``configs/entrants_v2.yaml`` and
the four screened Sobol recipes in round 0, round 0 scored by the CPU simulator on
``train_v2`` (``--workers``), and every slot checked by the full section-13 validators.

The slots are checked with the schema rules (bounds, strategy and fit rules, quantisation).
The report keeps every slot as the model sent it, so ``--recheck-geometry`` can re-validate
them later with track G's section-13 validators on track O's training parts under track S's
task_v2, with no model call (``report_geometry.json``).

Usage::

    python -m rtt.search.probe_v2 --live --real --workers 14 --cap-usd 0.10   # GV5: the real round-1 prompt
    python -m rtt.search.probe_v2 --live                # 5 live calls (cache first), cap $0.30
    python -m rtt.search.probe_v2 --replay              # the same report from the cache, $0
    python -m rtt.search.probe_v2 --recheck-geometry    # the slots against the geometry validators
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rtt.llm.ledger import read_ledger
from rtt.search.base import BudgetLedger, RunContext, derive_seed
from rtt.search.diversity import screen
from rtt.search.fakes_v2 import FakeEvaluatorV2, stub_entrants
from rtt.search.nemotron import PROPOSER_MODEL, ProposerConfig, make_proposer_client, reserve_names
from rtt.search.nemotron_v2 import NemotronProposerV2, json_safe
from rtt.search.random_search import round0_designs
from rtt.search.space import V2, get_space
from rtt.store.manifest import write_json_atomic

__all__ = ["CallResult", "geometry_context", "main", "recheck_geometry", "run_probe"]

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "runs" / "probe_v2"
DEFAULT_SEEDS = (9101, 9102, 9103, 9104, 9105)
PASS_VALID = 0.90
MAX_REPAIRS = 1


@dataclass
class CallResult:
    """What one probe call returned, slot by slot."""

    seed: int
    ok: bool
    error: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False
    valid_primary: int = 0
    valid_reserves: int = 0
    valid_after_reserves: int = 0
    accepted_after_reserves: int = 0
    needs_repair: bool = True
    errors: list[str] = field(default_factory=list)
    screen_reasons: dict[str, int] = field(default_factory=dict)
    inactive_given: int = 0
    cells: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    """Every slot as the model sent it (JSON-safe), for :func:`recheck_geometry`."""

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def geometry_context() -> Any:
    """Track O's training parts and track S's task_v2 for the geometry validators."""
    from rtt.parts.library import training_parts
    from rtt.search.space_v2 import V2Context
    from rtt.sim.task import default_task_v2

    return V2Context(parts=tuple(training_parts()), task=default_task_v2(), label="train_v2")


class _RealRound0:
    """Round 0 of each probe seed on the real training suite, scored once for all seeds."""

    def __init__(self, seeds: tuple[int, ...], config: ProposerConfig, workers: int | None) -> None:
        from rtt.search.runner import DEFAULT_ENTRANTS_V2, _load_entrants_v2
        from rtt.search.sim_adapter import SimEvaluatorV2
        from rtt.sim.exec_cpu import CpuExecutor

        space = get_space(V2)
        pool = CpuExecutor(workers)
        try:
            ev = SimEvaluatorV2(executor=pool, workers=workers)
            self.info = ev.describe()
            entrants = _load_entrants_v2(DEFAULT_ENTRANTS_V2)
            self.round0 = {
                seed: round0_designs(
                    entrants, seed, 8, self.info.gripper_task, space=space, min_distance=config.min_distance
                )
                for seed in seeds
            }
            unique: dict[str, Any] = {}
            for props in self.round0.values():
                for prop in props:
                    unique.setdefault(space.design_id(prop.design), prop.design)
            scored = ev.evaluate(list(unique.values()))
            self.scores = dict(zip(unique, scored, strict=True))
        finally:
            pool.close()

    def evaluate(self, designs: list[Any]) -> list[Any]:
        space = get_space(V2)
        return [self.scores[space.design_id(d)] for d in designs]


def _probe_call(
    seed: int, client: Any, config: ProposerConfig, real: _RealRound0 | None = None
) -> CallResult:
    space = get_space(V2)
    ev: Any = real if real is not None else FakeEvaluatorV2(0)
    info = real.info if real is not None else ev.describe()
    ctx = RunContext("nemotron", 0, seed, info.gripper_task, info, BudgetLedger(40, 8, 8, 16), space=space,
                     shared_min_distance=config.min_distance)  # fmt: skip
    r0 = (
        real.round0[seed]
        if real is not None
        else round0_designs(
            stub_entrants(), seed, 8, info.gripper_task, space=space, min_distance=config.min_distance
        )
    )
    ctx.budget.charge(0, 8)
    items = [ctx.history.add(0, p, s) for p, s in zip(r0, ev.evaluate([p.design for p in r0]), strict=True)]
    proposer = NemotronProposerV2(ctx, client, config)
    proposer.tell(items)
    messages = proposer.build_messages(1, 0)
    outcome = proposer._call("main", 1, 0, messages, keep=0)
    assert outcome is not None
    record = outcome.record
    result = CallResult(
        seed=seed,
        ok=outcome.args is not None,
        error=outcome.error,
        finish_reason=record.finish_reason,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        reasoning_tokens=record.reasoning_tokens,
        cost_usd=record.cost_usd,
        latency_s=record.latency_s,
        cached=record.cached,
    )
    if outcome.args is None:
        result.errors.append(outcome.error or "no usable call")
        return result
    vctx = ctx.gripper_task
    names = [spec.name for spec in proposer.slots] + reserve_names(proposer.n_reserves)
    result.raw = {name: json_safe(outcome.args.get(name)) for name in names}
    primaries, reserves = [], []
    for spec in proposer.slots:
        primaries.append(proposer.kit.parse_slot(outcome.args.get(spec.name), spec, vctx))
    for name in reserve_names(proposer.n_reserves):
        from rtt.search.nemotron import SlotSpec

        reserves.append(proposer.kit.parse_slot(outcome.args.get(name), SlotSpec(name, "reserve"), vctx))
    for design, errors, meta in primaries + reserves:
        result.errors.extend(errors[:2])
        result.inactive_given += sum(1 for n in meta.get("notes", ()) if "is inactive" in n)
    result.valid_primary = sum(1 for d, _, _ in primaries if d is not None)
    result.valid_reserves = sum(1 for d, _, _ in reserves if d is not None)
    pool = [d for d, _, _ in reserves if d is not None]
    filled: list[Any] = []
    for design, _, _ in primaries:
        if design is not None:
            filled.append(design)
        elif pool:
            filled.append(pool.pop(0))
    result.valid_after_reserves = len(filled)
    result.needs_repair = result.valid_after_reserves < len(primaries)
    # The screen against the run and the call's own accepted designs (reserves fill screened holes too).
    refs = [item.design for item in ctx.history]
    accepted: list[Any] = []
    candidates = [d for d, _, _ in primaries]
    spare = [d for d, _, _ in reserves if d is not None]
    reasons: Counter[str] = Counter()
    for design in candidates:
        options = ([design] if design is not None else []) + spare
        while options:
            pick = options.pop(0)
            found = screen(
                pick, refs + accepted, None, config.min_distance, space=space, context=ctx.gripper_task
            )
            if pick is not design and pick in spare:
                spare.remove(pick)
            if found is None:
                accepted.append(pick)
                break
            reasons[found.reason] += 1
    result.accepted_after_reserves = len(accepted)
    result.screen_reasons = dict(sorted(reasons.items()))
    result.cells = [" + ".join(space.cell(d) or ("?", "?")) for d, _, _ in primaries if d is not None]
    return result


def run_probe(
    out: Path = DEFAULT_OUT,
    *,
    mode: str = "readwrite",
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    cap_usd: float = 0.30,
    transport: Any = None,
    limiter: Any = None,
    real: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run (or replay) the probe and write ``<out>/report.json``; returns the report.

    ``real``: the training parts, the entrants and simulator-scored round 0 (see the module
    docstring); the slots then pass the full section-13 validators.
    """
    out.mkdir(parents=True, exist_ok=True)
    client = make_proposer_client(
        out / "cache",
        cache_mode=mode,
        ledger_path=out / "ledger.jsonl",
        budget_usd=cap_usd,
        transport=transport,
        limiter=limiter,
    )  # type: ignore[arg-type]
    config = ProposerConfig.for_space(V2)
    run_seeds = tuple(derive_seed(seed, "probe-v2") % (2**31) for seed in seeds)
    round0 = _RealRound0(run_seeds, config, workers) if real else None
    calls = [_probe_call(seed, client, config, round0) for seed in run_seeds]
    slots = 4 * len(calls)
    valid = sum(c.valid_after_reserves for c in calls)
    repairs = sum(1 for c in calls if c.needs_repair)
    ledger = read_ledger(out / "ledger.jsonl")
    report = {
        "model": PROPOSER_MODEL,
        "design_space": V2,
        "calls": [c.to_json() for c in calls],
        "slots": slots,
        "valid_primary_rate": sum(c.valid_primary for c in calls) / slots,
        "valid_after_reserves_rate": valid / slots,
        "accepted_after_reserves_rate": sum(c.accepted_after_reserves for c in calls) / slots,
        "calls_needing_repair": repairs,
        "pass": valid / slots >= PASS_VALID and repairs <= MAX_REPAIRS,
        "rule": f">= {PASS_VALID:.0%} of slots valid after reserves and <= {MAX_REPAIRS} call needing a repair",
        "cost_usd_this_report": round(sum(c.cost_usd for c in calls), 6),
        "live_cost_usd_ledger_total": round(sum(float(r.cost_usd or 0.0) for r in ledger if not r.cached), 6),
        "live_calls_ledger_total": sum(1 for r in ledger if not r.cached),
        "cap_usd": cap_usd,
        "validators": "schema + geometry (section 13 on the training parts, task_v2)" if real else "schema",
        "context": "real" if real else "fake",
        "note": (
            "Each call is arm C's round-1 prompt: the training parts, the v2 entrants and round 0 scored by the"
            " CPU simulator on train_v2."
            if real
            else "Round-0 scores in the prompts come from the synthetic FakeEvaluatorV2; the probe measures"
            " output validity."
        ),
        "min_distance": config.min_distance,
        "explore_min_distance": config.explore_min_distance,
    }
    write_json_atomic(out / "report.json", report)
    return report


def _check_slot(space: Any, slot: Any, context: Any) -> Any:
    return space.validate(space.slot_mapping(slot) if isinstance(slot, dict) else {}, context)


def recheck_geometry(report: dict[str, Any], context: Any) -> dict[str, Any]:
    """Re-validate every slot of a probe report under ``context`` (no model call).

    Same accounting as the probe (4 slots, reserves fill invalid slots in order); the duplicate
    and diversity screen is not repeated (it does not depend on the parts).
    """
    space = get_space(V2)
    primary = [spec.name for spec in NemotronProposerV2.slots]
    rows, stages = [], Counter()
    for call in report["calls"]:
        raw = call.get("raw") or {}
        if not raw:
            rows.append(
                {"seed": call["seed"], "valid_primary": 0, "valid_after_reserves": 0, "needs_repair": True}
            )
            continue

        firsts = [_check_slot(space, raw.get(n), context) for n in primary]
        spares = [_check_slot(space, raw[n], context) for n in raw if n not in primary]
        pool = sum(1 for c in spares if c.ok)
        filled = sum(1 for c in firsts if c.ok) + min(pool, sum(1 for c in firsts if not c.ok))
        for c in firsts + spares:
            if not c.ok:
                stages[c.stage or "unknown"] += 1
        rows.append(
            {
                "seed": call["seed"],
                "valid_primary": sum(1 for c in firsts if c.ok),
                "valid_reserves": pool,
                "valid_after_reserves": filled,
                "needs_repair": filled < len(primary),
                "errors": [e for c in firsts + spares for e in c.errors[:1]][:6],
            }
        )
    slots = len(primary) * len(rows)
    valid = sum(r["valid_after_reserves"] for r in rows)
    repairs = sum(1 for r in rows if r["needs_repair"])
    return {
        "validators": "schema + geometry (section 13 on the training parts, task_v2)",
        "parts": list(getattr(context, "part_ids", ())),
        "calls": rows,
        "slots": slots,
        "valid_primary_rate": sum(r["valid_primary"] for r in rows) / slots,
        "valid_after_reserves_rate": valid / slots,
        "calls_needing_repair": repairs,
        "rejections_by_stage": dict(sorted(stages.items())),
        "pass": valid / slots >= PASS_VALID and repairs <= MAX_REPAIRS,
        "rule": report.get("rule"),
        "source_report_cost_usd": report.get("cost_usd_this_report"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live", action="store_true", help="cache first; only cache misses are billed")
    group.add_argument("--replay", action="store_true", help="from the cache only: no key, no spend")
    group.add_argument(
        "--recheck-geometry",
        action="store_true",
        help="re-validate report.json's slots with the geometry validators (no model call)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cap-usd", type=float, default=0.30)
    parser.add_argument(
        "--real", action="store_true", help="the real parts, entrants and simulator-scored round 0 (GV5)"
    )
    parser.add_argument("--workers", type=int, default=None, help="--real: CPU workers for round 0")
    args = parser.parse_args(argv)
    if args.recheck_geometry:
        source = json.loads((args.out / "report.json").read_text(encoding="utf-8"))
        report = recheck_geometry(source, geometry_context())
        write_json_atomic(args.out / "report_geometry.json", report)
        keys = (
            "validators",
            "valid_primary_rate",
            "valid_after_reserves_rate",
            "calls_needing_repair",
            "pass",
        )
        json.dump({k: report[k] for k in (*keys, "rejections_by_stage")}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0 if report["pass"] else 1
    report = run_probe(
        args.out,
        mode="replay" if args.replay else "readwrite",
        cap_usd=args.cap_usd,
        real=args.real,
        workers=args.workers,
    )
    keys = (
        "validators",
        "valid_primary_rate",
        "valid_after_reserves_rate",
        "accepted_after_reserves_rate",
        "calls_needing_repair",
        "pass",
        "cost_usd_this_report",
        "live_cost_usd_ledger_total",
    )
    summary = {k: report[k] for k in keys}
    json.dump(summary, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
