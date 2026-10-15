"""``rtt heldout``: score frozen designs on the commit-reveal held-out suite.

This is the only code path that reads the held-out seed. It runs after every search
job has frozen its top-1 design (:mod:`rtt.search.select`):

1. The seed is read from an explicit file through :class:`~rtt.store.vault.HeldoutVault`
   (no default location; a file inside the repository must be git-ignored) and, when a
   commitment is given, checked against it. A search process cannot get here: the vault
   refuses to read while a :class:`~rtt.store.vault.VaultGuard` is active.
2. The suite is :func:`rtt.sim.conditions.heldout_suite` of that seed (100 conditions x
   2 jitter seeds by default), checked against the same commitment.
3. Every frozen design runs at its frozen controller setting on every held-out
   condition and seed; manual entrants (optional) run at the setting that scored best
   for them on the training suite, measured here with the same rule as every arm.
4. ``heldout.json`` (schema ``rtt.heldout/1``) records per-design and per-arm results,
   the suite digest, the commitment and provenance; ``episodes.parquet`` holds every
   episode. The seed itself is never written.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rtt.design.schema import FingerDesign
from rtt.search.select import FrozenDesign, read_frozen

__all__ = ["HELDOUT_SCHEMA", "HeldoutEntry", "collect_frozen", "run_heldout"]

HELDOUT_SCHEMA: Final = "rtt.heldout/1"
REPORT: Final = "heldout.json"


@dataclass(frozen=True)
class HeldoutEntry:
    """One design to score: a frozen pick or a manual entrant, at one controller setting."""

    key: str
    arm: str
    run_index: int | None
    design: FingerDesign
    controller_setting: int
    frozen_sha256: str | None
    training_success: float | None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def collect_frozen(paths: Iterable[str | Path]) -> list[FrozenDesign]:
    """Frozen records from files, ``frozen/`` directories or search output directories.

    Each record's SHA-256 is verified on load; duplicates (same arm and run) are an error.
    """
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            files.append(p)
        elif (p / "frozen").is_dir():
            files += sorted((p / "frozen").glob("*.json"))
        elif p.is_dir():
            files += sorted(p.glob("*.json"))
        else:
            raise FileNotFoundError(f"{p}: no frozen record or directory")
    frozen = [read_frozen(f) for f in files]
    keys = [(f.arm, f.run_index) for f in frozen]
    if len(set(keys)) != len(keys):
        raise ValueError("two frozen records share an arm and run index")
    return frozen


def _entrant_settings(
    entrants: Sequence[FingerDesign], task: Any, executor: Any, workers: int | None
) -> dict[str, tuple[int, float]]:
    """Each entrant's best training controller setting and its training score."""
    from rtt.search.sim_adapter import SimEvaluator

    sim = SimEvaluator(task=task, executor=executor, workers=workers)
    return {
        d.design_id(): (s.best_k, s.success) for d, s in zip(entrants, sim.evaluate(entrants), strict=True)
    }


def run_heldout(
    seed_file: str | Path,
    out_dir: str | Path,
    frozen: Sequence[FrozenDesign],
    *,
    commitment: str | None = None,
    entrants: Sequence[FingerDesign] = (),
    n_conditions: int | None = None,
    seeds_per_condition: int | None = None,
    executor: Any = "cpu",
    workers: int | None = None,
    task: Any | None = None,
) -> dict[str, Any]:
    """Score ``frozen`` (plus ``entrants``) on the held-out suite; write and return the report.

    Raises:
        ValueError: the seed does not match ``commitment``.
        VaultAccessError: called inside a search's vault guard.
    """
    from rtt.sim.conditions import HELDOUT_CONDITIONS, HELDOUT_SEEDS_PER_CONDITION, heldout_suite
    from rtt.sim.evaluate import evaluate_designs, write_parquet
    from rtt.sim.exec_cpu import CpuExecutor
    from rtt.sim.task import default_task, task_digest
    from rtt.store.manifest import JobContext, code_facts, write_json_atomic
    from rtt.store.vault import HeldoutVault

    out = Path(out_dir)
    if (out / REPORT).exists():
        raise FileExistsError(f"{out} already holds a held-out report; use a fresh directory")
    if not frozen and not entrants:
        raise ValueError("nothing to evaluate: pass frozen designs and/or entrants")
    task = task or default_task()
    vault = HeldoutVault(seed_file)
    seed = vault.verify(commitment) if commitment else vault.read_seed()
    suite = heldout_suite(
        seed,
        n=n_conditions or HELDOUT_CONDITIONS,
        seeds_per_condition=seeds_per_condition or HELDOUT_SEEDS_PER_CONDITION,
        commitment=commitment,
        task=task,
    )
    del seed  # nothing below needs it; it is never written
    started_at, t0 = _utc_now(), time.perf_counter()
    own = executor == "cpu"
    pool = CpuExecutor(workers) if own else executor
    try:
        settings = _entrant_settings(entrants, task, pool, workers) if entrants else {}
        items = [
            HeldoutEntry(
                key=f"{f.arm}-r{f.run_index:02d}",
                arm=f.arm,
                run_index=f.run_index,
                design=f.design,
                controller_setting=f.controller_setting,
                frozen_sha256=f.content_sha256,
                training_success=f.training_success,
            )
            for f in frozen
        ]
        for d in entrants:
            k, train = settings[d.design_id()]
            items.append(HeldoutEntry(f"entrant-{d.name}", "entrant", None, d, k, None, train))
        by_k: dict[int, list[FingerDesign]] = defaultdict(list)
        for item in items:
            if item.design.design_id() not in {d.design_id() for d in by_k[item.controller_setting]}:
                by_k[item.controller_setting].append(item.design)
        records = []
        for k, designs in sorted(by_k.items()):
            records += evaluate_designs(designs, suite, controller_settings=(k,), executor=pool, task=task)
    finally:
        if own:
            pool.close()
    wall = time.perf_counter() - t0
    write_parquet(records, out / "episodes.parquet")

    by_design: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for r in records:
        by_design[(r.design_id, r.k)].append(r)
    designs_out = []
    for item in items:
        recs = by_design[(item.design.design_id(), item.controller_setting)]
        fam: dict[str, list[bool]] = defaultdict(list)
        modes: dict[str, int] = defaultdict(int)
        for r in recs:
            fam[r.family].append(r.success)
            if not r.success:
                modes[r.failure_mode or "unknown"] += 1
        designs_out.append(
            {
                "key": item.key,
                "arm": item.arm,
                "run_index": item.run_index,
                "design_id": item.design.design_id(),
                "name": item.design.name,
                "controller_setting": item.controller_setting,
                "frozen_sha256": item.frozen_sha256,
                "training_success": item.training_success,
                "heldout_success": sum(r.success for r in recs) / len(recs),
                "episodes": len(recs),
                "per_family": {f: sum(v) / len(v) for f, v in sorted(fam.items())},
                "failure_modes": dict(sorted(modes.items())),
            }
        )
    arms: dict[str, list[float]] = defaultdict(list)
    for d in designs_out:
        if d["arm"] != "entrant":
            arms[d["arm"]].append(d["heldout_success"])
    job = JobContext.from_env()
    report = {
        "schema": HELDOUT_SCHEMA,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_s": round(wall, 3),
        "suite": {
            "name": suite.name,
            "digest": suite.digest(),
            "n_conditions": len(suite),
            "seeds_per_condition": len(suite.conditions[0].seeds),
            "families": sorted({c.family for c in suite}),
        },
        "commitment": commitment,
        "commitment_verified": commitment is not None,
        "task": {"name": task.name, "digest": task_digest(task)},
        "executor": "cpu" if own else getattr(executor, "__name__", type(executor).__name__),
        "episodes": len(records),
        "episodes_per_design": len(suite) * len(suite.conditions[0].seeds),
        "designs": designs_out,
        "arms": {
            arm: {
                "runs": values,
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else None,
            }
            for arm, values in sorted(arms.items())
        },
        "job": job.to_json(),
        "code": code_facts(),
    }
    write_json_atomic(out / REPORT, report)
    return report
