"""Run gate GV3 (calibration of ``task_v2`` / ``train_v2``) on the laptop.

The screening sample is what spec section 14 names: the four round-0 entrants and the two
reference entrants of ``configs/entrants_v2.yaml``, plus 64 Sobol designs drawn by arm A's own
sampler (:class:`rtt.search.random_search.SpaceSampler` over the ``freeform-v2`` space, every
draw canonicalised and checked by track G's validators on the six training parts, distinct
geometry ids) from the screening seed. Every design runs the 48 episodes of ``train_v2``
(:func:`rtt.sim.evaluate.evaluate_designs_v2`); the scores and the gate are
:mod:`rtt.analysis.calibrate_v2`'s.

Without ``--log`` this is a rehearsal: the report is written and nothing is logged or frozen.
With ``--log FILE`` the iteration is recorded in the calibration log (the procedure rules of
:class:`~rtt.analysis.calibrate_v2.CalibrationLog` apply: iteration 1 is the task as drafted,
later iterations must name their changes) and, when the decision is a freeze, the frozen
file hashes are printed for ``docs/TASK_V2.md``.

    python -m rtt.analysis.gv3_run --seed 20261020 --out runs/gv3 [--workers 14]
    python -m rtt.analysis.gv3_run --seed 20261020 --out runs/gv3 --log runs/gv3/calibration_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rtt.analysis import calibrate_v2 as cal

REPO_ROOT = Path(__file__).resolve().parents[3]
N_SOBOL = 64


def screening_designs(seed: int, parts: Sequence[Any], task: Any) -> tuple[list[Any], dict[str, Any]]:
    """The 64 Sobol designs of the screening sample (valid on every training part, distinct
    geometry), and the sampler's valid-rate log."""
    from rtt.search.space import get_space
    from rtt.search.space_v2 import V2Context

    space = get_space("freeform-v2")
    ctx = V2Context(parts=tuple(parts), task=task)
    sampler = space.sample_sobol(seed, ctx)
    seen: set[str] = set()

    def distinct(design: Any) -> str | None:
        gid = space.geometry_id(design, ctx)
        if gid in seen:
            return "duplicate_geometry"
        seen.add(gid)
        return None

    designs = sampler.sample(N_SOBOL, accept=distinct)
    log = {
        "draws": sampler.draws,
        "accepted": sampler.accepted,
        "rejected": sampler.rejected,
        "valid_rate": sampler.accepted / sampler.draws if sampler.draws else 0.0,
        "reasons": dict(sampler.reasons),
    }
    return designs, log


def run(
    seed: int,
    out: Path,
    *,
    workers: int | None = None,
    native_ccd: bool | None = None,
    iteration: int = 1,
) -> tuple[cal.CalibrationV2Report, list[dict[str, Any]], dict[str, Any]]:
    from rtt.design.freeform import load_entrants_v2, reference_entrants, round0_entrants
    from rtt.parts.suite import train_suite
    from rtt.sim.evaluate import evaluate_designs_v2
    from rtt.sim.task import default_task_v2, task_digest

    cal.check_screening_seed(seed)
    task = default_task_v2()
    if native_ccd is not None:
        task = task.with_overrides(physics={"native_ccd": bool(native_ccd)})
    suite = train_suite()
    parts = tuple(suite.parts())
    entrants = load_entrants_v2()
    t0 = time.perf_counter()
    sobol, sampler_log = screening_designs(seed, parts, task)
    t_sample = time.perf_counter() - t0
    designs: list[tuple[str, cal.Origin, Any]] = [(e.name, "entrant", e) for e in round0_entrants(entrants)]
    designs += [(e.name, "reference", e) for e in reference_entrants(entrants)]
    designs += [(f"sobol_{i:02d}", "sobol", d) for i, d in enumerate(sobol)]
    t0 = time.perf_counter()
    records = evaluate_designs_v2([d for _, _, d in designs], suite, workers=workers, task=task)
    t_sim = time.perf_counter() - t0
    rows = [r.to_dict() for r in records]
    per = suite.n_episodes
    if len(rows) != per * len(designs):
        raise RuntimeError(f"expected {per * len(designs)} episode records, got {len(rows)}")
    names: dict[str, tuple[str, cal.Origin]] = {}
    masses: dict[str, float] = {}
    for k, (name, origin, _) in enumerate(designs):
        chunk = rows[k * per : (k + 1) * per]
        ids = {r["design_id"] for r in chunk}
        if len(ids) != 1:
            raise RuntimeError(f"{name}: its records carry {len(ids)} design ids")
        did = ids.pop()
        if did in names:
            raise RuntimeError(f"{name} and {names[did][0]} share design id {did}")
        names[did] = (name, origin)
        built = [float(r["finger_mass_g"]) for r in chunk if r.get("finger_mass_g") is not None]
        if built:
            masses[did] = max(built)
    scored = cal.score_designs_v2(
        rows,
        names,
        episodes_per_design=per,
        masses_g=masses,
        seat_short_max_mm=float(task.registration.seat_short_max_mm),
    )
    report = cal.CalibrationV2Report(
        rows=scored,
        screening_seed=seed,
        iteration=iteration,
        task_digest=task_digest(task),
        suite_digest=suite.digest(),
        file_sha256=cal.freeze_hashes(REPO_ROOT),
        notes={
            "sampler": sampler_log,
            "native_ccd": bool(task.physics.native_ccd),
            "episodes": len(rows),
            "sample_s": round(t_sample, 1),
            "simulate_s": round(t_sim, 1),
        },
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "gv3_report.json").write_text(json.dumps(report.to_dict(), indent=1) + "\n", encoding="utf-8")
    with (out / "gv3_records.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=float) + "\n")
    designs_doc = []
    for name, origin, d in designs:
        dump = (
            d.model_dump(mode="json") if hasattr(d, "model_dump") else {"entrant": getattr(d, "name", str(d))}
        )
        designs_doc.append({"name": name, "origin": origin, "design": dump})
    (out / "gv3_designs.json").write_text(json.dumps(designs_doc, indent=1) + "\n", encoding="utf-8")
    return report, rows, sampler_log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=cal.SCREENING_SEED_V2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--native-ccd", choices=("true", "false"), default=None)
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--log", type=Path, default=None, help="record the iteration in this calibration log")
    ap.add_argument("--change", action="append", default=[], help="knob=old:new (iterations after the first)")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    native = None if args.native_ccd is None else args.native_ccd == "true"
    report, _, _ = run(args.seed, args.out, workers=args.workers, native_ccd=native, iteration=args.iteration)
    print(report.table())
    if args.log is not None:
        changes = {}
        for item in args.change:
            knob, _, values = item.partition("=")
            old, _, new = values.partition(":")
            changes[knob] = (old, new)
        log = cal.CalibrationLog.open(args.log)
        decision, failing = log.record(report, changes=changes, note=args.note)
        print(f"logged iteration {report.iteration}: {decision} (failing: {failing or 'none'})")
        if decision in ("freeze", "declare_miss_and_freeze"):
            print(json.dumps(report.file_sha256, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
