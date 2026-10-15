"""Simulator benchmarks: timestep parity, CPU throughput and the entrants on a suite.

Run as ``python -m rtt.sim.bench <command>`` (the ``__main__`` guard matters: the CPU
executor spawns worker processes that re-import this module):

* ``parity``: a fixed 200-episode random parity set (random valid designs, random
  conditions with pose errors, random controller settings) simulated at two
  timesteps; reports how often success flags and failure modes agree, and the
  episodes/s at each timestep. Also runs gate G1 at each timestep over several seeds.
* ``throughput``: episodes/s of :func:`rtt.sim.evaluate.evaluate_designs` on the
  training suite for a list of worker counts.
* ``entrants``: the manual entrants' scores on the training suite.

Every command prints JSON (``--out`` also writes it to a file). Results feed
``docs/BENCH.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rtt.design.schema import CATEGORICAL_OPTIONS, PARAM_BOUNDS, FingerDesign, GripperTask, load_entrants
from rtt.design.validate import validate
from rtt.sim.conditions import Suite, heldout_suite, training_suite
from rtt.sim.evaluate import EpisodeRecord, evaluate_designs, rank_designs, resolve_design
from rtt.sim.exec_cpu import CpuExecutor, EpisodeJob
from rtt.sim.m1 import DEFAULT_ENTRANTS, run_gate
from rtt.sim.task import TaskConfig, default_task

# Public (non-secret) seed text for the parity set; it is not the held-out secret.
PARITY_SEED_TEXT = "rtt-timestep-parity-set-v1-public-not-a-secret"
PARITY_DESIGN_SEED = 20261015
PARITY_SUITE_NAME = "parity_v1"


def random_valid_designs(n: int, seed: int, gripper: GripperTask) -> list[FingerDesign]:
    """``n`` designs sampled uniformly over the schema, rejection-sampled until valid."""
    rng = np.random.default_rng(seed)
    designs: list[FingerDesign] = []
    while len(designs) < n:
        params: dict[str, object] = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PARAM_BOUNDS.items()}
        params.update({k: str(rng.choice(opts)) for k, opts in CATEGORICAL_OPTIONS.items()})
        if params["inner_face"] == "flat":
            params["inner_face_depth_mm"] = 0.0
        design = FingerDesign.model_validate(params)
        if validate(design, gripper).ok:
            designs.append(design)
    return designs


def parity_jobs(task: TaskConfig, n: int = 200) -> list[tuple[str, str, EpisodeJob]]:
    """The fixed random parity set: ``(design_id, condition_id, job)`` per episode.

    Conditions come from the held-out *generator* run on a public seed text and are named
    ``parity_v1/...`` so they are never mistaken for the secret held-out suite.
    """
    conditions = heldout_suite(
        PARITY_SEED_TEXT, n=(n + 1) // 2, seeds_per_condition=2, task=task, name=PARITY_SUITE_NAME
    )
    gripper = conditions.gripper_task(task)
    designs = list(load_entrants(DEFAULT_ENTRANTS)) + random_valid_designs(46, PARITY_DESIGN_SEED, gripper)
    resolved = [resolve_design(d) for d in designs]
    rng = np.random.default_rng(PARITY_DESIGN_SEED)
    jobs = []
    for cond in conditions:
        for seed in cond.seeds:
            did, geom = resolved[int(rng.integers(len(resolved)))]
            k = int(rng.integers(len(task.controller.grasp_height_fracs)))
            jobs.append(
                (did, cond.condition_id, EpisodeJob(geom, cond.obj, cond.placement, k, seed, task=task))
            )
    return jobs[:n]


def timestep_parity(
    timesteps: Sequence[float] = (0.001, 0.002),
    n: int = 200,
    workers: int | None = None,
    gate_seeds: Sequence[int] = range(8),
    task: TaskConfig | None = None,
) -> dict[str, Any]:
    """Compare outcomes of the parity set and gate G1 across ``timesteps`` (first = reference)."""
    base = task or default_task()
    runs: dict[float, dict[str, Any]] = {}
    with CpuExecutor(workers) as ex:
        for dt in timesteps:
            t = base.with_overrides(physics={"timestep": dt})
            entries = parity_jobs(t, n)
            batch = ex.run([j for _, _, j in entries])
            gate = [run_gate(None, seed=s, video=False, task=t) for s in gate_seeds]
            runs[dt] = {
                "results": batch.results,
                "keys": [(d, c) for d, c, _ in entries],
                "episodes_per_sec": batch.episodes_per_sec,
                "steps_per_sec_per_worker": batch.steps_per_sec_per_worker,
                "wall_time_s": batch.wall_time_s,
                "workers": batch.workers,
                "gate_passed": [g.passed for g in gate],
                "gate_cases": sum(len(g.outcomes) for g in gate),
                "gate_cases_ok": sum(o.passed for g in gate for o in g.outcomes),
            }
    ref_dt = timesteps[0]
    ref = runs[ref_dt]["results"]
    out: dict[str, Any] = {"n": len(ref), "reference_timestep": ref_dt, "timesteps": {}}
    for dt in timesteps:
        res = runs[dt]["results"]
        same_success = sum(a.success == b.success for a, b in zip(ref, res, strict=True))
        same_mode = sum(a.failure_mode == b.failure_mode for a, b in zip(ref, res, strict=True))
        mismatches = [
            {
                "design_id": d,
                "condition_id": c,
                "k": a.controller_setting,
                "seed": a.seed,
                f"ref_{ref_dt}": a.failure_mode or "success",
                str(dt): b.failure_mode or "success",
            }
            for (d, c), a, b in zip(runs[dt]["keys"], ref, res, strict=True)
            if a.failure_mode != b.failure_mode
        ]
        flags = Counter(f.split("@", 1)[0] for r in res for f in r.flags)
        out["timesteps"][str(dt)] = {
            "success_rate": sum(r.success for r in res) / len(res),
            "success_agreement": same_success / len(res),
            "failure_mode_agreement": same_mode / len(res),
            "failure_modes": dict(Counter(r.failure_mode or "success" for r in res)),
            "flags": dict(flags),
            "penetration_max_mm_p99": float(np.percentile([r.metrics["penetration_max"] for r in res], 99))
            * 1000.0,
            "episodes_per_sec": runs[dt]["episodes_per_sec"],
            "steps_per_sec_per_worker": runs[dt]["steps_per_sec_per_worker"],
            "workers": runs[dt]["workers"],
            "gate_runs_passed": f"{sum(runs[dt]['gate_passed'])}/{len(runs[dt]['gate_passed'])}",
            "gate_cases_ok": f"{runs[dt]['gate_cases_ok']}/{runs[dt]['gate_cases']}",
            "mismatches": mismatches,
        }
    return out


def throughput(
    worker_counts: Sequence[int],
    n_designs: int = 8,
    suite: Suite | None = None,
    task: TaskConfig | None = None,
) -> dict[str, Any]:
    """Episodes/s for ``n_designs`` random valid designs x the suite x k = 3, per worker count.

    Pool start-up is excluded (a warm :class:`CpuExecutor` is reused, as a search loop
    would); a cold :func:`~rtt.sim.exec_cpu.run_jobs` call adds about 1 s.
    """
    task = task or default_task()
    suite = suite or training_suite()
    designs = random_valid_designs(n_designs, PARITY_DESIGN_SEED + 1, suite.gripper_task(task))
    out: dict[str, Any] = {"designs": n_designs, "suite": suite.name, "rows": []}
    for w in worker_counts:
        with CpuExecutor(w) as ex:
            evaluate_designs(designs[:1], suite, controller_settings=[1], executor=ex, task=task)  # warm up
            t0 = time.perf_counter()
            records = evaluate_designs(designs, suite, executor=ex, task=task)
            wall = time.perf_counter() - t0
        steps = sum(r.n_steps for r in records)
        out["rows"].append(
            {
                "workers": w,
                "episodes": len(records),
                "wall_s": round(wall, 2),
                "episodes_per_sec": round(len(records) / wall, 2),
                "physics_steps_per_sec": round(steps / wall),
                "mean_sim_s_per_episode": round(steps * task.physics.timestep / len(records), 2),
            }
        )
    return out


def entrant_scores(
    suite: Suite | None = None, workers: int | None = None, task: TaskConfig | None = None
) -> dict[str, Any]:
    """The manual entrants on the suite (default: training), best first."""
    suite = suite or training_suite()
    entrants = load_entrants(DEFAULT_ENTRANTS)
    names = {d.design_id(): d.name for d in entrants}
    records: list[EpisodeRecord] = evaluate_designs(entrants, suite, workers=workers, task=task)
    rows = []
    for s in rank_designs(records):
        rows.append(
            {
                "entrant": names[s.design_id],
                "design_id": s.design_id,
                "score": round(s.score, 3),
                "best_k": s.best_k,
                "success_by_k": {k: round(v, 3) for k, v in s.success_by_k.items()},
                "success_by_family": {k: round(v, 3) for k, v in s.success_by_family.items()},
                "failure_modes": s.failure_modes,
                "flags": s.flags,
            }
        )
    failures = [
        {"entrant": names[r.design_id], "condition": r.condition_id, "k": r.k, "mode": r.failure_mode}
        for r in records
        if not r.success
    ]
    return {"suite": suite.name, "suite_digest": suite.digest()[:12], "entrants": rows, "failures": failures}


def _machine() -> dict[str, Any]:
    try:
        load = os.getloadavg()
    except OSError:
        load = (float("nan"),) * 3
    return {
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "loadavg_at_start": [round(v, 1) for v in load],
    }


def main(argv: Sequence[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workers", type=int, default=None, help="worker processes (default: all CPUs)")
    common.add_argument("--timestep", type=float, default=None, help="override the task timestep")
    common.add_argument("--out", type=Path, default=None, help="also write the JSON here")
    parser = argparse.ArgumentParser(prog="python -m rtt.sim.bench", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    par = sub.add_parser("parity", parents=[common], help="timestep parity on the random parity set and G1")
    par.add_argument("--timesteps", type=float, nargs="+", default=[0.001, 0.002])
    par.add_argument("--n", type=int, default=200)
    par.add_argument("--gate-seeds", type=int, default=8)
    thr = sub.add_parser(
        "throughput", parents=[common], help="episodes/s on the training suite per worker count"
    )
    thr.add_argument("--worker-counts", type=int, nargs="+", default=[1, 8, 16])
    thr.add_argument("--designs", type=int, default=8)
    sub.add_parser("entrants", parents=[common], help="manual entrants on the training suite")
    args = parser.parse_args(argv)
    task = default_task()
    if args.timestep:
        task = task.with_overrides(physics={"timestep": args.timestep})
    t0 = time.perf_counter()
    info = _machine()
    if args.cmd == "parity":
        result = timestep_parity(args.timesteps, args.n, args.workers, range(args.gate_seeds), task)
    elif args.cmd == "throughput":
        result = throughput(args.worker_counts, args.designs, task=task)
    else:
        result = entrant_scores(workers=args.workers, task=task)
    payload = {"command": args.cmd, "machine": info, "wall_s": round(time.perf_counter() - t0, 1), **result}
    text = json.dumps(payload, indent=1, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
