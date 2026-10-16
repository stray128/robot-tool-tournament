"""Gate GV4: the v2 path gives the same episodes on a Nebius job as on the laptop (spec section 14).

A fixed set of v2 designs, the six entrants of ``configs/entrants_v2.yaml`` (the four round-0
entrants and the two references) and two Sobol recipes of the v2 space (seed
:data:`GV4_SOBOL_SEED`, which no gate, pilot or confirmatory run uses), is compiled and
simulated on the 48 episodes of ``train_v2`` exactly as a search scores it
(:class:`~rtt.search.sim_adapter.SimEvaluatorV2`): 8 x 48 = 384 episodes, more than the 100
the gate asks for. Jaws are built where the episodes run, so the job checks the generator
as well as the physics.

The report records, per episode, every field of the v2 record except its wall time and a
SHA-256 of them (``digest``); per design, its ``design_id`` and ``geometry_id``; per training
part, its geometry hash; and the stock-blade hull file's SHA-256. :func:`compare` matches two
reports episode by episode.

**Pass:** the same episodes on both sides, at least 99 % of them with the same outcome
(``success_v2``, lift-only success and failure mode), and every part, jaw-pair, design
geometry and hull hash equal. (Otherwise the spec's fallback applies: build the geometry
once, store it by hash and score the stored geometry.)

Usage::

    python -m rtt.sim.gv4 run --out runs/gates-v2/gv4/laptop.json
    python -m rtt.cloud run gv4 --no-fallback --vcpus 32      # the same inside a Nebius job
    python -m rtt.sim.gv4 compare runs/gates-v2/gv4/laptop.json runs/cloud/<run_id>/gv4.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

__all__ = [
    "GV4_AGREEMENT",
    "GV4_MIN_EPISODES",
    "GV4_SOBOL_SEED",
    "SCHEMA",
    "compare",
    "gv4_designs",
    "main",
    "record_digest",
    "run_parity",
]

SCHEMA: Final = "rtt.gv4/1"
#: Seed of the two Sobol recipes (public; no gate, pilot or confirmatory run uses it).
GV4_SOBOL_SEED: Final = 20261024
GV4_N_SOBOL: Final = 2
#: Pass: at least this share of the episodes agree on outcome ...
GV4_AGREEMENT: Final = 0.99
#: ... over at least this many episodes.
GV4_MIN_EPISODES: Final = 100
#: Record fields that are not part of the outcome comparison (wall time is not deterministic).
_SKIP: Final = frozenset({"wall_time_s"})


def record_digest(record: Mapping[str, Any]) -> str:
    """SHA-256 of a v2 record's fields (wall time excluded; NaN as ``null``, floats as ``repr``)."""
    body = {k: v for k, v in record.items() if k not in _SKIP and k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _clean(record: Any) -> dict[str, Any]:
    data = record.to_dict() if hasattr(record, "to_dict") else dict(record)
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k in _SKIP:
            continue
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        out[k] = v
    return out


def gv4_designs(context: Any) -> list[tuple[str, Any]]:
    """``(label, design)``: the v2 entrants, then :data:`GV4_N_SOBOL` valid Sobol recipes."""
    from rtt.search.runner import DEFAULT_ENTRANTS_V2, _all_entrants_v2
    from rtt.search.space import V2, get_space

    space = get_space(V2)
    out = [
        (str(space.label(e) or f"entrant_{i}"), e)
        for i, e in enumerate(_all_entrants_v2(DEFAULT_ENTRANTS_V2))
    ]
    sampler = space.sample_sobol(GV4_SOBOL_SEED, context)
    out += [(f"sobol_{i:02d}", d) for i, d in enumerate(sampler.sample(GV4_N_SOBOL))]
    return out


def _environment() -> dict[str, Any]:
    from importlib import metadata

    def version(dist: str) -> str | None:
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return None

    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "machine": platform.machine(),
        "cpu": cpu,
        "python": platform.python_version(),
        "mujoco": version("mujoco"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "trimesh": version("trimesh"),
        "manifold3d": version("manifold3d"),
    }


def run_parity(workers: int | None = None) -> dict[str, Any]:
    """Build and simulate the GV4 designs on ``train_v2``; the report :func:`compare` reads."""
    from rtt.render.gripper import load_stock_blade_hulls
    from rtt.search.sim_adapter import SimEvaluatorV2
    from rtt.search.space import V2, get_space
    from rtt.sim.exec_cpu import CpuExecutor

    space = get_space(V2)
    t0 = time.perf_counter()
    pool = CpuExecutor(workers)
    try:
        ev = SimEvaluatorV2(executor=pool, workers=workers)
        designs = gv4_designs(ev.context)
        t_build = time.perf_counter()
        scores = ev.evaluate([d for _, d in designs])
        records = [_clean(r) for r in ev.last_records]
    finally:
        pool.close()
    t1 = time.perf_counter()
    per_design = []
    for (label, design), score in zip(designs, scores, strict=True):
        per_design.append(
            {
                "label": label,
                "design_id": space.design_id(design),
                "geometry_id": space.geometry_id(design, ev.context),
                "score": float(score.success),
                "lift_score": float(score.lift_success) if score.lift_success is not None else None,
            }
        )
    for r in records:
        r["digest"] = record_digest(r)
    hulls = load_stock_blade_hulls()
    return {
        "schema": SCHEMA,
        "sobol_seed": GV4_SOBOL_SEED,
        "task_digest": ev.task_hash,
        "suite_digest": ev.suite_hash,
        "parts": {str(p.part_id): str(p.geometry_sha256) for p in ev.parts},
        "stock_hulls_sha256": hulls.sha256,
        "designs": per_design,
        "episodes": len(records),
        "records": records,
        "records_sha256": hashlib.sha256("".join(r["digest"] for r in records).encode()).hexdigest(),
        "environment": _environment(),
        "timing_s": {"setup": round(t_build - t0, 2), "evaluate": round(t1 - t_build, 2)},
    }


def _key(r: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(r["design_id"]), str(r["condition_id"]), int(r["seed"]))


def _outcome(r: Mapping[str, Any]) -> tuple[bool, bool, str | None]:
    return (bool(r["success"]), bool(r["lift_success"]), r.get("failure_mode"))


def compare(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Match two GV4 reports episode by episode (``a`` the reference, e.g. the laptop)."""
    for side in (a, b):
        if side.get("schema") != SCHEMA:
            raise ValueError(f"not a GV4 report (schema {side.get('schema')!r})")
    ra = {_key(r): r for r in a["records"]}
    rb = {_key(r): r for r in b["records"]}
    common = sorted(set(ra) & set(rb))
    outcome_agree = [k for k in common if _outcome(ra[k]) == _outcome(rb[k])]
    exact = [k for k in common if ra[k]["digest"] == rb[k]["digest"]]
    geometry_agree = [k for k in common if ra[k].get("geometry_hash") == rb[k].get("geometry_hash")]
    max_diff: dict[str, float] = {}
    for k in common:
        for name, va in ra[k].items():
            vb = rb[k].get(name)
            if isinstance(va, float) and isinstance(vb, float) and va != vb:
                max_diff[name] = max(max_diff.get(name, 0.0), abs(va - vb))
    differ = [
        {
            "design_id": k[0][:12],
            "condition_id": k[1],
            "seed": k[2],
            "a": _outcome(ra[k]),
            "b": _outcome(rb[k]),
        }
        for k in common
        if _outcome(ra[k]) != _outcome(rb[k])
    ]
    da = {d["design_id"]: d for d in a["designs"]}
    db = {d["design_id"]: d for d in b["designs"]}
    checks = {
        "same_episodes": set(ra) == set(rb),
        "enough_episodes": len(common) >= GV4_MIN_EPISODES,
        "outcome_agreement": bool(common) and len(outcome_agree) / len(common) >= GV4_AGREEMENT,
        "part_hashes_equal": a["parts"] == b["parts"],
        "jaw_pair_hashes_equal": len(geometry_agree) == len(common),
        "design_geometry_ids_equal": set(da) == set(db)
        and all(da[i]["geometry_id"] == db[i]["geometry_id"] for i in da),
        "stock_hulls_equal": a["stock_hulls_sha256"] == b["stock_hulls_sha256"],
        "same_task_and_suite": (a["task_digest"], a["suite_digest"]) == (b["task_digest"], b["suite_digest"]),
    }
    return {
        "schema": "rtt.gv4-compare/1",
        "episodes": len(common),
        "only_in_a": len(set(ra) - set(rb)),
        "only_in_b": len(set(rb) - set(ra)),
        "outcome_agree": len(outcome_agree),
        "outcome_agreement": round(len(outcome_agree) / len(common), 6) if common else 0.0,
        "bit_identical_records": len(exact),
        "records_sha256_equal": a.get("records_sha256") == b.get("records_sha256"),
        "max_abs_diff": dict(sorted(max_diff.items())),
        "outcome_differences": differ,
        "designs": [
            {
                "label": da[i]["label"],
                "design_id": i[:12],
                "geometry_id_equal": i in db and da[i]["geometry_id"] == db[i]["geometry_id"],
                "score_a": da[i]["score"],
                "score_b": db[i]["score"] if i in db else None,
            }
            for i in da
        ],
        "environment": {"a": a.get("environment"), "b": b.get("environment")},
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rtt.sim.gv4", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="simulate the GV4 episodes here and write the report")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--workers", type=int, default=None)
    cmp_ = sub.add_parser("compare", help="compare two GV4 reports (reference first)")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)
    cmp_.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        report = run_parity(args.workers)
        _write(args.out, report)
        print(
            json.dumps(
                {
                    "episodes": report["episodes"],
                    "records_sha256": report["records_sha256"],
                    **report["timing_s"],
                }
            )
        )
        return 0
    a = json.loads(args.a.read_text(encoding="utf-8"))
    b = json.loads(args.b.read_text(encoding="utf-8"))
    result = compare(a, b)
    if args.out:
        _write(args.out, result)
    summary = {k: result[k] for k in ("episodes", "outcome_agreement", "bit_identical_records", "passed")}
    print(json.dumps({**summary, "checks": result["checks"]}, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
