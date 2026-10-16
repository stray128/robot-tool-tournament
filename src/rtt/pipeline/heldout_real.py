"""Real-object held-out suite: select household objects, then score designs on them.

The suite is drawn from Google Scanned Objects by the rule in
``configs/suites/heldout_real_v1.yaml`` (:mod:`rtt.sim.gso`; protocol in
``docs/HELDOUT_REAL.md``). It is a secondary endpoint. Two steps, each writing a
self-contained directory:

1. :func:`select_real` draws the objects and writes ``selection.json`` (schema
   ``rtt.gso.selection/1``: accepted objects in draw order, each with its placement, jitter
   seeds and attribution, plus every exclusion with its reason), ``objects/<name>.npz``
   (the object pack: decomposed geometry, verified by SHA-256 on load) and
   ``ATTRIBUTION.md``.

   - ``mode="dev"`` uses the public ``draw.dev_seed``: the exploratory smoke subset.
   - ``mode="scored"`` uses the held-out secret, read only through
     :class:`~rtt.store.vault.HeldoutVault` and checked against its commitment, domain
     separated from the synthetic held-out suite. Every object in ``draw.exclude_names``
     (all objects simulated during development) is excluded. The current dev draw is
     recomputed and must be covered by that list, else the draw refuses to run, so no
     object seen during development is scored. The secret is never written.

2. :func:`run_real` scores frozen designs (at their frozen controller setting) and manual
   entrants (at their best training setting, or at every setting for exploration) on the
   selection's suite, through :func:`rtt.sim.evaluate.evaluate_designs`, and writes
   ``heldout_real.json`` (schema ``rtt.heldout_real/1``) and ``episodes.parquet``.

The object pack can be uploaded with the job inputs, so a scored job needs neither network
access nor CoACD. Run it as ``python -m rtt.pipeline.heldout_real {select,run} ...``.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from rtt.design.schema import FingerDesign
from rtt.search.select import FrozenDesign

#: The committed SHA-256 of the held-out seed (the same commitment as the procedural suite).
DEFAULT_COMMITMENT_FILE: Final = (
    Path(__file__).resolve().parents[3] / "configs" / "suites" / "heldout_v1.sha256"
)


def read_commitment(path: Path) -> str | None:
    """The first token of a commitment file (``<sha256>  <name>``), or ``None`` if it is missing."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").split()
    return text[0].lower() if text else None


__all__ = [
    "HELDOUT_REAL_SCHEMA",
    "RealEntry",
    "main",
    "read_selection",
    "run_real",
    "select_real",
    "width_band",
    "write_selection",
]

HELDOUT_REAL_SCHEMA: Final = "rtt.heldout_real/1"
SELECTION_FILE: Final = "selection.json"
REPORT: Final = "heldout_real.json"
EntrantSettings = Literal["training", "all"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- selection


def write_selection(selection: Any, out_dir: str | Path, cache: Any) -> Path:
    """Write ``selection.json``, the object pack and ``ATTRIBUTION.md`` into ``out_dir``.

    ``selection.json`` also records the code facts (git SHA, lock-file hash), outside the
    digest, so a scored draw names the code that screened it.
    """
    from rtt.sim.gso import GsoSource, attribution_markdown
    from rtt.store.manifest import code_facts, write_json_atomic

    out = Path(out_dir)
    if (out / SELECTION_FILE).exists():
        raise FileExistsError(f"{out} already holds a selection; use a fresh directory")
    pack = out / "objects"
    pack.mkdir(parents=True, exist_ok=True)
    src_dir = cache.objects_dir(selection.rules_key)
    for rec in selection.objects:
        shutil.copyfile(src_dir / f"{rec.name}.npz", pack / f"{rec.name}.npz")
    names = [o.name for o in selection.objects]
    (out / "ATTRIBUTION.md").write_text(
        attribution_markdown(names, GsoSource(**selection.source)), encoding="utf-8"
    )
    return write_json_atomic(
        out / SELECTION_FILE, {**selection.to_json(), "written_at": _utc_now(), "code": code_facts()}
    )


def select_real(
    out_dir: str | Path,
    *,
    mode: Literal["dev", "scored"],
    config: str | Path | None = None,
    seed_file: str | Path | None = None,
    commitment: str | None = None,
    n: int | None = None,
    workers: int = 3,
    cache_root: str | Path | None = None,
    fetch: Any | None = None,
    task: Any | None = None,
    progress: Callable[[int, Any], None] | None = None,
) -> Any:
    """Draw the dev or scored object set and write it to ``out_dir``; returns the Selection.

    ``fetch`` replaces the HTTPS fetcher (e.g. :class:`rtt.sim.gso.LocalMirror` for an
    offline mirror); ``cache_root`` the cache location.

    Raises:
        ValueError: scored mode without a seed file, a seed that does not match
            ``commitment``, a dev-draw object missing from ``draw.exclude_names``, or an
            excluded name that is not in the catalogue.
        VaultAccessError: called inside a search's vault guard.
    """
    from rtt.sim import gso
    from rtt.sim.task import default_task
    from rtt.store.vault import HeldoutVault

    cfg = gso.load_gso_config(config or gso.DEFAULT_CONFIG)
    task = task or default_task()
    cache = gso.GsoCache(cfg.source, root=cache_root, fetch=fetch)
    if (Path(out_dir) / SELECTION_FILE).exists():
        raise FileExistsError(f"{out_dir} already holds a selection; use a fresh directory")
    unknown = sorted(set(cfg.draw.exclude_names) - set(cache.catalogue()))
    if unknown:  # checked in dev mode too, so a typo shows up at the first dev run
        raise ValueError(f"draw.exclude_names lists objects not in the catalogue: {unknown}")
    dev_material = gso.dev_seed_material(cfg.draw.dev_seed)
    if mode == "dev":
        selection = gso.select_objects(
            cfg,
            dev_material,
            mode="dev",
            n=n,
            cache=cache,
            task=task,
            workers=workers,
            seed_info={"kind": "public", "dev_seed": cfg.draw.dev_seed},
            progress=progress,
        )
    elif mode == "scored":
        if seed_file is None:
            raise ValueError("the scored draw needs the held-out seed file")
        vault = HeldoutVault(seed_file)
        secret = vault.verify(commitment) if commitment else vault.read_seed()
        material = gso.secret_seed_material(secret)
        del secret
        exclude = {
            name: "simulated during development (draw.exclude_names)" for name in cfg.draw.exclude_names
        }
        dev_names: list[str] = []
        if cfg.draw.exclude_dev:
            dev = gso.select_objects(cfg, dev_material, mode="dev", cache=cache, task=task, workers=workers)
            dev_names = [o.name for o in dev.objects]
            unlisted = sorted(set(dev_names) - set(cfg.draw.exclude_names))
            if unlisted:
                # The dev draw changed since the list was written (a rule or code change), so
                # objects may have been simulated that the scored draw would not exclude.
                raise ValueError(
                    f"the current dev draw has {len(unlisted)} objects missing from draw.exclude_names "
                    f"({unlisted}); add them, since every object simulated in development stays out "
                    "of the scored draw"
                )
            exclude.update({name: "drawn for the dev subset" for name in dev_names})
        selection = gso.select_objects(
            cfg,
            material,
            mode="scored",
            n=n,
            cache=cache,
            task=task,
            exclude=exclude,
            workers=workers,
            seed_info={
                "kind": "heldout_secret",
                "commitment": commitment,
                "commitment_verified": commitment is not None,
                "dev_excluded": sorted(exclude),
                "dev_draw": dev_names,
            },
            progress=progress,
        )
        del material
    else:
        raise ValueError(f"mode must be 'dev' or 'scored', got {mode!r}")
    write_selection(selection, out_dir, cache)
    return selection


def read_selection(selection_dir: str | Path) -> tuple[Any, list[Any]]:
    """``(Selection, objects)`` from a directory written by :func:`select_real`.

    The selection digest and every object's geometry SHA-256 are verified.
    """
    import json

    from rtt.sim.gso import Selection, load_selection_objects

    d = Path(selection_dir)
    selection = Selection.from_json(json.loads((d / SELECTION_FILE).read_text(encoding="utf-8")))
    return selection, load_selection_objects(selection, d / "objects")


# --------------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class RealEntry:
    """One design to score: a frozen pick or a manual entrant, at one controller setting."""

    key: str
    arm: str
    run_index: int | None
    design: FingerDesign
    controller_setting: int
    frozen_sha256: str | None
    training_success: float | None


def _entrant_training(
    entrants: Sequence[FingerDesign], task: Any, executor: Any, workers: int | None
) -> dict[str, tuple[int, float]]:
    """Each entrant's best training controller setting and training score (same rule as every arm)."""
    from rtt.search.sim_adapter import SimEvaluator

    sim = SimEvaluator(task=task, executor=executor, workers=workers)
    return {
        d.design_id(): (s.best_k, s.success) for d, s in zip(entrants, sim.evaluate(entrants), strict=True)
    }


def _rate(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def width_band(width_m: float, training_widest_m: float) -> str:
    """``"within_training"`` when an object is no wider across the jaw than the widest
    training part (every valid design opens around it), else ``"wider"``.

    The design validator only guarantees an opening of the widest training part plus
    5 mm, so objects in the ``wider`` band also test how far a design opens, which the
    training suite never rewards. Reports give success in both bands.
    """
    return "within_training" if width_m <= training_widest_m + 1e-9 else "wider"


def run_real(
    selection_dir: str | Path,
    out_dir: str | Path,
    frozen: Sequence[FrozenDesign] = (),
    *,
    entrants: Sequence[FingerDesign] = (),
    entrant_settings: EntrantSettings = "training",
    limit: int | None = None,
    executor: Any = "cpu",
    workers: int | None = None,
    task: Any | None = None,
) -> dict[str, Any]:
    """Score ``frozen`` and ``entrants`` on a selection; write and return the report.

    ``entrant_settings="training"`` scores each entrant at its best training setting
    (measured here); ``"all"`` scores every setting and reports each (exploratory).
    ``limit`` keeps the first ``limit`` objects in draw order.
    """
    from rtt.design.generate import generate_finger
    from rtt.sim.conditions import training_suite
    from rtt.sim.evaluate import evaluate_designs, write_parquet
    from rtt.sim.exec_cpu import CpuExecutor
    from rtt.sim.gso import GsoSource, attribution, selection_suite
    from rtt.sim.objects import grip_width
    from rtt.sim.scene import open_gap
    from rtt.sim.task import default_task, task_digest
    from rtt.store.manifest import JobContext, code_facts, write_json_atomic

    out = Path(out_dir)
    if (out / REPORT).exists():
        raise FileExistsError(f"{out} already holds a report; use a fresh directory")
    if not frozen and not entrants:
        raise ValueError("nothing to evaluate: pass frozen designs and/or entrants")
    if entrant_settings not in ("training", "all"):
        raise ValueError("entrant_settings must be 'training' or 'all'")
    task = task or default_task()
    selection, objects = read_selection(selection_dir)
    if selection.task_digest != task_digest(task):
        raise ValueError("the selection was screened under a different task; re-select")
    suite = selection_suite(selection, objects, limit=limit)
    n_settings = len(task.controller.grasp_height_fracs)
    widest = training_suite().widest_grip_width
    band = {c.condition_id: width_band(grip_width(c.obj), widest) for c in suite}

    started_at, t0 = _utc_now(), time.perf_counter()
    own = executor == "cpu"
    pool = CpuExecutor(workers) if own else executor
    try:
        training = _entrant_training(entrants, task, pool, workers) if entrants else {}
        items = [
            RealEntry(
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
            best_k, train = training[d.design_id()]
            ks = [best_k] if entrant_settings == "training" else list(range(n_settings))
            for k in ks:
                suffix = "" if entrant_settings == "training" else f"-k{k}"
                items.append(RealEntry(f"entrant-{d.name}{suffix}", "entrant", None, d, k, None, train))
        by_k: dict[int, dict[str, FingerDesign]] = defaultdict(dict)
        for item in items:
            by_k[item.controller_setting][item.design.design_id()] = item.design
        records = []
        for k, designs in sorted(by_k.items()):
            records += evaluate_designs(
                list(designs.values()), suite, controller_settings=(k,), executor=pool, task=task
            )
    finally:
        if own:
            pool.close()
    wall = time.perf_counter() - t0
    write_parquet(records, out / "episodes.parquet")

    by_design: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for r in records:
        by_design[(r.design_id, r.k)].append(r)
    entrant_best = {did: k for did, (k, _) in training.items()}
    designs_out = []
    for item in items:
        recs = by_design[(item.design.design_id(), item.controller_setting)]
        per_object: dict[str, list[bool]] = defaultdict(list)
        per_band: dict[str, list[bool]] = {"within_training": [], "wider": []}
        for r in recs:
            per_object[r.size_label].append(r.success)
            per_band[band[r.condition_id]].append(r.success)
        designs_out.append(
            {
                "key": item.key,
                "arm": item.arm,
                "run_index": item.run_index,
                "design_id": item.design.design_id(),
                "name": item.design.name,
                "controller_setting": item.controller_setting,
                "grasp_height_frac": task.controller.grasp_height_fracs[item.controller_setting],
                "best_training_setting": entrant_best.get(item.design.design_id()),
                "frozen_sha256": item.frozen_sha256,
                "training_success": item.training_success,
                "open_gap_mm": round(1000.0 * open_gap(generate_finger(item.design), task), 1),
                "success": _rate([r.success for r in recs]),
                "episodes": len(recs),
                "success_by_width": {b: _rate(v) for b, v in per_band.items()},
                "objects_with_any_success": sum(any(v) for v in per_object.values()),
                "per_object": {name: _rate(v) for name, v in per_object.items()},
                "failure_modes": dict(
                    sorted(Counter(r.failure_mode or "unknown" for r in recs if not r.success).items())
                ),
                "flags": dict(sorted(Counter(f for r in recs for f in r.flags).items())),
                "penetration_flag_rate": _rate(["penetration" in r.flags for r in recs]),
            }
        )
    objects_out = []
    for cond, obj in zip(suite, (c.obj for c in suite), strict=True):
        recs = [r for r in records if r.condition_id == cond.condition_id]
        rec = next(o for o in selection.objects if o.name == obj.name)
        objects_out.append(
            {
                "object": obj.name,
                "condition_id": cond.condition_id,
                "size_mm": [round(1000.0 * s, 1) for s in obj.size],
                "width_band": band[cond.condition_id],
                "mass_kg": rec.mass_kg,
                "n_parts": rec.n_parts,
                "success": _rate([r.success for r in recs]),
                "penetration_flag_rate": _rate(["penetration" in r.flags for r in recs]),
                "failure_modes": dict(
                    sorted(Counter(r.failure_mode or "unknown" for r in recs if not r.success).items())
                ),
            }
        )
    arms: dict[str, list[float]] = defaultdict(list)
    for d in designs_out:
        if d["arm"] != "entrant" and d["success"] is not None:
            arms[d["arm"]].append(d["success"])
    source = GsoSource(**selection.source)
    report = {
        "schema": HELDOUT_REAL_SCHEMA,
        # a scored draw counts only when its seed was checked against the PREREG commitment
        "exploratory": selection.mode != "scored"
        or limit is not None
        or not selection.seed_info.get("commitment_verified", False),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_s": round(wall, 3),
        "selection": {
            "name": selection.name,
            "mode": selection.mode,
            "digest": selection.digest(),
            "config_digest": selection.config_digest,
            "seed": selection.seed_info,
            "objects": len(selection.objects),
            "requested": selection.requested,
            "examined": selection.examined,
            "exclusion_counts": selection.reason_counts(),
        },
        "suite": {
            "name": suite.name,
            "digest": suite.digest(),
            "n_conditions": len(suite),
            "seeds_per_condition": len(suite.conditions[0].seeds),
            "limit": limit,
            "training_widest_mm": round(1000.0 * widest, 1),
            "objects_by_width": dict(Counter(band.values())),
        },
        "task": {"name": task.name, "digest": task_digest(task)},
        "executor": "cpu" if own else getattr(executor, "__name__", type(executor).__name__),
        "entrant_settings": entrant_settings,
        "episodes": len(records),
        "designs": designs_out,
        "objects": objects_out,
        "arms": {
            arm: {
                "runs": values,
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else None,
            }
            for arm, values in sorted(arms.items())
        },
        "attribution": [attribution(c.obj.name, source) for c in suite],
        "job": JobContext.from_env().to_json(),
        "code": code_facts(),
    }
    write_json_atomic(out / REPORT, report)
    return report


# --------------------------------------------------------------------------- command line


def _print_progress(position: int, result: Any) -> None:
    status = "accept" if result.accepted else result.reason
    print(f"#{position:4d} {status:20s} {result.seconds:6.1f}s  {result.name}", file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m rtt.pipeline.heldout_real {select,run}``."""
    parser = argparse.ArgumentParser(prog="rtt heldout-real", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select", help="draw the dev or scored object set")
    sel.add_argument("--mode", choices=("dev", "scored"), required=True)
    sel.add_argument("--out", required=True, type=Path)
    sel.add_argument("--config", type=Path, default=None)
    sel.add_argument("--seed-file", type=Path, default=None, help="held-out seed (scored mode only)")
    sel.add_argument("--commitment", default=None, help="committed SHA-256 of the held-out seed")
    sel.add_argument(
        "--commitment-file",
        type=Path,
        default=DEFAULT_COMMITMENT_FILE,
        help="file holding the committed SHA-256 (default configs/suites/heldout_v1.sha256);"
        " used in scored mode when --commitment is not given",
    )
    sel.add_argument("--n", type=int, default=None, help="objects to accept (default from the config)")
    sel.add_argument(
        "--workers", type=int, default=3, help="parallel candidates (CoACD uses ~3 threads each)"
    )
    sel.add_argument(
        "--cache", type=Path, default=None, help="cache root (default $RTT_GSO_CACHE or ~/.cache/rtt/gso)"
    )
    sel.add_argument(
        "--mirror", type=Path, default=None, help="serve the source from a local mirror directory"
    )
    run = sub.add_parser("run", help="score designs on a selection")
    run.add_argument("--selection", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--frozen", nargs="*", default=[], help="frozen records or search output directories")
    run.add_argument("--entrants", type=Path, default=None, help="entrants YAML (e.g. configs/entrants.yaml)")
    run.add_argument("--entrant-settings", choices=("training", "all"), default="training")
    run.add_argument("--limit", type=int, default=None, help="first N objects only (exploratory)")
    run.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    if args.cmd == "select":
        from rtt.sim.gso import LocalMirror

        if args.mode == "scored" and args.commitment is None:
            # read the commitment from the committed file, never from a value typed at run time
            args.commitment = read_commitment(args.commitment_file)
            if args.commitment is None:
                print(
                    f"error: scored mode needs a commitment; {args.commitment_file} is missing",
                    file=sys.stderr,
                )
                return 2

        selection = select_real(
            args.out,
            mode=args.mode,
            config=args.config,
            seed_file=args.seed_file,
            commitment=args.commitment,
            n=args.n,
            workers=args.workers,
            cache_root=args.cache,
            fetch=LocalMirror(args.mirror) if args.mirror else None,
            progress=_print_progress,
        )
        print(
            f"{selection.mode} selection: {len(selection.objects)} objects after {selection.examined} examined; "
            f"exclusions {selection.reason_counts()}; digest {selection.digest()}"
        )
        if len(selection.objects) < selection.requested:
            print(f"warning: the pool ran out; {selection.requested} objects were requested", file=sys.stderr)
            return 1
        return 0

    from rtt.design.schema import load_entrants
    from rtt.pipeline.heldout import collect_frozen

    frozen = collect_frozen(args.frozen) if args.frozen else []
    entrants = load_entrants(args.entrants) if args.entrants else []
    report = run_real(
        args.selection,
        args.out,
        frozen,
        entrants=entrants,
        entrant_settings=args.entrant_settings,
        limit=args.limit,
        workers=args.workers,
    )
    for d in report["designs"]:
        print(
            f"{d['key']:28s} k={d['controller_setting']} success {d['success']:.3f} ({d['episodes']} episodes)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
