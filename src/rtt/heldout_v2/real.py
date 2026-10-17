"""H2: score every run's frozen ``freeform-v2`` pick on the scored scan draw (``heldout_real_v2``).

``rtt heldout-real {select,run} --space freeform-v2`` (``docs/PREREG.md`` A6, option A):

1. **Selection** (:func:`select_real_v2`): the ``heldout_real_v1`` selection rule, unchanged
   apart from the v2 decomposition in ``configs/suites/heldout_real_v2.yaml`` (hash in PREREG
   A5), through the v1 pipeline (:func:`rtt.pipeline.heldout_real.select_real`): the
   scored draw reads the secret through the vault, checks it against the commitment, and
   refuses to run unless every object of the current dev draw is in ``draw.exclude_names``.
   The screen runs under the rule's own task (``task_v1``, the default of
   :func:`~rtt.sim.gso.select_objects`), as every development draw of this config did. Next to
   ``selection.json`` it writes ``hull_hashes.json``: each accepted object's hull hash
   (``geometry_sha256`` of its convex pieces), width across the jaw and band, and the
   SHA-256 of the ``<hash>  <name>`` listing in draw order.
2. **Scoring** (:func:`run_real_v2`): the selection as a v2 part suite
   (:func:`rtt.sim.gso.selection_part_suite`; same objects, placements and 2 jitter seeds),
   every frozen pick and the six v2 entrants compiled against each object and scored under
   ``task_v2`` (:mod:`rtt.heldout_v2.common`). H2 is the run-level mean over the whole
   draw. Success is also reported per pre-registered width band (:func:`width_band`: at
   most 50 mm across the jaw, and above 50 mm), a planned split outside the Holm family.

``run`` without ``--selection`` makes the selection first (into ``OUT/selection``), then
scores it, so one command does both, as the held-out job does.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Literal

from rtt.heldout_v2 import common
from rtt.search.select import FrozenDesign

__all__ = [
    "BANDS",
    "BAND_EDGE_MM",
    "HULLS",
    "REPORT",
    "SCHEMA",
    "main",
    "run_real_v2",
    "select_real_v2",
    "width_band",
]

SCHEMA: Final = "rtt.heldout_real/2"
REPORT: Final = "heldout_real_v2.json"
EPISODES: Final = "episodes.parquet"
HULLS: Final = "hull_hashes.json"
SELECTION_DIR: Final = "selection"
#: The fixed band edge (founder, 2026-10-17: option A), in millimetres across the jaw.
BAND_EDGE_MM: Final = 50.0
BANDS: Final = ("width_le_50mm", "width_gt_50mm")


def width_band(width_m: float, edge_mm: float = BAND_EDGE_MM) -> str:
    """``"width_le_50mm"`` for an object at most ``edge_mm`` across the jaw, else ``"width_gt_50mm"``."""
    if edge_mm != BAND_EDGE_MM:
        raise ValueError(f"the band edge is fixed at {BAND_EDGE_MM} mm (PREREG A6)")
    return BANDS[0] if 1000.0 * width_m <= edge_mm + 1e-6 else BANDS[1]


def _v2_config() -> Path:
    from rtt.sim.gso import DEFAULT_CONFIG_V2

    return Path(DEFAULT_CONFIG_V2)


# --------------------------------------------------------------------------- selection


def hull_hashes(selection_dir: str | Path) -> dict[str, Any]:
    """Every accepted object's hull hash, width across the jaw and band, in draw order."""
    from rtt.pipeline.heldout_real import read_selection
    from rtt.sim.objects import grip_width

    selection, objects = read_selection(selection_dir)
    rows = []
    for rec, obj in zip(selection.objects, objects, strict=True):
        width = grip_width(obj)
        rows.append(
            {
                "name": rec.name,
                "geometry_sha256": rec.geometry_sha256,
                "n_parts": rec.n_parts,
                "width_mm": round(1000.0 * width, 3),
                "band": width_band(width),
                "file_sha256": rec.file_sha256,
            }
        )
    listing = "".join(f"{r['geometry_sha256']}  {r['name']}\n" for r in rows)
    return {
        "selection": selection.name,
        "mode": selection.mode,
        "selection_digest": selection.digest(),
        "config_digest": selection.config_digest,
        "band_edge_mm": BAND_EDGE_MM,
        "objects": rows,
        "objects_by_band": dict(sorted(Counter(r["band"] for r in rows).items())),
        "hull_listing_sha256": hashlib.sha256(listing.encode()).hexdigest(),
    }


def select_real_v2(
    out_dir: str | Path,
    *,
    mode: Literal["dev", "scored"],
    seed_file: str | Path | None = None,
    commitment: str | None = None,
    config: str | Path | None = None,
    n: int | None = None,
    workers: int = 3,
    cache_root: str | Path | None = None,
    fetch: Any | None = None,
    progress: Callable[[int, Any], None] | None = None,
) -> Any:
    """Draw the v2 scan set (``heldout_real_v2.yaml`` unless ``config``) and record its hull hashes.

    Raises:
        ValueError: scored mode without a seed file, a secret that does not match
            ``commitment``, or a dev-draw object missing from ``draw.exclude_names``.
        VaultAccessError: called inside a search's vault guard.
    """
    from rtt.pipeline.heldout_real import select_real
    from rtt.store.manifest import write_json_atomic

    selection = select_real(
        out_dir,
        mode=mode,
        config=config or _v2_config(),
        seed_file=seed_file,
        commitment=commitment,
        n=n,
        workers=workers,
        cache_root=cache_root,
        fetch=fetch,
        progress=progress,
    )
    write_json_atomic(Path(out_dir) / HULLS, hull_hashes(out_dir))
    return selection


# --------------------------------------------------------------------------- scoring


def run_real_v2(
    selection_dir: str | Path,
    out_dir: str | Path,
    frozen: Sequence[FrozenDesign] = (),
    *,
    entrants: Sequence[Any] | None = None,
    limit: int | None = None,
    executor: Any = "cpu",
    workers: int | None = None,
    task: Any = None,
) -> dict[str, Any]:
    """Score ``frozen`` and the entrants (default: the v2 file's six) on a v2 selection.

    The report is exploratory unless the selection is the scored draw of the frozen v2
    config, with a verified commitment, at full size.
    """
    from rtt.parts.suite import train_suite
    from rtt.pipeline.heldout_real import read_selection
    from rtt.sim import gso
    from rtt.sim.evaluate import write_parquet_v2
    from rtt.sim.objects import grip_width
    from rtt.sim.task import default_task, default_task_v2, task_digest
    from rtt.store.manifest import JobContext, code_facts, write_json_atomic

    out = Path(out_dir)
    if (out / REPORT).exists():
        raise FileExistsError(f"{out} already holds a report; use a fresh directory")
    entrants = common.load_entrants() if entrants is None else list(entrants)
    if not frozen and not entrants:
        raise ValueError("nothing to evaluate: pass frozen picks and/or entrants")
    for f in frozen:
        if f.design_space != common.DESIGN_SPACE:
            raise ValueError(f"{f.arm}-r{f.run_index:02d}: a {f.design_space} pick")
    task = task or default_task_v2()
    selection, objects = read_selection(selection_dir)
    screen_task = default_task()
    if selection.task_digest != task_digest(screen_task):
        raise ValueError(
            f"the selection was screened under task {selection.task_digest}, not the rule's screen task"
            f" {screen_task.name} ({task_digest(screen_task)}); re-select"
        )
    frozen_cfg = gso.load_gso_config(_v2_config())
    reasons = []
    if selection.mode != "scored":
        reasons.append(f"{selection.mode} selection")
    if not selection.seed_info.get("commitment_verified", False):
        reasons.append("commitment not checked")
    if selection.config_digest != frozen_cfg.digest():
        reasons.append("selection config differs from configs/suites/heldout_real_v2.yaml")
    if limit is not None:
        reasons.append(f"limited to the first {limit} objects")
    if selection.requested != frozen_cfg.draw.n_objects or len(selection.objects) != selection.requested:
        reasons.append(f"{len(selection.objects)} objects of {selection.requested} requested")

    t0, started_at = time.perf_counter(), common.utc_now()
    suite = gso.selection_part_suite(selection, objects, limit=limit)
    v1 = gso.selection_suite(selection, objects, limit=limit)
    obj_of = {c.condition_id: c.obj for c in v1}
    band_of = {cid: width_band(grip_width(obj)) for cid, obj in obj_of.items()}
    name_of = {cid: obj.name for cid, obj in obj_of.items()}

    entries = common.held_out_entries(frozen, entrants)
    sim = common.simulate(entries, suite, task, executor=executor, workers=workers)
    write_parquet_v2(sim.records, out / EPISODES)
    groups = {"by_band": lambda r: band_of[r.condition_id], "by_object": lambda r: name_of[r.condition_id]}
    designs = [common.design_summary(e, sim.by_design[e.design_id], groups) for e in entries]
    for d in designs:  # both bands always present; a band without objects has no rate
        for block in ("success", "lift_success", "episodes"):
            for band in BANDS:
                d["by_band"][block].setdefault(band, None)
            d["by_band"][block] = {b: d["by_band"][block][b] for b in BANDS}
    hull_record = hull_hashes(selection_dir)
    hulls = {r["name"]: r for r in hull_record["objects"]}
    objects_out = []
    for cond in suite:
        name = name_of[cond.condition_id]
        recs = [r for r in sim.records if r.condition_id == cond.condition_id]
        objects_out.append(
            {
                "object": name,
                "condition_id": cond.condition_id,
                "width_mm": hulls[name]["width_mm"],
                "band": band_of[cond.condition_id],
                "size_mm": [round(1000.0 * s, 1) for s in obj_of[cond.condition_id].size],
                "geometry_sha256": hulls[name]["geometry_sha256"],
                "n_parts": hulls[name]["n_parts"],
                "yaw_symmetry": cond.part.yaw_symmetry,
                "success_all_designs": common.mean_or_none(float(r.success) for r in recs),
            }
        )
    source = gso.GsoSource(**selection.source)
    report = {
        "schema": SCHEMA,
        "endpoint": "H2",
        "design_space": common.DESIGN_SPACE,
        "exploratory": bool(reasons),
        "exploratory_reasons": reasons,
        "started_at": started_at,
        "finished_at": common.utc_now(),
        "wall_s": round(time.perf_counter() - t0, 3),
        "timing_s": {"build": round(sim.build_s, 3), "simulate": round(sim.sim_s, 3)},
        "selection": {
            "name": selection.name,
            "mode": selection.mode,
            "digest": selection.digest(),
            "config_digest": selection.config_digest,
            "screen_task": {"name": screen_task.name, "digest": selection.task_digest},
            "seed": selection.seed_info,
            "objects": len(selection.objects),
            "requested": selection.requested,
            "examined": selection.examined,
            "exclusion_counts": selection.reason_counts(),
            "hull_listing_sha256": hull_record["hull_listing_sha256"],
        },
        "suite": {
            "name": suite.name,
            "digest": suite.digest(),
            "n_conditions": len(suite),
            "seeds_per_condition": len(suite.conditions[0].seeds),
            "episodes_per_design": suite.n_episodes,
            "limit": limit,
            "band_edge_mm": BAND_EDGE_MM,
            "objects_by_band": {b: sum(v == b for v in band_of.values()) for b in BANDS},
            "training_widest_mm": round(1000.0 * train_suite().widest_grip_width, 3),
        },
        "task": {"name": task.name, "digest": task_digest(task)},
        "generator_version": _generator_version(),
        "frozen": {"n": len(frozen), "digest": common.frozen_digest(frozen)},
        "builds": {"pairs": sim.builds, "invalid": sim.invalid_builds},
        "episodes": len(sim.records),
        "physics": common.physics_shares(sim.records),
        "designs": designs,
        "objects": objects_out,
        "arms": common.arm_summary(designs, extra=("by_band",)),
        "attribution": [gso.attribution(name_of[c.condition_id], source) for c in suite],
        "job": JobContext.from_env().to_json(),
        "code": code_facts(),
        "heldout_code": common.code_listing(),
    }
    write_json_atomic(out / REPORT, report)
    return report


def _generator_version() -> str:
    from rtt.design.freeform.hashing import GENERATOR_VERSION

    return str(GENERATOR_VERSION)


# --------------------------------------------------------------------------- command line


def _progress(position: int, result: Any) -> None:
    status = "accept" if result.accepted else result.reason
    print(f"#{position:4d} {status:20s} {result.seconds:6.1f}s", file=sys.stderr, flush=True)


def _commitment(args: argparse.Namespace) -> str | None:
    return args.commitment or common.read_commitment(args.commitment_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtt heldout-real --space freeform-v2",
        description="H2: the v2 scan draw (heldout_real_v2), selected by the hashed rule and scored under task_v2.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select", help="draw the dev or scored v2 object set; record its hull hashes")
    run = sub.add_parser(
        "run", help="score frozen v2 picks on a selection (selecting first when none is given)"
    )
    for p in (sel, run):
        p.add_argument("--space", choices=("freeform-v2", "v2"), default="freeform-v2")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--config", type=Path, default=None, help="suite file (default heldout_real_v2.yaml)")
        p.add_argument("--seed-file", type=Path, default=None, help="held-out seed (scored draw only)")
        p.add_argument("--commitment", default=None, help="committed SHA-256 (default: the committed file)")
        p.add_argument("--commitment-file", type=Path, default=common.COMMITMENT_FILE)
        p.add_argument("--n", type=int, default=None, help="objects to accept (development only)")
        p.add_argument("--cache", type=Path, default=None, help="GSO cache root")
        p.add_argument("--mirror", type=Path, default=None, help="serve the source from a local mirror")
    sel.add_argument("--mode", choices=("dev", "scored"), required=True)
    sel.add_argument(
        "--workers", type=int, default=3, help="parallel candidates (CoACD uses ~3 threads each)"
    )
    run.add_argument(
        "--mode", choices=("dev", "scored"), default="scored", help="draw made when --selection is absent"
    )
    run.add_argument("--selection", type=Path, default=None, help="an existing selection directory")
    run.add_argument("--frozen", nargs="*", default=[], help="frozen records, frozen/ dirs or search outputs")
    run.add_argument("--entrants", type=Path, default=None, help="entrants YAML (default entrants_v2.yaml)")
    run.add_argument("--no-entrants", action="store_true", help="score the frozen picks only")
    run.add_argument("--limit", type=int, default=None, help="first N objects only (exploratory)")
    run.add_argument("--workers", type=int, default=None, help="build and simulation processes")
    run.add_argument("--select-workers", type=int, default=3, help="parallel candidates while selecting")
    return parser


def _select(args: argparse.Namespace, out: Path, workers: int) -> Any:
    from rtt.sim.gso import LocalMirror

    commitment = None
    if args.mode == "scored":
        commitment = _commitment(args)
        if commitment is None:
            raise ValueError(f"scored mode needs a commitment; {args.commitment_file} is missing")
        if args.seed_file is None:
            raise ValueError("the scored draw needs --seed-file")
    return select_real_v2(
        out,
        mode=args.mode,
        seed_file=args.seed_file if args.mode == "scored" else None,
        commitment=commitment,
        config=args.config,
        n=args.n,
        workers=workers,
        cache_root=args.cache,
        fetch=LocalMirror(args.mirror) if args.mirror else None,
        progress=_progress,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "select":
            selection = _select(args, args.out, args.workers)
            print(
                f"{selection.mode} v2 selection: {len(selection.objects)} objects after {selection.examined}"
                f" examined; digest {selection.digest()}; hull hashes in {args.out / HULLS}"
            )
            return 0 if len(selection.objects) == selection.requested else 1
        selection_dir = args.selection
        if selection_dir is None:
            selection_dir = args.out / SELECTION_DIR
            _select(args, selection_dir, args.select_workers)
        frozen = common.collect_frozen_v2(args.frozen) if args.frozen else []
        entrants = [] if args.no_entrants else common.load_entrants(args.entrants)
        report = run_real_v2(
            selection_dir, args.out, frozen, entrants=entrants, limit=args.limit, workers=args.workers
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"H2 {'(exploratory) ' if report['exploratory'] else ''}{report['suite']['n_conditions']} objects x"
        f" {report['suite']['seeds_per_condition']} seeds, {report['episodes']} episodes;"
        f" bands {report['suite']['objects_by_band']}"
    )
    for arm, a in report["arms"].items():
        print(f"  arm {arm}: {a['n']} runs")
    print(f"Wrote {args.out / REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
