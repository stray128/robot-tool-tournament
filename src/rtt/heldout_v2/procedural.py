"""H1: score every run's frozen ``freeform-v2`` pick on ``heldout_procedural_v2``.

``rtt heldout --space freeform-v2 --seed-file PATH --frozen DIR... --out DIR``
(:func:`run_heldout_v2`; ``docs/PREREG.md`` A6):

1. The secret is read through :class:`~rtt.store.vault.HeldoutVault` from an explicit file
   and checked against the committed SHA-256 (``configs/suites/heldout_v1.sha256``); the
   vault refuses inside a search's :class:`~rtt.store.vault.VaultGuard`.
2. The suite is :func:`rtt.parts.heldout.heldout_procedural_suite` of that secret and the
   frozen ``configs/suites/heldout_procedural_v2.yaml``: 100 conditions x 2 jitter seeds =
   200 episodes per design. It is drawn with the module's default task, as frozen; its
   only task input, the grip's mass cap, is the same in ``task_v1`` and ``task_v2``.
3. Every frozen pick, the four round-0 entrants and the two reference entrants are compiled
   against each held-out part and scored under ``task_v2`` (:mod:`rtt.heldout_v2.common`).
4. ``heldout_v2.json`` (schema ``rtt.heldout/2``) holds per-design and per-arm results
   (registered and lift-only), the suite digest and composition, the commitment and the
   code listing; ``episodes.parquet`` every episode; ``suite.json`` the drawn parts and
   placements. The secret itself is never written.

A report is ``exploratory`` (never an H1 result) unless the commitment was verified and the
suite is the frozen one at full size.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from rtt.heldout_v2 import common
from rtt.search.select import FrozenDesign

__all__ = ["EPISODES", "REPORT", "SCHEMA", "SUITE", "run_heldout_v2"]

SCHEMA: Final = "rtt.heldout/2"
REPORT: Final = "heldout_v2.json"
EPISODES: Final = "episodes.parquet"
SUITE: Final = "suite.json"


def run_heldout_v2(
    seed_file: str | Path,
    out_dir: str | Path,
    frozen: Sequence[FrozenDesign],
    *,
    commitment: str | None,
    entrants: Sequence[Any] | None = None,
    config: Any = None,
    n_conditions: int | None = None,
    seeds_per_condition: int | None = None,
    executor: Any = "cpu",
    workers: int | None = None,
    task: Any = None,
) -> dict[str, Any]:
    """Score ``frozen`` (and ``entrants``, default the v2 file's six) on the procedural suite.

    ``config`` (a :class:`~rtt.parts.heldout.HeldoutConfig` or a path) and the size
    overrides are for development runs only; any of them makes the report exploratory.

    Raises:
        ValueError: the secret does not match ``commitment``, or nothing to score.
        VaultAccessError: called inside a search's vault guard.
        FileExistsError: ``out_dir`` already holds a report.
    """
    from rtt.parts import heldout as hp
    from rtt.sim.evaluate import write_parquet_v2
    from rtt.sim.task import default_task_v2, task_digest
    from rtt.store.manifest import JobContext, code_facts, file_sha256, write_json_atomic
    from rtt.store.vault import HeldoutVault

    out = Path(out_dir)
    if (out / REPORT).exists():
        raise FileExistsError(f"{out} already holds a held-out report; use a fresh directory")
    entrants = common.load_entrants() if entrants is None else list(entrants)
    if not frozen and not entrants:
        raise ValueError("nothing to evaluate: pass frozen picks and/or entrants")
    for f in frozen:
        if f.design_space != common.DESIGN_SPACE:
            raise ValueError(f"{f.arm}-r{f.run_index:02d}: a {f.design_space} pick")
    task = task or default_task_v2()
    frozen_cfg = hp.load_heldout_config(hp.DEFAULT_CONFIG)
    if config is None:
        cfg = frozen_cfg
    elif isinstance(config, hp.HeldoutConfig):
        cfg = config
    else:
        cfg = hp.load_heldout_config(config)
    if seeds_per_condition is not None:
        cfg = dataclasses.replace(cfg, seeds_per_condition=int(seeds_per_condition))
    reasons = []
    if commitment is None:
        reasons.append("commitment not checked")
    if cfg.digest() != frozen_cfg.digest():
        reasons.append("suite config differs from configs/suites/heldout_procedural_v2.yaml")
    if n_conditions is not None and n_conditions != cfg.n_conditions:
        reasons.append(f"n_conditions overridden ({n_conditions})")

    started_at, t0 = common.utc_now(), time.perf_counter()
    vault = HeldoutVault(seed_file)
    secret = vault.verify(commitment) if commitment else vault.read_seed()
    suite = hp.heldout_procedural_suite(secret, cfg, commitment=commitment, n=n_conditions)
    del secret  # nothing below needs it; it is never written
    draw_s = time.perf_counter() - t0

    entries = common.held_out_entries(frozen, entrants)
    sim = common.simulate(entries, suite, task, executor=executor, workers=workers)
    write_parquet_v2(sim.records, out / EPISODES)
    groups = {
        "by_family": lambda r: r.family,
        "by_size_mode": lambda r: r.size_label,
    }
    designs = [common.design_summary(e, sim.by_design[e.design_id], groups) for e in entries]
    seeds = {len(c.seeds) for c in suite}
    summary = hp.suite_summary(suite)
    write_json_atomic(out / SUITE, {"name": suite.name, "digest": suite.digest(), "summary": summary,
                                    **suite.describe()})  # fmt: skip
    report = {
        "schema": SCHEMA,
        "endpoint": "H1",
        "design_space": common.DESIGN_SPACE,
        "exploratory": bool(reasons),
        "exploratory_reasons": reasons,
        "started_at": started_at,
        "finished_at": common.utc_now(),
        "wall_s": round(time.perf_counter() - t0, 3),
        "timing_s": {
            "draw": round(draw_s, 3),
            "build": round(sim.build_s, 3),
            "simulate": round(sim.sim_s, 3),
        },
        "suite": {
            "name": suite.name,
            "digest": suite.digest(),
            "n_conditions": len(suite),
            "seeds_per_condition": seeds.pop() if len(seeds) == 1 else sorted(seeds),
            "episodes_per_design": suite.n_episodes,
            "summary": summary,
            "config_digest": cfg.digest(),
            "config_sha256": file_sha256(hp.DEFAULT_CONFIG) if cfg == frozen_cfg else None,
            "generator_module_sha256": file_sha256(Path(hp.__file__)),
            "domain": hp.DOMAIN.rstrip(b"\x00").decode(),
        },
        "commitment": commitment,
        "commitment_verified": commitment is not None,
        "task": {"name": task.name, "digest": task_digest(task)},
        "generator_version": _generator_version(),
        "frozen": {"n": len(frozen), "digest": common.frozen_digest(frozen)},
        "builds": {"pairs": sim.builds, "invalid": sim.invalid_builds},
        "episodes": len(sim.records),
        "physics": common.physics_shares(sim.records),
        "designs": designs,
        "arms": common.arm_summary(designs),
        "job": JobContext.from_env().to_json(),
        "code": code_facts(),
        "heldout_code": common.code_listing(),
    }
    write_json_atomic(out / REPORT, report)
    return report


def _generator_version() -> str:
    from rtt.design.freeform.hashing import GENERATOR_VERSION

    return str(GENERATOR_VERSION)
