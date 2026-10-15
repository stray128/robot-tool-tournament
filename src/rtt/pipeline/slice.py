"""The vertical slice (``rtt slice``): one Nemotron round, end to end, with replays.

1. **Round 0** is the shared protocol round: the manual entrants plus a Sobol fill
   (:func:`rtt.search.random_search.round0_designs`), simulated on the training suite.
2. **Round 1**: Nemotron 3 Super on Token Factory sees those results and proposes 4
   designs in one forced ``submit_designs`` tool call (one island: two mutations, one
   crossover, one exploration). Every design goes through the validator, at most one
   repair call and the diversity check; a slot that still fails is filled at random
   and logged as ``fallback_random``. This is :func:`rtt.search.runner.run_search` with
   one run, so the manifest, history, rejections, LLM-call log and response cache are
   exactly what a full search writes.
3. The 4 designs are simulated on the CPU executor and scored with the same rule as
   every arm (best controller setting's mean success over 36 episodes).
4. **Replays**: on one training condition, chosen by a fixed rule (the one where the
   lanes' scored outcomes differ most), every proposed design and the baseline
   entrant are re-simulated with recording on, at each design's own best controller
   setting and the condition's scored jitter seed, and must reproduce their scored
   outcome (:func:`rtt.pipeline.replays.record_replays`).
5. ``slice.json`` (schema ``rtt.slice/1``) ties it together for ``rtt export-web``:
   designs with their provenance (model, tokens, latency, cost of the call that made
   them), scores, lanes and compute facts.

Live model spend is capped (default $0.20) by the client's ledger; with
``cache_mode="replay"`` and ``cache_dir`` pointing at an earlier slice's ``llm_cache``,
the slice re-runs from that cache with no API key and must reproduce it exactly.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rtt.llm.cache import CacheMode
from rtt.llm.client import LLMClient
from rtt.search.base import Evaluated, LLMCallRecord
from rtt.search.nemotron import PROPOSER_SLOTS, ProposerConfig, make_proposer_client
from rtt.search.runner import SearchConfig, SearchOutcome, run_search
from rtt.search.sim_adapter import SimEvaluator, describe_condition
from rtt.store.manifest import JobContext, LLMUsage, code_facts, write_json_atomic

__all__ = ["SLICE_SCHEMA", "SliceConfig", "SliceOutcome", "choose_condition", "run_slice"]

SLICE_SCHEMA: Final = "rtt.slice/1"
SUMMARY: Final = "slice.json"


@dataclass(frozen=True)
class SliceConfig:
    """Knobs of one slice run.

    Args:
        seed: Run seed (round-0 Sobol fill and the random fallback stream).
        round0_size: Designs in round 0 (entrants first, then Sobol).
        designs: Designs Nemotron proposes (a multiple of the 4 tool slots; one island per 4).
        spend_cap_usd: Hard cap on live model spend for this slice.
        baseline: Entrant replayed beside the proposed designs.
        workers: CPU worker processes (default: all CPUs).
        cache_mode: ``readwrite`` (live calls, cached) or ``replay`` (cache only, no key).
        cache_dir: Response cache (default ``<out>/llm_cache``); point a replay at the
            cache of an earlier slice to re-run it with zero API calls.
    """

    seed: int = 17
    round0_size: int = 8
    designs: int = 4
    spend_cap_usd: float = 0.20
    baseline: str = "stock"
    workers: int | None = None
    cache_mode: CacheMode = "readwrite"
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.designs < len(PROPOSER_SLOTS) or self.designs % len(PROPOSER_SLOTS):
            raise ValueError(
                f"designs must be a positive multiple of {len(PROPOSER_SLOTS)} (one tool call each)"
            )
        if self.spend_cap_usd <= 0:
            raise ValueError("spend_cap_usd must be > 0")


@dataclass
class SliceOutcome:
    """What :func:`run_slice` produced."""

    out_dir: Path
    summary: dict[str, Any]
    search: SearchOutcome

    @property
    def summary_path(self) -> Path:
        return self.out_dir / SUMMARY


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _outcome_label(success: bool, failure_mode: str | None) -> str:
    return "success" if success else (failure_mode or "failure")


def choose_condition(lanes: Sequence[Evaluated], conditions: Sequence[Any]) -> Any:
    """The training condition whose scored outcomes differ most across ``lanes``.

    Each lane is read at its own best controller setting and the condition's first
    jitter seed. Ranking (fixed before any result is seen): most distinct outcomes
    (success or each failure mode), then the most successes among the lanes, then
    suite order. So a bout shows contrast when there is any, and a success when
    possible.
    """
    best = None
    best_key: tuple[int, int, int] | None = None
    for order, cond in enumerate(conditions):
        outcomes = []
        for lane in lanes:
            ep = _episode(lane, cond.condition_id, cond.seeds[0])
            outcomes.append(_outcome_label(ep.success, ep.failure_mode))
        key = (len(set(outcomes)), sum(o == "success" for o in outcomes), -order)
        if best_key is None or key > best_key:
            best, best_key = cond, key
    if best is None:
        raise ValueError("no conditions to choose from")
    return best


def _episode(item: Evaluated, condition_id: str, seed: int) -> Any:
    for ep in item.score.episodes:
        if ep.condition_id == condition_id and ep.controller_setting == item.score.best_k and ep.seed == seed:
            return ep
    raise LookupError(
        f"{item.label}: no scored episode for {condition_id} at k={item.score.best_k}, seed {seed}"
    )


def _call_by_hash(calls: Sequence[LLMCallRecord]) -> dict[str, LLMCallRecord]:
    return {c.request_hash: c for c in calls if c.request_hash}


def _design_entry(item: Evaluated, calls: dict[str, LLMCallRecord], info: Any) -> dict[str, Any]:
    p = item.proposal
    call = calls.get(p.request_hash or "")
    s = item.score
    entry: dict[str, Any] = {
        "label": item.label,
        "round": item.round,
        "design_id": s.design_id,
        "name": p.design.name,
        "params": p.design.geometry_params(),
        "origin": p.origin,
        "operator": p.operator,
        "slot": p.slot,
        "parents": list(p.parents),
        "hypothesis": p.hypothesis,
        "via": p.via,
        "fallback_reason": p.fallback_reason,
        "score": {
            "success": s.success,
            "best_k": s.best_k,
            "grasp_height_frac": info.controller_settings[s.best_k],
            "success_by_k": list(s.success_by_k),
            "hold_fraction": s.hold_fraction,
            "per_family": dict(s.per_family),
            "failure_modes": dict(s.failure_modes),
            "mass_g": s.mass_g,
            "n_episodes": s.n_episodes,
        },
        "llm_call": None,
    }
    if call is not None:
        entry["llm_call"] = {
            "kind": call.kind,
            "request_hash": call.request_hash,
            "cached": call.cached,
            "prompt_tokens": call.prompt_tokens,
            "completion_tokens": call.completion_tokens,
            "reasoning_tokens": call.reasoning_tokens,
            "cost_usd": call.cost_usd,
            "latency_s": call.latency_s,
            "finish_reason": call.finish_reason,
        }
    return entry


def run_slice(
    out_dir: str | Path,
    config: SliceConfig | None = None,
    *,
    llm_client: LLMClient | None = None,
    evaluator: SimEvaluator | None = None,
    executor: Any | None = None,
    job: JobContext | None = None,
) -> SliceOutcome:
    """Run the vertical slice into ``out_dir`` (must not hold a previous slice).

    ``llm_client`` defaults to the pre-registered proposer client with its cache in
    ``out_dir/llm_cache`` and a ledger capped at ``config.spend_cap_usd``;
    ``evaluator`` defaults to the training suite on ``executor`` (default: a CPU pool
    of ``config.workers``, reused for the replays).
    """
    from rtt.design.schema import load_entrants
    from rtt.pipeline.replays import ReplayRequest, record_replays
    from rtt.search.runner import DEFAULT_ENTRANTS
    from rtt.sim.exec_cpu import CpuExecutor
    from rtt.sim.task import task_digest

    config = config or SliceConfig()
    out = Path(out_dir)
    if (out / SUMMARY).exists():
        raise FileExistsError(f"{out} already holds a slice; use a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    started_at, t0 = _utc_now(), time.perf_counter()
    own_pool = executor is None and evaluator is None
    pool = executor if executor is not None else (CpuExecutor(config.workers) if own_pool else None)
    try:
        sim = evaluator or SimEvaluator(executor=pool)
        client = llm_client or make_proposer_client(
            config.cache_dir or out / "llm_cache",
            cache_mode=config.cache_mode,
            ledger_path=out / "llm_ledger.jsonl",
            budget_usd=config.spend_cap_usd,
        )
        entrants = load_entrants(DEFAULT_ENTRANTS)
        if config.baseline not in {e.name for e in entrants}:
            raise ValueError(f"baseline {config.baseline!r} is not an entrant")
        search = run_search(
            "nemotron",
            1,
            config.round0_size + config.designs,
            [config.seed],
            sim,
            out / "search",
            client,
            config=SearchConfig(
                round0_size=config.round0_size,
                round_size=config.designs,
                proposer=ProposerConfig(islands=config.designs // len(PROPOSER_SLOTS)),
            ),
            entrants=entrants,
            job=job,
        )
        run = search.runs[0].ctx
        info = run.info
        proposed = [item for item in run.history if item.round == 1]
        baseline = next(
            item for item in run.history if item.round == 0 and item.proposal.design.name == config.baseline
        )
        lanes = [baseline, *proposed]
        condition = choose_condition(lanes, list(sim.suite))
        seed = condition.seeds[0]
        requests = []
        for item in lanes:
            ep = _episode(item, condition.condition_id, seed)
            p = item.proposal
            stem = (
                "baseline-" + config.baseline
                if item is baseline
                else (p.slot or item.label).replace("_", "-")
            )
            requests.append(
                ReplayRequest(
                    stem=stem,
                    design=p.design,
                    condition=condition,
                    controller_setting=item.score.best_k,
                    seed=seed,
                    expect_success=ep.success,
                    expect_failure_mode=ep.failure_mode,
                    meta={"label": item.label, "origin": p.origin, "slot": p.slot, "entrant": p.design.name},
                )
            )
        replay_pool = pool if isinstance(pool, CpuExecutor) else None
        recorded = record_replays(
            requests, out / "replays", task=sim.task, executor=replay_pool, workers=config.workers
        )
    finally:
        if own_pool and pool is not None:
            pool.close()

    calls = _call_by_hash(run.llm_calls)
    usage = LLMUsage.from_records(c.to_json() for c in run.llm_calls)
    manifest = search.manifest
    job_ctx = job or JobContext.from_env()
    summary: dict[str, Any] = {
        "schema": SLICE_SCHEMA,
        "slice_id": out.name,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_s": round(time.perf_counter() - t0, 3),
        "config": {
            "seed": config.seed,
            "round0_size": config.round0_size,
            "designs": config.designs,
            "spend_cap_usd": config.spend_cap_usd,
            "baseline": config.baseline,
            "cache_mode": config.cache_mode,
            "cache_dir": str(config.cache_dir) if config.cache_dir else "llm_cache",
        },
        "task": {
            "name": sim.task.name,
            "digest": task_digest(sim.task),
            "grip_force_n": sim.task.gripper.grip_force,
            "max_opening_mm": round(2000.0 * sim.task.gripper.open_half_gap, 3),
        },
        "suite": {"name": sim.suite.name, "digest": sim.suite.digest(), "n_conditions": len(sim.suite)},
        "model": manifest.get("model"),
        "provider": manifest.get("provider"),
        "llm": usage.to_json(),
        "llm_calls": [c.to_json() for c in run.llm_calls],
        "rejections": [r.to_json() for r in run.rejections],
        "fallback_random": run.budget.fallbacks,
        "llm_disabled": manifest["per_run"][0].get("llm_disabled"),
        "sim": manifest.get("sim"),
        "designs": [_design_entry(item, calls, info) for item in run.history],
        "condition": {
            "condition_id": condition.condition_id,
            "description": describe_condition(condition),
            "family": condition.family,
            "kind": condition.obj.kind,
            "size_m": list(condition.obj.size),
            "seed": seed,
        },
        "lanes": [
            {
                "stem": rec.request.stem,
                "replay": f"replays/{rec.request.stem}",
                "label": item.label,
                "design_id": item.score.design_id,
                "origin": item.proposal.origin,
                "controller_setting": rec.request.controller_setting,
                "grasp_height_frac": info.controller_settings[rec.request.controller_setting],
                "seed": seed,
                "success": rec.success,
                "failure_mode": rec.failure_mode,
                "max_lift_m": rec.max_lift_m,
                "hold_time_s": rec.hold_time_s,
                "flags": list(rec.flags),
                "n_frames": rec.n_frames,
                "n_steps": rec.n_steps,
                "wall_time_s": round(rec.wall_time_s, 3),
            }
            for rec, item in zip(recorded, lanes, strict=True)
        ],
        "compute": {
            "executor": info.executor,
            "job": job_ctx.to_json(),
            "host": "nebius-job" if job_ctx.is_cloud_job else "local",
            "machine": {"node": _short_host(), "python": platform.python_version()},
        },
        "code": code_facts(),
        "search_manifest": "search/manifest.json",
    }
    write_json_atomic(out / SUMMARY, summary)
    return SliceOutcome(out_dir=out, summary=summary, search=search)


def _short_host() -> str:
    """A non-identifying host label: the machine class, never its hostname."""
    return f"{platform.system().lower()}-{platform.machine()}"


def read_slice(path: str | Path) -> dict[str, Any]:
    """Load a ``slice.json`` (a directory or the file itself) and check its schema."""
    p = Path(path)
    if p.is_dir():
        p = p / SUMMARY
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != SLICE_SCHEMA:
        raise ValueError(f"{p}: not an {SLICE_SCHEMA} summary")
    return data
