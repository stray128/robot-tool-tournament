"""Lockstep runner: R independent runs of one arm, equal budgets, manifests and frozen picks.

:func:`run_search` is the entry point the ``rtt search`` command wraps. For one arm
it runs ``R`` independent runs side by side, round by round:

1. Round 0 for run ``r`` is :func:`~rtt.search.random_search.round0_designs` with
   run seed ``s_r``: the manual entrants plus a Sobol fill, byte-identical in every
   arm that uses the same seeds.
2. Each later round, every run's strategy proposes its designs from *its own*
   history only (LLM runs in parallel threads; no state is shared between runs).
3. All runs' designs go to the evaluator in one batch (identical designs are
   simulated once; the evaluator is deterministic per design, so sharing a score
   carries no information between runs). Each run is charged exactly its round
   size on its :class:`~rtt.search.base.BudgetLedger` and told its own results.
4. After the last round each run's top-1 design is frozen (SHA-256 + timestamp).

The whole search runs inside a :class:`~rtt.store.vault.VaultGuard`, so nothing it
does can open the held-out seed. Outputs under ``out_dir``: ``manifest.json``,
``runs/<run_id>/{history,rejections,llm_calls}.jsonl``, ``results/*.parquet`` and
``frozen/<run_id>.json``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import yaml

from rtt.design.schema import FingerDesign, GripperTask, load_entrants
from rtt.design.validate import validate
from rtt.llm.client import LLMClient
from rtt.llm.providers import PROVIDER
from rtt.search.base import (
    DEFAULT_LLM_CALL_CAP,
    DEFAULT_ROUND_SIZE,
    BudgetError,
    BudgetLedger,
    DesignScore,
    Evaluated,
    Evaluator,
    Proposal,
    RunContext,
    SearchStrategy,
)
from rtt.search.diversity import EXPLORE_MIN_DISTANCE_V2, MIN_DISTANCE_V2, diversity_stats
from rtt.search.nemotron import LLMProposer, NemotronProposer, ProposerConfig, prompt_spec_hashes
from rtt.search.random_search import RandomSearch, round0_designs
from rtt.search.select import FrozenDesign, freeze_top1, write_frozen
from rtt.search.space import V1, V2, DesignSpace, get_space, space_of
from rtt.search.zeroshot import ZeroShotProposer
from rtt.store import runs as store
from rtt.store.manifest import JobContext, LLMUsage, code_facts, write_manifest
from rtt.store.vault import VaultGuard

__all__ = [
    "ARMS",
    "ARMS_V2",
    "DEFAULT_ENTRANTS",
    "DEFAULT_ENTRANTS_V2",
    "RunOutcome",
    "SearchConfig",
    "SearchOutcome",
    "load_seeds",
    "make_strategy",
    "run_search",
]

ARMS: Final[dict[str, type[SearchStrategy]]] = {
    "random": RandomSearch,
    "nemotron": NemotronProposer,
    "zeroshot": ZeroShotProposer,
}
DEFAULT_ENTRANTS: Final = Path(__file__).resolve().parents[3] / "configs" / "entrants.yaml"
#: v2 entrants (track G): stock blade, flat pad, configurator imprint, swept imprint.
DEFAULT_ENTRANTS_V2: Final = Path(__file__).resolve().parents[3] / "configs" / "entrants_v2.yaml"


def _arms_v2() -> dict[str, type[SearchStrategy]]:
    from rtt.search.nemotron_v2 import NemotronProposerV2
    from rtt.search.zeroshot import ZeroShotProposerV2

    return {"random": RandomSearch, "nemotron": NemotronProposerV2, "zeroshot": ZeroShotProposerV2}


#: The same three arms on the v2 design space (resolved lazily: see :func:`make_strategy`).
ARMS_V2: Final[tuple[str, ...]] = tuple(ARMS)


@dataclass(frozen=True)
class SearchConfig:
    """Protocol knobs shared by every arm (frozen in PREREG).

    ``shared_diversity`` (v2 only): every arm's designs, including arm A, the round-0 Sobol
    fill and ``fallback_random``, pass the same duplicate and diversity screen at the
    proposer's ``min_distance`` (spec section 13, step 6). v1 ignores it.
    """

    round0_size: int = DEFAULT_ROUND_SIZE
    round_size: int = DEFAULT_ROUND_SIZE
    llm_call_cap: int = DEFAULT_LLM_CALL_CAP
    llm_concurrency: int = 8
    reuse_scores: bool = True
    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    shared_diversity: bool = True

    @classmethod
    def for_space(cls, space: str) -> SearchConfig:
        """The pre-registered defaults of ``space`` (v2: the rule-set diversity and explore bars)."""
        if space == V1:
            return cls()
        return cls(
            proposer=ProposerConfig(
                min_distance=MIN_DISTANCE_V2, explore_min_distance=EXPLORE_MIN_DISTANCE_V2
            )
        )

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["proposer"] = self.proposer.to_json()
        return data


@dataclass
class RunOutcome:
    """One finished run."""

    ctx: RunContext
    frozen: FrozenDesign

    @property
    def fingerprint(self) -> str:
        return self.ctx.fingerprint()


@dataclass
class SearchOutcome:
    """Everything :func:`run_search` produced."""

    arm: str
    out_dir: Path
    runs: list[RunOutcome]
    manifest: dict[str, Any]
    paths: dict[str, Path]


def load_seeds(path: str | Path) -> list[int]:
    """Run seeds from YAML: a list, or a mapping with a ``seeds`` list."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    seeds = data.get("seeds") if isinstance(data, Mapping) else data
    if not isinstance(seeds, list) or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise TypeError(f"{path}: expected a list of integer seeds")
    return list(seeds)


def make_strategy(
    arm: str, ctx: RunContext, llm_client: LLMClient | None, config: SearchConfig
) -> SearchStrategy:
    """Instantiate ``arm``'s strategy for one run (the v2 classes when ``ctx.space`` is v2)."""
    cls = ARMS[arm] if ctx.space.name == V1 else _arms_v2()[arm]
    if issubclass(cls, LLMProposer):
        if llm_client is None:
            raise ValueError(f"arm {arm!r} needs an llm_client")
        return cls(ctx, llm_client, config.proposer)
    return cls(ctx)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _ScoreBook:
    """Evaluates batches, simulating each distinct geometry once (optionally across rounds)."""

    def __init__(self, evaluator: Evaluator, reuse: bool) -> None:
        self.evaluator = evaluator
        self.reuse = reuse
        self.scores: dict[str, DesignScore] = {}
        self.simulated = 0
        self.episodes = 0
        self.wall_s = 0.0

    def evaluate(self, batches: Sequence[Sequence[Any]]) -> list[list[DesignScore]]:
        known = self.scores if self.reuse else {}
        todo: dict[str, Any] = {}
        for designs in batches:
            for design in designs:
                key = _design_key(design)
                if key not in known and key not in todo:
                    todo[key] = design
        fresh: dict[str, DesignScore] = {}
        if todo:
            started = time.perf_counter()
            scores = self.evaluator.evaluate(list(todo.values()))
            self.wall_s += time.perf_counter() - started
            if len(scores) != len(todo):
                raise RuntimeError(f"evaluator returned {len(scores)} scores for {len(todo)} designs")
            for key, score in zip(todo, scores, strict=True):
                if score.design_id != key:
                    raise RuntimeError(f"evaluator returned a score for {score.design_id}, expected {key}")
                fresh[key] = score
            self.simulated += len(todo)
            self.episodes += sum(s.n_episodes for s in scores)
        if self.reuse:
            self.scores.update(fresh)
        lookup = {**known, **fresh}
        return [[lookup[_design_key(d)] for d in designs] for designs in batches]


def _design_key(design: Any) -> str:
    if isinstance(design, FingerDesign):
        return design.design_id()
    return space_of(design).design_id(design)


def _check_round(ctx: RunContext, round_index: int, proposals: Sequence[Proposal]) -> None:
    expected = ctx.budget.round_sizes[round_index]
    if len(proposals) != expected:
        raise BudgetError(
            f"{ctx.run_id} round {round_index}: {len(proposals)} proposals, expected {expected}"
        )
    for p in proposals:
        if ctx.space.name == V1:
            ok = validate(p.design, ctx.gripper_task).ok
        else:
            ok = ctx.space.is_design(p.design) and ctx.space.validate(p.design, ctx.gripper_task).ok
        if not ok:
            raise BudgetError(f"{ctx.run_id} round {round_index}: invalid design reached the simulator")


def _run_identity(ctx: RunContext) -> dict[str, Any]:
    return {"arm": ctx.arm, "run_id": ctx.run_id, "run_index": ctx.run_index, "run_seed": ctx.run_seed}


def _model_facts(client: LLMClient | None) -> dict[str, Any] | None:
    if client is None:
        return None
    return {
        "model_id": client.model,
        "provider": PROVIDER,
        "temperature": client.temperature,
        "top_p": client.top_p,
        "max_tokens": client.max_tokens,
        "extra_body": client.extra_body,
        "cache_mode": client.cache.mode if client.cache is not None else None,
    }


def run_search(
    arm: str,
    runs: int,
    budget: int,
    seeds: Sequence[int],
    evaluator: Evaluator,
    out_dir: str | Path,
    llm_client: LLMClient | None = None,
    *,
    config: SearchConfig | None = None,
    entrants: Sequence[Any] | None = None,
    gripper_task: GripperTask | Any | None = None,
    job: JobContext | None = None,
    protected_paths: Iterable[str | Path] = (),
    space: str | DesignSpace | None = None,
) -> SearchOutcome:
    """Run ``runs`` independent runs of ``arm`` in lockstep, each simulating exactly ``budget`` designs.

    Args:
        arm: ``random``, ``nemotron`` or ``zeroshot``.
        runs: Number of independent runs R.
        budget: Valid designs simulated per run (B, 40 in the protocol).
        seeds: One seed per run (the first ``runs`` are used); run ``r`` of every
            arm with the same seeds shares round 0.
        evaluator: Training-suite evaluator (:class:`~rtt.search.base.Evaluator`).
        out_dir: Output directory; must not already hold a manifest.
        llm_client: Required for the LLM arms. Its cache decides live vs replay.
        config: Protocol knobs (round sizes, LLM-call cap, proposer settings).
        entrants: Manual entrants for round 0 (default ``configs/entrants.yaml``; v2:
            ``configs/entrants_v2.yaml`` through track G's loader).
        gripper_task: Validator constants (default: the evaluator's). In v2, the
            :class:`~rtt.search.space_v2.V2Context` with the training parts.
        job: Job facts for the manifest (default :meth:`JobContext.from_env`).
        protected_paths: Extra paths the vault guard must keep unreadable.
        space: ``box-v1`` or ``freeform-v2`` (default: the evaluator's design space). A run
            never mixes spaces: the evaluator, the entrants and every proposal must agree.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    if runs < 1:
        raise ValueError("runs must be >= 1")
    run_seeds = [int(s) for s in list(seeds)[:runs]]
    if len(run_seeds) < runs:
        raise ValueError(f"{runs} runs need {runs} seeds, got {len(run_seeds)}")
    if len(set(run_seeds)) != len(run_seeds):
        raise ValueError("run seeds must be distinct")
    out = Path(out_dir)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already holds a search manifest; use a fresh directory")
    info = evaluator.describe()
    design_space = get_space(space) if isinstance(space, str) else space or get_space(info.design_space)
    if design_space.name != info.design_space:
        raise ValueError(f"the evaluator scores {info.design_space} designs, not {design_space.name}")
    config = config or SearchConfig.for_space(design_space.name)
    task = gripper_task or info.gripper_task
    if entrants is not None:
        entrant_designs = list(entrants)
    elif design_space.name == V1:
        entrant_designs = load_entrants(DEFAULT_ENTRANTS)
    else:
        entrant_designs = _load_entrants_v2(DEFAULT_ENTRANTS_V2)
    if any(not design_space.is_design(d) for d in entrant_designs):
        raise ValueError(f"every entrant must be a {design_space.name} design")
    job = job or JobContext.from_env()
    shared = config.proposer.min_distance if design_space.name != V1 and config.shared_diversity else None

    contexts = [
        RunContext(
            arm=arm,
            run_index=i,
            run_seed=seed,
            gripper_task=task,
            info=info,
            budget=BudgetLedger(budget, config.round0_size, config.round_size, config.llm_call_cap),
            space=design_space,
            shared_min_distance=shared,
        )
        for i, seed in enumerate(run_seeds)
    ]
    strategies = [make_strategy(arm, ctx, llm_client, config) for ctx in contexts]
    plan = contexts[0].budget.round_sizes
    book = _ScoreBook(evaluator, config.reuse_scores)
    started_at, t0 = _utc_now(), time.perf_counter()
    base_manifest: dict[str, Any] = {
        "kind": "search",
        "arm": arm,
        "runs": runs,
        "budget": budget,
        "round_sizes": list(plan),
        "seeds": run_seeds,
        "started_at": started_at,
        "job": job.to_json(),
        "code": code_facts(),
        "executor": info.executor,
        "evaluator": {
            "executor": info.executor,
            "task_hash": info.task_hash,
            "suite_hash": info.suite_hash,
            "n_conditions": len(info.conditions),
            "families": list(info.families),
            "controller_settings": list(info.controller_settings),
            "seeds_per_condition": info.seeds_per_condition,
            "episodes_per_design": info.episodes_per_design,
        },
        "gripper_task": asdict(task) if design_space.name == V1 else _context_json(task),
        "entrants": [
            {"name": design_space.label(d), "design_id": design_space.design_id(d)} for d in entrant_designs
        ],
        "config": config.to_json(),
        "model": _model_facts(llm_client) if strategies[0].uses_llm else None,
        "provider": PROVIDER if strategies[0].uses_llm else None,
        "prompt_hashes": _prompt_hashes(design_space.name) if strategies[0].uses_llm else None,
        "design_space": design_space.name,
    }
    write_manifest(out / "manifest.json", {**base_manifest, "status": "running"})

    written = {ctx.run_id: {"rejections": 0, "calls": 0} for ctx in contexts}

    def flush(ctx: RunContext, new_items: Sequence[Evaluated]) -> None:
        run_dir = out / "runs" / ctx.run_id
        store.append_jsonl(run_dir / "history.jsonl", (item.to_json() for item in new_items))
        marks = written[ctx.run_id]
        store.append_jsonl(
            run_dir / "rejections.jsonl", (r.to_json() for r in ctx.rejections[marks["rejections"] :])
        )
        store.append_jsonl(
            run_dir / "llm_calls.jsonl", (c.to_json() for c in ctx.llm_calls[marks["calls"] :])
        )
        marks["rejections"], marks["calls"] = len(ctx.rejections), len(ctx.llm_calls)

    try:
        with VaultGuard(protected_paths):
            uses_llm = strategies[0].uses_llm
            workers = max(1, min(config.llm_concurrency, runs)) if uses_llm else 1
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for round_index, size in enumerate(plan):
                    if round_index == 0:
                        proposals = [
                            round0_designs(entrant_designs, c.run_seed, size, task)
                            if design_space.name == V1
                            else round0_designs(
                                entrant_designs,
                                c.run_seed,
                                size,
                                task,
                                space=design_space,
                                min_distance=c.shared_min_distance,
                            )
                            for c in contexts
                        ]
                    else:
                        proposals = list(pool.map(lambda s, r=round_index, n=size: s.ask(r, n), strategies))
                    for ctx, props in zip(contexts, proposals, strict=True):
                        _check_round(ctx, round_index, props)
                    scores = book.evaluate([[p.design for p in props] for props in proposals])
                    for ctx, strategy, props, run_scores in zip(
                        contexts, strategies, proposals, scores, strict=True
                    ):
                        ctx.budget.charge(round_index, len(props))
                        new_items = [
                            ctx.history.add(round_index, p, s) for p, s in zip(props, run_scores, strict=True)
                        ]
                        strategy.tell(new_items)
                        flush(ctx, new_items)
            for ctx in contexts:
                if not ctx.budget.complete or len(ctx.history) != budget:
                    raise BudgetError(f"{ctx.run_id} simulated {len(ctx.history)} designs, budget {budget}")
            outcomes = [
                RunOutcome(ctx, freeze_top1(arm, ctx.run_index, ctx.run_seed, ctx.history, info))
                for ctx in contexts
            ]
            paths: dict[str, Path] = {}
            for outcome in outcomes:
                paths[f"frozen:{outcome.ctx.run_id}"] = write_frozen(outcome.frozen, out / "frozen")
            tables = store.write_tables if design_space.name == V1 else _write_tables_v2
            paths.update(
                tables(
                    out / "results",
                    [_run_identity(c) for c in contexts],
                    [[item.to_json() for item in c.history] for c in contexts],
                    [[r.to_json() for r in c.rejections] for c in contexts],
                    info.families,
                )
            )
    except BaseException as exc:
        write_manifest(
            out / "manifest.json",
            {
                **base_manifest,
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "finished_at": _utc_now(),
                "wall_s": round(time.perf_counter() - t0, 3),
            },
        )
        raise

    per_run = []
    usages = []
    for outcome, strategy in zip(outcomes, strategies, strict=True):
        ctx = outcome.ctx
        usage = LLMUsage.from_records(c.to_json() for c in ctx.llm_calls)
        usages.append(usage)
        reasons: dict[str, int] = {}
        for r in ctx.rejections:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
        best = ctx.history.best()
        per_run.append(
            {
                "run_id": ctx.run_id,
                "run_index": ctx.run_index,
                "run_seed": ctx.run_seed,
                "budget": ctx.budget.to_json(),
                "llm": usage.to_json(),
                "rejections": len(ctx.rejections),
                "rejections_by_reason": dict(sorted(reasons.items())),
                "fallback_random": ctx.budget.fallbacks,
                "llm_disabled": strategy.disabled_reason if isinstance(strategy, LLMProposer) else None,
                "best": {
                    "label": best.label,
                    "design_id": best.score.design_id,
                    "success": best.score.success,
                },
                "best_so_far": ctx.history.best_so_far(),
                "diversity": diversity_stats(
                    ctx.history.designs(), space=None if design_space.name == V1 else design_space
                ),
                "frozen_sha256": outcome.frozen.content_sha256,
                "frozen_at": outcome.frozen.frozen_at,
                "fingerprint": outcome.fingerprint,
            }
        )
        if design_space.name != V1:
            per_run[-1]["sampler"] = _sampler_stats(strategy)
            per_run[-1]["lift_best_so_far"] = _lift_best_so_far(ctx)
    manifest = {
        **base_manifest,
        "status": "complete",
        "finished_at": _utc_now(),
        "wall_s": round(time.perf_counter() - t0, 3),
        "sim": {
            "designs_charged": sum(c.budget.spent for c in contexts),
            "designs_simulated": book.simulated,
            "episodes_simulated": book.episodes,
            "evaluator_wall_s": round(book.wall_s, 3),
        },
        "llm": LLMUsage.combine(usages).to_json(),
        "rejections": sum(len(c.rejections) for c in contexts),
        "fallback_random": sum(c.budget.fallbacks for c in contexts),
        "per_run": per_run,
    }
    paths["manifest"] = write_manifest(out / "manifest.json", manifest)
    return SearchOutcome(arm=arm, out_dir=out, runs=outcomes, manifest=manifest, paths=paths)


# --------------------------------------------------------------------------- v2 helpers


def _load_entrants_v2(path: Path) -> list[Any]:
    """Track G's v2 entrants (``configs/entrants_v2.yaml``); a clear error until G lands."""
    try:
        from rtt.design.freeform.baselines import load_entrants as load_v2
    except ImportError as exc:
        raise FileNotFoundError(
            "v2 entrants come from track G (rtt.design.freeform.baselines.load_entrants and "
            f"{path.name}); pass entrants= explicitly until they land"
        ) from exc
    return list(load_v2(path))


def _context_json(task: Any) -> Any:
    to_json = getattr(task, "to_json", None)
    return to_json() if callable(to_json) else repr(task)


def _prompt_hashes(space: str) -> dict[str, str]:
    if space == V1:
        return prompt_spec_hashes()
    from rtt.search.nemotron_v2 import prompt_spec_hashes_v2

    return prompt_spec_hashes_v2()


def _sampler_stats(strategy: SearchStrategy) -> dict[str, Any] | None:
    """The random stream's valid rate and rejections by reason (arm A, or an LLM arm's fallback)."""
    sampler = getattr(strategy, "sampler", None) or getattr(strategy, "fallback", None)
    stats = getattr(sampler, "stats", None)
    return stats() if callable(stats) else None


def _lift_best_so_far(ctx: RunContext) -> list[float]:
    best, out = 0.0, []
    for item in ctx.history:
        lift = item.score.lift_success
        best = max(best, item.score.success if lift is None else lift)
        out.append(best)
    return out


def _design_row_v2(
    run: Mapping[str, Any], evaluated: Mapping[str, Any], families: Sequence[str]
) -> dict[str, Any]:
    """:func:`rtt.store.runs.design_row` for v2: recipe columns (``pad``, ``tip``, the 7 numbers)."""
    from rtt.search.space_v2 import SCORED_PARAMS

    proposal, score = evaluated["proposal"], evaluated["score"]
    design = proposal["design"]
    point = design.get("point") if "baseline" in design else design
    row: dict[str, Any] = {
        "arm": run["arm"],
        "run_id": run["run_id"],
        "run_index": run["run_index"],
        "run_seed": run["run_seed"],
        "round": evaluated["round"],
        "eval_index": evaluated["index"],
        "label": evaluated["label"],
        "design_id": proposal["design_id"],
        "design_space": proposal.get("design_space", V2),
        "origin": proposal["origin"],
        "operator": proposal["operator"],
        "island": proposal["island"],
        "slot": proposal["slot"],
        "parents": ",".join(proposal["parents"]),
        "via": proposal["via"],
        "fallback_reason": proposal["fallback_reason"],
        "request_hash": proposal["request_hash"],
        "hypothesis": proposal["hypothesis"],
        "name": design.get("name"),
        "baseline": design.get("baseline"),
    }
    for key in SCORED_PARAMS:
        row[key] = point.get(key) if point else None
    row.update(
        {
            "mass_g": score["mass_g"],
            "success": score["success"],
            "lift_success": score.get("lift_success"),
            "hold_fraction": score["hold_fraction"],
            "n_episodes": score["n_episodes"],
            "jam_on_close": score.get("jam_on_close"),
            "off_centre": score.get("off_centre"),
            "max_pieces": score.get("max_pieces"),
        }
    )
    for family in families:
        row[f"success_{family}"] = score["per_family"].get(family)
    for place, value in sorted((score.get("per_placement") or {}).items()):
        row[f"success_{place}"] = value
    return row


def _write_tables_v2(
    out_dir: str | Path,
    runs: Sequence[Mapping[str, Any]],
    histories: Sequence[Sequence[Mapping[str, Any]]],
    rejections: Sequence[Sequence[Mapping[str, Any]]],
    families: Sequence[str],
) -> dict[str, Path]:
    """:func:`rtt.store.runs.write_tables` with v2 design rows (same files, same episode and rejection rows)."""
    results = Path(out_dir)
    designs: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for run, history, run_rejections in zip(runs, histories, rejections, strict=True):
        for item in history:
            designs.append(_design_row_v2(run, item, families))
            episodes.extend(store.episode_rows(run, item))
        rejected.extend(store.rejection_row(run, r) for r in run_rejections)
    paths = {
        "designs_parquet": store.write_parquet(results / "designs.parquet", designs),
        "designs_jsonl": store.write_jsonl(results / "designs.jsonl", designs),
    }
    if episodes:
        paths["episodes_parquet"] = store.write_parquet(results / "episodes.parquet", episodes)
    if rejected:
        paths["rejections_parquet"] = store.write_parquet(results / "rejections.parquet", rejected)
    return paths
