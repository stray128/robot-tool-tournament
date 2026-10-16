"""One search invocation as the ``rtt search`` command and the Nebius search job run it.

:func:`run_search_command` builds the training-suite evaluator of the chosen design space
(``box-v1``: :class:`~rtt.search.sim_adapter.SimEvaluator` on ``train_v1``; ``freeform-v2``:
:class:`~rtt.search.sim_adapter.SimEvaluatorV2` on ``train_v2`` under ``task_v2``) on the
chosen executor and, for the LLM arms, the pre-registered proposer client with its
response cache and spend ledger inside the output directory, then calls
:func:`rtt.search.runner.run_search`. The LLM arms always run under a spend cap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rtt.llm.cache import CacheMode

__all__ = [
    "DEFAULT_SPEND_CAP_USD",
    "SPACE_ALIASES",
    "SearchRequest",
    "parse_seeds",
    "parse_space",
    "run_search_command",
]

#: Default cap on live model spend for one ``rtt search`` invocation.
DEFAULT_SPEND_CAP_USD: Final = 2.0
#: ``--space`` values: the design-space names and their short aliases.
SPACE_ALIASES: Final[dict[str, str]] = {
    "box-v1": "box-v1",
    "v1": "box-v1",
    "freeform-v2": "freeform-v2",
    "v2": "freeform-v2",
}


def parse_space(name: str | None) -> str:
    """``box-v1`` / ``freeform-v2`` from a ``--space`` value (``v1``/``v2`` accepted; default v1)."""
    if name is None:
        return "box-v1"
    try:
        return SPACE_ALIASES[str(name).strip().lower()]
    except KeyError:
        raise ValueError(f"--space: expected one of {sorted(SPACE_ALIASES)}, got {name!r}") from None


def parse_seeds(spec: str | None, runs: int) -> list[int]:
    """Run seeds from ``spec``: a YAML file (list, or ``seeds:`` list), ``a,b,c``, or ``None``.

    ``None`` gives ``1..runs``. Only the first ``runs`` seeds are used.
    """
    from rtt.search.runner import load_seeds

    if spec is None:
        seeds = list(range(1, runs + 1))
    elif Path(spec).is_file():
        seeds = load_seeds(spec)
    else:
        try:
            seeds = [int(s) for s in spec.split(",") if s.strip()]
        except ValueError as exc:
            raise ValueError(
                f"--seeds: expected a YAML file or comma-separated integers, got {spec!r}"
            ) from exc
    if len(seeds) < runs:
        raise ValueError(f"{runs} runs need {runs} seeds, got {len(seeds)}")
    return seeds[:runs]


@dataclass(frozen=True)
class SearchRequest:
    """Arguments of one search invocation (see ``rtt search --help``)."""

    arm: str
    runs: int
    budget: int
    out: Path
    seeds: Sequence[int]
    executor: str = "cpu"
    workers: int | None = None
    spend_cap_usd: float = DEFAULT_SPEND_CAP_USD
    cache_mode: CacheMode = "readwrite"
    cache_dir: Path | None = None
    space: str = "box-v1"


def run_search_command(req: SearchRequest) -> Any:
    """Run the search described by ``req``; returns the :class:`~rtt.search.runner.SearchOutcome`."""
    from rtt.search.nemotron import make_proposer_client
    from rtt.search.runner import ARMS, run_search
    from rtt.search.sim_adapter import SimEvaluator, SimEvaluatorV2
    from rtt.sim.exec_cpu import CpuExecutor

    if req.arm not in ARMS:
        raise ValueError(f"unknown arm {req.arm!r}; choose from {sorted(ARMS)}")
    space = parse_space(req.space)
    out = Path(req.out)
    client = None
    if ARMS[req.arm].uses_llm:
        client = make_proposer_client(
            req.cache_dir or out / "llm_cache",
            cache_mode=req.cache_mode,
            ledger_path=out / "llm_ledger.jsonl",
            budget_usd=req.spend_cap_usd,
        )
    pool = CpuExecutor(req.workers) if req.executor == "cpu" else None
    try:
        executor = pool if pool is not None else req.executor
        if space == "freeform-v2":
            if req.executor != "cpu":
                raise ValueError("freeform-v2 runs on the CPU executor only (MuJoCo Warp has no v2 path)")
            evaluator: Any = SimEvaluatorV2(executor=executor, workers=req.workers)
        else:
            evaluator = SimEvaluator(executor=executor, workers=req.workers)
        return run_search(req.arm, req.runs, req.budget, req.seeds, evaluator, out, client, space=space)
    finally:
        if pool is not None:
            pool.close()
