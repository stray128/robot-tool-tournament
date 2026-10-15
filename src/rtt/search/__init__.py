"""Design search: arms, budgets, the lockstep runner and top-1 selection.

* :mod:`rtt.search.base` - evaluator contract, scores, history, budget ledger, ask/tell.
* :mod:`rtt.search.random_search` - arm A (scrambled Sobol) and the shared round 0.
* :mod:`rtt.search.nemotron` - arm C (Nemotron with feedback) and the LLM proposal machinery.
* :mod:`rtt.search.zeroshot` - arm Z (Nemotron without feedback).
* :mod:`rtt.search.diversity` - the 0.08 design-distance check.
* :mod:`rtt.search.runner` - :func:`run_search`, R independent runs in lockstep.
* :mod:`rtt.search.select` - freeze each run's top-1 design.
* :mod:`rtt.search.sim_adapter` - evaluator backed by :mod:`rtt.sim` (imports MuJoCo).
* :mod:`rtt.search.fakes` - synthetic evaluator and scripted model for tests and dry runs.
"""

from rtt.search.base import (
    BudgetError,
    BudgetLedger,
    ConditionInfo,
    DesignScore,
    EpisodeOutcome,
    Evaluated,
    Evaluator,
    EvaluatorInfo,
    Proposal,
    Rejection,
    RunContext,
    RunHistory,
    SearchStrategy,
)
from rtt.search.nemotron import NemotronProposer, ProposerConfig, make_proposer_client
from rtt.search.random_search import DesignSampler, RandomSearch, round0_designs
from rtt.search.runner import ARMS, SearchConfig, SearchOutcome, load_seeds, run_search
from rtt.search.select import FrozenDesign, freeze_top1, read_frozen, verify_frozen
from rtt.search.zeroshot import ZeroShotProposer

__all__ = [
    "ARMS",
    "BudgetError",
    "BudgetLedger",
    "ConditionInfo",
    "DesignSampler",
    "DesignScore",
    "EpisodeOutcome",
    "Evaluated",
    "Evaluator",
    "EvaluatorInfo",
    "FrozenDesign",
    "NemotronProposer",
    "Proposal",
    "ProposerConfig",
    "RandomSearch",
    "Rejection",
    "RunContext",
    "RunHistory",
    "SearchConfig",
    "SearchOutcome",
    "SearchStrategy",
    "ZeroShotProposer",
    "freeze_top1",
    "load_seeds",
    "make_proposer_client",
    "read_frozen",
    "round0_designs",
    "run_search",
    "verify_frozen",
]
