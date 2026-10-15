"""Search harness core: evaluator contract, scores, run history, budget ledger and ask/tell.

Every search arm (random, Nemotron, zero-shot) is a :class:`SearchStrategy` with an
``ask(round, n) -> proposals`` / ``tell(evaluated)`` interface. The runner owns the
experiment protocol around it:

* **Budget.** A run simulates exactly ``B`` valid designs (default 40): a shared
  round 0 of 8, then rounds of 8. :class:`BudgetLedger` refuses anything else.
  Invalid proposals never reach the simulator, so they cost no budget; they are
  recorded as :class:`Rejection` rows instead. LLM arms also have a per-run cap on
  model calls (default 16).
* **Evaluator contract.** Scoring goes through the :class:`Evaluator` protocol:
  ``evaluate(designs) -> list[DesignScore]`` plus ``describe() -> EvaluatorInfo``.
  The simulator adapter (:mod:`rtt.search.sim_adapter`) and the synthetic
  :class:`rtt.search.fakes.FakeEvaluator` both implement it. Results must depend
  only on the design (common random numbers), never on batch composition.
* **Scores.** :meth:`DesignScore.from_episodes` is the one aggregation rule: the
  score is the success rate at the best controller setting ``k``; per-family,
  per-condition and failure-mode tallies are taken at that same ``k``; ties break
  on mean hold fraction, then lower finger mass (:meth:`DesignScore.rank_key`).

Seeds are derived with :func:`derive_seed` (SHA-256 of labelled parts), so every
random stream is reproducible from the run seed alone.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

from rtt.design.schema import GEOMETRY_PARAMS, FingerDesign, GripperTask
from rtt.llm.cache import canonical_json

__all__ = [
    "DEFAULT_BUDGET",
    "DEFAULT_LLM_CALL_CAP",
    "DEFAULT_ROUND_SIZE",
    "FAILURE_MODES",
    "BudgetError",
    "BudgetLedger",
    "ConditionInfo",
    "DesignScore",
    "EpisodeOutcome",
    "Evaluated",
    "Evaluator",
    "EvaluatorInfo",
    "LLMCallRecord",
    "Proposal",
    "Rejection",
    "RunContext",
    "RunHistory",
    "SearchStrategy",
    "derive_seed",
    "design_params",
    "round_plan",
]

DEFAULT_BUDGET: Final = 40
DEFAULT_ROUND_SIZE: Final = 8
DEFAULT_LLM_CALL_CAP: Final = 16

#: Failure labels produced by the simulator (``rtt.sim.score.FAILURE_MODES``), in its order.
FAILURE_MODES: Final[tuple[str, ...]] = (
    "missed_grasp",
    "slip_during_lift",
    "ejected_on_close",
    "unstable_hold",
    "finger_env_collision",
)


def derive_seed(base: int, *labels: object) -> int:
    """A 63-bit seed derived from ``base`` and ``labels`` (stable across platforms and runs)."""
    text = ":".join([str(int(base)), *(str(label) for label in labels)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") >> 1


def design_params(design: FingerDesign) -> dict[str, float | str]:
    """The 10 geometry parameters of ``design`` in canonical order."""
    return design.geometry_params()


def round_plan(
    budget: int = DEFAULT_BUDGET, round0_size: int = DEFAULT_ROUND_SIZE, round_size: int = DEFAULT_ROUND_SIZE
) -> tuple[int, ...]:
    """Designs simulated per round: ``(round0_size, round_size, ...)`` summing to ``budget``."""
    if round0_size <= 0 or round_size <= 0:
        raise ValueError("round sizes must be positive")
    rest = budget - round0_size
    if rest < 0 or rest % round_size:
        raise ValueError(
            f"budget {budget} must be round0 ({round0_size}) plus a whole number of rounds of {round_size}"
        )
    return (round0_size, *([round_size] * (rest // round_size)))


# --------------------------------------------------------------------------- evaluator contract


@dataclass(frozen=True)
class ConditionInfo:
    """One training condition, as the proposer may see it (never a held-out condition)."""

    condition_id: str
    family: str
    label: str
    grip_width_mm: float


@dataclass(frozen=True)
class EpisodeOutcome:
    """One simulated grasp of one design: condition x controller setting x jitter seed."""

    condition_id: str
    family: str
    controller_setting: int
    seed: int
    success: bool
    failure_mode: str | None
    max_lift_m: float
    hold_frac: float
    """Longest continuous hold above the success height / the required hold, in [0, 1]."""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> EpisodeOutcome:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class EvaluatorInfo:
    """What a search arm may know about the evaluator: training conditions and task facts.

    ``task_hash`` and ``suite_hash`` identify the frozen task and training suite in
    manifests. Physical facts are used to describe the task in prompts.
    """

    executor: str
    conditions: tuple[ConditionInfo, ...]
    controller_settings: tuple[float, ...]
    """Grasp-height fraction of each controller setting ``k``: where the fingertips stop
    within the band they can reach, from the lowest tip height that keeps the palm
    ``palm_clearance_m`` above the object's top up to the top (``k x`` the object
    height for fingers that reach that far)."""
    seeds_per_condition: int
    """Jitter seeds per condition and setting (the seeds themselves are in the suite hash)."""
    gripper_task: GripperTask
    task_hash: str
    suite_hash: str
    grip_force_n: float = 40.0
    success_lift_m: float = 0.10
    success_hold_s: float = 2.0
    max_time_s: float = 12.0
    friction: float = 0.8
    object_density_kg_m3: float = 700.0
    finger_density_kg_m3: float = 1240.0
    jitter_pos_mm: float = 2.0
    jitter_yaw_deg: float = 3.0
    jitter_mu_frac: float = 0.10
    palm_clearance_m: float = 0.01
    """Gap the controller keeps between the palm's underside and the object's top."""

    @property
    def families(self) -> tuple[str, ...]:
        """Object families in first-appearance order."""
        return tuple(dict.fromkeys(c.family for c in self.conditions))

    @property
    def episodes_per_design(self) -> int:
        return len(self.conditions) * len(self.controller_settings) * self.seeds_per_condition

    def condition(self, condition_id: str) -> ConditionInfo:
        for c in self.conditions:
            if c.condition_id == condition_id:
                return c
        raise KeyError(condition_id)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["gripper_task"] = asdict(self.gripper_task)
        return data


@dataclass(frozen=True)
class DesignScore:
    """Training result of one design. Build it with :meth:`from_episodes`."""

    design_id: str
    mass_g: float
    success: float
    """Mean success at the best controller setting (the design's score)."""
    best_k: int
    success_by_k: tuple[float, ...]
    hold_fraction: float
    """Mean hold fraction at ``best_k`` (first tie-breaker)."""
    per_family: Mapping[str, float]
    per_condition: Mapping[str, float]
    failure_modes: Mapping[str, int]
    """Failure-mode counts over the episodes at ``best_k``."""
    condition_failures: Mapping[str, Mapping[str, int]]
    """Failure-mode counts per condition at ``best_k`` (conditions with no failure omitted)."""
    n_episodes: int
    episodes: tuple[EpisodeOutcome, ...] = field(default=(), repr=False)

    @classmethod
    def from_episodes(
        cls, design_id: str, mass_g: float, episodes: Sequence[EpisodeOutcome], n_settings: int
    ) -> DesignScore:
        """Aggregate episodes: best ``k`` by (success, hold fraction), lowest ``k`` on a full tie."""
        if not episodes:
            raise ValueError(f"design {design_id}: no episodes to score")
        by_k: dict[int, list[EpisodeOutcome]] = {k: [] for k in range(n_settings)}
        for ep in episodes:
            if ep.controller_setting not in by_k:
                raise ValueError(f"controller setting {ep.controller_setting} outside 0..{n_settings - 1}")
            by_k[ep.controller_setting].append(ep)

        def mean(values: Iterable[float]) -> float:
            items = list(values)
            return sum(items) / len(items) if items else 0.0

        success_by_k = tuple(mean(float(e.success) for e in by_k[k]) for k in range(n_settings))
        hold_by_k = tuple(mean(e.hold_frac for e in by_k[k]) for k in range(n_settings))
        populated = [k for k in range(n_settings) if by_k[k]]
        best_k = max(populated, key=lambda k: (success_by_k[k], hold_by_k[k], -k))
        best = by_k[best_k]

        per_family: dict[str, list[bool]] = {}
        per_condition: dict[str, list[bool]] = {}
        condition_failures: dict[str, Counter[str]] = {}
        for ep in best:
            per_family.setdefault(ep.family, []).append(ep.success)
            per_condition.setdefault(ep.condition_id, []).append(ep.success)
            if not ep.success:
                condition_failures.setdefault(ep.condition_id, Counter())[ep.failure_mode or "unknown"] += 1
        failure_modes = Counter(ep.failure_mode or "unknown" for ep in best if not ep.success)
        return cls(
            design_id=design_id,
            mass_g=float(mass_g),
            success=success_by_k[best_k],
            best_k=best_k,
            success_by_k=success_by_k,
            hold_fraction=hold_by_k[best_k],
            per_family={f: mean(map(float, v)) for f, v in per_family.items()},
            per_condition={c: mean(map(float, v)) for c, v in per_condition.items()},
            failure_modes=dict(sorted(failure_modes.items(), key=lambda kv: (-kv[1], kv[0]))),
            condition_failures={c: dict(sorted(v.items())) for c, v in condition_failures.items()},
            n_episodes=len(episodes),
            episodes=tuple(episodes),
        )

    def rank_key(self) -> tuple[float, float, float]:
        """Sort key, larger is better: success, then hold fraction, then lower mass."""
        return (round(self.success, 12), round(self.hold_fraction, 12), -round(self.mass_g, 9))

    def to_json(self, *, include_episodes: bool = True) -> dict[str, Any]:
        data = {
            "design_id": self.design_id,
            "mass_g": self.mass_g,
            "success": self.success,
            "best_k": self.best_k,
            "success_by_k": list(self.success_by_k),
            "hold_fraction": self.hold_fraction,
            "per_family": dict(self.per_family),
            "per_condition": dict(self.per_condition),
            "failure_modes": dict(self.failure_modes),
            "condition_failures": {k: dict(v) for k, v in self.condition_failures.items()},
            "n_episodes": self.n_episodes,
        }
        if include_episodes:
            data["episodes"] = [e.to_json() for e in self.episodes]
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> DesignScore:
        return cls(
            design_id=str(data["design_id"]),
            mass_g=float(data["mass_g"]),
            success=float(data["success"]),
            best_k=int(data["best_k"]),
            success_by_k=tuple(float(v) for v in data["success_by_k"]),
            hold_fraction=float(data["hold_fraction"]),
            per_family=dict(data["per_family"]),
            per_condition=dict(data["per_condition"]),
            failure_modes=dict(data["failure_modes"]),
            condition_failures={k: dict(v) for k, v in data["condition_failures"].items()},
            n_episodes=int(data["n_episodes"]),
            episodes=tuple(EpisodeOutcome.from_json(e) for e in data.get("episodes", ())),
        )


@runtime_checkable
class Evaluator(Protocol):
    """Scores finger designs on the training suite.

    Implementations must be deterministic per design (common random numbers: the
    same conditions, controller settings and jitter seeds for every design) and
    batch-invariant, return one :class:`DesignScore` per input design in input
    order, and expose only training-suite facts through :meth:`describe`.
    """

    def evaluate(self, designs: Sequence[FingerDesign]) -> list[DesignScore]: ...

    def describe(self) -> EvaluatorInfo: ...


# --------------------------------------------------------------------------- proposals and history


@dataclass(frozen=True)
class Proposal:
    """A valid design an arm wants simulated, plus where it came from.

    ``origin`` is one of ``entrant``, ``sobol`` (shared round 0), ``random``,
    ``nemotron``, ``zeroshot`` or ``fallback_random``. LLM proposals carry the
    operator (``mutate``/``crossover``/``explore``/``zeroshot``), island, parent
    labels, hypothesis, the call that produced them (``via``: main, repair or
    reflection) and that call's request hash.
    """

    design: FingerDesign
    origin: str
    operator: str | None = None
    island: int | None = None
    slot: str | None = None
    parents: tuple[str, ...] = ()
    hypothesis: str | None = None
    via: str | None = None
    request_hash: str | None = None
    fallback_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "design": self.design.model_dump(mode="json"),
            "design_id": self.design.design_id(),
            "origin": self.origin,
            "operator": self.operator,
            "island": self.island,
            "slot": self.slot,
            "parents": list(self.parents),
            "hypothesis": self.hypothesis,
            "via": self.via,
            "request_hash": self.request_hash,
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Proposal:
        return cls(
            design=FingerDesign.model_validate(data["design"]),
            origin=str(data["origin"]),
            operator=data.get("operator"),
            island=data.get("island"),
            slot=data.get("slot"),
            parents=tuple(data.get("parents") or ()),
            hypothesis=data.get("hypothesis"),
            via=data.get("via"),
            request_hash=data.get("request_hash"),
            fallback_reason=data.get("fallback_reason"),
        )


@dataclass(frozen=True)
class Evaluated:
    """A simulated design in a run: evaluation order (1-based), round, proposal and score."""

    index: int
    round: int
    proposal: Proposal
    score: DesignScore

    @property
    def label(self) -> str:
        """Run-local name shown to the proposer, e.g. ``d07``."""
        return f"d{self.index:02d}"

    @property
    def design(self) -> FingerDesign:
        return self.proposal.design

    def to_json(self, *, include_episodes: bool = True) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "round": self.round,
            "proposal": self.proposal.to_json(),
            "score": self.score.to_json(include_episodes=include_episodes),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Evaluated:
        return cls(
            index=int(data["index"]),
            round=int(data["round"]),
            proposal=Proposal.from_json(data["proposal"]),
            score=DesignScore.from_json(data["score"]),
        )


@dataclass(frozen=True)
class Rejection:
    """A proposal that never reached the simulator, and why (free, but reported)."""

    round: int
    stage: str
    """``main``, ``repair`` or ``reflection`` (the LLM call that produced it)."""
    reason: str
    """``invalid``, ``parse_error``, ``too_similar`` or ``call_failed``."""
    errors: tuple[str, ...]
    island: int | None = None
    slot: str | None = None
    params: Mapping[str, Any] | None = None
    request_hash: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["errors"] = list(self.errors)
        data["params"] = dict(self.params) if self.params is not None else None
        return data


@dataclass(frozen=True)
class LLMCallRecord:
    """One model call made by an arm (no message bodies: the request hash points into the cache)."""

    tag: str
    kind: str
    """``main``, ``repair`` or ``reflection``."""
    round: int
    island: int
    request_hash: str | None
    ok: bool
    cached: bool = False
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    finish_reason: str | None = None
    notes: tuple[str, ...] = ()
    analysis: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data

    def replay_view(self) -> dict[str, Any]:
        """Fields that must be identical when the arm is replayed from cache."""
        return {
            "tag": self.tag,
            "kind": self.kind,
            "round": self.round,
            "island": self.island,
            "request_hash": self.request_hash,
            "ok": self.ok,
            "error": self.error,
            "analysis": self.analysis,
        }


class RunHistory:
    """The evaluated designs of one run, in evaluation order."""

    def __init__(self) -> None:
        self._items: list[Evaluated] = []

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Evaluated]:
        return iter(self._items)

    def __getitem__(self, index: int) -> Evaluated:
        return self._items[index]

    @property
    def items(self) -> tuple[Evaluated, ...]:
        return tuple(self._items)

    def add(self, round_index: int, proposal: Proposal, score: DesignScore) -> Evaluated:
        if score.design_id != proposal.design.design_id():
            raise ValueError("score does not belong to the proposed design")
        item = Evaluated(index=len(self._items) + 1, round=round_index, proposal=proposal, score=score)
        self._items.append(item)
        return item

    def by_label(self, label: str) -> Evaluated | None:
        for item in self._items:
            if item.label == label:
                return item
        return None

    def designs(self) -> list[FingerDesign]:
        return [item.design for item in self._items]

    def ranked(self, pool: Iterable[Evaluated] | None = None) -> list[Evaluated]:
        """Best first; ties keep evaluation order."""
        items = list(self._items if pool is None else pool)
        return sorted(items, key=lambda e: (*(-v for v in e.score.rank_key()), e.index))

    def best(self) -> Evaluated:
        if not self._items:
            raise ValueError("empty history")
        return self.ranked()[0]

    def best_so_far(self) -> list[float]:
        """Best training success after each evaluation (for sample-efficiency curves)."""
        out: list[float] = []
        best = -math.inf
        for item in self._items:
            best = max(best, item.score.success)
            out.append(best)
        return out

    def fingerprint(self, rejections: Sequence[Rejection] = (), calls: Sequence[LLMCallRecord] = ()) -> str:
        """SHA-256 over designs, scores, rejections and call identities (no timings)."""
        payload = {
            "evaluated": [item.to_json(include_episodes=False) for item in self._items],
            "rejections": [r.to_json() for r in rejections],
            "calls": [c.replay_view() for c in calls],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class BudgetError(RuntimeError):
    """A run tried to simulate more (or other) than its pre-registered budget allows."""


class BudgetLedger:
    """Per-run budget: simulator evaluations by round, plus the LLM-call cap.

    Args:
        budget: Valid designs simulated per run (``B``).
        round0_size: Designs in the shared round 0.
        round_size: Designs per later round.
        llm_call_cap: Maximum model calls per run (proposals, repairs, reflections).
    """

    def __init__(
        self,
        budget: int = DEFAULT_BUDGET,
        round0_size: int = DEFAULT_ROUND_SIZE,
        round_size: int = DEFAULT_ROUND_SIZE,
        llm_call_cap: int = DEFAULT_LLM_CALL_CAP,
    ) -> None:
        if llm_call_cap < 0:
            raise ValueError("llm_call_cap must be >= 0")
        self.budget = budget
        self.round_sizes = round_plan(budget, round0_size, round_size)
        self.llm_call_cap = llm_call_cap
        self.charged: list[int] = [0] * len(self.round_sizes)
        self.llm_calls = 0
        self.rejections = 0
        self.fallbacks = 0
        self.sampler_rejections = 0

    @property
    def n_rounds(self) -> int:
        return len(self.round_sizes)

    @property
    def spent(self) -> int:
        return sum(self.charged)

    @property
    def remaining(self) -> int:
        return self.budget - self.spent

    @property
    def complete(self) -> bool:
        return self.spent == self.budget

    def charge(self, round_index: int, n: int) -> None:
        """Record ``n`` simulated designs in ``round_index``; the round must be filled exactly once."""
        if not 0 <= round_index < self.n_rounds:
            raise BudgetError(f"round {round_index} is outside the plan {self.round_sizes}")
        if any(self.charged[r] != self.round_sizes[r] for r in range(round_index)):
            raise BudgetError(f"round {round_index} charged before earlier rounds were complete")
        if self.charged[round_index] or n != self.round_sizes[round_index]:
            raise BudgetError(
                f"round {round_index} must simulate exactly {self.round_sizes[round_index]} designs once "
                f"(already {self.charged[round_index]}, asked {n})"
            )
        self.charged[round_index] = n

    def llm_calls_left(self) -> int:
        return max(0, self.llm_call_cap - self.llm_calls)

    def take_llm_call(self, *, keep: int = 0) -> bool:
        """Consume one model call if at least ``keep`` calls remain afterwards."""
        if self.llm_calls + 1 + keep > self.llm_call_cap:
            return False
        self.llm_calls += 1
        return True

    def to_json(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "round_sizes": list(self.round_sizes),
            "charged": list(self.charged),
            "spent": self.spent,
            "llm_call_cap": self.llm_call_cap,
            "llm_calls": self.llm_calls,
            "rejections": self.rejections,
            "fallback_random": self.fallbacks,
            "sampler_rejections": self.sampler_rejections,
        }


@dataclass
class RunContext:
    """State of one independent run, shared by the runner and the run's strategy.

    Nothing in here refers to any other run: arms see only their own history.
    """

    arm: str
    run_index: int
    run_seed: int
    gripper_task: GripperTask
    info: EvaluatorInfo
    budget: BudgetLedger
    history: RunHistory = field(default_factory=RunHistory)
    rejections: list[Rejection] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return f"{self.arm}-r{self.run_index:02d}"

    def reject(self, rejection: Rejection) -> None:
        self.rejections.append(rejection)
        self.budget.rejections += 1

    def fingerprint(self) -> str:
        return self.history.fingerprint(self.rejections, self.llm_calls)


class SearchStrategy(ABC):
    """One arm's proposal policy for one run (ask/tell).

    ``ask`` returns exactly ``n`` proposals whose designs pass the validator under
    ``ctx.gripper_task``; the runner simulates them, appends them to
    ``ctx.history`` and then calls ``tell`` with the new :class:`Evaluated` items.
    Round 0 is shared across arms and filled by the runner, but it is still told
    to the strategy.
    """

    arm: ClassVar[str] = ""
    uses_llm: ClassVar[bool] = False

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def ask(self, round_index: int, n: int) -> list[Proposal]:
        """Propose ``n`` valid designs for ``round_index`` (>= 1)."""

    def tell(self, evaluated: Sequence[Evaluated]) -> None:
        """Receive this run's newly simulated designs (they are already in ``ctx.history``)."""


def params_key(params: Mapping[str, Any]) -> str:
    """Canonical JSON of geometry parameters only (for de-duplication and logs)."""
    return canonical_json({k: params[k] for k in GEOMETRY_PARAMS if k in params})
