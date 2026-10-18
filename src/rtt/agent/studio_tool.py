"""The Grasp Engineer in Effector Lab Studio (product mode, ``docs/STUDIO.md`` 7.7).

Three tools join the Grasp Engineer's toolbox (:class:`rtt.agent.tools.Toolbox`, argument ``studio``):

* ``plan_task(text)``: the Task Engineer (track S4, :func:`rtt.studio.engineer.engineer.plan_task`)
  turns a use case into a :class:`~rtt.studio.types.TaskSpec`. A description of the recorded gate
  run (:data:`rtt.studio.engineer.gate.RECORDED_GATE`) is answered from its replay cache, free and
  without the passcode. Any other description calls Nemotron: the session must be unlocked, the
  plan's calls may spend at most :attr:`StudioCaps.plan_usd` (both calls) and every reservation
  must also fit the session's :class:`~rtt.agent.caps.SpendGuard` (session and day caps), which is
  charged what the calls cost. The plan is saved as ``plan-<12 hex>``.
* ``run_task_tournament(plan_id, n_conditions?)``: the Studio search (track S5,
  :func:`rtt.studio.search.run_task_tournament`) on a valid plan of this session, on this machine
  (free), inside the tool call, one tournament at a time per process, at most
  :data:`HARD_MAX_WORKERS` worker processes and a wall-time cap (:attr:`StudioCaps.wall_s`, at most
  :data:`HARD_MAX_WALL_S`) enforced by :class:`DeadlinePool`. The report is saved as
  ``tour-<12 hex>``; the verdict (``recommended`` or ``no_recommendation``) is computed by code
  (:mod:`rtt.studio.recommend`).
* ``export_build_pack(tournament_id, setup_id?)``: the build pack with printed part jaws
  (:func:`rtt.studio.buildpack.part_jaw.export_part_jaw_pack`) of the recommendation, of a chosen
  finalist, or, with no recommendation, of the closest setup (said so), under a zip size cap
  (:attr:`StudioCaps.max_zip_bytes`). Saved as ``spack-<12 hex>``.

Every result carries the product-mode label and no local path. Outputs live under
``<store root>/studio/`` (``plans/``, ``tournaments/``, ``packs/``, and ``cache/`` for live plans).

:class:`StudioEngineer` is a Grasp Engineer session that offers only these tools (and
``cost_ledger``), with its own prompts and Lightning summaries; :class:`ScriptedStudioTransport`
plays it offline; :func:`record_studio_session` records one session of a hero use case::

    python -m rtt.agent.studio_tool record --case h1_tote --out DIR [--live --env-file PATH --budget USD]
                                           [--workers N] [--wall-s S]

Product mode: not part of the pre-registered experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Final, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator

from rtt.agent.caps import (
    AgentCaps,
    CapError,
    DailySpendBook,
    GuardedLedger,
    SpendCapExceededError,
    SpendGuard,
)
from rtt.agent.engineer import AgentLLM, GraspEngineer, SessionResult, ToolCallRequest
from rtt.agent.recorder import PRODUCT_MODE_LABEL, SCRIPTED_MODEL, SessionRecorder, redact_paths
from rtt.agent.schemas import ArgsError, ToolSpec, tool_spec
from rtt.agent.store import AgentStore, new_id
from rtt.agent.summarizer import (
    LightningSummarizer,
    drop_unsupported_passes,
    drop_verdict_claims,
    verdicts_in,
)
from rtt.agent.tools import Toolbox, ToolOutcome
from rtt.llm.cache import CacheMiss
from rtt.llm.client import LLMError
from rtt.llm.ledger import BudgetExceededError, Ledger, LedgerEntry
from rtt.llm.providers import NEMOTRON_LIGHTNING, OpenAITransport, TransportResponse
from rtt.store.manifest import write_json_atomic
from rtt.studio.engineer.engineer import (
    ENGINEER_MODEL,
    MAX_USE_CASE_CHARS,
    PlanResult,
    make_engineer_client,
    plan_task,
)
from rtt.studio.engineer.gate import RECORDED_GATE
from rtt.studio.recommend import APPROXIMATION_CAVEAT
from rtt.studio.runner import WorkerPool

__all__ = [
    "DEFAULT_SESSION_USD",
    "HARD_MAX_WALL_S",
    "HARD_MAX_WORKERS",
    "HERO_CASE_IDS",
    "STUDIO_SESSION_TOOLS",
    "STUDIO_TOOL_NAMES",
    "STUDIO_TOOL_SPECS",
    "DeadlinePool",
    "ExportBuildPackArgs",
    "PackSizeCapError",
    "PlanLedger",
    "PlanNotRecordedError",
    "PlanTaskArgs",
    "RunTaskTournamentArgs",
    "ScriptedStudioTransport",
    "StudioCaps",
    "StudioEngineer",
    "StudioSummarizer",
    "StudioTools",
    "ToolNotOfferedError",
    "TournamentBusyError",
    "WallTimeCapError",
    "hero_text",
    "main",
    "pack_brief",
    "plan_brief",
    "record_studio_session",
    "studio_system_prompt",
    "studio_toolbox",
    "studio_user_prompt",
    "tournament_brief",
    "validate_studio_args",
]

#: No configuration may run a tournament on more worker processes than this (the host is shared).
HARD_MAX_WORKERS: Final = 4
#: No configuration may let one tournament run longer than this (s).
HARD_MAX_WALL_S: Final = 1800.0
#: No configuration may allow a larger build pack zip than this (bytes).
HARD_MAX_ZIP_BYTES: Final = 100 * 1024 * 1024
#: Spend cap of one live plan, both calls (as ``StudioPlanSettings.plan_usd`` of the HTTP route).
DEFAULT_PLAN_USD: Final = 0.05
DEFAULT_WALL_S: Final = 900.0
DEFAULT_ZIP_BYTES: Final = 20 * 1024 * 1024
DEFAULT_CONDITIONS: Final = 8
#: ``run_task_tournament`` screens 3 conditions, so a grid has at least that many.
MIN_CONDITIONS: Final = 3
MAX_CONDITIONS: Final = 16
#: Session cap of a recorded session (agent turns and summaries; recorded plans are free).
DEFAULT_SESSION_USD: Final = 0.10
#: Tool calls of a Studio session (plan, tournament, pack, plus room for one retry each).
STUDIO_MAX_TOOL_CALLS: Final = 6
HERO_CASE_IDS: Final[tuple[str, ...]] = ("h1_tote", "h2_coupling", "h3_cabinet")
STUDIO_TOOL_NAMES: Final[tuple[str, ...]] = ("plan_task", "run_task_tournament", "export_build_pack")
#: What a Studio-only session accepts.
STUDIO_SESSION_TOOLS: Final[tuple[str, ...]] = (*STUDIO_TOOL_NAMES, "cost_ledger")
#: Studio tools whose results get a Lightning summary for the user.
STUDIO_SUMMARIZED_TOOLS: Final = frozenset(STUDIO_TOOL_NAMES)
STUDIO_GOAL: Final = (
    "Plan this use case, find the tool setup that does the whole task in simulation, and give me its "
    "build pack."
)
_MAX_TEXT_CHARS: Final = 4 * MAX_USE_CASE_CHARS
_MAX_LINES: Final = 6
_MAX_LINE_CHARS: Final = 240
_TIER_ORDER: Final = {"rigid": 0, "approximation": 1, "not_simulated": 2}
#: One tournament at a time in this process (each runs up to HARD_MAX_WORKERS processes).
_TOURNAMENT_LOCK = threading.Lock()

T = TypeVar("T")
R = TypeVar("R")


# --------------------------------------------------------------------------- caps and errors


class WallTimeCapError(CapError):
    """The tournament ran past its wall-time cap and was stopped."""

    code = "wall_time_cap"


class TournamentBusyError(CapError):
    """Another tournament is running in this process."""

    code = "tournament_busy"


class PackSizeCapError(CapError):
    """The build pack's zip is over the size cap (it was deleted)."""

    code = "pack_size_cap"


class ToolNotOfferedError(CapError):
    """A Studio session offers only the Studio tools and ``cost_ledger``."""

    code = "not_offered"


class PlanNotRecordedError(CapError):
    """This session plans only the recorded example descriptions (no live model)."""

    code = "not_recorded"


@dataclass(frozen=True)
class StudioCaps:
    """Limits of the Studio tools (hard ceilings cannot be raised by configuration).

    Attributes:
        plan_usd: Spend cap of one live plan (both calls), inside the session and day caps.
        wall_s: Wall-time cap of one tournament (at most :data:`HARD_MAX_WALL_S`).
        workers: Worker processes of one tournament (at most :data:`HARD_MAX_WORKERS`; 1 runs in
            this process).
        n_conditions: Conditions per finalist when the model does not ask for a number.
        max_zip_bytes: Largest build pack zip the tool hands over.
    """

    plan_usd: float = DEFAULT_PLAN_USD
    wall_s: float = DEFAULT_WALL_S
    workers: int = HARD_MAX_WORKERS
    n_conditions: int = DEFAULT_CONDITIONS
    max_zip_bytes: int = DEFAULT_ZIP_BYTES

    def __post_init__(self) -> None:
        if not math.isfinite(self.plan_usd) or self.plan_usd < 0:
            raise ValueError("plan_usd must be a finite non-negative number")
        if not (math.isfinite(self.wall_s) and 0 < self.wall_s <= HARD_MAX_WALL_S):
            raise ValueError(f"wall_s must be in (0, {HARD_MAX_WALL_S:.0f}] s, got {self.wall_s}")
        if not 1 <= self.workers <= HARD_MAX_WORKERS:
            raise ValueError(f"workers must be in 1..{HARD_MAX_WORKERS}, got {self.workers}")
        if not MIN_CONDITIONS <= self.n_conditions <= MAX_CONDITIONS:
            raise ValueError(f"n_conditions must be in {MIN_CONDITIONS}..{MAX_CONDITIONS}")
        if not 1 <= self.max_zip_bytes <= HARD_MAX_ZIP_BYTES:
            raise ValueError(f"max_zip_bytes must be in 1..{HARD_MAX_ZIP_BYTES}")

    def to_json(self) -> dict[str, Any]:
        return {
            "plan_usd": self.plan_usd,
            "wall_s": self.wall_s,
            "workers": self.workers,
            "n_conditions": self.n_conditions,
            "max_zip_bytes": self.max_zip_bytes,
        }


class DeadlinePool(WorkerPool):
    """A :class:`~rtt.studio.runner.WorkerPool` with a wall-time deadline (set when it is made).

    Every :meth:`map` checks the deadline first. With worker processes the map waits at most the
    time left (``map_async(...).get(timeout=...)``) and the pool is terminated on timeout; in this
    process (``workers = 1``) the deadline is checked before every item. Past the deadline,
    :class:`WallTimeCapError`. With more than one worker every map goes to the worker processes,
    a single item included, so no item runs where it cannot be stopped.
    """

    def __init__(
        self, workers: int | None = 1, *, wall_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        super().__init__(max(1, min(int(workers or 1), HARD_MAX_WORKERS)))
        if not (math.isfinite(wall_s) and wall_s > 0):
            raise ValueError("wall_s must be a positive number")
        self.wall_s = float(wall_s)
        self._clock = clock
        self.deadline = clock() + self.wall_s
        self.timed_out = False

    def remaining_s(self) -> float:
        return self.deadline - self._clock()

    def _refuse(self) -> WallTimeCapError:
        self.timed_out = True
        self.terminate()
        return WallTimeCapError(
            f"refused: the tournament ran past its {self.wall_s:.0f} s wall-time cap and was stopped"
        )

    def _check(self) -> None:
        if self.remaining_s() <= 0:
            raise self._refuse()

    def map(self, fn: Callable[[T], R], items: Sequence[T], chunksize: int = 1) -> list[R]:
        items = list(items)
        self._check()
        t0 = time.perf_counter()
        out: list[R] = []
        if self.workers <= 1:
            for item in items:
                self._check()
                out.append(fn(item))
        elif items:
            if self._pool is None:
                self._pool = mp.get_context("spawn").Pool(processes=self.workers)
            pending = self._pool.map_async(fn, items, chunksize=max(1, chunksize))
            try:
                out = pending.get(timeout=max(0.0, self.remaining_s()))
            except mp.TimeoutError:
                raise self._refuse() from None
        self.jobs_run += len(items)
        self.busy_s += time.perf_counter() - t0
        return out

    def terminate(self) -> None:
        """Stop the worker processes now (the jobs in flight are lost)."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None


class PlanLedger(Ledger):
    """The ledger of one live plan: its own cap (:attr:`StudioCaps.plan_usd`) on top of the
    session's ledger.

    A reservation must fit this plan's cap and then the session ledger's (for a
    :class:`~rtt.agent.caps.GuardedLedger`, the session and day caps of its spend guard); every
    entry is recorded here and in the session ledger, which charges the guard what the call cost.
    """

    def __init__(self, inner: Ledger, cap_usd: float) -> None:
        super().__init__(None, budget_usd=cap_usd)
        self.inner = inner

    def reserve(self, estimated_cost_usd: float) -> None:
        super().reserve(estimated_cost_usd)
        try:
            self.inner.reserve(estimated_cost_usd)
        except BaseException:
            super().release(estimated_cost_usd)
            raise

    def release(self, estimated_cost_usd: float) -> None:
        super().release(estimated_cost_usd)
        self.inner.release(estimated_cost_usd)

    @property
    def remaining_usd(self) -> float | None:
        own, inner = super().remaining_usd, self.inner.remaining_usd
        return own if inner is None else (inner if own is None else min(own, inner))

    def record(self, entry: LedgerEntry) -> None:
        super().record(entry)
        self.inner.record(entry)


# --------------------------------------------------------------------------- schemas

PLAN_ID_PATTERN: Final = r"^plan-[0-9a-f]{12}$"
TOURNAMENT_ID_PATTERN: Final = r"^tour-[0-9a-f]{12}$"
STUDIO_PACK_ID_PATTERN: Final = r"^spack-[0-9a-f]{12}$"
#: A finalist's setup ID as the tournament shows it (12 hex digits) or in full (64).
SETUP_ID_PATTERN: Final = r"^[0-9a-f]{12}([0-9a-f]{52})?$"
_ID_PATTERNS: Final = {
    "plan": re.compile(PLAN_ID_PATTERN),
    "tour": re.compile(TOURNAMENT_ID_PATTERN),
    "spack": re.compile(STUDIO_PACK_ID_PATTERN),
}

PlanId = Annotated[str, StringConstraints(pattern=PLAN_ID_PATTERN)]
TournamentId = Annotated[str, StringConstraints(pattern=TOURNAMENT_ID_PATTERN)]
SetupId = Annotated[str, StringConstraints(pattern=SETUP_ID_PATTERN)]


class _StudioArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanTaskArgs(_StudioArgs):
    text: str = Field(min_length=1, max_length=_MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("describe the use case")
        return value


class RunTaskTournamentArgs(_StudioArgs):
    plan_id: PlanId
    n_conditions: int | None = Field(default=None, ge=MIN_CONDITIONS, le=MAX_CONDITIONS)


class ExportBuildPackArgs(_StudioArgs):
    tournament_id: TournamentId
    setup_id: SetupId | None = None


def _object(properties: Mapping[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": required,
        "additionalProperties": False,
    }


STUDIO_TOOL_SPECS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        "plan_task",
        "Turn the user's use case, in their words, into a task plan: the objects as parametric "
        "templates and the ordered steps as primitive skills, each step with its fidelity tier. A "
        "description of one of the recorded examples is answered from the recording at no cost; any "
        "other description calls the Task Engineer model and costs money (the session needs the "
        "passcode). Returns a plan_id, the status (valid, refused, invalid or failed), the objects and "
        "the steps. Product mode, not the pre-registered experiment.",
        _object(
            {
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_TEXT_CHARS,
                    "description": "The use case in the user's words.",
                }
            },
            ["text"],
        ),
        PlanTaskArgs,
        may_spend=True,
    ),
    ToolSpec(
        "run_task_tournament",
        "Find the tool setup that does the whole planned task: every candidate tool screened on every "
        "step, single tools, tool changes and two-tool heads combined, the finalists run on the full "
        "task over a grid of placement conditions in MuJoCo physics, and one recommendation computed "
        "by code (or none, with the closest setups), next to the best single off-the-shelf effector. "
        "Runs on this machine at no cost and can take several minutes; call it once. Needs a valid "
        f"plan_id from plan_task. n_conditions: conditions per finalist ({MIN_CONDITIONS}-"
        f"{MAX_CONDITIONS}, default {DEFAULT_CONDITIONS}).",
        _object(
            {
                "plan_id": {
                    "type": "string",
                    "pattern": PLAN_ID_PATTERN,
                    "description": "Plan ID from plan_task, e.g. plan-0123456789ab.",
                },
                "n_conditions": {"type": "integer", "minimum": MIN_CONDITIONS, "maximum": MAX_CONDITIONS},
            },
            ["plan_id"],
        ),
        RunTaskTournamentArgs,
    ),
    ToolSpec(
        "export_build_pack",
        "Write the build pack of a tournament's recommended setup (or of setup_id, a finalist's ID as "
        "the tournament shows it): printable parts and jaws, the tested MJCF scenes, the bill of "
        "materials, the certificate, licences and upstream links, as one zip. Without a "
        "recommendation it packs the closest setup and says so. Runs locally, no cost.",
        _object(
            {
                "tournament_id": {
                    "type": "string",
                    "pattern": TOURNAMENT_ID_PATTERN,
                    "description": "Tournament ID from run_task_tournament.",
                },
                "setup_id": {
                    "type": "string",
                    "pattern": SETUP_ID_PATTERN,
                    "description": "A finalist's setup ID from run_task_tournament (default: the recommendation).",
                },
            },
            ["tournament_id"],
        ),
        ExportBuildPackArgs,
    ),
)
_BY_NAME: Final[dict[str, ToolSpec]] = {spec.name: spec for spec in STUDIO_TOOL_SPECS}


def validate_studio_args(name: str, args: Any) -> BaseModel:
    """Validate ``args`` for Studio tool ``name`` (as :func:`rtt.agent.schemas.validate_args`).

    Raises:
        ArgsError: unknown tool, arguments not an object, or any field problem.
    """
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ArgsError(name, [f"unknown Studio tool {name!r}; tools are {', '.join(STUDIO_TOOL_NAMES)}"])
    if not isinstance(args, Mapping):
        detail = getattr(args, "error", None)  # the agent passes its JSON parse error along
        raise ArgsError(name, [str(detail) if detail else "arguments must be a JSON object"])
    try:
        return spec.args_model.model_validate(dict(args))
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            where = ".".join(str(p) for p in err.get("loc", ())) or "arguments"
            problems.append(f"{where}: {err.get('msg', 'invalid')}")
        raise ArgsError(name, problems) from None


# --------------------------------------------------------------------------- what the model sees


def _pct(rate: Any) -> float | None:
    return None if rate is None else round(100.0 * float(rate), 1)


def _lines(items: Sequence[str]) -> list[str]:
    return [redact_paths(str(p))[:_MAX_LINE_CHARS] for p in list(items)[:_MAX_LINES]]


def _json_safe(value: Any) -> Any:
    """JSON-safe copy for ``allow_nan=False`` files (non-finite floats become ``None``)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def plan_brief(plan_id: str, source: str, result: PlanResult) -> dict[str, Any]:
    """The ``plan_task`` answer: the plan's status, objects and steps, with each step's fidelity tier."""
    spec = result.spec
    out: dict[str, Any] = {
        "kind": "studio_plan",
        "label": PRODUCT_MODE_LABEL,
        "plan_id": plan_id,
        "source": source,
        "status": result.status,
        "result": result.result,
        "task_id": spec.task_id if spec is not None and result.status == "valid" else None,
        "title": spec.title if spec is not None else None,
        "objects": [{"name": o.name, "kind": o.kind.value} for o in spec.objects] if spec else [],
        "steps": [
            {
                "id": s.id,
                "primitive": s.primitive.value,
                "object": s.object,
                "target": s.target,
                "fidelity": spec.fidelity(s.id).value,
            }
            for s in (spec.steps if spec is not None else ())
        ],
        "problems": _lines(result.problems),
        "warnings": _lines(result.warnings),
        "cost_usd": round(result.cost_usd, 6),
    }
    if spec is not None and result.status == "refused":
        out["refusal"] = redact_paths(spec.refusal or "")[:_MAX_LINE_CHARS]
        out["unsupported"] = [
            {
                "what": u.what[:120],
                "nearest_primitive": u.nearest_primitive.value if u.nearest_primitive else None,
            }
            for u in spec.unsupported[:_MAX_LINES]
        ]
    if result.layout_moves:
        out["layout_moves"] = _lines(result.layout_moves)
    out["next"] = (
        "run_task_tournament with this plan_id"
        if result.status == "valid"
        else "no tournament: say what the plan lacks"
    )
    return out


def _runs_by_setup(data: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    return {f["setup"]["setup_id"]: list(f.get("runs") or []) for f in data.get("finalists") or []}


def _first_stop(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Where the setup's runs stop most often (the first entry of the ranking row's ``stops_at``)."""
    for step_id, info in dict((row or {}).get("stops_at") or {}).items():
        return {
            "step_id": step_id,
            "runs": info.get("runs"),
            "failure_modes": dict(info.get("failure_modes") or {}),
        }
    return None


def _step_tiers(runs: Sequence[Mapping[str, Any]], task: Mapping[str, Any] | None) -> dict[str, str]:
    """Per step, the worst fidelity tier of the plan and of the setup's runs."""
    tiers = {str(k): str(v) for k, v in dict((task or {}).get("fidelity") or {}).items()}
    for run in runs:
        for step in run.get("steps") or []:
            sid, tier = str(step.get("step_id")), str(step.get("fidelity") or "rigid")
            if sid not in tiers or _TIER_ORDER.get(tier, 2) > _TIER_ORDER.get(tiers[sid], 2):
                tiers[sid] = tier
    return tiers


class _Briefer:
    """Builds the setup rows of one tournament report."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = data
        self.rec: Mapping[str, Any] = data.get("recommendation") or {}
        self.rows = list(self.rec.get("ranking") or []) or list(self.rec.get("reference_results") or [])
        self.by_id = {r["setup_id"]: r for r in self.rows}
        self.runs = _runs_by_setup(data)

    def counts(self, card: Mapping[str, Any]) -> tuple[int, int]:
        runs = self.runs.get(card["setup_id"])
        if runs:
            return sum(1 for r in runs if r.get("completed")), len(runs)
        total = int(card.get("n_runs") or 0)
        return round(float(card.get("completion_rate") or 0.0) * total), total

    def row(self, card: Mapping[str, Any]) -> dict[str, Any]:
        done, total = self.counts(card)
        sid = card["setup_id"]
        return {
            "setup_id": sid[:12],
            "label": card.get("label"),
            "kind": card.get("kind"),
            "completion_pct": _pct(card.get("completion_rate")),
            "completed_runs": done,
            "total_runs": total,
            "stops_at": _first_stop(self.by_id.get(sid) or card),
        }

    def setup(self, card: Mapping[str, Any]) -> dict[str, Any]:
        sid = card["setup_id"]
        tiers = _step_tiers(self.runs.get(sid, []), self.data.get("task"))
        fidelity = str(card.get("fidelity") or "rigid")
        approx = fidelity != "rigid" or any(t != "rigid" for t in tiers.values())
        return {
            **self.row(card),
            "tools": [t.get("display_name") for t in card.get("tools") or []],
            "cycle_time_s": card.get("cycle_time_s"),
            "fidelity": fidelity,
            "step_fidelity": tiers,
            "caveat": card.get("caveat") or (APPROXIMATION_CAVEAT if approx else None),
            "verify_on_hardware": approx,
        }


def tournament_brief(
    tournament_id: str, plan_id: str, data: Mapping[str, Any], *, wall_s: float, workers: int | None = None
) -> dict[str, Any]:
    """The ``run_task_tournament`` answer (the verdict and every number from the report, by code)."""
    b = _Briefer(data)
    rec = b.rec
    card = rec.get("recommendation")
    out: dict[str, Any] = {
        "kind": "studio_tournament",
        "label": PRODUCT_MODE_LABEL,
        "tournament_id": tournament_id,
        "plan_id": plan_id,
        "task_id": data.get("task_id"),
        "verdict": "recommended" if card else "no_recommendation",
        "evidence": rec.get("evidence"),
        "recommendation": None,
    }
    if card:
        out["recommendation"] = {**b.setup(card), "why": card.get("why")}
        ru = rec.get("runner_up_rigid")
        if ru:
            out["runner_up_rigid"] = (
                {**b.row(ru), "fidelity": ru.get("fidelity")}
                if ru.get("setup_id")
                else {"note": ru.get("basis")}
            )
    single = rec.get("best_single")
    out["best_single"] = (
        {**b.row(single), "cycle_time_s": single.get("cycle_time_s"), "fidelity": single.get("fidelity")}
        if single
        else None
    )
    vs = rec.get("versus_best_single")
    out["versus_best_single"] = (
        {
            "same_setup": bool(vs.get("same_setup")),
            "completion_delta_pct": _pct(vs.get("completion_delta")),
            "cycle_delta_s": vs.get("cycle_delta_s"),
            "extra_tools": vs.get("extra_tools"),
        }
        if vs
        else None
    )
    out["finalists"] = [b.row(r) for r in b.rows]
    if not card:
        nw = rec.get("no_winner") or {}
        closest = list(nw.get("closest") or []) or b.rows[:3]
        out["no_recommendation"] = {
            "reason": nw.get("reason") or rec.get("reason") or "no setup completed the task",
            "closest": [
                {**b.row(c), "why_not": c.get("reason"), "step_progress_pct": _pct(c.get("step_progress"))}
                for c in closest[:3]
            ],
        }
    out["runs"] = sum(len(v) for v in b.runs.values())
    out["conditions"] = len(data.get("conditions") or [])
    out["wall_time_s"] = round(float(wall_s), 1)
    out["workers"] = workers if workers is not None else data.get("workers")
    out["next"] = "export_build_pack with this tournament_id"
    return out


def tournament_verdict_row(brief: Mapping[str, Any]) -> dict[str, Any]:
    """The session file's record of a Studio tournament verdict (computed by code)."""
    pick = brief.get("recommendation") or {}
    single = brief.get("best_single") or {}
    return {
        "kind": "studio_recommendation",
        "tournament_id": brief["tournament_id"],
        "plan_id": brief["plan_id"],
        "task_id": brief.get("task_id"),
        "recommendation": brief["verdict"],
        "setup_id": pick.get("setup_id"),
        "setup_label": pick.get("label"),
        "completion_pct": pick.get("completion_pct"),
        "completed_runs": pick.get("completed_runs"),
        "total_runs": pick.get("total_runs"),
        "fidelity": pick.get("fidelity"),
        "verify_on_hardware": pick.get("verify_on_hardware"),
        "best_single_label": single.get("label"),
        "best_single_completion_pct": single.get("completion_pct"),
        "label": PRODUCT_MODE_LABEL,
    }


def pack_brief(
    pack_id: str, tournament_id: str, pack: Any, *, purpose: str, setup_label: str | None, zip_bytes: int
) -> dict[str, Any]:
    """The ``export_build_pack`` answer: the files (path, kind, licence, bytes), never a local path."""
    root = Path(pack.out_dir)
    manifest = pack.manifest
    files = [
        {
            "path": f.path,
            "kind": f.kind,
            "license": f.license,
            "bytes": (root / f.path).stat().st_size if (root / f.path).is_file() else None,
        }
        for f in manifest.files
    ]
    out: dict[str, Any] = {
        "kind": "studio_build_pack",
        "label": PRODUCT_MODE_LABEL,
        "pack_id": pack_id,
        "tournament_id": tournament_id,
        "setup_id": str(pack.setup_id)[:12],
        "setup_label": setup_label,
        "for": purpose,
    }
    if purpose == "closest":
        out["note"] = (
            "There is no recommendation: this pack is for the closest setup, which did not complete the task."
        )
    out.update(
        files=files,
        n_files=len(files),
        zip_bytes=int(zip_bytes),
        links=len(manifest.links),
        caveats=len(manifest.caveats),
    )
    return out


def _closest_setup_id(data: Mapping[str, Any]) -> str | None:
    """The closest setup of a report without a recommendation: ``ranking[0]``, else the first of the
    no-winner list or the reference results, else the first finalist."""
    rec = data.get("recommendation") or {}
    for rows in (
        rec.get("ranking"),
        (rec.get("no_winner") or {}).get("closest"),
        rec.get("reference_results"),
    ):
        if rows:
            return str(rows[0]["setup_id"])
    finalists = data.get("finalists") or []
    return str(finalists[0]["setup"]["setup_id"]) if finalists else None


# --------------------------------------------------------------------------- the tools


@dataclass
class _Plan:
    plan_id: str
    source: str
    result: PlanResult


@dataclass
class _Tournament:
    tournament_id: str
    plan_id: str
    data: dict[str, Any]
    brief: dict[str, Any] = field(default_factory=dict)


def _same_words(text: str) -> str:
    """``text`` with its spacing, letter case and surrounding quotes normalised (a lookup key)."""
    return " ".join(text.split()).strip("\"'` ").casefold()


def _new_studio_id(kind: str) -> str:
    value = f"{kind}-{uuid.uuid4().hex[:12]}"
    assert _ID_PATTERNS[kind].match(value)
    return value


def _remove(*paths: Path | None) -> None:
    for path in paths:
        if path is None:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _default_tournament(spec: Any, *, workers: int, n_conditions: int, pool: WorkerPool) -> Any:
    from rtt.studio.search import run_task_tournament

    return run_task_tournament(spec, workers=workers, n_conditions=n_conditions, pool=pool)


def _default_export(report: Mapping[str, Any], out_dir: Path, *, setup_id: str | None) -> Any:
    from rtt.studio.buildpack.part_jaw import export_part_jaw_pack

    return export_part_jaw_pack(report, out_dir, setup_id=setup_id)


class StudioTools:
    """The Studio tools of one session (module docstring); give it to :class:`~rtt.agent.tools.Toolbox`
    as ``studio``.

    Args:
        caps: The Studio limits.
        recorded_cache: The replay cache of recorded plans (``None``: nothing is recorded).
        tournament: ``tournament(spec, workers=, n_conditions=, pool=)`` returning a
            :class:`~rtt.studio.search.TournamentReport` or its ``to_dict()`` (default: the real
            search; tests pass a fake).
        export: ``export(report_dict, out_dir, setup_id=)`` returning a
            :class:`~rtt.studio.buildpack.BuildPack` (default: ``export_part_jaw_pack``).
        transport: The model transport of a live plan (default: Token Factory, built from
            ``env_file`` or the environment on the first live call).
        env_file: The ``.env`` holding ``NEBIUS_API_KEY`` for a live plan (read on first use).
        live_plans: Whether a description that is not recorded may be planned live at all (an
            offline session plans recorded descriptions only).
        exclusive: A Studio-only session: every tool but the Studio tools and ``cost_ledger`` is
            refused as not offered.
        clock: Monotonic clock for the wall-time cap (tests pass a fake).
    """

    def __init__(
        self,
        *,
        caps: StudioCaps | None = None,
        recorded_cache: str | Path | None = RECORDED_GATE / "cache",
        tournament: Callable[..., Any] | None = None,
        export: Callable[..., Any] | None = None,
        transport: Any | None = None,
        env_file: str | Path | None = None,
        live_plans: bool = True,
        exclusive: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.caps = caps or StudioCaps()
        self.recorded_cache = Path(recorded_cache) if recorded_cache is not None else None
        self.tournament = tournament or _default_tournament
        self.export = export or _default_export
        self.transport = transport
        self.env_file = Path(env_file) if env_file is not None else None
        self.live_plans = live_plans
        self.exclusive = exclusive
        self._clock = clock
        self._toolbox: Toolbox | None = None
        self._lock = threading.Lock()
        self._recorded: dict[str, PlanResult | None] = {}
        self._texts: dict[str, str] | None = None
        self._near: set[str] = set()
        self._plans: dict[str, _Plan] = {}
        self._tournaments: dict[str, _Tournament] = {}
        self._verdicts: dict[str, dict[str, Any]] = {}

    def __repr__(self) -> str:
        return f"StudioTools(caps={self.caps!r}, exclusive={self.exclusive})"

    # ------------------------------------------------------------------ the toolbox's hooks

    def owns(self, name: str) -> bool:
        return name in _BY_NAME

    def check_offered(self, name: str) -> None:
        if self.exclusive and name not in STUDIO_SESSION_TOOLS:
            raise ToolNotOfferedError(
                f"{name} is not offered in a Studio session; the tools are {', '.join(STUDIO_SESSION_TOOLS)}"
            )

    def validate(self, name: str, args: Any) -> BaseModel:
        return validate_studio_args(name, args)

    def may_spend(self, name: str, args: Any) -> bool:
        """Only ``plan_task`` spends, and only on a description that is not recorded."""
        return name == "plan_task" and self.recorded_plan(args.text) is None

    def passcode_text(self, name: str) -> str:
        return (
            f"{name} on a description that is not one of the recorded examples calls a paid model; "
            "this session was not unlocked with the passcode"
        )

    def bind(self, toolbox: Toolbox) -> dict[str, Callable[[Any], dict[str, Any]]]:
        """Attach to the session's toolbox; returns the handlers it registers."""
        if self._toolbox is not None and self._toolbox is not toolbox:
            raise ValueError("a StudioTools serves one session")
        self._toolbox = toolbox
        return {
            "plan_task": self._plan_task,
            "run_task_tournament": self._run_tournament,
            "export_build_pack": self._export_build_pack,
        }

    def verdicts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(v) for v in self._verdicts.values()]

    def openai_tools(self) -> list[dict[str, Any]]:
        """What a Studio session offers the model: the three tools, then ``cost_ledger``."""
        return [spec.openai() for spec in STUDIO_TOOL_SPECS] + [tool_spec("cost_ledger").openai()]

    # ------------------------------------------------------------------ helpers

    def _tb(self) -> Toolbox:
        if self._toolbox is None:
            raise RuntimeError("the Studio tools are not bound to a toolbox")
        return self._toolbox

    def _dir(self, name: str) -> Path:
        return Path(self._tb().store.root) / "studio" / name

    def _recorded_texts(self) -> dict[str, str]:
        """The recorded descriptions (the gate manifest beside the cache) by their :func:`_same_words` key."""
        if self._texts is None:
            texts: dict[str, str] = {}
            manifest = self.recorded_cache.parent / "manifest.json" if self.recorded_cache else None
            if manifest is not None and manifest.is_file():
                for case in json.loads(manifest.read_text(encoding="utf-8")).get("cases") or []:
                    if isinstance(case.get("text"), str):
                        texts.setdefault(_same_words(case["text"]), case["text"])
            self._texts = texts
        return self._texts

    def _replay(self, text: str) -> PlanResult | None:
        assert self.recorded_cache is not None
        try:
            return plan_task(text, client=make_engineer_client(self.recorded_cache, cache_mode="replay"))
        except CacheMiss:
            return None

    def recorded_plan(self, text: str) -> PlanResult | None:
        """The recorded plan of ``text`` from the replay cache (``None`` when it is not recorded).

        A text that is a recorded description up to spacing, letter case and surrounding quotes
        (a model copying the use case may change those) gets that description's plan."""
        with self._lock:
            if text in self._recorded:
                return self._recorded[text]
        hit: PlanResult | None = None
        if self.recorded_cache is not None:
            hit = self._replay(text)
            same = self._recorded_texts().get(_same_words(text))
            if hit is None and same is not None and same != text:
                hit = self._replay(same)
                if hit is not None:
                    with self._lock:
                        self._near.add(text)
        with self._lock:
            self._recorded[text] = hit
        return hit

    def _live_plan(self, text: str) -> PlanResult:
        tb = self._tb()
        inner = tb.ledger if tb.ledger is not None else GuardedLedger(tb.guard)
        ledger = PlanLedger(inner, self.caps.plan_usd)
        client = make_engineer_client(
            self._dir("cache"),
            ledger=ledger,
            transport=self.transport,
            env_file=self.env_file if self.transport is None else None,
        )
        return plan_task(text, client=client)

    def _emit_calls(self, result: PlanResult, source: str) -> None:
        tb = self._tb()
        if tb.recorder is None:
            return
        for a in result.attempts:
            if a.outcome == "spend_cap":
                continue  # refused before anything was sent
            replay = source == "recorded"
            tb.recorder.emit(
                "llm.call",
                role="task_engineer",
                model=ENGINEER_MODEL if replay or not tb.offline else SCRIPTED_MODEL,
                prompt_tokens=a.prompt_tokens,
                completion_tokens=a.completion_tokens,
                reasoning_tokens=a.reasoning_tokens,
                cost_usd=0.0 if a.cached or tb.offline else round(a.cost_usd, 6),
                latency_s=a.latency_s,
                cached=a.cached,
                finish_reason=a.finish_reason,
                stage=a.stage,
                replay=replay,
            )

    # ------------------------------------------------------------------ plan_task

    def _plan_task(self, args: PlanTaskArgs) -> dict[str, Any]:
        tb = self._tb()
        hit = self.recorded_plan(args.text)
        if hit is not None:
            result, source = hit, "recorded"
        else:
            if not self.live_plans:
                raise PlanNotRecordedError(
                    "this session plans only the recorded example descriptions (no live model here)"
                )
            result, source = self._live_plan(args.text), "live"
            capped = [a for a in result.attempts if a.outcome == "spend_cap"]
            if result.status == "failed" and capped:
                self._emit_calls(result, source)
                raise SpendCapExceededError(
                    redact_paths(capped[-1].error or "refused: the spend cap refused this plan")[:300]
                )
        self._emit_calls(result, source)
        plan_id = _new_studio_id("plan")
        doc = {**result.to_dict(), "plan_id": plan_id, "session_id": tb.session_id, "source": source}
        write_json_atomic(self._dir("plans") / f"{plan_id}.json", _json_safe(doc))
        with self._lock:
            self._plans[plan_id] = _Plan(plan_id, source, result)
            near = args.text in self._near
        brief = plan_brief(plan_id, source, result)
        if near:
            brief["note"] = "matched a recorded description up to spacing, letter case and quotes"
        return brief

    # ------------------------------------------------------------------ run_task_tournament

    def _run_tournament(self, args: RunTaskTournamentArgs) -> dict[str, Any]:
        tb = self._tb()
        with self._lock:
            plan = self._plans.get(args.plan_id)
        if plan is None:
            raise KeyError(f"plan {args.plan_id} (use a plan_id that plan_task returned in this session)")
        spec = plan.result.spec
        if plan.result.status != "valid" or spec is None or not spec.steps:
            raise ArgsError(
                "run_task_tournament",
                [f"plan_id: plan {args.plan_id} is {plan.result.status}, not valid; there is nothing to run"],
            )
        n_conditions = args.n_conditions or self.caps.n_conditions
        if not _TOURNAMENT_LOCK.acquire(blocking=False):
            raise TournamentBusyError(
                "another tournament is running in this process; try again when it finishes"
            )
        try:
            t0 = self._clock()
            pool = DeadlinePool(self.caps.workers, wall_s=self.caps.wall_s, clock=self._clock)
            done = False
            try:
                report = self.tournament(spec, workers=pool.workers, n_conditions=n_conditions, pool=pool)
                done = True
            finally:
                if done:
                    pool.close()
                else:
                    pool.terminate()
            wall = self._clock() - t0
        finally:
            _TOURNAMENT_LOCK.release()
        data = _json_safe(report.to_dict() if hasattr(report, "to_dict") else dict(report))
        tour_id = _new_studio_id("tour")
        brief = tournament_brief(tour_id, plan.plan_id, data, wall_s=wall, workers=pool.workers)
        doc = {**data, "tournament_id": tour_id, "plan_id": plan.plan_id, "session_id": tb.session_id}
        write_json_atomic(self._dir("tournaments") / f"{tour_id}.json", doc)
        with self._lock:
            self._tournaments[tour_id] = _Tournament(tour_id, plan.plan_id, data, brief)
            self._verdicts[tour_id] = tournament_verdict_row(brief)
        return brief

    # ------------------------------------------------------------------ export_build_pack

    def _export_build_pack(self, args: ExportBuildPackArgs) -> dict[str, Any]:
        with self._lock:
            tour = self._tournaments.get(args.tournament_id)
        if tour is None:
            raise KeyError(
                f"tournament {args.tournament_id} (use a tournament_id that run_task_tournament "
                "returned in this session)"
            )
        data = tour.data
        rec_id = ((data.get("recommendation") or {}).get("recommendation") or {}).get("setup_id")
        if args.setup_id:
            setup_id = args.setup_id
            purpose = "recommended" if rec_id and rec_id.startswith(args.setup_id) else "chosen"
        elif rec_id:
            setup_id, purpose = rec_id, "recommended"
        else:
            closest = _closest_setup_id(data)
            if closest is None:
                raise ArgsError("export_build_pack", ["this tournament has no finalist to pack"])
            setup_id, purpose = closest, "closest"
        pack_id = _new_studio_id("spack")
        out = self._dir("packs") / pack_id
        try:
            pack = self.export(data, out, setup_id=setup_id)
        except BaseException:
            _remove(out, out.with_name(out.name + ".zip"))
            raise
        zip_path = Path(pack.zip_path) if pack.zip_path is not None else None
        zip_bytes = zip_path.stat().st_size if zip_path is not None and zip_path.is_file() else 0
        if zip_bytes > self.caps.max_zip_bytes:
            _remove(Path(pack.out_dir), zip_path)
            raise PackSizeCapError(
                f"refused: the build pack zip is {zip_bytes} bytes, over the {self.caps.max_zip_bytes} "
                "byte cap; it was deleted"
            )
        label = next(
            (
                f["setup"].get("label")
                for f in data.get("finalists") or []
                if f["setup"]["setup_id"] == pack.setup_id
            ),
            None,
        )
        return pack_brief(
            pack_id, tour.tournament_id, pack, purpose=purpose, setup_label=label, zip_bytes=zip_bytes
        )


class _NoParts:
    """The part backend of a Studio session: it has no uploaded part."""

    name = "none"

    def __getattr__(self, attr: str) -> Any:
        raise AttributeError(f"a Studio session has no part backend ({attr})")


class _NoProposer:
    """The design proposer of a Studio session: it proposes no finger designs."""

    client = None

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("a Studio session proposes no finger designs")


def studio_toolbox(
    *,
    session_id: str,
    store: AgentStore,
    caps: AgentCaps,
    guard: SpendGuard,
    studio: StudioTools,
    unlocked: bool = False,
    ledger: Ledger | None = None,
    recorder: SessionRecorder | None = None,
    offline: bool = False,
) -> Toolbox:
    """A :class:`~rtt.agent.tools.Toolbox` for a Studio session: no part, no finger tools, the
    Studio tools under the session's caps and spend guard."""
    return Toolbox(
        session_id=session_id,
        part_id="",
        backend=_NoParts(),  # type: ignore[arg-type]
        proposer=_NoProposer(),  # type: ignore[arg-type]
        runners={},
        store=store,
        caps=caps,
        guard=guard,
        unlocked=unlocked,
        ledger=ledger,
        recorder=recorder,
        offline=offline,
        studio=studio,
    )


# --------------------------------------------------------------------------- the session


def studio_system_prompt(caps: AgentCaps, studio: StudioCaps) -> str:
    paragraphs = (
        (
            "You are the Grasp Engineer in Effector Lab Studio. A user describes a robot use case in plain "
            "words; you find the tool setup (one tool, a tool change, or a two-tool head) that does the "
            "whole task in simulation, and hand over its build pack. This is product mode, not the "
            "pre-registered experiment."
        ),
        (
            "Physics decides. A MuJoCo simulator runs every finalist on the full task, and code computes "
            "the recommendation (or none). You orchestrate tools and explain; you never claim a setup works "
            "without the tournament's results, and you report every number exactly as the tools give it."
        ),
        (
            "The plan: call plan_task once with the user's use case exactly as written after 'Use case:' "
            "(copy it, do not reword it). If the plan is valid, call run_task_tournament once with its "
            "plan_id and no n_conditions unless the user asks (it runs on this machine at no cost and can "
            "take several minutes). Then call export_build_pack with the tournament_id. Then answer. If plan_task "
            "refuses the use case or fails, run no tournament: say what is not simulated and the nearest "
            "primitive the plan names. Skip cost_ledger unless asked."
        ),
        (
            f"Hard limits, enforced by code: {caps.max_tool_calls} tool calls in total (every call counts, "
            f"including failed ones); one tournament at a time, on at most {studio.workers} worker "
            f"processes, stopped after {studio.wall_s:.0f} s; a live plan may spend at most "
            f"${studio.plan_usd:.2f} within the session's ${caps.session_usd:.2f} cap, and a description "
            "that is not one of the recorded examples needs the passcode; a build pack zip of at most "
            f"{studio.max_zip_bytes // (1024 * 1024)} MB. Use IDs exactly as tools return them."
        ),
        (
            "Final answer: at most 120 words, plain text. With a recommendation: name the recommended "
            "setup, its completion over the runs (completed of total runs), its fidelity tier, and for an "
            "approximation say verify on hardware; then name the best single off-the-shelf baseline and "
            "its completion. With no recommendation: say that no setup completes the task, name the "
            "closest setup and the step where it stops. Give the build pack ID. Use the numbers exactly "
            "as the tools give them."
        ),
    )
    return "\n\n".join(paragraphs)


def studio_user_prompt(use_case: str) -> str:
    """The user's turn; the use case comes last, after its marker line."""
    return f"Goal: {STUDIO_GOAL}\nUse case:\n{use_case.strip()}"


STUDIO_SUMMARY_SYSTEM: Final = (
    "You write one or two short sentences (at most 45 words) for an engineer watching an AI agent pick "
    "a robot tool setup for their use case in simulation. Use only the facts and numbers in the JSON you "
    "are given. The recommendation comes from the simulator and code: mention a verdict only if the JSON "
    "has a 'verdict' field, repeat it exactly, and never upgrade it; an approximation must be verified on "
    "hardware. Plain text, no markdown, no lists."
)
_MAX_SUMMARY_PAYLOAD: Final = 3500
_MAX_SUMMARY_CHARS: Final = 400


def whole_sentences(text: str, limit: int) -> str:
    """``text`` cut to at most ``limit`` characters at its last full sentence (or word), so a capped
    summary never ends mid-word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    end = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if end > 0:
        return head[: end + 1]
    cut = head.rfind(" ")
    return (head[:cut] if cut > 0 else head).rstrip(" ,;:") + " ..."


class StudioSummarizer(LightningSummarizer):
    """:class:`~rtt.agent.summarizer.LightningSummarizer` with the Studio's system prompt; the same
    code-side rules (no verdict claims without a verdict, never an unsupported pass)."""

    def summarize(self, kind: str, payload: Mapping[str, Any]) -> str | None:
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        if len(text) > _MAX_SUMMARY_PAYLOAD:
            text = text[:_MAX_SUMMARY_PAYLOAD] + " ...(truncated)"
        messages = [
            {"role": "system", "content": STUDIO_SUMMARY_SYSTEM},
            {"role": "user", "content": f"Step: {kind}\nJSON:\n{text}"},
        ]
        self.last_meta = None
        try:
            answer, meta = self.client.chat(messages, tag=f"agent.studio.summary.{kind}")
        except (LLMError, BudgetExceededError, CacheMiss):
            return None
        self.last_meta = meta
        answer = " ".join(answer.split())
        verdicts = verdicts_in(payload)
        if not verdicts:
            answer = drop_verdict_claims(answer)
        elif "pass" not in verdicts:
            answer = drop_unsupported_passes(answer)
        return whole_sentences(answer, _MAX_SUMMARY_CHARS) or None


class StudioEngineer(GraspEngineer):
    """A Grasp Engineer session in the Studio: the use case in, a recommendation and a build pack out.

    Its toolbox must hold :class:`StudioTools` (:func:`studio_toolbox`); the model is offered only the
    Studio tools and ``cost_ledger``. The loop, the caps, the recording and the summaries are the
    Grasp Engineer's; the prompts, the session facts and the summarized tools are the Studio's.
    """

    def __init__(
        self,
        llm: AgentLLM,
        toolbox: Toolbox,
        recorder: SessionRecorder,
        *,
        caps: AgentCaps,
        use_case: str,
        case_id: str | None = None,
        summarizer: LightningSummarizer | None = None,
        stop: threading.Event | None = None,
        offline: bool = False,
    ) -> None:
        if toolbox.studio is None:
            raise ValueError("a Studio session needs a toolbox with Studio tools (studio_toolbox)")
        super().__init__(
            llm,
            toolbox,
            recorder,
            caps=caps,
            goal=STUDIO_GOAL,
            summarizer=summarizer,
            stop=stop,
            offline=offline,
        )
        self.studio: StudioTools = toolbox.studio
        self.use_case = use_case
        self.case_id = case_id

    def _models(self) -> dict[str, str | None]:
        models: dict[str, str | None] = {
            "agent": self.llm.model,
            "task_engineer": ENGINEER_MODEL,  # recorded plans replay its calls; live plans call it
            "summarizer": self.summarizer.client.model if self.summarizer else None,
        }
        if self.offline:
            models["agent"] = SCRIPTED_MODEL
            models["summarizer"] = SCRIPTED_MODEL if self.summarizer else None
        return models

    def run(self) -> SessionResult:
        tools = self.studio.openai_tools()
        self.messages = [
            {"role": "system", "content": studio_system_prompt(self.caps, self.studio.caps)},
            {"role": "user", "content": studio_user_prompt(self.use_case)},
        ]
        self.recorder.emit(
            "session.started",
            kind="studio",
            case_id=self.case_id,
            use_case=self.use_case,
            goal=self.goal,
            label=PRODUCT_MODE_LABEL,
            models=self._models(),
            offline=self.offline,
            caps=self.caps.to_json(),
            studio_caps=self.studio.caps.to_json(),
            unlocked=self.toolbox.unlocked,
            tools=[t["function"]["name"] for t in tools],
        )
        self.recorder.set_messages(self.messages)
        status, reason, answer = "completed", None, ""
        try:
            status, reason, answer = self._loop(tools)
        except (BudgetExceededError, CacheMiss) as exc:
            status, reason = "stopped", redact_paths(f"{type(exc).__name__}: {exc}")
        except LLMError as exc:
            status, reason = "failed", redact_paths(f"{type(exc).__name__}: {str(exc)[:300]}")
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            status, reason = "failed", redact_paths(f"{type(exc).__name__}: {str(exc)[:300]}")
            self.recorder.emit("error", message=reason)
        finally:
            self.toolbox.close()
        verdicts = self.toolbox.verdicts()
        spend = self.toolbox.guard.snapshot()
        summary = None
        if self.summarizer is not None and status != "failed":
            summary = self._summarize(
                "session",
                {"status": status, "answer": answer, "verdicts": verdicts, "spend_usd": spend["spent_usd"]},
            )
            spend = self.toolbox.guard.snapshot()
        self.recorder.set_messages(self.messages)
        self.recorder.update(spend=spend, verdicts=verdicts, answer=answer, summary=summary)
        self.recorder.finish(
            status,
            reason=reason,
            answer=answer,
            summary=summary,
            verdicts=verdicts,
            spend=spend,
            turns=self.turns,
            tool_calls=self.toolbox.calls_used,
        )
        return SessionResult(
            status, answer, reason, self.turns, self.toolbox.calls_used, verdicts, spend, summary
        )

    def _run_tool(self, call: ToolCallRequest) -> ToolOutcome:
        outcome = super()._run_tool(call)
        if outcome.ok and call.name in STUDIO_SUMMARIZED_TOOLS and self.summarizer is not None:
            self._summarize(call.name, outcome.result or {})
        return outcome


# --------------------------------------------------------------------------- scripted model


def _response(
    message: dict[str, Any], model: str, prompt_tokens: int, completion_tokens: int
) -> TransportResponse:
    finish = "tool_calls" if message.get("tool_calls") else "stop"
    return TransportResponse(
        status_code=200,
        headers={},
        body={
            "id": "scripted-studio",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    )


class ScriptedStudioTransport:
    """A :class:`rtt.llm.providers.Transport` that plays a Studio session offline (deterministic).

    The agent plans the use case of the user message, runs the tournament on a valid plan, exports
    the build pack and answers from the tools' results; Lightning summary requests get a templated
    sentence; a forced ``submit_task_plan`` call (a live plan) is answered with ``plan_args`` when
    given. Usage is reported, so caps and ledgers see real token counts.

    Args:
        plan_args: Task Engineer tool arguments for a live plan (``None``: answer with text only).
        n_conditions: Conditions the scripted agent asks the tournament for (``None``: the default).
        bad_first_call: Make the agent's first tool call carry invalid arguments.
    """

    def __init__(
        self,
        *,
        plan_args: Mapping[str, Any] | None = None,
        n_conditions: int | None = None,
        bad_first_call: bool = False,
    ) -> None:
        self.plan_args = dict(plan_args) if plan_args is not None else None
        self.n_conditions = n_conditions
        self.bad_first_call = bad_first_call
        self.payloads: list[dict[str, Any]] = []
        self._calls = 0
        self._lock = threading.Lock()

    def list_models(self, timeout_s: float) -> TransportResponse:
        return TransportResponse(200, {}, {"data": []})

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> TransportResponse:
        with self._lock:
            self.payloads.append(dict(payload))
            self._calls += 1
            call_n = self._calls
        prompt_tokens = len(json.dumps(payload.get("messages", []))) // 4 + 200
        model = str(payload.get("model"))
        if model == NEMOTRON_LIGHTNING or "tools" not in payload:
            text = "Scripted summary: the numbers are in the step above."
            return _response({"role": "assistant", "content": text}, model, prompt_tokens, 30)
        choice = payload.get("tool_choice")
        if isinstance(choice, Mapping):  # the Task Engineer's forced submit_task_plan call
            name = str((choice.get("function") or {}).get("name") or "submit_task_plan")
            if self.plan_args is None:
                return _response({"role": "assistant", "content": "No plan."}, model, prompt_tokens, 10)
            call = {
                "id": "call_plan",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(self.plan_args)},
            }
            return _response(
                {"role": "assistant", "content": None, "tool_calls": [call]}, model, prompt_tokens, 900
            )
        return _response(self._agent_turn(payload, call_n), model, prompt_tokens, 120)

    def _agent_turn(self, payload: Mapping[str, Any], call_n: int) -> dict[str, Any]:
        messages = list(payload.get("messages", []))
        use_case = ""
        for m in messages:
            content = m.get("content")
            if m.get("role") == "user" and isinstance(content, str) and "Use case:\n" in content:
                use_case = content.split("Use case:\n", 1)[1]
        names: dict[str, str] = {}
        results: list[tuple[str, dict[str, Any]]] = []
        for m in messages:
            if m.get("role") == "assistant":
                for c in m.get("tool_calls") or []:
                    names[c["id"]] = c["function"]["name"]
            elif m.get("role") == "tool":
                results.append((names.get(m.get("tool_call_id", ""), "?"), json.loads(m["content"])))
        if payload.get("tool_choice") == "none":
            return {"role": "assistant", "content": self.final(results)}
        if not results:
            if self.bad_first_call:
                return self._call("plan_task", {"description": use_case}, call_n)
            return self._call("plan_task", {"text": use_case}, call_n)
        last_name, last = results[-1]
        retry = last_name == "plan_task" and last.get("code") == "invalid_args"
        if retry and sum(1 for n, _ in results if n == "plan_task") == 1:
            return self._call("plan_task", {"text": use_case}, call_n)
        if not last.get("ok"):
            return {"role": "assistant", "content": self.final(results)}
        result = last["result"]
        if last_name == "plan_task" and result.get("status") == "valid":
            args: dict[str, Any] = {"plan_id": result["plan_id"]}
            if self.n_conditions is not None:
                args["n_conditions"] = self.n_conditions
            return self._call("run_task_tournament", args, call_n)
        if last_name == "run_task_tournament":
            return self._call("export_build_pack", {"tournament_id": result["tournament_id"]}, call_n)
        return {"role": "assistant", "content": self.final(results)}

    @staticmethod
    def _call(name: str, args: Mapping[str, Any], n: int) -> dict[str, Any]:
        call = {
            "id": f"call_{n}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        return {"role": "assistant", "content": "", "tool_calls": [call]}

    @staticmethod
    def final(results: Sequence[tuple[str, dict[str, Any]]]) -> str:
        """The scripted final answer, built from the tools' results only."""

        def last(name: str) -> dict[str, Any] | None:
            return next((r["result"] for n, r in reversed(results) if n == name and r.get("ok")), None)

        def error(name: str) -> str:
            hit = next((r for n, r in reversed(results) if n == name and not r.get("ok")), None)
            return str(hit.get("error")) if hit else "it was not run"

        plan, tour, pack = last("plan_task"), last("run_task_tournament"), last("export_build_pack")
        tail = " Product mode, not the pre-registered experiment."
        if plan is None:
            return f"No plan was made: {error('plan_task')}.{tail}"
        if plan["status"] != "valid":
            why = plan.get("refusal") or "; ".join(plan.get("problems") or []) or plan["result"]
            return f"The use case was not planned ({plan['result']}): {why}.{tail}"
        if tour is None:
            return f"The tournament did not finish: {error('run_task_tournament')}.{tail}"
        if tour["verdict"] == "recommended":
            r = tour["recommendation"]
            text = (
                f"Recommended setup: {r['label']}, completed {r['completed_runs']}/{r['total_runs']} runs "
                f"in simulation, fidelity {r['fidelity']}"
            )
            text += " (verify on hardware)." if r.get("verify_on_hardware") else "."
            b = tour.get("best_single")
            if b:
                text += (
                    f" Best single off-the-shelf baseline: {b['label']}, "
                    f"{b['completed_runs']}/{b['total_runs']} runs."
                )
        else:
            text = "No setup completes the task in simulation."
            closest = (tour.get("no_recommendation") or {}).get("closest") or []
            if closest:
                c = closest[0]
                text += f" Closest: {c['label']}, {c['completed_runs']}/{c['total_runs']} runs"
                text += f", stops at {c['stops_at']['step_id']}." if c.get("stops_at") else "."
        if pack is not None:
            text += f" Build pack {pack['pack_id']}."
        return text + tail


# --------------------------------------------------------------------------- recording driver


def hero_text(case_id: str, manifest: str | Path = RECORDED_GATE / "manifest.json") -> str:
    """The description of ``case_id`` in the recorded gate manifest (``KeyError`` if missing)."""
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for case in data.get("cases") or []:
        if case.get("case_id") == case_id:
            return str(case["text"])
    raise KeyError(f"case {case_id!r} is not in the recorded gate manifest")


class _LazyTokenFactory:
    """Builds the Token Factory transport from an explicit ``.env`` on the first live call (the key
    is never printed or recorded)."""

    def __init__(self, env_file: Path) -> None:
        self._env_file = Path(env_file)
        self._inner: OpenAITransport | None = None
        self._lock = threading.Lock()

    def _get(self) -> OpenAITransport:
        with self._lock:
            if self._inner is None:
                self._inner = OpenAITransport.from_env(self._env_file)
            return self._inner

    def chat_completions(self, payload: Mapping[str, Any], timeout_s: float) -> Any:
        return self._get().chat_completions(payload, timeout_s)

    def list_models(self, timeout_s: float) -> Any:
        return self._get().list_models(timeout_s)

    def __repr__(self) -> str:
        return "LazyTokenFactory(env_file=<set>)"


class _EchoRecorder(SessionRecorder):
    """A session recorder that prints one line per event (as ``python -m rtt.agent`` does)."""

    def emit(self, type_: str, **data: Any) -> dict[str, Any]:
        from rtt.agent.__main__ import _line

        event = super().emit(type_, **data)
        try:
            print(_line(event), flush=True)
        except (KeyError, TypeError, ValueError):
            print(f"[{event['t_s']:7.1f}s] {type_}", flush=True)
        return event


def record_studio_session(
    case_id: str,
    out_dir: str | Path,
    *,
    live: bool = False,
    env_file: str | Path | None = None,
    budget_usd: float = DEFAULT_SESSION_USD,
    workers: int = HARD_MAX_WORKERS,
    wall_s: float = DEFAULT_WALL_S,
    transport: Any | None = None,
    tournament: Callable[..., Any] | None = None,
    export: Callable[..., Any] | None = None,
    echo: bool = False,
    stop: threading.Event | None = None,
) -> tuple[SessionResult, Path]:
    """Record one Studio session of a hero use case; returns its result and the session file.

    The description is the recorded gate manifest's (:func:`hero_text`). The session is not unlocked:
    a recorded plan is free, the tournament runs locally for free, so only the agent's turns and the
    summaries can spend. Offline (the default) the model is :class:`ScriptedStudioTransport`
    (recorded as ``scripted-offline``) and nothing is billed; ``live`` uses Token Factory with the
    key from ``env_file``, read on the first call. Spend goes through a :class:`SpendGuard` whose
    session cap is ``budget_usd``, on a day book under ``out_dir/store`` capped at the same.

    Layout: ``out_dir/<case_id>.<session_id>.json`` (the session) and ``out_dir/store/`` (plans,
    tournament reports, build packs, the daily spend book and, live, the call ledger).
    """
    if case_id not in HERO_CASE_IDS:
        raise ValueError(f"case must be one of {', '.join(HERO_CASE_IDS)}")
    if not math.isfinite(budget_usd) or budget_usd < 0:
        raise ValueError("budget_usd must be a finite non-negative number")
    if live and transport is None and env_file is None:
        raise ValueError("a live session needs env_file (the .env holding NEBIUS_API_KEY)")
    text = hero_text(case_id)
    out = Path(out_dir)
    store = AgentStore(out / "store")
    session_id = new_id("session")
    offline = not live
    if transport is None:
        transport = (
            _LazyTokenFactory(Path(env_file)) if live and env_file is not None else ScriptedStudioTransport()
        )
    caps = AgentCaps(max_tool_calls=STUDIO_MAX_TOOL_CALLS, session_usd=budget_usd, day_usd=budget_usd)
    day_book = DailySpendBook(store.daily_spend_path, budget_usd)
    guard = SpendGuard(budget_usd, day_book, session_id, billing=live)
    ledger = GuardedLedger(guard, store.ledger_path if live else None)
    path = out / f"{case_id}.{session_id}.json"
    recorder_cls = _EchoRecorder if echo else SessionRecorder
    recorder = recorder_cls(
        session_id, path, meta={"kind": "studio", "case_id": case_id, "use_case": text, "offline": offline}
    )
    studio = StudioTools(
        caps=StudioCaps(workers=workers, wall_s=wall_s),
        tournament=tournament,
        export=export,
        transport=transport if live else None,
        live_plans=live,
    )
    toolbox = studio_toolbox(
        session_id=session_id,
        store=store,
        caps=caps,
        guard=guard,
        studio=studio,
        unlocked=False,
        ledger=ledger,
        recorder=recorder,
        offline=offline,
    )
    engineer = StudioEngineer(
        AgentLLM.create(ledger, transport=transport),
        toolbox,
        recorder,
        caps=caps,
        use_case=text,
        case_id=case_id,
        summarizer=StudioSummarizer.create(ledger, transport=transport),
        stop=stop,
        offline=offline,
    )
    return engineer.run(), path


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m rtt.agent.studio_tool record --case CASE --out DIR [...]`` (module docstring)."""
    ap = argparse.ArgumentParser(
        prog="python -m rtt.agent.studio_tool",
        description="Record a Grasp Engineer session in the Studio (product mode).",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="record one Studio session of a hero use case")
    rec.add_argument("--case", required=True, choices=HERO_CASE_IDS)
    rec.add_argument("--out", required=True, type=Path, help="output folder (session file and store)")
    rec.add_argument(
        "--live", action="store_true", help="Nemotron on Token Factory (default: scripted, free)"
    )
    rec.add_argument("--env-file", type=Path, default=None, help=".env with NEBIUS_API_KEY (with --live)")
    rec.add_argument("--budget", type=float, default=DEFAULT_SESSION_USD, help="session and day cap (USD)")
    rec.add_argument("--workers", type=int, default=HARD_MAX_WORKERS, help=f"1..{HARD_MAX_WORKERS}")
    rec.add_argument("--wall-s", type=float, default=DEFAULT_WALL_S, help=f"at most {HARD_MAX_WALL_S:.0f}")
    a = ap.parse_args(argv)
    if a.live and a.env_file is None:
        ap.error("--live needs --env-file")
    if not 1 <= a.workers <= HARD_MAX_WORKERS:
        ap.error(f"--workers must be in 1..{HARD_MAX_WORKERS}")
    if not (math.isfinite(a.wall_s) and 0 < a.wall_s <= HARD_MAX_WALL_S):
        ap.error(f"--wall-s must be in (0, {HARD_MAX_WALL_S:.0f}]")
    if not (math.isfinite(a.budget) and a.budget >= 0):
        ap.error("--budget must be a non-negative number")
    print(
        f"studio session: case {a.case}, {'live' if a.live else 'scripted offline'}, budget ${a.budget:.2f}"
    )
    result, path = record_studio_session(
        a.case,
        a.out,
        live=a.live,
        env_file=a.env_file,
        budget_usd=a.budget,
        workers=a.workers,
        wall_s=a.wall_s,
        echo=True,
    )
    print(
        f"{result.status}: spent ${result.spend['spent_usd']:.4f} of ${a.budget:.2f} "
        f"({result.tool_calls} tool calls, {result.turns} model turns); session file {path}"
    )
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
