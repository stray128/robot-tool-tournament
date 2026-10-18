"""The Grasp Engineer's tools: argument validation, hard caps, and calls into real code.

:class:`Toolbox` is bound to one session and one part. :meth:`Toolbox.call` is the only
way the agent acts, and it applies every limit in order:

1. **Tool budget**: at most ``caps.max_tool_calls`` calls per session; a call past the
   budget is refused (and not counted). Every other call counts, valid or not.
2. **Arguments**: validated against the tool's schema (:mod:`rtt.agent.schemas`); IDs
   must have their fixed shapes, and the part must be this session's part.
3. **Passcode**: a tool that spends (``propose_designs``; ``launch_sim_job`` on Nebius)
   needs a session unlocked with the passcode.
4. **Dollars**: spend is reserved against the session and day caps before any call or
   job starts (:class:`rtt.agent.caps.SpendGuard`).
5. **Concurrency**: at most ``caps.max_concurrent_jobs`` jobs in flight.

Physics decides pass/fail. Each simulated design gets ``verdict`` = ``pass`` when its
success rate at its best grasp height reaches :data:`PASS_THRESHOLD`, and each
certificate gets ``verdict`` = ``pass`` when the nominal grasp holds (in every certified
pose) and its score reaches :data:`CERT_PASS_SCORE`. The agent only orchestrates and explains; the
verdicts it reports are these, recorded by code in the session file.

With an effector runner (``effectors``, product mode), a ninth tool, ``recommend_effector``,
starts an effector tournament (:mod:`rtt.agent.effector_tool`). It shares the caps above: it
counts toward the tool calls, its jobs count toward the jobs in flight, and ``design = true``
(Nemotron morphologies) needs the passcode and reserves its worst case before it starts. Its
``verdict`` is ``recommended`` or ``no_winner``, computed by :mod:`rtt.effectors.recommend`.

With Studio tools (``studio``, product mode, :mod:`rtt.agent.studio_tool`), three more tools are
registered: ``plan_task``, ``run_task_tournament`` and ``export_build_pack``. Their arguments are
validated by the Studio object's own schemas, the passcode rule is its ``may_spend`` (``plan_task``
is free for a recorded description), and a Studio-only session refuses every other tool but
``cost_ledger`` as not offered. Sessions without ``studio`` are unchanged.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

from rtt.agent import effector_tool as et
from rtt.agent.backends import PartBackend, PartNotFoundError
from rtt.agent.caps import (
    AgentCaps,
    CapError,
    ConcurrencyCapError,
    PasscodeRequiredError,
    SpendCapExceededError,
    SpendGuard,
    ToolBudgetExhaustedError,
)
from rtt.agent.jobs import JobRecord, JobRunner, UnknownJobError
from rtt.agent.proposer import DesignProposer, SimulatedDesign
from rtt.agent.recorder import PRODUCT_MODE_LABEL, SCRIPTED_MODEL, SessionRecorder, redact_paths
from rtt.agent.schemas import (
    ArgsError,
    CostLedgerArgs,
    ExportPrintPackArgs,
    FetchResultsArgs,
    InspectPartArgs,
    LaunchSimJobArgs,
    PollJobArgs,
    ProposeDesignsArgs,
    RecommendEffectorArgs,
    StressTestArgs,
    tool_spec,
    validate_args,
)
from rtt.agent.store import AgentStore, new_id
from rtt.design.schema import FingerDesign
from rtt.design.validate import validate
from rtt.llm.client import CallMeta
from rtt.llm.ledger import Ledger
from rtt.search.base import DesignScore

if TYPE_CHECKING:  # the Studio module imports this one, so it is never imported here at run time
    from rtt.agent.studio_tool import StudioTools

__all__ = ["CERT_PASS_SCORE", "PASS_THRESHOLD", "DesignEntry", "ToolOutcome", "Toolbox"]

#: A design passes when its simulated success at its best grasp height reaches this.
PASS_THRESHOLD: Final = 0.8
#: A certificate passes when the nominal grasp holds and its score reaches this.
CERT_PASS_SCORE: Final = 0.75
_MAX_ERROR_CHARS: Final = 400


@dataclass
class DesignEntry:
    """A design the session knows: a baseline entrant or a proposal."""

    design: FingerDesign
    origin: str  # "baseline" | "proposed"
    label: str
    hypothesis: str | None = None
    score: DesignScore | None = None
    certificate: dict[str, Any] | None = None

    @property
    def design_id(self) -> str:
        return self.design.design_id()


@dataclass
class _JobEntry:
    record: JobRecord
    reserved_usd: float
    settled: bool = False
    run_id: str | None = None
    kind: str = "design"  # "design" (launch_sim_job) or "effector" (recommend_effector)
    effector_result: dict[str, Any] | None = None
    charged_usd: float = 0.0  # what the settled job was charged


@dataclass(frozen=True)
class ToolOutcome:
    """Result of one tool call, as recorded and as shown to the model."""

    name: str
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    code: str | None = None
    args: dict[str, Any] | None = None
    refused: bool = False
    duration_s: float = 0.0
    calls_left: int = 0

    def for_model(self) -> str:
        """The tool message content: compact JSON with the calls left."""
        body: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            body["result"] = self.result
        else:
            body["error"] = self.error
            body["code"] = self.code
        body["tool_calls_left"] = self.calls_left
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"), default=str)

    def to_event(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "ok": self.ok, "duration_s": round(self.duration_s, 3)}
        if self.ok:
            data["result"] = self.result
        else:
            data["error"] = self.error
            data["code"] = self.code
        data["tool_calls_left"] = self.calls_left
        return data


def _short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}" if not isinstance(exc, CapError) else str(exc)
    return redact_paths(text)[:_MAX_ERROR_CHARS]


def _pct(value: float) -> float:
    return round(100.0 * value, 1)


def _episodes_at_best_height(row: Mapping[str, Any]) -> int:
    """Episodes behind a design's ``success`` (those at its best grasp height ``best_k``).

    ``success`` is the success rate at the best grasp height only, so the episode count
    shown next to it must be that height's, not the total over every height.
    """
    episodes = row.get("episodes") or []
    best_k = row.get("best_k")
    if episodes and best_k is not None:
        return sum(1 for e in episodes if e.get("controller_setting") == best_k)
    heights = len(row.get("success_by_k") or ()) or 1
    return int(row.get("n_episodes", 0)) // heights


class Toolbox:
    """One session's tools (see the module docstring).

    Args:
        session_id: The session's ID (for spend records).
        part_id: The only part this session may work on.
        backend: Physics services.
        proposer: Design proposer (Nemotron 3 Super in production).
        runners: Job runners by target (``"local"``, optionally ``"nebius"``).
        store: Where run results and print packs are written.
        caps: Session limits.
        guard: The session's spend guard.
        unlocked: Whether the session was unlocked with the passcode.
        ledger: The session's model-call ledger (for ``cost_ledger``).
        recorder: Session recorder (``job.update`` and proposer ``llm.call`` events).
        baselines: Entrant designs offered for comparison.
        pass_threshold: Success rate a design needs for a ``pass`` verdict.
        offline: Model calls come from the scripted transport (recorded as such).
        effectors: Effector tournament jobs (enables ``recommend_effector``; ``None``: not offered).
        design_cap_usd: Nemotron design spend this session may commit over all its
            ``recommend_effector(design=true)`` jobs (reservations in flight plus charges).
        studio: Studio tools (:class:`rtt.agent.studio_tool.StudioTools`; registers ``plan_task``,
            ``run_task_tournament`` and ``export_build_pack``; ``None``: not offered).
    """

    def __init__(
        self,
        *,
        session_id: str,
        part_id: str,
        backend: PartBackend,
        proposer: DesignProposer,
        runners: Mapping[str, JobRunner],
        store: AgentStore,
        caps: AgentCaps,
        guard: SpendGuard,
        unlocked: bool,
        ledger: Ledger | None = None,
        recorder: SessionRecorder | None = None,
        baselines: Sequence[FingerDesign] = (),
        pass_threshold: float = PASS_THRESHOLD,
        offline: bool = False,
        clock: Callable[[], float] = time.monotonic,
        effectors: et.EffectorJobRunner | None = None,
        design_cap_usd: float = et.DESIGN_SESSION_USD,
        studio: StudioTools | None = None,
    ) -> None:
        self.offline = offline
        self.effectors = effectors
        self.studio = studio
        self.design_cap_usd = float(design_cap_usd)
        self.session_id = session_id
        self.part_id = part_id
        self.backend = backend
        self.proposer = proposer
        self.runners = dict(runners)
        self.store = store
        self.caps = caps
        self.guard = guard
        self.unlocked = unlocked
        self.ledger = ledger
        self.recorder = recorder
        self.pass_threshold = pass_threshold
        self._clock = clock
        self._lock = threading.RLock()
        self.calls_used = 0
        self.designs: dict[str, DesignEntry] = {}
        self._jobs: dict[str, _JobEntry] = {}
        self._runs: dict[str, dict[str, Any]] = {}
        self._effector_verdicts: dict[str, dict[str, Any]] = {}
        for design in baselines:
            entry = DesignEntry(design, "baseline", design.name or design.design_id())
            self.designs.setdefault(entry.design_id, entry)
        self._handlers: dict[str, Callable[[Any], dict[str, Any]]] = {
            "inspect_part": self._inspect_part,
            "propose_designs": self._propose_designs,
            "launch_sim_job": self._launch_sim_job,
            "poll_job": self._poll_job,
            "fetch_results": self._fetch_results,
            "stress_test": self._stress_test,
            "export_print_pack": self._export_print_pack,
            "cost_ledger": self._cost_ledger,
            "recommend_effector": self._recommend_effector,
        }
        if studio is not None:
            self._handlers.update(studio.bind(self))

    # ------------------------------------------------------------------ dispatch

    @property
    def calls_left(self) -> int:
        with self._lock:
            return max(0, self.caps.max_tool_calls - self.calls_used)

    def call(self, name: str, raw_args: Any) -> ToolOutcome:
        """Run one tool call under every cap; never raises for model mistakes."""
        t0 = self._clock()
        with self._lock:
            if self.calls_used >= self.caps.max_tool_calls:
                exc = ToolBudgetExhaustedError(
                    f"the session's {self.caps.max_tool_calls} tool calls are used up; write your final answer"
                )
                return ToolOutcome(name, False, error=str(exc), code=exc.code, refused=True, calls_left=0)
            self.calls_used += 1
        args_dict: dict[str, Any] | None = None
        try:
            studio = self.studio if self.studio is not None and self.studio.owns(name) else None
            if self.studio is not None:
                self.studio.check_offered(name)  # a Studio-only session refuses every other tool
            args = studio.validate(name, raw_args) if studio is not None else validate_args(name, raw_args)
            args_dict = args.model_dump()
            self._check_part(name, args)
            if studio is not None:
                spends = studio.may_spend(name, args)
            else:
                spends = tool_spec(name).may_spend and not self._spend_free(name, args)
            if spends and not self.unlocked:
                raise PasscodeRequiredError(
                    studio.passcode_text(name)
                    if studio is not None
                    else f"{name} costs money; this session was not unlocked with the passcode"
                )
            result = self._handlers[name](args)
        except ArgsError as exc:
            return self._outcome(name, t0, error=str(exc), code="invalid_args", args=args_dict)
        except CapError as exc:
            return self._outcome(name, t0, error=str(exc), code=exc.code, args=args_dict, refused=True)
        except (PartNotFoundError, UnknownJobError, KeyError) as exc:
            return self._outcome(name, t0, error=f"not found: {exc}", code="not_found", args=args_dict)
        except Exception as exc:  # noqa: BLE001 - a failed tool is reported to the model, not raised
            return self._outcome(name, t0, error=_short_error(exc), code="error", args=args_dict)
        return ToolOutcome(
            name,
            True,
            result=result,
            args=args_dict,
            duration_s=self._clock() - t0,
            calls_left=self.calls_left,
        )

    def _outcome(
        self,
        name: str,
        t0: float,
        *,
        error: str,
        code: str,
        args: dict[str, Any] | None,
        refused: bool = False,
    ) -> ToolOutcome:
        return ToolOutcome(
            name,
            False,
            error=error,
            code=code,
            args=args,
            refused=refused,
            duration_s=self._clock() - t0,
            calls_left=self.calls_left,
        )

    def _check_part(self, name: str, args: Any) -> None:
        part_id = getattr(args, "part_id", None)
        if part_id is not None and part_id != self.part_id:
            raise ArgsError(name, [f"part_id: this session works on {self.part_id}, not {part_id}"])

    def _spend_free(self, name: str, args: Any) -> bool:
        """``launch_sim_job`` on a free target and ``recommend_effector`` without Nemotron need no passcode."""
        if name == "recommend_effector":
            return not args.design
        if name != "launch_sim_job":
            return False
        runner = self.runners.get(args.target)
        return runner is not None and runner.estimate_usd(len(args.designs)) <= 0.0

    # ------------------------------------------------------------------ helpers

    def _entry(self, design_id: str) -> DesignEntry:
        entry = self.designs.get(design_id)
        if entry is None:
            raise KeyError(f"design {design_id} (use IDs from propose_designs or the baselines list)")
        return entry

    def _emit(self, type_: str, **data: Any) -> None:
        if self.recorder is not None:
            self.recorder.emit(type_, **data)

    def emit_llm_call(self, role: str, meta: CallMeta) -> None:
        """Record one model call made on the session's behalf."""
        self._emit(
            "llm.call",
            role=role,
            model=SCRIPTED_MODEL if self.offline else meta.model,
            prompt_tokens=meta.prompt_tokens,
            completion_tokens=meta.completion_tokens,
            reasoning_tokens=meta.reasoning_tokens,
            cost_usd=0.0 if self.offline else round(meta.cost_usd, 6),  # scripted: nothing billed
            latency_s=meta.latency_s,
            cached=meta.cached,
            finish_reason=meta.finish_reason,
        )

    def _runner_for(self, entry: _JobEntry) -> JobRunner:
        if entry.kind == "effector":
            assert self.effectors is not None
            return self.effectors
        return self.runners[entry.record.target]

    def _settle(self, entry: _JobEntry, record: JobRecord) -> None:
        """Turn a finished job's reservation into its estimated charge (once).

        An effector job's reservation is its Nemotron worst case: it is charged what the
        proposals cost (``llm``), or the whole reservation when the job did not complete."""
        entry.record = record
        if record.terminal and not entry.settled:
            entry.settled = True
            if entry.reserved_usd > 0:
                self.guard.release(entry.reserved_usd)
            if entry.kind == "effector":
                spent = entry.reserved_usd
                if record.state == "COMPLETED":
                    try:
                        entry.effector_result = dict(self._runner_for(entry).results(record.job_id))
                        spent = min(entry.reserved_usd, float(entry.effector_result.get("llm_usd", 0.0)))
                    except Exception:  # noqa: BLE001 - keep the reservation as the charge
                        spent = entry.reserved_usd
                entry.charged_usd = spent
                self.guard.charge(spent, "llm")
                return
            self.guard.charge(record.est_usd, "compute")

    def _active_jobs(self) -> int:
        active = 0
        for job_id, entry in self._jobs.items():
            if entry.settled:
                continue
            record = self._runner_for(entry).status(job_id)
            if record.state != entry.record.state:
                self._emit("job.update", **record.to_json())
            self._settle(entry, record)
            if not record.terminal:
                active += 1
        return active

    def history(self) -> list[SimulatedDesign]:
        """Designs simulated in this session, for the proposer."""
        out = []
        for entry in self.designs.values():
            if entry.score is None and entry.certificate is None:
                continue
            out.append(
                SimulatedDesign(
                    design=entry.design,
                    label=entry.label if entry.origin == "baseline" else entry.design_id,
                    success=entry.score.success if entry.score else None,
                    failure_modes=dict(entry.score.failure_modes) if entry.score else {},
                    certificate_score=float(entry.certificate["score"]) if entry.certificate else None,
                )
            )
        return out

    def verdicts(self) -> list[dict[str, Any]]:
        """Code-computed verdicts for every simulated design and every fetched effector
        recommendation (what the session file reports)."""
        out = []
        for entry in self.designs.values():
            if entry.score is None and entry.certificate is None:
                continue
            row: dict[str, Any] = {"design_id": entry.design_id, "label": entry.label, "origin": entry.origin}
            if entry.score is not None:
                row["success_pct"] = _pct(entry.score.success)
                row["verdict"] = self._verdict(entry.score)
            if entry.certificate is not None:
                row["certificate_score_pct"] = _pct(float(entry.certificate.get("score", 0.0)))
                row["certificate_verdict"] = entry.certificate.get("verdict")
            out.append(row)
        out += [dict(v) for v in self._effector_verdicts.values()]
        if self.studio is not None:  # one row per Studio tournament, computed by code
            out += self.studio.verdicts()
        return out

    def _verdict(self, score: DesignScore) -> str:
        return "pass" if score.success >= self.pass_threshold - 1e-12 else "fail"

    # ------------------------------------------------------------------ tools

    def _inspect_part(self, args: InspectPartArgs) -> dict[str, Any]:
        info = self.backend.info(args.part_id)
        task = self.backend.gripper_task(args.part_id)
        baselines = []
        for entry in self.designs.values():
            if entry.origin != "baseline":
                continue
            check = validate(entry.design, task)
            row: dict[str, Any] = {
                "design_id": entry.design_id,
                "name": entry.label,
                "valid_for_part": check.ok,
            }
            if not check.ok:
                row["why_not"] = check.errors[:2]
            baselines.append(row)
        part = info.to_json()
        part.pop("backend", None)
        return {
            "part": part,
            "pass_threshold_pct": _pct(self.pass_threshold),
            "baselines": baselines,
            "targets": sorted(self.runners),
        }

    def _propose_designs(self, args: ProposeDesignsArgs) -> dict[str, Any]:
        if args.n > self.caps.max_designs_per_proposal:
            raise ArgsError("propose_designs", [f"n: at most {self.caps.max_designs_per_proposal}"])
        info = self.backend.info(args.part_id)
        task = self.backend.gripper_task(args.part_id)
        known = [e.design for e in self.designs.values()]
        batch = self.proposer.propose(info, args.n, gripper_task=task, history=self.history(), known=known)
        for meta in batch.calls:
            self.emit_llm_call("proposer", meta)
        designs = []
        for proposed in batch.designs:
            entry = DesignEntry(proposed.design, "proposed", proposed.design.design_id(), proposed.hypothesis)
            self.designs.setdefault(entry.design_id, entry)
            designs.append(
                {
                    "design_id": entry.design_id,
                    "hypothesis": proposed.hypothesis,
                    "params": proposed.design.geometry_params(),
                }
            )
        return {"designs": designs, "rejected": list(batch.rejections), "analysis": batch.analysis}

    def _launch_sim_job(self, args: LaunchSimJobArgs) -> dict[str, Any]:
        runner = self.runners.get(args.target)
        if runner is None:
            raise ArgsError(
                "launch_sim_job",
                [f"target {args.target!r} is not available here; use {sorted(self.runners)}"],
            )
        if len(args.designs) > self.caps.max_designs_per_job:
            raise ArgsError("launch_sim_job", [f"designs: at most {self.caps.max_designs_per_job} per job"])
        entries = [self._entry(d) for d in args.designs]
        task = self.backend.gripper_task(args.part_id)
        for entry in entries:
            check = validate(entry.design, task)
            if not check.ok:
                raise ArgsError(
                    "launch_sim_job", [f"{entry.design_id} does not fit this part: {check.errors[0]}"]
                )
        with self._lock:
            active = self._active_jobs()
            if active >= self.caps.max_concurrent_jobs:
                raise ConcurrencyCapError(
                    f"{active} jobs are already in flight (limit {self.caps.max_concurrent_jobs}); poll one first"
                )
            estimate = runner.estimate_usd(len(entries))
            if estimate > 0:
                self.guard.reserve(estimate)
            try:
                record = runner.launch(args.part_id, [e.design for e in entries])
            except BaseException:
                if estimate > 0:
                    self.guard.release(estimate)
                raise
            self._jobs[record.job_id] = _JobEntry(record, estimate)
        self._emit("job.update", **record.to_json())
        return record.to_json()

    def _poll_job(self, args: PollJobArgs) -> dict[str, Any]:
        entry = self._jobs.get(args.job_id)
        if entry is None:
            raise UnknownJobError(args.job_id)
        before = entry.record.state
        record = self._runner_for(entry).wait(args.job_id, self.caps.poll_wait_s)
        with self._lock:
            self._settle(entry, record)
        if record.state != before:
            self._emit("job.update", **record.to_json())
        return record.to_json()

    def _fetch_results(self, args: FetchResultsArgs) -> dict[str, Any]:
        entry = self._jobs.get(args.job_id)
        if entry is None:
            raise UnknownJobError(args.job_id)
        runner = self._runner_for(entry)
        record = runner.status(args.job_id)
        with self._lock:
            self._settle(entry, record)
        if record.state != "COMPLETED":
            raise RuntimeError(f"job {args.job_id} is {record.state}; poll_job until it is COMPLETED")
        if entry.kind == "effector":
            return self._effector_results(entry)
        if entry.run_id is not None:
            return self._compact_run(self._runs[entry.run_id])
        scores = runner.results(args.job_id)
        by_id = {s.design_id: s for s in scores}
        rows = []
        for design_id in record.design_ids:
            score = by_id.get(design_id)
            if score is None:
                raise RuntimeError(f"the job returned no score for design {design_id}")
            design_entry = self._entry(design_id)
            design_entry.score = score
            rows.append((design_entry, score))
        rows.sort(key=lambda r: r[1].rank_key(), reverse=True)
        run_id = new_id("run")
        doc = {
            "run_id": run_id,
            "session_id": self.session_id,
            "label": PRODUCT_MODE_LABEL,
            "part_id": record.part_id,
            "job": record.to_json(),
            "pass_threshold": self.pass_threshold,
            "best_design_id": rows[0][1].design_id,
            "designs": [
                {
                    "design_id": score.design_id,
                    "label": e.label,
                    "origin": e.origin,
                    "hypothesis": e.hypothesis,
                    "params": e.design.geometry_params(),
                    "verdict": self._verdict(score),
                    **score.to_json(include_episodes=True),
                }
                for e, score in rows
            ],
        }
        self.store.save_run(run_id, doc)
        with self._lock:
            entry.run_id = run_id
            self._runs[run_id] = doc
        return self._compact_run(doc)

    def _compact_run(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        designs = []
        for row in doc["designs"]:
            designs.append(
                {
                    "design_id": row["design_id"],
                    "label": row["label"],
                    "success_pct": _pct(row["success"]),
                    "verdict": row["verdict"],
                    "success_by_grasp_height_pct": [_pct(v) for v in row["success_by_k"]],
                    "hold_fraction": round(row["hold_fraction"], 3),
                    "mass_g": round(row["mass_g"], 1),
                    "failure_modes": row["failure_modes"],
                    # success_pct and failure_modes are over the best grasp height's episodes
                    "episodes": _episodes_at_best_height(row),
                    "episodes_all_heights": row["n_episodes"],
                }
            )
        return {
            "run_id": doc["run_id"],
            "job_id": doc["job"]["job_id"],
            "target": doc["job"]["target"],
            "pass_threshold_pct": _pct(doc["pass_threshold"]),
            "best_design_id": doc["best_design_id"],
            "designs": designs,
        }

    def _stress_test(self, args: StressTestArgs) -> dict[str, Any]:
        entry = self._entry(args.design_id)
        task = self.backend.gripper_task(args.part_id)
        check = validate(entry.design, task)
        if not check.ok:
            raise ArgsError("stress_test", [f"{entry.design_id} does not fit this part: {check.errors[0]}"])
        cert = dict(self.backend.certify(args.part_id, entry.design, score=entry.score))
        score = float(cert.get("score") or 0.0)
        tests = cert.get("tests") or []
        nominals = [
            bool(t.get("success")) for t in tests if str(t.get("name", "")).split("/")[-1] == "nominal"
        ]
        nominal_ok = all(nominals) if nominals else score > 0
        if "verdict" not in cert:
            cert["verdict"] = "pass" if nominal_ok and score >= CERT_PASS_SCORE - 1e-12 else "fail"
        cert["pass_score"] = CERT_PASS_SCORE
        entry.certificate = cert
        failed = [
            {"test": t.get("name"), "failure_mode": t.get("failure_mode")}
            for t in tests
            if not t.get("success")
        ]
        return {
            "design_id": entry.design_id,
            "score_pct": _pct(score),
            "verdict": cert["verdict"],
            "category_pct": {k: _pct(float(v)) for k, v in dict(cert.get("category_scores") or {}).items()},
            "placement_tolerance_mm": cert.get("tolerance_mm"),
            "passed": cert.get("passed"),
            "failed_tests": failed[:8],
        }

    def _export_print_pack(self, args: ExportPrintPackArgs) -> dict[str, Any]:
        doc = self._runs.get(args.run_id)
        if doc is None:
            try:
                doc = self.store.load_run(args.run_id)
            except FileNotFoundError:
                raise KeyError(f"run {args.run_id}") from None
            if doc.get("session_id") != self.session_id:
                raise KeyError(f"run {args.run_id}")
        design_id = args.design_id or doc["best_design_id"]
        row = next((r for r in doc["designs"] if r["design_id"] == design_id), None)
        if row is None:
            raise ArgsError("export_print_pack", [f"design {design_id} is not in run {args.run_id}"])
        entry = self._entry(design_id)
        pack_id = new_id("pack")
        pack_dir = self.store.pack_dir(pack_id)
        results = {k: v for k, v in row.items() if k != "episodes"}
        results.update(run_id=args.run_id, part_id=doc["part_id"], label=PRODUCT_MODE_LABEL)
        files = self.backend.print_pack(
            doc["part_id"], entry.design, pack_dir, results=results, certificate=entry.certificate
        )
        archive = self.store.zip_pack(pack_id, files)
        return {
            "pack_id": pack_id,
            "design_id": design_id,
            "run_id": args.run_id,
            "verdict": row["verdict"],
            "certificate_included": any(f.name == "certificate.json" for f in files),
            # a certificate made at export (no stress_test this session) has no verdict here
            "stress_tested": entry.certificate is not None,
            "certificate_verdict": entry.certificate.get("verdict") if entry.certificate else None,
            "files": [{"name": f.name, "bytes": f.stat().st_size} for f in sorted(files)],
            "zip_bytes": archive.stat().st_size,
            "download": f"/printpack/{pack_id}",
        }

    def _cost_ledger(self, args: CostLedgerArgs) -> dict[str, Any]:
        by_model: dict[str, dict[str, Any]] = {}
        if self.ledger is not None:  # one ledger instance per session: its entries are this session's
            for e in self.ledger.session_entries():
                model = SCRIPTED_MODEL if self.offline else e.model
                row = by_model.setdefault(model, {"calls": 0, "usd": 0.0})
                row["calls"] += 1
                row["usd"] = round(row["usd"] + (0.0 if e.cached or self.offline else e.cost_usd), 6)
        jobs = [
            {
                "job_id": e.record.job_id,
                "target": e.record.target,
                "state": e.record.state,
                "est_usd": round(e.record.est_usd, 4),
            }
            for e in self._jobs.values()
        ]
        out: dict[str, Any] = {
            "spend": self.guard.snapshot(),
            "llm_by_model": by_model,
            "jobs": jobs,
            "tool_calls_used": self.calls_used,
            "tool_calls_left": self.calls_left,
            "caps": self.caps.to_json(),
        }
        if self.studio is not None:
            out["studio_caps"] = self.studio.caps.to_json()
        return out

    def _design_committed(self) -> float:
        """Nemotron design spend this session has committed: reservations of design jobs in
        flight plus what settled ones were charged (call with the lock held)."""
        return sum(
            (e.charged_usd if e.settled else e.reserved_usd)
            for e in self._jobs.values()
            if e.kind == "effector"
        )

    def _recommend_effector(self, args: RecommendEffectorArgs) -> dict[str, Any]:
        if self.effectors is None:
            raise ArgsError("recommend_effector", ["the effector layer is not enabled on this server"])
        live = bool(args.design) and bool(getattr(self.effectors, "designs", True))
        reserve = et.DESIGN_RESERVE_USD if live else 0.0
        request = et.EffectorRequest(
            part_id=args.part_id,
            requirement=args.requirement,
            groups=et.groups_or_none(args.groups),
            design=live,
            budget_usd=reserve,
        )
        with self._lock:
            active = self._active_jobs()
            if active >= self.caps.max_concurrent_jobs:
                raise ConcurrencyCapError(
                    f"{active} jobs are already in flight (limit {self.caps.max_concurrent_jobs}); poll one first"
                )
            if reserve > 0:
                committed = self._design_committed()
                if committed + reserve > self.design_cap_usd + 1e-12:
                    raise SpendCapExceededError(
                        f"refused: ${reserve:.2f} more Nemotron design spend would exceed this session's "
                        f"${self.design_cap_usd:.2f} design cap (${committed:.4f} committed)"
                    )
                self.guard.reserve(reserve)
            try:
                record = self.effectors.launch_request(request)
            except BaseException:
                if reserve > 0:
                    self.guard.release(reserve)
                raise
            self._jobs[record.job_id] = _JobEntry(record, reserve, kind="effector")
        self._emit("job.update", **record.to_json(), kind="effector_tournament")
        return {
            **record.to_json(),
            "kind": "effector_tournament",
            "requirement": args.requirement,
            "groups": list(args.groups) if args.groups else "all",
            "design": live,
            **(
                {
                    "design_note": "this session has no live designer: the Nemotron group is skipped, nothing reserved"
                }
                if args.design and not live
                else {}
            ),
            "reserved_usd": round(reserve, 4),
            "label": PRODUCT_MODE_LABEL,
            "next": "poll_job, then fetch_results",
        }

    def _effector_results(self, entry: _JobEntry) -> dict[str, Any]:
        if entry.effector_result is None:
            entry.effector_result = dict(self._runner_for(entry).results(entry.record.job_id))
        compact = et.compact_result(
            entry.record.job_id, entry.record.part_id, entry.effector_result, CERT_PASS_SCORE
        )
        with self._lock:
            self._effector_verdicts[entry.record.job_id] = et.verdict_row(
                entry.record.job_id, entry.record.part_id, compact
            )
        return compact

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> list[dict[str, Any]]:
        """Cancel jobs still in flight and settle their spend; returns what was cancelled."""
        cancelled = []
        with self._lock:
            for job_id, entry in self._jobs.items():
                if entry.settled:
                    continue
                runner = self._runner_for(entry)
                try:
                    record = runner.status(job_id)
                    if not record.terminal:
                        record = runner.cancel(job_id)
                        cancelled.append(record.to_json())
                except Exception as exc:  # noqa: BLE001 - keep the reservation as the charge
                    record = replace(entry.record, est_usd=max(entry.record.est_usd, entry.reserved_usd))
                    record = replace(record, error=_short_error(exc))
                if record.terminal:
                    self._settle(entry, record)
                else:  # still running (a local thread cannot be interrupted): charge the reservation
                    entry.settled = True
                    if entry.reserved_usd > 0:
                        entry.charged_usd = entry.reserved_usd
                        self.guard.release(entry.reserved_usd)
                        self.guard.charge(
                            entry.reserved_usd, "llm" if entry.kind == "effector" else "compute"
                        )
        for record_json in cancelled:
            self._emit("job.update", **record_json)
        return cancelled
