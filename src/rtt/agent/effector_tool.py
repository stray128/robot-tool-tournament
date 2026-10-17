"""``recommend_effector``: the Grasp Engineer's effector tournament (product mode, ``docs/EFFECTORS.md`` 7).

The tool starts a tournament job on this machine (CPU MuJoCo, free) and returns its job ID;
``poll_job`` waits for it and ``fetch_results`` returns the recommendation. The verdict is
computed by code (:mod:`rtt.effectors.recommend`): the model only explains it.

* :class:`EffectorRequest`: what the model asked for (the session's part, the requirement, the
  groups, whether Nemotron designs morphologies).
* :class:`EffectorJobRunner`: tournament jobs in a small thread pool, with the same job records,
  states and IDs (``local-`` + 12 hex digits) as :class:`rtt.agent.jobs.LocalJobRunner`, so the
  existing ``poll_job`` and ``fetch_results`` serve them.
* :func:`tournament_job`: the default job: resolve the uploaded part, run
  :func:`rtt.effectors.tournament.run_tournament`, write its outputs, return the recommendation.
* :func:`compact_result`: what ``fetch_results`` shows the model, with ``verdict``
  (``recommended`` or ``no_winner``) and the winner's ``certificate_verdict`` (the
  ``stress_test`` rule: the nominal grasp passes the requirement and the score reaches
  :data:`rtt.agent.tools.CERT_PASS_SCORE`).

Caps are the toolbox's (:mod:`rtt.agent.tools`): the call counts toward the 12 tool calls, at
most 2 jobs (of any kind) run at once, and ``design = true`` needs the passcode and reserves
:data:`DESIGN_RESERVE_USD` against the session and day caps before the job starts, within the
session's Nemotron design cap :data:`DESIGN_SESSION_USD`; the Nemotron spend is charged when the
job ends. The API service gives a live (passcode) session a runner with the designer
(:func:`designer_factory`, its client capped at the reservation); an offline session's runner
has none (``designs = False``), and ``design = true`` there runs without Nemotron and reserves
nothing. Every result carries the product-mode label.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from rtt.agent.jobs import JobRecord, LocalJobRunner, _LocalJob, _new_local_id
from rtt.agent.recorder import PRODUCT_MODE_LABEL, redact_paths

__all__ = [
    "DESIGN_RESERVE_USD",
    "DESIGN_SESSION_USD",
    "EFFECTOR_GROUPS",
    "EffectorJobRunner",
    "EffectorRequest",
    "compact_result",
    "designer_factory",
    "resolve_upload",
    "tournament_job",
]

#: Budget groups the model may name (``rtt.effectors.types.EntrantGroup``).
EFFECTOR_GROUPS: Final[tuple[str, ...]] = (
    "reference",
    "catalog",
    "soft_arm",
    "suction",
    "designed",
    "random",
)
#: Worst case of one tournament's Nemotron proposals: 2 rounds of one forced call plus at most one
#: repair, each at most about $0.0125 as the client reserves it (its conservative prompt estimate
#: plus 8,192 output tokens, at $0.30 and $0.90 per million tokens). The designer's client is
#: capped at this, so a job never spends more. Measured: $0.011 to $0.015 per training part.
DESIGN_RESERVE_USD: Final = 0.05
#: Nemotron design spend one session may commit (reservations in flight plus what finished jobs
#: cost), on top of the session and day caps that every reservation must also fit.
DESIGN_SESSION_USD: Final = 0.10
_RANKING_ROWS: Final = 6


@dataclass(frozen=True)
class EffectorRequest:
    """One ``recommend_effector`` call, validated."""

    part_id: str
    requirement: str = "registered"
    groups: tuple[str, ...] | None = None
    design: bool = False
    budget_usd: float = 0.0  # what the job may spend on Nemotron (the reservation)


class EffectorJobRunner(LocalJobRunner):
    """Tournament jobs on this machine (no compute cost); ``results`` returns the job's document.

    ``designs``: whether a job of this runner can ask Nemotron for morphologies (the runner's job
    has a live designer). Without one, ``design = true`` is answered without Nemotron and nothing
    is reserved."""

    def __init__(
        self,
        run: Callable[[EffectorRequest], Mapping[str, Any]],
        *,
        max_workers: int = 2,
        clock: Callable[[], float] = time.time,
        designs: bool = True,
    ) -> None:
        super().__init__(lambda part_id, designs: [], max_workers=max_workers, clock=clock)
        self._run = run
        self.designs = designs

    def launch_request(self, request: EffectorRequest) -> JobRecord:
        job_id = _new_local_id()
        record = JobRecord(
            job_id=job_id,
            target="local",
            part_id=request.part_id,
            design_ids=(),
            state="QUEUED",
            created_at=self._clock(),
        )
        job = _LocalJob(record)
        with self._lock:
            self._jobs[job_id] = job

        def run() -> Any:
            with job.lock:
                job.record = replace(job.record, state="RUNNING", started_at=self._clock())
            return dict(self._run(request))

        job.future = self._pool.submit(run)
        job.future.add_done_callback(lambda fut: self._finish(job, fut))  # type: ignore[arg-type]
        return job.record


# --------------------------------------------------------------------------- the default job


def resolve_upload(backend: Any, part_id: str) -> tuple[Any, tuple[Any, ...], str]:
    """The session's uploaded part as a :class:`rtt.parts.types.PartSpec`, its screen
    conditions and the suite name.

    With the ``rtt.part`` intake (a part spec with resting poses), the most likely pose is used
    with its :func:`rtt.part.conditions.conditions_for_part` conditions (2 jitter seeds each);
    otherwise the part as loaded, at the four ``train_v2`` placements
    (:func:`rtt.effectors.tournament.upload_conditions`).
    """
    from rtt.effectors.tournament import ScreenCondition, upload_conditions
    from rtt.parts.library import part_from_mesh_object

    getter = getattr(backend, "_get", None)
    if getter is None:
        raise RuntimeError("effector tournaments need the simulator backend, not the offline fake")
    held = getter(part_id)
    intake = getattr(held, "spec", None)
    if intake is not None and getattr(intake, "poses", None):
        from rtt.part.conditions import conditions_for_part

        suite = conditions_for_part(intake, seeds_per_condition=2)
        first = suite.conditions[0]
        same = [c for c in suite.conditions if c.obj is first.obj][:4]
        part = dataclasses.replace(part_from_mesh_object(first.obj, family="upload"), name=part_id)
        conds = tuple(
            ScreenCondition(
                c.condition_id, c.placement_label or c.condition_id, c.placement, tuple(c.seeds[:2])
            )
            for c in same
        )
        return part, conds, suite.name
    part = dataclasses.replace(part_from_mesh_object(held.obj, family="upload"), name=part_id)
    return part, upload_conditions(part, suite="upload"), "upload"


def tournament_job(
    backend: Any,
    out_root: str | Path,
    *,
    workers: int | None = 2,
    designer_factory: Callable[[float], Any] | None = None,
) -> Callable[[EffectorRequest], dict[str, Any]]:
    """The default ``recommend_effector`` job: tournament on the uploaded part, outputs under
    ``out_root/<part_id>/``, and the recommendation back.

    ``designer_factory(budget_usd)`` builds the :class:`~rtt.effectors.tournament.MorphologyDesigner`
    for ``design = true`` (its model client capped at ``budget_usd``); without it a design request
    runs the groups that need no model.
    """
    from rtt.effectors.tournament import run_tournament, write_outputs

    def run(request: EffectorRequest) -> dict[str, Any]:
        part, conditions, suite = resolve_upload(backend, request.part_id)
        designer = designer_factory(request.budget_usd) if request.design and designer_factory else None
        result = run_tournament(
            part,
            requirement=request.requirement,
            groups=request.groups,
            workers=workers,
            conditions=conditions,
            suite=suite,
            design=request.design and designer is not None,
            designer=designer,
        )
        write_outputs(result, Path(out_root) / part.name)
        physics = result.physics()
        return {
            "recommendation": result.recommendation,
            "episodes": physics["episodes"] + physics["certificate_episodes"],
            "llm_usd": float(result.llm["usd"]),
            "llm_calls": int(result.llm["calls"]),
            "fair": result.fairness()["identical"],
            "digest": result.digest(),
        }

    return run


def designer_factory(
    root: str | Path, *, transport: Callable[[], Any] | None = None
) -> Callable[[float], Any]:
    """``factory(budget_usd)``: track B3's designer on a client capped at ``budget_usd`` (one job's
    reservation), with the scrubbing response cache under ``root/llm_cache`` and the call ledger
    ``root/llm_ledger.jsonl``. ``transport`` builds the model transport (default: Token Factory
    from ``NEBIUS_API_KEY``, read on the first live call)."""
    base = Path(root)

    def make(budget_usd: float) -> Any:
        from rtt.effectors.morphology import make_designer_client
        from rtt.effectors.tournament import default_designer

        client = make_designer_client(
            base / "llm_cache",
            ledger_path=base / "llm_ledger.jsonl",
            budget_usd=float(budget_usd),
            transport=transport() if transport is not None else None,
        )
        designer = default_designer(client)
        if designer is None:
            raise RuntimeError("the morphology designer (rtt.effectors.morphology) is not available")
        return designer

    return make


# --------------------------------------------------------------------------- what the model sees


def _pct(v: float | None) -> float | None:
    return None if v is None else round(100.0 * float(v), 1)


def _brief(card: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not card or not card.get("effector"):
        return None
    eff = card["effector"]
    mass = card.get("mass_kg")
    return {
        "effector_id": eff["effector_id"],
        "display_name": eff["display_name"],
        "family": eff["family"],
        "fidelity": card.get("fidelity") or eff.get("fidelity"),
        "certificate_score_pct": _pct(card.get("score")),
        "placement_tolerance_mm": card.get("tolerance_mm"),
        "screen": card.get("screen"),
        "n_actuators": card.get("n_actuators"),
        "mass_g": round(1000.0 * mass, 1) if mass is not None else None,
        "actuation": card.get("actuation"),
        "why": card.get("why"),
        "caveat": card.get("caveat"),
        "note": card.get("note"),
    }


def certificate_verdict(card: Mapping[str, Any] | None, requirement: str, pass_score: float) -> str | None:
    """``pass`` when the certificate's nominal grasp passes ``requirement`` and its score for it
    reaches ``pass_score`` (the ``stress_test`` rule), else ``fail``; ``None`` without one."""
    if not card or not card.get("certificate"):
        return None
    cert = card["certificate"]
    nominal = next((t for t in cert["tests"] if t["name"] == "nominal"), None)
    score = float(cert["held_score"] if requirement == "held" else cert["registered_score"])
    ok = bool(nominal and nominal["held" if requirement == "held" else "registered"])
    return "pass" if ok and score >= pass_score - 1e-12 else "fail"


def compact_result(job_id: str, part_id: str, doc: Mapping[str, Any], pass_score: float) -> dict[str, Any]:
    """The ``fetch_results`` answer for an effector job (labelled, verdicts computed here)."""
    rec = doc["recommendation"]
    req = rec["requirement"]
    winner = rec.get("winner")
    nw = rec.get("no_winner")
    out: dict[str, Any] = {
        "kind": "effector_recommendation",
        "label": PRODUCT_MODE_LABEL,
        "job_id": job_id,
        "part_id": part_id,
        "requirement": req,
        "verdict": "recommended" if winner else "no_winner",
        "recommendation": _brief(winner),
        "certificate_verdict": certificate_verdict(winner, req, pass_score),
        "runner_up_rigid": _brief(rec.get("runner_up_rigid")),
        "other_requirements": {
            k: (v["display_name"] if v else None) for k, v in (rec.get("alternatives") or {}).items()
        },
        "ranking": [
            {
                "display_name": r["display_name"],
                "family": r["family"],
                "fidelity": r["fidelity"],
                "eligible": r["eligible"],
                "certificate_score_pct": _pct(r.get("score")),
                "screen_success_pct": _pct(r.get("screen_rate")),
                "why_not": r.get("reason"),
                **({"note": r["note"]} if r.get("note") else {}),
            }
            for r in rec["ranking"][:_RANKING_ROWS]
        ],
        "groups": {g: {"status": s["status"], "reason": s["reason"]} for g, s in rec["groups"].items()},
        "episodes": int(doc.get("episodes", 0)),
        "nemotron_usd": round(float(doc.get("llm_usd", 0.0)), 6),
    }
    if winner and winner.get("fidelity") == "approximation" and out["runner_up_rigid"] is None:
        out["runner_up_note"] = (rec.get("runner_up_rigid") or {}).get("basis")
    if nw:
        out["no_winner"] = {
            "reason": nw["reason"],
            "closest": [
                {
                    "display_name": c["display_name"],
                    "why_not": c.get("reason"),
                    "failure_modes": c.get("failure_modes"),
                }
                for c in nw["closest"]
            ],
        }
    return out


def verdict_row(job_id: str, part_id: str, compact: Mapping[str, Any]) -> dict[str, Any]:
    """The session file's record of an effector verdict (computed by code)."""
    win = compact.get("recommendation") or {}
    return {
        "kind": "effector_recommendation",
        "job_id": job_id,
        "part_id": part_id,
        "requirement": compact["requirement"],
        "recommendation": compact["verdict"],
        "effector_id": win.get("effector_id"),
        "display_name": win.get("display_name"),
        "fidelity": win.get("fidelity"),
        "certificate_score_pct": win.get("certificate_score_pct"),
        "certificate_verdict": compact.get("certificate_verdict"),
        "label": PRODUCT_MODE_LABEL,
    }


def error_text(exc: BaseException) -> str:
    return redact_paths(f"{type(exc).__name__}: {exc}")[:400]


def groups_or_none(groups: Sequence[str] | None) -> tuple[str, ...] | None:
    return tuple(groups) if groups else None
