"""HTTP routes of the Task Engineer for the existing FastAPI app (product mode).

:func:`studio_router` returns an :class:`fastapi.APIRouter` over an
:class:`rtt.api.service.AgentService` (its passcode gate, its daily spend book and its part
store). The integrator mounts it with one line in :func:`rtt.api.app.create_app`::

    app.include_router(studio_router(svc))

Endpoints:

* ``POST /studio/plan``: ``{"text", "part_id"?, "passcode"?}`` (the passcode may also come in
  the ``X-RTT-Passcode`` header). A description the recorded gate run holds is answered from
  that cache at no cost and needs no passcode. Anything else calls Nemotron, so it needs the
  passcode (403 without it, 429 when locked out), runs one plan at a time (429 when busy) and
  spends at most ``plan_usd`` (``RTT_STUDIO_PLAN_USD``, default $0.05) inside the server's
  daily cap (429 when the cap refuses it). An offline server answers recorded descriptions
  only (409 otherwise). ``part_id`` names a part uploaded with ``POST /parts`` (404 if
  unknown; 422 if its intake record is missing): the plan must use it as ``upload:<part_id>``.
  200 with ``{"label", "source": "recorded"|"live", "result": PlanResult.to_dict(), "spend"}``.
* ``POST /studio/validate``: ``{"plan": <submit_task_plan arguments>, "part_ids"?}`` ->
  ``{"status", "problems", "warnings", "task_id", "fidelity"}``. No model call, no passcode.

Bodies larger than :data:`MAX_BODY_BYTES` get 413.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from rtt.agent.backends import PartNotFoundError
from rtt.agent.caps import GuardedLedger, PasscodeLockedError, SpendGuard
from rtt.agent.schemas import PART_ID_PATTERN
from rtt.effectors.types import PRODUCT_MODE_LABEL
from rtt.llm.cache import CacheMiss
from rtt.llm.client import LLMAuthError, LLMBillingError
from rtt.studio.engineer.engineer import (
    MAX_USE_CASE_CHARS,
    PlanResult,
    judge_args,
    make_engineer_client,
    normalise_nulls,
    plan_task,
)
from rtt.studio.engineer.gate import RECORDED_GATE
from rtt.studio.engineer.uploads import UploadInfo, UploadRegistry, install_upload_resolver

if TYPE_CHECKING:
    from rtt.api.service import AgentService

__all__ = ["MAX_BODY_BYTES", "PlanRequest", "StudioPlanSettings", "ValidateRequest", "studio_router"]

MAX_BODY_BYTES: Final = 256 * 1024


class PlanRequest(BaseModel):
    """Body of ``POST /studio/plan``."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4 * MAX_USE_CASE_CHARS)
    part_id: str | None = Field(default=None, pattern=PART_ID_PATTERN)
    passcode: str | None = Field(default=None, max_length=128)


class ValidateRequest(BaseModel):
    """Body of ``POST /studio/validate``."""

    model_config = ConfigDict(extra="forbid")

    plan: dict[str, Any]
    part_ids: list[Annotated[str, Field(pattern=PART_ID_PATTERN)]] = Field(default_factory=list, max_length=8)


@dataclass(frozen=True)
class StudioPlanSettings:
    """``plan_usd``: the cap of one live plan (both calls); ``recorded``: the free replay cache."""

    plan_usd: float = 0.05
    recorded: Path = RECORDED_GATE / "cache"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> StudioPlanSettings:
        env = os.environ if environ is None else environ
        return cls(plan_usd=float(env.get("RTT_STUDIO_PLAN_USD", "0.05")))


def _check_size(request: Request) -> None:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise HTTPException(413, f"body larger than {MAX_BODY_BYTES} bytes")


def studio_router(
    service: AgentService,
    settings: StudioPlanSettings | None = None,
    *,
    transport_factory: Callable[[], Any] | None = None,
) -> APIRouter:
    """The Studio's Task Engineer routes over ``service`` (module docstring).

    ``transport_factory`` builds the model transport of a live plan (default: Token Factory
    from the server's ``.env``; tests pass a scripted one).
    """
    cfg = settings or StudioPlanSettings.from_env()
    registry = UploadRegistry(service.store.parts_dir)
    install_upload_resolver(registry)
    busy = threading.Lock()
    studio_root = service.store.root / "studio"
    router = APIRouter(prefix="/studio", tags=["studio"])

    def upload_of(part_id: str) -> UploadInfo:
        try:
            service.part(part_id)
        except (ValueError, PartNotFoundError):
            raise HTTPException(404, "no such part") from None
        try:
            return registry.info(part_id)
        except (KeyError, ValueError):
            raise HTTPException(
                422,
                "this part has no intake record the Studio can use (upload it again with the full intake)",
            ) from None

    def recorded(text: str) -> PlanResult | None:
        client = make_engineer_client(cfg.recorded, cache_mode="replay")
        try:
            return plan_task(text, client=client)
        except CacheMiss:
            return None

    def live(text: str, uploads: list[UploadInfo]) -> tuple[PlanResult, dict[str, Any]]:
        guard = SpendGuard(cfg.plan_usd, service.day_book, f"plan-{uuid.uuid4().hex[:12]}")
        ledger = GuardedLedger(guard, studio_root / "ledger.jsonl")
        # The agent's transport factory scripts the agent's own tools, so it is not reused here;
        # without one of ours, the client reads NEBIUS_API_KEY from the server's .env on first use.
        transport = transport_factory() if transport_factory is not None else None
        client = make_engineer_client(studio_root / "cache", ledger=ledger, transport=transport)
        return plan_task(text, client=client, uploads=uploads), guard.snapshot()

    @router.post("/plan")
    async def plan(
        body: PlanRequest,
        request: Request,
        x_rtt_passcode: Annotated[str | None, Header(max_length=128)] = None,
    ) -> dict[str, Any]:
        _check_size(request)
        if not body.text.strip():
            raise HTTPException(422, "describe the use case first")
        if body.part_id is None:
            hit = await asyncio.to_thread(recorded, body.text)
            if hit is not None:
                return {
                    "label": PRODUCT_MODE_LABEL,
                    "source": "recorded",
                    "result": hit.to_dict(),
                    "spend": None,
                }
        if service.settings.offline:
            raise HTTPException(
                409, "this server is offline: it plans only the recorded example descriptions"
            )
        try:
            ok = service.gate.check(body.passcode or x_rtt_passcode)
        except PasscodeLockedError as exc:
            raise HTTPException(429, str(exc)) from None
        if not ok:
            raise HTTPException(
                403, "planning a new description calls a paid model; the passcode is required"
            )
        uploads = [upload_of(body.part_id)] if body.part_id else []
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "another plan is running; try again when it finishes")
        try:
            result, spend = await asyncio.to_thread(live, body.text, uploads)
        except (LLMAuthError, LLMBillingError) as exc:
            raise HTTPException(503, f"the model provider refused the call ({type(exc).__name__})") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        finally:
            busy.release()
        capped = [a for a in result.attempts if a.outcome == "spend_cap"]
        if result.status == "failed" and capped:
            raise HTTPException(429, "the spend cap refused this plan: " + (capped[-1].error or "")[:200])
        return {"label": PRODUCT_MODE_LABEL, "source": "live", "result": result.to_dict(), "spend": spend}

    @router.post("/validate")
    def validate(body: ValidateRequest, request: Request) -> dict[str, Any]:
        _check_size(request)
        uploads = [upload_of(pid) for pid in body.part_ids]
        args, notes = normalise_nulls(body.plan)
        spec, problems, warnings = judge_args(args, uploads=uploads, source="manual")
        status = "invalid" if problems or spec is None else ("refused" if spec.refusal else "valid")
        return {
            "label": PRODUCT_MODE_LABEL,
            "status": status,
            "problems": problems,
            "warnings": warnings,
            "notes": list(notes),
            "task_id": spec.task_id if spec is not None and status == "valid" else None,
            "fidelity": {s.id: spec.fidelity(s.id).value for s in spec.steps} if spec is not None else {},
        }

    return router
