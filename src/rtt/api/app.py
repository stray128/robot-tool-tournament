"""FastAPI app for the Grasp Engineer (local or passcode only; product mode).

Endpoints:

* ``POST /parts``: upload an STL as the raw request body (``Content-Type:
  application/octet-stream`` or ``model/stl``; query ``name``, ``units=mm|m``,
  ``density_kg_m3``). Larger than ``max_upload_bytes``: 413. Not an STL: 422. When the
  server listens beyond loopback (``ApiSettings.require_passcode``) the passcode must
  come in the ``X-RTT-Passcode`` header: 403 without it, 429 when locked out.
* ``GET /parts/{part_id}``: the part as the agent sees it.
* ``POST /sessions``: ``{"part_id", "passcode", "goal"}`` (the passcode may also come
  in the ``X-RTT-Passcode`` header) starts the agent; 201 with the session ID. Wrong or
  missing passcode: 403; too many wrong guesses or the server is busy: 429; the Nebius
  target is enabled but cannot be set up: 503.
* ``GET /sessions/{id}``: the session recording (live snapshot or finished file).
* ``GET /sessions/{id}/events``: Server-Sent Events, one per agent step
  (``id`` = event ``seq``, ``event`` = its type); resumes after ``Last-Event-ID`` or
  ``?after=``; ends with ``event: end``.
* ``POST /sessions/{id}/stop``: stop after the current step.
* ``GET /runs/{run_id}``: results of one simulation job (every design and episode).
* ``GET /printpack/{pack_id}``: the print pack as a zip.
* ``GET /health``.

CORS allows the local web dev servers only (``ApiSettings.cors_origins``), and requests
whose ``Host`` is not in ``ApiSettings.allowed_hosts`` (loopback names by default) get
400, which stops DNS-rebinding pages. ``POST /parts`` with any other content type gets
415. Run with
``python -m rtt.api`` (binds 127.0.0.1) or ``uvicorn rtt.api.app:create_app --factory``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Final

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from rtt.agent.backends import PartNotFoundError
from rtt.agent.caps import PasscodeLockedError, PasscodeRequiredError
from rtt.agent.recorder import PRODUCT_MODE_LABEL
from rtt.agent.schemas import PART_ID_PATTERN
from rtt.api.service import (
    AgentService,
    ApiSettings,
    BusyError,
    SessionView,
    TargetUnavailableError,
    UploadError,
)

__all__ = ["SessionRequest", "create_app", "sse_format"]

SSE_WAIT_S: Final = 10.0
#: Content types accepted for ``POST /parts`` (the first is the one to use).
STL_CONTENT_TYPES: Final = (
    "application/octet-stream",
    "model/stl",
    "model/x.stl-binary",
    "model/x.stl-ascii",
    "application/sla",
    "application/vnd.ms-pki.stl",
)


class SessionRequest(BaseModel):
    """Body of ``POST /sessions``."""

    model_config = ConfigDict(extra="forbid")

    part_id: str = Field(pattern=PART_ID_PATTERN)
    passcode: str | None = Field(default=None, max_length=128)
    goal: str | None = Field(default=None, max_length=500)


def sse_format(event: dict[str, Any]) -> str:
    """One SSE frame: ``id``, ``event`` and a single-line JSON ``data``."""
    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event['seq']}\nevent: {event['type']}\ndata: {data}\n\n"


async def _event_stream(view: SessionView, after: int) -> AsyncIterator[str]:
    last = after
    yield "retry: 2000\n\n"
    while True:
        events = await asyncio.to_thread(view.wait_events, last, SSE_WAIT_S)
        for event in events:
            yield sse_format(event)
            last = int(event["seq"])
        if view.done and not view.wait_events(last, 0.0):
            yield f"event: end\ndata: {json.dumps({'last_seq': last})}\n\n"
            return
        if not events:
            yield ": keep-alive\n\n"


def create_app(settings: ApiSettings | None = None, *, service: AgentService | None = None) -> FastAPI:
    """Build the app around ``service`` (default: an :class:`AgentService` from ``settings``)."""
    svc = service or AgentService(settings or ApiSettings.from_env())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        svc.shutdown()

    app = FastAPI(
        title="Robot Tool Tournament: Grasp Engineer",
        version="0.1.0",
        description=f"Upload a part, let the agent design fingers for it. {PRODUCT_MODE_LABEL}.",
        lifespan=lifespan,
    )
    app.state.service = svc
    if "*" not in svc.settings.allowed_hosts:
        # A browser page on another site can point its own host name at 127.0.0.1 (DNS
        # rebinding); refusing unknown Host headers keeps such pages off this API.
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(svc.settings.allowed_hosts))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(svc.settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Last-Event-ID", "X-RTT-Passcode"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "mode": "offline" if svc.settings.offline else "live",
            "label": PRODUCT_MODE_LABEL,
            "passcode_configured": svc.gate.enabled,
            "passcode_for_uploads": svc.settings.require_passcode,
            "active_sessions": svc.active_sessions(),
            "targets": (
                ["local", "nebius"] if svc.settings.enable_nebius and not svc.settings.offline else ["local"]
            ),
            "backend": svc.backend.name,
        }

    @app.post("/parts", status_code=201)
    async def upload_part(
        request: Request,
        name: Annotated[str | None, Query(max_length=120)] = None,
        units: Annotated[str, Query(pattern="^(mm|m)$")] = "mm",
        density_kg_m3: Annotated[float | None, Query(gt=0)] = None,
        x_rtt_passcode: Annotated[str | None, Header(max_length=128)] = None,
    ) -> dict[str, Any]:
        if svc.settings.require_passcode:
            # checked before the body is read, so a stranger cannot make the server parse anything
            try:
                ok = svc.gate.check(x_rtt_passcode)
            except PasscodeLockedError as exc:
                raise HTTPException(429, str(exc)) from None
            if not ok:
                raise HTTPException(403, "this server needs the passcode (X-RTT-Passcode) for uploads")
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type not in STL_CONTENT_TYPES:
            # Also keeps cross-site pages out: a form or text/plain POST needs no CORS
            # preflight, a binary content type does.
            raise HTTPException(415, f"send the STL as the raw body with Content-Type {STL_CONTENT_TYPES[0]}")
        limit = svc.settings.max_upload_bytes
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            raise HTTPException(413, f"upload larger than {limit} bytes")
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"upload larger than {limit} bytes")
            chunks.append(chunk)
        data = b"".join(chunks)
        try:
            info = await asyncio.to_thread(
                svc.upload_part, data, name=name, units=units, density=density_kg_m3, passcode=x_rtt_passcode
            )
        except UploadError as exc:
            raise HTTPException(422, str(exc)) from None
        except PasscodeLockedError as exc:
            raise HTTPException(429, str(exc)) from None
        except PasscodeRequiredError as exc:
            raise HTTPException(403, str(exc)) from None
        return {"part": info.to_json(), "label": PRODUCT_MODE_LABEL}

    @app.get("/parts/{part_id}")
    def get_part(part_id: str) -> dict[str, Any]:
        try:
            return {"part": svc.part(part_id).to_json()}
        except (ValueError, PartNotFoundError):
            raise HTTPException(404, "no such part") from None

    @app.post("/sessions", status_code=201)
    def start_session(
        body: SessionRequest,
        x_rtt_passcode: Annotated[str | None, Header(max_length=128)] = None,
    ) -> dict[str, Any]:
        try:
            session_id = svc.start_session(
                body.part_id, passcode=body.passcode or x_rtt_passcode, goal=body.goal
            )
        except PartNotFoundError:
            raise HTTPException(404, "no such part") from None
        except PasscodeLockedError as exc:
            raise HTTPException(429, str(exc)) from None
        except PasscodeRequiredError as exc:
            raise HTTPException(403, str(exc)) from None
        except BusyError as exc:
            raise HTTPException(429, str(exc)) from None
        except TargetUnavailableError as exc:
            raise HTTPException(503, str(exc)) from None
        return {
            "session_id": session_id,
            "events": f"/sessions/{session_id}/events",
            "label": PRODUCT_MODE_LABEL,
            "offline": svc.settings.offline,
        }

    def _view(session_id: str) -> SessionView:
        try:
            return svc.session(session_id)
        except (ValueError, KeyError):
            raise HTTPException(404, "no such session") from None

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return _view(session_id).to_json()

    @app.get("/sessions/{session_id}/events")
    def session_events(
        session_id: str,
        after: Annotated[int, Query(ge=0)] = 0,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        view = _view(session_id)
        start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after
        return StreamingResponse(
            _event_stream(view, start),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/sessions/{session_id}/stop")
    def stop_session(session_id: str) -> dict[str, Any]:
        _view(session_id)
        return {"stopping": svc.stop_session(session_id)}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> JSONResponse:
        try:
            return JSONResponse(svc.run(run_id))
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "no such run") from None

    @app.get("/printpack/{pack_id}")
    def get_print_pack(pack_id: str) -> FileResponse:
        try:
            path = svc.print_pack(pack_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "no such print pack") from None
        return FileResponse(path, media_type="application/zip", filename=f"{pack_id}.zip")

    return app
