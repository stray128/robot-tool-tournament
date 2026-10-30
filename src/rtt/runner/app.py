"""The runner's HTTP API and run lifecycle (``docs/RUNNER.md``).

:class:`Runner` answers Lambda Function URL events (payload format 2.0) through
:meth:`Runner.handle_event`; ``rtt runner serve`` feeds it the same events. Routes:

* ``GET /health``: the switches, what is left today and the spend so far.
* ``POST /plan``: the Task Engineer on ``nvidia``, ``tokenfactory`` (the visitor's key) or
  ``free`` (ours, capped).
* ``POST /runs``: a server-validated plan to one Nebius job, under the run and spend caps.
* ``GET /runs/{id}``: the run's status, its cost so far and, when done, the summary with a
  short-lived link to the full report.
* ``GET /runs/{id}/report``: a redirect to that link.

:meth:`Runner.sweep` (every two minutes from a schedule) polls the runs in flight, cancels a job
past the wall-time limit (or every job while the kill switch is on) and settles each run's cost
exactly once.

Errors are JSON ``{"error": ...}``: 400 bad input, 401 the provider rejected the visitor's key,
403 a browser origin that is not allowed, 404 unknown route or run, 405 wrong method, 429 a cap
(which one and when it resets), 503 kill switch, auto-off or not configured, 500 anything else
(its type is logged, never its text).
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from rtt.runner import caps as capmod
from rtt.runner.caps import KillSwitch, day_of
from rtt.runner.config import PROVIDERS, RunnerConfig, to_micros, to_usd
from rtt.runner.engineer import Planner, ProviderRejected
from rtt.runner.logs import log_event
from rtt.runner.nebius import (
    JobBackend,
    JobStatus,
    SubmitNotStarted,
    SubmitRefused,
    Submitted,
    job_name,
)
from rtt.runner.secrets import SecretError, SecretSource
from rtt.runner.store import Bump, Conflict, LimitReached, RunWrite, Store
from rtt.runner.summary import SUMMARY_FORMAT
from rtt.runner.validate import check_run_plan

__all__ = ["RUN_ID", "Request", "Response", "Runner", "new_run_id"]

#: 160 random bits in lowercase base32: url-safe, and valid in a bucket prefix and a job name.
RUN_ID: Final = re.compile(r"^[a-z2-7]{32}$")
_PRINTABLE_KEY: Final = re.compile(r"^[\x21-\x7e]+$")
ACTIVE: Final = ("queued", "running")
#: A job never seen terminal past this age is given up on (the platform's own 1 h timeout plus
#: slack); a run whose job could not be confirmed either way is then charged its reservation.
_GIVE_UP_S: Final = 3600 + 900


def new_run_id() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().lower()


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class Request:
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes = b""
    source_ip: str = ""
    request_id: str = ""


@dataclass
class Response:
    status: int
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)


class HTTPError(Exception):
    def __init__(self, status: int, body: Mapping[str, Any] | str) -> None:
        super().__init__(status)
        self.status = status
        self.body = {"error": body} if isinstance(body, str) else dict(body)


class Runner:
    """The API over a store, a planner and a job backend.

    ``jobs`` builds the job backend on first use (it needs the Nebius secret, so ``/health`` and
    ``/plan`` never touch it). ``secrets`` serves the two secrets by id. ``clock`` and
    ``run_id`` are injection points for tests.
    """

    def __init__(
        self,
        cfg: RunnerConfig,
        store: Store,
        planner: Planner,
        jobs: Callable[[], JobBackend],
        secrets_source: SecretSource,
        kill: KillSwitch | None = None,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        run_id: Callable[[], str] = new_run_id,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.planner = planner
        self._jobs_factory = jobs
        self._jobs: JobBackend | None = None
        self.secrets = secrets_source
        self.kill = kill or KillSwitch(cfg)
        self.clock = clock
        self.new_run_id = run_id
        self._salt: bytes | None = None

    # ------------------------------------------------------------------ plumbing

    def jobs(self) -> JobBackend:
        if self._jobs is None:
            self._jobs = self._jobs_factory()
        return self._jobs

    def _free_key(self) -> str:
        try:
            return self.secrets.get(self.cfg.tokenfactory_secret)
        except SecretError:
            raise HTTPError(503, "the runner is not configured (free-tier key)") from None

    def _ip(self, req: Request, now: datetime) -> str:
        if self._salt is None:
            self._salt = capmod.ip_salt(self._free_key())
        return capmod.ip_hash(self._salt, day_of(now), req.source_ip or "unknown")

    def _cors(self, origin: str | None) -> dict[str, str]:
        if not self.cfg.emit_cors or not origin or origin not in self.cfg.allowed_origins:
            return {}
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "content-type",
            "Access-Control-Max-Age": "600",
            "Vary": "Origin",
        }

    def _json(self, req: Request) -> dict[str, Any]:
        if len(req.body) > self.cfg.max_body_bytes:
            raise HTTPError(400, f"the body is larger than {self.cfg.max_body_bytes} bytes")
        try:
            data = json.loads(req.body.decode("utf-8")) if req.body else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HTTPError(400, "the body is not JSON") from None
        if not isinstance(data, dict):
            raise HTTPError(400, "the body must be a JSON object")
        return data

    def _gate(self, now: datetime) -> None:
        state = self.kill.state(now)
        if state.kill_switch:
            raise HTTPError(503, "the runner is switched off for now; recorded runs still replay")
        if state.auto_off:
            raise HTTPError(
                503,
                f"the hosted runner closed after {self.cfg.auto_off.isoformat()}; recorded runs still replay",
            )

    # ------------------------------------------------------------------ entry points

    def handle_event(self, event: Mapping[str, Any], context: Any = None) -> dict[str, Any]:
        """A Lambda Function URL event (payload 2.0) -> its response."""
        ctx = event.get("requestContext") or {}
        http = ctx.get("http") or {}
        body = event.get("body") or ""
        try:
            raw = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode("utf-8")
        except (binascii.Error, ValueError):
            raw = b"\xff"  # not JSON: answered 400 by the route
        headers = {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items()}
        req = Request(
            method=str(http.get("method") or "GET").upper(),
            path=str(event.get("rawPath") or http.get("path") or "/"),
            headers=headers,
            body=raw,
            source_ip=str(http.get("sourceIp") or ""),
            request_id=str(ctx.get("requestId") or getattr(context, "aws_request_id", "") or ""),
        )
        res = self.handle(req)
        out: dict[str, Any] = {"statusCode": res.status, "headers": res.headers}
        out["body"] = "" if res.body is None else json.dumps(res.body)
        return out

    def handle(self, req: Request) -> Response:
        t0 = time.perf_counter()
        route = "unknown"
        origin = req.headers.get("origin")
        try:
            if origin is not None and origin not in self.cfg.allowed_origins:
                route = "origin"
                res = Response(403, {"error": "this origin may not call the runner"})
            else:
                route, call = self._match(req)
                res = call()
        except HTTPError as exc:
            res = Response(exc.status, exc.body)
        except Exception as exc:  # noqa: BLE001 - never echo internals
            log_event("error", request_id=req.request_id, route=route, error_type=type(exc).__name__)
            res = Response(500, {"error": "internal error"})
        res.headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            **res.headers,
            **self._cors(origin),
        }
        log_event(
            "request",
            request_id=req.request_id,
            route=route,
            method=req.method,
            status=res.status,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return res

    def _match(self, req: Request) -> tuple[str, Callable[[], Response]]:
        """The route template and a call that answers ``req`` (404/405 answered directly)."""
        path = req.path.rstrip("/") or "/"
        parts = path.strip("/").split("/")
        if req.method == "OPTIONS":
            return "preflight", lambda: Response(204, None)
        found: tuple[str, str, Callable[..., Response], tuple[str, ...]] | None = None
        if path == "/health":
            found = ("GET", "/health", self.health, ())
        elif path == "/plan":
            found = ("POST", "/plan", self.plan, ())
        elif path == "/runs":
            found = ("POST", "/runs", self.create_run, ())
        elif len(parts) == 2 and parts[0] == "runs":
            found = ("GET", "/runs/{id}", self.get_run, (parts[1],))
        elif len(parts) == 3 and parts[0] == "runs" and parts[2] == "report":
            found = ("GET", "/runs/{id}/report", self.report, (parts[1],))
        if found is None:
            return "unknown", lambda: Response(404, {"error": "no such route"})
        method, route, fn, args = found
        if req.method != method:
            return route, lambda: Response(405, {"error": f"use {method}"}, {"Allow": method})
        return route, lambda: fn(req, *args)

    # ------------------------------------------------------------------ routes

    def health(self, req: Request) -> Response:
        now = self.clock()
        state = self.kill.state(now)
        day = day_of(now)
        caps = self.cfg.caps
        day_key, total_key = capmod.spend_keys(day)
        spent_day = self.store.numbers(day_key).get("committed", 0)
        spent_total = self.store.numbers(total_key).get("committed", 0)
        free_used = self.store.numbers(f"count#free-plans#{day}").get("n", 0)
        reserve = capmod.reservation_micros(self.cfg)
        room = min(
            to_micros(caps.spend_per_day_usd) - spent_day, to_micros(caps.spend_total_usd) - spent_total
        )
        runs_left = max(0, room // reserve) if reserve > 0 and not self.cfg.nebius.missing() else 0
        plans_left = max(0, caps.free_plans_per_day - free_used)
        if not state.open:
            runs_left = plans_left = 0
        return Response(
            200,
            {
                "ok": True,
                "kill_switch": state.kill_switch,
                "auto_off": self.cfg.auto_off.isoformat(),
                "plans_left_today": int(plans_left),
                "runs_left_today": int(runs_left),
                "spend_today_usd": to_usd(max(0, spent_day)),
                "spend_total_usd": to_usd(max(0, spent_total)),
                "open": state.open,
            },
        )

    def plan(self, req: Request) -> Response:
        now = self.clock()
        self._gate(now)
        body = self._json(req)
        unknown = sorted(set(body) - {"description", "provider", "key"})
        if unknown:
            raise HTTPError(400, f"unknown fields: {', '.join(unknown[:5])}")
        description = body.get("description")
        if not isinstance(description, str) or not description.strip():
            raise HTTPError(400, "description: describe the job in words")
        if len(description) > self.cfg.max_description_chars:
            raise HTTPError(400, f"description: at most {self.cfg.max_description_chars} characters")
        provider = body.get("provider")
        if provider not in PROVIDERS:
            raise HTTPError(400, f"provider: one of {', '.join(PROVIDERS)}")
        key = body.get("key")
        if provider == "free":
            if key not in (None, ""):
                raise HTTPError(400, "key: the free tier takes no key (send null)")
        elif (
            not isinstance(key, str)
            or not 8 <= len(key) <= self.cfg.max_key_chars
            or not _PRINTABLE_KEY.match(key)
        ):
            raise HTTPError(
                400,
                f"key: the {provider} provider needs your API key (8 to {self.cfg.max_key_chars} printable characters)",
            )
        ip = self._ip(req, now)
        try:
            self.store.commit(capmod.plan_bumps(self.cfg, provider, ip, now))
        except LimitReached as exc:
            raise HTTPError(429, capmod.cap_refusal(self.cfg, exc.bump, now).body()) from None
        spec = self.cfg.providers[provider]
        use_key = self._free_key() if provider == "free" else str(key)
        try:
            outcome = self.planner.plan(description, spec, use_key)
        except ProviderRejected as exc:
            if provider == "free":
                raise HTTPError(503, "the free tier is unavailable right now; try your own key") from None
            if exc.kind == "billing":
                raise HTTPError(
                    401, "the provider refused this key (billing or quota on its account)"
                ) from None
            raise HTTPError(401, "the provider rejected this key") from None
        if provider == "free" and outcome.cost_usd > 0:
            exp = int((now + timedelta(days=40)).timestamp())
            try:
                self.store.commit(
                    [Bump(f"spend#llm#day#{day_of(now)}", "micros", to_micros(outcome.cost_usd), exp)]
                )
            except Exception as exc:  # noqa: BLE001 - bookkeeping only; the plan is already paid for
                log_event("error", request_id=req.request_id, route="/plan", error_type=type(exc).__name__)
        return Response(200, outcome.body())

    def create_run(self, req: Request) -> Response:
        now = self.clock()
        self._gate(now)
        body = self._json(req)
        unknown = sorted(set(body) - {"plan"})
        if unknown:
            raise HTTPError(400, f"unknown fields: {', '.join(unknown[:5])}")
        spec, problems = check_run_plan(self.cfg, body.get("plan"))
        if spec is None:
            raise HTTPError(400, {"error": "the plan did not pass the runner's checks", "problems": problems})
        if self.cfg.nebius.missing():
            raise HTTPError(503, "runs are not configured on this runner")
        try:
            jobs = self.jobs()  # built before anything is reserved: a broken backend costs nothing
        except Exception as exc:  # noqa: BLE001 - unreadable or malformed secret, CLI profile failure
            log_event("error", request_id=req.request_id, route="/runs", error_type=type(exc).__name__)
            raise HTTPError(
                503, "the simulation backend is unavailable; nothing was reserved, try again later"
            ) from None
        ip = self._ip(req, now)
        reserve = capmod.reservation_micros(self.cfg)
        run_id = self.new_run_id()
        day = day_of(now)
        plan_json = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        record: dict[str, Any] = {
            "run_id": run_id,
            "status": "queued",
            "created_at": _iso(now),
            "spend_day": day,
            "reserved_micros": reserve,
            "settled": False,
            "settled_micros": None,
            "task_id": spec.task_id,
            "job_name": job_name(run_id),
            "job_id": None,
            "error": None,
            "result": None,
        }
        try:
            self.store.commit(
                capmod.run_bumps(self.cfg, ip, reserve, now), self._write(record, None, True, now)
            )
        except LimitReached as exc:
            raise HTTPError(429, capmod.cap_refusal(self.cfg, exc.bump, now).body()) from None
        try:
            sub: Submitted | None = jobs.submit(run_id, plan_json)
        except SubmitNotStarted as exc:  # no job can exist: give everything back
            log_event("error", request_id=req.request_id, route="/runs", error_type=type(exc).__name__)
            self._release_unstarted(record, ip, now)
            raise HTTPError(503, "the simulation job could not be started; try again later") from None
        except Exception as exc:  # noqa: BLE001 - the job may exist: look it up by name
            log_event("error", request_id=req.request_id, route="/runs", error_type=type(exc).__name__)
            found, sub = self._look_up(jobs, run_id)
            if sub is None and found == "absent" and isinstance(exc, SubmitRefused):
                # the platform refused the create and says the job does not exist: give it all back
                self._release_unstarted(record, ip, now)
                raise HTTPError(
                    503, "the platform refused the simulation job; this did not count against your runs"
                ) from None
            if sub is None:
                # Unknown: the reservation and the active record stay; the sweeper looks the job up
                # by name once this request can no longer be submitting it, and settles either way.
                return Response(202, {"run_id": run_id, "status": "queued", "eta_s": self.cfg.eta_s})
        assert sub is not None
        record.update(job_id=sub.job_id, job_created_at=_iso(sub.created_at) if sub.created_at else None)
        try:
            self.store.commit(run=self._write(record, 1, True, now))
        except Conflict:
            return self._after_conflict(req, jobs, record, sub)
        return Response(202, {"run_id": run_id, "status": "queued", "eta_s": self.cfg.eta_s})

    def _release_unstarted(self, record: dict[str, Any], ip: str, now: datetime) -> None:
        failed = {**record, "status": "failed", "settled": True, "settled_micros": 0}
        failed["error"] = "the simulation job could not be started; this did not count against your runs"
        self.store.commit(
            capmod.settle_bumps(
                self.cfg, record["spend_day"], int(record["reserved_micros"]), 0, now, release_ip=ip
            ),
            self._write(failed, 1, False, now),
        )

    def _after_conflict(
        self, req: Request, jobs: JobBackend, record: Mapping[str, Any], sub: Submitted
    ) -> Response:
        """The run's record changed while its job was being submitted (the sweeper got there
        first). If it already holds this job, all is well; otherwise the job is cancelled at once,
        so no job runs that the ledger does not count."""
        run_id = str(record["run_id"])
        fresh = self.store.get_run(run_id)
        if fresh is not None and fresh[0].get("job_id") == sub.job_id:
            return Response(202, {"run_id": run_id, "status": "queued", "eta_s": self.cfg.eta_s})
        try:
            jobs.cancel(sub.job_id)
        except Exception as exc:  # noqa: BLE001
            log_event("error", request_id=req.request_id, route="/runs", error_type=type(exc).__name__)
        raise HTTPError(503, "the run could not be recorded, so its job was cancelled; try again")

    def _look_up(self, jobs: JobBackend, run_id: str) -> tuple[str, Submitted | None]:
        """After a failed submission, the job by name: ``("found", job)`` if the platform created
        it anyway (a timeout after ``create``, a failed read-back), ``("absent", None)`` if the
        platform says it does not exist, ``("unknown", None)`` if the lookup itself failed."""
        try:
            sub = jobs.find(run_id)
        except Exception as exc:  # noqa: BLE001 - the lookup failed: whether the job exists is unknown
            log_event("error", route="/runs", error_type=type(exc).__name__)
            return "unknown", None
        return ("found", sub) if sub is not None else ("absent", None)

    def _lookup(self, run_id: str) -> tuple[dict[str, Any], int]:
        if not RUN_ID.match(run_id):
            raise HTTPError(404, "no such run")
        got = self.store.get_run(run_id)
        if got is None:
            raise HTTPError(404, "no such run")
        return got

    def get_run(self, req: Request, run_id: str) -> Response:
        record, version = self._lookup(run_id)
        try:
            record, _ = self.refresh(record, version)
        except Exception as exc:  # noqa: BLE001 - answer with what is stored
            log_event("error", request_id=req.request_id, route="/runs/{id}", error_type=type(exc).__name__)
        return Response(200, self.view(record))

    def report(self, req: Request, run_id: str) -> Response:
        record, _ = self._lookup(run_id)
        if record.get("status") != "done":
            raise HTTPError(404, "the run has no report yet")
        try:
            url = self.jobs().report_url(run_id, self.cfg.report_url_ttl_s)
        except Exception as exc:  # noqa: BLE001 - the backend could not be built
            log_event(
                "error", request_id=req.request_id, route="/runs/{id}/report", error_type=type(exc).__name__
            )
            url = None
        if not url:
            raise HTTPError(503, "the report link could not be made; try again")
        return Response(302, {"report_url": url}, {"Location": url})

    # ------------------------------------------------------------------ the run lifecycle

    def _write(self, record: Mapping[str, Any], version: int | None, active: bool, now: datetime) -> RunWrite:
        expires = int((now + timedelta(days=self.cfg.run_ttl_days)).timestamp())
        return RunWrite(dict(record), version, active, expires)

    def view(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if record.get("settled_micros") is not None:
            cost = to_usd(int(record["settled_micros"]))
        else:
            created = _parse_iso(record.get("job_created_at")) or _parse_iso(record.get("created_at"))
            wall = (self.clock() - created).total_seconds() if created and record.get("job_id") else 0.0
            cost = to_usd(capmod.job_cost_micros(self.cfg, wall))
        result = record.get("result")
        if record.get("status") == "done" and isinstance(result, dict):
            result = dict(result)
            try:
                result["report_url"] = self.jobs().report_url(
                    str(record["run_id"]), self.cfg.report_url_ttl_s
                )
            except Exception:  # noqa: BLE001
                result["report_url"] = None
        return {
            "status": record.get("status"),
            "created_at": record.get("created_at"),
            "cost_estimate_usd": cost,
            "error": record.get("error"),
            "result": result if record.get("status") == "done" else None,
        }

    def refresh(
        self, record: dict[str, Any], version: int, *, force: bool = False
    ) -> tuple[dict[str, Any], int]:
        """Poll the run's job and move the record on (module docstring); returns the stored record."""
        if record.get("settled"):
            return record, version
        now = self.clock()
        last = _parse_iso(record.get("last_poll_at"))
        if not force and last is not None and (now - last).total_seconds() < self.cfg.poll_interval_s:
            return record, version
        rec = dict(record)
        rec["last_poll_at"] = _iso(now)
        created = _parse_iso(rec.get("created_at")) or now
        age = (now - created).total_seconds()
        if not rec.get("job_id") and age < self.cfg.unsubmitted_grace_s:
            return record, version  # the request that submits it may still be running
        # Whenever the platform cannot be asked (the backend cannot be built, the lookup or the
        # status call fails), the run is left as it is until _GIVE_UP_S after its creation, then
        # settled on the safe side (_give_up). A run with no job id is left alone quietly (it has
        # nothing else to do); a run with a job lets the failure through, so the caller logs it.
        try:
            jobs = self.jobs()
        except Exception as exc:
            if age > _GIVE_UP_S:
                log_event("error", route="backend", error_type=type(exc).__name__)
                return self._give_up(rec, record, version, now, age)
            if rec.get("job_id"):
                raise
            log_event("error", route="backend", error_type=type(exc).__name__)
            return record, version
        bumps: list[Bump] = []
        active = True
        if not rec.get("job_id"):
            try:
                found = jobs.find(str(rec["run_id"]))
            except Exception as exc:  # noqa: BLE001 - unknown: keep the reservation, try again later
                log_event("error", route="find", error_type=type(exc).__name__)
                if age <= _GIVE_UP_S:
                    return record, version
                return self._give_up(rec, record, version, now, age)
            if found is None:
                rec.update(
                    status="failed",
                    error="the simulation job was never started",
                    settled=True,
                    settled_micros=0,
                )
                bumps = capmod.settle_bumps(self.cfg, rec["spend_day"], int(rec["reserved_micros"]), 0, now)
                return self._save(rec, version, bumps, False, now, record)
            rec.update(
                job_id=found.job_id, job_created_at=_iso(found.created_at) if found.created_at else None
            )
        try:
            st: JobStatus = jobs.status(str(rec["job_id"]))
        except Exception as exc:
            if age > _GIVE_UP_S:
                log_event("error", route="status", error_type=type(exc).__name__)
                return self._give_up(rec, record, version, now, age, jobs)
            # The job's state is unknown, but a run past the wall limit (from the job's creation)
            # or with the kill switch on is cancelled all the same; the failure is then let through
            # so the caller logs it, as without a cancel.
            start = _parse_iso(rec.get("job_created_at")) or created
            if self._cancel_if_due(rec, jobs, now, max(0.0, (now - start).total_seconds())):
                self._save(rec, version, [], True, now, record)
            raise
        rec["job_state"] = st.state
        start = st.created_at or _parse_iso(rec.get("job_created_at")) or created
        end = st.finished_at if (st.terminal and st.finished_at) else now
        wall = max(0.0, (end - start).total_seconds())
        if st.terminal:
            self._finish(rec, st, jobs)
            actual = capmod.job_cost_micros(self.cfg, wall)
            bumps = capmod.settle_bumps(self.cfg, rec["spend_day"], int(rec["reserved_micros"]), actual, now)
            rec.update(settled=True, settled_micros=actual, wall_s=round(wall, 1))
            active = False
        else:
            if rec.get("status") in ACTIVE:
                rec["status"] = "running" if st.state == "RUNNING" else "queued"
            self._cancel_if_due(rec, jobs, now, wall)
            if wall > _GIVE_UP_S:
                actual = capmod.job_cost_micros(self.cfg, wall)
                bumps = capmod.settle_bumps(
                    self.cfg, rec["spend_day"], int(rec["reserved_micros"]), actual, now
                )
                rec.update(settled=True, settled_micros=actual, wall_s=round(wall, 1), status="failed")
                rec["error"] = rec.get("error") or "the simulation job never finished"
                active = False
        return self._save(rec, version, bumps, active, now, record)

    def _cancel_if_due(self, rec: dict[str, Any], jobs: JobBackend, now: datetime, wall: float) -> bool:
        """Cancel the run's job when it is past the wall limit or the kill switch is on, unless a
        cancel went out less than ``cancel_grace_s`` ago (a job still running after that is
        cancelled again). The first time, the run turns ``failed`` with the reason. A failed cancel
        is logged by type and retried on the next poll; it never blocks anything. Returns whether
        ``rec`` changed."""
        over = wall > self.cfg.caps.job_wall_limit_s
        switched_off = not self.kill.state(now).open
        asked = _parse_iso(rec.get("cancel_requested_at"))
        again = asked is not None and (now - asked).total_seconds() > self.cfg.caps.cancel_grace_s
        if not (over or switched_off) or (rec.get("cancel_requested") and not again):
            return False
        if not rec.get("cancel_requested") and rec.get("status") != "failed":
            limit_min = round(self.cfg.caps.job_wall_limit_s / 60)
            rec["status"] = "failed"
            rec["error"] = (
                f"the run passed the {limit_min}-minute wall-time limit and was stopped"
                if over
                else "the runner was switched off and stopped this run"
            )
        try:
            jobs.cancel(str(rec["job_id"]))
            rec["cancel_requested"] = True
            rec["cancel_requested_at"] = _iso(now)
            rec["cancels"] = int(rec.get("cancels") or 0) + 1
        except Exception as exc:  # noqa: BLE001 - retried on the next poll
            log_event("error", route="cancel", error_type=type(exc).__name__)
        return True

    def _valid_result(
        self, rec: Mapping[str, Any], manifest: dict[str, Any] | None, summary: dict[str, Any] | None
    ) -> bool:
        return (
            isinstance(manifest, dict)
            and manifest.get("exit_status") == 0
            and isinstance(summary, dict)
            and summary.get("format") == SUMMARY_FORMAT
            and summary.get("run_id") == rec["run_id"]
            and summary.get("status") == "done"
        )

    def _give_up(
        self,
        rec: dict[str, Any],
        stored: dict[str, Any],
        version: int,
        now: datetime,
        age: float,
        jobs: JobBackend | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Settle a run the platform could not be asked about for _GIVE_UP_S.

        With the backend at hand and a job id, the job's published result is read first (Object
        Storage, not the CLI): a complete one finishes the run as ``done`` at its real cost (job
        creation to the manifest's ``finished_at``). Otherwise the job is cancelled (best effort:
        it may still be running) and the run settles on the safe side: with a job, at its cost
        from creation to now (more than the job, which the platform stops after an hour, can have
        run); with no job id, at its full reservation (the job may exist)."""
        reserved = int(rec["reserved_micros"])
        if rec.get("job_id") and jobs is not None:
            try:
                manifest, summary = jobs.read_result(str(rec["run_id"]))
            except Exception as exc:  # noqa: BLE001 - unreadable: settle on the safe side
                log_event("error", route="result", error_type=type(exc).__name__)
                manifest = summary = None
            if self._valid_result(rec, manifest, summary):
                assert manifest is not None
                start = _parse_iso(rec.get("job_created_at")) or _parse_iso(rec.get("created_at"))
                finished = _parse_iso(manifest.get("finished_at"))
                wall = age  # without a usable finish time: the safe side
                try:
                    if start is not None and finished is not None and start <= finished <= now:
                        wall = (finished - start).total_seconds()
                except TypeError:  # a time without a zone cannot be compared: keep the safe side
                    pass
                actual = capmod.job_cost_micros(self.cfg, wall)
                rec.update(
                    status="done",
                    error=None,
                    result=summary,
                    settled=True,
                    settled_micros=actual,
                    wall_s=round(wall, 1),
                )
                bumps = capmod.settle_bumps(self.cfg, rec["spend_day"], reserved, actual, now)
                return self._save(rec, version, bumps, False, now, stored)
            try:  # the job may still be running: stop it before the runner stops tracking it
                jobs.cancel(str(rec["job_id"]))
                rec["cancels"] = int(rec.get("cancels") or 0) + 1
            except Exception as exc:  # noqa: BLE001 - never blocks settling
                log_event("error", route="cancel", error_type=type(exc).__name__)
        if rec.get("job_id"):
            actual = capmod.job_cost_micros(self.cfg, age)
            error = "the simulation job's state could not be read; its cost to now was counted"
        else:
            actual = reserved
            error = "the simulation job could not be confirmed; its full reservation was counted"
        rec.update(
            status="failed",
            error=rec.get("error") or error,
            settled=True,
            settled_micros=actual,
            wall_s=round(age, 1),
        )
        bumps = capmod.settle_bumps(self.cfg, rec["spend_day"], reserved, actual, now)
        return self._save(rec, version, bumps, False, now, stored)

    def _finish(self, rec: dict[str, Any], st: JobStatus, jobs: JobBackend) -> None:
        manifest, summary = jobs.read_result(str(rec["run_id"]))
        if st.succeeded and self._valid_result(rec, manifest, summary):
            rec.update(status="done", error=None, result=summary)
            return
        rec["status"] = "failed"
        if rec.get("error"):
            return
        note = summary.get("error") if isinstance(summary, dict) else None
        if isinstance(note, str) and note:
            rec["error"] = f"the search failed: {note[:300]}"
        elif st.succeeded:
            rec["error"] = "the job finished but its summary could not be read"
        else:
            rec["error"] = f"the simulation job ended {st.state.lower()}"

    def _save(
        self,
        rec: dict[str, Any],
        version: int,
        bumps: list[Bump],
        active: bool,
        now: datetime,
        stored: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        """Write the moved-on record (and its spend) at ``version``. On a lost race, the winner's
        record; on any other store error, the stored record unchanged (logged by type): the run
        stays active, so the next poll or sweep tries again."""
        try:
            self.store.commit(bumps, self._write(rec, version, active, now))
        except Conflict:
            fresh = self.store.get_run(str(rec["run_id"]))
            return fresh if fresh is not None else (rec, version)
        except Exception as exc:  # noqa: BLE001 - nothing was written: the run stays active and is retried
            log_event("error", route="save", error_type=type(exc).__name__)
            return stored, version
        return rec, version + 1

    def sweep(self) -> dict[str, int]:
        """Poll every run in flight (the schedule calls this every two minutes)."""
        runs = self.store.active_runs()
        done = 0
        for record, version in runs:
            try:
                self.refresh(record, version, force=True)
                done += 1
            except Exception as exc:  # noqa: BLE001 - one bad run must not stop the rest
                log_event("error", route="sweep", error_type=type(exc).__name__)
        log_event("sweep", task="sweep", count=len(runs))
        return {"active": len(runs), "refreshed": done}
