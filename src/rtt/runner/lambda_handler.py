"""AWS Lambda entry point: ``rtt.runner.lambda_handler.handler``.

One :class:`~rtt.runner.app.Runner` per container, built on the first event from ``RUNNER_*``
environment variables (no secret among them). Secrets are read from Secrets Manager when first
needed and kept in memory for the container's lifetime.

Two kinds of event arrive:

* Function URL requests (``requestContext.http``): the API.
* ``{"runner_task": "sweep"}`` from the EventBridge schedule: poll, cancel and settle runs in
  flight. A Function URL request cannot produce this event (its fields arrive inside ``body``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from rtt.runner.app import Runner
from rtt.runner.caps import KillSwitch
from rtt.runner.config import RunnerConfig
from rtt.runner.engineer import Planner
from rtt.runner.logs import configure_logging, log_event
from rtt.runner.nebius import JobBackend, NebiusBackend
from rtt.runner.secrets import NebiusCredentials, SecretsManagerSource
from rtt.runner.store import DynamoStore

__all__ = ["build_runner", "handler"]

_RUNNER: Runner | None = None


def build_runner(environ: Mapping[str, str] | None = None) -> Runner:
    cfg = RunnerConfig.from_env(environ)
    if not cfg.table_name:
        raise RuntimeError("RUNNER_TABLE_NAME is not set")
    secrets = SecretsManagerSource()

    def jobs() -> JobBackend:
        creds = NebiusCredentials.parse(secrets.get(cfg.nebius_secret), cfg.nebius_secret)
        return NebiusBackend.from_credentials(cfg.nebius, creds, job_env=cfg.job_env())

    return Runner(cfg, DynamoStore(cfg.table_name), Planner(cfg), jobs, secrets, KillSwitch(cfg))


def _json_response(status: int, error: str) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json", "Cache-Control": "no-store"},
        "body": json.dumps({"error": error}),
    }


def handler(event: Mapping[str, Any], context: Any = None) -> Any:
    """Every failure, the runner's construction included, ends as a JSON 500 for a request (or an
    error type for the schedule); only the exception's type is logged."""
    global _RUNNER
    is_request = isinstance(event.get("requestContext"), Mapping)
    try:
        if _RUNNER is None:
            configure_logging()
            _RUNNER = build_runner()
        if event.get("runner_task") == "sweep":
            return _RUNNER.sweep()
        if is_request:
            return _RUNNER.handle_event(event, context)
        return _json_response(400, "unknown event")
    except Exception as exc:  # noqa: BLE001
        log_event("error", route="handler", error_type=type(exc).__name__)
        if is_request:
            return _json_response(500, "internal error")
        return {"error": type(exc).__name__}
