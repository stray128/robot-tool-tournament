"""Structured logs of the runner: one JSON line per request, nothing else about it.

A request line holds the request id, the route template (``/runs/{id}``, never the raw path), the
status and the duration. No body, header, IP, run id, key or exception text is ever logged; an
internal error logs the exception's type name only.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Final

__all__ = ["LOGGER_NAME", "configure_logging", "log_event"]

LOGGER_NAME: Final = "rtt.runner"
#: Libraries whose debug output could carry request details; kept at WARNING in production.
QUIET: Final = ("openai", "httpx", "httpx2", "httpcore", "botocore", "boto3", "urllib3", "s3transfer")
_ALLOWED: Final = frozenset(
    {"event", "request_id", "route", "method", "status", "duration_ms", "error_type", "task", "count"}
)


def log_event(event: str, **fields: Any) -> None:
    """Log one JSON line; only the fields named in ``_ALLOWED`` pass."""
    line = {"event": event, **{k: v for k, v in fields.items() if k in _ALLOWED}}
    logging.getLogger(LOGGER_NAME).info(json.dumps(line, sort_keys=True, default=str))


def configure_logging() -> None:
    """Production logging (Lambda, ``rtt runner serve``): the runner's lines to stdout, the noisy
    libraries at WARNING."""
    logger = logging.getLogger(LOGGER_NAME)
    if not any(getattr(h, "_rtt_runner", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._rtt_runner = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for name in QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
