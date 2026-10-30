"""A visitor's API key is used for one request and never logged, stored, cached or echoed.

Every path a key can take (a valid plan, a refusal, a rejected key, a billing refusal, a provider
error that echoes the key, a transport that raises with the key in its message, and the real
OpenAI-compatible transport over mocked HTTP) is run while every log record at DEBUG and
everything written to stdout and stderr is captured; the key must appear in none of them, nor in
the response, nor in the store."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx2
import pytest

from rtt.llm.client import TokenBucket
from rtt.llm.providers import OpenAITransport
from rtt.runner.engineer import Planner, redact
from rtt.runner.logs import LOGGER_NAME, QUIET, configure_logging, log_event
from runner_support import FREE_KEY, VISITOR_KEY, FakeModel, Harness, good_args


@pytest.fixture
def all_logs() -> Iterator[io.StringIO]:
    """Every record of every logger at DEBUG, formatted with its message and any exception."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    old = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    saved = {name: logging.getLogger(name).level for name in (*QUIET, LOGGER_NAME)}
    for name in saved:
        logging.getLogger(name).setLevel(logging.DEBUG)
    try:
        yield buf
    finally:
        root.removeHandler(handler)
        root.setLevel(old)
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


def _assert_absent(secret: str, *texts: str) -> None:
    for text in texts:
        assert secret not in text


def _store_text(h: Harness) -> str:
    return json.dumps(h.store.items, sort_keys=True, default=str)


@pytest.mark.parametrize(
    ("answers", "expect"),
    [
        ([None], 200),  # a valid plan
        ([401], 401),
        ([403], 401),
        ([402], 401),
        ([500, 500, 500, 500], 200),  # outage: a failed plan whose problems quote the provider
        ([RuntimeError(f"socket closed while sending Authorization: Bearer {VISITOR_KEY}")], 500),
    ],
)
@pytest.mark.parametrize("provider", ["nvidia", "tokenfactory"])
def test_the_visitors_key_never_leaves_the_request(
    answers: list[Any], expect: int, provider: str, all_logs: io.StringIO, capsys: pytest.CaptureFixture[str]
) -> None:
    script = [good_args() if a is None else a for a in answers]
    h = Harness(answers=script)
    status, body = h.plan(provider, VISITOR_KEY, text=f"Open a drawer. My key is {VISITOR_KEY}, plan it.")
    assert status == expect
    out = capsys.readouterr()
    _assert_absent(VISITOR_KEY, json.dumps(body), all_logs.getvalue(), out.out, out.err, _store_text(h))
    # the request log line is there, with nothing but its allowed fields
    lines = [
        json.loads(line.split(" ", 2)[2])
        for line in all_logs.getvalue().splitlines()
        if line.startswith(LOGGER_NAME)
    ]
    request = [ln for ln in lines if ln["event"] == "request"]
    assert request and set(request[-1]) == {"event", "request_id", "route", "method", "status", "duration_ms"}
    assert request[-1]["route"] == "/plan" and request[-1]["status"] == expect


def test_the_free_key_never_leaves_the_server(
    all_logs: io.StringIO, capsys: pytest.CaptureFixture[str]
) -> None:
    for answers in ([good_args()], [401], [500, 500, 500, 500]):
        h = Harness(answers=answers)
        status, body = h.plan("free")
        out = capsys.readouterr()
        _assert_absent(FREE_KEY, json.dumps(body), all_logs.getvalue(), out.out, out.err, _store_text(h))
        assert status in (200, 503)


def test_the_real_transport_over_mocked_http_never_leaks_the_key(
    all_logs: io.StringIO, capsys: pytest.CaptureFixture[str]
) -> None:
    """The OpenAI SDK path with every library logger at DEBUG: the provider echoes the bearer
    header in its error body."""
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        auth = request.headers["authorization"]
        seen.append(auth)
        return httpx2.Response(401, json={"error": {"message": f"invalid key: {auth}"}})

    h = Harness()

    def factory(provider: Any, key: str) -> OpenAITransport:
        return OpenAITransport(
            key, provider.base_url, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
        )

    h.runner.planner = Planner(h.cfg, transport_factory=factory, limiter=TokenBucket(1e12))
    status, body = h.plan("nvidia", VISITOR_KEY)
    assert status == 401 and body == {"error": "the provider rejected this key"}
    assert seen == [f"Bearer {VISITOR_KEY}"]  # the key did reach the provider, once
    out = capsys.readouterr()
    _assert_absent(VISITOR_KEY, json.dumps(body), all_logs.getvalue(), out.out, out.err, _store_text(h))


def test_production_logging_quiets_the_http_libraries() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    saved = {name: logging.getLogger(name).level for name in QUIET}
    try:
        configure_logging()
        configure_logging()  # idempotent: one handler
        assert sum(1 for x in logger.handlers if getattr(x, "_rtt_runner", False)) == 1
        assert all(logging.getLogger(name).level == logging.WARNING for name in QUIET)
        assert logger.propagate is False
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate
        for name, lvl in saved.items():
            logging.getLogger(name).setLevel(lvl)


def test_log_event_drops_every_field_it_does_not_know(all_logs: io.StringIO) -> None:
    log_event(
        "request", route="/plan", status=200, key=VISITOR_KEY, body={"key": VISITOR_KEY}, ip="203.0.113.7"
    )
    text = all_logs.getvalue()
    assert VISITOR_KEY not in text and "203.0.113.7" not in text and '"route": "/plan"' in text


def test_redact_reaches_every_string() -> None:
    value = {"a": [f"x{VISITOR_KEY}y", {"b": VISITOR_KEY}], VISITOR_KEY: 1, "n": 3}
    out = redact(value, VISITOR_KEY)
    assert VISITOR_KEY not in json.dumps(out) and out["n"] == 3


def test_ip_addresses_are_never_stored() -> None:
    h = Harness()
    h.plan("free", ip="198.51.100.77")
    h.start_run(ip="198.51.100.77")
    assert "198.51.100.77" not in _store_text(h)
    keys = [k for k in h.store.items if k.startswith("count#free-plans#") and k.count("#") == 3]
    (key,) = keys
    ip_hash = key.split("#")[2]
    assert len(ip_hash) == 32 and int(ip_hash, 16) >= 0


def test_fake_model_echoes_the_key_it_was_given() -> None:
    """Guard on the fake itself: the error body really does carry the key, so the tests above
    would catch a leak."""
    model = FakeModel([401], VISITOR_KEY)
    resp = model.chat_completions({"messages": []}, 1.0)
    assert VISITOR_KEY in json.dumps(resp.body) and VISITOR_KEY in resp.error_text
