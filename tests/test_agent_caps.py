"""Hard limits of the Grasp Engineer: caps, day book, spend guard, guarded ledger, passcode (no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from rtt.agent.caps import (
    HARD_MAX_CONCURRENT_JOBS,
    HARD_MAX_TOOL_CALLS,
    PASSCODE_DIGEST_ENV,
    AgentCaps,
    DailySpendBook,
    GuardedLedger,
    PasscodeGate,
    PasscodeLockedError,
    SpendCapExceededError,
    SpendGuard,
    passcode_digest,
)
from rtt.llm import NEMOTRON_SUPER, BudgetExceededError, LLMClient
from rtt.llm.providers import TransportResponse


class Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class CountingTransport:
    """Answers every chat call with a fixed usage; counts calls."""

    def __init__(self, prompt_tokens: int = 1000, completion_tokens: int = 200) -> None:
        self.calls = 0
        self.usage = (prompt_tokens, completion_tokens)

    def chat_completions(self, payload: Any, timeout_s: float) -> TransportResponse:
        self.calls += 1
        body = {
            "id": "x",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": self.usage[0], "completion_tokens": self.usage[1]},
        }
        return TransportResponse(200, {}, body)

    def list_models(self, timeout_s: float) -> TransportResponse:
        return TransportResponse(200, {}, {"data": []})


# --------------------------------------------------------------------------- caps


def test_caps_defaults_are_the_hard_ceilings() -> None:
    caps = AgentCaps()
    assert caps.max_tool_calls == HARD_MAX_TOOL_CALLS == 12
    assert caps.max_concurrent_jobs == HARD_MAX_CONCURRENT_JOBS == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tool_calls": 13},
        {"max_tool_calls": 0},
        {"max_concurrent_jobs": 3},
        {"session_usd": 3.0, "day_usd": 2.0},
        {"session_usd": -1.0},
        {"max_designs_per_job": 17},
        {"poll_wait_s": 601},
    ],
)
def test_caps_cannot_exceed_hard_limits(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AgentCaps(**kwargs)


# --------------------------------------------------------------------------- day book


def test_day_book_persists_and_rolls_over(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 10, 16, 23, 0, tzinfo=UTC))
    path = tmp_path / "daily.json"
    book = DailySpendBook(path, 1.0, now=clock)
    book.charge(0.4, kind="llm", session_id="sess-a")
    book.charge(0.3, kind="compute", session_id="sess-b")
    again = DailySpendBook(path, 1.0, now=clock)  # a restarted server sees today's spend
    assert again.spent_today() == pytest.approx(0.7)
    with pytest.raises(SpendCapExceededError):
        again.reserve(0.31)
    again.reserve(0.3)
    assert again.remaining_usd() == pytest.approx(0.0)
    again.release(0.3)
    data = json.loads(path.read_text())
    day = data["days"]["2026-10-16"]
    assert day["by_kind"] == {"llm": 0.4, "compute": 0.3}
    assert day["sessions"] == ["sess-a", "sess-b"]
    clock.now += timedelta(hours=2)  # next UTC day
    assert again.spent_today() == 0.0
    again.reserve(0.9)


def test_day_book_refuses_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "daily.json"
    path.write_text("{not json")
    with pytest.raises(RuntimeError, match="unreadable"):
        DailySpendBook(path, 1.0).spent_today()


# --------------------------------------------------------------------------- guard


def test_spend_guard_enforces_session_and_day_caps() -> None:
    book = DailySpendBook(None, 0.5)
    a = SpendGuard(0.3, book, "sess-a")
    b = SpendGuard(0.3, book, "sess-b")
    a.reserve(0.2)
    with pytest.raises(SpendCapExceededError, match="session cap"):
        a.reserve(0.11)
    a.release(0.2)
    a.charge(0.25, "llm")
    with pytest.raises(SpendCapExceededError, match="daily cap"):
        b.reserve(0.26)  # b's own cap allows it, the day does not
    b.reserve(0.25)
    assert book.remaining_usd() == pytest.approx(0.0)
    snap = a.snapshot()
    assert snap["spent_usd"] == pytest.approx(0.25)
    assert snap["by_kind_usd"] == {"llm": 0.25}
    assert snap["day_spent_usd"] == pytest.approx(0.25)


def test_spend_cap_error_is_a_budget_error() -> None:
    assert issubclass(SpendCapExceededError, BudgetExceededError)


def test_guarded_ledger_refuses_before_any_request_and_charges_after() -> None:
    book = DailySpendBook(None, 2.0)
    guard = SpendGuard(0.002, book, "sess-a")
    transport = CountingTransport()
    client = LLMClient(NEMOTRON_SUPER, max_tokens=4096, ledger=GuardedLedger(guard), transport=transport)
    with pytest.raises(BudgetExceededError):  # worst case: 4096 output tokens x $0.90/M > $0.002
        client.chat([{"role": "user", "content": "hi"}])
    assert transport.calls == 0
    assert guard.reserved_usd == 0.0

    roomy = SpendGuard(0.05, book, "sess-b")
    client = LLMClient(NEMOTRON_SUPER, max_tokens=256, ledger=GuardedLedger(roomy), transport=transport)
    client.chat([{"role": "user", "content": "hi"}])
    assert transport.calls == 1
    expected = (1000 * 0.30 + 200 * 0.90) / 1e6
    assert roomy.spent_usd == pytest.approx(expected)
    assert roomy.reserved_usd == 0.0
    assert book.spent_today() == pytest.approx(expected)


# --------------------------------------------------------------------------- passcode


def test_passcode_gate_without_digest_never_unlocks() -> None:
    gate = PasscodeGate(None)
    assert not gate.enabled
    assert gate.check("anything") is False
    assert gate.check(None) is False


def test_passcode_gate_checks_and_locks_out() -> None:
    now = [0.0]
    gate = PasscodeGate(passcode_digest("open sesame"), max_failures=3, window_s=60.0, clock=lambda: now[0])
    assert gate.check("open sesame") is True
    for _ in range(3):
        assert gate.check("wrong") is False
    with pytest.raises(PasscodeLockedError):
        gate.check("open sesame")  # locked even for the right code
    now[0] = 61.0
    assert gate.check("open sesame") is True


def test_passcode_gate_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(PASSCODE_DIGEST_ENV, passcode_digest("pw"))
    assert PasscodeGate.from_env(tmp_path / "missing.env").check("pw")
    monkeypatch.delenv(PASSCODE_DIGEST_ENV)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{PASSCODE_DIGEST_ENV}={passcode_digest('pw2')}\n")
    assert PasscodeGate.from_env(env_file).check("pw2")


def test_passcode_gate_rejects_a_malformed_digest() -> None:
    with pytest.raises(ValueError):
        PasscodeGate("not-a-digest")


# --------------------------------------------------------------------------- hardening (verifier, round 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"session_usd": float("nan")},
        {"day_usd": float("nan")},
        {"session_usd": 1.0, "day_usd": float("inf")},
        {"session_usd": float("inf"), "day_usd": float("inf")},
    ],
)
def test_caps_refuse_non_finite_dollars(kwargs: dict[str, Any]) -> None:
    # NaN compares false with everything: a NaN cap would never refuse a reservation
    with pytest.raises(ValueError):
        AgentCaps(**kwargs)
    with pytest.raises(ValueError):
        DailySpendBook(None, float("nan"))
    with pytest.raises(ValueError):
        SpendGuard(float("nan"), DailySpendBook(None, 1.0), "sess-0123456789ab")


def test_a_nan_amount_cannot_poison_the_caps() -> None:
    book = DailySpendBook(None, 1.0)
    guard = SpendGuard(0.5, book, "sess-0123456789ab")
    for bad in (float("nan"), float("inf")):
        with pytest.raises(SpendCapExceededError):
            guard.reserve(bad)
        with pytest.raises(SpendCapExceededError):
            book.reserve(bad)
        with pytest.raises(SpendCapExceededError):
            guard.charge(bad, "compute")
    assert guard.reserved_usd == 0.0 and book.reserved_usd == 0.0 and guard.spent_usd == 0.0
    with pytest.raises(SpendCapExceededError):  # the caps still hold afterwards
        guard.reserve(0.6)


def test_day_book_refuses_a_non_finite_total(tmp_path: Path) -> None:
    path = tmp_path / "daily.json"
    path.write_text(
        '{"format": "rtt-agent-daily-spend", "version": 1, "days": {"2026-10-16": {"usd": NaN}}}',
        encoding="utf-8",
    )
    book = DailySpendBook(path, 2.0, now=Clock(datetime(2026, 10, 16, 12, tzinfo=UTC)))
    with pytest.raises(RuntimeError, match="invalid total"):
        book.reserve(0.01)
