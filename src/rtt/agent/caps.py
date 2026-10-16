"""Hard limits for Grasp Engineer sessions: tool calls, concurrent jobs, dollars and the passcode.

Every limit here is enforced in code, not in the prompt. The model is told about the
limits so it can plan, but a request past a limit is refused by the toolbox or by the
spend guard before anything is sent or launched.

* :class:`AgentCaps`: per-session limits. Values above the hard ceilings
  (:data:`HARD_MAX_TOOL_CALLS`, :data:`HARD_MAX_CONCURRENT_JOBS`) are rejected.
* :class:`DailySpendBook`: spend per UTC day, persisted as JSON and shared by every
  session of a process, with a day cap and in-flight reservations.
* :class:`SpendGuard`: one session's spend (Token Factory plus estimated compute). A
  reservation must fit both the session cap and the day cap.
* :class:`GuardedLedger`: an :class:`rtt.llm.Ledger` whose reservations go through a
  :class:`SpendGuard`, so every model call of a session (agent, proposer, summarizer)
  draws on one budget.
* :class:`PasscodeGate`: anything that spends needs a passcode whose SHA-256 is set
  in the environment (``RTT_AGENT_PASSCODE_SHA256``) or ``.env``. With no digest
  configured, nothing can spend. Repeated wrong guesses lock the gate for a while.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rtt.llm.ledger import BudgetExceededError, Ledger, LedgerEntry
from rtt.store.manifest import write_json_atomic

__all__ = [
    "HARD_MAX_CONCURRENT_JOBS",
    "HARD_MAX_TOOL_CALLS",
    "PASSCODE_DIGEST_ENV",
    "AgentCaps",
    "CapError",
    "ConcurrencyCapError",
    "DailySpendBook",
    "GuardedLedger",
    "PasscodeGate",
    "PasscodeLockedError",
    "PasscodeRequiredError",
    "SpendCapExceededError",
    "SpendGuard",
    "ToolBudgetExhaustedError",
    "passcode_digest",
]

#: No configuration may allow more tool calls per session than this.
HARD_MAX_TOOL_CALLS: Final = 12
#: No configuration may allow more simulation jobs in flight per session than this.
HARD_MAX_CONCURRENT_JOBS: Final = 2
#: Environment variable (or ``.env`` key) holding the hex SHA-256 of the spend passcode.
PASSCODE_DIGEST_ENV: Final = "RTT_AGENT_PASSCODE_SHA256"
_DAY_FORMAT: Final = "rtt-agent-daily-spend"


# --------------------------------------------------------------------------- errors


class CapError(RuntimeError):
    """A request was refused by a hard limit. ``code`` is a stable machine-readable tag."""

    code = "cap"


class ToolBudgetExhaustedError(CapError):
    """The session has used all of its tool calls."""

    code = "tool_budget_exhausted"


class ConcurrencyCapError(CapError):
    """The session already has the maximum number of jobs in flight."""

    code = "too_many_jobs"


class SpendCapExceededError(CapError, BudgetExceededError):
    """Spending more would exceed the session or day cap.

    Also a :class:`~rtt.llm.ledger.BudgetExceededError`, so the LLM client treats it
    like its own budget refusal (no request is sent).
    """

    code = "spend_cap"


class PasscodeRequiredError(CapError):
    """The action spends money and the session was not unlocked with the passcode."""

    code = "passcode_required"


class PasscodeLockedError(CapError):
    """Too many wrong passcodes; the gate is locked for a while."""

    code = "passcode_locked"


# --------------------------------------------------------------------------- caps


@dataclass(frozen=True)
class AgentCaps:
    """Per-session limits (hard ceilings cannot be raised by configuration).

    Attributes:
        max_tool_calls: Tool calls the agent may make, including refused or invalid ones.
        max_concurrent_jobs: Simulation jobs in flight at once.
        session_usd: Dollar cap for one session (Token Factory plus estimated compute).
        day_usd: Dollar cap per UTC day over all sessions sharing a :class:`DailySpendBook`.
        max_llm_turns: Agent model calls, including the final answer.
        max_designs_per_job: Designs one ``launch_sim_job`` may simulate.
        max_designs_per_proposal: Designs one ``propose_designs`` call may ask for.
        poll_wait_s: How long ``poll_job`` blocks waiting for a job to finish.
    """

    max_tool_calls: int = HARD_MAX_TOOL_CALLS
    max_concurrent_jobs: int = HARD_MAX_CONCURRENT_JOBS
    session_usd: float = 0.50
    day_usd: float = 2.00
    max_llm_turns: int = 16
    max_designs_per_job: int = 8
    max_designs_per_proposal: int = 4
    poll_wait_s: float = 120.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_tool_calls <= HARD_MAX_TOOL_CALLS:
            raise ValueError(f"max_tool_calls must be in 1..{HARD_MAX_TOOL_CALLS}, got {self.max_tool_calls}")
        if not 1 <= self.max_concurrent_jobs <= HARD_MAX_CONCURRENT_JOBS:
            raise ValueError(
                f"max_concurrent_jobs must be in 1..{HARD_MAX_CONCURRENT_JOBS}, got {self.max_concurrent_jobs}"
            )
        # NaN compares false with everything, so a NaN cap would never refuse anything;
        # an infinite cap is no cap. Both are configuration errors, not limits.
        if not (math.isfinite(self.session_usd) and math.isfinite(self.day_usd)):
            raise ValueError("dollar caps must be finite numbers")
        if self.session_usd < 0 or self.day_usd < 0:
            raise ValueError("dollar caps must be non-negative")
        if self.session_usd > self.day_usd:
            raise ValueError("session_usd cannot exceed day_usd")
        if self.max_llm_turns < 1:
            raise ValueError("max_llm_turns must be at least 1")
        if not 1 <= self.max_designs_per_job <= 16:
            raise ValueError("max_designs_per_job must be in 1..16")
        if not 1 <= self.max_designs_per_proposal <= 8:
            raise ValueError("max_designs_per_proposal must be in 1..8")
        if not 0 <= self.poll_wait_s <= 600:
            raise ValueError("poll_wait_s must be in 0..600")

    def to_json(self) -> dict[str, Any]:
        return {
            "max_tool_calls": self.max_tool_calls,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "session_usd": self.session_usd,
            "day_usd": self.day_usd,
            "max_llm_turns": self.max_llm_turns,
            "max_designs_per_job": self.max_designs_per_job,
            "max_designs_per_proposal": self.max_designs_per_proposal,
            "poll_wait_s": self.poll_wait_s,
        }


# --------------------------------------------------------------------------- day book


def _check_amount(usd: float, what: str) -> None:
    """Refuse a NaN, infinite or negative dollar amount.

    NaN compares false with everything: one NaN reservation or charge would make every
    later cap comparison false, so nothing would ever be refused again.
    """
    if not math.isfinite(usd):
        raise SpendCapExceededError(f"refused: a non-finite {what} (${usd!r})")
    if usd < 0:
        raise ValueError(f"{what} must be non-negative")


def _utc_day(now: datetime) -> str:
    return now.astimezone(UTC).strftime("%Y-%m-%d")


class DailySpendBook:
    """Spend per UTC day, persisted as JSON, with a cap and in-flight reservations.

    Every charge is written through to ``path`` at once (re-reading the file first), so
    a restarted server still knows what today has cost. Reservations live in memory:
    they are the worst cases of calls and jobs in flight in this process.

    Args:
        path: JSON file (``None``: memory only).
        cap_usd: Maximum spend per UTC day.
        now: Clock returning an aware ``datetime`` (tests pass a fake).
    """

    def __init__(
        self,
        path: str | Path | None,
        cap_usd: float,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not math.isfinite(cap_usd) or cap_usd < 0:
            raise ValueError("cap_usd must be a finite non-negative number")
        self.path = Path(path) if path is not None else None
        self.cap_usd = float(cap_usd)
        self._now = now
        self._lock = threading.Lock()
        self._reserved = 0.0
        self._memory: dict[str, dict[str, Any]] = {}

    def __repr__(self) -> str:
        return f"DailySpendBook(path={str(self.path) if self.path else None!r}, cap_usd={self.cap_usd!r})"

    def today(self) -> str:
        return _utc_day(self._now())

    def _load(self) -> dict[str, dict[str, Any]]:
        if self.path is None:
            return self._memory
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # A corrupt book must not silently reset the day's spend to zero.
            raise RuntimeError(f"daily spend book {self.path} is unreadable: {exc}") from exc
        if not isinstance(data, Mapping) or data.get("format") != _DAY_FORMAT:
            raise RuntimeError(f"daily spend book {self.path} has an unknown format")
        days = {str(k): dict(v) for k, v in dict(data.get("days") or {}).items()}
        for day, entry in days.items():
            usd = entry.get("usd", 0.0)
            if not isinstance(usd, (int, float)) or not math.isfinite(usd) or usd < 0:
                raise RuntimeError(f"daily spend book {self.path} has an invalid total for {day}")
        return days

    def _save(self, days: dict[str, dict[str, Any]]) -> None:
        if self.path is None:
            self._memory = days
            return
        write_json_atomic(self.path, {"format": _DAY_FORMAT, "version": 1, "days": days})

    def spent_today(self) -> float:
        """Recorded spend for the current UTC day."""
        with self._lock:
            return float(self._load().get(self.today(), {}).get("usd", 0.0))

    @property
    def reserved_usd(self) -> float:
        with self._lock:
            return self._reserved

    def remaining_usd(self) -> float:
        """Cap minus today's spend and every open reservation."""
        with self._lock:
            spent = float(self._load().get(self.today(), {}).get("usd", 0.0))
            return self.cap_usd - spent - self._reserved

    def reserve(self, usd: float) -> None:
        """Hold ``usd`` of today's budget; raise :class:`SpendCapExceededError` if it does not fit."""
        _check_amount(usd, "reservation")
        with self._lock:
            spent = float(self._load().get(self.today(), {}).get("usd", 0.0))
            if spent + self._reserved + usd > self.cap_usd + 1e-12:
                raise SpendCapExceededError(
                    f"refused: ${usd:.4f} more would exceed the ${self.cap_usd:.2f} daily cap "
                    f"(${spent:.4f} spent today, ${self._reserved:.4f} in flight)"
                )
            self._reserved += usd

    def release(self, usd: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - usd)

    def charge(self, usd: float, *, kind: str, session_id: str | None = None) -> None:
        """Record ``usd`` of actual spend for today (written through to disk)."""
        _check_amount(usd, "charge")
        if usd <= 0:
            return
        with self._lock:
            days = self._load()
            day = days.setdefault(self.today(), {"usd": 0.0, "by_kind": {}, "sessions": []})
            day["usd"] = round(float(day.get("usd", 0.0)) + usd, 8)
            by_kind = day.setdefault("by_kind", {})
            by_kind[kind] = round(float(by_kind.get(kind, 0.0)) + usd, 8)
            sessions = day.setdefault("sessions", [])
            if session_id and session_id not in sessions:
                sessions.append(session_id)
            self._save(days)


# --------------------------------------------------------------------------- session guard


class SpendGuard:
    """One session's spend: Token Factory calls plus estimated compute, under two caps.

    ``reserve`` must succeed before a call or job is started; ``charge`` records the
    actual (or estimated) cost when it is known; ``release`` drops the reservation.

    ``billing=False`` is for scripted offline sessions: reservations are still checked,
    but nothing is charged, because no model was called and nothing was billed (their
    session files must not show dollars that were never spent).
    """

    def __init__(
        self, session_cap_usd: float, day_book: DailySpendBook, session_id: str, *, billing: bool = True
    ) -> None:
        if not math.isfinite(session_cap_usd) or session_cap_usd < 0:
            raise ValueError("session_cap_usd must be a finite non-negative number")
        self.cap_usd = float(session_cap_usd)
        self.day_book = day_book
        self.session_id = session_id
        self.billing = billing
        self._lock = threading.Lock()
        self._by_kind: dict[str, float] = {}
        self._reserved = 0.0

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return sum(self._by_kind.values())

    @property
    def reserved_usd(self) -> float:
        with self._lock:
            return self._reserved

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return self.cap_usd - sum(self._by_kind.values()) - self._reserved

    def reserve(self, usd: float) -> None:
        """Hold ``usd`` against the session cap and the day cap (both or neither)."""
        _check_amount(usd, "reservation")
        with self._lock:
            committed = sum(self._by_kind.values()) + self._reserved
            if committed + usd > self.cap_usd + 1e-12:
                raise SpendCapExceededError(
                    f"refused: ${usd:.4f} more would exceed the ${self.cap_usd:.2f} session cap "
                    f"(${committed:.4f} committed)"
                )
            self.day_book.reserve(usd)
            self._reserved += usd

    def release(self, usd: float) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - usd)
        self.day_book.release(usd)

    def charge(self, usd: float, kind: str) -> None:
        """Record spend of ``kind`` (``"llm"`` or ``"compute"``); nothing when not billing."""
        _check_amount(usd, "charge")
        if usd <= 0 or not self.billing:
            return
        with self._lock:
            self._by_kind[kind] = self._by_kind.get(kind, 0.0) + usd
        self.day_book.charge(usd, kind=kind, session_id=self.session_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            by_kind = dict(self._by_kind)
            reserved = self._reserved
        spent = sum(by_kind.values())
        return {
            "session_cap_usd": self.cap_usd,
            "spent_usd": round(spent, 6),
            "by_kind_usd": {k: round(v, 6) for k, v in sorted(by_kind.items())},
            "reserved_usd": round(reserved, 6),
            "session_remaining_usd": round(self.cap_usd - spent - reserved, 6),
            "day_cap_usd": self.day_book.cap_usd,
            "day_spent_usd": round(self.day_book.spent_today(), 6),
        }


class GuardedLedger(Ledger):
    """A call ledger whose budget is a :class:`SpendGuard`.

    :class:`~rtt.llm.client.LLMClient` reserves the worst case of every live call on its
    ledger; here that reservation must fit the session and day caps, and the cost of
    every recorded live call is charged to the guard as ``"llm"`` spend.
    """

    def __init__(self, guard: SpendGuard, path: str | Path | None = None) -> None:
        super().__init__(path, budget_usd=guard.cap_usd)
        self.guard = guard

    def reserve(self, estimated_cost_usd: float) -> None:
        self.guard.reserve(estimated_cost_usd)

    def release(self, estimated_cost_usd: float) -> None:
        self.guard.release(estimated_cost_usd)

    @property
    def remaining_usd(self) -> float | None:
        return self.guard.remaining_usd

    def record(self, entry: LedgerEntry) -> None:
        super().record(entry)
        if not entry.cached:
            self.guard.charge(entry.cost_usd, "llm")


# --------------------------------------------------------------------------- passcode


def passcode_digest(passcode: str) -> str:
    """Hex SHA-256 of a passcode (what goes in ``RTT_AGENT_PASSCODE_SHA256``)."""
    return hashlib.sha256(passcode.encode("utf-8")).hexdigest()


class PasscodeGate:
    """Checks a passcode against a configured SHA-256 digest, with a failure lockout.

    Args:
        digest_hex: Expected hex SHA-256, or ``None`` (no passcode configured: every
            check fails, so nothing can spend).
        max_failures: Wrong guesses allowed within ``window_s`` before locking.
        window_s: Lockout window in seconds.
        clock: Monotonic clock (tests pass a fake).
    """

    def __init__(
        self,
        digest_hex: str | None,
        *,
        max_failures: int = 5,
        window_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        digest = (digest_hex or "").strip().lower()
        if digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"{PASSCODE_DIGEST_ENV} must be 64 hex characters")
        self._digest = digest or None
        self.max_failures = max_failures
        self.window_s = window_s
        self._clock = clock
        self._failures: list[float] = []
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"PasscodeGate(enabled={self.enabled})"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None, **kwargs: Any) -> PasscodeGate:
        """Digest from ``RTT_AGENT_PASSCODE_SHA256`` in the environment, else in ``.env``."""
        digest = os.environ.get(PASSCODE_DIGEST_ENV, "").strip()
        if not digest:
            from dotenv import dotenv_values, find_dotenv

            path = str(env_file) if env_file is not None else find_dotenv(usecwd=True)
            if path:
                digest = (dotenv_values(path).get(PASSCODE_DIGEST_ENV) or "").strip()
        return cls(digest or None, **kwargs)

    @property
    def enabled(self) -> bool:
        """Whether a passcode is configured at all."""
        return self._digest is not None

    def check(self, passcode: str | None) -> bool:
        """``True`` for the right passcode.

        Raises:
            PasscodeLockedError: too many recent wrong guesses (checked before comparing).
        """
        with self._lock:
            now = self._clock()
            self._failures = [t for t in self._failures if now - t < self.window_s]
            if len(self._failures) >= self.max_failures:
                raise PasscodeLockedError("too many wrong passcodes; try again later")
            if self._digest is None or not passcode:
                if passcode:
                    self._failures.append(now)
                return False
            ok = hmac.compare_digest(passcode_digest(passcode), self._digest)
            if not ok:
                self._failures.append(now)
            return ok
