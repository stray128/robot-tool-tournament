"""Append-only JSONL ledger of LLM calls, with per-model totals and a spend cap.

Each line records one call: model, mode, request hash, whether it was served
from cache, latency, token usage, estimated cost and outcome. Message bodies
are not written (the request hash points into the cache for that), and no
credential ever reaches the ledger.

A :class:`Ledger` can carry a hard budget. Callers *reserve* a worst-case cost
estimate before a live request and release it afterwards; a reservation that
would push spend past the budget raises :class:`BudgetExceededError` before
any network traffic happens. Several clients can share one ledger, and so one
budget.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BudgetExceededError(RuntimeError):
    """A live call was refused because it could exceed the spend cap."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class LedgerEntry:
    """One LLM call as recorded in the ledger."""

    model: str
    mode: str
    request_hash: str
    cached: bool
    ok: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    attempts: int = 1
    provider: str = "nebius-tokenfactory"
    error: str | None = None
    tag: str | None = None
    ts: str = field(default_factory=utc_now_iso)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> LedgerEntry:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ModelTotals:
    """Aggregates for one model. ``cost_usd`` counts live (uncached) calls only."""

    calls: int = 0
    live_calls: int = 0
    cached_calls: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    live_latency_s: float = 0.0

    def add(self, entry: LedgerEntry) -> None:
        self.calls += 1
        if not entry.ok:
            self.errors += 1
        if entry.cached:
            self.cached_calls += 1
            return
        self.live_calls += 1
        self.prompt_tokens += entry.prompt_tokens
        self.completion_tokens += entry.completion_tokens
        self.cost_usd += entry.cost_usd
        self.live_latency_s += entry.latency_s


def summarize(entries: Iterable[LedgerEntry]) -> dict[str, ModelTotals]:
    """Totals keyed by model ID."""
    totals: dict[str, ModelTotals] = {}
    for entry in entries:
        totals.setdefault(entry.model, ModelTotals()).add(entry)
    return totals


class Ledger:
    """Thread-safe call ledger, optionally persisted as JSONL and budget-capped.

    Args:
        path: JSONL file to append to. ``None`` keeps entries in memory only.
        budget_usd: Hard cap on live spend recorded through this instance.
            ``None`` disables the cap.
    """

    def __init__(self, path: str | Path | None = None, budget_usd: float | None = None) -> None:
        if budget_usd is not None and budget_usd < 0:
            raise ValueError("budget_usd must be non-negative")
        self.path = Path(path) if path is not None else None
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        self._session: list[LedgerEntry] = []
        self._spent_usd = 0.0
        self._reserved_usd = 0.0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Ledger(path={str(self.path) if self.path else None!r}, budget_usd={self.budget_usd!r})"

    @property
    def spent_usd(self) -> float:
        """Live spend recorded through this instance (cached calls are free)."""
        with self._lock:
            return self._spent_usd

    @property
    def remaining_usd(self) -> float | None:
        """Budget left after spend and open reservations, or ``None`` if uncapped."""
        if self.budget_usd is None:
            return None
        with self._lock:
            return self.budget_usd - self._spent_usd - self._reserved_usd

    def reserve(self, estimated_cost_usd: float) -> None:
        """Hold ``estimated_cost_usd`` of budget for an in-flight live call.

        Raises:
            BudgetExceededError: if spend plus open reservations plus this
                estimate would exceed the budget.
        """
        with self._lock:
            if self.budget_usd is not None:
                projected = self._spent_usd + self._reserved_usd + estimated_cost_usd
                if projected > self.budget_usd:
                    raise BudgetExceededError(
                        f"call refused: worst-case ${estimated_cost_usd:.4f} on top of "
                        f"${self._spent_usd + self._reserved_usd:.4f} committed exceeds the "
                        f"${self.budget_usd:.4f} budget"
                    )
            self._reserved_usd += estimated_cost_usd

    def release(self, estimated_cost_usd: float) -> None:
        """Drop a reservation made with :meth:`reserve`."""
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - estimated_cost_usd)

    def record(self, entry: LedgerEntry) -> None:
        """Append ``entry`` (to memory and, if configured, the JSONL file)."""
        line = json.dumps(entry.to_json(), sort_keys=True, ensure_ascii=False)
        with self._lock:
            self._session.append(entry)
            if not entry.cached:
                self._spent_usd += entry.cost_usd
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def session_entries(self) -> list[LedgerEntry]:
        """Entries recorded through this instance, oldest first."""
        with self._lock:
            return list(self._session)

    def entries(self) -> list[LedgerEntry]:
        """All entries: the whole JSONL file if persisted, else this session's."""
        if self.path is None:
            return self.session_entries()
        with self._lock:
            return read_ledger(self.path)

    def totals_by_model(self) -> dict[str, ModelTotals]:
        """Per-model totals over :meth:`entries`."""
        return summarize(self.entries())

    def total_cost_usd(self) -> float:
        """Live spend over :meth:`entries`."""
        return sum(t.cost_usd for t in self.totals_by_model().values())


def read_ledger(path: str | Path) -> list[LedgerEntry]:
    """Parse a ledger JSONL file (missing file -> empty list)."""
    p = Path(path)
    if not p.is_file():
        return []
    entries = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(LedgerEntry.from_json(json.loads(line)))
    return entries
