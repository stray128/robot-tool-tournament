"""Session recording: every agent step as an event, persisted as one JSON file for playback.

A session file (``format: rtt-agent-session``) holds the session facts (part, goal,
models, caps, the product-mode label), the ordered event list and the full chat
transcript (system prompt, model messages with tool calls, tool results). The static
site and the video play a session back from this file; the API streams the same events
live over SSE while the session runs.

Events (``type``):

* ``session.started``: part, goal, models, caps, label.
* ``llm.call``: one model call (``role`` agent / proposer / summarizer, tokens, cost,
  latency, whether it came from cache).
* ``agent.message``: the agent's text for one turn and the tools it asked for (plus a
  truncated copy of its reasoning, when the model returned one).
* ``tool.call`` / ``tool.result`` / ``tool.refused``: each tool call with validated
  arguments, its result or error, or the cap that refused it.
* ``job.update``: a simulation job changed state.
* ``summary``: a short user-facing summary written by Nemotron Lightning.
* ``session.finished``: final status, the agent's answer, physics verdicts, spend.

The file is rewritten atomically after every event, so a crash loses at most the
step in progress. Nothing recorded here contains a local path or a credential.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from rtt.store.manifest import write_json_atomic

__all__ = [
    "PRODUCT_MODE_LABEL",
    "SCRIPTED_MODEL",
    "SESSION_FORMAT",
    "SessionRecorder",
    "load_session",
    "redact_paths",
]

SESSION_FORMAT: Final = "rtt-agent-session"
SESSION_VERSION: Final = 1
PRODUCT_MODE_LABEL: Final = "Product mode: not part of the pre-registered experiment"
#: Model name recorded for sessions played by :class:`rtt.agent.fakes.ScriptedAgentTransport`.
SCRIPTED_MODEL: Final = "scripted-offline"
TERMINAL_STATUSES: Final = frozenset({"completed", "stopped", "failed"})
#: An absolute filesystem path with at least two folders (``/a/b/file``). The look-behind
#: skips URLs (``https://host/a/b``) and API paths such as ``/printpack/<id>``.
_ABS_PATH: Final = re.compile(r"(?<![\w.:/~-])/(?:[^\s/'\"`(),;:<>]+/){2,}([^\s/'\"`(),;:<>]*)")


def redact_paths(text: str) -> str:
    """``text`` with every absolute path cut to its last component (``.../source.stl``).

    Error messages from libraries (file not found, a mesh that failed to load, a
    simulator error) often name local files. They are shown to the model and written
    to the session file, which the site and the video replay, so the folders go.
    """
    return _ABS_PATH.sub(lambda m: f".../{m.group(1)}", text)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class SessionRecorder:
    """Ordered, thread-safe event log of one session, mirrored to a JSON file.

    Args:
        session_id: The session's ID.
        path: JSON file to mirror to (``None``: memory only).
        meta: Session facts merged into the top level of the file.
        clock: Wall clock in seconds since the epoch.
    """

    def __init__(
        self,
        session_id: str,
        path: str | Path | None = None,
        *,
        meta: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.session_id = session_id
        self.path = Path(path) if path is not None else None
        self._clock = clock
        self._t0 = clock()
        self._cond = threading.Condition()
        self._events: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []
        self._meta: dict[str, Any] = dict(meta or {})
        self.status = "running"
        self._save()

    def __repr__(self) -> str:
        return f"SessionRecorder({self.session_id!r}, events={len(self._events)}, status={self.status!r})"

    # -- writing

    def emit(self, type_: str, **data: Any) -> dict[str, Any]:
        """Append an event and persist; returns the event (with ``seq``, ``ts`` and ``t_s``)."""
        now = self._clock()
        with self._cond:
            event = {
                "seq": len(self._events) + 1,
                "ts": _iso(now),
                "t_s": round(now - self._t0, 3),
                "type": type_,
                **_jsonable(data),
            }
            self._events.append(event)
            self._save_locked()
            self._cond.notify_all()
        return event

    def set_messages(self, messages: Sequence[Mapping[str, Any]]) -> None:
        """Replace the stored chat transcript (the agent passes its whole history)."""
        with self._cond:
            self._messages = [_jsonable(dict(m)) for m in messages]
            self._save_locked()

    def update(self, **fields: Any) -> None:
        """Set top-level session facts (spend, final answer, ...)."""
        with self._cond:
            self._meta.update(_jsonable(fields))
            self._save_locked()

    def finish(self, status: str, **data: Any) -> dict[str, Any]:
        """Emit ``session.finished`` and mark the session done (``completed``/``stopped``/``failed``)."""
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"status must be one of {sorted(TERMINAL_STATUSES)}")
        with self._cond:  # one lock hold: no reader sees the session done before the event exists
            self.status = status
            self._meta["finished_at"] = _iso(self._clock())
            return self.emit("session.finished", status=status, **data)

    # -- reading

    @property
    def done(self) -> bool:
        with self._cond:
            return self.status in TERMINAL_STATUSES

    def events_after(self, seq: int) -> list[dict[str, Any]]:
        with self._cond:
            return [dict(e) for e in self._events[max(0, seq) :]]

    def wait_events(self, after: int, timeout_s: float) -> list[dict[str, Any]]:
        """Events with ``seq > after``, blocking up to ``timeout_s`` for one to arrive."""
        with self._cond:
            if len(self._events) <= after and self.status not in TERMINAL_STATUSES:
                self._cond.wait(timeout=max(0.0, timeout_s))
            return [dict(e) for e in self._events[max(0, after) :]]

    def to_json(self) -> dict[str, Any]:
        with self._cond:
            return self._doc()

    # -- persistence

    def _doc(self) -> dict[str, Any]:
        return {
            "format": SESSION_FORMAT,
            "version": SESSION_VERSION,
            "session_id": self.session_id,
            "mode": "product",
            "label": PRODUCT_MODE_LABEL,
            "status": self.status,
            "created_at": _iso(self._t0),
            **self._meta,
            "events": list(self._events),
            "messages": list(self._messages),
        }

    def _save(self) -> None:
        with self._cond:
            self._save_locked()

    def _save_locked(self) -> None:
        if self.path is not None:
            write_json_atomic(self.path, self._doc())


def _jsonable(value: Any) -> Any:
    """JSON-safe copy (non-finite floats become ``None``; unknown objects become strings)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def load_session(path: str | Path) -> dict[str, Any]:
    """Read a session file (``ValueError`` if it is not one)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != SESSION_FORMAT:
        raise ValueError(f"{path} is not an {SESSION_FORMAT} file")
    return data
