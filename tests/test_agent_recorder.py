"""The session recorder's finish is atomic for readers (the SSE stream relies on it)."""

from __future__ import annotations

import threading
from typing import Any

from rtt.agent.recorder import SessionRecorder


def test_a_session_is_never_done_before_its_finished_event() -> None:
    rec = SessionRecorder("sess-000000000001", None)
    seen: list[tuple[bool, bool]] = []
    threads: list[threading.Thread] = []

    def observe() -> None:
        done = rec.done
        seen.append((done, any(e["type"] == "session.finished" for e in rec.events_after(0))))

    emit = rec.emit

    def emit_with_observer(type_: str, **data: Any) -> dict[str, Any]:
        if type_ == "session.finished":  # a reader arriving exactly as finish emits
            t = threading.Thread(target=observe)
            threads.append(t)
            t.start()
            t.join(timeout=0.2)  # with the lock held by finish, the reader waits
        return emit(type_, **data)

    rec.emit = emit_with_observer  # type: ignore[method-assign]
    rec.finish("completed")
    for t in threads:
        t.join(timeout=5.0)
    assert seen, "the observer never ran"
    (done, has_finished) = seen[0]
    assert not done or has_finished  # the stream's "done and nothing new" check cannot end early
