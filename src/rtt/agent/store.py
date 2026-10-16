"""On-disk layout of the agent's outputs (default ``runs/agent/``, git-ignored).

::

    <root>/parts/<part_id>/            source.stl, part.json and backend caches
    <root>/sessions/<session_id>.json  session recordings (see rtt.agent.recorder)
    <root>/results/<run_id>.json       results of one finished job (GET /runs/{id})
    <root>/printpacks/<pack_id>/       print pack files, plus <pack_id>.zip (GET /printpack/{id})
    <root>/daily_spend.json            spend per UTC day (rtt.agent.caps.DailySpendBook)
    <root>/llm_ledger.jsonl            every model call of every session

IDs are validated against fixed patterns before they touch a path, so a request can
never reach outside these folders.
"""

from __future__ import annotations

import json
import re
import uuid
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from rtt.store.manifest import write_json_atomic

__all__ = ["AgentStore", "new_id"]

_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "part": re.compile(r"^part-[0-9a-f]{12}$"),
    "run": re.compile(r"^run-[0-9a-f]{12}$"),
    "pack": re.compile(r"^pack-[0-9a-f]{12}$"),
    "session": re.compile(r"^sess-[0-9a-f]{12}$"),
}
_PREFIX: Final = {"part": "part", "run": "run", "pack": "pack", "session": "sess"}


def new_id(kind: str) -> str:
    """A fresh ID of ``kind`` (``part``, ``run``, ``pack`` or ``session``)."""
    return f"{_PREFIX[kind]}-{uuid.uuid4().hex[:12]}"


def check_id(kind: str, value: str) -> str:
    """``value`` if it is a well-formed ``kind`` ID, else ``ValueError``."""
    if not isinstance(value, str) or not _PATTERNS[kind].match(value):
        raise ValueError(f"malformed {kind} id")
    return value


class AgentStore:
    """Paths and small JSON helpers under one root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"AgentStore({str(self.root)!r})"

    @property
    def parts_dir(self) -> Path:
        return self.root / "parts"

    @property
    def daily_spend_path(self) -> Path:
        return self.root / "daily_spend.json"

    @property
    def ledger_path(self) -> Path:
        return self.root / "llm_ledger.jsonl"

    def part_dir(self, part_id: str) -> Path:
        return self.parts_dir / check_id("part", part_id)

    def session_path(self, session_id: str) -> Path:
        return self.root / "sessions" / f"{check_id('session', session_id)}.json"

    def run_path(self, run_id: str) -> Path:
        return self.root / "results" / f"{check_id('run', run_id)}.json"

    def pack_dir(self, pack_id: str) -> Path:
        return self.root / "printpacks" / check_id("pack", pack_id)

    def pack_zip(self, pack_id: str) -> Path:
        return self.root / "printpacks" / f"{check_id('pack', pack_id)}.zip"

    # -- runs

    def save_run(self, run_id: str, data: dict[str, Any]) -> Path:
        return write_json_atomic(self.run_path(run_id), data)

    def load_run(self, run_id: str) -> dict[str, Any]:
        path = self.run_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    # -- print packs

    def zip_pack(self, pack_id: str, files: Sequence[Path]) -> Path:
        """Zip ``files`` (paths inside the pack folder) into ``<pack_id>.zip``; returns it."""
        base = self.pack_dir(pack_id)
        target = self.pack_zip(pack_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(set(files)):
                path = Path(path)
                try:
                    arcname = path.resolve().relative_to(base.resolve()).as_posix()
                except ValueError:
                    arcname = path.name
                zf.write(path, f"{pack_id}/{arcname}")
        tmp.replace(target)
        return target
