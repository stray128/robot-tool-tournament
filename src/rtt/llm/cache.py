"""Content-addressed cache of LLM requests and raw responses.

Every request is serialised as compact JSON *in the order it is sent* and
hashed with SHA-256. Key order is deliberately significant: the server renders
tool and response schemas into the prompt in the order given, and constrained
decoding emits properties in schema order, so two payloads that differ only in
property order are different requests (for example "analysis before designs"
versus the reverse). The hash names a JSON file holding the request exactly as
sent, the raw provider response and the original call facts (latency and the
rate-limit headers; usage is in the response). Because the *raw* response is
stored, parsing is re-run on every read, so a replay reproduces parse failures
exactly as they happened. Reads re-hash the stored request, so an edited or
mismatched cache file is an error rather than a silent wrong answer.

Modes:

* ``readwrite``: return hits, call the provider on a miss and store the result.
* ``replay``: return hits, raise :class:`CacheMiss` on a miss. This is what
  ``make replay-llm`` uses to re-run a search arm with zero API calls.
* ``refresh``: always call the provider and overwrite the entry.
* ``off``: neither read nor write.

The cache never contains credentials: requests are the JSON payload sent to the
API (model, messages, tools or schema, sampling parameters), not HTTP headers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CacheMode = Literal["readwrite", "replay", "refresh", "off"]
CACHE_FORMAT_VERSION = 2


class CacheMiss(LookupError):
    """Raised in ``replay`` mode when a request has no stored response."""

    def __init__(self, request_hash: str) -> None:
        super().__init__(f"no cached response for request {request_hash} (replay mode)")
        self.request_hash = request_hash


def canonical_json(obj: Any) -> str:
    """Order-insensitive JSON text: sorted keys, compact separators, UTF-8 kept.

    Suitable for hashing data whose key order carries no meaning. Request
    hashing uses :func:`wire_json` instead, because key order inside schemas
    changes what the model sees.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def wire_json(obj: Any) -> str:
    """Compact JSON text that keeps key order (what goes over the wire)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def request_hash(request: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of ``request`` serialised with :func:`wire_json`.

    Deterministic for payloads built by the same code (Python dicts keep
    insertion order), and sensitive to schema property order on purpose.
    """
    return hashlib.sha256(wire_json(request).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """One stored exchange."""

    request_hash: str
    request: dict[str, Any]
    response: dict[str, Any]
    latency_s: float
    created_at: str
    headers: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "format": CACHE_FORMAT_VERSION,
            "request_hash": self.request_hash,
            "request": self.request,
            "response": self.response,
            "latency_s": self.latency_s,
            "created_at": self.created_at,
            "headers": self.headers,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> CacheEntry:
        return cls(
            request_hash=str(data["request_hash"]),
            request=dict(data["request"]),
            response=dict(data["response"]),
            latency_s=float(data.get("latency_s", 0.0)),
            created_at=str(data.get("created_at", "")),
            headers={str(k): str(v) for k, v in dict(data.get("headers") or {}).items()},
        )


class ResponseCache:
    """Directory-backed cache, sharded as ``<root>/<hash[:2]>/<hash>.json``."""

    def __init__(self, root: str | Path, mode: CacheMode = "readwrite") -> None:
        if mode not in ("readwrite", "replay", "refresh", "off"):
            raise ValueError(f"unknown cache mode {mode!r}")
        self.root = Path(root)
        self.mode: CacheMode = mode

    def __repr__(self) -> str:
        return f"ResponseCache(root={str(self.root)!r}, mode={self.mode!r})"

    @property
    def reads(self) -> bool:
        return self.mode in ("readwrite", "replay")

    @property
    def writes(self) -> bool:
        return self.mode in ("readwrite", "refresh")

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        """Stored entry for ``key``; ``None`` on a miss (or when reads are off).

        In ``replay`` mode a miss raises :class:`CacheMiss` instead.
        """
        if not self.reads:
            return None
        path = self.path_for(key)
        if not path.is_file():
            if self.mode == "replay":
                raise CacheMiss(key)
            return None
        entry = CacheEntry.from_json(json.loads(path.read_text(encoding="utf-8")))
        if entry.request_hash != key or request_hash(entry.request) != key:
            raise ValueError(f"cache file {path} does not hold the request with hash {key}")
        return entry

    def put(self, entry: CacheEntry) -> Path | None:
        """Atomically write ``entry``; returns its path (``None`` if writes are off)."""
        if not self.writes:
            return None
        path = self.path_for(entry.request_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                # No sort_keys: the stored request must keep the order it was sent in.
                json.dump(entry.to_json(), fh, ensure_ascii=False, indent=1)
                fh.write("\n")
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.path_for(key).is_file()

    def __len__(self) -> int:
        if not self.root.is_dir():
            return 0
        return sum(1 for _ in self.root.glob("??/*.json"))
