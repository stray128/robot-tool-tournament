"""Pinned-commit downloads of upstream files, with a local cache.

Every upstream file the library ships or derives from is fetched at a pinned commit from
``raw.githubusercontent.com`` (https only, at most 50 MB per file) and cached under a cache
directory keyed by repository and commit, so a re-run is offline and byte-identical. A
commit id must be the full 40-hex SHA: branch names are refused, because a branch moves.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Final

__all__ = ["MAX_BYTES", "Fetcher", "raw_url"]

MAX_BYTES: Final = 50 * 1024 * 1024
TIMEOUT_S: Final = 60
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")


def raw_url(repository: str, commit: str, path: str) -> str:
    """The raw download URL of ``path`` at ``commit`` (path components percent-encoded)."""
    m = _REPO.match(repository.rstrip("/"))
    if not m:
        raise ValueError(f"not a GitHub repository URL: {repository!r}")
    if not _COMMIT.match(commit):
        raise ValueError(f"pin a full 40-hex commit, not {commit!r}")
    quoted = "/".join(urllib.parse.quote(part) for part in path.split("/"))
    return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{commit}/{quoted}"


def blob_url(repository: str, commit: str, path: str) -> str:
    """The browsable URL of ``path`` at ``commit`` (for links on the catalog page)."""
    if not _COMMIT.match(commit):
        raise ValueError(f"pin a full 40-hex commit, not {commit!r}")
    quoted = "/".join(urllib.parse.quote(part) for part in path.split("/"))
    return f"{repository.rstrip('/')}/blob/{commit}/{quoted}"


class Fetcher:
    """Fetch upstream files at a pinned commit, caching them under ``cache_dir``."""

    def __init__(self, cache_dir: Path, *, offline: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self.downloaded = 0  # bytes fetched over the network in this run

    def _cached(self, repository: str, commit: str, path: str) -> Path:
        m = _REPO.match(repository.rstrip("/"))
        if not m:
            raise ValueError(f"not a GitHub repository URL: {repository!r}")
        safe = [p for p in path.split("/") if p not in ("", ".", "..")]
        return self.cache_dir / f"{m.group(1)}__{m.group(2)}" / commit / Path(*safe)

    def path(self, repository: str, commit: str, path: str) -> Path:
        """Local path of the cached file (downloading it first if needed)."""
        dest = self._cached(repository, commit, path)
        if dest.is_file():
            return dest
        if self.offline:
            raise FileNotFoundError(f"{repository}@{commit[:8]}:{path} is not cached (offline)")
        url = raw_url(repository, commit, path)
        req = urllib.request.Request(url, headers={"User-Agent": "rtt-vendor-library"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # https only (raw_url)
            data = resp.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"{url}: larger than {MAX_BYTES} bytes")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        self.downloaded += len(data)
        return dest

    def read(self, repository: str, commit: str, path: str) -> bytes:
        return self.path(repository, commit, path).read_bytes()

    def sha256(self, repository: str, commit: str, path: str) -> str:
        return hashlib.sha256(self.read(repository, commit, path)).hexdigest()
