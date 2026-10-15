"""Held-out seed vault: commit-reveal storage plus a guard that keeps search code away from it.

The held-out suite is generated from a secret seed drawn at pre-registration. Only
its SHA-256 is committed; the seed itself lives in a gitignored file
(``secrets/heldout_seed.txt`` locally, injected at ``/secret/heldout_seed`` into the
held-out job only). Two rules keep "held-out" checkable rather than asserted:

* **Explicit path only.** :class:`HeldoutVault` reads the seed from the path it is
  given and nowhere else: no default location, no environment variable, no search
  of the working tree. Only the ``heldout`` command constructs one. A path inside
  this repository is accepted only when git ignores it, so the seed can never be
  committed by accident (the same rule as ``rtt.sim.conditions.read_secret_seed``).
* **Search runs are fenced off.** :class:`VaultGuard` installs (once per process) a
  :func:`sys.addaudithook` hook that, while any guard is active, refuses to open or
  list anything under the protected roots or any file whose name starts with
  ``heldout_seed``. :func:`rtt.search.runner.run_search` runs inside a guard, so no
  search code path (including its LLM worker threads) can read the seed even by
  accident; the refusal raises :class:`VaultAccessError`.

Audit hooks cannot be removed, so the hook is a no-op whenever no guard is active.
The guard covers the search process and all its threads. Simulator worker processes
(spawned by the CPU executor) start fresh interpreters without the hook; they only
receive episode jobs and never open files.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets as _secrets
import subprocess
import sys
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

__all__ = [
    "DEFAULT_PROTECTED_ROOTS",
    "HELDOUT_SEED_PREFIX",
    "HeldoutVault",
    "VaultAccessError",
    "VaultGuard",
    "guard_active",
    "seed_sha256",
]

#: Any file whose name starts with this is treated as held-out material by the guard.
HELDOUT_SEED_PREFIX = "heldout_seed"

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Directories the guard protects by default: the repo's gitignored ``secrets/`` folder,
#: ``secrets/`` under the working directory, and the job's injected-secret mount point.
DEFAULT_PROTECTED_ROOTS: tuple[Path, ...] = (
    _REPO_ROOT / "secrets",
    Path("secrets"),
    Path("/secret"),
)

_SEED_RE = re.compile(r"^[0-9a-f]{64}$")
_LIST_EVENTS = frozenset({"os.listdir", "os.scandir", "glob.glob"})
_PATH_EVENTS = frozenset({"open", "shutil.copyfile", "shutil.copytree", "os.rename"}) | _LIST_EVENTS


class VaultAccessError(PermissionError):
    """Something tried to read held-out material while a :class:`VaultGuard` was active."""


def seed_sha256(seed: str) -> str:
    """The commitment for ``seed``: SHA-256 of its UTF-8 text (stripped of whitespace)."""
    return hashlib.sha256(seed.strip().encode("utf-8")).hexdigest()


class HeldoutVault:
    """The held-out seed file at an explicit path.

    Args:
        path: Location of the seed file. There is deliberately no default.
    """

    def __init__(self, path: str | Path) -> None:
        if path is None or str(path) == "":
            raise ValueError("HeldoutVault needs an explicit seed-file path")
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"HeldoutVault(path={str(self.path)!r})"

    @classmethod
    def create(cls, path: str | Path, seed: str | None = None) -> HeldoutVault:
        """Write a new seed file (``secrets.token_hex(32)`` unless ``seed`` is given), mode 600.

        Refuses to overwrite an existing file: the commitment must never change.
        """
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"{target} already exists; a committed seed is never replaced")
        _refuse_tracked_location(target)
        value = seed.strip() if seed is not None else _secrets.token_hex(32)
        _check_seed(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(value + "\n")
        return cls(target)

    def read_seed(self) -> str:
        """The seed (64 lowercase hex digits). Raises :class:`VaultAccessError` inside a guard."""
        if guard_active():
            raise VaultAccessError(f"held-out seed read refused: a search guard is active ({self.path})")
        _refuse_tracked_location(self.path)
        value = self.path.read_text(encoding="utf-8").strip()
        _check_seed(value)
        return value

    def sha256(self) -> str:
        """The commitment of the stored seed."""
        return seed_sha256(self.read_seed())

    def verify(self, expected_sha256: str) -> str:
        """Return the seed after checking it against the committed hash."""
        seed = self.read_seed()
        actual = seed_sha256(seed)
        if actual != expected_sha256.strip().lower():
            raise ValueError(f"held-out seed does not match its commitment ({actual[:12]} != expected)")
        return seed


def _check_seed(value: str) -> None:
    if not _SEED_RE.match(value):
        raise ValueError("held-out seed must be 64 lowercase hex digits (secrets.token_hex(32))")


def _git_ignores(path: Path) -> bool:
    """Whether git ignores ``path`` in this repository (a tracked file never counts as ignored)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "check-ignore", "-q", str(path)],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _refuse_tracked_location(path: Path) -> None:
    """Refuse a seed file inside the repository unless git ignores it."""
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(_REPO_ROOT):
        return
    if (_REPO_ROOT / ".git").exists() and _git_ignores(resolved):
        return
    raise PermissionError(
        f"{resolved} is inside the repository and not git-ignored; keep the held-out seed outside the "
        "repository or in a git-ignored folder"
    )


# --------------------------------------------------------------------------- guard

_LOCK = threading.Lock()
_ACTIVE: list[tuple[str, ...]] = []  # stack of active guards' protected roots (normalised)
_HOOK_INSTALLED = False


def guard_active() -> bool:
    """True while at least one :class:`VaultGuard` is active in this process."""
    return bool(_ACTIVE)


def _normalise(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _is_protected(raw: Any, roots: Iterable[str]) -> bool:
    if isinstance(raw, int) or raw is None:
        return False  # file descriptors were opened earlier, under the same rules
    try:
        text = os.fsdecode(raw)
    except TypeError:
        return False
    if os.path.basename(text.rstrip("/\\")).startswith(HELDOUT_SEED_PREFIX):
        return True
    resolved = _normalise(text)
    if os.path.basename(resolved).startswith(HELDOUT_SEED_PREFIX):
        return True
    return any(resolved == root or resolved.startswith(root + os.sep) for root in roots)


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _ACTIVE or event not in _PATH_EVENTS:
        return
    roots = [root for guard in _ACTIVE for root in guard]
    candidates = args[:2] if event in ("shutil.copyfile", "shutil.copytree", "os.rename") else args[:1]
    if event == "glob.glob" and args:
        candidates = (os.path.dirname(os.fsdecode(args[0])) or ".", args[0])
    for raw in candidates:
        if _is_protected(raw, roots):
            raise VaultAccessError(f"held-out vault access refused during a search run ({event})")


def _install_hook() -> None:
    global _HOOK_INSTALLED
    with _LOCK:
        if not _HOOK_INSTALLED:
            sys.addaudithook(_audit_hook)
            _HOOK_INSTALLED = True


class VaultGuard:
    """Context manager that forbids reading held-out material while it is active.

    Args:
        extra_roots: Further directories (or files) to protect, e.g. a test's
            temporary ``secrets`` folder or a bucket mount's held-out prefix.
        include_defaults: Also protect :data:`DEFAULT_PROTECTED_ROOTS`.
    """

    def __init__(self, extra_roots: Iterable[str | Path] = (), *, include_defaults: bool = True) -> None:
        roots = [*(DEFAULT_PROTECTED_ROOTS if include_defaults else ()), *map(Path, extra_roots)]
        self.roots: tuple[str, ...] = tuple(dict.fromkeys(_normalise(r) for r in roots))
        self._entered = False

    def __enter__(self) -> Self:
        _install_hook()
        with _LOCK:
            _ACTIVE.append(self.roots)
        self._entered = True
        return self

    def __exit__(self, *exc: object) -> None:
        if self._entered:
            with _LOCK:
                _ACTIVE.remove(self.roots)
            self._entered = False

    def protects(self, path: str | Path) -> bool:
        """Whether this guard would refuse ``path``."""
        return _is_protected(os.fspath(path), self.roots)


@contextmanager
def guarded(extra_roots: Iterable[str | Path] = ()) -> Iterator[VaultGuard]:
    """Functional form of :class:`VaultGuard`."""
    with VaultGuard(extra_roots) as guard:
        yield guard
