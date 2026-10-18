"""Prompt templates of the Task Engineer (``$name`` placeholders, :class:`string.Template`).

:func:`template_hashes` gives each template's SHA-256 for the plan records and the recorded
gate manifest (a changed template is a different request, so a recorded replay names them).
"""

from __future__ import annotations

import hashlib
from functools import cache
from importlib import resources
from string import Template
from typing import Final

__all__ = ["TEMPLATE_NAMES", "render", "template_hashes"]

TEMPLATE_NAMES: Final[tuple[str, ...]] = ("system.md", "user.md", "repair.md")


@cache
def _read(name: str) -> str:
    if name not in TEMPLATE_NAMES:
        raise KeyError(f"unknown prompt template {name!r}")
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


def render(name: str, **values: object) -> str:
    """Fill every placeholder of ``name`` (``KeyError`` if one is missing), trailing newline removed."""
    return Template(_read(name)).substitute({k: str(v) for k, v in values.items()}).rstrip("\n")


def template_hashes() -> dict[str, str]:
    return {name: hashlib.sha256(_read(name).encode("utf-8")).hexdigest() for name in TEMPLATE_NAMES}
