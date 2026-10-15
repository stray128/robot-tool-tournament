"""Prompt templates for the LLM arms, stored as files so they can be hashed into PREREG.

Templates use :class:`string.Template` placeholders (``$name``); rendering with a
missing placeholder is an error. :func:`template_hashes` gives the SHA-256 of every
template's bytes for manifests and the pre-registration.
"""

from __future__ import annotations

import hashlib
from functools import cache
from importlib import resources
from string import Template
from typing import Final

__all__ = ["TEMPLATE_NAMES", "load", "render", "template_hashes"]

TEMPLATE_NAMES: Final[tuple[str, ...]] = (
    "proposer_system.md",
    "proposer_user.md",
    "repair.md",
    "diversity.md",
    "zeroshot_system.md",
    "zeroshot_user.md",
)


@cache
def _read(name: str) -> str:
    if name not in TEMPLATE_NAMES:
        raise KeyError(f"unknown prompt template {name!r}")
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


def load(name: str) -> Template:
    """The template ``name`` (e.g. ``"proposer_user.md"``)."""
    return Template(_read(name))


def render(name: str, **values: object) -> str:
    """Fill every placeholder of ``name`` (``KeyError`` if one is missing), trailing newline removed."""
    return load(name).substitute({k: str(v) for k, v in values.items()}).rstrip("\n")


def template_hashes() -> dict[str, str]:
    """SHA-256 of each template file's text."""
    return {name: hashlib.sha256(_read(name).encode("utf-8")).hexdigest() for name in TEMPLATE_NAMES}
