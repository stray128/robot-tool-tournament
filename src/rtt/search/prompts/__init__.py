"""Prompt templates for the LLM arms, stored as files so they can be hashed into PREREG.

Templates use :class:`string.Template` placeholders (``$name``); rendering with a
missing placeholder is an error. :func:`template_hashes` gives the SHA-256 of every
template's bytes for manifests and the pre-registration.

The v1 templates sit at the top level (:data:`TEMPLATE_NAMES`); the ``freeform-v2`` ones,
including the hashed ``engineering_notes.md`` block shared by arms C and Z, sit under
``v2/`` (:data:`TEMPLATE_NAMES_V2`) and are hashed separately, so v1 hashes never change.
"""

from __future__ import annotations

import hashlib
from functools import cache
from importlib import resources
from string import Template
from typing import Final

__all__ = ["TEMPLATE_NAMES", "TEMPLATE_NAMES_V2", "load", "render", "template_hashes"]

TEMPLATE_NAMES: Final[tuple[str, ...]] = (
    "proposer_system.md",
    "proposer_user.md",
    "repair.md",
    "diversity.md",
    "zeroshot_system.md",
    "zeroshot_user.md",
)
TEMPLATE_NAMES_V2: Final[tuple[str, ...]] = (
    "v2/proposer_system.md",
    "v2/proposer_user.md",
    "v2/repair.md",
    "v2/diversity.md",
    "v2/zeroshot_system.md",
    "v2/zeroshot_user.md",
    "v2/engineering_notes.md",
)


@cache
def _read(name: str) -> str:
    if name not in TEMPLATE_NAMES and name not in TEMPLATE_NAMES_V2:
        raise KeyError(f"unknown prompt template {name!r}")
    return resources.files(__package__).joinpath(*name.split("/")).read_text(encoding="utf-8")


def load(name: str) -> Template:
    """The template ``name`` (e.g. ``"proposer_user.md"``)."""
    return Template(_read(name))


def render(name: str, **values: object) -> str:
    """Fill every placeholder of ``name`` (``KeyError`` if one is missing), trailing newline removed."""
    return load(name).substitute({k: str(v) for k, v in values.items()}).rstrip("\n")


def template_hashes(names: tuple[str, ...] = TEMPLATE_NAMES) -> dict[str, str]:
    """SHA-256 of each template file's text (the v1 set unless ``names`` says otherwise)."""
    return {name: hashlib.sha256(_read(name).encode("utf-8")).hexdigest() for name in names}
