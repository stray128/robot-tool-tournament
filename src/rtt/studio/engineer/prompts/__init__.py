"""Prompt templates of the Task Engineer (``$name`` placeholders, :class:`string.Template`), by version.

* ``v1/``: the prompt of the v1 recording (``recorded/gate-2026-10-18-s4e``), frozen: its three
  templates and, in ``frozen.json``, the catalog sections and the forced tool exactly as the v1
  code rendered them, so the v1 recording's requests are rebuilt byte for byte and replay from its
  cache (:func:`frozen`).
* ``v2/``: Studio v2's first prompt (the job templates, the wrist's pitch, the tools the search
  tries and rule set v2's first revision), frozen the same way: four of the v2 heroes of
  ``recorded/gate-2026-10-27-v2c`` answered it.
* ``v2.1/``: v2 with revision v2-r2's rules (a heavy part's lift leaves the tool's give under its
  load; a hold counts only above the hold threshold), frozen; a re-run of h4 and h5 answered it.
* ``v2.2/``: v2.1 with revision v2-r3's rules (a hook pull's force limit clears a sliding door's
  pull; the finger room around a part picked inside a machine), frozen; the current h4 and h5
  answered it.
* ``v2.3/``: the current prompt, rendered live from the contract
  (:mod:`rtt.studio.engineer.catalog`): v2.2 whose repair message says that problems name the field
  to change, as revision v2-r4's repairs do (:data:`FIELD_PATHS`).

A prompt version goes with a rule set (:data:`RULES_OF`, :mod:`rtt.studio.engineer.checks`): v1
with the frozen v1 rules, every v2 prompt with rule set v2. :func:`template_hashes` gives each
template's SHA-256 for the plan records and the recorded gate manifests (a changed template is a
different request, so a recorded replay names them).
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from importlib import resources
from string import Template
from typing import Any, Final, Literal

__all__ = [
    "CURRENT",
    "CURRENT_RULES",
    "FIELD_PATHS",
    "FROZEN",
    "RULES_OF",
    "TEMPLATE_NAMES",
    "VERSIONS",
    "PromptVersion",
    "field_paths",
    "frozen",
    "render",
    "rules_of",
    "template_hashes",
]

TEMPLATE_NAMES: Final[tuple[str, ...]] = ("system.md", "user.md", "repair.md")
PromptVersion = Literal["v1", "v2", "v2.1", "v2.2", "v2.3"]
VERSIONS: Final[tuple[PromptVersion, ...]] = ("v1", "v2", "v2.1", "v2.2", "v2.3")
#: The prompts whose catalog sections and tool are frozen in their ``frozen.json``.
FROZEN: Final[tuple[PromptVersion, ...]] = ("v1", "v2", "v2.1", "v2.2")
#: The prompt new plans use.
CURRENT: Final[PromptVersion] = "v2.3"
#: The rule set each prompt goes with.
RULES_OF: Final[dict[str, str]] = {"v1": "v1", "v2": "v2", "v2.1": "v2", "v2.2": "v2", "v2.3": "v2"}
#: The rule set new plans are judged by.
CURRENT_RULES: Final[str] = RULES_OF[CURRENT]
#: The prompts whose repairs name the field to change in every rule set v2 message (a recorded
#: prompt keeps the words its recorded repairs answered, so they replay request for request).
FIELD_PATHS: Final[frozenset[str]] = frozenset({"v2.3"})


def _check(version: str) -> None:
    if version not in VERSIONS:
        raise KeyError(f"unknown prompt version {version!r} (known: {', '.join(VERSIONS)})")


def rules_of(version: str) -> str:
    """The rule set prompt ``version`` goes with."""
    _check(version)
    return RULES_OF[version]


def field_paths(version: str) -> bool:
    """Whether the repairs of prompt ``version`` name the field to change (:data:`FIELD_PATHS`)."""
    _check(version)
    return version in FIELD_PATHS


@cache
def _read(name: str, version: str = CURRENT) -> str:
    _check(version)
    if name not in TEMPLATE_NAMES:
        raise KeyError(f"unknown prompt template {name!r}")
    return resources.files(__package__).joinpath(version, name).read_text(encoding="utf-8")


@cache
def frozen(version: str) -> dict[str, Any]:
    """A frozen prompt's rendered catalog sections and its tool (``<version>/frozen.json``)."""
    if version not in FROZEN:
        raise KeyError(f"prompt {version!r} is rendered live, not frozen (frozen: {', '.join(FROZEN)})")
    return json.loads(
        resources.files(__package__).joinpath(version, "frozen.json").read_text(encoding="utf-8")
    )


def render(name: str, version: str = CURRENT, **values: object) -> str:
    """Fill every placeholder of ``name`` (``KeyError`` if one is missing), trailing newline removed."""
    return Template(_read(name, version)).substitute({k: str(v) for k, v in values.items()}).rstrip("\n")


def template_hashes(version: str = CURRENT) -> dict[str, str]:
    return {name: hashlib.sha256(_read(name, version).encode("utf-8")).hexdigest() for name in TEMPLATE_NAMES}
