"""The library's brand guard: makers' and products' names stay off screen.

:data:`LIBRARY_BRAND_TOKENS` is the effector contract's :data:`~rtt.effectors.types.BRAND_TOKENS`
plus the makers and products the library adds. The contract list is frozen (its file is
hash-pinned), so the library keeps the extension here; :func:`find_library_brands` checks
against the union, with the same whole-word, any-case rule.

Two of the makers added here were on the project's private leak list until the founder
included their simulation models in the library (2026-10-18). Their names may now appear in
credits only: each item's ``SOURCE.json``, ``lock.json``, README, NOTICE,
``docs/ATTRIBUTION.md`` and the credits page. ``tests/test_library_brands.py`` scans every
other file of the repository (and the published library data) with the union.
"""

from __future__ import annotations

import re
from typing import Final

from rtt.effectors.types import BRAND_TOKENS

__all__ = ["LIBRARY_BRAND_TOKENS", "LIBRARY_ONLY_TOKENS", "check_library_name", "find_library_brands"]

#: Makers and products the library adds to the contract's list (whole words, any case).
LIBRARY_ONLY_TOKENS: Final[tuple[str, ...]] = (
    "robotiq",
    "franka",
    "fr3",
    "source robotics",
    "ssg-48",
    "ssg48",
    "parol6",
    "robonine",
    "robo9",
    "orca dexterity",
    "orca hand",
    "orcahand",
    "midas hand",
    "romela",
    "pollen robotics",
    "amazinghand",
    "amazing hand",
    "reachy",
    "hello robot",
    "hello-robot",
    "toyota research institute",
    "wsg 50",
    "wsg-50",
    "wsg50",
    "wsg 050",
    "ufactory",
    "xarm",
    "lite 6",
    "lite6",
    "widowx",
    "dynamixel",
    "robotis",
    "rh-p12",
    "feetech",
    "sts3215",
    "scs0009",
    "koch",
    "moveo",
    "bcn3d",
    "xlerobot",
    "lerobot",
    "assembler0",
    "grip4so101",
    "norma-core",
    "realsense",
)

LIBRARY_BRAND_TOKENS: Final[tuple[str, ...]] = tuple(dict.fromkeys((*BRAND_TOKENS, *LIBRARY_ONLY_TOKENS)))

_RE = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(re.escape(t) for t in sorted(LIBRARY_BRAND_TOKENS, key=len, reverse=True))
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)


def find_library_brands(text: str) -> list[str]:
    """Every brand token of the library's list in ``text`` (lower case, in order)."""
    return [m.group(1).lower() for m in _RE.finditer(text)]


def check_library_name(name: str, *, limit: int = 80) -> str:
    """``name`` if it can go on screen: non-empty, at most ``limit`` characters, brand-free."""
    if not isinstance(name, str) or not name.strip() or len(name) > limit:
        raise ValueError(f"a screen name must be a non-empty string of at most {limit} characters")
    spaced = name.replace("_", " ").replace("-", " ")
    split = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", spaced)  # "xarm7" -> "xarm 7"
    hits = find_library_brands(name) + find_library_brands(spaced) + find_library_brands(split)
    if hits:
        raise ValueError(f"screen name {name!r} names a brand ({', '.join(sorted(set(hits)))})")
    return name
