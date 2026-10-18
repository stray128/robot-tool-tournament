"""Makers' names stay off screen (product mode): the library's brand list over the whole repository.

The effector contract's scan (``tests/test_effectors_contract.py``) checks its frozen
``BRAND_TOKENS``. The library adds makers and products to that list
(:data:`rtt.library.brands.LIBRARY_ONLY_TOKENS`), two of them newly allowed in credits by the
founder (2026-10-18). This test applies the union with the same rule: names may appear only in
credit files (README, NOTICE, ``docs/ATTRIBUTION.md``, the credits page), provenance under
``assets/`` (``SOURCE.json``, ``lock.json``, upstream licence texts), the token lists themselves
and tests. Source URLs are ignored: a pinned URL is provenance, not an on-screen credit.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from rtt.effectors.types import BRAND_TOKENS
from rtt.library.brands import (
    LIBRARY_BRAND_TOKENS,
    LIBRARY_ONLY_TOKENS,
    check_library_name,
    find_library_brands,
)

REPO = Path(__file__).resolve().parents[1]
ALLOWED_FILES = {
    "README.md",
    "NOTICE",
    "docs/ATTRIBUTION.md",
    "docs/DEVIATIONS.md",
    "docs/EFFECTORS.md",
    "web/app/credits/page.tsx",
    "web/scripts/smoke_v2.mjs",
    "src/rtt/effectors/types.py",
    "src/rtt/library/brands.py",
}
ALLOWED_PREFIXES = ("tests/", "assets/gripper/", "assets/effectors/")
_URL = re.compile(r"https?://[^\s\"')>\]]+")


def test_the_library_list_extends_the_contract_list() -> None:
    assert set(BRAND_TOKENS) <= set(LIBRARY_BRAND_TOKENS)
    assert {"robotiq", "franka"} <= set(LIBRARY_ONLY_TOKENS)
    assert find_library_brands("a ROBOTIQ gripper and a Franka hand") == ["robotiq", "franka"]
    assert find_library_brands("frankly, a robotic gripper") == []  # whole words only
    with pytest.raises(ValueError, match="brand"):
        check_library_name("xarm7 gripper")
    assert check_library_name("Two-finger adaptive linkage gripper, 85 mm")


def _listed() -> list[str]:
    if not (REPO / ".git").exists():
        pytest.skip("needs the git checkout")
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return out


def test_library_brand_tokens_appear_only_in_credit_files() -> None:
    hits = {}
    for rel in _listed():
        if rel in ALLOWED_FILES or rel.startswith(ALLOWED_PREFIXES):
            continue
        path = REPO / rel
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found = find_library_brands(_URL.sub(" ", text))
        if found:
            hits[rel] = sorted(set(found))
    assert hits == {}


def test_rendered_pages_name_no_maker_except_credits() -> None:
    out = REPO / "web" / "out"
    if not out.is_dir():
        pytest.skip("no static export (run pnpm build in web/)")
    hits = {}
    for page in out.rglob("*.html"):
        rel = page.relative_to(out).as_posix()
        if rel.startswith("credits"):
            continue
        text = re.sub(r"<[^>]+>", " ", page.read_text(encoding="utf-8"))
        found = find_library_brands(_URL.sub(" ", text))
        if found:
            hits[rel] = sorted(set(found))
    assert hits == {}
