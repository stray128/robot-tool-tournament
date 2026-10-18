"""The open effector library on disk (product mode): every item re-hashes to its records,
the licence gate passes, the lock pins everything, and every item's status is explicit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rtt.library.brands import find_library_brands
from rtt.library.licence import Rating, classify
from rtt.library.mounts import ISO_PATTERNS
from rtt.library.records import (
    LIBRARY_DIR,
    LOCK_PATH,
    REPO_ROOT,
    build_lock,
    item_keys,
    manifest_sha256,
    read_item,
    read_lock,
    verify_item,
    verify_library,
)
from rtt.library.taxonomy import CAPABILITIES, SIM_STATUSES

KEYS = item_keys()
#: Founder brief 2026-10-18: keep the repository's library under about 60 MB.
SIZE_BUDGET_BYTES = 60 * 1024 * 1024


def test_the_library_has_items() -> None:
    assert len(KEYS) >= 25


@pytest.mark.parametrize("key", KEYS)
def test_every_item_rehashes_and_passes_the_gate(key: str) -> None:
    assert verify_item(key) == []


def test_lock_pins_every_item_and_its_licence_files() -> None:
    lock = read_lock()
    assert lock == build_lock()
    assert lock["manifest_sha256"] == manifest_sha256(lock["items"], lock["catalog_sha256"])
    for key, entry in lock["items"].items():
        assert entry["licence_files"], key  # a licence text pinned by hash for every item
    assert verify_library() == []


def test_vendor_check_command_passes() -> None:
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "vendor_library.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "lock.json: ok" in res.stdout


def test_a_changed_file_is_caught(tmp_path: Path) -> None:
    key = KEYS[0]
    item = read_item(key)
    target = REPO_ROOT / item.files[0].path
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n")
        read_item.cache_clear()
        assert any("sha256 differs" in p for p in verify_item(key))
    finally:
        target.write_bytes(original)
        read_item.cache_clear()
    assert verify_item(key) == []


@pytest.mark.parametrize("key", KEYS)
def test_status_tags_mount_and_specs_are_explicit(key: str) -> None:
    item = read_item(key)
    assert item.sim_status in SIM_STATUSES
    if item.sim_status in ("visual_only", "none"):
        assert item.sim.get("reason")
    assert set(item.capabilities) <= set(CAPABILITIES)
    assert item.mount.interface
    if item.mount.iso_pattern:
        assert item.mount.iso_pattern in ISO_PATTERNS
    for s in item.specs:
        assert s.tag in ("sourced", "derived", "assumed")
        if s.tag == "sourced":
            assert s.source.startswith("https://")


@pytest.mark.parametrize("key", KEYS)
def test_every_licence_is_green_and_kept_byte_for_byte(key: str) -> None:
    item = read_item(key)
    assert item.rating is Rating.GREEN
    for f in item.files:
        assert classify(f.license) is Rating.GREEN, f.path
        if f.role == "licence":
            assert f.sha256 == f.upstream_sha256, f"{f.path}: licence texts ship unmodified"
        if f.role in ("print", "cad", "doc") and f.upstream_commit != "generated":
            assert f.sha256 == f.upstream_sha256 and not f.changes, f"{f.path}: print files ship unmodified"


@pytest.mark.parametrize("key", KEYS)
def test_screen_strings_are_brand_free(key: str) -> None:
    item = read_item(key)
    for text in (key, item.display_name, item.summary, item.mount.interface, *item.notes):
        assert not find_library_brands(text), (key, text)
    for f in item.files:
        assert not find_library_brands(Path(f.path).name.replace("_", " ")), (
            f.path
        )  # shipped names are generic


def test_no_amber_or_red_item_is_ingested_and_exclusions_carry_reasons() -> None:
    catalog = json.loads((LIBRARY_DIR / "catalog.json").read_text())
    excluded = catalog["excluded"]
    assert sum(e["rating"] == "amber" for e in excluded) == 23
    assert sum(e["rating"] == "red" for e in excluded) == 26
    for e in excluded:
        assert e["reason"] and e["name"] and not find_library_brands(e["name"]), e
        assert classify(e["license"]) is not Rating.GREEN or e["rating"] == "amber", e
    for e in catalog["listed"]:
        assert e["reason"] and classify(e["license"]) is Rating.GREEN, e
    for key in KEYS:
        assert read_item(key).rating is Rating.GREEN


def test_repository_size_stays_in_budget() -> None:
    total = sum(p.stat().st_size for p in LIBRARY_DIR.rglob("*") if p.is_file())
    previews = REPO_ROOT / "web" / "public" / "data" / "library"
    total += sum(p.stat().st_size for p in previews.rglob("*") if p.is_file())
    assert total < SIZE_BUDGET_BYTES
    assert LOCK_PATH.is_file()
