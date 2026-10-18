"""The catalog page's data (product mode): ``web/public/data/library/index.json``."""

from __future__ import annotations

import json
import re

import pytest

from rtt.library.brands import find_library_brands
from rtt.library.preview import MAX_PREVIEW_BYTES
from rtt.library.records import REPO_ROOT, item_keys, read_item, read_lock
from rtt.library.web import INDEX_FORMAT, INDEX_PATH, LABEL, web_index

pytestmark = pytest.mark.skipif(not INDEX_PATH.is_file(), reason="run python -m rtt.library export-web")
_URL = re.compile(r"https?://\S+")


@pytest.fixture(scope="module")
def index() -> dict:
    return json.loads(INDEX_PATH.read_text())


def test_index_covers_every_item_with_the_label(index: dict) -> None:
    from rtt.effectors.types import PRODUCT_MODE_LABEL

    assert index["format"] == INDEX_FORMAT and index["label"] == LABEL == PRODUCT_MODE_LABEL
    assert [i["key"] for i in index["items"]] == list(item_keys())
    assert index["provenance"]["manifest_sha256"] == read_lock()["manifest_sha256"]
    assert index["counts"]["items"] == len(index["items"])
    assert index["counts"]["excluded_amber"] == 23 and index["counts"]["excluded_red"] == 26


def test_index_matches_the_records(index: dict) -> None:
    fresh = web_index(gate=False)
    for published, now in zip(index["items"], fresh["items"], strict=True):
        assert {k: v for k, v in published.items() if k != "gate"} == {
            k: v for k, v in now.items() if k != "gate"
        }


def test_every_entrant_passed_its_gate(index: dict) -> None:
    for item in index["items"]:
        if item["sim"]["entrant"]:
            assert item["gate"]["ran"] and item["gate"]["passed"], item["key"]
        else:
            assert item["sim"]["entrant_reason"], item["key"]


def test_previews_exist_fit_the_budget_and_match_their_records(index: dict) -> None:
    import hashlib

    for item in index["items"]:
        rec = read_item(item["key"]).preview
        assert item["preview"] and rec, item["key"]
        path = REPO_ROOT / "web" / "public" / item["preview"]["url"]
        data = path.read_bytes()
        assert len(data) <= MAX_PREVIEW_BYTES and data[:4] == b"glTF"
        assert hashlib.sha256(data).hexdigest() == rec["sha256"]


def test_published_library_data_names_no_brand() -> None:
    root = REPO_ROOT / "web" / "public" / "data" / "library"
    for path in root.rglob("*.json"):
        text = _URL.sub(" ", path.read_text())
        assert not find_library_brands(text), (path, sorted(set(find_library_brands(text))))


def test_downloads_point_at_shipped_files(index: dict) -> None:
    for item in index["items"]:
        for d in item["downloads"]:
            assert (REPO_ROOT / d["path"]).is_file(), d["path"]
        for link in item["links"]:
            assert link["url"].startswith("https://") and link["reason"]
