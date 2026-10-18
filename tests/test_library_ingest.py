"""Vendoring the library (product mode): licence drift stops a re-vendor, pins are full commits,
and rigid proxies and previews are deterministic. No network: upstream bytes come from the
shipped licence texts, which are byte-identical to upstream."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import trimesh

from rtt.library.fetch import Fetcher, raw_url
from rtt.library.ingest import LicenceDrift, vendor_item
from rtt.library.preview import cluster_decimate, scene_glb
from rtt.library.proxy import build_rigid_proxy
from rtt.library.records import REPO_ROOT, licence_key, read_item, read_lock

KEY = "gripper-2f-force-48"


class ShippedLicences(Fetcher):
    """Serves an item's upstream licence texts from its shipped copies (they are byte-identical)."""

    def __init__(self, key: str) -> None:
        super().__init__(Path("/nonexistent"), offline=True)
        self.files = {
            (f.upstream_url, f.upstream_path): (REPO_ROOT / f.path).read_bytes()
            for f in read_item(key).files
            if f.role == "licence"
        }

    def read(self, repository: str, commit: str, path: str) -> bytes:
        return self.files[(repository, path)]


def test_a_changed_licence_text_stops_the_vendoring() -> None:
    item = read_item(KEY)
    lock = copy.deepcopy(read_lock())
    lf = next(f for f in item.files if f.role == "licence")
    lock["items"][KEY]["licence_files"][licence_key(item, lf.upstream_url, lf.upstream_path)] = "0" * 64
    before = (item.directory / "SOURCE.json").read_bytes()
    with pytest.raises(LicenceDrift, match="review it before vendoring"):
        vendor_item(KEY, ShippedLicences(KEY), lock=lock)
    assert (item.directory / "SOURCE.json").read_bytes() == before  # nothing recorded


def test_pinned_licence_hashes_match_the_shipped_texts() -> None:
    lock = read_lock()
    for key, entry in lock["items"].items():
        item = read_item(key)
        for f in item.files:
            if f.role == "licence":
                k = licence_key(item, f.upstream_url, f.upstream_path)
                assert (
                    entry["licence_files"][k] == hashlib.sha256((REPO_ROOT / f.path).read_bytes()).hexdigest()
                )


def test_downloads_need_a_full_commit_and_https() -> None:
    assert raw_url("https://github.com/a/b", "a" * 40, "x y/z.stl") == (
        "https://raw.githubusercontent.com/a/b/" + "a" * 40 + "/x%20y/z.stl"
    )
    with pytest.raises(ValueError, match="40-hex"):
        raw_url("https://github.com/a/b", "main", "x")
    with pytest.raises(ValueError, match="GitHub"):
        raw_url("http://example.org/a/b", "a" * 40, "x")


def test_rigid_proxy_and_preview_are_deterministic() -> None:
    mesh = trimesh.creation.box(extents=(0.02, 0.03, 0.06))
    a = build_rigid_proxy({"tool": mesh}, down=(0, 0, -1), forward=(0, 1, 0), material="PLA", infill=0.5)
    b = build_rigid_proxy({"tool": mesh}, down=(0, 0, -1), forward=(0, 1, 0), material="PLA", infill=0.5)
    assert a.mass == pytest.approx(0.02 * 0.03 * 0.06 * 1240 * 0.5)
    assert a.reach == pytest.approx(0.06) and float(a.parts[0].mesh.vertices[:, 2].max()) == pytest.approx(
        0.0
    )
    assert [p.vertices.tolist() for p in a.parts[0].pieces] == [
        p.vertices.tolist() for p in b.parts[0].pieces
    ]
    sphere = trimesh.creation.icosphere(subdivisions=5)
    small = cluster_decimate(sphere, 500)
    assert len(small.faces) <= 500 and np.isfinite(small.vertices).all()
    assert scene_glb([sphere], max_faces=800) == scene_glb([sphere], max_faces=800)
