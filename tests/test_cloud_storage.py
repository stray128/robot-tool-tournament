"""rtt.cloud.storage against an in-memory S3 client (no network)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from rtt.cloud.config import ConfigError, NebiusConfig
from rtt.cloud.storage import BucketStore, container_path_to_key


class FakeS3:
    """The few S3 client calls BucketStore makes, over a dict; pages hold 2 keys."""

    page_size = 2

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def list_objects_v2(self, Bucket: str, Prefix: str, ContinuationToken: str | None = None) -> dict:
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start : start + self.page_size]
        out: dict[str, Any] = {
            "Contents": [{"Key": k, "Size": len(self.objects[k])} for k in page],
            "IsTruncated": start + self.page_size < len(keys),
        }
        if out["IsTruncated"]:
            out["NextContinuationToken"] = str(start + self.page_size)
        return out

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[key] = Path(filename).read_bytes()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[key])

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.objects[Key] = Body

    def get_object(self, Bucket: str, Key: str) -> dict:
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture
def store() -> BucketStore:
    return BucketStore("bucket", client=FakeS3())


def test_upload_and_download_round_trip(tmp_path: Path, store: BucketStore) -> None:
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.json").write_text("{}")
    (src / "b.bin").write_bytes(b"\x00\x01")
    (src / "sub" / "c.mp4").write_bytes(b"mp4")
    keys = store.upload_dir(src, "runs/m1-x/")
    assert keys == ["runs/m1-x/a.json", "runs/m1-x/b.bin", "runs/m1-x/sub/c.mp4"]

    # three objects span two list pages; a sibling prefix must not leak in
    store.client.objects["runs/m1-xy/other.json"] = b"{}"
    assert [o.key for o in store.list_objects("runs/m1-x/")] == keys
    files = store.download_prefix("runs/m1-x", tmp_path / "dest")
    assert sorted(p.relative_to(tmp_path / "dest").as_posix() for p in files) == [
        "a.json",
        "b.bin",
        "sub/c.mp4",
    ]
    assert (tmp_path / "dest" / "b.bin").read_bytes() == b"\x00\x01"
    assert store.exists("runs/m1-x/a.json") and not store.exists("runs/m1-x/zzz")


def test_download_skips_folder_markers(tmp_path: Path, store: BucketStore) -> None:
    store.client.objects.update({"runs/r/": b"", "runs/r/sub/": b"", "runs/r/sub/a.json": b"{}"})
    files = store.download_prefix("runs/r", tmp_path / "dest")
    assert [p.relative_to(tmp_path / "dest").as_posix() for p in files] == ["sub/a.json"]
    assert (tmp_path / "dest" / "sub").is_dir()


def test_download_refuses_keys_that_escape(tmp_path: Path, store: BucketStore) -> None:
    store.client.objects["runs/r/../../evil"] = b"x"
    with pytest.raises(ValueError, match="unsafe object key"):
        store.download_prefix("runs/r", tmp_path / "dest")
    assert not (tmp_path / "evil").exists()


def test_json_helpers(store: BucketStore) -> None:
    store.put_json("runs/r/job.json", {"state": "COMPLETED", "n": 1})
    assert store.get_json("runs/r/job.json") == {"n": 1, "state": "COMPLETED"}
    assert json.loads(store.client.objects["runs/r/job.json"])["state"] == "COMPLETED"


def test_upload_dir_needs_a_directory(tmp_path: Path, store: BucketStore) -> None:
    with pytest.raises(NotADirectoryError):
        store.upload_dir(tmp_path / "missing", "x")


def test_container_path_to_key() -> None:
    assert container_path_to_key("/out/runs/m1-x/report.json") == "runs/m1-x/report.json"
    with pytest.raises(ValueError):
        container_path_to_key("/tmp/elsewhere")


def test_from_config_needs_bucket_name_and_hides_it() -> None:
    with pytest.raises(ConfigError, match="NEBIUS_BUCKET_NAME"):
        BucketStore.from_config(NebiusConfig())
    store = BucketStore.from_config(NebiusConfig(bucket_name="private-bucket", s3_profile="nebius-eun1"))
    assert store.profile == "nebius-eun1" and "private-bucket" not in repr(store)
