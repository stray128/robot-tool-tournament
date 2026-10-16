"""Nebius Object Storage access from the laptop (S3 API through boto3).

Jobs see the results bucket as a filesystem mounted at ``/out``; the laptop reaches the
same objects over S3 with the named AWS profile (default ``nebius-eun1``, whose config
carries the Nebius endpoint). The profile is always named explicitly so the default AWS
credentials on the machine are never used by accident.

A run written by a job to ``/out/runs/<run_id>/`` is the prefix ``runs/<run_id>/`` here.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from rtt.cloud.config import NebiusConfig

MOUNT_POINT = "/out"


def container_path_to_key(path: str, mount: str = MOUNT_POINT) -> str:
    """``/out/runs/x/report.json`` -> ``runs/x/report.json``."""
    p = PurePosixPath(path)
    rel = p.relative_to(mount)
    return str(rel)


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    last_modified: datetime | None


def _safe_relative(key: str, prefix: str) -> PurePosixPath:
    """``key`` relative to ``prefix``, refusing anything that could escape the target dir."""
    rel = key[len(prefix) :].lstrip("/") if key.startswith(prefix) else key
    parts = PurePosixPath(rel).parts
    if not parts or any(part in ("..", "") for part in parts) or rel.startswith("/"):
        raise ValueError(f"refusing unsafe object key {key!r}")
    return PurePosixPath(*parts)


class BucketStore:
    """Upload, download and list objects in one bucket.

    Args:
        bucket: Bucket name.
        profile: AWS shared-config profile with the Nebius S3 keys and endpoint.
        endpoint_url: Overrides the profile's endpoint (e.g. ``NEBIUS_S3_ENDPOINT``).
        client: A ready S3 client (tests pass a fake); skips session creation.
    """

    def __init__(
        self,
        bucket: str,
        profile: str | None = "nebius-eun1",
        endpoint_url: str | None = None,
        client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.profile = profile
        self.endpoint_url = endpoint_url
        self._client = client

    def __repr__(self) -> str:
        return f"BucketStore(bucket=<set>, profile={self.profile!r})"

    @classmethod
    def from_config(cls, config: NebiusConfig) -> BucketStore:
        """The bucket of ``config``'s region (use ``config.for_region(...)`` for another region).

        The S3 profile is required, never defaulted: without it boto3 would fall back to
        the machine's default AWS credentials.
        """
        config.require("bucket_name", "s3_profile")
        assert config.bucket_name is not None
        return cls(config.bucket_name, config.s3_profile, config.s3_endpoint)

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            session = boto3.Session(profile_name=self.profile)
            self._client = session.client("s3", endpoint_url=self.endpoint_url)
        return self._client

    # -- listing

    def iter_objects(self, prefix: str) -> Iterator[ObjectInfo]:
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            for item in page.get("Contents", []) or []:
                yield ObjectInfo(item["Key"], int(item.get("Size", 0)), item.get("LastModified"))
            if not page.get("IsTruncated"):
                return
            token = page.get("NextContinuationToken")

    def list_objects(self, prefix: str) -> list[ObjectInfo]:
        """Every object under ``prefix``, sorted by key."""
        return sorted(self.iter_objects(prefix), key=lambda o: o.key)

    def exists(self, key: str) -> bool:
        return any(o.key == key for o in self.iter_objects(key))

    # -- transfer

    def upload_file(self, local: str | Path, key: str) -> str:
        self.client.upload_file(str(local), self.bucket, key)
        return key

    def upload_dir(self, local_dir: str | Path, prefix: str) -> list[str]:
        """Upload every file below ``local_dir`` to ``prefix/<relative path>``."""
        root = Path(local_dir)
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        keys = []
        base = prefix.rstrip("/")
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            key = f"{base}/{path.relative_to(root).as_posix()}" if base else path.relative_to(root).as_posix()
            keys.append(self.upload_file(path, key))
        return keys

    def download_file(self, key: str, local: str | Path) -> Path:
        dest = Path(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest))
        return dest

    def download_prefix(self, prefix: str, dest_dir: str | Path) -> list[Path]:
        """Download every object under ``prefix`` into ``dest_dir``, keeping relative paths.

        Keys ending in ``/`` are folder markers (written by consoles and some FUSE
        mounts), not files, and are skipped.
        """
        root = Path(dest_dir)
        norm = prefix if prefix.endswith("/") or not prefix else prefix + "/"
        out = []
        for obj in self.list_objects(norm):
            if obj.key.endswith("/"):
                continue
            rel = _safe_relative(obj.key, norm)
            out.append(self.download_file(obj.key, root.joinpath(*rel.parts)))
        return out

    # -- JSON helpers

    def put_json(self, key: str, data: Any) -> str:
        body = (json.dumps(data, indent=2, sort_keys=True, default=str) + "\n").encode()
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")
        return key

    def get_json(self, key: str) -> Any:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read())
