#!/usr/bin/env python
"""Fetch the pinned third-party asset files that the renderer and the stock entrant use.

Every vendored asset directory holds a ``lock.json`` that pins its upstream repository,
commit and, for each file, the upstream path, byte size and sha256. This script checks the
files against the lock and downloads any that are missing or wrong from the pinned commit,
so a fresh checkout, a CI runner or a job image always gets byte-identical inputs.

Usage:
    python scripts/fetch_assets.py              verify, download what is missing or wrong
    python scripts/fetch_assets.py --check      verify only; exit 1 if anything is off
    python scripts/fetch_assets.py --from DIR   copy from a local mirror of the upstream
                                                repository (same layout) instead of the network
    python scripts/fetch_assets.py --hulls      also rebuild the stock blade's convex hulls
                                                (CoACD, pinned settings) and print the
                                                inner-face bulge report

The fetch itself needs only the standard library. ``--hulls`` needs the repository package
(``pip install -e .``) because it runs CoACD through ``rtt.render.gripper``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCKS = (REPO / "assets" / "gripper" / "so_arm_102" / "lock.json",)
MAX_BYTES = 50 * 1024 * 1024  # refuse anything this large: the pinned files are < 1 MB
TIMEOUT_S = 60


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status(path: Path, entry: dict) -> str:
    if not path.is_file():
        return "missing"
    if path.stat().st_size != entry["bytes"]:
        return "wrong size"
    if _sha256(path) != entry["sha256"]:
        return "wrong sha256"
    return "ok"


def _download(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-https URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "rtt-fetch-assets"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # https only, checked above
        data = resp.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"{url}: larger than {MAX_BYTES} bytes")
    return data


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; these are ordinary repository files
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def sync_lock(lock_path: Path, *, check_only: bool, mirror: Path | None) -> list[str]:
    """Bring one locked directory up to date; returns the problems left (empty = all good)."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    root = lock_path.parent
    problems = []
    for rel, entry in sorted(lock["files"].items()):
        dest = root / rel
        state = _status(dest, entry)
        if state == "ok":
            print(f"  ok       {rel}")
            continue
        if check_only:
            problems.append(f"{root.name}/{rel}: {state}")
            print(f"  {state:8s} {rel}")
            continue
        if mirror is not None:
            src = mirror / entry["path"]
            data = src.read_bytes()
            where = f"mirror {entry['path']}"
        else:
            url = f"{lock['raw_base']}/{lock['commit']}/{entry['path']}"
            data = _download(url)
            where = f"{lock['commit'][:10]} {entry['path']}"
        got = hashlib.sha256(data).hexdigest()
        if len(data) != entry["bytes"] or got != entry["sha256"]:
            problems.append(f"{root.name}/{rel}: fetched file does not match the lock (sha256 {got[:12]}...)")
            print(f"  MISMATCH {rel} from {where}")
            continue
        _write_atomic(dest, data)
        print(f"  fetched  {rel} from {where}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify only, never download")
    ap.add_argument("--from", dest="mirror", type=Path, help="local mirror of the upstream repository")
    ap.add_argument("--hulls", action="store_true", help="rebuild the stock blade hulls (needs rtt)")
    args = ap.parse_args(argv)
    problems: list[str] = []
    for lock in LOCKS:
        print(f"{lock.parent.relative_to(REPO)}:")
        problems += sync_lock(lock, check_only=args.check, mirror=args.mirror)
    if problems:
        print("fetch_assets: " + "; ".join(problems), file=sys.stderr)
        return 1
    if args.hulls:
        from rtt.render.gripper import HULLS_PATH, build_stock_blade_hulls, stock_blade_report

        tmp = HULLS_PATH.with_suffix(".tmp.npz")
        build_stock_blade_hulls(tmp)
        shutil.move(tmp, HULLS_PATH)
        print(json.dumps(stock_blade_report(), indent=1))
    print("fetch_assets: all pinned files present and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
