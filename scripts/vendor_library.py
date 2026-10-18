#!/usr/bin/env python
"""Vendor and verify the open effector library (product mode, not pre-registered).

Each item directory ``assets/effectors/library/<key>/`` holds a hand-written ``SOURCE.json``
head: the upstream repository and the pinned commit, the licence chain, specs with sources,
the mount, capability tags, and under ``ingest`` what to fetch and derive
(:mod:`rtt.library.ingest`). This script fills in the rest from the upstream files at the
pinned commit and rewrites ``assets/effectors/library/lock.json``.

Usage (repository root):
    python scripts/vendor_library.py                 vendor every item (downloads, cached)
    python scripts/vendor_library.py --only KEY ...  limit to some items
    python scripts/vendor_library.py --check         re-hash every shipped file against
                                                     SOURCE.json and lock.json, apply the
                                                     licence gate (no network); exit 1 on a problem
    python scripts/vendor_library.py --drift         compare every pinned licence file with
                                                     the upstream default branch today; exit 1
                                                     if one changed (review before bumping)

Licence drift: a re-vendor that finds a licence file whose sha256 differs from the one in
``lock.json`` stops (exit 3) and names the file. Downloads are https only, capped at 50 MB per
file, cached under ``--cache`` (default ``~/.cache/rtt-library``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rtt.library.fetch import Fetcher
from rtt.library.ingest import LicenceDrift, vendor_item
from rtt.library.records import (
    LOCK_PATH,
    build_lock,
    item_keys,
    read_item,
    read_lock,
    verify_item,
    verify_library,
)

DEFAULT_CACHE = Path.home() / ".cache" / "rtt-library"


def check(keys: list[str]) -> int:
    problems = verify_library() if keys == list(item_keys()) else [p for k in keys for p in verify_item(k)]
    for key in keys:
        mine = [p for p in problems if p.startswith((key, f"assets/effectors/library/{key}/"))]
        print(f"{key}: {'ok' if not mine else f'{len(mine)} problem(s)'}")
    for p in problems:
        print(f"  {p}")
    if not problems:
        lock = read_lock()
        print(f"lock.json: ok, {len(lock['items'])} items, manifest {lock['manifest_sha256'][:16]}")
    return 1 if problems else 0


def drift(keys: list[str]) -> int:
    """Licence files at the upstream default branch today versus the pinned ones."""
    lock = read_lock()
    changed = 0
    for key in keys:
        item = read_item(key)
        for lkey, pinned in sorted(lock["items"].get(key, {}).get("licence_files", {}).items()):
            repo, _, path = lkey.rpartition("#") if "#" in lkey else (item.upstream["repository"], "", lkey)
            if not repo.startswith("https://github.com/"):
                continue
            owner_repo = repo.removeprefix("https://github.com/").rstrip("/")
            quoted = "/".join(urllib.parse.quote(p) for p in path.split("/"))
            url = f"https://raw.githubusercontent.com/{owner_repo}/HEAD/{quoted}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "rtt-vendor-library"})
                with urllib.request.urlopen(req, timeout=60) as resp:  # https only (built above)
                    now = hashlib.sha256(resp.read()).hexdigest()
            except OSError as exc:
                print(f"{key}: {path}: could not fetch ({exc})")
                changed += 1
                continue
            same = now == pinned
            changed += not same
            print(f"{key}: {path}: {'unchanged' if same else 'CHANGED upstream: review before bumping'}")
    return 1 if changed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify shipped files only (no network)")
    ap.add_argument("--drift", action="store_true", help="compare licence files with upstream today")
    ap.add_argument("--only", nargs="*", default=None, help="item keys to process")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="download cache directory")
    ap.add_argument("--offline", action="store_true", help="use the cache only")
    args = ap.parse_args(argv)
    keys = args.only or list(item_keys())
    if args.check:
        return check(keys)
    if args.drift:
        return drift(keys)
    fetcher = Fetcher(args.cache, offline=args.offline)
    lock = read_lock() if LOCK_PATH.is_file() else None
    for key in keys:
        try:
            report = vendor_item(key, fetcher, lock=lock)
        except LicenceDrift as exc:
            print(f"licence drift: {exc}", file=sys.stderr)
            return 3
        print(report.line())
    new = build_lock()
    LOCK_PATH.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"downloaded {fetcher.downloaded / 1048576:.1f} MB; lock.json manifest {new['manifest_sha256'][:16]}"
    )
    return check(list(item_keys()))


if __name__ == "__main__":
    sys.exit(main())
