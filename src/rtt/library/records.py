"""Library items on disk: ``SOURCE.json`` per item, ``lock.json`` for the library, verification.

Every item lives in ``assets/effectors/library/<key>/`` under a generic key. Its ``SOURCE.json``
(``format = "rtt-library-source"``, version 1) has a hand-written head and records that
``scripts/vendor_library.py`` fills in:

* head: ``key``, ``display_name`` and ``summary`` (brand-free, they go on screen), ``kind``,
  ``class``, ``capabilities``, ``upstream`` (``repository``, pinned ``commit``), ``licence``
  (one entry per licence and what it covers), ``copyright``, ``attribution`` (the credit line,
  may name the maker: credits only), ``chain`` (the licence chain, :class:`~rtt.library.licence.ChainLink`),
  ``reviewed``, ``mount`` (:class:`~rtt.library.mounts.Mount`), ``specs`` (published values,
  each with a source and a tag), ``sim`` (status and how the model was made), ``ingest`` (what
  the vendoring script fetches and derives);
* written by the script: ``files`` (one :class:`LibraryFile` per shipped file: path, sha256 as
  shipped, role, upstream path and sha256 at the pinned commit, licence, rating, changes),
  ``skipped`` (upstream files left out, for size, with a link instead), ``links`` (upstream
  download links), ``preview`` (the GLB the catalog page shows, a derived file) and ``derived``
  (mesh replacements and pinned inertials, the catalog loader's format).

``lock.json`` pins the whole library: per item the upstream repository and commit, the
sha256 of every upstream licence file (a changed licence at the pinned commit fails the
vendoring script: licence drift), the sha256 of every shipped file and of ``SOURCE.json``, and
one manifest hash over all of it.

:func:`verify_item` and :func:`verify_library` re-hash everything and apply the licence gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Final

from rtt.library.brands import check_library_name, find_library_brands
from rtt.library.licence import ChainLink, Rating, chain_rating, classify, gate
from rtt.library.mounts import Mount
from rtt.library.taxonomy import CAPABILITIES, CLASSES, KINDS, SIM_STATUSES

__all__ = [
    "GENERATED",
    "LIBRARY_DIR",
    "LOCK_FORMAT",
    "LOCK_PATH",
    "REPO_ROOT",
    "ROLES",
    "SOURCE_FORMAT",
    "LibraryFile",
    "LibraryItem",
    "Spec",
    "build_lock",
    "item_keys",
    "licence_key",
    "manifest_sha256",
    "read_item",
    "read_lock",
    "sha256_file",
    "verify_item",
    "verify_library",
]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
LIBRARY_DIR: Final = REPO_ROOT / "assets" / "effectors" / "library"
LOCK_PATH: Final = LIBRARY_DIR / "lock.json"
SOURCE_FORMAT: Final = "rtt-library-source"
LOCK_FORMAT: Final = "rtt-library-lock"
VERSION: Final = 1
#: What a shipped file is for.
ROLES: Final[tuple[str, ...]] = ("licence", "sim", "print", "cad", "derived", "doc")
SPEC_TAGS: Final[tuple[str, ...]] = ("sourced", "derived", "assumed")
#: ``upstream_commit`` of a file this project generates (our own work, no upstream commit).
GENERATED: Final = "generated"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_KEY = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class LibraryFile:
    """One shipped file of an item (a ``files`` entry of ``SOURCE.json``)."""

    path: str
    sha256: str
    size: int
    role: str
    upstream_url: str
    upstream_commit: str
    upstream_path: str
    upstream_sha256: str
    license: str
    copyright: str
    changes: tuple[str, ...] = ()
    chain: tuple[ChainLink, ...] = ()

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith("/") or ".." in self.path.split("/"):
            raise ValueError(f"path must be repository-relative, got {self.path!r}")
        for name in ("sha256", "upstream_sha256"):
            if not _SHA256.match(getattr(self, name)):
                raise ValueError(f"{self.path}: {name} must be a lowercase hex sha256")
        if self.role not in ROLES:
            raise ValueError(f"{self.path}: role must be one of {ROLES}")
        if not self.upstream_url.startswith("https://"):
            raise ValueError(f"{self.path}: upstream_url must be https")
        generated = self.upstream_commit == GENERATED
        if not (_COMMIT.match(self.upstream_commit) or generated):
            raise ValueError(f"{self.path}: upstream_commit must be a full 40-hex commit id")
        object.__setattr__(self, "changes", tuple(self.changes))
        object.__setattr__(self, "chain", tuple(self.chain))
        if self.sha256 != self.upstream_sha256 and not self.changes:
            raise ValueError(f"{self.path}: differs from upstream, so its changes must be listed")
        if not self.copyright:
            raise ValueError(f"{self.path}: a copyright line is required")

    @property
    def rating(self) -> Rating:
        """The worst of the file's own licence and its chain (the chain rule)."""
        own = ChainLink(self.license, "licensed", "the file's licence")
        return chain_rating((own, *self.chain))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "role": self.role,
            "upstream_url": self.upstream_url,
            "upstream_commit": self.upstream_commit,
            "upstream_path": self.upstream_path,
            "upstream_sha256": self.upstream_sha256,
            "license": self.license,
            "copyright": self.copyright,
            "changes": list(self.changes),
            "rating": self.rating.value,
        }
        if self.chain:
            out["chain"] = [c.to_dict() for c in self.chain]
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LibraryFile:
        return cls(
            path=str(raw["path"]),
            sha256=str(raw["sha256"]),
            size=int(raw["size"]),
            role=str(raw["role"]),
            upstream_url=str(raw["upstream_url"]),
            upstream_commit=str(raw["upstream_commit"]),
            upstream_path=str(raw["upstream_path"]),
            upstream_sha256=str(raw["upstream_sha256"]),
            license=str(raw["license"]),
            copyright=str(raw["copyright"]),
            changes=tuple(raw.get("changes", ())),
            chain=tuple(ChainLink.from_dict(c) for c in raw.get("chain", ())),
        )


@dataclass(frozen=True)
class Spec:
    """One published specification value, with where it comes from."""

    name: str
    value: str
    unit: str = ""
    tag: str = "sourced"
    source: str = ""

    def __post_init__(self) -> None:
        if self.tag not in SPEC_TAGS:
            raise ValueError(f"spec {self.name!r}: tag must be one of {SPEC_TAGS}")
        if self.tag == "sourced" and not self.source.startswith("https://"):
            raise ValueError(f"spec {self.name!r}: a sourced value needs an https source")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "tag": self.tag,
            "source": self.source,
        }


@dataclass(frozen=True)
class LibraryItem:
    """One item's ``SOURCE.json`` (module docstring)."""

    key: str
    directory: Path
    display_name: str
    summary: str
    kind: str
    item_class: str
    capabilities: tuple[str, ...]
    upstream: Mapping[str, str]
    licence: tuple[Mapping[str, str], ...]
    copyright: str
    attribution: str
    chain: tuple[ChainLink, ...]
    mount: Mount
    specs: tuple[Spec, ...]
    sim: Mapping[str, Any]
    files: tuple[LibraryFile, ...]
    skipped: tuple[Mapping[str, Any], ...] = ()
    links: tuple[Mapping[str, str], ...] = ()
    preview: Mapping[str, Any] | None = None
    derived: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def rating(self) -> Rating:
        """The worst link over the item's chain and every shipped file."""
        links = list(self.chain) + [ChainLink(f.license, "licensed", f.path) for f in self.files]
        links += [c for f in self.files for c in f.chain]
        return chain_rating(links) if links else Rating.RED

    @property
    def sim_status(self) -> str:
        return str(self.sim.get("status", ""))

    def files_with(self, role: str) -> tuple[LibraryFile, ...]:
        return tuple(f for f in self.files if f.role == role)


def item_keys() -> tuple[str, ...]:
    """Every item directory with a ``SOURCE.json`` (sorted)."""
    if not LIBRARY_DIR.is_dir():
        return ()
    return tuple(sorted(p.parent.name for p in LIBRARY_DIR.glob("*/SOURCE.json")))


def _tuple_of_dicts(raw: Iterable[Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(x) for x in raw)


@cache
def read_item(key: str) -> LibraryItem:
    """Parse ``assets/effectors/library/<key>/SOURCE.json`` (structure checked on load)."""
    directory = LIBRARY_DIR / key
    raw = json.loads((directory / "SOURCE.json").read_text(encoding="utf-8"))
    if raw.get("format") != SOURCE_FORMAT or raw.get("version") != VERSION:
        raise ValueError(f"{key}: SOURCE.json is not {SOURCE_FORMAT} version {VERSION}")
    if raw.get("key") != key:
        raise ValueError(f"{key}: SOURCE.json describes {raw.get('key')!r}")
    return LibraryItem(
        key=key,
        directory=directory,
        display_name=str(raw["display_name"]),
        summary=str(raw.get("summary", "")),
        kind=str(raw["kind"]),
        item_class=str(raw["class"]),
        capabilities=tuple(raw.get("capabilities", ())),
        upstream=dict(raw["upstream"]),
        licence=_tuple_of_dicts(raw.get("licence", ())),
        copyright=str(raw["copyright"]),
        attribution=str(raw["attribution"]),
        chain=tuple(ChainLink.from_dict(c) for c in raw.get("chain", ())),
        mount=Mount.from_dict(raw["mount"]),
        specs=tuple(Spec(**s) for s in raw.get("specs", ())),
        sim=dict(raw.get("sim", {})),
        files=tuple(LibraryFile.from_dict(f) for f in raw.get("files", ())),
        skipped=_tuple_of_dicts(raw.get("skipped", ())),
        links=_tuple_of_dicts(raw.get("links", ())),
        preview=dict(raw["preview"]) if raw.get("preview") else None,
        derived=dict(raw.get("derived", {})),
        notes=tuple(raw.get("notes", ())),
        raw=raw,
    )


def _screen_problems(item: LibraryItem) -> list[str]:
    """Strings that reach the screen must be brand-free (URLs aside)."""
    out = []
    url = re.compile(r"https?://\S+")
    screen = [item.key, item.display_name, item.summary, item.mount.interface, *item.mount.notes, *item.notes]
    screen += [f"{s.name} {s.value} {s.unit}" for s in item.specs]
    screen += [str(item.sim.get(k, "")) for k in ("source", "reason", "note")]
    for text in screen:
        hits = find_library_brands(url.sub(" ", text))
        if hits:
            out.append(
                f"{item.key}: on-screen text names a brand ({', '.join(sorted(set(hits)))}): {text[:60]!r}"
            )
    try:
        check_library_name(item.display_name)
    except ValueError as exc:
        out.append(f"{item.key}: {exc}")
    return out


def verify_item(key: str) -> list[str]:
    """Problems with one item (empty: every file re-hashes, the gate passes, the head is valid)."""
    item = read_item(key)
    problems: list[str] = []
    if not _KEY.match(key) or find_library_brands(key.replace("-", " ")):
        problems.append(f"{key}: keys are generic slugs")
    if item.kind not in KINDS:
        problems.append(f"{key}: unknown kind {item.kind!r}")
    if item.item_class not in CLASSES:
        problems.append(f"{key}: unknown class {item.item_class!r}")
    bad_caps = set(item.capabilities) - set(CAPABILITIES)
    if bad_caps or len(set(item.capabilities)) != len(item.capabilities):
        problems.append(f"{key}: capability tags must be distinct and from {CAPABILITIES}")
    status = item.sim_status
    if status not in SIM_STATUSES:
        problems.append(f"{key}: sim status must be one of {SIM_STATUSES}, got {status!r}")
    if status in ("visual_only", "none") and not item.sim.get("reason"):
        problems.append(f"{key}: a {status} item must say why it has no physics")
    if status in ("native", "proxy") and not (item.sim.get("model_file") or item.sim.get("family")):
        problems.append(f"{key}: a {status} item names its model file or family")
    problems += _screen_problems(item)
    # licence gate: the chain and every file must be green (the founder excluded amber)
    decision = gate(list(item.chain) + [ChainLink(f.license, "licensed", f.path) for f in item.files])
    if item.rating is not Rating.GREEN or not decision.shippable:
        problems.append(f"{key}: licence gate fails ({item.rating.value}): {'; '.join(decision.reasons)}")
    if not item.chain:
        problems.append(f"{key}: the licence chain is empty")
    for lic in item.licence:
        if classify(str(lic.get("spdx", ""))) is not Rating.GREEN:
            problems.append(f"{key}: licence {lic.get('spdx')!r} is not in the green lane")
    if item.files and not item.files_with("licence"):
        problems.append(f"{key}: ships files but no licence text")
    # every shipped file re-hashes, sits in the item directory and is listed
    listed = set()
    prefix = f"assets/effectors/library/{key}/"
    for f in item.files:
        path = REPO_ROOT / f.path
        listed.add(path.resolve())
        if not f.path.startswith(prefix):
            problems.append(f"{f.path}: outside the item's directory")
        if not path.is_file():
            problems.append(f"{f.path}: missing")
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != f.sha256:
            problems.append(f"{f.path}: sha256 differs from SOURCE.json")
        if len(data) != f.size:
            problems.append(f"{f.path}: size differs from SOURCE.json")
        pins = {(item.upstream.get("repository"), item.upstream.get("commit"))}
        pins |= {(e.get("repository"), e.get("commit")) for e in item.upstream.get("extra", ())}  # type: ignore[union-attr]
        if f.role != "derived" and (f.upstream_url, f.upstream_commit) not in pins:
            problems.append(
                f"{f.path}: pinned to {f.upstream_url}@{f.upstream_commit[:8]}, not one of the item's pins"
            )
    for path in sorted(item.directory.rglob("*")):
        if path.is_file() and path.name != "SOURCE.json" and path.resolve() not in listed:
            problems.append(f"{path.relative_to(REPO_ROOT)}: shipped but not in SOURCE.json")
    if item.preview:
        p = REPO_ROOT / str(item.preview.get("path", ""))
        if not p.is_file():
            problems.append(f"{key}: preview {item.preview.get('path')} missing")
        elif sha256_file(p) != item.preview.get("sha256"):
            problems.append(f"{key}: preview sha256 differs from SOURCE.json")
    model_file = item.sim.get("model_file")
    if model_file and not (item.directory / str(model_file)).is_file():
        problems.append(f"{key}: sim model file {model_file} missing")
    return problems


# --------------------------------------------------------------------------- lock


def manifest_sha256(items: Mapping[str, Mapping[str, Any]], catalog: str | None = None) -> str:
    """sha256 over every (item, path, sha256) and licence pin, in sorted order, and the catalog."""
    lines = [f"catalog {catalog}"] if catalog else []
    for key in sorted(items):
        entry = items[key]
        lines.append(f"item {key} {entry['upstream']['repository']} {entry['upstream']['commit']}")
        for extra in entry["upstream"].get("extra", ()):
            lines.append(f"pin {key} {extra['repository']} {extra['commit']}")
        lines.append(f"source {key} {entry['source_json_sha256']}")
        for path, sha in sorted(entry["licence_files"].items()):
            lines.append(f"licence {key} {path} {sha}")
        for path, sha in sorted(entry["files"].items()):
            lines.append(f"file {path} {sha}")
        if entry.get("preview"):
            lines.append(f"preview {key} {entry['preview']}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def licence_key(item: LibraryItem | Mapping[str, Any], repository: str, path: str) -> str:
    """The ``lock.json`` key of a licence file: its path, prefixed by its repository when that
    is not the item's own (two repositories may both have a ``LICENSE``)."""
    main = item.upstream["repository"] if isinstance(item, LibraryItem) else item["upstream"]["repository"]
    return path if repository == main else f"{repository}#{path}"


def _lock_entry(item: LibraryItem) -> dict[str, Any]:
    return {
        "upstream": {
            "repository": item.upstream["repository"],
            "commit": item.upstream["commit"],
            "extra": [dict(e) for e in item.upstream.get("extra", ())],  # type: ignore[union-attr]
        },
        "source_json_sha256": sha256_file(item.directory / "SOURCE.json"),
        "licence_files": {
            licence_key(item, f.upstream_url, f.upstream_path): f.upstream_sha256
            for f in item.files
            if f.role == "licence"
        },
        "files": {f.path: f.sha256 for f in item.files},
        "preview": item.preview.get("sha256") if item.preview else None,
    }


CATALOG_PATH: Final = LIBRARY_DIR / "catalog.json"


def build_lock(keys: Sequence[str] | None = None) -> dict[str, Any]:
    """The lock of the library as it is on disk now (items, and the listed/excluded catalog)."""
    items = {key: _lock_entry(read_item(key)) for key in (keys if keys is not None else item_keys())}
    catalog = sha256_file(CATALOG_PATH) if CATALOG_PATH.is_file() else None
    return {
        "format": LOCK_FORMAT,
        "version": VERSION,
        "items": items,
        "catalog_sha256": catalog,
        "manifest_sha256": manifest_sha256(items, catalog),
    }


def read_lock() -> dict[str, Any]:
    raw = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if raw.get("format") != LOCK_FORMAT or raw.get("version") != VERSION:
        raise ValueError(f"lock.json is not {LOCK_FORMAT} version {VERSION}")
    return raw


def verify_library() -> list[str]:
    """Every item verified, and ``lock.json`` equal to the library on disk."""
    problems: list[str] = []
    keys = item_keys()
    for key in keys:
        problems += verify_item(key)
    if not LOCK_PATH.is_file():
        return [*problems, "assets/effectors/library/lock.json: missing"]
    lock = read_lock()
    current = build_lock(keys)
    if set(lock["items"]) != set(keys):
        problems.append(f"lock.json lists {sorted(lock['items'])}, the library has {list(keys)}")
    for key in keys:
        if key in lock["items"] and lock["items"][key] != current["items"][key]:
            problems.append(f"lock.json entry for {key} differs from the files on disk")
    if lock.get("catalog_sha256") != current["catalog_sha256"]:
        problems.append("lock.json catalog_sha256 differs from catalog.json")
    if lock.get("manifest_sha256") != manifest_sha256(lock["items"], lock.get("catalog_sha256")):
        problems.append("lock.json manifest_sha256 does not match its entries")
    return problems
