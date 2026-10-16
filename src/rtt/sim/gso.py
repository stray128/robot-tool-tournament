"""Google Scanned Objects (GSO) as a real-object held-out set.

The objects are household items scanned by Google Research (Downs et al., 2022) in the
MuJoCo conversion ``kevinzakka/mujoco_scanned_objects``, pinned to one commit. Its MJCF
files are MIT; the meshes and textures are CC BY 4.0. **Nothing is vendored**: each object
the selection examines has its ``model.obj`` (the scan, in metres, 1.2 MB on average) fetched at
run time from the commit-pinned raw URL into a cache directory (``$RTT_GSO_CACHE``,
default ``~/.cache/rtt/gso``). The SHA-256 of every fetched file is recorded, and the
object listing is checked against the pinned git tree id before use.

**Selection rule** (:func:`select_objects`; parameters in
``configs/suites/heldout_real_v1.yaml``). It is a pure function of the config, the task,
this code and a seed:

1. *Catalogue*: every object directory under ``models/`` at the commit, sorted by name.
2. *Category rule* (:class:`Denylist`): drop any object with a denylisted name token,
   and the few objects listed individually by the SHA-256 of their full name.
3. *Draw order*: a permutation of the rest from the seed (SHA-256 domain separated).
4. *Walk*: examine objects in that order and accept one when it passes every rule in
   :class:`ScreenRules`, until ``n`` are accepted. An object passes when, in its most
   probable stable resting pose that fits (:func:`choose_pose`):

   - its width across the jaw (the narrowest horizontal extent, which the intake path
     turns onto the closing axis) is at most ``max_grip_width_mm``;
   - its height is at least ``min_height_mm``, and its length and height at most
     ``max_extent_mm``;
   - its mass as a solid at the task's object density is at most the cap
     :meth:`ScreenRules.mass_cap_kg` derived from the grip force;
   - its convex decomposition compiles in the task's MuJoCo scene, and it stays put for
     2 s on the table with the gripper parked above it (:func:`rest_test`).

   Every examined object that fails is recorded with its reason
   (:class:`Exclusion`). Objects listed by name hash are recorded by that hash only.

Each accepted object goes through the same intake path as an uploaded part
(:func:`rtt.sim.mesh.mesh_object_from_trimesh`: stable pose, grasp frame, CoACD), and is
cached as ``.npz``. :func:`selection_suite` turns a :class:`Selection` into an ordinary
:class:`~rtt.sim.conditions.Suite`, so :func:`rtt.sim.evaluate.evaluate_designs` and the
held-out runner score it like any other suite. :func:`attribution` gives the per-object
credit line that CC BY 4.0 requires.
"""

from __future__ import annotations

import collections
import concurrent.futures as cf
import functools
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import mujoco
import numpy as np
import trimesh
import yaml

from rtt.design.types import FingerGeometry
from rtt.sim import objects as objs
from rtt.sim.conditions import REPO_ROOT, Condition, Suite
from rtt.sim.mesh import mesh_object_from_trimesh, read_mesh_object, save_mesh_object, stable_poses
from rtt.sim.task import TaskConfig, default_task, task_digest

__all__ = [
    "ASSET_LICENSE",
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_V2",
    "GSO_CITATION",
    "MAX_PACK_FILE_BYTES",
    "CandidateResult",
    "DecompositionRules",
    "Denylist",
    "DrawConfig",
    "Exclusion",
    "GsoCache",
    "GsoConfig",
    "GsoSource",
    "HttpFetcher",
    "LocalMirror",
    "PlacementRanges",
    "PoseFit",
    "RestCheck",
    "RestRules",
    "ScreenRules",
    "Selection",
    "attribution",
    "attribution_markdown",
    "choose_pose",
    "dev_seed_material",
    "draw_order",
    "evaluate_candidate",
    "geometry_mesh",
    "git_tree_sha1",
    "load_gso_config",
    "load_selection_objects",
    "name_tokens",
    "pose_fits",
    "rest_test",
    "secret_seed_material",
    "select_objects",
    "selection_part_suite",
    "selection_suite",
]

DEFAULT_CONFIG = REPO_ROOT / "configs" / "suites" / "heldout_real_v1.yaml"
DEFAULT_CONFIG_V2 = REPO_ROOT / "configs" / "suites" / "heldout_real_v2.yaml"
TEXTURE_FILE = "texture.png"
CACHE_ENV = "RTT_GSO_CACHE"
DEFAULT_CACHE = Path("~/.cache/rtt/gso")
SCREEN_VERSION = 1  # bump whenever a change here can change a candidate's outcome
SELECTION_SCHEMA = "rtt.gso.selection/1"

ASSET_LICENSE = "CC-BY-4.0"
ASSET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
XML_LICENSE = "MIT"
GSO_PAPER_URL = "https://arxiv.org/abs/2204.11918"
GSO_CITATION = (
    "Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, "
    "Thomas B. McHugh and Vincent Vanhoucke. Google Scanned Objects: A High-Quality Dataset of "
    "3D Scanned Household Items. IEEE ICRA 2022. arXiv:2204.11918"
)
GSO_CREDIT = "Google Scanned Objects by Google Research, MuJoCo models by Kevin Zakka"
GSO_CHANGES = (
    "geometry only (no texture); re-posed to a stable resting pose and split into convex "
    "pieces (CoACD) for physics simulation"
)

ExclusionReason = Literal[
    "category",
    "dev_subset",
    "fetch_failed",
    "mesh_invalid",
    "too_wide",
    "too_flat",
    "too_long",
    "too_heavy",
    "decomposition_failed",
    "pose_mismatch",
    "load_failed",
    "rest_unstable",
    "rest_penetration",
]

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^\w[\w.\-]*$")  # GSO names are word characters (two are non-ASCII)
_DOMAIN = b"rtt-heldout-real-v1\x00"
_CAL_EPS = 1e-12


# --------------------------------------------------------------------------- source and fetching


class Fetcher(Protocol):
    """Anything that maps an ``https://`` URL to the bytes it serves (picklable for workers)."""

    def __call__(self, url: str) -> bytes: ...


@dataclass(frozen=True)
class HttpFetcher:
    """HTTPS GET with a timeout, a size cap and retries on transient errors (5xx, 429, network)."""

    timeout_s: float = 60.0
    retries: int = 3
    backoff_s: float = 2.0
    max_bytes: int = 64 << 20  # guards against runaway downloads; a scan is ~1 MB

    def __call__(self, url: str) -> bytes:
        if not url.startswith("https://"):
            raise ValueError(f"refusing a non-HTTPS URL: {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "rtt-gso/1"})
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    final = response.geturl()
                    if not final.startswith("https://"):  # urllib follows redirects to http
                        raise ValueError(f"GET {url} was redirected to a non-HTTPS URL")
                    data = response.read(self.max_bytes + 1)
                if len(data) > self.max_bytes:
                    raise ValueError(f"GET {url}: response larger than {self.max_bytes} bytes")
                return data
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(self.backoff_s * (2**attempt))
        raise ConnectionError(f"GET {url} failed after {self.retries + 1} attempts: {last}")


@dataclass(frozen=True)
class LocalMirror:
    """Serve the source from a local directory instead of the network (offline runs, tests).

    ``root`` mirrors the repository (``models/<name>/model.obj``, e.g. a sparse checkout)
    and holds ``models_tree.json``, the GitHub API listing of the pinned ``models/`` tree.
    The listing is still checked against the pinned tree id.
    """

    root: Path

    def __call__(self, url: str) -> bytes:
        parsed = urllib.parse.urlparse(url)
        root = Path(self.root).resolve()
        if parsed.netloc == "api.github.com":
            path = root / "models_tree.json"
        elif parsed.netloc == "raw.githubusercontent.com":
            parts = urllib.parse.unquote(parsed.path).split("/", 4)  # '', owner, repo, commit, path
            if len(parts) != 5:
                raise ValueError(f"unexpected raw URL {url}")
            path = (root / parts[4]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"{url} points outside the mirror")
        else:
            raise ValueError(f"the mirror does not serve {parsed.netloc}")
        return path.read_bytes()


@dataclass(frozen=True)
class GsoSource:
    """The pinned scanned-objects repository: commit, ``models/`` tree id and file layout."""

    repo: str
    commit: str
    models_tree: str
    n_models: int
    mesh_file: str = "model.obj"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", self.repo):
            raise ValueError(f"source.repo must be owner/name, got {self.repo!r}")
        if not _HEX40.match(self.commit) or not _HEX40.match(self.models_tree):
            raise ValueError("source.commit and source.models_tree must be full 40-hex git ids")
        if self.n_models < 1 or not _NAME.match(self.mesh_file):
            raise ValueError("source.n_models must be >= 1 and source.mesh_file a plain file name")

    def raw_url(self, name: str, file: str | None = None) -> str:
        """Commit-pinned (immutable) download URL of one object's file."""
        _check_name(name)
        path = urllib.parse.quote(f"models/{name}/{file or self.mesh_file}")
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{path}"

    def page_url(self, name: str) -> str:
        """Commit-pinned browse URL of one object's directory (for attribution)."""
        _check_name(name)
        return f"https://github.com/{self.repo}/tree/{self.commit}/models/{urllib.parse.quote(name)}"

    @property
    def tree_api_url(self) -> str:
        return f"https://api.github.com/repos/{self.repo}/git/trees/{self.models_tree}"


def _check_name(name: str) -> None:
    if not _NAME.match(name) or ".." in name:
        raise ValueError(f"not a valid object name: {name!r}")


def git_tree_sha1(entries: Sequence[Mapping[str, str]]) -> str:
    """The git object id of a tree with these entries (``path``, ``mode``, ``type``, ``sha``).

    Git sorts entries by name, comparing a subtree as if its name ended in ``/``, and
    writes a subtree's mode without the leading zero.
    """

    def sort_key(e: Mapping[str, str]) -> bytes:
        return (e["path"] + ("/" if e["type"] == "tree" else "")).encode()

    body = b"".join(
        f"{e['mode'].lstrip('0')} {e['path']}".encode() + b"\0" + bytes.fromhex(e["sha"])
        for e in sorted(entries, key=sort_key)
    )
    return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class GsoCache:
    """Local cache of the pinned source: the verified catalogue, meshes and screening results.

    Layout under ``root / commit[:12]``: ``catalogue.json``, ``models/<name>/model.obj``,
    ``screen/<rules key>/<name>.json`` and ``objects/<rules key>/<name>.npz``.
    """

    def __init__(self, source: GsoSource, root: str | Path | None = None, fetch: Fetcher | None = None):
        self.source = source
        base = root if root is not None else os.environ.get(CACHE_ENV) or DEFAULT_CACHE
        self.root = Path(base).expanduser()
        self.dir = self.root / source.commit[:12]
        self.fetch: Fetcher = fetch if fetch is not None else HttpFetcher()

    def __repr__(self) -> str:
        return f"GsoCache(root={str(self.root)!r}, commit={self.source.commit[:12]})"

    def catalogue(self) -> tuple[str, ...]:
        """Sorted object names at the pinned commit, verified against ``models_tree``."""
        path = self.dir / "catalogue.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("models_tree") == self.source.models_tree:
                return tuple(data["names"])
        raw = json.loads(self.fetch(self.source.tree_api_url))
        if raw.get("truncated"):
            raise ValueError("the models/ tree listing came back truncated")
        entries = raw.get("tree") or []
        actual = git_tree_sha1(entries)
        if actual != self.source.models_tree:
            raise ValueError(
                f"models/ listing does not hash to the pinned tree ({actual} != {self.source.models_tree})"
            )
        names = sorted(e["path"] for e in entries if e["type"] == "tree")
        if len(names) != self.source.n_models:
            raise ValueError(f"expected {self.source.n_models} objects, the pinned tree lists {len(names)}")
        for name in names:
            _check_name(name)
        _write_atomic(
            path,
            json.dumps({"models_tree": self.source.models_tree, "names": names}, indent=1).encode(),
        )
        return tuple(names)

    def mesh_path(self, name: str) -> Path:
        """Local copy of an object's mesh file, fetched on first use."""
        _check_name(name)
        path = self.dir / "models" / name / self.source.mesh_file
        if not path.is_file():
            data = self.fetch(self.source.raw_url(name))
            if not data:
                raise ValueError(f"{name}: empty download")
            _write_atomic(path, data)
        return path

    def texture_path(self, name: str, file: str = TEXTURE_FILE) -> Path:
        """Local copy of an object's texture image (fetched on first use, like the mesh).

        Only visual assets need it; screening and scoring never read textures."""
        _check_name(name)
        if not _NAME.match(file):
            raise ValueError(f"not a plain file name: {file!r}")
        path = self.dir / "models" / name / file
        if not path.is_file():
            data = self.fetch(self.source.raw_url(name, file))
            if not data:
                raise ValueError(f"{name}/{file}: empty download")
            _write_atomic(path, data)
        return path

    def screen_dir(self, key: str) -> Path:
        return self.dir / "screen" / key

    def objects_dir(self, key: str) -> Path:
        return self.dir / "objects" / key


# --------------------------------------------------------------------------- configuration


def name_tokens(name: str) -> tuple[str, ...]:
    """Lower-case alphanumeric tokens of an object name (``CITY_TAXI_2`` -> city, taxi, 2)."""
    return tuple(t for t in re.split(r"[^0-9a-z]+", name.lower()) if t)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Denylist:
    """Category rule: plain name tokens, plus individual objects by the SHA-256 of their name.

    ``name_sha256`` holds exact, full object names, never words: a hash of a dictionary
    word is reversed by hashing a word list, so it hides nothing. A hash of a full name
    still resolves against the public catalogue, so it only keeps the name itself out of
    the repository and its outputs.
    """

    tokens: frozenset[str] = frozenset()
    name_sha256: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        bad = [t for t in self.tokens if name_tokens(t) != (t,)]
        if bad:
            raise ValueError(f"denylist tokens must be single lower-case alphanumeric tokens: {sorted(bad)}")
        if any(not _HEX64.match(h) for h in self.name_sha256):
            raise ValueError("denylist.name_sha256 entries must be 64 lower-case hex digits")

    def match(self, name: str) -> tuple[Literal["token", "name_hash"], str | None] | None:
        """``("token", token)`` or ``("name_hash", None)`` for a denylisted name, else ``None``."""
        for tok in name_tokens(name):
            if tok in self.tokens:
                return ("token", tok)
        if _sha256_hex(name) in self.name_sha256:
            return ("name_hash", None)
        return None


@dataclass(frozen=True)
class RestRules:
    """The rest test: the object must stay put on the table for ``duration_s``."""

    duration_s: float = 2.0
    max_shift_mm: float = 2.0
    max_drop_mm: float = 2.0
    max_tilt_deg: float = 2.0
    max_speed_mm_s: float = 5.0


@dataclass(frozen=True)
class DecompositionRules:
    """Convex decomposition settings passed to :func:`rtt.sim.mesh.mesh_object_from_trimesh`.

    ``min_parts``, ``retry_threshold`` and ``coacd_version`` are the v2 rule (heldout_real_v2:
    CoACD 0.02, seed 0, at least 6 pieces else once more at 0.01, CoACD pinned). At their
    defaults they are left out of every digest and cache key, so v1 keys are unchanged.
    """

    method: Literal["coacd", "hull"] = "coacd"
    threshold: float = 0.05
    max_parts: int = 16
    max_part_vertices: int = 48
    seed: int = 0
    min_parts: int = 1
    retry_threshold: float | None = None
    coacd_version: str | None = None

    def __post_init__(self) -> None:
        if self.method not in ("coacd", "hull"):
            raise ValueError(f"decomposition.method must be coacd or hull, got {self.method!r}")
        if self.min_parts < 1 or self.min_parts > self.max_parts:
            raise ValueError("decomposition.min_parts must be in [1, max_parts]")
        if self.retry_threshold is not None and not 0 < self.retry_threshold < self.threshold:
            raise ValueError("decomposition.retry_threshold must be positive and below threshold")
        if self.coacd_version is not None and not re.fullmatch(r"\d+(\.\d+)*", self.coacd_version):
            raise ValueError(
                f"decomposition.coacd_version must be a version like 1.0.14, got {self.coacd_version!r}"
            )

    def payload(self) -> dict[str, Any]:
        """The settings for digests: v2-only fields only when they are not at their default."""
        out = asdict(self)
        for key, default in _V2_DECOMPOSITION_DEFAULTS.items():
            if out[key] == default:
                del out[key]
        return out


_V2_DECOMPOSITION_DEFAULTS: dict[str, Any] = {"min_parts": 1, "retry_threshold": None, "coacd_version": None}


def _rules_payload(rules: ScreenRules) -> dict[str, Any]:
    out = asdict(rules)
    out["decomposition"] = rules.decomposition.payload()
    return out


@dataclass(frozen=True)
class ScreenRules:
    """Per-object acceptance rules (lengths in millimetres, as in the config file)."""

    max_grip_width_mm: float = 79.0
    min_height_mm: float = 6.0
    max_extent_mm: float = 250.0
    min_pose_probability: float = 0.05
    density: float | None = None
    mass_safety_factor: float = 1.5
    rest: RestRules = field(default_factory=RestRules)
    decomposition: DecompositionRules = field(default_factory=DecompositionRules)

    def __post_init__(self) -> None:
        if not 0 < self.min_height_mm < self.max_extent_mm or self.max_grip_width_mm <= 0:
            raise ValueError("need 0 < min_height_mm < max_extent_mm and max_grip_width_mm > 0")
        if not 0 <= self.min_pose_probability < 1:
            raise ValueError("min_pose_probability must be in [0, 1)")
        if self.mass_safety_factor < 1 or (self.density is not None and self.density <= 0):
            raise ValueError("mass_safety_factor must be >= 1 and density positive")

    def object_density(self, task: TaskConfig) -> float:
        """kg/m^3: the rule's own density, or the task's default object density."""
        return self.density if self.density is not None else task.physics.object_density

    def mass_cap_kg(self, task: TaskConfig) -> float:
        """Heaviest part a friction grip holds at the worst jittered friction, with a margin.

        Two pads squeeze with ``F / 2`` each, so friction can carry ``mu * F``; jitter
        lowers mu by up to ``jitter.mu_frac``. Cap = ``mu_min * F / (g * safety)``.
        """
        mu_min = task.physics.friction * (1.0 - task.jitter.mu_frac)
        return mu_min * task.gripper.grip_force / (9.81 * self.mass_safety_factor)

    def key(self, task: TaskConfig) -> str:
        """16-hex fingerprint of everything that decides a candidate's outcome (cache key).

        Covers the rules, the task, :data:`SCREEN_VERSION` and the versions of the
        libraries that compute poses, pieces and physics.
        """
        payload = {
            "version": SCREEN_VERSION,
            "rules": _rules_payload(self),
            "task": task_digest(task),
            "libraries": _library_versions(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def _library_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for pkg in ("mujoco", "trimesh", "coacd", "numpy", "scipy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


@dataclass(frozen=True)
class DrawConfig:
    """Draw sizes and seeds.

    ``exclude_names`` lists every object that was simulated during development (any dev
    draw, including ones superseded by a rule change). The scored draw never includes
    them, and it refuses to run while the current dev draw has an object missing from
    this list, so "no object seen during development is scored" holds by construction.
    """

    n_objects: int = 64
    dev_seed: int = 20261016
    n_dev: int = 10
    exclude_dev: bool = True
    seeds_per_condition: int = 2
    exclude_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_objects < 1 or self.n_dev < 1 or self.seeds_per_condition < 1:
            raise ValueError("n_objects, n_dev and seeds_per_condition must be >= 1")
        for name in self.exclude_names:
            _check_name(name)
        if len(set(self.exclude_names)) != len(self.exclude_names):
            raise ValueError("draw.exclude_names has duplicates")


@dataclass(frozen=True)
class PlacementRanges:
    """Per-object placement draw: nominal x, y in +-xy, yaw over a full turn, plus a pose error."""

    xy_mm: float = 80.0
    err_xy_mm: float = 3.0
    err_yaw_deg: float = 8.0


@dataclass(frozen=True)
class GsoConfig:
    """A parsed ``heldout_real`` suite file."""

    name: str
    source: GsoSource
    draw: DrawConfig
    placement: PlacementRanges
    rules: ScreenRules
    denylist: Denylist
    path: Path | None = field(default=None, compare=False)

    def describe(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "source": asdict(self.source),
            "draw": asdict(self.draw),
            "placement": asdict(self.placement),
            "rules": _rules_payload(self.rules),
            "denylist": {
                "tokens": sorted(self.denylist.tokens),
                "name_sha256": sorted(self.denylist.name_sha256),
            },
        }
        return out

    def digest(self) -> str:
        """SHA-256 of the parsed configuration (independent of comments and key order)."""
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True).encode()).hexdigest()


def _section(raw: Mapping[str, Any], key: str, allowed: set[str], where: str) -> dict[str, Any]:
    value = raw.get(key) or {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}: '{key}' must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys in '{key}': {sorted(unknown)}")
    return dict(value)


def gso_config_from_dict(raw: Mapping[str, Any], where: str = "config") -> GsoConfig:
    """Validate and parse a suite mapping (see ``configs/suites/heldout_real_v1.yaml``)."""
    unknown = set(raw) - {"name", "source", "draw", "placement", "rules", "denylist"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    src = _section(raw, "source", {"repo", "commit", "models_tree", "n_models", "mesh_file"}, where)
    draw = _section(raw, "draw", set(DrawConfig.__dataclass_fields__), where)
    place = _section(raw, "placement", set(PlacementRanges.__dataclass_fields__), where)
    rules = _section(raw, "rules", set(ScreenRules.__dataclass_fields__), where)
    deny = _section(raw, "denylist", {"tokens", "name_sha256"}, where)
    if "exclude_names" in draw:
        names = draw["exclude_names"] or ()
        if isinstance(names, str) or not isinstance(names, Sequence):
            raise TypeError(f"{where}: 'draw.exclude_names' must be a list of object names")
        draw["exclude_names"] = tuple(sorted(str(n) for n in names))
    rest = rules.pop("rest", None) or {}
    decomp = rules.pop("decomposition", None) or {}
    for label, sub, cls in (("rest", rest, RestRules), ("decomposition", decomp, DecompositionRules)):
        extra = set(sub) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError(f"{where}: unknown keys in 'rules.{label}': {sorted(extra)}")
    if decomp.get("coacd_version") is not None:
        decomp["coacd_version"] = str(decomp["coacd_version"])
    return GsoConfig(
        name=str(raw.get("name", "heldout_real")),
        source=GsoSource(
            repo=str(src["repo"]),
            commit=str(src["commit"]),
            models_tree=str(src["models_tree"]),
            n_models=int(src["n_models"]),
            mesh_file=str(src.get("mesh_file", "model.obj")),
        ),
        draw=DrawConfig(**draw),
        placement=PlacementRanges(**{k: float(v) for k, v in place.items()}),
        rules=ScreenRules(**rules, rest=RestRules(**rest), decomposition=DecompositionRules(**decomp)),
        denylist=Denylist(
            tokens=frozenset(str(t) for t in deny.get("tokens") or ()),
            name_sha256=frozenset(str(h).lower() for h in deny.get("name_sha256") or ()),
        ),
    )


def load_gso_config(path: str | Path = DEFAULT_CONFIG) -> GsoConfig:
    """Load a ``heldout_real`` suite file."""
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        cfg = gso_config_from_dict(yaml.safe_load(fh) or {}, where=str(p))
    return GsoConfig(cfg.name, cfg.source, cfg.draw, cfg.placement, cfg.rules, cfg.denylist, path=p)


# --------------------------------------------------------------------------- seeds


def dev_seed_material(dev_seed: int) -> bytes:
    """Seed material of the public dev (smoke) draw."""
    return f"dev:{int(dev_seed)}".encode()


def secret_seed_material(secret: str) -> bytes:
    """Seed material of the scored draw: the held-out secret (64 hex digits).

    It is domain separated (:data:`_DOMAIN`) from the synthetic held-out suite drawn from
    the same secret, so one committed secret serves both without correlating them.
    """
    value = secret.strip()
    if not _HEX64.match(value):
        raise ValueError("the held-out secret must be 64 lower-case hex digits")
    return b"secret:" + value.encode()


def _rng(material: bytes, purpose: str) -> np.random.Generator:
    digest = hashlib.sha256(_DOMAIN + purpose.encode() + b"\x00" + material).digest()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(int.from_bytes(digest, "big"))))


def draw_order(names: Sequence[str], material: bytes) -> list[str]:
    """The seeded examination order of ``names`` (sorted first, so input order is irrelevant)."""
    ordered = sorted(names)
    perm = _rng(material, "order").permutation(len(ordered))
    return [ordered[int(i)] for i in perm]


# --------------------------------------------------------------------------- geometry screening


def geometry_mesh(data: bytes) -> trimesh.Trimesh:
    """The scan as plain geometry: OBJ bytes -> cleaned triangle mesh, texture dropped.

    Dropping UVs lets coincident seam vertices merge, so a closed scan becomes watertight
    and its mass properties come from the solid rather than from overlapping convex
    pieces. The cleaning repeats the intake path's (``rtt.sim.mesh``), which is idempotent,
    so stable-pose indices computed here match the ones it computes.
    """
    loaded = trimesh.load(io.BytesIO(data), file_type="obj", force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) < 4:
        raise ValueError("no triangle mesh in the file")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64),
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if mesh.is_watertight:
        mesh.fix_normals()
    if len(mesh.faces) < 4 or not np.isfinite(mesh.vertices).all():
        raise ValueError("degenerate mesh")
    if float(mesh.extents.max()) > objs.MAX_OBJECT_SIZE:
        raise ValueError(f"mesh is {mesh.extents.max():.3g} m across; expected metres")
    return mesh


def _encloses_volume(mesh: trimesh.Trimesh) -> bool:
    return bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 1e-9)


@dataclass(frozen=True)
class PoseFit:
    """One stable resting pose and the grasp-frame extents it gives (metres).

    ``width`` is the narrowest horizontal extent (the jaw axis), ``length`` the extent
    perpendicular to it and ``height`` the vertical extent.
    """

    index: int
    probability: float
    width: float
    length: float
    height: float

    def failures(self, rules: ScreenRules) -> tuple[ExclusionReason, ...]:
        out: list[ExclusionReason] = []
        if 1000.0 * self.width > rules.max_grip_width_mm:
            out.append("too_wide")
        if 1000.0 * self.height < rules.min_height_mm:
            out.append("too_flat")
        if 1000.0 * max(self.length, self.height) > rules.max_extent_mm:
            out.append("too_long")
        return tuple(out)

    def describe(self) -> str:
        return (
            f"pose {self.index} (p={self.probability:.3f}): width {1000 * self.width:.1f} mm, "
            f"length {1000 * self.length:.1f} mm, height {1000 * self.height:.1f} mm"
        )


def _min_width(points: np.ndarray) -> tuple[float, float]:
    """``(width, length)`` of 2D points: the minimum caliper width and the extent across it."""
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        span = points.max(axis=0) - points.min(axis=0)
        return float(span.min()), float(span.max())
    best = (math.inf, 0.0)
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        norm = float(np.hypot(*edge))
        if norm < _CAL_EPS:
            continue
        u = edge / norm
        n = np.array([-u[1], u[0]])
        width = float(np.ptp(hull @ n))
        if width < best[0]:
            best = (width, float(np.ptp(hull @ u)))
    return best


def pose_fits(mesh: trimesh.Trimesh) -> list[PoseFit]:
    """Grasp-frame extents of every stable pose of ``mesh``, most probable first.

    Pose indices and probabilities are those of :func:`rtt.sim.mesh.stable_poses`, i.e.
    the ``pose_index`` the intake path takes.
    """
    hull_pts = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
    out = []
    for index, (transform, prob) in enumerate(stable_poses(mesh)):
        pts = hull_pts @ transform[:3, :3].T + transform[:3, 3]
        width, length = _min_width(pts[:, :2])
        out.append(PoseFit(index, prob, width, length, float(np.ptp(pts[:, 2]))))
    return out


def choose_pose(
    fits: Sequence[PoseFit], rules: ScreenRules
) -> tuple[PoseFit | None, ExclusionReason | None, str]:
    """The most probable pose (at least ``min_pose_probability``) that passes the fit rules.

    Without one, the reason is that of the eligible pose failing the fewest rules (the more
    probable on ties), and the detail lists its extents.
    """
    eligible = [f for f in fits if f.probability >= rules.min_pose_probability] or list(fits[:1])
    if not eligible:
        return None, "mesh_invalid", "no stable pose"
    for fit in eligible:
        if not fit.failures(rules):
            return fit, None, fit.describe()
    closest = min(eligible, key=lambda f: (len(f.failures(rules)), -f.probability, f.index))
    return None, closest.failures(rules)[0], closest.describe()


# --------------------------------------------------------------------------- rest test


@dataclass(frozen=True)
class RestCheck:
    """Outcome of :func:`rest_test` (mm, degrees)."""

    shift_mm: float
    drop_mm: float
    tilt_deg: float
    speed_mm_s: float
    penetration_mm: float
    gripper_contact: bool
    passed: bool
    reason: ExclusionReason | None

    def describe(self) -> str:
        return (
            f"shift {self.shift_mm:.2f} mm, drop {self.drop_mm:.2f} mm, tilt {self.tilt_deg:.2f} deg, "
            f"speed {self.speed_mm_s:.2f} mm/s, penetration {self.penetration_mm:.2f} mm"
            + (", touched the gripper" if self.gripper_contact else "")
        )


@functools.lru_cache(maxsize=1)
def _rest_finger() -> FingerGeometry:
    from rtt.design.generate import generate_finger
    from rtt.design.schema import load_entrants

    designs = {d.name: d for d in load_entrants(REPO_ROOT / "configs" / "entrants.yaml")}
    return generate_finger(designs["straight"])


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    dot = min(1.0, abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)))))
    return math.degrees(2.0 * math.acos(dot))


def rest_test(
    obj: objs.AnyObject,
    task: TaskConfig | None = None,
    rules: RestRules | None = None,
    finger: FingerGeometry | None = None,
) -> RestCheck:
    """Rest ``obj`` on the table in the task's scene and check that it stays put.

    The object starts in its resting pose at the table centre (no jitter, no pose error)
    with the gripper parked, open, at the approach height above it, exactly as in an
    episode's settle phase, and runs for ``rules.duration_s``. It passes when its centre
    moves at most ``max_shift_mm`` sideways and ``max_drop_mm`` vertically, it turns at most
    ``max_tilt_deg``, it ends slower than ``max_speed_mm_s``, it sinks into the table no
    deeper than the task's penetration limit, and it never touches the gripper. Building
    the scene also checks that every convex piece compiles in MuJoCo.
    """
    from rtt.sim.controller import GraspController, plan_grasp
    from rtt.sim.scene import GROUP_ENV, GROUP_OBJECT, build_model

    task = task or default_task()
    rules = rules or RestRules()
    finger = finger or _rest_finger()
    model, ids = build_model(finger, obj, task)
    data = mujoco.MjData(model)
    placement = objs.Placement()
    plan = plan_grasp(finger, obj, placement, 0, task)
    data.qpos[ids.palm_qpos] = plan.palm_start
    data.qpos[ids.finger_qpos] = 0.0
    pos0, quat0 = objs.rest_pose(obj, placement, task.physics.table_top_z)
    q = ids.object_qpos
    data.qpos[q : q + 3] = pos0
    data.qpos[q + 3 : q + 7] = quat0
    data.ctrl[:] = GraspController(plan).command(0.0)
    mujoco.mj_forward(model, data)

    groups = ids.geom_group
    steps = max(1, round(rules.duration_s / model.opt.timestep))
    penetration = 0.0
    touched = False
    for _ in range(steps):
        mujoco.mj_step(model, data)
        for i in range(data.ncon):
            con = data.contact[i]
            a, b = sorted((int(groups[con.geom[0]]), int(groups[con.geom[1]])))
            if b != GROUP_OBJECT:
                continue
            if a == GROUP_ENV:
                penetration = max(penetration, -float(con.dist))
            elif a != GROUP_OBJECT:
                touched = True
    pos, quat = data.qpos[q : q + 3].copy(), data.qpos[q + 3 : q + 7].copy()
    d = ids.object_dof
    check = {
        "shift_mm": 1000.0 * float(np.hypot(*(pos[:2] - pos0[:2]))),
        "drop_mm": 1000.0 * abs(float(pos[2] - pos0[2])),
        "tilt_deg": _quat_angle_deg(quat0, quat),
        "speed_mm_s": 1000.0 * float(np.linalg.norm(data.qvel[d : d + 3])),
        "penetration_mm": 1000.0 * penetration,
    }
    reason: ExclusionReason | None = None
    if not all(math.isfinite(v) for v in check.values()):
        reason = "rest_unstable"
    elif check["penetration_mm"] > 1000.0 * task.score.penetration_max:
        reason = "rest_penetration"
    elif (
        touched
        or check["shift_mm"] > rules.max_shift_mm
        or check["drop_mm"] > rules.max_drop_mm
        or check["tilt_deg"] > rules.max_tilt_deg
        or check["speed_mm_s"] > rules.max_speed_mm_s
    ):
        reason = "rest_unstable"
    return RestCheck(**check, gripper_contact=touched, passed=reason is None, reason=reason)


# --------------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class CandidateResult:
    """Everything the screen found out about one object (lengths in metres, mass in kg).

    ``seconds`` is the only field that is not deterministic; it is left out of digests.
    """

    name: str
    accepted: bool
    reason: ExclusionReason | None
    detail: str
    file_sha256: str | None = None
    watertight: bool | None = None
    n_poses: int | None = None
    pose_index: int | None = None
    pose_probability: float | None = None
    size_m: tuple[float, float, float] | None = None
    volume_m3: float | None = None
    mass_kg: float | None = None
    n_parts: int | None = None
    geometry_sha256: str | None = None
    rest: dict[str, Any] | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["size_m"] = list(self.size_m) if self.size_m is not None else None
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CandidateResult:
        data = dict(raw)
        if data.get("size_m") is not None:
            data["size_m"] = tuple(float(v) for v in data["size_m"])
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def stable_dict(self) -> dict[str, Any]:
        """:meth:`to_dict` without the timing (for digests)."""
        out = self.to_dict()
        out.pop("seconds")
        return out


def _reject(name: str, reason: ExclusionReason, detail: str, t0: float, **facts: Any) -> CandidateResult:
    return CandidateResult(name, False, reason, detail, seconds=time.perf_counter() - t0, **facts)


def evaluate_candidate(
    name: str, cache: GsoCache, rules: ScreenRules, task: TaskConfig | None = None
) -> CandidateResult:
    """Fetch, screen, decompose and rest-test one object (cached on disk by rules key).

    An accepted object's decomposed geometry is saved as
    ``cache.objects_dir(key) / f"{name}.npz"``.
    """
    task = task or default_task()
    key = rules.key(task)
    cached = cache.screen_dir(key) / f"{name}.json"
    npz = cache.objects_dir(key) / f"{name}.npz"
    if cached.is_file():
        result = CandidateResult.from_dict(json.loads(cached.read_text(encoding="utf-8")))
        if not result.accepted or npz.is_file():
            return result
    result = _evaluate_uncached(name, cache, rules, task, npz)
    _write_atomic(cached, json.dumps(result.to_dict(), indent=1).encode())
    return result


def _evaluate_uncached(
    name: str, cache: GsoCache, rules: ScreenRules, task: TaskConfig, npz: Path
) -> CandidateResult:
    t0 = time.perf_counter()
    try:
        raw = cache.mesh_path(name).read_bytes()
    except (FileNotFoundError, urllib.error.HTTPError) as exc:
        # Only a file missing at the pinned commit is an exclusion. Any other failure
        # (network, rate limit) propagates: caching it would make the draw depend on luck.
        if isinstance(exc, urllib.error.HTTPError) and exc.code != 404:
            raise
        return _reject(name, "fetch_failed", f"{type(exc).__name__}: {exc}", t0)
    facts: dict[str, Any] = {"file_sha256": hashlib.sha256(raw).hexdigest()}
    try:
        mesh = geometry_mesh(raw)
        fits = pose_fits(mesh)
    except Exception as exc:  # noqa: BLE001
        return _reject(name, "mesh_invalid", f"{type(exc).__name__}: {exc}", t0, **facts)
    facts.update(watertight=bool(mesh.is_watertight), n_poses=len(fits))
    fit, reason, detail = choose_pose(fits, rules)
    if fit is None:
        return _reject(name, reason or "mesh_invalid", detail, t0, **facts)
    facts.update(pose_index=fit.index, pose_probability=fit.probability)
    density = rules.object_density(task)
    cap = rules.mass_cap_kg(task)
    if _encloses_volume(mesh) and float(mesh.volume) * density > cap:
        mass = float(mesh.volume) * density
        return _reject(name, "too_heavy", f"{mass:.3f} kg > cap {cap:.3f} kg", t0, mass_kg=mass, **facts)

    d = rules.decomposition
    try:
        obj = mesh_object_from_trimesh(
            mesh,
            name=name,
            pose_index=fit.index,
            decomposition=d.method,
            threshold=d.threshold,
            max_parts=d.max_parts,
            max_part_vertices=d.max_part_vertices,
            seed=d.seed,
            source_sha256=facts["file_sha256"],
            min_parts=d.min_parts,
            retry_threshold=d.retry_threshold,
            pin_coacd=d.coacd_version,
        )
    except MemoryError:
        raise  # a resource limit, not a property of the object
    except Exception as exc:  # noqa: BLE001
        return _reject(name, "decomposition_failed", f"{type(exc).__name__}: {exc}", t0, **facts)
    mass = obj.volume * density
    facts.update(
        size_m=obj.size,
        volume_m3=obj.volume,
        mass_kg=mass,
        n_parts=len(obj.parts),
        geometry_sha256=obj.geometry_sha256,
    )
    if abs(obj.pose_probability - fit.probability) > 1e-9:
        return _reject(name, "pose_mismatch", f"intake pose p={obj.pose_probability:.6f}", t0, **facts)
    built = PoseFit(fit.index, fit.probability, obj.size[1], obj.size[0], obj.size[2])
    if built.failures(rules):
        return _reject(
            name, built.failures(rules)[0], "after decomposition: " + built.describe(), t0, **facts
        )
    if mass > cap:
        return _reject(name, "too_heavy", f"{mass:.3f} kg > cap {cap:.3f} kg", t0, **facts)
    try:
        rest = rest_test(obj, task, rules.rest)
    except Exception as exc:  # noqa: BLE001 - MuJoCo refuses a piece: the object cannot be simulated
        return _reject(name, "load_failed", f"{type(exc).__name__}: {exc}", t0, **facts)
    facts["rest"] = {k: v for k, v in asdict(rest).items() if k not in ("passed", "reason")}
    if not rest.passed:
        return _reject(name, rest.reason or "rest_unstable", rest.describe(), t0, **facts)
    save_mesh_object(obj, npz)
    return CandidateResult(name, True, None, fit.describe(), seconds=time.perf_counter() - t0, **facts)


def _evaluate_job(args: tuple[str, GsoCache, ScreenRules, TaskConfig]) -> CandidateResult:
    return evaluate_candidate(*args)


# --------------------------------------------------------------------------- selection


@dataclass(frozen=True)
class Exclusion:
    """An object left out of a draw. ``position`` is its place in the draw order (-1: before it)."""

    position: int
    name: str | None  # None when only the hash may be shown
    name_sha256: str
    reason: ExclusionReason
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Selection:
    """The outcome of one draw: accepted objects in draw order, their conditions, exclusions."""

    name: str
    mode: Literal["dev", "scored"]
    seed_info: dict[str, Any]
    config_digest: str
    rules_key: str
    task_digest: str
    source: dict[str, Any]
    catalogue_size: int
    pool_size: int  # after the category rule and any excluded names
    examined: int
    requested: int  # n asked for; fewer objects means the pool ran out
    objects: tuple[CandidateResult, ...]
    conditions: tuple[dict[str, Any], ...]  # per object: placement and jitter seeds
    exclusions: tuple[Exclusion, ...]
    schema: str = SELECTION_SCHEMA

    def digest(self) -> str:
        """SHA-256 of everything that defines the scored suite (no timings)."""
        payload = {
            "schema": self.schema,
            "name": self.name,
            "mode": self.mode,
            "config_digest": self.config_digest,
            "rules_key": self.rules_key,
            "task_digest": self.task_digest,
            "requested": self.requested,
            "objects": [o.stable_dict() for o in self.objects],
            "conditions": list(self.conditions),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def reason_counts(self) -> dict[str, int]:
        return dict(sorted(collections.Counter(e.reason for e in self.exclusions).items()))

    def to_json(self) -> dict[str, Any]:
        src = GsoSource(**self.source)
        return {
            "schema": self.schema,
            "name": self.name,
            "mode": self.mode,
            "digest": self.digest(),
            "seed": self.seed_info,
            "config_digest": self.config_digest,
            "rules_key": self.rules_key,
            "task_digest": self.task_digest,
            "source": self.source,
            "catalogue_size": self.catalogue_size,
            "pool_size": self.pool_size,
            "examined": self.examined,
            "requested": self.requested,
            "accepted": len(self.objects),
            "exclusion_counts": self.reason_counts(),
            "objects": [{**o.to_dict(), "attribution": attribution(o.name, src)} for o in self.objects],
            "conditions": list(self.conditions),
            "exclusions": [e.to_dict() for e in self.exclusions],
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Selection:
        if raw.get("schema") != SELECTION_SCHEMA:
            raise ValueError(f"not a {SELECTION_SCHEMA} document (schema {raw.get('schema')!r})")
        sel = cls(
            name=raw["name"],
            mode=raw["mode"],
            seed_info=dict(raw["seed"]),
            config_digest=raw["config_digest"],
            rules_key=raw["rules_key"],
            task_digest=raw["task_digest"],
            source=dict(raw["source"]),
            catalogue_size=int(raw["catalogue_size"]),
            pool_size=int(raw["pool_size"]),
            examined=int(raw["examined"]),
            requested=int(raw["requested"]),
            objects=tuple(CandidateResult.from_dict(o) for o in raw["objects"]),
            conditions=tuple(dict(c) for c in raw["conditions"]),
            exclusions=tuple(Exclusion(**e) for e in raw["exclusions"]),
        )
        if raw.get("digest") != sel.digest():  # a missing digest fails too
            raise ValueError("selection digest is missing or does not match its contents")
        if [c.get("object") for c in sel.conditions] != [o.name for o in sel.objects]:
            raise ValueError("selection conditions do not list the accepted objects in draw order")
        for o in sel.objects:
            _check_name(o.name)
        return sel


def _draw_conditions(
    names: Sequence[str], material: bytes, ranges: PlacementRanges, seeds_per_condition: int
) -> tuple[dict[str, Any], ...]:
    """One placement and its jitter seeds per accepted object, in draw order.

    Each object's draw depends only on the seed material and its own name, so the
    placements of the first k objects do not change when more objects are accepted.
    """
    out = []
    for name in names:
        rng = _rng(material, f"placement/{name}")
        xy, err = ranges.xy_mm / 1000.0, ranges.err_xy_mm / 1000.0
        out.append(
            {
                "object": name,
                "x": round(float(rng.uniform(-xy, xy)), 6),
                "y": round(float(rng.uniform(-xy, xy)), 6),
                "yaw": round(float(rng.uniform(-math.pi, math.pi)), 6),
                "err_x": round(float(rng.uniform(-err, err)), 6),
                "err_y": round(float(rng.uniform(-err, err)), 6),
                "err_yaw": round(
                    math.radians(float(rng.uniform(-ranges.err_yaw_deg, ranges.err_yaw_deg))), 6
                ),
                "seeds": [int(s) for s in rng.integers(0, 2**63 - 1, size=seeds_per_condition)],
            }
        )
    return tuple(out)


def select_objects(
    config: GsoConfig,
    material: bytes,
    *,
    mode: Literal["dev", "scored"],
    n: int | None = None,
    cache: GsoCache | None = None,
    task: TaskConfig | None = None,
    exclude: Iterable[str] | Mapping[str, str] = (),
    workers: int = 1,
    seed_info: Mapping[str, Any] | None = None,
    progress: Callable[[int, CandidateResult], None] | None = None,
) -> Selection:
    """Walk the seeded draw order and accept the first ``n`` objects that pass the rules.

    ``n`` defaults to ``config.draw.n_objects`` (``n_dev`` in dev mode). ``exclude`` names
    (objects used in development, for the scored draw) are skipped and recorded with reason
    ``dev_subset``; pass a mapping to give each its own detail. A name that is not in the
    catalogue is refused, so a typo cannot silently exclude nothing. Candidates run in a
    ``spawn`` process pool when ``workers > 1``; the result does not depend on ``workers``,
    because acceptance is decided strictly in draw order. Fewer than ``n`` objects come back
    only when the pool runs out.
    """
    task = task or default_task()
    cache = cache or GsoCache(config.source)
    if n is None:
        n = config.draw.n_dev if mode == "dev" else config.draw.n_objects
    if n < 1:
        raise ValueError("n must be >= 1")
    rules = config.rules
    catalogue = cache.catalogue()
    if isinstance(exclude, Mapping):
        exclude_detail = {str(k): str(v) for k, v in exclude.items()}
    else:
        exclude_detail = {str(k): "drawn for the dev subset" for k in exclude}
    excluded_names = set(exclude_detail)
    unknown = sorted(excluded_names - set(catalogue))
    if unknown:
        raise ValueError(f"excluded names not in the catalogue: {unknown}")
    exclusions: list[Exclusion] = []
    pool: list[str] = []
    for name in catalogue:
        hit = config.denylist.match(name)
        if hit is not None:
            kind, token = hit
            shown = name if kind == "token" else None
            detail = f"name token '{token}'" if kind == "token" else "name listed by hash"
            exclusions.append(Exclusion(-1, shown, _sha256_hex(name), "category", detail))
        elif name in excluded_names:
            exclusions.append(Exclusion(-1, name, _sha256_hex(name), "dev_subset", exclude_detail[name]))
        else:
            pool.append(name)
    order = draw_order(pool, material)

    accepted: list[CandidateResult] = []
    examined = 0

    def consume(position: int, result: CandidateResult) -> None:
        nonlocal examined
        examined = position + 1
        if progress is not None:
            progress(position, result)
        if result.accepted:
            accepted.append(result)
        else:
            assert result.reason is not None
            exclusions.append(
                Exclusion(position, result.name, _sha256_hex(result.name), result.reason, result.detail)
            )

    if workers <= 1:
        for position, name in enumerate(order):
            if len(accepted) >= n:
                break
            consume(position, evaluate_candidate(name, cache, rules, task))
    else:
        window = 2 * workers
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            pending: collections.deque[cf.Future[CandidateResult]] = collections.deque()
            upcoming = iter(enumerate(order))
            try:
                while len(accepted) < n:
                    while len(pending) < window:
                        nxt = next(upcoming, None)
                        if nxt is None:
                            break
                        pending.append(ex.submit(_evaluate_job, (nxt[1], cache, rules, task)))
                    if not pending:
                        break
                    consume(examined, pending.popleft().result())
            finally:
                for fut in pending:
                    fut.cancel()
    names = [o.name for o in accepted]
    return Selection(
        name=config.name,
        mode=mode,
        seed_info=dict(seed_info or {}),
        config_digest=config.digest(),
        rules_key=rules.key(task),
        task_digest=task_digest(task),
        source=asdict(config.source),
        catalogue_size=len(catalogue),
        pool_size=len(pool),
        examined=examined,
        requested=n,
        objects=tuple(accepted),
        conditions=_draw_conditions(names, material, config.placement, config.draw.seeds_per_condition),
        exclusions=tuple(exclusions),
    )


# --------------------------------------------------------------------------- loading and suites


def load_selection_objects(
    selection: Selection, objects_dir: str | Path | None = None, cache: GsoCache | None = None
) -> list[objs.MeshObject]:
    """The accepted objects as :class:`~rtt.sim.objects.MeshObject`, in draw order.

    Each is read from ``objects_dir / f"{name}.npz"`` (an exported selection) or else from
    the cache, and its geometry SHA-256 must match the selection's record.
    """
    if objects_dir is None:
        if cache is None:
            raise ValueError("pass objects_dir or a cache")
        objects_dir = cache.objects_dir(selection.rules_key)
    out = []
    for rec in selection.objects:
        _check_name(rec.name)
        path = Path(objects_dir) / f"{rec.name}.npz"
        _check_pack_size(path)
        obj = read_mesh_object(path)
        if obj.geometry_sha256 != rec.geometry_sha256:
            raise ValueError(f"{rec.name}: geometry does not match the selection record")
        if obj.name != rec.name:
            raise ValueError(f"{rec.name}: the pack file names a different object ({obj.name!r})")
        out.append(obj)
    return out


#: Largest object-pack file accepted, compressed and uncompressed (a decomposed scan is
#: tens of kilobytes: at most 16 pieces of at most 48 vertices).
MAX_PACK_FILE_BYTES = 8 << 20


def _check_pack_size(path: Path) -> None:
    """Refuse an oversized or zip-bomb ``.npz`` before NumPy decompresses it."""
    import zipfile

    if path.stat().st_size > MAX_PACK_FILE_BYTES:
        raise ValueError(f"{path.name}: object pack file larger than {MAX_PACK_FILE_BYTES} bytes")
    with zipfile.ZipFile(path) as zf:
        total = sum(info.file_size for info in zf.infolist())
    if total > MAX_PACK_FILE_BYTES:
        raise ValueError(f"{path.name}: object pack file expands beyond {MAX_PACK_FILE_BYTES} bytes")


def selection_suite(
    selection: Selection, objects: Sequence[objs.MeshObject], *, limit: int | None = None
) -> Suite:
    """A :class:`~rtt.sim.conditions.Suite` with one condition per accepted object.

    Conditions keep draw order; ``limit`` keeps the first ``limit`` objects. Family is
    ``gso``; ``size_label`` is the object name and ``placement_label`` ``random``.
    """
    by_name = {o.name: o for o in objects}
    conds = selection.conditions if limit is None else selection.conditions[:limit]
    conditions = []
    for i, c in enumerate(conds):
        obj = by_name.get(c["object"])
        if obj is None:
            raise ValueError(f"no object loaded for {c['object']!r}")
        placement = objs.Placement(c["x"], c["y"], c["yaw"], c["err_x"], c["err_y"], c["err_yaw"])
        conditions.append(
            Condition(
                condition_id=f"{selection.name}/{i:03d}-{c['object']}",
                family="gso",
                obj=obj,
                placement=placement,
                seeds=tuple(int(s) for s in c["seeds"]),
                size_label=c["object"],
                placement_label="random",
            )
        )
    suffix = "" if selection.mode == "scored" else f"-{selection.mode}"
    return Suite(f"{selection.name}{suffix}", tuple(conditions))


def selection_part_suite(
    selection: Selection,
    objects: Sequence[objs.MeshObject],
    *,
    limit: int | None = None,
    visual_assets: Mapping[str, str] | None = None,
) -> Any:
    """The selection as a v2 :class:`rtt.parts.suite.PartSuite` of scanned parts.

    Same conditions, seeds and order as :func:`selection_suite`; each object becomes a
    :class:`~rtt.parts.types.PartSpec` of family ``gso`` (same pieces and geometry hash,
    yaw symmetry detected from the footprint, density from the screen rules' task).
    ``visual_assets`` maps object names to asset ids of their visual meshes.
    """
    from rtt.parts.library import part_from_mesh_object
    from rtt.parts.suite import PartCondition, PartSuite

    v1 = selection_suite(selection, objects, limit=limit)
    assets = dict(visual_assets or {})
    conditions = []
    for c in v1:
        assert isinstance(c.obj, objs.MeshObject)
        part = part_from_mesh_object(c.obj, visual_asset=assets.get(c.obj.name))
        conditions.append(
            PartCondition(
                condition_id=c.condition_id,
                part=part,
                placement=c.placement,
                seeds=c.seeds,
                placement_label=c.placement_label,
                size_label="scan",
            )
        )
    return PartSuite(v1.name, tuple(conditions))


# --------------------------------------------------------------------------- attribution


def attribution(name: str, source: GsoSource) -> dict[str, str]:
    """The CC BY 4.0 credit for one object: title, source, licence and the changes made."""
    return {
        "object": name,
        "credit": GSO_CREDIT,
        "source_url": source.page_url(name),
        "license": ASSET_LICENSE,
        "license_url": ASSET_LICENSE_URL,
        "xml_license": XML_LICENSE,
        "citation": GSO_CITATION,
        "changes": GSO_CHANGES,
    }


def attribution_markdown(names: Sequence[str], source: GsoSource) -> str:
    """An attribution file for a set of objects (ships with any selection or object pack)."""
    lines = [
        "# Object attribution",
        "",
        f"The objects below are from Google Scanned Objects ({GSO_PAPER_URL}), licensed under",
        f"CC BY 4.0 ({ASSET_LICENSE_URL}), via the MuJoCo models at",
        f"https://github.com/{source.repo} (commit {source.commit}; MJCF files MIT).",
        f"Changes: {GSO_CHANGES}.",
        "",
        f"Citation: {GSO_CITATION}.",
        "",
        "| Object | Source |",
        "|---|---|",
    ]
    lines += [f"| {n} | {source.page_url(n)} |" for n in names]
    return "\n".join(lines) + "\n"
