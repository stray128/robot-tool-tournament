"""Compact episode replays for the web viewer.

A replay is two files sharing a stem:

* ``<stem>.bin``: little-endian float32 array ``[frames x bodies x 7]`` holding
  each body frame's world pose ``(px, py, pz, qx, qy, qz, qw)``, sampled at
  ``fps`` (30). MuJoCo stores quaternions as wxyz; they are converted to the
  xyzw order three.js uses.
* ``<stem>.json``: a header with the layout, the body names, and for every
  body the primitive geoms needed to rebuild it (body-frame poses, xyzw
  quaternions, RGBA), plus the static table and floor. Coordinates are
  MuJoCo's Z-up world; a Y-up viewer rotates its root by -90 degrees about X.

Version 2 (``replay_v2.schema.json`` next to this module) keeps the pose file unchanged
and draws bodies from content-addressed GLB assets (``assets`` plus ``bodies[].visuals``);
the v1 geoms become an optional collision overlay, and ``meta`` and ``provenance`` say what
was simulated and where. Optional per-frame readouts live in ``<stem>.ovl.bin``
(little-endian float16 ``[frames x channels]``). :func:`read_replay` accepts both versions;
the scored recorder still writes version 1, and :func:`upgrade_header` lifts it to version 2.

Box geoms are described by ``half_size``; cylinders by ``radius`` and
``half_height`` (axis = geom local Z); capsules by ``radius`` and
``half_length`` of the cylindrical part (axis = geom local Z); convex mesh
pieces (mesh objects) by flat ``vertices`` (x, y, z triples, geom frame) and
``faces`` (vertex-index triples, counter-clockwise seen from outside).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from rtt.sim.scene import REPLAY_BODIES, SceneIds

FORMAT = "rtt-replay"
VERSION_V1 = 1
VERSION = 2  # the latest header version; the scored recorder writes VERSION_V1
SUPPORTED_VERSIONS = (VERSION_V1, VERSION)
POSE_FIELDS = ("px", "py", "pz", "qx", "qy", "qz", "qw")
OVERLAY_CHANNELS = ("offset_mm", "seat_short_mm", "hold_s", "n_contacts")
SCHEMA_PATH = Path(__file__).with_name("replay_v2.schema.json")
_FIXED = {
    "format": FORMAT,
    "dtype": "float32",
    "byte_order": "little",
    "layout": ["frames", "bodies", "pose"],
    "pose_fields": list(POSE_FIELDS),
    "up_axis": "z",
    "units": "m",
}
_V2_REQUIRED = (
    *_FIXED,
    "version",
    "fps",
    "n_frames",
    "n_bodies",
    "bin",
    "assets",
    "bodies",
    "meta",
    "provenance",
)
_V2_KEYS = frozenset((*_V2_REQUIRED, "static", "static_visuals", "overlays"))
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CODE_VERSION = re.compile(r"^[0-9a-f]{7,40}(\+dirty)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v)


# per-field value rules of the schema's meta and provenance blocks: (key, test, what it must be)
_META_RULES: tuple[tuple[str, Callable[[Any], bool], str], ...] = (
    ("design_space", lambda v: isinstance(v, str) and bool(_SLUG.match(v)), "a slug"),
    ("design_id", _nonempty_str, "a non-empty string"),
    (
        "geometry_id",
        lambda v: v is None or (isinstance(v, str) and bool(_SHA256.match(v))),
        "a sha256 or null",
    ),
    ("part_id", lambda v: isinstance(v, str) and bool(_SLUG.match(v)), "a slug"),
)
_PROVENANCE_RULES: tuple[tuple[str, Callable[[Any], bool], str], ...] = (
    ("engine_version", lambda v: isinstance(v, str) and bool(_SEMVER.match(v)), "a x.y.z version"),
    ("run_id", lambda v: isinstance(v, str) and bool(_SLUG.match(v)), "a slug"),
    (
        "code_version",
        lambda v: isinstance(v, str) and bool(_CODE_VERSION.match(v)),
        "a git commit (optionally +dirty)",
    ),
    ("product_mode", lambda v: isinstance(v, bool), "true or false"),
    ("episode", _nonempty_str, "a non-empty string"),
    ("suite", lambda v: isinstance(v, str) and bool(_SLUG.match(v)), "a slug"),
    ("condition", _nonempty_str, "a non-empty string"),
    ("seed", lambda v: _is_int(v) and v >= 0, "an integer >= 0"),
    ("task_sha256", lambda v: isinstance(v, str) and bool(_SHA256.match(v)), "a sha256"),
    ("record_version", lambda v: _is_int(v) and v >= 1, "an integer >= 1"),
    (
        "created_utc",
        lambda v: isinstance(v, str) and bool(_UTC.match(v)),
        "a UTC time like 2026-10-16T12:00:00Z",
    ),
)
# provenance keys the schema allows (it sets additionalProperties: false there)
PROVENANCE_KEYS = (
    "source",
    "engine",
    "engine_version",
    "run_id",
    "code_version",
    "product_mode",
    "episode",
    "suite",
    "condition",
    "seed",
    "task_sha256",
    "record_version",
    "created_utc",
)


@dataclass
class Replay:
    """Pose frames plus the JSON header describing how to draw them."""

    frames: np.ndarray  # float32, (n_frames, n_bodies, 7)
    header: dict[str, Any]
    overlays: np.ndarray | None = None  # float16, (n_frames, n_channels), version 2 only

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def version(self) -> int:
        return int(self.header.get("version", VERSION_V1))

    @property
    def body_names(self) -> list[str]:
        return [b["name"] for b in self.header["bodies"]]


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Reorder quaternion components along the last axis from wxyz to xyzw."""
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _round(values: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values]


def _geom_entry(model: mujoco.MjModel, g: int) -> dict[str, Any]:
    gtype = mujoco.mjtGeom(model.geom_type[g])
    size = model.geom_size[g]
    entry: dict[str, Any] = {"name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"}
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        entry.update(type="box", half_size=_round(size))
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        entry.update(type="cylinder", radius=round(float(size[0]), 6), half_height=round(float(size[1]), 6))
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        entry.update(type="capsule", radius=round(float(size[0]), 6), half_length=round(float(size[1]), 6))
    elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[g])
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        entry.update(
            type="mesh",
            vertices=_round(model.mesh_vert[va : va + vn].ravel()),
            faces=[int(i) for i in model.mesh_face[fa : fa + fn].ravel()],
        )
    elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        entry.update(type="plane", half_size=_round(size[:2]))
    else:
        raise ValueError(f"replay export does not support geom type {gtype.name}")
    entry.update(
        pos=_round(model.geom_pos[g]),
        quat_xyzw=_round(wxyz_to_xyzw(model.geom_quat[g])),
        rgba=_round(model.geom_rgba[g], 3),
    )
    return entry


def scene_header(model: mujoco.MjModel, ids: SceneIds, fps: int) -> dict[str, Any]:
    """Static description of the scene for a replay (no frame count yet)."""
    bodies = []
    for name, body_id in zip(REPLAY_BODIES, ids.replay_bodies, strict=True):
        geoms = [_geom_entry(model, g) for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        bodies.append({"name": name, "geoms": geoms})
    static = [_geom_entry(model, g) for g in range(model.ngeom) if model.geom_bodyid[g] == 0]
    return {
        "format": FORMAT,
        "version": VERSION_V1,
        "fps": fps,
        "dtype": "float32",
        "byte_order": "little",
        "layout": ["frames", "bodies", "pose"],
        "pose_fields": list(POSE_FIELDS),
        "up_axis": "z",
        "units": "m",
        "bodies": bodies,
        "static": static,
    }


class ReplayRecorder:
    """Samples body poses (and full qpos, for video rendering) at a fixed frame rate.

    With ``overlay_channels`` (a subset of :data:`OVERLAY_CHANNELS`) and ``overlay_fn``, each
    recorded frame also stores ``overlay_fn(data)[channel]`` for every channel; the readouts
    end up in :attr:`Replay.overlays` (float16). The scored loop never needs them.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        ids: SceneIds,
        fps: int,
        overlay_channels: Sequence[str] = (),
        overlay_fn: Callable[[mujoco.MjData], Mapping[str, float]] | None = None,
    ) -> None:
        unknown = set(overlay_channels) - set(OVERLAY_CHANNELS)
        if unknown:
            raise ValueError(f"unknown overlay channels {sorted(unknown)}; known: {OVERLAY_CHANNELS}")
        if bool(overlay_channels) != (overlay_fn is not None):
            raise ValueError("overlay_channels and overlay_fn go together")
        self._model = model
        self._ids = ids
        self.fps = fps
        self._next_t = 0.0
        self._poses: list[np.ndarray] = []
        self._qpos: list[np.ndarray] = []
        self._half_step = 0.5 * model.opt.timestep
        self._channels = tuple(overlay_channels)
        self._overlay_fn = overlay_fn
        self._overlay: list[list[float]] = []

    def maybe_record(self, data: mujoco.MjData) -> None:
        """Record a frame if the simulation clock has reached the next frame time."""
        if data.time + self._half_step >= self._next_t:
            body = self._ids.replay_bodies
            pose = np.empty((len(body), 7), dtype=np.float32)
            pose[:, :3] = data.xpos[body]
            pose[:, 3:] = wxyz_to_xyzw(data.xquat[body])
            self._poses.append(pose)
            self._qpos.append(data.qpos.copy())
            if self._overlay_fn is not None:
                values = self._overlay_fn(data)
                self._overlay.append([float(values[c]) for c in self._channels])
            self._next_t = len(self._poses) / self.fps

    def finish(self, meta: dict[str, Any] | None = None) -> tuple[Replay, np.ndarray]:
        """Return the replay and the matching ``(n_frames, nq)`` qpos trajectory."""
        frames = np.stack(self._poses).astype(np.float32, copy=False)
        header = scene_header(self._model, self._ids, self.fps)
        header["n_frames"] = int(frames.shape[0])
        header["n_bodies"] = int(frames.shape[1])
        header["meta"] = meta or {}
        overlays = None
        if self._channels:
            overlays = np.asarray(self._overlay, dtype=np.float16).reshape(
                len(self._overlay), len(self._channels)
            )
            header["overlay_channels"] = list(self._channels)  # consumed by upgrade_header
        return Replay(frames=frames, header=header, overlays=overlays), np.stack(self._qpos)


# --------------------------------------------------------------------------- version 2


def visual(
    asset: str,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    quat_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    scale: Sequence[float] = (1.0, 1.0, 1.0),
) -> dict[str, Any]:
    """One ``bodies[].visuals`` (or ``static_visuals``) entry: an asset placed in a frame."""
    if any(float(c) <= 0 for c in scale):
        raise ValueError("visual scale must be positive; a right jaw is its own asset, never a mirror")
    return {
        "asset": asset,
        "pos": _round(np.asarray(pos, dtype=np.float64)),
        "quat_xyzw": _round(np.asarray(quat_xyzw, dtype=np.float64)),
        "scale": _round(np.asarray(scale, dtype=np.float64)),
    }


def normalise_code_version(sha: str | None) -> str:
    """A git SHA as replay provenance wants it: 12 hex digits, ``+dirty`` for a modified tree.

    Accepts the run manifests' form (:func:`rtt.store.manifest.git_sha`: full SHA with an
    optional ``-dirty`` suffix) and this module's own form.
    """
    if not sha:
        raise ValueError("no code version: run from a git checkout or set RTT_GIT_SHA")
    sha = sha.strip().lower()
    dirty = sha.endswith(("-dirty", "+dirty"))
    core = sha[:-6] if dirty else sha
    if not re.fullmatch(r"[0-9a-f]{7,40}", core):
        raise ValueError(f"{sha!r} is not a git commit (optionally -dirty or +dirty)")
    return core[:12] + ("+dirty" if dirty else "")


def code_version(repo: str | Path | None = None) -> str:
    """The commit of the running code for ``provenance.code_version``.

    Uses :func:`rtt.store.manifest.git_sha` (so ``RTT_GIT_SHA`` wins, as in job images built
    from a clean export without ``.git``), normalised by :func:`normalise_code_version`.
    """
    from rtt.store.manifest import git_sha

    return normalise_code_version(git_sha(repo) if repo is not None else git_sha())


def provenance(*, run_id: str, code_version: str, product_mode: bool = False, **extra: Any) -> dict[str, Any]:
    """A replay v2 ``provenance`` block; the burned-in caption is built from it."""
    out: dict[str, Any] = {
        "source": "simulation",
        "engine": "mujoco",
        "engine_version": mujoco.__version__,
        "run_id": run_id,
        "code_version": code_version,
        "product_mode": bool(product_mode),
    }
    out.update({k: v for k, v in extra.items() if v is not None})
    out.setdefault("created_utc", datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return out


def upgrade_header(
    v1: Mapping[str, Any],
    *,
    visuals: Mapping[str, Sequence[Mapping[str, Any]]],
    assets: Mapping[str, Mapping[str, Any]],
    meta: Mapping[str, Any],
    provenance: Mapping[str, Any],
    keep_geoms: bool = True,
    static_visuals: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Lift a version 1 header (from :class:`ReplayRecorder`) to version 2.

    ``visuals`` maps body names to their visual entries (:func:`visual`); ``assets`` holds
    every referenced asset's manifest entry. With ``keep_geoms`` the v1 geoms stay as each
    body's collision overlay and the v1 static geoms (table, floor) as ``static``. ``meta``
    is merged over the v1 meta and must carry ``design_space``, ``design_id``,
    ``geometry_id`` (or ``None``) and ``part_id``.
    """
    if int(v1.get("version", VERSION_V1)) != VERSION_V1:
        raise ValueError("upgrade_header takes a version 1 header")
    names = [b["name"] for b in v1["bodies"]]
    unknown = set(visuals) - set(names)
    if unknown:
        raise ValueError(f"visuals for unknown bodies {sorted(unknown)}; bodies are {names}")
    bodies = []
    for b in v1["bodies"]:
        body: dict[str, Any] = {"name": b["name"], "visuals": [dict(v) for v in visuals.get(b["name"], ())]}
        if keep_geoms and b.get("geoms"):
            body["geoms"] = json.loads(json.dumps(b["geoms"]))
        bodies.append(body)
    header: dict[str, Any] = {k: v1[k] for k in _FIXED}
    header.update(
        version=VERSION,
        fps=int(v1["fps"]),
        n_frames=int(v1.get("n_frames", 0)),
        n_bodies=len(bodies),
        bin=v1.get("bin", "replay.bin"),
        assets={k: dict(v) for k, v in sorted(assets.items())},
        bodies=bodies,
        meta={**dict(v1.get("meta") or {}), **dict(meta)},
        provenance=dict(provenance),
    )
    if keep_geoms and v1.get("static"):
        header["static"] = json.loads(json.dumps(v1["static"]))
    if static_visuals:
        header["static_visuals"] = [dict(v) for v in static_visuals]
    channels = v1.get("overlay_channels")
    if channels:
        header["overlays"] = {
            "bin": "replay.ovl.bin",
            "dtype": "float16",
            "byte_order": "little",
            "layout": ["frames", "channels"],
            "channels": list(channels),
        }
    return header


def check_header(
    header: Mapping[str, Any], frames: np.ndarray | None = None, overlays: np.ndarray | None = None
) -> list[str]:
    """The v2 rules a writer must keep (required keys, fixed values and the schema's
    cross-field ``$comment`` rules); returns the problems found. The full JSON Schema is
    ``SCHEMA_PATH`` (checked by the tests)."""
    errs: list[str] = []
    missing = [k for k in _V2_REQUIRED if k not in header]
    if missing:
        return [f"missing keys {missing}"]
    extra = set(header) - _V2_KEYS
    if extra:
        errs.append(f"unknown keys {sorted(extra)}")
    for k, v in _FIXED.items():
        if header[k] != v:
            errs.append(f"{k} must be {v!r}")
    if header["version"] != VERSION:
        errs.append(f"version must be {VERSION}")
    if not (_is_int(header["fps"]) and 1 <= header["fps"] <= 1000):
        errs.append("fps must be an integer in [1, 1000]")
    if not (_is_int(header["n_frames"]) and header["n_frames"] >= 1):
        errs.append("n_frames must be an integer >= 1")
    if header["n_bodies"] != len(header["bodies"]):
        errs.append("n_bodies != len(bodies)")
    if frames is not None and tuple(frames.shape) != (
        header["n_frames"],
        header["n_bodies"],
        len(POSE_FIELDS),
    ):
        errs.append(
            f"frames have shape {frames.shape}, the header says {header['n_frames']} x {header['n_bodies']}"
        )
    assets = header["assets"]
    for aid, entry in assets.items():
        if entry.get("uri") != f"assets/{aid}.glb" or not str(entry.get("sha256", "")).startswith(aid):
            errs.append(f"asset {aid} is not content addressed")
        if entry.get("license") != "Apache-2.0" and not entry.get("changes"):
            errs.append(f"asset {aid}: a third-party asset must list its changes")
        if not entry.get("attribution"):
            errs.append(f"asset {aid}: attribution is required")
    visuals = [v for b in header["bodies"] for v in b.get("visuals", [])] + list(
        header.get("static_visuals", [])
    )
    used = set()
    for v in visuals:
        used.add(v.get("asset"))
        if v.get("asset") not in assets:
            errs.append(f"unknown asset {v.get('asset')}")
        if any(float(c) <= 0 for c in v.get("scale", [1, 1, 1])):
            errs.append(f"visual of {v.get('asset')} has a non-positive scale (mirroring is not allowed)")
    unused = set(assets) - used
    if unused:
        errs.append(f"assets not referenced by any visual: {sorted(unused)}")
    for b in header["bodies"]:
        if not _SLUG.match(str(b.get("name", ""))):
            errs.append(f"body name {b.get('name')!r} is not a slug")
        if not b.get("visuals") and not b.get("geoms"):
            errs.append(f"body {b.get('name')} has neither visuals nor geoms")
    meta = header["meta"]
    for k, ok, what in _META_RULES:
        if k not in meta:
            errs.append(f"meta.{k} is required")
        elif not ok(meta[k]):
            errs.append(f"meta.{k} must be {what}, got {meta[k]!r}")
    prov = header["provenance"]
    for k, want in (("source", "simulation"), ("engine", "mujoco")):
        if prov.get(k) != want:
            errs.append(f"provenance.{k} must be {want!r}")
    stray = set(prov) - set(PROVENANCE_KEYS)
    if stray:
        errs.append(f"provenance has keys the schema does not allow: {sorted(stray)} (put them in meta)")
    for k in ("engine_version", "run_id", "code_version", "product_mode"):
        if k not in prov:
            errs.append(f"provenance.{k} is required")
    for k, ok, what in _PROVENANCE_RULES:
        if k in prov and not ok(prov[k]):
            errs.append(f"provenance.{k} must be {what}, got {prov[k]!r}")
    ovl = header.get("overlays")
    if ovl is not None:
        if set(ovl.get("channels", [])) - set(OVERLAY_CHANNELS):
            errs.append("unknown overlay channels")
        if overlays is not None and tuple(overlays.shape) != (header["n_frames"], len(ovl["channels"])):
            errs.append(
                f"overlays have shape {overlays.shape}, expected ({header['n_frames']}, {len(ovl['channels'])})"
            )
    elif overlays is not None:
        errs.append("overlay data without an overlays block")
    return errs


def write_replay(replay: Replay, stem: str | Path) -> tuple[Path, Path]:
    """Write ``<stem>.bin`` and ``<stem>.json`` (and ``<stem>.ovl.bin`` for a v2 replay with
    overlays); returns the pose and header paths. A v2 header is checked first
    (:func:`check_header`) and nothing is written if it breaks a rule; it is written as
    compact JSON, since it is part of every lane's download."""
    stem = Path(stem)
    bin_path, json_path = stem.with_suffix(".bin"), stem.with_suffix(".json")
    frames = np.ascontiguousarray(replay.frames, dtype="<f4")
    header = dict(replay.header, bin=bin_path.name, n_frames=int(frames.shape[0]))
    header.pop("overlay_channels", None)
    ovl_path = None
    if int(header.get("version", VERSION_V1)) == VERSION:
        if "overlays" in header:
            ovl_path = stem.with_name(stem.name + ".ovl.bin")
            header["overlays"] = dict(header["overlays"], bin=ovl_path.name)
        errs = check_header(header, frames, replay.overlays)
        if replay.overlays is not None and ovl_path is None:
            errs.append("overlay data without an overlays block")
        if ovl_path is not None and replay.overlays is None:
            errs.append("an overlays block without overlay data")
        if errs:
            raise ValueError(f"{stem.name}: replay v2 header is invalid: " + "; ".join(errs))
    elif replay.overlays is not None:
        raise ValueError("overlays need a version 2 header (see upgrade_header)")
    stem.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_bytes(frames.tobytes())
    if ovl_path is not None:
        ovl_path.write_bytes(np.ascontiguousarray(replay.overlays, dtype="<f2").tobytes())
    if int(header.get("version", VERSION_V1)) == VERSION:  # compact: the header is part of the web payload
        text = json.dumps(header, separators=(",", ":"), ensure_ascii=False)
    else:  # version 1 headers keep their layout byte for byte (the F1 fallback)
        text = json.dumps(header, indent=1)
    json_path.write_text(text + "\n", encoding="utf-8")
    return bin_path, json_path


def read_replay(stem: str | Path) -> Replay:
    """Load a replay written by :func:`write_replay` (header version 1 or 2)."""
    stem = Path(stem)
    header = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    version = int(header.get("version", VERSION_V1))
    if header.get("format") != FORMAT or version not in SUPPORTED_VERSIONS:
        raise ValueError(f"{stem}: not an rtt replay of a supported version ({SUPPORTED_VERSIONS})")
    raw = np.frombuffer(stem.with_suffix(".bin").read_bytes(), dtype="<f4")
    frames = raw.reshape(header["n_frames"], header["n_bodies"], len(POSE_FIELDS)).astype(np.float32)
    overlays = None
    if version == VERSION and header.get("overlays"):
        ovl = header["overlays"]
        data = np.frombuffer((stem.parent / ovl["bin"]).read_bytes(), dtype="<f2")
        overlays = data.reshape(header["n_frames"], len(ovl["channels"])).astype(np.float16)
    return Replay(frames=frames, header=header, overlays=overlays)
