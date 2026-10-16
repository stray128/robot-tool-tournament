"""Textured visual meshes of scanned objects, and the brand screen for what may appear on screen.

**Visual intake.** A scan's collision pieces come from :func:`rtt.sim.mesh.mesh_object_from_trimesh`
(stable pose, grasp frame, CoACD). :func:`load_scan_visual` loads the same OBJ again with its
texture coordinates kept (so UV seams stay split) and replays the identical rigid steps
(:class:`rtt.sim.mesh.IntakeFrame`), so the visual sits exactly on the simulated object. The
bounds of the textured mesh must match the intake mesh's within ``bounds_tol`` (0.2 mm), or
the load is refused.

UV conventions: :attr:`ScanVisual.uv` is the OBJ convention (origin at the bottom left of the
image). :meth:`ScanVisual.uv_top_left` flips ``v -> 1 - v`` for consumers whose origin is the
top left: MuJoCo ``usertexcoord`` (verified by the render prototype) and raw glTF buffers.
:meth:`ScanVisual.to_trimesh` keeps the OBJ convention, because trimesh flips ``v`` itself
when it exports glTF/GLB; do not flip twice.

**Brand screen** (``configs/parts/visual_screen.yaml``): scans carry printed labels and logos.
Hero and video shots may show a scan *textured* only if the screen lists it with
``brand_visible: false`` (optionally only in the stable poses it was screened in). Unlisted
scans count as branded. Untextured ("clay") renders and the procedural industrial parts
always pass. :func:`check_shot` returns every violation for a shot's object list, so a
render or video pipeline can refuse a frame before it is written.

Attribution: a textured scan is a modified CC BY 4.0 asset; :func:`visual_attribution`
gives its credit with the changes made for the visual, and :func:`scan_asset_fields` the
licence, changes, ``brand_visible`` and pinned source of its asset-manifest entry.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml

from rtt.sim.conditions import REPO_ROOT
from rtt.sim.mesh import IntakeFrame, intake_frame

DEFAULT_SCREEN = REPO_ROOT / "configs" / "parts" / "visual_screen.yaml"
BOUNDS_TOL = 0.2e-3  # m
VISUAL_CHANGES = (
    "re-posed to the same stable resting pose as its physics model; texture kept; "
    "re-encoded for the web (simplified, compressed)"
)


@dataclass(frozen=True, eq=False)
class ScanVisual:
    """A scan's textured visual mesh in the object frame of its collision pieces (metres)."""

    name: str
    vertices: np.ndarray = field(repr=False)
    faces: np.ndarray = field(repr=False)
    uv: np.ndarray | None = field(repr=False)  # (n, 2), OBJ convention (origin bottom left)
    texture: Path | None
    frame: IntakeFrame = field(repr=False)
    bounds_error_mm: float

    @property
    def bounds(self) -> np.ndarray:
        return np.stack([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    def uv_top_left(self) -> np.ndarray:
        """Texture coordinates with ``v`` flipped (MuJoCo ``usertexcoord``, raw glTF)."""
        if self.uv is None:
            raise ValueError(f"{self.name}: the scan has no texture coordinates")
        out = self.uv.copy()
        out[:, 1] = 1.0 - out[:, 1]
        return out

    def to_trimesh(self, textured: bool = True) -> trimesh.Trimesh:
        """A ``trimesh.Trimesh`` (texture image attached when ``textured`` and available).

        UVs stay in the OBJ convention: trimesh's glTF exporter flips ``v`` itself.
        """
        mesh = trimesh.Trimesh(self.vertices.copy(), self.faces.copy(), process=False)
        if textured and self.uv is not None:
            from PIL import Image
            from trimesh.visual.material import PBRMaterial
            from trimesh.visual.texture import TextureVisuals

            image = Image.open(self.texture) if self.texture is not None else None
            mesh.visual = TextureVisuals(uv=self.uv.copy(), material=PBRMaterial(baseColorTexture=image))
        return mesh

    def clay(self) -> trimesh.Trimesh:
        """The untextured mesh (held-out gallery renders: no label can show)."""
        return self.to_trimesh(textured=False)


def load_scan_visual(
    obj_bytes: bytes,
    *,
    name: str,
    pose_index: int,
    scale: float = 1.0,
    texture: Path | None = None,
    bounds_tol: float = BOUNDS_TOL,
) -> ScanVisual:
    """The textured visual of a scan, moved by exactly the intake's rigid steps.

    ``pose_index`` must be the collision object's (``MeshObject.pose_index``). The geometry
    mesh is rebuilt from the same bytes the screen uses (:func:`rtt.sim.gso.geometry_mesh`),
    so the frame is the one the collision pieces were made in.
    """
    from rtt.sim.gso import geometry_mesh

    geom = geometry_mesh(obj_bytes)
    frame, rested = intake_frame(geom, name=name, scale=scale, pose_index=pose_index)
    raw = trimesh.load(io.BytesIO(obj_bytes), file_type="obj", force="mesh", process=False)
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise ValueError(f"{name}: no triangle mesh in the file")
    uv = getattr(raw.visual, "uv", None)
    uv = None if uv is None else np.asarray(uv, dtype=np.float64)
    visual = frame.apply(
        trimesh.Trimesh(np.asarray(raw.vertices, dtype=np.float64), np.asarray(raw.faces), process=False)
    )
    err = float(np.max(np.abs(np.asarray(visual.bounds) - np.asarray(rested.bounds))))
    if err > bounds_tol:
        raise ValueError(
            f"{name}: textured mesh bounds differ from the intake mesh by {1000 * err:.3f} mm "
            f"(> {1000 * bounds_tol:.1f} mm); the visual would not sit on the simulated object"
        )
    if uv is not None and len(uv) != len(visual.vertices):
        raise ValueError(f"{name}: {len(uv)} texture coordinates for {len(visual.vertices)} vertices")
    return ScanVisual(
        name=name,
        vertices=np.ascontiguousarray(visual.vertices, dtype=np.float64),
        faces=np.ascontiguousarray(visual.faces, dtype=np.int64),
        uv=uv,
        texture=texture,
        frame=frame,
        bounds_error_mm=1000.0 * err,
    )


def scan_visual_from_cache(cache: Any, name: str, pose_index: int, *, textured: bool = True) -> ScanVisual:
    """:func:`load_scan_visual` for an object of a :class:`rtt.sim.gso.GsoCache` (fetches the
    texture image on first use when ``textured``)."""
    data = cache.mesh_path(name).read_bytes()
    texture = cache.texture_path(name) if textured else None
    return load_scan_visual(data, name=name, pose_index=pose_index, texture=texture)


def collision_bounds_gap_mm(visual: ScanVisual, pieces: Sequence[np.ndarray]) -> float:
    """Largest gap (mm) between the visual's bounds and its collision pieces' bounds
    (convex hulls bulge a little beyond the scan; a large gap means the wrong frame)."""
    pts = np.concatenate([np.asarray(p) for p in pieces])
    hull_bounds = np.stack([pts.min(axis=0), pts.max(axis=0)])
    return 1000.0 * float(np.max(np.abs(hull_bounds - visual.bounds)))


def describe_scan(obj_bytes: bytes) -> dict[str, Any]:
    """Extents (mm) and whether the scan carries texture coordinates (the extents probe)."""
    raw = trimesh.load(io.BytesIO(obj_bytes), file_type="obj", force="mesh", process=False)
    return {
        "extents_mm": [round(1000.0 * float(e), 1) for e in raw.extents],
        "visual": type(raw.visual).__name__,
        "has_uv": getattr(raw.visual, "uv", None) is not None,
        "n_vertices": len(raw.vertices),
        "n_faces": len(raw.faces),
    }


def visual_attribution(name: str, source: Any) -> dict[str, str]:
    """The CC BY 4.0 credit of a textured scan visual (as :func:`rtt.sim.gso.attribution`,
    with the changes made for the visual)."""
    from rtt.sim.gso import attribution

    out = attribution(name, source)
    out["changes"] = VISUAL_CHANGES
    return out


# --------------------------------------------------------------------------- brand screen


@dataclass(frozen=True)
class ScreenEntry:
    """One screened scan. ``poses``: the stable-pose indices it was screened in (``None``:
    every pose); ``brand_visible`` applies in those poses only."""

    name: str
    brand_visible: bool
    poses: tuple[int, ...] | None = None
    method: str = ""
    note: str = ""


@dataclass(frozen=True)
class VisualScreen:
    """The parsed brand screen: listed scans, and the verdict for unlisted ones."""

    entries: Mapping[str, ScreenEntry]
    default_brand_visible: bool = True
    screened_branded_sha256: frozenset[str] = frozenset()
    path: Path | None = field(default=None, compare=False)

    def was_screened(self, name: str) -> bool:
        """Whether ``name`` was screened (cleared, or found branded and listed by hash)."""
        import hashlib

        return (
            name in self.entries or hashlib.sha256(name.encode()).hexdigest() in self.screened_branded_sha256
        )

    def brand_visible(self, name: str, pose_index: int | None = None) -> bool:
        """Whether a textured render of ``name`` (in ``pose_index``) may show a brand."""
        e = self.entries.get(name)
        if e is None:
            return self.default_brand_visible
        if e.poses is not None and (pose_index is None or pose_index not in e.poses):
            return True  # screened in other poses only
        return e.brand_visible

    def textured_ok(self) -> tuple[str, ...]:
        """Scans cleared for textured hero shots (in their screened poses)."""
        return tuple(sorted(n for n, e in self.entries.items() if not e.brand_visible))


def visual_screen_from_dict(raw: Mapping[str, Any], where: str = "visual_screen") -> VisualScreen:
    unknown = set(raw) - {"version", "default_brand_visible", "objects", "screened_branded_sha256"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    if int(raw.get("version", 1)) != 1:
        raise ValueError(f"{where}: unsupported version {raw.get('version')}")
    default = raw.get("default_brand_visible", True)
    if not isinstance(default, bool):
        raise TypeError(f"{where}: default_brand_visible must be true or false")
    entries: dict[str, ScreenEntry] = {}
    for name, item in (raw.get("objects") or {}).items():
        if not isinstance(item, Mapping):
            raise TypeError(f"{where}: {name}: must be a mapping")
        extra = set(item) - {"brand_visible", "poses", "method", "note"}
        if extra:
            raise ValueError(f"{where}: {name}: unknown keys {sorted(extra)}")
        visible = item.get("brand_visible")
        if not isinstance(visible, bool):
            raise TypeError(f"{where}: {name}: brand_visible must be true or false")
        poses = item.get("poses")
        if poses is not None:
            if not isinstance(poses, list) or not all(isinstance(p, int) and p >= 0 for p in poses):
                raise TypeError(f"{where}: {name}: poses must be a list of stable-pose indices")
            poses = tuple(poses)
        entries[str(name)] = ScreenEntry(
            str(name), visible, poses, str(item.get("method", "")), str(item.get("note", ""))
        )
    hashes = frozenset(str(h).lower() for h in raw.get("screened_branded_sha256") or ())
    if any(len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
        raise ValueError(f"{where}: screened_branded_sha256 entries must be 64 lower-case hex digits")
    return VisualScreen(entries, default, hashes)


def load_visual_screen(path: str | Path = DEFAULT_SCREEN) -> VisualScreen:
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        screen = visual_screen_from_dict(yaml.safe_load(fh) or {}, where=str(p))
    return VisualScreen(screen.entries, screen.default_brand_visible, screen.screened_branded_sha256, path=p)


@dataclass(frozen=True)
class ShotObject:
    """One object in a shot: its name, family (``"gso"`` for scans), stable pose, and whether
    it is drawn with its texture."""

    name: str
    family: str
    textured: bool
    pose_index: int | None = None


def check_shot(objects: Iterable[ShotObject], screen: VisualScreen | None = None) -> list[str]:
    """Every brand-screen violation in a shot (empty when the shot is clean)."""
    screen = screen or load_visual_screen()
    out = []
    for o in objects:
        if o.family != "gso" or not o.textured:
            continue
        if screen.brand_visible(o.name, o.pose_index):
            out.append(
                f"{o.name}: textured scan not cleared by the brand screen"
                + (f" in pose {o.pose_index}" if o.pose_index is not None else "")
                + "; render it clay or pick a cleared object"
            )
    return out


# --------------------------------------------------------------------------- asset manifest

#: Changes recorded in an asset-manifest entry for a scan's visual GLB (a list, as the
#: manifest wants it). Simulation-side changes (the CoACD pieces) belong to the physics model.
SCAN_ASSET_CHANGES_TEXTURED = (
    "re-posed to the stable resting pose of its physics model",
    "texture kept",
    "re-encoded as GLB",
)
SCAN_ASSET_CHANGES_CLAY = (
    "re-posed to the stable resting pose of its physics model",
    "texture removed (clay)",
    "re-encoded as GLB",
)


def scan_asset_fields(
    name: str,
    source: Any,
    *,
    pose_index: int,
    textured: bool,
    file_sha256: str,
    screen: VisualScreen | None = None,
) -> dict[str, Any]:
    """The licence-side fields of an asset-manifest entry for one scan's visual GLB.

    Returns ``license``, ``changes`` (a list), ``brand_visible`` and an upstream ``source``
    block (the commit-pinned URL of the scan file, the commit, the path and the file's
    SHA-256 ``file_sha256``). A clay GLB has no texture, so it never shows a brand; a textured
    one shows a brand unless the brand screen cleared ``name`` in ``pose_index``. The short
    ``attribution`` line is the asset store's (the render code owns it).
    """
    from rtt.sim.gso import ASSET_LICENSE

    sha = str(file_sha256)
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"{name}: file_sha256 must be 64 lower-case hex digits")
    visible = (screen or load_visual_screen()).brand_visible(name, pose_index) if textured else False
    return {
        "license": ASSET_LICENSE,
        "changes": list(SCAN_ASSET_CHANGES_TEXTURED if textured else SCAN_ASSET_CHANGES_CLAY),
        "brand_visible": bool(visible),
        "source": {
            "kind": "upstream",
            "url": source.raw_url(name),
            "commit": source.commit,
            "path": f"models/{name}/{source.mesh_file}",
            "sha256": sha,
        },
    }
