"""The open-source parallel gripper: visual meshes and the stock blade entrant.

Inputs are the gripper's simulation meshes, vendored unmodified in
``assets/gripper/so_arm_102/meshes/`` and pinned (commit, path, size, sha256) by that
directory's ``lock.json`` (``scripts/fetch_assets.py`` fetches and verifies them; the credit
line is in ``docs/ATTRIBUTION.md``). The three meshes share one pre-assembled frame in
millimetres with the jaws closed (upstream joint value 0):

* ``link5_1``: the gripper frame (wrist servo, frame, camera plate);
* ``clamp1_1``: the left clamp (+y side, closes toward -y);
* ``clamp2_1``: the right clamp, the same printed part turned 180 degrees.

Frame mapping (verified against the upstream URDF):

* assembly millimetres -> the upstream ``link5`` frame: ``0.001 v + LINK5_VIS_OFFSET``;
* ``link5`` -> the scored palm frame: a +90 degree rotation about y (``x -> -z``, ``z -> x``)
  and a shift that puts the frame's front face (assembly x = 297.3 mm) on the palm
  underside, palm z = -12 mm, the jaw root plane of :mod:`rtt.design.freeform.types`;
* palm (jaws closed) -> each jaw body frame: +12 mm in z. At full closure a jaw's body
  origin lies on the palm centre plane, so no y shift is needed for the left jaw, whose
  inner face lands on y = +0.4 mm (``CARRIAGE_INNER_Y``). Upstream places the right clamp
  1.0 mm off the mirror position (its inner face at y = +0.6 mm, overlapping the left one);
  it is moved by -1.0 mm in y so the pair is symmetric about the palm centre and the right
  inner face sits at y = -0.4 mm, as the coupled scored jaws require.

Each clamp is cut at the root plane with manifold3d: the part above is the **carriage**
(a moving visual on the guide rods), the part below is the **stock blade**, the "stock"
entrant (D9). Its collision geometry is a CoACD decomposition (threshold 0.02, seed 0,
coacd 1.0.14), run once per jaw and committed to ``stock_blade_hulls.npz`` with its
settings and input hashes; :func:`stock_blade_report` measures how far the hulls bulge out of
the blade's inner face (gate GV1: at most 0.2 mm, else the box approximation is used).

Nothing here enters the scored model by itself: the visual meshes are render-only, and the
stock hulls are handed to the scene by the stock entrant as ordinary jaw pieces.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import trimesh

from rtt.design.freeform.types import (
    CARRIAGE_INNER_Y,
    MAX_PIECES_PER_JAW,
    PLA_DENSITY,
    ROOT_PLANE_PALM_Z,
    JawGeometry,
)

__all__ = [
    "ASSET_DIR",
    "HULLS_PATH",
    "HULL_SETTINGS",
    "AssetError",
    "GripperAssetIds",
    "GripperMeshes",
    "StockBladeHulls",
    "add_gripper_assets",
    "assembly_to_palm",
    "build_stock_blade_hulls",
    "inner_face_deviation",
    "load_gripper",
    "load_lock",
    "load_stock_blade_hulls",
    "split_at_root",
    "stl_bytes_mm",
    "stock_blade_report",
    "stock_blade_shas",
    "stock_jaws",
    "verify_assets",
]

Side = Literal["left", "right"]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
ASSET_DIR: Final = REPO_ROOT / "assets" / "gripper" / "so_arm_102"
LOCK_PATH: Final = ASSET_DIR / "lock.json"
HULLS_PATH: Final = ASSET_DIR / "stock_blade_hulls.npz"
MESH_FILES: Final = {
    "frame": "meshes/link5_1.stl",
    "clamp_left": "meshes/clamp1_1.stl",
    "clamp_right": "meshes/clamp2_1.stl",
}

# Upstream URDF: link5's visual origin; every mesh vertex v (mm) sits at 0.001 v + this offset.
LINK5_VIS_OFFSET: Final = np.array([-0.22828777435, 6.48704621215e-07, -0.244954758855])
FRAME_FACE_X_MM: Final = 297.3  # assembly x of the frame's front face, where the blades leave it
R_LINK5_TO_PALM: Final = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # Ry(+90 deg)
RIGHT_CLAMP_SHIFT_Y: Final = -0.001  # m; makes the clamp pair symmetric about the palm centre
JAW_FROM_PALM_CLOSED: Final = np.array([0.0, 0.0, -ROOT_PLANE_PALM_Z])  # palm (closed) -> jaw frame

# Pinned CoACD settings for the stock blade (spec section 6, conflict row 15).
# max_convex_hull = 40 keeps the stock entrant inside the same contact-count rule as every designed
# jaw (MAX_PIECES_PER_JAW); uncapped, CoACD returns 44 pieces with the same pad fidelity.
HULL_SETTINGS: Final = {"threshold": 0.02, "seed": 0, "max_convex_hull": 40, "preprocess_mode": "auto"}
HULLS_FORMAT: Final = "rtt-stock-blade-hulls"
BULGE_LIMIT_MM: Final = 0.2  # GV1: the hulls may stand at most this far proud of the inner face
CONTACT_BAND_MM: Final = 1.0  # inner-face points within this of the frontmost point form the pad
_ZIP_TIME: Final = (1980, 1, 1, 0, 0, 0)


class AssetError(RuntimeError):
    """A vendored asset is missing or does not match its lock."""


# --------------------------------------------------------------------------- vendored files


def load_lock(root: Path = ASSET_DIR) -> dict[str, Any]:
    return json.loads((Path(root) / "lock.json").read_text(encoding="utf-8"))


def verify_assets(root: Path = ASSET_DIR) -> dict[str, Any]:
    """Check every locked file's size and sha256; returns the lock or raises :class:`AssetError`."""
    root = Path(root)
    lock = load_lock(root)
    bad = []
    for rel, entry in lock["files"].items():
        path = root / rel
        if not path.is_file():
            bad.append(f"{rel} missing")
        elif (
            path.stat().st_size != entry["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
        ):
            bad.append(f"{rel} does not match lock.json")
    if bad:
        raise AssetError(f"{root}: {', '.join(bad)}; run `python scripts/fetch_assets.py`")
    return lock


def assembly_to_palm() -> np.ndarray:
    """4x4 transform: an assembly-frame vertex in millimetres -> the palm frame in metres."""
    face_x_link5 = FRAME_FACE_X_MM * 1e-3 + LINK5_VIS_OFFSET[0]
    t = np.eye(4)
    t[:3, :3] = R_LINK5_TO_PALM * 1e-3
    t[:3, 3] = R_LINK5_TO_PALM @ LINK5_VIS_OFFSET + np.array([0.0, 0.0, ROOT_PLANE_PALM_Z + face_x_link5])
    return t


def _load_stl(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, file_type="stl", force="mesh", process=True)
    mesh.merge_vertices()
    return mesh


def split_at_root(clamp_palm: trimesh.Trimesh) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Cut a clamp (palm frame, jaws closed) at the root plane: ``(carriage, blade)``."""
    from rtt.part.mount import manifold_from_trimesh

    solid = manifold_from_trimesh(clamp_palm)
    above, below = solid.split_by_plane((0.0, 0.0, 1.0), ROOT_PLANE_PALM_Z)
    out = []
    for part in (above, below):
        mesh = part.to_mesh()
        tm = trimesh.Trimesh(
            np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
            np.asarray(mesh.tri_verts),
            process=True,
        )
        if not tm.is_watertight:
            raise AssetError("the clamp cut is not watertight")
        out.append(tm)
    return out[0], out[1]


@dataclass(frozen=True, eq=False)
class GripperMeshes:
    """The gripper's visual meshes in metres: the frame in the palm frame, the rest in jaw frames."""

    frame: trimesh.Trimesh
    carriage_left: trimesh.Trimesh
    carriage_right: trimesh.Trimesh
    blade_left: trimesh.Trimesh
    blade_right: trimesh.Trimesh
    lock: dict[str, Any]

    def carriage(self, side: Side) -> trimesh.Trimesh:
        return self.carriage_left if side == "left" else self.carriage_right

    def blade(self, side: Side) -> trimesh.Trimesh:
        return self.blade_left if side == "left" else self.blade_right

    def source(self, key: str) -> dict[str, Any]:
        """The ``source`` block (ASSET_MANIFEST.md) of the upstream file behind ``key``."""
        rel = MESH_FILES["frame" if key == "frame" else f"clamp_{key.rsplit('_', 1)[-1]}"]
        entry = self.lock["files"][rel]
        return {
            "kind": "upstream",
            "url": self.lock["repository"],
            "commit": self.lock["commit"],
            "path": entry["path"],
            "sha256": entry["sha256"],
        }


@functools.lru_cache(maxsize=4)
def _load_gripper_cached(root: str) -> GripperMeshes:
    lock = verify_assets(Path(root))
    to_palm = assembly_to_palm()
    meshes = {}
    for key, rel in MESH_FILES.items():
        mesh = _load_stl(Path(root) / rel)
        mesh.apply_transform(to_palm)
        meshes[key] = mesh
    meshes["clamp_right"].apply_translation([0.0, RIGHT_CLAMP_SHIFT_Y, 0.0])
    parts = {}
    for side in ("left", "right"):
        carriage, blade = split_at_root(meshes[f"clamp_{side}"])
        for name, mesh in (("carriage", carriage), ("blade", blade)):
            mesh.apply_translation(JAW_FROM_PALM_CLOSED)
            parts[f"{name}_{side}"] = mesh
    return GripperMeshes(frame=meshes["frame"], lock=lock, **parts)


def load_gripper(root: Path = ASSET_DIR) -> GripperMeshes:
    """Load, verify and re-pose the vendored gripper meshes (cached per directory)."""
    return _load_gripper_cached(str(Path(root).resolve()))


def stl_bytes_mm(mesh_m: trimesh.Trimesh) -> bytes:
    """Binary STL bytes (trimesh's writer: zero header) of a metre mesh, scaled to millimetres."""
    mm = mesh_m.copy()
    mm.apply_scale(1000.0)
    return mm.export(file_type="stl")


# --------------------------------------------------------------------------- stock blade hulls


@dataclass(frozen=True, eq=False)
class StockBladeHulls:
    """Convex collision pieces of the stock blade, one tuple per jaw (jaw frames, metres)."""

    left: tuple[np.ndarray, ...]
    right: tuple[np.ndarray, ...]
    meta: dict[str, Any]
    sha256: str  # of the committed .npz file

    def pieces(self, side: Side) -> tuple[np.ndarray, ...]:
        return self.left if side == "left" else self.right


def _mesh_digest(mesh: trimesh.Trimesh) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.round(mesh.vertices, 9), dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype="<i8").tobytes())
    return h.hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """``np.savez`` with fixed zip timestamps and sorted entries, so equal arrays give equal bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            arr_buf = io.BytesIO()
            np.lib.format.write_array(arr_buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, arr_buf.getvalue())
    Path(path).write_bytes(buf.getvalue())


def build_stock_blade_hulls(path: Path = HULLS_PATH, root: Path = ASSET_DIR) -> Path:
    """Run the pinned CoACD decomposition on both blades and write the hull file.

    This is run once and committed; scoring never calls it. The file records the CoACD
    version, the settings, the input mesh hashes and each piece's vertices (rounded to 1e-9 m).
    """
    import importlib.metadata

    import coacd

    grip = load_gripper(root)
    coacd.set_log_level("error")
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {
        "format": HULLS_FORMAT,
        "version": 1,
        "coacd": importlib.metadata.version("coacd"),
        "settings": dict(HULL_SETTINGS),
        "frame": "jaw body frame, metres (rtt.design.freeform.types)",
        "upstream_commit": grip.lock["commit"],
        "inputs": {},
        "pieces": {},
    }
    for side in ("left", "right"):
        blade = grip.blade(side)
        parts = coacd.run_coacd(
            coacd.Mesh(np.asarray(blade.vertices, np.float64), np.asarray(blade.faces, np.int32)),
            threshold=HULL_SETTINGS["threshold"],
            max_convex_hull=HULL_SETTINGS["max_convex_hull"],
            preprocess_mode=HULL_SETTINGS["preprocess_mode"],
            seed=HULL_SETTINGS["seed"],
        )
        hulls = []
        for verts, faces in parts:
            hull = trimesh.Trimesh(np.asarray(verts, np.float64), np.asarray(faces)).convex_hull
            hulls.append(np.round(np.asarray(hull.vertices, np.float64), 9))
        # deterministic order: by centroid z, then y, then x
        hulls.sort(key=lambda v: tuple(np.round(v.mean(axis=0)[::-1], 9)))
        for i, v in enumerate(hulls):
            arrays[f"{side}_{i:02d}"] = v
        meta["inputs"][side] = _mesh_digest(blade)
        meta["pieces"][side] = len(hulls)
    arrays["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    _write_npz(Path(path), arrays)
    return Path(path)


def load_stock_blade_hulls(path: Path = HULLS_PATH) -> StockBladeHulls:
    """Read the committed hull file (raises :class:`AssetError` if it is missing or malformed)."""
    path = Path(path)
    if not path.is_file():
        raise AssetError(f"{path} is missing; run `python scripts/fetch_assets.py --hulls`")
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        if meta.get("format") != HULLS_FORMAT:
            raise AssetError(f"{path}: not a stock blade hull file")
        sides = {
            side: tuple(np.array(z[f"{side}_{i:02d}"], dtype=np.float64) for i in range(meta["pieces"][side]))
            for side in ("left", "right")
        }
    return StockBladeHulls(
        left=sides["left"], right=sides["right"], meta=meta, sha256=hashlib.sha256(data).hexdigest()
    )


def _hull_front(pieces: tuple[np.ndarray, ...], xs: np.ndarray, zs: np.ndarray, sign: float) -> np.ndarray:
    """Per grid point, the hull union's frontmost ``sign * y`` minimum (inf where no hull)."""
    from scipy.spatial import ConvexHull

    best = np.full(xs.shape, np.inf)
    for v in pieces:
        eq = ConvexHull(v * [1.0, sign, 1.0]).equations  # n . p + d <= 0 inside
        n_x, n_y, n_z, d = eq.T
        rhs = -d[:, None] - n_x[:, None] * xs[None] - n_z[:, None] * zs[None]  # n_y * t <= rhs
        lo = np.where(
            n_y[:, None] < -1e-12, rhs / np.where(n_y[:, None] < -1e-12, n_y[:, None], -1.0), -np.inf
        )
        hi = np.where(n_y[:, None] > 1e-12, rhs / np.where(n_y[:, None] > 1e-12, n_y[:, None], 1.0), np.inf)
        flat_ok = np.all(np.where(np.abs(n_y[:, None]) <= 1e-12, rhs >= -1e-12, True), axis=0)
        t_min, t_max = lo.max(axis=0), hi.min(axis=0)
        ok = flat_ok & (t_min <= t_max + 1e-12)
        best = np.where(ok, np.minimum(best, t_min), best)
    return best


def _mesh_front(mesh: trimesh.Trimesh, xs: np.ndarray, zs: np.ndarray, sign: float) -> np.ndarray:
    """Per grid point, the mesh's frontmost ``sign * y`` along a line parallel to y (inf = miss)."""
    tri = mesh.triangles * [1.0, sign, 1.0]
    best = np.full(xs.shape, np.inf)
    for chunk in np.array_split(tri, max(1, len(tri) // 64)):
        a, b, c = chunk[:, 0], chunk[:, 1], chunk[:, 2]
        # barycentric coordinates in the x-z projection
        v0x, v0z = (b - a)[:, 0, None], (b - a)[:, 2, None]
        v1x, v1z = (c - a)[:, 0, None], (c - a)[:, 2, None]
        px, pz = xs[None] - a[:, 0, None], zs[None] - a[:, 2, None]
        den = v0x * v1z - v1x * v0z
        good = np.abs(den) > 1e-18
        den = np.where(good, den, 1.0)
        u = (px * v1z - v1x * pz) / den
        w = (v0x * pz - px * v0z) / den
        inside = good & (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1 + 1e-9)
        y = a[:, 1, None] + u * (b - a)[:, 1, None] + w * (c - a)[:, 1, None]
        best = np.minimum(best, np.where(inside, y, np.inf).min(axis=0))
    return best


def _point_triangle_distance(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Exact distance from each point ``(N, 3)`` to the nearest triangle of ``(T, 3, 3)``."""
    best = np.full(len(points), np.inf)
    for chunk in np.array_split(tris, max(1, len(tris) // 32)):
        a, b, c = (chunk[:, i][None] for i in range(3))  # (1, t, 3)
        p = points[:, None]  # (n, 1, 3)
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n, axis=-1, keepdims=True)
        n = n / np.where(nn > 0, nn, 1.0)
        d_plane = np.sum((p - a) * n, axis=-1)
        q = p - d_plane[..., None] * n  # projection onto the plane
        inside = np.ones(d_plane.shape, dtype=bool)
        for u, v in ((a, b), (b, c), (c, a)):
            inside &= np.sum(np.cross(v - u, q - u) * n, axis=-1) >= -1e-15
        dist = np.where(inside & (nn[..., 0] > 0), np.abs(d_plane), np.inf)
        for u, v in ((a, b), (b, c), (c, a)):
            e = v - u
            t = np.clip(np.sum((p - u) * e, axis=-1) / np.maximum(np.sum(e * e, axis=-1), 1e-30), 0.0, 1.0)
            dist = np.minimum(dist, np.linalg.norm(p - (u + t[..., None] * e), axis=-1))
        best = np.minimum(best, dist.min(axis=1))
    return best


def inner_face_deviation(
    blade: trimesh.Trimesh, pieces: tuple[np.ndarray, ...], side: Side, step: float = 2.5e-4
) -> dict[str, float]:
    """How far the hull union departs from the blade's inner (contact-facing) surface.

    The inner face is what a part touches: for the left jaw the blade seen from -y, for the
    right jaw from +y. On a grid in x-z (``step`` metres) the frontmost point of the mesh and of
    the hull union are found along the closing axis.

    * ``bulge_mm`` (the GV1 number): wherever the hull's front point lies in front of the
      blade's (or where the blade has no material at all), that point is outside the real solid;
      its exact 3D distance to the blade surface is how far the collision geometry stands proud
      of what is drawn. The maximum over the inner face is reported.
    * ``bulge_axis_mm``: the same difference measured along the closing axis only. At a step in
      the face a hull that chamfers the corner by a fraction of a millimetre shows up here as the
      full step depth, so this is a diagnostic, not the gate.
    * ``inset_mm``: how far the hulls sit behind the real face (along the axis).
    * ``*_pad_mm``: the same, on the contact pad only (inner-face points within
      ``CONTACT_BAND_MM`` of the frontmost point).
    * ``spill_mm2``: front-view hull area where the blade has none.
    """
    sign = 1.0 if side == "left" else -1.0  # left faces -y: frontmost = min y
    lo, hi = blade.bounds
    xs, zs = np.meshgrid(np.arange(lo[0], hi[0] + step, step), np.arange(lo[2], hi[2] + step, step))
    xs, zs = xs.ravel(), zs.ravel()
    m = _mesh_front(blade, xs, zs, sign)
    h = _hull_front(pieces, xs, zs, sign)
    hit, hull = np.isfinite(m), np.isfinite(h)
    both = hit & hull
    dev = np.full(xs.shape, np.nan)
    dev[both] = m[both] - h[both]  # > 0: the hull's front is in front of the blade's
    front = float(m[hit].min())
    pad = hit & (m <= front + CONTACT_BAND_MM * 1e-3)
    proud = hull & (~hit | (dev > 1e-9))
    pts = np.stack([xs[proud], sign * h[proud], zs[proud]], axis=1)
    dist = _point_triangle_distance(pts, np.asarray(blade.triangles)) if len(pts) else np.zeros(0)
    bulge = float(dist.max()) if len(dist) else 0.0
    over = proud.copy()
    over[proud] = dist > BULGE_LIMIT_MM * 1e-3
    recess = m[over & hit] - front  # how far behind the contact plane the over-limit points sit
    pad_dev = dev[pad & hull]

    def mm(v: float) -> float:
        return round(1000.0 * max(0.0, float(v)), 4)

    return {
        "bulge_mm": mm(bulge),
        "bulge_axis_mm": mm(np.nanmax(dev[both])),
        "inset_mm": mm(-np.nanmin(dev[both])),
        "bulge_pad_mm": mm(pad_dev.max()),
        "inset_pad_mm": mm(-pad_dev.min()),
        "pad_area_mm2": round(float(pad.sum()) * (step * 1000.0) ** 2, 1),
        "spill_mm2": round(float((~hit & hull).sum()) * (step * 1000.0) ** 2, 1),
        "over_limit_mm2": round(float(over.sum()) * (step * 1000.0) ** 2, 1),
        "over_limit_min_recess_mm": round(1000.0 * float(recess.min()), 3) if len(recess) else None,
        "inner_face_y_mm": round(1000.0 * sign * front, 4),
    }


def stock_blade_report(hulls_path: Path = HULLS_PATH, root: Path = ASSET_DIR) -> dict[str, Any]:
    """GV1 input: piece counts, convexity, volumes and the inner-face deviation of each jaw."""
    from scipy.spatial import ConvexHull

    grip = load_gripper(root)
    hulls = load_stock_blade_hulls(hulls_path)
    out: dict[str, Any] = {
        "hulls_sha256": hulls.sha256,
        "coacd": hulls.meta["coacd"],
        "settings": hulls.meta["settings"],
    }
    worst = 0.0
    for side in ("left", "right"):
        blade = grip.blade(side)
        pieces = hulls.pieces(side)
        stale = hulls.meta["inputs"][side] != _mesh_digest(blade)
        vol = sum(ConvexHull(p).volume for p in pieces)
        convex = all(len(ConvexHull(p).vertices) == len(p) for p in pieces)  # no interior vertices
        dev = inner_face_deviation(blade, pieces, side)
        worst = max(worst, dev["bulge_mm"])
        out[side] = {
            "pieces": len(pieces),
            "convex": convex,
            "stale_inputs": stale,
            "blade_volume_cm3": round(blade.volume * 1e6, 4),
            "hull_volume_cm3": round(vol * 1e6, 4),
            "mass_g_pla": round(blade.volume * PLA_DENSITY * 1000.0, 2),
            "length_mm": round(-1000.0 * blade.bounds[0][2], 3),
            **dev,
        }
    worst_pad = max(out[s]["bulge_pad_mm"] for s in ("left", "right"))
    out["bulge_limit_mm"] = BULGE_LIMIT_MM
    out["bulge_mm"] = worst
    out["bulge_pad_mm"] = worst_pad
    out["bulge_ok"] = worst <= BULGE_LIMIT_MM
    out["pad_ok"] = worst_pad <= BULGE_LIMIT_MM
    out["pieces_ok"] = all(out[s]["pieces"] <= MAX_PIECES_PER_JAW for s in ("left", "right"))
    return out


def stock_jaws(hulls_path: Path = HULLS_PATH, root: Path = ASSET_DIR) -> tuple[JawGeometry, JawGeometry]:
    """The stock entrant as two :class:`JawGeometry` (hull pieces; blade mass in PLA)."""
    grip = load_gripper(root)
    hulls = load_stock_blade_hulls(hulls_path)
    jaws = []
    for side in ("left", "right"):
        blade = grip.blade(side).copy()
        blade.density = PLA_DENSITY
        inertia = blade.moment_inertia  # about the centre of mass, jaw axes
        jaws.append(
            JawGeometry(
                pieces=hulls.pieces(side),
                mass=float(blade.mass),
                com=np.asarray(blade.center_mass),
                inertia6=np.array(
                    [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
                ),
                print_mesh_sha=hashlib.sha256(stl_bytes_mm(grip.blade(side))).hexdigest(),
            )
        )
    return jaws[0], jaws[1]


def stock_blade_shas(root: Path = ASSET_DIR) -> dict[str, str]:
    """``{"left": sha, "right": sha}``: the ``print_mesh_sha`` of each stock blade, so a jaw pair
    built from the stock entrant can be recognised (and drawn with the blade CAD)."""
    return dict(_stock_blade_shas_cached(str(Path(root).resolve())))


@functools.lru_cache(maxsize=4)
def _stock_blade_shas_cached(root: str) -> tuple[tuple[str, str], ...]:
    grip = load_gripper(Path(root))
    return tuple(
        (side, hashlib.sha256(stl_bytes_mm(grip.blade(side))).hexdigest()) for side in ("left", "right")
    )


@dataclass(frozen=True)
class GripperAssetIds:
    """Asset ids of the gripper visuals in a store (frame in the palm frame, rest in jaw frames)."""

    frame: str
    carriage_left: str
    carriage_right: str
    blade_left: str
    blade_right: str

    def carriage(self, side: Side) -> str:
        return self.carriage_left if side == "left" else self.carriage_right

    def blade(self, side: Side) -> str:
        return self.blade_left if side == "left" else self.blade_right


def add_gripper_assets(store: Any, root: Path = ASSET_DIR) -> GripperAssetIds:
    """Write the gripper's GLB assets (frame, both carriages, both stock blades) into ``store``
    (an :class:`rtt.render.glb.AssetStore`) with their credit, licence, changes and source."""
    from rtt.render import glb
    from rtt.render import materials as mat

    grip = load_gripper(root)
    ids = {}
    items = (
        ("frame", grip.frame, mat.FRAME, "gripper_frame", "open-source gripper frame"),
        ("carriage_left", grip.carriage_left, mat.CARRIAGE, "carriage", "gripper carriage, left jaw"),
        ("carriage_right", grip.carriage_right, mat.CARRIAGE, "carriage", "gripper carriage, right jaw"),
        ("blade_left", grip.blade_left, mat.STOCK_BLADE, "finger_left", "stock finger, left jaw"),
        ("blade_right", grip.blade_right, mat.STOCK_BLADE, "finger_right", "stock finger, right jaw"),
    )
    for key, mesh, look, role, label in items:
        changes = list(glb.GRIPPER_CHANGES)
        if key == "frame":
            changes = [c for c in changes if not c.startswith("cut")]
        extras = {"license": "CC-BY-4.0", "role": role, "changes": changes, "source": grip.source(key)}
        data = glb.glb_bytes(
            [glb.mesh_from_trimesh(mesh, look, name=key)], copyright=glb.GRIPPER_ATTRIBUTION, extras=extras
        )
        ids[key] = store.add(
            data,
            role=role,
            license="CC-BY-4.0",
            attribution=glb.GRIPPER_ATTRIBUTION,
            changes=changes,
            label=label,
            source=grip.source(key),
        )
    return GripperAssetIds(**ids)


def inner_face_y(side: Side) -> float:
    """The contract's carriage inner face at full closure (m, jaw frame)."""
    return CARRIAGE_INNER_Y if side == "left" else -CARRIAGE_INNER_Y
