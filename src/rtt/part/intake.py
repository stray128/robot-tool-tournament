"""Part intake: an uploaded mesh file -> a validated, simulator-ready :class:`PartSpec`.

:func:`intake` is the single entry point of "Bring Your Part". It checks the upload,
fixes what can be fixed, and refuses what cannot be simulated with an error a user can
act on (:class:`PartError`). In order:

1. **Load** (STL, OBJ, PLY, OFF; at most :data:`MAX_FILE_BYTES` and
   :data:`MAX_FACES` triangles). Several bodies in one file (at most
   :data:`MAX_BODIES`) are kept together as one rigid part (with a warning). The
   convex hull may have at most :data:`MAX_HULL_FACES` triangles, which bounds the
   stable-pose search below (its cost grows with the square of that count).
2. **Clean and repair**: merge duplicate vertices, drop degenerate and duplicate faces,
   then fill holes and fix the winding and normals. A mesh that is still open after
   that is simulated from its convex pieces and its mass is flagged approximate.
3. **Units**: ``units="auto"`` reads a file whose largest extent is above 0.5 as
   millimetres (no gripper part is half a metre across) and anything smaller as
   metres; ``"mm"``, ``"cm"``, ``"m"`` and ``"in"`` override that. The part must then
   be :data:`MIN_PART_SIZE_M` to :data:`MAX_OBJECT_SIZE` across.
4. **Mass** from the solid volume and a density (``density`` in kg/m^3, or a
   ``material`` from :data:`MATERIALS`; default the task's object density). Parts
   heavier than friction alone can hold get a warning; above :data:`MAX_PART_MASS_KG`
   they are refused.
5. **Stable poses** (:func:`rtt.sim.mesh.stable_poses`, i.e.
   ``trimesh.poses.compute_stable_poses``): resting orientations ranked by the chance a
   dropped part lands in them. Mirror-image and symmetric duplicates (same footprint,
   height and centre-of-mass offset) are merged and their probabilities added. The
   ``top_k`` most likely poses that fit the gripper are kept.
6. **Size vs the opening**: a pose fits when its narrowest horizontal extent, the
   width across the jaw, is at most the 84 mm opening minus the 5 mm clearance the
   design validator requires (79 mm). No fitting pose: refused.
7. **Convex decomposition** of each kept pose with CoACD through
   :func:`rtt.sim.mesh.mesh_object_from_trimesh` (poses run in parallel processes;
   results are cached as ``.npz`` when ``cache_dir`` is given; the whole step is
   limited to :data:`DECOMPOSE_TIMEOUT_S`).
8. **Grip candidates** per pose: the jaw axis (``narrow``: across the narrowest
   extent; ``wide``: turned 90 degrees, when that width also fits) and where along
   the part to grasp. The centre of the bounding box is always a candidate unless
   there is almost no material there; a second, off-centre point is added where the
   part's cross-section under a finger-wide slab is clearly largest (an L bracket's
   upright, a tub's wall). Only material the fingertips can reach counts: anything
   below the controller's lowest grasp height (30 % of the part's height) is ignored.

Everything is deterministic for a given file and arguments.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import trimesh
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull, QhullError

from rtt.sim.mesh import (
    Decomposition,
    coacd_available,
    mesh_object_from_trimesh,
    read_mesh_object,
    save_mesh_object,
    stable_poses,
)
from rtt.sim.objects import MAX_OBJECT_SIZE, MeshObject
from rtt.sim.task import TaskConfig, default_task, gripper_task_for

__all__ = [
    "DECOMPOSE_TIMEOUT_S",
    "MATERIALS",
    "MAX_BODIES",
    "MAX_FACES",
    "MAX_FILE_BYTES",
    "MAX_HULL_FACES",
    "MAX_PART_MASS_KG",
    "MIN_HULL_THICKNESS_M",
    "MIN_PART_SIZE_M",
    "SUPPORTED_SUFFIXES",
    "UNIT_SCALES",
    "GraspCandidate",
    "PartError",
    "PartPose",
    "PartSpec",
    "Units",
    "detect_units",
    "intake",
    "load_part",
    "rotate_quarter_turn",
    "save_part",
]

Units = Literal["auto", "mm", "cm", "m", "in"]
Axis = Literal["narrow", "wide"]

UNIT_SCALES: Final[dict[str, float]] = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254}
SUPPORTED_SUFFIXES: Final = (".stl", ".obj", ".ply", ".off")
MAX_FILE_BYTES: Final = 50 * 1024 * 1024
MAX_FACES: Final = 300_000
MAX_BODIES: Final = 64
"""Separate bodies in one file; more is an assembly or a point cloud, not one part (and
splitting tens of thousands of bodies takes minutes)."""
MAX_HULL_FACES: Final = 8_000
"""Triangles on the part's convex hull. Stable-pose search grows with the square of
this (a 20k-face ball takes ~100 s, a 300k-face one hours), so finer outer surfaces
are refused with a request to export a coarser tessellation."""
MIN_HULL_THICKNESS_M: Final = 1e-4
"""Thinnest convex hull accepted, as ``2 x volume / area`` (about a plate's thickness). A
flat or near-flat mesh has a centre of mass that is never strictly inside its hull, and
``trimesh.poses.compute_stable_poses`` (which rejection-samples it) then never returns."""
MIN_PART_SIZE_M: Final = 0.005
MAX_PART_MASS_KG: Final = 2.0
SMALL_MM_WARNING: Final = 20.0
"""An auto-detected millimetre part smaller than this (mm) may really be in cm or inches."""

#: Densities (kg/m^3) a user can name instead of giving a number.
MATERIALS: Final[dict[str, float]] = {
    "plastic": 1200.0,
    "pla": 1240.0,
    "petg": 1270.0,
    "abs": 1050.0,
    "nylon": 1150.0,
    "rubber": 1100.0,
    "wood": 700.0,
    "aluminium": 2700.0,
    "aluminum": 2700.0,
    "steel": 7850.0,
    "brass": 8500.0,
}

POSE_KEY_MM: Final = 1.0
"""Poses whose footprint, height and centre-of-mass offset agree to this (mm) are merged."""
MIN_POSE_PROBABILITY: Final = 0.05
WIDE_AXIS_MIN_RATIO: Final = 1.15
"""The wide jaw axis is only offered when it is this much wider than the narrow one."""
SLAB_MM: Final = 16.0
"""Finger-wide slab over which the cross-section is averaged for grasp-point candidates."""
GRID_MM: Final = 1.5
"""Occupancy-grid resolution for cross-section profiles."""
OFFSET_GAIN: Final = 1.5
"""An off-centre grasp point is offered when its slab cross-section is this much larger."""
MIN_OFFSET_MM: Final = 4.0
END_MARGIN_MM: Final = 8.0
"""A grasp point keeps half a finger width (8 mm) of part on both sides."""
EMPTY_CENTRE_FRAC: Final = 0.05
"""The centre is dropped as a grasp point when its cross-section is below this share of the largest."""
DECOMPOSE_TIMEOUT_S: Final = 300.0
"""Wall-time limit (s) on decomposing all kept poses with CoACD. Its run time depends on
the shape, not just the triangle count (a 1,200-triangle box with a sealed spherical
cavity takes ~80 s per pose), so it is bounded by a clock: workers are killed and the
part is refused with a way forward."""
PROFILE_CHUNK: Final = 2_000_000
"""Grid points x hull planes tested per batch in :func:`_section_profile` (~16 MB of
float64), so memory stays bounded for half-metre parts and many-faceted pieces."""
_SAFE_NAME = re.compile(r"[^a-z0-9_-]+")
logger = logging.getLogger(__name__)


class PartError(ValueError):
    """An upload that cannot be simulated; ``str(err)`` is written for the person who uploaded it."""


# --------------------------------------------------------------------------- data model


@dataclass(frozen=True)
class GraspCandidate:
    """Where and how the gripper takes hold of one resting pose of the part.

    ``offset_mm`` is the grasp point along the part's length (object-frame x, from
    the bounding-box centre). In a :class:`~rtt.sim.objects.Placement` it becomes
    ``err_x = -offset`` (the part sits that far from where the gripper aims).
    """

    axis: Axis
    offset_mm: float
    width_mm: float
    """Extent across the jaw (the bounding box, so the opening check is conservative)."""
    section_mm2: float
    """Mean reachable cross-section of the part under a finger-wide slab centred on the grasp point."""

    @property
    def label(self) -> str:
        if abs(self.offset_mm) < 1e-9:
            return f"{self.axis}-centre"
        return f"{self.axis}-at{self.offset_mm:+.0f}mm"

    def to_json(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "offset_mm": round(self.offset_mm, 3),
            "width_mm": round(self.width_mm, 3),
            "section_mm2": round(self.section_mm2, 1),
            "label": self.label,
        }


@dataclass(frozen=True, eq=False)
class PartPose:
    """One resting pose of the part, as the simulator will see it.

    ``rank`` is its place among the kept poses (0 = most likely); ``pose_index`` is the
    index into :func:`rtt.sim.mesh.stable_poses` of its representative and
    ``merged`` the indices folded into it. ``extents_mm`` are in the grasp frame:
    length (x), width across the jaw (y) and height (z).
    """

    rank: int
    pose_index: int
    probability: float
    merged: tuple[int, ...]
    extents_mm: tuple[float, float, float]
    com_height_mm: float
    candidates: tuple[GraspCandidate, ...]
    objects: Mapping[str, MeshObject] = field(default_factory=dict, repr=False)
    """Simulator objects by jaw axis (``narrow``, and ``wide`` when offered)."""

    @property
    def n_pieces(self) -> int:
        return len(self.objects["narrow"].parts) if "narrow" in self.objects else 0

    def obj(self, axis: Axis = "narrow") -> MeshObject:
        return self.objects[axis]

    def to_json(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "pose_index": self.pose_index,
            "probability": round(self.probability, 6),
            "merged": list(self.merged),
            "extents_mm": [round(v, 3) for v in self.extents_mm],
            "com_height_mm": round(self.com_height_mm, 3),
            "n_pieces": self.n_pieces,
            "candidates": [c.to_json() for c in self.candidates],
            "objects": {
                axis: {"name": o.name, "geometry_sha256": o.geometry_sha256, "n_pieces": len(o.parts)}
                for axis, o in self.objects.items()
            },
        }


@dataclass(frozen=True, eq=False)
class PartSpec:
    """A validated upload: facts for the user, poses and grip candidates for the simulator."""

    name: str
    source_name: str
    source_sha256: str
    units: str
    units_detected: bool
    scale_to_m: float
    n_faces: int
    n_bodies: int
    watertight: bool
    repairs: tuple[str, ...]
    extents_mm: tuple[float, float, float]
    """Axis-aligned extents of the file as uploaded, converted to millimetres."""
    volume_cm3: float
    material: str
    density_kg_m3: float
    mass_g: float
    mass_is_approximate: bool
    max_grip_width_mm: float
    friction_capacity_kg: float
    decomposition: str
    poses: tuple[PartPose, ...]
    other_poses: tuple[dict[str, Any], ...]
    """Stable poses that were not kept (too unlikely, beyond ``top_k`` or too wide)."""
    warnings: tuple[str, ...]
    mesh_mm: trimesh.Trimesh | None = field(default=None, repr=False)
    """The repaired mesh in millimetres, in the file's own orientation (previews, packs)."""

    @property
    def mass_kg(self) -> float:
        return self.mass_g / 1000.0

    @property
    def primary(self) -> PartPose:
        return self.poses[0]

    def to_json(self) -> dict[str, Any]:
        return {
            "format": "rtt-part",
            "version": 1,
            "name": self.name,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "units": self.units,
            "units_detected": self.units_detected,
            "scale_to_m": self.scale_to_m,
            "n_faces": self.n_faces,
            "n_bodies": self.n_bodies,
            "watertight": self.watertight,
            "repairs": list(self.repairs),
            "extents_mm": [round(v, 3) for v in self.extents_mm],
            "volume_cm3": round(self.volume_cm3, 4),
            "material": self.material,
            "density_kg_m3": self.density_kg_m3,
            "mass_g": round(self.mass_g, 3),
            "mass_is_approximate": self.mass_is_approximate,
            "max_grip_width_mm": round(self.max_grip_width_mm, 3),
            "friction_capacity_kg": round(self.friction_capacity_kg, 4),
            "decomposition": self.decomposition,
            "poses": [p.to_json() for p in self.poses],
            "other_poses": list(self.other_poses),
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """Plain-text summary for a terminal or a chat reply."""
        ex = " x ".join(f"{v:.1f}" for v in self.extents_mm)
        lines = [
            (
                f"{self.name}: {ex} mm, {self.mass_g:.0f} g of {self.material} "
                f"({'approximate' if self.mass_is_approximate else 'from the solid volume'}), units {self.units}"
                f"{' (detected)' if self.units_detected else ''}"
            ),
        ]
        for p in self.poses:
            cands = ", ".join(c.label for c in p.candidates)
            lines.append(
                f"  pose {p.rank}: lands this way {100 * p.probability:.0f}% of drops, "
                f"{p.extents_mm[0]:.0f} x {p.extents_mm[1]:.0f} x {p.extents_mm[2]:.0f} mm, "
                f"{p.n_pieces} convex pieces; grasps: {cands}"
            )
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


# --------------------------------------------------------------------------- helpers


def _safe_name(raw: str) -> str:
    name = _SAFE_NAME.sub("_", raw.strip().lower()).strip("_-")
    return (name or "part")[:48]


def detect_units(max_extent: float) -> str:
    """Guess the file units from its largest extent (see the module docstring)."""
    if not math.isfinite(max_extent) or max_extent <= 0:
        raise PartError("the mesh has no extent; it may be empty or flat")
    return "mm" if max_extent > MAX_OBJECT_SIZE else "m"


def _resolve_density(material: str | None, density: float | None, task: TaskConfig) -> tuple[str, float]:
    if material is not None and density is not None:
        raise PartError("give a material or a density, not both")
    if density is not None:
        if not math.isfinite(density) or not 50.0 <= density <= 25_000.0:
            raise PartError(f"density must be between 50 and 25000 kg/m^3, got {density:g}")
        return "custom", float(density)
    if material is not None:
        key = material.strip().lower()
        if key not in MATERIALS:
            raise PartError(f"unknown material {material!r}; choose one of {', '.join(sorted(MATERIALS))}")
        return key, MATERIALS[key]
    return "plastic", float(task.physics.object_density)


def _load(path: Path) -> tuple[trimesh.Trimesh, str]:
    """The mesh in ``path`` and the SHA-256 of the bytes it was parsed from.

    The file is read once (at most :data:`MAX_FILE_BYTES`, plus one byte to detect a
    larger one) and parsed from memory with no file resolver: an OBJ's ``mtllib`` or
    texture references cannot make intake open any other file, and the hash is of
    exactly the bytes that were simulated (no stat-then-read race).
    """
    if not path.is_file():
        raise PartError(f"no file at {path.name!r}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise PartError(
            f"{path.name}: unsupported file type {path.suffix!r}; upload one of {', '.join(SUPPORTED_SUFFIXES)}"
        )
    limit_mb = MAX_FILE_BYTES // 2**20
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise PartError(f"{path.name} is {size / 2**20:.0f} MB; the limit is {limit_mb} MB")
    with path.open("rb") as fh:
        data = fh.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise PartError(f"{path.name} is over {limit_mb} MB; the limit is {limit_mb} MB")
    sha = hashlib.sha256(data).hexdigest()
    try:
        mesh = trimesh.load(io.BytesIO(data), file_type=suffix[1:], force="mesh")
    except Exception as exc:  # trimesh raises many types for corrupt files
        raise PartError(f"{path.name} could not be read as a mesh ({type(exc).__name__})") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise PartError(f"{path.name} holds no triangles")
    if len(mesh.faces) > MAX_FACES:
        raise PartError(
            f"{path.name} has {len(mesh.faces):,} triangles; simplify it below {MAX_FACES:,} "
            "(grasping needs millimetre detail, not more)"
        )
    if not np.all(np.isfinite(mesh.vertices)):
        raise PartError(f"{path.name} has vertices that are not finite numbers")
    return mesh, sha


def _check_mass(file_name: str, mass_kg: float, material: str, rho: float) -> None:
    if not mass_kg <= MAX_PART_MASS_KG:
        raise PartError(
            f"{file_name} weighs {mass_kg:.2f} kg as {material} ({rho:g} kg/m^3); this gripper is tested up "
            f"to {MAX_PART_MASS_KG:g} kg. Check the units and the material"
        )


def _repair(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    """Clean the mesh and try to close it; returns the mesh and the repairs made."""
    repairs: list[str] = []
    mesh = mesh.copy()
    n_vertices = len(mesh.vertices)
    mesh.merge_vertices()
    if len(mesh.vertices) < n_vertices:
        repairs.append(f"merged {n_vertices - len(mesh.vertices)} duplicate vertices")
    n_faces = len(mesh.faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) < n_faces:
        repairs.append(f"removed {n_faces - len(mesh.faces)} degenerate or duplicate faces")
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        repairs.append("filled holes" if mesh.is_watertight else "tried to fill holes (mesh is still open)")
    if not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
        repairs.append("made face winding consistent")
    if mesh.is_watertight:
        before = float(mesh.volume)
        trimesh.repair.fix_normals(mesh)
        if before < 0 < float(mesh.volume):
            repairs.append("flipped inside-out normals")
    return mesh, repairs


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose_signature(
    hull: np.ndarray, com: np.ndarray, transform: np.ndarray
) -> tuple[np.ndarray, float, tuple]:
    """Grasp-frame extents (m), centre-of-mass height (m) and a dedup key for one stable pose."""
    rot, shift = transform[:3, :3], transform[:3, 3]
    pts = hull @ rot.T + shift
    c = com @ rot.T + shift
    # Narrowest horizontal extent along y (the closing axis), like rtt.sim.mesh does.
    # Extremes along any direction lie on the footprint's outline, so the 360-angle
    # sweep only needs the outline's points (a few dozen instead of thousands).
    xy = pts[:, :2]
    try:
        xy = xy[ConvexHull(xy).vertices]
    except QhullError:  # a degenerate (collinear) footprint: sweep every point
        pass
    angles = np.radians(np.arange(0.0, 180.0, 0.5))
    widths = [np.ptp(xy @ _rot_z(a)[:2, :2].T[:, 1]) for a in angles]
    turn = _rot_z(float(angles[int(np.argmin(widths))]))
    pts, c = pts @ turn.T, c @ turn.T
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2.0
    ext = hi - lo
    off = np.abs(c - centre)
    q = 1000.0 / POSE_KEY_MM
    key = (
        *np.round(ext * q).astype(int).tolist(),
        round((c[2] - lo[2]) * q),
        *np.round(off[:2] * q).astype(int).tolist(),
    )
    return ext, float(c[2] - lo[2]), tuple(key)


def rotate_quarter_turn(obj: MeshObject, name: str | None = None) -> MeshObject:
    """``obj`` turned 90 degrees about Z (the jaw then closes across its length).

    The bounding box stays centred on the origin, so the result is a valid
    :class:`MeshObject`; mass properties are rotated with it (``I' = R I R^T``).
    """
    rot = _rot_z(math.pi / 2.0)
    rot[np.abs(rot) < 1e-15] = 0.0
    ixx, iyy, izz, ixy, ixz, iyz = obj.unit_inertia
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    turned = rot @ inertia @ rot.T
    com = rot @ np.asarray(obj.com)
    return replace(
        obj,
        name=name or f"{obj.name}-wide",
        parts=tuple(np.ascontiguousarray(p @ rot.T) for p in obj.parts),
        com=(float(com[0]), float(com[1]), float(com[2])),
        unit_inertia=(
            float(turned[0, 0]),
            float(turned[1, 1]),
            float(turned[2, 2]),
            float(turned[0, 1]),
            float(turned[0, 2]),
            float(turned[1, 2]),
        ),
    )


def _section_profile(obj: MeshObject, min_height_frac: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Cross-section area (mm^2) of the convex pieces at sample positions x (mm) along the length.

    The pieces are rasterised on a :data:`GRID_MM` occupancy grid (union, so overlaps
    count once) and the occupied cells are counted per x column. Only material at least
    ``min_height_frac`` of the part's height above the table counts: the fingertips never
    stop lower than the controller's lowest grasp height, so a thin base plate under
    them cannot be pinched.
    """
    pts = np.concatenate(obj.parts) * 1000.0
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    axes = [np.arange(lo[i] + GRID_MM / 2.0, hi[i], GRID_MM) for i in range(3)]
    axes = [a if len(a) else np.array([(lo[i] + hi[i]) / 2.0]) for i, a in enumerate(axes)]
    inside = np.zeros(tuple(len(a) for a in axes), dtype=bool)
    for piece in obj.parts:
        piece_mm = piece * 1000.0
        try:
            eq = ConvexHull(piece_mm).equations
        except QhullError:  # a flat piece has no volume to rasterise
            logger.debug("skipping a flat convex piece of %s", obj.name)
            continue
        # Only grid points inside the piece's bounding box can be inside the piece (the
        # 1e-6 mm margin exceeds the 1e-9 plane tolerance); they are tested in batches of
        # at most PROFILE_CHUNK point-plane pairs, so memory stays bounded.
        plo, phi = piece_mm.min(axis=0) - 1e-6, piece_mm.max(axis=0) + 1e-6
        span = [
            (int(np.searchsorted(a, plo[i], side="left")), int(np.searchsorted(a, phi[i], side="right")))
            for i, a in enumerate(axes)
        ]
        if any(j1 <= j0 for j0, j1 in span):
            continue
        (x0, x1), (y0, y1), (z0, z1) = span
        ny, nz = y1 - y0, z1 - z0
        n_points = (x1 - x0) * ny * nz
        per_batch = max(1, PROFILE_CHUNK // len(eq))
        for b0 in range(0, n_points, per_batch):
            flat = np.arange(b0, min(b0 + per_batch, n_points))
            ix, rest = np.divmod(flat, ny * nz)
            iy, iz = np.divmod(rest, nz)
            ix, iy, iz = ix + x0, iy + y0, iz + z0
            batch = np.stack([axes[0][ix], axes[1][iy], axes[2][iz]], axis=1)
            hit = np.all(batch @ eq[:, :3].T + eq[:, 3] <= 1e-9, axis=1)
            inside[ix[hit], iy[hit], iz[hit]] = True
    reachable = axes[2] >= lo[2] + min_height_frac * (hi[2] - lo[2])
    counts = (inside & reachable[None, None, :]).sum(axis=(1, 2)).astype(np.float64)
    return axes[0], counts * GRID_MM * GRID_MM


def _grasp_candidates(obj: MeshObject, axis: Axis, min_height_frac: float) -> list[GraspCandidate]:
    xs, area = _section_profile(obj, min_height_frac)
    window = max(1, round(SLAB_MM / GRID_MM))
    smooth = uniform_filter1d(area, size=window, mode="constant", cval=0.0)
    width_mm = obj.size[1] * 1000.0
    half = obj.size[0] * 1000.0 / 2.0
    centre_area = float(np.interp(0.0, xs, smooth))
    reach = half - END_MARGIN_MM
    allowed = np.abs(xs) <= max(reach, 0.0)
    if not allowed.any():
        allowed = np.abs(xs) == np.abs(xs).min()
    best_i = int(np.flatnonzero(allowed)[np.argmax(smooth[allowed])])
    best_x, best_area = float(xs[best_i]), float(smooth[best_i])
    out = []
    if centre_area >= EMPTY_CENTRE_FRAC * max(best_area, 1e-9):
        out.append(GraspCandidate(axis, 0.0, width_mm, centre_area))
    if abs(best_x) >= MIN_OFFSET_MM and best_area >= OFFSET_GAIN * centre_area:
        out.append(GraspCandidate(axis, float(round(best_x)), width_mm, best_area))
    if not out:  # nothing clearly better than the (thin) centre: keep the centre
        out.append(GraspCandidate(axis, 0.0, width_mm, centre_area))
    return out


def _build_object(args: tuple[Any, ...]) -> MeshObject:
    mesh, kwargs = args
    return mesh_object_from_trimesh(mesh, **kwargs)


def _cache_key(sha: str, kwargs: Mapping[str, Any]) -> str:
    payload = json.dumps({"sha": sha, **{k: v for k, v in kwargs.items() if k != "name"}}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _stop_pool(pool: ProcessPoolExecutor) -> None:
    """Kill a process pool's workers now (a hung CoACD call never returns on its own)."""
    # Python < 3.14 has no public way to kill busy workers; _processes is the pool's own
    # pid -> Process map.
    procs = list((getattr(pool, "_processes", None) or {}).values())
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    pool.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()


def _build_in_pool(
    mesh: trimesh.Trimesh, jobs: Sequence[dict[str, Any]], workers: int, timeout_s: float | None
) -> list[MeshObject] | None:
    """Build ``jobs`` in spawned worker processes; ``None`` if the pool cannot start.

    Raises:
        PartError: The decomposition did not finish within ``timeout_s`` seconds (the
            workers are killed first).
    """
    pool = ProcessPoolExecutor(max(1, min(workers, len(jobs))), mp_context=get_context("spawn"))
    try:
        futures = [pool.submit(_build_object, (mesh, kwargs)) for kwargs in jobs]
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        built = []
        for fut in futures:
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            built.append(fut.result(timeout=left))
    except FutureTimeout:
        _stop_pool(pool)
        raise PartError(
            f"the convex decomposition did not finish within {timeout_s:g} s; simplify the part, "
            "or use decomposition='hull' (fast, but it fills concave regions)"
        ) from None
    except BrokenProcessPool:  # e.g. a __main__ that spawned workers cannot re-import
        _stop_pool(pool)
        logger.warning(
            "parallel decomposition failed; decomposing the poses in this process, without a time limit"
        )
        return None
    except BaseException:
        _stop_pool(pool)
        raise
    pool.shutdown(wait=True)
    return built


def _decompose(
    mesh: trimesh.Trimesh,
    jobs: Sequence[dict[str, Any]],
    sha: str,
    cache_dir: Path | None,
    workers: int,
    timeout_s: float | None = None,
) -> list[MeshObject]:
    results: list[MeshObject | None] = [None] * len(jobs)
    todo = []
    for i, kwargs in enumerate(jobs):
        if cache_dir is not None:
            cached = cache_dir / f"{_cache_key(sha, kwargs)}.npz"
            if cached.exists():
                obj = read_mesh_object(cached)
                results[i] = replace(obj, name=kwargs["name"])
                continue
        todo.append(i)
    if todo:
        built: list[MeshObject] | None = None
        if timeout_s is not None or (workers > 1 and len(todo) > 1):
            # Worker processes give parallelism and the only way to stop a CoACD call.
            built = _build_in_pool(mesh, [jobs[i] for i in todo], workers, timeout_s)
        if built is None:
            built = [_build_object((mesh, jobs[i])) for i in todo]
        for i, obj in zip(todo, built, strict=True):
            results[i] = obj
            if cache_dir is not None:
                save_mesh_object(obj, cache_dir / f"{_cache_key(sha, jobs[i])}.npz")
    return [r for r in results if r is not None]


# --------------------------------------------------------------------------- intake


def intake(
    path: str | Path,
    *,
    units: Units = "auto",
    material: str | None = None,
    density: float | None = None,
    name: str | None = None,
    top_k: int = 3,
    min_probability: float = MIN_POSE_PROBABILITY,
    decomposition: Decomposition = "auto",
    threshold: float = 0.05,
    max_parts: int = 16,
    wide_axis: bool = True,
    task: TaskConfig | None = None,
    cache_dir: str | Path | None = None,
    workers: int | None = None,
    decompose_timeout_s: float | None = DECOMPOSE_TIMEOUT_S,
) -> PartSpec:
    """Validate an uploaded part and prepare it for simulation (see the module docstring).

    Args:
        path: STL/OBJ/PLY/OFF file.
        units: File units, or ``"auto"`` to detect millimetres vs metres.
        material: A name from :data:`MATERIALS` (or give ``density``).
        density: kg/m^3 (or give ``material``); default the task's object density.
        name: Display name (default: the file stem), sanitised to ``[a-z0-9_-]``.
        top_k: Most stable poses to keep (those below ``min_probability`` are skipped,
            but the most likely fitting pose is always kept).
        decomposition: ``"auto"`` (CoACD when installed), ``"coacd"`` or ``"hull"``.
        threshold: CoACD concavity tolerance; ``max_parts`` caps the convex pieces.
        wide_axis: Also offer grasping across the length when that fits.
        task: Simulator task (default ``configs/task_v1.yaml``).
        cache_dir: Where decomposed poses are cached (``.npz``); ``None``: no cache.
        workers: Processes for decomposing poses in parallel (default: one per pose).
        decompose_timeout_s: Wall-time limit on the CoACD decomposition (``None``: no
            limit, and a single pose is decomposed in this process).

    Raises:
        PartError: The upload cannot be simulated; the message says why and what to do.
    """
    task = task or default_task()
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if decompose_timeout_s is not None and not (
        math.isfinite(decompose_timeout_s) and decompose_timeout_s > 0
    ):
        raise ValueError(
            f"decompose_timeout_s must be a positive number of seconds or None, got {decompose_timeout_s!r}"
        )
    if units not in ("auto", *UNIT_SCALES):
        raise PartError(f"units must be one of auto, {', '.join(UNIT_SCALES)}; got {units!r}")
    p = Path(path)
    raw_mesh, sha = _load(p)
    display = _safe_name(name or p.stem)
    material_name, rho = _resolve_density(material, density, task)
    warnings: list[str] = []

    mesh, repairs = _repair(raw_mesh)
    if len(mesh.faces) < 4:
        raise PartError(f"{p.name} has fewer than 4 usable triangles after cleaning")
    # Bodies = face-connected components (what mesh.split counts), labelled without
    # building a sub-mesh per body, which takes minutes for tens of thousands of them.
    n_bodies = int(
        trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces)).max() + 1
    )
    if n_bodies > MAX_BODIES:
        raise PartError(
            f"{p.name} holds {n_bodies:,} separate bodies; upload one part (at most {MAX_BODIES} bodies "
            "that move together), not an assembly or a point cloud"
        )
    if n_bodies > 1:
        warnings.append(
            f"the file holds {n_bodies} separate bodies; they are simulated as one rigid part, "
            "as if glued together"
        )
    watertight = bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0)
    if not watertight:
        warnings.append(
            "the mesh is not a closed solid even after repair; collisions use its convex pieces "
            "and the mass is estimated from them"
        )

    raw_extent = float(mesh.extents.max())
    detected = units == "auto"
    unit = detect_units(raw_extent) if detected else units
    scale = UNIT_SCALES[unit]
    size_m = raw_extent * scale
    if size_m > MAX_OBJECT_SIZE:
        hint = " even read as millimetres" if detected else f" in {unit}"
        raise PartError(
            f"{p.name} is {size_m:.3g} m across{hint}; parts must be under {MAX_OBJECT_SIZE} m "
            f"(the gripper opens {2000 * task.gripper.open_half_gap:.0f} mm)"
        )
    if size_m < MIN_PART_SIZE_M:
        raise PartError(
            f"{p.name} is only {size_m * 1000:.3g} mm across read as {unit}; the smallest part this "
            f"gripper can pick is {MIN_PART_SIZE_M * 1000:.0f} mm. If the file is not in {unit}, pass units="
            f"{'mm' if unit == 'm' else 'm'}, 'cm' or 'in'"
        )
    if detected and unit == "mm" and raw_extent < SMALL_MM_WARNING:
        warnings.append(
            f"units read as millimetres, so the part is {raw_extent:.3g} mm across; if it was exported in "
            "cm or inches, pass units='cm' or units='in'"
        )
    mesh_m = mesh.copy()
    mesh_m.apply_scale(scale)
    mesh_mm = mesh.copy()
    mesh_mm.apply_scale(scale * 1000.0)
    if watertight:  # refuse a part that is too heavy before paying for its decomposition
        _check_mass(p.name, float(mesh_m.volume) * rho, material_name, rho)

    # Stable poses, merged into distinct ones.
    try:
        convex = mesh_m.convex_hull
        hull_thickness = 2.0 * float(convex.volume) / float(convex.area)
    except Exception:  # noqa: BLE001 - qhull refuses a flat point set: it has no thickness
        hull_thickness = 0.0
    if not np.isfinite(hull_thickness) or hull_thickness < MIN_HULL_THICKNESS_M:
        raise PartError(
            f"{p.name} is flat (at most {max(hull_thickness, 0.0) * 1000:.3g} mm thick across its outer "
            "surface); upload a solid part, not a surface or a sketch"
        )
    if len(convex.faces) > MAX_HULL_FACES:
        raise PartError(
            f"{p.name} has {len(convex.faces):,} triangles on its outer (convex) surface; the resting-pose "
            f"search allows {MAX_HULL_FACES:,}. Export it with a coarser tessellation (a larger chord "
            "tolerance) or decimate it; millimetre detail is enough for grasping"
        )
    hull = np.asarray(convex.vertices)
    com = np.asarray(mesh_m.center_mass if watertight else mesh_m.convex_hull.center_mass)
    groups: dict[tuple, dict[str, Any]] = {}
    for i, (transform, prob) in enumerate(stable_poses(mesh_m)):
        ext, com_h, key = _pose_signature(hull, com, transform)
        g = groups.setdefault(key, {"index": i, "prob": 0.0, "merged": [], "ext": ext, "com_h": com_h})
        g["prob"] += prob
        g["merged"].append(i)
    ordered = sorted(groups.values(), key=lambda g: (-g["prob"], g["index"]))
    gtask = gripper_task_for(task)
    max_width_mm = gtask.max_opening_mm - gtask.open_gap_margin_mm
    fitting = [g for g in ordered if g["ext"][1] * 1000.0 <= max_width_mm + 1e-9]
    if not fitting:
        narrowest = min(g["ext"][1] for g in ordered) * 1000.0
        raise PartError(
            f"{p.name} is at least {narrowest:.0f} mm across the jaw in every resting pose; the gripper opens "
            f"{gtask.max_opening_mm:.0f} mm and needs {gtask.open_gap_margin_mm:.0f} mm of clearance, so parts "
            f"must be at most {max_width_mm:.0f} mm across their narrowest side"
        )
    kept = [g for g in fitting if g["prob"] >= min_probability][:top_k] or fitting[:1]
    kept_ids = {id(g) for g in kept}
    other = tuple(
        {
            "pose_index": g["index"],
            "probability": round(g["prob"], 6),
            "extents_mm": [round(v * 1000.0, 3) for v in g["ext"]],
            "reason": "too wide for the opening"
            if g["ext"][1] * 1000.0 > max_width_mm + 1e-9
            else "below the probability or top-k cut",
        }
        for g in ordered
        if id(g) not in kept_ids
    )
    unfit_share = sum(g["prob"] for g in ordered if g["ext"][1] * 1000.0 > max_width_mm + 1e-9)
    if unfit_share >= 0.25:
        warnings.append(
            f"{100 * unfit_share:.0f}% of drops land in poses too wide for the gripper; those are not tested"
        )
    kept_share = sum(g["prob"] for g in kept)

    # Convex decomposition of every kept pose (in parallel), then grip candidates.
    method = decomposition if decomposition != "auto" else ("coacd" if coacd_available() else "hull")
    jobs = [
        {
            "name": f"{display}_pose{rank}",
            "scale": scale,
            "density": rho,
            "pose_index": g["index"],
            "decomposition": method,
            "threshold": threshold,
            "max_parts": max_parts,
            "seed": 0,
            "source_sha256": sha,
        }
        for rank, g in enumerate(kept)
    ]
    n_workers = workers if workers is not None else min(len(jobs), os.cpu_count() or 1)
    cache = Path(cache_dir) if cache_dir is not None else None
    if method == "hull":
        n_workers = 1  # hulls take milliseconds; not worth a process pool
    timeout = decompose_timeout_s if method == "coacd" else None
    objects = _decompose(mesh, jobs, sha, cache, n_workers, timeout)

    low_frac = min(task.controller.grasp_height_fracs)
    poses = []
    for rank, (g, obj) in enumerate(zip(kept, objects, strict=True)):
        by_axis: dict[str, MeshObject] = {"narrow": obj}
        cands = _grasp_candidates(obj, "narrow", low_frac)
        length_mm, width_mm = obj.size[0] * 1000.0, obj.size[1] * 1000.0
        if wide_axis and length_mm <= max_width_mm and length_mm >= WIDE_AXIS_MIN_RATIO * width_mm:
            wide = rotate_quarter_turn(obj, name=f"{obj.name}-wide")
            by_axis["wide"] = wide
            cands.extend(_grasp_candidates(wide, "wide", low_frac))
        poses.append(
            PartPose(
                rank=rank,
                pose_index=int(g["index"]),
                probability=float(g["prob"]),
                merged=tuple(int(i) for i in g["merged"]),
                extents_mm=tuple(float(v) * 1000.0 for v in obj.size),  # type: ignore[arg-type]
                com_height_mm=(obj.com[2] + obj.size[2] / 2.0) * 1000.0,
                candidates=tuple(cands),
                objects=by_axis,
            )
        )

    volume_m3 = float(mesh_m.volume) if watertight else float(poses[0].obj().volume)
    mass_kg = volume_m3 * rho
    capacity = task.physics.friction * task.gripper.grip_force / 9.81
    _check_mass(p.name, mass_kg, material_name, rho)
    if mass_kg > capacity:
        warnings.append(
            f"at {1000 * mass_kg:.0f} g the part is heavier than the ~{1000 * capacity:.0f} g that friction alone "
            f"holds at {task.gripper.grip_force:g} N grip force; only cradling or hooking fingers can lift it"
        )
    if kept_share < 0.5:
        warnings.append(
            f"the tested poses cover {100 * kept_share:.0f}% of drops; raise top_k to test more of them"
        )
    narrowest = min(p_.extents_mm[1] for p_ in poses)
    if narrowest < 5.0:
        warnings.append(f"the part is only {narrowest:.1f} mm across the jaw; thin parts are easy to miss")

    return PartSpec(
        name=display,
        source_name=p.name,
        source_sha256=sha,
        units=unit,
        units_detected=detected,
        scale_to_m=scale,
        n_faces=len(mesh.faces),
        n_bodies=int(n_bodies),
        watertight=watertight,
        repairs=tuple(repairs),
        extents_mm=tuple(float(v) for v in mesh_mm.extents),  # type: ignore[arg-type]
        volume_cm3=volume_m3 * 1e6,
        material=material_name,
        density_kg_m3=rho,
        mass_g=mass_kg * 1000.0,
        mass_is_approximate=not watertight,
        max_grip_width_mm=max_width_mm,
        friction_capacity_kg=capacity,
        decomposition=method,
        poses=tuple(poses),
        other_poses=other,
        warnings=tuple(warnings),
        mesh_mm=mesh_mm,
    )


# --------------------------------------------------------------------------- persistence


def save_part(part: PartSpec, directory: str | Path) -> Path:
    """Write ``part.json``, ``part_mm.stl`` and one ``.npz`` per simulator object; returns ``part.json``."""
    d = Path(directory)
    (d / "objects").mkdir(parents=True, exist_ok=True)
    for pose in part.poses:
        for axis, obj in pose.objects.items():
            save_mesh_object(obj, d / "objects" / f"pose{pose.rank}_{axis}.npz")
    if part.mesh_mm is not None:
        (d / "part_mm.stl").write_bytes(part.mesh_mm.export(file_type="stl"))
    path = d / "part.json"
    path.write_text(json.dumps(part.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_part(directory: str | Path) -> PartSpec:
    """Read a part written by :func:`save_part` (objects, mesh and all)."""
    d = Path(directory)
    data = json.loads((d / "part.json").read_text(encoding="utf-8"))
    if data.get("format") != "rtt-part":
        raise ValueError(f"{d / 'part.json'} is not a part spec")
    poses = []
    for raw in data["poses"]:
        rank = int(raw["rank"])
        unknown = set(raw["objects"]) - {"narrow", "wide"}
        if unknown:  # the names become file names below; never let a spec point elsewhere
            raise ValueError(f"{d / 'part.json'}: unknown jaw axes {sorted(unknown)}")
        objects = {
            axis: read_mesh_object(d / "objects" / f"pose{rank}_{axis}.npz") for axis in raw["objects"]
        }
        poses.append(
            PartPose(
                rank=int(raw["rank"]),
                pose_index=int(raw["pose_index"]),
                probability=float(raw["probability"]),
                merged=tuple(raw["merged"]),
                extents_mm=tuple(raw["extents_mm"]),
                com_height_mm=float(raw["com_height_mm"]),
                candidates=tuple(
                    GraspCandidate(
                        c["axis"], float(c["offset_mm"]), float(c["width_mm"]), float(c["section_mm2"])
                    )
                    for c in raw["candidates"]
                ),
                objects=objects,
            )
        )
    stl = d / "part_mm.stl"
    mesh = trimesh.load(stl, force="mesh") if stl.exists() else None
    return PartSpec(
        name=data["name"],
        source_name=data["source_name"],
        source_sha256=data["source_sha256"],
        units=data["units"],
        units_detected=bool(data["units_detected"]),
        scale_to_m=float(data["scale_to_m"]),
        n_faces=int(data["n_faces"]),
        n_bodies=int(data["n_bodies"]),
        watertight=bool(data["watertight"]),
        repairs=tuple(data["repairs"]),
        extents_mm=tuple(data["extents_mm"]),
        volume_cm3=float(data["volume_cm3"]),
        material=data["material"],
        density_kg_m3=float(data["density_kg_m3"]),
        mass_g=float(data["mass_g"]),
        mass_is_approximate=bool(data["mass_is_approximate"]),
        max_grip_width_mm=float(data["max_grip_width_mm"]),
        friction_capacity_kg=float(data["friction_capacity_kg"]),
        decomposition=data["decomposition"],
        poses=tuple(poses),
        other_poses=tuple(data["other_poses"]),
        warnings=tuple(data["warnings"]),
        mesh_mm=mesh if isinstance(mesh, trimesh.Trimesh) else None,
    )
