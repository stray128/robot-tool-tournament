"""Mesh objects: load an STL/OBJ part, rest it in a stable pose, split it into convex pieces.

This is the intake path for "Bring Your Part" uploads and for scanned household
objects used as a real-object held-out set. :func:`load_mesh_object` turns a mesh
file into a :class:`~rtt.sim.objects.MeshObject` that the simulator places on the
table like any primitive:

1. **Load and clean** (``trimesh``): merge duplicate vertices, drop degenerate faces,
   apply ``scale`` (mesh units to metres; STL parts are usually millimetres, so pass
   ``scale=0.001``). Objects larger than 0.5 m are rejected as a units mistake.
2. **Stable pose** (``trimesh.poses.compute_stable_poses``): the resting orientations
   on a plane, ranked by the probability that a randomly dropped part lands in them.
   ``pose_index`` picks one (0, the most likely, by default).
3. **Grasp frame**: the part is turned about Z so its narrowest horizontal extent lies
   along Y (the closing axis), then centred on its bounding box.
4. **Convex decomposition**: CoACD (approximate convex decomposition; deterministic for
   a fixed ``seed``) when installed, else one convex hull per connected component
   (``decomposition="hull"`` forces this). MuJoCo collides every piece as a convex
   hull, so a concave part needs the decomposition to be held by its real surface.
   The v2 rule for scans (``min_parts``, ``retry_threshold``, ``pin_coacd``) re-runs
   CoACD at a finer threshold when it returns too few pieces, and pins the CoACD version.
5. **Mass properties** from the watertight mesh (volume, centre of mass and inertia,
   per unit density); for a mesh that does not enclose a volume they come from the
   convex pieces instead (overlaps between pieces are counted twice).

:func:`intake_frame` exposes steps 1-3 as an :class:`IntakeFrame`, which replays the same
rigid steps on another mesh of the object (the textured scan, see :mod:`rtt.sim.gso_visual`).

Decomposition takes from milliseconds (a box) to tens of seconds (a detailed scan),
so :func:`save_mesh_object` / :func:`read_mesh_object` cache the result as ``.npz``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh

from rtt.sim.objects import MAX_OBJECT_SIZE, MeshObject

Decomposition = Literal["auto", "coacd", "hull"]

MIN_PART_VOLUME = 1e-9  # m^3 (1 mm^3); smaller convex pieces are dropped
_YAW_COARSE_DEG = 1.0
_YAW_FINE_DEG = 0.05
_NPZ_FORMAT = "rtt-mesh-object"
_NPZ_VERSION = 1


def coacd_available() -> bool:
    """Whether the CoACD convex decomposition package can be imported."""
    try:
        import coacd  # noqa: F401
    except ImportError:
        return False
    return True


def _clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if mesh.is_watertight:
        mesh.fix_normals()
    return mesh


def _encloses_volume(mesh: trimesh.Trimesh) -> bool:
    return bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > MIN_PART_VOLUME)


def stable_poses(mesh: trimesh.Trimesh) -> list[tuple[np.ndarray, float]]:
    """Stable resting poses of ``mesh`` on a horizontal plane, most probable first.

    Each entry is a 4x4 transform (that rests the mesh on ``z = 0``) and the probability
    of the pose. The centre of mass is the mesh's own when it encloses a volume, else its
    convex hull's. Deterministic: no centre-of-mass sampling.
    """
    com = mesh.center_mass if _encloses_volume(mesh) else mesh.convex_hull.center_mass
    transforms, probs = trimesh.poses.compute_stable_poses(
        mesh, center_mass=com, sigma=0.0, n_samples=1, threshold=0.0, seed=0
    )
    order = sorted(range(len(probs)), key=lambda i: (-float(probs[i]), i))
    return [(np.asarray(transforms[i], dtype=np.float64), float(probs[i])) for i in order]


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    rot = np.eye(4)
    rot[:2, :2] = [[c, -s], [s, c]]
    return rot


def _narrowest_yaw(points: np.ndarray) -> float:
    """Yaw (rad, in [0, pi)) that minimises the Y extent of ``points`` after rotation about Z."""

    def y_extent(theta: np.ndarray) -> np.ndarray:
        y = np.sin(theta)[:, None] * points[None, :, 0] + np.cos(theta)[:, None] * points[None, :, 1]
        return y.max(axis=1) - y.min(axis=1)

    coarse = np.radians(np.arange(0.0, 180.0, _YAW_COARSE_DEG))
    best = float(coarse[int(np.argmin(y_extent(coarse)))])
    fine = best + np.radians(np.arange(-_YAW_COARSE_DEG, _YAW_COARSE_DEG + 1e-9, _YAW_FINE_DEG))
    return float(np.mod(fine[int(np.argmin(y_extent(fine)))], math.pi))


def _decompose_coacd(
    mesh: trimesh.Trimesh, threshold: float, max_parts: int, max_part_vertices: int, seed: int
) -> list[np.ndarray]:
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)),
        threshold=threshold,
        max_convex_hull=max_parts,
        preprocess_mode="auto",
        max_ch_vertex=max_part_vertices,
        seed=seed,
    )
    return [np.asarray(verts, dtype=np.float64) for verts, _ in parts]


def _decompose_hull(mesh: trimesh.Trimesh, max_parts: int) -> list[np.ndarray]:
    """One convex hull per connected component (one hull overall if there are too many)."""
    components = mesh.split(only_watertight=False)
    if not 1 <= len(components) <= max_parts:
        return [np.asarray(mesh.convex_hull.vertices, dtype=np.float64)]
    return [np.asarray(c.convex_hull.vertices, dtype=np.float64) for c in components]


def _hull(points: np.ndarray) -> trimesh.Trimesh | None:
    try:
        hull = trimesh.convex.convex_hull(points)
    except (RuntimeError, ValueError, IndexError):  # qhull (a RuntimeError) rejects flat point sets
        return None
    if len(hull.vertices) < 4 or hull.volume < MIN_PART_VOLUME:
        return None
    return hull


def _mass_properties(
    mesh: trimesh.Trimesh, hulls: Sequence[trimesh.Trimesh]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, centre of mass and inertia tensor about it, per unit density."""
    if _encloses_volume(mesh):
        solid = mesh.copy()
        solid.density = 1.0
        return float(solid.volume), np.asarray(solid.center_mass), np.asarray(solid.moment_inertia)
    vols = [float(h.volume) for h in hulls]
    total = sum(vols)
    com = sum(v * np.asarray(h.center_mass) for v, h in zip(vols, hulls, strict=True)) / total
    inertia = np.zeros((3, 3))
    for v, h in zip(vols, hulls, strict=True):
        part = h.copy()
        part.density = 1.0
        d = np.asarray(part.center_mass) - com
        inertia += np.asarray(part.moment_inertia) + v * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return total, com, inertia


@dataclass(frozen=True, eq=False)
class IntakeFrame:
    """The rigid steps that take a mesh into the grasp frame, in the order intake applies them.

    ``scale`` (file units to metres), then ``pose`` (rest on ``z = 0`` in stable pose
    ``pose_index``, with ``probability``), then ``turn`` (narrowest horizontal extent onto
    y), then ``centre`` (bounding box on the origin). :meth:`apply` replays them on another
    mesh of the same object, e.g. the textured scan, so a vertex that equals one of the
    geometry mesh's lands on bit-identical coordinates.
    """

    scale: float
    pose_index: int
    probability: float
    pose: np.ndarray
    turn: np.ndarray
    centre: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        """The composed 4x4 transform (for exporters; :meth:`apply` replays the steps instead)."""
        return self.centre @ self.turn @ self.pose @ np.diag([self.scale, self.scale, self.scale, 1.0])

    def apply(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        out = mesh.copy()
        out.apply_scale(self.scale)
        for step in (self.pose, self.turn, self.centre):
            out.apply_transform(step)
        return out


def intake_frame(
    mesh: trimesh.Trimesh, *, name: str, scale: float = 1.0, pose_index: int = 0
) -> tuple[IntakeFrame, trimesh.Trimesh]:
    """Clean and scale ``mesh``, rest it in stable pose ``pose_index`` and move it into the
    grasp frame (module docstring, steps 1-3): the frame and the rested, cleaned mesh."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    mesh = _clean(mesh)
    if len(mesh.faces) < 4:
        raise ValueError(f"mesh {name!r} has fewer than 4 usable faces")
    mesh.apply_scale(scale)
    if float(mesh.extents.max()) > MAX_OBJECT_SIZE:
        raise ValueError(
            f"mesh {name!r} is {mesh.extents.max():g} m across after scale={scale:g}; sizes must be in metres "
            f"(at most {MAX_OBJECT_SIZE} m); pass scale=0.001 for a mesh authored in millimetres"
        )

    poses = stable_poses(mesh)
    if not 0 <= pose_index < len(poses):
        raise ValueError(
            f"mesh {name!r} has {len(poses)} stable poses; pose_index {pose_index} is out of range"
        )
    pose, probability = poses[pose_index]
    rested = mesh.copy()
    rested.apply_transform(pose)
    turn = _rot_z(_narrowest_yaw(np.asarray(rested.convex_hull.vertices)))
    rested.apply_transform(turn)
    lo, hi = rested.bounds
    centre = np.eye(4)
    centre[:3, 3] = -(lo + hi) / 2.0
    rested.apply_transform(centre)
    return IntakeFrame(float(scale), int(pose_index), probability, pose, turn, centre), rested


def coacd_version() -> str | None:
    """The installed CoACD version, or ``None`` if it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("coacd")
    except PackageNotFoundError:
        return None


def decomposition_label(threshold: float, seed: int, version: str | None) -> str:
    """The ``decomposition`` string of a pinned v2 CoACD run, e.g. ``coacd-1.0.14:t=0.02:seed=0``."""
    return f"coacd-{version or 'unknown'}:t={threshold:g}:seed={seed}"


def mesh_object_from_trimesh(
    mesh: trimesh.Trimesh,
    *,
    name: str,
    scale: float = 1.0,
    density: float | None = None,
    pose_index: int = 0,
    decomposition: Decomposition = "auto",
    threshold: float = 0.05,
    max_parts: int = 16,
    max_part_vertices: int = 48,
    seed: int = 0,
    source_sha256: str = "",
    min_parts: int = 1,
    retry_threshold: float | None = None,
    pin_coacd: str | None = None,
) -> MeshObject:
    """Build a :class:`MeshObject` from an in-memory mesh (see the module docstring).

    v2 decomposition rule (all three default to off, which is the v1 behaviour): when CoACD
    returns fewer than ``min_parts`` solid pieces, it runs again at ``retry_threshold`` and
    that result is kept whatever its count; ``pin_coacd`` refuses to run unless that exact
    CoACD version is installed (hulls differ between versions). With any of them set the
    object's ``decomposition`` records the version, the threshold used and the seed.
    """
    if max_parts < 1 or max_part_vertices < 4 or min_parts < 1:
        raise ValueError("max_parts and min_parts must be >= 1 and max_part_vertices >= 4")
    frame, rested = intake_frame(mesh, name=name, scale=scale, pose_index=pose_index)
    probability = frame.probability

    v2_rule = min_parts > 1 or retry_threshold is not None or pin_coacd is not None
    label: str
    if decomposition == "auto":
        decomposition = "coacd" if coacd_available() else "hull"
    if decomposition == "coacd":
        installed = coacd_version()
        if pin_coacd is not None and installed != pin_coacd:
            raise RuntimeError(
                f"CoACD {pin_coacd} is pinned for this decomposition, but {installed} is installed"
            )
        hulls = _solid_hulls(_decompose_coacd(rested, threshold, max_parts, max_part_vertices, seed))
        used = threshold
        if len(hulls) < min_parts and retry_threshold is not None:
            hulls = _solid_hulls(
                _decompose_coacd(rested, retry_threshold, max_parts, max_part_vertices, seed)
            )
            used = retry_threshold
        label = decomposition_label(used, seed, installed) if v2_rule else "coacd"
    elif decomposition == "hull":
        hulls = _solid_hulls(_decompose_hull(rested, max_parts))
        label = "hull"
    else:
        raise ValueError(f"unknown decomposition {decomposition!r}")
    if not hulls:
        hull = _hull(np.asarray(rested.vertices))
        if hull is None:
            raise ValueError(f"mesh {name!r} is flat or degenerate; it has no solid convex pieces")
        hulls = [hull]
    hulls.sort(key=lambda h: (-float(h.volume), tuple(np.round(h.center_mass, 9))))

    vol, com, inertia = _mass_properties(rested, hulls)
    full = (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
    return MeshObject(
        name=name,
        parts=tuple(np.ascontiguousarray(h.vertices, dtype=np.float64) for h in hulls),
        faces=tuple(np.ascontiguousarray(h.faces, dtype=np.int32) for h in hulls),
        volume=vol,
        com=(float(com[0]), float(com[1]), float(com[2])),
        unit_inertia=tuple(float(v) for v in full),  # type: ignore[arg-type]
        density=density,
        source_sha256=source_sha256,
        pose_index=pose_index,
        pose_probability=probability,
        decomposition=label,
    )


def _solid_hulls(pieces: Sequence[np.ndarray]) -> list[trimesh.Trimesh]:
    return [h for h in (_hull(p) for p in pieces) if h is not None]


def load_mesh_object(
    path: str | Path,
    *,
    scale: float = 1.0,
    name: str | None = None,
    density: float | None = None,
    pose_index: int = 0,
    decomposition: Decomposition = "auto",
    threshold: float = 0.05,
    max_parts: int = 16,
    max_part_vertices: int = 48,
    seed: int = 0,
    min_parts: int = 1,
    retry_threshold: float | None = None,
    pin_coacd: str | None = None,
) -> MeshObject:
    """Load an STL/OBJ/PLY file as a simulator object resting in a stable pose.

    ``scale`` converts file units to metres (0.001 for millimetres). ``threshold`` is
    CoACD's concavity tolerance (smaller: more, tighter pieces); ``max_parts`` caps the
    number of convex pieces and ``max_part_vertices`` their vertex count. The object's
    name defaults to the file stem; ``source_sha256`` records the file's hash.
    """
    p = Path(path)
    raw = p.read_bytes()
    mesh = trimesh.load(p, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{p}: no triangle mesh found")
    return mesh_object_from_trimesh(
        mesh,
        name=name or p.stem,
        scale=scale,
        density=density,
        pose_index=pose_index,
        decomposition=decomposition,
        threshold=threshold,
        max_parts=max_parts,
        max_part_vertices=max_part_vertices,
        seed=seed,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        min_parts=min_parts,
        retry_threshold=retry_threshold,
        pin_coacd=pin_coacd,
    )


def save_mesh_object(obj: MeshObject, path: str | Path) -> Path:
    """Cache a decomposed object as ``.npz`` (pieces, faces and a JSON metadata block)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "format": _NPZ_FORMAT,
        "version": _NPZ_VERSION,
        "name": obj.name,
        "volume": obj.volume,
        "com": list(obj.com),
        "unit_inertia": list(obj.unit_inertia),
        "density": obj.density,
        "source_sha256": obj.source_sha256,
        "pose_index": obj.pose_index,
        "pose_probability": obj.pose_probability,
        "decomposition": obj.decomposition,
        "n_parts": len(obj.parts),
    }
    arrays: dict[str, np.ndarray] = {"meta": np.array(json.dumps(meta))}
    for i, (verts, tris) in enumerate(zip(obj.parts, obj.faces, strict=True)):
        arrays[f"verts_{i:03d}"] = verts
        arrays[f"faces_{i:03d}"] = tris
    with open(p, "wb") as fh:
        np.savez_compressed(fh, **arrays)  # type: ignore[arg-type]
    return p


def read_mesh_object(path: str | Path) -> MeshObject:
    """Load an object written by :func:`save_mesh_object`."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("format") != _NPZ_FORMAT:
            raise ValueError(f"{path}: not an {_NPZ_FORMAT} file")
        n = int(meta["n_parts"])
        parts = tuple(np.asarray(data[f"verts_{i:03d}"], dtype=np.float64) for i in range(n))
        faces = tuple(np.asarray(data[f"faces_{i:03d}"], dtype=np.int32) for i in range(n))
    return MeshObject(
        name=meta["name"],
        parts=parts,
        faces=faces,
        volume=float(meta["volume"]),
        com=tuple(meta["com"]),  # type: ignore[arg-type]
        unit_inertia=tuple(meta["unit_inertia"]),  # type: ignore[arg-type]
        density=meta["density"],
        source_sha256=meta["source_sha256"],
        pose_index=int(meta["pose_index"]),
        pose_probability=float(meta["pose_probability"]),
        decomposition=meta["decomposition"],
    )
