"""Graspable test objects and their placement on the table.

Objects are free rigid bodies of uniform density resting on a static table.
Three primitive families are supported by :class:`ObjectSpec`:

* ``box``: ``size = (sx, sy, sz)`` full extents in the object frame.
* ``cylinder``: ``size = (radius, height)``, standing upright (axis along Z).
* ``rod``: a capsule ``size = (radius, length)`` (tip to tip) lying on the table
  with its axis along the object-frame X axis.

Arbitrary shapes (an uploaded part, a scanned household object) are a
:class:`MeshObject`: convex pieces of the mesh, already turned into one of its
stable resting poses (see :mod:`rtt.sim.mesh`, which builds them). Its object frame
is centred on the bounding box, so it is used exactly like a box of size
``MeshObject.size``.

The gripper closes along the object-frame Y axis once the palm yaw matches the
placement yaw, so :func:`grip_width` (the Y extent) is the width the fingers see.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np

ObjectKind = Literal["box", "cylinder", "rod"]
MAX_OBJECT_SIZE = 0.5  # m; larger values almost certainly mean millimetres were passed

_ROD_QUAT = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # capsule axis Z -> X


@dataclass(frozen=True)
class ObjectSpec:
    """Shape of one primitive test object; density defaults to the task's object density."""

    kind: ObjectKind
    size: tuple[float, ...]
    density: float | None = None

    def __post_init__(self) -> None:
        expected = {"box": 3, "cylinder": 2, "rod": 2}
        if self.kind not in expected:
            raise ValueError(f"unknown object kind {self.kind!r}")
        if len(self.size) != expected[self.kind]:
            raise ValueError(f"{self.kind} needs {expected[self.kind]} size values, got {self.size}")
        if any(s <= 0 for s in self.size):
            raise ValueError(f"object sizes must be positive, got {self.size}")
        if any(s > MAX_OBJECT_SIZE for s in self.size):
            raise ValueError(f"object sizes are in metres (at most {MAX_OBJECT_SIZE}), got {self.size}")
        if self.kind == "rod" and self.size[1] < 2 * self.size[0]:
            raise ValueError("rod length must be at least its diameter")
        if self.density is not None and self.density <= 0:
            raise ValueError(f"density must be positive, got {self.density}")

    @property
    def label(self) -> str:
        """Short human-readable description, sizes in millimetres."""
        return f"{self.kind} " + "x".join(f"{1000.0 * s:g}" for s in self.size)


@dataclass(frozen=True, eq=False)
class MeshObject:
    """A rigid object given by convex pieces of a mesh, in one stable resting pose.

    Coordinates are metres in the object frame: Z up, the bounding box centred on the
    origin, and the narrowest horizontal extent along Y (the closing axis), so the
    object rests on the table when its centre is ``size[2] / 2`` above it.

    Mass properties are stored per unit density (``volume`` in m^3 and the inertia
    tensor about the centre of mass divided by density), so the task's object density
    applies unless ``density`` is set. Build instances with
    :func:`rtt.sim.mesh.load_mesh_object`.
    """

    name: str
    parts: tuple[np.ndarray, ...] = field(repr=False)  # (n_i, 3) convex-hull vertices per piece
    faces: tuple[np.ndarray, ...] = field(repr=False)  # (m_i, 3) triangle indices per piece
    volume: float  # m^3 of the solid
    com: tuple[float, float, float]
    unit_inertia: tuple[float, float, float, float, float, float]  # Ixx Iyy Izz Ixy Ixz Iyz / density
    density: float | None = None
    source_sha256: str = ""
    pose_index: int = 0
    pose_probability: float = 1.0
    decomposition: str = ""
    kind: Literal["mesh"] = "mesh"

    def __post_init__(self) -> None:
        if not self.parts or len(self.parts) != len(self.faces):
            raise ValueError("a mesh object needs at least one convex piece, each with faces")
        for verts, tris in zip(self.parts, self.faces, strict=True):
            if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 4:
                raise ValueError("every convex piece needs at least 4 vertices of shape (n, 3)")
            if tris.ndim != 2 or tris.shape[1] != 3 or int(tris.max()) >= len(verts):
                raise ValueError("every convex piece needs (m, 3) faces indexing its vertices")
        extent = self.size
        if any(s <= 0 for s in extent):
            raise ValueError(f"mesh object {self.name!r} is degenerate (extents {extent})")
        if any(s > MAX_OBJECT_SIZE for s in extent):
            raise ValueError(
                f"mesh object {self.name!r} is {max(extent):g} m across; sizes are in metres (at most "
                f"{MAX_OBJECT_SIZE}); pass scale=0.001 for a mesh authored in millimetres"
            )
        if self.volume <= 0:
            raise ValueError(f"mesh object {self.name!r} has non-positive volume {self.volume}")
        if self.density is not None and self.density <= 0:
            raise ValueError(f"density must be positive, got {self.density}")

    @property
    def size(self) -> tuple[float, float, float]:
        """Full bounding-box extents ``(sx, sy, sz)`` in the object frame."""
        pts = np.concatenate(self.parts)
        ext = pts.max(axis=0) - pts.min(axis=0)
        return (float(ext[0]), float(ext[1]), float(ext[2]))

    @property
    def label(self) -> str:
        return f"mesh {self.name} " + "x".join(f"{1000.0 * s:.0f}" for s in self.size)

    @property
    def geometry_sha256(self) -> str:
        """SHA-256 (hex) of the simulated geometry and mass properties.

        Covers every convex piece (vertices as little-endian float64, faces as int32, in
        order), the volume, centre of mass and unit inertia. Two objects with the same
        source file, size and name but a different stable pose or decomposition (piece
        count, CoACD threshold) collide differently, so suite digests must include this.
        """
        h = hashlib.sha256()
        for verts, tris in zip(self.parts, self.faces, strict=True):
            h.update(np.asarray(verts.shape, dtype="<i8").tobytes())
            h.update(np.ascontiguousarray(verts, dtype="<f8").tobytes())
            h.update(np.asarray(tris.shape, dtype="<i8").tobytes())
            h.update(np.ascontiguousarray(tris, dtype="<i4").tobytes())
        mass = np.array([self.volume, *self.com, *self.unit_inertia], dtype="<f8")
        h.update(mass.tobytes())
        return h.hexdigest()


AnyObject = ObjectSpec | MeshObject


@dataclass(frozen=True)
class Placement:
    """Nominal object pose on the table, plus a pose error the controller does not know.

    ``x``, ``y`` (m) and ``yaw`` (rad) are where the controller believes the object is
    and aims the gripper. ``err_x``/``err_y`` (m) and ``err_yaw`` (rad) displace the
    real object from that pose, in the nominal object frame, which is the gripper frame
    at the grasp: x along the finger width, y along the closing axis. Per-episode seed
    jitter is added on top (see :func:`rtt.sim.episode.sample_jitter`).
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    err_x: float = 0.0
    err_y: float = 0.0
    err_yaw: float = 0.0

    def actual(self, dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0) -> Placement:
        """The real object pose (no error left) after the pose error and a world-frame shift."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Placement(
            x=self.x + c * self.err_x - s * self.err_y + dx,
            y=self.y + s * self.err_x + c * self.err_y + dy,
            yaw=self.yaw + self.err_yaw + dyaw,
        )


def height(obj: AnyObject) -> float:
    """Vertical extent of the object when resting on the table."""
    if obj.kind in ("box", "mesh"):
        return obj.size[2]
    if obj.kind == "cylinder":
        return obj.size[1]
    return 2.0 * obj.size[0]


def grip_width(obj: AnyObject) -> float:
    """Extent along the object-frame Y axis, i.e. the width between the fingers."""
    if obj.kind in ("box", "mesh"):
        return obj.size[1]
    return 2.0 * obj.size[0]


def half_extents(obj: AnyObject) -> tuple[float, float, float]:
    """Half extents of the object's axis-aligned bounding box in its own frame."""
    if obj.kind in ("box", "mesh"):
        sx, sy, sz = obj.size
        return (sx / 2, sy / 2, sz / 2)
    if obj.kind == "cylinder":
        r, h = obj.size
        return (r, r, h / 2)
    r, length = obj.size
    return (length / 2, r, r)


def volume(obj: AnyObject) -> float:
    """Solid volume in m^3."""
    if isinstance(obj, MeshObject):
        return obj.volume
    if obj.kind == "box":
        sx, sy, sz = obj.size
        return sx * sy * sz
    if obj.kind == "cylinder":
        r, h = obj.size
        return math.pi * r * r * h
    r, length = obj.size
    return math.pi * r * r * (length - 2 * r) + 4.0 / 3.0 * math.pi * r**3


def geom_params(obj: ObjectSpec) -> tuple[mujoco.mjtGeom, list[float], tuple[float, float, float, float]]:
    """MuJoCo geom type, size vector and local orientation (wxyz) for a primitive object."""
    if obj.kind == "box":
        return mujoco.mjtGeom.mjGEOM_BOX, list(half_extents(obj)), (1.0, 0.0, 0.0, 0.0)
    if obj.kind == "cylinder":
        r, h = obj.size
        return mujoco.mjtGeom.mjGEOM_CYLINDER, [r, h / 2, 0.0], (1.0, 0.0, 0.0, 0.0)
    if obj.kind == "rod":
        r, length = obj.size
        return mujoco.mjtGeom.mjGEOM_CAPSULE, [r, length / 2 - r, 0.0], _ROD_QUAT
    raise ValueError(f"{obj.kind!r} objects are not a single primitive geom")


def yaw_quat(yaw: float) -> np.ndarray:
    """Quaternion (wxyz) for a rotation of ``yaw`` radians about +Z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def rest_pose(obj: AnyObject, placement: Placement, table_top_z: float) -> tuple[np.ndarray, np.ndarray]:
    """Free-joint position and quaternion (wxyz) with the object resting on the table.

    ``placement`` is taken literally (its pose-error fields are ignored); pass
    ``placement.actual(...)`` for the real object pose.
    """
    pos = np.array([placement.x, placement.y, table_top_z + height(obj) / 2])
    return pos, yaw_quat(placement.yaw)
