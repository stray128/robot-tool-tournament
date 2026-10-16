"""Frozen v2 contract for parts: exact convex pieces, registration symmetry, features, material.

A :class:`PartSpec` is one part in one resting pose, as the v2 generator, scene and scorer
see it. It carries every field of :class:`rtt.sim.objects.MeshObject` (same names, meaning
and units, with ``pieces`` as the name of the convex vertex sets) plus what v2 needs: the
registration symmetry, the declared features, the material and an optional visual asset.
:meth:`PartSpec.to_mesh_object` hands it to the v1 scene path unchanged.

Nothing here imports MuJoCo, trimesh or manifold3d.

Object frame (as for ``MeshObject``): metres, z up, the bounding box centred on the origin,
so the part rests on the table when its centre is ``size[2] / 2`` above it. y is the closing
axis: the left jaw approaches from +y and the right jaw from -y. The part is posed the way it
is gripped (for example the tee lies flat with its branch toward +y, the left jaw).

This package (procedural part library) is separate from :mod:`rtt.part`, the product-mode
intake of uploaded parts.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from rtt.sim.objects import MeshObject

FeatureKind = Literal["socket", "flange"]
FEATURE_KINDS: tuple[FeatureKind, ...] = ("socket", "flange")
RegisterAxis = Literal["x", "y", "yaw", "tilt"]
REGISTER_AXES: tuple[RegisterAxis, ...] = ("x", "y", "yaw", "tilt")
JawSide = Literal["left", "right"]
MAX_PART_SIZE = 0.5  # m; the same limit as rtt.sim.objects.MAX_OBJECT_SIZE

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ASSET_ID = re.compile(r"^[0-9a-f]{16}$")


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _vector(name: str, value: object, n: int) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence | np.ndarray) or len(value) != n:
        raise ValueError(f"{name} must be {n} numbers, got {value!r}")
    return tuple(_finite(f"{name}[{i}]", v) for i, v in enumerate(value))


def _unit_fraction(name: str, value: object) -> float:
    v = _finite(name, value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {v}")
    return v


def _check_name(what: str, value: object) -> None:
    if not isinstance(value, str) or not _NAME.match(value):
        raise ValueError(f"{what} must match {_NAME.pattern} (letters, digits, '_', '.', '-'), got {value!r}")


@dataclass(frozen=True)
class Material:
    """What a part is made of: its density (sets the mass) and its PBR look.

    * ``name``: a short identifier, for example ``"pvc_white"`` or ``"brass"``.
    * ``density``: kg/m^3. The part's mass is ``volume * density``.
    * ``base_color``: sRGB in [0, 1] (viewers convert to linear before lighting).
    * ``metallic``, ``roughness``: PBR factors in [0, 1].

    A material never changes friction: scored physics uses the task's friction for every part.
    """

    name: str
    density: float
    base_color: tuple[float, float, float]
    metallic: float = 0.0
    roughness: float = 0.5

    def __post_init__(self) -> None:
        _check_name("material name", self.name)
        density = _finite("density", self.density)
        if density <= 0:
            raise ValueError(f"density must be positive, got {density}")
        object.__setattr__(self, "density", density)
        color = _vector("base_color", self.base_color, 3)
        for i, c in enumerate(color):
            _unit_fraction(f"base_color[{i}]", c)
        object.__setattr__(self, "base_color", color)
        object.__setattr__(self, "metallic", _unit_fraction("metallic", self.metallic))
        object.__setattr__(self, "roughness", _unit_fraction("roughness", self.roughness))


@dataclass(frozen=True)
class Feature:
    """A declared geometric feature of a part, in the object frame.

    * ``kind``: ``"socket"`` (a blind or through bore a locator pin can enter) or
      ``"flange"`` (a rim a hook tip can catch under).
    * ``name``: unique within the part, for example ``"branch_socket"``.
    * ``origin``: m; the centre of the socket mouth, or the centre of the flange's free face.
    * ``axis``: unit vector out of the part: out of the socket mouth, or the flange face's
      outward normal.
    * ``radius``: m; the bore radius at the mouth, or the flange's outer radius.
    * ``depth``: m; from the mouth to the bottom of the socket, or the flange's thickness.
    """

    kind: FeatureKind
    name: str
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius: float
    depth: float

    def __post_init__(self) -> None:
        if self.kind not in FEATURE_KINDS:
            raise ValueError(f"feature kind must be one of {FEATURE_KINDS}, got {self.kind!r}")
        _check_name("feature name", self.name)
        object.__setattr__(self, "origin", _vector("origin", self.origin, 3))
        axis = _vector("axis", self.axis, 3)
        norm = math.sqrt(sum(a * a for a in axis))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(f"axis must be a unit vector, got norm {norm}")
        object.__setattr__(self, "axis", axis)
        for name in ("radius", "depth"):
            value = _finite(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            object.__setattr__(self, name, value)

    def facing(self, tol_deg: float = 10.0) -> JawSide | None:
        """The jaw this feature faces: ``"left"`` if its axis is within ``tol_deg`` of +y,
        ``"right"`` if within ``tol_deg`` of -y, else ``None``."""
        cos_tol = math.cos(math.radians(tol_deg))
        if self.axis[1] >= cos_tol:
            return "left"
        if -self.axis[1] >= cos_tol:
            return "right"
        return None


@dataclass(frozen=True, eq=False)
class PartSpec:
    """One part in one resting pose: exact convex pieces plus its v2 description.

    ``MeshObject`` fields (same meaning and units):

    * ``name``: the part's identifier, also its :attr:`part_id` in records, replays and
      :class:`~rtt.design.freeform.types.FingerPair`. Unique per instance (a held-out
      instance at a new scale gets its own name).
    * ``pieces``: convex vertex sets, each ``(n, 3)`` float64 with ``n >= 4``, metres, object
      frame (``MeshObject.parts``; :attr:`parts` is an alias). Procedural parts are exact
      pieces by construction; scans are CoACD hulls.
    * ``faces``: ``(m, 3)`` triangle indices per piece, counter-clockwise seen from outside.
    * ``volume`` (m^3), ``com`` (object frame) and ``unit_inertia`` (``Ixx Iyy Izz Ixy Ixz
      Iyz`` about the centre of mass, divided by density) of the solid.
    * ``source_sha256``, ``pose_index``, ``pose_probability``, ``decomposition``: as for
      ``MeshObject`` (``decomposition`` is ``"exact"`` for procedural parts).

    The density is the material's (:attr:`density`), so ``MeshObject.density`` is not a field.

    v2 fields:

    * ``family``: the part family, for example ``"pipe_tee"``, ``"block"`` or ``"gso"``.
    * ``material``: the :class:`Material`.
    * ``yaw_symmetry``: n-fold symmetry about z in this pose (1 = none, 6 for a hex), or 0
      for a part that is symmetric under every yaw. Yaw registration is taken modulo it
      (:meth:`yaw_residual_deg`).
    * ``register_axes``: the parts of the registration rule that apply to this part, a
      subset of :data:`REGISTER_AXES`. A part with ``yaw_symmetry == 0`` cannot list ``"yaw"``.
    * ``features``: the declared :class:`Feature` s (sockets, flanges).
    * ``visual_asset``: the asset-manifest id (sha256[:16] of the GLB) of the part's visual
      mesh, or ``None`` to draw the collision pieces in the material.
    """

    name: str
    family: str
    pieces: tuple[np.ndarray, ...] = field(repr=False)
    faces: tuple[np.ndarray, ...] = field(repr=False)
    volume: float
    com: tuple[float, float, float]
    unit_inertia: tuple[float, float, float, float, float, float]
    material: Material
    yaw_symmetry: int = 1
    register_axes: tuple[RegisterAxis, ...] = REGISTER_AXES
    features: tuple[Feature, ...] = ()
    visual_asset: str | None = None
    source_sha256: str = ""
    pose_index: int = 0
    pose_probability: float = 1.0
    decomposition: str = ""

    def __post_init__(self) -> None:
        _check_name("part name", self.name)
        _check_name("part family", self.family)
        self._check_pieces()
        volume = _finite("volume", self.volume)
        if volume <= 0:
            raise ValueError(f"part {self.name!r} has non-positive volume {volume}")
        object.__setattr__(self, "volume", volume)
        object.__setattr__(self, "com", _vector("com", self.com, 3))
        object.__setattr__(self, "unit_inertia", _vector("unit_inertia", self.unit_inertia, 6))
        if not isinstance(self.material, Material):
            raise TypeError("material must be a Material")
        self._check_registration()
        features = tuple(self.features)
        if not all(isinstance(f, Feature) for f in features):
            raise TypeError("features must be Feature instances")
        names = [f.name for f in features]
        if len(set(names)) != len(names):
            raise ValueError(f"feature names must be unique within a part, got {names}")
        object.__setattr__(self, "features", features)
        if self.visual_asset is not None and (
            not isinstance(self.visual_asset, str) or not _ASSET_ID.match(self.visual_asset)
        ):
            raise ValueError(
                f"visual_asset must be an asset id (16 lowercase hex), got {self.visual_asset!r}"
            )
        if isinstance(self.pose_index, bool) or not isinstance(self.pose_index, numbers.Integral):
            raise TypeError(f"pose_index must be an integer, got {self.pose_index!r}")
        prob = _finite("pose_probability", self.pose_probability)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"pose_probability must be in [0, 1], got {prob}")
        object.__setattr__(self, "pose_probability", prob)

    def _check_pieces(self) -> None:
        if isinstance(self.pieces, np.ndarray) or not isinstance(self.pieces, Sequence) or not self.pieces:
            raise ValueError("a part needs a non-empty sequence of convex pieces")
        if len(self.pieces) != len(self.faces):
            raise ValueError(f"{len(self.pieces)} pieces but {len(self.faces)} face arrays")
        pieces, faces = [], []
        for i, (verts, tris) in enumerate(zip(self.pieces, self.faces, strict=True)):
            v = np.array(verts, dtype=np.float64)
            t = np.array(tris)
            if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.all(np.isfinite(v)):
                raise ValueError(f"piece {i} needs at least 4 finite vertices of shape (n, 3), got {v.shape}")
            if t.ndim != 2 or t.shape[1] != 3 or not np.issubdtype(t.dtype, np.integer):
                raise ValueError(f"piece {i} needs (m, 3) integer faces, got {t.shape} {t.dtype}")
            if t.size == 0 or int(t.min()) < 0 or int(t.max()) >= len(v):
                raise ValueError(f"piece {i} faces must index its {len(v)} vertices")
            t = t.astype(np.int64)
            v.setflags(write=False)
            t.setflags(write=False)
            pieces.append(v)
            faces.append(t)
        object.__setattr__(self, "pieces", tuple(pieces))
        object.__setattr__(self, "faces", tuple(faces))
        extent = self.size
        if any(s <= 0 for s in extent):
            raise ValueError(f"part {self.name!r} is degenerate (extents {extent})")
        if any(s > MAX_PART_SIZE for s in extent):
            raise ValueError(f"part {self.name!r} is {max(extent):g} m across; sizes are in metres")

    def _check_registration(self) -> None:
        sym = self.yaw_symmetry
        if isinstance(sym, bool) or not isinstance(sym, numbers.Integral) or sym < 0:
            raise ValueError(f"yaw_symmetry must be an integer >= 0, got {sym!r}")
        object.__setattr__(self, "yaw_symmetry", int(sym))
        axes = tuple(self.register_axes)
        if not axes or len(set(axes)) != len(axes) or any(a not in REGISTER_AXES for a in axes):
            raise ValueError(f"register_axes must be a non-empty, duplicate-free subset of {REGISTER_AXES}")
        if sym == 0 and "yaw" in axes:
            raise ValueError("a part symmetric under every yaw (yaw_symmetry 0) cannot register yaw")
        object.__setattr__(self, "register_axes", axes)

    @property
    def part_id(self) -> str:
        return self.name

    @property
    def parts(self) -> tuple[np.ndarray, ...]:
        """``MeshObject``-style alias of :attr:`pieces`."""
        return self.pieces

    @property
    def density(self) -> float:
        return self.material.density

    @property
    def mass(self) -> float:
        return self.volume * self.material.density

    @property
    def size(self) -> tuple[float, float, float]:
        """Full bounding-box extents ``(sx, sy, sz)`` in the object frame."""
        pts = np.concatenate(self.pieces)
        ext = pts.max(axis=0) - pts.min(axis=0)
        return (float(ext[0]), float(ext[1]), float(ext[2]))

    @property
    def sockets(self) -> tuple[Feature, ...]:
        return tuple(f for f in self.features if f.kind == "socket")

    def yaw_residual_deg(self, yaw_error_deg: float) -> float:
        """The absolute yaw error in degrees once the part's yaw symmetry is taken out.

        With ``yaw_symmetry = n >= 1`` the error is wrapped to ``[-180/n, 180/n)`` first; with
        ``yaw_symmetry = 0`` every yaw is equivalent and the residual is 0.
        """
        err = _finite("yaw_error_deg", yaw_error_deg)
        if self.yaw_symmetry == 0:
            return 0.0
        period = 360.0 / self.yaw_symmetry
        return abs((err + period / 2.0) % period - period / 2.0)

    @property
    def geometry_sha256(self) -> str:
        """SHA-256 (hex) of the simulated geometry and mass properties.

        The same digest as :attr:`rtt.sim.objects.MeshObject.geometry_sha256` of
        :meth:`to_mesh_object`: every piece (vertices as little-endian float64, faces as
        int32, in order), then the volume, centre of mass and unit inertia.
        """
        h = hashlib.sha256()
        for verts, tris in zip(self.pieces, self.faces, strict=True):
            h.update(np.asarray(verts.shape, dtype="<i8").tobytes())
            h.update(np.ascontiguousarray(verts, dtype="<f8").tobytes())
            h.update(np.asarray(tris.shape, dtype="<i8").tobytes())
            h.update(np.ascontiguousarray(tris, dtype="<i4").tobytes())
        mass = np.array([self.volume, *self.com, *self.unit_inertia], dtype="<f8")
        h.update(mass.tobytes())
        return h.hexdigest()

    def to_mesh_object(self) -> MeshObject:
        """The part as a v1 :class:`rtt.sim.objects.MeshObject` (imports MuJoCo on first use)."""
        from rtt.sim.objects import MeshObject

        return MeshObject(
            name=self.name,
            parts=self.pieces,
            faces=self.faces,
            volume=self.volume,
            com=self.com,
            unit_inertia=self.unit_inertia,
            density=self.material.density,
            source_sha256=self.source_sha256,
            pose_index=int(self.pose_index),
            pose_probability=self.pose_probability,
            decomposition=self.decomposition,
        )
