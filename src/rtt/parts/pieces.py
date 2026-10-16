"""Exact convex pieces for procedural parts: prisms of convex polygons, built analytically.

Every procedural part is a set of convex pieces, each a right prism of a strictly convex
polygon (boxes, regular n-gon prisms, ring wedges, gear teeth). Because the pieces are
built analytically, their vertices, faces and mass properties are pure float64 functions of
the part's dimensions: no qhull, no CoACD and no mesh boolean is involved, so the geometry
hash does not depend on a library version or on the CPU.

* :class:`Piece` holds one prism: vertices (rounded to :data:`QUANTUM`), triangle faces
  (counter-clockwise seen from outside) and its polygon faces and halfspaces.
* :func:`prism` builds one from a 2D polygon and an axis; :func:`ngon` and
  :func:`ring_wedges` are the common cross-sections (polygons are circumscribed about the
  nominal circle, so flats sit exactly at the nominal radius: a part is never smaller than
  nominal outside, and a bore never smaller than nominal inside).
* :func:`mass_properties` gives the volume, centre of mass and inertia of the union of the
  pieces. Pieces may overlap (a pipe tee's branch core runs into its run core): overlaps are
  removed exactly by inclusion-exclusion over convex intersections, computed by clipping one
  piece by the other's halfspaces. Sums use :func:`math.fsum`, and results are rounded to
  12 significant digits, so they are identical on every machine.
* :func:`union_mesh` is the watertight union (manifold3d) for visuals and checks only; it
  never feeds a hash.

Units are metres throughout.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import trimesh

Axis = Literal["x", "y", "z"]

QUANTUM_DECIMALS = 7
QUANTUM = 10.0**-QUANTUM_DECIMALS  # m; vertices are rounded to 0.1 micrometre
MIN_EDGE = 1e-5  # m; polygon edges shorter than this are merged
CONTACT_TOL = 4 * QUANTUM  # m; pieces this far into each other still only touch
MIN_OVERLAP_VOLUME = 1e-15  # m^3 (1e-6 mm^3); smaller intersections are contacts, not overlaps
_PLANE_EPS = 1e-12
_MIN_FACE_AREA = 1e-18  # m^2; clipped faces smaller than this are slivers and dropped
_SIG_DIGITS = 12
_COM_FLOOR = 1e-12  # m; a centre-of-mass coordinate smaller than this is zero
_ZERO_REL = 1e-10  # inertia entries below this fraction of the largest moment are zero

# (u, v, w) axis permutation per extrusion axis: the polygon lives in (u, v) and w is the
# extrusion axis, with u x v = w so a polygon counter-clockwise in (u, v) is seen
# counter-clockwise from +w.
_FRAME: dict[str, tuple[int, int, int]] = {"z": (0, 1, 2), "x": (1, 2, 0), "y": (2, 0, 1)}


def quantize(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Round to :data:`QUANTUM` and normalise ``-0.0`` to ``0.0`` (stable bytes for hashing)."""
    return np.round(np.asarray(values, dtype=np.float64), QUANTUM_DECIMALS) + 0.0


def round_sig(value: float, digits: int = _SIG_DIGITS) -> float:
    """``value`` rounded to ``digits`` significant digits (``-0.0`` becomes ``0.0``)."""
    if value == 0.0 or not math.isfinite(value):
        return 0.0 if value == 0.0 else value
    return float(f"{value:.{digits}g}") + 0.0


# --------------------------------------------------------------------------- 2D polygons


def _cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_polygon(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """The strictly convex hull of 2D ``points``, counter-clockwise, collinear points dropped.

    Andrew's monotone chain on lexicographically sorted points: deterministic, no qhull.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError(f"need at least 3 points of shape (n, 2), got {pts.shape}")
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    uniq = [pts[0]]
    for p in pts[1:]:
        if np.hypot(*(p - uniq[-1])) > MIN_EDGE * 1e-3:
            uniq.append(p)
    if len(uniq) < 3:
        raise ValueError("polygon points are (nearly) coincident")

    def half(seq: Iterable[np.ndarray]) -> list[np.ndarray]:
        chain: list[np.ndarray] = []
        for p in seq:
            while len(chain) >= 2 and _cross2(chain[-2], chain[-1], p) <= 1e-18:
                chain.pop()
            chain.append(p)
        return chain

    lower = half(uniq)
    upper = half(reversed(uniq))
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("polygon points are collinear")
    poly = np.array(hull)
    keep = [i for i in range(len(poly)) if np.hypot(*(poly[(i + 1) % len(poly)] - poly[i])) > MIN_EDGE]
    poly = poly[keep]
    if len(poly) < 3:
        raise ValueError("polygon collapses after merging short edges")
    return poly


def polygon_area(poly: np.ndarray) -> float:
    """Signed area (positive for counter-clockwise)."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * math.fsum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def is_strictly_convex(poly: np.ndarray, tol: float = 1e-16) -> bool:
    """Whether a counter-clockwise polygon turns left at every vertex."""
    n = len(poly)
    return n >= 3 and all(_cross2(poly[i - 1], poly[i], poly[(i + 1) % n]) > tol for i in range(n))


def ngon(
    apothem: float, n: int, flat_at_deg: float = 0.0, centre: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Regular ``n``-gon circumscribed about a circle of radius ``apothem``.

    One flat's outward normal points at ``flat_at_deg`` (degrees from +u toward +v); with
    ``n`` divisible by 4 and ``flat_at_deg = 0`` there are flats facing +-u and +-v.
    Counter-clockwise.
    """
    if n < 3 or apothem <= 0:
        raise ValueError(f"need n >= 3 and a positive apothem, got n={n}, apothem={apothem}")
    r = apothem / math.cos(math.pi / n)
    ang = ngon_vertex_angles(n, flat_at_deg)
    return np.stack([centre[0] + r * np.cos(ang), centre[1] + r * np.sin(ang)], axis=1)


def ngon_vertex_angles(n: int, flat_at_deg: float = 0.0) -> np.ndarray:
    """Angles (rad) of the vertices of :func:`ngon`, in order: half a step either side of
    each flat normal, so vertex ``k`` sits at ``flat_at + pi/n + 2 pi k / n``."""
    return math.radians(flat_at_deg) + math.pi / n + np.arange(n) * (2.0 * math.pi / n)


def polygon_radius(poly: np.ndarray, angle: float) -> float:
    """Distance from the origin to the boundary of a convex polygon (containing the origin)
    along the ray at ``angle`` (rad)."""
    d = np.array([math.cos(angle), math.sin(angle)])
    best = math.inf
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        normal = np.array([e[1], -e[0]])  # outward for a counter-clockwise polygon
        denom = float(normal @ d)
        if denom > 1e-15:
            best = min(best, float(normal @ a) / denom)
    if not math.isfinite(best):
        raise ValueError("the ray does not leave the polygon; does it contain the origin?")
    return best


# --------------------------------------------------------------------------- pieces


@dataclass(frozen=True, eq=False)
class Piece:
    """One convex piece: quantised vertices, outward triangle faces, polygon faces, halfspaces.

    ``polys`` are the polygon faces (vertex index lists, counter-clockwise from outside);
    ``normals`` and ``offsets`` the matching halfspaces ``normals @ p <= offsets``.
    """

    verts: np.ndarray = field(repr=False)
    faces: np.ndarray = field(repr=False)
    polys: tuple[tuple[int, ...], ...] = field(repr=False)
    normals: np.ndarray = field(repr=False)
    offsets: np.ndarray = field(repr=False)

    @property
    def bounds(self) -> np.ndarray:
        return np.stack([self.verts.min(axis=0), self.verts.max(axis=0)])

    def polygon_faces(self) -> list[np.ndarray]:
        return [self.verts[list(p)] for p in self.polys]

    def contains(self, points: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """Boolean mask: which ``points`` (``(m, 3)``) lie inside (within ``tol``)."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return np.all(pts @ self.normals.T <= self.offsets + tol, axis=1)

    def translated(self, shift: Sequence[float]) -> Piece:
        return _piece_from_polys(self.verts + np.asarray(shift, dtype=np.float64), self.polys)


def _dot3(points: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Row-wise dot product with a 3-vector, as plain element-wise IEEE arithmetic (no BLAS,
    no fused multiply-add), so the result is bit-identical on every machine."""
    return points[:, 0] * n[0] + points[:, 1] * n[1] + points[:, 2] * n[2]


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ],
        axis=1,
    )


def _newell(poly_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Newell's normal (unnormalised, length = 2 x area) and the area of a planar polygon."""
    a = poly_pts
    b = np.roll(poly_pts, -1, axis=0)
    nx = math.fsum(((a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2])).tolist())
    ny = math.fsum(((a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0])).tolist())
    nz = math.fsum(((a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1])).tolist())
    n = np.array([nx, ny, nz])
    return n, 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _plane(poly_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Outward unit normal and offset of a planar counter-clockwise polygon (Newell's method)."""
    n, area = _newell(poly_pts)
    if area <= 0:
        raise ValueError("degenerate polygon face")
    n = n / (2.0 * area)
    return n, math.fsum(_dot3(poly_pts, n).tolist()) / len(poly_pts)


def _piece_from_polys(verts: np.ndarray, polys: Sequence[Sequence[int]]) -> Piece:
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    polys_t = tuple(tuple(int(i) for i in p) for p in polys)
    tris = [(p[0], p[i], p[i + 1]) for p in polys_t for i in range(1, len(p) - 1)]
    normals, offsets = [], []
    for p in polys_t:
        n, d = _plane(verts[list(p)])
        normals.append(n)
        offsets.append(d)
    faces = np.ascontiguousarray(tris, dtype=np.int32)
    v = verts.copy()
    v.setflags(write=False)
    faces.setflags(write=False)
    nrm = np.array(normals)
    off = np.array(offsets)
    nrm.setflags(write=False)
    off.setflags(write=False)
    return Piece(verts=v, faces=faces, polys=polys_t, normals=nrm, offsets=off)


def prism(polygon: np.ndarray | Sequence[Sequence[float]], axis: Axis, lo: float, hi: float) -> Piece:
    """A right prism: ``polygon`` (2D, convex) in the plane normal to ``axis``, from ``lo`` to
    ``hi`` along it. Polygon coordinates are ``(y, z)`` for axis x, ``(z, x)`` for axis y and
    ``(x, y)`` for axis z. Vertices are quantised; the polygon must stay strictly convex.
    """
    if axis not in _FRAME:
        raise ValueError(f"axis must be x, y or z, got {axis!r}")
    if not hi - lo > MIN_EDGE:
        raise ValueError(f"prism needs hi > lo (got {lo}, {hi})")
    poly = convex_polygon(quantize(polygon))
    if not is_strictly_convex(poly) or polygon_area(poly) <= 0:
        raise ValueError("prism polygon is not strictly convex")
    k = len(poly)
    iu, iv, iw = _FRAME[axis]
    verts = np.zeros((2 * k, 3))
    for layer, w in enumerate(quantize([lo, hi])):
        sl = slice(layer * k, (layer + 1) * k)
        verts[sl, iu] = poly[:, 0]
        verts[sl, iv] = poly[:, 1]
        verts[sl, iw] = w
    verts = quantize(verts)
    bottom = tuple(reversed(range(k)))
    top = tuple(range(k, 2 * k))
    sides = tuple((i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k))
    return _piece_from_polys(verts, (bottom, top, *sides))


def box(lo: Sequence[float], hi: Sequence[float]) -> Piece:
    """An axis-aligned box from corner ``lo`` to corner ``hi``."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    rect = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    return prism(rect, "z", float(lo[2]), float(hi[2]))


def to_axis_plane(axis: Axis, centre: Sequence[float]) -> tuple[float, float]:
    """The ``(u, v)`` coordinates of a 3D point in the polygon plane of ``axis``."""
    iu, iv, _ = _FRAME[axis]
    return (float(centre[iu]), float(centre[iv]))


def ring_wedges(
    r_in: float,
    r_out: float,
    n: int,
    axis: Axis,
    lo: float,
    hi: float,
    centre: Sequence[float] = (0.0, 0.0, 0.0),
    flat_at_deg: float = 0.0,
) -> list[Piece]:
    """A tube wall as ``n`` convex wedges: between the circumscribed ``n``-gons of apothem
    ``r_in`` and ``r_out``, coaxial with ``axis`` through ``centre``, from ``lo`` to ``hi``.

    Each wedge is a trapezoidal prism between two neighbouring vertex rays, so the wedges
    tile the wall exactly with no overlap.
    """
    if not 0 < r_in < r_out:
        raise ValueError(f"need 0 < r_in < r_out, got {r_in}, {r_out}")
    inner = ngon(r_in, n, flat_at_deg)
    outer = ngon(r_out, n, flat_at_deg)
    cu, cv = to_axis_plane(axis, centre)
    c = np.array([cu, cv])
    out = []
    for k in range(n):
        j = (k + 1) % n
        out.append(prism([inner[k] + c, outer[k] + c, outer[j] + c, inner[j] + c], axis, lo, hi))
    return out


def sector_pieces(
    outer: np.ndarray,
    bore_apothem: float,
    n_bore: int,
    axis: Axis,
    lo: float,
    hi: float,
    flat_at_deg: float = 0.0,
) -> list[Piece]:
    """A convex plate ``outer`` (2D polygon around the origin) with a central bore (the
    circumscribed ``n_bore``-gon of apothem ``bore_apothem``), as ``n_bore`` convex sectors
    between neighbouring bore-vertex rays. A sector's outer boundary may span several edges
    of ``outer``; it stays convex because it is the intersection of convex sets."""
    bore = ngon(bore_apothem, n_bore, flat_at_deg)
    ang = np.arctan2(bore[:, 1], bore[:, 0])
    outer_ang = np.arctan2(outer[:, 1], outer[:, 0])
    pieces = []
    for k in range(n_bore):
        j = (k + 1) % n_bore
        a0, a1 = float(ang[k]), float(ang[j])
        span = (a1 - a0) % (2 * math.pi)
        pts = [bore[k], bore[j]]
        for a in (a0, a1):
            r = polygon_radius(outer, a)
            pts.append(np.array([r * math.cos(a), r * math.sin(a)]))
        rel = (outer_ang - a0) % (2 * math.pi)
        for i in np.nonzero((rel > 1e-12) & (rel < span - 1e-12))[0]:
            pts.append(outer[int(i)])
        pieces.append(prism(np.array(pts), axis, lo, hi))
    return pieces


# --------------------------------------------------------------------------- convex clipping


def _clip_polygon(poly: np.ndarray, n: np.ndarray, d: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sutherland-Hodgman: keep the part of a planar polygon with ``n @ p <= d``.

    Returns the clipped polygon (possibly empty) and the points that lie on the plane.
    """
    out: list[np.ndarray] = []
    on_plane: list[np.ndarray] = []
    m = len(poly)
    s = _dot3(poly, n) - d
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        sa, sb = s[i], s[(i + 1) % m]
        a_in = sa <= _PLANE_EPS
        b_in = sb <= _PLANE_EPS
        if a_in:
            out.append(a)
            if abs(sa) <= _PLANE_EPS:
                on_plane.append(a)
        if a_in != b_in:
            t = sa / (sa - sb)
            p = a + t * (b - a)
            out.append(p)
            on_plane.append(p)
    dedup: list[np.ndarray] = []
    for p in out:
        if not dedup or np.max(np.abs(p - dedup[-1])) > 1e-13:
            dedup.append(p)
    if len(dedup) > 1 and np.max(np.abs(dedup[0] - dedup[-1])) <= 1e-13:
        dedup.pop()
    if len(dedup) < 3 or _newell(np.array(dedup))[1] <= _MIN_FACE_AREA:
        return np.zeros((0, 3)), on_plane
    return np.array(dedup), on_plane


def _order_on_plane(points: list[np.ndarray], n: np.ndarray) -> np.ndarray | None:
    """Unique points of a convex polygon on a plane with normal ``n``, counter-clockwise
    seen from ``+n``; ``None`` if fewer than 3 remain."""
    uniq: list[np.ndarray] = []
    for p in points:
        if all(np.linalg.norm(p - q) > 1e-12 for q in uniq):
            uniq.append(p)
    if len(uniq) < 3:
        return None
    pts = np.array(uniq)
    c = np.array([math.fsum(pts[:, i].tolist()) / len(pts) for i in range(3)])
    u = pts[0] - c
    if np.linalg.norm(u) < 1e-15:
        u = pts[1] - c
    u = u / np.linalg.norm(u)
    v = _cross3(n[None, :], u[None, :])[0]
    ang = np.arctan2(_dot3(pts - c, v), _dot3(pts - c, u))
    return pts[np.argsort(ang, kind="stable")]


def clip(faces: list[np.ndarray], normals: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Clip a convex polytope (polygon faces, counter-clockwise from outside) by halfspaces."""
    for n, d in zip(normals, offsets, strict=True):
        if not faces:
            return []
        new: list[np.ndarray] = []
        cap: list[np.ndarray] = []
        has_cap = False  # a face already lies on the plane, facing +n: it is the cap
        for f in faces:
            s = _dot3(f, n) - float(d)
            if np.all(np.abs(s) <= _PLANE_EPS):  # coplanar face
                fn, area = _newell(f)
                if area <= _MIN_FACE_AREA:
                    continue
                if float(fn @ n) > 0:
                    has_cap = True
                    new.append(f)
                else:  # facing -n: the polytope lies outside, touching the plane
                    return []
                continue
            clipped, on = _clip_polygon(f, n, float(d))
            if len(clipped):
                new.append(clipped)
            cap.extend(on)
        if not has_cap:
            ordered = _order_on_plane(cap, n)
            if ordered is not None and new:
                new.append(ordered)
        faces = new
    return faces


def intersection(a: Piece, b: Piece) -> list[np.ndarray]:
    """The intersection of two convex pieces as polygon faces (empty if they do not overlap)."""
    lo = np.maximum(a.bounds[0], b.bounds[0])
    hi = np.minimum(a.bounds[1], b.bounds[1])
    if np.any(hi - lo <= 1e-12) or _separated(a, b) or _separated(b, a):
        return []
    return clip(a.polygon_faces(), b.normals, b.offsets)


def _separated(a: Piece, b: Piece) -> bool:
    """Whether one of ``a``'s face planes has all of ``b`` on its outer side.

    Touching counts, and so does a penetration of up to :data:`CONTACT_TOL`: rounding vertices
    to :data:`QUANTUM` can push a piece that sits flush on another (a gear tooth on its hub
    flat) a fraction of a micrometre into it, which is a contact, not an overlap."""
    for n, d in zip(a.normals, a.offsets, strict=True):
        if float(np.min(_dot3(b.verts, n))) >= d - CONTACT_TOL:
            return True
    return False


# --------------------------------------------------------------------------- mass properties


@dataclass(frozen=True)
class MassIntegrals:
    """Volume integrals of a solid: ``1, x, y, z, x^2, y^2, z^2, xy, yz, zx`` (m^3 .. m^5)."""

    values: tuple[float, ...]

    def __add__(self, other: MassIntegrals) -> MassIntegrals:
        return MassIntegrals(tuple(math.fsum(p) for p in zip(self.values, other.values, strict=True)))

    def scaled(self, k: float) -> MassIntegrals:
        return MassIntegrals(tuple(k * v for v in self.values))


def _integrals(faces: list[np.ndarray]) -> MassIntegrals:
    """Volume integrals of a closed polyhedron given as outward polygon faces.

    Tetrahedra from the origin to each face triangle (a fan per polygon); the second moments
    of a tetrahedron with one vertex at the origin are closed-form. Sums use ``math.fsum``.
    """
    tris = []
    for f in faces:
        for i in range(1, len(f) - 1):
            tris.append((f[0], f[i], f[i + 1]))
    if not tris:
        return MassIntegrals((0.0,) * 10)
    t = np.array(tris)  # (m, 3, 3)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    bc = _cross3(b, c)
    det = a[:, 0] * bc[:, 0] + a[:, 1] * bc[:, 1] + a[:, 2] * bc[:, 2]  # 6 x signed tetra volume
    vol = det / 6.0
    s = a + b + c
    first = vol[:, None] * s / 4.0

    # integral of x_i x_j over a tetra (0, a, b, c) = det/120 * (sum_k p_ki p_kj + s_i s_j)
    def second(i: int, j: int) -> np.ndarray:
        pp = a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j]
        return det / 120.0 * (pp + s[:, i] * s[:, j])

    cols = [
        vol,
        first[:, 0],
        first[:, 1],
        first[:, 2],
        second(0, 0),
        second(1, 1),
        second(2, 2),
        second(0, 1),
        second(1, 2),
        second(2, 0),
    ]
    return MassIntegrals(tuple(math.fsum(col.tolist()) for col in cols))


@dataclass(frozen=True)
class MassProperties:
    """Per unit density: volume (m^3), centre of mass (m) and the inertia about it
    ``Ixx Iyy Izz Ixy Ixz Iyz`` (m^5), rounded to 12 significant digits."""

    volume: float
    com: tuple[float, float, float]
    unit_inertia: tuple[float, float, float, float, float, float]


def _properties(total: MassIntegrals) -> MassProperties:
    v, sx, sy, sz, xx, yy, zz, xy, yz, zx = total.values
    if v <= 0:
        raise ValueError(f"non-positive volume {v}")
    cx, cy, cz = sx / v, sy / v, sz / v
    # second moments about the centre of mass
    cxx = xx - v * cx * cx
    cyy = yy - v * cy * cy
    czz = zz - v * cz * cz
    cxy = xy - v * cx * cy
    cyz = yz - v * cy * cz
    czx = zx - v * cz * cx
    inertia = (cyy + czz, cxx + czz, cxx + cyy, -cxy, -czx, -cyz)
    # Values that are zero by symmetry come out as rounding noise (1e-19 m, 1e-26 m^5); floor
    # them to exact zeros so they cannot differ between machines.
    i_floor = _ZERO_REL * max(abs(i) for i in inertia[:3])
    return MassProperties(
        volume=round_sig(v),
        com=tuple(_floor(round_sig(x), _COM_FLOOR) for x in (cx, cy, cz)),  # type: ignore[arg-type]
        unit_inertia=tuple(_floor(round_sig(i), i_floor) for i in inertia),  # type: ignore[arg-type]
    )


def _floor(value: float, eps: float) -> float:
    return 0.0 if abs(value) < eps else value


def overlaps(pieces: Sequence[Piece]) -> list[tuple[int, int, float]]:
    """Pairs of pieces whose intersection has positive volume: ``(i, j, volume)``."""
    out = []
    for i, j in itertools.combinations(range(len(pieces)), 2):
        faces = intersection(pieces[i], pieces[j])
        if faces:
            vol = _integrals(faces).values[0]
            if vol > MIN_OVERLAP_VOLUME:
                out.append((i, j, vol))
    return out


def mass_properties(pieces: Sequence[Piece]) -> MassProperties:
    """Volume, centre of mass and inertia (per unit density) of the union of ``pieces``.

    Exact inclusion-exclusion up to triple overlaps; a part whose pieces overlap four at a
    time is refused (no procedural part needs it).
    """
    if not pieces:
        raise ValueError("no pieces")
    total = MassIntegrals((0.0,) * 10)
    for p in pieces:
        total = total + _integrals(p.polygon_faces())
    pairs = []
    for i, j in itertools.combinations(range(len(pieces)), 2):
        faces = intersection(pieces[i], pieces[j])
        if faces:
            ints = _integrals(faces)
            if ints.values[0] > MIN_OVERLAP_VOLUME:
                pairs.append((i, j))
                total = total + ints.scaled(-1.0)
    paired = set(pairs)
    for i, j, k in itertools.combinations(range(len(pieces)), 3):
        if (i, j) in paired and (i, k) in paired and (j, k) in paired:
            faces = clip(
                clip(pieces[i].polygon_faces(), pieces[j].normals, pieces[j].offsets),
                pieces[k].normals,
                pieces[k].offsets,
            )
            if faces:
                tri = _integrals(faces)
                if tri.values[0] > MIN_OVERLAP_VOLUME:
                    raise ValueError(f"pieces {i}, {j} and {k} overlap; triple overlaps are not supported")
    return _properties(total)


def piece_volume(piece: Piece) -> float:
    return _integrals(piece.polygon_faces()).values[0]


# --------------------------------------------------------------------------- union mesh (visual)


def union_mesh(
    pieces: Sequence[Piece] | Sequence[np.ndarray], faces: Sequence[np.ndarray] | None = None
) -> trimesh.Trimesh:
    """The watertight union of convex pieces (manifold3d), for visuals and checks.

    Pass :class:`Piece` objects, or vertex arrays plus their ``faces``. Never used for a hash.
    """
    import trimesh

    out = union_manifold(pieces, faces).to_mesh64()
    return trimesh.Trimesh(
        np.asarray(out.vert_properties, dtype=np.float64)[:, :3], np.asarray(out.tri_verts), process=False
    )


def union_manifold(
    pieces: Sequence[Piece] | Sequence[np.ndarray], faces: Sequence[np.ndarray] | None = None
) -> Any:
    """The union as a ``manifold3d.Manifold``, built in float64 (float32 input leaves
    zero-volume slivers where pieces meet)."""
    import manifold3d as m3d

    mans = []
    for i, p in enumerate(pieces):
        if isinstance(p, Piece):
            v, f = p.verts, p.faces
        else:
            if faces is None:
                raise ValueError("pass faces with raw vertex arrays")
            v, f = np.asarray(p), np.asarray(faces[i])
        # nanobind needs writable, C-ordered arrays (the pieces are read-only)
        verts = np.array(v, dtype=np.float64, order="C", copy=True)
        tris = np.array(f, dtype=np.uint64, order="C", copy=True)
        mans.append(m3d.Manifold(m3d.Mesh64(verts, tris)))
    return m3d.Manifold.batch_boolean(mans, m3d.OpType.Add) if len(mans) > 1 else mans[0]
